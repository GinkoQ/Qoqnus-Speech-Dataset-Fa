"""
Standalone builder for vhdm dataset.
Run from v2/: python builders/vhdm.py

Source: raw/vhdm/data/*.parquet
  audio: AudioDecoder with _hf_encoded["bytes"] (mp3 bytes)
  text:  corrected_sentence
  splits preserved as-is: train, test, validation
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import io
import re
import json
import time
import unicodedata
import numpy as np
import soundfile as sf
import resampy
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

SOURCE      = "vhdm"
RAW_DIR     = Path("raw/vhdm")
OUT_DIR     = Path("processed/vhdm")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
TEXT_COL    = "corrected_sentence"
SPEAKER     = "vhdm_spk0000"
TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 30.0
CLIP_AMP    = 0.999
CLIP_MS     = 10
NUM_WORKERS = min(32, (os.cpu_count() or 4) * 2)
BATCH_SIZE  = 1024

HOMOGLYPHS = str.maketrans({"ك": "ک", "ي": "ی", "ة": "ه", "ى": "ی", "ؤ": "و", "أ": "ا", "إ": "ا", "ء": ""})

SCHEMA = pa.schema([
    pa.field("utt_id",        pa.string()),
    pa.field("audio_flat",    pa.list_(pa.float32())),
    pa.field("sampling_rate", pa.int32()),
    pa.field("text",          pa.string()),
    pa.field("duration",      pa.float32()),
    pa.field("n_samples",     pa.int32()),
    pa.field("speaker_id",    pa.string()),
    pa.field("snr_db",        pa.float32()),
    pa.field("rms_db",        pa.float32()),
    pa.field("num_chars",     pa.int32()),
    pa.field("num_words",     pa.int32()),
    pa.field("speaking_rate", pa.float32()),
])

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(HOMOGLYPHS)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

def decode_audio(audio):
    encoded = audio._hf_encoded
    b = encoded.get("bytes")
    if b and len(b) > 0:
        arr, sr = sf.read(io.BytesIO(b))
        return np.asarray(arr, dtype=np.float32), int(sr)
    p = encoded.get("path")
    if p and Path(p).exists():
        arr, sr = sf.read(str(p))
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError("no bytes or valid path in _hf_encoded")

def process_audio(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)
    if sr != TARGET_SR:
        arr = resampy.resample(arr.astype(np.float64), sr, TARGET_SR).astype(np.float32)
    arr = (arr - arr.mean()).astype(np.float32)
    peak = np.abs(arr).max()
    if peak > 0: arr = (arr / peak).astype(np.float32)
    dur = len(arr) / TARGET_SR
    if dur < MIN_DUR:
        raise ValueError(f"too short {dur:.3f}s")
    if dur > MAX_DUR:
        raise ValueError(f"too long {dur:.1f}s")
    min_s = max(1, int(CLIP_MS * TARGET_SR / 1000))
    clipped = (np.abs(arr) >= CLIP_AMP).astype(np.uint8)
    if int(clipped.sum()) >= min_s:
        runs = np.convolve(clipped, np.ones(min_s, dtype=np.uint8), mode="valid")
        if int((runs >= min_s).sum()) > 0:
            raise ValueError("clipping detected")
    return arr

def snr_db(arr):
    sp = float(np.mean(arr ** 2))
    n = max(1, len(arr) // 10)
    np_ = float(np.mean(np.sort(np.abs(arr))[:n] ** 2))
    return round(10 * np.log10(max(sp, 1e-10) / max(np_, 1e-10)), 2)

def rms_db(arr):
    r = float(np.sqrt(np.mean(arr ** 2)))
    return round(20 * np.log10(r), 2) if r > 0 else -120.0

def process_row(args):
    i, row, split = args
    utt_id = f"{SOURCE}_{split}_{i:08d}"
    try:
        arr_raw, sr = decode_audio(row["audio"])
        arr = process_audio(arr_raw, sr)
    except Exception as e:
        return None, {"utt_id": utt_id, "reason": f"audio: {e}"}
    text = normalize(str(row.get(TEXT_COL) or ""))
    if not text:
        return None, {"utt_id": utt_id, "reason": "empty text"}
    ns = len(arr)
    dur = ns / TARGET_SR
    nw = len(text.split())
    return {
        "utt_id":        utt_id,
        "audio_flat":    arr,
        "sampling_rate": TARGET_SR,
        "text":          text,
        "duration":      float(round(dur, 4)),
        "n_samples":     ns,
        "speaker_id":    SPEAKER,
        "snr_db":        float(snr_db(arr)),
        "rms_db":        float(rms_db(arr)),
        "num_chars":     len(text),
        "num_words":     nw,
        "speaking_rate": float(round(nw / dur, 4)),
    }, None

def flush_batch(batch: list[dict], shard_path: Path):
    cols = {k: [] for k in SCHEMA.names}
    for r in batch:
        for k in SCHEMA.names:
            cols[k].append(r[k])
    arrays = []
    for field in SCHEMA:
        if field.name == "audio_flat":
            arrays.append(pa.array(cols[field.name], type=pa.list_(pa.float32())))
        else:
            arrays.append(pa.array(cols[field.name], type=field.type))
    table = pa.table({f.name: arrays[i] for i, f in enumerate(SCHEMA)}, schema=SCHEMA)
    pq.write_table(table, str(shard_path), compression="snappy", write_statistics=False)
    del table, arrays, cols

def process_split(split: str, files: list[str], all_drops: list[dict], smoke: int = 0) -> int:
    split_shards_dir = SHARDS_DIR / split
    split_shards_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_files=files, split="train")
    if smoke:
        ds = ds.select(range(min(smoke, len(ds))))
    n_in = len(ds)
    print(f"\n  [{split}] {n_in:,} rows  ({len(files)} file(s))")
    rows = [dict(r) for r in ds]
    del ds
    tasks = [(i, row, split) for i, row in enumerate(rows)]
    del rows

    batch: list[dict] = []
    n_ok = 0
    n_drop = 0
    split_hrs = 0.0
    shard_idx = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(process_row, t): t[0] for t in tasks}
        bar = tqdm(
            as_completed(futures),
            total=n_in,
            desc=f"    {split}",
            unit="row",
            dynamic_ncols=True,
            smoothing=0.05,
        )
        for fut in bar:
            record, drop = fut.result()
            if drop:
                n_drop += 1
                all_drops.append(drop)
            else:
                n_ok += 1
                split_hrs += record["duration"] / 3600
                batch.append(record)
                if len(batch) >= BATCH_SIZE:
                    flush_batch(batch, split_shards_dir / f"shard_{shard_idx:06d}.parquet")
                    shard_idx += 1
                    batch.clear()
            bar.set_postfix({"ok": n_ok, "drop": n_drop, "drop%": f"{100*n_drop/max(1,n_ok+n_drop):.1f}", "hrs": f"{split_hrs:.2f}"}, refresh=False)

    if batch:
        flush_batch(batch, split_shards_dir / f"shard_{shard_idx:06d}.parquet")
        batch.clear()

    drop_pct = 100 * n_drop / max(1, n_in)
    print(f"    → kept {n_ok:,}  dropped {n_drop:,} ({drop_pct:.1f}%)  {split_hrs:.2f}h")
    if n_drop:
        reason_counts: dict[str, int] = {}
        for d in all_drops[-n_drop:]:
            key = d["reason"].split(":")[0].strip()
            reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"      {reason:<30} {count:>6,}")
    return n_ok

def assemble_split(split: str) -> object:
    split_shards_dir = SHARDS_DIR / split
    shard_paths = sorted(split_shards_dir.glob("shard_*.parquet"))
    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in shard_paths],
        split="train",
        num_proc=min(8, os.cpu_count() or 1),
    )
    def rebuild_audio(batch):
        batch["audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(batch["audio_flat"], batch["sampling_rate"])]
        return batch
    ds = ds.map(
        rebuild_audio,
        batched=True,
        batch_size=1024,
        num_proc=min(8, os.cpu_count() or 1),
        desc=f"  rebuilding audio [{split}]",
    )
    ds = ds.remove_columns(["audio_flat"])
    return ds

def build(smoke: int = 0):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}")
    print(f"{'='*60}")
    print(f"  raw dir    : {RAW_DIR.resolve()}")
    print(f"  output dir : {OUT_DIR.resolve()}")
    print(f"  workers    : {NUM_WORKERS}")
    print(f"  batch size : {BATCH_SIZE}")

    data_dir = RAW_DIR / "data" if (RAW_DIR / "data").exists() else RAW_DIR
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {data_dir}")
    print(f"  parquets   : {len(parquet_files)} files")

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    split_files: dict[str, list] = {}
    for f in parquet_files:
        prefix = f.stem.split("-")[0]
        split_files.setdefault(prefix, []).append(str(f))
    print(f"  splits     : {sorted(split_files.keys())}")

    all_drops: list[dict] = []
    kept_per_split: dict[str, int] = {}

    for split, files in split_files.items():
        kept_per_split[split] = process_split(split, files, all_drops, smoke=smoke)

    total_kept = sum(kept_per_split.values())
    print(f"\n  {'─'*50}")
    print(f"  total kept   : {total_kept:,}")
    print(f"  total dropped: {len(all_drops):,}")

    if all_drops:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"{SOURCE}_dropped.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for d in all_drops:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  drop log     : {log_path}")

    print(f"\n  assembling DatasetDict...")
    split_datasets = {}
    for split in split_files:
        print(f"    assembling {split}...")
        split_datasets[split] = assemble_split(split)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd = DatasetDict(split_datasets)
    dd.save_to_disk(str(OUT_DIR))
    print(f"  saved ✓  ({OUT_DIR})")
    for split, ds in dd.items():
        hrs = sum(ds["duration"]) / 3600
        print(f"    {split:<20} {len(ds):>8,} rows  {hrs:.2f}h")

    import shutil
    shutil.rmtree(SHARDS_DIR)
    print(f"  shards cleaned")

    print(f"\n  {'─'*50}")
    print(f"  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    all_utt_ids = []
    for split, ds in dd_v.items():
        assert all(v == TARGET_SR for v in ds["sampling_rate"]), f"[{split}] sampling_rate != 16000"
        assert all(v > 0 for v in ds["duration"]),               f"[{split}] duration <= 0"
        assert all(v for v in ds["text"]),                        f"[{split}] empty text"
        assert all(v for v in ds["speaker_id"]),                  f"[{split}] empty speaker_id"
        assert "split" not in ds.column_names,                    f"[{split}] split column still present"
        all_utt_ids.extend(ds["utt_id"])
        for i in range(min(20, len(ds))):
            r = ds[i]
            assert r["n_samples"] == len(r["audio"]["array"]), f"[{split}] n_samples mismatch at row {i}"
            p = r["audio"].get("path")
            assert not (p and str(p).strip()),                 f"[{split}] external path at row {i}"
    assert len(set(all_utt_ids)) == len(all_utt_ids), "utt_id not globally unique"
    print(f"  all checks passed ✓")

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n  done in {h:02d}:{m:02d}:{s:02d}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    build(smoke=args.smoke)