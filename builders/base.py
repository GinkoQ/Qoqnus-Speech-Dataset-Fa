"""
Shared utilities for all dataset builders.
Import from builders that live in v2/builders/.
"""
import io
import os
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

TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 40.0
CLIP_AMP    = 0.999
CLIP_MS     = 10
BATCH_SIZE  = 1024
NUM_WORKERS = min(32, (os.cpu_count() or 4) * 2)

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

def extract_text(row, candidates):
    for col in candidates:
        v = row.get(col)
        if v and isinstance(v, str) and v.strip():
            return normalize(v.strip())
    return ""

def decode_hf_audio(audio):
    """Standard HF Audio feature dict: {"array": ndarray, "sampling_rate": int}"""
    if isinstance(audio, dict):
        arr = audio.get("array") or audio.get("bytes")
        if arr is None:
            raise ValueError("no array in audio dict")
        arr = np.asarray(arr, dtype=np.float32)
        sr = int(audio.get("sampling_rate", TARGET_SR))
        return arr, sr
    raise ValueError(f"unexpected audio type: {type(audio)}")

def decode_encoded_audio(audio):
    """AudioDecoder with _hf_encoded bytes (vhdm-style)"""
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

def decode_bytes_audio(audio):
    """Raw bytes or dict with bytes key"""
    if isinstance(audio, (bytes, bytearray)):
        arr, sr = sf.read(io.BytesIO(audio))
        return np.asarray(arr, dtype=np.float32), int(sr)
    if isinstance(audio, dict):
        b = audio.get("bytes")
        if b:
            arr, sr = sf.read(io.BytesIO(b))
            return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError(f"cannot decode bytes audio: {type(audio)}")

def process_audio(arr, sr):
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)
    if sr != TARGET_SR:
        arr = resampy.resample(arr.astype(np.float64), sr, TARGET_SR).astype(np.float32)
    arr = (arr - arr.mean()).astype(np.float32)
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

def make_record(utt_id, arr, text, speaker_id):
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
        "speaker_id":    speaker_id,
        "snr_db":        float(snr_db(arr)),
        "rms_db":        float(rms_db(arr)),
        "num_chars":     len(text),
        "num_words":     nw,
        "speaking_rate": float(round(nw / dur, 4)),
    }

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

def process_split_generic(source, split, rows, shards_dir, row_fn, all_drops):
    split_shards = shards_dir / split
    split_shards.mkdir(parents=True, exist_ok=True)
    n_in = len(rows)
    tasks = [(i, row, split) for i, row in enumerate(rows)]
    batch, n_ok, n_drop, split_hrs, shard_idx = [], 0, 0, 0.0, 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(row_fn, t): t[0] for t in tasks}
        bar = tqdm(as_completed(futures), total=n_in, desc=f"    {split}", unit="row", dynamic_ncols=True, smoothing=0.05)
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
                    flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
                    shard_idx += 1
                    batch.clear()
            bar.set_postfix({"ok": n_ok, "drop": n_drop, "drop%": f"{100*n_drop/max(1,n_ok+n_drop):.1f}", "hrs": f"{split_hrs:.2f}"}, refresh=False)
    if batch:
        flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
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

def assemble_split(shards_dir: Path, split: str):
    shard_paths = sorted((shards_dir / split).glob("shard_*.parquet"))
    ds = load_dataset("parquet", data_files=[str(p) for p in shard_paths], split="train", num_proc=min(8, os.cpu_count() or 1))
    def rebuild_audio(batch):
        batch["audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(batch["audio_flat"], batch["sampling_rate"])]
        return batch
    ds = ds.map(rebuild_audio, batched=True, batch_size=1024, num_proc=min(8, os.cpu_count() or 1), desc=f"  rebuilding audio [{split}]")
    ds = ds.remove_columns(["audio_flat"])
    return ds

def save_and_verify(source, out_dir, shards_dir, split_names, all_drops, total_kept, t0):
    log_dir = Path("logs")
    if all_drops:
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{source}_dropped.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for d in all_drops:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  drop log     : {log_path}")
    print(f"\n  assembling DatasetDict...")
    split_datasets = {}
    for split in split_names:
        print(f"    assembling {split}...")
        split_datasets[split] = assemble_split(shards_dir, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    dd = DatasetDict(split_datasets)
    dd.save_to_disk(str(out_dir))
    print(f"  saved ✓  ({out_dir})")
    for split, ds in dd.items():
        hrs = sum(ds["duration"]) / 3600
        print(f"    {split:<20} {len(ds):>8,} rows  {hrs:.2f}h")
    import shutil
    shutil.rmtree(shards_dir)
    print(f"  shards cleaned")
    print(f"\n  {'─'*50}")
    print(f"  verifying...")
    dd_v = load_from_disk(str(out_dir))
    all_utt_ids = []
    for split, ds in dd_v.items():
        assert all(v == TARGET_SR for v in ds["sampling_rate"]), f"[{split}] sampling_rate != 16000"
        assert all(v > 0 for v in ds["duration"]),               f"[{split}] duration <= 0"
        assert all(v for v in ds["text"]),                        f"[{split}] empty text"
        assert all(v for v in ds["speaker_id"]),                  f"[{split}] empty speaker_id"
        assert "split" not in ds.column_names,                    f"[{split}] split column present"
        all_utt_ids.extend(ds["utt_id"])
        for i in range(min(20, len(ds))):
            r = ds[i]
            assert r["n_samples"] == len(r["audio"]["array"]), f"[{split}] n_samples mismatch row {i}"
            p = r["audio"].get("path")
            assert not (p and str(p).strip()),                 f"[{split}] external path row {i}"
    assert len(set(all_utt_ids)) == len(all_utt_ids), "utt_id not globally unique"
    print(f"  all checks passed ✓")
    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n  done in {h:02d}:{m:02d}:{s:02d}")
    print(f"{'='*60}\n")
