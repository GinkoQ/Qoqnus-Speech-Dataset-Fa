"""
Standalone builder for srezas dataset.
Run from v2/: python builders/srezas.py

Source: raw/srezas/<config>/
  common_voice_17 : test-*.parquet, train-*.parquet
  fleurs          : test-*.parquet, train-*.parquet
  yazdi_accent    : test-*.parquet, train-*.parquet
  youtube         : <channel>-*.parquet  (all -> train, no split prefix)

  columns : ['audio', 'sentence']
  audio   : AudioDecoder (_hf_encoded bytes, mp3)
  text    : sentence
  speaker : gen:srezas_<config>_spk0000  (no speaker col in any config)

Smoke test  : python builders/srezas.py --smoke 5
Resume      : python builders/srezas.py
Force fresh : python builders/srezas.py --force
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import io, re, json, time, unicodedata, argparse
import numpy as np
import soundfile as sf
import soxr
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm

SOURCE      = "srezas"
RAW_DIR     = Path("raw/srezas")
OUT_DIR     = Path("processed/srezas")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
CKPT_FILE   = SHARDS_DIR / "_checkpoint.json"
TEXT_COL    = "sentence"
TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 40.0
CLIP_AMP    = 0.999
CLIP_MS     = 10
NUM_WORKERS = max(32, (os.cpu_count() or 4) * 2)
BATCH_SIZE  = 4096

CONFIG_SHORT = {
    "common_voice_17": "cv17",
    "fleurs":          "fleurs",
    "youtube":         "youtube",
    "yazdi_accent":    "yazdi",
}

HOMOGLYPHS = str.maketrans({"ك":"ک","ي":"ی","ة":"ه","ى":"ی","ؤ":"و","أ":"ا","إ":"ا","ء":""})

OUT_SCHEMA = pa.schema([
    pa.field("utt_id",        pa.string()),
    pa.field("audio",         pa.struct([
        pa.field("array",         pa.list_(pa.float32())),
        pa.field("sampling_rate", pa.int32()),
    ])),
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

# ── checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if not CKPT_FILE.exists():
        return {"done": set(), "shard_offsets": {}, "drop_counts": {}, "row_counts": {}, "hours": {}}
    with open(CKPT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["done"] = set(raw.get("done", []))
    return raw

def save_checkpoint(ckpt: dict):
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({**ckpt, "done": sorted(ckpt["done"])}, f, indent=2, ensure_ascii=False)
    tmp.replace(CKPT_FILE)

def verify_shards(split: str) -> bool:
    d = SHARDS_DIR / split
    return d.exists() and any(d.glob("shard_*.parquet"))

# ── audio / text ──────────────────────────────────────────────────────────────

def normalize(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()

def decode_audio(audio_val) -> tuple[np.ndarray, int]:
    """
    audio_val comes from pyarrow.to_pylist() — it is a plain dict:
      {"bytes": b"...", "path": "filename.mp3"}
    We decode it ourselves with soundfile. torchcodec never touches it.
    """
    if isinstance(audio_val, dict):
        b = audio_val.get("bytes")
        if b and len(b) > 0:
            arr, sr = sf.read(io.BytesIO(bytes(b)))
            return np.asarray(arr, dtype=np.float32), int(sr)
        p = audio_val.get("path")
        if p and Path(str(p)).exists():
            arr, sr = sf.read(str(p))
            return np.asarray(arr, dtype=np.float32), int(sr)
    if isinstance(audio_val, (bytes, bytearray)) and len(audio_val) > 0:
        arr, sr = sf.read(io.BytesIO(bytes(audio_val)))
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError(f"no usable audio bytes (type={type(audio_val).__name__})")

def process_audio(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1: arr = arr.mean(axis=-1).astype(np.float32)
    if sr != TARGET_SR:
        arr = soxr.resample(arr, sr, TARGET_SR).astype(np.float32)
    arr = (arr - arr.mean()).astype(np.float32)
    peak = np.abs(arr).max()
    if peak > 0: arr = (arr / peak).astype(np.float32)
    dur = len(arr) / TARGET_SR
    if dur < MIN_DUR: raise ValueError(f"too short {dur:.3f}s")
    if dur > MAX_DUR: raise ValueError(f"too long {dur:.1f}s")
    min_s = max(1, int(CLIP_MS * TARGET_SR / 1000))
    clipped = (np.abs(arr) >= CLIP_AMP).astype(np.uint8)
    if int(clipped.sum()) >= min_s:
        cs = np.cumsum(np.concatenate(([0], clipped)))
        if int(((cs[min_s:] - cs[:-min_s]) >= min_s).sum()) > 0:
            raise ValueError("clipping detected")
    return arr

def snr_db(arr):
    sp = float(np.mean(arr**2)); n = max(1, len(arr)//10)
    np_ = float(np.mean(np.partition(np.abs(arr), n)[:n]**2))
    return round(10*np.log10(max(sp,1e-10)/max(np_,1e-10)), 2)

def rms_db(arr):
    r = float(np.sqrt(np.mean(arr**2)))
    return round(20*np.log10(r), 2) if r > 0 else -120.0

def make_process_row(config):
    speaker = f"gen:{SOURCE}_{config}_spk0000"
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
        ns = len(arr); dur = ns / TARGET_SR; nw = len(text.split())
        return {
            "utt_id": utt_id,
            "audio": {"array": arr, "sampling_rate": TARGET_SR},
            "text": text, "duration": float(round(dur,4)), "n_samples": ns,
            "speaker_id": speaker, "snr_db": float(snr_db(arr)),
            "rms_db": float(rms_db(arr)), "num_chars": len(text),
            "num_words": nw, "speaking_rate": float(round(nw/dur,4)),
        }, None
    return process_row

def flush_batch(batch, shard_path):
    cols = {k: [] for k in OUT_SCHEMA.names}
    for r in batch:
        for k in OUT_SCHEMA.names:
            if k == "audio":
                cols[k].append({"array": r["audio"]["array"].tolist(), "sampling_rate": r["audio"]["sampling_rate"]})
            else:
                cols[k].append(r[k])
    audio_type = pa.struct([pa.field("array", pa.list_(pa.float32())), pa.field("sampling_rate", pa.int32())])
    arrays = []
    for field in OUT_SCHEMA:
        if field.name == "audio":
            arrays.append(pa.array(cols["audio"], type=audio_type))
        else:
            arrays.append(pa.array(cols[field.name], type=field.type))
    table = pa.table({f.name: arrays[i] for i,f in enumerate(OUT_SCHEMA)}, schema=OUT_SCHEMA)
    pq.write_table(table, str(shard_path), compression="snappy", write_statistics=False)
    del table, arrays, cols

def collect_config_splits(config_dir: Path, config: str) -> dict[str, list[str]]:
    parquet_files = sorted(config_dir.glob("*.parquet"))
    if not parquet_files:
        return {}
    short = CONFIG_SHORT.get(config, config)
    split_files: dict[str, list] = {}
    for f in parquet_files:
        file_prefix = f.stem.split("-")[0]
        split_name = f"{short}_{file_prefix}"
        split_files.setdefault(split_name, []).append(str(f))
    return split_files

def read_parquet_raw(files: list[str], smoke: int = 0) -> list[dict]:
    """
    Read parquet files directly with pyarrow — bypasses HF entirely.
    audio column arrives as a plain dict {"bytes": b"...", "path": "..."}.
    torchcodec / HF AudioDecoder never runs.
    """
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables)
    del tables
    if smoke:
        table = table.slice(0, min(smoke, len(table)))
    rows = table.to_pylist()
    del table
    return rows

def process_files(config, split, files, all_drops, shard_dir, shard_offset, smoke=0):
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = read_parquet_raw(files, smoke=smoke)
    n_in = len(rows)
    print(f"\n    [{config}/{split}] {n_in:,} rows  ({len(files)} files)")
    process_row = make_process_row(config)
    CHUNK = NUM_WORKERS * 16
    batch, n_ok, n_drop, hrs, shard_idx = [], 0, 0, 0.0, shard_offset
    bar = tqdm(total=n_in, desc=f"      {config}/{split}", unit="row", dynamic_ncols=True, smoothing=0.05)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for chunk_start in range(0, n_in, CHUNK):
            chunk = rows[chunk_start:chunk_start + CHUNK]
            tasks = [(chunk_start + i, row, split) for i, row in enumerate(chunk)]
            del chunk
            futures = {pool.submit(process_row, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                record, drop = fut.result()
                if drop:
                    n_drop += 1; all_drops.append(drop)
                else:
                    n_ok += 1; hrs += record["duration"]/3600; batch.append(record)
                    if len(batch) >= BATCH_SIZE:
                        flush_batch(batch, shard_dir/f"shard_{shard_idx:06d}.parquet")
                        shard_idx += 1; batch.clear()
                bar.update(1)
                bar.set_postfix({"ok":n_ok,"drop":n_drop,"drop%":f"{100*n_drop/max(1,n_ok+n_drop):.1f}","hrs":f"{hrs:.2f}"}, refresh=False)
    bar.close()
    del rows
    if batch:
        flush_batch(batch, shard_dir/f"shard_{shard_idx:06d}.parquet"); shard_idx += 1; batch.clear()
    print(f"      → kept {n_ok:,}  dropped {n_drop:,} ({100*n_drop/max(1,n_in):.1f}%)  {hrs:.2f}h")
    return n_ok, n_drop, hrs, shard_idx

def assemble_split(split):
    from datasets import load_dataset
    shards = sorted((SHARDS_DIR/split).glob("shard_*.parquet"))
    return load_dataset("parquet", data_files=[str(p) for p in shards], split="train", num_proc=10)

# ── main ──────────────────────────────────────────────────────────────────────

def build(smoke=0, force=False):
    t0 = time.time()
    print(f"\n{'='*60}\n[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}\n{'='*60}")
    if not RAW_DIR.exists(): raise FileNotFoundError(f"not found: {RAW_DIR}")
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    if force and CKPT_FILE.exists():
        CKPT_FILE.unlink()
        print(f"  --force: checkpoint cleared")
    ckpt = load_checkpoint()
    done = ckpt["done"]
    shard_offsets: dict[str, int] = ckpt.get("shard_offsets", {})
    if done:
        print(f"  resuming — {len(done)} config/split(s) already done: {sorted(done)}")
    else:
        print(f"  starting fresh")
    all_drops: list[dict] = []
    config_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"  configs found: {[d.name for d in config_dirs]}")
    for config_dir in config_dirs:
        config = config_dir.name
        split_files = collect_config_splits(config_dir, config)
        if not split_files:
            print(f"  [{config}] no parquet files, skipping")
            continue
        print(f"\n  [{config}]  splits: {sorted(split_files.keys())}")
        for split, files in split_files.items():
            key = f"{config}/{split}"
            if key in done:
                n_shards = len(list((SHARDS_DIR/split).glob("shard_*.parquet"))) if (SHARDS_DIR/split).exists() else 0
                print(f"\n    [{config}/{split}] ✓ already done ({n_shards} shards) — skipping")
                continue
            if not verify_shards(split):
                shard_offsets[split] = 0
            offset = shard_offsets.get(split, 0)
            shard_dir = SHARDS_DIR / split
            n_ok, n_drop, hrs, new_offset = process_files(config, split, files, all_drops, shard_dir, offset, smoke=smoke)
            shard_offsets[split] = new_offset
            done.add(key)
            ckpt["done"]             = done
            ckpt["shard_offsets"]    = shard_offsets
            ckpt["drop_counts"][key] = n_drop
            ckpt["row_counts"][key]  = n_ok
            ckpt["hours"][key]       = hrs
            save_checkpoint(ckpt)
            print(f"      ✓ checkpoint saved  ({len(done)} total done)")
    if all_drops:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"{SOURCE}_dropped.jsonl"
        mode = "a" if log_path.exists() else "w"
        with open(log_path, mode, encoding="utf-8") as f:
            for d in all_drops: f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n  drop log: {log_path} (this run: {len(all_drops)})")
    all_split_names = sorted(shard_offsets.keys())
    print(f"\n  assembling DatasetDict from splits: {all_split_names}")
    split_datasets = {}
    for split in all_split_names:
        if not verify_shards(split):
            print(f"  WARNING: no shards for '{split}', skipping")
            continue
        ds = assemble_split(split)
        split_datasets[split] = ds
        print(f"    {split:<25} {len(ds):>8,} rows  {sum(ds['duration'])/3600:.2f}h")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DatasetDict(split_datasets).save_to_disk(str(OUT_DIR))
    import shutil
    shutil.rmtree(SHARDS_DIR)
    print(f"  shards cleaned")
    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    all_ids = []
    for split, ds in dd_v.items():
        assert all(r["audio"]["sampling_rate"] == TARGET_SR for r in ds.select(range(min(50,len(ds))))), f"[{split}] sr != 16000"
        assert all(v > 0 for v in ds["duration"]),   f"[{split}] duration <= 0"
        assert all(v for v in ds["text"]),            f"[{split}] empty text"
        assert all(v for v in ds["speaker_id"]),      f"[{split}] empty speaker_id"
        all_ids.extend(ds["utt_id"])
        for i in range(min(20, len(ds))):
            r = ds[i]
            assert r["n_samples"] == len(r["audio"]["array"]), f"[{split}] n_samples mismatch row {i}"
            p = r["audio"].get("path"); assert not (p and str(p).strip()), f"[{split}] external path row {i}"
    assert len(set(all_ids)) == len(all_ids), "utt_id not globally unique"
    print(f"  ✓ all checks passed")
    m, s = divmod(int(time.time()-t0), 60); h, m = divmod(m, 60)
    print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(smoke=args.smoke, force=args.force)