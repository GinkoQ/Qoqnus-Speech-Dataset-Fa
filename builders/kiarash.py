"""
Standalone builder for kiarash dataset.
Run from v2/: python builders/kiarash.py

Source: raw/kiarash/*.parquet  (flat files, no split prefix)
  columns : ['path', 'text', 'audio', 'sampling_rate']
  audio   : raw bytes (wav/mp3 encoded)
  text    : text
  speaker : no speaker col -> gen:kiarash_spk0000
  splits  : all files -> train

Smoke test  : python builders/kiarash.py --smoke 5
Resume      : python builders/kiarash.py          (auto-detects checkpoint)
Force fresh : python builders/kiarash.py --force
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"
# Disable HF datasets cache to avoid filling disk with cached copies
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_kiarash"
os.environ["DISABLE_DATASETS_CACHING"] = "1"

import io, re, json, time, unicodedata, argparse
import numpy as np
import soundfile as sf
import soxr
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import datasets
datasets.disable_caching()
from tqdm import tqdm

SOURCE      = "kiarash"
RAW_DIR     = Path("raw/kiarash")
OUT_DIR     = Path("processed/kiarash")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
CKPT_FILE   = SHARDS_DIR / "_checkpoint.json"
TEXT_COL    = "text"
SPEAKER     = "gen:kiarash_spk0000"
TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 40.0
CLIP_AMP    = 0.999
CLIP_MS     = 10
NUM_WORKERS = min(32, (os.cpu_count() or 4) * 2)
BATCH_SIZE  = 1024

HOMOGLYPHS = str.maketrans({"ك":"ک","ي":"ی","ة":"ه","ى":"ی","ؤ":"و","أ":"ا","إ":"ا","ء":""})

SCHEMA = pa.schema([
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
        return {"done_files": [], "shard_idx": 0, "global_idx": 0, "n_ok": 0, "n_drop": 0, "hours": 0.0}
    with open(CKPT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(ckpt: dict):
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)
    tmp.replace(CKPT_FILE)

# ── audio / text ──────────────────────────────────────────────────────────────

def normalize(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()

def decode_audio(row):
    b = row["audio"]
    if not isinstance(b, (bytes, bytearray)) or len(b) == 0:
        raise ValueError("empty or invalid bytes")
    arr, sr = sf.read(io.BytesIO(b))
    return np.asarray(arr, dtype=np.float32), int(sr)

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
        runs = np.convolve(clipped, np.ones(min_s, dtype=np.uint8), mode="valid")
        if int((runs >= min_s).sum()) > 0: raise ValueError("clipping detected")
    return arr

def snr_db(arr):
    sp = float(np.mean(arr**2)); n = max(1, len(arr)//10)
    np_ = float(np.mean(np.sort(np.abs(arr))[:n]**2))
    return round(10*np.log10(max(sp,1e-10)/max(np_,1e-10)), 2)

def rms_db(arr):
    r = float(np.sqrt(np.mean(arr**2)))
    return round(20*np.log10(r), 2) if r > 0 else -120.0

def process_row(args):
    i, row, split = args
    utt_id = f"{SOURCE}_{split}_{i:08d}"
    try:
        arr_raw, sr = decode_audio(row)
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
        "speaker_id": SPEAKER, "snr_db": float(snr_db(arr)),
        "rms_db": float(rms_db(arr)), "num_chars": len(text),
        "num_words": nw, "speaking_rate": float(round(nw/dur,4)),
    }, None

def flush_batch(batch, shard_path):
    cols = {k: [] for k in SCHEMA.names}
    for r in batch:
        for k in SCHEMA.names:
            if k == "audio":
                cols[k].append({"array": r["audio"]["array"].tolist(), "sampling_rate": r["audio"]["sampling_rate"]})
            else:
                cols[k].append(r[k])
    audio_type = pa.struct([pa.field("array", pa.list_(pa.float32())), pa.field("sampling_rate", pa.int32())])
    arrays = []
    for field in SCHEMA:
        if field.name == "audio":
            arrays.append(pa.array(cols["audio"], type=audio_type))
        else:
            arrays.append(pa.array(cols[field.name], type=field.type))
    table = pa.table({f.name: arrays[i] for i,f in enumerate(SCHEMA)}, schema=SCHEMA)
    pq.write_table(table, str(shard_path), compression="snappy", write_statistics=False)
    del table, arrays, cols

# ── main ──────────────────────────────────────────────────────────────────────

def build(smoke=0, force=False):
    t0 = time.time()
    print(f"\n{'='*60}\n[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}\n{'='*60}")

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(RAW_DIR.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {RAW_DIR}")
    print(f"  source files : {len(parquet_files)}")
    print(f"  workers      : {NUM_WORKERS}  batch: {BATCH_SIZE}")

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    split = "train"
    split_shards = SHARDS_DIR / split
    split_shards.mkdir(parents=True, exist_ok=True)

    if force and CKPT_FILE.exists():
        CKPT_FILE.unlink()
        print(f"  --force: checkpoint cleared")

    ckpt       = load_checkpoint()
    done_files = set(ckpt["done_files"])
    shard_idx  = ckpt["shard_idx"]
    global_idx = ckpt["global_idx"]
    n_ok       = ckpt["n_ok"]
    n_drop     = ckpt["n_drop"]
    hrs        = ckpt["hours"]

    remaining = [f for f in parquet_files if f.stem not in done_files]

    if done_files:
        existing_shards = len(list(split_shards.glob("shard_*.parquet")))
        print(f"  resuming      : {len(done_files)}/{len(parquet_files)} files already done")
        print(f"  shards on disk: {existing_shards}")
        print(f"  rows so far   : ok={n_ok:,}  drop={n_drop:,}  hrs={hrs:.2f}")
        print(f"  remaining     : {len(remaining)} files")
    else:
        print(f"  starting fresh")

    if not remaining:
        print(f"  all files already processed — jumping to assembly")
    else:
        if smoke:
            remaining = remaining[:1]
            print(f"  smoke mode    : limiting to first remaining file only")

        all_drops: list[dict] = []

        bar = tqdm(
            desc=f"    {split}",
            unit="row",
            dynamic_ncols=True,
            smoothing=0.05,
            initial=n_ok + n_drop,
        )

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            for pf in remaining:
                file_ds = load_dataset("parquet", data_files=str(pf), split="train")
                if smoke:
                    file_ds = file_ds.select(range(min(smoke, len(file_ds))))
                rows = [dict(r) for r in file_ds]
                del file_ds

                bar.total = (bar.total or 0) + len(rows)
                bar.set_description(f"    {split} [{pf.stem}]")
                bar.refresh()

                tasks = [(global_idx + i, row, split) for i, row in enumerate(rows)]
                global_idx += len(rows)
                del rows

                batch: list[dict] = []
                futures = {pool.submit(process_row, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    record, drop = fut.result()
                    if drop:
                        n_drop += 1
                        all_drops.append(drop)
                    else:
                        n_ok += 1
                        hrs += record["duration"] / 3600
                        batch.append(record)
                        if len(batch) >= BATCH_SIZE:
                            flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
                            shard_idx += 1
                            batch.clear()
                    bar.update(1)
                    bar.set_postfix({"ok": n_ok, "drop": n_drop, "drop%": f"{100*n_drop/max(1,n_ok+n_drop):.1f}", "hrs": f"{hrs:.2f}"}, refresh=False)

                if batch:
                    flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
                    shard_idx += 1
                    batch.clear()

                done_files.add(pf.stem)
                save_checkpoint({
                    "done_files": sorted(done_files),
                    "shard_idx":  shard_idx,
                    "global_idx": global_idx,
                    "n_ok":       n_ok,
                    "n_drop":     n_drop,
                    "hours":      hrs,
                })

        bar.close()

        if all_drops:
            LOG_DIR.mkdir(exist_ok=True)
            log_path = LOG_DIR / f"{SOURCE}_dropped.jsonl"
            mode = "a" if log_path.exists() else "w"
            with open(log_path, mode, encoding="utf-8") as f:
                for d in all_drops:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"  drop log: {log_path} (this run: {len(all_drops)})")

    print(f"\n  total: kept {n_ok:,}  dropped {n_drop:,} ({100*n_drop/max(1,n_ok+n_drop):.1f}%)  {hrs:.2f}h")
    print(f"  shards: {shard_idx}")

    print(f"\n  assembling final dataset from {shard_idx} shards...")
    shards = sorted(split_shards.glob("shard_*.parquet"))

    # Validate shards first - remove corrupt ones
    valid_shards = []
    for shard in tqdm(shards, desc="  validating shards", unit="shard", dynamic_ncols=True):
        try:
            pq.read_metadata(str(shard))
            valid_shards.append(str(shard))
        except Exception as e:
            # Try partial recovery
            try:
                table = pq.read_table(str(shard))
                if len(table) > 0:
                    pq.write_table(table, str(shard), compression="snappy", write_statistics=False)
                    valid_shards.append(str(shard))
                    print(f"\n  ↳ repaired {shard.name}: {len(table)} rows")
                else:
                    shard.unlink()
                    print(f"\n  ↳ removed empty corrupt shard: {shard.name}")
            except Exception:
                shard.unlink()
                print(f"\n  ↳ removed unrecoverable shard: {shard.name} ({e})")

    print(f"  valid shards: {len(valid_shards)}/{len(shards)}")

    # Load all valid shards and save as DatasetDict
    print(f"  loading and saving...")
    ds_out = load_dataset(
        "parquet",
        data_files=valid_shards,
        split="train",
        num_proc=min(16, os.cpu_count() or 1),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds_out}).save_to_disk(str(OUT_DIR))
    print(f"  assembled: {len(ds_out):,} rows  {sum(ds_out['duration'])/3600:.2f}h")

    import shutil
    shutil.rmtree(SHARDS_DIR)
    print(f"  shards cleaned (checkpoint removed)")

    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    ds_v = dd_v["train"]
    assert all(v > 0 for v in ds_v["duration"]),       "duration <= 0"
    assert all(v for v in ds_v["text"]),                "empty text"
    assert all(v for v in ds_v["speaker_id"]),          "empty speaker_id"
    assert len(set(ds_v["utt_id"])) == len(ds_v),       "utt_id not unique"
    sample = ds_v.select(range(min(50, len(ds_v))))
    for r in sample:
        assert r["audio"]["sampling_rate"] == TARGET_SR, "sr != 16000"
        assert r["n_samples"] == len(r["audio"]["array"]), "n_samples mismatch"
        p = r["audio"].get("path")
        assert not (p and str(p).strip()), f"external path: {p}"
    print(f"  ✓ all checks passed")

    m, s = divmod(int(time.time()-t0), 60); h, m = divmod(m, 60)
    print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="limit to N rows from first remaining file")
    ap.add_argument("--force", action="store_true", help="ignore checkpoint and rebuild from scratch")
    args = ap.parse_args()
    build(smoke=args.smoke, force=args.force)