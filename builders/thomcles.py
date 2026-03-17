"""
Standalone builder for thomcles dataset.
Run from v2/: python builders/thomcles.py

Source: raw/thomcles/*.parquet  (flat files, no split prefix)
  columns : ['audio', 'sentence' / 'text' / 'transcription']
  audio   : HF Audio dict or AudioDecoder bytes
  text    : sentence / text / transcription
  speaker : no speaker col -> gen:thomcles_spk0000
  splits  : all files -> train

Modes:
  python builders/thomcles.py                  # resume from checkpoint / existing shards
  python builders/thomcles.py --assemble-only  # shards already done, just assemble + verify
  python builders/thomcles.py --smoke 5        # test with 5 rows from first file
  python builders/thomcles.py --force          # delete checkpoint and rebuild from scratch
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import io, re, json, time, unicodedata, argparse
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

SOURCE      = "thomcles"
RAW_DIR     = Path("raw/thomcles")
OUT_DIR     = Path("processed/thomcles")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
CKPT_FILE   = SHARDS_DIR / "_checkpoint.json"
TEXT_COLS   = ["sentence", "text", "transcription", "transcript"]
SPEAKER     = "gen:thomcles_spk0000"
TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 40.0
CLIP_AMP    = 0.999
CLIP_MS     = 10
NUM_WORKERS = min(32, (os.cpu_count() or 4) * 2)
BATCH_SIZE  = 1024

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

def existing_shards() -> list[Path]:
    d = SHARDS_DIR / "train"
    return sorted(d.glob("shard_*.parquet")) if d.exists() else []

def read_max_global_idx_from_shards(shards: list[Path]) -> int:
    """
    Read utt_id column only from all existing shards and find the
    highest integer index to avoid any utt_id collision.
    utt_id format: {SOURCE}_train_{index:08d}
    """
    if not shards:
        return 0
    prefix = f"{SOURCE}_train_"
    max_idx = 0
    for shard in tqdm(shards, desc="  scanning existing shards for max utt_id", unit="shard"):
        table = pq.read_table(str(shard), columns=["utt_id"])
        for uid in table.column("utt_id").to_pylist():
            if isinstance(uid, str) and uid.startswith(prefix):
                idx = int(uid[len(prefix):])
                if idx > max_idx:
                    max_idx = idx
        del table
    return max_idx + 1

def load_checkpoint() -> dict:
    n_shards = len(existing_shards())
    if not CKPT_FILE.exists():
        if n_shards > 0:
            print(f"  WARNING: {n_shards} shards on disk but no checkpoint.")
            real_next = read_max_global_idx_from_shards(existing_shards())
            print(f"  Scanned shards — actual next global_idx: {real_next:,}")
            print(f"  Use --assemble-only if all source files are already processed.")
        else:
            real_next = 0
        return {"done_files": [], "shard_idx": n_shards, "global_idx": real_next, "n_ok": 0, "n_drop": 0, "hours": 0.0}
    with open(CKPT_FILE, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    if n_shards > ckpt.get("shard_idx", 0):
        print(f"  NOTE: {n_shards} shards on disk > checkpoint shard_idx={ckpt['shard_idx']} — updating.")
        ckpt["shard_idx"] = n_shards
    return ckpt

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

def extract_text(row) -> str:
    for col in TEXT_COLS:
        v = row.get(col)
        if v and isinstance(v, str) and v.strip():
            return normalize(v.strip())
    return ""

def decode_audio(audio_val) -> tuple[np.ndarray, int]:
    if isinstance(audio_val, dict):
        arr = audio_val.get("array")
        if arr is not None:
            return np.asarray(arr, dtype=np.float32), int(audio_val.get("sampling_rate", TARGET_SR))
        b = audio_val.get("bytes")
        if b and len(b) > 0:
            a, sr = sf.read(io.BytesIO(bytes(b)))
            return np.asarray(a, dtype=np.float32), int(sr)
        p = audio_val.get("path")
        if p and Path(str(p)).exists():
            a, sr = sf.read(str(p))
            return np.asarray(a, dtype=np.float32), int(sr)
    if isinstance(audio_val, (bytes, bytearray)) and len(audio_val) > 0:
        a, sr = sf.read(io.BytesIO(bytes(audio_val)))
        return np.asarray(a, dtype=np.float32), int(sr)
    raise ValueError(f"no usable audio data (type={type(audio_val).__name__})")

def process_audio(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1: arr = arr.mean(axis=-1).astype(np.float32)
    if sr != TARGET_SR:
        import resampy
        arr = resampy.resample(arr.astype(np.float64), sr, TARGET_SR).astype(np.float32)
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
        arr_raw, sr = decode_audio(row["audio"])
        arr = process_audio(arr_raw, sr)
    except Exception as e:
        return None, {"utt_id": utt_id, "reason": f"audio: {e}"}
    text = extract_text(row)
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

def read_parquet_raw(path: str, smoke: int = 0) -> list[dict]:
    table = pq.read_table(path)
    if smoke:
        table = table.slice(0, min(smoke, len(table)))
    rows = table.to_pylist()
    del table
    return rows

# ── assemble + verify ─────────────────────────────────────────────────────────

def assemble_and_save() -> object:
    shards = existing_shards()
    if not shards:
        raise FileNotFoundError(f"no shards in {SHARDS_DIR / 'train'}")
    print(f"\n  assembling {len(shards)} shards...")
    ds_out = load_dataset("parquet", data_files=[str(p) for p in shards], split="train", num_proc=min(8, os.cpu_count() or 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds_out}).save_to_disk(str(OUT_DIR))
    hrs = sum(ds_out["duration"]) / 3600
    print(f"  saved ✓  train: {len(ds_out):,} rows  {hrs:.2f}h")
    import shutil
    shutil.rmtree(SHARDS_DIR)
    print(f"  shards cleaned")
    return ds_out

def verify():
    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    ds_v = dd_v["train"]
    utt_ids = ds_v["utt_id"]
    n_unique = len(set(utt_ids))
    if n_unique != len(ds_v):
        dupes = len(ds_v) - n_unique
        raise AssertionError(f"utt_id not unique: {dupes:,} duplicates — run thomcles_dedup_fix.py")
    assert all(v > 0 for v in ds_v["duration"]),        "duration <= 0"
    assert all(v for v in ds_v["text"]),                 "empty text"
    assert all(v for v in ds_v["speaker_id"]),           "empty speaker_id"
    sample = ds_v.select(range(min(50, len(ds_v))))
    for r in sample:
        assert r["audio"]["sampling_rate"] == TARGET_SR, "sr != 16000"
        assert r["n_samples"] == len(r["audio"]["array"]), "n_samples mismatch"
        p = r["audio"].get("path")
        assert not (p and str(p).strip()), f"external path: {p}"
    print(f"  ✓ all checks passed")

# ── main ──────────────────────────────────────────────────────────────────────

def build(smoke=0, force=False, assemble_only=False):
    t0 = time.time()
    print(f"\n{'='*60}\n[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}\n{'='*60}")
    print(f"  workers: {NUM_WORKERS}  batch: {BATCH_SIZE}")

    if assemble_only:
        print(f"  --assemble-only: using existing shards")
        assemble_and_save()
        verify()
        m, s = divmod(int(time.time()-t0), 60); h, m = divmod(m, 60)
        print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")
        return

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(RAW_DIR.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files under {RAW_DIR}")
    print(f"  source files : {len(parquet_files)}")

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    split_shards = SHARDS_DIR / "train"
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
        print(f"  resuming      : {len(done_files)}/{len(parquet_files)} files done")
        print(f"  shards on disk: {shard_idx}  next global_idx: {global_idx:,}")
        print(f"  rows so far   : ok={n_ok:,}  drop={n_drop:,}  hrs={hrs:.2f}")
        print(f"  remaining     : {len(remaining)} files")
    else:
        print(f"  shard_idx={shard_idx}  global_idx={global_idx:,}")

    if not remaining:
        print(f"\n  all source files processed — assembling")
        assemble_and_save()
        verify()
        m, s = divmod(int(time.time()-t0), 60); h, m = divmod(m, 60)
        print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")
        return

    if smoke:
        remaining = remaining[:1]
        print(f"  smoke: first remaining file only")

    all_drops: list[dict] = []
    bar = tqdm(desc=f"    train", unit="row", dynamic_ncols=True, smoothing=0.05, initial=n_ok + n_drop)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for pf in remaining:
            rows = read_parquet_raw(str(pf), smoke=smoke)
            bar.total = (bar.total or 0) + len(rows)
            bar.set_description(f"    train [{pf.stem}]")
            bar.refresh()
            tasks = [(global_idx + i, row, "train") for i, row in enumerate(rows)]
            global_idx += len(rows)
            del rows
            batch: list[dict] = []
            futures = {pool.submit(process_row, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                record, drop = fut.result()
                if drop:
                    n_drop += 1; all_drops.append(drop)
                else:
                    n_ok += 1; hrs += record["duration"] / 3600; batch.append(record)
                    if len(batch) >= BATCH_SIZE:
                        flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
                        shard_idx += 1; batch.clear()
                bar.update(1)
                bar.set_postfix({"ok": n_ok, "drop": n_drop, "drop%": f"{100*n_drop/max(1,n_ok+n_drop):.1f}", "hrs": f"{hrs:.2f}"}, refresh=False)
            if batch:
                flush_batch(batch, split_shards / f"shard_{shard_idx:06d}.parquet")
                shard_idx += 1; batch.clear()
            done_files.add(pf.stem)
            save_checkpoint({"done_files": sorted(done_files), "shard_idx": shard_idx, "global_idx": global_idx, "n_ok": n_ok, "n_drop": n_drop, "hours": hrs})

    bar.close()

    if all_drops:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / f"{SOURCE}_dropped.jsonl"
        mode = "a" if log_path.exists() else "w"
        with open(log_path, mode, encoding="utf-8") as f:
            for d in all_drops: f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  drop log: {log_path} (this run: {len(all_drops)})")

    print(f"\n  total: kept {n_ok:,}  dropped {n_drop:,} ({100*n_drop/max(1,n_ok+n_drop):.1f}%)  {hrs:.2f}h")

    assemble_and_save()
    verify()

    m, s = divmod(int(time.time()-t0), 60); h, m = divmod(m, 60)
    print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",         type=int,           default=0)
    ap.add_argument("--force",         action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    build(smoke=args.smoke, force=args.force, assemble_only=args.assemble_only)