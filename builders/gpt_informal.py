"""
Standalone builder for gpt_informal dataset.
Run from v2/: python builders/gpt_informal.py

Source: raw/gpt_informal/dataset/*.parquet  (flat files)
  columns : ['file name', 'transcript', 'subject', 'audio', 'samplerate', 'duration']
  audio   : list of float32 samples (pre-decoded)
  sr      : samplerate column (int, e.g. 44100)
  text    : transcript
  speaker : no speaker col -> gen:gpt_informal_spk0000
  splits  : all files -> train

Smoke test: python builders/gpt_informal.py --smoke 5
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import re, json, time, unicodedata, argparse
import numpy as np
import resampy
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

SOURCE      = "gpt_informal"
RAW_DIR     = Path("raw/gpt_informal/dataset")
OUT_DIR     = Path("processed/gpt_informal")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
TEXT_COL    = "transcript"
SR_COL      = "samplerate"
SPEAKER     = "gen:gpt_informal_spk0000"
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

def normalize(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", text).strip()

def decode_audio(row):
    audio = row["audio"]
    if not isinstance(audio, (list, tuple)) or len(audio) == 0:
        raise ValueError("empty or invalid audio list")
    arr = np.asarray(audio, dtype=np.float32)
    sr_raw = row.get(SR_COL)
    sr = int(float(sr_raw)) if sr_raw is not None else TARGET_SR
    return arr, sr

def process_audio(arr, sr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1: arr = arr.mean(axis=-1).astype(np.float32)
    if sr != TARGET_SR:
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
        arr_raw, sr = decode_audio(row)
        arr = process_audio(arr_raw, sr)
    except Exception as e:
        return None, {"utt_id": utt_id, "reason": f"audio: {e}"}
    text = normalize(str(row.get(TEXT_COL) or ""))
    if not text:
        return None, {"utt_id": utt_id, "reason": "empty text"}
    ns = len(arr); dur = ns / TARGET_SR; nw = len(text.split())
    return {
        "utt_id": utt_id, "audio": {"array": arr.tolist(), "sampling_rate": TARGET_SR},
        "text": text, "duration": float(round(dur,4)), "n_samples": ns,
        "speaker_id": SPEAKER, "snr_db": float(snr_db(arr)),
        "rms_db": float(rms_db(arr)), "num_chars": len(text),
        "num_words": nw, "speaking_rate": float(round(nw/dur,4)),
    }, None

def flush_batch(batch, shard_path):
    cols = {k: [] for k in SCHEMA.names}
    for r in batch:
        for k in SCHEMA.names: cols[k].append(r[k])
    audio_type = pa.struct([pa.field("array", pa.list_(pa.float32())), pa.field("sampling_rate", pa.int32())])
    arrays = []
    for field in SCHEMA:
        if field.name == "audio":
            arrays.append(pa.array(cols["audio"], type=audio_type))
        else:
            arrays.append(pa.array(cols[field.name], type=field.type))
    table = pa.table({f.name: arrays[i] for i,f in enumerate(SCHEMA)}, schema=SCHEMA)
    pq.write_table(table, str(shard_path), compression="snappy", write_statistics=False)

def build(smoke=0):
    t0 = time.time()
    print(f"\n{'='*60}\n[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}\n{'='*60}")
    if not RAW_DIR.exists(): raise FileNotFoundError(f"not found: {RAW_DIR}")
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(RAW_DIR.glob("**/*.parquet"))
    if not parquet_files: raise FileNotFoundError(f"no parquet files under {RAW_DIR}")
    print(f"  files: {len(parquet_files)}")
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    split = "train"
    split_shards = SHARDS_DIR / split
    split_shards.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
    if smoke: ds = ds.select(range(min(smoke, len(ds))))
    n_in = len(ds)
    print(f"  total rows: {n_in:,}")

    CHUNK = NUM_WORKERS * 4
    all_drops, batch, n_ok, n_drop, hrs, shard_idx = [], [], 0, 0, 0.0, 0
    bar = tqdm(total=n_in, desc=f"    {split}", unit="row", dynamic_ncols=True, smoothing=0.05)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for chunk_start in range(0, n_in, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n_in)
            chunk = [dict(ds[i]) for i in range(chunk_start, chunk_end)]
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
                        flush_batch(batch, split_shards/f"shard_{shard_idx:06d}.parquet")
                        shard_idx += 1; batch.clear()
                bar.update(1)
                bar.set_postfix({"ok":n_ok,"drop":n_drop,"drop%":f"{100*n_drop/max(1,n_ok+n_drop):.1f}","hrs":f"{hrs:.2f}"}, refresh=False)

    bar.close()
    if batch:
        flush_batch(batch, split_shards/f"shard_{shard_idx:06d}.parquet"); batch.clear()
    print(f"    → kept {n_ok:,}  dropped {n_drop:,} ({100*n_drop/max(1,n_in):.1f}%)  {hrs:.2f}h")

    if all_drops:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR/f"{SOURCE}_dropped.jsonl","w",encoding="utf-8") as f:
            for d in all_drops: f.write(json.dumps(d,ensure_ascii=False)+"\n")

    shards = sorted(split_shards.glob("shard_*.parquet"))
    ds_out = load_dataset("parquet", data_files=[str(p) for p in shards], split="train", num_proc=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd = DatasetDict({"train": ds_out})
    dd.save_to_disk(str(OUT_DIR))
    print(f"    train: {len(ds_out):,} rows  {sum(ds_out['duration'])/3600:.2f}h")
    import shutil; shutil.rmtree(SHARDS_DIR)

    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    ds_v = dd_v["train"]
    assert all(r["audio"]["sampling_rate"]==TARGET_SR for r in ds_v)
    assert all(v>0 for v in ds_v["duration"])
    assert all(v for v in ds_v["text"])
    assert len(set(ds_v["utt_id"]))==len(ds_v), "utt_id not unique"
    for i in range(min(20,len(ds_v))):
        r=ds_v[i]; assert r["n_samples"]==len(r["audio"]["array"])
        p=r["audio"].get("path"); assert not(p and str(p).strip())
    print(f"  ✓ all checks passed")
    m,s=divmod(int(time.time()-t0),60); h,m=divmod(m,60)
    print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    build(smoke=args.smoke)
