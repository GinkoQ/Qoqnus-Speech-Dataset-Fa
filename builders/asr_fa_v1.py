"""
Standalone builder for asr_fa_v1 dataset.
Run from v2/: python builders/asr_fa_v1.py

Source: raw/asr_fa_v1/data/*.parquet
  columns : ['audio', 'corrected_sentence']
  audio   : AudioDecoder (_hf_encoded bytes, mp3)
  text    : corrected_sentence
  speaker : no speaker col -> gen:asr_fa_v1_spk0000
  splits  : test, train

Smoke test: python builders/asr_fa_v1.py --smoke 5
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import io, re, json, time, unicodedata, argparse
import numpy as np
import soundfile as sf
import resampy
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, load_from_disk, DatasetDict
from tqdm import tqdm

SOURCE      = "asr_fa_v1"
RAW_DIR     = Path("raw/asr_fa_v1")
OUT_DIR     = Path("processed/asr_fa_v1")
SHARDS_DIR  = OUT_DIR / "_shards"
LOG_DIR     = Path("logs")
TEXT_COL    = "corrected_sentence"
SPEAKER     = "gen:asr_fa_v1_spk0000"
TARGET_SR   = 16000
MIN_DUR     = 0.3
MAX_DUR     = 30.0
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
        arr_raw, sr = decode_audio(row["audio"])
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

def process_split(split, files, all_drops, smoke=0):
    split_shards = SHARDS_DIR / split
    split_shards.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("parquet", data_files=files, split="train")
    if smoke: ds = ds.select(range(min(smoke, len(ds))))
    n_in = len(ds)
    print(f"\n  [{split}] {n_in:,} rows  ({len(files)} file(s))")

    CHUNK = NUM_WORKERS * 4
    batch, n_ok, n_drop, hrs, shard_idx = [], 0, 0, 0.0, 0
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
    drop_pct = 100*n_drop/max(1,n_in)
    print(f"    → kept {n_ok:,}  dropped {n_drop:,} ({drop_pct:.1f}%)  {hrs:.2f}h")
    if n_drop:
        rc = {}
        for d in all_drops[-n_drop:]: rc[d["reason"].split(":")[0].strip()] = rc.get(d["reason"].split(":")[0].strip(),0)+1
        for r,c in sorted(rc.items(), key=lambda x:-x[1]): print(f"      {r:<30} {c:>6,}")
    return n_ok

def assemble_split(split):
    shards = sorted((SHARDS_DIR/split).glob("shard_*.parquet"))
    return load_dataset("parquet", data_files=[str(p) for p in shards], split="train", num_proc=1)

def build(smoke=0):
    t0 = time.time()
    print(f"\n{'='*60}\n[{SOURCE}] dataset builder{'  [SMOKE]' if smoke else ''}\n{'='*60}")
    data_dir = RAW_DIR/"data" if (RAW_DIR/"data").exists() else RAW_DIR
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files: raise FileNotFoundError(f"no parquet files under {data_dir}")
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    split_files: dict[str,list] = {}
    for f in parquet_files: split_files.setdefault(f.stem.split("-")[0],[]).append(str(f))
    print(f"  splits: {sorted(split_files.keys())}")
    all_drops: list[dict] = []
    for split, files in split_files.items():
        process_split(split, files, all_drops, smoke=smoke)
    if all_drops:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR/f"{SOURCE}_dropped.jsonl","w",encoding="utf-8") as f:
            for d in all_drops: f.write(json.dumps(d,ensure_ascii=False)+"\n")
    print(f"\n  assembling DatasetDict...")
    dd = DatasetDict({split: assemble_split(split) for split in split_files})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(OUT_DIR))
    for split, ds in dd.items():
        print(f"    {split:<20} {len(ds):>8,} rows  {sum(ds['duration'])/3600:.2f}h")
    import shutil; shutil.rmtree(SHARDS_DIR)
    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    all_ids = []
    for split, ds in dd_v.items():
        assert all(r["audio"]["sampling_rate"]==TARGET_SR for r in ds)
        assert all(v>0 for v in ds["duration"])
        assert all(v for v in ds["text"])
        assert all(v for v in ds["speaker_id"])
        all_ids.extend(ds["utt_id"])
        for i in range(min(20,len(ds))):
            r=ds[i]; assert r["n_samples"]==len(r["audio"]["array"])
            p=r["audio"].get("path"); assert not(p and str(p).strip())
    assert len(set(all_ids))==len(all_ids), "utt_id not unique"
    print(f"  ✓ all checks passed")
    m,s=divmod(int(time.time()-t0),60); h,m=divmod(m,60)
    print(f"  done in {h:02d}:{m:02d}:{s:02d}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    build(smoke=args.smoke)
