"""
One-time fix for thomcles dataset that was saved with duplicate utt_ids.
Run from v2/: python builders/thomcles_dedup_fix.py

What this does:
  1. Loads processed/thomcles
  2. Finds all duplicate utt_ids
  3. Keeps only the first occurrence of each utt_id
  4. Re-saves in-place (atomic: saves to tmp dir, then replaces)
  5. Verifies the result
"""
import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict

OUT_DIR = Path("processed/thomcles")
TMP_DIR = Path("processed/thomcles_dedup_tmp")
TARGET_SR = 16000

def fix():
    print(f"\n{'='*60}")
    print(f"[thomcles] dedup fix")
    print(f"{'='*60}")

    print(f"  loading {OUT_DIR} ...")
    dd = load_from_disk(str(OUT_DIR))
    ds = dd["train"]
    n_before = len(ds)
    print(f"  rows before : {n_before:,}")

    utt_ids = ds["utt_id"]
    seen = set()
    keep_indices = []
    for i, uid in enumerate(utt_ids):
        if uid not in seen:
            seen.add(uid)
            keep_indices.append(i)

    n_dupes = n_before - len(keep_indices)
    print(f"  duplicates  : {n_dupes:,}")
    print(f"  rows after  : {len(keep_indices):,}")

    if n_dupes == 0:
        print(f"  no duplicates found — nothing to do")
        return

    print(f"  filtering...")
    ds_clean = ds.select(keep_indices)

    print(f"  saving to tmp dir {TMP_DIR} ...")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds_clean}).save_to_disk(str(TMP_DIR))

    print(f"  replacing {OUT_DIR} ...")
    import shutil
    shutil.rmtree(str(OUT_DIR))
    TMP_DIR.rename(OUT_DIR)

    print(f"\n  verifying...")
    dd_v = load_from_disk(str(OUT_DIR))
    ds_v = dd_v["train"]
    assert len(set(ds_v["utt_id"])) == len(ds_v),        "utt_id still not unique"
    assert all(v > 0 for v in ds_v["duration"]),          "duration <= 0"
    assert all(v for v in ds_v["text"]),                  "empty text"
    assert all(v for v in ds_v["speaker_id"]),            "empty speaker_id"
    sample = ds_v.select(range(min(50, len(ds_v))))
    for r in sample:
        assert r["audio"]["sampling_rate"] == TARGET_SR,  "sr != 16000"
        assert r["n_samples"] == len(r["audio"]["array"]), "n_samples mismatch"
        p = r["audio"].get("path")
        assert not (p and str(p).strip()),                f"external path: {p}"
    hrs = sum(ds_v["duration"]) / 3600
    print(f"  ✓ all checks passed")
    print(f"  final: {len(ds_v):,} rows  {hrs:.2f}h")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    fix()