from pathlib import Path
from datasets import load_from_disk, DatasetDict
import os
import json

PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("qoqnus")
NUM_PROC = os.cpu_count()

DATASETS = [
    "vhdm", "seyedali", "hezarai_cv13", "pourmand", "m522t",
    "kiarash", "mana_tts", "gpt_informal", "mshojaei",
    "thomcles", "srezas", "asr_fa_v1", "pertts",
]

def num_proc_for(ds):
    return min(NUM_PROC, len(ds))

def normalize(ds):
    if "sampling_rate" in ds.column_names:
        return ds.remove_columns(["sampling_rate"])
    return ds

def is_saved(path):
    return (path / "dataset_info.json").exists()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

keys = []
for name in DATASETS:
    path = PROCESSED_DIR / name
    if not path.exists():
        print(f"[{name}] not found")
        continue
    dd = load_from_disk(str(path), keep_in_memory=False)
    for split, ds in dd.items():
        key = f"{name}_{split}"
        out = OUTPUT_DIR / key
        keys.append(key)
        if is_saved(out):
            print(f"  [{key}] skip")
            continue
        print(f"  [{key}] {len(ds):,} rows → saving")
        ds = normalize(ds)
        ds.save_to_disk(str(out), num_proc=num_proc_for(ds), max_shard_size="2GB")

(OUTPUT_DIR / "dataset_dict.json").write_text(json.dumps({"splits": keys}))
print(f"Done → {OUTPUT_DIR} ({len(keys)} splits)")
