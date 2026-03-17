"""
Download all Persian ASR datasets using snapshot_download.
Each dataset is saved to RAW_DIR/<name> for later processing.
"""

import os
import time
from pathlib import Path
from huggingface_hub import login, snapshot_download

login(token=os.environ.get("HF_TOKEN"))

RAW_DIR = Path("raw")

# (hf_repo_id, local_name, repo_type)
# All are dataset repos unless noted
DATASETS = [
    ("vhdm/persian-voice-v1",                          "vhdm",          "dataset"),
    ("SeyedAli/Persian-Speech-Dataset",                "seyedali",      "dataset"),
    ("hezarai/common-voice-13-fa",                     "hezarai_cv13",  "dataset"),
    ("pourmand1376/asr-farsi-youtube-chunked-30-seconds", "pourmand",   "dataset"),
    ("m522t/rel_dataset",                              "m522t",         "dataset"),
    ("srezas/farsi_voice_dataset",                     "srezas",        "dataset"),  # all configs
    ("kiarashQ/farsi-asr-unified-cleaned",             "kiarash",       "dataset"),
    ("MahtaFetrat/Mana-TTS",                           "mana_tts",      "dataset"),
    ("MahtaFetrat/GPTInformal-Persian",                "gpt_informal",  "dataset"),
    ("mshojaei77/persian_tts_merged",                  "mshojaei",      "dataset"),
    ("Thomcles/Persian-Farsi-Speech",                  "thomcles",      "dataset"),
    ("SadeghK/datacula-pertts-amir",                   "pertts",        "dataset"),
    ("ASR-fa/ASR_fa_v1",                               "asr_fa_v1",     "dataset"),
]


def download_one(repo_id: str, local_name: str, repo_type: str, max_retries: int = 3) -> bool:
    dest = RAW_DIR / local_name
    if dest.exists() and any(dest.iterdir()):
        print(f"  ✓ already exists, skipping")
        return True

    dest.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(dest),
                ignore_patterns=["*.git*", "*.gitattributes"],
            )
            print(f"  ✓ saved to {dest}")
            return True
        except Exception as e:
            print(f"  attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    return False


def main():
    print("=" * 70)
    print("Persian ASR — snapshot download")
    print(f"Output: {RAW_DIR.absolute()}")
    print("=" * 70)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for idx, (repo_id, local_name, repo_type) in enumerate(DATASETS, 1):
        print(f"\n[{idx}/{len(DATASETS)}] {repo_id}  ->  {local_name}")
        ok = download_one(repo_id, local_name, repo_type)
        if not ok:
            failed.append(repo_id)

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED ({len(failed)}):")
        for r in failed:
            print(f"  - {r}")
    else:
        print("All datasets downloaded successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
