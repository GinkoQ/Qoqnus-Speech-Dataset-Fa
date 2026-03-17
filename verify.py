"""
Verify downloaded datasets - loads ONE row per dataset and inspects audio structure.
Run from v2/: python verify.py
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path("raw")

# (source_name, path_relative_to_raw, text_col)
SOURCES = [
    ("vhdm",         "vhdm",                          "corrected_sentence"),
    ("seyedali",     "seyedali",                      "transcript"),
    ("hezarai_cv13", "hezarai_cv13",                  "sentence"),
    ("pourmand",     "pourmand",                      "transcription"),
    ("m522t",        "m522t",                         "sentence"),
    ("srezas",       "srezas/common_voice_17",        "sentence"),
    ("kiarash",      "kiarash",                       "text"),
    ("mana_tts",     "mana_tts/dataset",              "transcript"),
    ("gpt_informal", "gpt_informal/dataset",          "transcript"),
    ("mshojaei",     "mshojaei",                      "sentence"),
    ("thomcles",     "thomcles",                      "sentence"),
    ("pertts",       "pertts",                        "sentence"),
    ("asr_fa_v1",    "asr_fa_v1",                     "corrected_sentence"),
]

AUDIO_FIELDS = ["audio", "speech", "waveform", "wav"]


def find_parquet(path: Path):
    """Find first parquet file under path, checking data/ subdir too."""
    for p in [path, path / "data"]:
        files = sorted(p.glob("*.parquet"))
        if files:
            return files[0]
    files = sorted(path.glob("**/*.parquet"))
    return files[0] if files else None


def find_arrow(path: Path):
    """Find first arrow file under path."""
    files = sorted(path.glob("**/*.arrow"))
    return files[0] if files else None


def load_one_row(path: Path):
    """Load exactly one row from a dataset path."""
    parquet = find_parquet(path)
    if parquet:
        ds = load_dataset("parquet", data_files=str(parquet), split="train", streaming=True)
        return next(iter(ds)), parquet.name

    arrow = find_arrow(path)
    if arrow:
        ds = load_dataset("arrow", data_files=str(arrow), split="train", streaming=True)
        return next(iter(ds)), arrow.name

    return None, None


def inspect_audio(audio_obj, row):
    """Inspect audio object and return human-readable info."""
    t = type(audio_obj)
    type_name = t.__name__
    module = t.__module__

    lines = [f"  type        : {module}.{type_name}"]

    # AudioDecoder (torchcodec)
    if "AudioDecoder" in type_name:
        lines.append(f"  kind        : AudioDecoder (torchcodec)")
        if hasattr(audio_obj, "_hf_encoded"):
            enc = audio_obj._hf_encoded
            keys = list(enc.keys()) if hasattr(enc, "keys") else str(enc)
            lines.append(f"  _hf_encoded : keys={keys}")
            b = enc.get("bytes") if hasattr(enc, "get") else None
            p = enc.get("path") if hasattr(enc, "get") else None
            lines.append(f"    bytes     : {f'{len(b)} bytes' if b else 'None'}")
            lines.append(f"    path      : {p}")
        return lines

    # dict-like
    if hasattr(audio_obj, "keys"):
        keys = list(audio_obj.keys())
        lines.append(f"  kind        : dict-like  keys={keys}")
        arr = audio_obj.get("array") if hasattr(audio_obj, "get") else None
        b   = audio_obj.get("bytes") if hasattr(audio_obj, "get") else None
        p   = audio_obj.get("path")  if hasattr(audio_obj, "get") else None
        sr  = audio_obj.get("sampling_rate") if hasattr(audio_obj, "get") else None
        if arr is not None:
            lines.append(f"    array     : {type(arr).__name__} len={len(arr) if hasattr(arr,'__len__') else '?'}")
        if b is not None:
            lines.append(f"    bytes     : {len(b)} bytes")
        if p is not None:
            lines.append(f"    path      : {p}")
        if sr is not None:
            lines.append(f"    sr        : {sr}")
        return lines

    # raw bytes
    if isinstance(audio_obj, bytes):
        lines.append(f"  kind        : raw bytes  len={len(audio_obj)}")
        return lines

    # raw list
    if isinstance(audio_obj, list):
        sr = row.get("samplerate") or row.get("sampling_rate")
        lines.append(f"  kind        : list of floats  len={len(audio_obj)}  sr_col={sr}")
        return lines

    # numpy
    try:
        import numpy as np
        if isinstance(audio_obj, np.ndarray):
            lines.append(f"  kind        : numpy array  shape={audio_obj.shape}  dtype={audio_obj.dtype}")
            return lines
    except ImportError:
        pass

    lines.append(f"  kind        : unknown")
    return lines


def main():
    print("=" * 70)
    print("Dataset Verification — one row per source")
    print("=" * 70)

    for source, rel_path, text_col in SOURCES:
        path = RAW_DIR / rel_path
        print(f"\n[{source}]")

        if not path.exists():
            print(f"  ❌ not found: {path}")
            continue

        row, fname = load_one_row(path)
        if row is None:
            print(f"  ❌ no parquet/arrow files found under {path}")
            continue

        print(f"  file        : {fname}")
        print(f"  columns     : {list(row.keys())}")

        # Find audio field
        audio_field = None
        audio_obj = None
        for f in AUDIO_FIELDS:
            if f in row and row[f] is not None:
                audio_field = f
                audio_obj = row[f]
                break

        if audio_obj is None:
            print(f"  ❌ no audio field found")
        else:
            print(f"  audio field : '{audio_field}'")
            for line in inspect_audio(audio_obj, row):
                print(line)

        # Text field
        text_val = row.get(text_col, "<not found>")
        if text_val and text_val != "<not found>":
            preview = str(text_val)[:60]
            print(f"  text ({text_col}): {preview!r}")
        else:
            print(f"  ❌ text col '{text_col}' not found")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
