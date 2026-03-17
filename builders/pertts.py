"""
Build pertts parquet from downloaded zip.
Run from v2/: python builders/pertts.py

Produces a SELF-CONTAINED train.parquet where audio bytes are embedded inline.
No external files needed after build.
"""
import io
import re
import builtins
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR  = Path("raw")
SRC_DIR  = RAW_DIR / "pertts"
ZIP_PATH = SRC_DIR / "pertts-speech-database-rokh-ljspeech.zip"
EXTRACT  = SRC_DIR / "extracted"
OUT_FILE = SRC_DIR / "train.parquet"
INNER    = "pertts-speech-database-rokh-ljspeech"


def extract():
    if EXTRACT.exists():
        return
    print("  Extracting zip...")
    with ZipFile(ZIP_PATH) as z:
        z.extractall(str(EXTRACT))
    print("  Done.")


def parse_metadata() -> pd.DataFrame:
    meta = EXTRACT / INNER / "metadata.csv"
    id_pat = re.compile(r"[A-Za-z0-9_-]{6,}\|")
    lines = []
    with open(meta, encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            matches = list(id_pat.finditer(line))
            if len(matches) <= 1:
                lines.append(line)
                continue
            starts = [m.start() for m in matches]
            if starts[0] != 0:
                starts = [0] + starts
            for a, b in builtins.zip(starts, starts[1:] + [len(line)]):
                seg = line[a:b]
                if seg:
                    lines.append(seg)
    buf = io.StringIO("\n".join(lines))
    return pd.read_csv(buf, sep="|", header=None, names=["id", "text"],
                       engine="python", encoding="utf-8")


def build():
    if OUT_FILE.exists():
        print(f"  Already built: {OUT_FILE}")
        return

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}\nRun download.py first.")

    extract()
    df = parse_metadata()
    wav_dir = EXTRACT / INNER / "wav"

    # Read audio bytes directly into memory - fully self-contained
    audio_bytes_list = []
    texts = []
    missing = 0

    for _, row in df.iterrows():
        wav = wav_dir / f"{row['id']}.wav"
        if not wav.exists():
            missing += 1
            continue
        audio_bytes_list.append(wav.read_bytes())
        texts.append(str(row["text"]))

    print(f"  Loaded {len(audio_bytes_list)} wav files ({missing} missing)")

    # Build parquet with audio stored as struct {bytes, path}
    # This matches the HF Audio feature format that datasets expects
    audio_structs = [{"bytes": b, "path": None} for b in audio_bytes_list]

    table = pa.table({
        "audio": pa.array(audio_structs, type=pa.struct([
            ("bytes", pa.binary()),
            ("path", pa.string()),
        ])),
        "sentence": pa.array(texts, type=pa.string()),
    })

    pq.write_table(table, str(OUT_FILE))
    print(f"  Saved self-contained: {OUT_FILE}  ({len(texts)} rows)")

    # Cleanup extracted folder
    import shutil
    shutil.rmtree(str(EXTRACT))
    print(f"  Removed extracted folder.")


if __name__ == "__main__":
    print("=" * 60)
    print("Building pertts dataset")
    print("=" * 60)
    build()
    print("=" * 60)
