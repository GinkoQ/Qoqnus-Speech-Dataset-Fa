
<div align="center">

<!-- Full-width white banner block (adaptive height by image ratio) -->
<div style="width:100%; background:#ffffff; padding:0; margin:0;">
<img
src="https://cdn-uploads.huggingface.co/production/uploads/62f65fc1c58915315c4eb5f6/Rs2d1hHBo2pgJy2b5MuEp.png"
alt="GinkgoQ banner"
style="width:40%; height:auto; display:block;"
/>
</div>

<!-- Logo below the banner -->
<div style="width:100%; display:flex; justify-content:center; margin-top:18px; margin-bottom:6px;">
<img
src="https://cdn-uploads.huggingface.co/production/uploads/62f65fc1c58915315c4eb5f6/TJvFpMXxCIqJjjG9lWLTW.png"
alt="Qoqnus logo"
style="max-width:280px; width:42%; height:auto; display:block;"
/>
</div>

<h1 style="margin:14px 0 10px; text-align:center; font-size:32px; line-height:1.15; font-weight:800; letter-spacing:-0.02em;">
Qoqnus Persian Speech Corpus
</h1>

**A large-scale, multi-source Persian speech dataset curated for ASR and spoken language research**

![Language](https://img.shields.io/badge/Language-Persian%20%28Farsi%29-blue) ![Hours](https://img.shields.io/badge/Audio-3,006%2B%20hours-green) ![Samples](https://img.shields.io/badge/Samples-2,192,858-orange) ![Sources](https://img.shields.io/badge/Sources-16-purple) ![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-red) ![Version](https://img.shields.io/badge/Version-1.0-lightgrey)

</div>

---

## Overview

**Qoqnus** (ققنوس — the Persian Phoenix) is a consolidated, production-grade Persian speech corpus assembled and released by [GinkgoQ](https://ginkgoq.com). It unifies **16 independent datasets** spanning read speech, conversational audio, podcast recordings, TTS synthesis, and crowd-sourced contributions — forming one of the largest open Persian ASR corpora available.

The corpus is designed for:
- Training and evaluating **Persian ASR models** (CTC, attention, transducer)
- **Speaker-conditioned** and **multi-speaker** speech synthesis
- **Speaking rate** and **prosody** research in Persian
- Benchmarking low-resource and cross-domain speech systems

All audio is stored at **16 kHz mono** in Apache Arrow format, directly loadable via 🤗 Hugging Face `datasets`.

---

## At a Glance

| Metric | Value |
|---|---|
| Total utterances | 2,192,858 |
| Total duration | 3,006h 12m (3006.2 hours) |
| Unique speakers | 3,814 |
| Source datasets | 16 |
| Total splits | 35 |
| Sampling rate | 16,000 Hz |
| Audio encoding | Mono PCM (Arrow) |
| Script | Persian (RTL, Unicode) |
| Release date | March 2026 |

---

## Dataset Composition

| Dataset | Utterances | Duration | Splits | Speakers |
|---|---|---|---|---|
| kiarash | 1,278,935 | 1,392h 45m | 1 | — |
| thomcles | 140,149 | 529h 01m | 1 | — |
| pourmand | 40,933 | 324h 19m | 3 | — |
| srezas | 298,962 | 234h 19m | 8 | — |
| srezas_cv17 | 132,862 | 149h 25m | 2 | — |
| mana_tts | 86,895 | 114h 59m | 1 | — |
| mshojaei | 82,135 | 88h 30m | 1 | — |
| hezarai_cv13 | 48,904 | 56h 31m | 3 | 3,713 |
| asr_fa_v1 | 29,778 | 31h 39m | 3 | — |
| vhdm | 28,892 | 30h 41m | 3 | — |
| srezas_fleurs | 4,336 | 17h 14m | 2 | — |
| m522t | 3,724 | 16h 03m | 1 | — |
| pertts | 7,086 | 10h 30m | 1 | — |
| gpt_informal | 5,874 | 6h 14m | 1 | — |
| seyedali | 2,838 | 3h 17m | 2 | 87 |
| srezas_yazdi | 555 | 0h 36m | 2 | — |

---

## Split Reference

<details>
<summary>Expand full split table (35 splits)</summary>

| Dataset | Split | Utterances | Duration | Speakers |
|---|---|---|---|---|
| kiarash | train | 1,278,935 | 1,392h 45m | — |
| thomcles | train | 140,149 | 529h 01m | — |
| pourmand | train | 32,746 | 259h 26m | — |
| srezas_cv17 | train | 131,862 | 148h 17m | — |
| mana_tts | train | 86,895 | 114h 59m | — |
| srezas | youtube_bpluspodcast | 133,000 | 94h 33m | — |
| mshojaei | train | 82,135 | 88h 30m | — |
| srezas | youtube_rokhpodcast | 29,004 | 33h 57m | — |
| srezas | youtube_Arantik | 27,535 | 33h 51m | — |
| pourmand | val | 4,093 | 32h 30m | — |
| pourmand | test | 4,094 | 32h 22m | — |
| hezarai_cv13 | train | 28,024 | 29h 49m | 146 |
| srezas | youtube_Kouman | 62,855 | 26h 30m | — |
| asr_fa_v1 | train | 23,822 | 25h 15m | — |
| vhdm | train | 23,113 | 24h 34m | — |
| srezas | youtube_MojtabaShakoori | 14,223 | 21h 30m | — |
| srezas | youtube_movarekhpodcast | 23,569 | 18h 22m | — |
| m522t | train | 3,724 | 16h 03m | — |
| hezarai_cv13 | test | 10,440 | 14h 25m | 2,681 |
| srezas_fleurs | train | 3,465 | 13h 32m | — |
| hezarai_cv13 | validation | 10,440 | 12h 17m | 886 |
| pertts | train | 7,086 | 10h 30m | — |
| gpt_informal | train | 5,874 | 6h 14m | — |
| srezas_fleurs | test | 871 | 3h 42m | — |
| asr_fa_v1 | test | 2,978 | 3h 12m | — |
| asr_fa_v1 | validation | 2,978 | 3h 12m | — |
| vhdm | validation | 2,889 | 3h 05m | — |
| vhdm | test | 2,890 | 3h 01m | — |
| srezas | youtube_TPM | 5,192 | 2h 58m | — |
| seyedali | train | 2,270 | 2h 39m | 87 |
| srezas | youtube_FarhangAdyani | 3,584 | 2h 35m | — |
| srezas_cv17 | test | 1,000 | 1h 07m | — |
| seyedali | test | 568 | 0h 38m | 81 |
| srezas_yazdi | train | 505 | 0h 33m | — |
| srezas_yazdi | test | 50 | 0h 03m | — |

</details>

---

## Audio Quality Analysis

All quality metrics are computed on raw audio at 16 kHz using energy-based SNR estimation and RMS normalization.

### Signal-to-Noise Ratio (SNR)

| Range | Count | Share |
|---|---|---|
| <10 dB | 39 | 0.0% |
| 10–20 dB | 120 | 0.0% |
| 20–30 dB | 292,516 | 13.3% |
| 30–40 dB | 699,158 | 31.9% |
| >40 dB | 1,201,025 | 54.8% |

**Mean SNR:** 45.5 dB  |  **Median:** 41.9 dB  |  **Std:** 14.7 dB

### RMS Level Distribution

| Metric | Value |
|---|---|
| Mean RMS | -17.12 dB |
| Median RMS | -17.04 dB |
| Std RMS | 2.64 dB |
| 5th percentile | -21.48 dB |
| 95th percentile | -13.00 dB |

---

## Utterance Duration Distribution

| Bucket | Count | Share |
|---|---|---|
| <2s | 376,826 | 17.2% |
| 2–5s | 1,227,740 | 56.0% |
| 5–10s | 431,425 | 19.7% |
| 10–20s | 73,360 | 3.3% |
| >20s | 83,507 | 3.8% |

**Mean duration:** 4.94s  |  **Median:** 3.45s  |  **Max:** 39.9s

---

## Linguistic Analysis

### Speaking Rate (characters per second)

| Bucket | Count | Share |
|---|---|---|
| <5 c/s | 2,184,182 | 99.6% |
| 5–10 c/s | 8,442 | 0.4% |
| 10–15 c/s | 176 | 0.0% |
| 15–20 c/s | 35 | 0.0% |
| >20 c/s | 23 | 0.0% |

**Mean:** 2.1 c/s  |  **Median:** 2.0 c/s  |  **Std:** 0.9 c/s

> Speaking rate is computed as Persian character count divided by utterance duration, excluding silence padding.

---

## Schema
```python
Features({
"utt_id": Value("string"), # unique utterance identifier
"text": Value("string"), # Persian transcript (Unicode, normalized)
"duration": Value("float32"), # seconds
"n_samples": Value("int64"), # number of audio samples at 16kHz
"speaker_id": Value("string"), # speaker label (dataset-scoped)
"snr_db": Value("float32"), # signal-to-noise ratio in dB
"rms_db": Value("float32"), # RMS loudness in dB
"num_chars": Value("int32"), # Persian character count
"num_words": Value("int32"), # word count
"speaking_rate":Value("float32"), # characters per second
"audio": Audio(16000), # audio array + sampling_rate
})
```

---

## Usage

### Load the full corpus
```python
from datasets import load_from_disk

ds = load_from_disk("qoqnus")
print(ds)
```

### Load a specific split
```python
train = load_from_disk("qoqnus/kiarash_train")
sample = train[0]
print(sample["text"])
# Audio: sample["audio"]["array"], sample["audio"]["sampling_rate"]
```

### Filter by quality
```python
split = load_from_disk("qoqnus/srezas_cv17_train")
clean = split.filter(lambda x: x["snr_db"] > 20 and x["duration"] > 1.0, num_proc=8)
```

### Stream for training (without loading all audio)
```python
from datasets import load_dataset
ds = load_dataset("path/to/qoqnus", split="kiarash_train", streaming=True)
for sample in ds:
audio = sample["audio"]["array"] # loaded on demand
```

---

## Curation Notes

- All audio resampled to **16 kHz mono** using high-quality sinc interpolation
- Transcripts normalized: ZWNJ preserved, Arabic Kaf/Yeh unified to Persian equivalents
- `sampling_rate` column removed from schema (redundant with `Audio(16000)` feature)
- Splits with fewer than 100 utterances retained as-is for benchmark completeness
- Speaker IDs are dataset-scoped — cross-dataset speaker identity is not resolved

---

## Source Datasets

| ID | Source | Domain |
|---|---|---|
| vhdm | VHDM | Read speech |
| seyedali | SeyedAli | Read speech |
| hezarai_cv13 | Common Voice 13 (Hezarai) | Crowd-sourced |
| pourmand | Pourmand | Read speech |
| m522t | M522T | Mixed |
| kiarash | Kiarash | Large-scale mixed |
| mana_tts | Mana TTS | Synthetic / TTS |
| gpt_informal | GPT Informal | Conversational |
| mshojaei | MShojaei | Read speech |
| thomcles | Thomcles | Podcast / long-form |
| srezas | SRezas (multi-source) | YouTube / CV / Fleurs |
| asr_fa_v1 | ASR-FA-v1 | Benchmark |
| pertts | PerTTS | Synthetic / TTS |

---

## Citation

If you use Qoqnus in your research, please cite:
```bibtex
@dataset{qoqnus2025,
title = {Qoqnus: A Large-Scale Multi-Source Persian Speech Corpus},
author = {GinkgoQ Research},
year = {2025},
publisher = {GinkgoQ},
url = {https://ginkgoq.com/qoqnus},
note = {Version 2.0. 2,192,858 utterances, 3006 hours, 16 sources.}
}
```

---

## License

This corpus inherits the licenses of its constituent sources. The unified schema, curation pipeline, and quality annotations are released under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

---

<div align="center">
<sub>Curated with care by <a href="https://ginkgoq.com">GinkgoQ</a> · Built for Persian AI</sub>
<br/>
<img
src="https://cdn-uploads.huggingface.co/production/uploads/62f65fc1c58915315c4eb5f6/Rs2d1hHBo2pgJy2b5MuEp.png"
alt="GinkgoQ banner"
style="width:10%; height:auto; display:block;"
/>
</div>
