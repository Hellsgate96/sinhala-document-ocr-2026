# Sinhala Document OCR

An end-to-end Optical Character Recognition (OCR) pipeline for **printed (primary)**
and **handwritten (secondary)** Sinhala documents — forms, invoices and ID-style
fields — with support for mixed **Sinhala–English** layouts.

This repository is the implementation scaffold for an MSc research project. The design
follows the approved proposal (Sinhala only; Tamil is out of scope) and is built to be
trained on **Google Colab (GPU)** with data captured from a **phone camera / flatbed
scanner** (no expensive hardware required).

## Pipeline (5 stages)

```
 (1) Acquisition      (2) Preprocessing        (3) Detection         (4) Recognition          (5) Post-processing
+---------------+    +-------------------+    +----------------+    +-------------------+    +---------------------+
| phone camera  | -> | deskew / denoise  | -> | text-line /    | -> | CRNN (CNN+BiLSTM  | -> | dictionary / LM     |
| flatbed scan  |    | binarize / CLAHE  |    | region boxes   |    | + CTC) recognizer |    | correction + field  |
| image files   |    | contrast enhance  |    | (OpenCV / DBNet|    | (TrOCR/PARSeq opt)|    | extraction          |
+---------------+    +-------------------+    +----------------+    +-------------------+    +---------------------+
```

| Stage | Module | Notes |
|-------|--------|-------|
| 1. Acquisition | (external) | Phone camera / flatbed scanner; images placed under `data/`. |
| 2. Preprocessing | `src/preprocessing/preprocess.py` | grayscale, deskew, denoise, binarization (Otsu/adaptive), CLAHE. |
| 3. Detection | `src/detection/text_detection.py` | OpenCV morphological + contour baseline; DBNet/CRAFT adapter slot. |
| 4. Recognition | `src/recognition/` | CRNN (CNN backbone -> BiLSTM -> CTC). Option to fine-tune TrOCR/PARSeq. |
| 5. Post-processing | `src/postprocess/correction.py` | edit-distance dictionary correction; n-gram/LM rescoring stub. |

## Project layout

```
sinhala-document-ocr/
  configs/default.yaml          central configuration
  src/
    charset.py                  Sinhala Unicode charset + CTC encode/decode
    data/                       synthetic generator + PyTorch Dataset
    preprocessing/              document preprocessing
    detection/                  text-line detection (OpenCV baseline + adapter)
    recognition/                CRNN model, train, predict
    evaluation/                 CER / WER / field accuracy / timing
    postprocess/                dictionary + LM correction
    utils/                      seeding, logging, IO, config loader
  notebooks/colab_pipeline.ipynb  end-to-end Colab notebook
  scripts/                      CLI wrappers
  data/  models/  tests/
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

On Windows the Sinhala-capable font **Nirmala UI** (`C:\Windows\Fonts\Nirmala.ttc`)
ships with the OS and is used as the default rendering font.

## How to run each stage

```bash
# 1) Generate synthetic Sinhala text-line data
python scripts/generate_data.py --config configs/default.yaml --num 2000

# 2) Preprocess a folder of documents
python scripts/run_preprocess.py --input data/raw --output data/preprocessed

# 3) Train the CRNN recognizer
python -m src.recognition.train --config configs/default.yaml

# 4) Run inference
python -m src.recognition.predict --checkpoint models/crnn_best.pth \
    --charset models/charset.json --image path/to/line.png

# 5) Evaluate on a test set
python -m src.evaluation.metrics --checkpoint models/crnn_best.pth \
    --charset models/charset.json --labels data/synthetic/test_labels.txt
```

## Datasets

- **Synthetic** Sinhala text lines rendered with Sinhala fonts (Noto Sans Sinhala,
  FM Abhaya, Iskoola Pota, Malithi Web, Nirmala UI) via `src/data/synthetic_generator.py`
  (SynthTIGER-style degradations: rotation, blur, noise, JPEG, shadow).
- **Real** small locally-annotated set of scanned/photographed Sinhala documents.
- **Split by document source**: Train / Val / Test = 70 / 15 / 15.

## Evaluation metrics

Character Error Rate (CER), Word Error Rate (WER), field-level accuracy, and average
**CPU inference time** (see `src/evaluation/metrics.py`).

## Google Colab

See `notebooks/colab_pipeline.ipynb` for an end-to-end run: mount Drive, install deps,
generate synthetic data, train the CRNN, evaluate (CER/WER) and run an inference demo.

## Reference methods (2021+)

TrOCR, PARSeq, Donut, PP-OCRv3, SynthTIGER, DBNet, CRNN.