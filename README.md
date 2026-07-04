# Sinhala Document OCR

An end-to-end Optical Character Recognition (OCR) pipeline for **printed (primary)**
and **handwritten (secondary)** Sinhala documents â€” forms, invoices and ID-style
fields â€” with support for mixed **Sinhalaâ€“English** layouts.

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
# optional: pip install -r requirements-optional.txt
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


## Running Locally (Windows + Jupyter)

Run the full baseline pipeline on your laptop without Google Colab.

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Optional:** NVIDIA GPU with CUDA for faster CRNN training

### NVIDIA GPU (local training, e.g. RTX 4060)


**Python 3.13 on Windows:** CUDA wheels are published for `cp313` on the `cu124` index (e.g. `torch-2.6.0+cu124`). The download is about **2.5 GB**; allow time on slower connections. If `pip` only installs `+cpu`, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_cuda_torch.ps1
```

For a dedicated GPU environment when only older Python versions are available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_venv.ps1
```

`requirements.txt` installs a **CPU-only** PyTorch wheel by default. For an **NVIDIA GeForce RTX 4060** (or similar) on Windows, install CUDA-enabled PyTorch **after** the base requirements:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify the GPU is visible:

```powershell
python -c "import torch; print('version:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

The local notebook (`notebooks/local_pipeline.ipynb`) picks `cuda` automatically when available and uses `train.batch_size: 32` from `configs/local.yaml`. `train.device: auto` in that config is overridden by the notebook during interactive runs.

- Sinhala-capable fonts: **Nirmala UI** (`C:\Windows\Fonts\Nirmala.ttc`) ships with Windows

### One-time setup

From the project root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1
# optional virtual environment:
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 -CreateVenv
```

This installs `requirements.txt`, registers the **Sinhala OCR** Jupyter kernel, and creates the `data/` layout.

Core setup does **not** require `editdistance`. On **Windows with Python 3.13**, `editdistance` often has no prebuilt wheel and needs [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile. CER/WER still work via a pure-Python Levenshtein fallback in `src/evaluation/metrics.py`.

- **Default:** run setup without optional native metrics extensions.
- **Optional speedup:** `pip install -r requirements-optional.txt` (includes `rapidfuzz`, which has Python 3.13 wheels, and `editdistance` when a wheel or compiler is available).
- **If you need `editdistance` specifically:** install Visual C++ Build Tools, or use **Python 3.11** where wheels are more common.


### Sinhala display in Jupyter

Training, evaluation, and OCR use Unicode label files and PNG line images. **Missing tofu boxes in plots do not block OCR** — they only affect how matplotlib/IPython render Sinhala in notebook titles and prints.

For readable Sinhala in previews:

1. **Windows:** `setup_matplotlib_sinhala()` registers **Nirmala UI** (`C:\Windows\Fonts\Nirmala.ttc`) automatically (Latin + Sinhala).
2. **Linux / Colab:** install `fonts-noto-core` or let `notebooks/colab_pipeline.ipynb` download Noto Sans Sinhala into `fonts/`.
3. **Fallback:** `powershell -ExecutionPolicy Bypass -File scripts/download_fonts.ps1` saves `fonts/NotoSansSinhala-Regular.ttf`.

Notebooks call `from src.utils.display import setup_matplotlib_sinhala` after font detection. `scripts/setup_local.ps1` prints the resolved font path.

Quick check:

```powershell
python -c "from src.utils.display import setup_matplotlib_sinhala; print(setup_matplotlib_sinhala())"
```

### Start the local notebook

```powershell
cd C:\path\to\sinhala-document-ocr
jupyter notebook notebooks/local_pipeline.ipynb
```

Work through the cells in order: synthetic data generation, training, evaluation, and upload OCR.

### Data layout (local)

| Path | Purpose |
|------|---------|
| `data/synthetic/` | Generated training lines (`images/`, `train_labels.txt`, …) — **gitignored** after generation |
| `data/uploads/` | Place test photos/scans for Section 8 |
| `data/real/images/` + `data/real/labels/` | Future real annotated documents (your local collection) |
| `data/debug/` | Optional inference debug dumps from the notebook |
| `models/` | `crnn_best.pth`, `crnn_last.pth`, `charset.json` (checkpoints gitignored) |

The first local run **generates synthetic samples** into `data/synthetic/` (default **5000** lines via `configs/local.yaml`).

### Train without Jupyter

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_local_train.ps1
# or
python scripts/run_local_train.py
```

Uses `configs/local.yaml` (15 epochs, batch size 16 by default).


## Google Colab

See `notebooks/colab_pipeline.ipynb` for an end-to-end run: mount Drive, install deps,
generate synthetic data, train the CRNN, evaluate (CER/WER) and run an inference demo.

## Reference methods (2021+)

TrOCR, PARSeq, Donut, PP-OCRv3, SynthTIGER, DBNet, CRNN.

