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


## v2 training: diverse corpus + projection detection

The v2 overhaul targets **one general baseline model** that reads arbitrary Sinhala
documents (no per-document fine-tuning):

* **Diverse real-text corpus** — `src/data/corpus_sinhala.txt` (3000+ distinct Sinhala
  lines: everyday/news sentences, names, addresses, verse, religious/formal phrases,
  school text, greetings, mixed Sinhala-English, wide grapheme coverage incl. ්‍ර / ්‍ය
  conjuncts, ඳ ඟ ඬ ෘ ...). Rebuild with `python scripts/build_corpus.py`.
* **Generator v2** — ~65% of lines sampled from the corpus (full sentences + random
  spans), rest word recombinations / numbers / dates; every available Sinhala font
  face (all 6 Nirmala UI/Text faces on Windows), sizes 24–72, dark-colour text,
  plain/gradient/textured light backgrounds, centered vs left layouts.
* **Projection line detection** (default `detection.method: projection`) — background-
  subtracted contrast binarization (drops faint watermarks), border/frame suppression,
  horizontal ink-profile bands, per-band ink extent (handles centered short lines).
  The legacy morphology detector remains available via `detection.method: contours`.
* **Training regime** — 40 epochs, ReduceLROnPlateau on val CER, early stopping
  (patience 8), per-epoch val CER logging.

### Retrain (required after upgrading to v2)

```powershell
# 1) rebuild the corpus (optional; committed file is current)
python scripts/build_corpus.py

# 2) generate 30000 diverse synthetic lines (GPU box; use --num 5000 on CPU-only)
python scripts/generate_data.py --config configs/local.yaml --large

# 3) train the general baseline (40 epochs, early stopping)
python -m src.recognition.train --config configs/local.yaml
```

Or in `notebooks/local_pipeline.ipynb`: set `RUN_GENERATE=True` and
`RUN_BASELINE_TRAIN=True` in Section 4 (defaults: `NUM_SAMPLES=30000`,
`BASELINE_EPOCHS=40`, `CHECKPOINT_MODE="baseline"`, `DETECTION_METHOD="projection"`)
and run Sections 5–7. Section 8 (poem fine-tune) is an **optional experiment** only.

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

### Typical workflow (`notebooks/local_pipeline.ipynb`)

Use **Section 4 — Pipeline control** (`RUN_*` flags) so you do not re-train on every test run.

| Goal | Flags (Section 4) |
|------|-------------------|
| **First full run** | `RUN_GENERATE`, `RUN_BASELINE_TRAIN`, `RUN_FINETUNE`, `RUN_UPLOAD_TEST`, `RUN_REAL_PHOTO` all `True` |
| **Inference / testing only** | All training flags `False`; enable `RUN_UPLOAD_TEST` and/or `RUN_REAL_PHOTO` |
| **Refresh synthetic data** | `RUN_GENERATE=True` only |

Notebook sections: setup → pipeline flags → optional generate → optional baseline train → synthetic eval → optional poem fine-tune → poem CER table → digital upload test → real phone photo → optional debug export.

Checkpoints: `models/crnn_best.pth` (baseline), `models/crnn_finetuned.pth` (poem fine-tune, inference height **64**, greedy decode).

### Real captured photo (notebook Section 11)

1. Run Sections **1–4** (setup + pipeline flags); leave training flags `False` if checkpoints already exist.
2. Set `RUN_REAL_PHOTO=True`, `TEST_MODE="upload"` (tkinter picker) or `file_path` + `REAL_PHOTO_PATH`.
3. Set **`CHECKPOINT_MODE="baseline"`** for new LaTeX/general documents (default `auto` uses `crnn_best.pth` unless poem flags are set).
4. Use **`CHECKPOINT_MODE="finetuned"`** or `USE_POEM_FINETUNE=True` (with `auto`) only for the Kanyawee poem crops/page.
5. Run Section 10/11 cells; each run prints which checkpoint loaded. Debug output under `data/debug/real_capture_<timestamp>/`.
6. Set `COMPARE_TO_POEM_GT=True` only for the same Kanyawee poem page (10 lines) — also selects finetuned weights in `auto` mode.


### Baseline vs poem fine-tuned checkpoint

| Checkpoint | When to use |
|------------|-------------|
| `models/crnn_best.pth` | **Default** for arbitrary uploads, LaTeX PDFs, forms, mixed Sinhala–English documents. |
| `models/crnn_finetuned.pth` | **Kanyawee poem only** (~10 repeated training lines). Using it on general documents often yields garbage OCR. |

In `notebooks/local_pipeline.ipynb` Section 4: `CHECKPOINT_MODE="auto"` (default) loads finetuned weights only when `COMPARE_TO_POEM_GT=True` or `USE_POEM_FINETUNE=True`; otherwise baseline.

## Google Colab

See `notebooks/colab_pipeline.ipynb` for an end-to-end run: mount Drive, install deps,
generate synthetic data, train the CRNN, evaluate (CER/WER) and run an inference demo.

## Reference methods (2021+)

TrOCR, PARSeq, Donut, PP-OCRv3, SynthTIGER, DBNet, CRNN.



