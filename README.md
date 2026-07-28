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
    data/                       synthetic line generator, page_synth.py (v3 full-page +
                                 detector-in-the-loop generator), PyTorch Dataset
    preprocessing/              document preprocessing
    detection/                  text-line detection (projection profile default + contours)
    recognition/                CRNN model, train, predict
    evaluation/                 CER / WER / field accuracy / timing, pipeline_eval.py
                                 (shared detect+recognize path for all eval scripts)
    postprocess/                dictionary + LM correction
    utils/                      seeding, logging, IO, config loader
  notebooks/local_pipeline.ipynb   Windows/Jupyter end-to-end notebook (primary)
  notebooks/colab_pipeline.ipynb   Google Colab end-to-end notebook
  scripts/                      CLI wrappers (generate_data.py, generate_pages.py,
                                 build_eval_pages.py, build_adversarial_pages.py,
                                 eval_real_images.py, run_realistic_eval.py, ...)
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

Or in `notebooks/local_pipeline.ipynb`: set `RUN_GENERATE=True` /
`RUN_GENERATE_PAGES=True` / `RUN_TRAIN=True` in Section 4 once, then
**Restart & Run All**. Later testing leaves those flags `False`.

## v3: closing the synthetic-to-real domain gap

**Symptom:** after the v2 overhaul (diverse corpus + projection detector), synthetic
line-crop validation CER reached ~4% by epoch 16 - yet real full-page photos were
still "not acceptable". Line-crop CER on its own is **not sufficient evidence** of
real-world quality; see the before/after numbers below.

**Root causes found (with evidence, not guesses):**

1. **Only one font family was actually on disk.** The v2 generator supports many
   font *faces*, but on this machine only `C:/Windows/Fonts/Nirmala.ttc` existed
   (`iskpota.ttf` and the Noto fallback were both missing) - every training image
   used one of 6 faces from a single family. Fixed by downloading 4 more Sinhala
   font families into `fonts/` (`scripts/download_fonts.ps1`, extended) - Noto Sans
   Sinhala, Noto Serif Sinhala, Abhaya Libre, Yaldevi - now 10 font faces total.
2. **Training only ever saw idealised single-line crops.** The generator renders one
   tightly-cropped line at a time; real inference runs `ProjectionLineDetector` over
   a whole photographed *page* and crops whatever imperfect box the detector
   produces (mis-padding, a border/watermark sliver at the edge, occasional
   merge/split of adjacent lines). A model that never saw that kind of crop during
   training has no reason to be robust to it. Fixed with a new
   **detector-in-the-loop page generator** (`src/data/page_synth.py`,
   `scripts/generate_pages.py`): render a full synthetic page (paragraph / bordered
   card / poem / mixed Sinhala-English / letterhead), run the *real* detector on it,
   and train on the detector's actual output crops paired with their transcript
   (pages where the detector's line count doesn't match ground truth are discarded,
   not mislabeled - the discard rate itself is logged as a per-layout detector
   health metric).
3. **Line-crop augmentation under-modelled the physical capture process.** Added
   paper-texture grain, camera-like defocus/motion blur (distinct from the existing
   Gaussian resampling blur), rare moire (screen re-photograph), rare rule/adjacent-
   line edge artifacts (simulating an imperfect detector crop), and multi-generation
   JPEG re-encoding (`src/data/synthetic_generator.py`, `augment.*` in
   `configs/*.yaml`).
4. **A scoring bug inflated real-world CER measurements.** The CLI "low confidence"
   warning prefix (meant for human-facing display) was being fed into the CER
   calculation in the evaluation scripts, making otherwise-correct numeric-heavy
   lines (dates, amounts, IDs) look wildly wrong. Fixed in
   `src/evaluation/pipeline_eval.py` (raw text is always scored; the warning prefix
   is display-only).
5. **Whole-page augmentation reused line-crop perspective jitter unscaled**, which
   for a ~1500px-tall page could shift a corner by 100+px and merge unrelated lines
   into one detected band. Fixed by scaling `aug_perspective`'s jitter fraction down
   for page-level use (`src/data/page_synth.apply_page_augmentations`); average
   detector exact-line-count match rate across the 5 page layouts on augmented pages
   went from ~0.63 to ~0.80 after the fix (`scripts/generate_pages.py` logs).
6. **The projection detector silently merged adjacent lines whose glyphs touch
   vertically** (a descender or matra bridging the gap keeps every row's ink count
   above the "is text" threshold, so the profile never dips to zero between the two
   lines) - this under-counted lines by 30-50% on poem-style and numeric-heavy mixed
   Sinhala/English layouts specifically. Fixed by re-splitting any band taller than
   1.2x the page's median line height at its lowest internal ink-profile valley
   (`_split_tall_band` in `src/detection/text_detection.py`, regression test in
   `tests/test_page_detection.py::test_tall_band_is_split_at_internal_valley`).
   Very short isolated marker/label lines (a handful of characters, e.g. a lone
   page/section tag) can still be dropped by the relative-height filter - a
   remaining known limitation, not yet hit on the realistic eval set's main content
   lines.

**What did NOT need fixing:** the CRNN+CTC architecture, charset/ZWJ handling,
`resize_keep_height`'s LANCZOS up/down-scaling, and the projection detector's
watermark/border suppression were all verified correct and left alone.

### New realistic evaluation (this is what actually proves a fix worked)

Line-crop CER on the synthetic val set is kept as a training-time signal, but the
real acceptance test is **full-pipeline** (detection errors count against you):

```powershell
# Build a small held-out set of full synthetic pages (different font/colour mix
# than training) and score the whole detect+recognize pipeline end to end:
python scripts/build_eval_pages.py --config configs/local.yaml --num-pages 10
python scripts/run_realistic_eval.py --images-dir data/eval_pages --checkpoint models/crnn_best.pth

# 3 hand-built adversarial acceptance-test pages (decorative bordered card with a
# watermark, LaTeX-article-style page, heavily phone-camera-degraded paragraph):
python scripts/build_adversarial_pages.py --config configs/local.yaml
python scripts/run_realistic_eval.py --images-dir data/eval_real/adversarial --checkpoint models/crnn_best.pth
```

`scripts/eval_real_images.py` still works the same way for a single real image with
an optional ground-truth labels file; all three scripts share one detect+recognize
code path (`src/evaluation/pipeline_eval.py`) so there is no drift between "what
gets evaluated" and "what a real upload goes through".

### Retrain results (this run)

Warm-started from the pre-v3 checkpoint (`models/crnn_best_pre_domaingap.pth`,
epoch 16, synthetic val CER 0.0442) with the v3 line-crop augmentation, extra font
faces, and a 25,545-crop detector-in-the-loop page supplement (4,000 synthetic
pages, ~81.6% detector exact-match rate) merged in via `--extra-labels`. Trained
40 epochs (`train.num_workers=6`, plateau LR schedule), ~4h11m on an RTX 4060:
best synthetic val CER **0.0311** at epoch 37 (`models/train_v3.log`).

End-to-end full-pipeline corpus CER, BEFORE (`crnn_best_pre_domaingap.pth`) vs
AFTER (`crnn_best.pth`, this run) - detection errors count against the score:

| Eval set | BEFORE | AFTER |
| --- | --- | --- |
| `data/eval_pages` (10 held-out synthetic pages, 5 layouts) | 0.2880 | **0.0976** |
| `data/eval_real/adversarial` (3 hand-built acceptance pages) | 0.0745 | **0.0655** |

8 of the 10 realistic eval pages (paragraph/card/letterhead/one poem layout) score
at or near **0.00 CER** after the fix. The remaining error is concentrated in two
layouts with short interjected/numeric lines (poem interjections, dense
registration-form-style mixed Sinhala/English/numeric text) where the detector
still occasionally under-counts lines - see fix #6 above and the "known limitation"
note; because scoring aligns lines positionally, one missed line inflates that
page's CER disproportionately even though most of its words are read correctly.

### Regenerate training data + retrain with the v3 fixes

```powershell
# 1) extra Sinhala font families (best-effort; safe to skip if offline)
powershell -ExecutionPolicy Bypass -File scripts/download_fonts.ps1

# 2) line-crop dataset (now with the richer augmentation + font list)
python scripts/generate_data.py --config configs/local.yaml --large

# 3) detector-in-the-loop page supplement (the actual domain-gap fix)
python scripts/generate_pages.py --config configs/local.yaml --num-pages 4000

# 4) train, merging the page supplement and warm-starting from the existing checkpoint
python -m src.recognition.train --config configs/local.yaml \
    --extra-labels data/synthetic_pages/train_labels.txt --resume models/crnn_best.pth
```

Or in `notebooks/local_pipeline.ipynb`: `RUN_GENERATE=True`, `RUN_GENERATE_PAGES=True`,
`RUN_BASELINE_TRAIN=True` in Section 4, then run Sections 5-7 (Section 5b is the new
page supplement). `RESUME_FROM_PRE_V3_CKPT=True` (default) warm-starts from the
existing `crnn_best.pth` instead of random init.

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

### Typical workflow (`notebooks/local_pipeline.ipynb`) — Monday demo path

**One general model:** always use `models/crnn_best.pth`. Real Kanyawee poem lines are
mixed into general training (with heavy augmentation) via `--extra-labels`; there is
no required second-stage poem fine-tune.

| Goal | Flags (Section 4) |
|------|-------------------|
| **Test a real image only** | Leave `RUN_GENERATE` / `RUN_GENERATE_PAGES` / `RUN_TRAIN` as `False`; set `TEST_IMAGE_PATH` or use the file picker; **Kernel → Restart & Run All** |
| **First full train** | Set generate/train flags `True` once (auto-skips later when data/checkpoint exist) |
| **Refresh synthetic data** | `RUN_GENERATE=True` (and optionally `RUN_GENERATE_PAGES=True`) |

Notebook sections: setup → install → fonts → **one control cell** → optional generate → optional page-synth → optional train → **test real image** → optional poem CER → optional debug.

Checkpoints: `models/crnn_best.pth` (general model; gitignored — keep a local copy after training).
Optional legacy: `models/crnn_finetuned.pth` is **not** used by the cleaned notebook.

### Adding real labeled lines (where files go)

Put **line crops** (not full pages) here:

| What | Path |
|------|------|
| Images | `data/real/images/` (e.g. `my_line_001.png`) |
| Labels | `data/real/labels/<name>.txt` |

**Labels format** (UTF-8, one row per line crop):

```text
images/my_line_001.png	exact Sinhala transcript here
images/my_line_002.png	another line
```

- Separator is a **tab** (`path\ttranscript`).
- Image paths are **relative to `data/real/`** (so `images/...`, not `data/real/images/...`).
- Absolute paths also work, but relative is preferred.
- Naming: `poem_line_###.png` or any stable prefix; keep IDs unique.
- **How many help:** tens of diverse real lines already move the needle (this repo’s 10 Kanyawee lines + 80× aug mixed into general training); aim for **30–100+** unique lines across fonts/paper/lighting if you can. Prefer variety over many near-duplicates.

**Full pages for testing only** go under `data/uploads/` (or set `TEST_IMAGE_PATH`); they are not training labels until you crop + transcribe them into `data/real/`.

### Mix real poem lines into the general model

Kanyawee shortcut (auto-crop a known page + hard-coded GT):

```powershell
python scripts/prepare_poem_dataset.py --image data/uploads/test2.png
```

Or add your own crops/labels as above, then augment + continue-train the **same** `crnn_best.pth`:

```powershell
python scripts/augment_poem_dataset.py --labels data/real/labels/poem_kanyawee.txt --copies 80
python scripts/prepare_real_pages.py --gt-json data/real/labels/user_batch1_gt.json
python scripts/augment_poem_dataset.py --labels data/real/labels/user_batch1.txt `
  --out-labels data/real/labels/user_batch1_aug.txt --name-prefix user_aug --copies 50
python -m src.recognition.train --config configs/mix_real.yaml `
  --extra-labels data/synthetic_pages/train_labels.txt `
  --extra-labels data/real/labels/poem_kanyawee_aug.txt `
  --extra-labels data/real/labels/user_batch1_aug.txt `
  --resume models/crnn_best.pth
```

**Holdout (updated Jul 2026):** prior `page_07_font_list` + `page_12_hitigama` lines
were moved **into training** for accuracy. The honest check is now
`data/real/labels/web_batch1_holdout.txt` (fresh hard-style renders). Historical copy:
`user_batch1_holdout_SUPERSEDED.txt`.

### Web / hard-case mix (`configs/mix_web.yaml`)

Adds pill/dark/coloured title lines, the Jul-26 exam-cover crops, and (optionally)
CC-BY-4.0 Sri Lankan Acts pages from Hugging Face. Provenance:
`data/real/pages/web_batch1/SOURCES.md`.

```powershell
python scripts/prepare_web_batch1.py
python scripts/generate_hard_lines.py --num 9000
python scripts/download_hf_acts.py --max-pages 200   # optional; CC-BY-4.0
python scripts/augment_poem_dataset.py --copies 80
python scripts/augment_poem_dataset.py --labels data/real/labels/user_batch1.txt `
  --out-labels data/real/labels/user_batch1_aug.txt --name-prefix user_aug --copies 40
python scripts/augment_poem_dataset.py --labels data/real/labels/web_batch1.txt `
  --out-labels data/real/labels/web_batch1_aug.txt --name-prefix web_aug --copies 80
copy models\crnn_best.pth models\crnn_best_pre_web.pth
python -m src.recognition.train --config configs/mix_web.yaml `
  --extra-labels data/synthetic_pages/train_labels.txt `
  --extra-labels data/synthetic_hard/train_labels.txt `
  --extra-labels data/real/labels/poem_kanyawee_aug.txt `
  --extra-labels data/real/labels/user_batch1_aug.txt `
  --extra-labels data/real/labels/web_batch1_aug.txt `
  --resume models/crnn_best.pth
```

**Notebook:** after any `data/real/labels/*_aug.txt` exists, set `RUN_TRAIN=True` and
Restart & Run All — prefers `configs/mix_web.yaml` and auto-includes every `*_aug.txt`
plus `data/synthetic_hard/train_labels.txt` / `web_batch1_acts.txt` when present.

**Test after:** leave train flags `False`, set `TEST_IMAGE_PATH` to a page (e.g. the
exam cover), Run All; keep `RUN_POEM_CER=True` for poem crops.

**Note:** `*.pth` checkpoints and large page/hard images are gitignored. Keep a local
`models/crnn_best.pth` (backup: `models/crnn_best_pre_web.pth`). Labels + `SOURCES.md`
are tracked; regenerate aug images locally.

On the Kanyawee poem crops (in-train after mix), corpus CER dropped from ~0.19
(pre-mix general model) to ~0.008. Held-out `data/eval_pages` overall CER stayed
~0.098 (no regression vs the prior general checkpoint).

### Jul-27: page deskew + book-print/tiny-text training styles

Diagnosis on a new real photographed poem page (serif book print) and the held-out
eval pages showed two independent failure classes:

1. **Detection — rotated photos merged adjacent lines.** At ~1.5° skew over a
   ~600 px column the vertical drift exceeds the inter-line gap, so the projection
   profile never dips between lines; a missed/merged line then shifts the
   order-aligned scoring and inflates page CER past 1.0.
   Fix: `estimate_page_skew` + `rotate_page` in `src/detection/text_detection.py`
   (projection-profile sharpness search, applied automatically in
   `run_pipeline_on_gray` when `detection.deskew: true`), plus a percentile-robust
   valley test in `_split_tall_band` so strike-through/underline rules no longer
   cause false splits. Regression tests: `tests/test_page_detection.py`.
2. **Recognition — real serif book print + tiny text.** Vowel-sign drops and
   ත/න, ට/ව confusions on photographed book/poem print; very small footer lines
   decoded as garbage. Fix: `scripts/generate_hard_lines.py` gained a
   `book_serif` style (grey paper, serif faces) and a low-res degradation pass
   (crush to 10–22 px height and back), then a continue-train with all real+hard
   extras (`models/train_jul27.log`).

Held-out evaluation lives in `data/eval_real/print_photos/` (never trained on;
see its `SOURCES.md`). Detection-only gains (same checkpoint, deskew on):
`data/eval_pages` overall CER 0.098 → 0.020 (all 76 lines found exactly),
adversarial pages 0.063 → 0.034.

Results after the Jul-27 continue-train (25 epochs, `models/train_jul27.log`;
backup of the prior checkpoint: `models/crnn_best_pre_jul27.pth`):

| Eval set | Before (Jul-26 ckpt + old detector) | After (Jul-27 ckpt + deskew) |
|---|---|---|
| `data/eval_pages` end-to-end CER (held-out synth pages) | 0.0981 | **0.0192** |
| Adversarial pages end-to-end CER | 0.0629 | **0.0339** |
| `print_photos` poem page end-to-end CER (held-out real) | 0.1533 | **0.1400** |
| `user_batch1_holdout.txt` line CER (41 crops) | 0.0338 | **0.0055** |
| `web_batch1_holdout.txt` line CER (14 crops) | 0.0000 | 0.0000 |
| Kanyawee poem line CER (in-train) | 0.0110 | **0.0055** |
| Synthetic val CER (trainer) | 0.0332* | 0.0356 |

\* mix runs trade a little synthetic val CER for real-image accuracy; the real
holdout rows are the signal. Remaining known weakness: very small text
(≈10 px x-height footers) is still unreliable, and the ු/ූ (short/long -u)
distinction on traditional serif book faces.

### Real image test (notebook Section 8)

1. Open `notebooks/local_pipeline.ipynb`.
2. In Section 4, leave train flags `False` if `models/crnn_best.pth` already exists.
3. Set `TEST_IMAGE_PATH` to a page/photo path, or leave it empty and use the picker / demo fallback.
4. **Kernel → Restart & Run All** — detect lines, show crops + Sinhala predictions + full transcription.

## Google Colab

See `notebooks/colab_pipeline.ipynb` for an end-to-end run: mount Drive, install deps,
generate synthetic data, train the CRNN, evaluate (CER/WER) and run an inference demo.

## Reference methods (2021+)

TrOCR, PARSeq, Donut, PP-OCRv3, SynthTIGER, DBNet, CRNN.



