# Sinhala Document OCR

An end-to-end Optical Character Recognition (OCR) pipeline for **printed** Sinhala
documents — poems, forms, book pages, exam covers and lyric cards — with support
for mixed **Sinhala–English** layouts.

Give it a photograph or scan of a Sinhala page; it deskews the page, finds the
text lines, recognises each line with a CRNN+CTC model trained for Sinhala, and
returns the transcription.

This repository is the implementation of an MSc research project (Sinhala only;
Tamil is out of scope). It trains on a single consumer GPU (developed on an
RTX 4060 laptop) with data captured from a phone camera / flatbed scanner plus a
synthetic Sinhala text-line generator.

---

## Quick start (examiner / supervisor)

**Final results, methodology and honest limitations: [`RESULTS.md`](RESULTS.md).**

```powershell
git clone <repo> && cd sinhala-document-ocr
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
# GPU (optional, much faster):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then place the trained checkpoint at **`models/crnn_best.pth`** (≈120 MB — it is
gitignored, so it is handed over separately; see *Reproducing the trained model*
below to rebuild it instead).

**Test an image:**

```powershell
jupyter lab notebooks/local_pipeline.ipynb
```

In Section 4 leave `RUN_GENERATE` / `RUN_GENERATE_PAGES` / `RUN_TRAIN` as `False`,
set `TEST_IMAGE_PATH` to your image (or leave it empty to get a file picker, or
set the `OCR_TEST_IMAGE` environment variable), then **Kernel → Restart & Run All**.
Section 8 shows the detected line boxes, each line crop with its prediction, and
the full transcription. With no image chosen it falls back to a bundled demo page
under `data/eval_real/print_photos/`.

The whole notebook runs in well under a minute on a GPU when the training flags
are `False`. It is verified to run headlessly:

```powershell
$env:OCR_TEST_IMAGE="data/eval_real/print_photos/page_poem_print.jpg"
jupyter nbconvert --to notebook --execute --inplace notebooks/local_pipeline.ipynb
```

**Command line instead of the notebook:**

```powershell
# one page, with the detected boxes and transcription
python scripts/eval_real_images.py --image path/to/page.jpg --checkpoint models/crnn_best.pth

# the full honest evaluation suite (held-out vs in-train is labelled per set)
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth
```

### ⚠ The trained model is not in git

`models/*.pth` is gitignored (≈120 MB per checkpoint), so **cloning this
repository is not enough to run the demo.** The file that must travel with the
handover is:

| File | Why |
|---|---|
| **`models/crnn_best.pth`** | the delivered model — Section 8 of the notebook and every eval script need it |
| `models/crnn_best_jul28_e12.pth` | identical byte-for-byte copy, kept as the restore point |
| `models/crnn_best_pre_jul28.pth` | the previous checkpoint, for before/after comparison |

Copy them onto a USB stick / cloud drive alongside the repo, or rebuild the
model from scratch with *Reproducing the trained model* below (~6 h on a GPU).
`models/charset.json` **is** in git and must match the checkpoint — do not
regenerate it separately.

Everything else degrades gracefully: a clean clone with no checkpoint and no
generated data still passes all 63 tests, and the notebook runs top to bottom
printing what is missing until Section 8, which stops with an explicit message
pointing back here.

**Headline accuracy** (end-to-end, detection errors included; full table and
methodology in [`RESULTS.md`](RESULTS.md)):

| Evaluation set | Status | Corpus CER |
|---|---|---|
| Real photographed pages (2 pages, 32 lines) | **held out** | **0.0877** |
| Synthetic eval pages (10 pages, 76 lines) | held out | **0.0088** |
| Adversarial acceptance pages (3) | held out | 0.0351 |
| Real line crops (`user_batch1`, 41) | *in training* | 0.0035 |
| Kanyawee poem lines (10) | *in training* | 0.0000 |

Line detection is exact on every page in the suite (76/76, 9/9, 23/23).

---

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
                                 eval_real_images.py, run_realistic_eval.py,
                                 run_eval_suite.py, report_errors.py,
                                 check_holdout_leakage.py, ...)
  data/  models/  tests/
  RESULTS.md                    final metrics, methodology, limitations
```

### Where things live

| Path | Tracked in git? | What |
|---|---|---|
| `src/`, `scripts/`, `tests/`, `configs/`, `notebooks/` | yes | all code |
| `src/data/corpus_sinhala.txt`, `sample_words.txt`, `form_vocab.txt` | yes | Sinhala text used by the generator and the character LM |
| `data/real/labels/*.txt` | yes | real line-crop transcriptions |
| `data/eval_pages/`, `data/eval_real/` | yes | evaluation pages + ground truth + `SOURCES.md` |
| `models/charset.json` | yes | character set (224 chars + CTC blank) |
| **`models/*.pth`** | **no** | trained checkpoints, ~120 MB each |
| **`data/synthetic*/`, `data/real/images/`, `data/uploads/`, `data/debug/`** | **no** | generated / captured images |

Everything untracked is regenerable — see *Reproducing the trained model*.

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

**Holdout (audited Jul-28 2026):** `user_batch1_holdout.txt` is **no longer a
holdout** — all 41 of its transcripts were folded into training in a later round
(historical copy: `user_batch1_holdout_SUPERSEDED.txt`).
`web_batch1_holdout.txt` is only a partial holdout (6 of its 14 transcripts are
also in training). The only fully clean real-image evidence is
`data/eval_real/print_photos/`. Run `python scripts/check_holdout_leakage.py` to
re-verify this at any time; `scripts/run_eval_suite.py` prints the status next to
every number.

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

**Correction (Jul-28 audit):** the `user_batch1_holdout.txt` row above is *not*
a holdout — all 41 of its transcripts were folded into the training mix in an
earlier round. `scripts/check_holdout_leakage.py` now proves this automatically
and `scripts/run_eval_suite.py` labels every set accordingly. See `RESULTS.md`.

### Jul-28: the small-text round (final delivered model)

Two independent fixes, measured separately. Full tables in
[`RESULTS.md`](RESULTS.md).

1. **A train/inference mismatch on short crops.** `inference.pad_to_height`
   white-padded a crop shorter than the 48 px model input, while training always
   *resized* to 48 px. An 18 px lyric line therefore reached the model with its
   glyphs at 18 px — a scale it had never been trained on. Setting
   `pad_to_height: false` (upscale instead) improved every single evaluation set
   with the checkpoint completely unchanged: real photos 0.1688 → 0.1218,
   `eval_pages` 0.0192 → 0.0120. Pinned by
   `tests/test_small_text_path.py`.
2. **A dedicated small-text curriculum.** `scripts/report_errors.py` showed the
   residual errors were specific graphemes, not generic blur — dominated by
   `ේ → ී` always paired with a spurious inserted `ෙ`. Sinhala renders `ේ` with
   a *pre-base* kombuva sitting visually between two consonants; at ~16 px the
   model attached it to the wrong consonant (`දෙරණේ` → `දෙරෙණී`). Also, the
   `...//` refrain notation on lyric pages had never appeared in training text.
   `scripts/generate_hard_lines.py` gained a `lyrics_small` style, a
   `--tiny-ratio` flag (used to build `data/synthetic_small`: 10,000 lines all
   crushed to 11–26 px, half left small so the dataset performs the same upscale
   inference does), oversampling of confusable graphemes, and refrain notation.
   A 12-epoch continue-train (`configs/mix_jul28.yaml`, ~70k rows, 3.5 h on an
   RTX 4060, `models/train_jul28.log`) took real photos 0.1218 → **0.0990** and
   `eval_pages` 0.0120 → **0.0100**. The `...//` errors disappeared entirely.

3. **Character n-gram LM fused into CTC beam search**
   (`src/postprocess/char_lm.py`, `src/recognition/decode.py`). A 6-gram
   character model built at load time from training-side Sinhala text only
   is combined with the acoustic score as
   `log P_ctc + lm_weight·log P_lm + insertion_bonus·|y|`. At the swept optimum
   `lm_weight: 0.2` it **improves or ties every held-out set** — real photos
   0.0990 → **0.0877**, `eval_pages` 0.0100 → **0.0088**, adversarial unchanged
   — so it is the shipped default (`decode: beam_lm`). It costs ~50% more
   inference time; set `decode: greedy` for the fast path. Weights above ~0.5
   overfit the LM corpus and badly damage out-of-domain text.

Sole regression across the whole round: the 3-page adversarial set,
0.0339 → 0.0351. Kept, because it moved against a 48% improvement on the real
photographs, which are the only fully clean real-image evidence in the project.

## Reproducing the trained model

Checkpoints and generated images are gitignored. From a fresh clone, on a
CUDA GPU (~6 h end to end; the continue-train alone is ~3.5 h on an RTX 4060):

```powershell
# 1) extra Sinhala font families (best-effort; safe to skip if offline)
powershell -ExecutionPolicy Bypass -File scripts/download_fonts.ps1

# 2) synthetic line crops and detector-in-the-loop page crops
python scripts/generate_data.py  --config configs/local.yaml --large
python scripts/generate_pages.py --config configs/local.yaml --num-pages 4000

# 3) hard-case + all-tiny supplements (the Jul-27 / Jul-28 rounds)
python scripts/generate_hard_lines.py --num 12000 --out data/synthetic_hard --seed 20260728
python scripts/generate_hard_lines.py --num 10000 --out data/synthetic_small `
    --name-prefix small --seed 20260729 --tiny-ratio 1.0 --tiny-min-h 11 --tiny-max-h 26

# 4) real line crops: images are not in git, the transcriptions are.
#    Re-crop them, then rebuild the augmented copies:
python scripts/prepare_poem_dataset.py --image data/uploads/test2.png
python scripts/prepare_real_pages.py --gt-json data/real/labels/user_batch1_gt.json
python scripts/prepare_web_batch1.py
python scripts/download_hf_acts.py --max-pages 200        # optional, CC-BY-4.0
python scripts/augment_poem_dataset.py --copies 80
python scripts/augment_poem_dataset.py --labels data/real/labels/user_batch1.txt `
    --out-labels data/real/labels/user_batch1_aug.txt --name-prefix user_aug --copies 40
python scripts/augment_poem_dataset.py --labels data/real/labels/web_batch1.txt `
    --out-labels data/real/labels/web_batch1_aug.txt --name-prefix web_aug --copies 60

# 5) base model, then the Jul-28 continue-train
python -m src.recognition.train --config configs/local.yaml
copy models\crnn_best.pth models\crnn_best_pre_jul28.pth
python -m src.recognition.train --config configs/mix_jul28.yaml `
    --extra-labels data/synthetic_pages/train_labels.txt `
    --extra-labels data/synthetic_hard/train_labels.txt `
    --extra-labels data/synthetic_small/train_labels.txt `
    --extra-labels data/real/labels/poem_kanyawee_aug.txt `
    --extra-labels data/real/labels/user_batch1_aug.txt `
    --extra-labels data/real/labels/web_batch1_aug.txt `
    --extra-labels data/real/labels/web_batch1_acts_aug.txt `
    --extra-labels data/real/labels/web_batch1_acts.txt `
    --resume models/crnn_best.pth

# 6) verify
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth
```

The trainer only overwrites `models/crnn_best.pth` when *synthetic validation*
CER improves; it writes `models/crnn_last.pth` every epoch. Because a
domain-mix run can trade synthetic CER for real accuracy, always decide which
one to keep with `scripts/run_eval_suite.py` on the held-out sets, and back up
the previous checkpoint first (`models/crnn_best_pre_*.pth`).

### Real image test (notebook Section 8)

1. Open `notebooks/local_pipeline.ipynb`.
2. In Section 4, leave train flags `False` if `models/crnn_best.pth` already exists.
3. Set `TEST_IMAGE_PATH` to a page/photo path, or leave it empty and use the picker / demo fallback.
   The `OCR_TEST_IMAGE` environment variable does the same thing for a headless run.
4. **Kernel → Restart & Run All** — detect lines, show crops + Sinhala predictions + full transcription.

If `models/crnn_best.pth` is missing, Sections 5–7 print a clear NOTE and skip
(they will *not* silently start hours of data generation or training) and
Section 8 raises a message telling you where to get the checkpoint.

## Architecture summary

| Stage | Implementation | Key settings (`configs/local.yaml`) |
|---|---|---|
| Deskew | Projection-profile sharpness search over ±5°, applied before detection | `detection.deskew`, `deskew_max_angle: 5.0` |
| Line detection | Contrast-binarised ink mask (drops watermarks), border/frame suppression, horizontal ink-profile bands, tall-band re-split at the lowest internal valley | `detection.method: projection` |
| Line crops | Padded boxes, minimum crop height 14 px | `crop_padding_x: 10`, `crop_padding_y: 5` |
| Recognition | CRNN — CNN backbone (512 ch) → 2-layer BiLSTM (256 hidden) → CTC over 224 Sinhala/ASCII characters + blank, fixed input height 48 px | `model.*`, `image.height: 48` |
| Inference prep | Grayscale, auto polarity inversion, **upscale** (not pad) to 48 px, LANCZOS | `inference.pad_to_height: false` |
| Decoding | CTC greedy (default); optional prefix beam search with character-LM shallow fusion | `inference.decode: greedy \| beam \| beam_lm` |
| Post-processing | Edit-distance dictionary correction available; character n-gram LM in the decoder | `src/postprocess/` |

One shared code path (`src/evaluation/pipeline_eval.py`) runs detection +
recognition for the notebook, the CLI scripts and every evaluation script, so
"what gets measured" and "what a user's upload goes through" cannot drift apart.

## Known limitations

Measured, not guessed — see [`RESULTS.md`](RESULTS.md) §4 for the error counts.

1. Fully held-out **real** evaluation data is only 2 photographs / 32 lines.
   More photographed pages with verified ground truth is the highest-value
   addition anyone could make next.
2. On very low-resolution lines (~16 px) the `ේ` / `ී` confusion — the pre-base
   kombuva being attached to the wrong consonant — is reduced but not solved.
3. Long `ූ` vs short `ු` on traditional serif book faces is still wrong on the
   poem page (4 occurrences); the character LM cannot help because the corpus
   itself prefers the short form.
4. Logos, decorative display fonts and graphic regions are not suppressed — the
   detector emits a box and the recogniser produces noise.
5. Very short isolated marker/label lines can be dropped by the detector's
   relative-height filter.
6. Handwriting is out of scope for the delivered model (printed text only).

## Google Colab

See `notebooks/colab_pipeline.ipynb` for an end-to-end run: mount Drive, install deps,
generate synthetic data, train the CRNN, evaluate (CER/WER) and run an inference demo.

## Reference methods (2021+)

TrOCR, PARSeq, Donut, PP-OCRv3, SynthTIGER, DBNet, CRNN.



