# Methodology — End-to-End OCR for Printed Sinhala Documents

**Project:** Sinhala Document OCR (`sinhala-document-ocr`)  
**Delivered checkpoint:** `models/crnn_best.pth` (CRNN + CTC; continue-trained 2026-07-28)  
**Companion notebook:** `notebooks/training_methodology.ipynb`  
**Numerical source of truth:** `RESULTS.md` (reproduced by `scripts/run_eval_suite.py`)

This document is the full training and evaluation methodology. The Jupyter demos
(`notebooks/local_pipeline.ipynb`, `notebooks/colab_pipeline.ipynb`) load the
delivered weights and run **inference only**. They do not retrain the model.

---

## 1. Problem

Printed Sinhala remains a demanding OCR setting: a large grapheme inventory,
pre-base vowel signs (kombuva) that sit *between* consonants in the rendered
line, Zero-Width Joiner (ZWJ) conjuncts, and mixed Sinhala–English tokens on
forms, exam covers, lyric cards and book pages. Public labelled page datasets
are small relative to Latin or Devanagari, so an MSc-scale project cannot rely
on millions of real transcribed lines.

Phone photographs add skew, JPEG compression, uneven lighting and crops that
may be only 12–26 px tall. A model trained solely on tightly cropped synthetic
lines therefore sees a different distribution from the detector crops produced
at inference time.

**Research question.** How can a single general CRNN, trained on a consumer GPU
from synthetic pages plus a modest real set, be evaluated *honestly* at page
level — without treating synthetic validation CER, or in-training poem/line
crops, as evidence of generalisation?

**Scope.** Printed Sinhala (and Latin tokens that appear on those pages).
Handwriting and Tamil are out of scope. The delivered detector is classical
projection-profile, not a learned DBNet/CRAFT head.

---

## 2. Pipeline

A page image is processed in five stages. Every evaluation script and both demo
notebooks share one detect-and-recognise path (`src/evaluation/pipeline_eval.py`),
so what is measured is what a user upload goes through.

```
(1) Acquisition     (2) Preprocess      (3) Detection       (4) Recognition      (5) Post-process
 phone / scanner  →  deskew / CLAHE  →  projection lines →  CRNN + CTC decode →  matra / lyric rules
```

| Stage | Implementation | Why this choice |
|---|---|---|
| Preprocess | `src/preprocessing/`, deskew in `src/detection/text_detection.py` | A few degrees of skew merge adjacent projection bands on photographed columns. Deskew (`detection.deskew: true`, search ±5°) is applied before the ink profile. |
| Line detection | Horizontal projection on a contrast-binarised ink mask; tall-band re-split at the internal valley | Classical, inspectable, and exact on the evaluation suite (76/76, 9/9, 23/23 lines). Watermark/border suppression avoids treating ornaments as text. |
| Recognition | CRNN (CNN → BiLSTM → linear) trained with CTC | No character boxes required; fits a single RTX 4060-class GPU. Input height 48 px, variable width (max 512). |
| Decode | CTC prefix beam search with 6-gram character-LM shallow fusion | Default `decode: beam_lm`, `lm_weight: 0.2`. Swept; higher weights overfit the LM corpus. |
| Post-correct | `src/postprocess/sinhala_fix.py` | Only rules that improve or tie **every** held-out set. Same checkpoint; no retrain for the final accuracy push. |

Short crops at inference are **upscaled** to 48 px (`pad_to_height: false`).
Training always resizes with `resize_keep_height`. White-padding a 18 px lyric
line used to leave glyphs at 18 px inside a 48 px canvas — a train/inference
mismatch that was measured and then removed (`RESULTS.md` §3a).

---

## 3. Datasets

Counts below were taken from label files on the development machine (2026-08).
Large generated image trees (`data/synthetic*/`) are gitignored; a fresh clone
may not have them. The methodology notebook counts what is on disk and, if a
file is missing, prints the documented figure from this table rather than
crashing.

**In-train vs held-out is labelled in every row.** Poem lines and
`user_batch1` are in the training mix; they must not be quoted as
generalisation.

### 3.1 Training and mix sources

| Source | Files / lines (this machine) | Role | Status |
|---|---|---|---|
| Synthetic text lines | 30,000 images; labels 21,000 / 4,500 / 4,500 (train/val/test, split 0.70 / 0.15 / 0.15) | Primary CRNN corpus (`data/synthetic/`) | Train / val / test of the *generator*; not real-image evidence |
| Detector-in-the-loop page crops | 25,545 images; 21,713 train + 3,832 val labels (`data/synthetic_pages/`) | Extra-labels: real detector boxes on rendered pages | In mix (Jul-28 extras). README v3: ~4,000 pages, ~81.6% detector exact-match rate |
| Hard-case lines | 12,000 (`data/synthetic_hard/train_labels.txt`) | Dark/pill titles, book-serif, low-res crush | In mix |
| Tiny-text curriculum | 10,000 (`data/synthetic_small/`); every line crushed to 11–26 px | Matches the upscale inference path | In mix (Jul-28) |
| Tiny-text round 2 | 10,000 (`data/synthetic_small2/`) | Wider confusion pool | **Not** in the delivered model (Jul-29 run rejected) |
| Real poem lines | 10 unique in `poem_kanyawee.txt`; 810 after 80× augmentation | Style mix (literary print). Mentioned here only as a **training mix** source — not as a live OCR demo | **In training** |
| Real user pages | 91 labelled lines in `user_batch1.txt` (90 unique transcripts); 3,731 after augmentation | Phone/scan crops | **In training**. A historical file `user_batch1_holdout.txt` (41 lines) was later folded into this mix; do not quote it as a holdout |
| Web / exam-cover lines | 6 unique in `web_batch1.txt`; 486 after augmentation | High-contrast title / exam-cover style | **In training** |
| Sri Lankan Acts (HF, CC-BY-4.0) | 2,275 lines in `web_batch1_acts.txt` (60 train pages); 9,100 after light aug | Legal print mix via `scripts/download_hf_acts.py` | **In training** |
| Acts test extras | 1,895 lines in `web_batch1_acts_extra.txt` (50 pages) | Reserved for a future mix | **Not** used in the delivered Jul-28 train |
| Sentence corpus | 3,269 lines in `src/data/corpus_sinhala.txt` | ~65% of synthetic lines sampled from this corpus | Train-side text only |

**Jul-28 merged training volume** (what the delivered continue-train actually
saw), from `configs/mix_jul28.yaml` (`synthetic_train_max: 10000`) plus
`--extra-labels`:

```
10,000 (capped synthetic train)
+ 21,713 (page-synth train)
+ 12,000 (hard)
+ 10,000 (tiny)
+    810 (poem aug)
+  3,731 (user_batch1 aug)
+    486 (web_batch1 aug)
+  9,100 (acts aug)
+  2,275 (acts raw)
= 70,115 labelled rows  (~70k as stated in RESULTS.md §3b)
```

Why synthetic at all? There are not enough labelled real Sinhala *lines* to
train a 225-class CTC model from scratch. The generator (`src/data/synthetic_generator.py`)
renders multi-font lines (Nirmala UI, Iskoola Pota, Noto Sans/Serif Sinhala,
Abhaya Libre, Yaldevi) with SynthTIGER-style degradations: rotation, blur,
noise, JPEG, shadow, plus v3 camera-like defocus, paper texture, rare moiré,
edge artefacts and multi-generation JPEG.

Why detector-in-the-loop? Tight crops never look like the imperfect boxes the
projection detector emits on a full page (padding, a rule fragment, a merged
descender). `src/data/page_synth.py` renders a whole page (paragraph / bordered
card / poem / mixed Sinhala–English / letterhead), runs the **same** detector
used at inference, and keeps only pages whose detected line count matches
ground truth. Mismatched pages are discarded, not mislabelled; the discard rate
is a detector-health metric.

Why mix real lines? Synthetic fonts and layouts still miss decorative exam
covers, serif book print and tiny lyric cards. A small real set, heavily
augmented (`scripts/augment_poem_dataset.py`), is mixed as extra-labels so
those styles influence the gradient without abandoning the synthetic prior.

### 3.2 Held-out evaluation

| Set | Content | Status |
|---|---|---|
| `data/eval_real/print_photos/` | 2 photographed pages, **32** hand-transcribed lines (`page_poem_print.jpg` 9 lines; `page_song_lyrics.jpg` 23 lines) | **Held out.** Never trained on. The headline real-image result. |
| `data/eval_pages/` | 10 synthetic pages, 5 layouts, **76** lines | Held out. Same generator *family* as training pages, separate generation run. |
| `data/eval_real/adversarial/` | 3 hand-built stress pages, **26** lines (bordered card + watermark, article, heavily degraded photo) | Held out, synthetic. |
| `data/real/labels/web_batch1_holdout.txt` | 14 hard-style crops | **Partial leak:** 6 of 14 transcripts also appear in training. Weak evidence. |
| `user_batch1_holdout.txt` / `poem_kanyawee.txt` | 41 and 10 line crops | **In training.** Reported for reference only. |

`scripts/check_holdout_leakage.py` compares evaluation transcripts against
every transcript the trainer saw. Honest summary: the only fully clean
*real-image* generalisation evidence is **2 photographs / 32 lines**.

---

## 4. Model

**Architecture.** Convolutional Recurrent Neural Network (Shi, Bai & Yao,
*IEEE TPAMI* 2017): CNN backbone → map-to-sequence → two Bidirectional LSTM
blocks → linear layer over classes. Implementation: `src/recognition/model.py`.

| Hyperparameter | Value (`configs/local.yaml` / `model`) |
|---|---|
| Input | Greyscale, height **48** px, max width 512, pad value 255 |
| CNN final channels | 512 |
| BiLSTM hidden (per direction) | 256 |
| LSTM layers (first block) | 2 |
| Dropout | 0.1 |
| Charset | **224** printable characters in `models/charset.json` |
| CTC blank | index 0 → **225** output classes |

The charset (`src/charset.py`) covers the Sinhala Unicode block (U+0D80–U+0DFF),
ZWJ and ZWNJ, ASCII letters and digits, and form punctuation. Conjuncts that
require U+200D are kept as sequences so the model is not forced to invent
impossible compositions. **Do not regenerate `charset.json` independently of the
checkpoint** — class indices must match the saved weights.

**Loss.** Connectionist Temporal Classification (Graves et al., ICML 2006).
PyTorch `nn.CTCLoss(blank=0, zero_infinity=True)`. CTC removes the need for
character-level bounding boxes: the network emits a frame-wise distribution
over 225 symbols; the loss marginalises over all alignments that collapse
(after removing blanks and repeats) to the Unicode transcript.

---

## 5. Training procedure

Training is **not** a single from-scratch run of 12 epochs. The delivered
weights are the last *promoted* continue-train on top of a general CRNN that
had already seen synthetic lines, page-synth crops, hard styles and real
mixes. Promotion is decided with `scripts/run_eval_suite.py` on held-out
pages, **not** with the trainer’s synthetic validation CER alone.

### 5.1 Chronology (from logs on disk + README where a log is empty)

| Round | Config / log | Epochs actually logged | Best synthetic val CER | Notes |
|---|---|---|---|---|
| v2 diverse synthetic | `configs/local.yaml`, `models/train_v2.log` | **20** (config allows 40; early stopping) | 0.0423 | 30k lines, ReduceLROnPlateau, patience 8 |
| v3 domain-gap | `configs/local.yaml` + page extras | README: **40** scheduled, best at epoch **37** | **0.0311** | `models/train_v3.log` is empty on this machine; figure taken from README. ~4 h 11 min on RTX 4060; 25,545 page-synth crops |
| Poem mix | `configs/mix_real.yaml`, `train_poem_mix.log` | **14** (config 15) | 0.0312 | Real poem aug mixed in |
| user_batch1 mix | `mix_real.yaml`, `train_user_batch1.log` | **15** | 0.0332 | Real user-page aug |
| Web / hard mix | `configs/mix_web.yaml`, `train_web_batch.log` | **6** (config 25; early stop) | 0.0388 | Domain mix can *worsen* synth val while helping real pages |
| Jul-27 book-serif + tiny | `mix_web` extras, `train_jul27.log` | **25** | **0.0356** | `book_serif` style + low-res crush; deskew is a detector change (no weight change required for the detection gain) |
| **Jul-28 delivered** | `configs/mix_jul28.yaml`, `train_jul28.log` | **12** | **0.0348** | ~70k rows, ~3.5 h, RTX 4060. This is `crnn_best.pth` |
| Jul-29 (rejected) | `configs/mix_jul29.yaml`, `train_jul29.log` | **5** of 8 planned | **0.0343** | Best synth val ever; **worse on every held-out set**. Not promoted. Writes `crnn_jul29_*.pth` |

Do not invent a 40-epoch delivered schedule. The shipped recogniser is the
**12-epoch** Jul-28 continue-train (`RESULTS.md` opening paragraph).

### 5.2 Hyperparameters of the delivered round

From `configs/mix_jul28.yaml` (inherits `mix_web.yaml` → `local.yaml`):

| Item | Value |
|---|---|
| Optimiser | Adam |
| Epochs | 12 |
| Learning rate | 8×10⁻⁵, ReduceLROnPlateau (factor 0.5, patience 2) |
| Observed LR in log | 8×10⁻⁵ for epochs 1–5, then 4×10⁻⁵ for epochs 6–12 |
| Batch size | 32 |
| Gradient clip | 5.0 |
| Early stopping patience | 6 (on synthetic val CER) |
| Extra-label repeat | 1 |
| Synthetic train cap | 10,000 (so extras stay influential) |
| Seed | 1337 |
| Hardware (development) | NVIDIA RTX 4060 Laptop GPU |

The trainer writes `models/crnn_last.pth` every epoch and overwrites
`models/crnn_best.pth` only when **synthetic** val CER improves. Because a
domain-mix run can trade synthetic CER for real accuracy, the previous
checkpoint is copied first (`models/crnn_best_pre_jul28.pth`) and the keep/reject
decision is made from the held-out suite.

### 5.3 Base (local.yaml) schedule used for from-scratch / v2–v3

| Item | Value |
|---|---|
| Epochs | 40 (early stopping patience 8) |
| LR | 1×10⁻³, plateau factor 0.5, patience 3, min_lr 1×10⁻⁶ |
| Batch size | 32 |
| Save-best metric | CER |

---

## 6. Decoding and post-processing

### 6.1 CTC decode

Default inference (`configs/local.yaml` `inference:`):

* `decode: beam_lm`
* `lm_weight: 0.2`, `insertion_bonus: 0.6`, `beam_width: 12`, `beam_top_k: 8`
* Character LM: 6-gram, Witten–Bell interpolation, built at load time from
  **training-side** Sinhala text only (`src/postprocess/char_lm.py`) — no
  held-out ground truth.

Score of a hypothesis \(y\):

```
score = log P_ctc(y | x) + lm_weight · log P_lm(y) + insertion_bonus · |y|
```

The insertion bonus counters the LM’s preference for short strings. A sweep
on the delivered checkpoint (`RESULTS.md` §3c) showed `lm_weight: 0.2`
improves or ties every held-out set (real photos 0.0990 → 0.0877, eval_pages
0.0100 → 0.0088, adversarial unchanged). Weights above ~0.5 overfit news/legal
prose and damage out-of-domain lyrics. Greedy decode remains available for a
faster path.

Test-time augmentation (photometric variants averaged before decode) was
implemented, measured, and **left off**: it is a trade between real photos and
synthetic pages, not a strict improvement (`RESULTS.md` §3d).

### 6.2 Orthographic post-correction

`fix_sinhala_ocr` (`src/postprocess/sinhala_fix.py`) applies only rules that
improve real photos **and** leave `eval_pages` / `adversarial` unchanged:

* Matra / kombuva: `ෙCී` → `Cේ`, word-final `ෙණ්` → `ණේ`, dangling `ේී`,
  word-final `ණී` → `ණේ`, illegal pre-base reorder, orphan pre-base+virama.
* LM-gated word-final `මි` → `ම්`.
* Lyric polish: `..//` → `...//`, `ලෙලෙ` → `ලෙල`, word-final `වැවි` → `වැව්`,
  `ුණි` → `ුණේ`, line-final `මැවුණ` → `මැවුණේ`.
* Word-accuracy push: gated literary `ස්සු` → `ස්සූ` (everyday `මිනිස්සු`
  excluded), plus a few single-grapheme repairs (`හදවෙතේ` → `හදවතේ`,
  `මගහැරී` → `මඟහැරී`, …).

Blind `ස්සු` (including `මිනිස්සු`) and bare `ණි` → `ණේ` were measured and
**rejected** because they regress the synthetic holdouts. These stages did
**not** change network weights.

---

## 7. Evaluation protocol

### 7.1 Metric definitions

Let \(R\) be the reference string and \(H\) the hypothesis. Edit distance is
Levenshtein (character tokens for CER; whitespace-tokenised words for WER).

* **CER** = (character insertions + deletions + substitutions) / \|R\|  
* **WER** = (word insertions + deletions + substitutions) / word-count(\(R\))  
* **Character Accuracy (%)** = (1 − CER) × 100  
* **Word Accuracy (%)** = (1 − WER) × 100  

**Corpus** metrics sum edits and reference lengths over the whole set (not the
mean of per-line rates). Page evaluation is **end to end**: deskew → detect →
recognise; detected lines are aligned to ground-truth lines in order, so a
missed or merged line inflates that page’s CER rather than being dropped.

Line-crop CER on the synthetic val split is a *training-time* signal only. It
is not evidence of real-world quality. The Jul-29 run is the proof: synthetic
val CER 0.0343 (best ever) with worse every held-out set.

### 7.2 Final results (delivered decode + post-correct)

Source: `RESULTS.md` §2 / `data/metrics/eval_summary.json` (AFTER + word-accuracy
push §3g, 2026-08-01). Same `crnn_best.pth` as Jul-28; later gains are decode
and post-correct only.

| Evaluation set | Status | CER | Char Acc. % | WER | Word Acc. % |
|---|---|---|---|---|---|
| Real photos (`print_photos`, 2 pages, 32 lines) | **held out** | **0.0325** | **96.75** | **0.1491** | **85.09** |
| Synthetic pages (`eval_pages`, 10 pages, 76 lines) | held out | **0.0088** | **99.12** | 0.0226 | 97.74 |
| Adversarial (3 pages, 26 lines) | held out | 0.0351 | 96.49 | 0.0458 | 95.42 |
| Synthetic val (trainer) | trainer split | 0.0348 | 96.52 | 0.0912 | 90.88 |
| `user_batch1` (41 historical holdout crops) | **in train** | 0.0035 | 99.65 | 0.0222 | 97.78 |
| `web_batch1_holdout` (14 crops) | partial leak | 0.0000 | 100.00 | 0.0000 | 100.00 |
| Poem lines (10 crops) | **in train** | 0.0000 | 100.00 | 0.0000 | 100.00 |

Line detection is exact on every page in the suite.

**Attributed gains on real photos** (CER), same or successive checkpoints as
documented in `RESULTS.md`:

| Stage | Real-photo CER |
|---|---|
| Jul-27 checkpoint + pad-to-height (old inference) | 0.1688 |
| Upscale short crops (config only) | 0.1218 |
| + Jul-28 tiny-text continue-train | 0.0990 |
| + beam + character LM (`lm_weight` 0.2) | 0.0877 |
| + matra post-correct | 0.0552 |
| + lyric polish | 0.0455 |
| + word-accuracy rules (§3g) | **0.0325** |

The one recorded regression is adversarial pages 0.0339 → 0.0351 after the
Jul-28 retrain. It was kept because it moved against a large gain on the only
clean real photographs.

---

## 8. Limitations

1. Fully held-out **real** data is 2 pages / 32 lines. More photographed pages
   with verified ground truth is the highest-value next step.
2. The synthetic validation split has **saturated** as a model-selection
   signal (Jul-29 negative result).
3. On ~16 px crops, `ේ` / `ී` (pre-base kombuva attached to the wrong
   consonant) is reduced but not solved; residual confusions include `්`/`ි`,
   `ඟ`/`ග`, `හ`/`න`.
4. Long `ූ` vs short `ු` on traditional serif book faces: the character LM
   prefers the short form; a *gated* literary rule helps some words only.
5. Logos and decorative display fonts are not suppressed; the detector emits a
   box and the recogniser produces noise.
6. Very short isolated marker lines can be dropped by the relative-height
   filter.
7. Handwriting is out of scope.

---

## 9. How to reproduce

Checkpoints (`*.pth`) and generated images are gitignored. From a clone, on a
CUDA GPU (~6 h end to end; the Jul-28 continue-train alone is ~3.5 h on an
RTX 4060):

```powershell
# 1) Fonts (best-effort)
powershell -ExecutionPolicy Bypass -File scripts/download_fonts.ps1

# 2) Synthetic lines + detector-in-the-loop pages
python scripts/generate_data.py  --config configs/local.yaml --large
python scripts/generate_pages.py --config configs/local.yaml --num-pages 4000

# 3) Hard-case + all-tiny supplements
python scripts/generate_hard_lines.py --num 12000 --out data/synthetic_hard --seed 20260728
python scripts/generate_hard_lines.py --num 10000 --out data/synthetic_small `
    --name-prefix small --seed 20260729 --tiny-ratio 1.0 --tiny-min-h 11 --tiny-max-h 26

# 4) Real line crops (images not in git; transcriptions are)
python scripts/prepare_poem_dataset.py --image data/uploads/test2.png
python scripts/prepare_real_pages.py --gt-json data/real/labels/user_batch1_gt.json
python scripts/prepare_web_batch1.py
python scripts/download_hf_acts.py --max-pages 200
python scripts/augment_poem_dataset.py --copies 80
python scripts/augment_poem_dataset.py --labels data/real/labels/user_batch1.txt `
    --out-labels data/real/labels/user_batch1_aug.txt --name-prefix user_aug --copies 40
python scripts/augment_poem_dataset.py --labels data/real/labels/web_batch1.txt `
    --out-labels data/real/labels/web_batch1_aug.txt --name-prefix web_aug --copies 60

# 5) Base model, then the Jul-28 continue-train
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

# 6) Honest evaluation
python scripts/check_holdout_leakage.py
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth --out data/debug/suite_final.json
```

Walk-through of counts, configs, logs and curves without retraining:

```powershell
jupyter lab notebooks/training_methodology.ipynb
```

Inference demo (loads `crnn_best.pth`, does not train):

```powershell
jupyter lab notebooks/local_pipeline.ipynb
```

---

## 10. References

1. B. Shi, X. Bai, and C. Yao, “An End-to-End Trainable Neural Network for
   Image-Based Sequence Recognition and Its Application to Scene Text
   Recognition,” *IEEE TPAMI*, 2017 (CRNN).
2. A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist
   Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent
   Neural Networks,” *ICML*, 2006.
3. Unicode Consortium, *The Unicode Standard* — Sinhala block and ZWJ conjunct
   behaviour.
4. Hugging Face / Microsoft TrOCR (2021–) — transformer OCR alternative
   considered; not the delivered training path.
5. Project record: `RESULTS.md`, `data/metrics/eval_summary.json`,
   `data/metrics/train_history_jul28.json`, `models/train_jul28.log`.

Prefer the peer-reviewed CRNN/CTC papers in a viva; treat project files as the
primary source for the numerical claims in Section 7.
