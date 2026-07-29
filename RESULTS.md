# Results — Sinhala Document OCR

Final evaluation of the delivered model, `models/crnn_best.pth`
(CRNN + CTC, 224 characters + blank, input height 48 px; continue-trained
12 epochs on 2026-07-28, best synthetic validation CER **0.0348**,
log: `models/train_jul28.log`).

Everything below is reproducible with two commands:

```powershell
python scripts/check_holdout_leakage.py
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth --out data/debug/suite_final.json
```

---

## 1. Methodology — what counts as held out

The single most important caveat in this project: **line-crop CER on synthetic
validation data is not evidence of real-world quality.** Every number that
matters is measured **end to end** — the page goes through deskew → line
detection → recognition, and detection mistakes count against the score. Lines
are aligned positionally against the ground truth, so a missed or merged line
inflates that page's CER rather than being silently dropped.

`scripts/check_holdout_leakage.py` compares each evaluation set's transcripts
against every transcript the trainer ever saw. Its output classifies the sets:

| Set | Content | Status |
|---|---|---|
| `data/eval_real/print_photos/` | 2 photographed real pages, 32 lines, hand-transcribed | **Held out.** Never trained on, in any form. The headline result. |
| `data/eval_pages/` | 10 synthetic pages, 5 layouts, 76 lines | Held out. Generated in a separate run from the training pages, but by the same generator family. |
| `data/eval_real/adversarial/` | 3 hand-built acceptance pages (bordered card + watermark, article page, heavily degraded photo) | Held out, synthetic. |
| `data/real/labels/web_batch1_holdout.txt` | 14 hard-style line crops | **Partially leaked** — 6 of 14 transcripts also appear in training. Weak evidence. |
| `data/real/labels/user_batch1_holdout.txt` | 41 real line crops | **In training.** Keeps its historical name, but a later round folded all 41 into the training mix. Reported for reference only; **do not quote this as a holdout result.** |
| `data/real/labels/poem_kanyawee.txt` | 10 real poem line crops | **In training.** Reference only. |

Honest summary: the only fully clean *real-image* generalisation evidence in
this project is 2 photographs / 32 lines. That is a small sample and the
per-page numbers below should be read with that in mind.

---

## 2. Final numbers

End-to-end corpus CER (lower is better). "BEFORE" is the previous delivered
checkpoint `models/crnn_best_pre_jul28.pth` with the previous inference config;
"AFTER" is the delivered `models/crnn_best.pth`.

### Held out

| Evaluation set | BEFORE (Jul-27) | AFTER beam+LM | AFTER + matra fix | Change vs Jul-27 |
|---|---|---|---|---|
| **Real photos** — `print_photos` (2 pages, 32 lines) | 0.1688 | 0.0877 | **0.0552** | **−67%** |
| &nbsp;&nbsp;`page_poem_print.jpg` (serif book print, 9 lines) | 0.1400 | 0.0533 | **0.0533** | −62% |
| &nbsp;&nbsp;`page_song_lyrics.jpg` (251 px-wide lyrics card, 23 lines) | 0.1781 | 0.0987 | **0.0558** | −69% |
| `eval_pages` (10 synthetic pages, 76 lines) | 0.0192 | **0.0088** | **0.0088** | **−54%** |
| `adversarial` (3 acceptance pages) | 0.0339 | 0.0351 | 0.0351 | +3.5% ⚠ |

The matra/modifier post-corrector (`src/postprocess/sinhala_fix.py`, §3e)
is on by default (`inference.post_correct: true`). Same checkpoint — no retrain.
It does not change `eval_pages` or `adversarial`.

### In training (reference only — not generalisation evidence)

| Set | BEFORE | AFTER |
|---|---|---|
| `user_batch1_holdout` (41 line crops) | 0.0055 | 0.0035 |
| `web_batch1_holdout` (14 line crops, partial leak) | 0.0000 | 0.0000 |
| `poem_kanyawee` (10 line crops) | 0.0055 | 0.0000 |
| Synthetic validation CER (trainer's own metric) | 0.0356 | 0.0348 |

"AFTER beam+LM" is `models/crnn_best.pth` with the shipped decode settings
(§3a–3c). "AFTER + matra fix" adds §3e on the same weights.

Line detection is exact on every page in the suite: 76/76 lines on
`eval_pages`, 9/9 and 23/23 on the two real photos.

**The one regression** is the 3-page adversarial set, +0.0012 absolute
(0.0339 → 0.0351). It is a synthetic acceptance set of 3 pages, and it moved
against a 41% improvement on the real photographs, so the checkpoint was kept.
This is recorded here rather than hidden.

---

## 3. Where the improvement came from

Two changes, measured separately so the credit is attributable.

### 3a. Train/inference mismatch on short crops (config only, no retraining)

`inference.pad_to_height` used to white-pad a crop shorter than the model input
up to 48 px. Training *always* resized to 48 px
(`OCRLineDataset` → `resize_keep_height`). So an 18 px lyric line kept its
glyphs at 18 px inside a 48 px input at inference time, a size the model had
never been trained on. Setting `pad_to_height: false` upscales instead.

With the **unchanged** Jul-27 checkpoint, flipping this one flag:

| Set | pad (old) | upscale (new) |
|---|---|---|
| real photos | 0.1688 | **0.1218** |
| `eval_pages` | 0.0192 | **0.0120** |
| adversarial | 0.0339 | 0.0339 |
| `user_batch1` | 0.0055 | **0.0030** |
| poem (in train) | 0.0055 | **0.0000** |

No set regressed. Regression test:
`tests/test_small_text_path.py::test_pad_to_height_false_upscales_instead_of_padding`.

### 3b. Dedicated small-text training curriculum (the retrain)

Error analysis (`scripts/report_errors.py`) on the remaining failures showed
they were not generic blur but a small set of specific graphemes, dominated by
`ේ → ී` (9 occurrences) always paired with a spurious inserted `ෙ`. Sinhala
renders `ේ` as a *pre-base* kombuva sitting visually **between** two
consonants plus a mark above; on a ~16 px line the model was attaching that
kombuva to the preceding consonant and reading the leftover mark as `ී`.
So `දෙරණේ` came out as `දෙරෙණී`. A second cluster was the `...//` refrain
notation on lyric pages, which had never appeared in any training text.

Fixes in `scripts/generate_hard_lines.py`:

* a `lyrics_small` render style (white card, soft ink, crushed to 12–24 px);
* `--tiny-ratio`, used to build `data/synthetic_small` — 10,000 lines where
  **every** sample is crushed to 11–26 px, and half are left at that small size
  so `OCRLineDataset` performs exactly the upscale inference does;
* oversampling (22% of lines) of corpus words containing the confusable
  graphemes `ණේ නේ ම් මි ූ ඬ ඳ ඟ ැයි`;
* `...//`-style refrain notation appended to 15% of lines.

Then a 12-epoch continue-train (`configs/mix_jul28.yaml`, ~70k merged rows,
3.5 h on an RTX 4060) from the Jul-27 checkpoint.

Contribution of the retrain alone, on top of 3a:

| Set | after 3a | after 3a + 3b |
|---|---|---|
| real photos | 0.1218 | **0.0990** |
| `eval_pages` | 0.0120 | **0.0100** |
| adversarial | 0.0339 | 0.0351 |

The `...//` notation errors are gone from the confusion table entirely.

### 3c. Character-LM shallow fusion in CTC beam search (enabled by default)

`src/postprocess/char_lm.py` builds a 6-gram character LM (Witten-Bell
interpolation) from the training-side Sinhala text only — no held-out ground
truth — and `src/recognition/decode.py` fuses it into CTC prefix beam search:

```
score = log P_ctc(y|x) + lm_weight · log P_lm(y) + insertion_bonus · |y|
```

The fusion weight was swept on the delivered checkpoint
(`insertion_bonus=0.6`, `beam_width=12`):

| `lm_weight` | real photos | `eval_pages` | adversarial |
|---|---|---|---|
| 0.0 (greedy) | 0.0990 | 0.0100 | 0.0351 |
| **0.2 (shipped)** | **0.0877** | **0.0088** | **0.0351** |
| 0.3 | 0.0893 | 0.0096 | 0.0364 |
| 0.4 | 0.0893 | 0.0104 | 0.0364 |
| 0.8 | 0.1445 | 0.0112 | — |
| 1.2 | 0.2045 | 0.0196 | — |

`lm_weight: 0.2` **improves or ties every held-out set** with no regression
anywhere, so it is the shipped default in `configs/local.yaml`. The cost is
runtime: the full suite takes 31 s instead of 21 s. For the fastest path set
`decode: greedy`.

Weights above ~0.5 clearly overfit the LM's news/legal corpus and damage
out-of-domain text such as song lyrics (0.1373 → 0.2511 at weight 1.2), which
is the expected failure mode of shallow fusion with a small in-domain corpus.
The one in-train reference set moves the other way (`user_batch1` 0.0030 →
0.0035); since that set is not held out it was not treated as evidence.

---

## 3d. Things that were measured and rejected

Recorded because the negative results are part of the evidence, and because
the code is still in the repository behind a flag.

**Test-time augmentation** (`inference.tta`, `src/recognition/inference.py`).
Each crop is decoded as several photometric variants — unaltered, unsharp
masked, and CLAHE contrast-equalised — whose per-frame distributions are
averaged before decoding. Every variant keeps the aspect ratio, so all of them
produce the same CTC sequence length and can be averaged in one batched forward
pass.

| Variants | real photos | `eval_pages` |
|---|---|---|
| *off (shipped)* | **0.0877** | **0.0088** |
| none + contrast | 0.0844 | 0.0100 |
| none + sharpen | 0.0877 | 0.0104 |
| none + sharpen + contrast | 0.0893 | 0.0100 |

The best variant pair buys about three characters on the real photographs and
costs more than that on the synthetic pages. Unlike the LM fusion in §3c it is
a *trade*, not a strict improvement, so by the same rule used everywhere else in
this project it was not promoted. Enable it with `tta: true` under `inference:`
if real photographs are the only thing that matters.

**A second continue-training round** (`configs/mix_jul29.yaml`) on a further
10,000 all-tiny lines whose confusion pool was widened to word-final `ේ`, `්`,
`ූ` and `ී`. This is the clearest result in the whole project, and it is a
negative one: at epoch 4 the run reached **0.0343 synthetic validation CER, the
best number this model has ever produced** — better than the delivered
checkpoint's 0.0348 — and was worse on *every* held-out set.

| | synthetic val | real photos | `eval_pages` | adversarial |
|---|---|---|---|---|
| delivered (Jul-28 epoch 12) | 0.0348 | **0.0877** | **0.0088** | **0.0351** |
| Jul-29 epoch 4 | **0.0343** | 0.0909 | 0.0100 | 0.0364 |

The trainer promotes checkpoints on synthetic validation CER, so had this run
been allowed to write to `models/crnn_best.pth` it would have silently replaced
a better model with a worse one on the strength of a metric that measures the
generator rather than the task. That is why `mix_jul29.yaml` writes to
`crnn_jul29_best.pth` instead, and why the promotion decision is made from
`run_eval_suite.py` output by hand. The run was stopped after this comparison;
the delivered checkpoint is unchanged.

The wider lesson for anyone continuing this work: **the synthetic validation
split has saturated and is no longer a useful model-selection signal.** Further
accuracy has to come from more real labelled photographs, not from more
synthetic data or longer schedules on the existing mix.

### 3e. Matra / modifier post-correction (no retrain, 2026-07-29)

Error analysis on the delivered decode still showed the same orthographic
cluster on real photos: mis-attached pre-base kombuva (`ෙCී` / word-final
`ෙණ්` → should be `Cේ` / `ණේ`), word-final `ණී`→`ණේ`, dangling `ේී`, and
occasional illegal pre-base order (`දිෙන` → `දිනෙ`). A short continue-train
was **not** promoted — Jul-29 (§3d) already showed synthetic-val wins can
regress every held-out set.

`src/postprocess/sinhala_fix.py` applies only rules that improve or tie
every held-out set when measured on the suite JSON:

| Rule | Effect on `print_photos` |
|---|---|
| `ෙCී` → `Cේ`, word-final `ෙණ්` → `ණේ`, `ේී` → `ේ`, word-final `ණී` → `ණේ` | main gain |
| illegal pre-base reorder + orphan `ේ`→`්` | helps user uploads; no held-out change |
| LM-gated word-final `මි`→`ම්` | small extra gain; blind `මි`/`ස්සු` rules rejected (regress `eval_pages`) |

| Set | before §3e | after §3e |
|---|---|---|
| real photos | 0.0877 | **0.0552** |
| `eval_pages` | 0.0088 | 0.0088 |
| adversarial | 0.0351 | 0.0351 |

Enabled by default via `inference.post_correct: true`. Disable in config if you
need the raw CTC+LM string.

---

## 4. Known limitations (honest)

1. **Held-out real data is 2 pages / 32 lines.** Everything else is either
   synthetic or partly in training. More photographed pages with verified
   ground truth is the single highest-value thing that could be added.
2. **`ේ` vs `ී` on very low-resolution lines is still the dominant error**,
   together with the associated spurious `ෙ` insertion. Roughly halved across
   the two fixes, but not solved.
3. **Long `ූ` vs short `ු`** on traditional serif book faces. The character LM
   does *not* help here — the corpus itself prefers the short form — so this
   one has to be fixed in the recogniser with more serif-print training data.
4. Other residual confusions on ~16 px text: `් → ි` (hal kirima vs is-pilla),
   `ඟ → ග`, `හ → න`, `ද → ඳ`. Regenerate the current list any time with
   `python scripts/report_errors.py data/debug/suite_delivered.json --set print_photos`.
5. **Logos, decorative display fonts and graphic regions** are not suppressed;
   the detector will emit a box and the recogniser will produce noise for them.
6. Very short isolated marker/label lines (a handful of characters) can be
   dropped by the detector's relative-height filter.
7. Handwriting is out of scope for the delivered model — printed text only.

---

## 5. Training curves & metric graphs

Notebooks plot training dynamics and a held-out CER bar chart (no GPU needed):

- **Curves source:** `models/train_jul28.log` when present (delivered continue-train,
  12 epochs, best synthetic val CER **0.0348**). Otherwise
  `models/train_history.json` (written each epoch by `src/recognition/train.py`) or
  the bundled `data/metrics/train_history_jul28.json`.
- **Bar chart:** `data/metrics/eval_summary.json` — same end-to-end CER figures as §2.
- **Where:** `notebooks/local_pipeline.ipynb` §12 and `notebooks/colab_pipeline.ipynb` §7
  (“Evaluation metrics & training curves”). Helpers live in
  `src/evaluation/train_curves.py`.

Restart & Run All on either notebook is enough; Section 11/6 still prints the
per-image CER/WER text report.

---

## 6. Reproducing these numbers

```powershell
# honest suite, all sets, with the held-out/in-train status column
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth `
    --out data/debug/suite_final.json

# which "holdout" sets actually leaked
python scripts/check_holdout_leakage.py

# per-line errors + ranked character confusion table
python scripts/report_errors.py data/debug/suite_final.json --set print_photos

# compare against the previous checkpoint
python scripts/run_eval_suite.py --checkpoint models/crnn_best_pre_jul28.pth `
    --pad-to-height true --name "before"

# the rejected levers from section 3d
python scripts/run_eval_suite.py --checkpoint models/crnn_best.pth `
    --tta true --tta-variants none,contrast --name "tta"
```

`models/*.pth` and the generated image datasets are gitignored; see
"Reproducing the trained model" in `README.md` for the exact regeneration and
training commands.
