# Review panel / viva FAQ — Sinhala Document OCR

Short answers intended for an MSc review panel. Numbers match `RESULTS.md` and
`data/metrics/eval_summary.json` for the delivered checkpoint
`models/crnn_best.pth` (beam+LM + post-correct, including §3g word-accuracy push).

---

### Why CRNN+CTC instead of TrOCR / a vision transformer?

CRNN+CTC is a proven line OCR baseline, trains on a single consumer GPU with the
available data budget, and needs no character boxes thanks to CTC. Transformer
OCR was considered as an optional path in the repo notes, but the fully trained
and evaluated delivery is CRNN. For an MSc project, a well-measured classical
stack is preferable to an under-trained larger model.

### Why report CER/WER instead of “accuracy” alone?

CER/WER are standard OCR error rates and expose edit distance, not only
exact-match. This project also reports **Character Accuracy = (1−CER)×100%** and
**Word Accuracy = (1−WER)×100%** so the panel can read both conventions. Word
Accuracy on real photos (85.09%) is still lower than Character Accuracy (96.75%)
because a one-character matra error flips a whole word.

### What is the headline number I should quote?

Held-out real photographs (`print_photos`, 2 pages / 32 lines):

* CER **0.0325** → Character Accuracy **96.75%**
* WER **0.1491** → Word Accuracy **85.09%**

Do **not** quote `user_batch1` / poem CERs near zero as generalisation; those
sets are in training.

### Why can synthetic validation look better than real photos?

The generator and the real camera domain differ (fonts, blur, compression, tiny
lyric crops). A Jul-29 continue-train improved synthetic val CER to 0.0343 yet
**worsened every held-out set**. Synthetic val is useful for monitoring training
but must not be the sole promotion criterion.

### How do you know the holdout is clean?

`scripts/check_holdout_leakage.py` compares evaluation transcripts against every
training transcript. `print_photos` has no overlap. Some historically named
“holdout” label files later entered the training mix and are labelled as
in-training / partial leak in `RESULTS.md`.

### Why projection-profile detection instead of a learned detector?

It is simple, fast, and sufficient for the mostly single-column printed pages in
this project. On the evaluation suite it recovered every GT line. A learned
detector is future work for denser multi-column layouts.

### What does deskew do, and when does it matter?

Before detection, the page rotation is estimated and corrected if above a small
threshold. Skewed phone photos otherwise merge or split projection bands. Upright
scans typically report ~0° and skip a meaningful rotation.

### How is Sinhala ZWJ handled?

The charset keeps ZWJ-capable sequences so conjuncts are not silently dropped.
Encode/decode utilities and unit tests live in `src/charset.py`. Display fonts
must also support Sinhala (Nirmala / Noto) for notebook figures.

### What is beam+LM shallow fusion?

CTC prefix beam search rescores partial hypotheses with a character *n*-gram LM
trained only on training-side text:

`score = log P_ctc + lm_weight · log P_lm + insertion_bonus · |y|`

Shipped `lm_weight=0.2` improved or tied all holdouts; higher weights overfit the
news/legal corpus and hurt song lyrics.

### Why a rule-based post-corrector after a neural decoder?

Error analysis showed a stable orthographic cluster (mis-attached kombuva,
`ණී`/`ණේ`, lyric punctuation). Rules were admitted only when they improved or
tied **every** held-out set. Same checkpoint—no retrain. Blind rules that helped
lyrics but hurt `eval_pages` were rejected.

### What still fails?

Very small lyric text (`ේ`/`ී`), serif long/short `ු`/`ූ`, occasional
logo/decorative regions, and proper names on credit lines. Handwriting is out of
scope.

### How should an examiner run the demo?

1. Place `models/crnn_best.pth` in `models/`.  
2. Open `notebooks/local_pipeline.ipynb` (or Colab — see `COLAB.md`).  
3. Pick a real image: local file picker, or on Colab the default
   `files.upload()`; or set `TEST_IMAGE_PATH` / cancel for the bundled demo.  
4. **Kernel → Restart & Run All** (inference-only by default).  
5. Read: Load model → Run OCR → Predictions → Evaluation metrics → Training
   curves. For train-loop questions, enable **Optional: Training**
   (`RUN_TRAIN = True`; short demo by default).

Long methodology stays in this FAQ / `docs/Project_Report.md` / `README.md`.

### Where is the full write-up?

* `docs/Project_Report.md` (and `.docx` if generated)  
* `RESULTS.md` — methodology of evaluation and ablations  
* `README.md` — setup and reproduction commands  
