# Training label audit (2026-08-01)

Systematic check of `data/real/labels/` transcripts against line-crop images
(plus model-vs-GT disagreement scan). Synthetic / hard / page-synth labels were
spot-checked only — they are correct by construction.

## Confirmed wrong → fixed (3)

| File | Image | Was | Corrected to |
|---|---|---|---|
| `web_batch1.txt` | `web_exam_line_001.png` | `11` + grade word | `II` + grade word (Roman numerals on cover) |
| `poem_kanyawee.txt` | `poem_line_007.png` | wrong final word (ව vs ල) | cuckoo plural with ල |
| `user_batch1.txt` (+ `user_batch1_gt.json`) | `user_p01_line_001.png` | Julius + wrong surname | Julius Ratnaweera |

Augmented mirrors (`*_aug.txt`) were text-patched the same way.

## Checked OK (no change)

- Remaining `poem_kanyawee.txt` lines
- `web_batch1.txt` exam lines 2–6; `web_holdout` Arabic `11` (synthetic hard render, not the exam cover)
- `user_batch1` short/clean samples; decorative-font disagreements are recognition hardness, not label noise
- Random sample of 80 `web_batch1_acts.txt` lines + first `acts_test_*` extras — match page GT

## Not the main real-photo failure mode

Held-out `print_photos` residuals after §3f were recognition confusions
(long-uu vs short-u, prenasal ga, lyric short-crop glyphs), not training-label
noise. Word Accuracy crossed 80% via measured post-correct rules (§3g in
`RESULTS.md`).
