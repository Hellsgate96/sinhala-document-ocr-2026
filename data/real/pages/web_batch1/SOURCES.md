# web_batch1 data provenance

## Real downloaded / curated pages

| Asset | Source | License | Notes |
|---|---|---|---|
| `page_exam_cover_2024.jpg` | User-provided Jul-26 notebook test image (exam cover style) | Authorized by user for MSc OCR research | Line crops `web_exam_line_*.png` labeled manually |
| Acts pages (60 train pages used) | Hugging Face `avishadilhara/sinhala-ocr-lk-acts-1010` | **CC-BY-4.0** | Detector-in-the-loop crops via `scripts/download_hf_acts.py` (exact line-count match only); **2275** labeled lines in `web_batch1_acts.txt` (+ light aug) |

## Synthetic / rendered (OFL fonts)

| Asset | Source | License |
|---|---|---|
| `data/synthetic_hard/` | Rendered from project corpus + `src/data/exam_style_lines.txt` using Noto Sans/Serif Sinhala, Abhaya Libre, Yaldevi (OFL), Nirmala UI | Font OFL / system fonts; text from project corpus |
| `web_holdout_line_*.png` | Fresh hard-style renders held out of training | Same |

## Training policy this round

- Prior holdout pages (`page_07_font_list`, `page_12_hitigama`) were **moved into training** (`user_batch1.txt`) to improve accuracy on decorative fonts.
- New held-out set: `data/real/labels/web_batch1_holdout.txt`.
- Mix: synthetic + page-synth + hard lines + poem aug + user_batch1 aug + web_batch1 (+ HF acts if present) via `configs/mix_web.yaml`.

## Avoided

- Wholesale scraping of copyrighted news/books without license.
- Out-of-domain palm-leaf manuscript scans as primary printed-OCR training.
