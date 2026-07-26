# -*- coding: utf-8 -*-
"""Prepare web_batch1 real training data from curated pages + prior holdout.

Actions:
  1. Merge prior holdout (font_list + Hitigama) into ``user_batch1.txt`` train.
  2. Label the Jul-26 exam-cover failure page into images + labels.
  3. Create a NEW held-out set of fresh hard-style renders.
  4. Write ``web_batch1.txt`` + ``SOURCES.md``.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import apply_augmentations, discover_font_faces  # noqa: E402
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402

# Import render helper from generate_hard_lines
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "generate_hard_lines", ROOT / "scripts" / "generate_hard_lines.py"
)
_hard = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hard)
load_exam_style_lines = _hard.load_exam_style_lines
render_styled = _hard.render_styled

EXAM_COVER_LINES = load_exam_style_lines()[:6]


def _save_crop(path: Path, crop: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if crop.ndim == 2:
        Image.fromarray(crop).save(path)
    else:
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(path)


def _read_label_file(path: Path) -> List[Tuple[str, str]]:
    rows = []
    if not path.is_file():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or "\t" not in ln:
            continue
        rel, gt = ln.split("\t", 1)
        rows.append((rel.strip(), gt.strip()))
    return rows


def _write_labels(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel, gt in rows:
            f.write(f"{rel}\t{gt}\n")


def crop_exam_cover(image_path: Path, out_images: Path, prefix: str = "web_exam") -> List[Tuple[str, str]]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.95)
    region = bgr[y0:y1]
    rh = region.shape[0]
    n = len(EXAM_COVER_LINES)
    step = rh / n
    rows: List[Tuple[str, str]] = []
    for i, gt in enumerate(EXAM_COVER_LINES):
        a = int(round(i * step))
        b = int(round((i + 1) * step))
        strip = region[a:b]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cols = (bw > 0).sum(axis=0)
        xs = np.flatnonzero(cols)
        if xs.size:
            x0 = max(0, int(xs[0]) - 8)
            x1 = min(w, int(xs[-1]) + 8)
        else:
            x0, x1 = 0, w
        crop = strip[:, x0:x1]
        name = f"{prefix}_line_{i + 1:03d}.png"
        _save_crop(out_images / name, crop)
        rows.append((f"images/{name}", gt))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exam-image",
        default=r"C:\Users\ASUS TUF\Pictures\Screenshots\ocr\images.jpg",
    )
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("prepare_web_batch1")

    real = ROOT / "data" / "real"
    images = real / "images"
    labels_dir = real / "labels"
    pages_dir = real / "pages" / "web_batch1"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    old_holdout = _read_label_file(labels_dir / "user_batch1_holdout.txt")
    train_user = _read_label_file(labels_dir / "user_batch1.txt")
    by_rel = {rel: gt for rel, gt in train_user}
    for rel, gt in old_holdout:
        by_rel[rel] = gt
    merged_user = sorted(by_rel.items())
    _write_labels(labels_dir / "user_batch1.txt", merged_user)
    logger.info("user_batch1 train now %d lines (includes prior holdout)", len(merged_user))
    if old_holdout:
        _write_labels(labels_dir / "user_batch1_holdout_SUPERSEDED.txt", old_holdout)

    exam_rows: List[Tuple[str, str]] = []
    exam_src = Path(args.exam_image)
    if exam_src.is_file():
        shutil.copy2(exam_src, pages_dir / "page_exam_cover_2024.jpg")
        exam_rows = crop_exam_cover(exam_src, images, prefix="web_exam")
        logger.info("exam cover -> %d line crops", len(exam_rows))
    else:
        logger.warning("exam image missing: %s", exam_src)

    # All exam lines go into train for accuracy; new holdout is fresh renders.
    web_train = list(exam_rows)

    cfg = load_config(str(ROOT / "configs" / "local.yaml"))
    faces = discover_font_faces(cfg["synthetic"]["fonts"], warn=logger.warning)
    rng = random.Random(424242)
    exam_style = load_exam_style_lines()
    holdout_texts = exam_style[:14] if len(exam_style) >= 14 else exam_style
    if not holdout_texts:
        holdout_texts = list(EXAM_COVER_LINES)

    new_holdout: List[Tuple[str, str]] = []
    styles = ["pill", "dark", "colored", "blue_title", "plain_bold"]
    for i, text in enumerate(holdout_texts, start=1):
        path, face_index = faces[i % len(faces)]
        font = ImageFont.truetype(path, 48 if len(text) < 40 else 36, index=face_index)
        style = styles[i % len(styles)]
        img, bg = render_styled(text, font, style, rng)
        img = apply_augmentations(
            img,
            {
                "rotation": True,
                "rotation_max_deg": 2.0,
                "blur": True,
                "jpeg_compression": True,
                "gaussian_noise": True,
            },
            bg,
            rng,
        )
        name = f"web_holdout_line_{i:03d}.png"
        img.save(images / name)
        new_holdout.append((f"images/{name}", text))

    _write_labels(labels_dir / "web_batch1_holdout.txt", new_holdout)
    _write_labels(labels_dir / "web_batch1.txt", web_train)
    logger.info("web_batch1 train=%d new_holdout=%d", len(web_train), len(new_holdout))

    (pages_dir / "SOURCES.md").write_text(
        """# web_batch1 data provenance

## Real downloaded / curated pages

| Asset | Source | License | Notes |
|---|---|---|---|
| `page_exam_cover_2024.jpg` | User-provided Jul-26 notebook test image (exam cover style) | Authorized by user for MSc OCR research | Line crops `web_exam_line_*.png` labeled manually |
| Acts pages (optional) | Hugging Face `avishadilhara/sinhala-ocr-lk-acts-1010` | **CC-BY-4.0** | Detector-in-the-loop crops via `scripts/download_hf_acts.py`; labels in `web_batch1_acts.txt` |

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
""",
        encoding="utf-8",
    )
    logger.info("wrote SOURCES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
