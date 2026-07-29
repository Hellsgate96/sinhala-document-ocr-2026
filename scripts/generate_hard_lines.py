# -*- coding: utf-8 -*-
"""Render hard-case Sinhala line crops for recognition training.

Covers failure modes seen on real exam/cover pages and decorative fonts:
  - dark background / white (or light) text
  - solid coloured bars / pill shapes behind text
  - large bold title sizes
  - mixed English + Sinhala
  - short centred headings
  - traditional serif book print on grey paper (jul27: photographed poem page
    showed vowel-sign drops and Tha/Na, Ta/Va confusions on this style)
  - tiny low-resolution text (jul27: small footer line came out as garbage) -
    simulated by crushing rendered lines down to 10-22 px

Writes ``data/synthetic_hard/images/*.png`` and ``train_labels.txt``.

A dedicated all-tiny set (jul28: a lyrics card whose every line is ~15-20 px
tall drove most of the remaining real-photo CER) is the same generator with
``--tiny-ratio 1.0``:

    python scripts/generate_hard_lines.py --num 10000 --out data/synthetic_small \\
        --name-prefix small --seed 20260729 --tiny-ratio 1.0 \\
        --tiny-min-h 11 --tiny-max-h 26
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.synthetic_generator import (  # noqa: E402
    apply_augmentations,
    compose_line,
    discover_font_faces,
    load_corpus,
    load_word_lists,
)
from src.utils.common import configure_stdout_utf8, get_logger, load_config  # noqa: E402

Color = Tuple[int, int, int]


def load_exam_style_lines(path: str | None = None) -> List[str]:
    p = path or os.path.join(ROOT, "src", "data", "exam_style_lines.txt")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _pill_bg(w: int, h: int, pill: Color, outer: Color) -> Image.Image:
    img = Image.new("RGB", (w, h), outer)
    draw = ImageDraw.Draw(img)
    margin = max(2, h // 8)
    radius = max(6, h // 3)
    draw.rounded_rectangle(
        [margin, margin, w - margin - 1, h - margin - 1],
        radius=radius,
        fill=pill,
    )
    return img


def render_styled(
    text: str,
    font: ImageFont.FreeTypeFont,
    style: str,
    rng: random.Random,
) -> Tuple[Image.Image, Color]:
    scratch = Image.new("RGB", (8, 8), (255, 255, 255))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad_y = rng.randint(6, max(8, int(th * 0.4)))
    pad_x = rng.randint(10, 36)
    extra = int(tw * rng.uniform(0.05, 0.55)) if rng.random() < 0.5 else 0
    w = tw + 2 * pad_x + extra
    h = th + 2 * pad_y
    x0 = pad_x + (extra // 2 if rng.random() < 0.7 else 0)
    y0 = pad_y

    if style == "lyrics_small":
        # Low-res lyrics-site card: white-ish page, soft dark text, later
        # crushed to a small pixel height + JPEG (see main loop).
        g = rng.randint(238, 255)
        bg = (g, g, g - rng.randint(0, 6))
        ink = rng.randint(0, 60)
        fg = (ink, ink, ink + rng.randint(0, 30))
        img = Image.new("RGB", (w, h), bg)
    elif style == "book_serif":
        # Photographed book/poem print: grey paper, soft dark ink.
        g = rng.randint(185, 225)
        bg = (g, g + rng.randint(-4, 4), g + rng.randint(-4, 4))
        ink = rng.randint(15, 75)
        fg = (ink, ink + rng.randint(-6, 6), ink + rng.randint(-6, 6))
        img = Image.new("RGB", (w, h), bg)
    elif style == "dark":
        bg = (rng.randint(8, 40), rng.randint(8, 40), rng.randint(8, 50))
        fg = (rng.randint(220, 255), rng.randint(220, 255), rng.randint(220, 255))
        img = Image.new("RGB", (w, h), bg)
    elif style == "colored":
        bg = (rng.randint(160, 230), rng.randint(170, 235), rng.randint(200, 255))
        fg = (rng.randint(0, 40), rng.randint(0, 40), rng.randint(0, 55))
        img = Image.new("RGB", (w, h), bg)
    elif style == "pill":
        outer = (255, 255, 255)
        pill = (rng.randint(180, 230), rng.randint(185, 235), rng.randint(200, 255))
        fg = (rng.randint(0, 35), rng.randint(0, 35), rng.randint(0, 50))
        img = _pill_bg(w, h, pill, outer)
        bg = outer
    elif style == "blue_title":
        bg = (255, 255, 255)
        fg = (rng.randint(10, 40), rng.randint(30, 80), rng.randint(100, 180))
        img = Image.new("RGB", (w, h), bg)
    else:
        bg = (rng.randint(235, 255), rng.randint(235, 255), rng.randint(235, 255))
        fg = (rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 30))
        img = Image.new("RGB", (w, h), bg)

    draw = ImageDraw.Draw(img)
    draw.text((x0 - bbox[0], y0 - bbox[1]), text, font=font, fill=fg)
    return img, bg


def degrade_low_res(img: Image.Image, rng: random.Random,
                    min_h: int = 10, max_h: int = 22,
                    keep_small_prob: float = 0.5) -> Image.Image:
    """Simulate tiny/low-resolution text (small footers, thumbnails, distant
    photos): crush the render down to a few pixels of x-height, so the model
    sees the same soft, aliased strokes at train time that the detector's
    small crops have at inference time.

    Half the time the crushed image is returned as-is, so ``OCRLineDataset``
    performs the upscale to the model height exactly the way inference does
    (``inference.pad_to_height=false`` -> ``resize_keep_height``). The other
    half is scaled back up here, which keeps some samples at the blur level of
    an already-upscaled scan. Aspect ratio is preserved either way, so the
    width after the model-height resize is unchanged.
    """
    w, h = img.size
    target = rng.randint(min_h, max_h)
    if h <= target:
        return img
    scale = target / float(h)
    down = img.resize((max(8, int(w * scale)), target),
                      Image.BILINEAR if rng.random() < 0.5 else Image.LANCZOS)
    if rng.random() < keep_small_prob:
        return down
    return down.resize((w, h), Image.BILINEAR)


SERIF_HINTS = ("serif", "abhaya", "iskpota")


def _serif_faces(faces):
    hits = [f for f in faces if any(hint in os.path.basename(f[0]).lower() for hint in SERIF_HINTS)]
    return hits or faces


# Jul-28 real-page confusions (low-res lyrics/poem photos): word-final ණේ/නේ
# misread as ණී/නී, hal-m ම් vs මි, short/long -u (ු/ූ), ඩ/ඬ and ද/ඳ pairs.
# Oversampling corpus words that contain these graphemes gives CTC many more
# clean examples of the exact distinctions that were failing.
CONFUSION_SUBSTRINGS = ("ණේ", "නේ", "ම්", "මි", "ූ", "ඬ", "ඳ", "ඟ", "ැයි")

# Word-FINAL graphemes, which is where the jul28/jul29 confusions actually live:
# the kombuva of a word-final "ේ" is the sign that gets re-attached to the wrong
# consonant, and a word-final hal kirima "්" is what gets read as "ි".
CONFUSION_SUFFIXES = ("ේ", "්", "ූ", "ී")

# Lyrics/poem pages end refrain lines with dot-runs and repeat marks;
# these sequences never appeared in the training text, so the model emitted
# garbage like ".Ll" for "...//".
NOTATION_SUFFIXES = ("...//", "...//", "..//", "...", "..", ".//")


def _is_confusable(word: str) -> bool:
    stripped = word.strip(".,:;!?()[]\"'")
    if any(s in word for s in CONFUSION_SUBSTRINGS):
        return True
    return bool(stripped) and stripped.endswith(CONFUSION_SUFFIXES)


def build_confusion_pool(corpus: List[str], words: List[str]) -> List[str]:
    """Words from the corpus/word-lists containing hard confusion graphemes."""
    pool: List[str] = []
    seen = set()
    for line in corpus:
        for w in line.split():
            if w in seen or len(w) < 2:
                continue
            if _is_confusable(w):
                seen.add(w)
                pool.append(w)
    for w in words:
        if w not in seen and _is_confusable(w):
            seen.add(w)
            pool.append(w)
    return pool


def compose_confusion_line(pool: List[str], rng: random.Random) -> str:
    n = rng.randint(3, 7)
    return " ".join(rng.choice(pool) for _ in range(n))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hard-case Sinhala OCR lines.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--num", type=int, default=4000)
    parser.add_argument("--out", default="data/synthetic_hard")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--tiny-ratio", type=float, default=None,
        help="force this fraction of every style through the low-res degradation "
             "(default: per-style 0.35 book_serif / 0.15 others). Use 1.0 to build "
             "a dedicated small-text set.",
    )
    parser.add_argument("--tiny-min-h", type=int, default=10)
    parser.add_argument("--tiny-max-h", type=int, default=22)
    parser.add_argument(
        "--name-prefix", default="hard",
        help="image filename prefix (keeps separate sets from colliding)",
    )
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("generate_hard_lines")
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    cfg = load_config(cfg_path)
    syn = cfg["synthetic"]
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    word_sources = [cfg["paths"]["word_list"]]
    if cfg["paths"].get("form_vocab"):
        word_sources.append(cfg["paths"]["form_vocab"])
    words = load_word_lists(word_sources, warn=logger.warning)
    exam_lines = load_exam_style_lines()
    corpus = list(exam_lines) + list(load_corpus(cfg["paths"].get("corpus"), warn=logger.warning))

    faces = discover_font_faces(syn["fonts"], warn=logger.warning)
    serif_faces = _serif_faces(faces)
    confusion_pool = build_confusion_pool(corpus, words)
    logger.info(f"confusion word pool: {len(confusion_pool)} words")
    sizes = list(syn["font_sizes"]) + [80, 96]
    styles = ["dark", "colored", "pill", "blue_title", "plain_bold", "book_serif", "lyrics_small"]
    style_weights = [0.11, 0.13, 0.15, 0.09, 0.12, 0.22, 0.18]
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    augment = dict(syn.get("augment") or {})
    font_cache = {}

    def get_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
        key = (path, index, size)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(path, size, index=index)
        return font_cache[key]

    labels: List[Tuple[str, str]] = []
    for i in range(args.num):
        if confusion_pool and rng.random() < 0.22:
            text = compose_confusion_line(confusion_pool, rng)
        elif exam_lines and rng.random() < 0.25:
            text = rng.choice(exam_lines)
        else:
            text = compose_line(
                words,
                1,
                10,
                rng,
                numeric_ratio=0.15,
                mixed_ratio=0.18,
                corpus=corpus,
                corpus_ratio=0.75,
            )
        # Refrain/dot-run notation seen on real lyric/poem pages.
        if rng.random() < 0.15:
            text = text.rstrip(".") + rng.choice(NOTATION_SUFFIXES)
        style = rng.choices(styles, weights=style_weights, k=1)[0]
        face_pool = serif_faces if style == "book_serif" else faces
        path, face_index = rng.choice(face_pool)
        size_choices = [s for s in sizes if s >= 32] if len(text) < 40 else sizes
        font = get_font(path, face_index, rng.choice(size_choices or sizes))
        img, bg = render_styled(text, font, style, rng)
        img = apply_augmentations(img, augment, bg, rng)
        # Tiny-text degradation on a fraction of every style (small footers,
        # low-res photos) - the jul27 real page's footer line failed on this.
        if args.tiny_ratio is not None:
            if rng.random() < args.tiny_ratio:
                img = degrade_low_res(img, rng, min_h=args.tiny_min_h, max_h=args.tiny_max_h)
        elif style == "lyrics_small":
            img = degrade_low_res(img, rng, min_h=12, max_h=24)
        elif rng.random() < (0.35 if style == "book_serif" else 0.15):
            img = degrade_low_res(img, rng)
        rel = f"images/{args.name_prefix}_{i:06d}.png"
        img.save(os.path.join(out_dir, rel.replace("/", os.sep)))
        labels.append((rel, text))
        if (i + 1) % 500 == 0:
            logger.info(f"rendered {i + 1}/{args.num}")

    labels_path = os.path.join(out_dir, "train_labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for rel, text in labels:
            f.write(f"{rel}\t{text}\n")
    logger.info(f"wrote {len(labels)} hard lines -> {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
