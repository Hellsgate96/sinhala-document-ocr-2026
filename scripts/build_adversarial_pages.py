"""Hand-built adversarial acceptance-test pages (v3 domain-gap fix, item 9).

Three pages built directly with PIL (not the randomized page_synth layouts),
each targeting one of the failure modes reported by the user:

  1. ``adv_card``   - decorative bordered greeting-card style page with a
     faint watermark and short centered lines.
  2. ``adv_article`` - LaTeX-article-style page: title, author/date line,
     section heading, justified-looking paragraph body.
  3. ``adv_photo``  - plain paragraph page put through heavier camera-photo
     degradation (perspective + defocus blur + shadow) than the average
     training/eval sample, simulating a poorly-taken phone photo.

Ground truth text is deterministic (fixed corpus line indices), so this
script is fully reproducible. Writes ``data/eval_real/adversarial/<name>.png``
+ ``<name>.gt.txt``; run through the full pipeline with
``scripts/run_realistic_eval.py --images-dir data/eval_real/adversarial``.

Usage:
    python scripts/build_adversarial_pages.py --config configs/local.yaml
"""
from __future__ import annotations

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.page_synth import _draw_border, _draw_watermark, _measure, aug_defocus_blur, aug_perspective, aug_rotate, aug_shadow
from src.data.synthetic_generator import aug_jpeg, discover_font_faces, load_corpus, random_date
from src.utils.common import configure_stdout_utf8, get_logger, load_config


def _font(faces, size, index=0):
    path, face_idx = faces[index % len(faces)]
    return ImageFont.truetype(path, size, index=face_idx)


def build_card(corpus, faces, out_dir):
    w, h = 1100, 1500
    bg = (250, 246, 238)
    fg = (60, 20, 20)
    canvas = Image.new("RGB", (w, h), bg)
    rng = random.Random(11)
    _draw_border(canvas, rng, (130, 40, 40))
    _draw_watermark(canvas, rng, bg[0])
    draw = ImageDraw.Draw(canvas)

    lines = [corpus[5], corpus[42], corpus[123], corpus[7]]
    font_sizes = [58, 44, 44, 50]
    y = int(h * 0.28)
    for text, size in zip(lines, font_sizes):
        font = _font(faces, size, index=1)
        tw, th = _measure(draw, text, font)
        draw.text(((w - tw) // 2, y), text, font=font, fill=fg)
        y += int(th * 2.0)

    canvas = aug_rotate(canvas, 1.5, bg, rng)
    canvas = aug_shadow(canvas, rng)
    canvas = aug_jpeg(canvas, rng)
    return canvas, lines


def build_article(corpus, faces, out_dir):
    w, h = 1275, 1650  # ~A4 @ 150dpi
    bg = (255, 255, 255)
    fg = (10, 10, 10)
    canvas = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(canvas)
    margin = 130
    y = 90

    title = corpus[300]
    title_font = _font(faces, 52, index=0)
    tw, th = _measure(draw, title, title_font)
    draw.text(((w - tw) // 2, y), title, font=title_font, fill=fg)
    y += th + 24

    byline = f"{corpus[301][:24]}   -   {random_date(random.Random(3))}"
    sub_font = _font(faces, 28, index=2)
    sw, sh = _measure(draw, byline, sub_font)
    draw.text(((w - sw) // 2, y), byline, font=sub_font, fill=fg)
    y += sh + 40

    heading = corpus[9]
    head_font = _font(faces, 36, index=0)
    draw.text((margin, y), heading, font=head_font, fill=fg)
    y += _measure(draw, heading, head_font)[1] + 20

    body_font = _font(faces, 30, index=3)
    lines = []
    body_idx = [10, 11, 12, 13, 14, 15, 16, 17]
    for idx in body_idx:
        text = corpus[idx]
        tw, th = _measure(draw, text, body_font)
        if tw > w - 2 * margin:
            words = text.split()
            text = " ".join(words[: max(1, int(len(words) * (w - 2 * margin) / tw))])
        draw.text((margin, y), text, font=body_font, fill=fg)
        lines.append(text)
        y += int(_measure(draw, text, body_font)[1] * 1.6)
        if y > h - 100:
            break

    rng = random.Random(22)
    canvas = aug_rotate(canvas, 1.0, bg, rng)
    canvas = aug_jpeg(canvas, rng)
    return canvas, [title, byline, heading] + lines


def build_photo(corpus, faces, out_dir):
    w, h = 1200, 1600
    bg = (245, 245, 240)
    fg = (25, 25, 25)
    canvas = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(canvas)
    margin = 110
    y = 100
    body_font = _font(faces, 34, index=4 % len(faces))
    lines = []
    for idx in range(400, 411):
        text = corpus[idx]
        tw, th = _measure(draw, text, body_font)
        if tw > w - 2 * margin:
            words = text.split()
            text = " ".join(words[: max(1, int(len(words) * (w - 2 * margin) / tw))])
        draw.text((margin, y), text, font=body_font, fill=fg)
        lines.append(text)
        y += int(_measure(draw, text, body_font)[1] * 1.6)
        if y > h - 120:
            break

    rng = random.Random(33)
    # Heavier-than-average camera degradation on purpose (worst-case phone photo).
    canvas = aug_rotate(canvas, 3.5, bg, rng)
    canvas = aug_perspective(canvas, bg, rng)
    canvas = aug_shadow(canvas, rng)
    canvas = aug_defocus_blur(canvas, rng)
    canvas = aug_jpeg(canvas, rng)
    canvas = aug_jpeg(canvas, rng)
    return canvas, lines


def main():
    parser = argparse.ArgumentParser(description="Build hand-crafted adversarial acceptance-test pages.")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--out", default="data/eval_real/adversarial")
    args = parser.parse_args()

    configure_stdout_utf8()
    logger = get_logger("build_adversarial_pages")
    cfg = load_config(args.config)
    corpus = load_corpus(cfg["paths"].get("corpus"), warn=logger.warning)
    faces = discover_font_faces(cfg["synthetic"]["fonts"], warn=logger.warning)

    os.makedirs(args.out, exist_ok=True)
    builders = {"adv_card": build_card, "adv_article": build_article, "adv_photo": build_photo}
    for name, fn in builders.items():
        img, lines = fn(corpus, faces, args.out)
        img_path = os.path.join(args.out, f"{name}.png")
        gt_path = os.path.join(args.out, f"{name}.gt.txt")
        img.convert("RGB").save(img_path)
        with open(gt_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logger.info(f"wrote {img_path} ({len(lines)} gt lines)")


if __name__ == "__main__":
    main()
