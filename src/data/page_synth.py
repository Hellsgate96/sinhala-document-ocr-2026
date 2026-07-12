"""Full synthetic PAGE composition + detector-in-the-loop training-crop
extraction (v3 domain-gap fix).

The line-level generator (:mod:`src.data.synthetic_generator`) renders one
tightly-cropped line at a time; the real inference pipeline instead runs a
projection-profile detector over a whole photographed PAGE and crops whatever
box the detector produces (which may be mis-padded, merge a descender from
the row above, or clip a border rule). Training only on idealised line crops
creates a train/inference distribution gap.

This module renders whole pages (paragraphs, decorative bordered cards,
poems, mixed Sinhala/English/numeric lines, letterhead-style pages), applies
page-level phone-camera degradation, and can run the *same* detector used at
inference time over the rendered page to produce (crop, transcript) training
pairs from the detector's actual output boxes - see
:func:`generate_detector_in_the_loop`.  It is also used to build a
realistic, non-leaking held-out evaluation set - see
:func:`generate_eval_pages`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.synthetic_generator import (
    aug_brightness_contrast,
    aug_defocus_blur,
    aug_gaussian_noise,
    aug_jpeg,
    aug_moire,
    aug_paper_texture,
    aug_perspective,
    aug_rotate,
    aug_shadow,
    compose_line,
    discover_font_faces,
    random_date,
    random_mixed_token,
    random_number,
)

LAYOUTS = ("paragraph", "bordered_card", "poem", "mixed_en_si", "letterhead")


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------
def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _page_colors(rng: random.Random) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    bg = rng.randint(224, 255)
    bg_color = (bg, bg, min(255, bg + rng.randint(-6, 6)))
    style = rng.random()
    if style < 0.7:
        fg = rng.randint(0, 60)
        text_color = (fg, fg, fg)
    else:
        base = [rng.randint(0, 50) for _ in range(3)]
        base[rng.randint(0, 2)] = rng.randint(30, 100)
        text_color = tuple(base)
    return text_color, bg_color


def _wrap_paragraph(draw, words: List[str], font, max_width: int,
                    n_lines: int) -> List[str]:
    """Greedy word-wrap ``words`` into at most ``n_lines`` lines of <= max_width px."""
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        tw, _ = _measure(draw, trial, font)
        if tw > max_width and cur:
            lines.append(" ".join(cur))
            cur = [w]
            if len(lines) >= n_lines:
                return lines
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines[:n_lines]


def _paragraph_words(rng: random.Random, corpus: Sequence[str], target: int) -> List[str]:
    words: List[str] = []
    guard = 0
    while len(words) < target and guard < 40:
        words.extend(rng.choice(corpus).split())
        guard += 1
    return words


def _draw_border(canvas: Image.Image, rng: random.Random, color: Tuple[int, int, int]) -> None:
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)
    margin = int(min(w, h) * rng.uniform(0.025, 0.045))
    thickness = rng.randint(3, 7)
    for t in range(thickness):
        draw.rectangle(
            [margin + t, margin + t, w - margin - t, h - margin - t], outline=color
        )
    if rng.random() < 0.5:
        inner = margin + thickness + rng.randint(6, 16)
        draw.rectangle([inner, inner, w - inner, h - inner], outline=color)


def _draw_watermark(canvas: Image.Image, rng: random.Random, bg_gray: int) -> None:
    """Low-contrast circular/logo-like blob - must survive detector suppression."""
    w, h = canvas.size
    arr = np.asarray(canvas).astype(np.float32)
    cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.3, 0.7)
    radius = min(w, h) * rng.uniform(0.15, 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    delta = rng.uniform(6, 16) * np.clip(1.0 - dist / radius, 0, 1)
    sign = -1 if rng.random() < 0.5 else 1
    arr -= sign * delta[:, :, None]
    canvas.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))


# ---------------------------------------------------------------------------
# Layout builders: each returns (image, transcripts_top_to_bottom)
# ---------------------------------------------------------------------------
def _get_font(cache, path, index, size):
    key = (path, index, size)
    if key not in cache:
        cache[key] = ImageFont.truetype(path, size, index=index)
    return cache[key]


@dataclass
class PageContext:
    rng: random.Random
    corpus: Sequence[str]
    faces: Sequence[Tuple[str, int]]
    font_sizes: Sequence[int]
    font_cache: Dict = field(default_factory=dict)
    _words: Optional[List[str]] = None

    def pick_font(self, size: Optional[int] = None) -> ImageFont.FreeTypeFont:
        path, idx = self.rng.choice(self.faces)
        size = size or self.rng.choice(self.font_sizes)
        return _get_font(self.font_cache, path, idx, size)

    @property
    def words(self) -> List[str]:
        """Flat word list derived from the corpus (fallback vocabulary for the
        token-by-token compose_line() path: numeric/mixed/random tokens)."""
        if self._words is None:
            seen = set()
            out: List[str] = []
            for line in self.corpus:
                for w in line.split():
                    if w not in seen:
                        seen.add(w)
                        out.append(w)
            self._words = out or ["පෙළ"]
        return self._words


def _layout_paragraph(ctx: PageContext, w: int, h: int, text_color, bg_color) -> List[str]:
    canvas = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(canvas)
    margin = int(w * ctx.rng.uniform(0.07, 0.12))
    y = int(h * ctx.rng.uniform(0.06, 0.1))
    transcripts: List[str] = []

    if ctx.rng.random() < 0.55:
        title = compose_line(["Gazette", "Notice", "Report"], 2, 4, ctx.rng, corpus=ctx.corpus, corpus_ratio=0.9)
        title_font = ctx.pick_font(size=ctx.rng.choice([44, 52, 60]))
        tw, th = _measure(draw, title, title_font)
        draw.text(((w - tw) // 2, y), title, font=title_font, fill=text_color)
        transcripts.append(title)
        y += th + int(h * 0.03)

    body_font = ctx.pick_font(size=ctx.rng.choice([28, 32, 36, 40]))
    n_lines = ctx.rng.randint(6, 13)
    words = _paragraph_words(ctx.rng, ctx.corpus, n_lines * 7)
    lines = _wrap_paragraph(draw, words, body_font, w - 2 * margin, n_lines)
    for line in lines:
        _, lh = _measure(draw, line, body_font)
        if y + lh > h - int(h * 0.05):
            break
        draw.text((margin, y), line, font=body_font, fill=text_color)
        transcripts.append(line)
        y += int(lh * ctx.rng.uniform(1.45, 1.75))
    return canvas, transcripts


def _layout_letterhead(ctx: PageContext, w: int, h: int, text_color, bg_color):
    canvas, transcripts = _layout_paragraph_base(ctx, w, h, text_color, bg_color, with_title=False)
    return canvas, transcripts


def _layout_paragraph_base(ctx, w, h, text_color, bg_color, with_title):
    canvas = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(canvas)
    margin = int(w * ctx.rng.uniform(0.07, 0.12))
    y = int(h * ctx.rng.uniform(0.05, 0.08))
    transcripts: List[str] = []

    title_font = ctx.pick_font(size=ctx.rng.choice([46, 54, 62]))
    title = compose_line(ctx.words, 2, 5, ctx.rng, corpus=ctx.corpus, corpus_ratio=1.0)
    tw, th = _measure(draw, title, title_font)
    draw.text(((w - tw) // 2, y), title, font=title_font, fill=text_color)
    transcripts.append(title)
    y += th + int(h * 0.015)

    sub_font = ctx.pick_font(size=ctx.rng.choice([24, 28]))
    subtitle = f"{random_date(ctx.rng)}  -  {random_mixed_token(ctx.rng)}"
    sw, sh = _measure(draw, subtitle, sub_font)
    draw.text(((w - sw) // 2, y), subtitle, font=sub_font, fill=text_color)
    transcripts.append(subtitle)
    y += sh + int(h * 0.02)

    rule_y = y + int(h * 0.015)
    draw.line([(margin, rule_y), (w - margin, rule_y)], fill=text_color, width=2)
    y = rule_y + int(h * 0.03)

    body_font = ctx.pick_font(size=ctx.rng.choice([26, 30, 34]))
    n_lines = ctx.rng.randint(5, 10)
    words = _paragraph_words(ctx.rng, ctx.corpus, n_lines * 7)
    lines = _wrap_paragraph(draw, words, body_font, w - 2 * margin, n_lines)
    for line in lines:
        _, lh = _measure(draw, line, body_font)
        if y + lh > h - int(h * 0.05):
            break
        draw.text((margin, y), line, font=body_font, fill=text_color)
        transcripts.append(line)
        y += int(lh * ctx.rng.uniform(1.45, 1.75))
    return canvas, transcripts


def _layout_bordered_card(ctx: PageContext, w: int, h: int, text_color, bg_color):
    canvas = Image.new("RGB", (w, h), bg_color)
    border_color = text_color if ctx.rng.random() < 0.5 else (
        ctx.rng.randint(80, 160), ctx.rng.randint(20, 90), ctx.rng.randint(20, 90)
    )
    _draw_border(canvas, ctx.rng, border_color)
    if ctx.rng.random() < 0.6:
        _draw_watermark(canvas, ctx.rng, bg_color[0])
    draw = ImageDraw.Draw(canvas)

    n_lines = ctx.rng.randint(2, 5)
    total_h = 0
    lines = []
    fonts = []
    for _ in range(n_lines):
        font = ctx.pick_font(size=ctx.rng.choice([36, 44, 52, 60]))
        text = compose_line(ctx.words, 1, 4, ctx.rng, corpus=ctx.corpus, corpus_ratio=1.0)
        _, lh = _measure(draw, text, font)
        lines.append(text)
        fonts.append(font)
        total_h += int(lh * 1.9)

    y = max(int(h * 0.1), (h - total_h) // 2)
    transcripts = []
    for text, font in zip(lines, fonts):
        tw, th = _measure(draw, text, font)
        if y + th > h * 0.92:
            break
        draw.text(((w - tw) // 2, y), text, font=font, fill=text_color)
        transcripts.append(text)
        y += int(th * 1.9)
    return canvas, transcripts


def _layout_poem(ctx: PageContext, w: int, h: int, text_color, bg_color):
    canvas = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(canvas)
    n_stanzas = ctx.rng.randint(2, 3)
    font = ctx.pick_font(size=ctx.rng.choice([32, 38, 44]))
    y = int(h * ctx.rng.uniform(0.08, 0.14))
    transcripts = []
    for _s in range(n_stanzas):
        n_lines = ctx.rng.randint(3, 5)
        for _ in range(n_lines):
            text = compose_line(ctx.words, 1, 5, ctx.rng, corpus=ctx.corpus, corpus_ratio=1.0)
            tw, th = _measure(draw, text, font)
            if y + th > h * 0.92:
                return canvas, transcripts
            draw.text(((w - tw) // 2, y), text, font=font, fill=text_color)
            transcripts.append(text)
            y += int(th * ctx.rng.uniform(1.5, 1.8))
        y += int(h * 0.04)  # stanza gap
    return canvas, transcripts


def _layout_mixed_en_si(ctx: PageContext, w: int, h: int, text_color, bg_color):
    canvas = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(canvas)
    margin = int(w * ctx.rng.uniform(0.08, 0.12))
    y = int(h * ctx.rng.uniform(0.06, 0.1))
    n_lines = ctx.rng.randint(6, 12)
    font = ctx.pick_font(size=ctx.rng.choice([28, 32, 36]))
    transcripts = []
    for _ in range(n_lines):
        text = compose_line(ctx.words, 3, 9, ctx.rng, numeric_ratio=0.28, mixed_ratio=0.28,
                            corpus=ctx.corpus, corpus_ratio=0.35)
        tw, th = _measure(draw, text, font)
        if y + th > h * 0.94:
            break
        draw.text((margin, y), text, font=font, fill=text_color)
        transcripts.append(text)
        y += int(th * ctx.rng.uniform(1.4, 1.65))
    return canvas, transcripts


_LAYOUT_FUNCS = {
    "paragraph": _layout_paragraph,
    "bordered_card": _layout_bordered_card,
    "poem": _layout_poem,
    "mixed_en_si": _layout_mixed_en_si,
    "letterhead": _layout_letterhead,
}


def make_page(rng: random.Random, corpus: Sequence[str], faces: Sequence[Tuple[str, int]],
             font_sizes: Sequence[int], layout: Optional[str] = None,
             page_w: Optional[int] = None, page_h: Optional[int] = None,
             font_cache: Optional[Dict] = None) -> Tuple[Image.Image, List[str], str]:
    """Render one synthetic page. Returns (RGB image, transcripts top->bottom, layout name)."""
    layout = layout or rng.choice(LAYOUTS)
    page_w = page_w or rng.choice([1100, 1240, 1400])
    page_h = page_h or rng.choice([1500, 1650, 1800])
    text_color, bg_color = _page_colors(rng)
    ctx = PageContext(rng=rng, corpus=corpus, faces=faces, font_sizes=font_sizes,
                      font_cache=font_cache if font_cache is not None else {})
    fn = _LAYOUT_FUNCS[layout]
    canvas, transcripts = fn(ctx, page_w, page_h, text_color, bg_color)
    return canvas, [t for t in transcripts if t.strip()], layout


# ---------------------------------------------------------------------------
# Page-level phone-capture augmentation
# ---------------------------------------------------------------------------
def apply_page_augmentations(img: Image.Image, rng: random.Random,
                             bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if rng.random() < 0.55:
        img = aug_rotate(img, 1.6, bg_color, rng)
    if rng.random() < 0.45:
        # Page-scale perspective: much smaller fraction than the line-crop
        # default (the same fraction of a ~1500px page would jitter a corner
        # by 50-100px and can visually merge adjacent text lines).
        img = aug_perspective(img, bg_color, rng, mx_range=(0.003, 0.015), my_range=(0.003, 0.018))
    if rng.random() < 0.6:
        img = aug_shadow(img, rng)
    if rng.random() < 0.5:
        img = aug_brightness_contrast(img, rng)
    if rng.random() < 0.45:
        img = aug_defocus_blur(img, rng)
    if rng.random() < 0.4:
        img = aug_paper_texture(img, rng)
    if rng.random() < 0.4:
        img = aug_gaussian_noise(img, rng)
    if rng.random() < 0.06:
        img = aug_moire(img, rng)
    if rng.random() < 0.55:
        img = aug_jpeg(img, rng)
        if rng.random() < 0.2:
            img = aug_jpeg(img, rng)
    return img


# ---------------------------------------------------------------------------
# Detector-in-the-loop training crop extraction
# ---------------------------------------------------------------------------
def _match_boxes_to_transcripts(boxes: List[Tuple[int, int, int, int]],
                                transcripts: List[str]) -> Optional[List[Tuple[Tuple[int, int, int, int], str]]]:
    """Pair detector boxes with ground-truth transcripts by top-to-bottom order.

    Only accepts an exact 1:1 count match (both lists are already sorted
    top-to-bottom by construction/detector). Returns ``None`` on any mismatch
    so the caller can discard the page and count it as a detector miss - this
    keeps training labels clean while the discard rate itself measures
    detector robustness across layouts (see ``generate_detector_in_the_loop``).
    """
    if len(boxes) != len(transcripts):
        return None
    return list(zip(boxes, transcripts))


def generate_detector_in_the_loop(
    out_dir: str,
    num_pages: int,
    font_paths: Sequence[str],
    font_sizes: Sequence[int],
    corpus: Sequence[str],
    detection_cfg: Optional[Dict] = None,
    crop_padding_x: int = 10,
    crop_padding_y: int = 5,
    min_crop_height: int = 14,
    layouts: Sequence[str] = LAYOUTS,
    split: Tuple[float, float] = (0.85, 0.15),
    seed: int = 2026,
    logger=None,
    progress: bool = True,
) -> Dict[str, object]:
    """Render synthetic pages, run the real detector, and save the crops the
    detector *actually produced* (imperfect padding, merges, fragments) paired
    with their ground-truth transcript.

    Writes ``images/page_XXXXXX_lNN.png`` + ``train_labels.txt`` /
    ``val_labels.txt`` under ``out_dir``. Returns stats including the
    per-layout detector exact-match rate (pages kept vs discarded).
    """
    from src.detection.text_detection import build_detector, crop_lines

    info = (logger.info if logger else print)
    warn = (logger.warning if logger else print)

    rng = random.Random(seed)
    faces = discover_font_faces(font_paths, warn=warn)
    detector = build_detector(detection_cfg or {})

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records: List[Tuple[str, str]] = []
    per_layout_total: Dict[str, int] = {}
    per_layout_kept: Dict[str, int] = {}
    font_cache: Dict = {}

    iterator = range(num_pages)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_pages, desc="detector-in-the-loop")
        except Exception:
            pass

    for i in iterator:
        layout = layouts[i % len(layouts)]
        page, transcripts, layout = make_page(
            rng, corpus, faces, font_sizes, layout=layout, font_cache=font_cache,
        )
        per_layout_total[layout] = per_layout_total.get(layout, 0) + 1
        if not transcripts:
            continue
        aug_page = apply_page_augmentations(page, rng)
        gray = np.asarray(aug_page.convert("L"))

        boxes = detector.detect(gray)
        pairs = _match_boxes_to_transcripts(boxes, transcripts)
        if pairs is None:
            continue
        per_layout_kept[layout] = per_layout_kept.get(layout, 0) + 1

        crops = crop_lines(
            gray, [b for b, _ in pairs],
            padding_x=crop_padding_x, padding_y=crop_padding_y,
            min_crop_height=min_crop_height,
        )
        for j, (crop, (_, text)) in enumerate(zip(crops, pairs)):
            if crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            rel = os.path.join("images", f"page_{i:06d}_l{j:02d}.png")
            Image.fromarray(crop).save(os.path.join(out_dir, rel))
            records.append((rel.replace("\\", "/"), text))

    rng.shuffle(records)
    n = len(records)
    n_train = int(n * split[0])
    parts = {"train": records[:n_train], "val": records[n_train:]}
    for name, rows in parts.items():
        path = os.path.join(out_dir, f"{name}_labels.txt")
        with open(path, "w", encoding="utf-8") as f:
            for rel, txt in rows:
                f.write(f"{rel}\t{txt}\n")
        info(f"[detector-in-the-loop] {name}: {len(rows)} crops -> {path}")

    match_rate = {
        layout: per_layout_kept.get(layout, 0) / max(1, per_layout_total.get(layout, 0))
        for layout in per_layout_total
    }
    for layout, rate in match_rate.items():
        info(f"[detector-in-the-loop] layout={layout} exact-match rate={rate:.3f} "
            f"({per_layout_kept.get(layout, 0)}/{per_layout_total.get(layout, 0)} pages)")

    return {
        "num_pages": num_pages,
        "num_crops": n,
        "per_layout_total": per_layout_total,
        "per_layout_kept": per_layout_kept,
        "match_rate": match_rate,
        "counts": {k: len(v) for k, v in parts.items()},
    }


# ---------------------------------------------------------------------------
# Realistic held-out evaluation pages (NOT used for training)
# ---------------------------------------------------------------------------
def generate_eval_pages(
    out_dir: str,
    num_pages: int,
    font_paths: Sequence[str],
    font_sizes: Sequence[int],
    corpus: Sequence[str],
    layouts: Sequence[str] = LAYOUTS,
    seed: int = 999999,
    logger=None,
) -> List[str]:
    """Render a small, disjoint-seed set of full pages + ground-truth transcript
    files for end-to-end (detector+recognizer) evaluation. Saved as
    ``page_XXX.png`` + ``page_XXX.gt.txt`` (one transcript line per row, in
    top-to-bottom order) under ``out_dir``. Not used anywhere in training.
    """
    info = (logger.info if logger else print)
    warn = (logger.warning if logger else print)
    rng = random.Random(seed)
    faces = discover_font_faces(font_paths, warn=warn)
    os.makedirs(out_dir, exist_ok=True)
    font_cache: Dict = {}

    paths = []
    for i in range(num_pages):
        layout = layouts[i % len(layouts)]
        page, transcripts, layout = make_page(rng, corpus, faces, font_sizes,
                                              layout=layout, font_cache=font_cache)
        aug_page = apply_page_augmentations(page, rng)
        img_path = os.path.join(out_dir, f"page_{i:03d}_{layout}.png")
        gt_path = os.path.join(out_dir, f"page_{i:03d}_{layout}.gt.txt")
        aug_page.convert("RGB").save(img_path)
        with open(gt_path, "w", encoding="utf-8") as f:
            for t in transcripts:
                f.write(t + "\n")
        paths.append(img_path)
        info(f"[eval-pages] wrote {img_path} ({len(transcripts)} lines, layout={layout})")
    return paths
