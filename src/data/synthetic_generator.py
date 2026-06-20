"""Synthetic Sinhala text-line image generator (SynthTIGER-style).

Renders random Sinhala (and mixed Sinhala-English/numeric) text lines using PIL
fonts, then applies phone-capture style degradations (rotation, blur, noise, JPEG
compression, shadow). Produces ``image`` + ``transcript`` pairs and a tab-separated
labels file. Runs on CPU with only Pillow + NumPy (OpenCV not required).

Font handling is robust on Windows: missing font files are skipped with a warning,
and ``C:/Windows/Fonts/Nirmala.ttf`` (Nirmala UI, ships with Windows, supports
Sinhala) is used as a fallback default.
"""

from __future__ import annotations

import io
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_WINDOWS_SINHALA_FONT = "C:/Windows/Fonts/Nirmala.ttc"


# --------------------------------------------------------------------------
# Word list + text sampling
# --------------------------------------------------------------------------
def load_word_list(path: str) -> List[str]:
    """Read a UTF-8 word/phrase list (one entry per non-empty line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def random_text(words: Sequence[str], min_words: int, max_words: int,
                rng: random.Random) -> str:
    """Compose a random line by joining 1..N sampled words/phrases."""
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(words) for _ in range(n))


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
def discover_fonts(font_paths: Sequence[str], warn=print) -> List[str]:
    """Return the subset of ``font_paths`` that exist on disk.

    Falls back to the bundled Windows Nirmala UI font if nothing else is found.
    """
    available: List[str] = []
    for p in font_paths:
        if os.path.isfile(p):
            available.append(p)
        else:
            warn(f"[font] missing, skipping: {p}")
    if not available and os.path.isfile(DEFAULT_WINDOWS_SINHALA_FONT):
        warn(f"[font] using fallback: {DEFAULT_WINDOWS_SINHALA_FONT}")
        available.append(DEFAULT_WINDOWS_SINHALA_FONT)
    if not available:
        raise RuntimeError(
            "No usable fonts found. Provide a Sinhala-capable .ttf in configs "
            "default.yaml (e.g. Noto Sans Sinhala) or run on Windows with Nirmala UI."
        )
    return available


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_text_image(text: str, font: ImageFont.FreeTypeFont,
                      text_color: Tuple[int, int, int],
                      bg_color: Tuple[int, int, int],
                      padding: int = 8) -> Image.Image:
    """Render ``text`` onto a tight RGB background with the given font."""
    # Measure with a scratch draw context.
    scratch = Image.new("RGB", (4, 4), bg_color)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0]) + 2 * padding
    h = max(1, bbox[3] - bbox[1]) + 2 * padding
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=text_color)
    return img


# --------------------------------------------------------------------------
# Augmentations (phone-capture simulation)
# --------------------------------------------------------------------------
def aug_rotate(img: Image.Image, max_deg: float, bg_color, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-max_deg, max_deg)
    return img.rotate(angle, expand=True, fillcolor=bg_color, resample=Image.BILINEAR)


def aug_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))


def aug_gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    sigma = rng.uniform(4.0, 18.0)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(25, 70)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def aug_shadow(img: Image.Image, rng: random.Random) -> Image.Image:
    """Overlay a soft linear brightness gradient to mimic uneven lighting."""
    w, h = img.size
    grad = np.linspace(rng.uniform(0.55, 0.85), 1.0, w, dtype=np.float32)
    if rng.random() < 0.5:
        grad = grad[::-1]
    mask = np.tile(grad, (h, 1))[:, :, None]
    arr = np.asarray(img).astype(np.float32) * mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_augmentations(img: Image.Image, augment: Dict, bg_color,
                        rng: random.Random) -> Image.Image:
    if augment.get("rotation") and rng.random() < 0.7:
        img = aug_rotate(img, float(augment.get("rotation_max_deg", 3.0)), bg_color, rng)
    if augment.get("shadow") and rng.random() < 0.5:
        img = aug_shadow(img, rng)
    if augment.get("blur") and rng.random() < 0.5:
        img = aug_blur(img, rng)
    if augment.get("gaussian_noise") and rng.random() < 0.6:
        img = aug_gaussian_noise(img, rng)
    if augment.get("jpeg_compression") and rng.random() < 0.5:
        img = aug_jpeg(img, rng)
    return img


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------
def _random_colors(rng: random.Random) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (text_color, bg_color) with a readable contrast (dark text)."""
    bg = rng.randint(200, 255)
    fg = rng.randint(0, 80)
    bg_color = (bg, bg, rng.randint(max(180, bg - 20), 255))
    text_color = (fg, fg, fg)
    return text_color, bg_color


def generate(out_dir: str,
             num_samples: int,
             font_paths: Sequence[str],
             font_sizes: Sequence[int],
             words: Sequence[str],
             min_words: int = 1,
             max_words: int = 4,
             augment: Optional[Dict] = None,
             split: Sequence[float] = (0.7, 0.15, 0.15),
             seed: int = 1337,
             logger=None) -> Dict[str, int]:
    """Generate a synthetic line-image dataset and write labels files.

    Returns a dict with the number of samples written per split.
    """
    warn = (logger.warning if logger else print)
    info = (logger.info if logger else print)

    rng = random.Random(seed)
    np.random.seed(seed)
    augment = augment or {}

    fonts = discover_fonts(font_paths, warn=warn)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Pre-load font objects keyed by (path, size).
    font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(path, size)
        return font_cache[key]

    records: List[Tuple[str, str]] = []
    for i in range(num_samples):
        text = random_text(words, min_words, max_words, rng)
        font = get_font(rng.choice(fonts), rng.choice(list(font_sizes)))
        text_color, bg_color = _random_colors(rng)
        img = render_text_image(text, font, text_color, bg_color)
        img = apply_augmentations(img, augment, bg_color, rng)

        rel_path = os.path.join("images", f"line_{i:06d}.png")
        img.save(os.path.join(out_dir, rel_path))
        records.append((rel_path.replace("\\", "/"), text))

    # Split by index (acts as a stand-in for "by document source").
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    parts = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }

    def write_labels(name: str, rows: List[Tuple[str, str]]):
        path = os.path.join(out_dir, f"{name}_labels.txt")
        with open(path, "w", encoding="utf-8") as f:
            for rel, txt in rows:
                f.write(f"{rel}\t{txt}\n")
        return path

    # Full labels + per-split labels.
    write_labels("all", records)
    counts = {}
    for name, rows in parts.items():
        write_labels(name, rows)
        counts[name] = len(rows)
        info(f"[split] {name}: {len(rows)} samples")

    info(f"[done] wrote {n} images to {images_dir}")
    return counts


if __name__ == "__main__":
    from src.utils.common import configure_stdout_utf8
    configure_stdout_utf8()
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    word_list = load_word_list(os.path.join(base, "src", "data", "sample_words.txt"))
    generate(
        out_dir=os.path.join(base, "data", "synthetic_sample"),
        num_samples=20,
        font_paths=[DEFAULT_WINDOWS_SINHALA_FONT],
        font_sizes=[26, 30, 34],
        words=word_list,
        augment={"rotation": True, "blur": True, "gaussian_noise": True,
                 "jpeg_compression": True, "shadow": True, "rotation_max_deg": 3.0},
    )
