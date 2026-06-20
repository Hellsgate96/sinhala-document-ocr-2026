"""Synthetic Sinhala text-line image generator (SynthTIGER-style).

Renders random Sinhala (and mixed Sinhala-English/numeric) text lines using PIL
fonts, then applies phone-capture style degradations (rotation, blur, noise,
brightness/contrast jitter, JPEG compression, perspective, shadow). Produces
``image`` + ``transcript`` pairs and tab-separated labels files with a seedable
train/val/test split.

This module scales from a tiny smoke-test (a handful of images) up to a large
training corpus (tens of thousands of images). It runs on CPU with only Pillow +
NumPy (OpenCV is NOT required); an optional ``tqdm`` progress bar is used when
available.

Font handling is robust on Windows and Linux/Colab: missing font files are skipped
with a warning, and ``C:/Windows/Fonts/Nirmala.ttc`` (Nirmala UI, ships with
Windows, supports Sinhala) is used as a final fallback on Windows.
"""

from __future__ import annotations

import io
import os
import random
import string
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

DEFAULT_WINDOWS_SINHALA_FONT = "C:/Windows/Fonts/Nirmala.ttc"

# Additional well-known Sinhala-capable fonts probed as a fallback when no
# configured font exists (covers both Windows and Linux/Colab installs).
FALLBACK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/NirmalaB.ttf",
    "C:/Windows/Fonts/iskpota.ttf",
    "C:/Windows/Fonts/Iskpota.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifSinhala-Regular.ttf",
    "/root/.fonts/NotoSansSinhala-Regular.ttf",
)


# --------------------------------------------------------------------------
# Word list + text sampling
# --------------------------------------------------------------------------
def load_word_list(path: str) -> List[str]:
    """Read a UTF-8 word/phrase list (one entry per non-empty line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_word_lists(paths: Sequence[str], warn=print) -> List[str]:
    """Read and concatenate several UTF-8 word lists, skipping missing files."""
    words: List[str] = []
    for p in paths:
        if p and os.path.isfile(p):
            words.extend(load_word_list(p))
        elif p:
            warn(f"[words] missing, skipping: {p}")
    return words


# -- numeric / mixed token synthesis ---------------------------------------
def random_number(rng: random.Random) -> str:
    """Generate a standalone number token (integer, decimal, money, percent)."""
    kind = rng.random()
    if kind < 0.35:
        return str(rng.randint(0, 999999))
    if kind < 0.55:
        whole = rng.randint(0, 99999)
        return f"{whole:,}.{rng.randint(0, 99):02d}"
    if kind < 0.70:
        return f"{rng.randint(1, 100)}%"
    if kind < 0.85:
        return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"
    return f"{rng.randint(0, 999999):,}"


def random_date(rng: random.Random) -> str:
    """Generate a date-like token in one of a few common formats."""
    y = rng.randint(1990, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    fmt = rng.randint(0, 3)
    if fmt == 0:
        return f"{y}.{m:02d}.{d:02d}"
    if fmt == 1:
        return f"{y}-{m:02d}-{d:02d}"
    if fmt == 2:
        return f"{d:02d}/{m:02d}/{y}"
    return f"{d}/{m}/{y}"


def random_mixed_token(rng: random.Random) -> str:
    """Generate a Sinhala-English / alphanumeric mixed token (IDs, refs)."""
    kind = rng.randint(0, 4)
    if kind == 0:  # NIC-style
        return f"{rng.randint(190000000000, 200099999999)}"
    if kind == 1:  # phone
        return f"0{rng.randint(700000000, 779999999)}"
    if kind == 2:  # reference code
        letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.randint(1, 3)))
        return f"{letters}{rng.randint(100, 9999)}"
    if kind == 3:  # No. style
        return f"No. {rng.randint(1, 999)}"
    return f"{rng.choice(['A/L', 'O/L', 'Grade', 'Ref'])} {rng.randint(1, 2026)}"


def compose_line(words: Sequence[str], min_words: int, max_words: int,
                 rng: random.Random, numeric_ratio: float = 0.12,
                 mixed_ratio: float = 0.10) -> str:
    """Compose a random line by joining 1..N sampled tokens.

    Each token is usually a sampled word/phrase, but with small probabilities a
    synthetic number/date or a Sinhala-English mixed token is injected to mimic
    real documents (amounts, dates, IDs, reference codes).
    """
    n = rng.randint(min_words, max_words)
    tokens: List[str] = []
    for _ in range(n):
        r = rng.random()
        if r < numeric_ratio:
            tokens.append(random_number(rng) if rng.random() < 0.6 else random_date(rng))
        elif r < numeric_ratio + mixed_ratio:
            tokens.append(random_mixed_token(rng))
        else:
            tokens.append(rng.choice(words))
    return " ".join(tokens)


# Backward-compatible alias (older callers used ``random_text``).
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

    Falls back to a list of well-known Sinhala-capable fonts (Windows Nirmala UI,
    Linux/Colab Noto Sans Sinhala) if none of the configured paths are found.
    """
    available: List[str] = []
    for p in font_paths:
        if p and os.path.isfile(p):
            available.append(p)
        elif p:
            warn(f"[font] missing, skipping: {p}")
    if not available:
        for cand in FALLBACK_FONT_CANDIDATES:
            if os.path.isfile(cand):
                warn(f"[font] using fallback: {cand}")
                available.append(cand)
                break
    if not available:
        raise RuntimeError(
            "No usable fonts found. Provide a Sinhala-capable .ttf in configs "
            "default.yaml (e.g. Noto Sans Sinhala), run on Windows with Nirmala UI, "
            "or `apt-get install fonts-noto` on Linux/Colab."
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


def aug_brightness_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    """Jitter brightness and contrast to mimic varied lighting/exposure."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 1.35))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.4))
    return img


def _perspective_coeffs(src: Sequence[Tuple[float, float]],
                        dst: Sequence[Tuple[float, float]]) -> List[float]:
    """Solve for the 8 PIL PERSPECTIVE coefficients mapping output->input."""
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.reshape(8).tolist()


def aug_perspective(img: Image.Image, bg_color, rng: random.Random) -> Image.Image:
    """Apply a slight perspective warp (corner jitter) for camera-angle realism."""
    w, h = img.size
    mx = rng.uniform(0.01, 0.05)
    my = rng.uniform(0.01, 0.06)
    jitter = lambda d: rng.uniform(-d, d)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (jitter(w * mx), jitter(h * my)),
        (w - jitter(w * mx), jitter(h * my)),
        (w - jitter(w * mx), h - jitter(h * my)),
        (jitter(w * mx), h - jitter(h * my)),
    ]
    try:
        coeffs = _perspective_coeffs(src, dst)
    except np.linalg.LinAlgError:
        return img
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BILINEAR, fillcolor=bg_color)


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
    if augment.get("perspective") and rng.random() < 0.4:
        img = aug_perspective(img, bg_color, rng)
    if augment.get("shadow") and rng.random() < 0.5:
        img = aug_shadow(img, rng)
    if augment.get("brightness_contrast") and rng.random() < 0.6:
        img = aug_brightness_contrast(img, rng)
    if augment.get("blur") and rng.random() < 0.5:
        img = aug_blur(img, rng)
    if augment.get("gaussian_noise") and rng.random() < 0.6:
        img = aug_gaussian_noise(img, rng)
    if augment.get("jpeg_compression") and rng.random() < 0.5:
        img = aug_jpeg(img, rng)
    return img


def aug_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(25, 70)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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


def _progress(iterable, total, enabled, desc="generate"):
    """Wrap an iterable with tqdm when available and enabled; else passthrough."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


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
             logger=None,
             numeric_ratio: float = 0.12,
             mixed_ratio: float = 0.10,
             progress: bool = True) -> Dict[str, int]:
    """Generate a synthetic line-image dataset and write labels files.

    Writes ``images/line_XXXXXX.png`` plus ``all_labels.txt`` and the per-split
    ``train_labels.txt`` / ``val_labels.txt`` / ``test_labels.txt`` files (each row
    ``relative_image_path<TAB>transcript``). Returns the per-split sample counts.

    The same ``seed`` reproduces both the rendered images and the split exactly.
    """
    warn = (logger.warning if logger else print)
    info = (logger.info if logger else print)

    rng = random.Random(seed)
    np.random.seed(seed)
    augment = augment or {}

    fonts = discover_fonts(font_paths, warn=warn)
    font_sizes = list(font_sizes)
    info(f"[fonts] using {len(fonts)} font(s); {len(words)} vocab entries; "
         f"{num_samples} samples")
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
    for i in _progress(range(num_samples), num_samples, progress, desc="render"):
        text = compose_line(words, min_words, max_words, rng,
                            numeric_ratio=numeric_ratio, mixed_ratio=mixed_ratio)
        font = get_font(rng.choice(fonts), rng.choice(font_sizes))
        text_color, bg_color = _random_colors(rng)
        img = render_text_image(text, font, text_color, bg_color)
        img = apply_augmentations(img, augment, bg_color, rng)

        rel_path = os.path.join("images", f"line_{i:06d}.png")
        img.save(os.path.join(out_dir, rel_path))
        records.append((rel_path.replace("\\", "/"), text))

    # Shuffle then split by index (acts as a stand-in for "by document source").
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
                 "jpeg_compression": True, "shadow": True,
                 "brightness_contrast": True, "perspective": True,
                 "rotation_max_deg": 3.0},
    )