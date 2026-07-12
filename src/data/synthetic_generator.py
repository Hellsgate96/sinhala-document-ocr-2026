"""Synthetic Sinhala text-line image generator (SynthTIGER-style).

Renders diverse Sinhala (and mixed Sinhala-English/numeric) text lines using PIL
fonts, then applies phone-capture style degradations (rotation, blur, noise,
brightness/contrast jitter, JPEG compression, perspective, shadow). Produces
``image`` + ``transcript`` pairs and tab-separated labels files with a seedable
train/val/test split.

Text sampling (v2): lines are drawn primarily from a large real-text corpus
(``src/data/corpus_sinhala.txt`` - thousands of distinct Sinhala sentences),
mixed with random sentence spans, word recombinations from the legacy word
lists, and synthetic numbers/dates/IDs. Rendering varies font face (all faces
in a .ttc collection), size, text colour, background (plain / textured /
gradient) and padding.

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
DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "corpus_sinhala.txt"
)

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
# Word list / corpus loading + text sampling
# --------------------------------------------------------------------------
def load_word_list(path: str) -> List[str]:
    """Read a UTF-8 word/phrase list (one entry per non-empty, non-comment line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.lstrip().startswith("#")]


def load_word_lists(paths: Sequence[str], warn=print) -> List[str]:
    """Read and concatenate several UTF-8 word lists, skipping missing files."""
    words: List[str] = []
    for p in paths:
        if p and os.path.isfile(p):
            words.extend(load_word_list(p))
        elif p:
            warn(f"[words] missing, skipping: {p}")
    return words


def load_corpus(path: Optional[str] = None, warn=print) -> List[str]:
    """Load the Sinhala sentence corpus (one line per row); [] when missing."""
    path = path or DEFAULT_CORPUS_PATH
    if not os.path.isfile(path):
        warn(f"[corpus] missing: {path} (run scripts/build_corpus.py)")
        return []
    return load_word_list(path)


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


def clamp_words(text: str, max_words: int, rng: random.Random) -> str:
    """Cut a random contiguous span of at most ``max_words`` words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    start = rng.randint(0, len(words) - max_words)
    return " ".join(words[start:start + max_words])


def compose_line(words: Sequence[str], min_words: int, max_words: int,
                 rng: random.Random, numeric_ratio: float = 0.12,
                 mixed_ratio: float = 0.10,
                 corpus: Optional[Sequence[str]] = None,
                 corpus_ratio: float = 0.65) -> str:
    """Compose one training line.

    With probability ``corpus_ratio`` (and a non-empty ``corpus``) the line is a
    real corpus sentence: either the full sentence or a random contiguous span,
    clamped to ``max_words`` words. Otherwise the line is built token-by-token
    from the word list, with small probabilities of synthetic numbers/dates and
    Sinhala-English mixed tokens (amounts, dates, IDs, reference codes).
    """
    if corpus and rng.random() < corpus_ratio:
        sentence = rng.choice(corpus)
        n_words = len(sentence.split())
        if n_words > 1 and rng.random() < 0.35:
            # random span: 1..min(max_words, n) words
            span = rng.randint(1, min(max_words, n_words))
            start = rng.randint(0, n_words - span)
            return " ".join(sentence.split()[start:start + span])
        return clamp_words(sentence, max_words, rng)

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


def discover_font_faces(font_paths: Sequence[str], warn=print,
                        probe_text: str = "\u0d9c\u0dd4") -> List[Tuple[str, int]]:
    """Expand font files into (path, face_index) pairs.

    ``.ttc`` collections (e.g. Nirmala UI Regular/Bold/Semilight + Nirmala Text
    faces) are enumerated so every face that renders Sinhala becomes an extra
    rendering style for free.
    """
    faces: List[Tuple[str, int]] = []
    for path in discover_fonts(font_paths, warn=warn):
        if path.lower().endswith(".ttc"):
            index = 0
            while True:
                try:
                    font = ImageFont.truetype(path, 24, index=index)
                except Exception:
                    break
                try:
                    bbox = font.getbbox(probe_text)
                    renders = bbox[2] > bbox[0]
                except Exception:
                    renders = True
                if renders:
                    faces.append((path, index))
                index += 1
                if index > 15:
                    break
        else:
            faces.append((path, 0))
    if not faces:
        raise RuntimeError("No usable font faces found.")
    return faces


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _make_background(w: int, h: int, bg_color: Tuple[int, int, int],
                     rng: random.Random) -> Image.Image:
    """Plain / subtle-gradient / lightly-textured light background."""
    style = rng.random()
    if style < 0.6:
        return Image.new("RGB", (w, h), bg_color)
    base = np.full((h, w, 3), bg_color, dtype=np.float32)
    if style < 0.85:  # subtle linear gradient
        drop = rng.uniform(4, 22)
        axis = rng.random() < 0.5
        ramp = np.linspace(0.0, drop, w if axis else h, dtype=np.float32)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        base -= ramp[None, :, None] if axis else ramp[:, None, None]
    else:  # light paper texture
        sigma = rng.uniform(2.0, 7.0)
        base += np.random.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def render_text_image(text: str, font: ImageFont.FreeTypeFont,
                      text_color: Tuple[int, int, int],
                      bg_color: Tuple[int, int, int],
                      padding: int = 8,
                      rng: Optional[random.Random] = None) -> Image.Image:
    """Render ``text`` onto a light background with the given font.

    When ``rng`` is provided, the horizontal/vertical padding is jittered and
    the background may carry a subtle gradient or paper texture. Extra
    horizontal padding (with the ink placed centre or off-centre) mimics
    centered lines on wider layouts.
    """
    scratch = Image.new("RGB", (4, 4), bg_color)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])

    if rng is None:
        pad_x = pad_y = padding
        w = text_w + 2 * pad_x
        h = text_h + 2 * pad_y
        img = Image.new("RGB", (w, h), bg_color)
        x0 = pad_x
    else:
        pad_y = rng.randint(4, max(5, int(text_h * 0.45)))
        pad_x = rng.randint(4, 24)
        extra = int(text_w * rng.uniform(0.0, 0.5)) if rng.random() < 0.3 else 0
        w = text_w + 2 * pad_x + extra
        h = text_h + 2 * pad_y
        img = _make_background(w, h, bg_color, rng)
        # centre or left-align the ink inside the extra space
        x0 = pad_x + (extra // 2 if rng.random() < 0.6 else 0)
    draw = ImageDraw.Draw(img)
    draw.text((x0 - bbox[0], pad_y - bbox[1]), text, font=font, fill=text_color)
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


def aug_perspective(img: Image.Image, bg_color, rng: random.Random,
                    mx_range=(0.01, 0.05), my_range=(0.01, 0.06)) -> Image.Image:
    """Apply a slight perspective warp (corner jitter) for camera-angle realism.

    ``mx_range``/``my_range`` are fractions of width/height and default to
    values tuned for small line crops. Full PAGE images (see
    :mod:`src.data.page_synth`) are hundreds of pixels taller/wider, so the
    same fraction would move a corner by 50-100px - pass smaller fractions
    for whole-page perspective (mild camera tilt, not a wild skew).
    """
    w, h = img.size
    mx = rng.uniform(*mx_range)
    my = rng.uniform(*my_range)
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


def aug_defocus_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """Camera-like defocus blur: box/disc blur, occasionally a slight motion streak.

    Distinct from :func:`aug_blur` (plain Gaussian) - simulates an out-of-focus
    phone camera rather than image resampling blur.
    """
    if rng.random() < 0.65:
        radius = rng.uniform(0.6, 2.2)
        return img.filter(ImageFilter.BoxBlur(radius))
    # short directional motion blur (hand shake)
    arr = np.asarray(img).astype(np.float32)
    length = rng.randint(3, 7)
    kernel = np.zeros((length, length), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[length // 2, :] = 1.0
    else:
        np.fill_diagonal(kernel, 1.0)
    kernel /= kernel.sum()
    try:
        import cv2
        blurred = cv2.filter2D(arr, -1, kernel)
    except Exception:
        return img
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def aug_paper_texture(img: Image.Image, rng: random.Random) -> Image.Image:
    """Overlay correlated grain (paper fibre / scan noise) across the whole crop,
    including ink pixels - plain Gaussian noise (:func:`aug_gaussian_noise`) is
    applied later at a lighter, uncorrelated level; this simulates paper grain
    that survives print + photograph."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    sigma = rng.uniform(3.0, 10.0)
    grain = np.random.normal(0.0, sigma, (max(1, h // 2), max(1, w // 2)))
    grain = np.asarray(Image.fromarray(grain.astype(np.float32)).resize((w, h), Image.BILINEAR))
    if arr.ndim == 3:
        grain = grain[:, :, None]
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug_moire(img: Image.Image, rng: random.Random) -> Image.Image:
    """Low-amplitude sinusoidal interference pattern (screen/re-photograph moire)."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    freq = rng.uniform(0.15, 0.5)
    angle = rng.uniform(0, np.pi)
    amp = rng.uniform(4.0, 14.0)
    yy, xx = np.mgrid[0:h, 0:w]
    phase = xx * np.cos(angle) * freq + yy * np.sin(angle) * freq
    pattern = amp * np.sin(phase)
    if arr.ndim == 3:
        pattern = pattern[:, :, None]
    arr = np.clip(arr + pattern, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def aug_edge_artifact(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate an imperfect detector crop: a thin rule/border fragment or a
    sliver of an adjacent line's strokes bleeding into the top/bottom edge."""
    arr = np.asarray(img).astype(np.float32).copy()
    h, w = arr.shape[:2]
    dark = float(rng.randint(20, 90))
    if rng.random() < 0.5:
        # straight rule fragment (table/border edge) near one border
        thickness = rng.randint(1, 3)
        band = rng.choice(["top", "bottom", "left", "right"])
        if band == "top":
            arr[0:thickness, :] = dark
        elif band == "bottom":
            arr[h - thickness:h, :] = dark
        elif band == "left":
            arr[:, 0:thickness] = dark
        else:
            arr[:, w - thickness:w] = dark
    else:
        # sliver of adjacent-line strokes: a few short dark dashes near top/bottom
        edge_h = max(2, int(h * rng.uniform(0.06, 0.16)))
        band = rng.choice(["top", "bottom"])
        n_dashes = rng.randint(2, 6)
        for _ in range(n_dashes):
            dash_w = rng.randint(max(2, w // 30), max(3, w // 10))
            x0 = rng.randint(0, max(1, w - dash_w))
            y0 = 0 if band == "top" else h - edge_h
            y1 = edge_h if band == "top" else h
            yy = rng.randint(y0, max(y0, y1 - 1))
            arr[yy:yy + 1, x0:x0 + dash_w] = dark
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_augmentations(img: Image.Image, augment: Dict, bg_color,
                        rng: random.Random) -> Image.Image:
    """Apply phone-capture-style degradations, roughly in physical order:
    geometry (rotation/perspective) -> crop-edge artifacts -> lighting ->
    optics (defocus/blur) -> surface (paper texture/moire) -> sensor noise ->
    (re-)compression. Every step is independently probabilistic so the
    aggregate distribution is broad rather than "every image gets every
    degradation applied the same way"."""
    if augment.get("rotation") and rng.random() < 0.7:
        img = aug_rotate(img, float(augment.get("rotation_max_deg", 3.0)), bg_color, rng)
    if augment.get("perspective") and rng.random() < 0.4:
        img = aug_perspective(img, bg_color, rng)
    if augment.get("edge_artifact", True) and rng.random() < 0.22:
        img = aug_edge_artifact(img, rng)
    if augment.get("shadow") and rng.random() < 0.5:
        img = aug_shadow(img, rng)
    if augment.get("brightness_contrast") and rng.random() < 0.6:
        img = aug_brightness_contrast(img, rng)
    if augment.get("defocus_blur", True) and rng.random() < 0.35:
        img = aug_defocus_blur(img, rng)
    elif augment.get("blur") and rng.random() < 0.4:
        img = aug_blur(img, rng)
    if augment.get("paper_texture", True) and rng.random() < 0.45:
        img = aug_paper_texture(img, rng)
    if augment.get("gaussian_noise") and rng.random() < 0.55:
        img = aug_gaussian_noise(img, rng)
    if augment.get("moire", True) and rng.random() < 0.08:
        img = aug_moire(img, rng)
    if augment.get("jpeg_compression") and rng.random() < 0.55:
        img = aug_jpeg(img, rng)
        if augment.get("multi_jpeg", True) and rng.random() < 0.25:
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
    """Return (text_color, bg_color): dark ink on a light background.

    Text is black / dark gray most of the time, occasionally a dark hue (blue,
    brown, green, maroon) as on printed cards and letterheads.
    """
    bg = rng.randint(200, 255)
    bg_color = (bg, bg, rng.randint(max(180, bg - 20), 255))
    style = rng.random()
    if style < 0.6:  # black / dark gray
        fg = rng.randint(0, 70)
        text_color = (fg, fg, fg)
    else:  # dark hue: one channel lifted slightly
        base = [rng.randint(0, 60) for _ in range(3)]
        base[rng.randint(0, 2)] = rng.randint(40, 110)
        text_color = tuple(base)
    return text_color, bg_color


def check_charset_coverage(texts: Sequence[str], warn=print) -> List[str]:
    """Warn about characters in ``texts`` missing from the default charset.

    Returns the sorted list of missing characters (empty when fully covered).
    Encoding silently drops unknown characters at train time, so any hit here
    means labels and rendered pixels would disagree - fix the charset instead.
    """
    from src.charset import Charset

    charset = Charset.build_default()
    used = set("".join(texts))
    missing = sorted(c for c in used if c not in charset.char_to_idx)
    if missing:
        codes = ", ".join(f"U+{ord(c):04X}" for c in missing)
        warn(f"[charset] {len(missing)} character(s) missing from charset: {codes}")
    return missing


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
             max_words: int = 12,
             augment: Optional[Dict] = None,
             split: Sequence[float] = (0.7, 0.15, 0.15),
             seed: int = 1337,
             logger=None,
             numeric_ratio: float = 0.12,
             mixed_ratio: float = 0.10,
             corpus: Optional[Sequence[str]] = None,
             corpus_path: Optional[str] = None,
             corpus_ratio: float = 0.65,
             progress: bool = True) -> Dict[str, int]:
    """Generate a synthetic line-image dataset and write labels files.

    Writes ``images/line_XXXXXX.png`` plus ``all_labels.txt`` and the per-split
    ``train_labels.txt`` / ``val_labels.txt`` / ``test_labels.txt`` files (each row
    ``relative_image_path<TAB>transcript``). Returns the per-split sample counts.

    ``corpus`` (or ``corpus_path``) supplies real Sinhala sentences; when
    omitted, ``src/data/corpus_sinhala.txt`` is loaded automatically. The same
    ``seed`` reproduces both the rendered images and the split exactly.
    """
    warn = (logger.warning if logger else print)
    info = (logger.info if logger else print)

    rng = random.Random(seed)
    np.random.seed(seed)
    augment = augment or {}

    if corpus is None:
        corpus = load_corpus(corpus_path, warn=warn)
    corpus = list(corpus or [])

    faces = discover_font_faces(font_paths, warn=warn)
    font_sizes = list(font_sizes)
    info(f"[fonts] using {len(faces)} font face(s); {len(words)} vocab entries; "
         f"{len(corpus)} corpus lines; {num_samples} samples")

    check_charset_coverage(list(corpus) + list(words), warn=warn)

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Pre-load font objects keyed by (path, face_index, size).
    font_cache: Dict[Tuple[str, int, int], ImageFont.FreeTypeFont] = {}

    def get_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
        key = (path, index, size)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(path, size, index=index)
        return font_cache[key]

    records: List[Tuple[str, str]] = []
    for i in _progress(range(num_samples), num_samples, progress, desc="render"):
        text = compose_line(words, min_words, max_words, rng,
                            numeric_ratio=numeric_ratio, mixed_ratio=mixed_ratio,
                            corpus=corpus, corpus_ratio=corpus_ratio)
        path, face_index = rng.choice(faces)
        font = get_font(path, face_index, rng.choice(font_sizes))
        text_color, bg_color = _random_colors(rng)
        img = render_text_image(text, font, text_color, bg_color, rng=rng)
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