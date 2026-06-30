"""Document preprocessing for Sinhala OCR.

Stage-2 of the pipeline: grayscale -> deskew -> denoise -> binarization ->
contrast enhancement (CLAHE). Exposes a single :func:`preprocess_document`
pipeline plus a small CLI to batch-process a folder of images.

Depends on OpenCV (``cv2``) and NumPy only.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/RGB image to single-channel grayscale (idempotent)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew (degrees) via the minimum-area rectangle of ink."""
    inv = cv2.bitwise_not(gray)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in [-90, 0); normalize to a small correction.
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def deskew(gray: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """Rotate the image to correct skew. Angle auto-estimated when not given."""
    if angle is None:
        angle = estimate_skew_angle(gray)
    if abs(angle) < 0.1:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray: np.ndarray, method: str = "nlmeans") -> np.ndarray:
    """Reduce noise using fastNlMeans (default) or a median blur."""
    if method == "median":
        return cv2.medianBlur(gray, 3)
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7,
                                    searchWindowSize=21)


def binarize(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """Binarize via adaptive Gaussian threshold (default) or global Otsu."""
    if method == "otsu":
        return cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, blockSize=31, C=15)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0,
                     tile: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def prepare_for_ocr(image: np.ndarray,
                    do_denoise: bool = True,
                    do_clahe: bool = True,
                    do_deskew: bool = False,
                    auto_invert: bool = False) -> np.ndarray:
    """Lightweight page/crop prep for recognition (no binarization).

    Keeps grayscale edges for stylized or logo-like text. Document-level
    :func:`preprocess_document` with binarization remains for detection-only paths.
    """
    gray = to_grayscale(image)
    if do_clahe:
        gray = enhance_contrast(gray)
    if do_denoise:
        gray = denoise(gray, method="median")
    if do_deskew:
        gray = deskew(gray)
    if auto_invert and float(gray.mean()) < 128.0:
        gray = cv2.bitwise_not(gray)
    return gray


def preprocess_document(image: np.ndarray,
                        do_deskew: bool = True,
                        do_denoise: bool = True,
                        do_clahe: bool = True,
                        do_binarize: bool = True,
                        binarize_method: str = "adaptive",
                        for_recognition: bool = False) -> np.ndarray:
    """Run the full preprocessing pipeline and return a cleaned image.

    When ``for_recognition`` is True, skips binarization (and uses a lighter path
    via :func:`prepare_for_ocr`) so CRNN inputs are not destroyed.
    """
    if for_recognition:
        return prepare_for_ocr(
            image,
            do_denoise=do_denoise,
            do_clahe=do_clahe,
            do_deskew=do_deskew,
            auto_invert=False,
        )
    gray = to_grayscale(image)
    if do_clahe:
        gray = enhance_contrast(gray)
    if do_denoise:
        gray = denoise(gray)
    if do_deskew:
        gray = deskew(gray)
    if do_binarize:
        gray = binarize(gray, method=binarize_method)
    return gray


def process_folder(input_dir: str, output_dir: str,
                   exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> int:
    """Preprocess every image in ``input_dir`` and write results to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for path in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if os.path.splitext(path)[1].lower() not in exts:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] could not read {path}, skipping")
            continue
        out = preprocess_document(img)
        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, out)
        count += 1
    print(f"[done] preprocessed {count} image(s) -> {output_dir}")
    return count


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess a folder of documents.")
    p.add_argument("--input", required=True, help="input image folder")
    p.add_argument("--output", required=True, help="output folder")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    process_folder(args.input, args.output)