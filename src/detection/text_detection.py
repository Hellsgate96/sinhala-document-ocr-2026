"""Text-line / region detection (pipeline stage 3).

Provides a dependency-light OpenCV baseline (morphological dilation + contour
analysis) that returns line bounding boxes, and a documented adapter interface
showing where a learned detector (DBNet / CRAFT / PP-OCRv3 det head) would plug in.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# Axis-aligned bounding box: (x, y, w, h)
BBox = Tuple[int, int, int, int]


class TextDetector:
    """Interface for text-region detectors.

    Implementations return a list of axis-aligned bounding boxes in reading order
    (top-to-bottom, then left-to-right).
    """

    def detect(self, image: np.ndarray) -> List[BBox]:
        raise NotImplementedError


class OpenCVLineDetector(TextDetector):
    """Baseline line detector using morphology + connected components.

    No deep-learning dependency; suitable for clean printed pages and as a fallback
    before a DBNet/CRAFT model is trained.
    """

    def __init__(self, dilate_kernel: Tuple[int, int] = (25, 3),
                 min_line_height: int = 8, min_line_width: int = 20):
        self.dilate_kernel = dilate_kernel
        self.min_line_height = min_line_height
        self.min_line_width = min_line_width

    def detect(self, image: np.ndarray) -> List[BBox]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarize so ink is white on black.
        binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_kernel)
        merged = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[BBox] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= self.min_line_height and w >= self.min_line_width:
                boxes.append((x, y, w, h))
        # Reading order: top-to-bottom, then left-to-right.
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes


class DBNetDetector(TextDetector):
    """Adapter placeholder for a learned detector (DBNet / CRAFT / PP-OCRv3).

    To integrate a real model:
      1. Load the trained weights in ``__init__`` (e.g. a Torch/ONNX module).
      2. In ``detect``: preprocess -> forward pass -> probability/threshold map
         -> post-process (binarize, find polygons, box-from-polygon).
      3. Return axis-aligned boxes (or extend the interface to return polygons).

    Until weights are wired in, this raises ``NotImplementedError`` so callers can
    fall back to :class:`OpenCVLineDetector`.
    """

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path
        self.model = None  # placeholder for the loaded network

    def detect(self, image: np.ndarray) -> List[BBox]:
        raise NotImplementedError(
            "DBNetDetector is a placeholder. Load weights and implement the "
            "forward + post-processing, or use OpenCVLineDetector."
        )




def filter_line_boxes(
    boxes: List[BBox],
    min_line_height: int = 8,
    min_line_width: int = 20,
) -> List[BBox]:
    """Drop boxes that are too small to be text lines on a document page."""
    return [
        b for b in boxes
        if b[3] >= min_line_height and b[2] >= min_line_width
    ]


def merge_adjacent_line_boxes(
    boxes: List[BBox],
    max_vertical_gap: int = 6,
    max_merge_height: int = 28,
) -> List[BBox]:
    """Merge very short boxes on the same text row (common OpenCV over-split)."""
    if len(boxes) < 2:
        return list(boxes)
    merged: List[BBox] = list(boxes)
    changed = True
    while changed:
        changed = False
        merged.sort(key=lambda b: (b[1], b[0]))
        out: List[BBox] = []
        i = 0
        while i < len(merged):
            x, y, w, h = merged[i]
            if i + 1 < len(merged):
                x2, y2, w2, h2 = merged[i + 1]
                same_row = abs((y + h // 2) - (y2 + h2 // 2)) <= max_vertical_gap
                short = h <= max_merge_height and h2 <= max_merge_height
                gap = x2 - (x + w)
                if same_row and short and 0 <= gap <= max(12, w // 2):
                    nx = min(x, x2)
                    ny = min(y, y2)
                    nx2 = max(x + w, x2 + w2)
                    ny2 = max(y + h, y2 + h2)
                    out.append((nx, ny, nx2 - nx, ny2 - ny))
                    i += 2
                    changed = True
                    continue
            out.append((x, y, w, h))
            i += 1
        merged = out
    return merged


def refine_document_boxes(
    boxes: List[BBox],
    min_line_height: int = 8,
    min_line_width: int = 20,
    merge_short: bool = True,
) -> List[BBox]:
    """Filter tiny boxes and optionally merge adjacent short fragments."""
    boxes = filter_line_boxes(boxes, min_line_height, min_line_width)
    if merge_short:
        boxes = merge_adjacent_line_boxes(boxes)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def crop_is_mostly_numeric(
    gray_crop: np.ndarray,
    ink_ratio_threshold: float = 0.85,
) -> bool:
    """True when the crop looks like digits/punctuation only (skip prose OCR)."""
    if gray_crop is None or gray_crop.size == 0:
        return False
    if gray_crop.ndim == 3:
        gray_crop = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)
    inv = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(inv > 0)
    if len(xs) == 0:
        return False
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    blobs = inv[y0 : y1 + 1, x0 : x1 + 1]
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    digit_like = 0
    total = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 4:
            continue
        total += 1
        aspect = w / max(1, h)
        if 0.15 <= aspect <= 0.85 and h >= 6:
            digit_like += 1
    if total == 0:
        return False
    return (digit_like / total) >= ink_ratio_threshold


def crop_lines(
    image: np.ndarray,
    boxes: List[BBox],
    padding: int | None = None,
    padding_x: int = 10,
    padding_y: int = 5,
    min_crop_height: int = 14,
) -> List[np.ndarray]:
    """Crop each detected box from the page (with padding and minimum height)."""
    if padding is not None:
        padding_x = padding_y = int(padding)
    h, w = image.shape[:2]
    crops: List[np.ndarray] = []
    for (x, y, bw, bh) in boxes:
        cy = y + bh // 2
        bh_eff = max(int(bh), int(min_crop_height))
        y0 = max(0, cy - bh_eff // 2)
        y1 = min(h, y0 + bh_eff)
        y0 = max(0, y1 - bh_eff)
        x0 = max(0, x - int(padding_x))
        x1 = min(w, x + bw + int(padding_x))
        y0 = max(0, y0 - int(padding_y))
        y1 = min(h, y1 + int(padding_y))
        crops.append(image[y0:y1, x0:x1])
    return crops


def draw_boxes(image: np.ndarray, boxes: List[BBox]) -> np.ndarray:
    """Return a copy of the image with detected boxes drawn (for debugging)."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return canvas