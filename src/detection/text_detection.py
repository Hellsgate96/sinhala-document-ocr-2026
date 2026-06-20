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


def crop_lines(image: np.ndarray, boxes: List[BBox],
               padding: int = 2) -> List[np.ndarray]:
    """Crop each detected box from the page (with optional padding)."""
    h, w = image.shape[:2]
    crops: List[np.ndarray] = []
    for (x, y, bw, bh) in boxes:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
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