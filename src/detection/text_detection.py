"""Text-line / region detection (pipeline stage 3).

Two strategies:

* :class:`ProjectionLineDetector` (default) - watermark-robust binarization,
  border/frame suppression, then horizontal projection-profile segmentation.
  Handles decorated pages (greeting cards, certificates) with centered lines.
* :class:`OpenCVLineDetector` (fallback) - morphological dilation + contours.
  Suitable for clean printed pages; tends to fragment decorated layouts.

Select via ``detection.method: "projection" | "contours"`` in the config or the
:func:`build_detector` factory. Core steps are pure functions for unit testing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------
def binarize_for_detection(gray: np.ndarray,
                           min_contrast: int = 25) -> np.ndarray:
    """Binarize a page image so ink is 255 on 0, dropping faint watermarks.

    Estimates the local background with a large median blur, takes the darkness
    relative to that background, and thresholds with Otsu. A ``min_contrast``
    floor keeps low-contrast elements (faint watermark logos, paper texture,
    subtle gradients) out of the ink mask.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Background estimate: median blur with a window larger than glyph strokes.
    k = max(15, (min(h, w) // 20) | 1)
    background = cv2.medianBlur(gray, min(k, 255))
    # Darkness relative to background (text is darker than its surroundings).
    diff = cv2.subtract(background, gray)
    otsu_thr, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu_thr), float(min_contrast))
    binary = (diff >= thr).astype(np.uint8) * 255
    return binary


def suppress_border_structures(binary: np.ndarray,
                               margin_frac: float = 0.04,
                               max_aspect: float = 8.0,
                               frame_frac: float = 0.75) -> np.ndarray:
    """Remove decorative borders/frames from an ink mask (255 = ink).

    A connected component is erased when it either
    * spans most of the page (``frame_frac`` of width or height) - a frame, or
    * touches the page margin band and is long-and-thin (aspect ratio above
      ``max_aspect``) - border rules and ornaments.
    """
    h, w = binary.shape[:2]
    margin_x = max(2, int(w * margin_frac))
    margin_y = max(2, int(h * margin_frac))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = binary.copy()
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        spans_page = bw >= frame_frac * w or bh >= frame_frac * h
        touches_margin = (x <= margin_x or y <= margin_y
                          or x + bw >= w - margin_x or y + bh >= h - margin_y)
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if spans_page or (touches_margin and aspect >= max_aspect):
            out[labels == i] = 0
    return out


def _split_tall_band(profile: np.ndarray, y0: int, y1: int, median_h: float,
                     min_sub_frac: float = 0.55,
                     valley_ratio: float = 0.6) -> List[Tuple[int, int]]:
    """Recursively split an anomalously tall band at local profile valleys.

    Two touching text lines (matras/descenders bridging the gap so the row
    ink count never drops below the global threshold) show up as one band
    much taller than the page's median line height. Within such a band we
    look for the row with the lowest ink count (a local valley) and split
    there when the valley is a real dip - i.e. clearly lower than the ink
    level on both sides - and both halves would still be a reasonable
    fraction of a normal line height. Genuine single tall lines (large
    titles, decorative caps) have no such valley and are left intact.
    """
    height = y1 - y0
    min_sub = max(3, int(median_h * min_sub_frac))
    if height < 2 * min_sub:
        return [(y0, y1)]
    segment = profile[y0:y1]
    # Search for the valley away from the very edges (avoid trivial splits).
    lo = min_sub
    hi = height - min_sub
    if hi <= lo:
        return [(y0, y1)]
    window = segment[lo:hi]
    if window.size == 0:
        return [(y0, y1)]
    rel_idx = int(np.argmin(window))
    valley_y = lo + rel_idx
    valley_val = float(segment[valley_y])
    left_peak = float(segment[:valley_y].max()) if valley_y > 0 else valley_val
    right_peak = float(segment[valley_y:].max()) if valley_y < height else valley_val
    surrounding_peak = min(left_peak, right_peak) if left_peak and right_peak else max(left_peak, right_peak)
    if surrounding_peak <= 0:
        return [(y0, y1)]
    if valley_val > valley_ratio * surrounding_peak:
        return [(y0, y1)]
    split_y = y0 + valley_y
    left = _split_tall_band(profile, y0, split_y, median_h, min_sub_frac, valley_ratio)
    right = _split_tall_band(profile, split_y, y1, median_h, min_sub_frac, valley_ratio)
    return left + right


def horizontal_projection_bands(binary: np.ndarray,
                                min_ink_frac: float = 0.004,
                                smooth_frac: float = 0.004,
                                merge_gap_frac: float = 0.4,
                                min_band_frac: float = 0.35,
                                split_tall_frac: float = 1.2) -> List[Tuple[int, int]]:
    """Find (y0, y1) text bands from the horizontal ink-density profile.

    * profile: ink pixels per row, lightly smoothed.
    * rows with at least ``min_ink_frac`` of the page width in ink are "text".
    * a neighbouring band is merged into the previous one only when the gap is
      small (``merge_gap_frac`` x median band height) AND one of the two bands
      is a fragment (shorter than half the median) - dots/diacritics split
      across rows. Two full-height lines are never merged.
    * bands more than ``split_tall_frac`` x median height are re-split at an
      internal profile valley (handles adjacent lines whose glyphs touch
      vertically so the ink count never drops below threshold between them).
    * bands shorter than ``min_band_frac`` x median band height are dropped.
    """
    h, w = binary.shape[:2]
    profile = (binary > 0).sum(axis=1).astype(np.float32)
    k = max(1, int(round(h * smooth_frac)))
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        profile = np.convolve(profile, kernel, mode="same")
    threshold = max(1.0, w * min_ink_frac)

    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, value in enumerate(profile):
        if value >= threshold and start is None:
            start = y
        elif value < threshold and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, h))
    if not bands:
        return []

    heights = sorted(y1 - y0 for y0, y1 in bands)
    median_h = heights[len(heights) // 2]

    merged: List[Tuple[int, int]] = [bands[0]]
    max_gap = max(1, int(median_h * merge_gap_frac))
    frag_h = max(2, median_h // 2)
    for y0, y1 in bands[1:]:
        py0, py1 = merged[-1]
        gap = y0 - py1
        fragment_pair = (y1 - y0) < frag_h or (py1 - py0) < frag_h
        if gap <= max_gap and fragment_pair:
            merged[-1] = (py0, y1)
        else:
            merged.append((y0, y1))

    heights = sorted(y1 - y0 for y0, y1 in merged)
    median_h = heights[len(heights) // 2]

    if median_h > 0 and split_tall_frac > 0:
        split_out: List[Tuple[int, int]] = []
        tall_thresh = split_tall_frac * median_h
        for y0, y1 in merged:
            if (y1 - y0) > tall_thresh:
                split_out.extend(_split_tall_band(profile, y0, y1, median_h))
            else:
                split_out.append((y0, y1))
        merged = split_out

    heights = sorted(y1 - y0 for y0, y1 in merged)
    median_h = heights[len(heights) // 2]
    min_h = max(3, int(median_h * min_band_frac))
    kept = [(y0, y1) for y0, y1 in merged if (y1 - y0) >= min_h]
    return kept or merged


def band_to_box(binary: np.ndarray, y0: int, y1: int,
                min_ink_density: float = 0.01,
                pad_x: int = 4) -> Optional[BBox]:
    """Horizontal ink extent of one text band -> box, or None when too sparse.

    Handles centered/short lines: the box hugs the ink instead of the page.
    """
    h, w = binary.shape[:2]
    strip = binary[y0:y1]
    cols = (strip > 0).sum(axis=0)
    xs = np.flatnonzero(cols)
    if xs.size == 0:
        return None
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    box_w, box_h = x1 - x0, y1 - y0
    if box_w < 2 or box_h < 2:
        return None
    ink = float((strip[:, x0:x1] > 0).sum()) / float(box_w * box_h)
    if ink < min_ink_density:
        return None
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    return (x0, y0, x1 - x0, y1 - y0)


def filter_boxes_relative(boxes: List[BBox],
                          min_height_frac: float = 0.35,
                          max_aspect: float = 80.0) -> List[BBox]:
    """Drop boxes much shorter than the median line height or absurdly thin."""
    if not boxes:
        return []
    heights = sorted(b[3] for b in boxes)
    median_h = heights[len(heights) // 2]
    min_h = max(3, int(median_h * min_height_frac))
    out = []
    for x, y, w, h in boxes:
        if h < min_h:
            continue
        if w / max(1, h) > max_aspect:
            continue
        out.append((x, y, w, h))
    return out


class ProjectionLineDetector(TextDetector):
    """Line detector: watermark-robust binarize -> border suppression ->
    horizontal projection bands -> per-band ink extent boxes.

    Designed for real pages: decorative frames, faint watermarks and centered
    short lines (greeting cards, certificates, letters).
    """

    def __init__(self,
                 min_line_height: int = 8,
                 min_line_width: int = 20,
                 min_contrast: int = 25,
                 suppress_borders: bool = True,
                 min_ink_frac: float = 0.004,
                 min_ink_density: float = 0.01,
                 min_height_frac: float = 0.35):
        self.min_line_height = min_line_height
        self.min_line_width = min_line_width
        self.min_contrast = min_contrast
        self.suppress_borders = suppress_borders
        self.min_ink_frac = min_ink_frac
        self.min_ink_density = min_ink_density
        self.min_height_frac = min_height_frac

    def ink_mask(self, image: np.ndarray) -> np.ndarray:
        """Binarized, border-suppressed ink mask (for debugging/tests)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = binarize_for_detection(gray, min_contrast=self.min_contrast)
        if self.suppress_borders:
            binary = suppress_border_structures(binary)
        return binary

    def detect(self, image: np.ndarray) -> List[BBox]:
        binary = self.ink_mask(image)
        bands = horizontal_projection_bands(binary, min_ink_frac=self.min_ink_frac)
        boxes: List[BBox] = []
        for y0, y1 in bands:
            box = band_to_box(binary, y0, y1, min_ink_density=self.min_ink_density)
            if box is None:
                continue
            if box[3] >= self.min_line_height and box[2] >= self.min_line_width:
                boxes.append(box)
        boxes = filter_boxes_relative(boxes, min_height_frac=self.min_height_frac)
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes


class OpenCVLineDetector(TextDetector):
    """Baseline line detector using morphology + connected components.

    No deep-learning dependency; suitable for clean printed pages and as a
    fallback when the projection method under-segments unusual layouts.
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


def build_detector(detection_cfg: Optional[Dict] = None) -> TextDetector:
    """Factory: build the detector selected by ``detection.method`` config.

    ``method: "projection"`` (default) or ``"contours"``.
    """
    cfg = detection_cfg or {}
    method = str(cfg.get("method", "projection")).lower()
    min_h = int(cfg.get("min_line_height", 8))
    min_w = int(cfg.get("min_line_width", 20))
    if method == "contours":
        return OpenCVLineDetector(
            dilate_kernel=tuple(cfg.get("dilate_kernel", (25, 3))),
            min_line_height=min_h,
            min_line_width=min_w,
        )
    return ProjectionLineDetector(
        min_line_height=min_h,
        min_line_width=min_w,
        min_contrast=int(cfg.get("min_contrast", 25)),
        suppress_borders=bool(cfg.get("suppress_borders", True)),
        min_ink_frac=float(cfg.get("min_ink_frac", 0.004)),
        min_ink_density=float(cfg.get("min_ink_density", 0.01)),
        min_height_frac=float(cfg.get("min_height_frac", 0.35)),
    )


class DBNetDetector(TextDetector):
    """Adapter placeholder for a learned detector (DBNet / CRAFT / PP-OCRv3).

    To integrate a real model:
      1. Load the trained weights in ``__init__`` (e.g. a Torch/ONNX module).
      2. In ``detect``: preprocess -> forward pass -> probability/threshold map
         -> post-process (binarize, find polygons, box-from-polygon).
      3. Return axis-aligned boxes (or extend the interface to return polygons).

    Until weights are wired in, this raises ``NotImplementedError`` so callers can
    fall back to :class:`ProjectionLineDetector` / :class:`OpenCVLineDetector`.
    """

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path
        self.model = None  # placeholder for the loaded network

    def detect(self, image: np.ndarray) -> List[BBox]:
        raise NotImplementedError(
            "DBNetDetector is a placeholder. Load weights and implement the "
            "forward + post-processing, or use ProjectionLineDetector."
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


def merge_vertically_overlapping_boxes(boxes: List[BBox],
                                       min_overlap_frac: float = 0.5) -> List[BBox]:
    """Merge boxes whose vertical spans overlap substantially (same text band)."""
    if len(boxes) < 2:
        return list(boxes)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    out: List[BBox] = [boxes[0]]
    for x, y, w, h in boxes[1:]:
        px, py, pw, ph = out[-1]
        overlap = min(py + ph, y + h) - max(py, y)
        if overlap > 0 and overlap >= min_overlap_frac * min(h, ph):
            nx, ny = min(px, x), min(py, y)
            nx2, ny2 = max(px + pw, x + w), max(py + ph, y + h)
            out[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            out.append((x, y, w, h))
    return out


def refine_document_boxes(
    boxes: List[BBox],
    min_line_height: int = 8,
    min_line_width: int = 20,
    merge_short: bool = True,
    merge_overlapping: bool = True,
) -> List[BBox]:
    """Filter tiny boxes and merge same-row / same-band fragments."""
    boxes = filter_line_boxes(boxes, min_line_height, min_line_width)
    if merge_short:
        boxes = merge_adjacent_line_boxes(boxes)
    if merge_overlapping:
        boxes = merge_vertically_overlapping_boxes(boxes)
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