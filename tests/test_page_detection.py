"""Detector regression tests against real synthetic PAGE renders (v3 domain-gap
fix). Complements the hand-drawn-bar tests in ``tests/test_v2_overhaul.py``
with actual PIL/font-rendered pages across every layout in
``src.data.page_synth`` - bordered cards, watermarks, poems, mixed
Sinhala/English/numeric lines and letterhead pages - and tracks the detector's
exact line-count match rate as an explicit, checked regression floor.
"""
from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from src.data.page_synth import LAYOUTS, apply_page_augmentations, make_page
from src.data.synthetic_generator import discover_font_faces, load_corpus
from src.detection.text_detection import ProjectionLineDetector, horizontal_projection_bands

FONT_CANDIDATES = [
    "C:/Windows/Fonts/Nirmala.ttc",
    "fonts/NotoSansSinhala-Regular.ttf",
    "fonts/NotoSerifSinhala-Regular.ttf",
    "fonts/AbhayaLibre-Regular.ttf",
    "fonts/Yaldevi-Variable.ttf",
]


def _available_faces():
    try:
        return discover_font_faces(FONT_CANDIDATES, warn=lambda *a, **k: None)
    except RuntimeError:
        return None


CORPUS = load_corpus()
FACES = _available_faces()

pytestmark = pytest.mark.skipif(
    not CORPUS or not FACES, reason="corpus or Sinhala font faces unavailable in this environment"
)


def _match_rate(layout: str, n: int = 12, augmented: bool = False) -> float:
    det = ProjectionLineDetector()
    ok = 0
    for seed in range(n):
        rng = random.Random(seed * 31 + 7)
        page, transcripts, _ = make_page(rng, CORPUS, FACES, [28, 32, 36, 44, 52], layout=layout)
        if augmented:
            page = apply_page_augmentations(page, rng)
        gray = np.asarray(page.convert("L"))
        boxes = det.detect(gray)
        if len(boxes) == len(transcripts):
            ok += 1
    return ok / n


@pytest.mark.parametrize("layout", LAYOUTS)
def test_detector_runs_without_crash_on_every_layout(layout):
    rng = random.Random(1)
    page, transcripts, _ = make_page(rng, CORPUS, FACES, [28, 32, 36, 44, 52], layout=layout)
    gray = np.asarray(page.convert("L"))
    boxes = ProjectionLineDetector().detect(gray)
    assert len(boxes) >= 1, f"detector found nothing on a non-empty {layout} page"
    assert len(transcripts) >= 1


@pytest.mark.parametrize("layout", LAYOUTS)
def test_detector_clean_page_match_rate_floor(layout):
    """On clean (un-augmented) pages the detector should get the exact line
    count right most of the time - this is the floor; augmented pages are
    allowed to be harder (see test_detector_augmented_match_rate_floor)."""
    rate = _match_rate(layout, n=12, augmented=False)
    assert rate >= 0.55, f"{layout}: clean-page exact-match rate too low ({rate:.2f})"


def test_detector_augmented_match_rate_floor_overall():
    """Average exact-match rate across layouts on phone-camera-augmented
    pages must not regress below this floor (v3 baseline ~0.80, see
    scripts/generate_pages.py logs). Guards against reintroducing an
    over-aggressive page-level augmentation (e.g. the perspective-jitter bug
    fixed in this same overhaul, which merged whole paragraphs into one box)."""
    rates = [_match_rate(layout, n=10, augmented=True) for layout in LAYOUTS]
    avg = sum(rates) / len(rates)
    assert avg >= 0.55, f"avg augmented match rate regressed: {avg:.2f} ({dict(zip(LAYOUTS, rates))})"


def test_bordered_card_border_not_detected_as_line():
    """A decorative border must not itself become a spurious detected box."""
    rng = random.Random(42)
    page, transcripts, _ = make_page(rng, CORPUS, FACES, [36, 44, 52], layout="bordered_card")
    gray = np.asarray(page.convert("L"))
    boxes = ProjectionLineDetector().detect(gray)
    w, h = page.size
    # no box should span nearly the whole page width AND height (that would be the frame)
    for (x, y, bw, bh) in boxes:
        assert not (bw >= 0.9 * w and bh >= 0.9 * h), "border frame leaked into detected boxes"


def test_tall_band_is_split_at_internal_valley():
    """Two lines whose glyphs touch vertically (a thin descender/matra
    bridging the gap with only a sliver of ink) can end up as one tall
    projection band instead of two, because the row ink-count never drops
    below the global "is text" threshold. Regression test for the fix that
    re-splits anomalously tall bands at an internal profile valley, which
    closed a real gap seen on poem/mixed-language eval pages (detector
    under-counted lines by 30-50%)."""
    import numpy as np

    w = 200
    line_h = 20
    gap_h = 10
    bridge_h = 2  # a thin connecting strip - few ink pixels, not full width
    bridge_ink_px = 8  # keeps the row above the "is text" threshold (0.004*w)
    n_normal_lines = 4

    def full_line_row():
        return np.full(w, 180, dtype=np.uint8)

    def empty_row():
        return np.zeros(w, dtype=np.uint8)

    def bridge_row():
        row = np.zeros(w, dtype=np.uint8)
        row[: bridge_ink_px] = 180
        return row

    binary_rows = []
    for _ in range(n_normal_lines):
        binary_rows.extend([full_line_row()] * line_h)
        binary_rows.extend([empty_row()] * gap_h)

    # Two touching lines: full-ink rows connected by a thin low-ink-count
    # bridge (simulates a descender/matra), so the profile never drops below
    # the global "is text" threshold but does dip sharply at the bridge.
    binary_rows.extend([full_line_row()] * line_h)
    binary_rows.extend([bridge_row()] * bridge_h)
    binary_rows.extend([full_line_row()] * line_h)
    binary_rows.extend([empty_row()] * gap_h)

    binary = np.stack(binary_rows, axis=0)
    bands = horizontal_projection_bands(binary, smooth_frac=0.0)
    # Expect n_normal_lines + the 2 touching lines split apart = n_normal_lines + 2
    assert len(bands) == n_normal_lines + 2, f"expected {n_normal_lines + 2} bands, got {len(bands)}: {bands}"
