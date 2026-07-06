"""Tests for the v2 overhaul: ZWJ round-trip, projection detection, corpus."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.charset import Charset

ZWJ = "\u200d"


# ---------------------------------------------------------------------------
# Charset: ZWJ conjunct handling
# ---------------------------------------------------------------------------
def test_zwj_in_charset():
    cs = Charset.build_default()
    assert ZWJ in cs.char_to_idx, "ZWJ (U+200D) must be encodable for conjuncts"


def test_zwj_round_trip_kanyawee():
    # "kanyaawee" contains the ්‍ය conjunct: DA(0DB1) AL(0DCA) ZWJ YA(0DBA)
    text = "\u0d9a\u0db1\u0dca\u200d\u0dba\u0dcf\u0dc0\u0dd3"
    cs = Charset.build_default()
    encoded = cs.encode(text)
    assert len(encoded) == len(text), "no character may be dropped"
    decoded = cs.decode(encoded)
    assert decoded == text
    assert ZWJ in decoded, "ZWJ must survive the encode/decode round-trip"


def test_zwj_round_trip_rakaransaya():
    # "shree" with ්‍ර conjunct as in ශ්‍රී
    text = "\u0dc1\u0dca\u200d\u0dbb\u0dd3 \u0dbd\u0d82\u0d9a\u0dcf"
    cs = Charset.build_default()
    assert cs.decode(cs.encode(text)) == text


# ---------------------------------------------------------------------------
# Projection detection on a synthetic multi-line page
# ---------------------------------------------------------------------------
def _synthetic_page(n_lines=3, w=800, h=600, with_border=False, with_watermark=False):
    """White page with dark horizontal text-like bars (centered, varying width)."""
    page = np.full((h, w), 245, dtype=np.uint8)
    if with_watermark:
        # faint blob in the middle (low contrast vs 245 background)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = ((yy - h // 2) ** 2 + (xx - w // 2) ** 2) < (h // 4) ** 2
        page[blob] = 232
    line_h = 24
    gap = (h - n_lines * line_h) // (n_lines + 1)
    boxes = []
    for i in range(n_lines):
        y0 = gap + i * (line_h + gap)
        line_w = int(w * (0.3 + 0.15 * (i % 3)))
        x0 = (w - line_w) // 2  # centered
        # text-like bar with per-column gaps (words)
        for x in range(x0, x0 + line_w):
            if ((x - x0) // 40) % 4 != 3:  # word gaps (ink starts at x0)
                page[y0:y0 + line_h, x] = 30
        boxes.append((x0, y0, line_w, line_h))
    if with_border:
        t = 6
        m = 15
        page[m:m + t, m:w - m] = 40
        page[h - m - t:h - m, m:w - m] = 40
        page[m:h - m, m:m + t] = 40
        page[m:h - m, w - m - t:w - m] = 40
    return page, boxes


def test_projection_detects_three_lines():
    from src.detection.text_detection import ProjectionLineDetector

    page, truth = _synthetic_page(n_lines=3)
    boxes = ProjectionLineDetector().detect(page)
    assert len(boxes) == 3, f"expected 3 lines, got {len(boxes)}: {boxes}"
    for (tx, ty, tw, th), (x, y, w, h) in zip(truth, boxes):
        assert abs(y - ty) <= 6 and abs((y + h) - (ty + th)) <= 6
        assert abs(x - tx) <= 12, "box must hug centered ink, not span the page"


def test_projection_ignores_border_frame():
    from src.detection.text_detection import ProjectionLineDetector

    page, truth = _synthetic_page(n_lines=2, with_border=True)
    boxes = ProjectionLineDetector().detect(page)
    assert len(boxes) == 2, f"border frame leaked into boxes: {boxes}"


def test_projection_ignores_faint_watermark():
    from src.detection.text_detection import ProjectionLineDetector

    page, truth = _synthetic_page(n_lines=3, with_watermark=True)
    boxes = ProjectionLineDetector().detect(page)
    assert len(boxes) == 3, f"watermark leaked into boxes: {boxes}"


def test_build_detector_switch():
    from src.detection.text_detection import (
        OpenCVLineDetector, ProjectionLineDetector, build_detector)

    assert isinstance(build_detector({"method": "projection"}), ProjectionLineDetector)
    assert isinstance(build_detector({"method": "contours"}), OpenCVLineDetector)
    assert isinstance(build_detector(None), ProjectionLineDetector)  # default


# ---------------------------------------------------------------------------
# Corpus file
# ---------------------------------------------------------------------------
def _corpus_lines():
    path = os.path.join(ROOT, "src", "data", "corpus_sinhala.txt")
    assert os.path.isfile(path), "run scripts/build_corpus.py"
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def test_corpus_loads_and_is_large():
    lines = _corpus_lines()
    assert len(lines) >= 3000, f"corpus too small: {len(lines)}"
    assert len(set(lines)) == len(lines), "corpus lines must be distinct"


def test_corpus_charset_coverage():
    cs = Charset.build_default()
    lines = _corpus_lines()
    used = set("".join(lines))
    missing = sorted(c for c in used if c not in cs.char_to_idx)
    assert not missing, f"corpus chars missing from charset: {[hex(ord(c)) for c in missing]}"


def test_corpus_has_conjuncts_and_rare_graphemes():
    text = "\n".join(_corpus_lines())
    assert "\u0dca\u200d\u0dbb" in text, "rakaransaya (\u0dca+ZWJ+\u0dbb) missing"
    assert "\u0dca\u200d\u0dba" in text, "yansaya (\u0dca+ZWJ+\u0dba) missing"
    for cp in (0x0DB3, 0x0D9F, 0x0DAC, 0x0DD8):  # nda, nga, ndda, vocalic r sign
        assert chr(cp) in text, f"U+{cp:04X} missing from corpus"


def test_generator_charset_check_warns_on_unknown():
    from src.data.synthetic_generator import check_charset_coverage

    warnings = []
    missing = check_charset_coverage(["hello", "\u4e16\u754c"], warn=warnings.append)
    assert missing and warnings, "CJK chars must be reported as missing"
    assert check_charset_coverage(["hello"], warn=warnings.append) == []