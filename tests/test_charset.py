"""Unit tests for the Sinhala charset (encode/decode round-trip + persistence)."""

import pytest
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.charset import Charset


def test_round_trip():
    cs = Charset.build_default()
    samples = ["ආයුබෝවන්", "ලංකාව", "ගම", "පාසල", "ශ්‍රී ලංකාව", "මිල 2024"]
    for text in samples:
        encoded = cs.encode(text)
        assert cs.decode(encoded) == text, f"round-trip failed for {text!r}"


def test_blank_index_reserved():
    cs = Charset.build_default()
    assert cs.BLANK_INDEX == 0
    assert all(idx >= 1 for idx in cs.char_to_idx.values())
    assert cs.num_classes == len(cs) + 1


def test_ctc_greedy_decode_collapses():
    cs = Charset.build_default()
    a = cs.char_to_idx["a"]
    b = cs.char_to_idx["b"]
    # frames: a a <blank> a b b -> "aab"
    frames = [a, a, 0, a, b, b]
    assert cs.ctc_greedy_decode(frames) == "aab"




def test_logaddexp_helper():
    import math

    from src.charset import _logaddexp

    assert _logaddexp(-math.inf, 1.0) == 1.0
    assert _logaddexp(0.0, 0.0) == pytest.approx(math.log(2.0))


def test_ctc_beam_search_decode_small():
    import numpy as np

    from src.charset import _logaddexp

    cs = Charset.build_default()
    rng = np.random.default_rng(0)
    T, C = 5, min(8, cs.num_classes)
    log_probs = rng.standard_normal((T, C)).astype(np.float64)
    log_probs -= log_probs.max(axis=1, keepdims=True)
    out = cs.ctc_beam_search_decode(log_probs, beam_width=3)
    assert isinstance(out, str)
    a, b = -1.2, -0.5
    np_val = np.logaddexp(a, b)
    assert _logaddexp(a, b) == pytest.approx(float(np_val))


def test_save_load():
    cs = Charset.build_default()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "charset.json")
        cs.save(path)
        loaded = Charset.load(path)
        assert loaded.chars == cs.chars
        assert loaded.num_classes == cs.num_classes




def test_ctc_greedy_decode_repetition_cases():
    cs = Charset.build_default()
    a = cs.char_to_idx["a"]
    b = cs.char_to_idx["b"]
    blank = cs.BLANK_INDEX
    assert cs.ctc_greedy_decode([blank, a, a, blank, b]) == "ab"
    assert cs.ctc_greedy_decode([a, a, a, b, b]) == "ab"
    assert cs.ctc_greedy_decode([a, blank, a]) == "aa"


def test_ctc_beam_search_collapses_repeats():
    import numpy as np

    cs = Charset.build_default()
    a = cs.char_to_idx["a"]
    b = cs.char_to_idx["b"]
    blank = cs.BLANK_INDEX
    T, C = 6, cs.num_classes
    log_probs = np.full((T, C), -50.0, dtype=np.float64)
    seq = [a, a, blank, blank, b, b]
    for t, idx in enumerate(seq):
        log_probs[t, idx] = 0.0
    assert cs.ctc_beam_search_decode(log_probs, beam_width=5) == "ab"
    assert cs.ctc_greedy_decode(seq) == "ab"


def test_ctc_beam_matches_greedy_on_peaked_path():
    import numpy as np

    cs = Charset.build_default()
    a = cs.char_to_idx["x"]
    b = cs.char_to_idx["y"]
    blank = cs.BLANK_INDEX
    seq = [blank, a, a, blank, b, b, blank]
    T, C = len(seq), cs.num_classes
    log_probs = np.full((T, C), -50.0, dtype=np.float64)
    for t, idx in enumerate(seq):
        log_probs[t, idx] = 0.0
    greedy = cs.ctc_greedy_decode(seq)
    beam = cs.ctc_beam_search_decode(log_probs, beam_width=3)
    assert beam == greedy == "xy"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    test_round_trip()
    test_blank_index_reserved()
    test_ctc_greedy_decode_collapses()
    test_save_load()
    print("test_charset: ALL PASSED")