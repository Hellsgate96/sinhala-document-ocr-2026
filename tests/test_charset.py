"""Unit tests for the Sinhala charset (encode/decode round-trip + persistence)."""

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


def test_save_load():
    cs = Charset.build_default()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "charset.json")
        cs.save(path)
        loaded = Charset.load(path)
        assert loaded.chars == cs.chars
        assert loaded.num_classes == cs.num_classes


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