"""Tests for the character n-gram LM and the LM-fused CTC beam decoder.

The jul28 real-photo failures were orthographic (ණේ -> ෙණී, ම් -> මි), so these
tests pin the two properties the fix relies on: the LM must rank the correct
Sinhala spelling above the confusable one, and the fused decoder must be able
to override the acoustic argmax when the LM is confident enough - while still
reducing to greedy decoding when the fusion weight is zero.
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from src.charset import Charset
from src.postprocess.char_lm import CharNGramLM, build_char_lm
from src.recognition.decode import ctc_beam_search_lm, decode_log_probs


def _log_probs_from_rows(rows, num_classes):
    """(T, C) log-probs from a list of {class_index: probability} dicts."""
    mat = np.full((len(rows), num_classes), 1e-9)
    for t, row in enumerate(rows):
        for idx, p in row.items():
            mat[t, idx] = p
    mat /= mat.sum(axis=1, keepdims=True)
    return np.log(mat)


def test_char_lm_is_a_distribution():
    lm = CharNGramLM(order=3).train(["abcabc", "abcab"])
    total = sum(math.exp(lm.logp("ab", ch)) for ch in "abc")
    # Backoff mass also covers unseen characters, so the seen chars must not
    # exceed 1 but should carry nearly all of it.
    assert 0.8 < total <= 1.0 + 1e-9


def test_char_lm_prefers_seen_continuations():
    lm = CharNGramLM(order=4).train(["ලංකාව " * 20, "ලංකාවේ " * 20])
    assert lm.logp("ලංකා", "ව") > lm.logp("ලංකා", "x")


def test_repo_lm_ranks_sinhala_confusions_correctly():
    """The exact word-final confusions seen on the jul28 lyrics page."""
    lm = build_char_lm(ROOT)
    pairs = [
        ("සිංහල දෙරණේ සරණ", "සිංහල දෙරෙණී සරණ"),   # ණේ -> ෙණී
        ("වීර හැගුම් මල් පිපුණේ", "විර හැගුමි මල් පිපුණේී"),  # ම් -> මි
    ]
    for good, bad in pairs:
        assert lm.logp_text(good) > lm.logp_text(bad), good


def test_beam_search_without_lm_matches_greedy():
    cs = Charset.build_default()
    a, b = cs.char_to_idx["a"], cs.char_to_idx["b"]
    rows = [{a: 0.9}, {a: 0.9}, {0: 0.9}, {a: 0.8}, {b: 0.95}, {b: 0.9}]
    log_probs = _log_probs_from_rows(rows, cs.num_classes)
    greedy = cs.ctc_greedy_decode(log_probs.argmax(-1).tolist())
    beam = ctc_beam_search_lm(log_probs, cs.idx_to_char, beam_width=8, top_k=6)
    assert beam == greedy == "aab"


def test_lm_fusion_overrides_a_marginal_acoustic_choice():
    cs = Charset.build_default()
    x, y = cs.char_to_idx["x"], cs.char_to_idx["y"]
    prefix = cs.char_to_idx["a"]
    # Frame 1 slightly favours "x"; a LM that has only ever seen "ay" should win.
    rows = [{prefix: 0.99}, {x: 0.52, y: 0.48}]
    log_probs = _log_probs_from_rows(rows, cs.num_classes)
    lm = CharNGramLM(order=3).train(["ay"] * 50)

    assert ctc_beam_search_lm(log_probs, cs.idx_to_char, beam_width=8, top_k=6) == "ax"
    fused = ctc_beam_search_lm(
        log_probs, cs.idx_to_char, beam_width=8, top_k=6,
        lm=lm, lm_weight=1.0, insertion_bonus=0.0,
    )
    assert fused == "ay"


def test_top_k_pruning_keeps_the_dominant_path():
    cs = Charset.build_default()
    a = cs.char_to_idx["a"]
    rows = [{a: 0.99}] * 4
    log_probs = _log_probs_from_rows(rows, cs.num_classes)
    assert ctc_beam_search_lm(log_probs, cs.idx_to_char, top_k=2) == "a"
    assert ctc_beam_search_lm(log_probs, cs.idx_to_char, top_k=0) == "a"


def test_decode_log_probs_dispatch():
    cs = Charset.build_default()
    a = cs.char_to_idx["a"]
    log_probs = _log_probs_from_rows([{a: 0.99}] * 3, cs.num_classes)
    assert decode_log_probs(log_probs, cs, mode="greedy") == "a"
    assert decode_log_probs(log_probs, cs, mode="beam") == "a"
    assert decode_log_probs(log_probs, cs, mode="beam_lm", lm_weight=0.0) == "a"
    with pytest.raises(ValueError):
        decode_log_probs(log_probs, cs, mode="nonsense")
