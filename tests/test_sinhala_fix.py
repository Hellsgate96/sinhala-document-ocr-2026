# -*- coding: utf-8 -*-
"""Tests for measured Sinhala OCR post-correction."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.postprocess.sinhala_fix import fix_sinhala_ocr


def test_kombuva_misattach_to_ee():
    assert fix_sinhala_ocr("රැකවරෙණී", use_lm=False) == "රැකවරණේ"
    assert fix_sinhala_ocr("දෙරෙණ් සරණ", use_lm=False) == "දෙරණේ සරණ"
    assert fix_sinhala_ocr("පිපුණේී", use_lm=False) == "පිපුණේ"


def test_word_final_na_ii():
    assert fix_sinhala_ocr("සරණී..//", use_lm=False) == "සරණේ..//"


def test_illegal_prebase_reorder():
    assert fix_sinhala_ocr("දිෙනක", use_lm=False) == "දිනෙක"
    # Already-correct syllables must not be scrambled.
    assert fix_sinhala_ocr("කොඩි වෙලා", use_lm=False) == "කොඩි වෙලා"


def test_orphan_prebase_before_virama():
    assert fix_sinhala_ocr("දෑෙස්", use_lm=False) == "දෑස්"


def test_legitimate_en_untouched():
    # Word-final ලෙන් must not become නේ.
    assert fix_sinhala_ocr("සිසිලෙන් තෙමුණේ", use_lm=False) == "සිසිලෙන් තෙමුණේ"
