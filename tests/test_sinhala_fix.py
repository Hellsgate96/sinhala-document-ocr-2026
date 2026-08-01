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
    assert fix_sinhala_ocr("සරණී..//", use_lm=False) == "සරණේ...//"


def test_illegal_prebase_reorder():
    assert fix_sinhala_ocr("දිෙනක", use_lm=False) == "දිනෙක"
    assert fix_sinhala_ocr("කොඩි වෙලා", use_lm=False) == "කොඩි වෙලා"


def test_orphan_prebase_before_virama():
    assert fix_sinhala_ocr("දෑෙස්", use_lm=False) == "දෑස්"


def test_legitimate_en_untouched():
    assert fix_sinhala_ocr("සිසිලෙන් තෙමුණේ", use_lm=False) == "සිසිලෙන් තෙමුණේ"


def test_lyric_short_crop_fixes():
    assert fix_sinhala_ocr("සිංහල සරණේ..//", use_lm=False) == "සිංහල සරණේ...//"
    assert fix_sinhala_ocr("ලෙලෙදෙන", use_lm=False) == "ලෙලදෙන"
    assert fix_sinhala_ocr("වැවි බැඳුණේ..", use_lm=False) == "වැව් බැඳුණේ.."
    assert fix_sinhala_ocr("නැගුණි...//", use_lm=False) == "නැගුණේ...//"
    assert fix_sinhala_ocr("ලොව මැවුණ", use_lm=False) == "ලොව මැවුණේ"
    assert fix_sinhala_ocr("පිළිගැන්විණි", use_lm=False) == "පිළිගැන්විණි"


def test_literary_ssu_long_uu_safe():
    assert fix_sinhala_ocr("තිගැස්සු", use_lm=False) == "තිගැස්සූ"
    assert fix_sinhala_ocr("හිනැස්සු", use_lm=False) == "හිනැස්සූ"
    assert fix_sinhala_ocr("ඇවිස්සු", use_lm=False) == "ඇවිස්සූ"
    assert fix_sinhala_ocr("ගමේ මිනිස්සු සහ", use_lm=False) == "ගමේ මිනිස්සු සහ"


def test_word_accuracy_residual_fixes():
    assert fix_sinhala_ocr("හමුවලා හදවෙතේ", use_lm=False) == "හමුවෙලා හදවතේ"
    assert fix_sinhala_ocr("යළි උපදිමි මේ", use_lm=False) == "යළි උපදිම් මේ"
    assert fix_sinhala_ocr("ගොනු හිත්", use_lm=False) == "ගොනු හිතේ"
    assert fix_sinhala_ocr("මගහැරී ගිලිහුනෙම", use_lm=False) == "මඟහැරී ගිලිහුනෙම"
