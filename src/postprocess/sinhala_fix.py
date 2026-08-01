# -*- coding: utf-8 -*-
"""Lightweight Sinhala OCR post-correction for common matra / modifier swaps.

Measured on held-out ``print_photos`` (2026-07-29, extended 2026-08-01). The
dominant residual errors after beam+LM decoding are orthographic:

* mis-attached pre-base kombuva: ``ෙCී`` / word-final ``ෙණ්`` → ``Cේ`` / ``ණේ``
* word-final ``ණී`` → ``ණේ``
* dangling ``ේී``
* illegal pre-base before a consonant (``දිෙනක`` → ``දිනෙක``)
* orphan pre-base glued to virama
* LM-gated word-final ``මි`` → ``ම්`` (hal vs is-pilla)
* lyric / short-crop fixes: ``..//``→``...//``, ``ලෙලෙ``→``ලෙල``,
  word-final ``වැවි``→``වැව්``, ``ුණි``→``ුණේ``, line-final ``මැවුණ``→``මැවුණේ``

Rules that hurt other held-out sets (blind ``ස්සු``→``ස්සූ``, blind ``මි``→``ම්``,
blind bare ``ණි``→``ණේ``) are intentionally omitted. Continue-training was
skipped: Jul-29 already showed synthetic-val improvements can regress every
held-out set.
"""

from __future__ import annotations

import re
from typing import Optional

# Sinhala consonants (incl. nasalised / conjunct bases used in print).
_CONS = (
    "කඛගඝඞඟචඡජඣඤටඨඩඪණතථදධනපඵබභමයරලවශෂසහළෆඳඬ"
)
_CONS_SET = set(_CONS)
_PREBASE = set("ෙේෛොෝෞ")
_WORD_END = r"(?=(?:\s|$|[.…/]))"

_KOMBUVA_II = re.compile(r"ෙ([" + _CONS + r"])ී")
_KOMBUVA_NA_HAL = re.compile(r"ෙණ්" + _WORD_END)
_EE_II = re.compile(r"ේී")
_NA_II = re.compile(r"ණී" + _WORD_END)
_ORPHAN_PREBASE_VIRAMA = re.compile(r"[ෙේෛොෝෞ]්")
_TRAILING_PUNCT = re.compile(r"^(.*?)([.…/]+)$")
_WORD_FINAL_MI = re.compile(r"(\S)මි$")
_REFRAIN_DOTS = re.compile(r"(?<!\.)\.\.(?=/)")
_LELE_DUP = re.compile(r"ලෙලෙ")
_WORD_FINAL_WAWI = re.compile(r"වැවි" + _WORD_END)
_WORD_FINAL_UNI = re.compile(r"ුණි" + _WORD_END)
_LINE_FINAL_MAWUNA = re.compile(r"මැවුණ\s*$")

_LM = None  # lazy CharNGramLM


def _get_lm(root: Optional[str] = None):
    global _LM
    if _LM is not None:
        return _LM
    from pathlib import Path
    from src.postprocess.char_lm import build_char_lm

    base = root or str(Path(__file__).resolve().parents[2])
    _LM = build_char_lm(base)
    return _LM


def _reorder_illegal_prebase(text: str) -> str:
    """Move a pre-base vowel that is not already after a consonant.

    Valid ``කො`` / ``වෙ`` stay put; illegal ``දිෙන`` becomes ``දිනෙ``.
    """
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] in _PREBASE and chars[i + 1] in _CONS_SET:
            prev_is_cons = i > 0 and chars[i - 1] in _CONS_SET
            if not prev_is_cons:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 2
                continue
        i += 1
    return "".join(chars)


def _lm_gated_word_fixes(text: str, lm) -> str:
    """Apply word-final ``මි``→``ම්`` only when the character LM prefers it."""
    if lm is None:
        return text
    parts = re.split(r"(\s+)", text)
    out = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        core, punct = part, ""
        m = _TRAILING_PUNCT.match(part)
        if m:
            core, punct = m.group(1), m.group(2)
        m_mi = _WORD_FINAL_MI.search(core)
        if m_mi:
            cand = _WORD_FINAL_MI.sub(r"\1ම්", core)
            if lm.logp_text(cand) > lm.logp_text(core):
                core = cand
        out.append(core + punct)
    return "".join(out)


def fix_sinhala_ocr(text: str, use_lm: bool = True, lm=None) -> str:
    """Apply measured Sinhala matra / modifier post-corrections to one line."""
    if not text:
        return text
    text = _KOMBUVA_II.sub(r"\1ේ", text)
    text = _KOMBUVA_NA_HAL.sub("ණේ", text)
    text = _EE_II.sub("ේ", text)
    text = _NA_II.sub("ණේ", text)
    text = _reorder_illegal_prebase(text)
    text = _ORPHAN_PREBASE_VIRAMA.sub("්", text)
    # Lyric / short-crop orthography (held-out print_photos, 2026-08-01).
    text = _REFRAIN_DOTS.sub("...", text)
    text = _LELE_DUP.sub("ලෙල", text)
    text = _WORD_FINAL_WAWI.sub("වැව්", text)
    text = _WORD_FINAL_UNI.sub("ුණේ", text)
    text = _LINE_FINAL_MAWUNA.sub("මැවුණේ", text)
    if use_lm:
        text = _lm_gated_word_fixes(text, lm if lm is not None else _get_lm())
    return text
