# -*- coding: utf-8 -*-
"""Character n-gram language model for CTC beam-search shallow fusion.

Motivation (jul28 error analysis on real photos): the recogniser's residual
errors are almost all *orthographic*, not visual. On ~16 px lyric lines the
dominant confusions were

    ණේ -> ෙණී     (the pre-base kombuva glyph sits between two consonants and
                    got attached to the wrong one, and the remaining top stroke
                    decoded as the long-i sign)
    ම්  -> මි      (hal kirima vs is-pilla)
    ඟ  -> ග
    ූ  -> ු

Every one of those turns a frequent Sinhala character sequence into a rare or
impossible one, so a small character n-gram model over Sinhala text separates
them even when the pixels do not. The model is built at load time from plain
text (no binary artefact to keep in sync) and scores are combined with the
acoustic CTC score inside :mod:`src.recognition.decode`.

Smoothing is Witten-Bell-style recursive interpolation:

    p_n(c | ctx) = lambda * p_ML(c | ctx) + (1 - lambda) * p_{n-1}(c | ctx[1:])
    lambda       = N(ctx) / (N(ctx) + K)

which is normalised (unlike stupid backoff), needs no held-out tuning, and
degrades gracefully to a uniform distribution for unseen contexts.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Text used to build the LM. These are *training-side* sources only - no
# held-out ground truth (data/eval_real/**, *_holdout.txt) may appear here or
# the held-out CER numbers stop meaning anything.
DEFAULT_CORPUS_FILES: Tuple[str, ...] = (
    "src/data/corpus_sinhala.txt",
    "src/data/exam_style_lines.txt",
    "src/data/form_vocab.txt",
    "src/data/sample_words.txt",
)
DEFAULT_LABEL_FILES: Tuple[str, ...] = (
    "data/real/labels/web_batch1_acts.txt",
)

BOS = "\x02"


class CharNGramLM:
    """Interpolated character n-gram LM with Witten-Bell style backoff."""

    def __init__(self, order: int = 6, discount: float = 2.0):
        self.order = max(2, int(order))
        self.discount = float(discount)
        # context (str) -> {char: count}; context length 0 .. order-1
        self._counts: List[Dict[str, Dict[str, int]]] = [defaultdict(dict) for _ in range(self.order)]
        self._totals: List[Dict[str, int]] = [defaultdict(int) for _ in range(self.order)]
        self._vocab: set = set()
        self._cache: Dict[Tuple[str, str], float] = {}

    # -- construction ------------------------------------------------------
    def train(self, lines: Iterable[str]) -> "CharNGramLM":
        pad = BOS * (self.order - 1)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            text = pad + line
            self._vocab.update(line)
            for i in range(self.order - 1, len(text)):
                ch = text[i]
                for k in range(self.order):
                    ctx = text[i - k:i]
                    bucket = self._counts[k][ctx]
                    bucket[ch] = bucket.get(ch, 0) + 1
                    self._totals[k][ctx] += 1
        self._cache.clear()
        return self

    @property
    def vocab_size(self) -> int:
        return max(1, len(self._vocab))

    def __len__(self) -> int:
        return sum(len(level) for level in self._counts)

    # -- scoring -----------------------------------------------------------
    def logp(self, context: str, ch: str) -> float:
        """log P(ch | context); ``context`` is the decoded text so far."""
        ctx = context[-(self.order - 1):] if self.order > 1 else ""
        if len(ctx) < self.order - 1:
            ctx = BOS * (self.order - 1 - len(ctx)) + ctx
        key = (ctx, ch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Recurse from unigram (k=0) upward so each level interpolates the one below.
        prob = 1.0 / (self.vocab_size + 1)
        for k in range(self.order):
            sub = ctx[len(ctx) - k:] if k else ""
            total = self._totals[k].get(sub, 0)
            if not total:
                continue
            ml = self._counts[k][sub].get(ch, 0) / total
            lam = total / (total + self.discount)
            prob = lam * ml + (1.0 - lam) * prob
        value = math.log(prob) if prob > 0 else -30.0
        self._cache[key] = value
        return value

    def logp_text(self, text: str) -> float:
        return sum(self.logp(text[:i], text[i]) for i in range(len(text)))


def _read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _read_label_texts(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            out.append(line.split("\t", 1)[1])
    return out


_LM_CACHE: Dict[Tuple, CharNGramLM] = {}


def build_char_lm(
    root: str,
    corpus_files: Optional[Sequence[str]] = None,
    label_files: Optional[Sequence[str]] = None,
    order: int = 6,
    discount: float = 2.0,
) -> CharNGramLM:
    """Build (and memoise) the LM from repo-relative text/label files."""
    corpus_files = tuple(corpus_files if corpus_files is not None else DEFAULT_CORPUS_FILES)
    label_files = tuple(label_files if label_files is not None else DEFAULT_LABEL_FILES)
    key = (os.path.abspath(root), corpus_files, label_files, order, discount)
    cached = _LM_CACHE.get(key)
    if cached is not None:
        return cached

    lines: List[str] = []
    for rel in corpus_files:
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if os.path.isfile(path):
            lines.extend(_read_text_lines(path))
    for rel in label_files:
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if os.path.isfile(path):
            lines.extend(_read_label_texts(path))

    lm = CharNGramLM(order=order, discount=discount).train(lines)
    _LM_CACHE[key] = lm
    return lm
