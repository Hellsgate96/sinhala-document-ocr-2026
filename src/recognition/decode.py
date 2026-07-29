# -*- coding: utf-8 -*-
"""CTC decoding strategies: greedy, prefix beam search, and beam search with
character-LM shallow fusion.

The fused score of a hypothesis is

    log P_ctc(y | x)  +  lm_weight * log P_lm(y)  +  insertion_bonus * |y|

The insertion bonus counteracts the LM's built-in preference for short strings
(every extra character costs one more negative log-probability).

Only the top ``top_k`` classes per frame are expanded. With 225 classes the
un-pruned expansion is ~20x slower for no measurable CER change - the CTC
posterior at any frame is concentrated on a handful of characters.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NEG_INF = -1e30


def _logaddexp(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_search_lm(
    log_probs,
    idx_to_char: Dict[int, str],
    blank: int = 0,
    beam_width: int = 12,
    lm: Optional[Any] = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
    top_k: int = 8,
) -> str:
    """Prefix beam search over ``log_probs`` of shape (T, num_classes)."""
    if hasattr(log_probs, "detach"):
        log_probs = log_probs.detach().cpu().numpy()
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, C = log_probs.shape
    use_lm = lm is not None and lm_weight != 0.0

    # prefix (tuple of class indices) -> [p_blank, p_nonblank, text, lm_score]
    beams: Dict[Tuple[int, ...], List[Any]] = {(): [0.0, NEG_INF, "", 0.0]}

    for t in range(T):
        row = log_probs[t]
        if 0 < top_k < C:
            cand = np.argpartition(-row, top_k - 1)[:top_k].tolist()
            if blank not in cand:
                cand.append(blank)
        else:
            cand = list(range(C))

        nxt: Dict[Tuple[int, ...], List[Any]] = {}
        for prefix, (p_b, p_nb, text, lm_s) in beams.items():
            p_total = _logaddexp(p_b, p_nb)
            last = prefix[-1] if prefix else None
            for c in cand:
                lp = float(row[c])
                if c == blank:
                    entry = nxt.get(prefix)
                    if entry is None:
                        entry = nxt[prefix] = [NEG_INF, NEG_INF, text, lm_s]
                    entry[0] = _logaddexp(entry[0], p_total + lp)
                    continue

                if c == last:
                    # Repeat of the current last label: stays the same string
                    # unless a blank separated them (then it is a new label).
                    entry = nxt.get(prefix)
                    if entry is None:
                        entry = nxt[prefix] = [NEG_INF, NEG_INF, text, lm_s]
                    entry[1] = _logaddexp(entry[1], p_nb + lp)
                    extend_from = p_b
                else:
                    extend_from = p_total
                if extend_from <= NEG_INF:
                    continue

                new_prefix = prefix + (c,)
                entry = nxt.get(new_prefix)
                if entry is None:
                    ch = idx_to_char.get(c, "")
                    new_lm = lm_s + (lm.logp(text, ch) if (use_lm and ch) else 0.0)
                    entry = nxt[new_prefix] = [NEG_INF, NEG_INF, text + ch, new_lm]
                entry[1] = _logaddexp(entry[1], extend_from + lp)

        if len(nxt) > beam_width:
            ranked = sorted(
                nxt.items(),
                key=lambda kv: -(
                    _logaddexp(kv[1][0], kv[1][1])
                    + lm_weight * kv[1][3]
                    + insertion_bonus * len(kv[1][2])
                ),
            )
            beams = dict(ranked[:beam_width])
        else:
            beams = nxt

    best = max(
        beams.values(),
        key=lambda v: _logaddexp(v[0], v[1]) + lm_weight * v[3] + insertion_bonus * len(v[2]),
    )
    return best[2]


def decode_log_probs(
    log_probs,
    charset,
    mode: str = "greedy",
    beam_width: int = 12,
    lm: Optional[Any] = None,
    lm_weight: float = 0.0,
    insertion_bonus: float = 0.0,
    top_k: int = 8,
) -> str:
    """Dispatch a (T, num_classes) log-prob matrix to the requested decoder."""
    mode = (mode or "greedy").lower()
    if mode == "greedy":
        if hasattr(log_probs, "argmax") and hasattr(log_probs, "detach"):
            indices = log_probs.argmax(-1).tolist()
        else:
            indices = np.asarray(log_probs).argmax(-1).tolist()
        return charset.ctc_greedy_decode(indices)
    if mode in ("beam_lm", "lm_beam", "lm"):
        return ctc_beam_search_lm(
            log_probs, charset.idx_to_char, blank=charset.BLANK_INDEX,
            beam_width=beam_width, lm=lm, lm_weight=lm_weight,
            insertion_bonus=insertion_bonus, top_k=top_k,
        )
    if mode == "beam":
        return ctc_beam_search_lm(
            log_probs, charset.idx_to_char, blank=charset.BLANK_INDEX,
            beam_width=beam_width, lm=None, lm_weight=0.0,
            insertion_bonus=0.0, top_k=top_k,
        )
    raise ValueError(f"unknown decode mode: {mode!r}")
