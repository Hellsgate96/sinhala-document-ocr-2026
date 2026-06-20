"""Sinhala-aware character set with CTC-style encode/decode.

The character set covers:
  * the Sinhala Unicode block (U+0D80-U+0DFF),
  * ZERO WIDTH JOINER (U+200D) which is essential for Sinhala conjuncts (e.g. "ශ්‍රී"),
  * ASCII digits, letters and common punctuation for mixed Sinhala-English forms,
  * the space character.

Index 0 is reserved for the CTC *blank* token, so model classes = len(charset) + 1.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Sequence

# --- Unicode ranges -------------------------------------------------------
SINHALA_START = 0x0D80
SINHALA_END = 0x0DFF
ZWJ = "\u200d"          # zero width joiner (conjunct formation)
ZWNJ = "\u200c"         # zero width non-joiner

# Common punctuation found in printed forms / invoices.
PUNCTUATION = " .,:;!?'\"()[]{}-+/\\@#%&*=_<>|~`$"
DIGITS = "0123456789"
ASCII_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _sinhala_characters() -> List[str]:
    """Return every assigned-or-reserved code point in the Sinhala block.

    We keep the whole block (minus a couple of unassigned gaps is unnecessary;
    PyTorch only needs a stable index map). Including the full range keeps the
    charset deterministic across machines.
    """
    return [chr(cp) for cp in range(SINHALA_START, SINHALA_END + 1)]


def default_characters() -> List[str]:
    """Build the ordered, de-duplicated default character list (excludes blank)."""
    chars: List[str] = []
    seen = set()
    for c in list(DIGITS) + list(ASCII_LETTERS) + list(PUNCTUATION) \
            + [ZWJ, ZWNJ] + _sinhala_characters():
        if c not in seen:
            seen.add(c)
            chars.append(c)
    return chars


class Charset:
    """Maps characters <-> integer indices for CTC training/decoding.

    Index 0 is always the CTC blank. Real characters start at index 1.
    """

    BLANK_INDEX = 0

    def __init__(self, chars: Sequence[str]):
        # Preserve order, drop duplicates and the blank if accidentally present.
        ordered: List[str] = []
        seen = set()
        for c in chars:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        self.chars: List[str] = ordered
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}

    # -- sizing ------------------------------------------------------------
    def __len__(self) -> int:
        """Number of real characters (excluding the blank)."""
        return len(self.chars)

    @property
    def num_classes(self) -> int:
        """Total model output classes including the CTC blank."""
        return len(self.chars) + 1

    # -- encode / decode ---------------------------------------------------
    def encode(self, text: str, warn_unknown: bool = False) -> List[int]:
        """Encode a string to a list of indices, skipping unknown characters."""
        out: List[int] = []
        for ch in text:
            idx = self.char_to_idx.get(ch)
            if idx is None:
                if warn_unknown:
                    import warnings
                    warnings.warn(f"Unknown character U+{ord(ch):04X} skipped")
                continue
            out.append(idx)
        return out

    def decode(self, indices: Iterable[int]) -> str:
        """Plain index->char mapping (blanks/0 are dropped). No CTC collapsing."""
        return "".join(self.idx_to_char[i] for i in indices if i in self.idx_to_char)

    def ctc_greedy_decode(self, indices: Sequence[int]) -> str:
        """Collapse a raw CTC frame sequence: merge repeats, then drop blanks."""
        collapsed: List[int] = []
        prev = None
        for i in indices:
            if i != prev:
                collapsed.append(i)
            prev = i
        return "".join(
            self.idx_to_char[i] for i in collapsed
            if i != self.BLANK_INDEX and i in self.idx_to_char
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the charset as JSON (UTF-8)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Charset":
        """Load a charset previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])

    @classmethod
    def build_default(cls) -> "Charset":
        """Construct the standard Sinhala + ASCII + punctuation charset."""
        return cls(default_characters())


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cs = Charset.build_default()
    sample = "ශ්‍රී ලංකාව 2024"
    enc = cs.encode(sample)
    print(f"num_classes (incl. blank) = {cs.num_classes}")
    print(f"sample           = {sample!r}")
    print(f"encoded          = {enc}")
    print(f"decoded          = {cs.decode(enc)!r}")
    assert cs.decode(enc) == sample, "round-trip failed"
    print("round-trip OK")