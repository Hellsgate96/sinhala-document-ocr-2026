"""Post-processing: dictionary-based correction + LM rescoring stub.

Stage-5 of the pipeline. The dictionary corrector snaps each recognized word to the
nearest in-vocabulary word by edit distance (within a threshold). A documented stub
shows where an n-gram (KenLM) or transformer language model would integrate for
context-aware rescoring.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from src.evaluation.metrics import edit_distance


def load_vocab(path: str) -> List[str]:
    """Load a newline-separated vocabulary (UTF-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


class DictionaryCorrector:
    """Snap words to the nearest dictionary entry within an edit-distance budget."""

    def __init__(self, vocab: Sequence[str], max_distance: int = 2):
        self.vocab = list(vocab)
        self.max_distance = max_distance
        self._exact = set(self.vocab)

    def correct_word(self, word: str) -> str:
        """Return the best in-vocab replacement, or the original if none is close."""
        if not word or word in self._exact:
            return word
        best_word = word
        best_dist = self.max_distance + 1
        for candidate in self.vocab:
            # Cheap length-based pruning before the full distance computation.
            if abs(len(candidate) - len(word)) > self.max_distance:
                continue
            dist = edit_distance(word, candidate)
            if dist < best_dist:
                best_dist, best_word = dist, candidate
        return best_word if best_dist <= self.max_distance else word

    def correct_text(self, text: str) -> str:
        """Correct each whitespace-separated token of a line."""
        return " ".join(self.correct_word(tok) for tok in text.split())


class NGramRescorer:
    """Placeholder for n-gram / neural LM rescoring.

    Integration plan:
      * KenLM: train an n-gram model on a Sinhala corpus, load with ``kenlm.Model``,
        then in :meth:`rescore` combine the recognizer's CTC score with the LM score
        (e.g. via beam search or weighted shallow fusion).
      * Transformer LM: load a Sinhala causal LM and score candidate hypotheses,
        choosing argmax of ``acoustic_weight * ctc + lm_weight * lm``.

    Until a model is wired in, :meth:`rescore` returns the top hypothesis unchanged.
    """

    def __init__(self, model_path: Optional[str] = None, lm_weight: float = 0.5):
        self.model_path = model_path
        self.lm_weight = lm_weight
        self.model = None  # placeholder for a loaded KenLM / transformer model

    def rescore(self, hypotheses: Sequence[str],
                acoustic_scores: Optional[Sequence[float]] = None) -> str:
        """Return the best hypothesis. Currently a pass-through (no LM loaded)."""
        if not hypotheses:
            return ""
        return hypotheses[0]


if __name__ == "__main__":
    from src.utils.common import configure_stdout_utf8
    configure_stdout_utf8()
    vocab = ["\u0dbd\u0d82\u0d9a\u0dcf\u0dc0", "\u0db4\u0dcf\u0dc3\u0dbd",
             "\u0d9c\u0db8"]  # ලංකාව, පාසල, ගම
    corrector = DictionaryCorrector(vocab, max_distance=2)
    noisy = "\u0dbd\u0d82\u0d9a\u0dcf\u0dc0\u0dca"  # ලංකාව් (extra virama)
    print(f"{noisy!r} -> {corrector.correct_word(noisy)!r}")