"""Post-processing: dictionary, LM correction, and Sinhala matra fixes."""

from src.postprocess.sinhala_fix import fix_sinhala_ocr

__all__ = ["fix_sinhala_ocr"]
