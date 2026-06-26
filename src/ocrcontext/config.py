"""Configuration for the analyzer / pipeline.

All knobs mirror constants from the original Modal service so OCR behaviour is
identical after decoupling.
"""

from __future__ import annotations

from dataclasses import dataclass

# PDF rasterization scale when falling back to image OCR (handwriting needs finer detail).
OCR_PDF_RENDER_SCALE = 2.75
OCR_PDF_RENDER_SCALE_HANDWRITING = 3.5

# Minimum expected non-empty lines per page before the line-band fallback kicks in.
MIN_EXPECTED_LINES_PER_PAGE = 3
MIN_EXPECTED_LINES_HANDWRITING = 1


@dataclass
class AnalyzerConfig:
    """Tunable settings for an :class:`~ocrcontext.analyzer.Analyzer`.

    Defaults reproduce the production pipeline's behaviour.
    """

    # Default document language (UI-style code, e.g. "en", "tr"). Mapped to a
    # PaddleOCR model via ocrcontext.utils.lang.normalize_paddle_lang.
    lang: str = "en"

    # Prefer a digital PDF's embedded text layer over OCR when it is sufficient.
    prefer_pdf_text_layer: bool = True

    # PDF rasterization scales.
    pdf_render_scale: float = OCR_PDF_RENDER_SCALE
    pdf_render_scale_handwriting: float = OCR_PDF_RENDER_SCALE_HANDWRITING

    # Line-band fallback thresholds.
    min_lines_per_page: int = MIN_EXPECTED_LINES_PER_PAGE
    min_lines_handwriting: int = MIN_EXPECTED_LINES_HANDWRITING

    # When True, automatically retry with the handwriting engine if printed OCR
    # returns insufficient text (mirrors the documents/process retry ladder).
    # Disabled by default: PaddleOCR is the sole default OCR engine; enable
    # explicitly when [vision] is installed.
    auto_handwriting_fallback: bool = False

    # Default refinement behaviour when Analyzer.analyze(refine=None):
    #   - refine when an LLM is configured AND the text did not come from an exact
    #     digital PDF text layer (which must not be "corrected").
    refine_by_default: bool = True
