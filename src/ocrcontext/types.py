"""Public result and value types."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RefinementMode(str, Enum):
    """LLM post-processing modes, ported verbatim from the original pipeline.

    - ``layout``: digital PDFs — reconstruct clean structure.
    - ``conservative``: printed OCR images/scans — minimal char-level correction.
    - ``handwriting_layout``: handwritten notes/lists/tables/diagrams.
    - ``handwriting_prose``: handwritten poems/paragraphs/letters.
    """

    LAYOUT = "layout"
    CONSERVATIVE = "conservative"
    HANDWRITING_LAYOUT = "handwriting_layout"
    HANDWRITING_PROSE = "handwriting_prose"


# String identifying which engine produced the text. Matches the original
# `text_source` contract so downstream behaviour (e.g. skip-refine) is preserved.
TextSource = Literal[
    "pdf_text_layer",
    "ocr",
    "vision_handwriting",
    "handwriting_ocr",
]


class OcrResult(BaseModel):
    """The output of an OCR / analysis run."""

    text: str = Field(description="Extracted (and optionally refined) plain text.")
    confidence: float = Field(default=0.0, description="Mean recognition confidence (0..1).")
    pages: int = Field(default=1, description="Number of pages processed.")
    text_source: TextSource = Field(
        default="ocr", description="Which engine produced the text."
    )
    has_dikw_structure: bool = Field(
        default=False, description="True when handwriting looks like a DIKW/pyramid diagram."
    )
    refined: bool = Field(default=False, description="True if an LLM refined the raw OCR text.")
    raw_text: Optional[str] = Field(
        default=None, description="Original OCR text before refinement (when refined)."
    )

    def __str__(self) -> str:  # convenient: print(result) -> the text
        return self.text
