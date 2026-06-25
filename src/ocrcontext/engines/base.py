"""Engine abstractions shared by all OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PageOcr:
    """Recognition output for a single page image."""

    text: str
    scores: list[float] = field(default_factory=list)
    # Set by the handwriting engine when the page looks like a DIKW/pyramid diagram.
    has_dikw_structure: bool = False
    # Engine-reported text source label (e.g. "vision_handwriting", "trocr_handwriting").
    text_source: str | None = None

    @property
    def line_count(self) -> int:
        return len([ln for ln in self.text.splitlines() if ln.strip()])


class OcrEngine(ABC):
    """Recognize text from a single page image on disk.

    Engines are responsible for their own preprocessing and for cleaning up any
    temporary files they create.
    """

    #: Default text_source label reported in OcrResult when this engine is used.
    text_source: str = "ocr"

    @abstractmethod
    def recognize(
        self,
        img_path: str,
        *,
        lang: str = "en",
        min_lines: int = 1,
        handwriting: bool = False,
    ) -> PageOcr:
        """Recognize a single page image and return its text + per-token scores."""
        raise NotImplementedError
