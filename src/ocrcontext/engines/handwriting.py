"""Handwriting engine: Google Vision only.

Mirrors ocr-service/modal_app.py::HandwritingOCRService per-page logic without
the Modal class wrapper. The engine is loaded lazily on first use.
"""

from __future__ import annotations

from ..preprocessing.image import preprocess_image_for_ocr
from ..utils.files import cleanup_paths
from .base import OcrEngine, PageOcr
from .vision import GoogleVisionHandwritingEngine, detect_dikw_structure, run_vision_on_page


class HandwritingEngine(OcrEngine):
    """Google Vision handwriting recognition."""

    text_source = "handwriting_ocr"

    def __init__(self) -> None:
        self._vision: GoogleVisionHandwritingEngine | None = None

    def _ensure_vision(self) -> GoogleVisionHandwritingEngine:
        if self._vision is None:
            engine = GoogleVisionHandwritingEngine()
            engine.load()
            self._vision = engine
        return self._vision

    def recognize(
        self,
        img_path: str,
        *,
        lang: str = "en",
        min_lines: int = 1,
        handwriting: bool = True,
    ) -> PageOcr:
        preprocessed: list[str] = []
        try:
            ocr_img_path = preprocess_image_for_ocr(img_path, handwriting=True)
            if ocr_img_path != img_path:
                preprocessed.append(ocr_img_path)

            page_text = ""
            page_conf = 0.0
            has_dikw = False

            try:
                vision = self._ensure_vision()
            except Exception:
                vision = None

            if vision is not None and vision.enabled:
                try:
                    page_text, page_conf = run_vision_on_page(
                        vision, ocr_img_path, ocr_lang=lang
                    )
                    if page_text and vision.last_has_dikw_structure:
                        has_dikw = True
                except Exception:
                    page_text = ""

            text_source = "vision_handwriting" if page_text else "handwriting_ocr"

            if not has_dikw and page_text.strip():
                has_dikw = detect_dikw_structure(page_text)

            scores = [page_conf] if page_conf > 0 else []
            return PageOcr(
                text=page_text.strip(),
                scores=scores,
                has_dikw_structure=has_dikw,
                text_source=text_source,
            )
        finally:
            cleanup_paths(preprocessed)
