"""Composite handwriting engine: Google Vision primary, TrOCR fallback.

Mirrors ocr-service/modal_app.py::HandwritingOCRService per-page logic without
the Modal class wrapper. Each sub-engine is loaded lazily on first use.
"""

from __future__ import annotations

from ..preprocessing.image import preprocess_image_for_ocr
from ..utils.files import cleanup_paths
from .base import OcrEngine, PageOcr
from .trocr import TrOCRHandwritingEngine, run_trocr_on_page
from .vision import GoogleVisionHandwritingEngine, detect_dikw_structure, run_vision_on_page


class HandwritingEngine(OcrEngine):
    """Vision-first handwriting recognition with a TrOCR fallback per page."""

    text_source = "handwriting_ocr"

    def __init__(self) -> None:
        self._vision: GoogleVisionHandwritingEngine | None = None
        self._trocr: TrOCRHandwritingEngine | None = None

    def _ensure_vision(self) -> GoogleVisionHandwritingEngine:
        if self._vision is None:
            engine = GoogleVisionHandwritingEngine()
            engine.load()  # no-op disable if creds missing; raises only if pkg absent
            self._vision = engine
        return self._vision

    def _ensure_trocr(self) -> TrOCRHandwritingEngine:
        if self._trocr is None:
            engine = TrOCRHandwritingEngine()
            engine.load()
            engine.warmup_inference()
            self._trocr = engine
        return self._trocr

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
            used_vision = False
            used_trocr = False
            has_dikw = False

            # Vision is optional: load() leaves it disabled when no credentials exist.
            try:
                vision = self._ensure_vision()
            except Exception:
                vision = None

            if vision is not None and vision.enabled:
                try:
                    page_text, page_conf = run_vision_on_page(
                        vision, ocr_img_path, ocr_lang=lang
                    )
                    if page_text:
                        used_vision = True
                        if vision.last_has_dikw_structure:
                            has_dikw = True
                except Exception:
                    page_text = ""

            if not page_text:
                trocr = self._ensure_trocr()
                page_text, page_conf = run_trocr_on_page(trocr, ocr_img_path)
                if page_text:
                    used_trocr = True

            text_source = (
                "vision_handwriting"
                if used_vision and not used_trocr
                else "trocr_handwriting"
                if used_trocr
                else "handwriting_ocr"
            )

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
