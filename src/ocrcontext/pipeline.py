"""Document routing + OCR orchestration (no LLM here — that's the analyzer's job).

Reproduces the retry/fallback ladder from app/api/documents/process/route.ts and
the page loops in OCRService / HandwritingOCRService, minus the web/Modal layer.
"""

from __future__ import annotations

from .config import AnalyzerConfig
from .engines.base import OcrEngine, PageOcr
from .engines.pdf_text import extract_pdf_text_preserve_layout, has_sufficient_pdf_text
from .engines.registry import EngineRegistry
from .quality import is_ocr_text_insufficient
from .types import OcrResult, TextSource
from .utils.files import (
    Source,
    cleanup_paths,
    is_pdf,
    load_source,
    new_temp_path,
    rasterize_pdf,
)


class Pipeline:
    """Routes a document to the right engine(s) and returns raw OCR text."""

    def __init__(
        self,
        registry: EngineRegistry | None = None,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self.registry = registry or EngineRegistry.shared()
        self.config = config or AnalyzerConfig()

    def run(
        self,
        source: Source,
        *,
        lang: str | None = None,
        handwriting: bool = False,
        filename: str | None = None,
    ) -> OcrResult:
        lang = lang or self.config.lang
        file_bytes, ext = load_source(source, filename=filename)

        # 1) Digital PDF text layer — exact text, no OCR / no GPU.
        if is_pdf(ext) and self.config.prefer_pdf_text_layer and not handwriting:
            full_text, page_count = extract_pdf_text_preserve_layout(file_bytes)
            if has_sufficient_pdf_text(full_text):
                return OcrResult(
                    text=full_text.strip(),
                    confidence=1.0,
                    pages=page_count,
                    text_source="pdf_text_layer",
                )

        # 2) Printed / scanned OCR (or handwriting if explicitly requested).
        result = self._ocr(file_bytes, ext, lang=lang, handwriting=handwriting)

        # 3) Auto handwriting fallback when printed OCR returns too little text.
        if (
            not handwriting
            and self.config.auto_handwriting_fallback
            and is_ocr_text_insufficient(result.text, result.pages)
        ):
            result = self._ocr(file_bytes, ext, lang=lang, handwriting=True)

        # 4) Vision→Paddle fallback: if Vision returned nothing, retry with PaddleOCR.
        if handwriting and is_ocr_text_insufficient(result.text, result.pages):
            result = self._ocr(file_bytes, ext, lang=lang, handwriting=False)

        return result

    def _ocr(
        self, file_bytes: bytes, ext: str, *, lang: str, handwriting: bool
    ) -> OcrResult:
        engine = (
            self.registry.handwriting() if handwriting else self.registry.paddle()
        )
        render_scale = (
            self.config.pdf_render_scale_handwriting
            if handwriting
            else self.config.pdf_render_scale
        )
        min_lines = (
            self.config.min_lines_handwriting
            if handwriting
            else self.config.min_lines_per_page
        )

        image_paths, owned = self._materialize_images(file_bytes, ext, render_scale)
        try:
            return self._ocr_pages(
                engine,
                image_paths,
                lang=lang,
                handwriting=handwriting,
                min_lines=min_lines,
            )
        finally:
            cleanup_paths(owned)

    @staticmethod
    def _materialize_images(
        file_bytes: bytes, ext: str, render_scale: float
    ) -> tuple[list[str], list[str]]:
        """Return (image_paths, owned_paths_to_cleanup)."""
        if is_pdf(ext):
            paths = rasterize_pdf(file_bytes, render_scale)
            return paths, list(paths)
        # Single image: write to a temp file the engines can read.
        path = new_temp_path(ext)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return [path], [path]

    @staticmethod
    def _ocr_pages(
        engine: OcrEngine,
        image_paths: list[str],
        *,
        lang: str,
        handwriting: bool,
        min_lines: int,
    ) -> OcrResult:
        full_text = ""
        all_scores: list[float] = []
        used_vision = False
        has_dikw = False

        for idx, img_path in enumerate(image_paths):
            if idx > 0:
                full_text += f"\n\n--- Page {idx + 1} ---\n\n"

            page: PageOcr = engine.recognize(
                img_path, lang=lang, min_lines=min_lines, handwriting=handwriting
            )
            full_text += page.text + ("\n" if (handwriting and page.text) else "")
            all_scores.extend(page.scores)
            if page.has_dikw_structure:
                has_dikw = True
            if page.text_source == "vision_handwriting":
                used_vision = True

        avg_conf = sum(all_scores) / len(all_scores) if all_scores else 0.0

        text_source: TextSource = engine.text_source  # type: ignore[assignment]
        if handwriting:
            text_source = "vision_handwriting" if used_vision else "handwriting_ocr"

        return OcrResult(
            text=full_text.strip(),
            confidence=round(avg_conf, 4),
            pages=len(image_paths),
            text_source=text_source,
            has_dikw_structure=has_dikw,
        )
