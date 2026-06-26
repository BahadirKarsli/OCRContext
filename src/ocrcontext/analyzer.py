"""The public facade: instantiate, pass a document, get text or a Pydantic model.

    from ocrcontext import Analyzer
    result = Analyzer().analyze("invoice.pdf")
    print(result.text)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

from pydantic import BaseModel

from .config import AnalyzerConfig
from .engines.registry import EngineRegistry
from .exceptions import LLMNotConfiguredError
from .pipeline import Pipeline
from .quality import handwriting_refinement_mode
from .types import OcrResult, RefinementMode
from .utils.files import Source

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from .llm.extractor import StructuredExtractor
    from .llm.refiner import Refiner

TSchema = TypeVar("TSchema", bound=BaseModel)

_HANDWRITING_SOURCES = {"vision_handwriting", "handwriting_ocr"}


class Analyzer:
    """High-level document analyzer.

    Parameters
    ----------
    llm:
        Optional LangChain ``BaseChatModel``. Required only for ``refine``/``extract``.
        Bring your own provider (``langchain_openai.ChatOpenAI`` etc.).
    lang:
        Default document language code (e.g. ``"en"``, ``"tr"``).
    config:
        Advanced pipeline tuning. Overrides ``lang`` if both are set.
    registry:
        Shared engine registry (singleton model cache). Defaults to a process-wide
        shared instance so PaddleOCR/TrOCR load at most once.
    """

    def __init__(
        self,
        llm: "Optional[BaseChatModel]" = None,
        *,
        lang: str = "en",
        config: Optional[AnalyzerConfig] = None,
        registry: Optional[EngineRegistry] = None,
    ) -> None:
        self._llm = llm
        self.config = config or AnalyzerConfig(lang=lang)
        self.registry = registry or EngineRegistry.shared()
        self._pipeline = Pipeline(registry=self.registry, config=self.config)
        self._refiner: "Refiner | None" = None
        self._extractor: "StructuredExtractor | None" = None

    # --- Public API ----------------------------------------------------------

    def analyze(
        self,
        source: Source,
        *,
        handwriting: bool = False,
        refine: Optional[bool] = None,
        lang: Optional[str] = None,
        mode: Optional[RefinementMode] = None,
        filename: Optional[str] = None,
    ) -> OcrResult:
        """OCR a document (PDF/image) and optionally LLM-refine the text.

        ``refine=None`` (default) refines only when an LLM is configured and the
        text did not come from an exact digital PDF text layer.
        """
        result = self._pipeline.run(
            source, lang=lang, handwriting=handwriting, filename=filename
        )

        if self._should_refine(result, refine):
            chosen_mode = mode or self._infer_mode(result)
            refined = self.refine(
                result.text, language=lang or self.config.lang, mode=chosen_mode
            )
            if refined != result.text:
                result.raw_text = result.text
                result.text = refined
                result.refined = True

        return result

    def extract(
        self,
        source: Source,
        schema: type[TSchema],
        *,
        handwriting: bool = False,
        refine: bool = False,
        lang: Optional[str] = None,
        system_prompt: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> TSchema:
        """OCR a document and extract a structured Pydantic model from it.

        Refinement is OFF by default for extraction (the LLM extractor reads raw
        OCR text directly, mirroring the original invoice pipeline).
        """
        result = self.analyze(
            source,
            handwriting=handwriting,
            refine=refine,
            lang=lang,
            filename=filename,
        )
        return self.extract_text(
            result.text,
            schema,
            language=lang or self.config.lang,
            system_prompt=system_prompt,
        )

    def extract_text(
        self,
        text: str,
        schema: type[TSchema],
        *,
        language: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> TSchema:
        """Extract a structured Pydantic model from already-OCR'd text.

        Useful when you already have text (e.g. from a prior ``analyze`` call) and
        want to avoid re-running OCR. Requires a configured LLM.
        """
        return self._get_extractor().extract(
            text,
            schema,
            language=language or self.config.lang,
            system_prompt=system_prompt,
        )

    def refine(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        mode: RefinementMode = RefinementMode.CONSERVATIVE,
    ) -> str:
        """Refine arbitrary OCR text directly (requires a configured LLM)."""
        return self._get_refiner().refine(
            text, language=language or self.config.lang, mode=mode
        )

    # --- Internals -----------------------------------------------------------

    def _should_refine(self, result: OcrResult, refine: Optional[bool]) -> bool:
        if refine is False:
            return False
        if refine is True:
            if self._llm is None:
                raise LLMNotConfiguredError("refine=True")
            return True
        # refine is None -> auto
        if self._llm is None or not self.config.refine_by_default:
            return False
        # Never "correct" an exact digital PDF text layer.
        return result.text_source != "pdf_text_layer"

    def _infer_mode(self, result: OcrResult) -> RefinementMode:
        if result.text_source in _HANDWRITING_SOURCES:
            return handwriting_refinement_mode(result.text, result.has_dikw_structure)
        if result.text_source == "pdf_text_layer":
            return RefinementMode.LAYOUT
        return RefinementMode.CONSERVATIVE

    def _get_refiner(self) -> "Refiner":
        if self._llm is None:
            raise LLMNotConfiguredError("Refinement")
        if self._refiner is None:
            from .llm.refiner import Refiner

            self._refiner = Refiner(self._llm)
        return self._refiner

    def _get_extractor(self) -> "StructuredExtractor":
        if self._llm is None:
            raise LLMNotConfiguredError("Structured extraction")
        if self._extractor is None:
            from .llm.extractor import StructuredExtractor

            self._extractor = StructuredExtractor(self._llm)
        return self._extractor
