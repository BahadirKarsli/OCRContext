"""LangChain document loader backed by ocrcontext.

    from ocrcontext.loaders import OCRContextLoader

    loader = OCRContextLoader("invoice.pdf")
    docs = loader.load()          # -> [Document(page_content=..., metadata={...})]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from .analyzer import Analyzer
from .config import AnalyzerConfig
from .engines.registry import EngineRegistry

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class OCRContextLoader(BaseLoader):
    """LangChain ``BaseLoader`` that OCRs a PDF or image file via ocrcontext.

    Each ``load()`` / ``lazy_load()`` call returns a single ``Document`` whose
    ``page_content`` is the extracted (and optionally LLM-refined) text.
    Metadata keys: ``source``, ``text_source``, ``pages``, ``confidence``,
    ``refined`` (and ``raw_text`` when refinement changed the text).

    Parameters
    ----------
    file_path:
        Path to a PDF or image file.
    llm:
        Optional LangChain ``BaseChatModel`` for text refinement.
        Bring your own provider (``langchain_openai.ChatOpenAI`` etc.).
    lang:
        Document language code (default ``"en"``).
    handwriting:
        Force the handwriting engine instead of the default OCR path.
    refine:
        Override auto-refine behaviour: ``None`` = auto, ``True`` = always,
        ``False`` = never.
    config:
        Advanced ``AnalyzerConfig`` (overrides ``lang`` if both are set).
    registry:
        Shared ``EngineRegistry`` (defaults to the process-wide singleton).
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        llm: "Optional[BaseChatModel]" = None,
        lang: str = "en",
        handwriting: bool = False,
        refine: Optional[bool] = None,
        config: Optional[AnalyzerConfig] = None,
        registry: Optional[EngineRegistry] = None,
    ) -> None:
        self.file_path = Path(file_path)
        self._analyzer = Analyzer(llm=llm, lang=lang, config=config, registry=registry)
        self._handwriting = handwriting
        self._refine = refine

    def lazy_load(self) -> Iterator[Document]:
        result = self._analyzer.analyze(
            self.file_path,
            handwriting=self._handwriting,
            refine=self._refine,
        )
        metadata: dict = {
            "source": str(self.file_path),
            "text_source": result.text_source,
            "pages": result.pages,
            "confidence": result.confidence,
            "refined": result.refined,
        }
        if result.refined and result.raw_text is not None:
            metadata["raw_text"] = result.raw_text
        yield Document(page_content=result.text, metadata=metadata)
