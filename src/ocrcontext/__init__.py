"""ocrcontext — decoupled, LLM-agnostic document OCR + structured extraction.

Quick start::

    from ocrcontext import Analyzer
    result = Analyzer().analyze("invoice.pdf")
    print(result.text)

With an injected LangChain model::

    from langchain_openai import ChatOpenAI
    from ocrcontext import Analyzer
    from ocrcontext.schemas import Invoice

    analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o"))
    invoice = analyzer.extract("invoice.pdf", schema=Invoice)
"""

from __future__ import annotations

from .analyzer import Analyzer
from .config import AnalyzerConfig
from .engines.registry import EngineRegistry
from .loaders import OCRContextLoader
from .exceptions import (
    EngineError,
    LLMNotConfiguredError,
    MissingDependencyError,
    NoTextDetectedError,
    OcrContextError,
    UnsupportedFileError,
)
from .types import OcrResult, RefinementMode

__version__ = "0.1.1"

__all__ = [
    "Analyzer",
    "AnalyzerConfig",
    "EngineRegistry",
    "OCRContextLoader",
    "OcrResult",
    "RefinementMode",
    "OcrContextError",
    "MissingDependencyError",
    "UnsupportedFileError",
    "NoTextDetectedError",
    "LLMNotConfiguredError",
    "EngineError",
    "__version__",
]
