"""Exception hierarchy for ocrcontext."""

from __future__ import annotations


class OcrContextError(Exception):
    """Base class for all ocrcontext errors."""


class MissingDependencyError(OcrContextError):
    """A required optional dependency (extra) is not installed.

    Raised lazily when an engine is first used so the base install stays light.
    """

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"'{package}' is required for this feature but is not installed. "
            f"Install it with:  pip install 'ocrcontext[{extra}]'"
        )


class UnsupportedFileError(OcrContextError):
    """The provided file type / source could not be interpreted."""


class NoTextDetectedError(OcrContextError):
    """OCR produced no usable text from the document."""


class LLMNotConfiguredError(OcrContextError):
    """An LLM-dependent operation was requested without injecting a chat model."""

    def __init__(self, operation: str = "this operation") -> None:
        super().__init__(
            f"{operation} requires a LangChain chat model. Pass one to Analyzer(llm=...), e.g.\n"
            "    from langchain_openai import ChatOpenAI\n"
            "    analyzer = Analyzer(llm=ChatOpenAI(model='gpt-4o'))"
        )


class EngineError(OcrContextError):
    """An OCR engine failed to initialize or run."""
