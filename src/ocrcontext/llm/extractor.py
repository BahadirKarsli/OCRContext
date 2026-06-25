"""LLM-agnostic structured extraction via LangChain's ``with_structured_output``.

Give it any Pydantic schema and a chat model; get a populated model instance
back. The Invoice schema in :mod:`ocrcontext.llm.schemas` is auto-detected so it
uses the verbatim invoice prompt, but any schema works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from .schemas import INVOICE_EXTRACTION_PROMPT, Invoice

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

TSchema = TypeVar("TSchema", bound=BaseModel)

_GENERIC_PROMPT = (
    "You are an expert document data extraction assistant. The text may come from "
    "OCR and may contain scanning errors and missing characters. Extract the "
    "requested fields faithfully from the document text. Do not invent values: if a "
    "field is not present in the text, leave it null/empty. Preserve the document's "
    "original language for textual fields and do not translate."
)


class StructuredExtractor:
    """Extract a Pydantic schema from raw text using an injected chat model."""

    def __init__(self, llm: "BaseChatModel") -> None:
        self._llm = llm

    def extract(
        self,
        text: str,
        schema: type[TSchema],
        *,
        language: str = "auto",
        system_prompt: str | None = None,
    ) -> TSchema:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = system_prompt or self._default_prompt(schema)
        user = (
            f"Language Context: {language}\n"
            f"Extract detailed data from this document text:\n\n{text}"
        )

        structured = self._llm.with_structured_output(schema)
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        # with_structured_output returns an instance of `schema`.
        return result  # type: ignore[return-value]

    @staticmethod
    def _default_prompt(schema: type[BaseModel]) -> str:
        if schema is Invoice:
            return INVOICE_EXTRACTION_PROMPT
        return _GENERIC_PROMPT
