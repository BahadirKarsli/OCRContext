"""Shared test fixtures: fake chat models and OCR engines (no GPU / no network)."""

from __future__ import annotations

import pytest

from ocrcontext.engines.base import OcrEngine, PageOcr


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """Minimal duck-typed stand-in for a LangChain BaseChatModel.

    - ``bind(**kwargs)`` records kwargs and returns self.
    - ``invoke(messages)`` returns the configured text.
    - ``with_structured_output(schema)`` returns a runnable that builds ``schema``.
    """

    def __init__(self, *, text: str = "", structured: dict | None = None) -> None:
        self._text = text
        self._structured = structured or {}
        self.bound_kwargs: dict = {}
        self.invocations: list = []

    def bind(self, **kwargs):
        self.bound_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return FakeResponse(self._text)

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            def invoke(self, messages):
                parent.invocations.append(messages)
                return schema(**parent._structured)

        return _Structured()


class FakeEngine(OcrEngine):
    """Returns canned page text; never touches disk or a real model."""

    text_source = "ocr"

    def __init__(self, pages_text: list[str], *, has_dikw: bool = False,
                 text_source: str | None = None) -> None:
        self._pages = pages_text
        self._has_dikw = has_dikw
        self._idx = 0
        if text_source:
            self.text_source = text_source

    def recognize(self, img_path, *, lang="en", min_lines=1, handwriting=False) -> PageOcr:
        text = self._pages[self._idx] if self._idx < len(self._pages) else ""
        self._idx += 1
        return PageOcr(
            text=text,
            scores=[0.9] * len(text.split()),
            has_dikw_structure=self._has_dikw,
            text_source=self.text_source if handwriting else None,
        )


@pytest.fixture
def fake_chat():
    return FakeChatModel
