"""Tests for OCRContextLoader (LangChain BaseLoader integration)."""

from __future__ import annotations

import pytest

from ocrcontext import AnalyzerConfig, EngineRegistry, OCRContextLoader

from .conftest import FakeChatModel, FakeEngine

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _registry(engine: FakeEngine) -> EngineRegistry:
    reg = EngineRegistry()
    reg._paddle = engine
    return reg


def _no_fallback() -> AnalyzerConfig:
    return AnalyzerConfig(auto_handwriting_fallback=False)


def test_loader_returns_one_document(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(PNG_BYTES)

    reg = _registry(FakeEngine(["hello world"]))
    loader = OCRContextLoader(png, registry=reg, config=_no_fallback())
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "hello world"


def test_loader_metadata_keys(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(PNG_BYTES)

    reg = _registry(FakeEngine(["hello world"]))
    loader = OCRContextLoader(png, registry=reg, config=_no_fallback())
    doc = loader.load()[0]

    assert doc.metadata["source"] == str(png)
    assert doc.metadata["text_source"] == "ocr"
    assert doc.metadata["pages"] == 1
    assert "confidence" in doc.metadata
    assert doc.metadata["refined"] is False
    assert "raw_text" not in doc.metadata


def test_loader_refined_metadata(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(PNG_BYTES)

    reg = _registry(FakeEngine(["helo wrld foo bar baz qux"]))
    llm = FakeChatModel(text="hello world foo bar baz qux")
    loader = OCRContextLoader(png, llm=llm, registry=reg, config=_no_fallback())
    doc = loader.load()[0]

    assert doc.metadata["refined"] is True
    assert doc.metadata["raw_text"] == "helo wrld foo bar baz qux"
    assert doc.page_content == "hello world foo bar baz qux"


def test_loader_refine_false_skips_llm(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(PNG_BYTES)

    reg = _registry(FakeEngine(["raw text"]))
    llm = FakeChatModel(text="should not be used")
    loader = OCRContextLoader(png, llm=llm, refine=False, registry=reg, config=_no_fallback())
    doc = loader.load()[0]

    assert doc.page_content == "raw text"
    assert doc.metadata["refined"] is False
    assert llm.invocations == []


def test_loader_lazy_load_is_iterator(tmp_path):
    png = tmp_path / "page.png"
    png.write_bytes(PNG_BYTES)

    reg = _registry(FakeEngine(["text"]))
    loader = OCRContextLoader(png, registry=reg, config=_no_fallback())
    it = loader.lazy_load()
    doc = next(it)
    assert doc.page_content == "text"
    with pytest.raises(StopIteration):
        next(it)
