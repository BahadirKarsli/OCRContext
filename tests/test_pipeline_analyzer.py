import pytest

from ocrcontext import Analyzer, AnalyzerConfig, EngineRegistry, LLMNotConfiguredError

from .conftest import FakeChatModel, FakeEngine

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # engines are faked; content unused


# Disable the auto-handwriting fallback so paddle-only tests never reach the
# (unfaked) handwriting engine on short strings.
def _no_fallback_config(lang: str = "en") -> AnalyzerConfig:
    return AnalyzerConfig(lang=lang, auto_handwriting_fallback=False)


def _registry_with_paddle(engine: FakeEngine) -> EngineRegistry:
    reg = EngineRegistry()
    reg._paddle = engine  # inject fake, bypass lazy load
    return reg


def test_analyze_image_raw_ocr_no_llm():
    reg = _registry_with_paddle(FakeEngine(["hello world from ocr"]))
    analyzer = Analyzer(registry=reg, config=_no_fallback_config())
    result = analyzer.analyze(PNG_BYTES, filename="page.png")
    assert result.text == "hello world from ocr"
    assert result.text_source == "ocr"
    assert result.refined is False


def test_analyze_refine_true_without_llm_raises():
    reg = _registry_with_paddle(FakeEngine(["text"]))
    analyzer = Analyzer(registry=reg, config=_no_fallback_config())
    with pytest.raises(LLMNotConfiguredError):
        analyzer.analyze(PNG_BYTES, filename="page.png", refine=True)


def test_analyze_auto_refines_with_llm():
    reg = _registry_with_paddle(FakeEngine(["helo wrld foo bar baz qux"]))
    llm = FakeChatModel(text="hello world foo bar baz qux")
    analyzer = Analyzer(llm=llm, registry=reg, config=_no_fallback_config())
    result = analyzer.analyze(PNG_BYTES, filename="page.png")
    assert result.refined is True
    assert result.text == "hello world foo bar baz qux"
    assert result.raw_text == "helo wrld foo bar baz qux"


def test_extract_uses_raw_text_by_default():
    from ocrcontext.schemas import Invoice

    reg = _registry_with_paddle(FakeEngine(["ACME invoice total 100"]))
    llm = FakeChatModel(structured={"supplier_name": "ACME", "total_amount": 100.0})
    analyzer = Analyzer(llm=llm, registry=reg, config=_no_fallback_config())
    invoice = analyzer.extract(PNG_BYTES, schema=Invoice, filename="inv.png")
    assert invoice.supplier_name == "ACME"


def test_auto_handwriting_fallback_triggers_on_insufficient_text():
    reg = EngineRegistry()
    reg._paddle = FakeEngine(["ab"])  # too little text -> insufficient
    reg._handwriting = FakeEngine(
        ["recovered handwritten sentence with plenty of words here"],
        text_source="trocr_handwriting",
    )
    analyzer = Analyzer(registry=reg)  # fallback enabled (default)
    result = analyzer.analyze(PNG_BYTES, filename="page.png")
    assert result.text_source == "trocr_handwriting"
    assert "recovered handwritten sentence" in result.text


def test_registry_singleton_shared():
    a = EngineRegistry.shared()
    b = EngineRegistry.shared()
    assert a is b


def test_config_lang_threads_through():
    cfg = _no_fallback_config(lang="tr")
    reg = _registry_with_paddle(FakeEngine(["metin"]))
    analyzer = Analyzer(config=cfg, registry=reg)
    result = analyzer.analyze(PNG_BYTES, filename="page.png")
    assert result.text == "metin"


def test_infer_mode_for_handwriting(monkeypatch):
    reg = EngineRegistry()
    reg._handwriting = FakeEngine(
        ["W Wisdom\nK Knowledge\nI Information\nD Data"],
        has_dikw=True,
        text_source="vision_handwriting",
    )
    llm = FakeChatModel(text="W Wisdom\nK Knowledge\nI Information\nD Data")
    analyzer = Analyzer(llm=llm, registry=reg)
    result = analyzer.analyze(PNG_BYTES, filename="note.png", handwriting=True)
    assert result.text_source == "vision_handwriting"
    assert result.has_dikw_structure is True
