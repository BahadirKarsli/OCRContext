from ocrcontext.llm.extractor import StructuredExtractor
from ocrcontext.llm.refiner import Refiner
from ocrcontext.schemas import Invoice
from ocrcontext.types import RefinementMode

from .conftest import FakeChatModel


def test_refiner_fixes_text_and_binds_temperature():
    llm = FakeChatModel(text="Hello world\nthis is a test\nof ocr text here")
    refiner = Refiner(llm)
    out = refiner.refine(
        "Helo world\nthis is a tset\nof ocr text here",
        language="en",
        mode=RefinementMode.CONSERVATIVE,
    )
    assert out == "Hello world\nthis is a test\nof ocr text here"
    # Conservative mode -> temperature 0 was bound.
    assert llm.bound_kwargs == {"temperature": 0.0}


def test_refiner_preserves_literals_even_if_model_changes_them():
    # Model returns refined text but "fixes" the email — enforcement must revert it.
    llm = FakeChatModel(text="Contact bahadirkarsli@outlook.com today please now")
    refiner = Refiner(llm)
    out = refiner.refine(
        "Contact bahadrkrsl@outlook.com today please now",
        language="en",
        mode=RefinementMode.CONSERVATIVE,
    )
    assert "bahadrkrsl@outlook.com" in out


def test_refiner_rejects_drift_keeps_original():
    original = "one short faithful line of source"
    llm = FakeChatModel(text="completely\ndifferent\nhallucinated\noutput\nhere\nnow\nmore")
    refiner = Refiner(llm)
    out = refiner.refine(original, language="en", mode=RefinementMode.CONSERVATIVE)
    assert out == original


def test_analyzer_extract_text_from_existing_text():
    from ocrcontext import Analyzer

    llm = FakeChatModel(structured={"supplier_name": "Beta Co", "currency": "USD"})
    analyzer = Analyzer(llm=llm)
    invoice = analyzer.extract_text("already ocr'd invoice text", Invoice, language="en")
    assert invoice.supplier_name == "Beta Co"
    assert invoice.currency == "USD"


def test_structured_extractor_returns_schema_instance():
    llm = FakeChatModel(
        structured={
            "supplier_name": "ACME Ltd",
            "currency": "TRY",
            "total_amount": 1200.50,
            "line_items": [
                {"description": "Widget", "unit_price": 100.0, "total": 500.0}
            ],
        }
    )
    extractor = StructuredExtractor(llm)
    invoice = extractor.extract("raw invoice text", Invoice, language="tr")
    assert isinstance(invoice, Invoice)
    assert invoice.supplier_name == "ACME Ltd"
    # quantity back-fill: 500 / 100 = 5
    assert invoice.line_items[0].quantity == 5.0
