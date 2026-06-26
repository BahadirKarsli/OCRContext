"""Tests for built-in extraction schemas."""

from __future__ import annotations

import pytest

from ocrcontext import Analyzer, AnalyzerConfig, EngineRegistry
from ocrcontext.schemas import (
    CONTRACT_EXTRACTION_PROMPT,
    IDCARD_EXTRACTION_PROMPT,
    INVOICE_EXTRACTION_PROMPT,
    MEDICAL_REPORT_EXTRACTION_PROMPT,
    RECEIPT_EXTRACTION_PROMPT,
    Contract,
    ContractParty,
    IdCard,
    Invoice,
    MedicalReport,
    Medication,
    Receipt,
    ReceiptItem,
)

from .conftest import FakeChatModel, FakeEngine

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _setup(structured: dict, ocr_text: str = "text"):
    reg = EngineRegistry()
    reg._paddle = FakeEngine([ocr_text])
    llm = FakeChatModel(structured=structured)
    cfg = AnalyzerConfig(auto_handwriting_fallback=False)
    analyzer = Analyzer(llm=llm, registry=reg, config=cfg)
    return analyzer


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def test_receipt_basic():
    analyzer = _setup(
        {
            "store_name": "MarketX",
            "date": "2024-03-15",
            "total_amount": 45.50,
            "currency": "TRY",
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=Receipt, filename="r.png")
    assert result.store_name == "MarketX"
    assert result.total_amount == 45.50
    assert result.currency == "TRY"


def test_receipt_items():
    analyzer = _setup(
        {
            "items": [
                {"description": "Bread", "quantity": 2.0, "unit_price": 5.0, "total": 10.0}
            ],
            "total_amount": 10.0,
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=Receipt, filename="r.png")
    assert len(result.items) == 1
    assert result.items[0].description == "Bread"
    assert result.items[0].total == 10.0


def test_receipt_optional_fields_default_empty():
    analyzer = _setup({})
    result = analyzer.extract(PNG_BYTES, schema=Receipt, filename="r.png")
    assert result.store_name is None
    assert result.items == []


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_contract_basic():
    analyzer = _setup(
        {
            "title": "Service Agreement",
            "parties": [
                {"name": "Acme Corp", "role": "Buyer"},
                {"name": "DevShop", "role": "Seller"},
            ],
            "effective_date": "2024-01-01",
            "contract_value": 50000.0,
            "currency": "USD",
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=Contract, filename="c.png")
    assert result.title == "Service Agreement"
    assert len(result.parties) == 2
    assert result.parties[0].name == "Acme Corp"
    assert result.contract_value == 50000.0


def test_contract_empty_parties():
    analyzer = _setup({"title": "NDA"})
    result = analyzer.extract(PNG_BYTES, schema=Contract, filename="c.png")
    assert result.parties == []
    assert result.title == "NDA"


# ---------------------------------------------------------------------------
# IdCard
# ---------------------------------------------------------------------------


def test_idcard_basic():
    analyzer = _setup(
        {
            "document_type": "passport",
            "full_name": "Jane Doe",
            "date_of_birth": "1990-05-20",
            "nationality": "USA",
            "document_number": "A12345678",
            "expiry_date": "2030-05-20",
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=IdCard, filename="id.png")
    assert result.document_type == "passport"
    assert result.full_name == "Jane Doe"
    assert result.nationality == "USA"


def test_idcard_optional_defaults():
    analyzer = _setup({"full_name": "John Smith"})
    result = analyzer.extract(PNG_BYTES, schema=IdCard, filename="id.png")
    assert result.full_name == "John Smith"
    assert result.gender is None
    assert result.address is None


# ---------------------------------------------------------------------------
# MedicalReport
# ---------------------------------------------------------------------------


def test_medical_report_basic():
    analyzer = _setup(
        {
            "patient_name": "Ali Veli",
            "report_date": "2024-06-01",
            "diagnosis": "Community-acquired pneumonia",
            "icd_codes": ["J18.9"],
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=MedicalReport, filename="med.png")
    assert result.patient_name == "Ali Veli"
    assert result.diagnosis == "Community-acquired pneumonia"
    assert "J18.9" in result.icd_codes


def test_medical_report_medications():
    analyzer = _setup(
        {
            "medications": [
                {"name": "Amoxicillin", "dosage": "500 mg", "frequency": "3x daily", "duration": "7 days"}
            ]
        }
    )
    result = analyzer.extract(PNG_BYTES, schema=MedicalReport, filename="med.png")
    assert len(result.medications) == 1
    assert result.medications[0].name == "Amoxicillin"
    assert result.medications[0].duration == "7 days"


def test_medical_report_empty_defaults():
    analyzer = _setup({})
    result = analyzer.extract(PNG_BYTES, schema=MedicalReport, filename="med.png")
    assert result.icd_codes == []
    assert result.medications == []


# ---------------------------------------------------------------------------
# Prompt constants exported
# ---------------------------------------------------------------------------


def test_all_prompts_are_strings():
    for prompt in [
        INVOICE_EXTRACTION_PROMPT,
        RECEIPT_EXTRACTION_PROMPT,
        CONTRACT_EXTRACTION_PROMPT,
        IDCARD_EXTRACTION_PROMPT,
        MEDICAL_REPORT_EXTRACTION_PROMPT,
    ]:
        assert isinstance(prompt, str)
        assert len(prompt) > 50
