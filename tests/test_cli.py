"""Tests for the ocrcontext CLI (typer-based)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ocrcontext.cli import app, _parse_refine, _SCHEMAS


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_png(tmp_path: Path, name: str = "page.png") -> Path:
    p = tmp_path / name
    p.write_bytes(PNG_BYTES)
    return p


def _patch_analyzer(monkeypatch, ocr_text: str = "hello world", structured: dict | None = None):
    """Monkeypatch Analyzer and _build_llm so no real OCR or LLM runs."""
    from ocrcontext import types

    fake_result = types.OcrResult(
        text=ocr_text,
        text_source="ocr",
        pages=1,
        confidence=0.9,
    )

    import ocrcontext.cli as cli_mod

    class _FakeAnalyzer:
        def __init__(self, **kwargs):
            pass

        def analyze(self, *args, **kwargs):
            return fake_result

        def extract(self, *args, schema=None, **kwargs):
            return schema(**(structured or {}))

    monkeypatch.setattr(cli_mod, "Analyzer", _FakeAnalyzer)
    monkeypatch.setattr(cli_mod, "_build_llm", lambda provider, model: None)


# ---------------------------------------------------------------------------
# _parse_refine
# ---------------------------------------------------------------------------


def test_parse_refine_auto():
    assert _parse_refine("auto") is None


def test_parse_refine_yes():
    assert _parse_refine("yes") is True


def test_parse_refine_no():
    assert _parse_refine("no") is False


# ---------------------------------------------------------------------------
# SCHEMA_MAP completeness
# ---------------------------------------------------------------------------


def test_schema_map_has_expected_keys():
    assert set(_SCHEMAS) == {"invoice", "receipt", "contract", "idcard", "medical"}


# ---------------------------------------------------------------------------
# extract — text output (no schema)
# ---------------------------------------------------------------------------


def test_extract_text_output(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(monkeypatch, ocr_text="extracted text here")

    result = runner.invoke(app, ["extract", str(png)])
    assert result.exit_code == 0
    assert "extracted text here" in result.output


def test_extract_json_output(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(monkeypatch, ocr_text="some text")

    result = runner.invoke(app, ["extract", str(png), "--output", "json"])
    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    assert data["text"] == "some text"
    assert data["text_source"] == "ocr"


# ---------------------------------------------------------------------------
# extract — schema output
# ---------------------------------------------------------------------------


def test_extract_invoice_schema(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(
        monkeypatch,
        structured={"supplier_name": "ACME", "total_amount": 250.0},
    )

    result = runner.invoke(app, ["extract", str(png), "--schema", "invoice"])
    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    assert data["supplier_name"] == "ACME"
    assert data["total_amount"] == 250.0


def test_extract_receipt_schema(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(monkeypatch, structured={"store_name": "Migros", "total_amount": 45.0})

    result = runner.invoke(app, ["extract", str(png), "--schema", "receipt"])
    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    assert data["store_name"] == "Migros"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unknown_schema_exits_with_error(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(monkeypatch)

    result = runner.invoke(app, ["extract", str(png), "--schema", "banana"])
    assert result.exit_code != 0
    assert "banana" in result.output or "Unknown schema" in result.output


def test_invalid_output_format_exits_with_error(ascii_tmp, monkeypatch):
    png = _write_png(ascii_tmp)
    _patch_analyzer(monkeypatch)

    result = runner.invoke(app, ["extract", str(png), "--output", "xml"])
    assert result.exit_code != 0


def test_missing_file_exits_with_error(ascii_tmp):
    result = runner.invoke(app, ["extract", str(ascii_tmp / "ghost.png")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --help sanity
# ---------------------------------------------------------------------------


def test_help_exits_cleanly():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output


def test_extract_help_shows_options():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--schema" in result.output
    assert "--lang" in result.output
    assert "--provider" in result.output
