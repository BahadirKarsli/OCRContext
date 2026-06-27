"""Command-line interface for ocrcontext.

Install the CLI extra to use:
    pip install "ocrcontext[cli]"

Then run:
    ocrcontext extract invoice.pdf
    ocrcontext extract scan.pdf --schema receipt --output json
    ocrcontext extract note.png --handwriting --provider anthropic --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:  # pragma: no cover
    sys.exit(
        "The CLI requires the 'cli' extra.\n"
        "Install it with:  pip install \"ocrcontext[cli]\""
    )

from .analyzer import Analyzer
from .config import AnalyzerConfig
from .types import OcrResult
from .schemas import (
    Contract,
    IdCard,
    Invoice,
    MedicalReport,
    Receipt,
)

def _suppress_paddle_noise() -> None:
    import logging
    import warnings

    # Set env vars BEFORE any paddle/paddlex import so they see the right paths.
    # _ensure_ascii_model_cache() in paddle.py does the same but only when the
    # engine lazy-loads; calling it here guarantees it runs first.
    from .engines.paddle import _ensure_ascii_model_cache, _ensure_paddle_runtime_flags
    _ensure_ascii_model_cache()
    _ensure_paddle_runtime_flags()

    os.environ.setdefault("GLOG_minloglevel", "3")

    # Silence Python-level loggers (no paddlex import — that would defeat the purpose).
    null = logging.NullHandler()
    for name in ("ppocr", "paddlex", "paddle", "paddle.utils", "paddle.fluid"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = [null]
        lg.propagate = False

    # Root-level filter catches sub-loggers that bypass the above (e.g. paddlex.utils.*).
    class _NoiseFilter(logging.Filter):
        _NOISE = ("Could not find files", "ccache", "oneDNN", "mkldnn")
        def filter(self, record: logging.LogRecord) -> bool:
            return not any(t in record.getMessage() for t in self._NOISE)

    logging.getLogger().addFilter(_NoiseFilter())

    warnings.filterwarnings("ignore", category=UserWarning, module="paddle")




def _route_label(result: OcrResult, file_path: Path) -> str:
    src = result.text_source
    if src == "pdf_text_layer":
        return "DIGITAL PDF -> text layer"
    if src == "ocr":
        return "SCANNED PDF -> rasterize + PaddleOCR" if file_path.suffix.lower() == ".pdf" else "IMAGE -> PaddleOCR"
    if src == "vision_handwriting":
        return "HANDWRITING -> Google Vision"
    if src == "handwriting_ocr":
        return "HANDWRITING -> PaddleOCR"
    return src


def _info(msg: str) -> None:
    typer.echo(f"[i] {msg}", err=True)


def _ok(msg: str) -> None:
    typer.echo(f"[OK] {msg}", err=True)


app = typer.Typer(
    name="ocrcontext",
    help="OCR a document and optionally extract structured data.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)


@app.callback()
def _root() -> None:
    """ocrcontext — document OCR and structured extraction."""

_SCHEMAS = {
    "invoice": Invoice,
    "receipt": Receipt,
    "contract": Contract,
    "idcard": IdCard,
    "medical": MedicalReport,
}

_SCHEMA_NAMES = list(_SCHEMAS)


def _build_llm(provider: str, model: str):
    """Dynamically import the right LangChain provider class."""
    _API_KEY_HINTS = {
        "openai":    ("OPENAI_API_KEY",    "platform.openai.com/api-keys"),
        "anthropic": ("ANTHROPIC_API_KEY", "console.anthropic.com/settings/keys"),
        "google":    ("GOOGLE_API_KEY",    "aistudio.google.com/apikey"),
        "ollama":    (None, None),
    }

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]
            return ChatOpenAI(model=model)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
            return ChatAnthropic(model=model)
        if provider == "ollama":
            from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
            return ChatOllama(model=model)
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-untyped]
            return ChatGoogleGenerativeAI(model=model)
    except ImportError:
        typer.echo(
            f"[ERROR] Provider '{provider}' requires its LangChain package.\n"
            f"Install it with:  pip install langchain-{provider}",
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        msg = str(exc)
        if "api_key" in msg.lower() or "credentials" in msg.lower() or "auth" in msg.lower():
            env_var, url = _API_KEY_HINTS.get(provider, (None, None))
            hint = f"Set it with:  $env:{env_var} = \"...\"" if env_var else ""
            url_hint = f"\nGet a key at: {url}" if url else ""
            typer.echo(
                f"[ERROR] No API key found for '{provider}'.\n{hint}{url_hint}",
                err=True,
            )
        else:
            typer.echo(f"[ERROR] Failed to initialize '{provider}': {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"[ERROR] Unknown provider '{provider}'. "
        f"Choose from: openai, anthropic, ollama, google",
        err=True,
    )
    raise typer.Exit(code=1)


def _parse_refine(value: str) -> Optional[bool]:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None  # auto


@app.command()
def extract(
    file: str = typer.Argument(..., help="PDF or image file to process."),
    schema: Optional[str] = typer.Option(
        None,
        "--schema", "-s",
        help=f"Built-in schema to extract: {', '.join(_SCHEMA_NAMES)}.",
    ),
    lang: str = typer.Option("en", "--lang", "-l", help="Document language code (en, tr, de, …)."),
    handwriting: bool = typer.Option(False, "--handwriting", help="Force handwriting engine."),
    refine: str = typer.Option(
        "auto",
        "--refine",
        help="LLM text refinement: auto (default), yes, no.",
    ),
    output: str = typer.Option(
        "text",
        "--output", "-o",
        help="Output format: text (default) or json.",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider", "-p",
        help="LLM provider when schema or refine is used: openai, anthropic, ollama, google.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model", "-m",
        help="LLM model name passed to the provider.",
    ),
) -> None:
    """OCR a document and optionally extract structured data."""

    _suppress_paddle_noise()

    file_path = Path(file)
    if not file_path.exists():
        typer.echo(f"[ERROR] File not found: {file}", err=True)
        raise typer.Exit(code=1)

    if schema is not None and schema not in _SCHEMAS:
        typer.echo(
            f"[ERROR] Unknown schema '{schema}'. "
            f"Choose from: {', '.join(_SCHEMA_NAMES)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if output not in ("text", "json"):
        typer.echo("[ERROR] --output must be 'text' or 'json'.", err=True)
        raise typer.Exit(code=1)

    refine_flag = _parse_refine(refine)
    needs_llm = schema is not None or refine_flag is True
    llm = _build_llm(provider, model) if needs_llm else None

    analyzer = Analyzer(llm=llm, config=AnalyzerConfig(lang=lang))

    try:
        _info(f"file: {file_path.name}")
        _info("OCR...")

        ocr_result = analyzer.analyze(
                file_path,
                handwriting=handwriting,
                refine=refine_flag,
            )

        conf = f"confidence: {ocr_result.confidence:.0%}" if ocr_result.confidence < 1.0 else "exact"
        _ok(f"route: {_route_label(ocr_result, file_path)}  ({conf})")

        if ocr_result.refined:
            _ok("refine: APPLIED")

        if schema is not None:
            schema_cls = _SCHEMAS[schema]
            _info(f"extract: {schema} schema...")
            structured = analyzer.extract_text(
                ocr_result.text,
                schema_cls,
                language=lang,
            )
            _ok(f"extract: {schema} [OK]")
            typer.echo(structured.model_dump_json(indent=2))
        else:
            if output == "json":
                typer.echo(ocr_result.model_dump_json(indent=2))
            else:
                typer.echo(ocr_result.text)

    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1)


def main() -> None:  # entry-point shim
    app()
