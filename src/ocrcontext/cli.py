"""Command-line interface for ocrcontext.

Install the CLI extra to use:
    pip install "ocrcontext[cli]"

Then run:
    ocrcontext extract invoice.pdf
    ocrcontext extract scan.pdf --schema receipt --output json
    ocrcontext extract note.png --handwriting --provider anthropic --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

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
from .schemas import (
    Contract,
    IdCard,
    Invoice,
    MedicalReport,
    Receipt,
)

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

    file_path = Path(file)
    if not file_path.exists():
        typer.echo(f"[ERROR] File not found: {file}", err=True)
        raise typer.Exit(code=1)

    # Validate --schema value early for a clear error message.
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

    # Build LLM only when needed.
    needs_llm = schema is not None or refine_flag is True
    llm = _build_llm(provider, model) if needs_llm else None

    analyzer = Analyzer(
        llm=llm,
        config=AnalyzerConfig(lang=lang),
    )

    try:
        if schema is not None:
            schema_cls = _SCHEMAS[schema]
            result = analyzer.extract(
                file_path,
                schema=schema_cls,
                handwriting=handwriting,
                refine=refine_flag or False,
            )
            typer.echo(result.model_dump_json(indent=2))
        else:
            result = analyzer.analyze(
                file_path,
                handwriting=handwriting,
                refine=refine_flag,
            )
            if output == "json":
                typer.echo(result.model_dump_json(indent=2))
            else:
                typer.echo(result.text)

    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1)


def main() -> None:  # entry-point shim
    app()
