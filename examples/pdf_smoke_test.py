"""End-to-end smoke test: the PDF routing ladder.

Drop any PDFs into examples/ (or pass paths) and this prints which path the
router chose for each:

    digital PDF  -> PyMuPDF text-layer extraction  (text_source=pdf_text_layer)
                    -> LLM refine AUTO-SKIPPED (exact text must not be "corrected")
    scanned PDF  -> rasterize pages -> PaddleOCR    (text_source=ocr)
                    -> LLM refine applied (if an LLM is configured)

Multi-page documents are joined with "--- Page N ---" separators.

Usage
-----
    python examples/pdf_smoke_test.py                       # all *.pdf in examples/
    python examples/pdf_smoke_test.py digital.pdf scan.pdf  # explicit files

Setup
-----
    pip install -e '.[paddle]'            # required for the scanned path
    pip install langchain-openai          # optional: to see refine vs skip live
    $env:OPENAI_API_KEY = "sk-..."        # optional (PowerShell)

Notes
-----
- The scanned path runs OCR per page (~tens of seconds/page on CPU).
- With an LLM configured, refine uses auto mode: digital PDFs are skipped without
  spending any tokens; only scanned text is sent to the model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ocrcontext import Analyzer, AnalyzerConfig
from ocrcontext.exceptions import MissingDependencyError, UnsupportedFileError
from ocrcontext.types import OcrResult

# Windows consoles often default to a legacy codec (e.g. cp1254) that can't encode
# all extracted characters. Force UTF-8 so printing never crashes on real text.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass


def _find_pdfs() -> list[Path]:
    if len(sys.argv) > 1:
        return [Path(a).expanduser() for a in sys.argv[1:]]
    here = Path(__file__).resolve().parent
    pdfs: list[Path] = []
    for d in (here, here.parent):
        pdfs.extend(sorted(d.glob("*.pdf")))
    # De-duplicate while preserving order.
    seen, unique = set(), []
    for p in pdfs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _build_llm():
    """Return a ChatOpenAI if available + keyed, else None (routing still works)."""
    if "OPENAI_API_KEY" not in os.environ:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None
    return ChatOpenAI(model=os.environ.get("OCRCONTEXT_MODEL", "gpt-4o-mini"), temperature=0)


def _preview(text: str, limit: int = 700) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... [truncated, {len(text)} chars total]"


def _report(path: Path, result: OcrResult, llm_configured: bool) -> None:
    is_digital = result.text_source == "pdf_text_layer"
    route = (
        "DIGITAL  -> PyMuPDF text layer (no OCR)"
        if is_digital
        else f"SCANNED  -> rasterize + PaddleOCR (source={result.text_source})"
    )

    print("\n" + "#" * 64)
    print(f"# {path.name}")
    print("#" * 64)
    print(f"route      : {route}")
    print(f"pages      : {result.pages}")
    print(f"confidence : {result.confidence}")
    print(f"chars      : {len(result.text)}")

    # The auto-skip-refine rule for exact digital text.
    if is_digital:
        assert result.refined is False, "BUG: digital PDF text should never be refined"
        print("refine     : AUTO-SKIPPED [OK]  (exact text layer - never 'corrected')")
    elif llm_configured:
        print(f"refine     : {'APPLIED [OK]' if result.refined else 'attempted (no change)'}")
    else:
        print("refine     : n/a (no LLM configured - set OPENAI_API_KEY to see it)")

    # Multi-page joining marker.
    if "--- Page " in result.text:
        print("multipage  : '--- Page N ---' separators present [OK]")

    print("-" * 64)
    print(_preview(result.text))


def main() -> int:
    pdfs = _find_pdfs()
    if not pdfs:
        print(
            "No PDFs found. Drop a digital PDF and a scanned PDF into examples/, or run:\n"
            "    python examples/pdf_smoke_test.py path/to/digital.pdf path/to/scan.pdf"
        )
        return 2

    missing = [p for p in pdfs if not p.exists()]
    if missing:
        for p in missing:
            print(f"[x] File not found: {p}")
        return 2

    llm = _build_llm()
    llm_configured = llm is not None
    print(f"[i] LLM configured: {llm_configured} "
          f"({'refine will run on scanned text' if llm_configured else 'routing only'})")
    print(f"[i] PDFs to test  : {', '.join(p.name for p in pdfs)}")

    # Pure routing test: disable handwriting fallback so a sparse scan doesn't pull
    # the Vision/TrOCR extras. refine=None (auto) honours the skip rule for digital.
    analyzer = Analyzer(
        llm=llm,
        config=AnalyzerConfig(lang=os.environ.get("OCRCONTEXT_LANG", "en"),
                              auto_handwriting_fallback=False),
    )

    digital = scanned = 0
    for path in pdfs:
        try:
            result = analyzer.analyze(path, refine=None)
        except MissingDependencyError as exc:
            print(f"\n[x] {path.name}: {exc}")
            return 1
        except UnsupportedFileError as exc:
            print(f"\n[x] {path.name}: {exc}")
            return 2

        _report(path, result, llm_configured)
        if result.text_source == "pdf_text_layer":
            digital += 1
        else:
            scanned += 1

    print("\n" + "=" * 64)
    print(f"SUMMARY: {len(pdfs)} PDF(s) -> {digital} digital (text layer), "
          f"{scanned} scanned (OCR)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
