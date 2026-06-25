"""End-to-end smoke test: OCR -> LLM refine -> structured extraction.

Pipeline exercised:
    image -> PaddleOCR -> (LLM refine) -> with_structured_output(schema) -> Pydantic model

Usage
-----
    python examples/structured_smoke_test.py                 # auto-find a sample image
    python examples/structured_smoke_test.py path/to/img.png

Setup
-----
    pip install -e '.[paddle]' langchain-openai

Then provide your OpenAI key for the session (PowerShell):
    $env:OPENAI_API_KEY = "sk-..."

Optional model override (defaults to gpt-4o-mini):
    $env:OCRCONTEXT_MODEL = "gpt-4o"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ocrcontext import Analyzer, AnalyzerConfig
from ocrcontext.exceptions import MissingDependencyError, UnsupportedFileError
from ocrcontext.types import RefinementMode


# --- The schema: what we want the "LLM brain" to pull out of the document ------
class LiteraryWork(BaseModel):
    """Generic literary / document entities."""

    title: Optional[str] = Field(None, description="The work's title, if present.")
    author: Optional[str] = Field(None, description="The author's name, if present.")
    main_text: Optional[str] = Field(
        None, description="The full body text of the work (poem, paragraph, etc.)."
    )
    date: Optional[str] = Field(None, description="Any date shown, format YYYY-MM-DD.")
    language: Optional[str] = Field(
        None, description="ISO language code of the text (e.g. 'tr', 'en')."
    )


_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _find_sample_image() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for d in (here, here.parent):
        for name in ("sample", "test", "image"):
            for ext in _IMAGE_EXTS:
                p = d / f"{name}{ext}"
                if p.exists():
                    return p
        for ext in _IMAGE_EXTS:
            matches = sorted(d.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


def _resolve_image() -> Optional[Path]:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    found = _find_sample_image()
    if found:
        print(f"[i] No path given — using discovered image: {found.name}")
    return found


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print(
            "[x] OPENAI_API_KEY is not set.\n"
            "    PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"\n"
            "    then re-run:  python examples/structured_smoke_test.py"
        )
        return 2

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("[x] langchain-openai is not installed.  pip install langchain-openai")
        return 2

    image_path = _resolve_image()
    if image_path is None or not image_path.exists():
        print("[x] No image found. Pass a path: python examples/structured_smoke_test.py img.png")
        return 2

    model = os.environ.get("OCRCONTEXT_MODEL", "gpt-4o-mini")
    print(f"[i] OCR target : {image_path}")
    print(f"[i] LLM        : {model} (via langchain-openai)\n")

    llm = ChatOpenAI(model=model, temperature=0)
    analyzer = Analyzer(
        llm=llm,
        config=AnalyzerConfig(lang="tr", auto_handwriting_fallback=False),
    )

    try:
        # 1) OCR + LLM refine in one call (refine=True). We keep the refined text so
        #    we can extract from it WITHOUT re-running OCR.
        result = analyzer.analyze(image_path, refine=True, mode=RefinementMode.CONSERVATIVE)
    except MissingDependencyError as exc:
        print(f"[x] {exc}")
        return 1
    except UnsupportedFileError as exc:
        print(f"[x] {exc}")
        return 2

    print("=" * 60)
    print("RAW OCR TEXT")
    print("=" * 60)
    print(result.raw_text or result.text)
    print("=" * 60)
    print("REFINED TEXT" + ("  (LLM applied)" if result.refined else "  (no change)"))
    print("=" * 60)
    print(result.text)

    # 2) Structured extraction from the refined text (no second OCR pass).
    work = analyzer.extract_text(result.text, schema=LiteraryWork, language="tr")

    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT (LiteraryWork)")
    print("=" * 60)
    print(work.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
