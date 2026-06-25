<div align="center">

# OCR Context

**Turn any PDF or image into clean text — or a typed Pydantic model — in three lines.**

Decoupled, LLM-agnostic document OCR + structured extraction. No web server, no vendor lock-in.

[![PyPI version](https://img.shields.io/pypi/v/ocrcontext.svg?color=blue)](https://pypi.org/project/ocrcontext/)
[![Python versions](https://img.shields.io/pypi/pyversions/ocrcontext.svg)](https://pypi.org/project/ocrcontext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

</div>

```python
from ocrcontext import Analyzer

result = Analyzer().analyze("invoice.pdf")
print(result.text)
```

---

`ocrcontext` is the extraction core of a production document-analysis platform, lifted out of its
FastAPI/Next.js stack into a pure, pip-installable library. It does the hard parts — OCR engine
routing, fidelity-first LLM cleanup, and schema-based extraction — and gets out of your way.

## Why OCR Context

- **🚀 3-line DX** — instantiate, pass a file, get a result. That's the whole API surface.
  
- **🔌 LLM-agnostic** — inject *any* LangChain chat model (OpenAI, Anthropic, Ollama, local). Only
  `langchain-core` is required; you bring the provider.

- **🧠 Fidelity-first refinement** — fixes OCR errors without paraphrasing, translating, or inventing.
  Emails/URLs/IBANs are frozen so the model can't "correct" them, and drifting output is rejected.
  
- **📐 Structured extraction** — hand it a Pydantic schema, get a populated instance back via
  `with_structured_output`.
  
- **⚡ Resource-efficient** — heavy models (PaddleOCR, TrOCR) load lazily and are cached as
  process-wide singletons. They never reload per call.
  
- **🪶 Lightweight base install** — engines are opt-in extras. Core stays tiny.
  
- **🪟 Windows-hardened** — survives non-ASCII usernames and the PaddlePaddle 3.x CPU oneDNN issue
  out of the box.

## Contents

- [Install](#install)
- [Usage](#usage)
- [How it routes a document](#how-it-routes-a-document)
- [Refinement modes](#refinement-modes)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

## Install

```bash
pip install ocrcontext              # core only (PDF text layer + the full API surface)
pip install 'ocrcontext[paddle]'    # printed text + scanned PDFs (PaddleOCR)
pip install 'ocrcontext[trocr]'     # handwriting fallback (Microsoft TrOCR)
pip install 'ocrcontext[vision]'    # handwriting primary (Google Cloud Vision)
pip install 'ocrcontext[all]'       # everything
```

Pick an LLM provider for refinement / extraction:

```bash
pip install langchain-openai        # or langchain-anthropic, langchain-ollama, ...
```

## Usage

### Raw OCR (no LLM, no API key)

```python
from ocrcontext import Analyzer

result = Analyzer().analyze("scan.png")
print(result.text, result.confidence, result.pages, result.text_source)
```

### LLM-refined OCR

Refinement fixes OCR errors **without** paraphrasing, translating, or inventing text. Emails, URLs
and IBANs are masked before the model sees them and restored verbatim after; output that drifts too
far from the source is rejected in favour of the raw text.

Set your API key once in the environment — `langchain-openai` (and every other provider) picks it
up automatically:

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-...
```

Or pass it inline if you prefer (useful in notebooks):

```python
from langchain_openai import ChatOpenAI
ChatOpenAI(api_key="sk-...", model="gpt-4o")
```

Then:

```python
from langchain_openai import ChatOpenAI
from ocrcontext import Analyzer

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o"), lang="tr")
result = analyzer.analyze("handwritten_note.jpg", handwriting=True)

print(result.text)        # refined
print(result.raw_text)    # original OCR, kept alongside
```

### Structured extraction

```python
from langchain_openai import ChatOpenAI
from ocrcontext import Analyzer
from ocrcontext.schemas import Invoice

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))
invoice = analyzer.extract("invoice.pdf", schema=Invoice)   # -> Invoice instance

print(invoice.total_amount, invoice.currency)
for item in invoice.line_items:
    print(item.description, item.quantity, item.unit_price)
```

Define your own schema with plain Pydantic — field descriptions *are* the prompt:

```python
from pydantic import BaseModel, Field

class Receipt(BaseModel):
    merchant: str | None = Field(None, description="Store / merchant name")
    date: str | None = Field(None, description="Purchase date, YYYY-MM-DD")
    total: float | None = Field(None, description="Grand total")

receipt = analyzer.extract("receipt.jpg", schema=Receipt)
```

### Same code, local model (no API key)

```python
from langchain_ollama import ChatOllama
from ocrcontext import Analyzer

analyzer = Analyzer(llm=ChatOllama(model="llama3.1"))
print(analyzer.analyze("scan.png").text)
```

## How it routes a document

```
                ┌─────────────┐
   document ───▶│   Analyzer  │
                └──────┬──────┘
                       ▼
        ┌──────────────────────────────┐
        │ 1. Digital PDF? ──► text layer (PyMuPDF, no OCR)
        │                    └─► LLM refine AUTO-SKIPPED (exact text)
        │ 2. Image / scanned PDF ──► PaddleOCR (preprocess +
        │                            coverage-first + line-band fallback)
        │ 3. Handwriting ──► Google Vision → TrOCR fallback
        │ 4. (optional) LLM refine ──► fidelity-first, literal-safe
        │ 5. (optional) extract(schema) ──► typed Pydantic model
        └──────────────────────────────┘
```

Multi-page documents are joined with `--- Page N ---` separators. Handwriting kicks in
automatically when printed OCR returns too little text.

## Refinement modes

`RefinementMode`: `conservative` (scans), `layout` (digital PDFs), `handwriting_prose`,
`handwriting_layout`. The handwriting mode is auto-selected based on whether the text looks like a
DIKW/pyramid diagram. Modes and prompts are ported verbatim from the production pipeline and tuned
for fidelity.

## Configuration

```python
from ocrcontext import Analyzer, AnalyzerConfig

cfg = AnalyzerConfig(
    lang="tr",
    prefer_pdf_text_layer=True,
    auto_handwriting_fallback=True,
)
analyzer = Analyzer(llm=..., config=cfg)
```

## Development

```bash
git clone https://github.com/bahadirkarsli/ocrcontext
cd ocrcontext
pip install -e '.[dev]'
pytest            # runs without GPU/network — engines and LLM are faked
ruff check .
```

See the [`examples/`](examples/) folder for runnable smoke tests (image OCR, structured extraction,
and the PDF routing ladder).

## License

[MIT](LICENSE) © Bahadır Karslı
