<div align="center">

# OCR Context

**Turn any PDF or image into clean text — or a typed Pydantic model — in three lines.**

Decoupled, LLM-agnostic document OCR + structured extraction. No web server, no vendor lock-in.

[![CI](https://github.com/BahadirKarsli/OCRContext/actions/workflows/ci.yml/badge.svg)](https://github.com/BahadirKarsli/OCRContext/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ocrcontext.svg?color=blue)](https://pypi.org/project/ocrcontext/)
[![Python versions](https://img.shields.io/pypi/pyversions/ocrcontext.svg)](https://pypi.org/project/ocrcontext/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

</div>

**Try it in 30 seconds — no Python script needed:**

```bash
pip install 'ocrcontext[paddle,cli]'
ocrcontext extract invoice.pdf
ocrcontext extract receipt.jpg --output json
```

**Or use the Python API:**

```python
from ocrcontext import Analyzer

result = Analyzer().analyze("invoice.pdf")
print(result.text)
```

---

`ocrcontext` is the extraction core of a production document-analysis platform, lifted out of its FastAPI/Next.js stack into a pure, pip-installable library. It handles OCR engine routing, fidelity-first LLM cleanup, and schema-based structured extraction — and gets out of your way.

## Demo

**Structured invoice extraction from an image:**

<video src="https://github.com/user-attachments/assets/851c1aaf-9752-473f-9461-7b269b9ac42a" controls width="100%"></video>

**Digital PDF text extraction:**

<video src="https://github.com/user-attachments/assets/67580f43-ae84-40fc-aa7c-db0aeee50876" controls width="100%"></video>

## Contents

- [Demo](#demo)
- [Install](#install)
- [CLI](#cli)
- [Quick start (Python API)](#quick-start-python-api)
  - [GPU acceleration](#gpu-acceleration)
- [LangChain integration](#langchain-integration)
- [Built-in schemas](#built-in-schemas)
- [How it routes a document](#how-it-routes-a-document)
- [Refinement modes](#refinement-modes)
- [Configuration](#configuration)
- [Development](#development)
- [License](#license)

---

## Install

Engines are opt-in so your base install stays small:

| Command | What you get |
|---|---|
| `pip install ocrcontext` | Digital PDFs only (PyMuPDF text-layer — no OCR, no GPU, no API key) |
| `pip install 'ocrcontext[paddle]'` | + printed images & scanned PDFs (PaddleOCR, CPU/GPU) |
| `pip install 'ocrcontext[vision]'` | + handwriting (Google Cloud Vision) |
| `pip install 'ocrcontext[cli]'` | + terminal CLI (`ocrcontext extract`) |
| `pip install 'ocrcontext[all]'` | everything above |

Add an LLM provider for refinement and structured extraction:

```bash
pip install langchain-openai        # or langchain-anthropic, langchain-ollama, ...
```

> **Images and scanned PDFs require `[paddle]`.** Passing an image file to a bare `pip install ocrcontext` raises an `EngineError` with a clear install hint.

### Google Cloud Vision (`[vision]`)

1. Enable the **Cloud Vision API** in [Google Cloud Console](https://console.cloud.google.com/)
2. Create a service account key (JSON) under IAM & Admin → Service Accounts → Keys
3. Export the path:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"   # Linux/macOS
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\key.json" # PowerShell
```

---

## CLI

Install the `[cli]` extra to use `ocrcontext` straight from the terminal — no Python script needed.

```bash
pip install 'ocrcontext[paddle,cli]'
```

**Extract plain text:**

```bash
ocrcontext extract invoice.pdf
ocrcontext extract scan.png --output json
```

**Extract structured data with a built-in schema:**

```bash
ocrcontext extract invoice.pdf    --schema invoice
ocrcontext extract receipt.jpg    --schema receipt
ocrcontext extract contract.pdf   --schema contract
ocrcontext extract passport.jpg   --schema idcard
ocrcontext extract lab_report.pdf --schema medical
```

**Choose your LLM provider:**

```bash
ocrcontext extract invoice.pdf --schema invoice \
  --provider openai --model gpt-4o-mini

ocrcontext extract invoice.pdf --schema invoice \
  --provider anthropic --model claude-haiku-4-5-20251001

ocrcontext extract invoice.pdf --schema invoice \
  --provider ollama --model llama3.1
```

**All options:**

```
ocrcontext extract FILE [OPTIONS]

  --schema    -s   invoice | receipt | contract | idcard | medical
  --lang      -l   Language code (default: en)
  --handwriting    Force handwriting engine
  --refine         auto (default) | yes | no
  --output    -o   text (default) | json
  --provider  -p   openai | anthropic | ollama | google
  --model     -m   Model name (default: gpt-4o-mini)
```

---

## Quick start (Python API)

### Digital PDF

```python
from ocrcontext import Analyzer

result = Analyzer().analyze("document.pdf")
print(result.text)          # extracted text
print(result.pages)         # page count
print(result.text_source)   # "pdf_text_layer"
```

### Image / scanned PDF

```bash
pip install 'ocrcontext[paddle]'
```

```python
from ocrcontext import Analyzer

result = Analyzer().analyze("scan.png")
print(result.text, result.confidence)
```

### GPU acceleration

If you have a CUDA-capable GPU, swap the CPU PaddlePaddle build for the GPU one and pass `use_gpu=True`:

```bash
pip install 'ocrcontext[paddle]'
pip install paddlepaddle-gpu   # replaces the CPU build; pick the wheel that matches your CUDA version
```

```python
from ocrcontext import Analyzer

analyzer = Analyzer(use_gpu=True)
result = analyzer.analyze("scan.png")
print(result.text, result.confidence)
```

> PaddleOCR is typically 5–10× faster on GPU for large documents or batch workloads.
> CPU (`use_gpu=False`, the default) works out of the box with no extra steps.

### LLM-refined OCR

Refinement fixes character-level OCR errors without paraphrasing, translating, or inventing.
Emails, URLs, and IBANs are masked before the model sees them and restored verbatim after.
Output that drifts too far from the source is rejected in favour of the raw OCR text.

```bash
pip install 'ocrcontext[paddle]' langchain-openai
export OPENAI_API_KEY="sk-..."
```

```python
from langchain_openai import ChatOpenAI
from ocrcontext import Analyzer

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o-mini"), lang="en")
result = analyzer.analyze("scan.jpg")

print(result.text)       # refined
print(result.raw_text)   # original OCR output
print(result.refined)    # True
```

### Structured extraction

Hand the analyzer a Pydantic schema and get a populated instance back.

```python
from langchain_openai import ChatOpenAI
from ocrcontext import Analyzer
from ocrcontext.schemas import Invoice

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))
invoice = analyzer.extract("invoice.pdf", schema=Invoice)

print(invoice.supplier_name, invoice.total_amount, invoice.currency)
for item in invoice.line_items:
    print(item.description, item.quantity, item.unit_price)
```

Define your own schema — field descriptions are the prompt:

```python
from pydantic import BaseModel, Field

class ShippingLabel(BaseModel):
    sender: str | None = Field(None, description="Sender full name and address")
    recipient: str | None = Field(None, description="Recipient full name and address")
    tracking_number: str | None = Field(None, description="Carrier tracking number")

label = analyzer.extract("label.jpg", schema=ShippingLabel)
```

### No API key? Use a local model

```python
from langchain_ollama import ChatOllama
from ocrcontext import Analyzer

analyzer = Analyzer(llm=ChatOllama(model="llama3.1"))
result = analyzer.analyze("scan.png")
print(result.text)
```

---

## LangChain integration

`OCRContextLoader` is a drop-in LangChain `BaseLoader`. It slots into any LangChain pipeline — RAG, document Q&A, chain-of-thought — without glue code.

```python
from ocrcontext.loaders import OCRContextLoader

# Plain OCR
loader = OCRContextLoader("contract.pdf")
docs = loader.load()  # -> [Document(page_content="...", metadata={...})]

# With LLM refinement
from langchain_openai import ChatOpenAI

loader = OCRContextLoader(
    "scan.pdf",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    lang="en",
    refine="yes",
)
docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)
# {
#   "source": "scan.pdf",
#   "text_source": "ocr",
#   "pages": 3,
#   "confidence": 0.94,
#   "refined": True,
#   "raw_text": "..."
# }
```

**In a RAG pipeline:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ocrcontext.loaders import OCRContextLoader

docs = OCRContextLoader("annual_report.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
```

---

## Built-in schemas

Five ready-to-use Pydantic schemas with system prompts, importable from `ocrcontext.schemas`.
Pass them directly to `analyzer.extract()` or the CLI `--schema` flag.

### Invoice

```python
from ocrcontext.schemas import Invoice

invoice = analyzer.extract("invoice.pdf", schema=Invoice)
# invoice.supplier_name, .invoice_number, .invoice_date, .total_amount,
# .currency, .tax_id, .tax_rate, .line_items (list[LineItem])
```

### Receipt

```python
from ocrcontext.schemas import Receipt

receipt = analyzer.extract("receipt.jpg", schema=Receipt)
# receipt.store_name, .date, .time, .total_amount, .tax_amount,
# .subtotal, .payment_method, .currency, .items (list[ReceiptItem])
```

### Contract

```python
from ocrcontext.schemas import Contract

contract = analyzer.extract("agreement.pdf", schema=Contract)
# contract.title, .effective_date, .expiration_date, .contract_value,
# .currency, .governing_law, .key_obligations,
# .parties (list[ContractParty] with .name, .role)
```

### IdCard

Supports national_id, passport, driver_license, residence_permit.

```python
from ocrcontext.schemas import IdCard

card = analyzer.extract("passport.jpg", schema=IdCard)
# card.document_type, .full_name, .date_of_birth, .gender,
# .nationality, .document_number, .issue_date, .expiry_date,
# .issuing_authority, .address
```

### MedicalReport

```python
from ocrcontext.schemas import MedicalReport

report = analyzer.extract("lab_report.pdf", schema=MedicalReport)
# report.patient_name, .patient_dob, .report_date, .doctor_name,
# .institution, .diagnosis, .icd_codes (list[str]),
# .medications (list[Medication]), .notes
```

---

## How it routes a document

```
              ┌─────────────┐
 document ───▶│   Analyzer  │
              └──────┬──────┘
                     ▼
      ┌──────────────────────────────────────┐
      │ 1. Digital PDF?                       │
      │    └─▶ PyMuPDF text layer             │
      │        LLM refine auto-skipped        │
      │                                       │
      │ 2. Image / scanned PDF?               │
      │    └─▶ PaddleOCR                      │
      │        (preprocess → coverage-first   │
      │         → line-band fallback)         │
      │                                       │
      │ 3. Handwriting (explicit or auto)?    │
      │    └─▶ Google Cloud Vision            │
      │        → PaddleOCR if Vision empty    │
      │                                       │
      │ 4. (optional) LLM refine              │
      │    fidelity-first · literal-safe      │
      │                                       │
      │ 5. (optional) extract(schema)         │
      │    └─▶ typed Pydantic model           │
      └──────────────────────────────────────┘
```

Multi-page documents are joined with `--- Page N ---` separators.
Handwriting step 3 is explicit-only by default; set `auto_handwriting_fallback=True` to enable automatic retry.

---

## Refinement modes

| Mode | When it's used |
|---|---|
| `conservative` | Scanned images — minimal char-level correction only |
| `layout` | Digital PDFs — reconstruct clean structure |
| `handwriting_layout` | Handwritten notes / lists / diagrams |
| `handwriting_prose` | Handwritten poems / paragraphs / letters |

Modes are auto-selected based on the document type and text content. The handwriting mode choice is driven by whether the text looks like a DIKW/pyramid diagram. All prompts are ported verbatim from the production pipeline.

Override manually:

```python
from ocrcontext import Analyzer, RefinementMode

result = analyzer.analyze("scan.png", mode=RefinementMode.CONSERVATIVE)
```

---

## Configuration

```python
from ocrcontext import Analyzer, AnalyzerConfig

cfg = AnalyzerConfig(
    lang="tr",                        # default document language
    prefer_pdf_text_layer=True,       # skip OCR when a text layer exists
    auto_handwriting_fallback=False,  # keep PaddleOCR as sole engine (default); set True to enable Vision fallback
    refine_by_default=True,           # auto-refine whenever an LLM is configured
)
analyzer = Analyzer(llm=..., config=cfg, use_gpu=False)  # set use_gpu=True for CUDA-capable devices
```

---

## Development

```bash
git clone https://github.com/BahadirKarsli/OCRContext
cd OCRContext
pip install -e '.[dev]'
pytest            # runs without GPU or network — engines and LLM are faked
ruff check .
```

See [`examples/`](examples/) for runnable smoke tests (image OCR, structured extraction, PDF routing).

---

## License

[MIT](LICENSE) © Bahadır Karslı
