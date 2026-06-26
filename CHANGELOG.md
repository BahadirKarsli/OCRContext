# Changelog

All notable changes to **ocrcontext** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-06-26

### Added
- **`OCRContextLoader`** — LangChain `BaseLoader` integration. Drop-in loader for
  any LangChain pipeline: `OCRContextLoader("file.pdf").load()` returns a
  `Document` with OCR text and metadata (`source`, `text_source`, `pages`,
  `confidence`, `refined`).
- **Built-in extraction schemas** — four new ready-to-use Pydantic schemas with
  system prompts, importable from `ocrcontext.schemas`:
  - `Receipt` / `ReceiptItem` — store name, date, items, subtotal, tax, total,
    payment method.
  - `Contract` / `ContractParty` — parties, effective/expiry dates, value,
    governing law, key obligations.
  - `IdCard` — national_id / passport / driver_license / residence_permit with
    ICD-standard date normalisation and ISO 3166-1 nationality codes.
  - `MedicalReport` / `Medication` — diagnosis, ICD codes, prescriptions, notes.
- **CLI** (`ocrcontext extract`) — terminal-first developer experience via the
  new `[cli]` extra (`pip install "ocrcontext[cli]"`):
  - `ocrcontext extract invoice.pdf` — plain OCR to stdout.
  - `ocrcontext extract scan.pdf --schema receipt --output json` — structured
    extraction as JSON.
  - `--provider openai|anthropic|ollama|google --model <name>` — bring-your-own
    LLM provider.
  - `--handwriting`, `--lang`, `--refine auto|yes|no` flags.

## [0.1.0] - 2026-06-25

Initial release — the document extraction core, decoupled from its web stack
into a standalone, LLM-agnostic library.

### Added
- **`Analyzer` facade** — 3-line developer experience:
  `Analyzer().analyze("file.pdf").text`.
- **Routing ladder** (`pipeline.py`):
  - Digital PDFs → PyMuPDF text-layer extraction (no OCR); LLM refine is
    auto-skipped so exact text/identifiers are never altered.
  - Images / scanned PDFs → PaddleOCR with image preprocessing, multi-language
    *coverage-first* candidate selection, and a line-band recovery fallback.
  - Handwriting (explicit or auto on insufficient text) → Google Vision primary,
    Microsoft TrOCR fallback.
  - Multi-page documents joined with `--- Page N ---` separators.
- **LLM-agnostic LLM layer** — works with any LangChain `BaseChatModel`
  (`langchain-openai`, `langchain-anthropic`, `langchain-ollama`, ...). Only
  `langchain-core` is required at the core.
  - `Refiner` — fidelity-first OCR refinement (4 modes) with literal/contact
    preservation (`{{OCRLITn}}` masking) and drift/hallucination rejection.
  - `StructuredExtractor` + `Analyzer.extract()` / `Analyzer.extract_text()` —
    structured extraction into any Pydantic schema via `with_structured_output`.
  - Built-in `Invoice` / `LineItem` schemas and prompt.
- **Resource efficiency** — `EngineRegistry` singleton caches PaddleOCR/TrOCR
  engines (and per-language models) so they load at most once per process.
- **Windows robustness** — model cache and temp files are routed through ASCII
  8.3 short paths to survive non-ASCII usernames; oneDNN is disabled on CPU to
  avoid the PaddlePaddle 3.x PIR/oneDNN `NotImplementedError`.
- **Packaging** — optional extras `[paddle]`, `[trocr]`, `[vision]`, `[all]`;
  PEP 561 typed (`py.typed`); examples and a GPU/network-free test suite.

[Unreleased]: https://github.com/bahadirkarsli/ocrcontext/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/bahadirkarsli/ocrcontext/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bahadirkarsli/ocrcontext/releases/tag/v0.1.0
