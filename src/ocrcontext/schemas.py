"""Public re-export of built-in extraction schemas.

    from ocrcontext.schemas import Invoice, LineItem
"""

from .llm.schemas import INVOICE_EXTRACTION_PROMPT, Invoice, LineItem

__all__ = ["Invoice", "LineItem", "INVOICE_EXTRACTION_PROMPT"]
