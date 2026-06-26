"""Public re-export of built-in extraction schemas.

    from ocrcontext.schemas import Invoice, Receipt, Contract, IdCard, MedicalReport
"""

from .llm.schemas import (
    CONTRACT_EXTRACTION_PROMPT,
    IDCARD_EXTRACTION_PROMPT,
    INVOICE_EXTRACTION_PROMPT,
    MEDICAL_REPORT_EXTRACTION_PROMPT,
    RECEIPT_EXTRACTION_PROMPT,
    Contract,
    ContractParty,
    IdCard,
    Invoice,
    LineItem,
    MedicalReport,
    Medication,
    Receipt,
    ReceiptItem,
)

__all__ = [
    # Invoice
    "Invoice",
    "LineItem",
    "INVOICE_EXTRACTION_PROMPT",
    # Receipt
    "Receipt",
    "ReceiptItem",
    "RECEIPT_EXTRACTION_PROMPT",
    # Contract
    "Contract",
    "ContractParty",
    "CONTRACT_EXTRACTION_PROMPT",
    # IdCard
    "IdCard",
    "IDCARD_EXTRACTION_PROMPT",
    # MedicalReport
    "MedicalReport",
    "Medication",
    "MEDICAL_REPORT_EXTRACTION_PROMPT",
]
