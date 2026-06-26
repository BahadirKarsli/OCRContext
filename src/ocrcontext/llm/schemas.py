"""Built-in extraction schemas.

The Invoice schema + extraction prompt are ported from
app/api/invoices/process/route.ts, including the quantity back-fill rule.
These double as ready-to-use schemas and as worked examples for users defining
their own Pydantic schemas.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    description: Optional[str] = Field(None, description="Product/Service name.")
    quantity: Optional[float] = Field(
        None,
        description=(
            "Numeric quantity. If missing, calculate it as total / unit_price. "
            "Default 1 only if neither is available."
        ),
    )
    unit: Optional[str] = Field(None, description="Unit of measure as written on the document (e.g. pcs, kg, hrs). Null if not present.")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    tax_rate: Optional[str] = Field(
        None, description="Tax percentage (e.g., 20, 10, 0) or pattern."
    )
    total: Optional[float] = Field(None, description="Total price for this line.")


class Invoice(BaseModel):
    supplier_name: Optional[str] = Field(None, description="Name of the vendor/supplier.")
    invoice_date: Optional[str] = Field(None, description="Format YYYY-MM-DD.")
    invoice_number: Optional[str] = Field(None, description="The invoice ID/number.")
    tax_id: Optional[str] = Field(None, description="Tax ID / VAT registration number.")
    tax_rate: Optional[str] = Field(
        None, description="Tax/VAT rate as written on the document (e.g. 'VAT 20%', 'GST 10%')."
    )
    currency: Optional[str] = Field(None, description="Currency code (TRY, USD, EUR, etc.).")
    total_amount: Optional[float] = Field(None, description="Final total amount (numeric).")
    line_items: list[LineItem] = Field(
        default_factory=list, description="Array of items/services."
    )

    @model_validator(mode="after")
    def _backfill_line_item_quantities(self) -> "Invoice":
        """Port of the route's quantity = total / unit_price correction."""
        for item in self.line_items:
            if item.unit_price is None or item.total is None:
                continue
            try:
                unit_price = float(item.unit_price)
                total = float(item.total)
            except (TypeError, ValueError):
                continue
            if unit_price > 0:
                calculated_qty = total / unit_price
                qty = item.quantity
                if (qty is None or qty == 1) and abs(calculated_qty - 1) > 0.01:
                    item.quantity = round(calculated_qty, 2)
        return self


INVOICE_EXTRACTION_PROMPT = """You are an expert invoice data extraction assistant.

CRITICAL RULES:
1. **OCR REPAIR**: The text may come from OCR and may have missing or garbled characters.
   Use context to infer the correct value — do not invent values that are not on the document.

2. **NUMBER PARSING**:
    - Be careful with comma (,) and dot (.) as thousand separators vs decimal points.
    - European format: '1.200,50' = 1200.50. US/UK format: '1,200.50' = 1200.50.
    - NEVER confuse a quantity (e.g., 2) with a unit price (e.g., 45.00).

3. **CURRENCY DETECTION**:
    - Look for symbols or codes on the document: $, USD, €, EUR, £, GBP, ₺, TRY, etc.
    - Use ONLY what is explicitly stated. Do not default to any currency.

4. **UNITS**: Copy the unit exactly as written on the document (pcs, kg, hrs, m², etc.).
   If no unit is shown, use null — never invent one.

Extract the following fields if present:
- 'supplier_name': Name of the vendor/supplier.
- 'invoice_date': Format YYYY-MM-DD.
- 'invoice_number': The invoice ID/number.
- 'tax_id': Tax ID or VAT registration number.
- 'tax_rate': Tax/VAT rate as written (e.g. 'VAT 20%', 'GST 10%').
- 'currency': ISO currency code (USD, EUR, GBP, TRY, etc.).
- 'total_amount': Final total amount (numeric).
- 'line_items': An array of items/services. Each item should have:
  - 'description': Product/Service name.
  - 'quantity': Numeric quantity. If missing, calculate it as total / unit_price. Default 1 only if neither is available.
  - 'unit': Unit of measure exactly as written on the document. Null if not present.
  - 'unit_price': Price per unit.
  - 'tax_rate': Tax percentage (e.g., 20, 10, 0) or null.
  - 'total': Total price for this line.

Return ONLY a valid JSON object. If a field is not found, use null."""


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class ReceiptItem(BaseModel):
    description: Optional[str] = Field(None, description="Item or product name.")
    quantity: Optional[float] = Field(None, description="Numeric quantity.")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    total: Optional[float] = Field(None, description="Total price for this line.")


class Receipt(BaseModel):
    store_name: Optional[str] = Field(None, description="Name of the store or merchant.")
    date: Optional[str] = Field(None, description="Transaction date, format YYYY-MM-DD.")
    time: Optional[str] = Field(None, description="Transaction time, format HH:MM.")
    items: list[ReceiptItem] = Field(default_factory=list, description="Purchased items.")
    subtotal: Optional[float] = Field(None, description="Amount before tax.")
    tax_amount: Optional[float] = Field(None, description="Tax amount charged.")
    total_amount: Optional[float] = Field(None, description="Final total paid.")
    payment_method: Optional[str] = Field(
        None, description="e.g. Cash, Credit Card, Debit Card."
    )
    currency: Optional[str] = Field(None, description="Currency code (TRY, USD, EUR, etc.).")


RECEIPT_EXTRACTION_PROMPT = """You are an expert receipt data extraction assistant.

CRITICAL RULES:
1. **NUMBER PARSING**: Watch commas vs dots. In Turkish/European receipts '1.200,50'
   means 1200.50. Parse every numeric field as a plain float.
2. **DATE/TIME**: Output date as YYYY-MM-DD and time as HH:MM when visible.
3. **ITEMS**: List every line item. If quantity is missing, default to 1.
4. **TOTALS**: subtotal is pre-tax; total_amount is what was actually paid.

Extract: store_name, date, time, items (description, quantity, unit_price, total),
subtotal, tax_amount, total_amount, payment_method, currency.

Return ONLY a valid JSON object. If a field is not found, use null."""


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class ContractParty(BaseModel):
    name: Optional[str] = Field(None, description="Full legal name of the party.")
    role: Optional[str] = Field(
        None, description="Role in the contract (e.g. Buyer, Seller, Employer, Employee)."
    )


class Contract(BaseModel):
    title: Optional[str] = Field(None, description="Official title or subject of the contract.")
    parties: list[ContractParty] = Field(
        default_factory=list, description="All parties involved."
    )
    effective_date: Optional[str] = Field(
        None, description="Date the contract comes into force, format YYYY-MM-DD."
    )
    expiration_date: Optional[str] = Field(
        None, description="Date the contract expires, format YYYY-MM-DD."
    )
    contract_value: Optional[float] = Field(
        None, description="Monetary value stated in the contract (numeric)."
    )
    currency: Optional[str] = Field(None, description="Currency code (TRY, USD, EUR, etc.).")
    governing_law: Optional[str] = Field(
        None, description="Jurisdiction or governing law (e.g. 'Turkish Law', 'New York')."
    )
    key_obligations: Optional[str] = Field(
        None, description="One-sentence summary of the main obligations."
    )


CONTRACT_EXTRACTION_PROMPT = """You are an expert legal document extraction assistant.

CRITICAL RULES:
1. **PARTIES**: Extract every named party and their role (Buyer/Seller, Lessor/Lessee, etc.).
2. **DATES**: Format all dates as YYYY-MM-DD. If only a year is visible, use YYYY-01-01.
3. **VALUE**: Extract the numeric contract value only; omit currency symbols from the number.
4. **OBLIGATIONS**: Write a single concise sentence summarising what the contract requires.

Extract: title, parties (name, role), effective_date, expiration_date, contract_value,
currency, governing_law, key_obligations.

Return ONLY a valid JSON object. If a field is not found, use null."""


# ---------------------------------------------------------------------------
# IdCard
# ---------------------------------------------------------------------------


class IdCard(BaseModel):
    document_type: Optional[str] = Field(
        None,
        description=(
            "Type of identity document: 'national_id', 'passport', "
            "'driver_license', or 'residence_permit'."
        ),
    )
    full_name: Optional[str] = Field(None, description="Full name as printed on the document.")
    date_of_birth: Optional[str] = Field(
        None, description="Date of birth, format YYYY-MM-DD."
    )
    gender: Optional[str] = Field(None, description="'M', 'F', or 'X'.")
    nationality: Optional[str] = Field(
        None, description="Nationality or issuing country (ISO 3166-1 alpha-3 if possible)."
    )
    document_number: Optional[str] = Field(
        None, description="ID number, passport number, or licence number."
    )
    issue_date: Optional[str] = Field(None, description="Issue date, format YYYY-MM-DD.")
    expiry_date: Optional[str] = Field(None, description="Expiry date, format YYYY-MM-DD.")
    issuing_authority: Optional[str] = Field(
        None, description="Authority or institution that issued the document."
    )
    address: Optional[str] = Field(
        None, description="Registered address if printed on the document."
    )


IDCARD_EXTRACTION_PROMPT = """You are an expert identity document extraction assistant.

CRITICAL RULES:
1. **DOCUMENT TYPE**: Classify as 'national_id', 'passport', 'driver_license', or
   'residence_permit'. Default to 'national_id' if unclear.
2. **DATES**: Format all dates as YYYY-MM-DD. Two-digit years: 00-30 → 20xx, 31-99 → 19xx.
3. **GENDER**: Normalise to 'M', 'F', or 'X' regardless of the source language.
4. **NATIONALITY**: Prefer ISO 3166-1 alpha-3 codes (TUR, USA, DEU, etc.).
5. **PRIVACY**: Extract exactly what is printed — do not infer or fill in missing data.

Extract: document_type, full_name, date_of_birth, gender, nationality,
document_number, issue_date, expiry_date, issuing_authority, address.

Return ONLY a valid JSON object. If a field is not found, use null."""


# ---------------------------------------------------------------------------
# MedicalReport
# ---------------------------------------------------------------------------


class Medication(BaseModel):
    name: Optional[str] = Field(None, description="Drug or medication name.")
    dosage: Optional[str] = Field(None, description="Dosage (e.g. '500 mg').")
    frequency: Optional[str] = Field(None, description="Frequency (e.g. '2x daily', 'at night').")
    duration: Optional[str] = Field(None, description="Duration of use (e.g. '7 days').")


class MedicalReport(BaseModel):
    patient_name: Optional[str] = Field(None, description="Full name of the patient.")
    patient_dob: Optional[str] = Field(
        None, description="Patient date of birth, format YYYY-MM-DD."
    )
    report_date: Optional[str] = Field(
        None, description="Date the report was issued, format YYYY-MM-DD."
    )
    doctor_name: Optional[str] = Field(None, description="Name of the attending physician.")
    institution: Optional[str] = Field(
        None, description="Hospital, clinic, or laboratory name."
    )
    diagnosis: Optional[str] = Field(
        None, description="Primary diagnosis or clinical finding."
    )
    icd_codes: list[str] = Field(
        default_factory=list, description="ICD-10 or ICD-11 codes if present."
    )
    medications: list[Medication] = Field(
        default_factory=list, description="Prescribed medications."
    )
    notes: Optional[str] = Field(
        None, description="Additional clinical notes or recommendations."
    )


MEDICAL_REPORT_EXTRACTION_PROMPT = """You are an expert medical document extraction assistant.

CRITICAL RULES:
1. **DATES**: Format all dates as YYYY-MM-DD.
2. **DIAGNOSIS**: Copy the exact wording from the document; do not interpret or summarise.
3. **ICD CODES**: Include only explicitly printed codes (e.g. 'J18.9'); do not infer.
4. **MEDICATIONS**: For each drug list name, dosage, frequency, and duration when visible.
5. **PRIVACY**: Extract only what is printed. Never add, infer, or correct clinical data.

Extract: patient_name, patient_dob, report_date, doctor_name, institution,
diagnosis, icd_codes, medications (name, dosage, frequency, duration), notes.

Return ONLY a valid JSON object. If a field is not found, use null."""
