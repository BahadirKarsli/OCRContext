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
    unit: Optional[str] = Field(None, description="Unit type (Adet, Kg, Saat, etc.).")
    unit_price: Optional[float] = Field(None, description="Price per unit.")
    tax_rate: Optional[str] = Field(
        None, description="Tax percentage (e.g., 20, 10, 0) or pattern."
    )
    total: Optional[float] = Field(None, description="Total price for this line.")


class Invoice(BaseModel):
    supplier_name: Optional[str] = Field(None, description="Name of the vendor/supplier.")
    invoice_date: Optional[str] = Field(None, description="Format YYYY-MM-DD.")
    invoice_number: Optional[str] = Field(None, description="The invoice ID/number.")
    tax_id: Optional[str] = Field(None, description="Tax ID / VKN / TCKN.")
    tax_rate: Optional[str] = Field(
        None, description="e.g. 'KDV %20' when KDV is 20%."
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


# Verbatim system prompt from app/api/invoices/process/route.ts.
INVOICE_EXTRACTION_PROMPT = """You are an expert invoice data extraction assistant.

CRITICAL RULES:
1. **LANGUAGE REPAIR**:
    - The text may come from OCR and may have missing characters.
    - If language is 'tr' (Turkish), intelligently fix missing Turkish characters.

2. **NUMBER PARSING**:
    - Be extremely careful with comma (,) and dot (.).
    - In Turkish/European invoices, '1.200,50' means One Thousand Two Hundred and 50 cents.
    - NEVER confuse a quantity (e.g., 500) with a price (e.g. 5,00).

3. **CURRENCY DETECTION**:
    - Look for symbols: ₺, TL, TRY, USD, $, EUR, €.
    - Prioritize 'TRY' / 'TL' unless explicitly stated otherwise.

Extract the following fields if it exists:
- 'supplier_name': Name of the vendor/supplier.
- 'invoice_date': Format YYYY-MM-DD.
- 'invoice_number': The invoice ID/number.
- 'tax_id': Tax ID / VKN / TCKN.
- 'tax_rate': It can be like 'KDV' and for example if it is 'KDV' and it is %20, write it as 'KDV %20' in excel.
- 'currency': Currency code (TRY, USD, EUR, etc.).
- 'total_amount': Final total amount (numeric).
- 'line_items': An array of items/services. Each item should have:
  - 'description': Product/Service name.
  - 'quantity': Numeric quantity. If missing, calculate it as total / unit_price. Default 1 only if neither is available.
  - 'unit': Unit type (Adet, Kg, Saat, etc.).
  - 'unit_price': Price per unit.
  - 'tax_rate': Tax percentage (e.g., 20, 10, 0) or pattern.
  - 'total': Total price for this line.

Return ONLY a valid JSON object. If a field is not found, use null."""
