"""Structured extraction into a Pydantic model.

    pip install 'ocrcontext[paddle]' langchain-openai
    export OPENAI_API_KEY=sk-...
"""

from langchain_openai import ChatOpenAI

from ocrcontext import Analyzer
from ocrcontext.schemas import Invoice

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), lang="tr")

invoice = analyzer.extract("invoice.pdf", schema=Invoice)

print(invoice.supplier_name, invoice.total_amount, invoice.currency)
for item in invoice.line_items:
    print(f"  - {item.description}: {item.quantity} x {item.unit_price} = {item.total}")


# --- Or define your own schema -------------------------------------------------
from pydantic import BaseModel, Field  # noqa: E402


class Receipt(BaseModel):
    merchant: str | None = Field(None, description="Store / merchant name")
    date: str | None = Field(None, description="Purchase date, YYYY-MM-DD")
    total: float | None = Field(None, description="Grand total")


receipt = analyzer.extract("receipt.jpg", schema=Receipt)
print(receipt)
