"""Raw OCR in 3 lines — no LLM, no API key required.

    pip install 'ocrcontext[paddle]'
"""

from ocrcontext import Analyzer

result = Analyzer().analyze("invoice.pdf")
print(result.text)

# `result` is a Pydantic model with extra metadata:
print("source:", result.text_source, "| pages:", result.pages, "| conf:", result.confidence)
