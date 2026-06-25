"""LLM-refined OCR with OpenAI.

    pip install 'ocrcontext[paddle]' langchain-openai
    export OPENAI_API_KEY=sk-...
"""

from langchain_openai import ChatOpenAI

from ocrcontext import Analyzer

analyzer = Analyzer(llm=ChatOpenAI(model="gpt-4o"), lang="tr")

# refine=None (default) auto-refines OCR output (but never an exact PDF text layer).
result = analyzer.analyze("handwritten_note.jpg", handwriting=True)

print("Refined:", result.refined)
print(result.text)
if result.raw_text:
    print("\n--- raw OCR (before refine) ---\n", result.raw_text)
