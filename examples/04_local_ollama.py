"""LLM-agnostic: the exact same code with a local Ollama model — no API key.

    pip install 'ocrcontext[paddle]' langchain-ollama
    ollama pull llama3.1
"""

from langchain_ollama import ChatOllama

from ocrcontext import Analyzer

# Swap ChatOpenAI -> ChatOllama; nothing else changes.
analyzer = Analyzer(llm=ChatOllama(model="llama3.1"))

result = analyzer.analyze("scan.png")
print(result.text)
