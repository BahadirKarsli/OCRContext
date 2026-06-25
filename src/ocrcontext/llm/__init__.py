"""LLM layer: refinement, structured extraction, and fidelity guards.

Only ``langchain-core`` is required here. Bring your own provider package
(``langchain-openai``, ``langchain-anthropic``, ``langchain-ollama``, ...).
"""

from .extractor import StructuredExtractor
from .refiner import Refiner

__all__ = ["Refiner", "StructuredExtractor"]
