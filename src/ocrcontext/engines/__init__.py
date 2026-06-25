"""OCR engines and the singleton model registry."""

from .base import OcrEngine, PageOcr
from .registry import EngineRegistry

__all__ = ["OcrEngine", "PageOcr", "EngineRegistry"]
