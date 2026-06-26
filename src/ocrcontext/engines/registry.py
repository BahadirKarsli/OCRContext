"""Singleton registry for heavy OCR engines.

PaddleOCR and TrOCR models are expensive to load. The registry guarantees each
engine (and therefore each model) is instantiated at most once per process,
satisfying the resource-efficiency requirement. Loading is lazy: an engine is
only created the first time it is requested.

Thread-safe so the same singleton is shared across threads (e.g. a web worker
pool that wraps this library).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .handwriting import HandwritingEngine
    from .paddle import PaddleEngine


class EngineRegistry:
    """Process-wide lazy cache of OCR engines.

    A default shared instance is exposed via :meth:`shared`, but callers may also
    create isolated registries (useful for tests).
    """

    _shared: "EngineRegistry | None" = None
    _shared_lock = threading.Lock()

    def __init__(self, *, use_gpu: bool = False) -> None:
        self._lock = threading.Lock()
        self._use_gpu = use_gpu
        self._paddle: "PaddleEngine | None" = None
        self._handwriting: "HandwritingEngine | None" = None

    @classmethod
    def shared(cls) -> "EngineRegistry":
        if cls._shared is None:
            with cls._shared_lock:
                if cls._shared is None:
                    cls._shared = cls()
        return cls._shared

    def paddle(self) -> "PaddleEngine":
        if self._paddle is None:
            with self._lock:
                if self._paddle is None:
                    from .paddle import PaddleEngine

                    self._paddle = PaddleEngine(use_gpu=self._use_gpu)
        return self._paddle

    def handwriting(self) -> "HandwritingEngine":
        if self._handwriting is None:
            with self._lock:
                if self._handwriting is None:
                    from .handwriting import HandwritingEngine

                    self._handwriting = HandwritingEngine()
        return self._handwriting

    def reset(self) -> None:
        """Drop cached engines (frees model memory). Mainly for tests."""
        with self._lock:
            self._paddle = None
            self._handwriting = None
