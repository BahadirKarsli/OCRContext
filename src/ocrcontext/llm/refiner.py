"""LLM-agnostic OCR refinement, ported from lib/ocr/refine.ts::refineOcrText.

Works with any LangChain ``BaseChatModel``. The fidelity pipeline is preserved:
mask literals -> prompt -> invoke -> unmask -> enforce literals -> strip markdown
-> drift/hallucination rejection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..types import RefinementMode
from .drift import refine_hallucinated_length, refinement_drifted
from .formatting import strip_markdown_formatting
from .literal_preserve import (
    enforce_original_literals,
    mask_protected_literals,
    unmask_protected_literals,
)
from .prompts import build_refinement_prompt, refine_temperature

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("ocrcontext.refine")


class Refiner:
    """Post-OCR refinement using an injected LangChain chat model."""

    def __init__(self, llm: "BaseChatModel", *, apply_temperature: bool = True) -> None:
        self._llm = llm
        # When True, bind the mode's recommended temperature to the model call.
        # Falls back gracefully for providers that reject the kwarg.
        self._apply_temperature = apply_temperature

    def refine(
        self,
        text: str,
        language: str = "auto",
        mode: RefinementMode = RefinementMode.CONSERVATIVE,
    ) -> str:
        """Refine OCR ``text``. Returns the original text unchanged on drift/empty."""
        mask = mask_protected_literals(text)
        system, user = build_refinement_prompt(mask.masked_text, language, mode)

        raw = self._invoke(system, user, refine_temperature(mode))
        refined = raw or mask.masked_text

        unmasked = unmask_protected_literals(refined, mask.literals)
        literal_safe = enforce_original_literals(text, unmasked)

        convert_bullets = mode in (RefinementMode.HANDWRITING_LAYOUT, RefinementMode.LAYOUT)
        cleaned = strip_markdown_formatting(literal_safe, convert_bullets=convert_bullets)

        # Handwritten notes/prose: trust word + layout fixes; reject only wholesale
        # hallucination (size bears little resemblance to source).
        if mode in (RefinementMode.HANDWRITING_PROSE, RefinementMode.HANDWRITING_LAYOUT):
            if not cleaned.strip():
                return text
            if refine_hallucinated_length(text, cleaned):
                logger.warning(
                    "Handwriting output length diverged too far; keeping original OCR text "
                    "(mode=%s)",
                    mode.value,
                )
                return text
            return cleaned

        if refinement_drifted(text, cleaned):
            logger.warning(
                "Output drifted too far from source; keeping original OCR text "
                "(mode=%s, original_lines=%d, refined_lines=%d)",
                mode.value,
                len(text.split("\n")),
                len(cleaned.split("\n")),
            )
            return text

        return cleaned

    def _invoke(self, system: str, user: str, temperature: float) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]

        if self._apply_temperature:
            try:
                bound = self._llm.bind(temperature=temperature)
                response = bound.invoke(messages)
                return _message_text(response)
            except Exception:
                # Provider may not accept a temperature kwarg — fall back to plain invoke.
                logger.debug("temperature bind failed; retrying without it", exc_info=True)

        response = self._llm.invoke(messages)
        return _message_text(response)


def _message_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)
