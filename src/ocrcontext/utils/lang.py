"""Language code helpers.

``normalize_paddle_lang`` and the language map are ported verbatim from
``ocr-service/modal_app.py`` and ``lib/ocr/refine.ts`` respectively.
"""

from __future__ import annotations

from typing import Optional

# Mirrors languageMap in lib/ocr/refine.ts — UI code -> human-readable name used
# inside the refinement prompts.
LANGUAGE_MAP: dict[str, str] = {
    "tr": "Turkish",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


def language_full_name(lang: Optional[str]) -> Optional[str]:
    """Return the human-readable language name for a UI code, or the code itself."""
    if not lang:
        return None
    return LANGUAGE_MAP.get(lang, lang)


def normalize_paddle_lang(lang: Optional[str]) -> str:
    """Map UI / document language codes to PaddleOCR recognition models.

    Turkish is not a separate 'tr' pack in many PaddleOCR builds; 'latin' covers
    Latin-script languages with a wider charset than 'en' alone.

    Ported verbatim from ocr-service/modal_app.py::normalize_paddle_lang.
    """
    if not lang:
        return "en"
    code = str(lang).strip().lower()
    if code in ("auto", "unknown"):
        return "en"
    # Turkish / similar Latin-extended -> latin model (better s, g, i, o, u than en-only)
    if code in ("tr", "tur", "turkish"):
        return "latin"
    return {
        "en": "en",
        "english": "en",
        "de": "german",
        "german": "german",
        "fr": "french",
        "french": "french",
        "es": "es",
        "spanish": "es",
        "pt": "portuguese",
        "portuguese": "portuguese",
        "it": "it",
        "italian": "it",
    }.get(code, code if len(code) <= 20 else "en")


def candidate_langs(lang: Optional[str]) -> list[str]:
    """Ordered, de-duplicated PaddleOCR model candidates: primary -> latin -> en.

    Mirrors the candidate selection in OCRService.process.
    """
    primary = normalize_paddle_lang(lang)
    out: list[str] = []
    for code in (primary, "latin", "en"):
        if code not in out:
            out.append(code)
    return out
