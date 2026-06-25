"""OCR text-quality heuristics, ported from lib/ocr/ocr-quality.ts and
lib/ocr/handwriting-refine.ts / detect-dikw.ts.
"""

from __future__ import annotations

import re

from .types import RefinementMode

_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)  # unicode letters/digits
_WS_RE = re.compile(r"\s+")


def is_ocr_text_insufficient(text: str, page_count: int = 1) -> bool:
    """Heuristic: OCR returned too little usable text (common on handwriting)."""
    stripped = _WS_RE.sub(" ", text).strip()
    if not stripped:
        return True

    pages = max(1, page_count)
    min_chars = max(50, pages * 25)
    if len(stripped) < min_chars:
        return True

    alnum = len(_ALNUM_RE.findall(stripped))
    ratio = alnum / max(len(stripped), 1)
    return ratio < 0.2


# --- DIKW detection (ported from lib/ocr/detect-dikw.ts) -----------------------

_DIKW_LETTERS = {"W", "K", "I", "D"}


def _is_dikw_letter_token(token: str) -> bool:
    t = re.sub(r"[^a-zA-Z]", "", token)
    return len(t) == 1 and t.upper() in _DIKW_LETTERS


def _is_pyramid_header(line: str) -> bool:
    low = line.lower()
    return "piramid" in low or "pyramid" in low or "dikw" in low


def _row_looks_like_dikw_pair(line: str) -> bool:
    tokens = [t for t in re.split(r"[\s·]+", line) if t]
    if len(tokens) < 2:
        return False
    has_letter = any(_is_dikw_letter_token(t) for t in tokens)
    has_long = any(len(re.sub(r"\W", "", t)) > 2 for t in tokens)
    return has_letter and has_long


def detect_dikw_structure(text: str) -> bool:
    """True when OCR text looks like a DIKW / pyramid diagram (not plain prose)."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    pair_count = 0
    for line in lines:
        if _is_pyramid_header(line):
            return True
        if _row_looks_like_dikw_pair(line):
            pair_count += 1
            if pair_count >= 2:
                return True
    return False


def handwriting_refinement_mode(
    raw_text: str, has_dikw_from_ocr: bool | None = None
) -> RefinementMode:
    """Pick the handwriting refinement mode (layout for DIKW, else prose)."""
    has_dikw = has_dikw_from_ocr is True or detect_dikw_structure(raw_text)
    if has_dikw:
        return RefinementMode.HANDWRITING_LAYOUT
    return RefinementMode.HANDWRITING_PROSE
