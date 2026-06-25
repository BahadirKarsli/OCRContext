"""Hallucination / drift guards, ported from lib/ocr/refine-drift.ts.

If LLM refinement diverges too far from the source OCR text, the raw text is
kept instead — fidelity over fluency.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^\w]", re.UNICODE)


def _normalize_token(t: str) -> str:
    return _NON_ALNUM.sub("", t.lower())


def refine_hallucinated_length(original: str, refined: str) -> bool:
    """Light guard for generous prose mode: flags wholesale length divergence only."""
    o_words = len([w for w in original.split() if w])
    r_words = len([w for w in refined.split() if w])
    if o_words == 0:
        return False
    ratio = r_words / o_words
    return ratio < 0.5 or ratio > 1.8


def refinement_drifted(original: str, refined: str) -> bool:
    """Reject LLM refine output that diverges too far from source."""
    o_lines = len([ln for ln in original.split("\n") if ln.strip()])
    r_lines = len([ln for ln in refined.split("\n") if ln.strip()])
    if o_lines > 0 and abs(r_lines - o_lines) / o_lines > 0.35:
        return True

    o_words = [w for w in original.split() if w]
    r_words = [w for w in refined.split() if w]
    if len(o_words) > 0 and abs(len(r_words) - len(o_words)) / len(o_words) > 0.25:
        return True

    # Line-by-line: too many wholly different words (e.g. var -> vakit, Elinde -> içinde)
    o_line_arr = [ln for ln in original.split("\n") if ln.strip()]
    r_line_arr = [ln for ln in refined.split("\n") if ln.strip()]
    line_count = min(len(o_line_arr), len(r_line_arr))
    if line_count >= 2:
        changed_lines = 0
        for i in range(line_count):
            o_t = [t for t in (_normalize_token(x) for x in o_line_arr[i].split()) if t]
            r_t = [t for t in (_normalize_token(x) for x in r_line_arr[i].split()) if t]
            if len(o_t) != len(r_t):
                changed_lines += 1
                continue
            diff = sum(1 for j in range(len(o_t)) if o_t[j] != r_t[j])
            if diff > max(1, int(len(o_t) * 0.34)):
                changed_lines += 1
        if changed_lines / line_count > 0.4:
            return True

    return False
