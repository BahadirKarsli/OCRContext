"""Plain-text formatting, ported from lib/ocr/plain-text-format.ts.

Strip Markdown syntax so the model's structured output stays clean plain text
while keeping layout (blank lines, headings, bullets).
"""

from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_BLOCKQUOTE_RE = re.compile(r"^(\s{0,3})>\s?")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")
_CODE_FENCE_RE = re.compile(r"```[^\n`]*\n?")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_UNDERSCORE_BOLD_RE = re.compile(r"__([^_\n]+)__")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANKS_RE = re.compile(r"\n{3,}")


def strip_markdown_formatting(text: str, *, convert_bullets: bool = False) -> str:
    lines = []
    for line in text.split("\n"):
        line = _HEADING_RE.sub("", line)
        line = _BLOCKQUOTE_RE.sub(r"\1", line)
        if convert_bullets:
            line = _BULLET_RE.sub(r"\1• ", line)
        lines.append(line)

    result = "\n".join(lines)
    result = _CODE_FENCE_RE.sub("", result)
    result = _BOLD_RE.sub(r"\1", result)
    result = _UNDERSCORE_BOLD_RE.sub(r"\1", result)
    result = _INLINE_CODE_RE.sub(r"\1", result)
    result = _TRAILING_WS_RE.sub("", result)
    result = _EXTRA_BLANKS_RE.sub("\n\n", result)

    return result.strip()
