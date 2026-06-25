"""Literal / contact-data preservation, ported from lib/ocr/literal-preserve.ts.

Emails, URLs, IBANs and card numbers are masked to ``{{OCRLITn}}`` placeholders
before the LLM sees the text and restored verbatim afterwards, so the model
cannot "fix" identifiers (e.g. bahadrkrsl@... -> bahadirkarsli@...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Placeholders injected before LLM refine; restored verbatim after.
def _token_for(index: int) -> str:
    return f"{{{{OCRLIT{index}}}}}"


EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9](?:[a-zA-Z0-9._%+-]*[a-zA-Z0-9])?"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}"
)

_LITERAL_PATTERNS: list[re.Pattern[str]] = [
    EMAIL_PATTERN,
    re.compile(r"https?://[^\s<>\"'\])}+]+", re.IGNORECASE),
    re.compile(r"www\.[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}[^\s<>\"']*", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
]


@dataclass
class MaskResult:
    masked_text: str
    literals: list[str]


def preprocess_literal_text(text: str) -> str:
    """OCR often inserts spaces/newlines around @ - join before masking."""
    text = re.sub(
        r"([a-zA-Z0-9._%+-]+)\s*\n\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.-]*)", r"\1@\2", text
    )
    text = re.sub(
        r"([a-zA-Z0-9._%+-]+)\s+@\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)", r"\1@\2", text
    )
    return text


def extract_emails(text: str) -> list[str]:
    normalized = preprocess_literal_text(text)
    return [m.group(0) for m in EMAIL_PATTERN.finditer(normalized)]


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def _is_likely_same_email(candidate: str, original: str) -> bool:
    c = candidate.lower().split("@")
    o = original.lower().split("@")
    if len(c) != 2 or len(o) != 2:
        return False
    c_local, c_domain = c
    o_local, o_domain = o
    if not c_local or not o_local or c_domain != o_domain:
        return False
    if candidate == original:
        return True
    max_dist = max(2, int(len(o_local) * 0.35))
    return _levenshtein(c_local, o_local) <= max_dist


def enforce_original_literals(original_text: str, refined_text: str) -> str:
    """Force the OCR/original email spelling back if the model rewrote it."""
    originals = extract_emails(original_text)
    if not originals:
        return refined_text

    output = refined_text
    for orig in originals:
        parts = orig.split("@")
        if len(parts) != 2:
            continue
        domain = parts[1]
        if not domain:
            continue
        domain_escaped = re.escape(domain)
        domain_re = re.compile(
            r"[a-zA-Z0-9](?:[a-zA-Z0-9._%+-]*[a-zA-Z0-9])?@" + domain_escaped,
            re.IGNORECASE,
        )

        def _replace(match: re.Match[str], _orig: str = orig) -> str:
            text = match.group(0)
            if text == _orig:
                return _orig
            return _orig if _is_likely_same_email(text, _orig) else text

        output = domain_re.sub(_replace, output)

    return output


def _collect_non_overlapping_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), match.group(0)))

    # start asc; at equal start, the longer span wins (so it is kept, shorter dropped).
    spans.sort(key=lambda s: (s[0], -s[1]))

    merged: list[tuple[int, int, str]] = []
    for span in spans:
        if not merged or span[0] >= merged[-1][1]:
            merged.append(span)
    return merged


def mask_protected_literals(text: str) -> MaskResult:
    preprocessed = preprocess_literal_text(text)
    spans = _collect_non_overlapping_spans(preprocessed)
    literals: list[str] = []
    masked_text = ""
    cursor = 0
    for start, end, value in spans:
        masked_text += preprocessed[cursor:start]
        literals.append(value)
        masked_text += _token_for(len(literals) - 1)
        cursor = end
    masked_text += preprocessed[cursor:]
    return MaskResult(masked_text=masked_text, literals=literals)


def unmask_protected_literals(text: str, literals: list[str]) -> str:
    output = text
    for i, literal in enumerate(literals):
        placeholder = _token_for(i)
        if placeholder in output:
            output = output.replace(placeholder, literal)
            continue
        fuzzy = re.compile(r"\{\{\s*OCRLIT\s*" + str(i) + r"\s*\}\}", re.IGNORECASE)
        output = fuzzy.sub(literal, output)
    return output


LITERAL_PRESERVE_PROMPT = """
LITERAL / CONTACT DATA (CRITICAL):
- Tokens like {{OCRLIT0}}, {{OCRLIT1}}, ... are frozen placeholders for emails, URLs, IBANs, and similar identifiers.
- Copy every {{OCRLITn}} token EXACTLY — same spelling, same characters, same position in the sentence.
- NEVER "fix", complete, or guess emails/usernames (e.g. do NOT change bahadrkrsl@outlook.com to bahadirkarsli@outlook.com).
- NEVER invent @ symbols or domains. If a placeholder is present, output it unchanged.
- Apply OCR fixes only to normal words around these placeholders, not to the placeholders themselves.
"""
