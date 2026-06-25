"""Refinement prompts, ported VERBATIM from lib/ocr/refine.ts.

These prompts were heavily tuned for fidelity; do not paraphrase them. Model
selection (gpt-4.1 vs gpt-4o) is intentionally dropped — the chat model is
injected by the caller. Only the per-mode temperature recommendation is kept.
"""

from __future__ import annotations

from ..types import RefinementMode
from ..utils.lang import language_full_name
from .literal_preserve import LITERAL_PRESERVE_PROMPT


def refine_temperature(mode: RefinementMode) -> float:
    """Deterministic for handwriting/conservative; light creativity for layout."""
    if mode == RefinementMode.CONSERVATIVE:
        return 0.0
    if mode == RefinementMode.HANDWRITING_PROSE:
        return 0.0
    if mode == RefinementMode.HANDWRITING_LAYOUT:
        return 0.0
    return 0.1


_LAYOUT = """
  LAYOUT MODE (for digital PDFs):
  - Reconstruct clean document structure in plain text.
  - Keep clear section separation with blank lines between paragraphs/sections.
  - Preserve list structure as plain text list items (no markdown markers unless present in source).
  - Preserve the original reading order and do not merge unrelated sections.
"""

_HANDWRITING_LAYOUT = """
  HANDWRITING LAYOUT MODE (for handwritten notes, lists, tables and diagrams of any topic):
  - FIDELITY FIRST: stay faithful to what is actually written. Your job is to fix OCR errors, NOT to
    improve the text. Do NOT paraphrase, summarize, complete unfinished sentences, smooth awkward
    phrasing, or add connecting words. A slightly awkward but faithful transcription is the goal.
  - Fix only clear OCR errors: missing diacritics, visually confused letters, and words the OCR split
    or joined. If a word is plausible as written, keep it exactly.
  - If a word or phrase is illegible, give your closest LITERAL character reading. NEVER replace it
    with a fluent but made-up sentence, and never add facts that are not in the source. It is better
    to leave a rough/partial phrase than to invent a clean one.
  - Remove lines that are ONLY margin ruler numbers (e.g. lone "23", "2213").
  - PRESERVE all of the source's content: keep every heading, label, list item, definition,
    and arrow ("→") line. Never drop a line. If you are unsure about a line, keep it.
  - PRESERVE THE ORIGINAL ORDER: keep lines and sentences in the exact order they appear in the source.
    Do NOT reorder, move, or merge sentences to make the text flow better. If the source order looks
    odd, leave it odd — do not "fix" it.
  - Keep numbered lists and definition paragraphs as they are; fix only OCR errors in them.
  - LAYOUT: only tidy spacing — keep the source's own line and section breaks, put each existing heading
    on its own line, leave ONE blank line between existing sections, and use "• " for bullets that are
    already bullets. Do NOT restructure content. Keep it PLAIN TEXT: no Markdown symbols (#, *, _,
    backticks, >) and no code fences.
  - Do NOT translate.
"""

_HANDWRITING_PROSE = """
  HANDWRITING PROSE MODE (poems, paragraphs, letters, notes — no sensitive data):
  - FIDELITY FIRST: fix OCR errors but stay faithful to what is written. Do NOT paraphrase, rewrite
    style, complete unfinished sentences, or add new words/sentences/ideas.
  - Fix OCR errors: missing diacritics, visually confused letters, and words the OCR split or joined
    (e.g. "düşünmedin sen" → "düşünmediysen"; split a wrongly-joined word). If a word is plausible as
    written, keep it exactly — do NOT swap it for a synonym or a "better" word.
  - Fix line breaks the OCR got wrong and keep real verse/paragraph breaks. Remove duplicate words and
    stray margin numbers caused by OCR.
  - Keep lines and sentences in their original order; do NOT reorder or move content.
  - If a word is illegible, give your closest LITERAL reading; never invent a fluent replacement.
  - Keep signatures, author names, and titles; only fix obvious typos in them.
  - Do NOT translate. Keep the original language.
"""

_CONSERVATIVE = """
  CONSERVATIVE MODE (for OCR images/scans):
  - Perform minimal, character-level OCR correction only.
  - Do NOT replace a valid-looking word with a different semantic word.
  - If uncertain, keep the original token exactly as-is.
  - Do NOT infer missing entities (names, places, brands, email local-parts) from context.
  - Preserve line order and keep output close to source line-by-line.
"""

_MODE_INSTRUCTIONS = {
    RefinementMode.LAYOUT: _LAYOUT,
    RefinementMode.HANDWRITING_LAYOUT: _HANDWRITING_LAYOUT,
    RefinementMode.HANDWRITING_PROSE: _HANDWRITING_PROSE,
    RefinementMode.CONSERVATIVE: _CONSERVATIVE,
}

_SYSTEM_HANDWRITING_LAYOUT = (
    "You are a world-class OCR post-processor for handwritten notes. "
    "Fix OCR errors and tidy the layout, but stay faithful to what is written: "
    "never paraphrase, complete, or invent text you cannot read. "
    "Never alter frozen {{OCRLITn}} placeholders. Output plain text only."
)
_SYSTEM_HANDWRITING_PROSE = (
    "You are a world-class OCR post-processor for handwritten prose and poetry. "
    "Fix misread words and broken line breaks, but stay faithful: never paraphrase, "
    "complete, or invent content, and never translate. "
    "Never alter frozen {{OCRLITn}} placeholders. Output plain text only."
)
_SYSTEM_DEFAULT = (
    "You are a world-class OCR post-processor. Fix OCR noise in normal prose only. "
    "Never alter frozen {{OCRLITn}} placeholders (emails, URLs, banking IDs). "
    "Never guess or complete email addresses or usernames. Output plain text only."
)


def build_refinement_prompt(
    masked_text: str, language: str, mode: RefinementMode
) -> tuple[str, str]:
    """Return ``(system, user)`` prompt strings for the given mode."""
    full_language = language_full_name(language) if language else None
    if full_language and full_language != "auto":
        language_prompt = (
            f"The text is in {full_language}. Preserve the original language and only fix "
            f"OCR errors using {full_language} spelling rules."
        )
    else:
        language_prompt = (
            "Preserve the original language of the text. Do not translate or change the language."
        )

    mode_instructions = _MODE_INSTRUCTIONS[mode]

    user = f"""
  You are an expert OCR post-processing AI. Your ONLY task is to reconstruct the original text from OCR output that contains scanning errors.
  Never add new information and never remove existing information.
  {mode_instructions}
  {LITERAL_PRESERVE_PROMPT}

  UNDERSTANDING OCR ERRORS:
  OCR engines make SYSTEMATIC errors — they don't understand language, they only recognize shapes.

  A) DIACRITIC STRIPPING (Turkish, French, German, Spanish, etc.)
  B) VISUALLY SIMILAR CHARACTER CONFUSION (0↔O, rn→m, ...)
  C) TRUNCATION & MISSING CHARACTERS
  D) WORD BOUNDARY ERRORS

  YOUR TASK:
  1. Fix OCR errors in regular words using sentence context.
  2. DO NOT translate. DO NOT change the language. DO NOT add or remove content.
  3. DO NOT add commentary. Output ONLY the corrected plain text.
  4. Output PLAIN TEXT only. Do NOT use Markdown syntax (#, *, _, backticks, >) and do NOT wrap the output in code fences.
  5. {language_prompt}

  Input Text:
  {masked_text}
  """

    if mode == RefinementMode.HANDWRITING_LAYOUT:
        system = _SYSTEM_HANDWRITING_LAYOUT
    elif mode == RefinementMode.HANDWRITING_PROSE:
        system = _SYSTEM_HANDWRITING_PROSE
    else:
        system = _SYSTEM_DEFAULT

    return system, user
