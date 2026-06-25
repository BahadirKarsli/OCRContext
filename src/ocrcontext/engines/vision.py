"""Google Cloud Vision handwriting engine.

Ported verbatim from ocr-service/vision_handwriting.py. Primary engine for
handwriting mode; TrOCR is the fallback. ``google-cloud-vision`` is imported
lazily (install the ``vision`` extra).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from ..exceptions import MissingDependencyError

_DIKW_LETTERS = frozenset("WKID")


def _language_hints(ocr_lang: str) -> list[str]:
    """Vision BCP-47 hints. Paddle uses 'latin' for Turkish; UI sends explicit codes."""
    code = (ocr_lang or "").strip().lower()
    if code in ("tr", "tur", "turkish", "latin", "auto", "unknown", ""):
        return ["tr", "en"]
    if code in ("en", "english"):
        return ["en"]
    return ["en", code]


@dataclass
class _WordBox:
    text: str
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float


_DIKW_MAP = {
    "wisdom": ("W", "Wisdom"),
    "knowledge": ("K", "Knowledge"),
    "information": ("I", "Information"),
    "data": ("D", "Data"),
}


def _vertices_box(vertices) -> tuple[float, float, float, float]:
    xs = [float(v.x) for v in vertices]
    ys = [float(v.y) for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _word_text(word) -> str:
    return "".join(s.text for s in word.symbols).strip()


def _collect_words(full_annotation) -> list[_WordBox]:
    words: list[_WordBox] = []
    if not full_annotation or not full_annotation.pages:
        return words
    for page in full_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = _word_text(word)
                    if not text:
                        continue
                    x0, y0, x1, y1 = _vertices_box(word.bounding_box.vertices)
                    words.append(
                        _WordBox(
                            text=text,
                            cx=(x0 + x1) / 2,
                            cy=(y0 + y1) / 2,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                        )
                    )
    return words


def _row_tolerance(words: list[_WordBox]) -> float:
    if not words:
        return 20.0
    heights = [w.y1 - w.y0 for w in words if w.y1 > w.y0]
    if not heights:
        return 20.0
    heights.sort()
    med = heights[len(heights) // 2]
    return max(12.0, min(35.0, med * 0.75))


def _cluster_rows(words: list[_WordBox], tol: float) -> list[list[_WordBox]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w.cy)
    rows: list[list[_WordBox]] = []
    for w in sorted_words:
        placed = False
        for row in rows:
            row_cy = sum(x.cy for x in row) / len(row)
            if abs(w.cy - row_cy) <= tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w.x0)
    rows.sort(key=lambda r: sum(w.cy for w in r) / len(r))
    return rows


def _is_dikw_letter_token(token: str) -> bool:
    """Single-letter DIKW side labels only (W, K, I, D) - not 'ne', 'to', etc."""
    t = re.sub(r"[^a-zA-Z]", "", token.strip())
    return len(t) == 1 and t.upper() in _DIKW_LETTERS


def _is_short_label(token: str) -> bool:
    return _is_dikw_letter_token(token)


def _normalize_dikw_word(token: str) -> Optional[tuple[str, str]]:
    key = re.sub(r"[^a-z]", "", token.lower())
    return _DIKW_MAP.get(key)


def _row_has_dikw_pattern(tokens: list[str]) -> bool:
    letters = [t for t in tokens if _is_dikw_letter_token(t)]
    longs = [
        t for t in tokens if not _is_dikw_letter_token(t) and len(re.sub(r"\W", "", t)) > 1
    ]
    return bool(letters) and bool(longs) and len(tokens) <= 8


def _format_row_tokens(tokens: list[str]) -> str:
    """Default: space-joined prose. DIKW letter+word merge only when pattern matches."""
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    if not _row_has_dikw_pattern(tokens):
        return " ".join(tokens)

    shorts = [t for t in tokens if _is_dikw_letter_token(t)]
    longs = [t for t in tokens if not _is_dikw_letter_token(t)]

    pairs: list[str] = []
    used_long: set[int] = set()

    for s in shorts:
        letter = re.sub(r"[^a-zA-Z]", "", s).upper()
        matched = False
        for i, lng in enumerate(longs):
            if i in used_long:
                continue
            mapped = _normalize_dikw_word(lng)
            if mapped and mapped[0] == letter:
                pairs.append(f"{mapped[1]} ({mapped[0]})")
                used_long.add(i)
                matched = True
                break
        if not matched and len(longs) == 1 and 0 not in used_long:
            pairs.append(f"{longs[0]} ({letter})")
            used_long.add(0)
        elif not matched:
            pairs.append(s)

    for i, lng in enumerate(longs):
        if i not in used_long:
            pairs.append(lng)

    return " · ".join(pairs) if len(pairs) > 1 else (pairs[0] if pairs else " ".join(tokens))


def _is_margin_number_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if re.fullmatch(r"[\d\s]+", s):
        return True
    if re.fullmatch(r"\d{2,4}", s):
        return True
    return False


def _is_pyramid_header(line: str) -> bool:
    low = line.lower()
    return "piramid" in low or "pyramid" in low or "dikw" in low


def _row_looks_like_dikw_pair(line: str) -> bool:
    tokens = re.split(r"\s+|[·]", line.replace("·", " "))
    tokens = [t for t in tokens if t]
    if len(tokens) < 2:
        return False
    has_letter = any(_is_dikw_letter_token(t) for t in tokens)
    has_long = any(len(re.sub(r"\W", "", t)) > 2 for t in tokens)
    return has_letter and has_long


def document_has_dikw_structure(lines: list[str]) -> bool:
    """True when text looks like a DIKW / pyramid diagram (not plain prose)."""
    pair_count = 0
    for line in lines:
        if _is_pyramid_header(line):
            return True
        if _row_looks_like_dikw_pair(line):
            pair_count += 1
            if pair_count >= 2:
                return True
    return False


def detect_dikw_structure(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return document_has_dikw_structure(lines)


def _dedupe_consecutive_tokens(line: str) -> str:
    tokens = line.split()
    if len(tokens) < 2:
        return line

    def norm(t: str) -> str:
        return re.sub(r"[^\w]", "", t.lower())

    out = [tokens[0]]
    for t in tokens[1:]:
        if norm(t) != norm(out[-1]):
            out.append(t)
    return " ".join(out)


def dedupe_prose_lines(lines: list[str]) -> list[str]:
    return [_dedupe_consecutive_tokens(ln) for ln in lines]


def _line_ends_complete(line: str) -> bool:
    s = line.rstrip()
    if not s:
        return True
    if s.endswith(("...", "…")):
        return True
    return s[-1] in ".!?;:"


def _line_starts_continuation(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return False
    return s[0].islower()


def _looks_like_signature_line(line: str) -> bool:
    words = [w for w in line.split() if w]
    if len(words) < 2 or len(words) > 5:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps >= 2 and not _line_starts_continuation(line)


def _capitalize_line_start(line: str) -> str:
    for i, ch in enumerate(line):
        if ch.isalpha():
            return line[:i] + ch.upper() + line[i + 1:]
    return line


def merge_wrapped_prose_lines(lines: list[str]) -> list[str]:
    """Join Vision line breaks that split one sentence/verse across rows."""
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    if len(cleaned) < 2:
        return [_capitalize_line_start(ln) for ln in cleaned]

    merged: list[str] = [cleaned[0]]
    for nxt in cleaned[1:]:
        prev = merged[-1]
        should_merge = False

        if _looks_like_signature_line(nxt) and _line_ends_complete(prev):
            should_merge = False
        elif (
            _line_starts_continuation(nxt)
            and not _line_ends_complete(prev)
            and len(prev.split()) <= 3
        ):
            should_merge = True
        elif len(prev.split()) <= 2 and not _line_ends_complete(prev):
            should_merge = True

        if should_merge:
            merged[-1] = f"{prev} {nxt}"
        else:
            merged.append(nxt)

    return [_capitalize_line_start(ln) for ln in merged]


def _format_dikw_hierarchy(header: str, pair_lines: list[str], side_notes: list[str]) -> str:
    entries: list[tuple[int, str]] = []
    order_key = {"W": 0, "K": 1, "I": 2, "D": 3}

    for line in pair_lines:
        tokens = [t for t in re.split(r"\s+|[·]", line.strip()) if t]
        shorts = [t.upper() for t in tokens if _is_dikw_letter_token(t)]
        longs = [t for t in tokens if not _is_dikw_letter_token(t)]
        letter = shorts[0] if shorts else ""
        label = longs[0] if longs else (shorts[0] if shorts else line)
        mapped = _normalize_dikw_word(label)
        if mapped:
            letter, label = mapped
        rank = order_key.get(letter, 99)
        if letter == "W":
            entries.append((rank, f"- {label} ({letter}) — en üst"))
        elif letter == "D":
            entries.append((rank, f"- {label} ({letter}) — taban"))
        elif letter:
            entries.append((rank, f"- {label} ({letter})"))
        else:
            entries.append((rank, f"- {line.strip()}"))

    entries.sort(key=lambda x: x[0])
    out = [header.rstrip(":") + ":"]
    out.extend(e[1] for e in entries)
    if side_notes:
        note_parts = [n.strip() for n in side_notes if n.strip()]
        if note_parts:
            out.append("Not: " + " · ".join(note_parts))
    return "\n".join(out)


def _restructure_document_lines(lines: list[str]) -> list[str]:
    """Detect Bilgi Piramidi / DIKW blocks and emit hierarchical list."""
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _is_pyramid_header(line):
            result.append(line)
            i += 1
            continue

        header = line
        i += 1
        pair_lines: list[str] = []
        side_notes: list[str] = []

        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                i += 1
                break
            low = nxt.lower()
            if _is_pyramid_header(nxt) and pair_lines:
                break
            if len(nxt) > 80 and not _row_looks_like_dikw_pair(nxt):
                break
            if low in ("value", "meaning") or low.startswith("value") or low.startswith("meaning"):
                if "value" in low:
                    side_notes.append("Value (↑) değer artar")
                if "meaning" in low:
                    side_notes.append("Meaning (↓) anlam artar")
                i += 1
                continue
            if (
                _row_looks_like_dikw_pair(nxt)
                or _is_dikw_letter_token(nxt)
                or _normalize_dikw_word(nxt)
            ):
                pair_lines.append(nxt)
                i += 1
                continue
            if len(pair_lines) >= 2:
                break
            pair_lines.append(nxt)
            i += 1

        if len(pair_lines) >= 2:
            result.append(_format_dikw_hierarchy(header, pair_lines, side_notes))
        else:
            result.append(header)
            result.extend(pair_lines)
            result.extend(side_notes)

    return result


def spatial_text_from_annotation(full_annotation) -> tuple[str, bool]:
    """Build reading-order text using word bounding boxes. Returns (text, has_dikw)."""
    words = _collect_words(full_annotation)
    if not words:
        return "", False

    tol = _row_tolerance(words)
    rows = _cluster_rows(words, tol)
    lines: list[str] = []
    for row in rows:
        tokens = [w.text for w in row]
        line = _format_row_tokens(tokens)
        if line and not _is_margin_number_line(line):
            lines.append(line)

    if not lines:
        return "", False

    has_dikw = document_has_dikw_structure(lines)
    if has_dikw:
        lines = _restructure_document_lines(lines)
    lines = dedupe_prose_lines(lines)
    if not has_dikw:
        lines = merge_wrapped_prose_lines(lines)
    return "\n".join(lines).strip(), has_dikw


class GoogleVisionHandwritingEngine:
    """Thin wrapper around the Vision API client. Call :meth:`load` once."""

    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        self._last_spatial = False
        self._last_has_dikw_structure = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_used_spatial(self) -> bool:
        return self._last_spatial

    @property
    def last_has_dikw_structure(self) -> bool:
        return self._last_has_dikw_structure

    def load(self) -> None:
        """Load Vision client from env-based credentials.

        Supported env keys:
        - GOOGLE_VISION_SERVICE_ACCOUNT_JSON
        - GOOGLE_APPLICATION_CREDENTIALS_JSON
        """
        try:
            from google.cloud import vision
            from google.oauth2 import service_account
        except ImportError as exc:  # pragma: no cover - exercised via install matrix
            raise MissingDependencyError("google-cloud-vision", "vision") from exc

        raw = (
            os.environ.get("GOOGLE_VISION_SERVICE_ACCOUNT_JSON")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            or ""
        ).strip()

        if not raw:
            self._enabled = False
            self._client = None
            return

        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            self._enabled = False
            self._client = None
            return

        creds = service_account.Credentials.from_service_account_info(info)
        self._client = vision.ImageAnnotatorClient(credentials=creds)
        self._enabled = True

    def extract_text_from_bytes(self, image_bytes: bytes, ocr_lang: str = "en") -> str:
        from google.cloud import vision

        self._last_spatial = False
        self._last_has_dikw_structure = False
        if not self._enabled or self._client is None:
            return ""

        image = vision.Image(content=image_bytes)
        hints = _language_hints(ocr_lang)
        context = vision.ImageContext(language_hints=hints)
        response = self._client.document_text_detection(image=image, image_context=context)

        if response.error and response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")

        annotation = response.full_text_annotation
        if annotation and annotation.pages:
            flat = (annotation.text or "").strip()
            has_dikw = detect_dikw_structure(flat) if flat else False
            self._last_has_dikw_structure = has_dikw

            # Always prefer Vision's flat reading-order text: it is the most faithful
            # transcription. Layout (incl. DIKW diagrams) is reconstructed downstream by
            # the LLM, which handles tightly-stacked notes far better than bbox clustering.
            if flat:
                lines = [ln.strip() for ln in flat.splitlines() if ln.strip()]
                lines = [ln for ln in lines if not _is_margin_number_line(ln)]
                lines = dedupe_prose_lines(lines)
                if not has_dikw:
                    lines = merge_wrapped_prose_lines(lines)
                text = "\n".join(lines)
                return text

            spatial, has_dikw_spatial = spatial_text_from_annotation(annotation)
            if spatial and len(spatial) >= 10:
                self._last_spatial = True
                self._last_has_dikw_structure = has_dikw_spatial
                return spatial

        if response.text_annotations:
            flat = (response.text_annotations[0].description or "").strip()
            self._last_has_dikw_structure = detect_dikw_structure(flat)
            return flat

        return ""


def run_vision_on_page(
    engine: GoogleVisionHandwritingEngine,
    img_path: str,
    ocr_lang: str = "en",
) -> tuple[str, float]:
    """OCR one page image with Google Vision. Returns (text, pseudo_confidence 0..1)."""
    if not engine.enabled:
        return "", 0.0

    with open(img_path, "rb") as f:
        text = engine.extract_text_from_bytes(f.read(), ocr_lang=ocr_lang)

    conf = min(1.0, len(text) / 250.0) if text else 0.0
    return text, conf
