"""Digital PDF text-layer extraction (no GPU / no OCR).

Ported verbatim from ocr-service/modal_app.py. Used to skip OCR entirely when a
PDF already carries an accurate text layer.
"""

from __future__ import annotations

import re

# PowerPoint / Google Slides PDFs often expose internal image names in the text layer.
_PDF_IMAGE_ARTIFACT_RE = re.compile(
    r"^[\w.\-]{1,120}\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg)$",
    re.IGNORECASE,
)
_PDF_KNOWN_ARTIFACTS = frozenset(
    {
        "preencoded.png",
        "image.png",
        "image1.png",
        "image2.png",
    }
)


def is_pdf_text_artifact(line: str) -> bool:
    """Filter embedded image filenames leaked into PDF text extraction."""
    s = (line or "").strip()
    if not s:
        return False
    lower = s.lower()
    if lower in _PDF_KNOWN_ARTIFACTS:
        return True
    if " " in s or "/" in s or "\\" in s:
        return False
    if _PDF_IMAGE_ARTIFACT_RE.match(s):
        return True
    return False


def extract_pdf_text_preserve_layout(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from digital PDFs while preserving line order/layout."""
    import fitz

    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(pdf_document)
    pages_output: list[str] = []

    for page in pdf_document:
        # Use block-level extraction to preserve paragraph breaks and reading order.
        blocks = page.get_text("blocks")
        if not blocks:
            pages_output.append("")
            continue

        # block tuple: (x0, y0, x1, y1, text, block_no, block_type) - block_type 0=text, 1=image
        text_blocks = [
            b
            for b in blocks
            if len(b) >= 5
            and (len(b) < 7 or b[6] == 0)
            and isinstance(b[4], str)
            and b[4].strip()
        ]
        text_blocks.sort(key=lambda b: (round(float(b[1]), 1), round(float(b[0]), 1)))

        if not text_blocks:
            pages_output.append("")
            continue

        page_lines: list[str] = []
        prev_bottom = None

        for block in text_blocks:
            y0, y1 = float(block[1]), float(block[3])
            block_text = block[4].replace("\r\n", "\n").replace("\r", "\n").strip()
            if not block_text:
                continue

            # Insert paragraph gap if there is visible vertical space between blocks.
            if prev_bottom is not None and (y0 - prev_bottom) > 8:
                if page_lines and page_lines[-1] != "":
                    page_lines.append("")

            block_lines = [
                ln.rstrip()
                for ln in block_text.split("\n")
                if ln.strip() and not is_pdf_text_artifact(ln)
            ]
            page_lines.extend(block_lines)
            prev_bottom = y1

        # Collapse accidental triple+ gaps while keeping intentional paragraph breaks.
        compact_lines: list[str] = []
        empty_streak = 0
        for ln in page_lines:
            if ln.strip() == "":
                empty_streak += 1
                if empty_streak <= 1:
                    compact_lines.append("")
            else:
                empty_streak = 0
                compact_lines.append(ln)

        pages_output.append("\n".join(compact_lines).strip())

    pdf_document.close()

    full_text = ""
    for idx, page_text in enumerate(pages_output):
        if idx > 0:
            full_text += f"\n\n--- Page {idx + 1} ---\n\n"
        full_text += page_text

    return full_text, page_count


def has_sufficient_pdf_text(text: str) -> bool:
    """True when the PDF text layer is rich enough to use instead of OCR."""
    stripped = (text or "").strip()
    if len(stripped) < 80:
        return False

    alnum_count = sum(ch.isalnum() for ch in stripped)
    ratio = alnum_count / max(len(stripped), 1)
    return ratio >= 0.25
