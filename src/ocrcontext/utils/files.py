"""File / source loading and PDF rasterization helpers.

Replaces the Modal service's ``/tmp`` plumbing with cross-platform temp handling
(uses the OS temp dir via ``tempfile`` so it works on Windows too).
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Union

from ..exceptions import UnsupportedFileError

# What the public API accepts as a document source.
Source = Union[str, Path, bytes, bytearray, IO[bytes]]

IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"}


def is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def short_path(path: str) -> str:
    """Return the Windows 8.3 short path (ASCII) for an existing path.

    PaddlePaddle/OpenCV's C++ file readers fail on paths containing non-ASCII
    characters (a common case: a non-ASCII Windows username). The 8.3 short path
    aliases the same file with ASCII-only characters. No-op off Windows or when
    the path already resolves cleanly.
    """
    if sys.platform != "win32":
        return path
    try:
        import ctypes
        from ctypes import wintypes

        _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        _GetShortPathNameW.restype = wintypes.DWORD

        buf = ctypes.create_unicode_buffer(560)
        result = _GetShortPathNameW(str(path), buf, len(buf))
        if result:
            return buf.value
    except Exception:
        pass
    return path


def ascii_safe_dir(path: str) -> str:
    """ASCII-safe form of an existing directory (8.3 short path on Windows)."""
    if is_ascii(path):
        return path
    candidate = short_path(path)
    return candidate if is_ascii(candidate) else path


def _temp_dir() -> str:
    """ASCII-safe temp base. Override with ``OCRCONTEXT_TMPDIR``."""
    base = os.environ.get("OCRCONTEXT_TMPDIR") or tempfile.gettempdir()
    return ascii_safe_dir(base)


def new_temp_path(ext: str) -> str:
    """Absolute, ASCII-safe path to a unique temp file with the given extension."""
    ext = ext.lstrip(".")
    return os.path.join(_temp_dir(), f"ocrctx_{uuid.uuid4().hex}.{ext}")


def load_source(source: Source, *, filename: str | None = None) -> tuple[bytes, str]:
    """Normalize any accepted source into ``(file_bytes, extension)``.

    ``extension`` is a lowercase string without a leading dot (e.g. ``"pdf"``).
    """
    # Path / path-like string
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise UnsupportedFileError(f"File not found: {path}")
        ext = path.suffix.lower().lstrip(".")
        if not ext:
            raise UnsupportedFileError(f"Cannot determine file type from path: {path}")
        return path.read_bytes(), ext

    # Raw bytes — need a filename hint for the extension
    if isinstance(source, (bytes, bytearray)):
        ext = _ext_from_filename(filename)
        return bytes(source), ext

    # File-like object
    if hasattr(source, "read"):
        data = source.read()
        if not isinstance(data, (bytes, bytearray)):
            raise UnsupportedFileError("File-like source must be opened in binary mode.")
        name = filename or getattr(source, "name", None)
        ext = _ext_from_filename(name)
        return bytes(data), ext

    raise UnsupportedFileError(f"Unsupported source type: {type(source)!r}")


def _ext_from_filename(filename: str | None) -> str:
    if not filename or "." not in filename:
        raise UnsupportedFileError(
            "Could not infer file extension. Pass `filename=` when supplying raw bytes."
        )
    return filename.rsplit(".", 1)[-1].lower()


def is_pdf(ext: str) -> bool:
    return ext.lower() == "pdf"


def is_image(ext: str) -> bool:
    return ext.lower() in IMAGE_EXTS


@contextmanager
def temp_file(file_bytes: bytes, ext: str) -> Iterator[str]:
    """Write bytes to a temp file, yield its path, and clean up afterwards."""
    path = new_temp_path(ext)
    with open(path, "wb") as f:
        f.write(file_bytes)
    try:
        yield path
    finally:
        _safe_remove(path)


def rasterize_pdf(file_bytes: bytes, scale: float, prefix: str = "page") -> list[str]:
    """Render every PDF page to a PNG on disk; return the image paths.

    Caller owns cleanup (use :func:`cleanup_paths`). Ported from the PDF render
    loop in OCRService.process / HandwritingOCRService.process.
    """
    import fitz  # PyMuPDF — core dependency

    from PIL import Image

    image_paths: list[str] = []
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out_path = new_temp_path("png")
            img.save(out_path)
            image_paths.append(out_path)
    finally:
        pdf_document.close()
    return image_paths


def cleanup_paths(paths: list[str]) -> None:
    for p in paths:
        _safe_remove(p)


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
