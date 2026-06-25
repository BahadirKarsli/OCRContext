"""Image preprocessing, ported verbatim from ocr-service/modal_app.py.

OpenCV / NumPy are imported lazily so the base install does not require the
``paddle`` extra unless an image OCR path is actually taken.
"""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from ..utils.files import new_temp_path


def _require_cv2():
    """Import OpenCV, raising a friendly error if the image extras aren't installed."""
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised via install matrix
        raise MissingDependencyError("opencv-python-headless", "paddle") from exc


def _deskew_grayscale(img):
    """Correct slight rotation on notebook photos."""
    import numpy as np

    cv2 = _require_cv2()

    inv = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 200:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.4 or abs(angle) > 12:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _suppress_notebook_ruling(img):
    """Reduce horizontal ruled lines that cross handwritten strokes."""
    cv2 = _require_cv2()

    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return img

    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    line_w = max(25, w // 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    if cv2.countNonZero(horizontal) < 50:
        return img

    cleaned = cv2.inpaint(img, horizontal, 2, cv2.INPAINT_TELEA)
    return cleaned


def preprocess_image_for_ocr(img_path: str, handwriting: bool = False) -> str:
    """Contrast + denoise before OCR (helps faint / handwritten scans).

    Returns a path to a preprocessed PNG, or the original path if the image
    could not be read. Caller is responsible for cleaning up new files.
    """
    cv2 = _require_cv2()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return img_path

    if handwriting:
        h, w = img.shape[:2]
        # Notebook margin rulers (e.g. 23, 22, 21...) confuse Vision - trim narrow left strip.
        if w > h * 0.9 and w > 400:
            crop_x = min(int(w * 0.08), 100)
            img = img[:, crop_x:]
        # Deskew only; skip ruled-line inpainting - it can erase strokes that touch notebook lines.
        img = _deskew_grayscale(img)
        img = cv2.fastNlMeansDenoising(img, None, h=8, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img = clahe.apply(img)
        # Slight upscale for thin strokes
        img = cv2.resize(img, None, fx=1.15, fy=1.15, interpolation=cv2.INTER_CUBIC)
    else:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        img = clahe.apply(img)

    out_path = new_temp_path("png")
    cv2.imwrite(out_path, img)
    return out_path


def split_image_into_line_bands(img_path: str) -> list[tuple[int, str]]:
    """Create horizontal candidate bands for line recovery via projection profile.

    Returns ``[(y_offset, band_image_path), ...]``. Caller owns cleanup of the
    band image files. Ported from OCRService.process.split_image_into_line_bands.
    """
    import numpy as np

    cv2 = _require_cv2()

    bands: list[tuple[int, str]] = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return bands
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return bands

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )

    # Horizontal projection profile
    row_sum = np.sum(bw > 0, axis=1)
    threshold = max(8, int(0.02 * w))
    active_rows = row_sum > threshold

    segments: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(active_rows):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= 10:
                segments.append((start, i))
            start = None
    if start is not None and (len(active_rows) - start) >= 10:
        segments.append((start, len(active_rows)))

    if not segments:
        return bands

    # Merge close segments (same line broken by weak strokes)
    merged: list[list[int]] = []
    for s, e in segments:
        if not merged:
            merged.append([s, e])
        elif s - merged[-1][1] <= 8:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    for idx_band, (s, e) in enumerate(merged):
        pad = 10
        y0 = max(0, s - pad)
        y1 = min(h, e + pad)
        crop = img[y0:y1, :]
        if crop.shape[0] < 12:
            continue
        upscaled = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        band_path = new_temp_path("png")
        cv2.imwrite(band_path, upscaled)
        bands.append((y0, band_path))

    return bands
