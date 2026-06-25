"""Microsoft TrOCR handwriting engine (line-by-line).

Ported verbatim from ocr-service/handwriting_ocr.py. Used as the fallback when
Google Vision is unavailable or returns nothing. ``transformers``/``torch`` are
imported lazily (install the ``trocr`` extra).
"""

from __future__ import annotations

from ..exceptions import MissingDependencyError
from ..utils.files import cleanup_paths, new_temp_path

TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
MIN_BAND_HEIGHT = 12
MAX_NEW_TOKENS = 128
TARGET_HEIGHT = 384
MAX_WIDTH = 1280


def split_image_into_line_bands(img_path: str) -> list[tuple[int, str]]:
    """Horizontal projection -> line crops for TrOCR (one line per image)."""
    import cv2

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

    import numpy as np

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

    merged: list[list[int]] = []
    for s, e in segments:
        if not merged:
            merged.append([s, e])
        elif s - merged[-1][1] <= 12:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    for idx_band, (s, e) in enumerate(merged):
        pad = 12
        y0 = max(0, s - pad)
        y1 = min(h, e + pad)
        crop = img[y0:y1, :]
        if crop.shape[0] < MIN_BAND_HEIGHT:
            continue
        upscaled = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        band_path = new_temp_path("png")
        cv2.imwrite(band_path, upscaled)
        bands.append((y0, band_path))

    return bands


def prepare_image_for_trocr(image):
    """Resize to TrOCR-friendly dimensions (avoids ViT tensor errors on tiny/huge crops)."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    if h < 1 or w < 1:
        return image

    if h < 32 or w < 32:
        scale = max(32 / w, 32 / h)
        w, h = max(32, int(w * scale)), max(32, int(h * scale))
        image = image.resize((w, h), Image.Resampling.LANCZOS)

    if h != TARGET_HEIGHT:
        new_w = max(32, int(w * (TARGET_HEIGHT / h)))
        image = image.resize((new_w, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        w, h = image.size

    if w > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, int(h * MAX_WIDTH / w)), Image.Resampling.LANCZOS)

    return image


class TrOCRHandwritingEngine:
    def __init__(self, model_id: str = TROCR_MODEL_ID) -> None:
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._device = None

    def load(self) -> None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:  # pragma: no cover - exercised via install matrix
            raise MissingDependencyError("transformers", "trocr") from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = TrOCRProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self._model.to(self._device)
        self._model.eval()

    def warmup_inference(self) -> None:
        if self._processor is None or self._model is None:
            return
        from PIL import Image

        dummy = Image.new("RGB", (384, 96), color=(255, 255, 255))
        try:
            _ = self.recognize_pil(dummy)
        except Exception:
            pass

    def recognize_line_image_path(self, path: str) -> str:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return self.recognize_pil(image)

    def recognize_pil(self, image) -> str:
        import torch

        if self._processor is None or self._model is None:
            raise RuntimeError("TrOCRHandwritingEngine.load() was not called")

        image = prepare_image_for_trocr(image)
        # Positional call matches HF docs; avoids kwarg edge cases in older processors.
        pixel_values = self._processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
            )

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return (text or "").strip()


def run_trocr_on_page(engine: TrOCRHandwritingEngine, img_path: str) -> tuple[str, float]:
    """OCR one page image with TrOCR line bands. Returns (text, pseudo_confidence 0..1)."""
    bands = split_image_into_line_bands(img_path)
    created: list[str] = []
    lines: list[str] = []

    try:
        if not bands:
            text = engine.recognize_line_image_path(img_path)
            if text:
                lines.append(text)
        else:
            for _, band_path in sorted(bands, key=lambda x: x[0]):
                created.append(band_path)
                line = engine.recognize_line_image_path(band_path)
                line = " ".join(line.split())
                if line:
                    lines.append(line)
    finally:
        cleanup_paths(created)

    full = "\n".join(lines).strip()
    conf = min(1.0, len(full) / 200.0) if full else 0.0
    return full, conf
