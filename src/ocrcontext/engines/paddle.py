"""PaddleOCR engine for printed text and scanned documents.

Ported from ocr-service/modal_app.py::OCRService — the lazy per-language model
cache, multi-language *coverage-first* candidate selection, and the line-band
recovery fallback are preserved exactly. The Modal/GPU plumbing is removed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..exceptions import EngineError, MissingDependencyError
from ..preprocessing.image import preprocess_image_for_ocr, split_image_into_line_bands
from ..utils.files import ascii_safe_dir, cleanup_paths, is_ascii
from ..utils.lang import candidate_langs
from .base import OcrEngine, PageOcr


def _ensure_ascii_model_cache() -> None:
    """Point PaddleX/HuggingFace model caches at an ASCII-safe path on Windows.

    PaddlePaddle's C++ model loader cannot open files whose path contains
    non-ASCII characters (e.g. a non-ASCII Windows username), failing with an
    "attempting to parse an empty input" JSON error. Redirecting the cache to the
    8.3 short path of the home directory aliases the very same files via ASCII.
    Respects any cache env vars the user already set.
    """
    if sys.platform != "win32":
        return
    home = str(Path.home())
    if is_ascii(home):
        return
    safe_home = ascii_safe_dir(home)
    if not is_ascii(safe_home):
        return  # no ASCII short path available; nothing we can safely do
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(safe_home, ".paddlex"))
    os.environ.setdefault("HF_HOME", os.path.join(safe_home, ".cache", "huggingface"))


def _ensure_paddle_runtime_flags() -> None:
    """Disable oneDNN/MKLDNN on CPU.

    PaddlePaddle 3.x's new-IR (PIR) executor hits an unimplemented oneDNN op for
    some PP-OCR models on CPU ("ConvertPirAttribute2RuntimeAttribute not
    support"). Turning oneDNN off routes inference through the standard kernels.
    Set as an env FLAG so it applies regardless of constructor support, and
    respects any value the user already chose.
    """
    os.environ.setdefault("FLAGS_use_mkldnn", "0")


def _extract_from_result(result):
    """Normalize PaddleOCR / PaddleX result objects into (text, scores)."""
    extracted_text = ""
    extracted_scores: list[float] = []
    if not result:
        return extracted_text, extracted_scores
    first_page_result = result[0] if isinstance(result, list) and len(result) > 0 else result
    if hasattr(first_page_result, "keys") and "rec_texts" in first_page_result:
        texts = first_page_result.get("rec_texts", [])
        scores = first_page_result.get("rec_scores", [])
        for i, text in enumerate(texts):
            extracted_text += str(text) + "\n"
            if i < len(scores):
                extracted_scores.append(scores[i])
    elif isinstance(first_page_result, list):
        for line in first_page_result:
            try:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        extracted_text += str(line[1][0]) + "\n"
                        extracted_scores.append(line[1][1])
            except Exception:
                continue
    return extracted_text, extracted_scores


class PaddleEngine(OcrEngine):
    """Lazy, per-language singleton-style PaddleOCR wrapper.

    A single instance caches one PaddleOCR model per language code, so models are
    loaded into memory at most once (resource-efficiency requirement).
    """

    text_source = "ocr"

    def __init__(self) -> None:
        self._ocr_by_lang: dict[str, object] = {}

    def _get_ocr(self, paddle_lang: str):
        """Lazy-load + cache a PaddleOCR model for a language (ported loader)."""
        if paddle_lang in self._ocr_by_lang:
            return self._ocr_by_lang[paddle_lang]
        _ensure_ascii_model_cache()
        _ensure_paddle_runtime_flags()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - exercised via install matrix
            raise MissingDependencyError("paddleocr", "paddle") from exc

        import logging

        logging.getLogger("ppocr").setLevel(logging.ERROR)
        requested = paddle_lang
        ocr, errors = self._try_init(PaddleOCR, paddle_lang)
        if ocr is None and paddle_lang != "en":
            ocr, en_errors = self._try_init(PaddleOCR, "en")
            errors.extend(en_errors)
        if ocr is None:
            detail = "; ".join(errors[-3:]) if errors else "no profiles attempted"
            raise EngineError(
                f"PaddleOCR could not be initialized for lang={paddle_lang!r}. "
                f"Last errors: {detail}"
            )
        self._ocr_by_lang[requested] = ocr
        return ocr

    @staticmethod
    def _try_init(PaddleOCR, lang: str):
        """Try several constructor signatures, newest model first.

        Order: PP-OCRv6 → PP-OCRv5 → PP-OCRv4 → 3.x default → legacy 2.x.
        Returns (engine_or_None, [error_strings]). All exceptions are kept so a
        total failure can be diagnosed rather than silently swallowed.
        """
        # Shared 3.x flags: disable sub-models unneeded for plain OCR and oneDNN
        # (CPU PIR incompatibility on PaddlePaddle 3.x).
        base_3x = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        }
        profiles = [
            # PP-OCRv6: 50-language unified model, +5.1% rec over v5 (PaddleOCR 3.7+)
            {**base_3x, "ocr_version": "PP-OCRv6"},
            # PP-OCRv5: strong handwriting support (PaddleOCR 3.x)
            {**base_3x, "ocr_version": "PP-OCRv5"},
            # PP-OCRv4: stable 3.x baseline
            {**base_3x, "ocr_version": "PP-OCRv4"},
            # 3.x default — version determined by installed package, no pin
            base_3x,
            # Minimal 3.x (for builds that reject the sub-model flags)
            {"lang": lang, "enable_mkldnn": False},
            {"lang": lang},
            # Legacy 2.x (use_angle_cls; use_doc_* / show_log don't exist in 2.x)
            {"use_angle_cls": True, "lang": lang},
        ]
        errors: list[str] = []
        for kwargs in profiles:
            try:
                return PaddleOCR(**kwargs), errors
            except Exception as exc:  # noqa: BLE001 - we record and try the next profile
                errors.append(f"{type(exc).__name__}: {exc}")
        return None, errors

    @staticmethod
    def _run_ocr(ocr_engine, path: str):
        """Run recognition across PaddleOCR 2.x (.ocr) and 3.x (.predict)."""
        predict = getattr(ocr_engine, "predict", None)
        if callable(predict):
            try:
                return predict(path)
            except Exception:
                pass
        return ocr_engine.ocr(path)

    def recognize(
        self,
        img_path: str,
        *,
        lang: str = "en",
        min_lines: int = 3,
        handwriting: bool = False,
    ) -> PageOcr:
        langs = candidate_langs(lang)
        preprocessed_paths: list[str] = []

        try:
            ocr_img_path = preprocess_image_for_ocr(img_path, handwriting=handwriting)
            if ocr_img_path != img_path:
                preprocessed_paths.append(ocr_img_path)

            best_text = ""
            best_scores: list[float] = []
            best_line_count = 0

            for lang_code in langs:
                ocr_engine = self._get_ocr(lang_code)
                result = self._run_ocr(ocr_engine, ocr_img_path)
                candidate_text, candidate_scores = _extract_from_result(result)
                candidate_line_count = len(
                    [ln for ln in candidate_text.splitlines() if ln.strip()]
                )

                # Coverage-first selection (confidence ignored):
                #   1) more non-empty lines wins
                #   2) on a tie, longer non-whitespace text wins
                if candidate_line_count > best_line_count or (
                    candidate_line_count == best_line_count
                    and len(candidate_text.strip()) > len(best_text.strip())
                ):
                    best_line_count = candidate_line_count
                    best_text = candidate_text
                    best_scores = candidate_scores

            text = best_text
            scores = list(best_scores)

            # If full-image OCR still sees too few lines, run line-band fallback.
            best_line_count = len([ln for ln in best_text.splitlines() if ln.strip()])
            if best_line_count < min_lines:
                text, scores = self._line_band_fallback(
                    ocr_img_path, langs, base_text=best_text, base_scores=scores,
                    best_line_count=best_line_count,
                )

            return PageOcr(text=text.strip(), scores=scores)
        finally:
            cleanup_paths(preprocessed_paths)

    def _line_band_fallback(
        self,
        ocr_img_path: str,
        langs: list[str],
        *,
        base_text: str,
        base_scores: list[float],
        best_line_count: int,
    ) -> tuple[str, list[float]]:
        band_paths = split_image_into_line_bands(ocr_img_path)
        if not band_paths:
            return base_text, base_scores

        recovered_lines: list[str] = []
        recovered_scores: list[float] = []
        created_paths = [bp for _, bp in band_paths]
        try:
            for _, band_path in sorted(band_paths, key=lambda x: x[0]):
                band_best_text = ""
                band_best_len = 0
                band_best_scores: list[float] = []
                for lang_code in langs:
                    ocr_engine = self._get_ocr(lang_code)
                    result = self._run_ocr(ocr_engine, band_path)
                    txt, sc = _extract_from_result(result)
                    txt_len = len(txt.strip())
                    if txt_len > band_best_len:
                        band_best_len = txt_len
                        band_best_text = txt
                        band_best_scores = sc
                line = " ".join(
                    [p.strip() for p in band_best_text.splitlines() if p.strip()]
                ).strip()
                if line:
                    recovered_lines.append(line)
                    recovered_scores.extend(band_best_scores)
        finally:
            cleanup_paths(created_paths)

        if len(recovered_lines) > best_line_count:
            text = base_text.rstrip()
            if text and not text.endswith("\n"):
                text += "\n"
            text += "\n".join(recovered_lines) + "\n"
            return text, base_scores + recovered_scores

        return base_text, base_scores
