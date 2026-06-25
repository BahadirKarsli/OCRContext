"""End-to-end smoke test: raw OCR on a single image — no LLM, no schemas.

This exercises the *pure* PaddleOCR path:
    load image -> preprocess -> candidate-language OCR -> coverage-first text.

Usage
-----
    python examples/image_smoke_test.py                 # auto-find a sample image
    python examples/image_smoke_test.py path/to/img.png # explicit path

Setup
-----
    pip install -e '.[paddle]'      # installs PaddleOCR + OpenCV
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from ocrcontext import Analyzer, AnalyzerConfig
from ocrcontext.exceptions import MissingDependencyError, UnsupportedFileError

# Common sample names / extensions to look for when no path is given.
_SAMPLE_NAMES = ["sample", "test", "image", "ocr", "smoke"]
_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def _find_sample_image() -> Path | None:
    """Look for an image next to this script and in the project root."""
    here = Path(__file__).resolve().parent
    search_dirs = [here, here.parent]  # examples/ then ocrcontext_lib/
    # 1) Prefer files whose name hints they're samples.
    for d in search_dirs:
        for stem in _SAMPLE_NAMES:
            for ext in _IMAGE_EXTS:
                candidate = d / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
    # 2) Otherwise, the first image we can find in those dirs.
    for d in search_dirs:
        for ext in _IMAGE_EXTS:
            matches = sorted(d.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


def main() -> int:
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1]).expanduser()
    else:
        found = _find_sample_image()
        if found is None:
            print(
                "No image given and none found automatically.\n"
                "Drop an image (e.g. sample.png) into the examples/ folder, or run:\n"
                "    python examples/image_smoke_test.py path/to/your/image.png"
            )
            return 2
        image_path = found
        print(f"[i] No path given — using discovered image: {image_path.name}")

    if not image_path.exists():
        print(f"[x] File not found: {image_path}")
        return 2

    print(f"[i] OCR target : {image_path}")
    print("[i] Engine     : PaddleOCR (raw OCR, no LLM)\n")

    # Pure PaddleOCR: disable the handwriting fallback so a sparse image doesn't try
    # to load the Vision/TrOCR extras during this smoke test.
    analyzer = Analyzer(config=AnalyzerConfig(lang="en", auto_handwriting_fallback=False))

    try:
        t0 = time.perf_counter()
        result = analyzer.analyze(image_path)
        elapsed = time.perf_counter() - t0
    except MissingDependencyError as exc:
        print(f"[x] {exc}")
        return 1
    except UnsupportedFileError as exc:
        print(f"[x] {exc}")
        return 2

    print("=" * 60)
    print("EXTRACTED TEXT")
    print("=" * 60)
    print(result.text if result.text else "(no text detected)")
    print("=" * 60)
    print(
        f"source={result.text_source}  pages={result.pages}  "
        f"confidence={result.confidence}  chars={len(result.text)}  "
        f"time={elapsed:.2f}s"
    )

    # Show the singleton in action: a second call reuses the loaded model (fast).
    t0 = time.perf_counter()
    analyzer.analyze(image_path)
    print(f"[i] 2nd run (warm model): {time.perf_counter() - t0:.2f}s")

    return 0 if result.text.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
