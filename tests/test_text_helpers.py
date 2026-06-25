from ocrcontext.engines.pdf_text import has_sufficient_pdf_text, is_pdf_text_artifact
from ocrcontext.llm.drift import refine_hallucinated_length, refinement_drifted
from ocrcontext.llm.formatting import strip_markdown_formatting
from ocrcontext.quality import (
    detect_dikw_structure,
    handwriting_refinement_mode,
    is_ocr_text_insufficient,
)
from ocrcontext.types import RefinementMode
from ocrcontext.utils.lang import candidate_langs, normalize_paddle_lang


# --- lang ----------------------------------------------------------------------

def test_normalize_paddle_lang():
    assert normalize_paddle_lang("tr") == "latin"
    assert normalize_paddle_lang("english") == "en"
    assert normalize_paddle_lang("de") == "german"
    assert normalize_paddle_lang(None) == "en"
    assert normalize_paddle_lang("auto") == "en"


def test_candidate_langs_dedup_and_order():
    assert candidate_langs("tr") == ["latin", "en"]
    assert candidate_langs("en") == ["en", "latin"]
    assert candidate_langs("de") == ["german", "latin", "en"]


# --- drift ---------------------------------------------------------------------

def test_refinement_not_drifted_when_minor_fixes():
    original = "Helo world\nthis is a tset\nof ocr text here"
    refined = "Hello world\nthis is a test\nof ocr text here"
    assert refinement_drifted(original, refined) is False


def test_refinement_drifted_when_lines_explode():
    original = "one line only here friend"
    refined = "a\nb\nc\nd\ne\nf\ng"
    assert refinement_drifted(original, refined) is True


def test_hallucinated_length():
    assert refine_hallucinated_length("a b c d", "a b c d e f g h i j") is True
    assert refine_hallucinated_length("a b c d", "a b c d") is False


# --- formatting ----------------------------------------------------------------

def test_strip_markdown():
    md = "# Title\n\n**bold** and `code`\n> quote\n```\nfenced\n```"
    out = strip_markdown_formatting(md)
    assert "#" not in out
    assert "**" not in out
    assert "`" not in out
    assert "Title" in out and "bold" in out and "quote" in out


def test_convert_bullets():
    out = strip_markdown_formatting("- item one\n- item two", convert_bullets=True)
    assert "• item one" in out


# --- quality / dikw ------------------------------------------------------------

def test_is_ocr_text_insufficient():
    assert is_ocr_text_insufficient("") is True
    assert is_ocr_text_insufficient("ab") is True
    assert is_ocr_text_insufficient("x" * 200, page_count=1) is False  # enough chars


def test_detect_dikw_and_mode():
    pyramid = "Bilgi Piramidi\nW Wisdom\nK Knowledge\nI Information\nD Data"
    assert detect_dikw_structure(pyramid) is True
    assert handwriting_refinement_mode(pyramid) == RefinementMode.HANDWRITING_LAYOUT

    prose = "this is just a normal\nhandwritten paragraph of text"
    assert detect_dikw_structure(prose) is False
    assert handwriting_refinement_mode(prose) == RefinementMode.HANDWRITING_PROSE


# --- pdf text ------------------------------------------------------------------

def test_pdf_artifact_filter():
    assert is_pdf_text_artifact("image1.png") is True
    assert is_pdf_text_artifact("logo.svg") is True
    assert is_pdf_text_artifact("Real sentence here.png too") is False
    assert is_pdf_text_artifact("Hello world") is False


def test_has_sufficient_pdf_text():
    assert has_sufficient_pdf_text("short") is False
    assert has_sufficient_pdf_text("A real paragraph of document text. " * 5) is True
