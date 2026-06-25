from ocrcontext.llm.literal_preserve import (
    enforce_original_literals,
    extract_emails,
    mask_protected_literals,
    preprocess_literal_text,
    unmask_protected_literals,
)


def test_mask_and_unmask_roundtrip():
    text = "Contact me at bahadrkrsl@outlook.com or https://example.com/page now."
    mask = mask_protected_literals(text)
    assert "bahadrkrsl@outlook.com" not in mask.masked_text
    assert "{{OCRLIT0}}" in mask.masked_text
    assert "https://example.com/page" in mask.literals

    restored = unmask_protected_literals(mask.masked_text, mask.literals)
    assert "bahadrkrsl@outlook.com" in restored
    assert "https://example.com/page" in restored


def test_unmask_tolerates_fuzzy_placeholder():
    literals = ["a@b.com"]
    # Model put spaces inside the placeholder.
    out = unmask_protected_literals("write {{ OCRLIT 0 }} here", literals)
    assert out == "write a@b.com here"


def test_preprocess_joins_split_emails():
    assert preprocess_literal_text("user\n@ domain.com") == "user@domain.com"
    assert preprocess_literal_text("user @ domain.com") == "user@domain.com"


def test_enforce_original_email_spelling():
    original = "Email: bahadrkrsl@outlook.com"
    refined = "Email: bahadirkarsli@outlook.com"  # model "corrected" the local part
    fixed = enforce_original_literals(original, refined)
    assert fixed == "Email: bahadrkrsl@outlook.com"


def test_enforce_leaves_unrelated_emails():
    original = "a@x.com"
    refined = "totally-different@y.com"
    assert enforce_original_literals(original, refined) == refined


def test_extract_emails():
    assert extract_emails("x a@b.com y c@d.org") == ["a@b.com", "c@d.org"]


def test_iban_and_card_masked():
    text = "IBAN TR330006100519786457841326 card 1234 5678 9012 3456"
    mask = mask_protected_literals(text)
    assert "{{OCRLIT0}}" in mask.masked_text
    assert any("TR33" in lit for lit in mask.literals)
    assert any("1234" in lit for lit in mask.literals)
