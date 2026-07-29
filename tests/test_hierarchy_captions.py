"""Unit tests for caption cleaning.

The four `REAL_*` cases are taken verbatim from
`~/pdfdrill-library/2209.00445/model.docmodel.json` — they are the actual
captions the pipeline must survive, not invented ones.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.captions import (_fallback_clean, clean_caption,
                                             lost_macros)

REAL_ALG = "\\ALG\\ Application"
REAL_EMPH = "\\emph{Siblings} score"
REAL_REMOVEP = "Testing the effect of the $removeP$ parameter"
REAL_TAU = "Testing the effect of the $\\tau$ function"


# --------------------------------------------------------------------------
# Plain input must survive untouched
# --------------------------------------------------------------------------

def test_plain_text_is_unchanged():
    assert clean_caption("Introduction") == "Introduction"
    assert clean_caption("The Conceptualization Algorithm") == \
        "The Conceptualization Algorithm"


def test_whitespace_is_collapsed():
    assert clean_caption("  Empirical   Evaluation \n") == "Empirical Evaluation"


def test_empty_input_returns_empty_string():
    assert clean_caption("") == ""
    assert clean_caption("   ") == ""


# --------------------------------------------------------------------------
# The real captions
# --------------------------------------------------------------------------

def test_emph_keeps_its_argument():
    assert clean_caption(REAL_EMPH) == "Siblings score"


def test_inline_math_identifier_is_kept():
    """'the $removeP$ parameter' must not become 'the parameter' — the
    identifier is the subject of the section."""
    out = clean_caption(REAL_REMOVEP)
    assert "removeP" in out
    assert "$" not in out


def test_greek_symbol_is_not_deleted():
    """Deleting $\\tau$ would turn 'the tau function' into 'the function'."""
    out = clean_caption(REAL_TAU)
    assert "$" not in out and "\\" not in out
    assert out != "Testing the effect of the function"


def test_no_latex_markup_survives_any_real_caption():
    for raw in (REAL_ALG, REAL_EMPH, REAL_REMOVEP, REAL_TAU):
        out = clean_caption(raw)
        assert "\\" not in out, raw
        assert "$" not in out, raw
        assert "{" not in out and "}" not in out, raw


def test_cleaning_never_leaves_ragged_whitespace():
    """A dropped macro must not leave a leading space, as ' Application' would."""
    for raw in (REAL_ALG, REAL_EMPH, REAL_REMOVEP, REAL_TAU):
        out = clean_caption(raw)
        assert out == out.strip()
        assert "  " not in out


# --------------------------------------------------------------------------
# Macro loss must be detectable, not silent
# --------------------------------------------------------------------------

def test_dropped_macro_is_reported():
    """\\ALG carries the paper's own name for its algorithm. Cleaning drops it,
    leaving 'Application' — the caller must be able to notice."""
    cleaned = clean_caption(REAL_ALG)
    assert "ALG" in lost_macros(REAL_ALG, cleaned)


def test_preserved_macro_is_not_reported_as_lost():
    assert lost_macros(REAL_EMPH, clean_caption(REAL_EMPH)) == ()


def test_plain_caption_loses_nothing():
    assert lost_macros("Introduction", "Introduction") == ()


def test_lost_macros_deduplicates():
    raw = "\\foo and \\foo again"
    assert lost_macros(raw, "and again").count("foo") == 1


# --------------------------------------------------------------------------
# The fallback path must work on its own
# --------------------------------------------------------------------------

def test_fallback_handles_every_real_caption():
    """pylatexenc is optional; the pipeline must not depend on it."""
    for raw in (REAL_ALG, REAL_EMPH, REAL_REMOVEP, REAL_TAU):
        out = _fallback_clean(raw)
        assert "\\" not in out and "$" not in out, raw
        assert out == out.strip()


def test_fallback_keeps_emph_argument():
    assert _fallback_clean(REAL_EMPH) == "Siblings score"


def test_fallback_spells_out_greek():
    assert "tau" in _fallback_clean(REAL_TAU)


def test_fallback_used_when_pylatexenc_is_broken(monkeypatch):
    """An import failure inside the cleaner must degrade, not raise."""
    import builtins
    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name.startswith("pylatexenc"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert clean_caption(REAL_EMPH) == "Siblings score"


def test_cleaner_is_deterministic():
    for raw in (REAL_ALG, REAL_EMPH, REAL_REMOVEP, REAL_TAU):
        assert clean_caption(raw) == clean_caption(raw)


def test_greek_preserved_as_unicode_is_not_reported_lost():
    """pylatexenc renders $\\tau$ as 'τ' while the fallback writes 'tau'.
    Checking only the word reported a preserved symbol as lost."""
    cleaned = clean_caption(REAL_TAU)
    assert lost_macros(REAL_TAU, cleaned) == ()


def test_greek_preserved_as_word_is_not_reported_lost():
    assert lost_macros(REAL_TAU, _fallback_clean(REAL_TAU)) == ()


def test_genuinely_deleted_symbol_is_still_reported():
    """The check must not become vacuous — a symbol that really vanished
    should still be flagged."""
    assert "tau" in lost_macros(REAL_TAU, "Testing the effect of the function")


# --------------------------------------------------------------------------
# clean_body_text — DocModel inline placeholders
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.captions import clean_body_text        # noqa: E402

REAL_FORMULA = "Let {{2209.00445_FO0001||FO}} be a space of textual objects"
REAL_CITATION = "as shown by {{2209.00445_REF_roberta||CIT}} in prior work"


def test_formula_placeholder_becomes_a_word():
    """Deleting it outright would leave 'Let be a space of textual objects'."""
    out = clean_body_text(REAL_FORMULA)
    assert "{{" not in out and "||" not in out
    assert "formula" in out
    assert out.startswith("Let formula be a space")


def test_citation_placeholder_becomes_its_citekey():
    """The key is real signal: a section citing roberta is partly about it."""
    out = clean_body_text(REAL_CITATION)
    assert "roberta" in out
    assert "{{" not in out and "REF" not in out


def test_no_placeholder_syntax_survives():
    for raw in (REAL_FORMULA, REAL_CITATION):
        out = clean_body_text(raw)
        assert "{{" not in out and "}}" not in out and "||" not in out


def test_plain_prose_is_untouched():
    text = "The concept space is built from the document's own structure."
    assert clean_body_text(text) == text


def test_whitespace_is_tidied_after_substitution():
    out = clean_body_text("a {{x_FO0001||FO}} b")
    assert "  " not in out


def test_unknown_placeholder_kind_degrades_to_its_name():
    assert "widget" in clean_body_text("see {{x_W1||WIDGET}} here")


def test_multiple_placeholders_in_one_paragraph():
    out = clean_body_text("{{a_FO1||FO}} and {{b_REF_lime||CIT}} and {{c_FO2||FO}}")
    assert out.count("formula") == 2 and "lime" in out


def test_empty_input_is_handled():
    assert clean_body_text("") == "" and clean_body_text(None) == ""


def test_newlines_are_preserved():
    """Paragraph structure matters for sentence splitting later."""
    assert "\n" in clean_body_text("line one\nline two")
