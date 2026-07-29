"""Unit tests for LLM output sanitisation.

The guiding distinction: normalise what is invisible or a lookalike, keep what
is content. A sanitiser that eats Greek letters or accented names is worse than
none, because it destroys meaning to fix cosmetics.
"""
from __future__ import annotations

from conceptdrill.hierarchy.sanitize import (find_suspect_characters,
                                             sanitize_summary_fields,
                                             sanitize_text)


# --------------------------------------------------------------------------
# The characters actually observed in a real LLM-written prompt
# --------------------------------------------------------------------------

def test_non_breaking_hyphen_becomes_ascii():
    """Nine of these arrived in the improved prompt, invisible on screen."""
    assert sanitize_text("embedding‑ready") == "embedding-ready"


def test_en_dash_becomes_ascii():
    assert sanitize_text("30–42 words") == "30-42 words"


def test_em_dash_becomes_ascii():
    assert sanitize_text("a — b") == "a - b"


def test_curly_quotes_become_ascii():
    assert sanitize_text("“hello”") == '"hello"'
    assert sanitize_text("author’s") == "author's"


def test_ellipsis_is_expanded():
    assert sanitize_text("wait…") == "wait..."


# --------------------------------------------------------------------------
# Invisible characters
# --------------------------------------------------------------------------

def test_zero_width_space_is_removed():
    assert sanitize_text("con​cept") == "concept"


def test_zero_width_joiner_and_non_joiner_are_removed():
    assert sanitize_text("a‌b‍c") == "abc"


def test_byte_order_mark_is_removed():
    assert sanitize_text("﻿concept") == "concept"


def test_soft_hyphen_is_removed():
    assert sanitize_text("con­cept") == "concept"


def test_bidi_controls_are_removed():
    assert sanitize_text("a‮b") == "ab"


def test_non_breaking_space_becomes_a_normal_space():
    """`30\\u00a0words` otherwise survives a naive split on ASCII space."""
    assert sanitize_text("30 words") == "30 words"
    assert len(sanitize_text("30 words").split()) == 2


def test_narrow_no_break_space_becomes_a_space():
    assert sanitize_text("a b") == "a b"


# --------------------------------------------------------------------------
# Real content must survive
# --------------------------------------------------------------------------

def test_greek_letters_survive():
    """tau is meaning, not decoration -- the prompt asks for symbols as words,
    but where one appears it must not be destroyed."""
    assert "τ" in sanitize_text("the τ function")


def test_accented_names_survive():
    assert sanitize_text("Müller and Gödel") == "Müller and Gödel"


def test_cjk_survives():
    assert sanitize_text("概念空間") == "概念空間"


def test_ordinary_ascii_is_untouched():
    text = "A canonical definition, reusable across documents (see section 2)."
    assert sanitize_text(text) == text


def test_mathematical_symbols_survive():
    assert "≤" in sanitize_text("x ≤ y")


# --------------------------------------------------------------------------
# Control characters and whitespace
# --------------------------------------------------------------------------

def test_tab_from_an_eaten_latex_command_is_removed():
    """`\\tau` arrives as TAB + 'au'; the tab must not reach an embedder."""
    assert "\t" not in sanitize_text("the \tau function")


def test_backspace_and_formfeed_are_removed():
    assert sanitize_text("a\bb\fc") == "a b c"


def test_newlines_are_preserved():
    assert "\n" in sanitize_text("line one\nline two")


def test_runs_of_blank_lines_are_collapsed():
    assert sanitize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_repeated_spaces_are_collapsed():
    assert sanitize_text("a     b") == "a b"


def test_leading_and_trailing_whitespace_is_stripped():
    assert sanitize_text("  concept  ") == "concept"


def test_whitespace_collapse_can_be_disabled():
    assert sanitize_text("a     b", collapse_whitespace=False) == "a     b"


def test_empty_and_none_input():
    assert sanitize_text("") == "" and sanitize_text(None) == ""


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_suspects_are_named_not_merely_counted():
    found = find_suspect_characters("embedding‑ready")
    assert any("U+2011" in f and "NON-BREAKING HYPHEN" in f for f in found)


def test_clean_text_has_no_suspects():
    assert find_suspect_characters("a perfectly ordinary sentence") == ()


def test_content_characters_are_not_flagged():
    """Greek and accents are content; flagging them would cry wolf."""
    assert find_suspect_characters("Müller studied τ decay") == ()


def test_suspects_are_deduplicated():
    found = find_suspect_characters("a‑b‑c‑d")
    assert len(found) == 1


def test_summary_fields_are_sanitised_and_reported():
    clean, warnings = sanitize_summary_fields({
        "label": "cross‑document key",
        "summary": "a clean sentence",
    })
    assert clean["label"] == "cross-document key"
    assert clean["summary"] == "a clean sentence"
    assert len(warnings) == 1 and "label" in warnings[0]


def test_no_warnings_when_nothing_was_changed():
    _, warnings = sanitize_summary_fields({"label": "clean text"})
    assert warnings == ()


def test_sanitisation_is_idempotent():
    once = sanitize_text("embedding‑ready “quoted”…")
    assert sanitize_text(once) == once
