"""Mechanical checks on what the model returned.

Every clause of GATE 5 is here. A prompt is an instruction, not a guarantee,
and these are what turn "the labels look better" into a number.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.labelcheck import (BANNED_CONSTRUCTIONS,
                                               bare_acronyms,
                                               banned_constructions,
                                               check_abstraction, check_label,
                                               check_summary, check_tiers,
                                               word_count)

GOOD = ("supervised classification of short keyword search requests by their "
        "orientation in time, distinguishing past, recency, future and "
        "atemporal intent using surface wording, lexical cues, corpus "
        "timestamps and temporal expression resolution over document text")


def test_the_reference_good_label_passes_every_clause():
    assert 30 <= word_count(GOOD) <= 42
    assert check_label(GOOD) == []


# --------------------------------------------------------------------------
# Banned constructions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", BANNED_CONSTRUCTIONS)
def test_every_declared_banned_construction_is_detected(phrase):
    assert banned_constructions(f"Prose containing {phrase} inside it") == [phrase]


def test_detection_is_case_insensitive():
    assert banned_constructions("This Section Outlines the approach")


def test_detection_survives_a_line_break():
    """A phrase split across lines is the same phrase."""
    assert "this section" in banned_constructions("text\nthis\nsection more")


def test_the_observed_defect_is_caught():
    """The real output that motivated the ban."""
    assert banned_constructions(
        "The section outlines the approach to the TQIC task") == \
        ["the section", "outlines"]


def test_ordinary_prose_trips_nothing():
    assert banned_constructions(
        "Temporal expressions are resolved to calendar intervals.") == []


def test_a_banned_word_inside_a_longer_word_is_not_a_match():
    """`presents` is banned; `representsational` must not trip it."""
    assert "presents" not in banned_constructions("the model represents intent")


# --------------------------------------------------------------------------
# Label clauses
# --------------------------------------------------------------------------

def test_a_short_label_fails_with_its_count():
    problems = check_label("far too short a label")
    assert any("below 30" in p for p in problems)


def test_a_long_label_fails_with_its_count():
    problems = check_label(" ".join(["word"] * 50))
    assert any("above 42" in p for p in problems)


def test_an_empty_label_fails():
    assert check_label("") == ["label is empty"]


@pytest.mark.parametrize("text,found", [
    ("temporal query intent classification (TQIC) of search requests", "(TQIC)"),
    ("bidirectional encoder representations (BERT) applied to formulae", "(BERT)"),
    ("relational database management system (DBMS) selection", "(DBMS)"),
])
def test_a_parenthesised_acronym_is_caught(text, found):
    assert any(found in p for p in check_label(text))


def test_a_parenthesised_word_is_not_an_acronym():
    """`(Windows)` is a proper noun, not a document-local shorthand."""
    assert not any("acronym" in p for p in check_label(GOOD + " (Windows)"))


def test_a_copula_makes_it_a_sentence_not_a_noun_phrase():
    """The observed defect: every label following `X is a Y that Zs` puts a
    constant template across the whole corpus."""
    problems = check_label(
        "temporal query intent classification is a supervised learning task "
        "that uses feature engineering over query wording and document text "
        "to assign each request a temporal orientation label")
    assert any("copula" in p for p in problems)


def test_participles_are_not_copulas():
    assert not any("copula" in p for p in check_label(GOOD))


@pytest.mark.parametrize("marker", ["[3]", "[3, 4]", "[3-5]",
                                    "[Martin et al., 1999]",
                                    "(Allen et al., 2000)"])
def test_citation_markers_are_caught(marker):
    assert any("citation" in p for p in check_label(f"{GOOD} {marker}"))


@pytest.mark.parametrize("ref", ["Figure 3", "Table 4", "Equation 12",
                                 "Section 5", "Algorithm 2", "Fig. 1"])
def test_document_local_references_are_caught(ref):
    assert any("document-local reference" in p
               for p in check_label(f"{GOOD} shown in {ref}"))


def test_a_year_is_not_a_float_reference():
    assert not any("document-local" in p
                   for p in check_label(GOOD.replace("document text", "text 2019")))


# --------------------------------------------------------------------------
# Abstraction and summary
# --------------------------------------------------------------------------

def test_abstraction_is_checked_only_for_banned_constructions():
    """Word budget and noun-phrase form are label clauses, not these."""
    assert check_abstraction("A short glossary entry.") == []


def test_abstraction_catches_the_defect():
    assert check_abstraction("This paper presents a method.")


def test_summary_catches_the_defect():
    assert check_summary("The authors describe a framework.")


def test_an_empty_tier_says_so():
    assert check_abstraction("") == ["abstraction is empty"]
    assert check_summary("  ") == ["summary is empty"]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def test_bare_acronyms_are_reported_not_gated():
    """Whether a proper noun is this paper's own cannot be decided from the
    text, so candidates are surfaced for a human rather than judged."""
    assert bare_acronyms("uses BERT and SOTorrent with SQL") == ["BERT", "SQL"]
    assert check_label(GOOD + " via BERT") == []


def test_check_tiers_reports_an_absent_tier_rather_than_passing_it():
    out = check_tiers(None, "an abstraction", "a summary")
    assert out["label"] == ["label is absent"]
    assert out["abstraction"] == [] and out["summary"] == []
