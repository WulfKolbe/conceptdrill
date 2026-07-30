"""Dimension zero: the classifier, and the reserved row it feeds.

The recall-side tests are the ones that matter. A structural section that
reaches a concept row contaminates every coordinate derived from it, and unlike
an over-absorbed concept that damage is invisible in the record afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conceptdrill.hierarchy.basis import (STRUCTURAL_LEVEL, STRUCTURAL_ROW_ID,
                                          ConceptBasis)
from conceptdrill.hierarchy.structural import (classify_section, is_structural,
                                               normalise_title)

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "docs" / "measurements" / "structural-labels-10docs.json"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("7. REFERENCES", "references"),
    ("Keywords:", "keywords"),
    ("  References  ", "references"),
    ("A. Acknowledgements", "acknowledgements"),
    ("IV. Bibliography", "bibliography"),
    ("Literaturverzeichnis", "literaturverzeichnis"),
    ("Abkürzungsverzeichnis", "abkuerzungsverzeichnis"),
    ("Danksagung", "danksagung"),
])
def test_titles_normalise_to_their_comparison_form(raw, want):
    assert normalise_title(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("A framework for search", "a framework for search"),
    ("I Introduction", "i introduction"),
    ("A. Acknowledgements", "acknowledgements"),
    ("3 Methods", "methods"),
])
def test_a_single_letter_ordinal_needs_punctuation_to_be_stripped(raw, want):
    """Without that requirement `A framework` loses its article and
    `I Introduction` is indistinguishable from a title starting with a
    pronoun."""
    assert normalise_title(raw) == want


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected_class", [
    ("References", "bibliography"),
    ("7. REFERENCES", "bibliography"),
    ("Bibliography", "bibliography"),
    ("Literaturverzeichnis", "bibliography"),
    ("Acknowledgments", "acknowledgement"),
    ("Danksagung", "acknowledgement"),
    ("Funding", "funding"),
    ("Declaration of competing interest", "competing-interests"),
    ("CRediT authorship contribution statement", "author-contributions"),
    ("Data availability", "data-availability"),
    ("Ethical statement", "ethics"),
    ("Table of Contents", "table-of-contents"),
    ("Inhaltsverzeichnis", "table-of-contents"),
    ("List of Figures", "list-of-floats"),
    ("Tabellenverzeichnis", "list-of-floats"),
    ("Index", "index"),
    ("Supplementary Material", "supplementary"),
    ("Impressum", "front-matter"),
    ("Vorwort", "front-matter"),
    ("Keywords", "metadata-block"),
    ("ARTICLE INFO", "metadata-block"),
    ("Team Name", "metadata-block"),
    ("Subtasks", "metadata-block"),
])
def test_structural_titles_are_classified(title, expected_class):
    assert classify_section(title)[0] == expected_class


@pytest.mark.parametrize("title", [
    "Abstract", "1 Abstract", "Introduction", "2 Related Work",
    "3 Dialogue System Framework", "Experimental Evaluation", "Conclusion",
    "A framework for federated search", "Discussion", "Preliminaries",
    "Datasets", "Results over Benchmark Graphs",
])
def test_content_titles_are_left_alone(title):
    assert classify_section(title) == (None, None)


def test_abstract_is_content_not_structural():
    """Stated in the brief and easy to get wrong: an abstract says what the
    document is about, which is what a concept space wants."""
    assert not is_structural("Abstract")
    assert not is_structural("ABSTRACT")
    assert not is_structural("Zusammenfassung")


@pytest.mark.parametrize("title", ["Box 1", "Box 12", "Figure 3", "Table 4",
                                   "Listing 2", "Algorithm 5"])
def test_numbered_floats_are_absorbed(title):
    assert classify_section(title) == ("float", "float-container")


def test_a_float_word_with_a_real_topic_is_not_a_float():
    """`Table understanding in documents` is a topic, not a table."""
    assert classify_section("Table understanding in documents") == (None, None)


@pytest.mark.parametrize("title", ["#1", "#12", "3.4.2"])
def test_drill_artifacts_are_absorbed(title):
    assert classify_section(title) == ("artifact", "drill-artifact")


def test_a_title_of_only_punctuation_normalises_away_to_untitled():
    """`---` is not an artifact with a name; it has no name at all."""
    assert classify_section("---") == ("untitled", "untitled-section")


def test_an_untitled_section_is_absorbed_rather_than_embedded():
    assert classify_section("") == ("untitled", "untitled-section")
    assert classify_section("   ") == ("untitled", "untitled-section")


@pytest.mark.parametrize("title", ["Appendix", "Appendix A", "Appendix. Additional Details",
                                   "Appendix B: Measurements", "Anhang",
                                   "Anhang B", "Appendices"])
def test_appendix_prefixes_are_absorbed(title):
    """`Appendix. Additional Details` keeps its period through normalisation;
    a space-based prefix test missed it."""
    assert classify_section(title)[0] == "appendix"


def test_the_docmodel_flag_is_a_safety_net():
    """False on all 173 sections of the measured corpus, so it must not be the
    only path -- but it must still work."""
    assert classify_section("Measurements", is_appendix=True) == \
        ("appendix", "docmodel-is-appendix")
    assert classify_section("Measurements", is_appendix=False) == (None, None)


def test_every_classification_names_a_rule():
    for title in ["References", "Box 1", "#1", "", "Appendix A", "Keywords"]:
        structural_class, rule = classify_section(title)
        assert structural_class and rule, title


# --------------------------------------------------------------------------
# Against the hand labels
# --------------------------------------------------------------------------

@pytest.mark.skipif(not LABELS.exists(), reason="hand labels not present")
def test_recall_against_the_hand_labels_is_one():
    """The binding constraint of gate 4, pinned as a unit test so a later
    vocabulary edit cannot quietly lose a category."""
    labels = json.loads(LABELS.read_text())["labels"]
    missed = [l["title"] for l in labels
              if l["structural"] and classify_section(l["title"])[0] is None]
    assert missed == []


@pytest.mark.skipif(not LABELS.exists(), reason="hand labels not present")
def test_precision_is_reported_not_asserted_at_one():
    """Precision is not gated. This records what it currently is, so a drop
    is visible rather than silent."""
    labels = json.loads(LABELS.read_text())["labels"]
    fp = [l["title"] for l in labels
          if not l["structural"] and classify_section(l["title"])[0] is not None]
    assert len(fp) <= 5, f"over-absorption grew: {fp}"


# --------------------------------------------------------------------------
# The reserved row
# --------------------------------------------------------------------------

def unit(*xs):
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def basis_with_sink():
    b = ConceptBasis()
    b.integrate(1, "concept a", unit(1, 0, 0), document="d1")
    b.integrate(1, "concept b", unit(0, 1, 0), document="d1")
    b.absorb_structural("References", unit(0, 0, 1), document="d1",
                        rule="reference-list")
    return b


def test_the_sink_is_row_zero():
    assert basis_with_sink().row_ids()[0] == STRUCTURAL_ROW_ID


def test_the_sink_stays_at_row_zero_however_large_its_support():
    """Support ordering must never displace a reserved row."""
    b = basis_with_sink()
    for i in range(20):
        b.absorb_structural(f"Bibliography {i}", unit(0, 0, 1), document="d2")
    assert b.row_ids()[0] == STRUCTURAL_ROW_ID


def test_absorption_bypasses_tau_entirely():
    """A structural label orthogonal to the sink is still absorbed."""
    b = basis_with_sink()
    before = b.structural.support
    b.absorb_structural("Funding", unit(1, 0, 0), document="d2")
    assert b.structural.support == before + 1
    assert len(b.rows) == 2, "no concept row may be created by absorption"


def test_the_sink_never_changes_identity():
    b = basis_with_sink()
    b.absorb_structural("Acknowledgements", unit(0.5, 0.5, 0.5), document="d2")
    assert b.structural.row_id == STRUCTURAL_ROW_ID
    assert b.structural.level == STRUCTURAL_LEVEL
    assert b.structural.label == "[structural]"


def test_the_sink_records_what_it_absorbed():
    b = basis_with_sink()
    b.absorb_structural("Funding", unit(1, 0, 0), document="d2")
    assert set(b.structural.documents) == {"d1", "d2"}
    assert "References" in b.structural.merged_labels
    assert "Funding" in b.structural.merged_labels


def test_row_counts_exclude_the_sink():
    """The clause that keeps every downstream number honest."""
    stats = basis_with_sink().stats()
    assert stats["rows"] == 2
    assert stats["rows_including_structural"] == 3
    assert stats["rows_including_structural"] - stats["rows"] == 1


def test_a_basis_with_no_structural_section_has_no_sink():
    b = ConceptBasis()
    b.integrate(1, "concept a", unit(1, 0, 0), document="d1")
    stats = b.stats()
    assert b.structural is None
    assert stats["structural_row"] is None
    assert stats["rows"] == stats["rows_including_structural"] == 1


def test_concept_rows_and_ordered_rows_differ_by_the_sink():
    b = basis_with_sink()
    assert len(b.ordered_rows()) == len(b.concept_rows()) + 1
    assert b.matrix().shape[0] == len(b.concept_rows()) + 1


def test_absorption_reports_its_own_decision():
    result = basis_with_sink().absorb_structural("Index", unit(0, 1, 1))
    assert result.action == "absorbed"
    assert result.row_id == STRUCTURAL_ROW_ID
