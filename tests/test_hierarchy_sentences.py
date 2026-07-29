"""Unit tests for sentence splitting and extraction.

The abbreviation cases are ones the drilled corpus actually contains.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.docmodel_tree import build_tree
from conceptdrill.hierarchy.sentences import (ABBREVIATIONS, MAY_BE_FINAL,
                                              NEVER_FINAL, Sentence,
                                              sentence_stats,
                                              sentences_from_tree,
                                              split_sentences)


# --------------------------------------------------------------------------
# Basic splitting
# --------------------------------------------------------------------------

def test_two_sentences_split():
    assert split_sentences("One idea. Another idea.") == \
        ["One idea.", "Another idea."]


def test_question_and_exclamation_end_sentences():
    assert len(split_sentences("Is it so? Indeed! Yes.")) == 3


def test_a_single_sentence_stays_whole():
    assert split_sentences("Just the one idea here.") == \
        ["Just the one idea here."]


def test_trailing_text_without_punctuation_is_kept():
    """Drilled text often loses a final period; dropping it would lose content."""
    assert split_sentences("First one. A tail with no period")[-1] == \
        "A tail with no period"


def test_empty_input():
    assert split_sentences("") == [] and split_sentences("   ") == []


def test_whitespace_is_normalised():
    assert split_sentences("One   idea\n  here.") == ["One idea here."]


def test_no_empty_strings_are_produced():
    assert all(s.strip() for s in split_sentences("A. . B.  ...  C."))


# --------------------------------------------------------------------------
# The cases that break naive splitters
# --------------------------------------------------------------------------

def test_eg_does_not_split():
    """`e.g.` otherwise leaves a stray 'g.' -- present in the real corpus."""
    got = split_sentences("Some models, e.g. BERT, are large. That is known.")
    assert len(got) == 2
    assert "e.g. BERT" in got[0]


def test_ie_does_not_split():
    assert len(split_sentences("The space, i.e. The basis, is shared.")) == 1


def test_et_al_does_not_split():
    got = split_sentences("Shown by Vaswani et al. The result holds.")
    assert len(got) == 2, "et al. must not split, but the next sentence must"


def test_decimals_do_not_split():
    assert len(split_sentences("The threshold is 0.85 for merging.")) == 1


def test_version_numbers_do_not_split():
    assert len(split_sentences("We used version 3.14 throughout.")) == 1


def test_initials_do_not_split():
    assert len(split_sentences("Written by J. Smith and A. B. Jones here.")) == 1


def test_enumerators_do_not_split():
    """'3. Method' is a list marker, not a sentence end."""
    assert len(split_sentences("See 3. Method for details.")) == 1


def test_rendered_latex_macro_abbreviation():
    """`\\ALG` renders to 'ALG.' and appears four times in the reference paper."""
    assert "alg" in ABBREVIATIONS


def test_never_final_and_may_be_final_are_disjoint():
    assert not (NEVER_FINAL & MAY_BE_FINAL)


def test_eg_is_never_final():
    """It always introduces something, so its period is never a boundary."""
    assert "e.g" in NEVER_FINAL and "e.g" not in MAY_BE_FINAL


def test_et_al_may_be_final():
    """'Shown by Vaswani et al.' is a complete sentence. English writes one
    period for both jobs, so the ambiguity cannot be resolved from punctuation;
    splitting is the safer default, since fusing two ideas into one vector is
    the worse error for a projection unit."""
    assert "al" in MAY_BE_FINAL


def test_fig_and_eq_do_not_split():
    assert len(split_sentences("As Fig. 2 shows, the value holds.")) == 1
    assert len(split_sentences("Given Eq. 5 the bound follows.")) == 1


def test_a_real_sentence_after_an_abbreviation_still_splits():
    """The guard must not become a blanket suppression."""
    got = split_sentences("We cite Smith et al. Results follow in the table.")
    assert len(got) == 2


def test_closing_quote_or_bracket_after_the_period():
    got = split_sentences('He said "it works." Then he left.')
    assert len(got) == 2


def test_paragraph_break_is_a_hard_boundary():
    """Even without punctuation -- drilled headings often lack a period."""
    got = split_sentences("A heading with no period\n\nThe body follows.")
    assert len(got) == 2


def test_lowercase_after_a_period_does_not_split():
    """Genuine sentences start with a capital; a lowercase follower signals
    that the period was part of something else."""
    assert len(split_sentences("the value is x.y in this notation")) == 1


def test_splitting_is_deterministic():
    text = "One idea, e.g. this. Another, per Smith et al. Done."
    assert split_sentences(text) == split_sentences(text)


# --------------------------------------------------------------------------
# Extraction from a tree
# --------------------------------------------------------------------------

def _sec(sid, cap, level, flow):
    return {"id": sid, "type": "Section",
            "props": {"caption": cap, "level": level, "flow_index": flow}}


def _par(pid, text, flow, parent=None):
    props = {"text": text, "flow_index": flow}
    if parent:
        props["parent_section"] = parent
    return {"id": pid, "type": "Paragraph", "props": props}


LONG_A = "The concept space is built from the document's own structure here."
LONG_B = "Projection is a single matrix multiplication over the basis rows."


@pytest.fixture
def tree():
    return build_tree({"meta": {"bibkey": "t"}, "objects": [
        _sec("s1", "Method", 2, 1),
        _par("p1", f"{LONG_A} {LONG_B}", 2, "s1"),
        _sec("s2", "Scoring", 3, 3),
        _par("p2", LONG_A, 4, "s2"),
    ]})


def test_sentences_carry_their_section(tree):
    got = sentences_from_tree(tree)
    assert {s.section_id for s in got} == {"s1", "s2"}


def test_sentences_carry_their_source_paragraph(tree):
    got = sentences_from_tree(tree)
    assert {s.source_id for s in got} == {"p1", "p2"}


def test_sentence_ids_are_unique(tree):
    got = sentences_from_tree(tree)
    assert len({s.id for s in got}) == len(got)


def test_sentence_ids_trace_back_to_the_source(tree):
    assert all(s.id.startswith(s.source_id + "#s") for s in sentences_from_tree(tree))


def test_sentences_are_in_reading_order(tree):
    got = sentences_from_tree(tree)
    assert [s.flow_index for s in got] == sorted(s.flow_index for s in got)


def test_a_paragraph_yields_several_sentences(tree):
    got = [s for s in sentences_from_tree(tree) if s.source_id == "p1"]
    assert len(got) == 2


def test_level_filter_restricts_extraction(tree):
    got = sentences_from_tree(tree, levels={2})
    assert {s.section_id for s in got} == {"s1"}


def test_short_fragments_are_dropped(tree):
    """A five-character fragment embeds to a near-arbitrary direction."""
    t = build_tree({"objects": [_sec("s1", "M", 2, 1),
                                _par("p1", f"Ok. {LONG_A}", 2, "s1")]})
    got = sentences_from_tree(t)
    assert all(len(s.text) >= 20 for s in got)
    assert len(got) == 1


def test_orphans_are_included_by_default():
    """Front matter is usually the abstract -- concept-dense text."""
    t = build_tree({"objects": [_par("p0", LONG_A, 1), _sec("s1", "M", 2, 5)]})
    got = sentences_from_tree(t)
    assert any(s.section_id is None for s in got)


def test_orphans_can_be_excluded():
    t = build_tree({"objects": [_par("p0", LONG_A, 1), _sec("s1", "M", 2, 5)]})
    assert sentences_from_tree(t, include_orphans=False) == []


def test_empty_tree_yields_nothing():
    assert sentences_from_tree(build_tree({"objects": []})) == []


def test_stats_describe_the_set(tree):
    st = sentence_stats(sentences_from_tree(tree))
    assert st["sentences"] == 3 and st["sections"] == 2
    assert st["words_min"] <= st["words_median"] <= st["words_max"]


def test_stats_of_an_empty_set():
    assert sentence_stats([])["sentences"] == 0
