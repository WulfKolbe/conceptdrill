r"""The cleaning contract for `basis_text`.

The postcondition tests come first because they are the contract; the rule
tests follow because they are how it is met. A change that keeps the
postconditions and alters the rules is fine. The reverse is not.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.basistext import (FORBIDDEN_CHARS, BasisTextViolation,
                                              check_basis_text, clean_basis_text)


def clean(text, title=""):
    return clean_basis_text(text, title=title).text


def rules(text, title=""):
    return set(clean_basis_text(text, title=title).rules_fired)


# --------------------------------------------------------------------------
# The postconditions, stated as tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("char", sorted(FORBIDDEN_CHARS))
def test_no_forbidden_character_survives(char):
    out = clean(f"Some prose {char} and more prose about embeddings.")
    assert char not in out


def test_the_real_corpus_failure_is_removed():
    r"""`\section*{` appeared in every label of the void run."""
    out = clean(r"\section*{2 Related Work} Prominent examples of dialogue "
                r"platforms include OOA and TRIPS.",
                title="2 Related Work")
    assert "section" not in out.lower()
    assert out.startswith("Prominent examples")


def test_no_command_residue():
    out = clean(r"We \cite{smith} show in \ref{sec:x} that \label{a} it holds.")
    assert check_basis_text(out) == []
    assert "cite" not in out and "ref" not in out


def test_does_not_begin_with_the_raw_title():
    out = clean("Dialogue System Framework. It fuses multimodal input.",
                title="Dialogue System Framework")
    assert out.startswith("It fuses")


def test_does_not_begin_with_the_normalized_title():
    """Case and punctuation differ; the title is still the title."""
    out = clean("3. APPROACH -- temporal classification uses features.",
                title="3 Approach")
    assert "APPROACH" not in out
    assert out.startswith("temporal classification")


@pytest.mark.parametrize("prefix", ["3. ", "3.1 ", "3.1.4 ", "IV. ", "A. ",
                                    "(3) ", "12 "])
def test_no_leading_ordinal(prefix):
    out = clean(f"{prefix}Temporal query classification for retrieval.")
    assert out.startswith("Temporal")


def test_a_leading_article_is_not_mistaken_for_an_ordinal():
    """`A framework...` must survive; `A. Framework` must not."""
    assert clean("A framework for federated search.").startswith("A framework")
    assert clean("I introduce a method.").startswith("I introduce")


def test_check_reports_every_clause_it_breaks():
    problems = check_basis_text(r"3. \section{X} $y$ text", title="3 X")
    assert len(problems) >= 2


def test_a_clean_text_reports_nothing():
    assert check_basis_text("Temporal query intent classification for search.") == []


# --------------------------------------------------------------------------
# The assertion is real
# --------------------------------------------------------------------------

def test_the_cleaner_raises_rather_than_returning_dirty_text(monkeypatch):
    """If a rule stops firing, the failure must be loud and immediate."""
    import conceptdrill.hierarchy.basistext as bt
    monkeypatch.setattr(bt, "_strip_latex", lambda s, r, depth=0: s)
    with pytest.raises(BasisTextViolation, match="postcondition"):
        bt.clean_basis_text(r"\section*{X} body text")


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------

def test_formatting_commands_keep_their_argument():
    assert "Siblings" in clean(r"The \emph{Siblings} score is degenerate.")


def test_structural_commands_lose_their_argument():
    out = clean(r"Text \footnote{an aside that is not the concept} continues.")
    assert "aside" not in out
    assert out.startswith("Text") and out.endswith("continues.")


def test_an_unknown_command_drops_its_name_and_keeps_its_prose():
    out = clean(r"\somethingodd{real prose about embeddings} follows.")
    assert "real prose about embeddings" in out
    assert "somethingodd" not in out


def test_symbols_become_words_rather_than_disappearing():
    r"""Deleting `\tau` turns "the $\tau$ function" into "the function"."""
    assert "tau" in clean(r"Testing the effect of the $\tau$ function.")


def test_nested_braces_do_not_leave_residue():
    out = clean(r"\section*{The $f(x)$ case} Real content here.")
    assert check_basis_text(out) == []
    assert out.startswith("Real content")


def test_an_unbalanced_brace_is_still_removed():
    out = clean("Prose with a stray { brace and $ sign.")
    assert check_basis_text(out) == []


def test_a_comment_is_dropped():
    assert "hidden" not in clean("Visible text % hidden editorial note\nmore.")


def test_every_applied_rule_is_named():
    fired = rules(r"\section*{2 Related Work} The \emph{score} is $\tau$.",
                  title="2 Related Work")
    assert "drop-command:section*" in fired
    assert "unwrap-command:emph" in fired
    assert "expand-symbol:tau" in fired


def test_clean_text_fires_no_rules():
    assert rules("Temporal query intent classification for search.") == set()


def test_the_title_does_not_contribute_by_default():
    """The documented decision, asserted rather than described."""
    out = clean("The framework fuses multimodal input.",
                title="Dialogue System Framework")
    assert "Dialogue System Framework" not in out


def test_including_the_title_is_possible_but_explicit():
    got = clean_basis_text("It fuses multimodal input.",
                           title="3 Dialogue System Framework",
                           include_title=True)
    assert got.text.startswith("Dialogue System Framework")
    assert "prepend-title" in got.rules_fired


def test_the_title_contributes_at_most_once():
    got = clean_basis_text("Dialogue System Framework. It fuses input.",
                           title="Dialogue System Framework",
                           include_title=True)
    assert got.text.lower().count("dialogue system framework") == 1


def test_empty_input_is_empty_output():
    assert clean("") == "" and clean("   ") == ""


def test_a_span_that_is_only_markup_becomes_empty():
    assert clean(r"\label{sec:intro}\ref{fig:1}") == ""


# --------------------------------------------------------------------------
# Titles that are not Latin, or not words at all
# --------------------------------------------------------------------------

def test_a_non_latin_title_does_not_condemn_every_text():
    """`[^a-z0-9]` normalised a Coptic title to the empty string, and
    `text.startswith("")` is always true -- so every concept in that document
    "began with its marker title". Two chunks of the overnight run died here."""
    title = "ΠετηαΝογογ. Μπεο-οογ ΝΑΚιΜ ΑΝ ΩΜΠεφΗϊ"
    assert check_basis_text("neural network training for Coptic recognition",
                            title) == []


def test_a_title_of_pure_punctuation_matches_nothing():
    assert check_basis_text("geometric text-line model with a baseline",
                            r"\({") == []


def test_non_latin_scripts_survive_normalisation():
    from conceptdrill.hierarchy.basistext import _normalise
    assert _normalise("ΠετηαΝογογ") == "πετηανογογ"
    assert _normalise("Причинность") == "причинность"
    assert _normalise("因果推論") == "因果推論"


def test_a_non_latin_title_is_still_stripped_when_it_leads():
    """Widening the alphabet must not lose the rule it exists for."""
    title = "ΠετηαΝογογ"
    out = clean_basis_text(f"{title} the model recognises characters",
                           title=title).text
    assert out.startswith("the model recognises")


def test_a_latin_title_is_still_caught():
    assert any("marker title" in p for p in
               check_basis_text("2 Related Work. Prominent examples",
                                "2 Related Work"))
