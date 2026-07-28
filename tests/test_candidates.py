"""Candidate generators."""
from __future__ import annotations

import pytest

from conceptdrill.abstractor import NullAbstractor
from conceptdrill.candidates import (build_generators, generate_candidates,
                                     merge_candidates)
from conceptdrill.candidates.base import structural_weight
from conceptdrill.candidates.bibliography import BibliographyGenerator, clean_title
from conceptdrill.candidates.equations import EquationGenerator
from conceptdrill.candidates.glossary import (GlossaryGenerator, find_long_form,
                                              _is_short_form)
from conceptdrill.candidates.headings import HeadingGenerator, strip_numbering
from conceptdrill.candidates.nounphrases import NounPhraseGenerator
from conceptdrill.types import Candidate


@pytest.fixture
def abstractor():
    return NullAbstractor()


# --------------------------------------------------------------------------
# 1.1 Headings
# --------------------------------------------------------------------------

def test_strip_numbering():
    assert strip_numbering("2.1 Semantic Projection") == "Semantic Projection"
    assert strip_numbering("IV. Results") == "Results"
    assert strip_numbering("Method") == "Method"


def test_headings_become_candidates(mock_document, abstractor):
    names = {c.name for c in
             HeadingGenerator().generate(mock_document, abstractor=abstractor)}
    assert "Semantic Projection" in names
    assert "Concept Scoring" in names


def test_boilerplate_headings_are_dropped(mock_document, abstractor):
    names = {c.name.lower() for c in
             HeadingGenerator().generate(mock_document, abstractor=abstractor)}
    assert "introduction" not in names
    assert "glossary" not in names


def test_hierarchical_path_candidates(mock_document, abstractor):
    cands = HeadingGenerator().generate(mock_document, abstractor=abstractor)
    paths = [c for c in cands if c.kind == "heading_path"]
    assert paths, "nested sections must produce breadcrumb concepts"
    assert any(" > " in c.name for c in paths)


def test_path_tau_is_prose_not_separator(mock_document, abstractor):
    cands = HeadingGenerator().generate(mock_document, abstractor=abstractor)
    path = next(c for c in cands if c.kind == "heading_path")
    assert ">" not in path.tau


# --------------------------------------------------------------------------
# 1.2 Glossary / definitions
# --------------------------------------------------------------------------

def test_short_form_detection():
    assert _is_short_form("CNN")
    assert _is_short_form("BERT")
    assert not _is_short_form("the")
    assert not _is_short_form("a very long phrase")


def test_schwartz_hearst_finds_expansion():
    long = find_long_form("CNN", "we use a Convolutional Neural Network ")
    assert long is not None
    assert "Convolutional Neural Network" in long


def test_schwartz_hearst_rejects_mismatch():
    assert find_long_form("XYZ", "there is nothing matching here ") is None


def test_acronym_becomes_concept_with_alias(mock_document, abstractor):
    cands = GlossaryGenerator().generate(mock_document, abstractor=abstractor)
    acronyms = [c for c in cands if c.kind == "acronym"]
    assert acronyms
    assert any("CNN" in (c.metadata.get("aliases") or []) for c in acronyms)


def test_definition_environment_is_captured(mock_document, abstractor):
    names = {c.name.lower() for c in
             GlossaryGenerator().generate(mock_document, abstractor=abstractor)}
    assert "concept projection" in names


def test_glossary_section_entries_are_captured(mock_document, abstractor):
    names = {c.name.lower() for c in
             GlossaryGenerator().generate(mock_document, abstractor=abstractor)}
    assert "concept space" in names or "latent vector" in names


# --------------------------------------------------------------------------
# 1.3 Bibliography
# --------------------------------------------------------------------------

def test_clean_title_strips_markers_and_authors():
    assert clean_title("[12] A Great Paper Title") == "A Great Paper Title"
    raw = "Cao, Z., Long, M.: Hashnet Deep Learning To Hash"
    assert "Hashnet" in clean_title(raw)


def test_clean_title_keeps_original_when_stripping_is_too_aggressive():
    """A short title must survive: over-stripping would lose the concept."""
    assert clean_title("Support Vector Networks") == "Support Vector Networks"


def test_bibliography_titles_become_candidates(mock_document, abstractor):
    cands = BibliographyGenerator().generate(mock_document, abstractor=abstractor)
    names = {c.name for c in cands}
    assert "Attention Is All You Need" in names
    assert "Support Vector Networks" in names


def test_long_title_is_shortened_but_tau_keeps_everything(mock_document, abstractor):
    cands = BibliographyGenerator().generate(mock_document, abstractor=abstractor)
    ces = next(c for c in cands if "Conceptualizing" in c.tau)
    assert len(ces.name.split()) < len(ces.tau.split())
    assert "Conceptualizing Embedding Spaces" in ces.tau


def test_citation_metadata_is_carried(mock_document, abstractor):
    cands = BibliographyGenerator().generate(mock_document, abstractor=abstractor)
    attention = next(c for c in cands if c.name == "Attention Is All You Need")
    assert attention.metadata["citations"] == 100000
    assert attention.metadata["year"] == 2017


# --------------------------------------------------------------------------
# 1.4 Noun phrases
# --------------------------------------------------------------------------

def test_noun_phrases_respect_min_count(mock_document, abstractor):
    gen = NounPhraseGenerator(min_count=3)
    for c in gen.generate(mock_document, abstractor=abstractor):
        assert c.frequency >= 3


def test_noun_phrases_respect_max_tokens(mock_document, abstractor):
    gen = NounPhraseGenerator(min_count=2, max_tokens=5)
    for c in gen.generate(mock_document, abstractor=abstractor):
        assert len(c.name.split()) <= 5


def test_substring_absorption_drops_redundant_phrases(abstractor):
    from conceptdrill.nlp import Phrase
    gen = NounPhraseGenerator()
    kept = gen._absorb([
        Phrase("concept space", 10),
        Phrase("the concept space", 8),      # adds nothing over the shorter
        Phrase("concept scoring", 6),
    ])
    names = {p.text for p in kept}
    assert "concept space" in names
    assert "the concept space" not in names
    assert "concept scoring" in names


def test_nlp_backend_is_recorded(mock_document, abstractor):
    cands = NounPhraseGenerator(min_count=2).generate(
        mock_document, abstractor=abstractor)
    if cands:
        assert cands[0].metadata.get("nlp_backend") in {"stanza", "spacy", "regex"}


# --------------------------------------------------------------------------
# 1.6 Equations
# --------------------------------------------------------------------------

def test_equation_abstraction_is_deterministic(mock_document, abstractor):
    gen = EquationGenerator()
    a = [c.name for c in gen.generate(mock_document, abstractor=abstractor)]
    b = [c.name for c in gen.generate(mock_document, abstractor=abstractor)]
    assert a == b


def test_equation_abstraction_reads_operators(abstractor):
    assert "summation" in abstractor.describe_equation(r"\sum_{i} x_i")
    assert "integral" in abstractor.describe_equation(r"\int_0^1 f(x) dx")
    assert "loss function" in abstractor.describe_equation(r"\mathcal{L} = 1")


def test_equation_candidates_record_their_exemplar(mock_document, abstractor):
    cands = EquationGenerator().generate(mock_document, abstractor=abstractor)
    if cands:
        assert cands[0].metadata["example_object_id"]
        assert cands[0].metadata["abstractor_deterministic"] is True


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def test_merge_keeps_strongest_source():
    merged = merge_candidates([
        Candidate(name="Deep Hashing", source="nounphrase", frequency=5),
        Candidate(name="deep hashing", source="heading", frequency=1),
        Candidate(name="Deep  Hashing", source="ner", frequency=2),
    ])
    assert len(merged) == 1
    assert merged[0].source == "heading"          # highest structural weight
    assert merged[0].frequency == 8               # frequencies summed
    assert merged[0].metadata["sources"] == ["heading", "ner", "nounphrase"]


def test_merge_is_order_independent():
    a = [Candidate(name="X", source="ner"), Candidate(name="X", source="heading")]
    assert (merge_candidates(a)[0].source
            == merge_candidates(list(reversed(a)))[0].source)


def test_merge_preserves_metadata_from_losers():
    merged = merge_candidates([
        Candidate(name="BERT", source="heading"),
        Candidate(name="BERT", source="bibliography", metadata={"year": 2019}),
    ])
    assert merged[0].metadata.get("year") == 2019


def test_structural_weights_follow_the_spec():
    assert structural_weight("glossary") == 1.0
    assert structural_weight("heading") == 1.0
    assert structural_weight("bibliography") == 0.7
    assert structural_weight("nounphrase") == 0.5
    assert structural_weight("equation") == 0.4
    assert structural_weight("ner") == 0.3


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown concept source"):
        build_generators(["not-a-generator"])


def test_generate_candidates_runs_every_generator(mock_document):
    cands = generate_candidates(mock_document)
    sources = {s for c in cands for s in c.metadata.get("sources", [c.source])}
    assert "heading" in sources
    assert "bibliography" in sources
    assert len(cands) > 5
