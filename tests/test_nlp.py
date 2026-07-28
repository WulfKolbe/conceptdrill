"""The NLP tier: backend selection, batching, and the regex fallback.

The neural tiers are exercised only when their models are actually installed.
Everything else asserts the fallback, which is the tier that must always work.
"""
from __future__ import annotations

import pytest

from conceptdrill import nlp
from conceptdrill.nlp import (VALID_BACKENDS, analyse, backend_name,
                              is_acceptable_phrase, named_entities,
                              normalise_phrase, noun_phrases, word_frequencies)

SAMPLE = [
    "Deep Hashing is used for Image Similarity Search in Elasticsearch.",
    "The Convolutional Neural Network (CNN) encodes each image.",
    "Deep Hashing produces short binary codes. Deep Hashing is efficient.",
]


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

def test_backend_is_pinned_in_tests():
    assert backend_name() == "regex"


def test_explicit_backend_is_honoured(monkeypatch):
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "regex")
    assert backend_name() == "regex"


def test_invalid_backend_value_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "nonsense")
    assert backend_name() in VALID_BACKENDS - {"auto"}


def test_unavailable_backend_degrades_rather_than_raising(monkeypatch):
    """Asking for spaCy without spaCy installed must cost quality, not the run."""
    monkeypatch.setattr(nlp, "_spacy_pipeline", lambda: None)
    nlp._resolve_backend.cache_clear()
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "spacy")
    assert backend_name() == "regex"
    nlp._resolve_backend.cache_clear()


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_normalise_strips_leading_and_trailing_stopwords():
    assert normalise_phrase("the concept space of") == "concept space"
    assert normalise_phrase("  Concept   Space.  ") == "Concept Space"


def test_acceptable_phrase_rejects_junk():
    assert not is_acceptable_phrase("")
    assert not is_acceptable_phrase("the of and")
    assert not is_acceptable_phrase("a b c d e f g")     # too many tokens
    assert not is_acceptable_phrase("xy")                # too short, not an acronym
    assert not is_acceptable_phrase("123 456")           # not enough letters


def test_acceptable_phrase_allows_acronyms():
    assert is_acceptable_phrase("CNN")
    assert is_acceptable_phrase("concept space")


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def test_noun_phrases_respect_min_count():
    phrases = noun_phrases(SAMPLE, min_count=2)
    assert all(p.count >= 2 for p in phrases)
    assert any("Deep Hashing" in p.text for p in phrases)


def test_noun_phrases_respect_max_tokens():
    for p in noun_phrases(SAMPLE, min_count=1, max_tokens=3):
        assert len(p.text.split()) <= 3


def test_acronyms_are_found_by_the_fallback():
    entities = named_entities(SAMPLE, min_count=1)
    assert any(e.text == "CNN" for e in entities)


def test_extraction_is_deterministic():
    assert ([(p.text, p.count) for p in noun_phrases(SAMPLE, min_count=1)]
            == [(p.text, p.count) for p in noun_phrases(SAMPLE, min_count=1)])


def test_empty_input_is_handled():
    assert noun_phrases([]) == []
    assert named_entities([]) == []
    assert analyse([]).phrases == {}


def test_blank_texts_are_skipped():
    assert noun_phrases(["", "   ", "\n"], min_count=1) == []


def test_one_pass_serves_both_consumers():
    """Noun phrases and entities must come from the same analysis, not two runs."""
    result = analyse(SAMPLE)
    assert result.phrases
    assert result.entities
    assert result.backend == "regex"


def test_analysis_is_cached_across_calls():
    a = analyse(SAMPLE)
    b = analyse(list(SAMPLE))
    assert a is b


def test_word_frequencies_are_lowercased():
    freqs = word_frequencies(["Deep deep DEEP hashing"])
    assert freqs["deep"] == 3
    assert freqs["hashing"] == 1


# --------------------------------------------------------------------------
# Neural tiers, when present
# --------------------------------------------------------------------------

@pytest.mark.skipif(nlp._stanza_pipeline() is None, reason="stanza models absent")
def test_stanza_tier_produces_phrases(monkeypatch):
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "stanza")
    nlp._resolve_backend.cache_clear()
    try:
        assert backend_name() == "stanza"
        result = analyse(tuple(SAMPLE))
        assert result.backend == "stanza"
        assert result.phrases
    finally:
        nlp._resolve_backend.cache_clear()


@pytest.mark.skipif(nlp._spacy_pipeline() is None, reason="spaCy models absent")
def test_spacy_tier_produces_phrases(monkeypatch):
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "spacy")
    nlp._resolve_backend.cache_clear()
    try:
        assert backend_name() == "spacy"
        assert analyse(tuple(SAMPLE)).backend == "spacy"
    finally:
        nlp._resolve_backend.cache_clear()
