"""The seven quality metrics and the weighted sum."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from conceptdrill.scoring.metrics import (DEFAULT_WEIGHTS, METRICS,
                                          ScoringContext, _coverage_curve,
                                          _entropy, citation_importance,
                                          coverage, embedding_variance,
                                          information_gain, purity, reusability,
                                          structural)
from conceptdrill.scoring.scorer import QualityScorer
from conceptdrill.types import Candidate


def make_context(sims: list[float], sections: list, *,
                 theta: float = 0.5, **kw) -> ScoringContext:
    """A context whose similarities are dictated directly.

    Metrics are tested against known similarity rows rather than real
    embeddings, so a metric's behaviour is pinned independently of the embedder.
    """
    n = len(sims)
    ctx = ScoringContext(
        block_embeddings=np.eye(n, dtype=np.float32),
        block_sections=sections,
        candidate_embeddings=np.zeros((1, n), dtype=np.float32),
        candidate_index={"c": 0},
        theta=theta,
        word_frequencies=kw.pop("word_frequencies", Counter()),
        max_word_frequency=kw.pop("max_word_frequency", 1),
        **kw,
    )
    ctx._sims["c"] = np.array(sims, dtype=np.float32)
    return ctx


CAND = Candidate(name="c", source="heading")


# --------------------------------------------------------------------------
# 2.1 Structural
# --------------------------------------------------------------------------

def test_structural_reflects_provenance():
    ctx = make_context([0.9], ["s1"])
    assert structural(Candidate(name="c", source="glossary"), ctx) == 1.0
    assert structural(Candidate(name="c", source="ner"), ctx) == 0.3


# --------------------------------------------------------------------------
# 2.2 Coverage
# --------------------------------------------------------------------------

def test_coverage_curve_matches_the_spec():
    assert _coverage_curve(0.0) == pytest.approx(0.1)
    assert _coverage_curve(0.01) == pytest.approx(0.1)
    assert _coverage_curve(0.3) == pytest.approx(1.0)
    assert _coverage_curve(0.9) == pytest.approx(0.3)
    assert _coverage_curve(1.0) == pytest.approx(0.3)


def test_coverage_curve_peaks_on_the_plateau():
    """Matching a fifth to two fifths of the document is the sweet spot."""
    plateau = _coverage_curve(0.3)
    assert plateau > _coverage_curve(0.05)
    assert plateau > _coverage_curve(0.95)


def test_coverage_penalises_a_concept_matching_everything():
    ctx = make_context([0.9] * 10, ["s1"] * 10)
    assert coverage(CAND, ctx) == pytest.approx(0.3)


def test_coverage_penalises_a_concept_matching_nothing():
    ctx = make_context([0.1] * 10, ["s1"] * 10)
    assert coverage(CAND, ctx) == pytest.approx(0.1)


# --------------------------------------------------------------------------
# 2.3 Purity
# --------------------------------------------------------------------------

def test_purity_low_when_confined_to_one_section():
    ctx = make_context([0.9, 0.9, 0.1, 0.1], ["s1", "s1", "s2", "s2"])
    assert purity(CAND, ctx) == pytest.approx(0.0)


def test_purity_high_when_spread_across_sections():
    ctx = make_context([0.9] * 4, ["s1", "s2", "s3", "s4"])
    assert purity(CAND, ctx) == pytest.approx(0.75)


def test_purity_is_zero_without_evidence():
    """No matches is not the same as a perfectly pure concept."""
    ctx = make_context([0.1, 0.1], ["s1", "s2"])
    assert purity(CAND, ctx) == 0.0


# --------------------------------------------------------------------------
# 2.4 Information gain
# --------------------------------------------------------------------------

def test_entropy_of_uniform_labels():
    assert _entropy(["a", "b"]) == pytest.approx(1.0)
    assert _entropy(["a", "a"]) == pytest.approx(0.0)


def test_information_gain_is_maximal_for_a_clean_split():
    ctx = make_context([0.9, 0.9, 0.1, 0.1], ["s1", "s1", "s2", "s2"])
    assert information_gain(CAND, ctx) == pytest.approx(1.0)


def test_information_gain_is_zero_for_a_useless_split():
    ctx = make_context([0.9, 0.1, 0.9, 0.1], ["s1", "s1", "s2", "s2"])
    assert information_gain(CAND, ctx) == pytest.approx(0.0)


def test_information_gain_is_neutral_with_one_section():
    """No entropy to reduce means the metric has nothing to say — neutral is the
    honest answer, not zero."""
    ctx = make_context([0.9, 0.1], ["s1", "s1"])
    assert information_gain(CAND, ctx) == 0.5


def test_information_gain_zero_when_split_is_degenerate():
    ctx = make_context([0.9, 0.9], ["s1", "s2"])
    assert information_gain(CAND, ctx) == 0.0


# --------------------------------------------------------------------------
# 2.5 Embedding variance
# --------------------------------------------------------------------------

def test_variance_neutral_with_fewer_than_two_matches():
    ctx = make_context([0.9, 0.1], ["s1", "s2"])
    assert embedding_variance(CAND, ctx) == 0.5


def test_variance_rewards_a_coherent_cluster():
    ctx = ScoringContext(
        block_embeddings=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], np.float32),
        block_sections=["s1", "s1", "s2"],
        candidate_embeddings=np.zeros((1, 2), np.float32),
        candidate_index={"c": 0},
    )
    ctx._sims["c"] = np.array([0.9, 0.9, 0.9], np.float32)
    assert embedding_variance(CAND, ctx) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 2.6 Citation importance
# --------------------------------------------------------------------------

def test_citation_importance_neutral_for_non_bibliography():
    ctx = make_context([0.9], ["s1"])
    assert citation_importance(Candidate(name="c", source="heading"), ctx) == 0.5


def test_citation_importance_neutral_when_count_is_absent():
    """PDF-parsed references carry no counts. Inventing one would be worse than
    declining to score."""
    ctx = make_context([0.9], ["s1"])
    cand = Candidate(name="c", source="bibliography", metadata={"year": 2020})
    assert citation_importance(cand, ctx) == 0.5


def test_citation_importance_scales_with_count():
    ctx = make_context([0.9], ["s1"])
    low = citation_importance(
        Candidate(name="c", source="bibliography",
                  metadata={"citations": 10, "year": 2024}), ctx)
    high = citation_importance(
        Candidate(name="c", source="bibliography",
                  metadata={"citations": 100000, "year": 2024}), ctx)
    assert high > low
    assert high == pytest.approx(1.0)


def test_citation_importance_applies_recency_factor():
    ctx = make_context([0.9], ["s1"])
    recent = citation_importance(
        Candidate(name="c", source="bibliography",
                  metadata={"citations": 1000, "year": 2025}), ctx)
    old = citation_importance(
        Candidate(name="c", source="bibliography",
                  metadata={"citations": 1000, "year": 1990}), ctx)
    assert recent > old
    assert old == pytest.approx(recent * 0.8)


# --------------------------------------------------------------------------
# 2.7 Reusability
# --------------------------------------------------------------------------

def test_reusability_penalises_common_words():
    ctx = make_context([0.9], ["s1"],
                       word_frequencies=Counter({"the": 1000, "hashing": 3}),
                       max_word_frequency=1000)
    common = reusability(Candidate(name="the", source="nounphrase"), ctx)
    rare = reusability(Candidate(name="hashing", source="nounphrase"), ctx)
    assert rare > common


def test_reusability_rewards_multiword_phrases():
    ctx = make_context([0.9], ["s1"],
                       word_frequencies=Counter({"graph": 5, "neural": 5,
                                                 "network": 5}),
                       max_word_frequency=100)
    single = reusability(Candidate(name="graph", source="nounphrase"), ctx)
    phrase = reusability(Candidate(name="graph neural network",
                                   source="nounphrase"), ctx)
    assert phrase > single


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_metric_has_a_weight():
    assert set(DEFAULT_WEIGHTS) == set(METRICS)


def test_scores_are_bounded(mock_document, embedder):
    from conceptdrill.candidates import generate_candidates
    scorer = QualityScorer()
    scored, _ = scorer.score(mock_document,
                             generate_candidates(mock_document), embedder)
    assert scored
    for item in scored:
        assert 0.0 <= item.score <= 1.0
        for value in item.metrics.values():
            assert 0.0 <= value <= 1.0


def test_scored_output_is_sorted_by_quality(mock_document, embedder):
    from conceptdrill.candidates import generate_candidates
    scored, _ = QualityScorer().score(
        mock_document, generate_candidates(mock_document), embedder)
    scores = [s.score for s in scored]
    assert scores == sorted(scores, reverse=True)


def test_weights_are_normalised_after_override():
    scorer = QualityScorer(weights={"structural": 10.0})
    assert sum(scorer.weights.values()) == pytest.approx(1.0)
    assert scorer.weights["structural"] > 0.5


def test_unknown_weight_is_rejected():
    with pytest.raises(ValueError, match="unknown metrics"):
        QualityScorer(weights={"nonsense": 1.0})


def test_a_failing_metric_does_not_sink_the_run():
    def explode(candidate, ctx):
        raise RuntimeError("boom")

    scorer = QualityScorer(metrics={**METRICS, "structural": explode})
    ctx = make_context([0.9], ["s1"])
    result = scorer.score_one(CAND, ctx)
    assert result.metrics["structural"] == 0.5


def test_metrics_are_individually_replaceable():
    """The spec asks that any scoring function be swappable."""
    scorer = QualityScorer(metrics={**METRICS,
                                    "coverage": lambda c, ctx: 1.0})
    ctx = make_context([0.1] * 10, ["s1"] * 10)
    assert scorer.score_one(CAND, ctx).metrics["coverage"] == 1.0
