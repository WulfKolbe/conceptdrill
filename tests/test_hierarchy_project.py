"""Unit tests for CES projection.

Vectors are hand-built so every similarity is predictable without a model.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.hierarchy.basis import ConceptBasis
from conceptdrill.hierarchy.project import (ConceptHit, DEFAULT_TOP_K,
                                            SentenceProjection,
                                            project_sentences, project_vectors,
                                            projection_stats)
from conceptdrill.hierarchy.sentences import Sentence


E1, E2, E3 = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


class FixedEmbedder:
    """Maps each text to a preset vector, so similarities are known exactly."""
    name, revision, dim = "fixed", "v1", 3

    def __init__(self, table):
        self.table = table

    def encode(self, texts):
        return np.vstack([self.table.get(t, E3) for t in texts])


@pytest.fixture
def basis():
    b = ConceptBasis(tau=2.0)          # never merge; keep rows distinct
    b.integrate(2, "alpha concept", E1)
    b.integrate(2, "beta concept", E2)
    return b


def sent(sid, text, section="s1"):
    return Sentence(id=sid, text=text, section_id=section, source_id="p1")


# --------------------------------------------------------------------------
# project_vectors
# --------------------------------------------------------------------------

def test_projection_shape_is_sentences_by_rows(basis):
    assert project_vectors(np.vstack([E1, E2, E3]), basis).shape == (3, 2)


def test_a_matching_sentence_scores_one(basis):
    scores = project_vectors(E1.reshape(1, -1), basis)
    assert float(scores.max()) == pytest.approx(1.0)


def test_an_orthogonal_sentence_scores_zero(basis):
    assert np.allclose(project_vectors(E3.reshape(1, -1), basis), 0.0)


def test_input_is_normalised_before_projection(basis):
    """An unnormalised vector must not inflate every coordinate."""
    scores = project_vectors((E1 * 17.0).reshape(1, -1), basis)
    assert float(scores.max()) == pytest.approx(1.0)


def test_a_one_dimensional_input_is_accepted(basis):
    assert project_vectors(E1, basis).shape == (1, 2)


def test_dimension_mismatch_is_reported_clearly(basis):
    with pytest.raises(ValueError, match="different embedding model"):
        project_vectors(np.ones((1, 7)), basis)


def test_an_empty_basis_projects_to_no_coordinates():
    assert project_vectors(np.vstack([E1]), ConceptBasis()).shape == (1, 0)


def test_projection_is_float64(basis):
    """A CES coordinate gets thresholded and ranked downstream."""
    got = project_vectors(np.vstack([E1]).astype(np.float32), basis)
    assert got.dtype == np.float64


# --------------------------------------------------------------------------
# project_sentences
# --------------------------------------------------------------------------

def test_every_sentence_is_projected(basis):
    emb = FixedEmbedder({"a": E1, "b": E2})
    got = project_sentences([sent("s#0", "a"), sent("s#1", "b")], basis, emb)
    assert [p.sentence_id for p in got] == ["s#0", "s#1"]


def test_top_concept_is_the_nearest_row(basis):
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], basis, emb)[0]
    assert got.best.label == "alpha concept"
    assert got.best.similarity == pytest.approx(1.0)


def test_hits_are_ranked(basis):
    emb = FixedEmbedder({"a": E1})
    hits = project_sentences([sent("s#0", "a")], basis, emb)[0].top_concepts
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].similarity >= hits[1].similarity


def test_top_k_is_capped_by_the_basis_size(basis):
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], basis, emb, top_k=99)[0]
    assert len(got.top_concepts) == 2


def test_provenance_travels_with_the_projection(basis):
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a", section="sec9")], basis, emb)[0]
    assert got.section_id == "sec9" and got.source_id == "p1"


def test_basis_version_is_recorded(basis):
    """Coordinate 4 means whatever row 4 was; the version makes that checkable."""
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], basis, emb)[0]
    assert got.basis_version == basis.basis_version()


def test_embedding_model_is_recorded(basis):
    """A vector from another encoder is not in the same space at all."""
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], basis, emb)[0]
    assert got.embedding_model == "fixed" and got.embedding_revision == "v1"


def test_vectors_are_not_stored_by_default(basis):
    emb = FixedEmbedder({"a": E1})
    assert project_sentences([sent("s#0", "a")], basis, emb)[0].vector == ()


def test_vectors_can_be_stored(basis):
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], basis, emb, store_vectors=True)[0]
    assert len(got.vector) == 2


def test_batching_does_not_change_the_result(basis):
    emb = FixedEmbedder({"a": E1, "b": E2})
    sents = [sent(f"s#{i}", "a" if i % 2 else "b") for i in range(6)]
    one = project_sentences(sents, basis, emb, batch_size=100)
    many = project_sentences(sents, basis, emb, batch_size=2)
    assert [p.to_dict() for p in one] == [p.to_dict() for p in many]


def test_no_sentences_yields_no_projections(basis):
    assert project_sentences([], basis, FixedEmbedder({})) == []


def test_an_empty_basis_still_projects_the_sentences():
    """The sentences are real; a caller who built no basis should see that."""
    emb = FixedEmbedder({"a": E1})
    got = project_sentences([sent("s#0", "a")], ConceptBasis(), emb)
    assert len(got) == 1 and got[0].top_concepts == ()
    assert got[0].best is None


# --------------------------------------------------------------------------
# Margin
# --------------------------------------------------------------------------

def test_margin_separates_confidence_from_ambiguity(basis):
    """A high top-1 with a near-zero margin is ambiguity, not confidence."""
    emb = FixedEmbedder({"tied": (E1 + E2) / np.sqrt(2)})
    got = project_sentences([sent("s#0", "tied")], basis, emb)[0]
    assert got.best.similarity > 0.7
    assert got.margin == pytest.approx(0.0, abs=1e-6)


def test_margin_is_large_for_an_unambiguous_sentence(basis):
    emb = FixedEmbedder({"a": E1})
    assert project_sentences([sent("s#0", "a")], basis, emb)[0].margin > 0.9


def test_margin_is_zero_with_a_single_concept():
    b = ConceptBasis()
    b.integrate(2, "only", E1)
    emb = FixedEmbedder({"a": E1})
    assert project_sentences([sent("s#0", "a")], b, emb)[0].margin == 0.0


# --------------------------------------------------------------------------
# Serialisation and stats
# --------------------------------------------------------------------------

def test_to_dict_omits_the_vector_by_default(basis):
    emb = FixedEmbedder({"a": E1})
    d = project_sentences([sent("s#0", "a")], basis, emb, store_vectors=True)[0].to_dict()
    assert "vector" not in d and d["top_concepts"]


def test_to_dict_can_include_the_vector(basis):
    emb = FixedEmbedder({"a": E1})
    p = project_sentences([sent("s#0", "a")], basis, emb, store_vectors=True)[0]
    assert "vector" in p.to_dict(include_vector=True)


def test_stats_report_the_basis_version(basis):
    emb = FixedEmbedder({"a": E1, "b": E2})
    st = projection_stats(project_sentences(
        [sent("s#0", "a"), sent("s#1", "b")], basis, emb))
    assert st["basis_versions"] == [basis.basis_version()]


def test_stats_report_concept_usage(basis):
    """A row that never wins is dead weight; one that wins everything is a sink."""
    emb = FixedEmbedder({"a": E1})
    st = projection_stats(project_sentences(
        [sent("s#0", "a"), sent("s#1", "a")], basis, emb))
    assert st["concepts_used"] == 1
    assert st["most_frequent_concept"] == "alpha concept"


def test_stats_of_an_empty_projection():
    assert projection_stats([])["sentences"] == 0
