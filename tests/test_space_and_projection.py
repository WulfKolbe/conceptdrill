"""Concept space construction, the CES matrix form, and projection records."""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.candidates import generate_candidates
from conceptdrill.projection import project_blocks, project_document
from conceptdrill.scoring.scorer import QualityScorer, ScoredCandidate
from conceptdrill.space import (attach_hierarchy, build_space, refine_space,
                                select_concepts)
from conceptdrill.types import Block, Candidate, Concept


@pytest.fixture
def scored(mock_document, embedder):
    cands = generate_candidates(mock_document)
    items, _ = QualityScorer().score(mock_document, cands, embedder)
    return items


@pytest.fixture
def space(scored, mock_document, embedder):
    # Smaller than the candidate pool, so refinement has somewhere to grow.
    return build_space(scored, mock_document, embedder=embedder, max_concepts=12)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def test_selection_respects_the_size_cap(scored, embedder):
    concepts, matrix = select_concepts(scored, embedder=embedder, max_concepts=5)
    assert len(concepts) <= 5
    assert matrix.shape[0] == len(concepts)


def test_selection_takes_the_best_first(scored, embedder):
    concepts, _ = select_concepts(scored, embedder=embedder, max_concepts=5,
                                  diversity_threshold=1.1)
    assert [c.score for c in concepts] == sorted([c.score for c in concepts],
                                                 reverse=True)


def test_diversity_filter_rejects_near_duplicates(embedder):
    """Two identical strings must not both enter the vocabulary."""
    items = [
        ScoredCandidate(Candidate(name="concept space", source="heading"), 0.9, {}),
        ScoredCandidate(Candidate(name="concept space", source="nounphrase"), 0.8, {}),
        ScoredCandidate(Candidate(name="entirely different topic", source="ner"), 0.7, {}),
    ]
    concepts, _ = select_concepts(items, embedder=embedder, max_concepts=10,
                                  diversity_threshold=0.95)
    assert len(concepts) == 2


def test_diversity_threshold_of_one_keeps_everything(embedder):
    items = [
        ScoredCandidate(Candidate(name="same text", source="heading"), 0.9, {}),
        ScoredCandidate(Candidate(name="same text", source="ner"), 0.8, {}),
    ]
    concepts, _ = select_concepts(items, embedder=embedder,
                                  diversity_threshold=1.1)
    assert len(concepts) == 2


def test_min_score_filters_weak_candidates(scored, embedder):
    concepts, _ = select_concepts(scored, embedder=embedder, min_score=0.99)
    assert all(c.score >= 0.99 for c in concepts)


def test_concept_ids_are_stable():
    a = Concept.make_id("Concept Space", "heading")
    b = Concept.make_id("concept  space", "heading")
    assert a == b                                    # normalised
    assert a != Concept.make_id("Concept Space", "ner")


# --------------------------------------------------------------------------
# Hierarchy
# --------------------------------------------------------------------------

def test_hierarchy_links_subsections_to_parents(mock_document):
    concepts = [
        Concept(id="c_method", name="Method", source="heading",
                metadata={"kind": "heading", "section_id": "s2"}),
        Concept(id="c_proj", name="Semantic Projection", source="heading",
                metadata={"kind": "heading", "section_id": "s2a"}),
    ]
    linked = {c.id: c for c in attach_hierarchy(concepts, mock_document)}
    assert linked["c_proj"].parent_id == "c_method"
    assert "c_proj" in linked["c_method"].children


def test_hierarchy_leaves_unanchored_concepts_as_roots(mock_document):
    concepts = [Concept(id="c1", name="Floating", source="bibliography")]
    assert attach_hierarchy(concepts, mock_document)[0].parent_id is None


def test_hierarchy_breaks_cycles(mock_document):
    """A malformed section tree must not produce an infinite parent chain."""
    from conceptdrill.types import Section
    doc = mock_document
    doc.sections["x"] = Section(id="x", title="X", parent_id="y")
    doc.sections["y"] = Section(id="y", title="Y", parent_id="x")
    concepts = [
        Concept(id="cx", name="X", source="heading",
                metadata={"kind": "heading", "section_id": "x"}),
        Concept(id="cy", name="Y", source="heading",
                metadata={"kind": "heading", "section_id": "y"}),
    ]
    linked = attach_hierarchy(concepts, doc)
    assert any(c.parent_id is None for c in linked)


# --------------------------------------------------------------------------
# The CES matrix form
# --------------------------------------------------------------------------

def test_matrix_rows_are_unit_norm(space):
    norms = np.linalg.norm(space.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_project_vector_equals_explicit_cosine(space, embedder):
    """`M @ l` must agree with computing each cosine separately — that identity
    is the whole justification for the matrix shortcut."""
    vec = embedder.encode(["semantic projection of a document object"])[0]
    fast = space.project_vector(vec)
    slow = np.array([float(row @ vec) for row in space.matrix], dtype=np.float32)
    assert np.allclose(fast, slow, atol=1e-5)


def test_project_vector_normalises_its_input(space):
    """An unnormalised vector must not silently inflate every similarity."""
    raw = np.ones(space.dim, dtype=np.float32) * 5.0
    scores = space.project_vector(raw)
    assert np.all(np.abs(scores) <= 1.0 + 1e-5)


def test_dimension_mismatch_is_reported_clearly(space):
    with pytest.raises(ValueError, match="dimension"):
        space.project_vector(np.ones(space.dim + 3, dtype=np.float32))


def test_top_k_is_ranked_and_bounded(space, embedder):
    vec = embedder.encode(["concept scoring and coverage"])[0]
    hits = space.top_k(vec, k=5)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_top_k_clamps_to_space_size(space, embedder):
    vec = embedder.encode(["anything"])[0]
    assert len(space.top_k(vec, k=10_000)) == len(space)


def test_empty_space_projects_to_nothing(embedder):
    from conceptdrill.space import ConceptSpace
    empty = ConceptSpace()
    assert empty.project_vector(np.ones(4, dtype=np.float32)).size == 0
    assert empty.top_k(np.ones(4, dtype=np.float32)) == []


def test_space_info_reports_structure(space):
    info = space.info()
    assert info["size"] == len(space)
    assert info["dimension"] == space.dim
    assert info["similarity_metric"] == "cosine"
    assert "sources" in info and "levels" in info


# --------------------------------------------------------------------------
# Static levels
# --------------------------------------------------------------------------

def test_static_levels_are_concatenated(scored, mock_document, embedder):
    domain = [Concept(id="", name="Machine Learning", source="domain"),
              Concept(id="", name="Graph Theory", source="domain")]
    space = build_space(scored, mock_document, embedder=embedder,
                        max_concepts=10,
                        extra_levels=[("domain", domain)])
    assert len(space.by_level("domain")) == 2
    assert len(space.by_level("document")) <= 10
    assert space.matrix.shape[0] == len(space)


# --------------------------------------------------------------------------
# 3.3 Refinement
# --------------------------------------------------------------------------

def test_refine_grows_the_space(space, scored, embedder):
    before = len(space)
    refined = refine_space(space, "concept scoring and coverage",
                           desired_size=before + 5, embedder=embedder,
                           pool=scored)
    assert len(refined) > before
    assert refined.matrix.shape[0] == len(refined)


def test_refine_is_a_noop_when_already_large_enough(space, embedder):
    assert refine_space(space, "anything", desired_size=1,
                        embedder=embedder) is space


def test_refine_does_not_mutate_the_original(space, scored, embedder):
    before = len(space)
    refine_space(space, "concept scoring", desired_size=before + 3,
                 embedder=embedder, pool=scored)
    assert len(space) == before


# --------------------------------------------------------------------------
# Projection records
# --------------------------------------------------------------------------

def test_projection_records_carry_provenance(space, embedder):
    blocks = [Block(id="b1", type="paragraph", text="semantic projection method")]
    proj = project_blocks(blocks, space, embedder=embedder, top_k=3)[0]
    assert proj.object_id == "b1"
    assert proj.embedding_model == embedder.name
    assert proj.embedding_revision == embedder.revision
    assert proj.similarity_metric == "cosine"
    assert len(proj.concepts) == 3


def test_projection_id_is_stable_and_timestamp_free(space, embedder):
    blocks = [Block(id="b1", type="paragraph", text="text")]
    a = project_blocks(blocks, space, embedder=embedder, created_at="2020-01-01")[0]
    b = project_blocks(blocks, space, embedder=embedder, created_at="2026-07-28")[0]
    assert a.projection_id == b.projection_id


def test_projection_id_changes_with_the_model(space, embedder):
    from conceptdrill.types import Projection
    a = Projection.make_id("o1", "hash", "r1", "src", "cosine", 5)
    b = Projection.make_id("o1", "sentencebert", "r1", "src", "cosine", 5)
    assert a != b


def test_confidence_reports_margin(space, embedder):
    blocks = [Block(id="b1", type="paragraph", text="semantic projection")]
    proj = project_blocks(blocks, space, embedder=embedder, top_k=5)[0]
    top1 = proj.concepts[0].similarity
    top2 = proj.concepts[1].similarity
    assert proj.confidence.top1 == pytest.approx(top1)
    assert proj.confidence.margin == pytest.approx(top1 - top2, abs=1e-5)


def test_embedding_can_be_omitted(space, embedder):
    blocks = [Block(id="b1", type="paragraph", text="text")]
    proj = project_blocks(blocks, space, embedder=embedder,
                          store_embedding=False)[0]
    assert proj.embedding == ()


def test_empty_blocks_are_not_projected(space, embedder):
    blocks = [Block(id="b1", type="paragraph", text="   ")]
    assert project_blocks(blocks, space, embedder=embedder) == []


def test_document_projection_accounts_for_every_block(mock_document, space, embedder):
    projections, skipped = project_document(mock_document, space, embedder=embedder)
    covered = {p.object_id for p in projections} | {s.object_id for s in skipped}
    assert {b.id for b in mock_document.blocks} <= covered


def test_type_filter_records_what_it_excluded(mock_document, space, embedder):
    projections, skipped = project_document(
        mock_document, space, embedder=embedder, types=["paragraph"])
    assert all(p.object_type == "paragraph" for p in projections)
    assert any("filter" in s.reason for s in skipped)
