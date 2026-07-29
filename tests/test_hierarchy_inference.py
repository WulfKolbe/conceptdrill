"""Unit tests for querying and the query log."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conceptdrill.hierarchy.basis import ConceptBasis
from conceptdrill.hierarchy.inference import (QueryEngine, QueryLog,
                                              QueryResult, _cosine_rows)

E1, E2, E3 = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))


class FixedEmbedder:
    name, revision, dim = "fixed", "v1", 3

    def __init__(self, table=None):
        self.table = table or {}

    def encode(self, texts):
        return np.vstack([self.table.get(t, E3) for t in texts])


@pytest.fixture
def basis():
    b = ConceptBasis(tau=2.0)
    b.integrate_document("doc1", [(2, "alpha concept", E1),
                                  (2, "beta concept", E2)])
    return b


def records_and_vectors():
    recs = [
        {"sentence_id": "s0", "text": "about alpha", "section_id": "sec1",
         "document": "doc1",
         "top_concepts": [{"label": "alpha concept", "similarity": 0.9}]},
        {"sentence_id": "s1", "text": "about beta", "section_id": "sec2",
         "document": "doc2",
         "top_concepts": [{"label": "beta concept", "similarity": 0.9}]},
    ]
    return recs, np.array([[1.0, 0.0], [0.0, 1.0]])


# --------------------------------------------------------------------------
# Categories — the query's own coordinates
# --------------------------------------------------------------------------

def test_query_finds_its_concept(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    got = eng.query("q")
    assert got.best.label == "alpha concept"
    assert got.best.similarity == pytest.approx(1.0)


def test_categories_are_ranked(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    cats = eng.query("q").categories
    assert [c.rank for c in cats] == [1, 2]
    assert cats[0].similarity >= cats[1].similarity


def test_top_k_is_capped_by_the_basis(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    assert len(eng.query("q", top_concepts=99).categories) == 2


def test_an_unrelated_query_scores_low(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E3}))
    assert eng.query("q").best.similarity == pytest.approx(0.0, abs=1e-9)


def test_basis_version_is_recorded(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    assert eng.query("q").basis_version == basis.basis_version()


def test_embedding_model_is_recorded(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    assert eng.query("q").embedding_model == "fixed"


def test_margin_distinguishes_ambiguity(basis):
    tied = (E1 + E2) / np.sqrt(2)
    eng = QueryEngine(basis, FixedEmbedder({"q": tied}))
    assert eng.query("q").margin == pytest.approx(0.0, abs=1e-6)


def test_an_empty_basis_yields_no_categories():
    eng = QueryEngine(ConceptBasis(), FixedEmbedder({"q": E1}))
    got = eng.query("q")
    assert got.categories == () and got.best is None


# --------------------------------------------------------------------------
# The annotated query — step 7's stated output
# --------------------------------------------------------------------------

def test_the_query_comes_back_with_its_categories_injected(basis):
    eng = QueryEngine(basis, FixedEmbedder({"deep hashing": E1}))
    got = eng.query("deep hashing", top_concepts=1)
    assert got.annotated == "deep hashing  [concepts: alpha concept]"


def test_annotation_degrades_to_the_bare_query(basis):
    eng = QueryEngine(ConceptBasis(), FixedEmbedder({"q": E1}))
    assert eng.query("q").annotated == "q"


# --------------------------------------------------------------------------
# Neighbours — where the corpus discusses it
# --------------------------------------------------------------------------

def test_nearest_sentence_is_found(basis):
    recs, vecs = records_and_vectors()
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}), records=recs, vectors=vecs)
    got = eng.query("q")
    assert got.neighbours[0].sentence_id == "s0"


def test_neighbours_are_ranked(basis):
    recs, vecs = records_and_vectors()
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}), records=recs, vectors=vecs)
    sims = [n.similarity for n in eng.query("q").neighbours]
    assert sims == sorted(sims, reverse=True)


def test_neighbours_carry_their_document(basis):
    recs, vecs = records_and_vectors()
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}), records=recs, vectors=vecs)
    assert eng.query("q").neighbours[0].document == "doc1"


def test_shared_concepts_explain_the_match(basis):
    """Raw cosine cannot say why two things matched; named coordinates can."""
    recs, vecs = records_and_vectors()
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}), records=recs, vectors=vecs)
    assert "alpha concept" in eng.query("q").neighbours[0].shared_concepts


def test_search_uses_cosine_not_magnitude(basis):
    """A sentence matching every concept weakly must not outrank one matching a
    single concept strongly, purely because its vector is longer."""
    recs = [{"sentence_id": "long", "text": "weak on everything",
             "top_concepts": []},
            {"sentence_id": "sharp", "text": "strong on alpha",
             "top_concepts": []}]
    vecs = np.array([[5.0, 5.0], [1.0, 0.0]])
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}), records=recs, vectors=vecs)
    assert eng.query("q").neighbours[0].sentence_id == "sharp"


def test_no_index_means_no_neighbours(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    assert eng.query("q").neighbours == () and not eng.has_index


def test_misaligned_records_and_vectors_are_refused(basis):
    recs, vecs = records_and_vectors()
    with pytest.raises(ValueError, match="records"):
        QueryEngine(basis, FixedEmbedder(), records=recs, vectors=vecs[:1])


def test_cosine_rows_handles_a_zero_row():
    sims = _cosine_rows(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([1.0, 0.0]))
    assert np.isfinite(sims).all()


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def test_result_serialises_without_the_vector(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    d = eng.query("q", store_vector=True).to_dict()
    assert "vector" not in d and d["categories"]


def test_result_can_include_the_vector(basis):
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    d = eng.query("q", store_vector=True).to_dict(include_vector=True)
    assert len(d["vector"]) == 2


# --------------------------------------------------------------------------
# Step 8: the query log
# --------------------------------------------------------------------------

def test_a_query_is_logged(tmp_path, basis):
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"))
    assert len(log) == 1


def test_the_log_is_append_only(tmp_path, basis):
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1, "r": E2}))
    log.append(eng.query("q"))
    log.append(eng.query("r"))
    assert [e["query"] for e in log.read()] == ["q", "r"]


def test_the_answer_is_stored_beside_the_query(tmp_path, basis):
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"), answer="because alpha")
    assert log.read()[0]["answer"] == "because alpha"


def test_the_query_vector_is_stored(tmp_path, basis):
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q", store_vector=True))
    assert len(log.read()[0]["vector"]) == 2


def test_query_id_is_stable_for_the_same_query_and_basis(tmp_path, basis):
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"))
    log.append(eng.query("q"))
    ids = [e["query_id"] for e in log.read()]
    assert ids[0] == ids[1]


def test_entries_can_be_filtered_by_basis_version(tmp_path, basis):
    """A log spanning a rebuild is still readable; the entries simply are not
    comparable across the boundary."""
    log = QueryLog(tmp_path / "queries.jsonl")
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"))
    version = basis.basis_version()
    assert len(log.read(basis_version=version)) == 1
    assert log.read(basis_version="other") == []


def test_a_malformed_line_does_not_break_the_log(tmp_path, basis):
    """A log is diagnostic; one bad entry must not make the rest unreadable."""
    path = tmp_path / "queries.jsonl"
    log = QueryLog(path)
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n")
    log.append(eng.query("q"))
    assert len(log.read()) == 2


def test_reading_a_missing_log_is_empty(tmp_path):
    assert QueryLog(tmp_path / "absent.jsonl").read() == []


def test_the_log_is_json_lines(tmp_path, basis):
    """One truncated final line costs one entry, not the file."""
    path = tmp_path / "queries.jsonl"
    log = QueryLog(path)
    eng = QueryEngine(basis, FixedEmbedder({"q": E1}))
    log.append(eng.query("q"))
    log.append(eng.query("q"))
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert all(json.loads(l) for l in lines)
