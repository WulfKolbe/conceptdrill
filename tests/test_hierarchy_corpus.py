"""Unit tests for the corpus store and for inference.

The invariant under test throughout: a CES vector is meaningless without the
basis it was computed against, so a mismatch must be refused, never guessed at.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conceptdrill.hierarchy.basis import ConceptBasis
from conceptdrill.hierarchy.corpus import (BASIS_JSON, BasisMismatch,
                                           CorpusStore)
from conceptdrill.hierarchy.inference import (QueryEngine, QueryLog,
                                              QueryResult)
from conceptdrill.hierarchy.project import ConceptHit, SentenceProjection

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


@pytest.fixture
def store(tmp_path):
    return CorpusStore(tmp_path / "corpus")


def _proj(sid, vector, version, label="alpha concept", doc="doc1"):
    return SentenceProjection(
        sentence_id=sid, text=f"sentence {sid}", span_id="s1",
        source_id="p1", basis_version=version, embedding_model="fixed",
        embedding_revision="v1",
        top_concepts=(ConceptHit("row_a", label, 2, 0.9, 1),),
        vector=tuple(vector))


# --------------------------------------------------------------------------
# Saving and loading the basis
# --------------------------------------------------------------------------

def test_nothing_exists_before_saving(store):
    assert not store.exists()
    assert store.info()["exists"] is False


def test_basis_round_trips(store, basis):
    store.save_basis(basis)
    got = store.load_basis()
    assert len(got) == len(basis)
    assert got.basis_version() == basis.basis_version()


def test_row_content_survives(store, basis):
    store.save_basis(basis)
    got = store.load_basis()
    labels = {r.label for r in got.ordered_rows()}
    assert labels == {"alpha concept", "beta concept"}


def test_vectors_survive(store, basis):
    store.save_basis(basis)
    got = store.load_basis()
    assert np.allclose(got.matrix(), basis.matrix())


def test_tau_and_document_order_survive(store, basis):
    store.save_basis(basis)
    got = store.load_basis()
    assert got.tau == basis.tau
    assert got.document_order == ("doc1",)


def test_support_and_provenance_survive(store):
    b = ConceptBasis(tau=0.5)
    b.integrate_document("d1", [(2, "shared", E1)])
    b.integrate_document("d2", [(2, "shared again", np.array([0.99, 0.14, 0.0]))])
    store.save_basis(b)
    row = store.load_basis().ordered_rows()[0]
    assert row.support == 2 and set(row.documents) == {"d1", "d2"}
    assert len(row.merged_labels) == 2


def test_embedding_model_is_recorded(store, basis):
    store.save_basis(basis, embedding_model="fixed", embedding_revision="v1")
    assert store.info()["embedding_model"] == "fixed"


def test_loading_a_missing_store_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load_basis()


def test_an_empty_basis_round_trips(store):
    store.save_basis(ConceptBasis())
    assert len(store.load_basis()) == 0


def test_no_temporary_files_are_left(store, basis):
    store.save_basis(basis)
    assert list(store.path.glob("*.tmp")) == []


# --------------------------------------------------------------------------
# The mismatch guards
# --------------------------------------------------------------------------

def test_inconsistent_json_and_npz_are_refused(store, basis):
    """A matrix from one basis beside metadata from another."""
    store.save_basis(basis)
    meta = json.loads(store.basis_json.read_text())
    meta["basis_version"] = "tampered"
    store.basis_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BasisMismatch, match="inconsistent"):
        store.load_basis()


def test_an_edited_row_is_detected(store, basis):
    """The rebuilt basis must reproduce its stored version."""
    store.save_basis(basis)
    meta = json.loads(store.basis_json.read_text())
    meta["rows"][0]["label"] = "something else entirely"
    store.basis_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BasisMismatch):
        store.load_basis()


def test_projections_from_another_basis_are_refused(store, basis):
    store.save_basis(basis)
    with pytest.raises(BasisMismatch, match="re-project"):
        store.save_projections([_proj("s0", [1.0, 0.0], "other-version")],
                               basis_version=basis.basis_version())


def test_index_from_another_basis_is_refused(store, basis):
    store.save_basis(basis)
    store.save_projections([_proj("s0", [1.0, 0.0], basis.basis_version())],
                           basis_version=basis.basis_version())
    with pytest.raises(BasisMismatch, match="re-project"):
        store.load_index(basis_version="a-different-version")


# --------------------------------------------------------------------------
# The sentence index
# --------------------------------------------------------------------------

def test_index_round_trips(store, basis):
    v = basis.basis_version()
    store.save_basis(basis)
    store.save_projections([_proj("s0", [1.0, 0.0], v),
                            _proj("s1", [0.0, 1.0], v)], basis_version=v)
    records, vectors = store.load_index(basis_version=v)
    assert [r["sentence_id"] for r in records] == ["s0", "s1"]
    assert vectors.shape == (2, 2)


def test_index_records_carry_provenance(store, basis):
    v = basis.basis_version()
    store.save_basis(basis)
    store.save_projections([_proj("s0", [1.0, 0.0], v)], basis_version=v,
                           document_of=lambda p: "2209.00445")
    records, _ = store.load_index(basis_version=v)
    assert records[0]["document"] == "2209.00445"
    assert records[0]["span_id"] == "s1"


def test_projections_without_vectors_are_skipped(store, basis):
    v = basis.basis_version()
    store.save_basis(basis)
    bare = SentenceProjection("s9", "t", "s1", "p1", v, "fixed", "v1")
    store.save_projections([bare], basis_version=v)
    records, _ = store.load_index(basis_version=v)
    assert records == []


def test_missing_index_reads_as_empty(store, basis):
    store.save_basis(basis)
    records, vectors = store.load_index()
    assert records == [] and vectors.shape[0] == 0


def test_info_reports_the_sentence_count(store, basis):
    v = basis.basis_version()
    store.save_basis(basis)
    store.save_projections([_proj("s0", [1.0, 0.0], v)], basis_version=v)
    assert store.info()["sentences"] == 1


def test_vectors_are_stored_as_float64(store, basis):
    v = basis.basis_version()
    store.save_basis(basis)
    store.save_projections([_proj("s0", [1.0, 0.0], v)], basis_version=v)
    _, vectors = store.load_index(basis_version=v)
    assert vectors.dtype == np.float64
