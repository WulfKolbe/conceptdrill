"""Unit tests for the self-adapting basis.

Vectors are hand-built and orthogonal or deliberately close, so every merge
decision is predictable without an embedding model in the loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.hierarchy.basis import (ConceptBasis, DEFAULT_TAU, BasisRow,
                                          canonical_label, make_row_id, merge,
                                          new_row)


def vec(*values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


E1, E2, E3 = vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)
NEAR_E1 = vec(0.99, 0.14, 0.0)      # cosine ~0.99 with E1


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_canonical_label_ignores_case_and_whitespace():
    assert canonical_label("  Concept   Space ") == canonical_label("concept space")


def test_row_id_is_stable_for_the_same_concept():
    assert make_row_id(2, "Concept Space") == make_row_id(2, "concept  space")


def test_row_id_separates_levels():
    """A chapter concept and a subsection concept are different things."""
    assert make_row_id(2, "Method") != make_row_id(3, "Method")


def test_row_id_separates_different_labels():
    assert make_row_id(2, "A") != make_row_id(2, "B")


# --------------------------------------------------------------------------
# Rows and merging
# --------------------------------------------------------------------------

def test_new_row_is_unit_norm():
    row = new_row(2, "L", vec(3, 4, 0))
    assert np.isclose(np.linalg.norm(row.vector), 1.0)


def test_new_row_records_its_origin():
    row = new_row(2, "Concept", E1, document="doc1")
    assert row.support == 1
    assert row.documents == ("doc1",) and row.merged_labels == ("Concept",)


def test_merge_increments_support():
    assert merge(new_row(2, "L", E1), E1).support == 2


def test_merge_moves_the_vector_toward_the_newcomer():
    row = merge(new_row(2, "L", E1), E2)
    assert 0.0 < float(row.vector @ E2) < 1.0


def test_merge_keeps_the_vector_unit_norm():
    row = merge(merge(new_row(2, "L", E1), E2), E3)
    assert np.isclose(np.linalg.norm(row.vector), 1.0)


def test_merge_is_a_running_mean_not_a_replacement():
    """After one merge the row must sit between its two inputs, nearer neither
    exclusively -- a replacement would land exactly on the newcomer."""
    row = merge(new_row(2, "L", E1), E2)
    assert float(row.vector @ E1) == pytest.approx(float(row.vector @ E2))


def test_later_merges_move_the_vector_less():
    """Support is weight: the tenth contributor should not overturn nine."""
    early = merge(new_row(2, "L", E1), E2)
    heavy = BasisRow("r", 2, "L", E1, support=50)
    late = merge(heavy, E2)
    assert float(late.vector @ E1) > float(early.vector @ E1)


def test_merge_never_changes_identity():
    """An id that drifted on every merge would invalidate every CES vector
    already stored against it."""
    row = new_row(2, "Concept Space", E1)
    assert merge(row, E2, label="Something Else").row_id == row.row_id
    assert merge(row, E2, label="Something Else").label == "Concept Space"


def test_merge_records_the_contributing_label_and_document():
    row = merge(new_row(2, "A", E1, "doc1"), E2, label="B", document="doc2")
    assert row.merged_labels == ("A", "B")
    assert row.documents == ("doc1", "doc2")


def test_merge_does_not_duplicate_a_repeat_document():
    row = merge(new_row(2, "A", E1, "doc1"), E2, label="B", document="doc1")
    assert row.documents == ("doc1",)


# --------------------------------------------------------------------------
# The adaptive core
# --------------------------------------------------------------------------

def test_first_candidate_is_always_added():
    b = ConceptBasis()
    assert b.integrate(2, "Concept", E1).action == "added"
    assert len(b) == 1


def test_a_distant_candidate_becomes_a_new_row():
    b = ConceptBasis()
    b.integrate(2, "A", E1)
    assert b.integrate(2, "B", E2).action == "merged" or len(b) == 2
    assert len(b) == 2


def test_a_close_candidate_merges():
    b = ConceptBasis(tau=0.85)
    b.integrate(2, "A", E1)
    result = b.integrate(2, "B", NEAR_E1)
    assert result.action == "merged"
    assert len(b) == 1 and b.rows[result.row_id].support == 2


def test_tau_controls_the_decision():
    close = ConceptBasis(tau=0.99999)
    close.integrate(2, "A", E1)
    close.integrate(2, "B", NEAR_E1)
    assert len(close) == 2, "a strict tau must keep them apart"

    loose = ConceptBasis(tau=0.5)
    loose.integrate(2, "A", E1)
    loose.integrate(2, "B", NEAR_E1)
    assert len(loose) == 1, "a loose tau must merge them"


def test_tau_can_be_overridden_per_call():
    b = ConceptBasis(tau=0.99999)
    b.integrate(2, "A", E1)
    assert b.integrate(2, "B", NEAR_E1, tau=0.5).action == "merged"


def test_levels_do_not_compete():
    """Letting a chapter concept merge with a subsection concept would flatten
    the hierarchy the basis exists to represent."""
    b = ConceptBasis(tau=0.5)
    b.integrate(2, "A", E1)
    assert b.integrate(3, "A-sub", E1).action == "added"
    assert len(b) == 2


def test_identical_label_merges_even_when_vectors_drifted():
    """Same canonical label at the same level IS the same concept."""
    b = ConceptBasis(tau=0.99999)
    b.integrate(2, "Concept Space", E1)
    result = b.integrate(2, "concept  space", E2)
    assert result.action == "merged" and len(b) == 1


def test_a_zero_vector_is_skipped_not_added():
    """It has no direction to compare, and would sit in the basis forever."""
    b = ConceptBasis()
    assert b.integrate(2, "Empty", vec(0, 0, 0)).action == "skipped"
    assert len(b) == 0


def test_similarity_is_reported():
    b = ConceptBasis(tau=0.5)
    b.integrate(2, "A", E1)
    assert b.integrate(2, "B", NEAR_E1).similarity > 0.9


# --------------------------------------------------------------------------
# Document integration
# --------------------------------------------------------------------------

def test_document_integration_records_the_document():
    b = ConceptBasis()
    b.integrate_document("doc1", [(2, "A", E1), (2, "B", E2)])
    assert b.document_order == ("doc1",)
    assert all("doc1" in r.documents for r in b.rows.values())


def test_candidate_order_within_a_document_is_normalised():
    """Otherwise the basis depends on span ordering, which nobody intended."""
    a = ConceptBasis(tau=0.9)
    a.integrate_document("d", [(2, "B", E2), (2, "A", E1)])
    b = ConceptBasis(tau=0.9)
    b.integrate_document("d", [(2, "A", E1), (2, "B", E2)])
    assert a.row_ids() == b.row_ids()


def test_a_repeated_document_is_not_listed_twice():
    b = ConceptBasis()
    b.integrate_document("doc1", [(2, "A", E1)])
    b.integrate_document("doc1", [(2, "A", E1)])
    assert b.document_order == ("doc1",)


def test_shared_concepts_accumulate_support_across_documents():
    b = ConceptBasis(tau=0.85)
    b.integrate_document("doc1", [(2, "Concept", E1)])
    b.integrate_document("doc2", [(2, "Concept", NEAR_E1)])
    row = next(iter(b.rows.values()))
    assert row.support == 2 and set(row.documents) == {"doc1", "doc2"}


# --------------------------------------------------------------------------
# Ordering, identity and version
# --------------------------------------------------------------------------

def test_rows_are_level_major():
    b = ConceptBasis(tau=2.0)
    b.integrate(3, "sub", E1)
    b.integrate(2, "top", E2)
    assert [r.level for r in b.ordered_rows()] == [2, 3]


def test_support_orders_within_a_level():
    b = ConceptBasis(tau=0.85)
    b.integrate(2, "A", E1)
    b.integrate(2, "A2", NEAR_E1)      # merges -> support 2
    b.integrate(2, "B", E2)            # stays at support 1
    assert [r.support for r in b.ordered_rows()] == [2, 1]


def test_label_breaks_ties_totally():
    """Without it, equal-support rows fall back to dict insertion order."""
    b = ConceptBasis(tau=2.0)
    b.integrate(2, "zebra", E1)
    b.integrate(2, "apple", E2)
    assert [r.label for r in b.ordered_rows()] == ["apple", "zebra"]


def test_matrix_rows_match_the_canonical_order():
    b = ConceptBasis(tau=2.0)
    b.integrate(2, "b", E1)
    b.integrate(2, "a", E2)
    matrix = b.matrix()
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix[0], b.ordered_rows()[0].vector)


def test_matrix_rows_are_unit_norm():
    b = ConceptBasis()
    b.integrate(2, "a", vec(3, 4, 0))
    assert np.allclose(np.linalg.norm(b.matrix(), axis=1), 1.0)


def test_matrix_is_float64():
    """A wrong cosine adds a row that should have merged, unrecoverably."""
    b = ConceptBasis()
    b.integrate(2, "a", np.asarray([1, 0, 0], dtype=np.float32))
    assert b.matrix().dtype == np.float64


def test_basis_version_changes_when_a_row_is_added():
    b = ConceptBasis(tau=2.0)
    b.integrate(2, "a", E1)
    before = b.basis_version()
    b.integrate(2, "b", E2)
    assert b.basis_version() != before


def test_basis_version_changes_when_support_reorders_rows():
    """The hazard the version exists for: positions move as a corpus grows."""
    b = ConceptBasis(tau=0.85)
    b.integrate(2, "a", E1)
    b.integrate(2, "b", E2)
    before = b.basis_version()
    b.integrate(2, "b2", vec(0.01, 0.999, 0.0))   # merges into b, support 2
    assert b.basis_version() != before


def test_row_id_survives_reordering():
    b = ConceptBasis(tau=0.85)
    r1 = b.integrate(2, "a", E1).row_id
    b.integrate(2, "b", E2)
    b.integrate(2, "b2", vec(0.01, 0.999, 0.0))
    assert r1 in b.row_ids()


def test_position_is_only_valid_for_one_version():
    b = ConceptBasis(tau=0.85)
    rid = b.integrate(2, "a", E1).row_id
    first = b.position_of(rid)
    b.integrate(2, "b", E2)
    b.integrate(2, "b2", vec(0.01, 0.999, 0.0))
    assert b.position_of(rid) != first


def test_empty_basis_has_an_empty_matrix():
    assert ConceptBasis().matrix().shape == (0, 0)


def test_stats_report_sharing():
    b = ConceptBasis(tau=0.85)
    b.integrate_document("d1", [(2, "A", E1), (2, "B", E2)])
    b.integrate_document("d2", [(2, "A2", NEAR_E1)])
    s = b.stats()
    assert s["rows"] == 2 and s["documents"] == 2
    assert s["shared_across_documents"] == 1
    assert s["singletons"] == 1


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.basis import calibrate, similarity_profile  # noqa: E402


def test_default_tau_is_the_measured_value():
    """0.85 produced zero merges across three related papers; the highest
    cross-document similarity observed was 0.647."""
    assert DEFAULT_TAU == 0.65


def test_profile_separates_within_from_cross():
    a = np.vstack([E1, NEAR_E1])       # a document that repeats itself
    b = np.vstack([E2])
    p = similarity_profile([a, b])
    assert p["within_document"]["max"] > 0.9
    assert p["cross_document"]["max"] < 0.3


def test_profile_handles_a_single_set():
    p = similarity_profile([np.vstack([E1, E2])])
    assert p["within_document"]["n"] == 1
    assert p["cross_document"]["n"] == 0


def test_profile_handles_singleton_sets():
    p = similarity_profile([np.vstack([E1]), np.vstack([E2])])
    assert p["within_document"]["n"] == 0
    assert p["cross_document"]["n"] == 1


def test_calibrate_suggests_from_the_cross_distribution():
    a = np.vstack([E1, E2])
    b = np.vstack([NEAR_E1, E3])
    out = calibrate([a, b])
    assert 0.45 <= out["suggested_tau"] <= 0.95
    assert "cross-document" in out["reason"]


def test_calibrate_reports_its_evidence():
    """A number without its distribution invites the same unexamined 0.85."""
    out = calibrate([np.vstack([E1]), np.vstack([E2])])
    assert "profile" in out and "cross_document" in out["profile"]


def test_calibrate_respects_the_floor():
    """Orthogonal documents must not drive tau to zero."""
    out = calibrate([np.vstack([E1]), np.vstack([E2])])
    assert out["suggested_tau"] >= 0.45


def test_calibrate_respects_the_ceiling():
    out = calibrate([np.vstack([E1]), np.vstack([E1])])
    assert out["suggested_tau"] <= 0.95


def test_calibrate_without_cross_pairs_falls_back():
    out = calibrate([np.vstack([E1, E2])])
    assert out["suggested_tau"] == DEFAULT_TAU
    assert "no cross-document" in out["reason"]
