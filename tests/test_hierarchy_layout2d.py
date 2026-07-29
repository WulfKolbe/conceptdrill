"""Unit tests for 2-D layout.

Deliberately structured inputs, so the expected geometry is known without a
plotting library in the loop.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.hierarchy.layout2d import (Layout, Point, available_backends,
                                             layout, layout_projections,
                                             pca_2d)
from conceptdrill.hierarchy.project import ConceptHit, SentenceProjection


def line(n=10, dim=5):
    """Points strung along one axis: PCA must recover a single direction."""
    data = np.zeros((n, dim))
    data[:, 0] = np.arange(n, dtype=float)
    return data


# --------------------------------------------------------------------------
# pca_2d
# --------------------------------------------------------------------------

def test_shape_is_two_dimensional():
    coords, _, _ = pca_2d(line())
    assert coords.shape == (10, 2)


def test_a_one_dimensional_cloud_loads_onto_one_axis():
    _, explained, _ = pca_2d(line())
    assert explained[0] == pytest.approx(1.0)
    assert explained[1] == pytest.approx(0.0, abs=1e-9)


def test_ordering_along_the_line_is_preserved():
    coords, _, _ = pca_2d(line())
    xs = coords[:, 0]
    assert list(xs) == sorted(xs) or list(xs) == sorted(xs, reverse=True)


def test_output_is_centred():
    coords, _, _ = pca_2d(line())
    assert coords.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_signs_are_pinned_so_runs_are_not_mirrored():
    """SVD may return v or -v; both correct. Without a convention two runs give
    mirrored plots and a reader sees a change that did not happen."""
    a, _, ca = pca_2d(line())
    b, _, cb = pca_2d(line())
    assert np.allclose(a, b) and np.allclose(ca, cb)


def test_sign_convention_is_largest_loading_positive():
    _, _, components = pca_2d(line())
    row = components[0]
    assert row[np.argmax(np.abs(row))] > 0


def test_identical_points_do_not_produce_nan():
    coords, explained, _ = pca_2d(np.ones((5, 4)))
    assert np.isfinite(coords).all() and np.isfinite(explained).all()


def test_a_single_point_is_handled():
    coords, _, _ = pca_2d(np.ones((1, 4)))
    assert coords.shape == (1, 2) and np.isfinite(coords).all()


def test_no_points_is_handled():
    assert pca_2d(np.zeros((0, 4)))[0].shape == (0, 2)


def test_a_one_dimensional_input_is_padded_to_two_axes():
    coords, explained, _ = pca_2d(np.array([[1.0], [2.0], [3.0]]))
    assert coords.shape == (3, 2) and explained.shape == (2,)


def test_pca_is_float64():
    coords, _, _ = pca_2d(line().astype(np.float32))
    assert coords.dtype == np.float64


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def test_every_vector_becomes_a_point():
    got = layout(line(), ids=[f"p{i}" for i in range(10)])
    assert len(got) == 10 and got.points[0].id == "p0"


def test_ids_must_match_the_vectors():
    with pytest.raises(ValueError, match="ids"):
        layout(line(), ids=["only-one"])


def test_pca_is_reported_as_deterministic():
    got = layout(line(), ids=[f"p{i}" for i in range(10)])
    assert got.backend == "pca" and got.deterministic
    assert got.seed is None


def test_explained_variance_is_reported_for_pca():
    got = layout(line(), ids=[f"p{i}" for i in range(10)])
    assert got.explained_variance and got.explained_variance[0] > 0.99


def test_loadings_name_the_driving_dimension():
    """A CES coordinate is the cosine to one named concept, so the loading
    says which concept drives an axis."""
    got = layout(line(), ids=[f"p{i}" for i in range(10)])
    assert got.loadings and got.loadings[0][0] == 0


def test_labels_and_groups_are_carried():
    got = layout(line(3), ids=["a", "b", "c"],
                 labels=["A", "B", "C"], groups=["g1", "g1", "g2"])
    assert got.points[0].label == "A" and got.points[2].group == "g2"


def test_metadata_is_carried_into_the_point():
    got = layout(line(2), ids=["a", "b"],
                 meta=[{"document": "d1"}, {"document": "d2"}])
    assert got.points[0].to_dict()["document"] == "d1"


def test_unavailable_backend_falls_back_to_pca():
    """A visualisation stage must never stop a pipeline."""
    got = layout(line(), ids=[f"p{i}" for i in range(10)], backend="umap")
    assert got.backend in available_backends()
    if "umap" not in available_backends():
        assert got.backend == "pca"


def test_auto_picks_an_available_backend():
    got = layout(line(), ids=[f"p{i}" for i in range(10)], backend="auto")
    assert got.backend in available_backends()


def test_pca_is_always_available():
    assert "pca" in available_backends()


def test_layout_is_repeatable():
    ids = [f"p{i}" for i in range(10)]
    a = layout(line(), ids=ids).to_dict()
    b = layout(line(), ids=ids).to_dict()
    assert a == b


def test_serialisation_can_omit_points():
    got = layout(line(3), ids=["a", "b", "c"]).to_dict(include_points=False)
    assert "points" not in got and got["n_points"] == 3


# --------------------------------------------------------------------------
# layout_projections
# --------------------------------------------------------------------------

def _proj(sid, vector, version="v1", label="concept"):
    return SentenceProjection(
        sentence_id=sid, text=f"text for {sid}", section_id="s1",
        source_id="p1", basis_version=version,
        embedding_model="m", embedding_revision="r",
        top_concepts=(ConceptHit("row_1", label, 2, 0.5, 1),),
        vector=tuple(vector))


def test_projections_are_laid_out():
    projs = [_proj(f"s{i}", [float(i), 0.0, 0.0]) for i in range(6)]
    got = layout_projections(projs)
    assert len(got) == 6 and got.basis_version == "v1"


def test_points_are_grouped_by_top_concept():
    projs = [_proj("s0", [1.0, 0, 0], label="alpha"),
             _proj("s1", [0, 1.0, 0], label="beta")]
    assert {p.group for p in layout_projections(projs).points} == {"alpha", "beta"}


def test_margin_travels_into_the_metadata():
    got = layout_projections([_proj("s0", [1.0, 0, 0])])
    assert "margin" in got.points[0].to_dict()


def test_mixed_basis_versions_are_refused():
    """Coordinates from different bases are not comparable; silently plotting
    them together would invent structure."""
    projs = [_proj("s0", [1.0, 0, 0], version="v1"),
             _proj("s1", [0, 1.0, 0], version="v2")]
    with pytest.raises(ValueError, match="basis versions"):
        layout_projections(projs)


def test_projections_without_vectors_yield_an_empty_layout():
    """top-k concepts alone are not a position in the space."""
    p = SentenceProjection("s0", "t", "s1", "p1", "v1", "m", "r")
    assert len(layout_projections([p])) == 0


def test_no_projections_yields_an_empty_layout():
    assert len(layout_projections([])) == 0
