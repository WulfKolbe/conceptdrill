"""Laying CES vectors out in two dimensions, for looking at.

A separate stage on purpose. UMAP and t-SNE are heavy dependencies, are
stochastic, and answer a question — "what does this look like" — that nothing
downstream depends on. Keeping them out of the core keeps the core
reproducible, and lets this run or not run without consequence.

Three backends behind one protocol:

    pca     pure numpy, deterministic, always available. The default.
    umap    optional (`umap-learn`), better at local structure.
    tsne    optional (`scikit-learn`), better at cluster separation.

## Why PCA is the default rather than a fallback

It is not a lesser UMAP. On CES vectors it is arguably the *right* tool: a CES
vector's coordinates already mean something — each is the cosine to one named
basis concept — so a linear projection keeps the axes interpretable, and the
loadings say which concepts drive each axis. UMAP's axes mean nothing at all.

## Eigenvector signs are arbitrary, and that matters here

`numpy.linalg.svd` may return `v` or `-v` for the same input; both are correct
eigenvectors. Two runs would then produce mirrored plots, and a reader
comparing them would see a change that is not there. `_fix_signs` pins the
convention: the largest-magnitude loading of each component is made positive.
Without it "deterministic PCA" is only deterministic within one process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

#: Backends, in order of preference when `auto` is asked for.
BACKENDS = ("umap", "tsne", "pca")

#: Fixed seed for the stochastic backends. Without one they are not repeatable
#: at all; with one they are repeatable per library version, which is the most
#: that can be promised.
DEFAULT_SEED = 0

PRECISION = 6


@dataclass(frozen=True)
class Point:
    """One laid-out item, with enough context to colour and label a plot."""
    id: str
    x: float
    y: float
    label: str = ""
    group: str = ""
    #: Free-form, e.g. document, span, top concept, similarity.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x": self.x, "y": self.y,
                "label": self.label, "group": self.group, **self.meta}


@dataclass
class Layout:
    """A 2-D layout plus how it was produced."""
    points: list[Point] = field(default_factory=list)
    backend: str = ""
    seed: Optional[int] = None
    #: PCA only: fraction of variance each axis carries.
    explained_variance: tuple[float, ...] = ()
    #: PCA only: which input dimensions drive each axis.
    loadings: tuple[tuple[int, float], ...] = ()
    deterministic: bool = True
    basis_version: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def to_dict(self, include_points: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.backend,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "basis_version": self.basis_version,
            "n_points": len(self.points),
        }
        if self.explained_variance:
            out["explained_variance"] = list(self.explained_variance)
        if self.loadings:
            out["top_loadings"] = [{"dimension": d, "weight": w}
                                   for d, w in self.loadings]
        if include_points:
            out["points"] = [p.to_dict() for p in self.points]
        return out


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """Pin the arbitrary sign of each eigenvector.

    SVD may return `v` or `-v`; both are correct. Left alone, two runs produce
    mirrored plots and a reader sees a change that did not happen. Convention:
    the largest-magnitude loading of each component is positive.
    """
    for i, row in enumerate(components):
        if row.size and row[np.argmax(np.abs(row))] < 0:
            components[i] = -row
    return components


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-component PCA. Returns `(coords, explained_variance, components)`.

    float64 throughout: this is an SVD, exactly the kind of linear algebra this
    project has learned not to trust in float32.
    """
    data = np.asarray(vectors, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(2), np.zeros((2, data.shape[1] or 1))

    centred = data - data.mean(axis=0, keepdims=True)
    # A single point, or identical points, have no variance to decompose.
    if data.shape[0] == 1 or not np.any(centred):
        return (np.zeros((data.shape[0], 2)), np.zeros(2),
                np.zeros((2, data.shape[1])))

    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    k = min(2, vt.shape[0])
    components = _fix_signs(vt[:k].copy())
    coords = centred @ components.T

    total = float((s ** 2).sum())
    explained = (s[:k] ** 2) / total if total > 0 else np.zeros(k)

    # Pad to two axes so callers never have to special-case a 1-D input.
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
        explained = np.concatenate([explained, np.zeros(2 - explained.shape[0])])
        components = np.vstack([components,
                                np.zeros((2 - components.shape[0], data.shape[1]))])
    return coords, explained, components


def _umap_2d(vectors: np.ndarray, seed: int) -> Optional[np.ndarray]:
    try:
        import umap
    except Exception:
        return None
    try:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return np.asarray(reducer.fit_transform(np.asarray(vectors, dtype=np.float64)))
    except Exception:
        return None


def _tsne_2d(vectors: np.ndarray, seed: int) -> Optional[np.ndarray]:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    data = np.asarray(vectors, dtype=np.float64)
    if data.shape[0] < 5:
        return None                     # t-SNE needs more points than that
    try:
        # perplexity must stay below the sample count or sklearn refuses.
        perplexity = min(30.0, max(5.0, (data.shape[0] - 1) / 3.0))
        return np.asarray(TSNE(n_components=2, random_state=seed,
                               perplexity=perplexity,
                               init="pca").fit_transform(data))
    except Exception:
        return None


def available_backends() -> tuple[str, ...]:
    """Which backends this installation can actually run."""
    out = ["pca"]
    try:
        import umap  # noqa: F401
        out.append("umap")
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE  # noqa: F401
        out.append("tsne")
    except Exception:
        pass
    return tuple(out)


def layout(vectors, *, ids: Sequence[str],
           labels: Optional[Sequence[str]] = None,
           groups: Optional[Sequence[str]] = None,
           meta: Optional[Sequence[dict]] = None,
           backend: str = "pca", seed: int = DEFAULT_SEED,
           basis_version: str = "") -> Layout:
    """Lay vectors out in 2-D.

    An unavailable or failing backend falls back to PCA rather than raising:
    a visualisation stage must never be the reason a pipeline stops, and the
    `backend` field records what actually ran.
    """
    data = np.asarray(vectors, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = data.shape[0]
    if n != len(ids):
        raise ValueError(f"{n} vectors but {len(ids)} ids")

    requested = backend.lower().strip()
    if requested == "auto":
        requested = next((b for b in BACKENDS if b in available_backends()), "pca")

    coords = None
    used = requested
    if requested == "umap":
        coords = _umap_2d(data, seed)
    elif requested == "tsne":
        coords = _tsne_2d(data, seed)

    explained: np.ndarray = np.zeros(2)
    components: np.ndarray = np.zeros((2, data.shape[1] if data.size else 1))
    if coords is None:
        coords, explained, components = pca_2d(data)
        used = "pca"

    coords = np.asarray(coords, dtype=np.float64)
    points = [
        Point(id=str(ids[i]),
              x=round(float(coords[i, 0]), PRECISION),
              y=round(float(coords[i, 1]), PRECISION),
              label=str(labels[i]) if labels is not None else "",
              group=str(groups[i]) if groups is not None else "",
              meta=dict(meta[i]) if meta is not None else {})
        for i in range(n)
    ]

    top_loadings: list[tuple[int, float]] = []
    if used == "pca" and components.size:
        for axis in range(min(2, components.shape[0])):
            row = components[axis]
            if row.size:
                idx = int(np.argmax(np.abs(row)))
                top_loadings.append((idx, round(float(row[idx]), PRECISION)))

    return Layout(
        points=points, backend=used, seed=seed if used != "pca" else None,
        explained_variance=tuple(round(float(v), PRECISION) for v in explained)
        if used == "pca" else (),
        loadings=tuple(top_loadings),
        deterministic=(used == "pca"),
        basis_version=basis_version,
    )


def layout_projections(projections: Sequence[Any], *, backend: str = "pca",
                       seed: int = DEFAULT_SEED,
                       document_of=None) -> Layout:
    """Lay out `SentenceProjection`s, colouring by their top concept.

    Requires projections carrying full vectors (`store_vectors=True`): the
    top-k concepts alone are not a position in the space.
    """
    usable = [p for p in projections if getattr(p, "vector", ())]
    if not usable:
        return Layout(backend=backend, basis_version="")

    versions = {p.basis_version for p in usable}
    if len(versions) > 1:
        raise ValueError(
            f"projections span {len(versions)} basis versions; coordinates from "
            f"different bases are not comparable. Re-project first.")

    return layout(
        np.vstack([np.asarray(p.vector, dtype=np.float64) for p in usable]),
        ids=[p.sentence_id for p in usable],
        labels=[p.text[:80] for p in usable],
        groups=[p.best.label if p.best else "" for p in usable],
        meta=[{"span_id": p.span_id,
               "document": document_of(p) if document_of else "",
               "top_similarity": p.best.similarity if p.best else 0.0,
               "margin": p.margin} for p in usable],
        backend=backend, seed=seed,
        basis_version=next(iter(versions)),
    )
