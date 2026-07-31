"""Projecting sentences into the shared concept basis.

    CES(s) = M @ f(s)

with `M` the basis matrix in canonical row order and `f(s)` a unit-norm
sentence embedding. Because both sides are unit-norm, each coordinate is the
cosine between the sentence and one basis concept.

## What a CES vector must carry with it

A bare vector of numbers is not interpretable and, worse, is not *safely*
interpretable: coordinate 4 means whatever row 4 of `M` was at the time. Support
counts change as a corpus grows, so rows reorder. Every projection therefore
records:

  * `basis_version` — a hash of the ordered row ids. Comparing two CES vectors
    with different versions is meaningless, and the mismatch is detectable
    rather than silent.
  * `embedding_model` and its revision — a vector from a different encoder is
    not in the same space at all.

`top_concepts` names the coordinates, so a projection can be read without the
basis to hand.

**float64**, like the basis. A CES coordinate is a cosine that downstream code
will threshold and rank; on the development host float32 GEMM was measurably
wrong, and a projection is exactly a GEMM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

#: Coordinates stored per sentence by default. The full vector is one float per
#: basis row and grows with the corpus; the top few are what anyone reads.
DEFAULT_TOP_K = 8

#: Stored coordinates are rounded so two runs compare equal despite last-bit
#: jitter, and so a sidecar does not carry meaningless precision.
PRECISION = 6


@dataclass(frozen=True)
class ConceptHit:
    """One basis concept's similarity to one sentence."""
    row_id: str
    label: str
    level: int
    similarity: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "label": self.label, "level": self.level,
                "similarity": self.similarity, "rank": self.rank}


@dataclass(frozen=True)
class SentenceProjection:
    """One sentence's CES vector, with everything needed to read it later."""
    sentence_id: str
    text: str
    span_id: Optional[str]
    source_id: str
    basis_version: str
    embedding_model: str
    embedding_revision: str
    top_concepts: tuple[ConceptHit, ...] = ()
    #: The full CES vector, optional because it grows with the basis.
    vector: tuple[float, ...] = ()

    @property
    def best(self) -> Optional[ConceptHit]:
        return self.top_concepts[0] if self.top_concepts else None

    @property
    def margin(self) -> float:
        """Top-1 minus top-2.

        Reported alongside the top similarity because they answer different
        questions: a high top-1 with a near-zero margin means several concepts
        describe the sentence equally well, which is ambiguity, not confidence.
        """
        if len(self.top_concepts) < 2:
            return 0.0
        return round(self.top_concepts[0].similarity
                     - self.top_concepts[1].similarity, PRECISION)

    def to_dict(self, include_vector: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "span_id": self.span_id,
            "source_id": self.source_id,
            "basis_version": self.basis_version,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "top_concepts": [h.to_dict() for h in self.top_concepts],
            "margin": self.margin,
        }
        if include_vector and self.vector:
            out["vector"] = list(self.vector)
        return out


def project_vectors(vectors, basis) -> np.ndarray:
    """`M @ l` for a stack of sentence vectors. Returns `(n_sentences, n_rows)`.

    One matmul for the whole document: that is the entire point of the matrix
    form, and doing it per sentence would be the same arithmetic many times over.
    """
    matrix = basis.matrix()
    if matrix.size == 0:
        return np.zeros((len(vectors), 0), dtype=np.float64)

    stacked = np.asarray(vectors, dtype=np.float64)
    if stacked.ndim == 1:
        stacked = stacked.reshape(1, -1)
    if stacked.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"sentence embeddings are {stacked.shape[1]}-dimensional but the "
            f"basis is {matrix.shape[1]}-dimensional; the basis was built with "
            f"a different embedding model")

    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (stacked / norms) @ matrix.T


def project_sentences(sentences: Sequence[Any], basis, embedder, *,
                      top_k: int = DEFAULT_TOP_K,
                      store_vectors: bool = False,
                      batch_size: int = 256) -> list[SentenceProjection]:
    """Project every sentence into the basis.

    Sentences are embedded in batches and projected in one multiply per batch.
    An empty basis yields projections with no concepts rather than an error:
    the sentences are still real, and a caller that built no basis should see
    that plainly.
    """
    usable = [s for s in sentences if getattr(s, "is_usable", True)]
    if not usable:
        return []

    rows = basis.ordered_rows()
    version = basis.basis_version()
    model = getattr(embedder, "name", "?")
    revision = getattr(embedder, "revision", "?")
    k = max(1, min(int(top_k), len(rows))) if rows else 0

    out: list[SentenceProjection] = []
    for start in range(0, len(usable), batch_size):
        batch = usable[start:start + batch_size]
        vectors = embedder.encode([s.text for s in batch])
        scores = project_vectors(vectors, basis)

        for sentence, row_scores in zip(batch, scores):
            hits: list[ConceptHit] = []
            if k:
                cut = np.argpartition(-row_scores, k - 1)[:k]
                ordered = cut[np.argsort(-row_scores[cut], kind="stable")]
                hits = [
                    ConceptHit(row_id=rows[i].row_id, label=rows[i].label,
                               level=rows[i].level,
                               similarity=round(float(row_scores[i]), PRECISION),
                               rank=rank)
                    for rank, i in enumerate(ordered, start=1)
                ]
            out.append(SentenceProjection(
                sentence_id=sentence.id, text=sentence.text,
                span_id=getattr(sentence, "span_id", None),
                source_id=getattr(sentence, "source_id", ""),
                basis_version=version,
                embedding_model=model, embedding_revision=revision,
                top_concepts=tuple(hits),
                vector=(tuple(round(float(v), PRECISION) for v in row_scores)
                        if store_vectors else ()),
            ))
    return out


def projection_stats(projections: Sequence[SentenceProjection]) -> dict[str, Any]:
    """How the projection went, in terms worth acting on."""
    if not projections:
        return {"sentences": 0}

    tops = [p.best.similarity for p in projections if p.best]
    margins = [p.margin for p in projections if p.best]
    versions = {p.basis_version for p in projections}
    concept_use: dict[str, int] = {}
    for p in projections:
        if p.best:
            concept_use[p.best.label] = concept_use.get(p.best.label, 0) + 1

    tops_sorted = sorted(tops)
    return {
        "sentences": len(projections),
        "basis_versions": sorted(versions),
        "top1_min": round(tops_sorted[0], 4) if tops else None,
        "top1_median": round(tops_sorted[len(tops_sorted) // 2], 4) if tops else None,
        "top1_max": round(tops_sorted[-1], 4) if tops else None,
        "margin_median": round(sorted(margins)[len(margins) // 2], 4) if margins else None,
        # A basis row that never wins is dead weight; one that wins everything
        # is a sink. Both are worth seeing.
        "concepts_used": len(concept_use),
        "most_frequent_concept": max(concept_use.items(),
                                     key=lambda kv: (kv[1], kv[0]))[0]
        if concept_use else None,
    }
