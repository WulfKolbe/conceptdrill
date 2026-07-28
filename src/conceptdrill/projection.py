"""Building `Projection` records from a concept space.

A projection is an *additional view* of a document object. Nothing here reads or
writes the object itself — a projection carries only its id and type, so the
"never modify the original" rule cannot be broken by accident.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .document import Document
from .embeddings.base import Embedder
from .space import ConceptSpace
from .types import (Block, Confidence, ConceptHit, Projection, SIM_PRECISION,
                    SkippedObject)

DEFAULT_TOP_K = 10


def project_blocks(blocks: Sequence[Block], space: ConceptSpace, *,
                   embedder: Embedder,
                   top_k: int = DEFAULT_TOP_K,
                   projection_type: str = "concepts",
                   concept_source: str = "document",
                   store_embedding: bool = True,
                   created_at: str = "",
                   parameters: Optional[dict] = None,
                   ) -> list[Projection]:
    """One `Projection` per block, in the order given.

    Blocks are embedded in a single batch: the whole point of the matrix form is
    that projecting n spans is one `(n, d) @ (d, k)` multiply.
    """
    blocks = [b for b in blocks if not b.is_empty]
    if not blocks or not len(space):
        return []

    vectors = embedder.encode([b.text for b in blocks])
    scores = vectors @ space.matrix.T          # (n_blocks, n_concepts)

    k = max(1, min(int(top_k), len(space)))
    out: list[Projection] = []

    for row, block in enumerate(blocks):
        row_scores = scores[row]
        cut = np.argpartition(-row_scores, k - 1)[:k]
        ordered = cut[np.argsort(-row_scores[cut], kind="stable")]
        hits = tuple(
            ConceptHit(
                concept_id=space.concepts[i].id,
                concept_name=space.concepts[i].name,
                similarity=round(float(row_scores[i]), SIM_PRECISION),
                rank=rank,
            )
            for rank, i in enumerate(ordered, start=1)
        )
        out.append(Projection(
            projection_id=Projection.make_id(
                block.id, embedder.name, embedder.revision,
                concept_source, space.similarity_metric, k),
            object_id=block.id,
            object_type=block.type,
            projection_type=projection_type,
            embedding_model=embedder.name,
            embedding_revision=embedder.revision,
            concept_source=concept_source,
            similarity_metric=space.similarity_metric,
            top_k=k,
            concepts=hits,
            confidence=Confidence.from_hits(hits),
            embedding=(tuple(round(float(v), SIM_PRECISION)
                             for v in vectors[row]) if store_embedding else ()),
            parameters=dict(parameters or {}),
            created_at=created_at,
        ))
    return out


def project_document(doc: Document, space: ConceptSpace, *,
                     embedder: Embedder,
                     top_k: int = DEFAULT_TOP_K,
                     types: Optional[Iterable[str]] = None,
                     **kwargs) -> tuple[list[Projection], list[SkippedObject]]:
    """Project every projectable block, and report the ones that were not.

    Skipped objects are returned rather than dropped so a caller can account for
    every object in the input.
    """
    wanted = {t.lower() for t in types} if types else None

    projectable: list[Block] = []
    skipped: list[SkippedObject] = []
    for block in doc.blocks:
        if wanted is not None and block.type.lower() not in wanted:
            skipped.append(SkippedObject(
                block.id, block.type, f"type not in --types filter"))
        elif block.is_empty:
            skipped.append(SkippedObject(
                block.id, block.type, "no text to embed"))
        else:
            projectable.append(block)

    # Objects the DocModel adapter already discarded (pages, citations, ...).
    for entry in doc.meta.get("skipped", ()) or ():
        if isinstance(entry, dict):
            skipped.append(SkippedObject(
                str(entry.get("object_id", "")),
                str(entry.get("object_type", "")),
                str(entry.get("reason", "")),
            ))

    projections = project_blocks(projectable, space, embedder=embedder,
                                 top_k=top_k, **kwargs)
    skipped.sort(key=lambda s: (s.object_type, s.object_id))
    return projections, skipped


def explain(text: str, space: ConceptSpace, *, embedder: Embedder,
            top_k: int = 5) -> list[tuple[str, float]]:
    """`(concept_name, similarity)` pairs for a free-text span."""
    vector = embedder.encode([text])[0]
    return [(hit.concept_name, hit.similarity) for hit in space.top_k(vector, top_k)]
