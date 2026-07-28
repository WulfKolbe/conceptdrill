"""Request interface for other CLI agents.

ConceptDrill does not index, schedule, or call out to anything. It exposes the
four operations another agent would drive and returns a status envelope:

    {"status": "completed" | "updated" | "failed", "operation": ..., ...}

The envelope is the contract. An agent reads `status` and, on failure, `error` —
it never has to introspect an exception type or parse a traceback.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .core import ConceptDrill
from .document import Document
from .storage import (build_payload, read_sidecar, sidecar_path, verify_sidecar,
                      write_sidecar)

STATUS_COMPLETED = "completed"
STATUS_UPDATED = "updated"
STATUS_FAILED = "failed"


def _ok(operation: str, status: str = STATUS_COMPLETED, **payload) -> dict[str, Any]:
    return {"status": status, "operation": operation, **payload}


def _fail(operation: str, exc: BaseException) -> dict[str, Any]:
    return {
        "status": STATUS_FAILED,
        "operation": operation,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(limit=5),
    }


def request_concept_generation(input_path: str | Path, *,
                               sources: Optional[Iterable[str]] = None,
                               model: str = "sentencebert",
                               max_concepts: int = 100,
                               **kwargs) -> dict[str, Any]:
    """Build a concept space and return its vocabulary without projecting."""
    op = "concept_generation"
    try:
        drill = ConceptDrill.from_path(
            input_path, embedding_model_name=model, sources=sources,
            max_concepts=max_concepts, **kwargs)
        return _ok(op,
                   source=str(input_path),
                   concepts=[{"id": c.id, "name": c.name, "score": c.score,
                              "source": c.source, "level": c.level}
                             for c in drill.space.concepts],
                   info=drill.get_concept_space_info())
    except Exception as exc:
        return _fail(op, exc)


def request_embedding(texts: Sequence[str], *, model: str = "sentencebert",
                      cache: bool = True,
                      cache_dir: Optional[str] = None) -> dict[str, Any]:
    """Embed raw text. No document, no concept space — just f(text)."""
    op = "embedding"
    try:
        from .embeddings import get_embedder
        embedder = get_embedder(model, cache=cache, cache_dir=cache_dir)
        vectors = embedder.encode(list(texts))
        flush = getattr(embedder, "flush", None)
        if callable(flush):
            flush()
        return _ok(op,
                   model=embedder.name, revision=embedder.revision,
                   dimension=int(vectors.shape[1]) if vectors.size else 0,
                   count=int(vectors.shape[0]),
                   embeddings=[[round(float(v), 6) for v in row] for row in vectors])
    except Exception as exc:
        return _fail(op, exc)


def request_projection(input_path: str | Path, *,
                       model: str = "sentencebert",
                       sources: Optional[Iterable[str]] = None,
                       top_k: int = 10,
                       max_concepts: int = 100,
                       text: Optional[str] = None,
                       **kwargs) -> dict[str, Any]:
    """Project a document, or a single span against the document's space."""
    op = "projection"
    try:
        drill = ConceptDrill.from_path(
            input_path, embedding_model_name=model, sources=sources,
            max_concepts=max_concepts, **kwargs)

        if text is not None:
            hits = drill.explain_hits(text, top_k=top_k)
            return _ok(op, source=str(input_path), text=text,
                       concepts=[{"concept_id": h.concept_id,
                                  "concept_name": h.concept_name,
                                  "similarity": h.similarity, "rank": h.rank}
                                 for h in hits])

        projections, skipped = drill.project_document(top_k=top_k)
        return _ok(op,
                   source=str(input_path),
                   projections=[p.to_dict(include_embedding=False)
                                for p in projections],
                   n_projections=len(projections),
                   n_skipped=len(skipped),
                   info=drill.get_concept_space_info())
    except Exception as exc:
        return _fail(op, exc)


def request_storage(input_path: str | Path, *,
                    output: Optional[str | Path] = None,
                    model: str = "sentencebert",
                    sources: Optional[Iterable[str]] = None,
                    top_k: int = 10,
                    max_concepts: int = 100,
                    store_embeddings: bool = False,
                    **kwargs) -> dict[str, Any]:
    """Project and persist. Reports `updated` when it replaced a sidecar."""
    op = "storage"
    try:
        target = sidecar_path(input_path, output)
        existed = target.exists()
        previous_hash = ""
        if existed:
            try:
                previous_hash = str(read_sidecar(target).get("content_hash", ""))
            except Exception:
                previous_hash = ""

        drill = ConceptDrill.from_path(
            input_path, embedding_model_name=model, sources=sources,
            max_concepts=max_concepts, **kwargs)
        projections, skipped = drill.project_document(
            top_k=top_k, store_embedding=store_embeddings)

        payload = build_payload(
            source_path=str(input_path), space=drill.space,
            projections=projections, skipped=skipped,
            meta=drill.get_concept_space_info(),
            store_embeddings=store_embeddings,
        )
        write_sidecar(payload, target)

        new_hash = str(payload.get("content_hash", ""))
        return _ok(op,
                   status=STATUS_UPDATED if existed else STATUS_COMPLETED,
                   source=str(input_path), output=str(target),
                   content_hash=new_hash,
                   changed=(previous_hash != new_hash) if existed else True,
                   n_projections=len(projections), n_skipped=len(skipped))
    except Exception as exc:
        return _fail(op, exc)


def request_verification(sidecar: str | Path) -> dict[str, Any]:
    """Recompute a stored sidecar's content hash. Detects hand-edits."""
    op = "verification"
    try:
        matches, stored, recomputed = verify_sidecar(sidecar)
        return _ok(op, path=str(sidecar), matches=matches,
                   stored_hash=stored, recomputed_hash=recomputed)
    except Exception as exc:
        return _fail(op, exc)


__all__ = [
    "STATUS_COMPLETED", "STATUS_FAILED", "STATUS_UPDATED",
    "request_concept_generation", "request_embedding", "request_projection",
    "request_storage", "request_verification",
]
