"""The `ConceptDrill` facade.

Constructing one runs the whole build: parse -> candidates -> score -> select ->
embed. After that, `project_text` and `explain_text` are cheap.

    drill = ConceptDrill.from_path("paper.json")
    drill.explain_text("Deep learning for graphs", top_k=5)

Everything is injectable — embedder, scorer, abstractor, generator set — so any
stage can be replaced without subclassing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .abstractor import Abstractor, NullAbstractor
from .candidates import DEFAULT_SOURCES, generate_candidates
from .document import Document
from .embeddings import get_embedder
from .embeddings.base import Embedder
from .nlp import backend_name
from .projection import DEFAULT_TOP_K, project_document, project_blocks
from .scoring.scorer import QualityScorer, ScoredCandidate
from .space import (ConceptSpace, DEFAULT_DIVERSITY_THRESHOLD,
                    DEFAULT_MAX_CONCEPTS, build_space, refine_space)
from .types import Candidate, Concept, Projection, SkippedObject

DEFAULT_MODEL = "sentencebert"


@dataclass
class BuildReport:
    """What the build did. Lands in the sidecar's `meta`, so a stored projection
    explains how its vocabulary was produced."""
    n_candidates: int = 0
    n_selected: int = 0
    sources: tuple[str, ...] = ()
    nlp_backend: str = ""
    abstractor: str = ""
    abstractor_deterministic: bool = True
    candidates_by_source: dict[str, int] = field(default_factory=dict)
    selected_by_source: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_selected": self.n_selected,
            "sources": list(self.sources),
            "nlp_backend": self.nlp_backend,
            "abstractor": self.abstractor,
            "abstractor_deterministic": self.abstractor_deterministic,
            "candidates_by_source": dict(sorted(self.candidates_by_source.items())),
            "selected_by_source": dict(sorted(self.selected_by_source.items())),
        }


class ConceptDrill:
    """Document-tailored concept space with a CES projection interface."""

    def __init__(self, document_json: dict[str, Any] | Document,
                 embedding_model_name: str = DEFAULT_MODEL,
                 *,
                 sources: Optional[Iterable[str]] = None,
                 max_concepts: int = DEFAULT_MAX_CONCEPTS,
                 diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
                 min_score: float = 0.0,
                 weights: Optional[dict[str, float]] = None,
                 embedder: Optional[Embedder] = None,
                 scorer: Optional[QualityScorer] = None,
                 abstractor: Optional[Abstractor] = None,
                 generator_options: Optional[dict] = None,
                 static_levels: Optional[Sequence[tuple[str, Sequence[Concept]]]] = None,
                 cache: bool = True,
                 cache_dir: Optional[str] = None,
                 source_path: Optional[str] = None,
                 build: bool = True) -> None:

        self.document = (document_json if isinstance(document_json, Document)
                         else Document.from_generic(document_json, source_path)
                         if not _is_docmodel(document_json)
                         else Document.from_docmodel(document_json, source_path))

        self.embedder: Embedder = embedder or get_embedder(
            embedding_model_name, cache=cache, cache_dir=cache_dir)
        self.scorer = scorer or QualityScorer(weights=weights)
        self.abstractor: Abstractor = abstractor or NullAbstractor()
        self.sources = tuple(sources) if sources is not None else DEFAULT_SOURCES
        self.max_concepts = max_concepts
        self.diversity_threshold = diversity_threshold
        self.min_score = min_score
        self.generator_options = dict(generator_options or {})
        self.static_levels = list(static_levels or [])

        self.candidates: list[Candidate] = []
        self.scored: list[ScoredCandidate] = []
        self.space = ConceptSpace()
        self.report = BuildReport()

        if build:
            self.build()

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path, **kwargs) -> "ConceptDrill":
        doc = Document.load(path)
        kwargs.setdefault("source_path", str(path))
        return cls(doc, **kwargs)

    def build(self) -> "ConceptDrill":
        """Run the full pipeline. Idempotent — calling twice rebuilds."""
        self.candidates = generate_candidates(
            self.document, sources=self.sources, abstractor=self.abstractor,
            generator_options=self.generator_options,
        )

        self.scored, ctx = self.scorer.score(
            self.document, self.candidates, self.embedder)

        # Reuse the candidate embeddings computed during scoring: re-encoding
        # them for selection would double the model calls for no benefit.
        reordered = _align_embeddings(self.scored, ctx)

        self.space = build_space(
            self.scored, self.document, embedder=self.embedder,
            max_concepts=self.max_concepts,
            diversity_threshold=self.diversity_threshold,
            min_score=self.min_score,
            extra_levels=self.static_levels,
            embeddings=reordered,
        )

        self.report = BuildReport(
            n_candidates=len(self.candidates),
            n_selected=len(self.space),
            sources=self.sources,
            nlp_backend=backend_name(),
            abstractor=self.abstractor.name,
            abstractor_deterministic=self.abstractor.is_deterministic,
            candidates_by_source=_count_by(c.source for c in self.candidates),
            selected_by_source=_count_by(c.source for c in self.space.concepts),
        )
        self._flush_cache()
        return self

    # ---- the CES interface ----------------------------------------------

    def project_text(self, text: str) -> np.ndarray:
        """The concept vector for `text`: one similarity per concept."""
        vector = self.embedder.encode([text])[0]
        return self.space.project_vector(vector)

    def explain_text(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """The top-k concepts describing `text`, strongest first."""
        vector = self.embedder.encode([text])[0]
        return [(h.concept_name, h.similarity)
                for h in self.space.top_k(vector, top_k)]

    def explain_hits(self, text: str, top_k: int = 5):
        """`explain_text` with full `ConceptHit`s — ids and ranks included."""
        vector = self.embedder.encode([text])[0]
        return self.space.top_k(vector, top_k)

    def get_concept_space_info(self) -> dict[str, Any]:
        """Metadata about the space: sizes, levels, sources, parameters."""
        return {
            **self.space.info(),
            "document": {
                "blocks": len(self.document.blocks),
                "prose_blocks": len(self.document.prose_blocks),
                "math_blocks": len(self.document.math_blocks),
                "code_blocks": len(self.document.code_blocks),
                "sections": len(self.document.sections),
                "bibliography": len(self.document.bibliography),
                "source_path": self.document.source_path,
            },
            "build": self.report.to_dict(),
            "scorer": self.scorer.describe(),
        }

    # ---- document-wide projection ---------------------------------------

    def project_document(self, *, top_k: int = DEFAULT_TOP_K,
                         types: Optional[Iterable[str]] = None,
                         store_embedding: bool = True,
                         created_at: str = "",
                         concept_source: Optional[str] = None,
                         ) -> tuple[list[Projection], list[SkippedObject]]:
        """Project every object in the document."""
        return project_document(
            self.document, self.space, embedder=self.embedder, top_k=top_k,
            types=types, store_embedding=store_embedding, created_at=created_at,
            concept_source=concept_source or "+".join(self.sources),
            parameters={
                "max_concepts": self.max_concepts,
                "diversity_threshold": self.diversity_threshold,
                "min_score": self.min_score,
                "sources": list(self.sources),
                "weights": dict(sorted(self.scorer.weights.items())),
                "theta": self.scorer.theta,
            },
        )

    def project_spans(self, texts: Sequence[str], *,
                      top_k: int = DEFAULT_TOP_K) -> list[list[tuple[str, float]]]:
        """Batch `explain_text` — one matrix multiply for all spans."""
        if not texts:
            return []
        vectors = self.embedder.encode(list(texts))
        scores = vectors @ self.space.matrix.T
        k = max(1, min(int(top_k), len(self.space)))
        out: list[list[tuple[str, float]]] = []
        for row in scores:
            cut = np.argpartition(-row, k - 1)[:k]
            ordered = cut[np.argsort(-row[cut], kind="stable")]
            out.append([(self.space.concepts[i].name, round(float(row[i]), 6))
                        for i in ordered])
        return out

    # ---- granularity ----------------------------------------------------

    def refine(self, context: str, desired_size: int) -> ConceptSpace:
        """Expand the space around `context`, in place. Returns the new space."""
        self.space = refine_space(
            self.space, context, desired_size=desired_size,
            embedder=self.embedder, pool=self.scored,
        )
        return self.space

    # ---- helpers --------------------------------------------------------

    def _flush_cache(self) -> None:
        flush = getattr(self.embedder, "flush", None)
        if callable(flush):
            flush()

    def __len__(self) -> int:
        return len(self.space)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<ConceptDrill concepts={len(self.space)} "
                f"model={self.embedder.name} sources={','.join(self.sources)}>")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _is_docmodel(data: Any) -> bool:
    from .document import is_docmodel
    return isinstance(data, dict) and is_docmodel(data)


def _count_by(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _align_embeddings(scored: Sequence[ScoredCandidate], ctx) -> Optional[np.ndarray]:
    """Reorder the scoring context's candidate embeddings to match `scored`.

    Scoring sorts candidates by quality, so the context's row order no longer
    lines up. Returning None makes selection re-encode, which is correct but
    wasteful; this keeps the single-pass guarantee.
    """
    if ctx is None or getattr(ctx, "candidate_embeddings", None) is None:
        return None
    embeddings = ctx.candidate_embeddings
    if embeddings.size == 0:
        return None
    rows = []
    for item in scored:
        idx = ctx.candidate_index.get(item.candidate.key)
        if idx is None:
            return None
        rows.append(embeddings[idx])
    return np.stack(rows) if rows else None
