"""`ConceptSpace` — the selected vocabulary and the CES projection matrix.

Selection is greedy over quality score with a diversity constraint: walk the
scored candidates best-first and reject any whose embedding exceeds
`diversity_threshold` cosine similarity to something already chosen. That keeps
"semantic image similarity search" from crowding out everything else with five
paraphrases of itself.

The projection itself is the matrix form from the CES paper. With `M`'s rows
being unit-norm concept vectors and `l` a unit-norm text vector::

    project(s) = M @ l

which is every cosine similarity in one BLAS call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .document import Document
from .embeddings.base import Embedder, l2_normalise
from .scoring.scorer import ScoredCandidate
from .types import Concept, ConceptHit, SIM_PRECISION

DEFAULT_MAX_CONCEPTS = 100
DEFAULT_DIVERSITY_THRESHOLD = 0.95

#: Concept levels, ordered outermost-first. `global` and `domain` are static
#: sets a caller supplies; `document` is what this module mines.
LEVELS = ("global", "domain", "document")


@dataclass
class ConceptSpace:
    """A concept vocabulary plus its embedding matrix."""

    concepts: list[Concept] = field(default_factory=list)
    #: (n_concepts, dim) unit-norm rows, aligned with `concepts`.
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    embedding_model: str = ""
    embedding_revision: str = ""
    similarity_metric: str = "cosine"
    parameters: dict = field(default_factory=dict)

    # ---- basics ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.concepts)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.concepts]

    def index_of(self, concept_id: str) -> Optional[int]:
        for i, c in enumerate(self.concepts):
            if c.id == concept_id:
                return i
        return None

    def by_level(self, level: str) -> list[Concept]:
        return [c for c in self.concepts if c.level == level]

    # ---- projection -----------------------------------------------------

    def project_vector(self, vector: np.ndarray) -> np.ndarray:
        """`M @ l` — the concept vector for an already-embedded span."""
        if not len(self.concepts) or self.matrix.size == 0:
            return np.zeros((0,), dtype=np.float32)
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"embedding dimension {vec.shape[0]} does not match the concept "
                f"space dimension {self.matrix.shape[1]}; the space was built "
                f"with model {self.embedding_model!r}"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return (self.matrix @ vec).astype(np.float32)

    def top_k(self, vector: np.ndarray, k: int = 5) -> list[ConceptHit]:
        """Top-k concepts for an embedded span, rank 1 first."""
        scores = self.project_vector(vector)
        if scores.size == 0:
            return []
        k = max(1, min(int(k), scores.size))
        # argpartition for the cut, argsort for the order within it.
        cut = np.argpartition(-scores, k - 1)[:k]
        ordered = cut[np.argsort(-scores[cut], kind="stable")]
        return [
            ConceptHit(
                concept_id=self.concepts[i].id,
                concept_name=self.concepts[i].name,
                similarity=round(float(scores[i]), SIM_PRECISION),
                rank=rank,
            )
            for rank, i in enumerate(ordered, start=1)
        ]

    # ---- metadata -------------------------------------------------------

    def info(self) -> dict:
        """The `get_concept_space_info()` payload."""
        by_level: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for c in self.concepts:
            by_level[c.level] = by_level.get(c.level, 0) + 1
            by_source[c.source] = by_source.get(c.source, 0) + 1
        roots = [c.id for c in self.concepts if c.parent_id is None]
        return {
            "size": len(self.concepts),
            "dimension": self.dim,
            "levels": dict(sorted(by_level.items())),
            "sources": dict(sorted(by_source.items())),
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "similarity_metric": self.similarity_metric,
            "hierarchy": {
                "roots": len(roots),
                "with_parent": sum(1 for c in self.concepts if c.parent_id),
                "max_children": max((len(c.children) for c in self.concepts),
                                    default=0),
            },
            "parameters": dict(self.parameters),
        }

    def save_matrix(self, path) -> None:
        """Persist `M` and the concept ids. Rebuilding the matrix is the
        expensive part of loading a space; the ids make it verifiable."""
        np.savez_compressed(
            path, matrix=self.matrix,
            concept_ids=np.array([c.id for c in self.concepts], dtype=object),
            model=self.embedding_model, revision=self.embedding_revision,
        )


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def select_concepts(scored: Sequence[ScoredCandidate], *,
                    embedder: Embedder,
                    max_concepts: int = DEFAULT_MAX_CONCEPTS,
                    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
                    min_score: float = 0.0,
                    level: str = "document",
                    embeddings: Optional[np.ndarray] = None,
                    ) -> tuple[list[Concept], np.ndarray]:
    """Greedy best-first selection under a diversity constraint.

    `embeddings` may be supplied to reuse vectors already computed during
    scoring; it must be aligned with `scored`.
    """
    scored = [s for s in scored if s.score >= min_score]
    if not scored:
        return [], np.zeros((0, 0), dtype=np.float32)

    if embeddings is None:
        embeddings = embedder.encode([s.candidate.tau for s in scored])
    else:
        embeddings = l2_normalise(np.asarray(embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(scored):
        raise ValueError("embeddings and scored candidates are misaligned")

    chosen_idx: list[int] = []
    chosen_vecs: list[np.ndarray] = []

    for i, item in enumerate(scored):
        if len(chosen_idx) >= max_concepts:
            break
        vec = embeddings[i]
        if chosen_vecs:
            # One dot product against the chosen block; cheaper than pairwise.
            sims = np.stack(chosen_vecs) @ vec
            if float(sims.max()) > diversity_threshold:
                continue
        chosen_idx.append(i)
        chosen_vecs.append(vec)

    concepts: list[Concept] = []
    for i in chosen_idx:
        item = scored[i]
        cand = item.candidate
        meta = dict(cand.metadata)
        aliases = tuple(str(a) for a in (meta.pop("aliases", ()) or ()))
        description = str(meta.pop("description", "") or "")
        concepts.append(Concept(
            id=Concept.make_id(cand.name, cand.source, level),
            name=cand.name,
            description=description,
            aliases=aliases,
            source=cand.source,
            tau=cand.tau,
            score=item.score,
            level=level,
            metrics=dict(item.metrics),
            metadata={**meta, "kind": cand.kind, "frequency": cand.frequency,
                      "section_id": cand.section_id},
        ))

    matrix = (np.stack(chosen_vecs) if chosen_vecs
              else np.zeros((0, embeddings.shape[1]), dtype=np.float32))
    return concepts, l2_normalise(matrix)


def attach_hierarchy(concepts: Sequence[Concept], doc: Document) -> list[Concept]:
    """Wire parent/child links from the document's section tree.

    A heading concept's parent is the concept for its parent section; concepts
    mined from inside a section hang off that section's concept. Concepts with
    no section anchor stay roots. Purely additive — nothing is dropped.
    """
    # Section id -> the concept representing it.
    section_concept: dict[str, str] = {}
    for c in concepts:
        sid = c.metadata.get("section_id")
        if c.source == "heading" and c.metadata.get("kind") == "heading" and sid:
            section_concept.setdefault(str(sid), c.id)

    if not section_concept:
        return list(concepts)

    parent_of: dict[str, Optional[str]] = {}
    for c in concepts:
        sid = c.metadata.get("section_id")
        parent: Optional[str] = None
        if sid:
            sid = str(sid)
            own = section_concept.get(sid)
            if own == c.id:
                # A section's own concept hangs off its parent section.
                section = doc.sections.get(sid)
                pid = section.parent_id if section else None
                parent = section_concept.get(str(pid)) if pid else None
            else:
                parent = own
        if parent == c.id:
            parent = None
        parent_of[c.id] = parent

    # Break any cycle the section tree might have introduced.
    for cid in list(parent_of):
        seen = {cid}
        cur = parent_of.get(cid)
        while cur:
            if cur in seen:
                parent_of[cid] = None
                break
            seen.add(cur)
            cur = parent_of.get(cur)

    children: dict[str, list[str]] = {}
    for cid, pid in parent_of.items():
        if pid:
            children.setdefault(pid, []).append(cid)

    return [
        Concept(
            id=c.id, name=c.name, description=c.description, aliases=c.aliases,
            source=c.source, tau=c.tau, score=c.score, level=c.level,
            parent_id=parent_of.get(c.id),
            children=tuple(sorted(children.get(c.id, ()))),
            metrics=c.metrics, metadata=c.metadata,
        )
        for c in concepts
    ]


def build_space(scored: Sequence[ScoredCandidate], doc: Document, *,
                embedder: Embedder,
                max_concepts: int = DEFAULT_MAX_CONCEPTS,
                diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
                min_score: float = 0.0,
                extra_levels: Optional[Iterable[tuple[str, Sequence[Concept]]]] = None,
                embeddings: Optional[np.ndarray] = None,
                ) -> ConceptSpace:
    """Select, wire the hierarchy, and concatenate any static levels."""
    concepts, matrix = select_concepts(
        scored, embedder=embedder, max_concepts=max_concepts,
        diversity_threshold=diversity_threshold, min_score=min_score,
        level="document", embeddings=embeddings,
    )
    concepts = attach_hierarchy(concepts, doc)

    all_concepts = list(concepts)
    matrices = [matrix] if matrix.size else []

    # Static global/domain levels, embedded on the fly. Ordered before the
    # document level so `LEVELS` order holds in the final space.
    for level, static in (extra_levels or ()):
        static = list(static)
        if not static:
            continue
        vecs = embedder.encode([c.tau or c.name for c in static])
        all_concepts = [
            Concept(
                id=c.id or Concept.make_id(c.name, c.source or level, level),
                name=c.name, description=c.description, aliases=c.aliases,
                source=c.source or level, tau=c.tau or c.name, score=c.score,
                level=level, parent_id=c.parent_id, children=c.children,
                metrics=c.metrics, metadata=c.metadata,
            ) for c in static
        ] + all_concepts
        matrices.insert(0, vecs)

    combined = (np.vstack(matrices) if matrices
                else np.zeros((0, max(1, embedder.dim)), dtype=np.float32))

    return ConceptSpace(
        concepts=all_concepts,
        matrix=l2_normalise(combined) if combined.size else combined,
        embedding_model=embedder.name,
        embedding_revision=embedder.revision,
        similarity_metric="cosine",
        parameters={
            "max_concepts": max_concepts,
            "diversity_threshold": diversity_threshold,
            "min_score": min_score,
        },
    )


# --------------------------------------------------------------------------
# 3.3 On-demand granularity
# --------------------------------------------------------------------------

def refine_space(space: ConceptSpace, context: str, *,
                 desired_size: int,
                 embedder: Embedder,
                 pool: Sequence[ScoredCandidate] = (),
                 ) -> ConceptSpace:
    """Expand the space's most context-relevant concepts with their children.

    The C* adaptation from the CES paper, with the document's own candidates as
    the expansion pool rather than a Wikipedia hierarchy:

      1. Rank the current concepts by similarity to `context`.
      2. Add the children of the most relevant ones, best-first.
      3. If children run out, fall back to the unselected candidate pool,
         ranked by relevance to the context.

    Returns a new space; the input is untouched.
    """
    if desired_size <= len(space):
        return space

    ctx_vec = embedder.encode_one(context) if hasattr(embedder, "encode_one") \
        else embedder.encode([context])[0]
    relevance = space.project_vector(ctx_vec)

    by_id = {c.id: c for c in space.concepts}
    present = set(by_id)
    ranked = sorted(range(len(space.concepts)),
                    key=lambda i: (-float(relevance[i]) if relevance.size else 0.0,
                                   space.concepts[i].name))

    additions: list[Concept] = []
    for i in ranked:
        if len(space) + len(additions) >= desired_size:
            break
        for child_id in space.concepts[i].children:
            if child_id in present or len(space) + len(additions) >= desired_size:
                continue
            child = by_id.get(child_id)
            if child is not None:
                additions.append(child)
                present.add(child_id)

    # Children exhausted: draw from the unselected pool by context relevance.
    if len(space) + len(additions) < desired_size and pool:
        remaining = [s for s in pool
                     if Concept.make_id(s.candidate.name, s.candidate.source,
                                        "document") not in present]
        if remaining:
            vecs = embedder.encode([s.candidate.tau for s in remaining])
            sims = vecs @ np.asarray(ctx_vec, dtype=np.float32).reshape(-1)
            order = sorted(range(len(remaining)),
                           key=lambda i: (-float(sims[i]), remaining[i].candidate.name))
            for i in order:
                if len(space) + len(additions) >= desired_size:
                    break
                item = remaining[i]
                cand = item.candidate
                cid = Concept.make_id(cand.name, cand.source, "document")
                if cid in present:
                    continue
                present.add(cid)
                additions.append(Concept(
                    id=cid, name=cand.name, source=cand.source, tau=cand.tau,
                    score=item.score, level="document", metrics=dict(item.metrics),
                    metadata={**cand.metadata, "added_by": "refine_space"},
                ))

    if not additions:
        return space

    new_vecs = embedder.encode([c.tau or c.name for c in additions])
    return ConceptSpace(
        concepts=list(space.concepts) + additions,
        matrix=l2_normalise(np.vstack([space.matrix, new_vecs])),
        embedding_model=space.embedding_model,
        embedding_revision=space.embedding_revision,
        similarity_metric=space.similarity_metric,
        parameters={**space.parameters, "refined_to": desired_size,
                    "refined_from": len(space)},
    )
