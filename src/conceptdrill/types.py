"""Core data types.

Everything here is a frozen dataclass. Projections are records, not living
objects: once built they are serialised and compared by hash, so immutability
keeps the determinism guarantee cheap to reason about.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

# Similarities are rounded before storage so two runs on different machines
# compare equal despite last-bit float jitter.
SIM_PRECISION = 6


def _stable_hash(*parts: Any) -> str:
    """sha256 over a canonical rendering of `parts`. Used for every id in the
    system, so it must never depend on dict ordering or wall-clock time."""
    payload = "\x1f".join(
        json.dumps(p, sort_keys=True, separators=(",", ":"), default=str)
        if not isinstance(p, str) else p
        for p in parts
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Document structure
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    """A typed text block: paragraph, equation, definition, caption, ..."""
    id: str
    type: str
    text: str
    section_id: Optional[str] = None
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class Section:
    """A node in the document's section tree."""
    id: str
    title: str
    level: int = 1
    parent_id: Optional[str] = None
    children: tuple[str, ...] = ()

    @property
    def is_top_level(self) -> bool:
        return self.parent_id is None


@dataclass(frozen=True)
class BibEntry:
    """A bibliography entry. `citations` and `year` drive citation importance."""
    id: str
    title: str
    label: str = ""
    year: Optional[int] = None
    citations: Optional[int] = None
    keywords: tuple[str, ...] = ()
    props: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """A concept candidate before scoring.

    `source` names the generator that produced it and drives the structural
    weight. `tau` is the text representation fed to the embedding model — the
    tau mapping from the CES paper. It defaults to `name` but generators may
    override it (a paper title, an equation description).
    """
    name: str
    source: str
    tau: str = ""
    kind: str = ""
    section_id: Optional[str] = None
    frequency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tau:
            object.__setattr__(self, "tau", self.name)

    @property
    def key(self) -> str:
        """Deduplication key. Candidates from different generators that name the
        same thing collapse into one, keeping the highest structural weight."""
        return " ".join(self.name.lower().split())


@dataclass(frozen=True)
class Concept:
    """A selected, scored concept in the final vocabulary."""
    id: str
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    source: str = ""
    tau: str = ""
    score: float = 0.0
    level: str = "document"          # document | domain | global
    parent_id: Optional[str] = None
    children: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(name: str, source: str, level: str = "document") -> str:
        return "cpt_" + _stable_hash(level, source, " ".join(name.lower().split()))[:12]


@dataclass(frozen=True)
class ConceptHit:
    """One concept's similarity to one object, with its rank."""
    concept_id: str
    concept_name: str
    similarity: float
    rank: int

    def rounded(self) -> "ConceptHit":
        return ConceptHit(
            self.concept_id, self.concept_name,
            round(self.similarity, SIM_PRECISION), self.rank,
        )


@dataclass(frozen=True)
class Confidence:
    """Two numbers, because one is not enough.

    `top1` says how strong the best match is. `margin` says whether the mapping
    is ambiguous — a high top1 with a near-zero margin means several concepts
    describe the object equally well, which is exactly the signal
    multi-model-disagreement analysis looks for.
    """
    top1: float
    margin: float

    @classmethod
    def from_hits(cls, hits: Sequence[ConceptHit]) -> "Confidence":
        if not hits:
            return cls(0.0, 0.0)
        top1 = hits[0].similarity
        second = hits[1].similarity if len(hits) > 1 else 0.0
        return cls(round(top1, SIM_PRECISION), round(top1 - second, SIM_PRECISION))


@dataclass(frozen=True)
class Projection:
    """An additional semantic view of one document object.

    Never replaces the object; carries only its id. `projection_id` excludes the
    timestamp so it is stable across runs.
    """
    projection_id: str
    object_id: str
    object_type: str
    projection_type: str
    embedding_model: str
    embedding_revision: str
    concept_source: str
    similarity_metric: str
    top_k: int
    concepts: tuple[ConceptHit, ...]
    confidence: Confidence
    embedding: tuple[float, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @staticmethod
    def make_id(object_id: str, model: str, revision: str,
                concept_source: str, metric: str, top_k: int) -> str:
        return "prj_" + _stable_hash(
            object_id, model, revision, concept_source, metric, top_k)[:16]

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_embedding:
            d.pop("embedding", None)
        return d


@dataclass(frozen=True)
class SkippedObject:
    """An object that carried no projectable text.

    Recorded rather than dropped, so a run accounts for every object in the
    input and an empty projection set is never mistaken for a bug.
    """
    object_id: str
    object_type: str
    reason: str
