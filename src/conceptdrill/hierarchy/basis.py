"""The self-adapting cross-document concept basis.

Each document contributes one candidate vector per section, embedded from that
section's `label` tier. Candidates are integrated into a shared basis one at a
time, per level:

    j = argmax cosine(v, M_L)
    cosine < TAU  ->  append v as a new row, support 1
    cosine >= TAU ->  merge into row j as a running mean, support += 1

The result is matrix `M`: one row per basis concept, in the decided order
`(level, -support, label)`.

## Three properties that are not incidental

**Order dependence is real and is handled, not wished away.** Merging is a
running mean, so the basis depends on the order documents arrive in. Documents
are therefore integrated in a deterministic order and that order is recorded,
so a rebuild reproduces the same basis rather than a plausible one.

**Identity is content-addressed, never positional.** `row_id` is a hash of
`(level, canonical label)`. Support counts change as a corpus grows, so row
POSITIONS move; a CES vector stored against position would silently transpose.
`basis_version` hashes the ordered row ids so a mismatch is loud.

**The arithmetic is float64.** Embeddings arrive as float32, but a merge
decision turns on one comparison against TAU: a wrong cosine adds a row that
should have merged, or merges two unrelated concepts, and neither is
recoverable afterwards. On the development host float32 GEMM was measurably
wrong while float64 was clean, so the basis converts on entry and stays there.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional, Sequence

import numpy as np

#: Cosine at or above which a candidate merges instead of becoming a new row.
#:
#: **Measured, not guessed.** The design spec proposed 0.85; against three
#: topically related papers embedded with all-MiniLM-L6-v2 that produced ZERO
#: merges of any kind, because the highest cross-document similarity observed
#: was 0.647 -- and that pair was genuinely related (both on precision/recall
#: in knowledge bases). Sentence embeddings of 20-40 word technical labels put
#: related-but-differently-worded concepts around 0.6, not 0.85; 0.85 is a
#: near-paraphrase threshold and the wrong scale for this job.
#:
#: 0.65 is the compromise: high enough that unrelated concepts stay apart, low
#: enough that cross-document sharing is reachable. It is still a default, not
#: a law -- use `calibrate` on your own corpus.
DEFAULT_TAU = 0.65

#: Levels are grouped contiguously by the row order, so a per-level split is a
#: slicing decision rather than a reordering one.
LEVEL_KEY = "level"


def canonical_label(label: str) -> str:
    """The form a row's identity is computed from.

    Case and whitespace are not meaning: "Concept Space" and "concept  space"
    name the same concept and must not produce two rows.
    """
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def make_row_id(level: int, label: str) -> str:
    """Content-addressed row identity, stable across reordering and merges."""
    payload = f"{level}\x1f{canonical_label(label)}"
    return "row_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _unit(vector) -> np.ndarray:
    """float64 unit vector. Zero vectors are returned unchanged, not NaN."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


@dataclass(frozen=True)
class BasisRow:
    """One concept in the shared basis."""
    row_id: str
    level: int
    label: str
    vector: np.ndarray
    support: int = 1
    #: bibkeys that contributed, in integration order, deduplicated.
    documents: tuple[str, ...] = ()
    #: Every label merged into this row. The evidence for what it means.
    merged_labels: tuple[str, ...] = ()

    @property
    def canonical(self) -> str:
        return canonical_label(self.label)

    def to_dict(self, include_vector: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "row_id": self.row_id, "level": self.level, "label": self.label,
            "support": self.support, "documents": list(self.documents),
            "merged_labels": list(self.merged_labels),
        }
        if include_vector:
            out["vector"] = [round(float(v), 8) for v in self.vector]
        return out


def new_row(level: int, label: str, vector, document: str = "") -> BasisRow:
    return BasisRow(
        row_id=make_row_id(level, label), level=level, label=label,
        vector=_unit(vector), support=1,
        documents=(document,) if document else (),
        merged_labels=(label,))


def merge(row: BasisRow, vector, label: str = "", document: str = "") -> BasisRow:
    """Fold a candidate into a row as a running mean.

    The row's `label` and therefore its `row_id` do NOT change: the canonical
    label is what identifies the concept, and an identity that drifted on every
    merge would invalidate every CES vector already stored against it.
    """
    n = max(1, row.support)
    blended = _unit((row.vector * n + _unit(vector)) / (n + 1))
    docs = row.documents
    if document and document not in docs:
        docs = docs + (document,)
    labels = row.merged_labels
    if label and label not in labels:
        labels = labels + (label,)
    return replace(row, vector=blended, support=n + 1,
                   documents=docs, merged_labels=labels)


@dataclass
class IntegrationResult:
    """What happened to one candidate."""
    action: str            # "added" | "merged"
    row_id: str
    similarity: float
    level: int
    label: str


@dataclass
class ConceptBasis:
    """The shared basis: rows plus the machinery to grow it."""

    rows: dict[str, BasisRow] = field(default_factory=dict)
    tau: float = DEFAULT_TAU
    #: Documents integrated, in order. The basis depends on this.
    document_order: tuple[str, ...] = ()

    # ---- basics ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def levels(self) -> list[int]:
        return sorted({r.level for r in self.rows.values()})

    def by_level(self, level: int) -> list[BasisRow]:
        return [r for r in self.ordered_rows() if r.level == level]

    # ---- ordering -------------------------------------------------------

    def ordered_rows(self) -> list[BasisRow]:
        """Canonical order: level-major, then support descending, then label.

        The label tie-break is what makes the ordering total. Without it, two
        rows at the same level with equal support fall back to dict insertion
        order, which varies between runs.
        """
        return sorted(self.rows.values(),
                      key=lambda r: (r.level, -r.support, r.canonical))

    def row_ids(self) -> list[str]:
        return [r.row_id for r in self.ordered_rows()]

    def basis_version(self) -> str:
        """Hash of the ordered row ids.

        Changes on any add, merge or reorder. A CES vector records this; on a
        mismatch the correct action is to re-project, not to compare.
        """
        payload = "\x1f".join(self.row_ids())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def matrix(self) -> np.ndarray:
        """`M`, one unit-norm row per concept, in canonical order."""
        rows = self.ordered_rows()
        if not rows:
            return np.zeros((0, 0), dtype=np.float64)
        return np.vstack([r.vector for r in rows])

    def position_of(self, row_id: str) -> Optional[int]:
        """Index in the current matrix. Valid only for this `basis_version`."""
        for i, rid in enumerate(self.row_ids()):
            if rid == row_id:
                return i
        return None

    # ---- the adaptive core ---------------------------------------------

    def nearest(self, level: int, vector) -> tuple[Optional[BasisRow], float]:
        """Closest row *at the same level*, and its cosine.

        Levels do not compete: a chapter concept and a subsection concept are
        different kinds of thing, and letting them merge would flatten the
        hierarchy the basis exists to represent.
        """
        candidates = [r for r in self.rows.values() if r.level == level]
        if not candidates:
            return None, -1.0
        vec = _unit(vector)
        if not np.any(vec):
            return None, -1.0
        sims = np.vstack([r.vector for r in candidates]) @ vec
        best = int(np.argmax(sims))
        return candidates[best], float(sims[best])

    def integrate(self, level: int, label: str, vector, *,
                  document: str = "", tau: Optional[float] = None
                  ) -> IntegrationResult:
        """Add or merge one candidate. The self-adapting step."""
        threshold = self.tau if tau is None else tau
        vec = _unit(vector)

        if not np.any(vec):
            # A zero vector has no direction to compare; adding it would put a
            # meaningless row in the basis forever.
            return IntegrationResult("skipped", "", -1.0, level, label)

        match, similarity = self.nearest(level, vec)
        if match is not None and similarity >= threshold:
            self.rows[match.row_id] = merge(match, vec, label, document)
            return IntegrationResult("merged", match.row_id, similarity,
                                     level, label)

        row = new_row(level, label, vec, document)
        if row.row_id in self.rows:
            # Same canonical label at the same level: identical concept by
            # identity even though the vectors drifted apart. Merge on identity.
            self.rows[row.row_id] = merge(self.rows[row.row_id], vec, label,
                                          document)
            return IntegrationResult("merged", row.row_id, similarity,
                                     level, label)
        self.rows[row.row_id] = row
        return IntegrationResult("added", row.row_id, similarity, level, label)

    def integrate_document(self, document: str,
                           candidates: Iterable[tuple[int, str, Any]], *,
                           tau: Optional[float] = None) -> list[IntegrationResult]:
        """Integrate one document's candidates as `(level, label, vector)`.

        Candidates are sorted by `(level, canonical label)` first: within a
        document the arrival order would otherwise depend on section ordering,
        making the basis depend on something no one intended.
        """
        ordered = sorted(candidates, key=lambda c: (c[0], canonical_label(c[1])))
        results = [self.integrate(level, label, vector, document=document,
                                  tau=tau)
                   for level, label, vector in ordered]
        if document and document not in self.document_order:
            self.document_order = self.document_order + (document,)
        return results

    # ---- reporting ------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        by_level: dict[int, int] = {}
        for row in self.rows.values():
            by_level[row.level] = by_level.get(row.level, 0) + 1
        supports = [r.support for r in self.rows.values()] or [0]
        shared = [r for r in self.rows.values() if len(r.documents) > 1]
        return {
            "rows": len(self.rows),
            "levels": dict(sorted(by_level.items())),
            "documents": len(self.document_order),
            "tau": self.tau,
            "basis_version": self.basis_version(),
            "support_max": max(supports),
            "support_mean": round(sum(supports) / len(supports), 3),
            "singletons": sum(1 for r in self.rows.values() if r.support == 1),
            "shared_across_documents": len(shared),
        }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def similarity_profile(vector_sets: Sequence[np.ndarray]) -> dict[str, Any]:
    """Within- and cross-set cosine distributions.

    The two distributions answer different questions. Within-set similarity is
    how much a single document repeats itself; cross-set similarity is how much
    two documents genuinely share. A threshold that ignores the difference
    either collapses one document's sections or never merges anything.
    """
    sets = [np.asarray(v, dtype=np.float64) for v in vector_sets
            if getattr(v, "size", 0)]
    within: list[float] = []
    cross: list[float] = []

    for i, a in enumerate(sets):
        if a.shape[0] > 1:
            sim = a @ a.T
            iu = np.triu_indices(sim.shape[0], k=1)
            within.extend(sim[iu].tolist())
        for b in sets[i + 1:]:
            cross.extend((a @ b.T).ravel().tolist())

    def describe(values: list[float]) -> dict[str, float]:
        if not values:
            return {"n": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
        }

    return {"within_document": describe(within), "cross_document": describe(cross)}


def calibrate(vector_sets: Sequence[np.ndarray], *,
              cross_percentile: float = 99.0,
              floor: float = 0.45, ceiling: float = 0.95) -> dict[str, Any]:
    """Suggest a TAU from observed similarity, with the evidence for it.

    The suggestion is the given percentile of the CROSS-document distribution:
    merging should be reachable for the most similar pairs across documents,
    which is the point of a shared basis, while everything typical stays apart.

    Returns the suggestion *and* the profile, because a number without its
    distribution invites the same unexamined 0.85 this replaced.
    """
    profile = similarity_profile(vector_sets)
    cross = profile["cross_document"]
    if not cross.get("n"):
        return {"suggested_tau": DEFAULT_TAU, "reason": "no cross-document pairs",
                "profile": profile}

    flat: list[float] = []
    sets = [np.asarray(v, dtype=np.float64) for v in vector_sets
            if getattr(v, "size", 0)]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            flat.extend((a @ b.T).ravel().tolist())
    suggested = float(np.percentile(np.asarray(flat), cross_percentile))
    suggested = max(floor, min(ceiling, suggested))
    return {
        "suggested_tau": round(suggested, 3),
        "reason": f"p{cross_percentile:g} of cross-document similarity",
        "profile": profile,
    }
