"""Answering a query against a stored corpus basis.

    query text -> f(q) -> CES(q) = M @ f(q) -> categories + nearest sentences

Two different searches fall out of the same CES vector, and they answer
different questions:

  * **categories** — the query's own coordinates. "What concepts is this
    about?" Read straight off the vector, no corpus needed.
  * **neighbours** — cosine between the query's CES vector and stored sentence
    CES vectors. "Where in the corpus is this discussed?" Needs the index.

Comparing in **CES space rather than embedding space** is the point of the
exercise: the similarity is then explainable, because the coordinates are named
concepts and the reason two things matched can be named too.

Every answer carries the `basis_version` it was computed against, and a query
run against a different basis than the stored vectors is refused rather than
quietly compared.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .project import PRECISION, ConceptHit, project_vectors

DEFAULT_TOP_CONCEPTS = 5
DEFAULT_TOP_NEIGHBOURS = 5


@dataclass(frozen=True)
class Neighbour:
    """A stored sentence near the query in CES space."""
    sentence_id: str
    text: str
    similarity: float
    document: str = ""
    span_id: Optional[str] = None
    #: Concepts both the query and this sentence rank highly. The explanation
    #: for *why* they matched, which raw cosine cannot give.
    shared_concepts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"sentence_id": self.sentence_id, "text": self.text,
                "similarity": self.similarity, "document": self.document,
                "span_id": self.span_id,
                "shared_concepts": list(self.shared_concepts)}


@dataclass(frozen=True)
class QueryResult:
    """A query, its concept categories, and where the corpus discusses it."""
    query: str
    basis_version: str
    embedding_model: str
    categories: tuple[ConceptHit, ...] = ()
    neighbours: tuple[Neighbour, ...] = ()
    vector: tuple[float, ...] = ()
    created_at: str = ""

    @property
    def best(self) -> Optional[ConceptHit]:
        return self.categories[0] if self.categories else None

    @property
    def margin(self) -> float:
        if len(self.categories) < 2:
            return 0.0
        return round(self.categories[0].similarity
                     - self.categories[1].similarity, PRECISION)

    @property
    def annotated(self) -> str:
        """The query with its categories injected — the spec's step 7 output."""
        if not self.categories:
            return self.query
        names = ", ".join(h.label for h in self.categories)
        return f"{self.query}  [concepts: {names}]"

    def to_dict(self, include_vector: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": self.query,
            "basis_version": self.basis_version,
            "embedding_model": self.embedding_model,
            "categories": [h.to_dict() for h in self.categories],
            "neighbours": [n.to_dict() for n in self.neighbours],
            "margin": self.margin,
            "annotated": self.annotated,
            "created_at": self.created_at,
        }
        if include_vector and self.vector:
            out["vector"] = list(self.vector)
        return out


def _cosine_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine of `vector` against every row. float64, zero-safe."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    mat = np.asarray(matrix, dtype=np.float64)
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)

    row_norms = np.linalg.norm(mat, axis=1)
    row_norms[row_norms == 0.0] = 1.0
    vec_norm = float(np.linalg.norm(vec)) or 1.0
    return (mat @ vec) / (row_norms * vec_norm)


class QueryEngine:
    """Answers queries against a basis, optionally with a sentence index."""

    def __init__(self, basis, embedder, *,
                 records: Optional[Sequence[dict]] = None,
                 vectors: Optional[np.ndarray] = None) -> None:
        self.basis = basis
        self.embedder = embedder
        self.records = list(records or [])
        self.vectors = (np.asarray(vectors, dtype=np.float64)
                        if vectors is not None else np.zeros((0, 0)))
        if self.records and self.vectors.shape[0] != len(self.records):
            raise ValueError(
                f"{len(self.records)} records but {self.vectors.shape[0]} vectors")

    @property
    def has_index(self) -> bool:
        return bool(self.records) and self.vectors.size > 0

    def ces_vector(self, text: str) -> np.ndarray:
        """`M @ f(q)` — the query's coordinates in the concept basis."""
        embedded = self.embedder.encode([text])
        return project_vectors(embedded, self.basis)[0]

    def categories(self, ces: np.ndarray, *,
                   top_k: int = DEFAULT_TOP_CONCEPTS) -> list[ConceptHit]:
        rows = self.basis.ordered_rows()
        if not rows or ces.size == 0:
            return []
        k = max(1, min(int(top_k), len(rows)))
        cut = np.argpartition(-ces, k - 1)[:k]
        ordered = cut[np.argsort(-ces[cut], kind="stable")]
        return [
            ConceptHit(row_id=rows[i].row_id, label=rows[i].label,
                       level=rows[i].level,
                       similarity=round(float(ces[i]), PRECISION), rank=rank)
            for rank, i in enumerate(ordered, start=1)
        ]

    def neighbours(self, ces: np.ndarray, *,
                   top_k: int = DEFAULT_TOP_NEIGHBOURS,
                   query_concepts: Sequence[ConceptHit] = ()) -> list[Neighbour]:
        """Stored sentences nearest the query **in CES space**.

        Cosine, not raw dot product: CES vectors are not unit-norm — a sentence
        matching every concept weakly would otherwise outrank one matching a
        single concept strongly, purely on magnitude.
        """
        if not self.has_index:
            return []
        sims = _cosine_rows(self.vectors, ces)
        k = max(1, min(int(top_k), sims.shape[0]))
        cut = np.argpartition(-sims, k - 1)[:k]
        ordered = cut[np.argsort(-sims[cut], kind="stable")]

        wanted = {h.label for h in query_concepts}
        out: list[Neighbour] = []
        for i in ordered:
            rec = self.records[int(i)]
            theirs = [c.get("label", "") for c in rec.get("top_concepts", [])]
            out.append(Neighbour(
                sentence_id=rec.get("sentence_id", ""),
                text=rec.get("text", ""),
                similarity=round(float(sims[i]), PRECISION),
                document=rec.get("document", ""),
                span_id=rec.get("span_id"),
                shared_concepts=tuple(t for t in theirs if t in wanted),
            ))
        return out

    def query(self, text: str, *, top_concepts: int = DEFAULT_TOP_CONCEPTS,
              top_neighbours: int = DEFAULT_TOP_NEIGHBOURS,
              store_vector: bool = False,
              created_at: str = "") -> QueryResult:
        """One query, answered."""
        ces = self.ces_vector(text)
        cats = self.categories(ces, top_k=top_concepts)
        near = self.neighbours(ces, top_k=top_neighbours, query_concepts=cats)
        return QueryResult(
            query=text,
            basis_version=self.basis.basis_version(),
            embedding_model=getattr(self.embedder, "name", "?"),
            categories=tuple(cats), neighbours=tuple(near),
            vector=(tuple(round(float(v), PRECISION) for v in ces)
                    if store_vector else ()),
            created_at=created_at,
        )


# --------------------------------------------------------------------------
# Step 8: the query log
# --------------------------------------------------------------------------

class QueryLog:
    """Append-only record of queries and their answers.

    JSON Lines rather than one JSON document: a log is appended to far more
    often than it is read whole, and a truncated final line costs one entry
    instead of the file.

    Each entry carries its `basis_version`, so a log spanning a basis rebuild
    can still be read — the entries simply are not comparable across the
    boundary, and `read` can filter to one version.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, result: QueryResult, *,
               answer: Optional[str] = None,
               include_vector: bool = True) -> dict[str, Any]:
        entry = result.to_dict(include_vector=include_vector)
        entry["query_id"] = hashlib.sha256(
            f"{result.basis_version}\x1f{result.query}".encode("utf-8")
        ).hexdigest()[:16]
        if answer is not None:
            entry["answer"] = answer

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def read(self, *, basis_version: Optional[str] = None) -> list[dict[str, Any]]:
        """Every entry, optionally filtered to one basis version.

        A malformed line is skipped rather than raising: a log is diagnostic,
        and one bad entry must not make the rest unreadable.
        """
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if basis_version is None or entry.get("basis_version") == basis_version:
                out.append(entry)
        return out

    def __len__(self) -> int:
        return len(self.read())
