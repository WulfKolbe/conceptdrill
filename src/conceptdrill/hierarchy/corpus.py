"""Persisting the corpus-level basis and its CES vectors.

This is the one artefact that deliberately does **not** live in a drill folder.
A marker tree belongs to one document; a shared basis is a property of the
corpus, and writing it beside one document would make that document silently
authoritative for every other.

Layout:

    <corpus_dir>/
        ces-basis.json     rows, document order, tau, basis_version
        ces-basis.npz      the matrix M, float64
        ces-index.json     one record per stored sentence
        ces-vectors.npz    sentence CES vectors, float64

Metadata is JSON so it is readable and diffable; vectors are `npz` because a
few thousand sentences times a few dozen coordinates is megabytes of JSON
floats for no benefit.

## The invariant this file exists to protect

A CES vector is meaningless without the basis it was computed against.
`basis_version` is written into every file here and checked on load, so a
vectors file left behind by an earlier basis is refused rather than silently
misread. That is the whole reason the version exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .basis import BasisRow, ConceptBasis

BASIS_JSON = "ces-basis.json"
BASIS_NPZ = "ces-basis.npz"
INDEX_JSON = "ces-index.json"
VECTORS_NPZ = "ces-vectors.npz"

FORMAT = "conceptdrill.ces.corpus"
FORMAT_VERSION = 1


class BasisMismatch(ValueError):
    """Raised when stored vectors do not belong to the loaded basis."""


@dataclass
class CorpusStore:
    """A directory holding one corpus basis and the vectors projected into it."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ---- paths ----------------------------------------------------------

    @property
    def basis_json(self) -> Path:
        return self.path / BASIS_JSON

    @property
    def basis_npz(self) -> Path:
        return self.path / BASIS_NPZ

    @property
    def index_json(self) -> Path:
        return self.path / INDEX_JSON

    @property
    def vectors_npz(self) -> Path:
        return self.path / VECTORS_NPZ

    def exists(self) -> bool:
        return self.basis_json.exists() and self.basis_npz.exists()

    # ---- writing --------------------------------------------------------

    def _write_json(self, target: Path, payload: dict[str, Any]) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                  default=str) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target

    def _write_npz(self, target: Path, **arrays) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        # Written through a file object: given a path, np.savez appends ".npz"
        # when the name does not already end in it, and the rename target would
        # never exist.
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(target)
        return target

    def save_basis(self, basis: ConceptBasis, *, embedding_model: str = "",
                   embedding_revision: str = "",
                   summarizer: str = "") -> dict[str, Any]:
        """Write the basis. Rows keep canonical order so the matrix aligns."""
        rows = basis.ordered_rows()
        payload = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "basis_version": basis.basis_version(),
            "tau": basis.tau,
            "document_order": list(basis.document_order),
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "summarizer": summarizer,
            "stats": basis.stats(),
            "rows": [r.to_dict(include_vector=False) for r in rows],
        }
        self._write_json(self.basis_json, payload)

        matrix = (np.vstack([r.vector for r in rows]) if rows
                  else np.zeros((0, 0), dtype=np.float64))
        self._write_npz(self.basis_npz, matrix=matrix,
                        row_ids=np.array([r.row_id for r in rows], dtype=object),
                        basis_version=np.array(basis.basis_version()))
        return payload

    def save_projections(self, projections: Sequence[Any], *,
                         basis_version: str,
                         document_of=None) -> dict[str, Any]:
        """Write sentence CES vectors and their metadata.

        Projections carrying a different `basis_version` are refused: mixing
        them would produce an index whose coordinates mean different things in
        different rows.
        """
        usable = [p for p in projections if getattr(p, "vector", ())]
        wrong = {p.basis_version for p in usable} - {basis_version}
        if wrong:
            raise BasisMismatch(
                f"projections carry basis versions {sorted(wrong)} but the "
                f"basis is {basis_version}; re-project before storing")

        records = []
        for p in usable:
            records.append({
                "sentence_id": p.sentence_id,
                "text": p.text,
                "span_id": p.span_id,
                "source_id": p.source_id,
                "document": document_of(p) if document_of else "",
                "top_concepts": [h.to_dict() for h in p.top_concepts],
                "margin": p.margin,
            })

        payload = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "basis_version": basis_version,
            "n_sentences": len(records),
            "records": records,
        }
        self._write_json(self.index_json, payload)

        vectors = (np.vstack([np.asarray(p.vector, dtype=np.float64)
                              for p in usable])
                   if usable else np.zeros((0, 0), dtype=np.float64))
        self._write_npz(self.vectors_npz, vectors=vectors,
                        sentence_ids=np.array([p.sentence_id for p in usable],
                                              dtype=object),
                        basis_version=np.array(basis_version))
        return payload

    # ---- reading --------------------------------------------------------

    def load_basis(self) -> ConceptBasis:
        """Rebuild the basis. Refuses a matrix that belongs to another version."""
        if not self.exists():
            raise FileNotFoundError(f"no corpus basis at {self.path}")

        meta = json.loads(self.basis_json.read_text(encoding="utf-8"))
        with np.load(self.basis_npz, allow_pickle=True) as z:
            matrix = np.asarray(z["matrix"], dtype=np.float64)
            row_ids = [str(x) for x in z["row_ids"]]
            stored_version = str(z["basis_version"])

        if stored_version != meta.get("basis_version"):
            raise BasisMismatch(
                f"{BASIS_NPZ} is version {stored_version} but {BASIS_JSON} is "
                f"{meta.get('basis_version')}; the pair is inconsistent")
        if len(row_ids) != matrix.shape[0]:
            raise BasisMismatch(
                f"{len(row_ids)} row ids but {matrix.shape[0]} matrix rows")

        basis = ConceptBasis(tau=float(meta.get("tau", 0.65)),
                             document_order=tuple(meta.get("document_order", ())))
        by_id = {r["row_id"]: r for r in meta.get("rows", [])}
        for i, row_id in enumerate(row_ids):
            rec = by_id.get(row_id)
            if rec is None:
                continue
            basis.rows[row_id] = BasisRow(
                row_id=row_id, level=int(rec["level"]), label=rec["label"],
                vector=matrix[i], support=int(rec.get("support", 1)),
                documents=tuple(rec.get("documents", ())),
                merged_labels=tuple(rec.get("merged_labels", ())))

        if basis.basis_version() != meta.get("basis_version"):
            raise BasisMismatch(
                "the rebuilt basis does not reproduce its stored version; the "
                "store has been edited or written by an incompatible release")
        return basis

    def load_index(self, *, basis_version: Optional[str] = None
                   ) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Sentence records and their CES vectors, aligned.

        Refuses vectors belonging to a different basis than the one asked for.
        """
        if not self.index_json.exists() or not self.vectors_npz.exists():
            return [], np.zeros((0, 0), dtype=np.float64)

        meta = json.loads(self.index_json.read_text(encoding="utf-8"))
        with np.load(self.vectors_npz, allow_pickle=True) as z:
            vectors = np.asarray(z["vectors"], dtype=np.float64)
            sentence_ids = [str(x) for x in z["sentence_ids"]]
            stored_version = str(z["basis_version"])

        if stored_version != meta.get("basis_version"):
            raise BasisMismatch(
                f"{VECTORS_NPZ} is version {stored_version} but {INDEX_JSON} is "
                f"{meta.get('basis_version')}")
        if basis_version is not None and stored_version != basis_version:
            raise BasisMismatch(
                f"stored vectors are for basis {stored_version} but the loaded "
                f"basis is {basis_version}; re-project before searching")

        records = meta.get("records", [])
        if len(records) != vectors.shape[0] or len(sentence_ids) != len(records):
            raise BasisMismatch(
                f"{len(records)} records but {vectors.shape[0]} vectors")
        return records, vectors

    def info(self) -> dict[str, Any]:
        """What is here, without loading the arrays."""
        if not self.basis_json.exists():
            return {"exists": False, "path": str(self.path)}
        meta = json.loads(self.basis_json.read_text(encoding="utf-8"))
        out = {
            "exists": True,
            "path": str(self.path),
            "basis_version": meta.get("basis_version"),
            "tau": meta.get("tau"),
            "rows": len(meta.get("rows", [])),
            "documents": len(meta.get("document_order", [])),
            "embedding_model": meta.get("embedding_model"),
            "has_index": self.index_json.exists(),
        }
        if self.index_json.exists():
            idx = json.loads(self.index_json.read_text(encoding="utf-8"))
            out["sentences"] = idx.get("n_sentences", 0)
        return out
