"""Reading and writing `math-concepts.json`.

One file per document, holding every `MathConcept` extracted from it.

## Where it does NOT go

Not in a drill folder, and not beside `ces-basis.json`. A MathConcept is not
versioned by `basis_version` — a definition does not stop being true when TAU
changes — so storing it with artefacts that *are* would tie it to a lifetime it
does not share. `corpus.py` makes the same argument for the opposite case: a
shared basis must not live beside one document.

Records are validated on write AND on read. On write because a malformed
record must not reach disk; on read because a file may have been produced by an
older release, and silently accepting an unknown field is how a contract stops
being one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import (FORMAT, FORMAT_VERSION, SchemaViolation, assert_record,
                     check_record)

FILENAME = "math-concepts.json"


class StoreMismatch(ValueError):
    """A file that is not what this module writes."""


@dataclass
class MathConceptStore:
    """A directory holding one document's MathConcepts."""

    path: Path

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def file(self) -> Path:
        return self.path / FILENAME

    def exists(self) -> bool:
        return self.file.exists()

    def save(self, records: Sequence[dict[str, Any]], *, doc_id: str = "",
             extractor: str = "", verified_against: str = "") -> dict[str, Any]:
        """Write every record, atomically. Raises before writing anything bad.

        `verified_against` names what checked the Lean anchors — a Mathlib
        revision, a Loogle endpoint and date, or "" for unverified. Without it
        a reader cannot tell a `supported` anchor from a stale one.
        """
        for record in records:
            assert_record(record)

        payload = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "doc_id": doc_id or None,
            "extractor": extractor or None,
            "verified_against": verified_against or None,
            "n_concepts": len(records),
            "concepts": list(records),
        }
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_name(self.file.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(self.file)
        return payload

    def load(self) -> dict[str, Any]:
        """The stored payload. Refuses a file this module did not write."""
        if not self.exists():
            raise FileNotFoundError(f"no {FILENAME} at {self.path}")

        payload = json.loads(self.file.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise StoreMismatch(
                f"{self.file} declares format {payload.get('format')!r}, "
                f"not {FORMAT!r}")
        if payload.get("format_version") != FORMAT_VERSION:
            raise StoreMismatch(
                f"{self.file} is format version {payload.get('format_version')}"
                f" but this release writes {FORMAT_VERSION}")

        problems: list[str] = []
        for record in payload.get("concepts") or []:
            for problem in check_record(record):
                problems.append(f"{record.get('concept')!r}: {problem}")
        if problems:
            raise SchemaViolation(f"{self.file} holds invalid records: {problems}")
        return payload

    def concepts(self) -> list[dict[str, Any]]:
        return list(self.load().get("concepts") or [])


def merge(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold records naming the same concept into one, keeping every difference.

    Two spans of one document both discussing linear maps produce two records
    with the same content-addressed `id`. Merging unions their symbols,
    relations, anchors and evidence rather than picking a winner, because which
    span mentioned a symbol is exactly the thing a reader needs.

    Deduplication is by the WHOLE part, not by its `latex` or `name`: the same
    symbol seen once as `observed` and once as `predicted` is two facts, and
    collapsing them would lose the observation or invent one.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["id"]
        if key not in by_id:
            by_id[key] = json.loads(json.dumps(record))
            continue
        target = by_id[key]
        for field in ("latex", "loogle"):
            for item in record.get(field) or []:
                if item not in (target.get(field) or []):
                    target.setdefault(field, []).append(item)
        for field in ("symbols", "related", "lean"):
            for item in record.get(field) or []:
                if item not in (target.get(field) or []):
                    target.setdefault(field, []).append(item)
        left, right = target.get("evidence") or {}, record.get("evidence") or {}
        for field in ("span_ids", "ces_row_ids"):
            merged = list(left.get(field) or [])
            for item in right.get(field) or []:
                if item not in merged:
                    merged.append(item)
            left[field] = merged
        target["evidence"] = left

    # Confidence is derived, so it must be recomputed after the parts change.
    from .schema import confidence
    for record in by_id.values():
        record["confidence"] = confidence(record)
    return [by_id[k] for k in sorted(by_id)]
