"""Sidecar persistence.

Output goes beside the input as `<input>.conceptdrill.json`. The input file is
opened read-only and never written to, so a projection run cannot damage a
document model.

The payload carries a `content_hash` computed over everything *except* the
timestamp. Two runs of the same input with the same parameters produce the same
hash, which is how the reproducibility claim is checked rather than asserted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .space import ConceptSpace
from .types import Concept, Projection, SkippedObject

SIDECAR_SUFFIX = ".conceptdrill.json"
FORMAT_VERSION = 1

#: Excluded from `content_hash` — these legitimately differ between identical runs.
VOLATILE_KEYS = frozenset({"created_at", "content_hash", "source_mtime",
                           "elapsed_seconds"})


def sidecar_path(input_path: str | Path, output: Optional[str | Path] = None) -> Path:
    if output:
        return Path(output)
    p = Path(input_path)
    # `model.docmodel.json` -> `model.docmodel.conceptdrill.json`, not
    # `model.docmodel.json.conceptdrill.json`.
    stem = p.name[:-len(".json")] if p.name.endswith(".json") else p.name
    return p.with_name(stem + SIDECAR_SUFFIX)


def _strip_volatile(obj: Any) -> Any:
    """Recursively drop volatile keys so the hash is stable across runs."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in sorted(obj.items())
                if k not in VOLATILE_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_strip_volatile(v) for v in obj]
    return obj


def content_hash(payload: dict[str, Any]) -> str:
    """sha256 over the payload minus volatile fields."""
    canonical = json.dumps(_strip_volatile(payload), sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_payload(*, source_path: Optional[str],
                  space: ConceptSpace,
                  projections: Sequence[Projection],
                  skipped: Sequence[SkippedObject] = (),
                  meta: Optional[dict[str, Any]] = None,
                  store_embeddings: bool = True,
                  ) -> dict[str, Any]:
    """Assemble the sidecar document."""
    payload: dict[str, Any] = {
        "format": "conceptdrill",
        "format_version": FORMAT_VERSION,
        "source": source_path,
        "meta": dict(meta or {}),
        "concept_space": {
            **space.info(),
            "concepts": [asdict(c) for c in space.concepts],
        },
        "projections": [p.to_dict(include_embedding=store_embeddings)
                        for p in projections],
        "skipped": [asdict(s) for s in skipped],
    }
    payload["content_hash"] = content_hash(payload)
    return payload


def write_sidecar(payload: dict[str, Any], path: str | Path) -> Path:
    """Write atomically: a crash mid-write must not leave a truncated sidecar."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    tmp.replace(out)
    return out


def read_sidecar(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def verify_sidecar(path: str | Path) -> tuple[bool, str, str]:
    """Recompute the hash of a stored sidecar.

    Returns `(matches, stored, recomputed)`. Detects hand-edits and confirms
    that two runs agree.
    """
    payload = read_sidecar(path)
    stored = str(payload.get("content_hash", ""))
    recomputed = content_hash(payload)
    return stored == recomputed, stored, recomputed
