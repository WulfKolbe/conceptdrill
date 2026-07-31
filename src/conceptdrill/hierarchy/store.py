"""Persisting CES artefacts into the drilled document's own folder.

The drill folder is the integration point. `pdfdrill` already writes
`model.docmodel.json` and `model.docpack.json` there, so this stage joins that
family as `model.ces.json` — same directory, same `model.<stage>.json`
convention, so a later tool finds it by looking where everything else is.

    ~/pdfdrill-library/2209.00445/
        model.docmodel.json     <- input, opened read-only, never modified
        model.docpack.json
        model.ces.json          <- written here

**Per-document only.** The marker tree and its summaries belong to one
document. The cross-document basis matrix does not: it is a property of a
corpus, not of any single drill folder, and writing it into one would make that
folder silently authoritative for all the others. It gets a separate corpus
store.

Every artefact records the identity of the docmodel it came from, so a stale
`model.ces.json` left beside a re-drilled document is detectable rather than
quietly wrong.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

#: Joins pdfdrill's `model.<stage>.json` family.
CES_FILENAME = "model.ces.json"

FORMAT = "conceptdrill.ces"
FORMAT_VERSION = 1

#: Excluded from `content_hash`: these legitimately differ between identical runs.
VOLATILE_KEYS = frozenset({"created_at", "content_hash", "elapsed_seconds"})


def drill_dir(docmodel_path: str | Path) -> Path:
    """The drill folder containing a docmodel."""
    return Path(docmodel_path).resolve().parent


def ces_path(docmodel_path: str | Path,
             output: Optional[str | Path] = None) -> Path:
    """Where this document's CES artefacts belong."""
    if output:
        return Path(output)
    return drill_dir(docmodel_path) / CES_FILENAME


def source_fingerprint(docmodel_path: str | Path) -> dict[str, Any]:
    """Identity of the input docmodel.

    Size and a digest of the object list rather than mtime: a re-drill that
    produces identical content should not invalidate the artefact, and a
    touched file should not either.
    """
    path = Path(docmodel_path)
    info: dict[str, Any] = {"path": str(path), "name": path.name}
    try:
        raw = path.read_bytes()
        info["bytes"] = len(raw)
        info["sha256"] = hashlib.sha256(raw).hexdigest()
    except Exception:
        info["bytes"] = None
        info["sha256"] = ""
    return info


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in sorted(obj.items())
                if k not in VOLATILE_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_strip_volatile(v) for v in obj]
    return obj


def content_hash(payload: Mapping[str, Any]) -> str:
    """sha256 over the payload minus volatile fields."""
    canonical = json.dumps(_strip_volatile(dict(payload)), sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_payload(tree, summaries: Optional[Mapping[str, Any]] = None, *,
                  summary_stats: Optional[Mapping[str, Any]] = None,
                  created_at: str = "") -> dict[str, Any]:
    """Assemble this document's CES artefact.

    The marker tree is stored in full: rebuilding it is cheap, but a consumer
    that only wants the concepts should not have to re-parse the docmodel and
    re-derive a hierarchy the DocModel does not actually store.
    """
    nodes = []
    for node in tree.iter_document_order():
        nodes.append({
            "id": node.id,
            "title": node.title,
            "title_raw": node.title_raw,
            "level": node.level,
            "flow_index": node.flow_index,
            "is_appendix": node.is_appendix,
            "parent_id": node.parent_id,
            "children": list(node.children),
            "lost_macros": list(node.lost_macros),
            "paragraph_ids": [p.id for p in node.paragraphs],
            "body_chars": len(node.body_text),
            "subtree_chars": len(tree.subtree_text(node.id)),
        })

    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "bibkey": tree.bibkey,
        "source": source_fingerprint(tree.source_path) if tree.source_path else {},
        "created_at": created_at,
        "marker_tree": {"stats": tree.stats(), "nodes": nodes,
                         "roots": list(tree.roots),
                         "orphan_paragraph_ids": [p.id for p in tree.orphans]},
    }

    if summaries is not None:
        payload["summaries"] = {
            sid: (asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s))
            for sid, s in summaries.items()
        }
        if summary_stats is not None:
            payload["summary_stats"] = dict(summary_stats)

    payload["content_hash"] = content_hash(payload)
    return payload


def write(payload: Mapping[str, Any], docmodel_path: str | Path,
          output: Optional[str | Path] = None) -> Path:
    """Write atomically into the drill folder.

    A crash mid-write must not leave a truncated `model.ces.json` beside a
    perfectly good docmodel.
    """
    target = ces_path(docmodel_path, output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    tmp.replace(target)
    return target


def read(docmodel_path: str | Path,
         output: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    """Read this document's artefact, or None if absent or unreadable."""
    target = ces_path(docmodel_path, output)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_stale(docmodel_path: str | Path,
             output: Optional[str | Path] = None) -> Optional[bool]:
    """Has the docmodel changed since the artefact was written?

    Returns None when there is no artefact to judge.

    **Prefer `sidecar.capability_valid`.** pdfdrill already answers this through
    facts and content-hash proofs, and that is the answer other tools consult.
    This remains for the case where an artefact exists but no sidecar does, and
    it deliberately agrees with the sidecar rather than competing with it.
    """
    payload = read(docmodel_path, output)
    if payload is None:
        return None
    stored = (payload.get("source") or {}).get("sha256", "")
    if not stored:
        return True
    return stored != source_fingerprint(docmodel_path).get("sha256", "")


def save(tree, summaries=None, *, summary_stats=None, created_at: str = "",
         output: Optional[str | Path] = None,
         register_capability: bool = True,
         params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Write the artefact into the drill folder AND register it in the sidecar.

    One call so the two never drift apart: an artefact on disk that no sidecar
    knows about is invisible to every other pdfdrill stage, and a fact without
    an artefact is a lie.
    """
    from . import sidecar as _sidecar

    docmodel_path = tree.source_path
    payload = build_payload(tree, summaries, summary_stats=summary_stats,
                            created_at=created_at)
    target = write(payload, docmodel_path, output)

    registered = None
    if register_capability and docmodel_path:
        stats = payload.get("marker_tree", {}).get("stats", {})
        registered = _sidecar.register(
            docmodel_path, ces_path=target,
            params=dict(params or {}),
            evidence={
                "markers": stats.get("markers"),
                "paragraphs": stats.get("paragraphs"),
                "summaries": len(payload.get("summaries") or {}),
                "content_hash": payload.get("content_hash"),
            })
    return {"path": target, "sidecar": registered,
            "content_hash": payload.get("content_hash")}


def verify(docmodel_path: str | Path,
           output: Optional[str | Path] = None) -> tuple[bool, str, str]:
    """Recompute a stored artefact's content hash. Detects hand-edits."""
    payload = read(docmodel_path, output)
    if payload is None:
        return False, "", ""
    stored = str(payload.get("content_hash", ""))
    return stored == content_hash(payload), stored, content_hash(payload)
