"""Copy a run's per-document results into each document's library folder.

Owner, 2026-11-01 (T-0117): "we need at least a copy of the plain results in
the folder of each document in pdfdrill-library. The final storage location
will be given later." So this is a copy, not the store: the run directory under
`output_root()` stays the original, and the library gets

    <library>/<doc>/conceptdrill/<run_id>/<doc>.json
    <library>/<doc>/conceptdrill/<run_id>/<doc>.md
    <library>/<doc>/conceptdrill/<run_id>/provenance.json

Nothing else in the document folder is touched, and a document whose folder
does not exist is reported and skipped -- creating it would invent a library
entry pdfdrill never drilled.

What made the files is read from the run's `manifest.json`, written by the run
itself with `write_manifest`. A run that was not made by conceptdrill says so
in `produced_by`; the provenance repeats it rather than dressing it up.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

MANIFEST = "manifest.json"
PROVENANCE = "provenance.json"
SUBFOLDER = "conceptdrill"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(run_dir: Path, *, run_id: str, command: str,
                   commit: Optional[str],
                   documents: Mapping[str, Iterable[Path]],
                   produced_by: str = "conceptdrill",
                   producer: Optional[dict] = None,
                   skipped: Iterable[dict] = (),
                   usage: Optional[dict] = None) -> Path:
    """Record what made this run, with each document's inputs and their sha256.

    Inputs are hashed now, while they are the files the run read. An input
    given as a mapping is recorded as it is: that is how a run recorded after
    the fact says its input was not hashed, rather than hashing today's file.
    """
    manifest = {
        "run_id": run_id,
        "command": command,
        "conceptdrill_commit": commit,
        "produced_by": produced_by,
        "producer": producer,
        "documents": {doc: [dict(p) if isinstance(p, Mapping)
                            else {"path": str(p), "sha256": _sha256(p)}
                            for p in inputs]
                      for doc, inputs in documents.items()},
        "skipped": list(skipped),
        "usage": usage,
        "written_at": _now(),
    }
    path = Path(run_dir) / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def copy_to_library(run_dir: Path, library: Path) -> dict:
    """Copy every document of `run_dir` into `library`. Returns what happened.

    Idempotent: copying the same run again rewrites identical bytes, keeping
    the first copy's `written_at` when nothing else in the provenance changed.
    """
    run_dir, library = Path(run_dir), Path(library)
    manifest_path = run_dir / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: no {MANIFEST}, so nothing "
                                f"says what made this run; not copied")
    manifest = json.loads(manifest_path.read_text())
    run_id = manifest["run_id"]
    copied, skipped = [], []
    for doc, inputs in sorted(manifest["documents"].items()):
        sources = [run_dir / f"{doc}.json", run_dir / f"{doc}.md"]
        if not (library / doc).is_dir():
            skipped.append({"doc": doc, "reason": f"no folder {library / doc}"})
            continue
        missing = [s.name for s in sources if not s.is_file()]
        if missing:
            skipped.append({"doc": doc, "reason": f"run has no {', '.join(missing)}"})
            continue
        dest = library / doc / SUBFOLDER / run_id
        dest.mkdir(parents=True, exist_ok=True)
        for src in sources:
            data = src.read_bytes()
            target = dest / src.name
            if not target.is_file() or target.read_bytes() != data:
                target.write_bytes(data)
        prov = {
            "run_id": run_id,
            "command": manifest["command"],
            "conceptdrill_commit": manifest["conceptdrill_commit"],
            "produced_by": manifest["produced_by"],
            "producer": manifest.get("producer"),
            "run_dir": str(run_dir),
            "inputs": inputs,
            "files": [{"name": s.name, "sha256": _sha256(s)} for s in sources],
        }
        prov_path = dest / PROVENANCE
        old = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
        written_at = old.pop("written_at", None)
        prov["written_at"] = written_at if old == prov and written_at else _now()
        prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
        copied.append(doc)
    return {"run_id": run_id, "copied": copied, "skipped": skipped}
