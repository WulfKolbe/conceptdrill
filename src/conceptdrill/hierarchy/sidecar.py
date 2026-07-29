"""Registering CES output in pdfdrill's sidecar, the document's state file.

pdfdrill already solves "what has been built for this document, and is it still
valid". Its sidecar (`<doc>.drill.json`) holds:

    facts         cumulative capability names -- MODEL_BUILT, LATEX_INGESTED
    capabilities  {fact: proof}, a proof recording the content-hash of every
                  input plus a params hash
    evidence      arbitrary per-document metadata

`capability_valid(fact)` re-hashes the recorded inputs, so a rebuilt document
invalidates the capability without anyone comparing mtimes. That is exactly the
staleness question this package had started answering for itself, worse.

So CES output registers as a normal capability, `CES_BUILT`, and stops
reinventing provenance. The proof format is reproduced byte-compatibly here
rather than imported: conceptdrill must not acquire a dependency on pdfdrill,
and the format is small, stable and documented in `pdfdrill/proofs.py`.

Two layouts exist and both are handled (see `pdfdrill.sidecar.blob_dir_for`):

    self-contained   2209.00445/2209.00445.drill.json   <- pdfdrill-library
    legacy           paper.pdf.drill.json               <- beside paper.pdf.drill/

Writing is additive and preserves everything already in the file. A sidecar is
another tool's state, and clobbering a key we do not understand would be worse
than not writing at all.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

#: The capability this package produces.
CES_FACT = "CES_BUILT"

#: Recorded in every proof so the algorithm can never be misread later.
#: pdfdrill prefers blake3 when installed and records which it used; sha256 is
#: always available and verifies identically because the prefix is stored.
_ALGO = "sha256"


def _hash_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def content_hash(path: str | Path) -> Optional[str]:
    """`<algo>:<hex>` of a file's bytes, or None if unreadable."""
    try:
        return _hash_bytes(Path(path).read_bytes())
    except OSError:
        return None


def params_hash(params: Optional[Mapping[str, Any]]) -> str:
    """Order-independent hash of a parameter mapping."""
    blob = json.dumps(dict(params or {}), sort_keys=True, ensure_ascii=False,
                      default=str)
    return _hash_bytes(blob.encode("utf-8"))


def make_proof(produced_by: str, inputs: Optional[Iterable] = None,
               params: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """A proof object in pdfdrill's format."""
    recorded: dict[str, str] = {}
    for path in (inputs or []):
        digest = content_hash(path)
        if digest is not None:
            recorded[str(path)] = digest
    return {
        "produced_by": produced_by,
        "inputs": recorded,
        "params_hash": params_hash(params),
        "algo": _ALGO,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def verify_proof(proof: Mapping[str, Any]) -> bool:
    """True iff every recorded input still hashes to its stored value.

    A proof with no inputs is valid: there is nothing that could invalidate it.
    This mirrors pdfdrill's `proofs.verify` exactly.
    """
    for path, want in (proof.get("inputs") or {}).items():
        got = content_hash(path)
        if got is None or got != want:
            # A proof written with blake3 cannot be checked with sha256. Say
            # "cannot verify" by treating it as invalid only when the algorithm
            # matches; otherwise defer rather than cry wolf.
            if str(want).split(":", 1)[0] != _ALGO:
                continue
            return False
    return True


def find_sidecar(docmodel_path: str | Path) -> Path:
    """The sidecar for the document owning this docmodel.

    Resolves both layouts. Returns the self-contained path when neither exists,
    which is the modern layout used by pdfdrill-library.
    """
    blob = Path(docmodel_path).resolve().parent
    self_contained = blob / f"{blob.name}.drill.json"
    if self_contained.exists():
        return self_contained
    # Legacy: blob dir `paper.pdf.drill/` has sidecar `paper.pdf.drill.json`.
    legacy = blob.parent / f"{blob.name}.json"
    if legacy.exists():
        return legacy
    return self_contained


def read_sidecar(path: str | Path) -> dict[str, Any]:
    """Load a sidecar, or an empty dict when absent or unreadable."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def register(docmodel_path: str | Path, *,
             ces_path: str | Path,
             produced_by: str = "conceptdrill",
             params: Optional[Mapping[str, Any]] = None,
             evidence: Optional[Mapping[str, Any]] = None,
             fact: str = CES_FACT,
             sidecar_path: Optional[str | Path] = None) -> Optional[Path]:
    """Record `CES_BUILT` in the document's sidecar. Additive.

    Returns the sidecar path, or None if it could not be written — a failure to
    register must not lose the artefact that was already produced.

    The docmodel is recorded as the proof's input, so re-drilling the document
    invalidates this capability automatically.
    """
    target = Path(sidecar_path) if sidecar_path else find_sidecar(docmodel_path)
    data = read_sidecar(target)

    facts = data.setdefault("facts", [])
    if isinstance(facts, list) and fact not in facts:
        facts.append(fact)

    caps = data.setdefault("capabilities", {})
    if isinstance(caps, dict):
        caps[fact] = make_proof(produced_by, inputs=[docmodel_path],
                                params=params)

    ev = data.setdefault("evidence", {})
    if isinstance(ev, dict):
        ev["ces_path"] = str(ces_path)
        for key, value in (evidence or {}).items():
            ev[f"ces_{key}" if not str(key).startswith("ces_") else key] = value

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                  default=str) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target
    except OSError:
        return None


def has_fact(docmodel_path: str | Path, fact: str = CES_FACT,
             sidecar_path: Optional[str | Path] = None) -> bool:
    data = read_sidecar(Path(sidecar_path) if sidecar_path
                        else find_sidecar(docmodel_path))
    return fact in (data.get("facts") or [])


def capability_valid(docmodel_path: str | Path, fact: str = CES_FACT,
                     sidecar_path: Optional[str | Path] = None) -> bool:
    """Is the capability held *and* still backed by matching inputs?

    A fact held without a proof is trusted, matching pdfdrill: proofs only ever
    make a capability False, never invent one.
    """
    data = read_sidecar(Path(sidecar_path) if sidecar_path
                        else find_sidecar(docmodel_path))
    if fact not in (data.get("facts") or []):
        return False
    proof = (data.get("capabilities") or {}).get(fact)
    if not proof:
        return True
    return verify_proof(proof)
