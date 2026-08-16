"""The `MathConcept` record and its closed vocabularies.

One record names one mathematical concept found in one span, the symbols that
stand for it, what the text says about it, and where it might live in Mathlib.

The shape is `runlog.py`'s, for the reason `runlog.py` gives: **an unknown key
raises and a missing key becomes `None`**, because `null` is a measurement (the
text named no symbol) while an absent key is an unanswered question. That
module caught a real bug this way — `bool(strict_mode)` silently turning `None`
into `False` — and the same class of mistake is available here.

## Status is the whole point

Every symbol and every relation carries where it came from:

    observed   the string is in the text
    implied    it follows from a phrase or notation in the text
    predicted  concepts like this usually carry it, and it is NOT here
    external   true of the concept, absent from this text

Collapsing these is how an ontology fills with facts nobody wrote, and it is
also how a prediction launders itself into an observation — see
`docs/math-layer-ocr.md` §17. Anything this layer contributed must stay
re-readable as `predicted` forever.

## No embeddings

`confidence()` is arithmetic over declared statuses. Step 15 measured that
cosine similarity into the corpus basis is worse than uninformative for this
task — one basis row was top-1 for four of six mathematical continuations — so
nothing here touches an embedder, a basis or a projection.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional, Sequence

FORMAT = "conceptdrill.mathlayer.concepts"
FORMAT_VERSION = 1

#: Every key on every MathConcept, in write order. The contract.
MATHCONCEPT_FIELDS: tuple[str, ...] = (
    "id", "concept", "spoken", "latex", "symbols", "related",
    "lean", "loogle", "evidence", "confidence",
)

#: Relations a concept may stand in. Closed, and checked on write.
#:
#: `related-to` is the escape hatch. A layer emitting mostly `related-to` has
#: produced a similarity graph, which CES already is, and has added nothing.
RELATIONS = frozenset({
    "is-a", "has-dimension", "over-field", "maps-to",
    "domain", "codomain", "related-to",
})

#: Where a symbol or a relation came from. See the module docstring.
EVIDENCE_STATUS = frozenset({"observed", "implied", "predicted", "external"})

#: What a proposed Lean name is worth.
#:
#: `not_expected` is distinct from `unresolved` on purpose, exactly as
#: `absorbed` is distinct from `not_integrated` in `runlog.py`: the domain of a
#: linear map has no Mathlib declaration because it is a type index, and that
#: is a fact about Mathlib, not a failed lookup. Scoring the two the same would
#: punish a concept for someone else's design.
ANCHOR_STATUS = frozenset({"supported", "unresolved", "unverified", "not_expected"})

#: What a symbol does for its concept. Advisory: not checked, because a closed
#: list here would be guessing at mathematics rather than at this pipeline.
SYMBOL_ROLES = ("instance", "value", "signature", "derived", "operator",
                "application", "representation")


class SchemaViolation(ValueError):
    """A record that breaks the contract in this module."""


def canonical_concept(name: str) -> str:
    """The form identity is computed from. Case and spacing are not meaning."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def concept_id(name: str) -> str:
    """Content-addressed identity, stable across reordering and re-extraction.

    Positional ids transpose the moment a document is re-read, which is the
    argument `basis.make_row_id` already makes for basis rows.
    """
    payload = canonical_concept(name)
    return "mc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# The parts
# --------------------------------------------------------------------------

def symbol(latex: str, *, role: str = "", status: str = "predicted"
           ) -> dict[str, Any]:
    """One symbol standing for a concept.

    `latex` is preserved EXACTLY and never normalised. `T \\colon V
    \\rightarrow W` and `T : V \\to W` are two entries, not one: which spelling
    an author used is evidence about the document, and SRE speaks both
    identically, so the spoken form cannot recover it.
    """
    if not isinstance(latex, str) or not latex.strip():
        raise SchemaViolation("a symbol needs non-empty latex")
    if status not in EVIDENCE_STATUS:
        raise SchemaViolation(
            f"symbol status {status!r} not in {sorted(EVIDENCE_STATUS)}")
    return {"latex": latex, "role": role or None, "status": status}


def relation(kind: str, target: str, *, status: str = "implied"
             ) -> dict[str, Any]:
    """One edge. `kind` must be in `RELATIONS`; `status` in `EVIDENCE_STATUS`."""
    if kind not in RELATIONS:
        raise SchemaViolation(
            f"relation {kind!r} not in {sorted(RELATIONS)}")
    if status not in EVIDENCE_STATUS:
        raise SchemaViolation(
            f"relation status {status!r} not in {sorted(EVIDENCE_STATUS)}")
    if not isinstance(target, str) or not target.strip():
        raise SchemaViolation("a relation needs a target")
    return {"relation": kind, "target": target, "status": status}


def anchor(name: str, *, status: str = "unverified",
           hits: Optional[int] = None, module: str = "",
           error: str = "") -> dict[str, Any]:
    """One candidate Lean/Mathlib declaration.

    `unverified` is the default and is not a formality. Measured in step 11:
    two of five names asserted as exact did not exist — `Basis` is
    `Module.Basis`, `FiniteDimensional.finrank` is `Module.finrank` — both
    stale in the same direction. A bare string list would have read as
    confirmed.
    """
    if status not in ANCHOR_STATUS:
        raise SchemaViolation(
            f"anchor status {status!r} not in {sorted(ANCHOR_STATUS)}")
    if not isinstance(name, str) or not name.strip():
        raise SchemaViolation("an anchor needs a name")
    return {"name": name, "status": status, "hits": hits,
            "module": module or None, "error": error or None}


def evidence(*, doc_id: str = "", span_ids: Sequence[str] = (),
             ces_row_ids: Sequence[str] = (), quote: Optional[str] = None,
             note: str = "") -> dict[str, Any]:
    """Where a concept came from.

    `ces_row_ids` is PROVENANCE ONLY and nothing in this package reads it.
    Step 15 measured CES retrieval as worse than uninformative for mathematics
    — the wrong answers scored higher than the right ones — so a CES
    coordinate must never rank, gate or select a MathConcept.
    """
    return {"doc_id": doc_id or None,
            "span_ids": list(span_ids),
            "ces_row_ids": list(ces_row_ids),
            "quote": quote,
            "note": note or None}


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

#: Weights of the three terms. Declared here so a change to the number is a
#: change to a constant that tests read, not a judgement inside a function.
WEIGHT_EVIDENCE = 0.5
WEIGHT_SYMBOL = 0.3
WEIGHT_ANCHOR = 0.2

_EVIDENCE_SCORE = {"observed": 1.0, "implied": 0.6,
                   "predicted": 0.2, "external": 0.2}

#: What an anchor contributes. **`unverified` sits between the two**, and that
#: is load-bearing: if an unchecked name scored the same as one Loogle
#: rejected, running `verify()` could only ever leave confidence flat or raise
#: it, and a verification pass that cannot lower a score is not a check.
#:
#: It is also the `absorbed` / `not_integrated` distinction again. "Nobody
#: looked" and "we looked and it is not there" are different states, and a
#: scalar that collapses them hides which one it is in.
_ANCHOR_SCORE = {"supported": 1.0, "unverified": 0.5, "unresolved": 0.0}


def confidence(record: Mapping[str, Any]) -> float:
    """Support for this concept, in `[0, 1]`. Never a model's self-report.

        0.5 * evidence + 0.3 * symbol + 0.2 * anchor

    `evidence` is the *strongest* relation status, because one observed
    relation is enough to establish that the text talks about the concept, and
    averaging would let a pile of `external` background facts dilute it.

    The anchor term takes the **best** status present, scored by
    `_ANCHOR_SCORE`: verified support beats an unchecked guess, which beats a
    name Loogle rejected.

    **When no anchor is expected, the anchor term is dropped and the remaining
    weights renormalise.** A concept with no Mathlib declaration — the domain
    of a linear map is a type index, not a name — has not failed a lookup, and
    scoring it as though it had would be the `absorbed`/`not_integrated`
    conflation again.

    A value of 1.0 means every available check passed. It does NOT mean the
    concept is certainly correct.
    """
    statuses = [r.get("status") for r in (record.get("related") or [])]
    known = [s for s in statuses if s in _EVIDENCE_SCORE]
    evidence_score = max((_EVIDENCE_SCORE[s] for s in known), default=0.2)

    symbols = record.get("symbols") or []
    symbol_score = 1.0 if any(s.get("status") == "observed"
                              for s in symbols) else 0.0

    anchors = record.get("lean") or []
    expected = [a for a in anchors if a.get("status") != "not_expected"]
    if not anchors or all(a.get("status") == "not_expected" for a in anchors):
        total = WEIGHT_EVIDENCE + WEIGHT_SYMBOL
        return round((WEIGHT_EVIDENCE * evidence_score
                      + WEIGHT_SYMBOL * symbol_score) / total, 4)

    anchor_score = max(_ANCHOR_SCORE.get(a.get("status"), 0.0)
                       for a in expected)
    return round(WEIGHT_EVIDENCE * evidence_score
                 + WEIGHT_SYMBOL * symbol_score
                 + WEIGHT_ANCHOR * anchor_score, 4)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def mathconcept_record(**fields: Any) -> dict[str, Any]:
    """One MathConcept, with every field present.

    An unknown key raises; a missing key becomes `None`. `id` and `confidence`
    are derived when not supplied, so a caller cannot hand-write an id that
    disagrees with the concept name.
    """
    unknown = sorted(set(fields) - set(MATHCONCEPT_FIELDS))
    if unknown:
        raise SchemaViolation(
            f"unknown MathConcept fields {unknown}; "
            f"the contract is {list(MATHCONCEPT_FIELDS)}")

    name = fields.get("concept")
    if not isinstance(name, str) or not name.strip():
        raise SchemaViolation("a MathConcept needs a non-empty concept name")

    record = {key: fields.get(key) for key in MATHCONCEPT_FIELDS}
    record["concept"] = canonical_concept(name)
    record["id"] = fields.get("id") or concept_id(name)
    for key in ("latex", "symbols", "related", "lean", "loogle"):
        if record[key] is None:
            record[key] = []
    if record["evidence"] is None:
        record["evidence"] = evidence()
    if fields.get("confidence") is None:
        record["confidence"] = confidence(record)
    return record


def check_record(record: Mapping[str, Any]) -> list[str]:
    """Every way `record` breaks the contract. Empty means it is well formed."""
    problems: list[str] = []
    missing = [k for k in MATHCONCEPT_FIELDS if k not in record]
    if missing:
        problems.append(f"missing fields {missing}")
    unknown = sorted(set(record) - set(MATHCONCEPT_FIELDS))
    if unknown:
        problems.append(f"unknown fields {unknown}")

    name = record.get("concept")
    if not isinstance(name, str) or not name.strip():
        problems.append("concept name is empty")
    elif record.get("id") != concept_id(name):
        problems.append(
            f"id {record.get('id')!r} is not the content address of "
            f"{name!r} ({concept_id(name)})")

    for i, sym in enumerate(record.get("symbols") or []):
        if sym.get("status") not in EVIDENCE_STATUS:
            problems.append(f"symbol {i} status {sym.get('status')!r}")
        if not (sym.get("latex") or "").strip():
            problems.append(f"symbol {i} has no latex")
    for i, rel in enumerate(record.get("related") or []):
        if rel.get("relation") not in RELATIONS:
            problems.append(f"relation {i} kind {rel.get('relation')!r}")
        if rel.get("status") not in EVIDENCE_STATUS:
            problems.append(f"relation {i} status {rel.get('status')!r}")
    for i, anc in enumerate(record.get("lean") or []):
        if anc.get("status") not in ANCHOR_STATUS:
            problems.append(f"anchor {i} status {anc.get('status')!r}")

    score = record.get("confidence")
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        problems.append(f"confidence {score!r} is not in [0, 1]")
    elif not problems:
        # Confidence is DERIVED. A stored value that disagrees with the
        # function means the parts changed underneath it or someone typed a
        # number they preferred, and both make every comparison against it
        # meaningless. `corpus.py` refuses a basis that does not reproduce its
        # stored version for the same reason.
        # Only checked once the parts are known good, or a bad relation would
        # be reported twice.
        expected = confidence(record)
        if abs(float(score) - expected) > 1e-9:
            problems.append(
                f"confidence {float(score)} is not what the declared function "
                f"gives for this record ({expected})")
    return problems


def assert_record(record: Mapping[str, Any]) -> None:
    """Raise unless `record` is well formed."""
    problems = check_record(record)
    if problems:
        raise SchemaViolation(
            f"{record.get('concept')!r}: {problems}")
