"""Verification against Loogle, with a fake client. No network in the suite.

The distinction these tests exist to protect: **"we could not reach the
service" is not "the name does not exist".** One is a transport failure and
must raise; the other is an answer and must be recorded.
"""
from __future__ import annotations

import pytest

from conceptdrill.mathlayer import (LoogleUnavailable, anchor, confidence,
                                    evidence, mathconcept_record, relation,
                                    symbol, verified_against, verify,
                                    verify_name)


class FakeLoogle:
    """Replies from a table. Anything not in it is an unknown identifier."""

    def __init__(self, table):
        self.table = table
        self.asked = []

    def query(self, pattern):
        self.asked.append(pattern)
        if pattern in self.table:
            count, module = self.table[pattern]
            return {"error": None, "count": count,
                    "hits": [{"name": pattern, "module": module}]}
        return {"error": f"unknown identifier '{pattern}'", "count": 0,
                "hits": []}


class DeadLoogle:
    def query(self, pattern):
        raise LoogleUnavailable("connection refused")


REAL = {
    "Module": (25323, "Mathlib.Algebra.Module.Defs"),
    "LinearMap": (11118, "Mathlib.Algebra.Module.LinearMap.Defs"),
    "Module.Basis": (1493, "Mathlib.LinearAlgebra.Basis.Defs"),
    "Module.finrank": (1539, "Mathlib.LinearAlgebra.Dimension.Finrank"),
}


def record(concept, anchors):
    return mathconcept_record(
        concept=concept,
        symbols=[symbol("T", status="observed")],
        related=[relation("domain", "vector space", status="observed")],
        lean=anchors,
        evidence=evidence(doc_id="fixture", span_ids=["s3"]))


# --------------------------------------------------------------------------
# One name
# --------------------------------------------------------------------------

def test_a_real_name_becomes_supported_with_its_module():
    got = verify_name("Module.Basis", FakeLoogle(REAL))
    assert got["status"] == "supported"
    assert got["hits"] == 1493
    assert got["module"] == "Mathlib.LinearAlgebra.Basis.Defs"


def test_a_stale_name_becomes_unresolved():
    """Measured against live Loogle: `Basis` is unknown, `Module.Basis` is not."""
    got = verify_name("Basis", FakeLoogle(REAL))
    assert got["status"] == "unresolved"


def test_the_error_text_is_kept_because_it_is_the_clue():
    """"unknown identifier 'Basis'" is what leads a reader to Module.Basis."""
    got = verify_name("FiniteDimensional.finrank", FakeLoogle(REAL))
    assert "unknown identifier" in got["error"]


def test_an_unreachable_service_raises_rather_than_reporting_absence():
    with pytest.raises(LoogleUnavailable):
        verify_name("Module", DeadLoogle())


# --------------------------------------------------------------------------
# Whole records
# --------------------------------------------------------------------------

def test_verification_replaces_every_unverified_anchor():
    before = record("basis", [anchor("Basis"), anchor("Module.Basis")])
    assert {a["status"] for a in before["lean"]} == {"unverified"}
    after = verify([before], FakeLoogle(REAL))[0]
    assert [a["status"] for a in after["lean"]] == ["unresolved", "supported"]


def test_a_not_expected_anchor_is_never_queried():
    """The domain of a map is a type index. Looking it up would turn a fact
    about Mathlib's design into a failed lookup."""
    client = FakeLoogle(REAL)
    before = record("domain", [anchor("nothing", status="not_expected")])
    after = verify([before], client)[0]
    assert client.asked == []
    assert after["lean"][0]["status"] == "not_expected"


def test_confidence_is_recomputed_after_verification():
    before = record("basis", [anchor("Basis")])
    after = verify([before], FakeLoogle(REAL))[0]
    assert after["confidence"] == confidence(after)
    assert after["confidence"] < before["confidence"]


def test_verification_does_not_mutate_the_input():
    before = record("basis", [anchor("Basis")])
    verify([before], FakeLoogle(REAL))
    assert before["lean"][0]["status"] == "unverified"


def test_a_dead_service_stops_the_pass_rather_than_marking_everything_absent():
    with pytest.raises(LoogleUnavailable):
        verify([record("basis", [anchor("Basis")])], DeadLoogle())


def test_progress_reports_every_name_checked():
    seen = []
    verify([record("basis", [anchor("Basis"), anchor("Module.Basis")])],
           FakeLoogle(REAL), progress=lambda n, r: seen.append((n, r["status"])))
    assert seen == [("Basis", "unresolved"), ("Module.Basis", "supported")]


def test_provenance_needs_a_date_to_mean_anything():
    """Loogle tracks Mathlib master, so the endpoint alone identifies nothing."""
    assert verified_against(when="2026-08-16").endswith("2026-08-16")
    assert "loogle" in verified_against()
