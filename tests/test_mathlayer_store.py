"""Persisting MathConcepts, and refusing files this release did not write."""
from __future__ import annotations

import json

import pytest

from conceptdrill.mathlayer import (MathConceptStore, SchemaViolation,
                                    StoreMismatch, anchor, evidence,
                                    mathconcept_record, merge, relation, symbol)


def record(concept="linear map", **over):
    base = dict(
        concept=concept,
        symbols=[symbol("T", status="observed")],
        related=[relation("domain", "vector space", status="observed")],
        lean=[anchor("LinearMap", status="supported", hits=11118)],
        evidence=evidence(doc_id="fixture", span_ids=["s3"]))
    base.update(over)
    return mathconcept_record(**base)


def test_a_saved_file_round_trips(tmp_path):
    store = MathConceptStore(tmp_path)
    store.save([record()], doc_id="fixture", extractor="test")
    assert store.concepts() == [record()]


def test_the_write_is_atomic(tmp_path):
    """A `.tmp` left behind means a crash mid-write would have been visible."""
    store = MathConceptStore(tmp_path)
    store.save([record()])
    assert store.file.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_an_invalid_record_never_reaches_disk(tmp_path):
    store = MathConceptStore(tmp_path)
    bad = record()
    bad["confidence"] = 7.0
    with pytest.raises(SchemaViolation):
        store.save([bad])
    assert not store.exists()


def test_a_foreign_file_is_refused(tmp_path):
    store = MathConceptStore(tmp_path)
    store.path.mkdir(parents=True, exist_ok=True)
    store.file.write_text(json.dumps({"format": "something.else",
                                      "concepts": []}), encoding="utf-8")
    with pytest.raises(StoreMismatch, match="format"):
        store.load()


def test_a_future_format_version_is_refused(tmp_path):
    store = MathConceptStore(tmp_path)
    store.save([record()])
    payload = json.loads(store.file.read_text(encoding="utf-8"))
    payload["format_version"] = 99
    store.file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StoreMismatch, match="format version"):
        store.load()


def test_a_corrupted_record_is_caught_on_read(tmp_path):
    """Validated on read as well as write: a file may predate this release."""
    store = MathConceptStore(tmp_path)
    store.save([record()])
    payload = json.loads(store.file.read_text(encoding="utf-8"))
    payload["concepts"][0]["related"][0]["relation"] = "is-basically"
    store.file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaViolation):
        store.load()


def test_verification_provenance_survives(tmp_path):
    """A `supported` anchor is meaningless without saying what checked it."""
    store = MathConceptStore(tmp_path)
    store.save([record()], verified_against="loogle on 2026-08-16")
    assert store.load()["verified_against"] == "loogle on 2026-08-16"


def test_an_unverified_run_says_so(tmp_path):
    store = MathConceptStore(tmp_path)
    store.save([record()])
    assert store.load()["verified_against"] is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MathConceptStore(tmp_path).load()


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def test_two_spans_naming_one_concept_become_one_record():
    a = record(evidence=evidence(doc_id="d", span_ids=["s1"]))
    b = record(evidence=evidence(doc_id="d", span_ids=["s7"]))
    merged = merge([a, b])
    assert len(merged) == 1
    assert merged[0]["evidence"]["span_ids"] == ["s1", "s7"]


def test_merging_keeps_both_spellings_of_a_symbol():
    a = record(symbols=[symbol(r"T : V \to W", status="observed")])
    b = record(symbols=[symbol(r"T \colon V \rightarrow W", status="observed")])
    latex = {s["latex"] for s in merge([a, b])[0]["symbols"]}
    assert latex == {r"T : V \to W", r"T \colon V \rightarrow W"}


def test_the_same_symbol_observed_and_predicted_stays_two_facts():
    """Collapsing them would either lose the observation or invent one."""
    a = record(symbols=[symbol(r"\ker T", status="observed")])
    b = record(symbols=[symbol(r"\ker T", status="predicted")])
    statuses = sorted(s["status"] for s in merge([a, b])[0]["symbols"])
    assert statuses == ["observed", "predicted"]


def test_distinct_concepts_are_not_merged():
    assert len(merge([record("linear map"), record("vector space")])) == 2


def test_confidence_is_recomputed_after_a_merge():
    """It reads anchor and symbol status, so it cannot survive the parts
    changing underneath it."""
    weak = record(symbols=[symbol("T", status="predicted")],
                  related=[relation("related-to", "x", status="external")],
                  lean=[anchor("Nope", status="unresolved")])
    strong = record()
    merged = merge([weak, strong])[0]
    assert merged["confidence"] == 1.0
    assert merged["confidence"] > weak["confidence"]


def test_merging_does_not_mutate_its_input():
    a = record(evidence=evidence(doc_id="d", span_ids=["s1"]))
    before = json.dumps(a, sort_keys=True)
    merge([a, record(evidence=evidence(doc_id="d", span_ids=["s2"]))])
    assert json.dumps(a, sort_keys=True) == before


def test_merged_output_is_deterministically_ordered():
    """Two runs must produce byte-identical files."""
    records = [record("linear map"), record("basis"), record("vector space")]
    assert [r["id"] for r in merge(records)] == \
        [r["id"] for r in merge(list(reversed(records)))]
