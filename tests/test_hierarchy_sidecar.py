"""Unit tests for sidecar registration.

The contract that matters: CES output registers as a normal pdfdrill
capability, byte-compatibly, and never clobbers state written by another tool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptdrill.hierarchy.sidecar import (CES_FACT, capability_valid,
                                            content_hash, find_sidecar,
                                            has_fact, make_proof, params_hash,
                                            read_sidecar, register,
                                            verify_proof)


@pytest.fixture
def library_doc(tmp_path):
    """Self-contained layout, as used by ~/pdfdrill-library."""
    folder = tmp_path / "2209.00445"
    folder.mkdir()
    (folder / "2209.00445.drill.json").write_text(json.dumps({
        "pdf": "2209.00445.pdf", "pdfdrill_version": "0.4.0",
        "facts": ["MODEL_BUILT", "LATEX_INGESTED"],
        "evidence": {"bibkey": "2209.00445"},
    }), encoding="utf-8")
    docmodel = folder / "model.docmodel.json"
    docmodel.write_text(json.dumps({"objects": []}), encoding="utf-8")
    return docmodel


@pytest.fixture
def legacy_doc(tmp_path):
    """Legacy layout: paper.pdf.drill/ beside paper.pdf.drill.json."""
    blob = tmp_path / "paper.pdf.drill"
    blob.mkdir()
    (tmp_path / "paper.pdf.drill.json").write_text(
        json.dumps({"facts": ["MODEL_BUILT"]}), encoding="utf-8")
    docmodel = blob / "model.docmodel.json"
    docmodel.write_text(json.dumps({"objects": []}), encoding="utf-8")
    return docmodel


# --------------------------------------------------------------------------
# Locating the sidecar
# --------------------------------------------------------------------------

def test_self_contained_layout_is_found(library_doc):
    assert find_sidecar(library_doc).name == "2209.00445.drill.json"


def test_legacy_layout_is_found(legacy_doc):
    assert find_sidecar(legacy_doc).name == "paper.pdf.drill.json"


def test_absent_sidecar_defaults_to_the_modern_layout(tmp_path):
    folder = tmp_path / "newdoc"
    folder.mkdir()
    docmodel = folder / "model.docmodel.json"
    docmodel.write_text("{}", encoding="utf-8")
    assert find_sidecar(docmodel).name == "newdoc.drill.json"


# --------------------------------------------------------------------------
# Proof format, matching pdfdrill/proofs.py
# --------------------------------------------------------------------------

def test_proof_records_input_hashes(library_doc):
    proof = make_proof("conceptdrill", inputs=[library_doc])
    assert str(library_doc) in proof["inputs"]
    assert proof["inputs"][str(library_doc)].startswith("sha256:")


def test_proof_carries_the_documented_fields(library_doc):
    proof = make_proof("conceptdrill", inputs=[library_doc], params={"k": 1})
    assert set(proof) == {"produced_by", "inputs", "params_hash", "algo", "ts"}
    assert proof["produced_by"] == "conceptdrill"


def test_content_hash_is_prefixed_with_its_algorithm(library_doc):
    """pdfdrill records the algorithm so it can never be misread later."""
    assert content_hash(library_doc).split(":", 1)[0] == "sha256"


def test_content_hash_of_a_missing_file_is_none(tmp_path):
    assert content_hash(tmp_path / "gone") is None


def test_params_hash_is_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_changes_with_values():
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_verify_accepts_unchanged_inputs(library_doc):
    assert verify_proof(make_proof("conceptdrill", inputs=[library_doc]))


def test_verify_rejects_changed_inputs(library_doc):
    proof = make_proof("conceptdrill", inputs=[library_doc])
    library_doc.write_text(json.dumps({"objects": [{"id": "x"}]}), encoding="utf-8")
    assert not verify_proof(proof)


def test_verify_rejects_a_deleted_input(library_doc):
    proof = make_proof("conceptdrill", inputs=[library_doc])
    library_doc.unlink()
    assert not verify_proof(proof)


def test_proof_without_inputs_is_valid():
    """Nothing recorded means nothing can invalidate it -- pdfdrill's rule."""
    assert verify_proof(make_proof("conceptdrill"))


def test_a_foreign_algorithm_does_not_produce_a_false_negative():
    """A blake3 proof cannot be checked with sha256. Defer, do not cry wolf."""
    assert verify_proof({"inputs": {"/nonexistent": "blake3:deadbeef"}})


# --------------------------------------------------------------------------
# Registration is additive
# --------------------------------------------------------------------------

def test_fact_is_registered(library_doc):
    register(library_doc, ces_path="model.ces.json")
    assert has_fact(library_doc)


def test_existing_facts_are_preserved(library_doc):
    register(library_doc, ces_path="model.ces.json")
    facts = read_sidecar(find_sidecar(library_doc))["facts"]
    assert "MODEL_BUILT" in facts and "LATEX_INGESTED" in facts
    assert CES_FACT in facts


def test_existing_evidence_is_preserved(library_doc):
    register(library_doc, ces_path="model.ces.json")
    ev = read_sidecar(find_sidecar(library_doc))["evidence"]
    assert ev["bibkey"] == "2209.00445"


def test_unknown_keys_are_preserved(library_doc):
    """A sidecar is another tool's state; clobbering an unrecognised key would
    be worse than not writing at all."""
    before = read_sidecar(find_sidecar(library_doc))
    register(library_doc, ces_path="model.ces.json")
    after = read_sidecar(find_sidecar(library_doc))
    assert after["pdf"] == before["pdf"]
    assert after["pdfdrill_version"] == before["pdfdrill_version"]


def test_registering_twice_does_not_duplicate_the_fact(library_doc):
    register(library_doc, ces_path="model.ces.json")
    register(library_doc, ces_path="model.ces.json")
    assert read_sidecar(find_sidecar(library_doc))["facts"].count(CES_FACT) == 1


def test_evidence_is_namespaced(library_doc):
    """CES keys must not collide with another stage's evidence."""
    register(library_doc, ces_path="m.json", evidence={"sections": 24})
    ev = read_sidecar(find_sidecar(library_doc))["evidence"]
    assert ev["ces_sections"] == 24
    assert ev["ces_path"] == "m.json"


def test_registration_creates_a_sidecar_when_absent(tmp_path):
    folder = tmp_path / "newdoc"
    folder.mkdir()
    docmodel = folder / "model.docmodel.json"
    docmodel.write_text("{}", encoding="utf-8")
    assert register(docmodel, ces_path="model.ces.json") is not None
    assert has_fact(docmodel)


def test_registration_works_on_the_legacy_layout(legacy_doc):
    register(legacy_doc, ces_path="model.ces.json")
    assert has_fact(legacy_doc)
    assert (legacy_doc.parent.parent / "paper.pdf.drill.json").exists()


def test_no_temporary_file_is_left_behind(library_doc):
    register(library_doc, ces_path="model.ces.json")
    assert list(find_sidecar(library_doc).parent.glob("*.tmp")) == []


# --------------------------------------------------------------------------
# Capability validity replaces hand-rolled staleness
# --------------------------------------------------------------------------

def test_capability_is_valid_right_after_registering(library_doc):
    register(library_doc, ces_path="model.ces.json")
    assert capability_valid(library_doc)


def test_capability_is_invalid_after_a_redrill(library_doc):
    """The whole point: a rebuilt document invalidates CES output without
    anyone comparing mtimes."""
    register(library_doc, ces_path="model.ces.json")
    library_doc.write_text(json.dumps({"objects": [{"id": "new"}]}),
                           encoding="utf-8")
    assert not capability_valid(library_doc)


def test_capability_is_false_when_the_fact_is_absent(library_doc):
    assert not capability_valid(library_doc)


def test_a_fact_without_a_proof_is_trusted(library_doc):
    """Proofs only ever make a capability False, never invent one."""
    path = find_sidecar(library_doc)
    data = read_sidecar(path)
    data["facts"].append(CES_FACT)
    path.write_text(json.dumps(data), encoding="utf-8")
    assert capability_valid(library_doc)


def test_corrupt_sidecar_reads_as_empty_rather_than_raising(tmp_path):
    bad = tmp_path / "x.drill.json"
    bad.write_text("not json", encoding="utf-8")
    assert read_sidecar(bad) == {}
