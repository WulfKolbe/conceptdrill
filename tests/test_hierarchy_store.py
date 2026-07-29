"""Unit tests for drill-folder persistence.

The load-bearing guarantee: artefacts land beside `model.docmodel.json`, and
that input is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptdrill.hierarchy.docmodel_tree import build_tree
from conceptdrill.hierarchy.store import (CES_FILENAME, build_payload,
                                          ces_path, content_hash, drill_dir,
                                          is_stale, read, source_fingerprint,
                                          verify, write)
from conceptdrill.hierarchy.summarize import ExtractiveSummarizer, summarize_tree

DOC = {"meta": {"bibkey": "2209.00445"}, "objects": [
    {"id": "s1", "type": "Section",
     "props": {"caption": "Method", "level": 2, "flow_index": 1}},
    {"id": "p1", "type": "Paragraph",
     "props": {"text": "The method embeds each section once.",
               "flow_index": 2, "parent_section": "s1"}},
    {"id": "s2", "type": "Section",
     "props": {"caption": "Scoring", "level": 3, "flow_index": 3}},
    {"id": "p2", "type": "Paragraph",
     "props": {"text": "Scoring combines weight and coverage.",
               "flow_index": 4, "parent_section": "s2"}},
]}


@pytest.fixture
def drill(tmp_path):
    """A drill folder shaped like the real ones."""
    folder = tmp_path / "2209.00445"
    folder.mkdir()
    path = folder / "model.docmodel.json"
    path.write_text(json.dumps(DOC), encoding="utf-8")
    return path


@pytest.fixture
def tree(drill):
    return build_tree(json.loads(drill.read_text()), str(drill))


# --------------------------------------------------------------------------
# Placement — the integration requirement
# --------------------------------------------------------------------------

def test_output_lands_in_the_drill_folder(drill, tree):
    write(build_payload(tree), drill)
    assert (drill.parent / CES_FILENAME).exists()


def test_filename_follows_the_model_stage_convention(drill):
    """pdfdrill writes model.docmodel.json and model.docpack.json there."""
    assert ces_path(drill).name == "model.ces.json"
    assert ces_path(drill).name.startswith("model.")


def test_drill_dir_is_the_docmodel_parent(drill):
    assert drill_dir(drill) == drill.parent


def test_output_sits_beside_the_docmodel(drill, tree):
    target = write(build_payload(tree), drill)
    assert target.parent == drill.parent


def test_explicit_output_path_overrides(drill, tree, tmp_path):
    other = tmp_path / "elsewhere.json"
    assert write(build_payload(tree), drill, other) == other


# --------------------------------------------------------------------------
# The input must never be touched
# --------------------------------------------------------------------------

def test_docmodel_is_left_byte_identical(drill, tree):
    before = drill.read_bytes()
    write(build_payload(tree), drill)
    assert drill.read_bytes() == before


def test_writing_twice_does_not_touch_the_docmodel(drill, tree):
    before = drill.read_bytes()
    write(build_payload(tree), drill)
    write(build_payload(tree), drill)
    assert drill.read_bytes() == before


# --------------------------------------------------------------------------
# Payload contents
# --------------------------------------------------------------------------

def test_payload_carries_the_section_tree(drill, tree):
    payload = build_payload(tree)
    ids = [n["id"] for n in payload["section_tree"]["nodes"]]
    assert ids == ["s1", "s2"]


def test_payload_records_hierarchy_and_sizes(drill, tree):
    nodes = {n["id"]: n for n in build_payload(tree)["section_tree"]["nodes"]}
    assert nodes["s2"]["parent_id"] == "s1"
    assert nodes["s1"]["children"] == ["s2"]
    assert nodes["s1"]["subtree_chars"] > nodes["s1"]["body_chars"]


def test_payload_carries_the_bibkey(drill, tree):
    assert build_payload(tree)["bibkey"] == "2209.00445"


def test_summaries_are_optional(drill, tree):
    assert "summaries" not in build_payload(tree)


def test_summaries_are_stored_when_given(drill, tree):
    run = summarize_tree(tree, ExtractiveSummarizer())
    payload = build_payload(tree, run.summaries, summary_stats=run.stats())
    assert set(payload["summaries"]) == {"s1", "s2"}
    assert payload["summaries"]["s1"]["label"]
    assert payload["summary_stats"]["sections"] == 2


def test_round_trip_through_disk(drill, tree):
    run = summarize_tree(tree, ExtractiveSummarizer())
    write(build_payload(tree, run.summaries), drill)
    got = read(drill)
    assert got["summaries"]["s1"]["label"] == run.summaries["s1"].label


def test_read_returns_none_when_absent(drill):
    assert read(drill) is None


def test_read_survives_a_corrupt_artefact(drill):
    (drill.parent / CES_FILENAME).write_text("not json", encoding="utf-8")
    assert read(drill) is None


# --------------------------------------------------------------------------
# Integrity and staleness
# --------------------------------------------------------------------------

def test_content_hash_ignores_the_timestamp():
    a = {"x": 1, "created_at": "2020-01-01"}
    b = {"x": 1, "created_at": "2026-07-29"}
    assert content_hash(a) == content_hash(b)


def test_content_hash_notices_real_change():
    assert content_hash({"x": 1}) != content_hash({"x": 2})


def test_verify_accepts_an_untouched_artefact(drill, tree):
    write(build_payload(tree), drill)
    assert verify(drill)[0]


def test_verify_detects_a_hand_edit(drill, tree):
    write(build_payload(tree), drill)
    target = drill.parent / CES_FILENAME
    payload = json.loads(target.read_text())
    payload["bibkey"] = "tampered"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert not verify(drill)[0]


def test_fresh_artefact_is_not_stale(drill, tree):
    write(build_payload(tree), drill)
    assert is_stale(drill) is False


def test_artefact_is_stale_after_a_redrill(drill, tree):
    """A re-drilled document must not keep silently-wrong CES output."""
    write(build_payload(tree), drill)
    changed = dict(DOC)
    changed["objects"] = DOC["objects"] + [
        {"id": "s3", "type": "Section",
         "props": {"caption": "New", "level": 2, "flow_index": 9}}]
    drill.write_text(json.dumps(changed), encoding="utf-8")
    assert is_stale(drill) is True


def test_staleness_is_unknown_without_an_artefact(drill):
    assert is_stale(drill) is None


def test_fingerprint_uses_content_not_mtime(drill):
    """Touching a file must not invalidate an artefact."""
    first = source_fingerprint(drill)
    drill.touch()
    assert source_fingerprint(drill)["sha256"] == first["sha256"]


def test_fingerprint_of_a_missing_file_is_empty(tmp_path):
    assert source_fingerprint(tmp_path / "gone.json")["sha256"] == ""


# --------------------------------------------------------------------------
# Atomicity
# --------------------------------------------------------------------------

def test_no_temporary_file_is_left_behind(drill, tree):
    write(build_payload(tree), drill)
    leftovers = list(drill.parent.glob("*.tmp"))
    assert leftovers == []
