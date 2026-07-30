"""The run-record contract.

These tests are the reason a run is auditable: they assert that a record cannot
be written with a field missing, that a run cannot be written when it has lost
track of a section, and that the manifest cannot claim not to know what produced
it.
"""
from __future__ import annotations

import json

import pytest

from conceptdrill.hierarchy.captions import (caption_cleaner_tier,
                                             clean_caption,
                                             clean_caption_traced)
from conceptdrill.hierarchy.runlog import (MANIFEST_FIELDS, MANIFEST_REQUIRED,
                                           SECTION_FIELDS, IncompleteRun,
                                           RunLog, gemm_state, section_record)


# --------------------------------------------------------------------------
# The section record
# --------------------------------------------------------------------------

def test_a_record_has_every_field_even_when_nothing_is_known():
    rec = section_record()
    assert set(rec) == set(SECTION_FIELDS)
    assert all(v is None for v in rec.values())


def test_absent_is_not_allowed_but_null_is():
    """The distinction the whole contract rests on: null is a measurement,
    a missing key is an unanswered question."""
    rec = section_record(doc_id="d", section_id="s", tier_label=None)
    assert "tier_label" in rec and rec["tier_label"] is None
    assert "structural_class" in rec


def test_an_unknown_field_raises_rather_than_being_dropped():
    with pytest.raises(KeyError, match="contract"):
        section_record(doc_id="d", tier_lable="typo")


def test_field_order_is_the_declared_order():
    assert list(section_record()) == list(SECTION_FIELDS)


def test_an_unknown_merge_decision_raises():
    with pytest.raises(ValueError, match="merge_decision"):
        section_record(merge_decision="probably-merged")


@pytest.mark.parametrize("decision",
                         ["added", "merged", "skipped", "not_integrated"])
def test_the_decision_vocabulary_is_accepted(decision):
    assert section_record(merge_decision=decision)["merge_decision"] == decision


# --------------------------------------------------------------------------
# Caption tracing
# --------------------------------------------------------------------------

def test_clean_caption_is_unchanged_by_tracing():
    for raw in ["", "  ", "Plain Title", r"\emph{Siblings} score",
                r"\ALG\ Application", r"Testing the $\tau$ function"]:
        assert clean_caption(raw) == clean_caption_traced(raw)[0]


def test_tracing_names_the_cleaner_that_ran():
    _, rules = clean_caption_traced(r"\emph{Siblings} score")
    assert rules[0] in {"pylatexenc"} or rules[0].startswith("regex-fallback")


def test_an_empty_caption_is_reported_as_empty():
    assert clean_caption_traced("   ") == ("", ("empty",))


def test_the_cleaner_tier_is_one_of_two_known_values():
    assert caption_cleaner_tier() in {"pylatexenc", "regex"}


# --------------------------------------------------------------------------
# The run directory
# --------------------------------------------------------------------------

def finish(log, **over):
    kwargs = dict(summarizer_class="StubSummarizer", embedder_backend="hash",
                  embedder_resolved_revision="rev0", nlp_backend="regex",
                  tau=0.65, strict_mode=True, corpus_paths=["/x/model.docmodel.json"],
                  doc_count=1, basis_rows=[])
    kwargs.update(over)
    return log.finish(**kwargs)


def test_a_run_directory_is_named_for_its_commit(tmp_path):
    log = RunLog.open(tmp_path, timestamp="20260730T120000Z")
    assert log.run_id.startswith("run-20260730T120000Z-")
    assert log.root.is_dir()


def test_the_three_files_are_written(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    root = finish(log)
    assert (root / "manifest.json").exists()
    assert (root / "sections.jsonl").exists()
    assert (root / "basis.json").exists()


def test_the_line_count_equals_the_manifest_section_count(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    for i in range(4):
        log.add_section(doc_id="d", section_id=f"s{i}")
    root = finish(log)
    lines = (root / "sections.jsonl").read_text().strip().splitlines()
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(lines) == manifest["section_count"] == 4


def test_every_written_line_carries_every_field(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1", tier_label="a label")
    root = finish(log)
    for line in (root / "sections.jsonl").read_text().strip().splitlines():
        assert set(json.loads(line)) == set(SECTION_FIELDS)


def test_losing_a_section_refuses_to_write(tmp_path):
    """The gate that makes the ledger mean something."""
    log = RunLog.open(tmp_path, timestamp="t")
    log.expect(3)
    log.add_section(doc_id="d", section_id="s1")
    with pytest.raises(IncompleteRun, match="not auditable"):
        finish(log)


def test_a_duplicated_section_refuses_to_write(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    log.add_section(doc_id="d", section_id="s1")
    with pytest.raises(IncompleteRun, match="duplicate"):
        finish(log)


def test_the_same_section_id_in_two_documents_is_not_a_duplicate(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="a", section_id="s1")
    log.add_section(doc_id="b", section_id="s1")
    assert finish(log)


@pytest.mark.parametrize("field", MANIFEST_REQUIRED)
def test_a_null_in_a_required_manifest_field_refuses_to_write(tmp_path, field):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    if field == "gemm_check_result":
        pytest.skip("measured, not caller-supplied")
    with pytest.raises(IncompleteRun, match="must not be null"):
        finish(log, **{field: None})


def test_the_manifest_carries_every_declared_field(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    manifest = json.loads((finish(log) / "manifest.json").read_text())
    assert set(MANIFEST_FIELDS) <= set(manifest)


def test_strict_mode_false_is_recorded_not_treated_as_missing(tmp_path):
    """`False` is an answer. A required-field check written with a falsiness
    test rather than an is-None test would reject it."""
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    manifest = json.loads((finish(log, strict_mode=False) / "manifest.json").read_text())
    assert manifest["strict_mode"] is False


def test_the_manifest_records_whether_the_tree_was_dirty(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1")
    manifest = json.loads((finish(log) / "manifest.json").read_text())
    assert "git_dirty" in manifest


def test_gemm_state_reports_a_verdict():
    state = gemm_state()
    assert state["verdict"] in {"green", "red", "unknown"} or \
        state["verdict"].startswith("unavailable")


def test_basis_rows_are_written_with_their_sections(tmp_path):
    log = RunLog.open(tmp_path, timestamp="t")
    log.add_section(doc_id="d", section_id="s1", row_id_assigned="row_a")
    root = finish(log, basis_rows=[{"row_id": "row_a", "label": "L", "support": 1,
                                    "level": 1, "documents": ["d"],
                                    "contributing_section_ids": ["s1"]}])
    rows = json.loads((root / "basis.json").read_text())["rows"]
    assert rows[0]["contributing_section_ids"] == ["s1"]


# --------------------------------------------------------------------------
# Gate 1 as a check over written artefacts
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.gates import gate1_persistence, read_run  # noqa: E402


def docmodel_with(tmp_path, doc_id, section_ids):
    """A minimal real docmodel, so gate 1 has an input to compare against."""
    doc = tmp_path / doc_id / "model.docmodel.json"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(json.dumps({"objects": [
        {"id": sid, "type": "Section",
         "props": {"caption": f"Section {i}", "level": 1, "flow_index": i}}
        for i, sid in enumerate(section_ids, start=1)]}))
    return doc


def written_run(tmp_path, records, **over):
    doc = docmodel_with(tmp_path, records[0]["doc_id"],
                        [r["section_id"] for r in records])
    log = RunLog.open(tmp_path, timestamp="t")
    for rec in records:
        log.add_section(**rec)
    over.setdefault("corpus_paths", [str(doc)])
    return finish(log, **over)


def test_gate1_passes_on_a_well_formed_run(tmp_path):
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"},
                                  {"doc_id": "d", "section_id": "s2"}])
    result = gate1_persistence(root)
    assert result.passed, result.report()
    assert result.checks["records"] == 2


def test_gate1_fails_when_the_line_count_disagrees_with_the_manifest(tmp_path):
    """Hand-edit the artefact: the gate must re-derive, not trust."""
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"}])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["section_count"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = gate1_persistence(root)
    assert not result.passed
    assert any("manifest says 99" in f for f in result.failures)


def test_gate1_fails_on_a_record_with_a_field_removed(tmp_path):
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"}])
    rec = json.loads((root / "sections.jsonl").read_text().strip())
    rec.pop("basis_text")
    (root / "sections.jsonl").write_text(json.dumps(rec) + "\n")
    result = gate1_persistence(root)
    assert not result.passed
    assert any("absent fields" in f for f in result.failures)


def test_gate1_fails_on_a_null_required_manifest_field(tmp_path):
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"}])
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["embedder_backend"] = None
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = gate1_persistence(root)
    assert not result.passed
    assert any("null" in f for f in result.failures)


def test_gate1_fails_when_a_section_of_the_input_has_no_record(tmp_path):
    """The clause that matters most: silent section loss."""
    root = written_run(tmp_path, [{"doc_id": "lib", "section_id": "s1"}])
    doc = docmodel_with(tmp_path, "lib", ["s1", "s2"])   # input gains a section
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["corpus_paths"] = [str(doc)]
    (root / "manifest.json").write_text(json.dumps(manifest))
    result = gate1_persistence(root)
    assert not result.passed
    assert any("no record" in f for f in result.failures)


def test_gate1_fails_when_the_manifest_names_no_inputs(tmp_path):
    """An un-evaluable clause must not be reported as a satisfied one."""
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"}],
                       corpus_paths=[])
    result = gate1_persistence(root)
    assert not result.passed
    assert any("cannot be evaluated" in f for f in result.failures)


def test_read_run_round_trips(tmp_path):
    root = written_run(tmp_path, [{"doc_id": "d", "section_id": "s1"}])
    manifest, records, basis = read_run(root)
    assert manifest["run_id"] == basis["run_id"]
    assert records[0]["section_id"] == "s1"


# --------------------------------------------------------------------------
# Gate 2 over written artefacts
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.gates import gate2_basis_text  # noqa: E402


def test_gate2_passes_on_clean_basis_text(tmp_path):
    root = written_run(tmp_path, [
        {"doc_id": "d", "section_id": "s1", "title_raw": "1 Introduction",
         "basis_text": "Temporal query intent classification for retrieval."}])
    result = gate2_basis_text(root)
    assert result.passed, result.report()
    assert result.checks["clean_fraction"] == 1.0


def test_gate2_fails_on_the_real_corpus_failure(tmp_path):
    root = written_run(tmp_path, [
        {"doc_id": "d", "section_id": "s1", "title_raw": "2 Related Work",
         "basis_text": r"\section*{2 Related Work} Prominent examples."}])
    result = gate2_basis_text(root)
    assert not result.passed
    assert any("s1" in f for f in result.failures)


def test_gate2_fails_on_a_single_violation_among_many(tmp_path):
    """Zero tolerance: one dirty section in a hundred still fails."""
    records = [{"doc_id": "d", "section_id": f"s{i}", "title_raw": "T",
                "basis_text": "Clean prose about embeddings and retrieval."}
               for i in range(20)]
    records[7]["basis_text"] = "Prose with a $ sign in it."
    result = gate2_basis_text(written_run(tmp_path, records))
    assert not result.passed
    assert result.checks["sections_violating"] == 1
    assert result.checks["clean_fraction"] < 1.0


def test_gate2_fails_when_basis_text_begins_with_its_title(tmp_path):
    root = written_run(tmp_path, [
        {"doc_id": "d", "section_id": "s1", "title_raw": "Dialogue Framework",
         "basis_text": "Dialogue Framework. It fuses multimodal input."}])
    assert not gate2_basis_text(root).passed


def test_gate2_ignores_records_with_no_basis_text(tmp_path):
    """A section that produced nothing is a gate 1 concern, not a gate 2 one."""
    root = written_run(tmp_path, [
        {"doc_id": "d", "section_id": "s1", "basis_text": None},
        {"doc_id": "d", "section_id": "s2", "basis_text": "Clean prose here."}])
    result = gate2_basis_text(root)
    assert result.passed
    assert result.checks["basis_texts_checked"] == 1
    assert result.checks["records_without_basis_text"] == 1


# --------------------------------------------------------------------------
# Gate 3 over written artefacts
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.gates import gate3_tier_independence  # noqa: E402


def test_gate3_passes_on_independent_tiers(tmp_path):
    root = written_run(tmp_path, [{
        "doc_id": "d", "section_id": "s1",
        "tier_summary": "The system classifies keyword queries by temporal "
                        "intent into past, recency, future or atemporal.",
        "tier_abstraction": "Assigning search requests to time orientations "
                            "using linguistic evidence.",
        "tier_label": "temporal orientation assignment for retrieval requests"}])
    result = gate3_tier_independence(root)
    assert result.passed, result.report()
    assert result.checks["records_with_two_or_more_tiers"] == 1


def test_gate3_fails_on_a_prefix_relation(tmp_path):
    root = written_run(tmp_path, [{
        "doc_id": "d", "section_id": "s1",
        "tier_summary": "alpha beta gamma delta epsilon",
        "tier_abstraction": "alpha beta gamma",
        "tier_label": "entirely separate vocabulary"}])
    result = gate3_tier_independence(root)
    assert not result.passed
    assert result.checks["prefix_relations"] == 1


def test_gate3_fails_on_high_jaccard(tmp_path):
    root = written_run(tmp_path, [{
        "doc_id": "d", "section_id": "s1",
        "tier_summary": "alpha beta gamma delta",
        "tier_abstraction": "delta gamma beta alpha",
        "tier_label": "unrelated words here"}])
    result = gate3_tier_independence(root)
    assert not result.passed
    assert result.checks["jaccard_above_threshold"] == 1


def test_gate3_does_not_compare_a_single_tier_ablation(tmp_path):
    """Arm B has one tier by design and must not fail for having it."""
    root = written_run(tmp_path, [
        {"doc_id": "d", "section_id": "s1", "tier_label": "3 Related Work"},
        {"doc_id": "d", "section_id": "s2", "tier_label": "4 Method"}])
    result = gate3_tier_independence(root)
    assert result.passed
    assert result.checks["records_with_two_or_more_tiers"] == 0


def test_gate3_reports_word_budgets_without_gating_them(tmp_path):
    """Budgets are Gate 5's business; Gate 3 reports what it saw."""
    root = written_run(tmp_path, [{
        "doc_id": "d", "section_id": "s1",
        "tier_label": "two words", "tier_abstraction": "quite different text"}])
    result = gate3_tier_independence(root)
    assert result.passed
    assert result.checks["tier_word_budgets"]["label"] == {"under": 1}


# --------------------------------------------------------------------------
# Gate 4 over written artefacts
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.basis import STRUCTURAL_ROW_ID  # noqa: E402
from conceptdrill.hierarchy.gates import gate4_structural   # noqa: E402


def labelled_run(tmp_path, records, labels, rows):
    root = written_run(tmp_path, records,
                       extra=None) if False else None
    log = RunLog.open(tmp_path, timestamp="t")
    doc = docmodel_with(tmp_path, records[0]["doc_id"],
                        [r["section_id"] for r in records])
    for rec in records:
        log.add_section(**rec)
    root = finish(log, corpus_paths=[str(doc)], basis_rows=rows)
    # Gate 4 compares basis.json against manifest basis_stats.
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["basis_stats"] = {"rows": sum(1 for r in rows
                                           if not r.get("structural"))}
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": labels}))
    return root, path


def concept_row(row_id="row_a"):
    return {"row_id": row_id, "label": "L", "support": 1, "level": 1,
            "documents": ["d"], "contributing_section_ids": [],
            "structural": False}


def sink_row():
    return {"row_id": STRUCTURAL_ROW_ID, "label": "[structural]", "support": 1,
            "level": 0, "documents": ["d"], "contributing_section_ids": [],
            "structural": True}


def test_gate4_passes_when_nothing_structural_reaches_a_concept_row(tmp_path):
    root, labels = labelled_run(
        tmp_path,
        [{"doc_id": "d", "section_id": "s1", "title_cleaned": "References",
          "structural_class": "bibliography",
          "structural_rule_fired": "reference-list",
          "row_id_assigned": STRUCTURAL_ROW_ID, "merge_decision": "absorbed"},
         {"doc_id": "d", "section_id": "s2", "title_cleaned": "Method",
          "row_id_assigned": "row_a", "merge_decision": "added"}],
        [{"doc_id": "d", "section_id": "s1", "structural": True},
         {"doc_id": "d", "section_id": "s2", "structural": False}],
        [sink_row(), concept_row()])
    result = gate4_structural(root, labels)
    assert result.passed, result.report()
    assert result.checks["recall"] == 1.0
    assert result.checks["basis_rows_with_row0"] - \
        result.checks["basis_rows_without_row0"] == 1


def test_gate4_fails_when_a_structural_section_holds_a_concept_row(tmp_path):
    """The binding clause. Recall is what this gate is for."""
    root, labels = labelled_run(
        tmp_path,
        [{"doc_id": "d", "section_id": "s1", "title_cleaned": "References",
          "row_id_assigned": "row_a", "merge_decision": "added"}],
        [{"doc_id": "d", "section_id": "s1", "structural": True}],
        [concept_row()])
    result = gate4_structural(root, labels)
    assert not result.passed
    assert result.checks["reached_a_concept_row"] == 1
    assert result.checks["recall"] == 0.0


def test_gate4_fails_when_an_absorbed_section_names_no_rule(tmp_path):
    root, labels = labelled_run(
        tmp_path,
        [{"doc_id": "d", "section_id": "s1", "title_cleaned": "References",
          "structural_class": "bibliography", "structural_rule_fired": None,
          "row_id_assigned": STRUCTURAL_ROW_ID, "merge_decision": "absorbed"}],
        [{"doc_id": "d", "section_id": "s1", "structural": True}],
        [sink_row()])
    result = gate4_structural(root, labels)
    assert not result.passed
    assert any("name no rule" in f for f in result.failures)


def test_gate4_fails_on_more_than_one_structural_row(tmp_path):
    """Row 0 is reserved; a second sink means the reservation broke."""
    second = dict(sink_row(), row_id="row_other_sink")
    root, labels = labelled_run(
        tmp_path,
        [{"doc_id": "d", "section_id": "s1", "title_cleaned": "References",
          "structural_class": "bibliography",
          "structural_rule_fired": "reference-list",
          "row_id_assigned": STRUCTURAL_ROW_ID, "merge_decision": "absorbed"}],
        [{"doc_id": "d", "section_id": "s1", "structural": True}],
        [sink_row(), second])
    result = gate4_structural(root, labels)
    assert not result.passed
    assert any("row 0 is reserved" in f for f in result.failures)


def test_gate4_reports_precision_without_gating_it(tmp_path):
    """Over-absorption is recoverable; it must be visible, not fatal."""
    root, labels = labelled_run(
        tmp_path,
        [{"doc_id": "d", "section_id": "s1", "title_cleaned": "References",
          "structural_class": "bibliography",
          "structural_rule_fired": "reference-list",
          "row_id_assigned": STRUCTURAL_ROW_ID, "merge_decision": "absorbed"},
         {"doc_id": "d", "section_id": "s2", "title_cleaned": "Contributions",
          "structural_class": "author-contributions",
          "structural_rule_fired": "author-contributions",
          "row_id_assigned": STRUCTURAL_ROW_ID, "merge_decision": "absorbed"}],
        [{"doc_id": "d", "section_id": "s1", "structural": True},
         {"doc_id": "d", "section_id": "s2", "structural": False}],
        [sink_row()])
    result = gate4_structural(root, labels)
    assert result.passed
    assert result.checks["recall"] == 1.0
    assert result.checks["precision"] == 0.5
