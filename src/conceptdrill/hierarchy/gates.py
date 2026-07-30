"""Checks that must pass before a run's numbers mean anything.

Each gate re-derives its answer from the written artefacts rather than from the
process that produced them. A run directory is checked the same way whether it
was written a minute ago or a month ago, and the check does not trust the run's
own summary of itself: gate 1 re-reads the input docmodels and compares section
ids, because `section_count` is exactly the number a broken run would get wrong.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runlog import MANIFEST_REQUIRED, SECTION_FIELDS


@dataclass
class GateResult:
    """Pass or fail, with the evidence either way."""
    name: str
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def report(self) -> str:
        head = f"{self.name}: {'PASS' if self.passed else 'FAIL'}"
        lines = [head]
        for key, value in self.checks.items():
            lines.append(f"  {key}: {value}")
        for failure in self.failures:
            lines.append(f"  ! {failure}")
        return "\n".join(lines)


def read_run(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]],
                                           dict[str, Any]]:
    """`(manifest, section records, basis)` from a run directory."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line
               in (root / "sections.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    basis = json.loads((root / "basis.json").read_text(encoding="utf-8"))
    return manifest, records, basis


#: Flat per-concept keys as they appeared before the concept became the unit.
_LEGACY_CONCEPT_KEYS = ("tier_label", "tier_abstraction", "tier_summary",
                        "basis_text", "embedding_model", "embedding_revision",
                        "row_id_assigned", "merge_decision", "merge_cosine",
                        "merge_target_row_id", "warnings", "error")


def concepts_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Every concept on a section record, whichever schema wrote it.

    The concept is the unit that becomes a basis row. Records written before
    that change carry exactly one concept in flat fields; reading them as a
    one-element list keeps older run directories checkable rather than
    silently passing gates that never looked at them.
    """
    concepts = record.get("concepts")
    if isinstance(concepts, list):
        return [c for c in concepts if isinstance(c, dict)]
    legacy = {k: record.get(k) for k in _LEGACY_CONCEPT_KEYS if k in record}
    return [{"concept_index": 0, **legacy}] if legacy else []


def gate1_persistence(run_dir: str | Path) -> GateResult:
    """Gate 1: the run accounts for its input, in full, with no holes.

    Four clauses, each of which a plausible bug would break on its own:
    the line count matches the manifest; the section ids match the input trees
    exactly; every record carries every field; the manifest knows what made it.
    """
    from .docmodel_tree import load_tree

    manifest, records, _ = read_run(run_dir)
    result = GateResult(name="GATE 1 (persistence)", passed=True)

    # 1. line count == manifest section_count
    declared = manifest.get("section_count")
    result.checks["records"] = len(records)
    result.checks["manifest.section_count"] = declared
    if declared != len(records):
        result.failures.append(
            f"sections.jsonl has {len(records)} lines, manifest says {declared}")

    # 2. exactly the sections in the input trees, each once
    paths = manifest.get("corpus_paths") or []
    if not paths:
        # Without the inputs there is nothing to compare against, and a clause
        # that cannot be evaluated must not be reported as satisfied.
        result.failures.append(
            "manifest lists no corpus_paths: the input-comparison clause "
            "cannot be evaluated, so the run is not auditable")
    expected: set[tuple[str, str]] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            result.failures.append(f"input no longer readable: {path}")
            continue
        tree = load_tree(p)
        for node in tree.iter_document_order():
            expected.add((p.parent.name, node.id))

    got: dict[tuple[str, str], int] = {}
    for rec in records:
        key = (rec.get("doc_id"), rec.get("section_id"))
        got[key] = got.get(key, 0) + 1

    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)
    duplicated = sorted(k for k, n in got.items() if n > 1)
    result.checks["input_sections"] = len(expected)
    result.checks["missing"] = len(missing)
    result.checks["unexpected"] = len(extra)
    result.checks["duplicated"] = len(duplicated)
    if missing:
        result.failures.append(f"sections in the input with no record: {missing[:5]}")
    if extra:
        result.failures.append(f"records for sections not in the input: {extra[:5]}")
    if duplicated:
        result.failures.append(f"sections recorded more than once: {duplicated[:5]}")

    # 3. every field on every record
    holes: dict[str, int] = {}
    for rec in records:
        for name in SECTION_FIELDS:
            if name not in rec:
                holes[name] = holes.get(name, 0) + 1
        for name in set(rec) - set(SECTION_FIELDS):
            result.failures.append(f"record carries an undeclared field {name!r}")
            break
    result.checks["records_missing_a_field"] = sum(holes.values())
    if holes:
        result.failures.append(f"absent fields: {dict(sorted(holes.items()))}")

    # 4. the manifest knows what produced it
    nulls = [k for k in MANIFEST_REQUIRED if manifest.get(k) is None]
    result.checks["manifest_required_nulls"] = nulls
    if nulls:
        result.failures.append(f"manifest fields are null: {nulls}")

    result.passed = not result.failures
    return result


def gate2_basis_text(run_dir: str | Path) -> GateResult:
    """Gate 2: every `basis_text` in the run satisfies the cleaning contract.

    Zero tolerance by construction — the gate reports each violating section
    with the offending substring, because "99% clean" describes a corpus with
    a constant substring in one section in a hundred, which is exactly the
    failure mode this exists to catch.
    """
    from .basistext import check_basis_text

    manifest, records, _ = read_run(run_dir)
    result = GateResult(name="GATE 2 (basis text)", passed=True)

    # In the title-only ablation the title IS the content, so `basis_text`
    # begins with it by construction and `clean_basis_text` is called with no
    # title to strip. The gate has to agree with the arm the runner declared,
    # or it fails a baseline for being the baseline.
    ablation = bool(manifest.get("is_ablation"))
    result.checks["is_ablation"] = ablation

    checked = 0
    concepts_seen = 0
    violations: list[tuple[str, str, str]] = []
    for rec in records:
        for concept in concepts_of(rec):
            concepts_seen += 1
            text = concept.get("basis_text")
            if text is None:
                continue
            checked += 1
            where = f"{rec.get('section_id')}#{concept.get('concept_index')}"
            title = "" if ablation else (rec.get("title_raw") or "")
            for problem in check_basis_text(text, title):
                violations.append((rec.get("doc_id"), where, problem))

    result.checks["concepts"] = concepts_seen
    result.checks["basis_texts_checked"] = checked
    result.checks["records_without_basis_text"] = concepts_seen - checked
    result.checks["violations"] = len(violations)
    result.checks["sections_violating"] = len({(d, s) for d, s, _ in violations})
    if checked:
        clean_fraction = 1.0 - len({(d, s) for d, s, _ in violations}) / checked
        result.checks["clean_fraction"] = round(clean_fraction, 6)
    for doc, sid, problem in violations[:20]:
        result.failures.append(f"{doc}/{sid}: {problem}")
    if len(violations) > 20:
        result.failures.append(f"... and {len(violations) - 20} more")

    result.passed = not violations
    return result


def gate3_tier_independence(run_dir: str | Path,
                            max_jaccard: float = 0.6) -> GateResult:
    """Gate 3: the three tiers are independent derivations, not cut points.

    Recomputed from the written tiers rather than from the summariser's own
    report, so a summariser that never checked is checked anyway. Records with
    fewer than two tiers present are counted but not compared — an ablation arm
    has one tier by design, and `null` is a declared absence.
    """
    from .summarize import TIER_WORDS, jaccard

    manifest, records, _ = read_run(run_dir)
    result = GateResult(name="GATE 3 (tier independence)", passed=True)

    tier_field = {tier: f"tier_{tier}" for tier in TIER_WORDS}
    prefix_hits: list[str] = []
    jaccard_hits: list[str] = []
    comparable = 0
    worst = 0.0
    budgets: dict[str, dict[str, int]] = {t: {} for t in TIER_WORDS}

    total_concepts = 0
    for rec in records:
      for concept in concepts_of(rec):
        total_concepts += 1
        present = {}
        for tier, key in tier_field.items():
            text = (concept.get(key) or "").strip()
            if text:
                present[tier] = text
                n = len(text.split())
                lo, hi = TIER_WORDS[tier]
                fit = "under" if n < lo else "over" if n > hi else "ok"
                budgets[tier][fit] = budgets[tier].get(fit, 0) + 1

        if len(present) < 2:
            continue
        comparable += 1
        names = sorted(present)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = present[left], present[right]
                where = (f"{rec.get('doc_id')}/{rec.get('section_id')}"
                         f"#{concept.get('concept_index')}")
                if a.startswith(b) or b.startswith(a):
                    prefix_hits.append(f"{where}: {left} and {right} share a prefix")
                    continue
                overlap = jaccard(a, b)
                worst = max(worst, overlap)
                if overlap > max_jaccard:
                    jaccard_hits.append(
                        f"{where}: {left}/{right} Jaccard {overlap:.3f}")

    result.checks["records"] = len(records)
    result.checks["concepts"] = total_concepts
    result.checks["concepts_with_two_or_more_tiers"] = comparable
    result.checks["prefix_relations"] = len(prefix_hits)
    result.checks["jaccard_above_threshold"] = len(jaccard_hits)
    result.checks["worst_jaccard"] = round(worst, 4)
    result.checks["tier_word_budgets"] = {t: dict(sorted(v.items()))
                                          for t, v in budgets.items()}
    result.checks["summarizer_class"] = manifest.get("summarizer_class")
    result.checks["is_ablation"] = manifest.get("is_ablation")

    for hit in (prefix_hits + jaccard_hits)[:20]:
        result.failures.append(hit)
    extra = len(prefix_hits) + len(jaccard_hits) - 20
    if extra > 0:
        result.failures.append(f"... and {extra} more")

    result.passed = not (prefix_hits or jaccard_hits)
    return result


def gate4_structural(run_dir: str | Path,
                     labels_path: str | Path) -> GateResult:
    """Gate 4: no hand-labelled structural section reaches the concept basis.

    Recall is the binding constraint and precision is reported only. A concept
    wrongly absorbed is visible in the record and recoverable; a reference list
    that became a basis row contaminates every coordinate derived from it and
    is not.
    """
    from .basis import STRUCTURAL_ROW_ID

    manifest, records, basis = read_run(run_dir)
    truth = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    want = {(l["doc_id"], l["section_id"]) for l in truth["labels"]
            if l["structural"]}

    result = GateResult(name="GATE 4 (structural layer)", passed=True)
    by_key = {(r["doc_id"], r["section_id"]): r for r in records}

    missing_labels = want - set(by_key)
    if missing_labels:
        result.failures.append(
            f"labelled sections absent from the run: {sorted(missing_labels)[:5]}")

    classified = {k for k, r in by_key.items() if r.get("structural_class")}
    tp = len(want & classified)
    fn = len(want - classified)
    fp = len(classified - want)

    # The binding clause: a structural section must not hold a concept row.
    leaked = []
    for key in want:
        rec = by_key.get(key)
        if rec is None:
            continue
        for concept in concepts_of(rec):
            row = concept.get("row_id_assigned")
            if row and row != STRUCTURAL_ROW_ID:
                leaked.append(
                    f"{key[0]}/{key[1]}#{concept.get('concept_index')} "
                    f"({rec.get('title_cleaned')!r}) holds concept row {row}")

    unnamed = [k for k, r in by_key.items()
               if r.get("structural_class") and not r.get("structural_rule_fired")]

    rows = basis.get("rows") or []
    sink_rows = [r for r in rows if r.get("structural")]
    concept_rows = len(rows) - len(sink_rows)

    result.checks["hand_labelled_structural"] = len(want)
    result.checks["classified_structural"] = len(classified)
    result.checks["true_positives"] = tp
    result.checks["false_negatives"] = fn
    result.checks["false_positives"] = fp
    result.checks["recall"] = round(tp / len(want), 4) if want else None
    result.checks["precision"] = (round(tp / len(classified), 4)
                                  if classified else None)
    result.checks["reached_a_concept_row"] = len(leaked)
    result.checks["basis_rows_without_row0"] = concept_rows
    result.checks["basis_rows_with_row0"] = len(rows)
    result.checks["manifest_stats_rows"] = (
        (manifest.get("basis_stats") or {}).get("rows"))

    for entry in leaked[:10]:
        result.failures.append(entry)
    if unnamed:
        result.failures.append(
            f"{len(unnamed)} absorbed sections name no rule: {unnamed[:5]}")
    if len(sink_rows) > 1:
        result.failures.append(f"{len(sink_rows)} structural rows; row 0 is reserved")
    if sink_rows and len(rows) - concept_rows != 1:
        result.failures.append(
            f"basis size with and without row 0 differ by "
            f"{len(rows) - concept_rows}, not 1")
    if (manifest.get("basis_stats") or {}).get("rows") != concept_rows:
        result.failures.append(
            f"manifest reports {(manifest.get('basis_stats') or {}).get('rows')} "
            f"rows but basis.json holds {concept_rows} concept rows")

    result.passed = not result.failures
    return result
