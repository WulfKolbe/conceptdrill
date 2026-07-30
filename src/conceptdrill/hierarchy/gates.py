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

    _, records, _ = read_run(run_dir)
    result = GateResult(name="GATE 2 (basis text)", passed=True)

    checked = 0
    violations: list[tuple[str, str, str]] = []
    for rec in records:
        text = rec.get("basis_text")
        if text is None:
            continue
        checked += 1
        for problem in check_basis_text(text, rec.get("title_raw") or ""):
            violations.append((rec.get("doc_id"), rec.get("section_id"), problem))

    result.checks["basis_texts_checked"] = checked
    result.checks["records_without_basis_text"] = len(records) - checked
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
