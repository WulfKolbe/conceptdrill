#!/usr/bin/env python3
"""Run every gate against a run directory and write the verdict beside it.

    python3 tools/gate_report.py ~/conceptdrill-corpus-llm/current

Writes `gates.json` into the run directory, so a run and its verdict are one
artefact rather than a directory plus a number someone remembered. Exits
non-zero when any gate fails, so it is usable as a check rather than only as a
report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

from conceptdrill.hierarchy.gates import (gate1_persistence,      # noqa: E402
                                          gate2_basis_text,
                                          gate3_tier_independence,
                                          gate4_structural)

LABELS = (Path(__file__).resolve().parents[1] / "docs" / "measurements"
          / "structural-labels-10docs.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--labels", default=str(LABELS))
    args = ap.parse_args()

    run = Path(args.run_dir)
    results = [gate1_persistence(run), gate2_basis_text(run),
               gate3_tier_independence(run)]
    if Path(args.labels).exists():
        results.append(gate4_structural(run, args.labels))

    payload = {
        "run_dir": str(run),
        "run_id": json.loads((run / "manifest.json").read_text())["run_id"],
        "passed": all(r.passed for r in results),
        "gates": [{"name": r.name, "passed": r.passed, "checks": r.checks,
                   "failures": r.failures} for r in results],
    }
    (run / "gates.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    for r in results:
        print(r.report())
        print()
    print(f"overall: {'PASS' if payload['passed'] else 'FAIL'}")
    print(f"written -> {run / 'gates.json'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
