#!/usr/bin/env python3
"""Step 11: run every proposed Loogle pattern and record what came back.

    python3 experiments/math-layer/loogle_query.py

There is NO existing Lean or Loogle mechanism to run. Measured before writing
this: no `lean`, `lake`, `elan` or `loogle` binary is installed; ConceptDrill
contains no Lean code; and `~/MX/PDFDRILL/src/pdfdrill/lean_export.py`, the only
Lean code in reach, *generates* Lean 4 from a LaTeX theorem via an LLM and
projects it to a file -- it queries nothing and has no search path. So this
script queries the public Loogle service directly.

It is deliberately dumb. It sends each pattern verbatim, stores the raw reply,
and makes no attempt to repair a query the service rejects: whether a proposed
pattern even *parses* is one of the things step 11 is supposed to find out, and
silently rewriting it would destroy that measurement.

Nothing here is imported by ConceptDrill and nothing it writes is read by the
pipeline.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates.json"
OUT = HERE / "loogle-results.json"

ENDPOINT = "https://loogle.lean-lang.org/json"
#: One second between calls. A public service run by volunteers; a burst of
#: twenty queries is rude and this experiment is not in a hurry.
DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30
#: Hits are truncated in the record. Loogle returns hundreds for a bare name
#: and the question here is "does this resolve", not "what is everything".
MAX_HITS_RECORDED = 5


def query(pattern: str) -> dict:
    """One Loogle query. Errors are returned as data, never raised.

    A rejected query is a result -- it says the pattern was not valid syntax --
    so it has to survive into the record rather than stopping the run.
    """
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'q': pattern})}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as fh:
            payload = json.loads(fh.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:400]
        except Exception:
            pass
        return {"transport": "http_error", "status": exc.code, "body": body}
    except Exception as exc:
        return {"transport": f"{type(exc).__name__}", "detail": str(exc)[:200]}

    hits = payload.get("hits") or []
    return {
        "transport": "ok",
        # Loogle reports a parse or elaboration failure in `error` with HTTP 200.
        "error": payload.get("error"),
        "header": payload.get("header"),
        "count": payload.get("count", len(hits)),
        "hits": [{"name": h.get("name"), "module": h.get("module"),
                  "type": (h.get("type") or "")[:200]}
                 for h in hits[:MAX_HITS_RECORDED]],
    }


def main() -> int:
    spec = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    records = []

    for entry in spec["concepts"]:
        concept = entry["concept"]

        # Every proposed Lean name is checked as a query in its own right, so
        # "exact" and "uncertain" are decided by the service rather than by the
        # confidence someone typed next to them.
        for anchor in entry.get("lean", []):
            result = query(anchor["name"])
            resolved = bool(result.get("transport") == "ok"
                            and not result.get("error")
                            and result.get("count"))
            records.append({
                "concept": concept, "kind": "lean_name",
                "query": anchor["name"], "claimed_certainty": anchor["certainty"],
                "resolved": resolved, "result": result,
            })
            print(f"{concept:14s} name    {anchor['name']:32s} "
                  f"{'OK ' if resolved else 'NO '} {result.get('count', '-')}"
                  f" {result.get('error') or ''}"[:150])
            time.sleep(DELAY_SECONDS)

        for pattern in entry.get("loogle", []):
            result = query(pattern)
            resolved = bool(result.get("transport") == "ok"
                            and not result.get("error")
                            and result.get("count"))
            records.append({
                "concept": concept, "kind": "loogle_pattern",
                "query": pattern, "claimed_certainty": None,
                "resolved": resolved, "result": result,
            })
            print(f"{concept:14s} pattern {pattern:32s} "
                  f"{'OK ' if resolved else 'NO '} {result.get('count', '-')}"
                  f" {result.get('error') or ''}"[:150])
            time.sleep(DELAY_SECONDS)

    payload = {
        "step": 11,
        "endpoint": ENDPOINT,
        "existing_mechanism": ("none: no lean/lake/elan/loogle binary, no Lean "
                               "code in conceptdrill, and PDFDRILL's lean_export "
                               "generates Lean rather than searching it"),
        "queries": len(records),
        "resolved": sum(1 for r in records if r["resolved"]),
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"\n{payload['resolved']}/{payload['queries']} resolved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
