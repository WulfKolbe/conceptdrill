#!/usr/bin/env python3
r"""What merge threshold does THIS embedder need?

    PYTHONPATH=src python3 tools/calibrate_tau.py ~/conceptdrill-corpus-llm/tc50

TAU = 0.65 was measured against all-MiniLM-L6-v2, whose off-diagonal cosines
had median 0.18. A different encoder has a different similarity scale, and the
threshold does not travel with it. Measured on ModernBERT the nearest-row
cosine at integration ran p5 0.653 to p95 0.851, so 0.65 put 95% of candidates
above the line: 482 concepts collapsed into 25 rows.

A threshold is only meaningful against a distribution, so this reports both.
The two distributions answer different questions and must not be pooled:

  within-document   two concepts from ONE document. High similarity here is
                    often genuine redundancy and merging is usually right.
  cross-document    two concepts from DIFFERENT documents. This is what a
                    shared basis exists to find, and where the threshold has
                    to sit: high enough that typical unrelated pairs stay
                    apart, low enough that the genuinely-shared ones meet.

Reads concepts from a completed run, so it costs embedding time and no LLM
calls. The embedder is taken from the run's manifest -- calibrating with a
different one than built the basis would measure a scale nothing uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

import numpy as np                                                # noqa: E402

from conceptdrill.embeddings import get_embedder                  # noqa: E402
from conceptdrill.hierarchy.basis import calibrate                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="percentile of the CROSS-document distribution to "
                         "suggest as tau")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    model = manifest.get("embedder_backend") or "modernbert"
    spans = [json.loads(line)
             for line in (run / "spans.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]

    # One vector set per document, of the texts that actually became candidates.
    by_doc: dict[str, list[str]] = defaultdict(list)
    for span in spans:
        if span.get("structural_class"):
            continue                     # the sink is not a concept
        for concept in span.get("concepts") or []:
            text = concept.get("basis_text")
            if text:
                by_doc[span["doc_id"]].append(text)

    docs = sorted(by_doc)
    print(f"calibrating {model} on {sum(len(v) for v in by_doc.values())} "
          f"concepts from {len(docs)} documents")
    embedder = get_embedder(model, cache=True)
    sets = [np.asarray(embedder.encode(by_doc[d]), dtype=np.float64) for d in docs]

    result = calibrate(sets, cross_percentile=args.percentile)
    profile = result["profile"]
    print(f"\ncurrent tau in this run: {manifest.get('tau')}")
    print(f"suggested tau at the p{args.percentile:g} cross-document point: "
          f"{result['suggested_tau']:.3f}")
    for name in ("within_document", "cross_document"):
        d = profile.get(name) or {}
        if not d.get("n"):
            continue
        print(f"\n{name.replace('_', '-')} ({d['n']} pairs):")
        for key in ("p10", "p25", "p50", "p75", "p90", "p99", "max"):
            if key in d:
                print(f"  {key:>4s} {d[key]:+.4f}")

    # What each threshold would do, since that is the decision being made.
    flat = []
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            flat.extend((a @ b.T).ravel().tolist())
    cross = np.asarray(flat)
    within = np.concatenate([(s @ s.T)[np.triu_indices(len(s), 1)]
                             for s in sets if len(s) > 1]) if sets else np.array([])
    print(f"\n{'tau':>6s} {'cross merges':>13s} {'within merges':>14s}")
    for tau in (0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95):
        print(f"{tau:6.2f} {(cross >= tau).mean():12.2%} "
              f"{(within >= tau).mean() if within.size else 0:13.2%}")

    payload = {"run_dir": str(run), "model": model,
               "tau_in_run": manifest.get("tau"),
               "suggested_tau": result["suggested_tau"],
               "cross_percentile": args.percentile,
               "profile": profile,
               "merge_fraction": {
                   str(t): {"cross": float((cross >= t).mean()),
                            "within": float((within >= t).mean()) if within.size else 0.0}
                   for t in (0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95)}}
    target = Path(args.out) if args.out else run / "tau-calibration.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                      encoding="utf-8")
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
