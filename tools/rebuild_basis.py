#!/usr/bin/env python3
r"""Rebuild a run's basis at other thresholds, without touching the LLM.

    PYTHONPATH=src python3 tools/rebuild_basis.py ~/conceptdrill-corpus-llm/tc50 \
        --tau 0.65 0.75 0.80 0.85 0.88 0.90 0.92

The concepts are already written in `spans.jsonl`; only the integration step
depends on tau. So the whole tau question can be answered by re-integrating
them, at embedding cost and no API cost.

WHY A PAIRWISE CALIBRATION IS NOT ENOUGH. `calibrate` reports how many PAIRS
exceed a threshold: at 0.65 that was 8.6% of cross-document pairs. But
`integrate` compares each candidate against its NEAREST existing row, and the
nearest of many rows is far closer than a typical pair. Fewer merges leaves
more rows, which makes the next candidate's nearest closer still. That
feedback cannot be read off a pairwise distribution -- it has to be run.

Documents are integrated in the run's own order, because the basis depends on
it and a different order would answer a different question.
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
from conceptdrill.hierarchy.basis import ConceptBasis             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--tau", type=float, nargs="+",
                    default=[0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    model = manifest.get("embedder_backend") or "modernbert"
    spans = [json.loads(line)
             for line in (run / "spans.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]

    # (document, level, text) in the run's own order.
    order: list[str] = []
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    structural: list[tuple[str, str]] = []
    for span in sorted(spans, key=lambda s: (s["doc_id"], s["flow_index"])):
        doc = span["doc_id"]
        if doc not in order:
            order.append(doc)
        for concept in span.get("concepts") or []:
            text = concept.get("basis_text")
            if not text:
                continue
            if span.get("structural_class"):
                structural.append((doc, text))
            else:
                candidates[doc].append((span["level"], text))

    every = [t for d in order for _, t in candidates[d]] + [t for _, t in structural]
    print(f"{len(every)} concept texts from {len(order)} documents, model {model}")
    embedder = get_embedder(model, cache=True)
    vectors = {t: v for t, v in zip(every, np.asarray(embedder.encode(every),
                                                      dtype=np.float64))}

    print(f"\n{'tau':>6s} {'rows':>6s} {'added':>7s} {'merged':>7s} "
          f"{'merge%':>7s} {'shared':>7s} {'singletons':>11s} {'rows/doc':>9s}")
    results = []
    for tau in args.tau:
        basis = ConceptBasis(tau=tau)
        added = merged = 0
        for doc in order:
            for level, text in candidates[doc]:
                r = basis.integrate(level, text, vectors[text], document=doc)
                added += r.action == "added"
                merged += r.action == "merged"
        for doc, text in structural:
            basis.absorb_structural(text, vectors[text], document=doc)
        stats = basis.stats()
        total = added + merged
        results.append({"tau": tau, **{k: stats[k] for k in
                                       ("rows", "singletons", "shared_across_documents")},
                        "added": added, "merged": merged,
                        "merge_rate": merged / total if total else 0.0})
        print(f"{tau:6.2f} {stats['rows']:6d} {added:7d} {merged:7d} "
              f"{merged / total if total else 0:6.1%} "
              f"{stats['shared_across_documents']:7d} {stats['singletons']:11d} "
              f"{stats['rows'] / len(order):9.1f}")

    payload = {"run_dir": str(run), "model": model,
               "tau_in_run": manifest.get("tau"),
               "concepts": len(every), "documents": len(order),
               "results": results}
    target = Path(args.out) if args.out else run / "tau-sweep.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
