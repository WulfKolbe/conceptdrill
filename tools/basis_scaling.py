#!/usr/bin/env python3
"""Does the shared basis stay small as the corpus grows?

Builds one `ConceptBasis` incrementally over the drilled library, reporting
after every batch. The question is the growth curve: if the basis grows
linearly with documents the merge rule is doing nothing and CES dimensionality
is unbounded; if it flattens, concepts are genuinely being shared.

    python3 tools/basis_scaling.py --batch 100 --limit 400

**The summariser here is the extractive floor, not an LLM.** Running the real
summariser over the library would be tens of thousands of API calls. The
extractive labels are title-plus-opening-clause, so they are document-faithful
by construction — which is exactly what the `label` tier is supposed to avoid.
Treat the resulting merge rate as a PROXY for the algorithm's scaling shape,
not as the quality the pipeline would deliver. Measured earlier on three
papers, extractive labels are *more* cross-document similar than LLM ones
(max 0.723 vs 0.647), so this proxy is optimistic about merging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Numerics before numpy loads: this host miscomputes float32 GEMM in roughly one
# process in three, and a wrong cosine silently changes a merge decision.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()
os.environ.setdefault("CONCEPTDRILL_NLP_BACKEND", "regex")

import numpy as np                                                # noqa: E402

from conceptdrill.embeddings import get_embedder                  # noqa: E402
from conceptdrill.hierarchy.basis import ConceptBasis             # noqa: E402
from conceptdrill.hierarchy.docmodel_tree import load_tree        # noqa: E402
from conceptdrill.hierarchy.summarize import (ExtractiveSummarizer,  # noqa: E402
                                              summarize_tree)


def candidates_for(tree, summarizer):
    """(level, label) pairs for one document, or [] when it has no sections."""
    if not len(tree):
        return []
    run = summarize_tree(tree, summarizer)
    return [(tree.nodes[sid].level, s.basis_text)
            for sid, s in run.usable().items() if sid in tree.nodes]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"))
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the merge threshold")
    ap.add_argument("--model", default="sentencebert")
    ap.add_argument("--out", default="", help="write the report as JSON")
    args = ap.parse_args()

    blas = blasfix.ensure_sane_blas(verbose=True)
    if blas.get("status") not in ("ok", "fixed"):
        print(f"note: float32 arithmetic status {blas.get('status')}", file=sys.stderr)

    docs = sorted(Path(args.library).glob("*/model.docmodel.json"))
    if args.limit:
        docs = docs[:args.limit]

    embedder = get_embedder(args.model, cache=True)
    summarizer = ExtractiveSummarizer()
    basis = ConceptBasis() if args.tau is None else ConceptBasis(tau=args.tau)

    print(f"{len(docs)} documents, batches of {args.batch}, "
          f"tau={basis.tau}, model={args.model}\n")
    header = (f"{'docs':>5s} {'usable':>6s} {'cands':>7s} {'rows':>6s} "
              f"{'added':>6s} {'merged':>6s} {'merge%':>7s} "
              f"{'rows/doc':>8s} {'shared':>7s} {'secs':>6s}")
    print(header)
    print("-" * len(header))

    report = []
    usable = total_cands = total_added = total_merged = 0
    started = time.monotonic()

    for start in range(0, len(docs), args.batch):
        chunk = docs[start:start + args.batch]
        t0 = time.monotonic()
        batch_cands = batch_added = batch_merged = 0

        for path in chunk:
            try:
                tree = load_tree(path)
                cands = candidates_for(tree, summarizer)
            except Exception:
                continue
            if not cands:
                continue
            usable += 1
            texts = [t for _, t in cands]
            vectors = embedder.encode(texts)
            results = basis.integrate_document(
                path.parent.name,
                [(lvl, txt, vec) for (lvl, txt), vec in zip(cands, vectors)])
            batch_cands += len(results)
            batch_added += sum(1 for r in results if r.action == "added")
            batch_merged += sum(1 for r in results if r.action == "merged")

        total_cands += batch_cands
        total_added += batch_added
        total_merged += batch_merged
        stats = basis.stats()
        elapsed = time.monotonic() - t0
        rate = (total_merged / total_cands) if total_cands else 0.0

        row = {
            "documents_seen": min(start + args.batch, len(docs)),
            "documents_usable": usable,
            "candidates": total_cands,
            "rows": stats["rows"],
            "added": total_added,
            "merged": total_merged,
            "merge_rate": round(rate, 4),
            "rows_per_document": round(stats["rows"] / usable, 3) if usable else 0.0,
            "shared_across_documents": stats["shared_across_documents"],
            "levels": stats["levels"],
            "support_max": stats["support_max"],
            "seconds": round(elapsed, 1),
        }
        report.append(row)
        print(f"{row['documents_seen']:>5d} {usable:>6d} {total_cands:>7d} "
              f"{stats['rows']:>6d} {total_added:>6d} {total_merged:>6d} "
              f"{rate:>6.1%} {row['rows_per_document']:>8.2f} "
              f"{stats['shared_across_documents']:>7d} {elapsed:>6.1f}")
        sys.stdout.flush()

    flush = getattr(embedder, "flush", None)
    if callable(flush):
        flush()

    print(f"\ntotal {time.monotonic() - started:.0f}s | final basis: {basis.stats()}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "tau": basis.tau, "model": args.model,
            "summarizer": "extractive (proxy -- see module docstring)",
            "batches": report, "final": basis.stats(),
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
