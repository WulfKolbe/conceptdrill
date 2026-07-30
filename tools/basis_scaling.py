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
                                              SummaryCache, summarize_tree)


def make_summarizer(kind: str, llm_model: str):
    """The summariser and a human name for it."""
    if kind == "extractive":
        return ExtractiveSummarizer()
    from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                               load_dotenv, make_openai_chat)
    load_dotenv()                       # .env is gitignored; never a repo file
    model = llm_model or os.environ.get("NOVITA_MODEL") or DEFAULT_MODEL
    return NovitaSummarizer(make_openai_chat(model=model), model=model)


def candidates_for(tree, summarizer, cache=None):
    """One record per usable section, or [] when the document has none.

    Returns the summaries themselves rather than bare `(level, text)` pairs so
    the caller can persist the basis input. An aggregate row count is not
    inspectable; the labels that produced it are.
    """
    if not len(tree):
        return [], None
    run = summarize_tree(tree, summarizer, cache=cache)
    out = []
    for sid, s in run.usable().items():
        node = tree.nodes.get(sid)
        if node is None:
            continue
        out.append({"section_id": sid, "level": node.level, "title": node.title,
                    "basis_text": s.basis_text, "label": s.label,
                    "abstraction": s.abstraction, "summary": s.summary,
                    "warnings": list(s.warnings), "error": s.error})
    return out, run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"))
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = all documents")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the merge threshold")
    ap.add_argument("--model", default="sentencebert")
    ap.add_argument("--summarizer", default="extractive",
                    choices=["extractive", "novita"],
                    help="'extractive' is the offline floor (see the module "
                         "docstring); 'novita' calls the chat model")
    ap.add_argument("--llm-model", default="",
                    help="chat model id; defaults to NOVITA_MODEL or the "
                         "package default")
    ap.add_argument("--summary-cache", default=".conceptdrill_cache/summaries.json",
                    help="content-addressed summary cache; '' disables it")
    ap.add_argument("--out", default="", help="write the report as JSON")
    ap.add_argument("--store", default="",
                    help="write the built basis here as a CorpusStore "
                         "(ces-basis.json + ces-basis.npz)")
    ap.add_argument("--summaries", default="",
                    help="write one <bibkey>.json of per-section summaries per "
                         "document here -- the input the basis was built from")
    args = ap.parse_args()

    blas = blasfix.ensure_sane_blas(verbose=True)
    if blas.get("status") not in ("ok", "fixed"):
        print(f"note: float32 arithmetic status {blas.get('status')}", file=sys.stderr)

    docs = sorted(Path(args.library).glob("*/model.docmodel.json"))
    if args.limit:
        docs = docs[:args.limit]

    embedder = get_embedder(args.model, cache=True)
    summarizer = make_summarizer(args.summarizer, args.llm_model)
    cache = SummaryCache(args.summary_cache) if args.summary_cache else None
    basis = ConceptBasis() if args.tau is None else ConceptBasis(tau=args.tau)

    print(f"{len(docs)} documents, batches of {args.batch}, tau={basis.tau}, "
          f"embedder={args.model}, summarizer={summarizer.name}\n")
    header = (f"{'docs':>5s} {'usable':>6s} {'cands':>7s} {'rows':>6s} "
              f"{'added':>6s} {'merged':>6s} {'merge%':>7s} "
              f"{'rows/doc':>8s} {'shared':>7s} {'secs':>6s}")
    print(header)
    print("-" * len(header))

    summaries_dir = None
    if args.summaries:
        summaries_dir = Path(args.summaries)
        summaries_dir.mkdir(parents=True, exist_ok=True)

    report = []
    failed = attempted = cached = generated = 0
    usable = total_cands = total_added = total_merged = 0
    started = time.monotonic()

    for start in range(0, len(docs), args.batch):
        chunk = docs[start:start + args.batch]
        t0 = time.monotonic()
        batch_cands = batch_added = batch_merged = 0

        for path in chunk:
            bibkey = path.parent.name
            try:
                tree = load_tree(path)
                cands, run = candidates_for(tree, summarizer, cache)
            except Exception:
                continue
            if not cands:
                continue
            usable += 1
            failed += len(run.failed)
            attempted += len(run.summaries) + len(run.failed)
            cached += run.cached
            generated += run.generated
            texts = [c["basis_text"] for c in cands]
            vectors = embedder.encode(texts)
            if summaries_dir is not None:
                (summaries_dir / f"{bibkey}.json").write_text(json.dumps(
                    {"document": bibkey, "summarizer": summarizer.name,
                     "sections": cands,
                     "failed_sections": list(run.failed),
                     "cached": run.cached, "generated": run.generated},
                    indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
            results = basis.integrate_document(
                bibkey,
                [(c["level"], c["basis_text"], vec)
                 for c, vec in zip(cands, vectors)])
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

    if cache is not None:
        cache.flush()
    if attempted:
        print(f"\nsummaries: {attempted - failed}/{attempted} usable, "
              f"{failed} failed | {cached} from cache, {generated} generated")

    flush = getattr(embedder, "flush", None)
    if callable(flush):
        flush()

    print(f"\ntotal {time.monotonic() - started:.0f}s | final basis: {basis.stats()}")
    if args.store:
        from conceptdrill.hierarchy.corpus import CorpusStore
        store = CorpusStore(args.store)
        store.save_basis(basis, embedding_model=args.model,
                         summarizer=summarizer.name)
        print(f"basis  -> {store.basis_json} (+ .npz)")
    if summaries_dir is not None:
        print(f"summaries -> {summaries_dir}/<bibkey>.json  "
              f"({len(list(summaries_dir.glob('*.json')))} documents)")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "tau": basis.tau, "model": args.model,
            "summarizer": summarizer.name,
            "summaries_attempted": attempted, "summaries_failed": failed,
            "summaries_cached": cached, "summaries_generated": generated,
            "batches": report, "final": basis.stats(),
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
