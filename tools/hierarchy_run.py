#!/usr/bin/env python3
"""One auditable hierarchy run: documents in, run directory out.

    python3 tools/hierarchy_run.py --limit 3 --out ~/conceptdrill-runs

Writes `run-<timestamp>-<git-sha>/` per `hierarchy/runlog.py`. Every section in
every input tree gets a line in `sections.jsonl`, including sections that were
never summarised and sections that never reached the basis — a run that cannot
account for its input is not evidence.

This is the persistence layer only. Input cleaning, tier independence and the
structural layer are separate steps; the fields that belong to them are written
as `null` here rather than omitted, so adding them changes values and never the
schema.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

from conceptdrill.embeddings import get_embedder                  # noqa: E402
from conceptdrill.hierarchy.basis import ConceptBasis             # noqa: E402
from conceptdrill.hierarchy.basistext import clean_basis_text      # noqa: E402
from conceptdrill.hierarchy.captions import clean_caption_traced  # noqa: E402
from conceptdrill.hierarchy.docmodel_tree import load_tree        # noqa: E402
from conceptdrill.hierarchy.runlog import RunLog                  # noqa: E402
from conceptdrill.hierarchy.summarize import (ExtractiveSummarizer,  # noqa: E402
                                              SummaryCache, summarize_tree)


def make_summarizer(kind: str, llm_model: str):
    if kind == "extractive":
        return ExtractiveSummarizer()
    from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                               load_dotenv, make_openai_chat)
    load_dotenv()
    model = llm_model or os.environ.get("NOVITA_MODEL") or DEFAULT_MODEL
    return NovitaSummarizer(make_openai_chat(model=model), model=model)


def _error_for(summary, basis_text) -> str | None:
    """Why this section has no basis vector, or None when it has one."""
    if summary is None:
        return "no summary produced"
    if summary.error:
        return summary.error
    if not summary.is_usable:
        return "summary produced no usable tier text"
    if basis_text is not None and not basis_text.text:
        return "cleaning left no text: the section was entirely markup"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"))
    ap.add_argument("--out", default=str(Path.home() / "conceptdrill-runs"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--docs", nargs="*", default=None,
                    help="explicit bibkeys; overrides --limit")
    ap.add_argument("--model", default="sentencebert")
    ap.add_argument("--summarizer", default="extractive",
                    choices=["extractive", "novita"])
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--summary-cache", default=".conceptdrill_cache/summaries.json")
    args = ap.parse_args()

    strict = os.environ.get("CONCEPTDRILL_STRICT", "") == "1"
    nlp_backend = os.environ.get("CONCEPTDRILL_NLP_BACKEND")

    library = Path(args.library)
    if args.docs:
        paths = [library / key / "model.docmodel.json" for key in args.docs]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"no docmodel at: {missing}")
    else:
        paths = sorted(library.glob("*/model.docmodel.json"))

    embedder = get_embedder(args.model, cache=True)
    summarizer = make_summarizer(args.summarizer, args.llm_model)
    cache = SummaryCache(args.summary_cache) if args.summary_cache else None
    basis = ConceptBasis() if args.tau is None else ConceptBasis(tau=args.tau)
    log = RunLog.open(args.out)

    used_paths: list[str] = []
    math_sources: dict[str, int] = {}
    docs_done = 0

    for path in paths:
        if not args.docs and docs_done >= args.limit:
            break
        bibkey = path.parent.name
        tree = load_tree(path)
        if not len(tree):
            continue                       # no sections: nothing to account for
        docs_done += 1
        used_paths.append(str(path))
        for source, n in (tree.math_sources or {}).items():
            math_sources[source] = math_sources.get(source, 0) + n

        # Every node in the tree, in document order. This list is the ledger.
        nodes = list(tree.iter_document_order())
        log.expect(len(nodes))

        run = summarize_tree(tree, summarizer, cache=cache)

        # Every section's basis text goes through the one cleaner, whether or
        # not it reaches the basis, so `cleaning_rules_fired` is populated for
        # sections that were later dropped.
        cleaned: dict[str, object] = {}
        for node in nodes:
            summary = run.summaries.get(node.id)
            if summary is None:
                continue
            cleaned[node.id] = clean_basis_text(summary.basis_text,
                                                title=node.title_raw)

        # Integrate only what survives cleaning, but keep the decision per
        # section so the sections that did not reach the basis say why.
        usable = [(n, cleaned[n.id]) for n in nodes
                  if n.id in run.summaries and run.summaries[n.id].is_usable
                  and n.id in cleaned and cleaned[n.id].text]
        decisions: dict[str, object] = {}
        if usable:
            vectors = embedder.encode([c.text for _, c in usable])
            for (node, basis_text), vector in zip(usable, vectors):
                decisions[node.id] = basis.integrate(
                    node.level, basis_text.text, vector, document=bibkey)
            if bibkey not in basis.document_order:
                basis.document_order = basis.document_order + (bibkey,)

        for node in nodes:
            summary = run.summaries.get(node.id)
            title_clean, title_rules = clean_caption_traced(node.title_raw)
            basis_text = cleaned.get(node.id)
            result = decisions.get(node.id)
            fired = [f"title:{r}" for r in title_rules]
            if basis_text is not None:
                fired += [f"basis:{r}" for r in basis_text.rules_fired]
            log.add_section(
                doc_id=bibkey,
                section_id=node.id,
                level=node.level,
                flow_index=node.flow_index,
                is_appendix=node.is_appendix,
                title_raw=node.title_raw,
                title_cleaned=title_clean,
                cleaning_rules_fired=fired,
                structural_class=None,          # step 4
                structural_rule_fired=None,     # step 4
                tier_label=summary.label if summary else None,
                tier_abstraction=summary.abstraction if summary else None,
                tier_summary=summary.summary if summary else None,
                basis_text=basis_text.text if basis_text is not None else None,
                embedding_model=embedder.name if result is not None else None,
                embedding_revision=(embedder.revision if result is not None
                                    else None),
                row_id_assigned=getattr(result, "row_id", None) or None,
                merge_decision=(getattr(result, "action", None)
                                if result is not None else "not_integrated"),
                merge_cosine=(round(float(result.similarity), 6)
                              if result is not None else None),
                merge_target_row_id=(result.row_id
                                     if result is not None
                                     and result.action == "merged" else None),
                warnings=list(summary.warnings) if summary else [],
                error=_error_for(summary, basis_text),
            )

    if cache is not None:
        cache.flush()
    flush = getattr(embedder, "flush", None)
    if callable(flush):
        flush()

    rows = [{"row_id": r.row_id, "label": r.label, "support": r.support,
             "level": r.level, "documents": list(r.documents),
             "contributing_section_ids": sorted(
                 rec["section_id"] for rec in log._sections
                 if rec["row_id_assigned"] == r.row_id)}
            for r in basis.ordered_rows()]

    root = log.finish(
        summarizer_class=type(summarizer).__name__,
        embedder_backend=embedder.name,
        embedder_resolved_revision=embedder.revision,
        nlp_backend=nlp_backend,
        tau=basis.tau,
        strict_mode=strict,
        corpus_paths=used_paths,
        doc_count=docs_done,
        basis_rows=rows,
        mathtext_source_counts=math_sources,
        extra={"summarizer_name": getattr(summarizer, "name", ""),
               "basis_version": basis.basis_version(),
               "basis_stats": basis.stats()})

    print(f"sections: {len(log)}  documents: {docs_done}  "
          f"basis rows: {len(rows)}")
    print(f"run -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
