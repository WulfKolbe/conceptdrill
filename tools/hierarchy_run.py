#!/usr/bin/env python3
"""One auditable hierarchy run: documents in, run directory out.

    python3 tools/hierarchy_run.py --limit 3

Runs land in `~/conceptdrill-corpus-llm/current/` by default. Not /tmp: a
measurement that a reboot can delete is not a record of anything.

Writes `run-<timestamp>-<git-sha>/` per `hierarchy/runlog.py`. Every span in
every input tree gets a line in `spans.jsonl`, including spans that were
never summarised and spans that never reached the basis — a run that cannot
account for its input is not evidence.

This is the persistence layer only. Input cleaning, tier independence and the
structural layer are separate steps; the fields that belong to them are written
as `null` here rather than omitted, so adding them changes values and never the
schema.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402
import numpy as np                                                # noqa: E402

blasfix.apply_env_mitigations()

from conceptdrill.embeddings import get_embedder                  # noqa: E402
from conceptdrill.hierarchy.basis import (STRUCTURAL_ROW_ID,      # noqa: E402
                                          ConceptBasis)
from conceptdrill.hierarchy.basistext import clean_basis_text      # noqa: E402
from conceptdrill.hierarchy.captions import clean_caption_traced  # noqa: E402
from conceptdrill.hierarchy.docmodel_tree import load_tree        # noqa: E402
from conceptdrill.hierarchy.runlog import (RunLog,                 # noqa: E402
                                           concept_record)
from conceptdrill.hierarchy.structural import classify_marker    # noqa: E402
from conceptdrill.hierarchy.summarize import (SummaryCache,  # noqa: E402
                                              TitleOnlySummarizer,
                                              check_tier_independence,
                                              summarize_tree)


class NotMeasurementSafe(RuntimeError):
    """A summariser that must never stand in for a real one was requested."""


def make_summarizer(kind: str, llm_model: str, *, temperature: float = 0.2):
    """The requested summariser, or a raise. Never a substitute.

    There is no fallback path here on purpose. When the LLM is unreachable the
    correct outcome is that the run does not happen: a run that quietly
    produced extractive output while reporting an LLM question is what made the
    previous corpus build void.
    """
    if kind == "title-only":
        return TitleOnlySummarizer()
    if kind == "novita":
        from conceptdrill.hierarchy.novita import (DEFAULT_MODEL,
                                                   NovitaSummarizer,
                                                   load_dotenv, make_openai_chat)
        load_dotenv()
        model = llm_model or os.environ.get("NOVITA_MODEL") or DEFAULT_MODEL
        summarizer = NovitaSummarizer(
            make_openai_chat(model=model, temperature=temperature), model=model)
    else:
        raise NotMeasurementSafe(f"unknown summariser {kind!r}")

    if not getattr(summarizer, "measurement_safe", False):
        raise NotMeasurementSafe(
            f"{type(summarizer).__name__} is not measurement safe and must not "
            f"stand in for a summariser")
    return summarizer


def _error_for(summary, basis_text) -> str | None:
    """Why this span has no basis vector, or None when it has one."""
    if summary is None:
        return "no summary produced"
    if summary.error:
        return summary.error
    if not summary.is_usable:
        return "summary produced no usable tier text"
    if basis_text is not None and not basis_text.text:
        return "cleaning left no text: the span was entirely markup"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"))
    ap.add_argument("--out",
                    default=str(Path.home() / "conceptdrill-corpus-llm"),
                    help="run directories land here; keep them out of /tmp "
                         "so a measurement survives a reboot and is findable")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--docs", nargs="*", default=None,
                    help="explicit bibkeys; overrides --limit")
    ap.add_argument("--model", default="modernbert")
    ap.add_argument("--summarizer", default="novita",
                    choices=["novita", "title-only"],
                    help="'novita' is arm A; 'title-only' is the arm B "
                         "ablation. ExtractiveSummarizer is a test fixture and "
                         "is refused here.")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--token-ceiling", type=int, default=None,
                    help="tokens each tier must fit. Word bands are derived "
                         "from it at the measured 1.441 tokens/word and "
                         "substituted into the prompt, so a sweep never needs "
                         "the prompt edited by hand.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="decoding temperature. Part of the cache signature, "
                         "so changing it is a cache miss by construction.")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--summary-cache", default=".conceptdrill_cache/summaries.json")
    ap.add_argument("--speech", default="",
                    help="path to the la2speech project; renders maths through "
                         "SRE instead of the coarse fallback. Loaded by path, "
                         "never imported.")
    ap.add_argument("--name", default="current",
                    help="directory name under --out. Fixed by default so the "
                         "current run is findable without knowing a timestamp; "
                         "manifest.json still records the stamped run_id.")
    args = ap.parse_args()

    strict = os.environ.get("CONCEPTDRILL_STRICT", "") == "1"
    nlp_backend = os.environ.get("CONCEPTDRILL_NLP_BACKEND")

    speaker = None
    speech_info = None
    if args.speech:
        from conceptdrill.hierarchy.speech import describe, load_speaker
        speaker = load_speaker(args.speech)     # raises rather than degrading
        speech_info = describe(speaker)
        print(f"spoken maths: {speech_info}")

    library = Path(args.library)
    if args.docs:
        paths = [library / key / "model.docmodel.json" for key in args.docs]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"no docmodel at: {missing}")
    else:
        paths = sorted(library.glob("*/model.docmodel.json"))

    embedder = get_embedder(args.model, cache=True)
    summarizer = make_summarizer(args.summarizer, args.llm_model,
                                 temperature=args.temperature)
    is_ablation = bool(getattr(summarizer, "is_ablation", False))
    cache = SummaryCache(args.summary_cache) if args.summary_cache else None
    basis = ConceptBasis() if args.tau is None else ConceptBasis(tau=args.tau)
    from conceptdrill.hierarchy.summarize import bands_for, render_prompt
    prompt_text = render_prompt(args.token_ceiling)
    if args.token_ceiling and summarizer is not None:
        # The summariser carries its own prompt for the cache signature and
        # for the call itself; both must be the rendered one.
        summarizer.prompt = prompt_text

    log = RunLog.open(args.out, name=args.name)
    print(f"building in {log.root} -> {log.final_root} on success")

    used_paths: list[str] = []
    tier_violations = 0
    token_stats: dict[str, int] = {}
    math_sources: dict[str, int] = {}
    docs_done = 0

    for path in paths:
        if not args.docs and docs_done >= args.limit:
            break
        bibkey = path.parent.name
        tree = load_tree(path, speaker=speaker)
        if not len(tree):
            continue                       # no spans: nothing to account for
        docs_done += 1
        used_paths.append(str(path))
        for source, n in (tree.math_sources or {}).items():
            math_sources[source] = math_sources.get(source, 0) + n

        # Every node in the tree, in document order. This list is the ledger.
        nodes = list(tree.iter_document_order())
        ordered_markers = sorted(nodes, key=lambda n: (n.flow_index, n.id))
        log.expect(len(nodes))
        run = summarize_tree(tree, summarizer, cache=cache,
                             prompt=prompt_text)

        # Every concept's basis text goes through the one cleaner, whether or
        # not it reaches the basis, so `cleaning_rules_fired` is populated for
        # concepts that were later dropped. The CONCEPT is the unit: a span
        # defining three ideas contributes three candidates, not one blend.
        cleaned: dict[tuple[str, int], object] = {}
        for node in nodes:
            summary = run.summaries.get(node.id)
            if summary is None:
                continue
            for i, concept in enumerate(summary.concepts):
                # For the ablation arm the title *is* the content, so it must
                # not be stripped from the front of itself.
                cleaned[(node.id, i)] = clean_basis_text(
                    concept.basis_text,
                    title="" if is_ablation else node.title_raw)

        # Dimension zero. Classified before embedding so a structural span
        # never competes for a concept row, whatever it looks like in vector
        # space.
        classes: dict[str, tuple] = {
            node.id: classify_marker(node.title,
                                      is_appendix=node.is_appendix)
            for node in nodes}

        # Integrate only what survives cleaning, but keep the decision per
        # concept so the concepts that did not reach the basis say why.
        usable = []
        for node in nodes:
            summary = run.summaries.get(node.id)
            if summary is None:
                continue
            for i, concept in enumerate(summary.concepts):
                text = cleaned.get((node.id, i))
                if concept.is_usable and text is not None and text.text:
                    usable.append((node, i, text))

        decisions: dict[tuple[str, int], object] = {}
        if usable:
            texts = [t.text for _, _, t in usable]
            report = getattr(embedder, "token_report", None)
            if callable(report):
                r = report(texts)
                for k, v in r.items():
                    if k in ("texts", "truncated", "tokens_lost",
                             "over_70_token_window"):
                        token_stats[k] = token_stats.get(k, 0) + v
                    else:
                        token_stats[k] = max(token_stats.get(k, 0), v)
            vectors = embedder.encode(texts)
            for (node, i, basis_text), vector in zip(usable, vectors):
                if classes[node.id][0] is not None:
                    decisions[(node.id, i)] = basis.absorb_structural(
                        basis_text.text, vector, document=bibkey,
                        rule=classes[node.id][1])
                    continue
                decisions[(node.id, i)] = basis.integrate(
                    node.level, basis_text.text, vector, document=bibkey)
            if bibkey not in basis.document_order:
                basis.document_order = basis.document_order + (bibkey,)

        for node in nodes:
            summary = run.summaries.get(node.id)
            title_clean, title_rules = clean_caption_traced(node.title_raw)
            title_fired = [f"title:{r}" for r in title_rules]
            structural_class, structural_rule = classes.get(node.id, (None, None))

            own_chars = len(node.body_text)
            derivation = "own_text" if own_chars else "empty"

            # What the model was actually given, recorded so a future input
            # bug is visible in the artefact without re-running anything.
            # last_200 is the one that matters: it proves the model saw the
            # END of the span.
            sent = node.body_text
            after = [m.flow_index for m in ordered_markers
                     if m.flow_index > node.flow_index]
            extent_ids = [u.id for u in node.paragraphs]

            concepts = []
            for i, concept in enumerate(summary.concepts if summary else ()):
                basis_text = cleaned.get((node.id, i))
                result = decisions.get((node.id, i))
                degenerate = check_tier_independence(concept)
                tier_violations += len(degenerate)
                concepts.append(concept_record(
                    concept_index=i,
                    tier_label=concept.label or None,
                    tier_abstraction=concept.abstraction or None,
                    tier_summary=concept.summary or None,
                    basis_text=(basis_text.text
                                if basis_text is not None else None),
                    cleaning_rules_fired=(
                        [f"basis:{r}" for r in basis_text.rules_fired]
                        if basis_text is not None else []),
                    embedding_model=embedder.name if result is not None else None,
                    embedding_revision=(embedder.revision
                                        if result is not None else None),
                    row_id_assigned=getattr(result, "row_id", None) or None,
                    merge_decision=(getattr(result, "action", None)
                                    if result is not None else "not_integrated"),
                    merge_cosine=(round(float(result.similarity), 6)
                                  if result is not None else None),
                    merge_target_row_id=(result.row_id
                                         if result is not None
                                         and result.action == "merged" else None),
                    warnings=(list(concept.warnings)
                              + [f"tier: {d}" for d in degenerate]),
                    error=_error_for(concept, basis_text)))

            log.add_span(
                doc_id=bibkey,
                # One marker opens exactly one span, so the ids coincide today.
                # They are separate fields because they are separate things:
                # the marker is a DocModel object that never carries text, the
                # span is the content it opens.
                span_id=node.id,
                marker_id=node.id,
                level=node.level,
                flow_index=node.flow_index,
                is_appendix=node.is_appendix,
                title_raw=node.title_raw,
                title_cleaned=title_clean,
                cleaning_rules_fired=title_fired,
                structural_class=structural_class,
                structural_rule_fired=structural_rule,
                derivation=derivation,
                own_text_chars=own_chars,
                extent_object_ids=extent_ids,
                extent_object_count=len(extent_ids),
                extent_start_flow_index=node.flow_index,
                extent_end_flow_index=(min(after) if after else None),
                extent_end_reason=("next_marker" if after else "end_of_document"),
                subtree_object_count=len(
                    [u for n2 in [node] + list(tree.descendants(node.id))
                     for u in n2.paragraphs]),
                child_marker_ids=list(node.children),
                is_leaf=not node.children,
                input_text_sha256=(hashlib.sha256(sent.encode()).hexdigest()
                                   if sent else None),
                input_text_first_200=sent[:200] or None,
                input_text_last_200=sent[-200:] or None,
                concept_count=len(concepts),
                concepts=concepts,
                warnings=[],
                error=(None if summary else
                       ("no paragraphs of its own: the content belongs to its "
                        "subsections, which are summarised separately"
                        if derivation == "empty" else "no summary produced")))

    if cache is not None:
        cache.flush()
    flush = getattr(embedder, "flush", None)
    if callable(flush):
        flush()

    rows = [{"row_id": r.row_id, "label": r.label, "support": r.support,
             "structural": r.row_id == STRUCTURAL_ROW_ID,
             "level": r.level, "documents": list(r.documents),
             "contributing_concepts": sorted(
                 f"{rec['span_id']}#{c['concept_index']}"
                 for rec in log._spans for c in (rec["concepts"] or [])
                 if c["row_id_assigned"] == r.row_id)}
            for r in basis.ordered_rows()]

    # The matrix IS the basis. Writing row metadata without the vectors makes
    # every downstream question -- projection, dimension ranking, search --
    # require re-embedding the labels and hoping for the same numbers.
    matrix = basis.matrix()
    np.savez_compressed(
        str(log.root / "basis.npz"),
        matrix=matrix,
        row_ids=np.array(basis.row_ids(), dtype=object),
        basis_version=np.array(basis.basis_version()))

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
               "is_ablation": is_ablation,
               "tier_violations": tier_violations,
               "basis_text_tokens": token_stats,
               "speech_backend": speech_info,
               "temperature": args.temperature,
               "token_ceiling": args.token_ceiling,
               "tier_bands": {k: list(v) for k, v in
                              bands_for(args.token_ceiling or 50).items()},
               "summary_cache": args.summary_cache or None,
               "basis_version": basis.basis_version(),
               "basis_stats": basis.stats()})

    stats = basis.stats()
    print(f"spans: {len(log)}  documents: {docs_done}  "
          f"concept rows: {stats['rows']}  "
          f"(+1 structural sink, support {(stats['structural_row'] or {}).get('support', 0)})")
    print(f"run -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
