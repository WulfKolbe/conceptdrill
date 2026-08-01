#!/usr/bin/env python3
"""The concepts a run produced, laid out for a human to read against the source.

    PYTHONPATH=src python3 tools/concepts_report.py ~/conceptdrill-corpus-llm/current

Every gate so far answers a mechanical question -- is it truncated, is it in
budget, do the tiers overlap. None of them answers the only question that
finally matters: does this concept describe what the span actually says.

That is read, not computed. This writes `concepts.md`, grouped by document and
span in reading order, with each span's own text beside the concepts derived
from it, so the comparison needs no other file open.

Spans that produced nothing appear too, with the reason, because a reader
scanning for gaps should see them rather than infer them from absence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--source-chars", type=int, default=0,
                    help="truncate span text to this many characters; 0 shows "
                         "all of it, which is the point of the report")
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"),
                    help="where to re-read the spans' full text from")
    ap.add_argument("--speech", default=str(Path.home() / "la2speech"),
                    help="render maths the same way the run did")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    spans = [json.loads(line)
             for line in (run / "spans.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]

    # spans.jsonl keeps only the first and last 200 characters of each input --
    # enough to prove what the model saw, not enough to read. A reader looking
    # for a specific equation found "…" where it should have been. Re-derive
    # the full text from the documents the run recorded.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from conceptdrill.hierarchy.docmodel_tree import load_tree
    speaker = None
    if manifest.get("speech_backend") and args.speech:
        try:
            from conceptdrill.hierarchy.speech import load_speaker
            speaker = load_speaker(args.speech)
        except Exception as exc:
            print(f"note: rendering maths without a speech backend ({exc})")
    full: dict[tuple[str, str], str] = {}
    for path in manifest.get("corpus_paths") or []:
        tree = load_tree(Path(path), speaker=speaker)
        doc = Path(path).parent.name
        for node in tree.iter_document_order():
            full[(doc, node.id)] = node.body_text

    bands = manifest.get("tier_bands") or {}
    out = [f"# Concepts — {manifest['run_id']}", "",
           f"- model `{manifest.get('embedder_backend')}` "
           f"rev `{(manifest.get('embedder_resolved_revision') or '')[:12]}`",
           f"- summariser `{manifest.get('summarizer_name')}` "
           f"temperature {manifest.get('temperature')}",
           f"- token ceiling {manifest.get('token_ceiling') or 50}, bands "
           + ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in bands.items()),
           f"- {manifest.get('span_count')} spans, "
           f"{sum(s['concept_count'] for s in spans)} concepts, "
           f"{(manifest.get('basis_stats') or {}).get('rows')} basis rows",
           "",
           "Each span shows the text the model was given, then the concepts it "
           "returned. Read the concepts against the text.", ""]

    by_doc: dict[str, list] = {}
    for span in spans:
        by_doc.setdefault(span["doc_id"], []).append(span)

    for doc in sorted(by_doc):
        rows = sorted(by_doc[doc], key=lambda s: s["flow_index"])
        out += [f"\n## {doc}", ""]
        for span in rows:
            title = span["title_cleaned"] or "(untitled)"
            head = f"### L{span['level']} {title}"
            if span["structural_class"]:
                head += f"  _[structural: {span['structural_class']}]_"
            out += [head, ""]
            if span["derivation"] == "empty":
                out += ["_no paragraphs of its own; its subsections carry the "
                        "content_", ""]
                continue

            body = full.get((span["doc_id"], span["span_id"]))
            if body is None:
                head = " ".join((span.get("input_text_first_200") or "").split())
                tail = " ".join((span.get("input_text_last_200") or "").split())
                shown = f"{head} … …{tail}" if span["own_text_chars"] > 400 else head
            else:
                shown = body if not args.source_chars else body[:args.source_chars]
            quoted = "\n".join(f"> {line}" if line.strip() else ">"
                                for line in shown.splitlines())
            out += [quoted, "",
                    f"_{span['own_text_chars']} chars, "
                    f"{span['extent_object_count']} objects, "
                    f"{span['concept_count']} concepts_", ""]

            for concept in span["concepts"] or []:
                label = concept["tier_label"] or "(no label)"
                out += [f"**{concept['concept_index'] + 1}. {label}**", ""]
                if concept["tier_abstraction"]:
                    out += [f"- abstraction: {concept['tier_abstraction']}"]
                if concept["tier_summary"]:
                    out += [f"- summary: {concept['tier_summary']}"]
                decision = concept.get("merge_decision")
                cos = concept.get("merge_cosine")
                out += [f"- basis: {decision}"
                        + (f", cosine {cos:.3f}" if isinstance(cos, float)
                           and cos >= 0 else ""), ""]
                if concept.get("error"):
                    out += [f"- ERROR: {concept['error']}", ""]

    target = Path(args.out) if args.out else run / "concepts.md"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(spans)} spans, {sum(s['concept_count'] for s in spans)} concepts")
    print(f"written -> {target}  ({target.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
