#!/usr/bin/env python3
"""GATE 5: does the revised prompt actually produce what it asks for?

    CONCEPTDRILL_STRICT=1 python3 tools/gate5_prompt.py

Summarises the frozen 20-section fixture and checks the result mechanically.
Four clauses, all hard:

    parse failures                     0
    labels inside 30-42 words          100%
    labels/abstractions free of the    100%
      banned constructions
    labels with a parenthesised        0
      acronym

The fixture is frozen so two prompt revisions are compared on identical input.
Results are written next to the run directories rather than to /tmp: a
measurement a reboot can delete is not a record of anything.

The summary cache is keyed on the prompt text, so a revised prompt is a cache
miss by construction and this always exercises the model rather than serving
last revision's answers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

from conceptdrill.hierarchy.docmodel_tree import load_tree        # noqa: E402
from conceptdrill.hierarchy.labelcheck import (bare_acronyms,     # noqa: E402
                                               check_abstraction, check_label,
                                               check_summary, word_count)
from conceptdrill.hierarchy.summarize import (SummaryCache,       # noqa: E402
                                              load_prompt, summary_key)

FIXTURE = (Path(__file__).resolve().parents[1] / "docs" / "measurements"
           / "gate5-fixture-20.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", default=str(Path.home() / "pdfdrill-library"))
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--out", default=str(Path.home() / "conceptdrill-corpus-llm"
                                         / "gate5"))
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--summary-cache",
                    default=".conceptdrill_cache/summaries.json")
    args = ap.parse_args()

    from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                               load_dotenv, make_openai_chat)
    load_dotenv()
    model = args.llm_model or os.environ.get("NOVITA_MODEL") or DEFAULT_MODEL
    summarizer = NovitaSummarizer(make_openai_chat(model=model), model=model)
    prompt = load_prompt()
    cache = SummaryCache(args.summary_cache) if args.summary_cache else None

    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    wanted = fixture["sections"]

    trees: dict[str, object] = {}
    results = []
    started = time.monotonic()

    for i, entry in enumerate(wanted, start=1):
        doc = entry["doc_id"]
        if doc not in trees:
            trees[doc] = load_tree(Path(args.library) / doc
                                   / "model.docmodel.json")
        tree = trees[doc]
        node = tree.nodes[entry["section_id"]]
        title = node.summarizer_title
        body = tree.subtree_text(node.id)

        key = summary_key(title, body,
                          getattr(summarizer, "cache_signature", summarizer.name),
                          prompt)
        summary = cache.get(key) if cache else None
        served_from_cache = summary is not None
        if summary is None:
            summary = summarizer.summarize(node.id, title, body)
            if cache is not None and summary.is_usable:
                cache.put(key, summary)

        label_problems = check_label(summary.label)
        abstraction_problems = check_abstraction(summary.abstraction)
        summary_problems = check_summary(summary.summary)

        results.append({
            "doc_id": doc, "section_id": node.id, "title": entry["title"],
            "cached": served_from_cache,
            "parse_failed": bool(summary.error),
            "error": summary.error or None,
            "label": summary.label, "abstraction": summary.abstraction,
            "summary": summary.summary,
            "label_words": word_count(summary.label),
            "abstraction_words": word_count(summary.abstraction),
            "summary_words": word_count(summary.summary),
            "label_problems": label_problems,
            "abstraction_problems": abstraction_problems,
            "summary_problems": summary_problems,
            "bare_acronyms_in_label": bare_acronyms(summary.label),
            "warnings": list(summary.warnings),
        })
        print(f"  [{i:2d}/{len(wanted)}] {doc[:14]:14s} "
              f"{entry['title'][:34]:34s} "
              f"{'CACHED' if served_from_cache else '      '} "
              f"label={word_count(summary.label):3d}w "
              f"{'FAIL' if (label_problems or abstraction_problems or summary.error) else 'ok'}")
        sys.stdout.flush()

    if cache is not None:
        cache.flush()

    n = len(results)
    parse_failures = [r for r in results if r["parse_failed"]]
    out_of_band = [r for r in results if not 30 <= r["label_words"] <= 42]
    banned = [r for r in results
              if any("banned construction" in p
                     for p in r["label_problems"] + r["abstraction_problems"])]
    acronyms = [r for r in results
                if any("parenthesised acronym" in p for p in r["label_problems"])]

    clauses = {
        "parse_failures": (len(parse_failures), 0),
        "labels_outside_30_42_words": (len(out_of_band), 0),
        "banned_constructions_in_label_or_abstraction": (len(banned), 0),
        "labels_with_a_parenthesised_acronym": (len(acronyms), 0),
    }
    passed = all(got == want for got, want in clauses.values())

    report = {
        "gate": "GATE 5 (prompt)", "passed": passed,
        "model": model, "sections": n,
        "prompt_sha256": __import__("hashlib").sha256(
            prompt.encode()).hexdigest()[:16],
        "clauses": {k: {"observed": got, "threshold": want}
                    for k, (got, want) in clauses.items()},
        "label_words": {
            "min": min((r["label_words"] for r in results), default=0),
            "max": max((r["label_words"] for r in results), default=0),
            "mean": round(sum(r["label_words"] for r in results) / n, 1) if n else 0,
        },
        "seconds": round(time.monotonic() - started, 1),
        "results": results,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = out_dir / f"gate5-{stamp}.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    print(f"\nGATE 5 (prompt): {'PASS' if passed else 'FAIL'}")
    for name, (got, want) in clauses.items():
        mark = "ok" if got == want else "FAIL"
        print(f"  {name}: {got} (threshold {want}) {mark}")
    print(f"  label words: min {report['label_words']['min']} "
          f"max {report['label_words']['max']} "
          f"mean {report['label_words']['mean']}")
    print(f"report -> {target}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
