#!/usr/bin/env python3
r"""Does fluent nonsense occupy its own region of concept space?

    CONCEPTDRILL_STRICT=1 PYTHONPATH=src python3 tools/bullshit_basis.py

Every gate in this project asks whether a concept was faithfully extracted from
a span. None asks whether the span said anything. `docs/benchmarks/` holds 100
questions that sound professionally competent and are incoherent, each labelled
with the technique that made it so -- so for once the absence of meaning is
ground truth rather than a judgement.

WHAT THIS MEASURES. Each question is put through the same summariser the corpus
uses, producing concepts and therefore basis vectors. Separately, the 13
technique descriptions are embedded: they are the ontology, and each question
is a labelled instance of one class. Then two questions:

  1. SEPARABILITY. Is a question's concept vector closer to ITS OWN technique's
     description than to the other twelve? If yes, the failure mode survives
     summarisation and is visible in concept space. If the assignment is at
     chance -- 1/13, about 7.7% -- then summarising nonsense produces concepts
     that carry no trace of what was wrong with it.

  2. VOLUME. How much of the space do these concepts span, against real
     concepts from the corpus? A tight cluster means fluent nonsense is
     recognisable by position alone. A cloud interleaved with real concepts
     means it is not, and no threshold on similarity will separate them.

WHAT IT DOES NOT MEASURE. Whether the summariser NOTICED. The benchmark's own
scoring (0 engaged, 1 hedged, 2 flagged) is about a model answering the
question; here the model is asked to extract concepts, not to judge. A concept
faithfully describing a fabricated framework is correct behaviour for this
pipeline and a failure for a detector. Keeping those apart is the point of
running it at all.
"""
from __future__ import annotations

import argparse
import json
import os
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
from conceptdrill.hierarchy.basistext import clean_basis_text     # noqa: E402
from conceptdrill.hierarchy.summarize import (SummaryCache,       # noqa: E402
                                              render_prompt, summary_key)

BENCH = (Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
         / "bullshitbench-v2.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default=str(BENCH))
    ap.add_argument("--model", default="modernbert")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--token-ceiling", type=int, default=50)
    ap.add_argument("--compare", default="",
                    help="a run directory whose concepts stand for real text")
    ap.add_argument("--summary-cache", default=".conceptdrill_cache/bullshit.json")
    ap.add_argument("--out", default=str(Path.home() / "conceptdrill-corpus-llm"
                                         / "bullshit"))
    args = ap.parse_args()

    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    techniques = {t["technique"]: t["description"] for t in bench["techniques"]}
    questions = [q for t in bench["techniques"] for q in t["questions"]]
    print(f"{len(questions)} questions, {len(techniques)} techniques")

    from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                               load_dotenv, make_openai_chat)
    load_dotenv()
    llm = args.llm_model or os.environ.get("NOVITA_MODEL") or DEFAULT_MODEL
    prompt = render_prompt(args.token_ceiling)
    summarizer = NovitaSummarizer(
        make_openai_chat(model=llm, temperature=args.temperature), model=llm)
    summarizer.prompt = prompt
    cache = SummaryCache(args.summary_cache)
    embedder = get_embedder(args.model, cache=True)

    # 1. Concepts from each question, exactly as a span would be summarised.
    rows = []
    for i, q in enumerate(questions, start=1):
        title = f"{q['domain']} ({q['domain_group']})"
        body = q["question"]
        key = summary_key(title, body,
                          getattr(summarizer, "cache_signature", llm), prompt)
        summary = cache.get(key)
        if summary is None:
            summary = summarizer.summarize(q["id"], title, body)
            if summary.is_usable:
                cache.put(key, summary)
        for j, concept in enumerate(summary.concepts):
            text = clean_basis_text(concept.basis_text, title="").text
            if text:
                rows.append({"id": q["id"], "technique": q["technique"],
                             "domain_group": q["domain_group"],
                             "concept_index": j, "basis_text": text,
                             "label": concept.label})
        print(f"  [{i:3d}/{len(questions)}] {q['id']:14s} "
              f"{q['technique'][:26]:26s} {len(summary.concepts)} concepts"
              f"{'  ERROR' if summary.error else ''}")
        sys.stdout.flush()
    cache.flush()
    if not rows:
        raise SystemExit("no concepts produced")

    # 2. The ontology: one vector per technique description.
    names = sorted(techniques)
    onto = np.asarray(embedder.encode([techniques[n] for n in names]),
                      dtype=np.float64)
    vecs = np.asarray(embedder.encode([r["basis_text"] for r in rows]),
                      dtype=np.float64)

    # 3. Separability: is a concept nearest its own technique's description?
    sims = vecs @ onto.T
    index = {n: i for i, n in enumerate(names)}
    nearest = sims.argmax(axis=1)
    correct = sum(1 for r, k in zip(rows, nearest)
                  if names[k] == r["technique"])
    chance = 1.0 / len(names)
    print(f"\nconcepts: {len(rows)}   techniques: {len(names)}")
    print(f"nearest-technique accuracy: {correct}/{len(rows)} "
          f"({correct / len(rows):.1%}) against {chance:.1%} chance")

    per = defaultdict(lambda: {"n": 0, "hit": 0, "own": [], "other": []})
    for r, row in zip(rows, sims):
        d = per[r["technique"]]
        d["n"] += 1
        d["hit"] += names[int(row.argmax())] == r["technique"]
        own = float(row[index[r["technique"]]])
        d["own"].append(own)
        d["other"].extend(float(v) for i, v in enumerate(row)
                          if names[i] != r["technique"])
    print(f"\n{'technique':34s} {'n':>4s} {'nearest':>8s} {'own':>7s} "
          f"{'other':>7s} {'margin':>7s}")
    for name in sorted(per, key=lambda k: -per[k]["n"]):
        d = per[name]
        own, other = float(np.mean(d["own"])), float(np.mean(d["other"]))
        print(f"{name:34s} {d['n']:4d} {d['hit'] / d['n']:7.0%} "
              f"{own:7.3f} {other:7.3f} {own - other:+7.3f}")

    # 4. Volume: how tightly do these concepts sit, and where?
    basis = ConceptBasis()
    for r, v in zip(rows, vecs):
        basis.integrate(1, r["basis_text"], v, document=r["technique"])
    gram = vecs @ vecs.T
    off = gram[~np.eye(len(vecs), dtype=bool)]
    print(f"\nbasis rows from nonsense: {basis.stats()['rows']} "
          f"of {len(rows)} concepts (tau {basis.tau})")
    print(f"pairwise cosine  p25 {np.percentile(off, 25):.3f}  "
          f"p50 {np.percentile(off, 50):.3f}  p75 {np.percentile(off, 75):.3f}")

    comparison = None
    if args.compare:
        run = Path(args.compare)
        real = [c["basis_text"]
                for line in (run / "spans.jsonl").read_text().splitlines()
                if line.strip()
                for c in (json.loads(line).get("concepts") or [])
                if c.get("basis_text")]
        real = real[:len(rows) * 3]
        rv = np.asarray(embedder.encode(real), dtype=np.float64)
        rg = rv @ rv.T
        ro = rg[~np.eye(len(rv), dtype=bool)]
        cross = (vecs @ rv.T).ravel()
        comparison = {
            "real_concepts": len(real),
            "real_pairwise_p50": float(np.percentile(ro, 50)),
            "nonsense_pairwise_p50": float(np.percentile(off, 50)),
            "cross_p50": float(np.percentile(cross, 50)),
            "cross_p99": float(np.percentile(cross, 99)),
        }
        print(f"\nagainst {len(real)} real concepts from {run.name}:")
        print(f"  real pairwise     p50 {comparison['real_pairwise_p50']:.3f}")
        print(f"  nonsense pairwise p50 {comparison['nonsense_pairwise_p50']:.3f}")
        print(f"  real vs nonsense  p50 {comparison['cross_p50']:.3f}  "
              f"p99 {comparison['cross_p99']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bullshit-basis.json").write_text(json.dumps({
        "benchmark": bench["benchmark"], "version": bench["version"],
        "llm": llm, "embedder": args.model, "token_ceiling": args.token_ceiling,
        "questions": len(questions), "concepts": len(rows),
        "nearest_technique_accuracy": correct / len(rows),
        "chance": chance,
        "per_technique": {k: {"n": v["n"], "nearest": v["hit"] / v["n"],
                              "own_mean": float(np.mean(v["own"])),
                              "other_mean": float(np.mean(v["other"]))}
                          for k, v in per.items()},
        "basis_rows": basis.stats()["rows"],
        "comparison": comparison,
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(str(out / "bullshit-vectors.npz"),
                        vectors=vecs, ontology=onto,
                        names=np.array(names, dtype=object),
                        ids=np.array([r["id"] for r in rows], dtype=object))
    print(f"\nwritten -> {out / 'bullshit-basis.json'} and bullshit-vectors.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
