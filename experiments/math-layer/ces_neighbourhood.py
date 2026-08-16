#!/usr/bin/env python3
"""Steps 13-15: what does CES actually retrieve for the fixture?

    PYTHONPATH=src python3 experiments/math-layer/ces_neighbourhood.py

The math layer's whole premise is that CES concepts are a useful input to it.
That is a claim, and this measures it instead of assuming it. Three probes:

  13. The fixture's own sentences, and the phrase "vector space" alone.
  14. The step-14 variant -- "a linear transformation" for "linear" -- so the
      delta between two near-identical statements is measured rather than
      narrated.
  15. Candidate continuations, ranked by what the corpus puts near them.

## What the rows are, exactly

The 16 chunk bases are NOT merged: each was built independently, so the same
concept can hold a row in several of them. Concatenating them gives a retrieval
pool, not a basis, and it is used here only to answer "what does this corpus
put near this text". Every count below is over that pool and would change under
a real merge.

float64 before every matmul. This host's float32 GEMM is measurably wrong; the
basis is float64 for the same reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The installed torchvision (6 Aug) is broken against the installed torch
# (29 Jul): importing it raises `operator torchvision::nms does not exist`, and
# transformers 5.x touches it while resolving ModernBertModel, so the encoder
# will not load. Setting the entry to None makes `import torchvision` raise a
# clean ImportError, which transformers' lazy loader handles.
#
# Local to this experiment, and NOT a fix. ConceptDrill does not depend on
# torchvision; the installed copy needs repairing or removing. Verified after
# the guard: the encoder resolves revision d556a88e332558790b210f7bdbe87da2fa94a8d8,
# the same revision the corpus basis was built with, so these vectors are
# comparable with the stored rows.
sys.modules.setdefault("torchvision", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

import numpy as np                                                # noqa: E402

from conceptdrill.embeddings import get_embedder                  # noqa: E402

FULL = Path.home() / "conceptdrill-corpus-llm" / "full"
OUT = Path(__file__).resolve().parent / "ces-neighbourhood.json"
TOP_K = 8

#: Step 13/14. The pair differs only in `linear` versus `a linear
#: transformation`; the mathematics is identical, so any large difference in
#: what CES returns is a fact about wording, not about meaning.
PROBES = {
    "concept:vector space": "vector space",
    "concept:linear map": "linear map",
    "s1": "V is an n-dimensional vector space over K.",
    "s3-fixture": "Let T : V right arrow W be linear.",
    "s3-variant": "Let T : V right arrow W be a linear transformation.",
    "fixture-whole": ("V is an n-dimensional vector space over K. "
                      "W is an m-dimensional vector space over K. "
                      "Let T colon V right arrow W be linear."),
    "variant-whole": ("V is an n-dimensional vector space over K. "
                      "W is an m-dimensional vector space over K. "
                      "Let T colon V right arrow W be a linear transformation."),
}

#: Step 15. Predicted continuations, spoken as SRE would render them, so the
#: comparison is like-for-like with a corpus whose maths arrived as speech.
CONTINUATIONS = {
    "kernel": "the kernel of T is the set of v with T of v equals zero",
    "rank-nullity": ("dimension of the kernel of T plus dimension of the image "
                     "of T equals n"),
    "basis": "let v sub 1 through v sub n be a basis of V",
    "matrix": "the matrix of T with respect to the basis is an m by n matrix",
    "linearity": ("T of alpha u plus beta v equals alpha T of u plus beta T of v"),
    "isomorphism": "T is an isomorphism if it is injective and surjective",
}


def load_pool():
    """Every chunk's rows as one retrieval pool. Not a merge -- see the docstring."""
    vectors, labels, sources = [], [], []
    for chunk in sorted(FULL.iterdir()):
        npz, meta = chunk / "basis.npz", chunk / "basis.json"
        if not (npz.exists() and meta.exists()):
            continue
        with np.load(npz, allow_pickle=True) as z:
            matrix = np.asarray(z["matrix"], dtype=np.float64)
            row_ids = [str(x) for x in z["row_ids"]]
        by_id = {r["row_id"]: r for r in
                 json.loads(meta.read_text(encoding="utf-8"))["rows"]}
        for i, rid in enumerate(row_ids):
            rec = by_id.get(rid)
            if rec is None or rec.get("structural"):
                continue          # the sink is not a concept
            vectors.append(matrix[i])
            labels.append(rec["label"])
            sources.append(chunk.name)
    return np.vstack(vectors), labels, sources


def main() -> int:
    pool, labels, sources = load_pool()
    print(f"retrieval pool: {pool.shape[0]} concept rows, {pool.shape[1]}-d, "
          f"from {len(set(sources))} chunk bases (NOT merged)")

    embedder = get_embedder("modernbert", cache=True)
    keys = list(PROBES) + [f"next:{k}" for k in CONTINUATIONS]
    texts = list(PROBES.values()) + list(CONTINUATIONS.values())
    queries = np.asarray(embedder.encode(texts), dtype=np.float64)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    sims = (queries / norms) @ pool.T

    out = {}
    for key, row in zip(keys, sims):
        top = np.argsort(-row)[:TOP_K]
        out[key] = {
            "top1": round(float(row[top[0]]), 4),
            "hits": [{"similarity": round(float(row[i]), 4),
                      "label": labels[i], "chunk": sources[i]} for i in top],
        }
        print(f"\n=== {key}   top1 {out[key]['top1']}")
        for h in out[key]["hits"][:4]:
            print(f"  {h['similarity']:.3f}  {h['label'][:104]}")

    # Step 14: the delta between two statements that mean the same thing.
    def label_set(key):
        return {h["label"] for h in out[key]["hits"]}

    for a, b in (("s3-fixture", "s3-variant"),
                 ("fixture-whole", "variant-whole")):
        shared = label_set(a) & label_set(b)
        out[f"delta:{a}->{b}"] = {
            "shared_of_top_k": len(shared),
            "top_k": TOP_K,
            "added": sorted(label_set(b) - label_set(a)),
            "removed": sorted(label_set(a) - label_set(b)),
            "top1_change": round(out[b]["top1"] - out[a]["top1"], 4),
        }
        print(f"\n=== delta {a} -> {b}: {len(shared)}/{TOP_K} shared, "
              f"top1 {out[b]['top1'] - out[a]['top1']:+.4f}")

    OUT.write_text(json.dumps(
        {"pool_rows": int(pool.shape[0]),
         "pool_note": "16 independent chunk bases concatenated; not a merge",
         "embedder": "modernbert", "top_k": TOP_K,
         "probes": PROBES, "continuations": CONTINUATIONS,
         "results": out}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
