#!/usr/bin/env python3
r"""Which CES dimensions carry the corpus's mass?

    PYTHONPATH=src python3 tools/ces_gravity.py

The paper builds the full conceptual embedding and then applies "feature
selection on the full embedding to choose the top 20% concepts"
(arXiv 2209.00445, section 6.2). It does not prescribe the selection method.
This is one unsupervised instance of it: project the corpus into the full CES
space, sum each coordinate across everything projected, and rank the
dimensions by that total.

    CES(s) = M @ f(s)          one coordinate per basis row
    mass[j] = sum over s of CES(s)[j]

A dimension with large mass is one the corpus keeps projecting onto. A
dimension near zero is a concept nothing in the corpus resembles, and it costs
a coordinate in every stored vector.

TWO SUMS, REPORTED SEPARATELY. Cosine is signed, so a plain sum lets a
dimension that is strongly negative for half the corpus and strongly positive
for the other half cancel to nothing while being highly discriminative.
`mass` is the plain sum the brief asked for; `abs_mass` and `positive_mass`
are reported beside it so a cancelling dimension is visible rather than
silently ranked last.

The projection unit is the SENTENCE, per the design: spans built the basis, so
projecting them back would mostly measure a row's own support.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

import numpy as np                                                # noqa: E402

from conceptdrill.embeddings import get_embedder                  # noqa: E402
from conceptdrill.hierarchy.basis import STRUCTURAL_ROW_ID        # noqa: E402
from conceptdrill.hierarchy.docmodel_tree import load_tree        # noqa: E402
from conceptdrill.hierarchy.sentences import sentences_from_tree  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--model", default="sentencebert")
    ap.add_argument("--top", type=int, default=30, help="how many to print")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    basis = json.loads((run / "basis.json").read_text(encoding="utf-8"))
    arrays = np.load(run / "basis.npz", allow_pickle=True)
    matrix = np.asarray(arrays["matrix"], dtype=np.float64)
    row_ids = [str(r) for r in arrays["row_ids"]]
    rows = {r["row_id"]: r for r in basis["rows"]}

    if matrix.shape[0] != len(row_ids):
        raise SystemExit(f"matrix has {matrix.shape[0]} rows, "
                         f"{len(row_ids)} row_ids: refusing to guess")

    embedder = get_embedder(args.model, cache=True)

    # Sentences, from the same documents the run used.
    texts, origins = [], []
    for path in manifest["corpus_paths"]:
        tree = load_tree(Path(path))
        for sentence in sentences_from_tree(tree):
            texts.append(sentence.text)
            origins.append(Path(path).parent.name)
    if not texts:
        raise SystemExit("no sentences to project")

    print(f"projecting {len(texts)} sentences from "
          f"{len(set(origins))} documents into {matrix.shape[0]} dimensions")
    vectors = np.asarray(embedder.encode(texts), dtype=np.float64)
    ces = vectors @ matrix.T                       # (sentences, dimensions)

    mass = ces.sum(axis=0)
    abs_mass = np.abs(ces).sum(axis=0)
    positive_mass = np.clip(ces, 0, None).sum(axis=0)
    peak = ces.max(axis=0)

    order = np.argsort(-mass)
    ranked = []
    for rank, j in enumerate(order, start=1):
        rid = row_ids[j]
        row = rows.get(rid, {})
        ranked.append({
            "rank": rank, "dimension": int(j), "row_id": rid,
            "structural": rid == STRUCTURAL_ROW_ID,
            "level": row.get("level"), "support": row.get("support"),
            "documents": len(row.get("documents") or []),
            "mass": round(float(mass[j]), 4),
            "abs_mass": round(float(abs_mass[j]), 4),
            "positive_mass": round(float(positive_mass[j]), 4),
            "peak": round(float(peak[j]), 4),
            "mean": round(float(mass[j] / len(texts)), 6),
            "label": row.get("label", ""),
        })

    total = float(mass.sum())
    cumulative, curve = 0.0, []
    for entry in ranked:
        cumulative += entry["mass"]
        curve.append(cumulative / total if total else 0.0)
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = max(1, int(round(fraction * len(ranked))))
        print(f"  top {fraction:4.0%} of dimensions ({k:4d}) carry "
              f"{curve[k - 1]:6.1%} of the total mass")

    print(f"\n{'rank':>4s} {'mass':>9s} {'mean':>8s} {'peak':>6s} "
          f"{'sup':>4s} {'doc':>4s}  label")
    for entry in ranked[:args.top]:
        tag = "  [STRUCTURAL SINK]" if entry["structural"] else ""
        print(f"{entry['rank']:4d} {entry['mass']:9.2f} {entry['mean']:8.4f} "
              f"{entry['peak']:6.3f} {entry['support']:4} {entry['documents']:4} "
              f" {entry['label'][:74]}{tag}")

    payload = {
        "run_dir": str(run), "run_id": manifest["run_id"],
        "basis_version": str(arrays["basis_version"]),
        "sentences": len(texts), "dimensions": len(ranked),
        "criterion": "sum of CES coordinate over all projected sentences",
        "mass_curve": [round(c, 6) for c in curve],
        "dimensions_ranked": ranked,
    }
    target = Path(args.out) if args.out else run / "ces-gravity.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"\nwritten -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
