r"""The geometry of the basis: scalar products, difference lengths, parallelism.

    PYTHONPATH=src python3 tools/basis_geometry.py ~/conceptdrill-corpus-llm/current

Visualisation and dimensionality-reduction methods start from one of two
matrices, so both are written rather than derived on demand:

    gram[i][j] = c_i . c_j          scalar products
    dist[i][j] = || c_i - c_j ||    length of the difference vector

For unit rows the two are related, `dist^2 = 2 - 2*gram`, but they are not
interchangeable in use: PCA and kernel methods want the Gram matrix, MDS and
t-SNE want distances, and writing only one leaves the caller to rederive the
other and to get the normalisation assumption right. `basis.py` guarantees
unit rows; this tool CHECKS that rather than assuming it, because the identity
above is false the moment it stops holding.

NEARLY PARALLEL ROWS ARE THE POINT. Two basis rows at cosine 0.95 are one
concept occupying two CES coordinates: every vector projected into the space
spends two numbers saying the same thing, and any later pruning that keeps one
and drops the other loses nothing while pruning that keeps both gains nothing.
They are reported with both rows' labels so the pair can be judged.

The structural sink is included in the matrices -- it is a real row of M -- but
excluded from the parallelism report, since it is a sink and its neighbours
mean something different.
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

from conceptdrill.hierarchy.basis import STRUCTURAL_ROW_ID        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?",
                    default=str(Path.home() / "conceptdrill-corpus-llm" / "current"))
    ap.add_argument("--parallel-at", type=float, default=0.9,
                    help="cosine above which two rows are called near-parallel")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    run = Path(args.run_dir)
    arrays = np.load(run / "basis.npz", allow_pickle=True)
    matrix = np.asarray(arrays["matrix"], dtype=np.float64)
    row_ids = [str(r) for r in arrays["row_ids"]]
    rows = {r["row_id"]: r
            for r in json.loads((run / "basis.json").read_text())["rows"]}
    n = matrix.shape[0]

    norms = np.linalg.norm(matrix, axis=1)
    unit = bool(np.allclose(norms, 1.0, atol=1e-6))
    print(f"basis {n} x {matrix.shape[1]}   rows unit-norm: {unit} "
          f"(min {norms.min():.6f}, max {norms.max():.6f})")
    if not unit:
        print("  NOT unit-norm: dist^2 = 2 - 2*gram does not hold here, and "
              "the two matrices are genuinely independent.")

    gram = matrix @ matrix.T                              # scalar products
    sq = np.maximum(0.0, norms[:, None]**2 + norms[None, :]**2 - 2.0 * gram)
    dist = np.sqrt(sq)                                    # difference lengths
    np.fill_diagonal(dist, 0.0)

    off = ~np.eye(n, dtype=bool)
    cos = gram[off]
    print(f"\nscalar products, off-diagonal ({cos.size} pairs):")
    for q in (1, 25, 50, 75, 99):
        print(f"  p{q:<3d} {np.percentile(cos, q):+.4f}")
    print(f"  min  {cos.min():+.4f}   max {cos.max():+.4f}   "
          f"mean {cos.mean():+.4f}")
    d = dist[off]
    print(f"difference lengths ||c_i - c_j||:")
    for q in (1, 25, 50, 75, 99):
        print(f"  p{q:<3d} {np.percentile(d, q):.4f}")
    print(f"  min  {d.min():.4f}   max {d.max():.4f}")

    # Effective dimensionality: how many directions the basis really spans.
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    total = float(eigenvalues.sum())
    cumulative = np.cumsum(np.clip(eigenvalues, 0, None)) / total
    print(f"\nrank of the Gram matrix: "
          f"{int(np.linalg.matrix_rank(gram, tol=1e-8))} of {n}")
    for frac in (0.5, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(cumulative, frac) + 1)
        print(f"  {frac:4.0%} of the spectrum in {k:4d} directions "
              f"({k / n:5.1%} of the rows)")

    # Near-parallel pairs: one concept holding two coordinates.
    concept = [i for i, rid in enumerate(row_ids) if rid != STRUCTURAL_ROW_ID]
    pairs = []
    for a_pos, i in enumerate(concept):
        for j in concept[a_pos + 1:]:
            if gram[i, j] >= args.parallel_at:
                pairs.append((float(gram[i, j]), float(dist[i, j]), i, j))
    pairs.sort(reverse=True)
    print(f"\nnear-parallel concept pairs at cosine >= {args.parallel_at}: "
          f"{len(pairs)}")
    for c, dd, i, j in pairs[:args.top]:
        li = rows.get(row_ids[i], {}).get("label", "")[:56]
        lj = rows.get(row_ids[j], {}).get("label", "")[:56]
        same_level = (rows.get(row_ids[i], {}).get("level")
                      == rows.get(row_ids[j], {}).get("level"))
        print(f"  cos {c:.3f}  dist {dd:.3f}  same_level={same_level}")
        print(f"     A {li}")
        print(f"     B {lj}")

    np.savez_compressed(str(run / "basis-geometry.npz"),
                        gram=gram, dist=dist, row_ids=np.array(row_ids, dtype=object),
                        eigenvalues=eigenvalues)
    summary = {
        "run_dir": str(run), "rows": n, "dimensions": int(matrix.shape[1]),
        "rows_unit_norm": unit,
        "scalar_products": {f"p{q}": float(np.percentile(cos, q))
                            for q in (1, 25, 50, 75, 99)},
        "difference_lengths": {f"p{q}": float(np.percentile(d, q))
                               for q in (1, 25, 50, 75, 99)},
        "gram_rank": int(np.linalg.matrix_rank(gram, tol=1e-8)),
        "directions_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1),
        "near_parallel_threshold": args.parallel_at,
        "near_parallel_pairs": [
            {"cosine": round(c, 6), "distance": round(dd, 6),
             "row_a": row_ids[i], "row_b": row_ids[j],
             "label_a": rows.get(row_ids[i], {}).get("label", ""),
             "label_b": rows.get(row_ids[j], {}).get("label", "")}
            for c, dd, i, j in pairs],
    }
    (run / "basis-geometry.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nmatrices -> {run / 'basis-geometry.npz'}  (gram, dist, eigenvalues)")
    print(f"summary  -> {run / 'basis-geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
