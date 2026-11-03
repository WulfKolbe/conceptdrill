"""kb build's clustering: the same clusters, fast enough for a real set.

The first `_agglomerate` recomputed every pair's average similarity on every
merge: 2 s for 100 sections, 16 s for 200, 130 s for 400 -- hours for the 87
OCR-set papers. The fast version keeps a cluster-similarity matrix updated by
the size-weighted average (exactly the mean over member pairs). It must give
the SAME clusters as the first: a changed cluster changes a writer prompt, and
a changed prompt is a cache miss that spends. `reference` below is the first
version, kept verbatim as the oracle.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from conceptdrill.kb.build import _agglomerate


def reference(vectors, sizes, *, max_size, tau=None, target=None):
    clusters = [[i] for i in range(len(vectors))]
    weight = list(sizes)
    sims = vectors @ vectors.T if len(vectors) else np.zeros((0, 0))
    while len(clusters) > 1:
        if target is not None and len(clusters) <= target:
            break
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if weight[a] + weight[b] > max_size:
                    continue
                s = float(np.mean(sims[np.ix_(clusters[a], clusters[b])]))
                if s > best + 1e-12:
                    best, pair = s, (a, b)
        if pair is None or (tau is not None and best < tau):
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        weight[a] += weight[b]
        del clusters[b], weight[b]
    return clusters


def _vectors(n, seed, dup=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 16))
    for i in range(dup):                 # exact duplicates: real sets repeat sections
        v[n - 1 - i] = v[i]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kw", [dict(max_size=6, tau=0.3), dict(max_size=6, tau=-1.0),
                                dict(max_size=8, target=4), dict(max_size=3, target=2)])
def test_the_fast_clustering_gives_the_reference_clusters(seed, kw):
    v = _vectors(40, seed, dup=3)
    assert _agglomerate(v, [1] * len(v), **kw) == reference(v, [1] * len(v), **kw)


def test_weighted_start_sizes_are_respected():
    v = _vectors(20, 9)
    sizes = [1, 2, 3] * 6 + [1, 1]
    assert _agglomerate(v, sizes, max_size=6, tau=0.0) == reference(v, sizes, max_size=6, tau=0.0)


def test_it_is_fast_enough_for_a_real_set():
    v = _vectors(400, 1)
    t = time.monotonic()
    _agglomerate(v, [1] * 400, max_size=6, tau=0.0)
    assert time.monotonic() - t < 10, "was 130 s at n=400"
