"""The seven quality metrics.

Each is a pure function `(candidate, ctx) -> float` in [0, 1], registered in
`METRICS`. Swapping one out means registering a different callable under the
same key — no other module is aware of how any metric is computed.

`ScoringContext` carries everything the metrics share: block embeddings, the
concept's own embedding, section labels, corpus statistics. Computing it once
for the whole candidate set is what keeps scoring to a single embedding pass.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..candidates.base import structural_weight
from ..types import Candidate

#: Similarity above which a block counts as "about" a concept.
DEFAULT_THETA = 0.5

#: Variance normaliser for the embedding-variance metric.
DEFAULT_MAX_VARIANCE = 2.0


@dataclass
class ScoringContext:
    """Shared state for one scoring pass over one document."""

    #: (n_blocks, dim) unit-norm block embeddings.
    block_embeddings: np.ndarray
    #: Top-level section id per block, aligned with `block_embeddings`.
    block_sections: Sequence[Optional[str]]
    #: (n_candidates, dim) unit-norm candidate embeddings.
    candidate_embeddings: np.ndarray
    #: Candidate name -> row index into `candidate_embeddings`.
    candidate_index: dict[str, int]
    theta: float = DEFAULT_THETA
    max_variance: float = DEFAULT_MAX_VARIANCE
    #: Lowercased word -> corpus frequency, for the reusability heuristic.
    word_frequencies: Counter = field(default_factory=Counter)
    #: Highest single word frequency, cached for normalisation.
    max_word_frequency: int = 1
    #: Entropy of the section-label distribution, cached for information gain.
    _base_entropy: Optional[float] = None
    #: Similarity row cache, keyed by candidate name.
    _sims: dict[str, np.ndarray] = field(default_factory=dict)

    def similarities(self, candidate: Candidate) -> np.ndarray:
        """Cosine similarity of `candidate` against every block.

        Both sides are unit-norm, so this is a plain dot product. Cached because
        four of the seven metrics need the same row.
        """
        key = candidate.key
        cached = self._sims.get(key)
        if cached is not None:
            return cached
        idx = self.candidate_index.get(key)
        if idx is None or self.block_embeddings.size == 0:
            sims = np.zeros(len(self.block_sections), dtype=np.float32)
        else:
            sims = self.block_embeddings @ self.candidate_embeddings[idx]
        self._sims[key] = sims
        return sims

    def matching_mask(self, candidate: Candidate) -> np.ndarray:
        return self.similarities(candidate) > self.theta

    @property
    def n_blocks(self) -> int:
        return len(self.block_sections)

    @property
    def base_entropy(self) -> float:
        if self._base_entropy is None:
            self._base_entropy = _entropy(self.block_sections)
        return self._base_entropy


def _entropy(labels: Sequence[Optional[str]]) -> float:
    """Shannon entropy in bits over a label sequence."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


# --------------------------------------------------------------------------
# 2.1 Structural importance
# --------------------------------------------------------------------------

def structural(candidate: Candidate, ctx: ScoringContext) -> float:
    """Where the candidate came from. The one metric that needs no embedding."""
    return structural_weight(candidate.source)


# --------------------------------------------------------------------------
# 2.2 Coverage
# --------------------------------------------------------------------------

def _coverage_curve(fraction: float) -> float:
    """Piecewise-linear penalty on the fraction of blocks a concept matches.

    A concept matching almost nothing is useless; one matching almost everything
    is not a distinction. The reward peaks on the 0.2-0.4 plateau, per the spec:

        <=0.01 -> 0.10      0.20-0.40 -> 1.00
         0.10  -> 0.55       0.90+    -> 0.30
    """
    f = float(max(0.0, min(1.0, fraction)))
    if f <= 0.01:
        return 0.1
    if f < 0.2:
        # 0.01 -> 0.1 rising to 0.2 -> 1.0
        return 0.1 + (f - 0.01) * (1.0 - 0.1) / (0.2 - 0.01)
    if f <= 0.4:
        return 1.0
    if f < 0.9:
        # 0.4 -> 1.0 falling to 0.9 -> 0.3
        return 1.0 - (f - 0.4) * (1.0 - 0.3) / (0.9 - 0.4)
    return 0.3


def coverage(candidate: Candidate, ctx: ScoringContext) -> float:
    if ctx.n_blocks == 0:
        return 0.0
    fraction = float(ctx.matching_mask(candidate).sum()) / ctx.n_blocks
    return _coverage_curve(fraction)


# --------------------------------------------------------------------------
# 2.3 Purity
# --------------------------------------------------------------------------

def purity(candidate: Candidate, ctx: ScoringContext) -> float:
    """How confined the concept is to one part of the document.

    1 minus the share of matching blocks that fall in the concept's single most
    frequent top-level section. A concept spread evenly across ten sections
    scores high; one that *is* a section scores low.

    A concept matching nothing gets 0.0 rather than a vacuous 1.0 — no evidence
    is not the same as good evidence.
    """
    mask = ctx.matching_mask(candidate)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    sections = [ctx.block_sections[i] for i in np.flatnonzero(mask)]
    dominant = Counter(sections).most_common(1)[0][1]
    return float(1.0 - dominant / total)


# --------------------------------------------------------------------------
# 2.4 Information gain
# --------------------------------------------------------------------------

def information_gain(candidate: Candidate, ctx: ScoringContext) -> float:
    """Entropy reduction from splitting blocks on "is about this concept".

    Straight decision-tree information gain over section labels, normalised by
    the base entropy so the result lands in [0, 1]. A concept that cleanly
    separates one section from the rest scores high.
    """
    if ctx.n_blocks == 0:
        return 0.0
    base = ctx.base_entropy
    if base <= 0.0:
        # One section, or none: no entropy to reduce, so the metric is
        # uninformative rather than zero-valued. Neutral is the honest answer.
        return 0.5

    mask = ctx.matching_mask(candidate)
    n_pos = int(mask.sum())
    n_neg = ctx.n_blocks - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos = [ctx.block_sections[i] for i in np.flatnonzero(mask)]
    neg = [ctx.block_sections[i] for i in np.flatnonzero(~mask)]
    weighted = (n_pos / ctx.n_blocks) * _entropy(pos) + \
               (n_neg / ctx.n_blocks) * _entropy(neg)
    return float(max(0.0, min(1.0, (base - weighted) / base)))


# --------------------------------------------------------------------------
# 2.5 Embedding variance
# --------------------------------------------------------------------------

def embedding_variance(candidate: Candidate, ctx: ScoringContext) -> float:
    """Coherence of the blocks a concept matches.

    Total variance across dimensions of the matching blocks' embeddings; low
    variance means the concept picks out a tight cluster. Scored as
    `1 - variance / max_variance`, clamped.

    Fewer than two matches leaves variance undefined, so the metric returns
    neutral rather than a misleading 1.0.
    """
    mask = ctx.matching_mask(candidate)
    n = int(mask.sum())
    if n < 2:
        return 0.5
    subset = ctx.block_embeddings[mask]
    variance = float(subset.var(axis=0).sum())
    if ctx.max_variance <= 0:
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - variance / ctx.max_variance)))


# --------------------------------------------------------------------------
# 2.6 Citation importance
# --------------------------------------------------------------------------

CURRENT_YEAR = 2026
RECENCY_WINDOW = 5


def citation_importance(candidate: Candidate, ctx: ScoringContext) -> float:
    """`min(1, log10(citations+1)/4) * recency`, for bibliography candidates.

    Everything else is neutral at 0.5, as specified. A bibliography entry with
    no citation count is *also* neutral: PDF-parsed references carry no counts,
    and inventing one would be worse than declining to score.
    """
    if candidate.source != "bibliography":
        return 0.5

    raw = candidate.metadata.get("citations")
    if raw is None:
        return 0.5
    try:
        citations = max(0, int(raw))
    except (TypeError, ValueError):
        return 0.5

    magnitude = min(1.0, math.log10(citations + 1) / 4.0)

    year = candidate.metadata.get("year")
    recency = 0.8
    try:
        if year is not None and CURRENT_YEAR - int(year) <= RECENCY_WINDOW:
            recency = 1.0
    except (TypeError, ValueError):
        pass

    return float(max(0.0, min(1.0, magnitude * recency)))


# --------------------------------------------------------------------------
# 2.7 Reusability
# --------------------------------------------------------------------------

def reusability(candidate: Candidate, ctx: ScoringContext) -> float:
    """How specific the term is, as a proxy for cross-document usefulness.

    With a corpus this would be IDF. With one document it is the spec's
    heuristic: rare words score high, common words score low. Frequency comes
    from the document itself rather than a bundled English list — a shipped word
    list would go stale and would not know the document's domain.

    Multi-word phrases get a small bonus: "graph neural network" is more
    specific than any of its words.
    """
    words = [w for w in candidate.key.split() if w]
    if not words:
        return 0.0

    max_freq = max(1, ctx.max_word_frequency)
    scores: list[float] = []
    for word in words:
        freq = ctx.word_frequencies.get(word, 0)
        # Log-scaled: raw frequency ratios are dominated by the top few words.
        scores.append(1.0 - math.log1p(freq) / math.log1p(max_freq))

    base = sum(scores) / len(scores)
    length_bonus = min(0.15, 0.05 * (len(words) - 1))
    return float(max(0.0, min(1.0, base + length_bonus)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

MetricFn = Callable[[Candidate, ScoringContext], float]

#: Metric name -> implementation. Replace an entry to replace a metric.
METRICS: dict[str, MetricFn] = {
    "structural": structural,
    "coverage": coverage,
    "purity": purity,
    "information_gain": information_gain,
    "embedding_variance": embedding_variance,
    "citation_importance": citation_importance,
    "reusability": reusability,
}

#: Default weights, from the spec. They sum to 1.0.
DEFAULT_WEIGHTS: dict[str, float] = {
    "structural": 0.25,
    "coverage": 0.20,
    "purity": 0.10,
    "information_gain": 0.15,
    "embedding_variance": 0.10,
    "citation_importance": 0.10,
    "reusability": 0.10,
}
