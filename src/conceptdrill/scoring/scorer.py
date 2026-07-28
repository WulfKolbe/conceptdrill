"""`QualityScorer` — builds the scoring context and applies the weighted sum.

    Q(c) = sum_m weight[m] * metric[m](c)

Weights are normalised so a partial or reweighted metric set still yields scores
in [0, 1] and stays comparable to a default run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from ..document import Document
from ..embeddings.base import Embedder
from ..nlp import word_frequencies
from ..types import Candidate
from .metrics import (DEFAULT_MAX_VARIANCE, DEFAULT_THETA, DEFAULT_WEIGHTS,
                      METRICS, MetricFn, ScoringContext)


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with its metric breakdown. The breakdown is kept, not just
    the total, so `explain` can say *why* a concept scored as it did."""
    candidate: Candidate
    score: float
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.candidate.name


class QualityScorer:
    """Scores candidates against a document."""

    def __init__(self, *, weights: Optional[Mapping[str, float]] = None,
                 metrics: Optional[Mapping[str, MetricFn]] = None,
                 theta: float = DEFAULT_THETA,
                 max_variance: float = DEFAULT_MAX_VARIANCE) -> None:
        self.metrics: dict[str, MetricFn] = dict(metrics or METRICS)
        raw = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.metrics)
            if unknown:
                raise ValueError(
                    f"weights given for unknown metrics: {', '.join(sorted(unknown))}"
                )
            raw.update(weights)
        # Keep only weights that correspond to an active metric, then normalise.
        raw = {k: float(v) for k, v in raw.items() if k in self.metrics}
        total = sum(abs(v) for v in raw.values())
        self.weights = ({k: v / total for k, v in raw.items()} if total > 0
                        else {k: 0.0 for k in raw})
        self.theta = theta
        self.max_variance = max_variance

    # ---- context --------------------------------------------------------

    def build_context(self, doc: Document, candidates: Sequence[Candidate],
                      embedder: Embedder) -> ScoringContext:
        """Embed blocks and candidates once; derive everything else from that."""
        blocks = doc.prose_blocks
        block_texts = [b.text for b in blocks]
        block_sections = [doc.top_level_section(b.section_id) for b in blocks]

        block_embeddings = (embedder.encode(block_texts) if block_texts
                            else np.zeros((0, max(1, embedder.dim)), dtype=np.float32))

        cand_texts = [c.tau for c in candidates]
        cand_embeddings = (embedder.encode(cand_texts) if cand_texts
                           else np.zeros((0, block_embeddings.shape[1]),
                                         dtype=np.float32))
        cand_index = {c.key: i for i, c in enumerate(candidates)}

        freqs = word_frequencies(block_texts)
        return ScoringContext(
            block_embeddings=block_embeddings,
            block_sections=block_sections,
            candidate_embeddings=cand_embeddings,
            candidate_index=cand_index,
            theta=self.theta,
            max_variance=self.max_variance,
            word_frequencies=freqs,
            max_word_frequency=max(freqs.values()) if freqs else 1,
        )

    # ---- scoring --------------------------------------------------------

    def score_one(self, candidate: Candidate,
                  ctx: ScoringContext) -> ScoredCandidate:
        values: dict[str, float] = {}
        for name in sorted(self.metrics):
            try:
                value = float(self.metrics[name](candidate, ctx))
            except Exception:
                # A failing metric must not sink the run; neutral is the safe
                # value and the failure shows up as an exactly-0.5 entry.
                value = 0.5
            values[name] = round(max(0.0, min(1.0, value)), 6)

        total = sum(self.weights.get(name, 0.0) * value
                    for name, value in values.items())
        return ScoredCandidate(candidate, round(float(total), 6), values)

    def score(self, doc: Document, candidates: Sequence[Candidate],
              embedder: Embedder,
              ctx: Optional[ScoringContext] = None,
              ) -> tuple[list[ScoredCandidate], ScoringContext]:
        """Score every candidate. Returns the scored list and the context, which
        `space.py` reuses so block embeddings are computed exactly once."""
        candidates = list(candidates)
        if ctx is None:
            ctx = self.build_context(doc, candidates, embedder)
        scored = [self.score_one(c, ctx) for c in candidates]
        # Deterministic: highest score first, ties broken on name.
        scored.sort(key=lambda s: (-s.score, s.candidate.name))
        return scored, ctx

    def describe(self) -> dict[str, object]:
        return {
            "weights": dict(sorted(self.weights.items())),
            "metrics": sorted(self.metrics),
            "theta": self.theta,
            "max_variance": self.max_variance,
        }
