"""Deterministic offline embedder — the signed hashing trick.

Why not random-vector-per-text: a hash of the whole string gives every text an
independent random direction, so all similarities collapse toward zero and the
coverage / purity / variance metrics degenerate into noise. Hashing *tokens*
into a shared space instead means texts that share vocabulary share directions,
so the metrics produce meaningful numbers with no model download.

This backend is what the test suite runs against. It is a lexical model, not a
semantic one — it will not see that "car" and "automobile" are related. It is
for reproducibility and offline operation, not for quality.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from .base import BaseEmbedder

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d+|\\[A-Za-z]+")

# LaTeX control words carry the mathematical signal, so `\sum` and `\int` are
# kept as tokens (the third alternative above) rather than split apart.


def _hash(token: str, salt: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{salt}\x1f{token}".encode("utf-8"), digest_size=8).digest(),
        "big",
    )


class HashingEmbedder(BaseEmbedder):
    """Signed hashing of word unigrams, bigrams and character n-grams.

    Deterministic across processes and platforms: blake2b, no PRNG state, no
    dict iteration order dependence.
    """

    def __init__(self, dim: int = 256, *, use_bigrams: bool = True,
                 char_ngram: int = 4, salt: str = "conceptdrill/v1") -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.name = "hash"
        self.dim = int(dim)
        self.salt = salt
        self.use_bigrams = use_bigrams
        self.char_ngram = int(char_ngram)
        # The revision pins every knob that changes the output, so a cache entry
        # from a differently-configured embedder can never be reused.
        self.revision = (f"hash-d{self.dim}-b{int(use_bigrams)}"
                         f"-c{self.char_ngram}-{salt}")
        self.batch_size = 512

    def _features(self, text: str) -> Counter[str]:
        """Sublinear term weighting over words, word bigrams, and char n-grams."""
        lowered = text.lower()
        words = _TOKEN.findall(lowered)
        feats: Counter[str] = Counter()
        for w in words:
            feats[f"w:{w}"] += 1
        if self.use_bigrams:
            for a, b in zip(words, words[1:]):
                feats[f"b:{a}_{b}"] += 1
        if self.char_ngram > 0:
            # Char n-grams give partial credit for morphology ("hashing" vs
            # "hash"), which matters for short headings and single-word concepts.
            squeezed = re.sub(r"\s+", " ", lowered)
            n = self.char_ngram
            for i in range(max(0, len(squeezed) - n + 1)):
                feats[f"c:{squeezed[i:i + n]}"] += 1
        return feats

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            if not text or not text.strip():
                continue
            for feat, count in self._features(text).items():
                h = _hash(feat, self.salt)
                col = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                # 1 + log(tf): damps the char-n-gram flood so a long block does
                # not simply outweigh a short one on shared substrings.
                out[row, col] += sign * (1.0 + math.log(count))
        return out
