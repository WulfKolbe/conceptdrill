"""The `Embedder` protocol — the CES function f.

Every backend returns **L2-normalised** row vectors. That is load-bearing: it
makes cosine similarity a plain dot product, which is what lets `space.py`
collapse the whole projection into one matrix multiply `M @ l`.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """A text -> vector model."""

    name: str
    revision: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an `(len(texts), dim)` float32 array of unit-norm rows."""
        ...


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows are left at zero rather than
    producing NaN — an empty string must not poison a whole batch."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class BaseEmbedder:
    """Shared plumbing: batching and the empty-input edge case."""

    name: str = "base"
    revision: str = "0"
    dim: int = 0
    batch_size: int = 32

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            out.append(self._encode_batch(texts[i:i + self.batch_size]))
        return l2_normalise(np.vstack(out))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name} rev={self.revision} dim={self.dim}>"
