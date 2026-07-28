"""Content-addressed embedding cache.

Keyed by `sha256(text | model | revision)`, so a cache entry can never be served
to a different model or a different checkpoint of the same model. That is what
makes the cache safe to keep across runs while preserving reproducibility.

Two layers: an in-memory dict for the current process, and an optional on-disk
`.npz` shard per (model, revision). Concept-vocabulary embeddings are the point
— they are recomputed on every projection run otherwise.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import Embedder, l2_normalise

DEFAULT_CACHE_DIR = ".conceptdrill_cache"


def cache_key(text: str, model: str, revision: str) -> str:
    return hashlib.sha256(
        f"{model}\x1f{revision}\x1f{text}".encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Wraps an `Embedder`, memoising per text.

    Presents the same protocol as the wrapped embedder, so callers cannot tell
    the difference and nothing downstream needs to know about caching.
    """

    def __init__(self, inner: Embedder, *, cache_dir: Optional[str | Path] = None,
                 persist: bool = True) -> None:
        self.inner = inner
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self.cache_dir = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
        self.persist = persist
        self._loaded = False

    # ---- protocol passthrough ------------------------------------------

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def revision(self) -> str:
        return self.inner.revision

    @property
    def dim(self) -> int:
        return self.inner.dim

    # ---- disk layer -----------------------------------------------------

    def _shard_path(self) -> Path:
        # The revision goes in the filename via a digest: revisions can be long
        # or contain characters that are awkward in a path.
        tag = hashlib.sha256(
            f"{self.inner.name}\x1f{self.inner.revision}".encode()).hexdigest()[:16]
        return self.cache_dir / f"emb-{self.inner.name}-{tag}.npz"

    def _load_shard(self) -> None:
        if self._loaded or not self.persist:
            self._loaded = True
            return
        self._loaded = True
        path = self._shard_path()
        if not path.exists():
            return
        try:
            with np.load(path) as z:
                for key in z.files:
                    self._mem.setdefault(key, z[key])
        except Exception:
            # A corrupt or truncated shard must never break a run; recompute.
            pass

    def flush(self) -> None:
        """Write new entries to disk. Safe to call repeatedly."""
        if not self.persist or not self._dirty or not self._mem:
            return
        path = self._shard_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # Write through a file object: given a *path*, np.savez appends ".npz"
        # when the name does not already end in it, and the rename target would
        # never exist.
        with tmp.open("wb") as fh:
            np.savez(fh, **self._mem)
        tmp.replace(path)
        self._dirty = False

    # ---- encoding -------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return self.inner.encode(texts)

        self._load_shard()
        keys = [cache_key(t, self.inner.name, self.inner.revision) for t in texts]

        with self._lock:
            missing_idx = [i for i, k in enumerate(keys) if k not in self._mem]

        if missing_idx:
            # Deduplicate within the batch: a document repeats short strings.
            unique: dict[str, int] = {}
            order: list[int] = []
            for i in missing_idx:
                if keys[i] not in unique:
                    unique[keys[i]] = i
                    order.append(i)
            fresh = self.inner.encode([texts[i] for i in order])
            with self._lock:
                for pos, i in enumerate(order):
                    self._mem[keys[i]] = fresh[pos]
                self._dirty = True

        with self._lock:
            rows = [self._mem[k] for k in keys]
        return l2_normalise(np.vstack(rows))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CachedEmbedder inner={self.inner!r} entries={len(self._mem)}>"
