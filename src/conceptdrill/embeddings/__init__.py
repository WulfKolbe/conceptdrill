"""Embedding backend registry.

`get_embedder(name)` is the only entry point callers should use. Names:

  * `hash`                          — offline deterministic (default in tests)
  * `sentencebert` / `mathbert` / `codebert`  — transformer checkpoints
  * any `org/model` string          — an arbitrary HuggingFace checkpoint

`all-MiniLM-L6-v2` is accepted as an alias for `sentencebert` because the spec's
`--embedding-model` example uses it.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseEmbedder, Embedder, l2_normalise
from .cache import DEFAULT_CACHE_DIR, CachedEmbedder, cache_key
from .hashing import HashingEmbedder
from .transformer import MODEL_SPECS, ModelSpec, TransformerEmbedder

ALIASES: dict[str, str] = {
    "all-minilm-l6-v2": "sentencebert",
    "sentence-transformers/all-minilm-l6-v2": "sentencebert",
    "sbert": "sentencebert",
    "minilm": "sentencebert",
    "math": "mathbert",
    "code": "codebert",
    "hashing": "hash",
    "offline": "hash",
    "deterministic": "hash",
}

#: Names accepted by `--model`, for help text and validation.
KNOWN_MODELS = ("hash", *sorted(MODEL_SPECS))


def resolve_name(name: str) -> str:
    key = (name or "").strip()
    return ALIASES.get(key.lower(), key)


def get_embedder(name: str = "sentencebert", *, cache: bool = True,
                 cache_dir: Optional[str] = None, dim: int = 256,
                 device: Optional[str] = None,
                 batch_size: Optional[int] = None) -> Embedder:
    """Build an embedder by name, wrapped in the cache unless disabled."""
    resolved = resolve_name(name)

    if resolved == "hash":
        inner: Embedder = HashingEmbedder(dim=dim)
    elif resolved in MODEL_SPECS:
        spec = MODEL_SPECS[resolved]
        inner = TransformerEmbedder(
            spec, device=device,
            batch_size=batch_size or 16,
            cache_dir=cache_dir,
        )
    elif "/" in resolved:
        # An arbitrary HuggingFace checkpoint, so a new model needs no code change.
        inner = TransformerEmbedder(
            ModelSpec(resolved, resolved), device=device,
            batch_size=batch_size or 16, cache_dir=cache_dir,
        )
    else:
        raise ValueError(
            f"unknown embedding model {name!r}; expected one of "
            f"{', '.join(KNOWN_MODELS)} or an 'org/checkpoint' path"
        )

    if not cache:
        return inner
    return CachedEmbedder(inner, cache_dir=cache_dir or DEFAULT_CACHE_DIR)


__all__ = [
    "BaseEmbedder", "CachedEmbedder", "DEFAULT_CACHE_DIR", "Embedder",
    "HashingEmbedder", "KNOWN_MODELS", "MODEL_SPECS", "ModelSpec",
    "TransformerEmbedder", "cache_key", "get_embedder", "l2_normalise",
    "resolve_name",
]
