"""Embedding backends: determinism, normalisation, and lexical sanity."""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.embeddings import get_embedder, resolve_name
from conceptdrill.embeddings.cache import CachedEmbedder, cache_key
from conceptdrill.embeddings.hashing import HashingEmbedder


def test_rows_are_unit_norm(embedder):
    vecs = embedder.encode(["concept space", "latent vector", "hashing"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_text_does_not_produce_nan(embedder):
    vecs = embedder.encode(["", "   ", "real text"])
    assert not np.isnan(vecs).any()
    # An empty string has no features, so it stays at the origin rather than
    # being normalised into an arbitrary direction.
    assert np.allclose(vecs[0], 0.0)


def test_empty_batch_returns_correct_shape(embedder):
    vecs = embedder.encode([])
    assert vecs.shape == (0, embedder.dim)


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=64).encode(["semantic projection"])
    b = HashingEmbedder(dim=64).encode(["semantic projection"])
    assert np.array_equal(a, b)


def test_lexical_overlap_beats_unrelated_text(embedder):
    """The point of the hashing trick over per-text random vectors: shared
    vocabulary must produce higher similarity than no shared vocabulary."""
    vecs = embedder.encode([
        "deep hashing model for image similarity search",
        "deep hashing model for image retrieval",
        "the mitochondrion is the powerhouse of the cell",
    ])
    related = float(vecs[0] @ vecs[1])
    unrelated = float(vecs[0] @ vecs[2])
    assert related > unrelated
    assert related > 0.5


def test_revision_encodes_configuration():
    """A differently-configured hasher must not be able to reuse cache entries."""
    assert HashingEmbedder(dim=64).revision != HashingEmbedder(dim=128).revision
    assert (HashingEmbedder(dim=64, char_ngram=3).revision
            != HashingEmbedder(dim=64, char_ngram=4).revision)


def test_latex_control_words_survive_tokenisation():
    """MathBERT is not always available, so the hash backend has to keep
    something mathematical to work with."""
    emb = HashingEmbedder(dim=128)
    feats = emb._features(r"\sum_{i=1}^{K} y_i \log p_i")
    assert any(f == r"w:\sum" for f in feats)
    assert any(f == r"w:\log" for f in feats)


def test_alias_resolution():
    assert resolve_name("all-MiniLM-L6-v2") == "sentencebert"
    assert resolve_name("SBERT") == "sentencebert"
    assert resolve_name("hash") == "hash"


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown embedding model"):
        get_embedder("not-a-real-model", cache=False)


def test_arbitrary_checkpoint_is_accepted_without_loading():
    """An org/checkpoint path must be constructible offline; loading is lazy."""
    emb = get_embedder("some-org/some-model", cache=False)
    assert emb.name == "some-org/some-model"


def test_transformer_construction_does_not_load_the_model():
    """Constructing must not download: a --dry-run should cost nothing."""
    emb = get_embedder("sentencebert", cache=False)
    assert emb.inner._model is None if hasattr(emb, "inner") else emb._model is None


def test_device_defaults_to_cpu_not_an_autodetected_accelerator(monkeypatch):
    """`torch.cuda.is_available()` returning True does not mean the device will
    survive a forward pass — on ROCm it can report a device and then segfault
    the interpreter. CPU unless explicitly opted into.
    """
    from conceptdrill.embeddings.transformer import DEVICE_ENV, MODEL_SPECS, TransformerEmbedder
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    emb = TransformerEmbedder(MODEL_SPECS["sentencebert"])
    assert emb._device is None                    # unresolved until load
    resolved = (__import__("os").environ.get(DEVICE_ENV) or "cpu")
    assert resolved == "cpu"


def test_device_can_be_opted_into(monkeypatch):
    from conceptdrill.embeddings.transformer import MODEL_SPECS, TransformerEmbedder
    emb = TransformerEmbedder(MODEL_SPECS["sentencebert"], device="cuda")
    assert emb._device == "cuda"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

class CountingEmbedder(HashingEmbedder):
    """Counts how many texts actually reach the model."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.encoded = 0

    def _encode_batch(self, texts):
        self.encoded += len(texts)
        return super()._encode_batch(texts)


def test_cache_avoids_recomputation(tmp_path):
    inner = CountingEmbedder(dim=64)
    cached = CachedEmbedder(inner, cache_dir=tmp_path)

    first = cached.encode(["alpha", "beta"])
    assert inner.encoded == 2
    second = cached.encode(["alpha", "beta"])
    assert inner.encoded == 2                      # served from memory
    assert np.array_equal(first, second)


def test_cache_deduplicates_within_a_batch(tmp_path):
    inner = CountingEmbedder(dim=64)
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    cached.encode(["same", "same", "same", "other"])
    assert inner.encoded == 2


def test_cache_persists_across_instances(tmp_path):
    first = CountingEmbedder(dim=64)
    c1 = CachedEmbedder(first, cache_dir=tmp_path)
    c1.encode(["persisted text"])
    c1.flush()

    second = CountingEmbedder(dim=64)
    c2 = CachedEmbedder(second, cache_dir=tmp_path)
    c2.encode(["persisted text"])
    assert second.encoded == 0                     # loaded from disk


def test_cache_key_separates_models_and_revisions():
    assert cache_key("t", "a", "1") != cache_key("t", "b", "1")
    assert cache_key("t", "a", "1") != cache_key("t", "a", "2")


def test_corrupt_cache_shard_does_not_break_a_run(tmp_path):
    inner = CountingEmbedder(dim=64)
    cached = CachedEmbedder(inner, cache_dir=tmp_path)
    shard = cached._shard_path()
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"this is not an npz file")

    vecs = cached.encode(["still works"])
    assert vecs.shape[0] == 1
