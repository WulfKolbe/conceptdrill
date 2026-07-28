"""HuggingFace transformer embedder with attention-masked mean pooling.

One code path serves all three named models. Mean pooling plus L2 normalisation
is exactly what `sentence-transformers` does for `all-MiniLM-L6-v2`, so nothing
is lost by not depending on that package, and MathBERT / CodeBERT — which have
no sentence-transformers head at all — work through the same class.

Imports of `torch` and `transformers` are deferred to construction time so the
rest of ConceptDrill stays importable without them.

**The device defaults to CPU, deliberately.** `torch.cuda.is_available()`
returning True does not mean the device will survive a forward pass — on a ROCm
box it can report one device and then segfault the interpreter. Embedding a
document is a small-batch workload where CPU is entirely adequate, so silently
opting into an accelerator that might take the process down is a bad trade. Set
`CONCEPTDRILL_DEVICE=cuda` or pass `device=` to opt in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .base import BaseEmbedder

#: Opt into an accelerator explicitly. Values are passed straight to torch.
DEVICE_ENV = "CONCEPTDRILL_DEVICE"


@dataclass(frozen=True)
class ModelSpec:
    """A named model. `revision` pins the checkpoint for reproducibility; None
    means "resolve and record whatever was actually loaded"."""
    key: str
    checkpoint: str
    revision: Optional[str] = None
    max_length: int = 512
    note: str = ""


# The three models named in the spec, plus the type-routing aliases.
MODEL_SPECS: dict[str, ModelSpec] = {
    "sentencebert": ModelSpec(
        "sentencebert", "sentence-transformers/all-MiniLM-L6-v2",
        max_length=256, note="prose, headings, bibliography titles"),
    "mathbert": ModelSpec(
        "mathbert", "tbs17/MathBERT",
        max_length=512, note="LaTeX formulae and display equations"),
    "codebert": ModelSpec(
        "codebert", "microsoft/codebert-base",
        max_length=512, note="source listings and algorithms"),
}


class TransformerEmbedder(BaseEmbedder):
    """Mean-pooled `AutoModel` embeddings.

    Loads lazily on first `encode` so constructing one is cheap and a run that
    never reaches the model (for example `--dry-run`) never downloads it.
    """

    def __init__(self, spec: ModelSpec, *, device: Optional[str] = None,
                 batch_size: int = 16, cache_dir: Optional[str] = None) -> None:
        self.spec = spec
        self.name = spec.key
        self.revision = spec.revision or "unresolved"
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._device = device
        self._model = None
        self._tokenizer = None
        self.dim = 0

    # ---- lazy load ------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"the '{self.name}' model needs torch + transformers: "
                f"pip install torch transformers  (or use --model hash for an "
                f"offline deterministic backend)"
            ) from exc

        kwargs = {}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.checkpoint, **kwargs)
        model = AutoModel.from_pretrained(self.spec.checkpoint, **kwargs)
        model.eval()

        if self._device is None:
            self._device = (os.environ.get(DEVICE_ENV) or "cpu").strip() or "cpu"
        try:
            model.to(self._device)
        except Exception:
            # An unusable accelerator must cost speed, not the run.
            self._device = "cpu"
            model.to("cpu")
        self._model = model
        self._torch = torch

        self.dim = int(getattr(model.config, "hidden_size", 0))
        self.revision = self._resolve_revision()

    def _resolve_revision(self) -> str:
        """Record the commit actually loaded, so a projection made against
        `main` still says which `main`."""
        if self.spec.revision:
            return self.spec.revision
        for obj in (self._model, self._tokenizer):
            cfg = getattr(obj, "config", None)
            for holder in (cfg, obj):
                sha = getattr(holder, "_commit_hash", None)
                if isinstance(sha, str) and sha:
                    return sha
        return f"{self.spec.checkpoint}@unpinned"

    # ---- encoding -------------------------------------------------------

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        # Empty strings tokenize to nothing and would divide by a zero mask.
        safe = [t if t and t.strip() else " " for t in texts]
        enc = self._tokenizer(
            list(safe), padding=True, truncation=True,
            max_length=self.spec.max_length, return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            out = self._model(**enc)

        hidden = out.last_hidden_state                      # (b, t, h)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return pooled.float().cpu().numpy()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # `dim` is unknown until the model loads, so the empty-input shortcut in
        # BaseEmbedder needs the load to have happened first.
        if not list(texts):
            self._ensure_loaded()
        return super().encode(texts)
