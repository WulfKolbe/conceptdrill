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

#: Torch CPU thread count. See `_configure_threads`.
THREADS_ENV = "CONCEPTDRILL_TORCH_THREADS"


def _configure_threads() -> int:
    """Pin torch to one CPU thread unless told otherwise.

    Multi-threaded CPU inference is **not** bit-reproducible: the reduction
    order inside a matmul depends on thread scheduling, and repeated encodes of
    the same text drift by ~6e-4. That is far coarser than the 6-decimal
    precision stored in a projection, so with 16 threads two identical runs
    produce different `content_hash` values and the reproducibility guarantee is
    simply false.

    Single-threaded inference is bit-identical across repeats, fresh model
    instances, and processes. Embedding a document is a small-batch workload, so
    the throughput cost is modest and worth paying for a guarantee the tool
    actually advertises. Set `CONCEPTDRILL_TORCH_THREADS=0` to restore torch's
    default and trade reproducibility for speed.
    """
    import torch

    raw = (os.environ.get(THREADS_ENV) or "1").strip()
    try:
        requested = int(raw)
    except ValueError:
        requested = 1
    if requested > 0:
        torch.set_num_threads(requested)
    return torch.get_num_threads()


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
                 batch_size: int = 16, cache_dir: Optional[str] = None,
                 attn_implementation: str = "eager") -> None:
        self.spec = spec
        self.attn_implementation = attn_implementation
        self.name = spec.key
        self.revision = spec.revision or "unresolved"
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._device = device
        self._model = None
        self._tokenizer = None
        self.dim = 0
        self.torch_threads = 0

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

        self.torch_threads = _configure_threads()

        kwargs = {}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.checkpoint, **kwargs)

        # Force the eager attention path. Transformers defaults to SDPA, which
        # picks a kernel at runtime; the choice varies between processes and the
        # same text then embeds ~7e-4 apart, enough to reorder near-tied
        # concepts. Eager brings cross-process drift down to float32 epsilon
        # (~7e-7). Override with attn_implementation= if you need the speed.
        attn = self.attn_implementation
        try:
            model = AutoModel.from_pretrained(
                self.spec.checkpoint, attn_implementation=attn, **kwargs)
        except (TypeError, ValueError):
            # Older transformers, or an architecture with no eager path.
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

    def token_report(self, texts: Sequence[str]) -> dict:
        """How many tokens each text contributes, and how many are lost.

        Truncation is otherwise invisible. `_encode_batch` passes
        `return_tensors="pt"`, and transformers only populates
        `num_truncated_tokens` on the Python tokenizer path with tensors
        disabled — and for BERT in transformers 5.x there is no Python path
        left, `use_fast=False` is ignored and `is_fast` is True regardless.
        So the count is derived here instead: full length minus the length
        actually fed. Verified against the fast tokenizer's own overflow rows,
        which agreed on all 2358 texts checked.

        `over_window` is the number that fit but exceed the 50-70 token window
        CES targets. Those are not truncated; they are diluted, because mean
        pooling averages the concept over more tokens than intended.
        """
        self._ensure_loaded()
        cap = self.spec.max_length
        fed, lost, truncated, over_window = [], 0, 0, 0
        for text in texts:
            body = text if text and text.strip() else " "
            full = len(self._tokenizer(body, truncation=False)["input_ids"])
            kept = len(self._tokenizer(body, truncation=True,
                                       max_length=cap)["input_ids"])
            fed.append(kept)
            if full > kept:
                truncated += 1
                lost += full - kept
            if kept > 70:
                over_window += 1
        ordered = sorted(fed)
        pick = lambda p: (ordered[min(len(ordered) - 1, int(p * len(ordered)))]
                          if ordered else 0)
        return {
            "texts": len(fed), "max_length": cap,
            "tokens_p50": pick(0.5), "tokens_p90": pick(0.9),
            "tokens_max": ordered[-1] if ordered else 0,
            "truncated": truncated, "tokens_lost": lost,
            "over_70_token_window": over_window,
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # `dim` is unknown until the model loads, so the empty-input shortcut in
        # BaseEmbedder needs the load to have happened first.
        if not list(texts):
            self._ensure_loaded()
        return super().encode(texts)
