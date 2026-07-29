"""Turning a section into the three basis texts.

Each section yields three progressively abstract descriptions, per
`prompts/section-concept.md`:

    summary      80-150 words   faithful to this document's terminology
    abstraction  ~70 words      the idea, independent of this document
    label        30-42 words    canonical, reusable across documents

Only `label` fits the 50-70 BERT token window the basis needs. Measured on the
reference paper at 1.604 tokens/word, the tiers land at 128-241, 96-128 and
**48-67** tokens respectively. The other two are kept for per-document views and
for comparing which tier projects better.

`Summarizer` is a protocol with two implementations:

  * `ExtractiveSummarizer` — deterministic, offline, no model. Selects text
    rather than generating it, so it is honest about being a floor rather than
    a substitute. This is what the test suite runs against.
  * `NovitaSummarizer` (in `novita.py`) — the real thing.

Realised token lengths are **measured, never assumed**: a target of 30-42 words
is an instruction to a model, not a guarantee from one.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .sanitize import sanitize_text

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "section-concept.md"

#: Word budgets per tier, from the prompt.
TIER_WORDS = {"summary": (80, 150), "abstraction": (55, 85), "label": (30, 42)}

#: The tier used to build the cross-document basis.
BASIS_TIER = "label"

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class SectionSummary:
    """Three basis texts for one section, plus how they were produced."""
    section_id: str
    title: str
    summary: str = ""
    abstraction: str = ""
    label: str = ""
    model: str = ""
    #: False when a language model was involved, so a run can say whether its
    #: vocabulary is reproducible.
    deterministic: bool = True
    #: Populated when the reply looked damaged; see `replyparse.control_corruption`.
    warnings: tuple[str, ...] = ()
    error: str = ""

    @property
    def basis_text(self) -> str:
        """The text that becomes a basis vector."""
        return getattr(self, BASIS_TIER, "") or self.abstraction or self.summary

    @property
    def is_usable(self) -> bool:
        return bool(self.basis_text.strip()) and not self.error

    def word_counts(self) -> dict[str, int]:
        return {tier: len(getattr(self, tier, "").split()) for tier in TIER_WORDS}

    def tier_fit(self) -> dict[str, str]:
        """Whether each tier hit its word budget: `under` | `ok` | `over`."""
        out = {}
        for tier, (lo, hi) in TIER_WORDS.items():
            n = len(getattr(self, tier, "").split())
            out[tier] = "under" if n < lo else "over" if n > hi else "ok"
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Summarizer(Protocol):
    """Produces the three basis texts for one section."""

    name: str
    deterministic: bool

    def summarize(self, section_id: str, title: str, body: str) -> SectionSummary:
        ...


def load_prompt() -> str:
    """The section-concept prompt shipped with the package."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _words(text: str, limit: int) -> str:
    """First `limit` words, cut at a word boundary."""
    parts = text.split()
    return " ".join(parts[:limit])


def _sentences_upto(text: str, max_words: int) -> str:
    """Whole sentences until the word budget would be exceeded.

    Preferring sentence boundaries matters: a basis text cut mid-clause embeds
    as a fragment, and the fragment is what the whole corpus then compares
    against.
    """
    out: list[str] = []
    used = 0
    for sentence in _SENTENCE.split(text.strip()):
        n = len(sentence.split())
        if not n:
            continue
        if used + n > max_words and out:
            break
        out.append(sentence.strip())
        used += n
    return " ".join(out) if out else _words(text, max_words)


class ExtractiveSummarizer:
    """Deterministic, offline, no model. Selects text; does not generate it.

    A genuine floor, not a stand-in for the real summariser: it cannot abstract
    away from this document's wording, which is exactly what the `abstraction`
    and `label` tiers are supposed to do. It exists so the pipeline runs, and
    so the test suite is offline and reproducible.
    """

    name = "extractive"
    deterministic = True

    def summarize(self, section_id: str, title: str, body: str) -> SectionSummary:
        # Sanitised even though the source is the document rather than a
        # model: drilled text carries OCR and LLM-authored characters too.
        text = re.sub(r"\s+", " ", sanitize_text(body or ""))
        clean_title = re.sub(r"\s+", " ", sanitize_text(title or ""))

        if not text:
            # A section with no body still has a title, which is a weak but
            # real concept signal. Better than an empty basis vector.
            return SectionSummary(
                section_id=section_id, title=clean_title,
                summary=clean_title, abstraction=clean_title, label=clean_title,
                model=self.name, deterministic=True,
                warnings=("empty body: title used as the basis text",))

        summary = _sentences_upto(text, TIER_WORDS["summary"][1])
        abstraction = _sentences_upto(text, TIER_WORDS["abstraction"][1])
        # The label leads with the title so the concept is named, then borrows
        # the opening clause for context.
        lead = _words(text, max(0, TIER_WORDS["label"][1] - len(clean_title.split())))
        label = f"{clean_title}. {lead}".strip(". ").strip()

        return SectionSummary(
            section_id=section_id, title=clean_title,
            summary=summary, abstraction=abstraction, label=label,
            model=self.name, deterministic=True)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def summary_key(title: str, body: str, model: str, prompt: str) -> str:
    """Content address for one summarisation request.

    Includes the prompt: changing the instructions changes the output, so a
    cached summary from an older prompt must not be served.
    """
    payload = "\x1f".join([model, prompt, title or "", body or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SummaryCache:
    """On-disk cache of summaries, keyed by content.

    The language model is the slowest and least reproducible stage. Caching
    makes re-runs cheap and, more importantly, makes a corpus build comparable
    across runs even though the model itself is not deterministic.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._dirty = False

    def load(self) -> "SummaryCache":
        if self._loaded:
            return self
        self._loaded = True
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = {k: v for k, v in data.items()
                                     if isinstance(v, dict)}
            except Exception:
                # A corrupt cache must cost time, not correctness.
                self._entries = {}
        return self

    def get(self, key: str) -> Optional[SectionSummary]:
        self.load()
        raw = self._entries.get(key)
        if not raw:
            return None
        try:
            return SectionSummary(**{**raw, "warnings": tuple(raw.get("warnings", ()))})
        except TypeError:
            return None

    def put(self, key: str, summary: SectionSummary) -> None:
        self.load()
        self._entries[key] = summary.to_dict()
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._entries, indent=2, sort_keys=True,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        self.load()
        return len(self._entries)


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

@dataclass
class SummaryRun:
    """Every section's basis texts, plus how the run went."""
    summaries: dict[str, SectionSummary] = field(default_factory=dict)
    model: str = ""
    deterministic: bool = True
    cached: int = 0
    generated: int = 0
    failed: tuple[str, ...] = ()

    def usable(self) -> dict[str, SectionSummary]:
        return {sid: s for sid, s in self.summaries.items() if s.is_usable}

    def stats(self) -> dict[str, Any]:
        fits: dict[str, int] = {}
        for summary in self.summaries.values():
            fits[summary.tier_fit()[BASIS_TIER]] = \
                fits.get(summary.tier_fit()[BASIS_TIER], 0) + 1
        warned = [s.section_id for s in self.summaries.values() if s.warnings]
        return {
            "sections": len(self.summaries),
            "usable": len(self.usable()),
            "cached": self.cached,
            "generated": self.generated,
            "failed": len(self.failed),
            "with_warnings": len(warned),
            "model": self.model,
            "deterministic": self.deterministic,
            f"{BASIS_TIER}_word_budget": dict(sorted(fits.items())),
        }


def summarize_tree(tree, summarizer: Summarizer, *,
                   cache: Optional[SummaryCache] = None,
                   levels: Optional[set[int]] = None,
                   prompt: Optional[str] = None,
                   progress=None) -> SummaryRun:
    """Summarise every section of a `SectionTree`.

    Two choices here are load-bearing:

    * **`subtree_text`, not `body_text`.** Summarising "Empirical Evaluation"
      from its own 1350 characters while ignoring the 9791 in its subsections
      would describe almost nothing.
    * **`summarizer_title`, not `title`.** Where caption cleaning dropped a
      macro the cleaned title can be meaningless — `\\ALG\\ Application` becomes
      `Application` — so the raw form is restored for the model.

    A section that fails is recorded and the run continues; one unreachable
    section must not cost a whole corpus.
    """
    prompt_text = prompt if prompt is not None else load_prompt()
    run = SummaryRun(model=getattr(summarizer, "name", "?"),
                     deterministic=bool(getattr(summarizer, "deterministic", False)))
    failed: list[str] = []

    for node in tree.iter_document_order():
        if levels is not None and node.level not in levels:
            continue

        title = node.summarizer_title
        body = tree.subtree_text(node.id)
        key = summary_key(title, body, run.model, prompt_text)

        summary = cache.get(key) if cache else None
        if summary is not None:
            run.cached += 1
        else:
            summary = summarizer.summarize(node.id, title, body)
            run.generated += 1
            # Only cache what is worth serving again.
            if cache is not None and summary.is_usable:
                cache.put(key, summary)

        run.summaries[node.id] = summary
        if not summary.is_usable:
            failed.append(node.id)
        if progress:
            progress(node, summary)

    if cache is not None:
        cache.flush()
    run.failed = tuple(failed)
    return run
