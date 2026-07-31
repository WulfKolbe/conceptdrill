"""Turning a span into the three basis texts.

Each span yields three progressively abstract descriptions, per
`prompts/section-concept.md`:

    summary      80-150 words   faithful to this document's terminology
    abstraction  ~70 words      the idea, independent of this document
    label        30-42 words    canonical, reusable across documents

Only `label` fits the 50-70 BERT token window the basis needs. Measured on the
reference paper at 1.604 tokens/word, the tiers land at 128-241, 96-128 and
**48-67** tokens respectively. The other two are kept for per-document views and
for comparing which tier projects better.

The three tiers must be **independent derivations**, not cut points on one
string. `check_tier_independence` enforces that: no tier may be a prefix of
another, and no two may share more than `MAX_TIER_JACCARD` of their token
vocabulary. Budgets alone were never a sufficient contract — three truncations
of the same paragraph satisfy every word count and answer none of the roles in
`TIER_ROLES`.

`Summarizer` is a protocol with three implementations, and only one of them
belongs in a measurement:

  * `NovitaSummarizer` (in `novita.py`) — the real thing. `measurement_safe`.
  * `TitleOnlySummarizer` — the ablation baseline. `measurement_safe`, and
    flagged `is_ablation` so it is never mistaken for a summariser.
  * `ExtractiveSummarizer` — **a test fixture**, `measurement_safe = False`.
    It fails `check_tier_independence` by construction; using it as a stand-in
    voided a whole corpus build.

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
#:
#: MEASURED against the embedder's own tokenizer, tokenising exactly as
#: `TransformerEmbedder._encode_batch` does (all-MiniLM-L6-v2, truncation=True,
#: max_length=256), over 786 cached summaries:
#:
#:     tier          tokens p50   p90   truncated
#:     summary          116       189   3.1%
#:     abstraction       81       114   0.8%
#:     label             39        66   0.1%
#:
#: Nothing was being silently truncated -- the cap is 256, not 70 -- but only
#: `label` sat inside the 50-70 token window CES targets. `summary` at 116
#: tokens is roughly double it, and mean pooling over twice the tokens dilutes
#: the concept signal.
#:
#: All three budgets now fit 70 tokens at the measured **1.441** tokens per
#: word (median, n=786). The long-standing 1.604 figure was an estimate and is
#: 11% high; 70 tokens is 49 words, not 44.
#:
#: The tiers no longer differ much in length, so their independence rests
#: entirely on role and form -- see `TIER_ROLES` and `check_tier_independence`.
TIER_WORDS = {"summary": (28, 34), "abstraction": (24, 30), "label": (22, 28)}

#: The embedder's window, in tokens. Every tier is sized to fit it.
#:
#: The CES design targets 50-70. The ceiling now sits at the BOTTOM of that
#: range rather than the top: at 70 tokens the measured output was p50 41,
#: p90 50, max 61, so the band was never binding and the vectors were longer
#: than they needed to be. Mean pooling averages the concept over every token
#: it is given, so a shorter text is a sharper vector.
#:
#: This is expected to press on GATE 3: three tiers describing one concept
#: converge as they shorten, and tier independence is already the failing
#: gate. That cost is measured, not assumed -- see the run that follows.
EMBEDDING_TOKEN_WINDOW = (35, 50)

#: Measured tokens per word on this corpus, all-MiniLM-L6-v2, n=786.
TOKENS_PER_WORD = 1.441

#: The tier used to build the cross-document basis.
BASIS_TIER = "label"

#: What each tier is *for*. Three cut points on one string satisfy the word
#: budgets and answer none of these questions, which is why the budgets alone
#: were never a sufficient contract.
TIER_ROLES = {
    "summary": "document-faithful prose",
    "abstraction": "document-independent prose, no document-local references",
    "label": "a noun phrase, canonical across documents: the basis tier",
}

#: Two tiers sharing more than this fraction of their token vocabulary are not
#: independent derivations. Jaccard rather than containment: a label that is a
#: scattered subset of its abstraction is as degenerate as a prefix of it.
MAX_TIER_JACCARD = 0.6

_TOKEN = re.compile(r"[a-z0-9]+")

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class SpanSummary:
    """Three basis texts for one span, plus how they were produced."""
    span_id: str
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
    #: The span's OTHER concepts, if it defines more than one.
    #:
    #: A span covering three ideas used to yield one label -- a compromise
    #: matching none of them, and one basis row where CES wants three. The
    #: prompt now emits an entry per concept; this record is the dominant one
    #: and the rest live here. Siblings never nest: a sibling's own `siblings`
    #: is always empty.
    siblings: tuple["SpanSummary", ...] = ()

    @property
    def concepts(self) -> tuple["SpanSummary", ...]:
        """Every concept this span defines, dominant first."""
        return (self,) + self.siblings

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


def token_set(text: str) -> frozenset[str]:
    """Lowercased alphanumeric tokens. The unit the Jaccard test counts."""
    return frozenset(_TOKEN.findall((text or "").lower()))


def jaccard(left: str, right: str) -> float:
    """Token-set overlap. 0.0 when either side is empty."""
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_tier_independence(summary: "SpanSummary", *,
                            max_jaccard: float = MAX_TIER_JACCARD) -> list[str]:
    """Every way this summary's tiers fail to be independent derivations.

    Two relations, because they catch different degeneracies. A **prefix**
    relation means one tier was produced by truncating another — the
    `ExtractiveSummarizer` failure. A high **Jaccard** means the tiers were
    written separately but say the same thing in the same words, which the
    prefix test alone would pass.

    Tiers that are absent are not compared: `null` is a declared absence, and
    an ablation arm legitimately has only one tier.
    """
    problems: list[str] = []
    present = {tier: getattr(summary, tier, "") or "" for tier in TIER_WORDS}
    present = {tier: text.strip() for tier, text in present.items() if text.strip()}

    names = sorted(present)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = present[left], present[right]
            if a.startswith(b) or b.startswith(a):
                shorter, longer = (right, left) if len(a) >= len(b) else (left, right)
                problems.append(f"{shorter} is a prefix of {longer}")
                continue
            overlap = jaccard(a, b)
            if overlap > max_jaccard:
                problems.append(
                    f"{left}/{right} token Jaccard {overlap:.3f} > {max_jaccard}")
    return problems


def assert_tier_independence(summary: "SpanSummary", *,
                             max_jaccard: float = MAX_TIER_JACCARD) -> None:
    """Raise when the tiers are not independent derivations."""
    problems = check_tier_independence(summary, max_jaccard=max_jaccard)
    if problems:
        raise TierDegeneracy(
            f"{summary.span_id}: {problems}")


class TierDegeneracy(ValueError):
    """Tiers that are truncations of one another, not separate derivations."""


@runtime_checkable
class Summarizer(Protocol):
    """Produces the three basis texts for one span."""

    name: str
    deterministic: bool

    def summarize(self, span_id: str, title: str, body: str) -> SpanSummary:
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
    """TEST FIXTURE ONLY. Not a summariser and not a measurement floor.

    It does not summarise. `summary` and `abstraction` are sentence-boundary
    prefixes of one raw string and `label` is the title plus a word-prefix of
    that same string, so all three fail `check_tier_independence` by
    construction. Used as a stand-in for a real summariser it produced a corpus
    in which `\\span*{` appeared in every label, and every number derived
    from that corpus was void.

    `measurement_safe = False` keeps it out of measurement runs mechanically
    rather than by convention. The honest floor is the title-only ablation
    (`TitleOnlySummarizer`), which does not claim to be a summary.

    It stays because the test suite must run offline and deterministically.
    """

    name = "extractive"
    deterministic = True
    #: Refused by measurement runners. See the class docstring.
    measurement_safe = False

    def summarize(self, span_id: str, title: str, body: str) -> SpanSummary:
        # Sanitised even though the source is the document rather than a
        # model: drilled text carries OCR and LLM-authored characters too.
        text = re.sub(r"\s+", " ", sanitize_text(body or ""))
        clean_title = re.sub(r"\s+", " ", sanitize_text(title or ""))

        if not text:
            # A span with no body still has a title, which is a weak but
            # real concept signal. Better than an empty basis vector.
            return SpanSummary(
                span_id=span_id, title=clean_title,
                summary=clean_title, abstraction=clean_title, label=clean_title,
                model=self.name, deterministic=True,
                warnings=("empty body: title used as the basis text",))

        summary = _sentences_upto(text, TIER_WORDS["summary"][1])
        abstraction = _sentences_upto(text, TIER_WORDS["abstraction"][1])
        # The label leads with the title so the concept is named, then borrows
        # the opening clause for context.
        lead = _words(text, max(0, TIER_WORDS["label"][1] - len(clean_title.split())))
        label = f"{clean_title}. {lead}".strip(". ").strip()

        return SpanSummary(
            span_id=span_id, title=clean_title,
            summary=summary, abstraction=abstraction, label=label,
            model=self.name, deterministic=True)


class TitleOnlySummarizer:
    """The ablation arm: the cleaned marker title, and nothing else.

    Not a degraded summariser — a deliberate baseline. If a run using real
    summaries is not clearly distinguishable from this one, the summarisation
    tier contributed nothing and the run is void. That comparison is the
    cheapest sanity check available and it is why this class exists.

    Only `label` is populated. `abstraction` and `summary` are empty because a
    title is neither, and claiming otherwise is the failure mode that made
    `ExtractiveSummarizer` unusable.
    """

    name = "title-only"
    deterministic = True
    measurement_safe = True
    #: A baseline to compare against, never a stand-in for a summariser.
    is_ablation = True

    def summarize(self, span_id: str, title: str, body: str) -> SpanSummary:
        clean = re.sub(r"\s+", " ", sanitize_text(title or "")).strip()
        return SpanSummary(
            span_id=span_id, title=clean, label=clean,
            model=self.name, deterministic=True,
            warnings=() if clean else ("span has no title",))


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def summary_key(title: str, body: str, model: str, prompt: str) -> str:
    """Content address for one summarisation request.

    `model` is the generator's full identity, not just its name — pass
    `summarizer.cache_signature` where one exists, so decoding parameters are
    part of the address.

    Includes the prompt: changing the instructions changes the output, so a
    cached summary from an older prompt must not be served. The same argument
    applies to `max_tokens`, which is why the signature exists.
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

    @staticmethod
    def _revive(raw: dict[str, Any]) -> "SpanSummary":
        """Rebuild a summary from JSON.

        `asdict` flattens nested summaries to dicts and tuples to lists, so a
        naive `SpanSummary(**raw)` returns siblings as a list of dicts --
        objects with no `.label`, which every consumer would then fail on.
        """
        siblings = tuple(
            SummaryCache._revive(s) if isinstance(s, dict) else s
            for s in raw.get("siblings", ()) or ())
        return SpanSummary(**{**raw,
                                 "warnings": tuple(raw.get("warnings", ()) or ()),
                                 "siblings": siblings})

    def get(self, key: str) -> Optional[SpanSummary]:
        self.load()
        raw = self._entries.get(key)
        if not raw:
            return None
        try:
            return self._revive(raw)
        except TypeError:
            return None

    def put(self, key: str, summary: SpanSummary) -> None:
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
    """Every span's basis texts, plus how the run went."""
    summaries: dict[str, SpanSummary] = field(default_factory=dict)
    model: str = ""
    deterministic: bool = True
    cached: int = 0
    generated: int = 0
    failed: tuple[str, ...] = ()
    #: Markers with no paragraphs of their own, so nothing to summarise.
    #: Recorded rather than dropped: a run must account for every span.
    empty: tuple[str, ...] = ()

    def usable(self) -> dict[str, SpanSummary]:
        return {sid: s for sid, s in self.summaries.items() if s.is_usable}

    def stats(self) -> dict[str, Any]:
        fits: dict[str, int] = {}
        for summary in self.summaries.values():
            fits[summary.tier_fit()[BASIS_TIER]] = \
                fits.get(summary.tier_fit()[BASIS_TIER], 0) + 1
        warned = [s.span_id for s in self.summaries.values() if s.warnings]
        return {
            "spans": len(self.summaries),
            "usable": len(self.usable()),
            "cached": self.cached,
            "generated": self.generated,
            "failed": len(self.failed),
            "empty": len(self.empty),
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
    r"""Summarise every span of a `MarkerTree`.

    Two choices here are load-bearing:

    * **`body_text`, not `subtree_text`.** The unit is the smallest set of
      paragraphs that belongs to one heading. Given

          \span{A}    paragraph P0
          \subsection{B} paragraphs P1
          \subsection{C} paragraphs P2

      that is three units: A's own P0, then B, then C. A is NOT P0+P1+P2.

      Sending the subtree meant every child's text was summarised twice, once
      under itself and once under its parent. Measured on the arXiv corpus it
      sent 848,819 characters where the own extents total 541,825 -- 1.57x,
      all of it duplication -- and it made parent and child summaries share
      source text, which manufactures parent-child merges in `basis.py`.

      A span whose own extent is empty -- a heading running straight into
      its first subsection -- is not summarised at all. There is nothing there
      to summarise, and a title is not a concept. 10 of the corpus's 36 parents
      are like this; they are recorded, not silently skipped.
    * **`summarizer_title`, not `title`.** Where caption cleaning dropped a
      macro the cleaned title can be meaningless — `\\ALG\\ Application` becomes
      `Application` — so the raw form is restored for the model.

    A span that fails is recorded and the run continues; one unreachable
    span must not cost a whole corpus.
    """
    prompt_text = prompt if prompt is not None else load_prompt()
    run = SummaryRun(model=getattr(summarizer, "name", "?"),
                     deterministic=bool(getattr(summarizer, "deterministic", False)))
    # Decoding parameters belong in the cache address, not just the model name.
    signature = getattr(summarizer, "cache_signature", None) or run.model
    failed: list[str] = []
    empty: list[str] = []

    for node in tree.iter_document_order():
        if levels is not None and node.level not in levels:
            continue

        title = node.summarizer_title
        body = node.body_text
        if not body.strip():
            # A heading with no paragraphs of its own. Its children carry the
            # content and are summarised in their own right.
            empty.append(node.id)
            continue
        key = summary_key(title, body, signature, prompt_text)

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
    run.empty = tuple(empty)
    return run
