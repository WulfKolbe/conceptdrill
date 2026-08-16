r"""Turning a span into `MathConcept` candidates.

Shaped like `summarize.Summarizer`, deliberately: a protocol, several
implementations, and a `measurement_safe` flag that keeps fixtures out of
measurements **mechanically** rather than by convention. `ExtractiveSummarizer`
being usable as a stand-in for a real summariser voided a 416-document corpus
build. The same trap is available here and gets the same guard.

    LLMExtractor       the real thing.  measurement_safe = True
    FixtureExtractor   a stored answer. measurement_safe = False

## What the extractor sees

Span text with the mathematics already spoken — that is what `body_text` holds
— plus the LaTeX of the formulas in the span, which speech cannot recover.
Both are needed and neither substitutes for the other:

    \to  and  \rightarrow   speak identically ("right arrow")

so a layer given only speech cannot record which spelling the author used, and
`MathConcept.latex` exists to preserve exactly that.

## What it must not do

Not embed anything, not consult a basis, not open a network connection. Step 15
measured CES retrieval as worse than uninformative for mathematics: one basis
row was top-1 for four of six continuations, and the wrong answers outscored
the right ones. Verification of Lean anchors is `loogle.py`, invoked
separately.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from ..hierarchy.sanitize import sanitize_text
from .schema import (RELATIONS, SchemaViolation, anchor, evidence,
                     mathconcept_record, relation, symbol)

#: The instruction. Kept in this module rather than a prompt file because it is
#: short and because the field vocabulary it names must not drift from
#: `schema.py` without both changing together.
PROMPT = """You extract mathematical concepts from one section of a research \
paper. Mathematics reaches you as SPOKEN TEXT, because formulas were read \
aloud by a speech engine: "T colon V right arrow W" is $T : V \\to W$.

Return ONLY a JSON object:

{"concepts": [
  {"concept": "linear map",
   "spoken": "T colon V right arrow W",
   "latex": ["T : V \\\\to W"],
   "symbols": [{"latex": "T", "role": "instance", "status": "observed"}],
   "related": [{"relation": "domain", "target": "vector space",
                "status": "observed"}]}
]}

RULES, and the first is the one that matters:

1. STATUS IS NOT OPTIONAL AND MUST NOT BE GUESSED.
   observed  - the string is literally in the text I gave you
   implied   - it follows from a phrase or notation in the text
   predicted - concepts like this usually carry it, and it is NOT here
   external  - true of the concept generally, absent from this text
   If the word "field" does not appear, a field concept is implied, not
   observed. Marking a prediction as observed is the worst error you can make.

2. INVENT NOTHING. If the text does not say a map is injective, no relation
   says so. Absence of a fact is a fact.

3. relation must be one of: %s

4. Preserve LaTeX EXACTLY as written. Never normalise \\colon to : or
   \\rightarrow to \\to. Which spelling was used is evidence.

5. Name concepts canonically and in lower case: "linear map", not "the linear
   map T" and not "Linear Maps".

6. Propose no Lean names. That is a separate, verified step.

Return between 1 and 6 concepts. Fewer, correct, is better than more.""" % (
    ", ".join(sorted(RELATIONS)),)


class ExtractionFailed(RuntimeError):
    """The extractor could not produce records. Never returns empty silently."""


class NotMeasurementSafe(RuntimeError):
    """A fixture extractor was handed to something that measures."""


@runtime_checkable
class Extractor(Protocol):
    """Produces MathConcept records for one span."""

    name: str
    measurement_safe: bool

    def extract(self, span_id: str, title: str, body: str, *,
                latex: Sequence[str] = (), doc_id: str = "",
                ces_row_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        ...


def require_measurement_safe(extractor: Extractor) -> Extractor:
    """Raise unless `extractor` may appear in a measurement."""
    if not getattr(extractor, "measurement_safe", False):
        raise NotMeasurementSafe(
            f"{getattr(extractor, 'name', extractor)!r} is not measurement "
            f"safe and must not be used in a run whose numbers are reported")
    return extractor


# --------------------------------------------------------------------------
# Parsing a reply
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_reply(reply: str, *, span_id: str = "", doc_id: str = "",
                ces_row_ids: Sequence[str] = (),
                quote: Optional[str] = None) -> list[dict[str, Any]]:
    """Model reply to validated records.

    Every part goes through `schema.symbol` / `schema.relation`, so a bad
    status or an unknown relation raises here rather than reaching a file. A
    concept whose parts are all rejected is dropped and the rest survive: one
    malformed entry must not cost the whole span.
    """
    text = (reply or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise ExtractionFailed("empty reply")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionFailed(f"reply was not JSON: {exc}") from exc
    if not isinstance(payload, dict) or "concepts" not in payload:
        raise ExtractionFailed("reply has no `concepts` key")

    out: list[dict[str, Any]] = []
    for entry in payload.get("concepts") or []:
        if not isinstance(entry, dict):
            continue
        name = sanitize_text(str(entry.get("concept") or "")).strip()
        if not name:
            continue
        try:
            symbols = [symbol(sanitize_text(str(s.get("latex") or "")),
                              role=str(s.get("role") or ""),
                              status=str(s.get("status") or "predicted"))
                       for s in entry.get("symbols") or []
                       if isinstance(s, dict) and (s.get("latex") or "").strip()]
            relations = [relation(str(r.get("relation") or ""),
                                  sanitize_text(str(r.get("target") or "")),
                                  status=str(r.get("status") or "implied"))
                         for r in entry.get("related") or []
                         if isinstance(r, dict)]
        except SchemaViolation:
            # The model used a vocabulary that is not ours. Dropping this
            # concept is right; repairing it would be inventing the status.
            continue

        out.append(mathconcept_record(
            concept=name,
            spoken=sanitize_text(str(entry.get("spoken") or "")) or None,
            latex=[str(x) for x in entry.get("latex") or [] if str(x).strip()],
            symbols=symbols,
            related=relations,
            lean=[],
            loogle=[],
            evidence=evidence(doc_id=doc_id,
                              span_ids=[span_id] if span_id else [],
                              ces_row_ids=list(ces_row_ids),
                              quote=quote),
        ))
    if not out:
        raise ExtractionFailed("no usable concepts in the reply")
    return out


def render_input(title: str, body: str, latex: Sequence[str] = ()) -> str:
    """What the model is shown. LaTeX is listed separately from the prose.

    Separately because speech is lossy about spelling: `\\to` and
    `\\rightarrow` are one spoken string, and `MathConcept.latex` has to record
    which one the author wrote.
    """
    parts = [f"SECTION TITLE: {title}".strip(), "", "SECTION TEXT (mathematics "
             "appears as spoken words):", body.strip()]
    forms = [x for x in latex if (x or "").strip()]
    if forms:
        parts += ["", "LATEX OF THE FORMULAS IN THIS SECTION, exactly as "
                  "written:"] + [f"  {x}" for x in forms]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------

class LLMExtractor:
    """The real extractor. Takes a chat callable, like `NovitaSummarizer`."""

    measurement_safe = True

    def __init__(self, chat: Callable[[str, str], str], *,
                 model: str = "", prompt: str = PROMPT) -> None:
        self.chat = chat
        self.model = model
        self.prompt = prompt
        self.name = f"llm:{model}" if model else "llm"

    @property
    def cache_signature(self) -> str:
        params = getattr(self.chat, "params", {}) or {}
        return json.dumps({"name": self.name, **params}, sort_keys=True)

    def extract(self, span_id: str, title: str, body: str, *,
                latex: Sequence[str] = (), doc_id: str = "",
                ces_row_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        if not (body or "").strip():
            raise ExtractionFailed(f"{span_id}: empty body")
        reply = self.chat(self.prompt, render_input(title, body, latex))
        return parse_reply(reply, span_id=span_id, doc_id=doc_id,
                           ces_row_ids=ces_row_ids,
                           quote=(body or "").strip()[:200])


class FixtureExtractor:
    """TEST FIXTURE ONLY. Replays stored records; extracts nothing.

    `measurement_safe = False` keeps it out of anything that reports numbers,
    mechanically. It exists so the suite runs offline and deterministically,
    which is the same reason `ExtractiveSummarizer` exists and the same reason
    that one is flagged.
    """

    name = "fixture"
    measurement_safe = False

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = [dict(r) for r in records]

    def extract(self, span_id: str, title: str, body: str, *,
                latex: Sequence[str] = (), doc_id: str = "",
                ces_row_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        if not self.records:
            raise ExtractionFailed("fixture holds no records")
        return [dict(r) for r in self.records]
