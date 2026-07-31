"""The real summariser: an OpenAI-compatible chat endpoint (Novita by default).

Follows `~/Gemma4/section_concepts.py`, which was built and tuned against these
models — same prompt, same temperature, same rate discipline, same tolerant
JSON recovery.

Credentials come from the **environment only**. There is deliberately no code
path that reads `~/Gemma4/env.txt`: that file holds live keys for five
services, and a library that quietly harvests credentials from a neighbouring
directory is a liability. Set `NOVITA_API_KEY`, or pass `api_key=`.

The network call is injected as a plain callable, so every behaviour below —
retries, throttling, malformed replies, refusals — is testable offline.
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Callable, Optional

from .replyparse import control_corruption, parse_reply
from .sanitize import sanitize_summary_fields
from .summarize import SectionSummary, load_prompt

DEFAULT_MODEL = "inclusionai/ling-3.0-flash"
DEFAULT_BASE_URL = "https://api.novita.ai/v3/openai"

#: Ling 3.0 Flash caps at roughly 30 requests/minute. 2.2s between calls keeps
#: a single-threaded run comfortably underneath without needing a token bucket.
DEFAULT_MIN_INTERVAL = 2.2

#: Chars of section body sent to the model. Enough for a long section, short
#: enough to stay cheap; the model is asked for the concept, not a précis.
#: Model limits, read from the provider rather than assumed.
#:
#: `client.models.list()` reports for inclusionai/ling-3.0-flash:
#:     context_size       262144
#:     max_output_tokens   32768
#: Confirmed by probe: max_tokens=32000 is accepted, 65536 returns
#: "must be between 0 and 32768".
MODEL_CONTEXT_TOKENS = 262_144
MODEL_MAX_OUTPUT_TOKENS = 32_768

#: Characters of section body handed to the model.
#:
#: MEASURED. This was 6000 -- roughly 2000 tokens against a 262144-token
#: context, so 0.8% of what the model can read. Across the 10-document arXiv
#: corpus it silently discarded 172,494 characters from 39 of 203 sections;
#: the largest section is 26,831 characters and 78% of it never reached the
#: model. Sections were being summarised from their opening pages.
#:
#: 200,000 characters is about 65,000 tokens, which leaves the full output
#: budget and most of the context free while still bounding a pathological
#: input. Nothing in the library comes close to it.
DEFAULT_MAX_BODY = 200_000

#: Completion-token ceiling: the model's own maximum.
#:
#: MEASURED, not guessed. `inclusionai/ling-3.0-flash` reasons at length before
#: emitting JSON, and it counts label words aloud because the prompt asks for a
#: word budget. At 900 and at 2000 every probed section returned reasoning with
#: no JSON at all; at 4000 two of three completed. 8000 was still not enough:
#: 5 of 203 sections in the arXiv corpus hit it and raised TruncatedReply.
#:
#: A ceiling costs nothing unspent -- billing is on tokens produced -- so there
#: is no reason to sit below the provider's limit and convert a long reply into
#: a failed section.
DEFAULT_MAX_TOKENS = MODEL_MAX_OUTPUT_TOKENS

#: How much the model may think before answering.
#:
#: `client.models.list()` reports `features: [..., 'reasoning']` for
#: inclusionai/ling-3.0-flash, and reasoning tokens are spent from the SAME
#: completion budget as the answer. That is what the max_tokens ladder was
#: really fighting: at 900, 2000 and 4000 every probed section returned
#: reasoning and no JSON, and at 8000 five of 203 sections still hit the wall
#: (see TruncatedReply in the arXiv corpus). Raising the ceiling treats the
#: symptom; capping the reasoning treats the cause.
#:
#: PROBED against the provider -- `max_tokens`, `max_completion_tokens`, both
#: together, and `reasoning_effort` in ("low", "minimal") are all accepted.
#: Note the provider returns `completion_tokens_details = None`, so the split
#: between reasoning and answer tokens is NOT observable here; the effort cap
#: is the only lever.
#:
#: Set to None to omit the parameter for a provider that rejects it.
DEFAULT_REASONING_EFFORT = "minimal"

#: Fields the prompt asks for.
TIERS = ("summary", "abstraction", "label")

#: A chat call: (system_prompt, user_prompt) -> reply text.
ChatFn = Callable[[str, str], str]


def load_dotenv(path: Optional[str] = None) -> dict[str, str]:
    """Load `KEY=value` pairs from the project's own `.env` into os.environ.

    Only ever this project's `.env`, which is gitignored. Existing environment
    variables win, so an explicit export always overrides the file. This is not
    the same as harvesting credentials from a neighbouring directory: a `.env`
    beside the code is the conventional place for a developer to put their own
    keys, and it is opt-in by existing.
    """
    from pathlib import Path as _Path
    target = _Path(path) if path else _Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not target.exists():
        return loaded
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and not os.environ.get(key):
                os.environ[key] = value
                loaded[key] = value
    except Exception:
        pass
    return loaded


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """API key from the argument, the environment, or the project's .env."""
    if explicit:
        return explicit
    if not any(os.environ.get(n) for n in ("NOVITA_API_KEY", "OPENAI_API_KEY")):
        load_dotenv()
    for name in ("NOVITA_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def make_openai_chat(*, api_key: Optional[str] = None,
                     base_url: Optional[str] = None,
                     model: str = DEFAULT_MODEL,
                     temperature: float = 0.2,
                     max_tokens: int = DEFAULT_MAX_TOKENS,
                     reasoning_effort: Optional[str] = DEFAULT_REASONING_EFFORT,
                     max_retries: int = 4) -> ChatFn:
    """Build a chat callable backed by the `openai` client.

    Imported lazily so the rest of the package works without the dependency.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "the Novita summariser needs the openai package: pip install openai "
            "(or use ExtractiveSummarizer for an offline deterministic floor)"
        ) from exc

    key = resolve_api_key(api_key)
    if not key:
        raise RuntimeError(
            "no API key: set NOVITA_API_KEY (or pass api_key=). "
            "Credentials are never read from a file.")

    client = OpenAI(api_key=key,
                    base_url=base_url or os.environ.get("NOVITA_API_BASE")
                    or DEFAULT_BASE_URL,
                    max_retries=max_retries)

    def chat(system: str, user: str) -> str:
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        reply = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **extra)
        choice = reply.choices[0]
        if choice.finish_reason == "length":
            # A truncated reply is a budget failure, not a malformed one.
            # Reporting it as "reply was not JSON" sent an entire diagnosis
            # down the wrong path: the model had not misunderstood the format,
            # it had never reached it.
            used = getattr(reply, "usage", None)
            raise TruncatedReply(
                f"reply truncated at max_tokens={max_tokens} "
                f"(completion_tokens="
                f"{getattr(used, 'completion_tokens', '?')}); no JSON emitted")
        return choice.message.content or ""

    # The cache key must cover everything that changes the output. max_tokens
    # does: at 900 this model returned truncated reasoning, at 8000 it returns
    # JSON with correctly-sized labels. Without this the cache served the
    # truncated generation's answers into a run that had already fixed it.
    # reasoning_effort belongs here for the same reason max_tokens does: it
    # changes the generation, so a cache built at one setting must not serve a
    # run made at another.
    chat.params = {"model": model, "temperature": temperature,
                   "max_tokens": max_tokens,
                   "reasoning_effort": reasoning_effort}
    return chat


class TruncatedReply(RuntimeError):
    """The model ran out of completion budget before finishing its reply."""


class Throttle:
    """Minimum interval between calls.

    A plain gate rather than a token bucket: the rate limit is per-minute and
    the workload is a couple of dozen sections, so spacing calls evenly is both
    sufficient and easier to reason about.
    """

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._last: Optional[float] = None

    def wait(self) -> float:
        """Block until the next call is allowed. Returns seconds slept."""
        now = self._clock()
        slept = 0.0
        if self._last is not None and self.min_interval > 0:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                slept = remaining
        self._last = self._clock()
        return slept


class NovitaSummarizer:
    """Generates the three basis texts through a chat model."""

    deterministic = False
    #: The only summariser that actually summarises. See
    #: `summarize.ExtractiveSummarizer` for why this flag exists.
    measurement_safe = True

    def __init__(self, chat: ChatFn, *, model: str = DEFAULT_MODEL,
                 prompt: Optional[str] = None,
                 max_body: int = DEFAULT_MAX_BODY,
                 throttle: Optional[Throttle] = None) -> None:
        self._chat = chat
        self.name = model
        self.model = model
        self.prompt = prompt if prompt is not None else load_prompt()
        self.max_body = max_body
        self.throttle = throttle if throttle is not None else Throttle()

    @property
    def cache_signature(self) -> str:
        """Identity of the generator for cache purposes: model AND parameters.

        A summary is a function of the prompt, the text, the model and the
        decoding parameters. Keying on the first three alone let a cache built
        under `max_tokens=900` answer a run made under 8000.
        """
        params = getattr(self._chat, "params", None)
        if not params:
            return self.model
        return "|".join(f"{k}={params[k]}" for k in sorted(params))

    def build_user_prompt(self, title: str, body: str) -> str:
        return self.build_user_prompt_traced(title, body)[0]

    def build_user_prompt_traced(self, title: str, body: str
                                 ) -> tuple[str, int]:
        """The user message, and how many characters of body were dropped.

        Returned rather than discarded. Cutting the input silently is how a
        section came to be summarised from its opening pages with nothing in
        the record to say so.
        """
        text = body or ""
        dropped = max(0, len(text) - self.max_body)
        return f"TITLE: {title}\n\nBODY:\n{text[:self.max_body]}", dropped

    def summarize(self, section_id: str, title: str, body: str) -> SectionSummary:
        """One section. Failures are returned as a record, never raised.

        A single unreachable section must not abort a corpus build, so the
        error travels with the summary and the caller decides what to do.
        """
        self.throttle.wait()
        user, dropped = self.build_user_prompt_traced(title, body)
        body_warnings = ()
        if dropped:
            body_warnings = (f"input truncated: {dropped} characters of the "
                             f"section body were not sent (max_body="
                             f"{self.max_body})",)
        try:
            reply = self._chat(self.prompt, user)
        except Exception as exc:
            return SectionSummary(
                section_id=section_id, title=title, model=self.model,
                deterministic=False, warnings=body_warnings,
                error=f"{type(exc).__name__}: {exc}")

        parsed = parse_reply(reply)
        if isinstance(parsed, dict) and isinstance(parsed.get("concepts"), list):
            # One entry per concept. The first is the dominant one and becomes
            # this SectionSummary; the rest travel in `siblings` so a caller
            # that wants every concept gets them and one that does not is
            # unaffected.
            entries = [e for e in parsed["concepts"] if isinstance(e, dict)]
            if entries:
                built = [self._build(section_id, title, e, body_warnings)
                         for e in entries]
                head, rest = built[0], tuple(built[1:])
                return replace(head, siblings=rest) if rest else head
            parsed = None
        if not parsed:
            return SectionSummary(
                section_id=section_id, title=title, model=self.model,
                deterministic=False,
                error="reply was not JSON",
                warnings=(f"raw reply: {(reply or '')[:200]}",))

        return self._build(section_id, title, parsed, body_warnings)

    def _build(self, section_id: str, title: str, parsed: dict,
               body_warnings: tuple = ()) -> SectionSummary:
        """One concept entry to a `SectionSummary`, sanitised and checked."""
        raw_values = {tier: str(parsed.get(tier) or "").strip() for tier in TIERS}

        warnings: list[str] = list(body_warnings)
        # Detect the eaten-escape damage BEFORE sanitising, because sanitising
        # strips the very control characters that evidence it.
        for tier, text in raw_values.items():
            for offender in control_corruption(text):
                warnings.append(
                    f"{tier}: a LaTeX command was eaten by the legal JSON "
                    f"escape {offender}")

        # Every model-produced string is sanitised. Models emit characters that
        # are invisible or indistinguishable from ASCII -- non-breaking hyphens,
        # zero-width joiners, curly quotes -- and each one changes how the text
        # tokenizes, so a basis vector built from unsanitised text is not the
        # vector it appears to be.
        values, sanitize_warnings = sanitize_summary_fields(raw_values)
        warnings.extend(sanitize_warnings)
        if not any(values.values()):
            return SectionSummary(
                section_id=section_id, title=title, model=self.model,
                deterministic=False, error="reply contained no tier text",
                warnings=tuple(warnings))

        return SectionSummary(
            section_id=section_id, title=title,
            summary=values["summary"], abstraction=values["abstraction"],
            label=values["label"], model=self.model, deterministic=False,
            warnings=tuple(warnings))
