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
from typing import Callable, Optional

from .replyparse import control_corruption, parse_reply
from .summarize import SectionSummary, load_prompt

DEFAULT_MODEL = "inclusionai/ling-3.0-flash"
DEFAULT_BASE_URL = "https://api.novita.ai/v3/openai"

#: Ling 3.0 Flash caps at roughly 30 requests/minute. 2.2s between calls keeps
#: a single-threaded run comfortably underneath without needing a token bucket.
DEFAULT_MIN_INTERVAL = 2.2

#: Chars of section body sent to the model. Enough for a long section, short
#: enough to stay cheap; the model is asked for the concept, not a précis.
DEFAULT_MAX_BODY = 6000

#: Fields the prompt asks for.
TIERS = ("summary", "abstraction", "label")

#: A chat call: (system_prompt, user_prompt) -> reply text.
ChatFn = Callable[[str, str], str]


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """API key from the argument or the environment. Never from a dotfile."""
    if explicit:
        return explicit
    for name in ("NOVITA_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def make_openai_chat(*, api_key: Optional[str] = None,
                     base_url: Optional[str] = None,
                     model: str = DEFAULT_MODEL,
                     temperature: float = 0.2,
                     max_tokens: int = 900,
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
        reply = client.chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return reply.choices[0].message.content or ""

    return chat


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

    def build_user_prompt(self, title: str, body: str) -> str:
        return f"TITLE: {title}\n\nBODY:\n{(body or '')[:self.max_body]}"

    def summarize(self, section_id: str, title: str, body: str) -> SectionSummary:
        """One section. Failures are returned as a record, never raised.

        A single unreachable section must not abort a corpus build, so the
        error travels with the summary and the caller decides what to do.
        """
        self.throttle.wait()
        try:
            reply = self._chat(self.prompt, self.build_user_prompt(title, body))
        except Exception as exc:
            return SectionSummary(
                section_id=section_id, title=title, model=self.model,
                deterministic=False,
                error=f"{type(exc).__name__}: {exc}")

        parsed = parse_reply(reply)
        if not parsed:
            return SectionSummary(
                section_id=section_id, title=title, model=self.model,
                deterministic=False,
                error="reply was not JSON",
                warnings=(f"raw reply: {(reply or '')[:200]}",))

        values = {tier: str(parsed.get(tier) or "").strip() for tier in TIERS}

        warnings: list[str] = []
        for tier, text in values.items():
            for offender in control_corruption(text):
                warnings.append(
                    f"{tier}: a LaTeX command was eaten by the legal JSON "
                    f"escape {offender}")
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
