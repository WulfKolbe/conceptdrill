"""The writer: the one stage of `kb build` that calls a language model.

It names leaves, topics and the knowledge base and writes their prose, from
the cited sections only. It never sees payload content: payload is copied by
code, so the model cannot paraphrase an equation or a line of code into the
knowledge base. Every reply is cached by content (model, parameters, prompt,
input), so a rebuild spends nothing and a failed build resumes where it
stopped.
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Callable, Optional

from ..hierarchy.replyparse import parse_reply
from ..hierarchy.sanitize import sanitize_text
from . import contract as C
from .build import BuildError

#: A cited section's text is cut to this many words in the prompt.
SECTION_WORDS = 1200

#: USD per million tokens (input, output) on the provider's price list,
#: read 2026-10-28; the list's unit is assumed, as in the provider test.
PRICES = {
    "inclusionai/ling-3.0-flash": (0.06, 0.18),
    "inclusionai/ling-3.1-flash": (0.30, 0.90),
}
#: Completion tokens per call, measured for ling-3.0-flash at minimal
#: reasoning effort on 16 section prompts (88765 / 16, 2026-10-28). The model
#: reasons in its reply, so this is most of the cost; it is an assumption for
#: other models.
COMPLETION_PER_CALL = 5500

ChatFn = Callable[[str, str], str]


def chat_params(model: str) -> dict:
    """The parameters `make_openai_chat(model=model)` reports as `.params`.

    The writer's cache key includes them, and it must know them before any
    chat exists: a replay from the cache needs no key, so no chat is made.
    A test holds this equal to the chat's own.
    """
    from ..hierarchy.novita import DEFAULT_MAX_TOKENS, DEFAULT_REASONING_EFFORT
    return {"model": model, "temperature": 0.2, "max_tokens": DEFAULT_MAX_TOKENS,
            "reasoning_effort": DEFAULT_REASONING_EFFORT}


def prompt(kind: str) -> str:
    return resources.files("conceptdrill.kb").joinpath(
        "prompts", f"{kind}.md").read_text(encoding="utf-8")


def _clip(text: str, words: int) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " …"


def leaf_input(sections: list[dict], payload: list[dict]) -> dict:
    return {"sections": [{"id": s["id"], "title": s["title"],
                          "text": _clip(s["text"], SECTION_WORDS)} for s in sections],
            "payload": [{"id": p["id"], "kind": p["kind"]} for p in payload]}


def _str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _valid(kind: str, reply: Optional[dict]) -> Optional[dict]:
    """The reply in the shape the renderer needs, or None."""
    if not isinstance(reply, dict):
        return None
    clean = lambda s: sanitize_text(str(s))
    if kind == "leaf":
        if not all(_str(reply.get(k)) for k in ("title", "use_when", "summary")):
            return None
        concepts = reply.get("concepts")
        if not isinstance(concepts, list) or not concepts or not all(
                isinstance(c, dict) and _str(c.get("name")) for c in concepts):
            return None
        lists = {k: reply.get(k, []) for k in ("method", "pitfalls")}
        if not all(isinstance(v, list) and all(isinstance(x, str) for x in v)
                   for v in lists.values()):
            return None
        return {"title": clean(reply["title"]), "use_when": clean(reply["use_when"]),
                "summary": clean(reply["summary"]),
                "concepts": [{"name": clean(c["name"]),
                              "definition": clean(c.get("definition", ""))}
                             for c in concepts[:6]],
                "method": [clean(x) for x in lists["method"] if x.strip()],
                "pitfalls": [clean(x) for x in lists["pitfalls"] if x.strip()]}
    keys = ("title", "purpose", "differ") if kind == "topic" else ("title", "purpose")
    if not all(_str(reply.get(k)) for k in keys):
        return None
    return {k: clean(reply[k]) for k in keys}


class LLMWriter:
    """`chat` may be given, or made by `chat_factory` at the first cache miss
    -- so a build served wholly from the cache needs no key. With
    `cache_only`, a miss is an error naming the entry, and nothing is spent."""
    stub = False

    def __init__(self, chat: Optional[ChatFn] = None, *, model: str, cache_path: Path,
                 params: Optional[dict] = None,
                 chat_factory: Optional[Callable[[], ChatFn]] = None,
                 cache_only: bool = False) -> None:
        self.chat = chat
        self.chat_factory = chat_factory
        self.cache_only = cache_only
        self.params = params if params is not None else \
            getattr(chat, "params", {"model": model})
        self.model = model
        self.cache_path = Path(cache_path)
        self._cache: dict = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._calls = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def usage(self) -> dict:
        if self.chat is None:
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        spent = getattr(self.chat, "usage", None)
        return dict(spent) if spent is not None else dict(self._calls)

    @staticmethod
    def _what(data: dict) -> str:
        items = data.get("sections") or data.get("leaves") or data.get("topics") or []
        return ", ".join(str(i.get("id", i.get("title", "?"))) for i in items)

    def _ask(self, kind: str, data: dict) -> dict:
        system = prompt(kind)
        user = json.dumps(data, ensure_ascii=False, indent=1)
        key = hashlib.sha256(json.dumps([self.model, self.params, system, user],
                                        sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self.cache_only:
            raise BuildError(f"cache miss for the {kind} of {self._what(data)} "
                             f"(key {key[:12]}); --cache-only spends nothing")
        if self.chat is None:
            if self.chat_factory is None:
                raise BuildError(f"cache miss for the {kind} of {self._what(data)} "
                                 f"and no way to make a chat")
            try:
                self.chat = self.chat_factory()
            except RuntimeError as exc:      # no key: fail before any call
                raise BuildError(f"cache miss for the {kind} of {self._what(data)}: "
                                 f"{exc}") from exc
        for _ in range(2):
            self._calls["calls"] += 1
            reply = _valid(kind, parse_reply(self.chat(system, user) or ""))
            if reply is not None:
                self._cache[key] = reply
                self.flush()
                return reply
        raise BuildError(f"the writer gave no usable {kind} reply twice for "
                         f"{self._what(data)}")

    def leaf(self, sections, payload):
        return self._ask("leaf", leaf_input(sections, payload))

    def topic(self, leaves):
        return self._ask("topic", {"leaves": leaves})

    def kb(self, topics):
        return self._ask("kb", {"topics": topics})

    def flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name(self.cache_path.name + ".partial")
        tmp.write_text(json.dumps(self._cache, indent=1, sort_keys=True,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache_path)


class CountingWriter:
    """For `--estimate`: renders every prompt the real writer would send and
    counts its tokens, answering with numbered placeholders. Calls nothing."""
    stub = True
    model = None

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self._n = 0

    @property
    def usage(self) -> dict:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _count(self, kind: str, data: dict) -> None:
        self.calls += 1
        self.prompt_tokens += C.tokens(prompt(kind)) + C.tokens(
            json.dumps(data, ensure_ascii=False))
        self._n += 1

    def leaf(self, sections, payload):
        self._count("leaf", leaf_input(sections, payload))
        return {"title": f"leaf {self._n}", "use_when": "x", "summary": "x",
                "concepts": [{"name": f"c{self._n}", "definition": "x"}],
                "method": [], "pitfalls": []}

    def topic(self, leaves):
        self._count("topic", {"leaves": leaves})
        return {"title": f"topic {self._n}", "purpose": "x", "differ": "x"}

    def kb(self, topics):
        self._count("kb", {"topics": topics})
        return {"title": "kb", "purpose": "x"}


def estimate_cost(model: str, *, calls: int, prompt_tokens: int) -> dict:
    completion = calls * COMPLETION_PER_CALL
    price = PRICES.get(model)
    usd = None if price is None else (prompt_tokens * price[0] + completion * price[1]) / 1e6
    return {"model": model, "calls": calls, "prompt_tokens": prompt_tokens,
            "completion_per_call": COMPLETION_PER_CALL, "completion_tokens": completion,
            "usd": usd}
