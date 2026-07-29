"""Recovering JSON from a language model's reply.

Models asked for JSON return *nearly* JSON. Four defect classes account for
almost all of it, and each is worth repairing rather than discarding a whole
section's summary:

  1. markdown fences        ```json ... ```
  2. raw LaTeX backslashes  "\\sum" is not a legal JSON escape -- the dominant
                            failure on scientific text, where the model echoes
                            the author's notation back
  3. unquoted keys          {summary: "..."} instead of {"summary": "..."}
  4. trailing prose         a JSON object followed by an explanation

Repairs are attempted in increasing order of aggression, and clean JSON is
parsed untouched, so a well-behaved reply is never mangled by a repair it did
not need.

This module is pure: text in, dict out. It knows nothing about HTTP, sections,
or files. The approach follows `~/Gemma4/gemmatester.parse_objects`, which was
built against the same models.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator, Optional

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

#: A backslash that does not begin a legal JSON escape. LaTeX in a summary --
#: `\sum`, `\tau`, `\mathcal` -- is otherwise a hard parse error.
_BAD_ESCAPE = re.compile(r'\\(?![\"\\/bfnrtu]|u[0-9a-fA-F]{4})')

#: A bare identifier used as an object key.
_UNQUOTED_KEY = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')

#: Control characters that never occur in real prose. Their presence means a
#: LaTeX command was silently eaten by a *legal* JSON escape -- see
#: `control_corruption`.
_CORRUPTION = {
    "\b": "\\b...",   # \beta  -> BACKSPACE + 'eta'
    "\f": "\\f...",   # \frac  -> FORMFEED  + 'rac'
    "\t": "\\t...",   # \tau   -> TAB       + 'au'
    "\r": "\\r...",   # \rho   -> CR        + 'ho'
}


def control_corruption(value: str) -> tuple[str, ...]:
    """Control characters in `value` that indicate an eaten LaTeX command.

    This is the defect `repair_escapes` *cannot* fix. `\\t`, `\\n`, `\\b`,
    `\\f`, `\\r` are all legal JSON escapes, so a model writing `\\tau` emits
    valid JSON containing a TAB followed by `au`. Nothing fails; the text is
    just quietly wrong. Likewise `\\nabla` -> newline + `abla`, `\\beta` ->
    backspace + `eta`, `\\frac` -> formfeed + `rac`.

    Newline is deliberately absent: a multi-line summary is legitimate, so
    flagging `\\n` would fire on almost every reply. That means `\\nabla`
    escapes detection — a known gap, mitigated by the prompt asking for plain
    text with no LaTeX.

    Returns the offending escape names, or `()` when the text looks clean.
    """
    return tuple(name for ch, name in _CORRUPTION.items() if ch in (value or ""))


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if any."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = _FENCE.sub("", stripped)
    return stripped.strip()


def repair_escapes(text: str) -> str:
    """Double backslashes that are not legal JSON escapes, so LaTeX survives."""
    return _BAD_ESCAPE.sub(r"\\\\", text)


def repair_keys(text: str) -> str:
    """Quote bare identifier keys."""
    return _UNQUOTED_KEY.sub(r'\1"\2"\3', text)


def _candidates(body: str) -> Iterator[str]:
    """Progressively repaired parse attempts, least aggressive first."""
    yield body
    yield repair_escapes(body)
    yield repair_keys(repair_escapes(body))


def iter_json_objects(text: str) -> Iterator[str]:
    """Yield each balanced top-level `{...}` block.

    String-aware, so a brace inside a quoted value does not throw off the
    depth count. This is what rescues an object followed by trailing prose.
    """
    depth = 0
    start: Optional[int] = None
    in_string = escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def parse_reply(text: str) -> Optional[dict[str, Any]]:
    """The first JSON object in a model reply, or None.

    Returns a dict even when the model wrapped its answer in a list, because
    every caller here wants one object per section.
    """
    body = strip_fences(text)
    if not body:
        return None

    for candidate in _candidates(body):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
        return None

    # Last resort: salvage the first object that parses on its own, so trailing
    # prose or a second malformed object cannot cost us the whole reply.
    for block in iter_json_objects(body):
        for candidate in _candidates(block):
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict):
                return data
    return None
