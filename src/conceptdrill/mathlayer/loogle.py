"""Verifying Lean anchors against Loogle. An explicit pass, never automatic.

Extraction is offline. This module is the only thing in the package that
touches a network, it is never called from `extract.py`, and it has no default
that quietly does nothing: `verify()` requires a client, and the network client
must be asked for by name.

That shape is `speech.load_speaker`'s — probe the backend, raise
`SpeechUnavailable` rather than degrade — for the same reason. An anchor
silently left `unverified` because a request timed out is indistinguishable
from one nobody checked, and the whole value of the status is telling those
apart.

## Why this is not decoration

MEASURED, step 11 of `newtask.md`, 31 queries against live Loogle:

    Basis                       unknown identifier   -> Module.Basis (1493 hits)
    FiniteDimensional.finrank   unknown identifier   -> Module.finrank (1539 hits)

Two of five names asserted `exact` from memory did not exist, both stale the
same way — moved into the `Module` namespace. A model's recollection of
Mathlib is a snapshot of whenever it was trained, so `unverified` is the only
honest default and this module is what changes it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

from .schema import anchor

ENDPOINT = "https://loogle.lean-lang.org/json"

#: A public service run by volunteers. A burst of queries is rude and nothing
#: here is in a hurry.
DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30


class LoogleUnavailable(RuntimeError):
    """The service could not be reached. NOT the same as "no such name"."""


@runtime_checkable
class LoogleClient(Protocol):
    """One query, one reply. Implemented by `HttpLoogleClient` and by fakes."""

    def query(self, pattern: str) -> dict[str, Any]:
        ...


class HttpLoogleClient:
    """The real thing. Must be constructed deliberately.

    A transport failure raises `LoogleUnavailable`; a query the service
    *answered* with `unknown identifier` returns normally with that error, and
    the difference is the point. The first means "we do not know", the second
    means "we checked and it is not there".
    """

    def __init__(self, *, endpoint: str = ENDPOINT,
                 delay: float = DELAY_SECONDS,
                 timeout: int = TIMEOUT_SECONDS) -> None:
        self.endpoint = endpoint
        self.delay = delay
        self.timeout = timeout
        self._last = 0.0

    def query(self, pattern: str) -> dict[str, Any]:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = f"{self.endpoint}?{urllib.parse.urlencode({'q': pattern})}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as fh:
                payload = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:
            raise LoogleUnavailable(
                f"{type(exc).__name__} querying {pattern!r}: {exc}") from exc
        finally:
            self._last = time.monotonic()

        hits = payload.get("hits") or []
        return {"error": payload.get("error"),
                "count": int(payload.get("count", len(hits)) or 0),
                "hits": hits}


def verify_name(name: str, client: LoogleClient) -> dict[str, Any]:
    """Query one name and return the anchor it justifies.

    A name the service rejects becomes `unresolved` **with its error text
    kept**, because "unknown identifier 'Basis'" is what leads a reader to
    `Module.Basis`. Discarding it would throw away the only clue.
    """
    result = client.query(name)
    error = result.get("error")
    if error:
        return anchor(name, status="unresolved", error=str(error))
    count = int(result.get("count") or 0)
    if not count:
        return anchor(name, status="unresolved", error="no hits")
    module = ""
    for hit in result.get("hits") or []:
        if hit.get("name") == name:
            module = hit.get("module") or ""
            break
    return anchor(name, status="supported", hits=count, module=module)


def verify(records: Iterable[dict[str, Any]], client: LoogleClient, *,
           progress: Optional[Callable[[str, dict], None]] = None
           ) -> list[dict[str, Any]]:
    """Re-check every anchor of every record. Returns new records.

    `not_expected` anchors are left alone: there is nothing to look up, and
    querying them would turn a design fact about Mathlib into a failed lookup.

    `confidence` is recomputed, since it reads anchor status.
    """
    from .schema import confidence

    out: list[dict[str, Any]] = []
    for record in records:
        updated = json.loads(json.dumps(record))
        checked = []
        for anc in updated.get("lean") or []:
            if anc.get("status") == "not_expected":
                checked.append(anc)
                continue
            result = verify_name(anc["name"], client)
            checked.append(result)
            if progress:
                progress(anc["name"], result)
        updated["lean"] = checked
        updated["confidence"] = confidence(updated)
        out.append(updated)
    return out


def verified_against(endpoint: str = ENDPOINT, *, when: str = "") -> str:
    """The provenance string for `MathConceptStore.save`.

    Loogle tracks Mathlib's master rather than a pinned revision, so the date
    is the only thing that identifies what was checked. Recording the endpoint
    without it would claim more precision than exists.
    """
    return f"{endpoint} on {when}" if when else endpoint
