"""`kb reach`: can the reading protocol get to each payload item? (T-0130)

A measure, not a conformance rule, and deterministic: no model is asked.

A placed payload item's **method name** is the title of the section it sits
in, without its numbering. The item is **reachable** when an index entry
*names* that method -- the entry holds at least half of the title's content
words -- and points at a leaf that carries the item. That is INDEX -> letter
file -> leaf -> payload, inside the reading protocol.

Content words are the title's words of three or more characters, minus
function words and the generic words every paper's outline uses ("Results",
"Introduction", "Page 3"): a section called "Results" names no method, so its
payload cannot be found by its method's name at all, and the report says so.
Nothing here knows a field.
"""
from __future__ import annotations

import json
import re
from math import ceil
from pathlib import Path

NUMBERING = re.compile(r"^\s*(?:[IVX]+|\d+(?:\.\d+)*|[A-H])\.?\s+")

#: Function words, and the words of a generic paper outline.
STOP = set("""
a an and are as at be by for from in into is it its of on or that the this to
with without via using based new our their than then these those which while
abstract introduction background related work works preliminaries preliminary
overview summary conclusion conclusions discussion results result experimental
experiment experiments evaluation evaluations method methods methodology
approach approaches proposed proposal analysis performance implementation
details appendix acknowledgements acknowledgments acknowledgement references
page section chapter part future remarks concluding notation definitions
problem statement setup settings dataset datasets data case study studies
""".split())


def method_name(title: str) -> str:
    return NUMBERING.sub("", title or "").strip()


def _tokens(text: str) -> list[str]:
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in re.findall(r"[a-z0-9]+", (text or "").casefold())]


def content_words(name: str) -> set[str]:
    return {w for w in _tokens(name) if len(w) >= 3 and w not in STOP
            and not w.isdigit()}


def names(entry: str, method: str) -> bool:
    """Does the index entry name the method? At least half its content words."""
    want = content_words(method)
    if not want:
        return False
    return len(want & set(_tokens(entry))) >= ceil(len(want) / 2)


def reach(kb: dict) -> dict:
    sections = {s["id"]: s for s in kb["sections"]}
    leaves = {l["id"]: l for l in kb["leaves"]}
    rows = []
    for item in kb["payload"]:
        if item.get("status") != "placed":
            continue
        title = sections.get(item["section"], {}).get("title", "")
        method = method_name(title)
        row = {"id": item["id"], "kind": item["kind"], "section": item["section"],
               "method": method, "reachable": False, "reason": None, "via": []}
        if not content_words(method):
            row["reason"] = "section title names no method"
        else:
            naming = [c for c in kb["concepts"] if names(c["name"], method)]
            via = [c for c in naming
                   if item["id"] in leaves.get(c["leaf"], {}).get("payload", [])]
            if via:
                row["reachable"] = True
                row["via"] = [f"{c['name']} -> {c['leaf']}" for c in via]
            elif naming:
                row["reason"] = "no leaf carrying it is indexed under its method"
            else:
                row["reason"] = "no index entry names its method"
        rows.append(row)
    n = len(rows)
    ok = sum(r["reachable"] for r in rows)
    reasons: dict[str, int] = {}
    for r in rows:
        if r["reason"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {"items": n, "reachable": ok, "share": (ok / n) if n else None,
            "reasons": reasons, "rows": rows}


def reach_dir(kb_dir: Path) -> dict:
    return reach(json.loads((Path(kb_dir) / "kb.json").read_text(encoding="utf-8")))
