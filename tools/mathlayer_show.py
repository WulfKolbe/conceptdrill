#!/usr/bin/env python3
r"""Step 21: show everything the pipeline knows about one span.

    PYTHONPATH=src python3 tools/mathlayer_show.py --fixture
    PYTHONPATH=src python3 tools/mathlayer_show.py \
        --docmodel ~/pdfdrill-library/1702.06385/model.docmodel.json --span SE0003
    PYTHONPATH=src python3 tools/mathlayer_show.py --fixture --verify

Prints, in order: the span text, its CES concepts, its mathematical concepts,
the notation candidates, the Lean anchors and the Loogle queries.

Read-only. It opens a docmodel, never writes to it, and touches no basis.
`--verify` is the one thing here that reaches the network, and it has to be
asked for by name.

## Why CES appears at all

It is printed and it is not used. Step 15 measured CES retrieval as worse than
uninformative for mathematics: one basis row was top-1 for four of six
mathematical continuations and the wrong answers outscored the right ones. The
CES section is there so a reader can see that for themselves on their own
document, which is more convincing than the claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conceptdrill.mathlayer import (HttpLoogleClient, LoogleUnavailable,  # noqa: E402
                                    MathConceptStore, verified_against, verify)

FIXTURE_CONCEPTS = (Path(__file__).resolve().parents[1] / "experiments"
                    / "math-layer" / "mathconcepts.json")
FIXTURE_TEXT = (Path(__file__).resolve().parents[1] / "tests" / "fixtures"
                / "vector-space.md")

BAR = "-" * 72


def heading(text: str) -> None:
    print(f"\n{text}\n{BAR}")


def show_span(title: str, body: str, latex) -> None:
    heading(f"SPAN  {title}")
    print(body.strip()[:1200] or "(empty)")
    if latex:
        heading("LATEX IN THIS SPAN (exactly as written)")
        for form in latex[:20]:
            print(f"  {form}")


def show_ces(concepts) -> None:
    heading("CES CONCEPTS  (printed for inspection; NOT used to rank anything)")
    if not concepts:
        print("  none recorded for this span")
        return
    for i, label in enumerate(concepts, start=1):
        print(f"  {i}. {label}")


def show_math(records) -> None:
    heading("MATHEMATICAL CONCEPTS")
    for record in sorted(records, key=lambda r: (-r["confidence"], r["concept"])):
        print(f"\n  {record['concept']}   confidence {record['confidence']}")
        if record.get("spoken"):
            print(f"    spoken:  {record['spoken']}")
        for form in record.get("latex") or []:
            print(f"    latex:   {form}")
        for rel in record.get("related") or []:
            print(f"    {rel['relation']:<14} {rel['target']}  "
                  f"[{rel['status']}]")
        evid = record.get("evidence") or {}
        if evid.get("quote"):
            print(f"    from:    {evid['quote'][:80]!r}")
        elif not evid.get("span_ids"):
            print("    from:    NOT IN THE TEXT -- reached by inference only")


def show_notation(records) -> None:
    heading("NOTATION CANDIDATES")
    print(f"  {'symbol':<28} {'status':<10} {'role':<14} concept")
    for record in records:
        for sym in record.get("symbols") or []:
            print(f"  {sym['latex']:<28} {sym['status']:<10} "
                  f"{(sym.get('role') or '-'):<14} {record['concept']}")


def show_anchors(records) -> None:
    heading("LEAN ANCHORS")
    print(f"  {'name':<30} {'status':<12} {'hits':>7}  concept")
    any_shown = False
    for record in records:
        for anc in record.get("lean") or []:
            any_shown = True
            hits = anc.get("hits")
            print(f"  {anc['name']:<30} {anc['status']:<12} "
                  f"{(hits if hits is not None else '-'):>7}  "
                  f"{record['concept']}"
                  + (f"   {anc['error']}" if anc.get("error") else ""))
        if not (record.get("lean") or []):
            any_shown = True
            print(f"  {'(none expected)':<30} {'not_expected':<12} "
                  f"{'-':>7}  {record['concept']}")
    if not any_shown:
        print("  none proposed")


def show_loogle(records) -> None:
    heading("LOOGLE QUERIES")
    seen = set()
    for record in records:
        for pattern in record.get("loogle") or []:
            if pattern in seen:
                continue
            seen.add(pattern)
            print(f"  {pattern:<34} ({record['concept']})")
    if not seen:
        print("  none")


def load_docmodel_span(path: Path, span_id: str):
    """Span title, body and formula LaTeX, straight from a docmodel.

    Read-only, and it goes through the hierarchy tree reader rather than
    reimplementing extent assignment -- the positional rule is subtle and
    duplicating it would produce a second, slightly different answer.
    """
    from conceptdrill.hierarchy.docmodel_tree import build_tree
    from conceptdrill.hierarchy.mathtext import latex_from_props

    objects = json.loads(path.read_text(encoding="utf-8")).get("objects") or []
    tree = build_tree(objects, bibkey=path.parent.name, source_path=str(path))
    node = tree.nodes.get(span_id)
    if node is None:
        raise SystemExit(
            f"no span {span_id!r}. Available: "
            f"{', '.join(list(tree.nodes)[:12])} ...")

    owned = {p.id for p in node.paragraphs}
    latex = [latex_from_props(o.get("props") or {}) for o in objects
             if str(o.get("id") or "") in owned]
    return node.summarizer_title, node.body_text, [x for x in latex if x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="store_true",
                    help="use tests/fixtures/vector-space.md")
    ap.add_argument("--docmodel", default="")
    ap.add_argument("--span", default="")
    ap.add_argument("--concepts", default="",
                    help="a math-concepts.json; defaults to the fixture's")
    ap.add_argument("--verify", action="store_true",
                    help="re-check every Lean anchor against live Loogle")
    args = ap.parse_args()

    if args.fixture:
        title = "vector-space fixture"
        body = ("V is an n-dimensional vector space over K. "
                "W is an m-dimensional vector space over K. "
                "Let T colon V right arrow W be linear.")
        latex = [r"V", r"W", r"K", r"T : V \to W", r"T \colon V \rightarrow W"]
        ces = []
    elif args.docmodel and args.span:
        title, body, latex = load_docmodel_span(Path(args.docmodel).expanduser(),
                                                args.span)
        ces = []
    else:
        ap.error("give --fixture, or --docmodel with --span")

    source = Path(args.concepts).expanduser() if args.concepts else FIXTURE_CONCEPTS
    if source.name == "math-concepts.json":
        records = MathConceptStore(source.parent).concepts()
    else:
        records = json.loads(source.read_text(encoding="utf-8"))["concepts"]

    if args.verify:
        heading("VERIFYING ANCHORS AGAINST LIVE LOOGLE")
        try:
            records = verify(records, HttpLoogleClient(),
                             progress=lambda n, r: print(
                                 f"  {n:<30} {r['status']}"))
            print(f"\n  verified against {verified_against(when='now')}")
        except LoogleUnavailable as exc:
            # Reported, never swallowed: an anchor left unverified because the
            # network failed must not read as one nobody proposed.
            print(f"  UNAVAILABLE: {exc}")
            print("  anchors below keep their stored status")

    show_span(title, body, latex)
    show_ces(ces)
    show_math(records)
    show_notation(records)
    show_anchors(records)
    show_loogle(records)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
