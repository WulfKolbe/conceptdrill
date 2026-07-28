"""Command-line interface.

Subcommands::

    conceptdrill project  INPUT [--model M]... [--concept-source S]...
    conceptdrill concepts INPUT [--top N]
    conceptdrill explain  INPUT --text "..." [--top N]
    conceptdrill verify   SIDECAR
    conceptdrill routing

The flat form from the spec is also accepted and dispatches to `explain`::

    conceptdrill --input paper.json --text "Deep learning for graphs" --top 5

Repeating `--model` produces one independent projection set per model, which is
what makes multi-model agreement a read over stored output rather than new code.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .candidates import DEFAULT_SOURCES, GENERATORS
from .embeddings import KNOWN_MODELS
from .routing import routing_table
from .scoring.metrics import DEFAULT_WEIGHTS
from .storage import build_payload, sidecar_path, verify_sidecar, write_sidecar


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def _add_build_args(p: argparse.ArgumentParser) -> None:
    """Options controlling how the concept space is built."""
    p.add_argument("--model", "-m", action="append", metavar="NAME",
                   help=f"embedding model, repeatable. One of "
                        f"{', '.join(KNOWN_MODELS)} or an org/checkpoint path "
                        f"(default: sentencebert)")
    p.add_argument("--concept-source", "-s", action="append", metavar="SOURCE",
                   choices=sorted(GENERATORS),
                   help=f"candidate generator, repeatable "
                        f"(default: all of {', '.join(DEFAULT_SOURCES)})")
    p.add_argument("--max-concepts", type=int, default=100,
                   help="size of the selected vocabulary (default: 100)")
    p.add_argument("--diversity", type=float, default=0.95, metavar="T",
                   help="reject a concept more similar than T to one already "
                        "selected (default: 0.95)")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="discard candidates scoring below this (default: 0.0)")
    p.add_argument("--theta", type=float, default=0.5,
                   help="similarity above which a block counts as 'about' a "
                        "concept, used by the quality metrics (default: 0.5)")
    p.add_argument("--weight", action="append", metavar="METRIC=VALUE",
                   help="override a scoring weight, repeatable. Metrics: "
                        + ", ".join(sorted(DEFAULT_WEIGHTS)))
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write the embedding cache")
    p.add_argument("--cache-dir", metavar="DIR",
                   help="embedding cache location (default: .conceptdrill_cache)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptdrill",
        description="Build a document-tailored concept space and project text "
                    "into it (CES over the document's own structure).",
    )
    parser.add_argument("--version", action="version",
                        version=f"conceptdrill {__version__}")

    sub = parser.add_subparsers(dest="command")

    # -- project ---------------------------------------------------------
    p_project = sub.add_parser(
        "project", help="project every object in a document and write a sidecar")
    p_project.add_argument("input", help="document JSON or model.docmodel.json")
    p_project.add_argument("--projection", default="concepts",
                           help="projection type label (default: concepts)")
    p_project.add_argument("--output", "-o", help="sidecar path "
                           "(default: <input>.conceptdrill.json)")
    p_project.add_argument("--top", "-k", type=int, default=10,
                           help="concepts stored per object (default: 10)")
    p_project.add_argument("--types", nargs="*", metavar="TYPE",
                           help="restrict to these object types")
    p_project.add_argument("--store-embeddings", action="store_true",
                           help="include each object's embedding in the sidecar")
    p_project.add_argument("--dry-run", action="store_true",
                           help="report what would be written, write nothing")
    _add_build_args(p_project)

    # -- concepts --------------------------------------------------------
    p_concepts = sub.add_parser(
        "concepts", help="print the selected concept vocabulary")
    p_concepts.add_argument("input")
    p_concepts.add_argument("--top", "-k", type=int, default=0,
                            help="show only the top N (default: all)")
    p_concepts.add_argument("--metrics", action="store_true",
                            help="show the per-metric breakdown")
    p_concepts.add_argument("--json", action="store_true", dest="as_json")
    _add_build_args(p_concepts)

    # -- explain ---------------------------------------------------------
    p_explain = sub.add_parser(
        "explain", help="show the top concepts for a text span")
    p_explain.add_argument("input", nargs="?")
    p_explain.add_argument("--input", "-i", dest="input_opt",
                           help="document JSON (alternative to positional)")
    p_explain.add_argument("--text", "-t", required=True, help="text to project")
    p_explain.add_argument("--top", "-k", type=int, default=5)
    p_explain.add_argument("--json", action="store_true", dest="as_json")
    _add_build_args(p_explain)

    # -- verify / routing ------------------------------------------------
    p_verify = sub.add_parser("verify", help="recompute a sidecar's content hash")
    p_verify.add_argument("sidecar")

    sub.add_parser("routing", help="print the object-type -> model routing table")

    return parser


def _parse_weights(pairs: Optional[Sequence[str]]) -> Optional[dict[str, float]]:
    if not pairs:
        return None
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--weight expects METRIC=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in DEFAULT_WEIGHTS:
            raise SystemExit(
                f"unknown metric {key!r}; expected one of "
                f"{', '.join(sorted(DEFAULT_WEIGHTS))}")
        try:
            out[key] = float(value)
        except ValueError:
            raise SystemExit(f"--weight value must be a number, got {value!r}")
    return out


def _build_kwargs(args: argparse.Namespace) -> dict:
    from .scoring.scorer import QualityScorer
    weights = _parse_weights(getattr(args, "weight", None))
    return {
        "sources": args.concept_source or None,
        "max_concepts": args.max_concepts,
        "diversity_threshold": args.diversity,
        "min_score": args.min_score,
        "scorer": QualityScorer(weights=weights, theta=args.theta),
        "cache": not args.no_cache,
        "cache_dir": args.cache_dir,
    }


def _models(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.model or ["sentencebert"]))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_project(args: argparse.Namespace) -> int:
    from .core import ConceptDrill

    src = Path(args.input)
    if not src.exists():
        print(f"conceptdrill: no such file: {src}", file=sys.stderr)
        return 2

    kwargs = _build_kwargs(args)
    models = _models(args)
    created_at = _timestamp()

    all_projections = []
    all_skipped = []
    infos: dict[str, dict] = {}
    space = None

    for model in models:
        drill = ConceptDrill.from_path(src, embedding_model_name=model, **kwargs)
        projections, skipped = drill.project_document(
            top_k=args.top, types=args.types,
            store_embedding=args.store_embeddings,
            created_at=created_at,
        )
        all_projections.extend(projections)
        # Skips are identical across models; record them once.
        if not all_skipped:
            all_skipped = skipped
        infos[model] = drill.get_concept_space_info()
        space = drill.space
        print(f"  {model:14s} {len(drill.space):4d} concepts  "
              f"{len(projections):4d} projections", file=sys.stderr)

    if space is None:
        print("conceptdrill: nothing to project", file=sys.stderr)
        return 1

    payload = build_payload(
        source_path=str(src), space=space, projections=all_projections,
        skipped=all_skipped,
        meta={"models": models, "projection_type": args.projection,
              "created_at": created_at, "per_model": infos},
        store_embeddings=args.store_embeddings,
    )

    if args.dry_run:
        print(json.dumps({
            "would_write": str(sidecar_path(src, args.output)),
            "projections": len(all_projections),
            "skipped": len(all_skipped),
            "concepts": len(space),
            "content_hash": payload["content_hash"],
        }, indent=2))
        return 0

    out = write_sidecar(payload, sidecar_path(src, args.output))
    print(f"{out}  ({len(all_projections)} projections, "
          f"{len(all_skipped)} skipped, hash {payload['content_hash'][:12]})")
    return 0


def cmd_concepts(args: argparse.Namespace) -> int:
    from .core import ConceptDrill

    src = Path(args.input)
    if not src.exists():
        print(f"conceptdrill: no such file: {src}", file=sys.stderr)
        return 2

    model = _models(args)[0]
    drill = ConceptDrill.from_path(src, embedding_model_name=model,
                                   **_build_kwargs(args))
    concepts = drill.space.concepts
    if args.top:
        concepts = concepts[:args.top]

    if args.as_json:
        print(json.dumps({
            "info": drill.get_concept_space_info(),
            "concepts": [
                {"id": c.id, "name": c.name, "score": c.score,
                 "source": c.source, "level": c.level,
                 "parent_id": c.parent_id, "children": list(c.children),
                 "description": c.description, "aliases": list(c.aliases),
                 "metrics": c.metrics}
                for c in concepts
            ],
        }, indent=2))
        return 0

    info = drill.get_concept_space_info()
    print(f"{len(drill.space)} concepts from {info['build']['n_candidates']} "
          f"candidates  [model={drill.embedder.name} "
          f"nlp={info['build']['nlp_backend']}]")
    print()
    for i, c in enumerate(concepts, 1):
        print(f"{i:4d}. {c.score:.3f}  {c.name}   [{c.source}]")
        if args.metrics:
            parts = "  ".join(f"{k}={v:.2f}" for k, v in sorted(c.metrics.items()))
            print(f"        {parts}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .core import ConceptDrill

    target = args.input or args.input_opt
    if not target:
        print("conceptdrill explain: an input document is required "
              "(positional or --input)", file=sys.stderr)
        return 2
    src = Path(target)
    if not src.exists():
        print(f"conceptdrill: no such file: {src}", file=sys.stderr)
        return 2

    model = _models(args)[0]
    drill = ConceptDrill.from_path(src, embedding_model_name=model,
                                   **_build_kwargs(args))
    hits = drill.explain_hits(args.text, top_k=args.top)

    if args.as_json:
        print(json.dumps({
            "text": args.text, "model": drill.embedder.name,
            "revision": drill.embedder.revision,
            "concepts": [{"concept_id": h.concept_id, "concept_name": h.concept_name,
                          "similarity": h.similarity, "rank": h.rank}
                         for h in hits],
        }, indent=2))
        return 0

    print(f"{args.text!r}  [{drill.embedder.name}, {len(drill.space)} concepts]")
    for h in hits:
        print(f"  {h.rank:2d}. {h.similarity:+.4f}  {h.concept_name}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.sidecar)
    if not path.exists():
        print(f"conceptdrill: no such file: {path}", file=sys.stderr)
        return 2
    matches, stored, recomputed = verify_sidecar(path)
    print(f"{'OK  ' if matches else 'FAIL'} {path}")
    print(f"  stored:     {stored}")
    print(f"  recomputed: {recomputed}")
    return 0 if matches else 1


def cmd_routing(args: argparse.Namespace) -> int:
    table = routing_table()
    width = max(len(k) for k in table)
    for btype, entry in table.items():
        note = "  (dormant: no such object in the DocModel yet)" if entry["dormant"] else ""
        print(f"{btype:<{width}}  ->  {entry['model']}{note}")
    return 0


COMMANDS = {
    "project": cmd_project,
    "concepts": cmd_concepts,
    "explain": cmd_explain,
    "verify": cmd_verify,
    "routing": cmd_routing,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # The spec's flat form: `conceptdrill --input X --text Y --top 5`.
    # No subcommand, but --text present -> explain.
    if argv and argv[0].startswith("-") and "--text" in argv or \
       (argv and argv[0].startswith("-") and any(a.startswith("-t") for a in argv)):
        if not any(a in COMMANDS for a in argv):
            argv = ["explain", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"conceptdrill: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
