"""Object type -> default embedding model.

The defaults are the spec's table. `code` and `algorithm` are registered even
though the Semantic Compiler emits neither today: when it starts to, they route
to CodeBERT without a code change here, and until then they are simply never
hit. Registering them costs nothing and documents the intent.

An explicit `--model` always wins over routing.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_MODEL = "sentencebert"

#: Block type (lowercase) -> model name.
TYPE_MODEL: dict[str, str] = {
    # prose
    "paragraph": "sentencebert",
    "abstract": "sentencebert",
    "heading": "sentencebert",
    "section": "sentencebert",
    "caption": "sentencebert",
    "footnote": "sentencebert",
    "sidenote": "sentencebert",
    "listitem": "sentencebert",
    "bibliography": "sentencebert",
    "bibitem": "sentencebert",
    "bibentry": "sentencebert",
    "table": "sentencebert",
    "figure": "sentencebert",

    # mathematics
    "equation": "mathbert",
    "formula": "mathbert",
    "displayequation": "mathbert",
    "inlinemath": "mathbert",

    # code — dormant: the DocModel emits no such objects yet
    "code": "codebert",
    "sourcecode": "codebert",
    "listing": "codebert",
    "algorithm": "codebert",
}

#: Types routed to a model that no current DocModel object reaches. Surfaced in
#: `--explain-routing` so the gap is visible rather than mysterious.
DORMANT_TYPES: frozenset[str] = frozenset({"code", "sourcecode", "listing",
                                           "algorithm"})


def model_for_type(block_type: str, override: Optional[str] = None) -> str:
    """Which model should embed this block type."""
    if override:
        return override
    return TYPE_MODEL.get((block_type or "").lower(), DEFAULT_MODEL)


def routing_table() -> dict[str, dict[str, object]]:
    """The full table, annotated with dormancy. For introspection and docs."""
    return {
        btype: {"model": model, "dormant": btype in DORMANT_TYPES}
        for btype, model in sorted(TYPE_MODEL.items())
    }


def group_by_model(block_types: list[str],
                   override: Optional[str] = None) -> dict[str, list[str]]:
    """Invert the routing for a set of types, so each model loads once."""
    grouped: dict[str, list[str]] = {}
    for btype in block_types:
        grouped.setdefault(model_for_type(btype, override), []).append(btype)
    return {model: sorted(set(types)) for model, types in sorted(grouped.items())}
