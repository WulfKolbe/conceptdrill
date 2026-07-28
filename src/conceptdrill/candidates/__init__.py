"""Candidate generator registry.

`generate_candidates()` runs the requested generators, merges duplicates, and
returns a deterministically ordered list.

Merging matters: the same string frequently arrives from several generators —
"ElasticHash" as a heading, as a frequent noun phrase, and as an ORG entity. The
merged candidate keeps the **highest-structural-weight source** as its primary
(a heading beats a noun phrase beats NER), sums frequencies, and records every
contributing source in metadata. That is what makes multi-source agreement
visible instead of collapsing it.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..abstractor import Abstractor, NullAbstractor
from ..document import Document
from ..types import Candidate
from .base import (BaseGenerator, CandidateGenerator, DEFAULT_STRUCTURAL_WEIGHT,
                   STRUCTURAL_WEIGHTS, structural_weight)
from .bibliography import BibliographyGenerator
from .equations import EquationGenerator
from .glossary import GlossaryGenerator
from .headings import HeadingGenerator
from .ner import NERGenerator
from .nounphrases import NounPhraseGenerator

#: Registry key -> factory. Add an entry to add a generator; nothing else changes.
GENERATORS: dict[str, type] = {
    "heading": HeadingGenerator,
    "glossary": GlossaryGenerator,
    "bibliography": BibliographyGenerator,
    "nounphrase": NounPhraseGenerator,
    "ner": NERGenerator,
    "equation": EquationGenerator,
}

#: Run by default. All six — the spec's whole candidate surface.
DEFAULT_SOURCES: tuple[str, ...] = (
    "heading", "glossary", "bibliography", "nounphrase", "ner", "equation",
)


def build_generators(sources: Optional[Iterable[str]] = None,
                     **overrides) -> list[CandidateGenerator]:
    """Instantiate generators by name. `overrides` maps a name to kwargs."""
    names = list(sources) if sources is not None else list(DEFAULT_SOURCES)
    out: list[CandidateGenerator] = []
    for name in names:
        key = name.strip().lower()
        factory = GENERATORS.get(key)
        if factory is None:
            raise ValueError(
                f"unknown concept source {name!r}; expected one of "
                f"{', '.join(sorted(GENERATORS))}"
            )
        out.append(factory(**(overrides.get(key) or {})))
    return out


def merge_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Deduplicate by normalised name, keeping the strongest source.

    Deterministic: the winner is chosen by structural weight, then by source
    name, never by arrival order.
    """
    grouped: dict[str, list[Candidate]] = {}
    for cand in candidates:
        grouped.setdefault(cand.key, []).append(cand)

    merged: list[Candidate] = []
    for key in sorted(grouped):
        group = grouped[key]
        primary = max(
            group,
            key=lambda c: (structural_weight(c.source), c.source, c.name),
        )
        sources = sorted({c.source for c in group})
        total_frequency = sum(c.frequency for c in group)

        metadata = dict(primary.metadata)
        # Merge non-conflicting metadata from the other contributors so a
        # bibliography year is not lost when a heading wins the name.
        for other in group:
            if other is primary:
                continue
            for mk, mv in other.metadata.items():
                metadata.setdefault(f"{other.source}_{mk}"
                                    if mk in metadata else mk, mv)

        aliases = sorted({
            a for c in group
            for a in (c.metadata.get("aliases") or ())
            if isinstance(a, str)
        })
        if aliases:
            metadata["aliases"] = aliases
        metadata["sources"] = sources
        metadata["source_count"] = len(sources)

        merged.append(Candidate(
            name=primary.name,
            source=primary.source,
            tau=primary.tau,
            kind=primary.kind,
            section_id=primary.section_id,
            frequency=total_frequency,
            metadata=metadata,
        ))

    # Stable final ordering: strongest structural signal first, then frequency,
    # then name. Selection reads this order, so it must not vary between runs.
    merged.sort(key=lambda c: (-structural_weight(c.source), -c.frequency, c.name))
    return merged


def generate_candidates(doc: Document, *,
                        sources: Optional[Iterable[str]] = None,
                        abstractor: Optional[Abstractor] = None,
                        generator_options: Optional[dict] = None,
                        ) -> list[Candidate]:
    """Run the generators over `doc` and return merged candidates."""
    abstractor = abstractor or NullAbstractor()
    generators = build_generators(sources, **(generator_options or {}))
    collected: list[Candidate] = []
    for generator in generators:
        collected.extend(generator.generate(doc, abstractor=abstractor))
    return merge_candidates(collected)


__all__ = [
    "BaseGenerator", "BibliographyGenerator", "CandidateGenerator",
    "DEFAULT_SOURCES", "DEFAULT_STRUCTURAL_WEIGHT", "EquationGenerator",
    "GENERATORS", "GlossaryGenerator", "HeadingGenerator", "NERGenerator",
    "NounPhraseGenerator", "STRUCTURAL_WEIGHTS", "build_generators",
    "generate_candidates", "merge_candidates", "structural_weight",
]
