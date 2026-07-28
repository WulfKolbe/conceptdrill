"""1.1 Section headings.

Emits two candidates per section: the bare title, and — for nested sections —
the breadcrumb path ("Method > Semantic Projection"). The path form disambiguates
generic titles: three papers' worth of "Results" sections are not the same
concept, but "Evaluation > Results" and "Ablation > Results" are distinguishable.

The tau mapping for a path candidate is the *readable* form, because the embedder
should see prose rather than an angle-bracket separator.
"""
from __future__ import annotations

import re
from typing import Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import is_acceptable_phrase, normalise_phrase
from ..types import Candidate
from .base import BaseGenerator

PATH_SEPARATOR = " > "

# Headings that name a document part rather than a concept. Keeping them would
# pollute every document's vocabulary with the same handful of terms.
BOILERPLATE = frozenset({
    "introduction", "abstract", "conclusion", "conclusions", "references",
    "bibliography", "acknowledgements", "acknowledgments", "appendix",
    "appendices", "related work", "background", "discussion", "summary",
    "future work", "outline", "overview", "notation", "preliminaries",
    "contents", "table of contents", "index", "glossary", "nomenclature",
})

_NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLCivxlc]+|[A-Za-z])[.)]?\s+")


def strip_numbering(title: str) -> str:
    """Drop a leading "3.2 " or "IV. " so the concept is the words alone."""
    prev = None
    out = title.strip()
    while out != prev:
        prev = out
        out = _NUMBERING.sub("", out).strip()
    return out


class HeadingGenerator(BaseGenerator):
    """Section and subsection titles, plus hierarchical path concepts."""

    source = "heading"

    def __init__(self, *, include_paths: bool = True,
                 max_path_depth: int = 3,
                 drop_boilerplate: bool = True) -> None:
        self.include_paths = include_paths
        self.max_path_depth = max_path_depth
        self.drop_boilerplate = drop_boilerplate

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()

        for section in doc.iter_sections_sorted():
            title = strip_numbering(section.title)
            title = normalise_phrase(title)
            if not title or not is_acceptable_phrase(title, max_tokens=8):
                continue
            if self.drop_boilerplate and title.lower() in BOILERPLATE:
                continue

            if title.lower() not in seen:
                seen.add(title.lower())
                out.append(Candidate(
                    name=title, source=self.source, kind="heading",
                    section_id=section.id,
                    metadata={"level": section.level, "section_title": section.title},
                ))

            if not self.include_paths:
                continue

            # Breadcrumb: only worth emitting when there is genuine nesting.
            path = [strip_numbering(t) for t in doc.section_path(section.id)]
            path = [normalise_phrase(p) for p in path if normalise_phrase(p)]
            if self.drop_boilerplate:
                # A boilerplate ancestor still provides useful scoping, but a
                # boilerplate leaf does not.
                path = [p for p in path if p.lower() not in BOILERPLATE] or path
            path = path[-self.max_path_depth:]
            if len(path) < 2:
                continue

            name = PATH_SEPARATOR.join(path)
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            out.append(Candidate(
                name=name,
                source=self.source,
                kind="heading_path",
                section_id=section.id,
                # The embedder sees prose, not the separator glyph.
                tau=", ".join(reversed(path)),
                metadata={"level": section.level, "path": path},
            ))

        return self._sorted(out)
