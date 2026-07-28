"""The `Document` — ConceptDrill's read-only view of a parsed document.

Two entry points:

  * `Document.from_generic(json)` — the simple schema described in the spec:
    sections, typed text blocks, a bibliography list.
  * `Document.from_docmodel(json)` — the Semantic Compiler's
    `model.docmodel.json` (see `docmodel.py`).

`load()` sniffs which one it is. Nothing here mutates the input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .types import BibEntry, Block, Section

# Block types that count as prose for noun-phrase / NER mining and for the
# coverage-style metrics. Equations and captions are projectable but are not
# prose, so they are excluded from linguistic mining.
#
# `bibitem` is deliberately absent: reference lists are projectable but mining
# them for noun phrases yields author surnames and venue names, not concepts.
# The bibliography generator reads them through `Document.bibliography` instead.
PROSE_TYPES = frozenset({
    "paragraph", "abstract", "text", "definition", "theorem", "proof",
    "footnote", "sidenote", "listitem", "remark", "example",
})

MATH_TYPES = frozenset({"equation", "formula", "displayequation", "inlinemath"})
CODE_TYPES = frozenset({"code", "sourcecode", "algorithm", "listing"})


@dataclass
class Document:
    """A document reduced to what concept extraction needs."""

    blocks: list[Block] = field(default_factory=list)
    sections: dict[str, Section] = field(default_factory=dict)
    bibliography: list[BibEntry] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    # ---- access helpers -------------------------------------------------

    def blocks_of_type(self, *types: str) -> list[Block]:
        wanted = {t.lower() for t in types}
        return [b for b in self.blocks if b.type.lower() in wanted]

    @property
    def prose_blocks(self) -> list[Block]:
        """Non-empty prose, in document order. The unit of `coverage`."""
        return [b for b in self.blocks
                if b.type.lower() in PROSE_TYPES and not b.is_empty]

    @property
    def math_blocks(self) -> list[Block]:
        return [b for b in self.blocks
                if b.type.lower() in MATH_TYPES and not b.is_empty]

    @property
    def code_blocks(self) -> list[Block]:
        """Empty for Semantic Compiler input today — the DocModel emits no code
        objects. Kept because the routing table registers CodeBERT for them."""
        return [b for b in self.blocks
                if b.type.lower() in CODE_TYPES and not b.is_empty]

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.prose_blocks)

    def section(self, section_id: Optional[str]) -> Optional[Section]:
        return self.sections.get(section_id) if section_id else None

    def top_level_section(self, section_id: Optional[str]) -> Optional[str]:
        """Walk up to the outermost ancestor. `purity` groups by this."""
        seen: set[str] = set()
        cur = section_id
        while cur and cur in self.sections and cur not in seen:
            seen.add(cur)
            parent = self.sections[cur].parent_id
            if not parent or parent not in self.sections:
                return cur
            cur = parent
        return cur

    def section_path(self, section_id: Optional[str]) -> list[str]:
        """Titles from the root down to `section_id`. Builds the "Method >
        Semantic Projection" path concepts."""
        path: list[str] = []
        seen: set[str] = set()
        cur = section_id
        while cur and cur in self.sections and cur not in seen:
            seen.add(cur)
            sec = self.sections[cur]
            path.append(sec.title)
            cur = sec.parent_id
        return list(reversed(path))

    def iter_sections_sorted(self) -> Iterator[Section]:
        """Deterministic section iteration — by (level, id), never dict order."""
        for sec in sorted(self.sections.values(), key=lambda s: (s.level, s.id)):
            yield sec

    # ---- constructors ---------------------------------------------------

    @classmethod
    def from_generic(cls, data: dict[str, Any],
                     source_path: Optional[str] = None) -> "Document":
        """Parse the generic schema.

        Expected (all keys optional)::

            {"meta": {...},
             "sections": [{"id","title","level","parent"|"parent_id","children"}],
             "blocks":   [{"id","type","text","section"|"section_id","props"}],
             "bibliography": [{"id","title","label","year","citations","keywords"}]}
        """
        sections: dict[str, Section] = {}
        raw_sections = data.get("sections") or []
        for i, s in enumerate(raw_sections):
            sid = str(s.get("id") or f"sec_{i}")
            sections[sid] = Section(
                id=sid,
                title=str(s.get("title") or s.get("heading") or "").strip(),
                level=int(s.get("level", 1) or 1),
                parent_id=(str(s["parent_id"]) if s.get("parent_id") else
                           str(s["parent"]) if s.get("parent") else None),
                children=tuple(str(c) for c in (s.get("children") or ())),
            )

        blocks: list[Block] = []
        raw_blocks = data.get("blocks") or data.get("text_blocks") or []
        for i, b in enumerate(raw_blocks):
            blocks.append(Block(
                id=str(b.get("id") or f"blk_{i}"),
                type=str(b.get("type") or "paragraph"),
                text=str(b.get("text") or b.get("content") or ""),
                section_id=(str(b["section_id"]) if b.get("section_id") else
                            str(b["section"]) if b.get("section") else None),
                props={k: v for k, v in b.items()
                       if k not in {"id", "type", "text", "content",
                                    "section", "section_id"}},
            ))

        bib: list[BibEntry] = []
        raw_bib = data.get("bibliography") or data.get("references") or []
        for i, e in enumerate(raw_bib):
            if isinstance(e, str):
                bib.append(BibEntry(id=f"bib_{i}", title=e))
                continue
            year = e.get("year")
            bib.append(BibEntry(
                id=str(e.get("id") or e.get("citekey") or f"bib_{i}"),
                title=str(e.get("title") or "").strip(),
                label=str(e.get("label") or e.get("citekey") or ""),
                year=int(year) if isinstance(year, (int, str)) and str(year).isdigit() else None,
                citations=(int(e["citations"])
                           if str(e.get("citations", "")).isdigit() else None),
                keywords=tuple(str(k) for k in (e.get("keywords") or ())),
                props={k: v for k, v in e.items()
                       if k not in {"id", "citekey", "title", "label", "year",
                                    "citations", "keywords"}},
            ))

        return cls(blocks=blocks, sections=sections, bibliography=bib,
                   meta=dict(data.get("meta") or {}), source_path=source_path)

    @classmethod
    def from_docmodel(cls, data: dict[str, Any],
                      source_path: Optional[str] = None) -> "Document":
        from .docmodel import docmodel_to_document
        return docmodel_to_document(data, source_path=source_path)

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        """Load from disk, sniffing the schema.

        The Semantic Compiler's DocModel is recognised by its `objects` list of
        `{id, type, props, realizations}` records.
        """
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a JSON object at the top level")
        if is_docmodel(data):
            return cls.from_docmodel(data, source_path=str(p))
        return cls.from_generic(data, source_path=str(p))


def is_docmodel(data: dict[str, Any]) -> bool:
    """True for a Semantic Compiler `model.docmodel.json`."""
    objs = data.get("objects")
    if not isinstance(objs, list) or not objs:
        return False
    first = objs[0]
    return isinstance(first, dict) and "props" in first and "type" in first
