"""Build a section tree from a Semantic Compiler `model.docmodel.json`.

Verified against `~/pdfdrill-library/2209.00445/`. Three properties of the
DocModel drive this module, and each cost a debugging session to find:

  * **Section titles live under `props.caption`**, not `props.title`.
  * **`parent` is `null` on every Section.** The hierarchy is implied by
    `level` + `flow_index` and must be reconstructed.
  * **Captions carry unresolved LaTeX macros**, cleaned by `captions.py`.

Levels start at **2** in real documents, not 1, so nothing here may assume the
root level is 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .captions import clean_caption, lost_macros

SECTION_TYPES = frozenset({"section", "heading"})


@dataclass(frozen=True)
class RawSection:
    """One Section object, cleaned but not yet linked into a tree."""
    id: str
    title: str
    title_raw: str
    level: int
    flow_index: int
    is_appendix: bool = False
    #: Macro names dropped by cleaning. Non-empty means `title` lost meaning
    #: the author had encoded (see `captions.lost_macros`).
    lost_macros: tuple[str, ...] = ()

    @property
    def title_is_degraded(self) -> bool:
        """True when cleaning cost the title information.

        `\\ALG\\ Application` -> `Application` is the real case: the title alone
        no longer identifies the section, so a summariser should be given the
        raw form as well.
        """
        return bool(self.lost_macros)


def read_sections(objects: Sequence[dict[str, Any]]) -> list[RawSection]:
    """Extract Section records, ordered by `flow_index`.

    Objects without a usable id are skipped — an unidentifiable section cannot
    be linked to paragraphs or referenced by a basis vector.

    `flow_index` is the DocModel's document-order key. When absent it falls
    back to the object's position in the input, so ordering stays deterministic
    rather than becoming dict-order dependent.
    """
    out: list[RawSection] = []
    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "").lower() not in SECTION_TYPES:
            continue
        oid = str(obj.get("id") or "").strip()
        if not oid:
            continue

        props = obj.get("props") or {}
        if not isinstance(props, dict):
            props = {}

        raw_title = str(props.get("caption") or props.get("title") or "").strip()
        title = clean_caption(raw_title)

        try:
            level = int(props.get("level", 1) or 1)
        except (TypeError, ValueError):
            level = 1
        try:
            flow = int(props.get("flow_index", position))
        except (TypeError, ValueError):
            flow = position

        out.append(RawSection(
            id=oid,
            title=title,
            title_raw=raw_title,
            level=level,
            flow_index=flow,
            is_appendix=bool(props.get("is_appendix", False)),
            lost_macros=lost_macros(raw_title, title),
        ))

    # Ties on flow_index break on id, never on input order.
    out.sort(key=lambda s: (s.flow_index, s.id))
    return out


def link_parents(sections: Sequence[RawSection]) -> dict[str, Optional[str]]:
    """Reconstruct parent links from `level` + document order.

    The DocModel leaves `parent` null on every Section, so the tree is implied
    rather than stored. Walking sections in `flow_index` order, a section's
    parent is the nearest preceding section of *strictly lower* level — the
    standard interpretation of a numbered outline.

    Two properties this must respect:

    * **Level jumps are legal.** A document may go L2 -> L4 with no L3. The L4
      then hangs off the L2 rather than becoming a root, because that is where
      it sits in the printed document.
    * **Appendix and body are separate trees.** An appendix subsection must not
      hang off the last body section merely because it follows it. In the
      reference document every appendix entry is L2 so this never fires, but a
      document with appendix subsections would otherwise graft its whole
      appendix onto "Ethics Statement".

    Returns `{section_id: parent_id or None}` for every input section.
    """
    parents: dict[str, Optional[str]] = {}
    stack: list[RawSection] = []

    for sec in sections:
        # Pop until the top is a legal parent: lower level, same tree.
        while stack and (stack[-1].level >= sec.level
                         or stack[-1].is_appendix != sec.is_appendix):
            stack.pop()
        parents[sec.id] = stack[-1].id if stack else None
        stack.append(sec)

    return parents


#: Object types whose text belongs to the owning section's body.
BODY_TYPES = frozenset({"paragraph", "abstract", "listitem"})

#: A paragraph that is nothing but LaTeX commands. The reference document has
#: one: a Paragraph object whose entire text is `\maketitle`. It carries no
#: content, and feeding it to a summariser or an embedder is pure noise.
_LATEX_ONLY = re.compile(r"^(?:\s*\\[A-Za-z@]+\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})*)+\s*$")


def is_latex_artifact(text: str) -> bool:
    """True when the text is only LaTeX commands, with no prose to speak of.

    Conservative by design: a paragraph containing commands *and* words is
    real content and is kept. Only a paragraph that is entirely markup is
    dropped.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if not _LATEX_ONLY.match(stripped):
        return False
    # Belt and braces: if removing the commands leaves real words, keep it.
    remainder = re.sub(r"\\[A-Za-z@]+|[{}\[\]$\\]", " ", stripped)
    return len(re.sub(r"[^A-Za-z]", "", remainder)) < 3


@dataclass(frozen=True)
class Paragraph:
    """A body text unit, already located within a section."""
    id: str
    text: str
    section_id: Optional[str]
    flow_index: int


def read_paragraphs(objects: Sequence[dict[str, Any]]) -> list[Paragraph]:
    """Body text units in document order, each with its owning section.

    `props.parent_section` holds the section id. In the reference document it
    is present on 84 of 85 paragraphs; the one without is `flow_index=1`,
    front matter that precedes the first section. That paragraph gets
    `section_id=None` rather than being dropped — see `attach_paragraphs`.
    """
    out: list[Paragraph] = []
    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "").lower() not in BODY_TYPES:
            continue
        props = obj.get("props") or {}
        if not isinstance(props, dict):
            continue
        text = str(props.get("text") or props.get("content") or "").strip()
        if not text or is_latex_artifact(text):
            continue
        parent = props.get("parent_section")
        try:
            flow = int(props.get("flow_index", position))
        except (TypeError, ValueError):
            flow = position
        out.append(Paragraph(
            id=str(obj.get("id") or f"para_{position}"),
            text=text,
            section_id=str(parent) if parent else None,
            flow_index=flow,
        ))
    out.sort(key=lambda p: (p.flow_index, p.id))
    return out


def attach_paragraphs(paragraphs: Sequence[Paragraph],
                      section_ids: Sequence[str],
                      ) -> tuple[dict[str, list[Paragraph]], list[Paragraph]]:
    """Group paragraphs by section. Returns `(by_section, orphans)`.

    A paragraph is an orphan when it has no `parent_section`, or names one that
    is not a known section. **Orphans are returned, never discarded** — front
    matter (title block, abstract lead) legitimately precedes the first section,
    and a run must be able to account for every paragraph in the document
    rather than quietly losing text before it is ever summarised.
    """
    known = set(section_ids)
    by_section: dict[str, list[Paragraph]] = {sid: [] for sid in section_ids}
    orphans: list[Paragraph] = []

    for para in paragraphs:
        if para.section_id in known:
            by_section[para.section_id].append(para)
        else:
            orphans.append(para)

    return by_section, orphans


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionNode:
    """A section, linked into the tree and holding its own body text."""
    id: str
    title: str
    title_raw: str
    level: int
    flow_index: int
    is_appendix: bool
    lost_macros: tuple[str, ...]
    parent_id: Optional[str]
    children: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]

    @property
    def title_is_degraded(self) -> bool:
        return bool(self.lost_macros)

    @property
    def body_text(self) -> str:
        """This section's own paragraphs, excluding its subsections."""
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def summarizer_title(self) -> str:
        """Title to hand a summariser.

        When cleaning dropped a macro the cleaned title can be meaningless on
        its own — `\\ALG\\ Application` becomes `Application` — so the raw form
        is appended to restore the author's intent.
        """
        if self.title_is_degraded and self.title_raw:
            return f"{self.title} ({self.title_raw})"
        return self.title


@dataclass
class SectionTree:
    """The section hierarchy of one document, with its body text attached."""
    nodes: dict[str, SectionNode] = field(default_factory=dict)
    roots: tuple[str, ...] = ()
    #: Paragraphs belonging to no known section — front matter, usually.
    orphans: tuple[Paragraph, ...] = ()
    bibkey: str = ""
    source_path: Optional[str] = None

    def __len__(self) -> int:
        return len(self.nodes)

    def by_level(self, level: int) -> list[SectionNode]:
        """Nodes at one level, in document order."""
        return [n for n in self.iter_document_order() if n.level == level]

    @property
    def levels(self) -> list[int]:
        return sorted({n.level for n in self.nodes.values()})

    def iter_document_order(self):
        """Every node, ordered by `flow_index` — the printed reading order."""
        for node in sorted(self.nodes.values(),
                           key=lambda n: (n.flow_index, n.id)):
            yield node

    def depth(self, node_id: str) -> int:
        """Edges from this node up to its root."""
        depth, seen, cur = 0, {node_id}, self.nodes[node_id].parent_id
        while cur is not None and cur in self.nodes:
            if cur in seen:                       # malformed input; do not hang
                break
            seen.add(cur)
            depth += 1
            cur = self.nodes[cur].parent_id
        return depth

    def descendants(self, node_id: str) -> list[SectionNode]:
        """All nodes beneath this one, in document order."""
        out, stack = [], list(self.nodes[node_id].children)
        seen = set()
        while stack:
            cid = stack.pop()
            if cid in seen or cid not in self.nodes:
                continue
            seen.add(cid)
            out.append(self.nodes[cid])
            stack.extend(self.nodes[cid].children)
        return sorted(out, key=lambda n: (n.flow_index, n.id))

    def subtree_text(self, node_id: str) -> str:
        """This section's text *including* its subsections.

        This is what a level-2 summary needs: summarising "Empirical
        Evaluation" from its own two paragraphs, while ignoring the three
        subsections beneath it, would describe almost nothing.
        """
        node = self.nodes[node_id]
        parts = [node.body_text]
        parts += [d.body_text for d in self.descendants(node_id)]
        return "\n\n".join(p for p in parts if p)

    def stats(self) -> dict[str, Any]:
        counts: dict[int, int] = {}
        for node in self.nodes.values():
            counts[node.level] = counts.get(node.level, 0) + 1
        return {
            "sections": len(self.nodes),
            "roots": len(self.roots),
            "levels": dict(sorted(counts.items())),
            "appendix_sections": sum(1 for n in self.nodes.values() if n.is_appendix),
            "paragraphs": sum(len(n.paragraphs) for n in self.nodes.values()),
            "orphan_paragraphs": len(self.orphans),
            "degraded_titles": sum(1 for n in self.nodes.values()
                                   if n.title_is_degraded),
        }


def build_tree(docmodel: dict[str, Any],
               source_path: Optional[str] = None) -> SectionTree:
    """Assemble a `SectionTree` from a parsed `model.docmodel.json`.

    Composes the four verified units: read sections, link parents, read
    paragraphs, attach them. Adds only the child lists, which are the inverse
    of the parent map.
    """
    objects = docmodel.get("objects") or []
    if not isinstance(objects, list):
        objects = []

    sections = read_sections(objects)
    parents = link_parents(sections)
    paragraphs = read_paragraphs(objects)
    by_section, orphans = attach_paragraphs(paragraphs, [s.id for s in sections])

    # Children are the inverse of the parent map, kept in document order.
    children: dict[str, list[str]] = {s.id: [] for s in sections}
    for sec in sections:
        parent = parents.get(sec.id)
        if parent in children:
            children[parent].append(sec.id)

    nodes = {
        sec.id: SectionNode(
            id=sec.id, title=sec.title, title_raw=sec.title_raw,
            level=sec.level, flow_index=sec.flow_index,
            is_appendix=sec.is_appendix, lost_macros=sec.lost_macros,
            parent_id=parents.get(sec.id),
            children=tuple(children[sec.id]),
            paragraphs=tuple(by_section.get(sec.id, ())),
        )
        for sec in sections
    }

    meta = docmodel.get("meta") or {}
    return SectionTree(
        nodes=nodes,
        roots=tuple(s.id for s in sections if parents.get(s.id) is None),
        orphans=tuple(orphans),
        bibkey=str(meta.get("bibkey") or "") if isinstance(meta, dict) else "",
        source_path=source_path,
    )


def load_tree(path) -> SectionTree:
    """Read a `model.docmodel.json` from disk. The file is opened read-only."""
    import json
    from pathlib import Path

    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        return build_tree(json.load(fh), source_path=str(p))
