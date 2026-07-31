"""Build a marker tree from a Semantic Compiler `model.docmodel.json`.

Verified against `~/pdfdrill-library/2209.00445/`. Three properties of the
DocModel drive this module, and each cost a debugging session to find:

  * **Span titles live under `props.caption`**, not `props.title`.
  * **`parent` is `null` on every Span.** The hierarchy is implied by
    `level` + `flow_index` and must be reconstructed.
  * **Captions carry unresolved LaTeX macros**, cleaned by `captions.py`.

Levels start at **2** in real documents, not 1, so nothing here may assume the
root level is 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .captions import clean_body_text, clean_caption, lost_macros
from .mathtext import math_text

MARKER_TYPES = frozenset({"section", "heading"})


@dataclass(frozen=True)
class SectionMarker:
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
        no longer identifies the span, so a summariser should be given the
        raw form as well.
        """
        return bool(self.lost_macros)


def read_markers(objects: Sequence[dict[str, Any]]) -> list[SectionMarker]:
    """Extract Span records, ordered by `flow_index`.

    Objects without a usable id are skipped — an unidentifiable span cannot
    be linked to paragraphs or referenced by a basis vector.

    `flow_index` is the DocModel's document-order key. When absent it falls
    back to the object's position in the input, so ordering stays deterministic
    rather than becoming dict-order dependent.
    """
    out: list[SectionMarker] = []
    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "").lower() not in MARKER_TYPES:
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

        out.append(SectionMarker(
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


def link_parents(markers: Sequence[SectionMarker]) -> dict[str, Optional[str]]:
    """Reconstruct parent links from `level` + document order.

    The DocModel leaves `parent` null on every Span, so the tree is implied
    rather than stored. Walking markers in `flow_index` order, a span's
    parent is the nearest preceding span of *strictly lower* level — the
    standard interpretation of a numbered outline.

    Two properties this must respect:

    * **Level jumps are legal.** A document may go L2 -> L4 with no L3. The L4
      then hangs off the L2 rather than becoming a root, because that is where
      it sits in the printed document.
    * **Appendix and body are separate trees.** An appendix subsection must not
      hang off the last body span merely because it follows it. In the
      reference document every appendix entry is L2 so this never fires, but a
      document with appendix subsections would otherwise graft its whole
      appendix onto "Ethics Statement".

    Returns `{marker_id: parent_id or None}` for every input span.
    """
    parents: dict[str, Optional[str]] = {}
    stack: list[SectionMarker] = []

    for sec in markers:
        # Pop until the top is a legal parent: lower level, same tree.
        while stack and (stack[-1].level >= sec.level
                         or stack[-1].is_appendix != sec.is_appendix):
            stack.pop()
        parents[sec.id] = stack[-1].id if stack else None
        stack.append(sec)

    return parents


#: Object type -> the props that may carry its prose, in preference order.
#:
#: MEASURED. `BODY_TYPES` was `{paragraph, abstract, listitem}`, and five types
#: carrying text under the DocModel mapping were consulted by nothing at all.
#: Across the 10-document set that silently discarded 80 Sidenotes holding
#: 148,772 characters, 61 Diagrams, 33 Tables, 15 Footnotes and 7 Pictures. In
#: `0864` alone, 34 of 208 content objects and 25,156 characters never reached
#: any summariser -- one span, `6 Conclusion`, lost 83% of its text to a
#: single unread Sidenote.
#:
#: `Picture` and `Diagram` read `caption` ONLY. Their `url`, `cdn_url` and
#: `image_id` props hold mathpix CDN links, which are longer than the caption
#: and are pure noise in an embedding.
BODY_PROPS: dict[str, tuple[str, ...]] = {
    "paragraph": ("text", "content"),
    "listitem": ("content", "text"),
    "sidenote": ("content",),
    "footnote": ("content",),
    "picture": ("caption",),
    "diagram": ("caption",),
}

#: Object types whose text belongs to the owning span's body.
BODY_TYPES = frozenset(BODY_PROPS)

#: Types deliberately not read, each with the reason. Recorded per object, not
#: dropped: a run must be able to account for every object in its input.
#:
#: `Table` is the interesting one. Its `raw_text` is tab-separated numbers --
#: "NaiveBayes\nJ48\nsurface (baseline)\n56.30\n42.20" -- and the summariser
#: prompt explicitly excludes numerical results. Feeding it in would add digits
#: to concept labels, not concepts.
SKIPPED_TYPES: dict[str, str] = {
    # Deferred by decision, not by oversight. An abstract is likely the most
    # valuable span in a document for concept extraction, but a typical one
    # states what was done rather than what the solution is, so it needs its
    # own prompt. Reading it under the span prompt would put process
    # description into the basis.
    "abstract": "deferred: needs its own prompt, not the span prompt",
    # An Algorithm plausibly opens a unit and its steps are its content. That
    # is a boundary rule, and boundary rules are decided, not inferred.
    "algorithm": "deferred: boundary semantics not yet decided",
    "algorithmstep": "deferred: boundary semantics not yet decided",
    "table": "tabular data: numeric results, which the concept prompt excludes",
    "tablecell": "fragment of a Table, not an independent content object",
    "tablerow": "fragment of a Table, not an independent content object",
    "citation": "reference marker; the citekey is already inlined by clean_body_text",
    "reference": "bibliography entry: author surnames outrank concepts",
    "page": "page image, no text",
    "document": "the document object itself carries no body text",
    # Surfaced by the skip ledger itself: 40 objects across two documents,
    # every one a spacing or layout command (\smallskip, \noindent). They
    # were invisible before, which is the point of recording unknowns.
    "ltxcommand": "a LaTeX spacing or layout command, not content",
}

#: A reference-list entry opener: a citation marker followed immediately by a
#: capitalised author name. `[Alcala-Fdez et al., 2011] Jesus Alcala-Fdez` is an
#: entry; `model [7, 10, 16, 18] or inform` is prose that cites one. The capital
#: is the whole discriminator, and it separates cleanly: across the 10-document
#: set, marker *density* does not -- real prose reaches 8.6 markers per 1000
#: characters, higher than some genuine reference lists.
_REFERENCE_ENTRY = re.compile(r"\[[^\]\n]{1,80}\]\s+[A-Z]")

#: Longest gap between consecutive entries still counted as one run. Reference
#: entries are 100-300 characters; 600 tolerates a long one without swallowing
#: a paragraph of prose that happens to sit between two citations.
_REFERENCE_GAP = 600

#: Entries needed before a run is called a reference list at all.
_REFERENCE_MIN_ENTRIES = 3


def reference_tail(text: str) -> int:
    """Offset where a trailing reference list begins, or `len(text)`.

    Drill output merges a span's closing prose with the bibliography that
    follows it: one Sidenote in `03-NTCIR11` holds the paper's conclusion and
    then eight numbered references. Dropping the object loses the conclusion;
    keeping it puts author surnames into a concept label. Cutting at the start
    of the trailing run keeps the prose and drops the list.
    """
    starts = [m.start() for m in _REFERENCE_ENTRY.finditer(text or "")]
    if len(starts) < _REFERENCE_MIN_ENTRIES:
        return len(text or "")
    # Walk back from the end while entries stay closely spaced.
    cut = starts[-1]
    run = 1
    for earlier, later in zip(reversed(starts[:-1]), reversed(starts[1:])):
        if later - earlier > _REFERENCE_GAP:
            break
        cut, run = earlier, run + 1
    return cut if run >= _REFERENCE_MIN_ENTRIES else len(text or "")


#: A caption that is only a URL. `Picture.caption` sometimes holds the CDN link
#: rather than a caption, and a URL embeds as noise.
_URL_ONLY = re.compile(r"^\s*(?:!\[\]\()?https?://\S+\)?\s*$")

#: Math objects. Their only content is LaTeX, which embeds as noise, so they
#: are rendered to prose by `mathtext` first. Excluding them entirely -- as this
#: module originally did -- dropped 74 of the reference paper's objects, and
#: with them the whole of its mathematics, from every summary.
MATH_TYPES = frozenset({"formula", "equation", "displayequation", "inlinemath"})

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
    """A body text unit, already located within a span."""
    id: str
    text: str
    marker_id: Optional[str]
    flow_index: int


@dataclass(frozen=True)
class SkippedObject:
    """An object the tree reader did not turn into body text, and why.

    The single-document path has recorded skips since the beginning
    (`docmodel.SkippedObject`, enforced by `test_every_object_is_accounted_for`).
    The hierarchy path did not, which is how five content types went unread
    through four steps without anything noticing.
    """
    object_id: str
    object_type: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"object_id": self.object_id, "object_type": self.object_type,
                "reason": self.reason}


def read_math(objects: Sequence[dict[str, Any]], *,
              speaker=None, min_latex_chars: int = 3
              ) -> tuple[list[Paragraph], dict[str, int]]:
    """Math objects rendered to prose, plus a tally of where the text came from.

    The tally matters: it says whether a run used the docmodel's own spoken
    field, a speech engine, or the coarse fallback, so basis quality is never
    a mystery after the fact.
    """
    out: list[Paragraph] = []
    sources: dict[str, int] = {}
    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "").lower() not in MATH_TYPES:
            continue
        props = obj.get("props") or {}
        if not isinstance(props, dict):
            continue
        text, source = math_text(props, speaker=speaker,
                                 min_latex_chars=min_latex_chars)
        sources[source] = sources.get(source, 0) + 1
        if not text:
            continue
        parent = props.get("parent_section")
        try:
            flow = int(props.get("flow_index", position))
        except (TypeError, ValueError):
            flow = position
        out.append(Paragraph(id=str(obj.get("id") or f"math_{position}"),
                             text=text,
                             marker_id=str(parent) if parent else None,
                             flow_index=flow))
    out.sort(key=lambda p: (p.flow_index, p.id))
    return out, sources


def read_content(objects: Sequence[dict[str, Any]]
                 ) -> tuple[list[Paragraph], list[SkippedObject]]:
    """Body text units in document order, plus a reason for everything else.

    `props.parent_section` holds the span id. In the reference document it
    is present on 84 of 85 paragraphs; the one without is `flow_index=1`,
    front matter that precedes the first span. That paragraph gets
    `marker_id=None` rather than being dropped — see `attach_paragraphs`.

    Every object that does not become a `Paragraph` is returned as a
    `SkippedObject` with a reason. Math objects are not skipped here: they go
    through `read_math`, which renders them to prose, and `build_tree` records
    only the ones that produced nothing.
    """
    out: list[Paragraph] = []
    skipped: list[SkippedObject] = []

    def skip(obj: Any, position: int, reason: str) -> None:
        source = obj if isinstance(obj, dict) else {}
        skipped.append(SkippedObject(
            str(source.get("id") or f"object_{position}"),
            str(source.get("type") or ""), reason))

    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            skip(obj, position, "not an object")
            continue
        otype = str(obj.get("type") or "").lower()

        if otype in MARKER_TYPES:
            continue                      # a marker, accounted for as a span
        if otype in MATH_TYPES:
            continue                      # handled by read_math
        if otype in SKIPPED_TYPES:
            skip(obj, position, SKIPPED_TYPES[otype])
            continue
        if otype not in BODY_PROPS:
            # An unknown type is a reason to look, not a reason to be silent.
            skip(obj, position, f"unhandled object type {otype!r}")
            continue

        props = obj.get("props") or {}
        if not isinstance(props, dict):
            skip(obj, position, "props is not an object")
            continue

        raw_text = ""
        for prop in BODY_PROPS[otype]:
            value = str(props.get(prop) or "").strip()
            if value:
                raw_text = value
                break
        if not raw_text:
            skip(obj, position, f"no text under {'/'.join(BODY_PROPS[otype])}")
            continue
        if _URL_ONLY.match(raw_text):
            skip(obj, position, "caption is only a URL")
            continue

        # Drill output merges a span's closing prose with the bibliography
        # that follows it. Cut the trailing entry run, keep the prose.
        cut = reference_tail(raw_text)
        if cut < len(raw_text):
            raw_text = raw_text[:cut].strip()
            if not raw_text:
                skip(obj, position,
                     "reference list: no prose before the entries")
                continue

        if is_latex_artifact(raw_text):
            skip(obj, position, "text is only LaTeX markup")
            continue

        # Strip DocModel placeholders before the text reaches a summariser or
        # an embedder; 55% of paragraphs in the reference document carry them.
        text = clean_body_text(raw_text)
        if not text:
            skip(obj, position, "placeholder cleaning left no text")
            continue
        parent = props.get("parent_section")
        try:
            flow = int(props.get("flow_index", position))
        except (TypeError, ValueError):
            flow = position
        out.append(Paragraph(
            id=str(obj.get("id") or f"para_{position}"),
            text=text,
            marker_id=str(parent) if parent else None,
            flow_index=flow,
        ))
    out.sort(key=lambda p: (p.flow_index, p.id))
    return out, skipped


def read_paragraphs(objects: Sequence[dict[str, Any]]) -> list[Paragraph]:
    """`read_content` without the skip ledger. See it for the contract."""
    return read_content(objects)[0]


def assign_by_flow(units: Sequence[Paragraph],
                   markers: Sequence[SectionMarker]) -> list[Paragraph]:
    """Give units without an explicit owner the span they fall under.

    `Formula` and `Equation` objects carry NO `parent_section` -- zero of the
    reference paper's 74 do -- but all carry `flow_index`. Position is the
    relationship: a formula belongs to the span it is printed under, which is
    the last span whose `flow_index` precedes it.

    Units that already name a parent keep it: an explicit link always beats an
    inferred one. Units appearing before the first span stay unowned, which
    is correct -- they are front matter.
    """
    ordered = sorted(markers, key=lambda s: (s.flow_index, s.id))
    if not ordered:
        return list(units)

    boundaries = [s.flow_index for s in ordered]
    out: list[Paragraph] = []
    for unit in units:
        if unit.marker_id:
            out.append(unit)
            continue
        import bisect
        idx = bisect.bisect_right(boundaries, unit.flow_index) - 1
        owner = ordered[idx].id if idx >= 0 else None
        out.append(Paragraph(id=unit.id, text=unit.text,
                             marker_id=owner, flow_index=unit.flow_index))
    return out


def assign_by_extent(units: Sequence[Paragraph],
                     markers: Sequence[SectionMarker],
                     ) -> tuple[dict[str, list[Paragraph]], list[Paragraph],
                                list[str]]:
    r"""Group content into spans by flow position. `(by_marker, preamble, moved)`.

    THE RULE. A span is the content between one marker and the immediately
    following marker **at any level**. So a unit belongs to the nearest marker
    that precedes it in `flow_index`, and units before the first marker belong
    to no span at all.

    Position decides, not `props.parent_section`. MEASURED across the arXiv
    corpus: the two agree on 4249 of 4253 units, 99.91%, and all four
    disagreements are units whose `parent_section` was missing or unknown, so
    attachment made them orphans while position places them correctly. `moved`
    names those, because a rule that silently rescues four units should say so.

    This also removes the special case for maths. `Formula` and `Equation`
    carry no `parent_section` and previously needed `assign_by_flow`; under a
    positional rule they are ordinary content and need no exception.
    """
    ordered = sorted(markers, key=lambda m: (m.flow_index, m.id))
    by_marker: dict[str, list[Paragraph]] = {m.id: [] for m in ordered}
    preamble: list[Paragraph] = []
    moved: list[str] = []

    for unit in sorted(units, key=lambda u: (u.flow_index, u.id)):
        owner = None
        for marker in ordered:
            if marker.flow_index < unit.flow_index:
                owner = marker.id
            else:
                break
        if owner is None:
            preamble.append(unit)
            continue
        if unit.marker_id and unit.marker_id != owner:
            # A genuine disagreement: the docmodel named a different owner.
            # Units with no `parent_section` at all -- every Formula -- are
            # positional by design and are not a disagreement.
            moved.append(unit.id)
        by_marker[owner].append(unit)

    return by_marker, preamble, moved


def attach_paragraphs(paragraphs: Sequence[Paragraph],
                      marker_ids: Sequence[str],
                      ) -> tuple[dict[str, list[Paragraph]], list[Paragraph]]:
    """Group paragraphs by span. Returns `(by_marker, orphans)`.

    A paragraph is an orphan when it has no `parent_section`, or names one that
    is not a known span. **Orphans are returned, never discarded** — front
    matter (title block, abstract lead) legitimately precedes the first span,
    and a run must be able to account for every paragraph in the document
    rather than quietly losing text before it is ever summarised.
    """
    known = set(marker_ids)
    by_marker: dict[str, list[Paragraph]] = {sid: [] for sid in marker_ids}
    orphans: list[Paragraph] = []

    for para in paragraphs:
        if para.marker_id in known:
            by_marker[para.marker_id].append(para)
        else:
            orphans.append(para)

    return by_marker, orphans


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkerNode:
    """A span, linked into the tree and holding its own body text."""
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
        """This span's own paragraphs, excluding its subsections."""
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
class MarkerTree:
    """The marker hierarchy of one document, with its body text attached."""
    nodes: dict[str, MarkerNode] = field(default_factory=dict)
    roots: tuple[str, ...] = ()
    #: Content before the first marker. It belongs to no span: in this corpus
    #: it is title-block LaTeX -- \author, \affiliation, CCSXML -- 78 units
    #: and 6,610 characters across 8 documents, none of it content. Assigned to
    #: the structural sink, which is what Step 4 exists for.
    orphans: tuple[Paragraph, ...] = ()
    #: Units the docmodel assigned to a different marker than position does.
    #: Recorded because a rule that overrides the docmodel should say so.
    moved_by_position: tuple[str, ...] = ()
    bibkey: str = ""
    source_path: Optional[str] = None
    #: How each math object's text was obtained: docmodel | speech | fallback | none.
    math_sources: dict[str, int] = field(default_factory=dict)
    #: Every object that did not become body text, with a reason. A run must be
    #: able to account for its whole input; see `accounting`.
    skipped: tuple[SkippedObject, ...] = ()

    def __len__(self) -> int:
        return len(self.nodes)

    def accounting(self, objects: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Does this tree account for every object in the docmodel?

        The conservation law, computed rather than asserted: markers, attached
        body units, orphans and recorded skips should partition the input with
        no object counted twice and none counted zero times.
        """
        all_ids = [str(o.get("id") or "") for o in objects if isinstance(o, dict)]
        markers = {n.id for n in self.nodes.values()}
        attached = {p.id for n in self.nodes.values() for p in n.paragraphs}
        orphans = {p.id for p in self.orphans}
        skipped = {s.object_id for s in self.skipped}

        seen: dict[str, int] = {}
        for group in (markers, attached, orphans, skipped):
            for oid in group:
                seen[oid] = seen.get(oid, 0) + 1

        return {
            "objects": len(all_ids),
            "markers": len(markers),
            "attached": len(attached),
            "orphans": len(orphans),
            "skipped": len(skipped),
            "unaccounted": sorted(set(all_ids) - set(seen)),
            "double_counted": sorted(k for k, n in seen.items() if n > 1),
            "accounted_for": len(set(all_ids) & set(seen)),
        }

    def by_level(self, level: int) -> list[MarkerNode]:
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

    def descendants(self, node_id: str) -> list[MarkerNode]:
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
        """This span's text *including* its subsections.

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
            "markers": len(self.nodes),
            "roots": len(self.roots),
            "levels": dict(sorted(counts.items())),
            "appendix_markers": sum(1 for n in self.nodes.values() if n.is_appendix),
            "paragraphs": sum(len(n.paragraphs) for n in self.nodes.values()),
            "preamble_units": len(self.orphans),
            "moved_by_position": len(self.moved_by_position),
            "degraded_titles": sum(1 for n in self.nodes.values()
                                   if n.title_is_degraded),
            "math_sources": dict(sorted(self.math_sources.items())),
        }


def build_tree(docmodel: dict[str, Any],
               source_path: Optional[str] = None, *,
               include_math: bool = True,
               speaker=None) -> MarkerTree:
    """Assemble a `MarkerTree` from a parsed `model.docmodel.json`.

    Composes the four verified units: read markers, link parents, read
    paragraphs, attach them. Adds only the child lists, which are the inverse
    of the parent map.
    """
    objects = docmodel.get("objects") or []
    if not isinstance(objects, list):
        objects = []

    markers = read_markers(objects)
    parents = link_parents(markers)
    paragraphs, skipped = read_content(objects)

    math_sources: dict[str, int] = {}
    if include_math:
        math_units, math_sources = read_math(objects, speaker=speaker)
        rendered = {u.id for u in math_units}
        skipped.extend(
            SkippedObject(str(o.get("id") or ""), str(o.get("type") or ""),
                          "math object rendered to no prose")
            for o in objects
            if isinstance(o, dict)
            and str(o.get("type") or "").lower() in MATH_TYPES
            and str(o.get("id") or "") not in rendered)
        # No special case: under the positional rule a formula is ordinary
        # content and finds its span the same way a paragraph does.
        paragraphs = sorted(paragraphs + math_units,
                            key=lambda p: (p.flow_index, p.id))
    else:
        skipped.extend(
            SkippedObject(str(o.get("id") or ""), str(o.get("type") or ""),
                          "math objects excluded from this build")
            for o in objects
            if isinstance(o, dict)
            and str(o.get("type") or "").lower() in MATH_TYPES)

    by_marker, preamble, moved = assign_by_extent(paragraphs, markers)

    # Children are the inverse of the parent map, kept in document order.
    children: dict[str, list[str]] = {s.id: [] for s in markers}
    for sec in markers:
        parent = parents.get(sec.id)
        if parent in children:
            children[parent].append(sec.id)

    nodes = {
        sec.id: MarkerNode(
            id=sec.id, title=sec.title, title_raw=sec.title_raw,
            level=sec.level, flow_index=sec.flow_index,
            is_appendix=sec.is_appendix, lost_macros=sec.lost_macros,
            parent_id=parents.get(sec.id),
            children=tuple(children[sec.id]),
            paragraphs=tuple(by_marker.get(sec.id, ())),
        )
        for sec in markers
    }

    meta = docmodel.get("meta") or {}
    return MarkerTree(
        nodes=nodes,
        roots=tuple(s.id for s in markers if parents.get(s.id) is None),
        orphans=tuple(preamble),
        moved_by_position=tuple(moved),
        bibkey=str(meta.get("bibkey") or "") if isinstance(meta, dict) else "",
        source_path=source_path,
        math_sources=math_sources,
        skipped=tuple(skipped),
    )


def load_tree(path) -> MarkerTree:
    """Read a `model.docmodel.json` from disk. The file is opened read-only."""
    import json
    from pathlib import Path

    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        return build_tree(json.load(fh), source_path=str(p))
