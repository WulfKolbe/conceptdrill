"""Adapter: Semantic Compiler `model.docmodel.json` -> `Document`.

The DocModel keeps every object's text under a *different* prop depending on
type — `props.text` for prose, `props.latex` for equations, `props.content` for
list items, `props.caption` for figures. `TEXT_EXTRACTORS` is that mapping in one
place, and it is the only thing that needs touching when the DocModel grows a
new object type.

Read-only: nothing in this module writes to the input structure.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .types import BibEntry, Block, Section, SkippedObject

Props = dict[str, Any]

# Types that carry no projectable text at all. Skipped with a reason so a run
# can account for every object rather than silently shrinking.
NON_PROJECTABLE: dict[str, str] = {
    "page": "page objects carry layout, not content",
    "document": "root container",
    "toc": "table of contents mirrors headings already mined as candidates",
    "tablerow": "covered by the parent Table projection",
    "citation": "carries only a citekey; no text to embed",
}


def _first_prop(props: Props, *keys: str) -> str:
    for k in keys:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _text(props: Props) -> str:
    return _first_prop(props, "text")


def _latex(props: Props) -> str:
    return _first_prop(props, "latex", "latex_raw", "latex_original")


def _content(props: Props) -> str:
    return _first_prop(props, "content", "text")


def _caption(props: Props) -> str:
    return _first_prop(props, "caption", "text")


def _table(props: Props) -> str:
    """Caption first — it is the semantically loaded part — then the tabular
    source, which contributes column and row labels."""
    parts = [_first_prop(props, "caption"),
             _first_prop(props, "latex_code", "latex", "latex_original")]
    return "\n".join(p for p in parts if p)


def _heading(props: Props) -> str:
    return _first_prop(props, "title", "heading", "text", "caption")


# Object type (lowercased) -> text extractor.
#
# `code` and `algorithm` are registered but the DocModel does not emit them
# today. They stay here so they light up automatically if it ever does — see
# `routing.py`, which points them at CodeBERT.
TEXT_EXTRACTORS: dict[str, Callable[[Props], str]] = {
    "paragraph": _text,
    "abstract": _text,
    "footnote": _text,
    "sidenote": _text,
    "equation": _latex,
    "formula": _latex,
    "displayequation": _latex,
    "inlinemath": _latex,
    "listitem": _content,
    "table": _table,
    "picture": _caption,
    "diagram": _caption,
    "figure": _caption,
    "section": _heading,
    "heading": _heading,
    "caption": _caption,
    "code": _text,
    "sourcecode": _text,
    "algorithm": _text,
}

# DocModel type -> the block type ConceptDrill reasons about. Mostly identity,
# lowercased; a few are normalised so `PROSE_TYPES` and friends match.
TYPE_ALIASES: dict[str, str] = {
    "displayequation": "equation",
    "inlinemath": "formula",
    "picture": "figure",
    "diagram": "figure",
    "sourcecode": "code",
}


def extract_text(obj: dict[str, Any]) -> str:
    """Text representation of one DocModel object, or "" if it has none."""
    otype = str(obj.get("type") or "").lower()
    props = obj.get("props") or {}
    if not isinstance(props, dict):
        return ""
    extractor = TEXT_EXTRACTORS.get(otype)
    if extractor is None:
        # Unknown type: try the common props rather than dropping it outright.
        return _first_prop(props, "text", "content", "caption", "latex", "title")
    return extractor(props)


def skip_reason(obj: dict[str, Any]) -> Optional[str]:
    """Why this object is not projectable, or None if it is."""
    otype = str(obj.get("type") or "").lower()
    if otype in NON_PROJECTABLE:
        return NON_PROJECTABLE[otype]
    if not extract_text(obj).strip():
        return f"no text under any known prop for type '{obj.get('type')}'"
    return None


def _looks_like_bibliography(text: str) -> bool:
    """A heuristic, and labelled as one.

    The DocModel surfaces reference lists as `ListItem`s with no marker that
    they are bibliography. Requiring two of three signals — a year in
    parentheses or bare, an initials pattern, a venue word — keeps ordinary
    prose list items out.
    """
    import re
    signals = 0
    if re.search(r"\b(19|20)\d{2}\b", text):
        signals += 1
    if re.search(r"\b[A-Z]\.\s*[A-Z]?\.?,", text) or re.search(r"[A-Z][a-z]+,\s+[A-Z]\.", text):
        signals += 1
    if re.search(r"(?i)\b(proc\.|proceedings|journal|conf\.|conference|"
                 r"trans\.|arxiv|vol\.|pp\.|In:)\b", text):
        signals += 1
    return signals >= 2


def docmodel_to_document(data: dict[str, Any],
                         source_path: Optional[str] = None):
    """Convert a DocModel dict into a `Document`.

    Sections come from `Section` objects; parentage from the DocModel's own
    `parent` field, falling back to `props.parent_section`.
    """
    from .document import Document

    objects = [o for o in (data.get("objects") or []) if isinstance(o, dict)]

    # ---- sections -------------------------------------------------------
    sections: dict[str, Section] = {}
    for obj in objects:
        if str(obj.get("type") or "").lower() not in {"section", "heading"}:
            continue
        props = obj.get("props") or {}
        sid = str(obj.get("id") or "")
        if not sid:
            continue
        parent = obj.get("parent") or props.get("parent_section")
        sections[sid] = Section(
            id=sid,
            title=_heading(props),
            level=int(props.get("level", props.get("depth", 1)) or 1),
            parent_id=str(parent) if parent else None,
            children=tuple(str(c) for c in (obj.get("children") or ())),
        )
    # Drop parent pointers that leave the section tree (e.g. straight to the
    # Document root) so `top_level_section` terminates on a real section.
    sections = {
        sid: (sec if (sec.parent_id in sections or sec.parent_id is None)
              else Section(sec.id, sec.title, sec.level, None, sec.children))
        for sid, sec in sections.items()
    }

    # ---- blocks + bibliography -----------------------------------------
    blocks: list[Block] = []
    bibliography: list[BibEntry] = []
    skipped: list[SkippedObject] = []
    bib_counter = 0

    for obj in objects:
        raw_type = str(obj.get("type") or "")
        otype = raw_type.lower()
        oid = str(obj.get("id") or "")
        props = obj.get("props") or {}

        reason = skip_reason(obj)
        if reason is not None:
            skipped.append(SkippedObject(oid, raw_type, reason))
            continue

        text = extract_text(obj)

        # A ListItem that reads like a reference becomes a bibliography entry
        # *and* is retyped `bibitem`.
        #
        # The retype matters: reference lists are still projectable, but they are
        # not prose. Leaving them in the prose pool floods noun-phrase and NER
        # mining with author surnames and venue names, which then outrank the
        # document's actual concepts.
        block_type = TYPE_ALIASES.get(otype, otype)
        if otype == "listitem" and _looks_like_bibliography(text):
            import re
            m = re.search(r"\b(19|20)\d{2}\b", text)
            bibliography.append(BibEntry(
                id=f"bib_{bib_counter}", title=text,
                label=str(props.get("marker") or ""),
                year=int(m.group(0)) if m else None,
                props={"origin_object": oid},
            ))
            bib_counter += 1
            block_type = "bibitem"

        section_id = obj.get("parent") or props.get("parent_section")
        section_id = str(section_id) if section_id in sections else None

        blocks.append(Block(
            id=oid,
            type=block_type,
            text=text,
            section_id=section_id,
            props=dict(props),
        ))

    meta = dict(data.get("meta") or {})
    meta["docmodel_object_count"] = len(objects)
    meta["skipped"] = [
        {"object_id": s.object_id, "object_type": s.object_type, "reason": s.reason}
        for s in skipped
    ]

    return Document(blocks=blocks, sections=sections, bibliography=bibliography,
                    meta=meta, source_path=source_path)
