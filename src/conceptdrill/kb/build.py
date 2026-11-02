"""`kb build`: drilled documents in, a kb/1 knowledge base out (T-0119).

Deterministic stages decide everything structural:

    documents -> sections -> payload -> placement -> leaves -> topics

and a `Writer` -- the only stage that may use a language model -- names leaves
and topics and writes their prose from the cited sections alone. Payload never
passes through the writer: equations, code and pseudocode are copied by code,
byte for byte. The structure is the code's, so it cannot change with the
model's mood; `kb check` runs last and a build that does not conform fails.

Nothing here knows what the documents are about.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import yaml

from . import contract as C
from ..hierarchy.glyphsections import MIN_WORDS, split_glyphs

#: Two sections whose embeddings are at least this similar may share a leaf.
LEAF_TAU = 0.75
#: Bounds on the structure, so it never grows a shape kb/1 cannot hold.
MAX_SECTIONS_PER_LEAF = 6
MAX_LEAVES_PER_TOPIC = 8
MIN_TOPICS, MAX_TOPICS = 3, 12
#: A leaf's See also: at most this many neighbours, at least this similar.
RELATED_MAX, RELATED_MIN = 2, 0.5
#: How much of a section's text goes into its embedding.
EMBED_CHARS = 2000

PSEUDOCODE = re.compile(
    r"^\s*(?:\d+\s*[:.)]\s*\S|(?:for|while|repeat|until|if|else|return|end|"
    r"procedure|function|algorithm)\b)|←|:=", re.IGNORECASE)
PLACEHOLDER = re.compile(r"<!--\s*DEFERRED", re.IGNORECASE)


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# The writer: the only stage allowed a language model
# --------------------------------------------------------------------------

class Writer(Protocol):
    stub: bool
    model: Optional[str]

    @property
    def usage(self) -> dict: ...

    def leaf(self, sections: list[dict], payload: list[dict]) -> dict:
        """{"title", "use_when", "summary", "concepts": [{"name", "definition"}],
        "method": [str], "pitfalls": [str]} from the cited sections only."""

    def topic(self, leaves: list[dict]) -> dict:
        """{"title", "purpose", "differ"} from the topic's leaves."""

    def kb(self, topics: list[dict]) -> dict:
        """{"title", "purpose"} from the topics."""


class StubWriter:
    """Fixed strings in place of a model: no key, no network, no content.

    Names are numbered in call order, which the build makes deterministic, so
    two runs over the same input give the same tree.
    """
    stub = True
    model = None

    def __init__(self) -> None:
        self._n = {"leaf": 0, "topic": 0}

    @property
    def usage(self) -> dict:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def leaf(self, sections, payload):
        self._n["leaf"] += 1
        n = self._n["leaf"]
        return {"title": f"NAME-{n}", "use_when": "PROSE", "summary": "PROSE",
                "concepts": [{"name": f"CONCEPT-{n}", "definition": "PROSE"}],
                "method": ["PROSE"], "pitfalls": []}

    def topic(self, leaves):
        self._n["topic"] += 1
        return {"title": f"TOPIC-{self._n['topic']}", "purpose": "PROSE",
                "differ": "PROSE"}

    def kb(self, topics):
        return {"title": "KNOWLEDGE BASE", "purpose": "PROSE"}


# --------------------------------------------------------------------------
# Stage 1-2: documents and sections
# --------------------------------------------------------------------------

@dataclass
class Section:
    doc: str
    order: int
    title: str
    number: list
    page: int
    line_start: int
    line_end: int
    text: str                      # the text file's content, verbatim lines + "\n"
    embed_text: str
    status: str = "unused"
    reason: Optional[str] = None
    cited_by: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.doc}#{self.order}"

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def text_file(self) -> str:
        return f"text/{self.doc}/{self.order}.txt"


@dataclass
class Document:
    folder: str
    doc: str
    title: str
    glyphs: Path
    glyphs_sha: str
    docmodel: Optional[Path]
    docmodel_sha: Optional[str]
    sections: list[Section]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _title_of(text: str, folder: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            title = _one_line(re.sub(r"!\[[^\]]*\]\([^)]*\)|<!--.*?-->", " ", line[2:]))
            if title:
                return title
    return folder


def load_documents(folders: Sequence[str], library: Path) -> tuple[list[Document], list[dict]]:
    present, skipped = [], []
    for folder in dict.fromkeys(folders):            # keep order, drop repeats
        glyphs = library / folder / f"{folder}.glyphs.md"
        if glyphs.is_file():
            present.append(folder)
        else:
            skipped.append({"folder": folder,
                            "reason": f"no {folder}.glyphs.md in the library"})
    slugs = C.doc_slugs(present)
    docs = []
    for folder in sorted(present, key=lambda f: slugs[f]):
        glyphs = library / folder / f"{folder}.glyphs.md"
        raw = glyphs.read_bytes()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        dm = library / folder / "model.docmodel.json"
        doc = slugs[folder]
        sections = []
        for order, s in enumerate(split_glyphs(text)):
            body = "\n".join(lines[s["line_start"] - 1:s["line_end"]]) + "\n"
            sections.append(Section(
                doc=doc, order=order, title=_one_line(s["title"]), number=s["number"],
                page=s["page"], line_start=s["line_start"], line_end=s["line_end"],
                text=body, embed_text=f"{s['title']}\n{s['text'][:EMBED_CHARS]}"))
        docs.append(Document(folder, doc, _title_of(text, folder), glyphs, _sha(raw),
                             dm if dm.is_file() else None,
                             _sha(dm.read_bytes()) if dm.is_file() else None,
                             sections))
    return docs, skipped


# --------------------------------------------------------------------------
# Stage 3-4: payload and its placement
# --------------------------------------------------------------------------

@dataclass
class Payload:
    doc: str
    kind: str
    members: list[str]
    field: str
    content: str
    page: Optional[int]
    probe: str
    labels: list[str]
    section: Optional[str] = None
    placed_by: Optional[str] = None
    reason: Optional[str] = None
    caption: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.doc}#{self.members[0]}"

    @property
    def file(self) -> Optional[str]:
        if self.section is None:
            return None
        ext = "tex" if self.kind == "equation" else "txt"
        return f"payload/{self.doc}/{C.slugify(self.members[0])}.{ext}"


def _page(props: dict) -> Optional[int]:
    try:
        return int(str(props.get("page")))
    except (TypeError, ValueError):
        return None


def _flow(props: dict) -> Optional[int]:
    try:
        return int(str(props.get("flow_index")))
    except (TypeError, ValueError):
        return None


def extract_payload(doc: Document, verified: set[str]) -> list[Payload]:
    """Equations, code listings and pseudocode runs, verbatim, in document order."""
    if doc.docmodel is None:
        return []
    objects = json.loads(doc.docmodel.read_text(encoding="utf-8")).get("objects", [])
    items: list[Payload] = []

    # Code: consecutive CodeListing objects (page apart at most 1) are one listing.
    run: list[dict] = []
    def close_code():
        if run:
            code = "\n".join(o["props"].get("code", "") for o in run)
            items.append(Payload(doc.doc, "code", [o["id"] for o in run], "code", code,
                                 _page(run[0]["props"]), _probe(code),
                                 ["line numbers not separated"],
                                 caption=run[0]["props"].get("caption") or None))
            run.clear()
    for o in objects:
        if o.get("type") != "CodeListing":
            continue
        if run and abs((_page(o["props"]) or 0) - (_page(run[-1]["props"]) or 0)) > 1:
            close_code()
        run.append(o)
    close_code()

    # Pseudocode: runs of >= 3 flow-consecutive list items or paragraphs that
    # each read as a step.
    for otype, fieldname in (("ListItem", "content"), ("Paragraph", "text")):
        typed = sorted((o for o in objects if o.get("type") == otype
                        and _flow(o["props"]) is not None),
                       key=lambda o: _flow(o["props"]))
        group: list[dict] = []
        def close_ps():
            if len(group) >= 3:
                text = "\n".join(o["props"].get(fieldname, "") for o in group)
                items.append(Payload(doc.doc, "pseudocode", [o["id"] for o in group],
                                     fieldname, text, _page(group[0]["props"]),
                                     _probe(text), ["detected pseudocode"]))
            group.clear()
        for o in typed:
            text = str(o["props"].get(fieldname, ""))
            step = bool(PSEUDOCODE.search(text))
            adjacent = group and _flow(o["props"]) - _flow(group[-1]["props"]) <= 2
            if step and (not group or adjacent):
                group.append(o)
            else:
                close_ps()
                if step:
                    group.append(o)
        close_ps()

    # Equations: one item each here; grouped per section after placement.
    for o in objects:
        if o.get("type") != "Equation":
            continue
        latex = str(o["props"].get("latex") or "")
        if not latex.strip() or PLACEHOLDER.search(latex):
            continue
        items.append(Payload(doc.doc, "equation", [o["id"]], "latex", latex,
                             _page(o["props"]), latex.strip(),
                             ["verified" if o["id"] in verified else "unverified"],
                             caption=o["props"].get("caption") or None))
    return items


def _probe(text: str) -> str:
    for line in text.splitlines():
        if len(line.strip()) >= 6:
            return line.strip()
    return text.strip()


def place(doc: Document, items: list[Payload]) -> list[Payload]:
    """Put each item in the section that holds it: by its text, else by page."""
    ordered = sorted(doc.sections, key=lambda s: s.order)
    for item in items:
        hit = next((s for s in ordered if item.probe and item.probe in s.text), None)
        if hit is not None:
            item.section, item.placed_by = hit.id, "text"
            continue
        if item.page is None:
            item.reason = "no page, and its text is in no section"
            continue
        before = [s for s in ordered if s.page <= item.page]
        if not before:
            item.reason = f"page {item.page} precedes the first section"
            continue
        item.section, item.placed_by = before[-1].id, "page"
    # A section's equations become one item, in document order.
    out: list[Payload] = [i for i in items if i.kind != "equation" or i.section is None]
    by_section: dict[str, list[Payload]] = {}
    for i in items:
        if i.kind == "equation" and i.section is not None:
            by_section.setdefault(i.section, []).append(i)
    for sid, eqs in by_section.items():
        labels = ["verified"] if all(e.labels == ["verified"] for e in eqs) else ["unverified"]
        out.append(Payload(doc.doc, "equation", [m for e in eqs for m in e.members],
                           "latex", "\n".join(e.content for e in eqs), eqs[0].page,
                           eqs[0].probe, labels, sid,
                           "text" if all(e.placed_by == "text" for e in eqs) else "page"))
    order = {s.id: s.order for s in doc.sections}
    kind_rank = {"equation": 0, "pseudocode": 1, "code": 2}
    return sorted(out, key=lambda i: (order.get(i.section, 10 ** 9),
                                      kind_rank[i.kind], i.members[0]))


# --------------------------------------------------------------------------
# Stage 5-6: leaves and topics, by clustering in float64
# --------------------------------------------------------------------------

def _embed(texts: list[str], embedder_name: str) -> np.ndarray:
    from ..embeddings import get_embedder
    if not texts:
        return np.zeros((0, 1))
    embedder = get_embedder(embedder_name, cache=True)
    vectors = np.asarray(embedder.encode(texts), dtype=np.float64)
    embedder.flush()          # persist: the next build must not pay this pass again
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _agglomerate(vectors: np.ndarray, sizes: list[int], *, max_size: int,
                 tau: Optional[float] = None, target: Optional[int] = None) -> list[list[int]]:
    """Average-linkage clustering, deterministic: ties go to the lowest indices.

    Merges while the best pair is at least `tau` similar (or while there are
    more than `target` clusters), never past `max_size` members.
    """
    clusters = [[i] for i in range(len(vectors))]
    weight = list(sizes)
    sims = vectors @ vectors.T if len(vectors) else np.zeros((0, 0))
    while len(clusters) > 1:
        if target is not None and len(clusters) <= target:
            break
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if weight[a] + weight[b] > max_size:
                    continue
                s = float(np.mean(sims[np.ix_(clusters[a], clusters[b])]))
                if s > best + 1e-12:
                    best, pair = s, (a, b)
        if pair is None or (tau is not None and best < tau):
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        weight[a] += weight[b]
        del clusters[b], weight[b]
    return clusters


@dataclass
class Leaf:
    sections: list[Section]
    payload: list[Payload]
    vector: np.ndarray
    slug: str = ""
    topic: str = ""
    folder: str = ""
    title: str = ""
    prose: dict = field(default_factory=dict)
    related: list = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.topic}/{self.slug}"

    @property
    def path(self) -> str:
        return f"{self.folder}/{self.slug}.md"


@dataclass
class Topic:
    leaves: list[Leaf]
    slug: str = ""
    folder: str = ""
    prose: dict = field(default_factory=dict)


def _sort_key(section: Section) -> tuple:
    return (section.doc, section.order)


def entry_name(section: Section, doc_sections: list[Section], kind: str, *,
               caption: Optional[str]) -> Optional[str]:
    """The index entry for a payload item placed in `section` (T-0130).

    Named from the item's own caption or listing label; else from the nearest
    preceding numbered subsection of the same chapter; else from the chapter's
    title. Never from `section`'s own title -- that is what `kb reach` queries
    with -- and never from the paper's title, its second query. None when no
    such name exists or it names nothing.
    """
    from .reach import content_words, method_name
    if caption and content_words(caption):
        return f"{_one_line(caption)} — {kind}"
    if not section.number:
        return None
    chapter = section.number[0]
    before = [s for s in doc_sections if s.order < section.order and s.number
              and s.number[0] == chapter]
    subs = [s for s in before if len(s.number) >= 2]
    heads = [s for s in before if len(s.number) == 1]
    source = subs[-1] if subs else (heads[-1] if heads else None)
    if source is None:
        return None
    name = method_name(source.title)
    return f"{name} — {kind}" if content_words(name) else None


def attach_payload(leaves: list["Leaf"], payload: list[Payload]) -> None:
    """Give each leaf the payload of its chapters (T-0130).

    An item placed in a numbered section is carried by every leaf citing a
    section of the same document with the same top-level number: a paper's
    method chapter and its subsections hold the method's payload between them,
    and clustering by text may split them across leaves. An item in an
    unnumbered section stays with the leaves citing that section only.
    """
    sections = {s.id: s for l in leaves for s in l.sections}
    def chapter(sid: str):
        s = sections.get(sid)
        return (s.doc, s.number[0]) if s is not None and s.number else None
    by_chapter: dict[tuple, list[Payload]] = {}
    by_section: dict[str, list[Payload]] = {}
    for p in payload:
        if p.section is None:
            continue
        by_section.setdefault(p.section, []).append(p)
        ch = chapter(p.section)
        if ch is not None:
            by_chapter.setdefault(ch, []).append(p)
    order = {s.id: (s.doc, s.order) for s in sections.values()}
    rank = {"equation": 0, "pseudocode": 1, "code": 2}
    for leaf in leaves:
        mine: dict[str, Payload] = {}
        for s in leaf.sections:
            ch = (s.doc, s.number[0]) if s.number else None
            for p in (by_chapter.get(ch, []) if ch else by_section.get(s.id, [])):
                mine[p.id] = p
        leaf.payload = sorted(mine.values(), key=lambda p: (
            order.get(p.section, ("", 10 ** 9)), rank[p.kind], p.members[0]))


def structure(sections: list[Section], payload: list[Payload],
              embedder_name: str, embedding: Optional[dict] = None) -> list[Topic]:
    usable = [s for s in sections if s.status == "cited"]
    if not usable:
        raise BuildError("no section has enough text or payload to cite")
    vectors = _embed([s.embed_text for s in usable], embedder_name)
    if embedding is not None:
        # What the structure was clustered from: on a host with faulty float32
        # BLAS, uncached embeddings can differ between runs, so "the same
        # tree" is checked by this hash, not assumed.
        embedding.update(model=embedder_name, vectors=int(len(vectors)),
                         sha256=_sha(np.ascontiguousarray(vectors, dtype="<f8").tobytes()))
    groups = _agglomerate(vectors, [1] * len(usable), max_size=MAX_SECTIONS_PER_LEAF,
                          tau=LEAF_TAU)
    leaves = []
    for g in groups:
        secs = sorted((usable[i] for i in g), key=_sort_key)
        centroid = vectors[g].mean(axis=0)
        n = np.linalg.norm(centroid)
        leaves.append(Leaf(secs, [], centroid / n if n else centroid))
    attach_payload(leaves, payload)
    leaves.sort(key=lambda l: _sort_key(l.sections[0]))
    lv = np.vstack([l.vector for l in leaves])
    want = min(len(leaves), max(MIN_TOPICS, min(MAX_TOPICS, round(len(leaves) ** 0.5)),
                                -(-len(leaves) // MAX_LEAVES_PER_TOPIC)))
    tgroups = _agglomerate(lv, [1] * len(leaves), max_size=MAX_LEAVES_PER_TOPIC,
                           target=want)
    topics = [Topic(sorted((leaves[i] for i in g), key=lambda l: _sort_key(l.sections[0])))
              for g in tgroups]
    topics.sort(key=lambda t: (-len(t.leaves), _sort_key(t.leaves[0].sections[0])))
    # See also: nearest leaves, by centroid cosine.
    sims = lv @ lv.T
    index = {id(l): i for i, l in enumerate(leaves)}
    for leaf in leaves:
        i = index[id(leaf)]
        order = sorted((j for j in range(len(leaves)) if j != i),
                       key=lambda j: (-sims[i, j], j))
        leaf.related = [leaves[j] for j in order[:RELATED_MAX] if sims[i, j] >= RELATED_MIN]
    return topics


# --------------------------------------------------------------------------
# Stage 7: naming and prose (the writer), then rendering
# --------------------------------------------------------------------------

def _unique(slug: str, taken: set[str]) -> str:
    slug = slug or "untitled"
    base, n, out = slug, 1, slug
    while out in taken:
        n += 1
        out = f"{base[:C.SLUG_MAX - 4]}-{n}"
    taken.add(out)
    return out


def write_prose(topics: list[Topic], writer) -> dict:
    taken_topics: set[str] = set()
    for t_index, topic in enumerate(topics, 1):
        taken_leaves: set[str] = set()
        for leaf in topic.leaves:
            # The writer sees the payload of the sections it cites. Payload a
            # leaf carries for its chapter (attach_payload) is added by code and
            # left out here, so carrying it changes no prompt and costs nothing.
            own = {s.id for s in leaf.sections}
            leaf.prose = writer.leaf(
                [{"id": s.id, "title": s.title, "text": s.text} for s in leaf.sections],
                [{"id": p.id, "kind": p.kind, "content": p.content}
                 for p in leaf.payload if p.section in own])
            leaf.title = _one_line(leaf.prose["title"]) or "Untitled"
            leaf.slug = _unique(C.slugify(leaf.title), taken_leaves)
        topic.prose = writer.topic([{"title": l.title, "summary": l.prose["summary"]}
                                    for l in topic.leaves])
        topic.slug = _unique(C.slugify(_one_line(topic.prose["title"])), taken_topics)
        topic.folder = f"topics/{t_index:02d}-{topic.slug}"
        for leaf in topic.leaves:
            leaf.topic, leaf.folder = topic.slug, topic.folder
    return writer.kb([{"title": _one_line(t.prose["title"]), "purpose": t.prose["purpose"]}
                      for t in topics])


def _prose(text: str) -> str:
    """Writer text made safe for a kb/1 section: no heading, no fence."""
    lines = [re.sub(r"^\s*(#+|```+)\s*", "", l) for l in str(text).strip().splitlines()]
    return "\n".join(lines).strip() or "None."


def _front(kind: str, values: dict) -> str:
    ordered = {k: values[k] for k in C.FRONTMATTER[kind]}
    return "---\n" + yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True,
                                    width=10 ** 6) + "---\n"


def _doc_md(kind: str, front: dict, title_line: str, sections: list[tuple[str, str]]) -> str:
    body = "".join(f"## {h}\n\n{(t.strip() or 'None.')}\n\n" for h, t in sections)
    return _front(kind, front) + f"{title_line}\n\n" + body.rstrip("\n") + "\n"


def _bullets(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "None."


def _rel(from_dir: str, to: str) -> str:
    import os
    return os.path.relpath(to, from_dir or ".").replace("\\", "/")


def render(docs: list[Document], topics: list[Topic], payload: list[Payload],
           kb_prose: dict, name: str) -> dict[str, str]:
    files: dict[str, str] = {}
    leaves = [l for t in topics for l in t.leaves]
    for t in topics:
        for l in t.leaves:
            for s in l.sections:
                s.cited_by.append(l.id)
    sections = [s for d in docs for s in d.sections]
    by_id = {s.id: s for s in sections}

    # leaves
    doc_sections = {d.doc: d.sections for d in docs}
    for leaf in leaves:
        concepts = [_one_line(c["name"]) for c in leaf.prose.get("concepts", [])
                    if _one_line(c["name"])]
        # One index entry per payload item this leaf's own sections hold.
        own = {s.id: s for s in leaf.sections}
        for p in leaf.payload:
            if p.section in own:
                entry = entry_name(own[p.section], doc_sections[p.doc], p.kind,
                                   caption=p.caption)
                if entry and entry not in concepts:
                    concepts.append(entry)
        def payload_lines(inline: bool) -> str:
            out = []
            for p in leaf.payload:
                out.append(f"- `{p.id}` {p.kind} · {', '.join(p.labels)} · from "
                           f"`{p.section}` → [file]({_rel(leaf.folder, p.file)})")
                lines = p.content.split("\n")
                if inline and len(lines) <= C.INLINE_MAX_LINES and "```" not in p.content:
                    out.append(f"\n```\n{p.content}\n```\n")
            return "\n".join(out) if out else "None."
        sources = [f"- `{s.id}` {s.title} → [text, {s.words} words]"
                   f"({_rel(leaf.folder, s.text_file)}) · "
                   f"[{s.doc}]({_rel(leaf.folder, f'sources/{s.doc}.md')})"
                   for s in leaf.sections]
        see_also = [f"- [{r.title}]({_rel(leaf.folder, r.path)})" for r in leaf.related]
        method = [f"{i}. {_one_line(step)}" for i, step in
                  enumerate(leaf.prose.get("method", []), 1) if _one_line(step)]
        pitfalls = [f"- {_one_line(p)}" for p in leaf.prose.get("pitfalls", []) if _one_line(p)]
        front = {"format": C.FORMAT, "id": leaf.id, "title": leaf.title,
                 "topic": leaf.topic, "concepts": concepts,
                 "sources": [s.id for s in leaf.sections],
                 "payload": [p.id for p in leaf.payload],
                 "related": [r.id for r in leaf.related]}
        for inline in (True, False):
            text = _doc_md("leaf", front, f"# {leaf.title}", [
                ("Use when", _prose(leaf.prose.get("use_when", ""))),
                ("Summary", _prose(leaf.prose.get("summary", ""))),
                ("Key concepts", _bullets([f"- {c}" for c in concepts])),
                ("Method", _bullets(method)),
                ("Payload", payload_lines(inline)),
                ("Pitfalls", _bullets(pitfalls)),
                ("Sources", _bullets(sources)),
                ("See also", _bullets(see_also))])
            if C.tokens(text) <= C.BUDGETS["leaf"]:
                break
        files[leaf.path] = text
        leaf.concepts = concepts

    # topics
    for t in topics:
        rel_topics = sorted({r.topic for l in t.leaves for r in l.related if r.topic != t.slug})
        folder_of = {x.slug: x for x in topics}
        files[f"{t.folder}/README.md"] = _doc_md("topic", {
            "format": C.FORMAT, "id": t.slug, "title": _one_line(t.prose["title"]),
            "leaves": [l.id for l in t.leaves],
            "concepts": sorted({c for l in t.leaves for c in l.concepts}),
            "related": rel_topics}, f"# {_one_line(t.prose['title'])}", [
            ("Purpose", _prose(t.prose.get("purpose", ""))),
            ("Leaves", _bullets([f"- [{l.title}]({l.slug}.md) — "
                                 f"{_one_line(l.prose.get('use_when', '')) or l.title}"
                                 for l in t.leaves])),
            ("How the approaches differ", _prose(t.prose.get("differ", ""))),
            ("Related topics", _bullets([
                f"- [{_one_line(folder_of[r].prose['title'])}]"
                f"({_rel(t.folder, folder_of[r].folder + '/README.md')})"
                for r in rel_topics]))])

    # sources and text
    for d in docs:
        cited, unused = [], []
        for s in d.sections:
            head = (f"- `{s.id}` p.{s.page} l.{s.line_start}–{s.line_end} {s.title} → "
                    f"[text, {s.words} words](../{s.text_file})")
            if s.status == "cited":
                links = ", ".join(f"[{l.title}](../{l.path})" for l in leaves
                                  if l.id in s.cited_by)
                cited.append(f"{head} · {links}")
            else:
                unused.append(f"{head} — {s.reason}")
            files[s.text_file] = s.text
        files[f"sources/{d.doc}.md"] = _doc_md("source", {
            "format": C.FORMAT, "doc": d.doc, "title": d.title,
            "sections": len(d.sections)}, f"# {d.title}",
            [("Sections", _bullets(cited)), ("Unused sections", _bullets(unused))])

    # payload files
    for p in payload:
        if p.file:
            files[p.file] = p.content

    # index and glossary
    entries = sorted(((c, l) for l in leaves for c in l.concepts),
                     key=lambda e: (e[0].casefold(), e[1].id))
    terms = sorted(((_one_line(c["name"]), _one_line(c.get("definition", "")), l)
                    for l in leaves for c in l.prose.get("concepts", [])
                    if _one_line(c["name"])), key=lambda e: (e[0].casefold(), e[2].id))
    counts_i, counts_g = {}, {}
    for letter in C.LETTERS:
        mine = [f"- {c} → [{l.title}](../{l.path})" for c, l in entries
                if C.letter_of(c) == letter]
        counts_i[letter] = len(mine)
        files[f"index/{letter}.md"] = _doc_md("index_letter", {
            "format": C.FORMAT, "kb": name, "letter": letter, "entries": len(mine)},
            C.letter_title("index_letter", letter), [("Index", _bullets(mine))])
        mine = [f"- **{t}** — {d or 'None.'} → [{l.title}](../{l.path})"
                for t, d, l in terms if C.letter_of(t) == letter]
        counts_g[letter] = len(mine)
        files[f"glossary/{letter}.md"] = _doc_md("glossary_letter", {
            "format": C.FORMAT, "kb": name, "letter": letter, "terms": len(mine)},
            C.letter_title("glossary_letter", letter), [("Glossary", _bullets(mine))])
    shown = lambda l: "Other" if l == "other" else l.upper()
    files["INDEX.md"] = _doc_md("index", {"format": C.FORMAT, "kb": name,
                                          "letters": C.LETTERS}, "# Index", [
        ("Letters", "\n".join(f"- [{shown(l)}](index/{l}.md) — {counts_i[l]} entries"
                              for l in C.LETTERS))])
    files["GLOSSARY.md"] = _doc_md("glossary", {"format": C.FORMAT, "kb": name,
                                                "letters": C.LETTERS}, "# Glossary", [
        ("Letters", "\n".join(f"- [{shown(l)}](glossary/{l}.md) — {counts_g[l]} terms"
                              for l in C.LETTERS))])

    # README
    kb_title = _one_line(kb_prose.get("title", "")) or name
    files["README.md"] = _doc_md("readme", {
        "format": C.FORMAT, "kb": name, "title": kb_title,
        "documents": [d.doc for d in docs], "topics": [t.slug for t in topics]},
        f"# {kb_title}", [
        ("Purpose", _prose(kb_prose.get("purpose", ""))),
        ("Topics", _bullets([f"- [{_one_line(t.prose['title'])}]({t.folder}/README.md) — "
                             f"{_one_line(t.prose.get('purpose', '')) or 'None.'}"
                             for t in topics])),
        ("How to read this knowledge base", C.PROTOCOL),
        ("Sources", _bullets([f"- [{d.doc}](sources/{d.doc}.md) — {d.title}" for d in docs]))])

    # kb.json
    files["kb.json"] = json.dumps({
        "format": C.FORMAT, "kb": name,
        "documents": [{"doc": d.doc, "folder": d.folder, "title": d.title,
                       "glyphs": {"path": str(d.glyphs), "sha256": d.glyphs_sha},
                       "docmodel": ({"path": str(d.docmodel), "sha256": d.docmodel_sha}
                                    if d.docmodel else None)} for d in docs],
        "sections": [{"id": s.id, "doc": s.doc, "order": s.order, "title": s.title,
                      "number": s.number, "page": s.page, "line_start": s.line_start,
                      "line_end": s.line_end, "status": s.status, "cited_by": s.cited_by,
                      "reason": s.reason, "text_file": s.text_file,
                      "text_sha256": _sha(s.text.encode("utf-8")), "words": s.words,
                      "nul": s.text.count("\x00")}
                     for s in sections],
        "payload": [{"id": p.id, "doc": p.doc, "kind": p.kind, "members": p.members,
                     "field": p.field, "section": p.section, "placed_by": p.placed_by,
                     "status": "placed" if p.section else "unplaced", "labels": p.labels,
                     "file": p.file,
                     "sha256": _sha(p.content.encode("utf-8")) if p.section else None,
                     "reason": None if p.section else p.reason} for p in payload],
        "topics": [{"id": t.slug, "title": _one_line(t.prose["title"]), "folder": t.folder,
                    "leaves": [l.id for l in t.leaves]} for t in topics],
        "leaves": [{"id": l.id, "title": l.title, "topic": l.topic, "path": l.path,
                    "concepts": l.concepts, "sources": [s.id for s in l.sections],
                    "payload": [p.id for p in l.payload],
                    "related": [r.id for r in l.related]} for l in leaves],
        "concepts": [{"name": c, "leaf": l.id} for c, l in entries],
        "terms": [{"term": t, "definition": d, "leaf": l.id} for t, d, l in terms],
    }, indent=2, ensure_ascii=False) + "\n"
    del by_id
    return files


# --------------------------------------------------------------------------

def build_kb(folders: Sequence[str], *, name: str, library: Path, out_root: Path,
             writer, embedder: str, command: str, commit: Optional[str],
             replace: bool = False, verified: Optional[set[str]] = None,
             dry: bool = False) -> dict:
    """Build the knowledge base. `dry` runs every stage and writes nothing."""
    if not C.SLUG_RE.match(name):
        raise BuildError(f"--name {name!r} is not a slug")
    target = out_root / "kb" / name
    if target.exists() and not dry:
        if not replace:
            raise BuildError(f"{target} exists; pass --replace to rebuild it")
        if not (target / "kb.json").is_file():
            raise BuildError(f"{target} exists and is not a knowledge base; not replaced")
    docs, skipped = load_documents(folders, library)
    if not docs:
        raise BuildError("no requested document has a glyphs.md in the library")
    payload: list[Payload] = []
    for d in docs:
        payload += place(d, extract_payload(d, verified or set()))
    with_payload = {p.section for p in payload if p.section}
    for s in (s for d in docs for s in d.sections):
        if s.words >= MIN_WORDS or s.id in with_payload:
            s.status = "cited"
        else:
            s.reason = f"only {s.words} words of text and no payload"
    embedding: dict = {}
    topics = structure([s for d in docs for s in d.sections], payload, embedder, embedding)
    kb_prose = write_prose(topics, writer)
    files = render(docs, topics, payload, kb_prose, name)
    counts = {"path": target, "documents": len(docs), "skipped": skipped,
              "sections": sum(len(d.sections) for d in docs),
              "leaves": sum(len(t.leaves) for t in topics), "topics": len(topics),
              "payload": len(payload), "files": len(files) + 1}
    if dry:
        return counts
    inputs = []
    for d in docs:
        inputs.append({"path": str(d.glyphs), "sha256": d.glyphs_sha})
        if d.docmodel:
            inputs.append({"path": str(d.docmodel), "sha256": d.docmodel_sha})
    files["manifest.json"] = json.dumps({
        "format": C.FORMAT, "kb": name, "command": command,
        "conceptdrill_commit": commit, "inputs": inputs, "skipped": skipped,
        "embedding": embedding,
        "llm": {"stub": bool(writer.stub), "model": writer.model, "usage": writer.usage},
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False) + "\n"
    if target.exists():
        shutil.rmtree(target)
    for rel, text in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return counts
