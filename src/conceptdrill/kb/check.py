"""`kb check`: does a folder conform to its format, kb/2 (docs/kb-format/kb-2.md)
or kb/1 (docs/kb-format/kb-1.md)?

The format is the one the tree names (`kb.json`, else the root README); a
tree that names none is judged by its shape, else as `contract.CURRENT`.

Every violation is collected, never only the first: a report that stops at
one break hides the next. Notes are things worth knowing that do not make the
tree invalid -- an input that has changed since the build, for instance.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Optional

from . import contract as C

LINK = re.compile(r"\]\(([^)\s]*)\)")


@dataclass
class Report:
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    files: int = 0
    format: str = C.CURRENT.format

    @property
    def ok(self) -> bool:
        return not self.violations


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schema(name: str) -> dict:
    return json.loads(resources.files("conceptdrill.kb").joinpath(
        "schemas", name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Markdown files
# --------------------------------------------------------------------------

@dataclass
class MdFile:
    rel: str
    kind: str
    front: Optional[dict]
    title: Optional[str]
    sections: list[tuple[str, list[str]]]
    fences: list[str]                  # headings under which a fence appeared
    links: list[str]
    text: str


def detect(root: Path) -> C.Spec:
    """The spec a tree is checked against: the format it names, else its shape."""
    import yaml
    try:
        fmt = json.loads((root / "kb.json").read_text(encoding="utf-8")).get("format")
        if fmt in C.SPECS:
            return C.SPECS[fmt]
    except Exception:
        pass
    try:
        text = (root / "README.md").read_text(encoding="utf-8")
        fmt = yaml.safe_load(text[4:text.index("\n---\n", 3) + 1]).get("format")
        if fmt in C.SPECS:
            return C.SPECS[fmt]
    except Exception:
        pass
    if (root / "areas").is_dir():
        return C.KB2
    if (root / "topics").is_dir():
        return C.KB1
    return C.CURRENT


def kind_of(rel: str, spec: C.Spec = C.CURRENT) -> Optional[str]:
    parts = rel.split("/")
    if rel == "README.md":
        return "readme"
    if rel == "INDEX.md":
        return "index"
    if rel == "GLOSSARY.md":
        return "glossary"
    if rel == "SOURCES.md" and spec is C.KB2:
        return "sources_list"
    if len(parts) == 2 and parts[0] == "index":
        return "index_letter"
    if len(parts) == 2 and parts[0] == "glossary":
        return "glossary_letter"
    if len(parts) == 2 and parts[0] == "sources":
        return "source"
    if spec is C.KB1:
        if len(parts) == 3 and parts[0] == "topics":
            return "topic" if parts[2] == "README.md" else "leaf"
        return None
    if parts[0] != "areas":
        return None
    if len(parts) == 3 and parts[2] == "README.md":
        return "area"
    if len(parts) == 5 and parts[2] == "topics":
        return "topic" if parts[4] == "README.md" else "leaf"
    return None


def parse_md(rel: str, text: str, problems: list[str],
             spec: C.Spec = C.CURRENT) -> MdFile:
    import yaml
    kind = kind_of(rel, spec) or "?"
    front, body = None, text
    if text.startswith("---\n") and "\n---\n" in text[3:]:
        end = text.index("\n---\n", 3)
        try:
            front = yaml.safe_load(text[4:end + 1]) or {}
            if not isinstance(front, dict):
                raise ValueError("not a mapping")
        except Exception as exc:          # malformed YAML is a violation, not a crash
            problems.append(f"{rel}: frontmatter is not valid YAML ({exc})")
            front = None
        body = text[end + 5:]
    else:
        problems.append(f"{rel}: no frontmatter")
    title, sections, fences, links = None, [], [], []
    in_fence = False
    current: Optional[list[str]] = None
    for line in body.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            if sections:
                fences.append(sections[-1][0])
            if current is not None:
                current.append(line)
            continue
        if in_fence:
            if current is not None:
                current.append(line)
            continue
        links += LINK.findall(line)
        if line.startswith("## "):
            current = []
            sections.append((line[3:].strip(), current))
        elif line.startswith("# "):
            if title is None and not sections:
                title = line
            else:
                problems.append(f"{rel}: extra '# ' heading {line!r}")
        elif current is not None:
            current.append(line)
    return MdFile(rel, kind, front, title, sections, fences, links, text)


def check_md(md: MdFile, problems: list[str], spec: C.Spec = C.CURRENT) -> None:
    rel, kind = md.rel, md.kind
    if kind not in spec.headings:
        return
    # frontmatter keys
    if md.front is not None:
        want, got = spec.frontmatter[kind], list(md.front)
        if sorted(want) != sorted(got):
            missing = [k for k in want if k not in got]
            extra = [k for k in got if k not in want]
            problems.append(f"{rel}: frontmatter keys: missing {missing}, extra {extra}")
        if md.front.get("format", spec.format) != spec.format:
            problems.append(f"{rel}: frontmatter format is not {spec.format}")
    # title
    if kind in ("index_letter", "glossary_letter"):
        letter = Path(rel).stem
        expected = C.letter_title(kind, letter)
    elif kind in ("index", "glossary"):
        expected = "# Index" if kind == "index" else "# Glossary"
    elif kind == "sources_list":
        expected = "# Sources"
    else:
        expected = f"# {(md.front or {}).get('title', '')}"
    if md.title != expected:
        problems.append(f"{rel}: title {md.title!r}, expected {expected!r}")
    # headings, in order, each once
    names = [h for h, _ in md.sections]
    if names != spec.headings[kind]:
        problems.append(f"{rel}: headings {names}, expected {spec.headings[kind]}")
    for heading, lines in md.sections:
        content = [l for l in lines if l.strip()]
        if not content:
            problems.append(f"{rel}: ## {heading} is empty; write None.")
            continue
        if content == ["None."]:
            continue
        form = spec.line_forms.get((kind, heading))
        if form is None:
            continue
        n = 0
        in_fence = False
        for line in content:
            if line.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            n += 1
            if not form.match(line):
                problems.append(f"{rel}: ## {heading} line {n} is not of the {spec.format} "
                                f"form: {line[:80]!r}")
    for heading in md.fences:
        if (kind, heading) not in spec.fences_allowed:
            problems.append(f"{rel}: fenced block under ## {heading}; "
                            f"only a leaf's ## Payload may hold one")
    if kind == "readme":
        proto = dict(md.sections).get("How to read this knowledge base")
        if proto is not None and "\n".join(l for l in proto if l.strip()) != spec.protocol:
            problems.append(f"{rel}: ## How to read this knowledge base is not the "
                            f"{spec.format} reading protocol")
    budget = spec.budgets.get(kind)
    used = C.tokens(md.text)
    if budget and used > budget:
        problems.append(f"{rel}: {used} tokens, budget {budget}")
    if kind in ("index_letter", "glossary_letter", "sources_list") and md.front:
        key = {"index_letter": "entries", "glossary_letter": "terms",
               "sources_list": "documents"}[kind]
        lines = [l for l in (md.sections[0][1] if md.sections else []) if l.strip()]
        count = 0 if lines == ["None."] else len(lines)
        if md.front.get(key) != count:
            problems.append(f"{rel}: {key} {md.front.get(key)} but {count} lines")


def check_links(root: Path, md: MdFile, problems: list[str],
                graph: dict[str, set[str]], spec: C.Spec = C.CURRENT) -> None:
    here = (root / md.rel).parent
    for target in md.links:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            problems.append(f"{md.rel}: link {target} is a URL; {spec.format} links stay in the tree")
            continue
        if target.startswith("/"):
            problems.append(f"{md.rel}: link {target} is absolute")
            continue
        path = (here / target.split("#", 1)[0]).resolve()
        try:
            rel = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            problems.append(f"{md.rel}: link {target} points outside the knowledge base")
            continue
        if not path.is_file():
            problems.append(f"{md.rel}: dead link {target}")
            continue
        graph.setdefault(md.rel, set()).add(rel)


# --------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------

def check_tree(root: Path, problems: list[str], spec: C.Spec = C.CURRENT) -> list[str]:
    """Shape of the folder. Returns every file's relative path."""
    files: list[str] = []
    for name in spec.root_files:
        if not (root / name).is_file():
            problems.append(f"{name}: missing")
    for name in spec.required_dirs:
        if not (root / name).is_dir():
            problems.append(f"{name}/: missing")
    for folder in ("index", "glossary"):
        for letter in C.LETTERS:
            if not (root / folder / f"{letter}.md").is_file():
                problems.append(f"{folder}/{letter}.md: missing")
        if spec.letter_pages and (root / folder).is_dir():
            for p in sorted((root / folder).glob("*-*.md")):
                letter, n = C.letter_page(p.stem)
                before = letter if n == 2 else f"{letter}-{n - 1}"
                if n >= 2 and not (root / folder / f"{before}.md").is_file():
                    problems.append(f"{folder}/{p.name}: page {n} without page {n - 1}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(p.startswith(".") for p in parts):
            problems.append(f"{rel}: hidden entry")
            continue
        if path.is_dir():
            if not any(path.iterdir()):
                problems.append(f"{rel}/: empty folder")
            if not _dir_allowed(parts, spec):
                problems.append(f"{rel}: not in the {spec.format} tree")
            continue
        files.append(rel)
        if not _file_allowed(parts, spec):
            problems.append(f"{rel}: not in the {spec.format} tree")
        elif rel not in spec.root_files and rel.lower() != rel and parts[-1] != "README.md":
            problems.append(f"{rel}: not lower case")
    if spec is C.KB1:
        _numbered(root, "topics", "topic", None, problems)
    else:
        areas = _numbered(root, "areas", "area", spec.max_areas, problems)
        for area in areas:
            if not (root / "areas" / area / "topics").is_dir():
                problems.append(f"areas/{area}/topics/: missing")
                continue
            _numbered(root, f"areas/{area}/topics", "topic", spec.max_topics_per_area,
                      problems)
    return files


def _numbered(root: Path, rel: str, what: str, most: Optional[int],
              problems: list[str]) -> list[str]:
    """Folders NN-<slug> under rel, numbered 01.. without a gap, each with a
    README.md; at least one, at most `most`."""
    here = root / rel
    if not here.is_dir():
        return []
    names = sorted(p.name for p in here.iterdir() if p.is_dir() and not p.name.startswith("."))
    for i, name in enumerate(names, 1):
        if not name.startswith(f"{i:02d}-"):
            problems.append(f"{rel}/{name}: expected {i:02d}-<slug>")
        if not (here / name / "README.md").is_file():
            problems.append(f"{rel}/{name}/README.md: missing")
    if not names:
        problems.append(f"{rel}/: no {what}")
    if most is not None and len(names) > most:
        problems.append(f"{rel}/: {len(names)} {what}s, at most {most}")
    return names


_NN_SLUG = re.compile(rf"^[0-9]{{2}}-{C.SLUG}$")


def _dir_allowed(parts: list[str], spec: C.Spec = C.CURRENT) -> bool:
    if len(parts) == 1:
        return parts[0] in spec.root_dirs
    if len(parts) == 2 and parts[0] in ("payload", "text"):
        return bool(C.SLUG_RE.match(parts[1]))
    if spec is C.KB1:
        return len(parts) == 2 and parts[0] == "topics" and bool(_NN_SLUG.match(parts[1]))
    if parts[0] != "areas":
        return False
    if len(parts) == 2:
        return bool(_NN_SLUG.match(parts[1]))
    if len(parts) == 3:
        return bool(_NN_SLUG.match(parts[1])) and parts[2] == "topics"
    if len(parts) == 4:
        return bool(_NN_SLUG.match(parts[1])) and parts[2] == "topics" \
            and bool(_NN_SLUG.match(parts[3]))
    return False


def _file_allowed(parts: list[str], spec: C.Spec = C.CURRENT) -> bool:
    if len(parts) == 1:
        return parts[0] in spec.root_files
    head = parts[0]
    if head in ("index", "glossary") and len(parts) == 2:
        if not parts[1].endswith(".md"):
            return False
        stem = parts[1][:-3]
        letter, n = C.letter_page(stem)
        if stem == letter:
            return letter in C.LETTERS
        return (spec.letter_pages and letter in C.LETTERS
                and bool(re.fullmatch(r"[a-z]+-[1-9][0-9]*", stem)) and n >= 2)
    if head == "sources" and len(parts) == 2:
        return bool(re.match(rf"^{C.SLUG}\.md$", parts[1]))
    if head == "topics" and len(parts) == 3 and spec is C.KB1:
        return parts[2] == "README.md" or bool(re.match(rf"^{C.SLUG}\.md$", parts[2]))
    if head == "areas" and spec is C.KB2 and _dir_allowed(parts[:-1], spec):
        if len(parts) == 3:
            return parts[2] == "README.md"
        if len(parts) == 5:
            return parts[4] == "README.md" or bool(re.match(rf"^{C.SLUG}\.md$", parts[4]))
    if head == "payload" and len(parts) == 3:
        return bool(re.match(rf"^{C.SLUG}\.(tex|txt)$", parts[2]))
    if head == "text" and len(parts) == 3:
        return bool(re.match(r"^[0-9]+\.txt$", parts[2]))
    return False


# --------------------------------------------------------------------------
# kb.json and its rules (kb-1 section 7a)
# --------------------------------------------------------------------------

def _load_json(root: Path, name: str, schema: str, problems: list[str]) -> Optional[dict]:
    path = root / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"{name}: not JSON ({exc})")
        return None
    import jsonschema
    validator = jsonschema.Draft202012Validator(_schema(schema))
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    for e in errors:
        where = "/".join(str(p) for p in e.path) or "(root)"
        problems.append(f"{name}: schema: {where}: {e.message[:160]}")
    return None if errors else data


def check_kb_json(root: Path, kb: dict, problems: list[str], notes: list[str]) -> None:
    P = lambda m: problems.append(f"kb.json: {m}")
    sections = {s["id"]: s for s in kb["sections"]}
    leaves = {l["id"]: l for l in kb["leaves"]}
    topics = {t["id"]: t for t in kb["topics"]}
    payload = {p["id"]: p for p in kb["payload"]}
    docs = {d["doc"]: d for d in kb["documents"]}

    # rule 8: doc slugs
    slugs = C.doc_slugs(d["folder"] for d in kb["documents"])
    for d in kb["documents"]:
        if slugs.get(d["folder"]) != d["doc"]:
            P(f"doc {d['doc']} is not the slug of {d['folder']!r}")

    # rule 7: orders without a gap, per document
    by_doc: dict[str, list[int]] = {}
    for s in kb["sections"]:
        if s["doc"] not in docs:
            P(f"{s['id']} belongs to unknown document {s['doc']}")
        if s["id"] != f"{s['doc']}#{s['order']}":
            P(f"{s['id']} is not {s['doc']}#{s['order']}")
        by_doc.setdefault(s["doc"], []).append(s["order"])
    for doc, orders in by_doc.items():
        missing = sorted(set(range(max(orders) + 1)) - set(orders))
        if missing:
            P(f"{doc} sections skip {', '.join(f'{doc}#{o}' for o in missing)}")

    # rules 1 and 2: accounting
    for s in kb["sections"]:
        sid = s["id"]
        if s["status"] == "cited":
            if s["reason"] is not None:
                P(f"{sid} cited but has a reason")
            if not s["cited_by"]:
                P(f"{sid} cited by no leaf")
        else:
            if s["cited_by"]:
                P(f"{sid} unused but cited by {', '.join(s['cited_by'])}")
            if not s["reason"]:
                P(f"{sid} unused without a reason")
        for lid in s["cited_by"]:
            if lid not in leaves:
                P(f"{sid} cited by {lid}, which does not exist")
            elif sid not in leaves[lid]["sources"]:
                P(f"{sid} cited by {lid}, but {lid} does not cite it")
    for lid, leaf in leaves.items():
        for sid in leaf["sources"]:
            if sid not in sections:
                P(f"{lid} cites {sid}, which does not exist")
            elif lid not in sections[sid]["cited_by"]:
                P(f"{lid} cites {sid}, which does not name it")

    # rule 3: placed / unplaced
    four = ("file", "sha256", "section", "placed_by")
    for pid, item in payload.items():
        if item["status"] == "placed":
            for k in four:
                if item[k] is None:
                    P(f"{pid} placed without {k}")
        else:
            for k in four:
                if item[k] is not None:
                    P(f"{pid} unplaced but has {k}")
            if not item["reason"]:
                P(f"{pid} unplaced without a reason")
        if item["section"] is not None and item["section"] not in sections:
            P(f"{pid} section {item['section']} does not exist")
        if item["doc"] not in docs:
            P(f"{pid} belongs to unknown document {item['doc']}")

    # rule 4: collisions
    def collide(kind, pairs):
        seen: dict[str, str] = {}
        for owner, path in pairs:
            if path is None:
                continue
            if path in seen:
                P(f"{path} used by {seen[path]} and {owner}")
            else:
                seen[path] = owner
    collide("payload", [(p["id"], p["file"]) for p in kb["payload"]])
    collide("text", [(s["id"], s["text_file"]) for s in kb["sections"]])
    collide("leaf", [(l["id"], l["path"]) for l in kb["leaves"]])

    # rule 5: references
    for lid, leaf in leaves.items():
        if leaf["topic"] not in topics:
            P(f"{lid} topic {leaf['topic']} does not exist")
        for r in leaf["related"]:
            if r not in leaves:
                P(f"{lid} related {r} does not exist")
        for r in leaf["payload"]:
            if r not in payload:
                P(f"{lid} payload {r} does not exist")
    for tid, topic in topics.items():
        for r in topic["leaves"]:
            if r not in leaves:
                P(f"topic {tid} leaf {r} does not exist")
            elif leaves[r]["topic"] != tid:
                P(f"topic {tid} lists {r}, whose topic is {leaves[r]['topic']}")
        listed = {l for l in topic["leaves"]}
        for lid, leaf in leaves.items():
            if leaf["topic"] == tid and lid not in listed:
                P(f"topic {tid} does not list its leaf {lid}")
    if "areas" in kb:                                     # kb/2
        areas = {a["id"]: a for a in kb["areas"]}
        for aid, area in areas.items():
            for r in area["topics"]:
                if r not in topics:
                    P(f"area {aid} topic {r} does not exist")
                elif topics[r]["area"] != aid:
                    P(f"area {aid} lists {r}, whose area is {topics[r]['area']}")
            if not area["folder"].endswith("-" + aid):
                P(f"area {aid} folder {area['folder']} is not named after it")
        for tid, topic in topics.items():
            if topic["area"] not in areas:
                P(f"topic {tid} area {topic['area']} does not exist")
                continue
            if tid not in areas[topic["area"]]["topics"]:
                P(f"area {topic['area']} does not list its topic {tid}")
            if tid != f"{topic['area']}/{tid.split('/', 1)[-1]}":
                P(f"topic {tid} is not <area>/<topic> of area {topic['area']}")
            if not topic["folder"].startswith(areas[topic["area"]]["folder"] + "/topics/") \
                    or not topic["folder"].endswith("-" + tid.split("/", 1)[-1]):
                P(f"topic {tid} folder {topic['folder']} is not in its area or not named after it")
        for lid, leaf in leaves.items():
            if leaf["topic"] in topics and \
                    leaf["path"] != f"{topics[leaf['topic']]['folder']}/{lid.rsplit('/', 1)[-1]}.md":
                P(f"{lid} path {leaf['path']} is not in its topic's folder")
    for c in kb["concepts"]:
        if c["leaf"] not in leaves:
            P(f"concept {c['name']!r} points at {c['leaf']}, which does not exist")
    for t in kb["terms"]:
        if t["leaf"] not in leaves:
            P(f"term {t['term']!r} points at {t['leaf']}, which does not exist")

    # rule 7, on disk: text files
    for s in kb["sections"]:
        path = root / s["text_file"]
        if not path.is_file():
            problems.append(f"{s['text_file']}: missing")
            continue
        data = path.read_bytes()
        if _sha(data) != s["text_sha256"]:
            problems.append(f"{s['text_file']}: sha256 differs from kb.json")
        if len(data.decode("utf-8", "replace").split()) != s["words"]:
            problems.append(f"{s['text_file']}: word count differs from kb.json")
        if data.count(b"\x00") != s["nul"]:
            problems.append(f"{s['text_file']}: {data.count(b'\x00')} NUL bytes, "
                            f"kb.json says {s['nul']}")
    for item in kb["payload"]:
        if item["file"] is None:
            continue
        path = root / item["file"]
        if not path.is_file():
            problems.append(f"{item['file']}: missing")
        elif _sha(path.read_bytes()) != item["sha256"]:
            problems.append(f"{item['file']}: sha256 differs from kb.json")

    _rederive(kb, problems, notes)


def _rederive(kb: dict, problems: list[str], notes: list[str]) -> None:
    """Inputs that still exist unchanged are re-read: split and payload bytes."""
    from ..hierarchy.glyphsections import split_glyphs
    for d in kb["documents"]:
        g = Path(d["glyphs"]["path"])
        if not g.is_file() or _sha(g.read_bytes()) != d["glyphs"]["sha256"]:
            notes.append(f"{d['doc']}: glyphs input gone or changed since the build; "
                         f"section split not re-derived")
            continue
        text = g.read_text(encoding="utf-8")
        lines = text.splitlines()
        derived = split_glyphs(text)
        mine = sorted((s for s in kb["sections"] if s["doc"] == d["doc"]),
                      key=lambda s: s["order"])
        if len(derived) != len(mine):
            problems.append(f"kb.json: {d['doc']} has {len(mine)} sections, its input "
                            f"splits into {len(derived)}")
            continue
        for s, ds in zip(mine, derived):
            body = "\n".join(lines[ds["line_start"] - 1:ds["line_end"]]) + "\n"
            if (s["line_start"], s["line_end"]) != (ds["line_start"], ds["line_end"]) \
                    or _sha(body.encode("utf-8")) != s["text_sha256"]:
                problems.append(f"kb.json: {s['id']} text is not its input's lines "
                                f"{ds['line_start']}-{ds['line_end']}")
    for d in kb["documents"]:
        dm = d["docmodel"]
        if dm is None:
            continue
        p = Path(dm["path"])
        if not p.is_file() or _sha(p.read_bytes()) != dm["sha256"]:
            notes.append(f"{d['doc']}: docmodel gone or changed since the build; "
                         f"payload bytes not re-derived")
            continue
        objects = {o.get("id"): o for o in json.loads(p.read_text(encoding="utf-8"))
                   .get("objects", [])}
        for item in kb["payload"]:
            if item["doc"] != d["doc"] or item["sha256"] is None:
                continue
            try:
                data = "\n".join(objects[m]["props"][item["field"]]
                                 for m in item["members"])
            except KeyError:
                problems.append(f"kb.json: {item['id']} members not in the docmodel")
                continue
            if _sha(data.encode("utf-8")) != item["sha256"]:
                problems.append(f"kb.json: {item['id']} is not the docmodel's bytes")


# --------------------------------------------------------------------------
# Frontmatter == kb.json, and reachability
# --------------------------------------------------------------------------

def _related_groups(kb: dict, key, own) -> list[str]:
    """The groups (topics or areas) of a group's leaves' related leaves, other
    than itself, sorted. `key` maps a leaf to its group."""
    by_id = {l["id"]: l for l in kb["leaves"]}
    return sorted({key(by_id[r]) for l in kb["leaves"] if key(l) == own
                   for r in l["related"] if r in by_id and key(by_id[r]) != own})


def check_against_kb(kb: dict, mds: dict[str, MdFile], graph: dict[str, set[str]],
                     problems: list[str], spec: C.Spec = C.CURRENT) -> None:
    leaves = {l["path"]: l for l in kb["leaves"]}
    topics = {t["folder"] + "/README.md": t for t in kb["topics"]}
    areas = {a["folder"] + "/README.md": a for a in kb.get("areas", [])}
    topic_of = {t["id"]: t for t in kb["topics"]}
    area_key = lambda l: topic_of.get(l["topic"], {}).get("area")
    def differ(md, key, want):
        if md.front is not None and md.front.get(key) != want:
            problems.append(f"{md.rel}: frontmatter {key} differs from kb.json")
    for rel, md in mds.items():
        if md.kind == "leaf":
            leaf = leaves.get(rel)
            if leaf is None:
                problems.append(f"{rel}: leaf not in kb.json")
                continue
            for key in ("id", "title", "topic", "concepts", "sources", "payload", "related"):
                differ(md, key, leaf[key])
        elif md.kind == "topic":
            topic = topics.get(rel)
            if topic is None:
                problems.append(f"{rel}: topic not in kb.json")
                continue
            concepts = sorted({c for l in kb["leaves"] if l["topic"] == topic["id"]
                               for c in l["concepts"]})
            related = _related_groups(kb, lambda l: l["topic"], topic["id"])
            for key, want in (("id", topic["id"]), ("title", topic["title"]),
                              ("leaves", topic["leaves"]), ("concepts", concepts),
                              ("related", related)):
                differ(md, key, want)
        elif md.kind == "area":
            area = areas.get(rel)
            if area is None:
                problems.append(f"{rel}: area not in kb.json")
                continue
            for key, want in (("id", area["id"]), ("title", area["title"]),
                              ("topics", area["topics"]),
                              ("related", _related_groups(kb, area_key, area["id"]))):
                differ(md, key, want)
        elif md.kind == "readme":
            differ(md, "kb", kb["kb"])
            if spec is C.KB1:
                differ(md, "documents", [d["doc"] for d in kb["documents"]])
                differ(md, "topics", [t["id"] for t in kb["topics"]])
            else:
                differ(md, "areas", [a["id"] for a in kb.get("areas", [])])
        elif md.kind == "sources_list":
            differ(md, "kb", kb["kb"])
            differ(md, "documents", len(kb["documents"]))
        elif md.kind == "source":
            doc = Path(rel).stem
            d = next((x for x in kb["documents"] if x["doc"] == doc), None)
            if d is None:
                problems.append(f"{rel}: document not in kb.json")
                continue
            differ(md, "doc", doc)
            differ(md, "title", d["title"])
            differ(md, "sections", sum(1 for s in kb["sections"] if s["doc"] == doc))
    for a in kb.get("areas", []):
        if a["folder"] + "/README.md" not in mds:
            problems.append(f"{a['folder']}/README.md: missing")
    for t in kb["topics"]:
        if t["folder"] + "/README.md" not in mds:
            problems.append(f"{t['folder']}/README.md: missing")
    for l in kb["leaves"]:
        if l["path"] not in mds:
            problems.append(f"{l['path']}: missing")
    for d in kb["documents"]:
        if f"sources/{d['doc']}.md" not in mds:
            problems.append(f"sources/{d['doc']}.md: missing")
        elif spec is C.KB2 and "SOURCES.md" not in {
                src for src, ts in graph.items() if f"sources/{d['doc']}.md" in ts}:
            problems.append(f"sources/{d['doc']}.md: not linked from SOURCES.md")

    linked_from = lambda target: {src for src, ts in graph.items() if target in ts}
    leaf_paths = {l["path"] for l in kb["leaves"]}
    for a in kb.get("areas", []):
        rel = a["folder"] + "/README.md"
        if "README.md" not in linked_from(rel):
            problems.append(f"{rel}: not linked from README.md")
    for t in kb["topics"]:
        rel = t["folder"] + "/README.md"
        parent = "README.md" if spec is C.KB1 else t["folder"].split("/topics/")[0] + "/README.md"
        if parent not in linked_from(rel):
            problems.append(f"{rel}: not linked from {parent}")
    for l in kb["leaves"]:
        sources = linked_from(l["path"])
        topic_readme = l["path"].rsplit("/", 1)[0] + "/README.md"
        if topic_readme not in sources:
            problems.append(f"{l['path']}: not linked from {topic_readme}")
        if not any(s.startswith("index/") for s in sources):
            problems.append(f"{l['path']}: not linked from any index entry")
    for p in kb["payload"]:
        if p["file"] and not any(s in leaf_paths for s in linked_from(p["file"])):
            problems.append(f"{p['file']}: not linked from any leaf")
    for s in kb["sections"]:
        if f"sources/{s['doc']}.md" not in linked_from(s["text_file"]):
            problems.append(f"{s['text_file']}: not linked from sources/{s['doc']}.md")
        for lid in s["cited_by"]:
            leaf = next((l for l in kb["leaves"] if l["id"] == lid), None)
            if leaf and leaf["path"] not in linked_from(s["text_file"]):
                problems.append(f"{s['text_file']}: not linked from citing leaf {leaf['path']}")


# --------------------------------------------------------------------------

def check(root: Path) -> Report:
    root = Path(root)
    spec = detect(root)
    report = Report(format=spec.format)
    problems = report.violations
    files = check_tree(root, problems, spec)
    report.files = len(files)
    kb = _load_json(root, "kb.json", spec.schema, problems)
    manifest = _load_json(root, "manifest.json", "manifest.schema.json", problems)
    if manifest is not None and manifest.get("format") != spec.format:
        problems.append(f"manifest.json: format {manifest.get('format')}, the tree is "
                        f"{spec.format}")
    mds: dict[str, MdFile] = {}
    graph: dict[str, set[str]] = {}
    for rel in files:
        if not rel.endswith(".md") or kind_of(rel, spec) is None:
            continue
        md = parse_md(rel, (root / rel).read_text(encoding="utf-8"), problems, spec)
        mds[rel] = md
        check_md(md, problems, spec)
        check_links(root, md, problems, graph, spec)
    if kb is not None:
        check_kb_json(root, kb, problems, report.notes)
        check_against_kb(kb, mds, graph, problems, spec)
    return report


def form_of(root: Path) -> dict:
    """A knowledge base's form, without its names or content.

    Two trees built from unrelated documents must have equal forms: the same
    root entries, and for every file kind the same headings and frontmatter
    keys. Which optional folders exist (payload/) depends on the content.
    """
    root = Path(root)
    spec = detect(root)
    kinds: dict[str, set] = {}
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        kind = kind_of(rel, spec)
        if kind is None:
            continue
        md = parse_md(rel, path.read_text(encoding="utf-8"), [], spec)
        kinds.setdefault(kind, set()).add(
            (tuple(h for h, _ in md.sections), tuple(md.front or ())))
    present = sorted(p.name for p in root.iterdir()
                     if p.name in list(spec.root_files) + spec.required_dirs + ["text"])
    letters = {folder: sorted(p.stem for p in (root / folder).glob("*.md"))
               for folder in ("index", "glossary")}
    return {"format": spec.format, "root": present, "letters": letters,
            "kinds": {k: sorted(v) for k, v in sorted(kinds.items())}}
