"""The knowledge-base formats, as constants: kb/1 (docs/kb-format/kb-1.md)
and kb/2 (docs/kb-format/kb-2.md).

This module holds each contract's mechanical parts so the generator and the
checker read one copy. The specs stay the source: tests compare the heading
tables and the reading protocols here against the documents. Nothing here
knows what a knowledge base is about.

`kb build` writes `CURRENT` (kb/2). `kb check` checks a tree against the spec
its `format` names, so a kb/1 tree built before kb/2 stays checkable.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional

#: The 27 letter files under index/ and glossary/.
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["other"]

SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
SLUG_RE = re.compile(rf"^{SLUG}$")
SLUG_MAX = 60
DOC_SLUG_MAX = 56

#: The files at the root.
ROOT_FILES = ["README.md", "INDEX.md", "GLOSSARY.md", "kb.json", "manifest.json"]

#: The measured tokens-per-word of the project (summarize.TOKENS_PER_WORD).
TOKENS_PER_WORD = 1.441

PAYLOAD_KINDS = ["equation", "code", "pseudocode"]
PAYLOAD_LABELS = ["verified", "unverified", "line numbers not separated",
                  "detected pseudocode"]

#: Inline a payload file's content under its leaf line up to this many lines.
INLINE_MAX_LINES = 40

_SID = rf"{SLUG}#[0-9]+"
_PID = rf"{SLUG}#[^#\s`]+"
_LABELS = "(?:" + "|".join(map(re.escape, PAYLOAD_LABELS)) + ")"
_LETTER_LINK = r"(?:[A-Z]|Other)"
_NN = r"[0-9]{2}"


@dataclass(frozen=True)
class Spec:
    """One format's mechanical parts."""
    format: str
    root_dirs: list[str]
    required_dirs: list[str]
    headings: dict[str, list[str]]
    frontmatter: dict[str, list[str]]
    budgets: dict[str, int]
    protocol: str
    line_forms: dict[tuple[str, str], re.Pattern]
    #: The leaf folder relative to the root, as a regex: where leaves live.
    leaf_dir: str
    schema: str
    max_areas: Optional[int] = None
    max_topics_per_area: Optional[int] = None
    max_leaves_per_topic: int = 8
    fences_allowed: frozenset = field(default_factory=lambda: frozenset({("leaf", "Payload")}))


def _forms(up: str, leaf_dir: str, extra: dict) -> dict:
    """The line forms both formats share; `up` climbs from a leaf to the root."""
    u = re.escape(up)
    forms = {
        ("readme", "Sources"): rf"^- \[{SLUG}\]\(sources/{SLUG}\.md\) — .*$",
        ("index", "Letters"): rf"^- \[{_LETTER_LINK}\]\(index/[a-z]+\.md\) — [0-9]+ entries$",
        ("glossary", "Letters"): rf"^- \[{_LETTER_LINK}\]\(glossary/[a-z]+\.md\) — [0-9]+ terms$",
        ("index_letter", "Index"): rf"^- .+ → \[.+\]\(\.\./{leaf_dir}/{SLUG}\.md\)$",
        ("glossary_letter", "Glossary"):
            rf"^- \*\*.+\*\* — .* → \[.+\]\(\.\./{leaf_dir}/{SLUG}\.md\)$",
        ("topic", "Leaves"): rf"^- \[.+\]\({SLUG}\.md\) — .+$",
        ("leaf", "Key concepts"): r"^- \S.*$",
        ("leaf", "Payload"):
            rf"^- `{_PID}` (?:equation|code|pseudocode) · {_LABELS}(?:, {_LABELS})* · "
            rf"from `{_SID}` → \[file\]\({u}payload/{SLUG}/[^)\s]+\)$",
        ("leaf", "Sources"):
            rf"^- `{_SID}` .* → \[text, [0-9]+ words\]\({u}text/{SLUG}/[0-9]+\.txt\)"
            rf" · \[{SLUG}\]\({u}sources/{SLUG}\.md\)$",
        ("leaf", "See also"): r"^- \[.+\]\([^)\s]+\.md\)$",
        ("source", "Sections"):
            rf"^- `{_SID}` p\.[0-9]+ l\.[0-9]+–[0-9]+ .* → "
            rf"\[text, [0-9]+ words\]\(\.\./text/{SLUG}/[0-9]+\.txt\) · .+$",
        ("source", "Unused sections"):
            rf"^- `{_SID}` p\.[0-9]+ l\.[0-9]+–[0-9]+ .* → "
            rf"\[text, [0-9]+ words\]\(\.\./text/{SLUG}/[0-9]+\.txt\) — .+$",
    }
    forms.update(extra)
    return {k: re.compile(v) for k, v in forms.items()}


_LEAF = ["Use when", "Summary", "Key concepts", "Method", "Payload", "Pitfalls",
         "Sources", "See also"]
_COMMON_HEADINGS = {
    "index": ["Letters"], "glossary": ["Letters"], "index_letter": ["Index"],
    "glossary_letter": ["Glossary"],
    "topic": ["Purpose", "Leaves", "How the approaches differ", "Related topics"],
    "leaf": _LEAF, "source": ["Sections", "Unused sections"],
}
_COMMON_FRONT = {
    "index": ["format", "kb", "letters"],
    "glossary": ["format", "kb", "letters"],
    "index_letter": ["format", "kb", "letter", "entries"],
    "glossary_letter": ["format", "kb", "letter", "terms"],
    "topic": ["format", "id", "title", "leaves", "concepts", "related"],
    "leaf": ["format", "id", "title", "topic", "concepts", "sources", "payload", "related"],
    "source": ["format", "doc", "title", "sections"],
}
_COMMON_BUDGETS = {"readme": 1500, "topic": 1500, "leaf": 2500, "index_letter": 4000,
                   "glossary_letter": 4000, "index": 500, "glossary": 500, "source": 4000}

_KB1_LEAF_DIR = rf"topics/{_NN}-{SLUG}"
KB1 = Spec(
    format="kb/1",
    root_dirs=["index", "glossary", "topics", "payload", "text", "sources"],
    required_dirs=["index", "glossary", "topics", "sources"],
    headings={"readme": ["Purpose", "Topics", "How to read this knowledge base", "Sources"],
              **_COMMON_HEADINGS},
    frontmatter={"readme": ["format", "kb", "title", "documents", "topics"], **_COMMON_FRONT},
    budgets=dict(_COMMON_BUDGETS),
    protocol="""\
1. Read this file and INDEX.md. Do not open other files yet.
2. Pick the one topic, or the one index entry, that matches the question.
3. Open that topic's README.md, then at most two of its leaves.
4. Open a payload file only when the exact equation, code or pseudocode is needed.
5. Follow See also or Related topics only if the answer is still missing.
6. Open a section's text (text/<doc>/<order>.txt) only when the summary is not enough.
7. Open sources/<doc>.md only to cite or check where a statement comes from.""",
    line_forms=_forms("../../", _KB1_LEAF_DIR, {
        ("readme", "Topics"): rf"^- \[.+\]\(topics/{_NN}-{SLUG}/README\.md\) — .+$",
        ("topic", "Related topics"): rf"^- \[.+\]\(\.\./{_NN}-{SLUG}/README\.md\)$",
    }),
    leaf_dir=_KB1_LEAF_DIR,
    schema="kb.schema.json",
)

_KB2_LEAF_DIR = rf"areas/{_NN}-{SLUG}/topics/{_NN}-{SLUG}"
KB2 = Spec(
    format="kb/2",
    root_dirs=["index", "glossary", "areas", "payload", "text", "sources"],
    required_dirs=["index", "glossary", "areas", "sources"],
    headings={"readme": ["Purpose", "Areas", "How to read this knowledge base", "Sources"],
              "area": ["Purpose", "Topics", "How the topics differ", "Related areas"],
              **_COMMON_HEADINGS},
    frontmatter={"readme": ["format", "kb", "title", "documents", "areas"],
                 # No `concepts`: an area's union overflowed its budget (one area of 66 leaves: 275
                 # names, 1836 tokens); topics, leaves and the index carry them.
                 "area": ["format", "id", "title", "topics", "related"],
                 **_COMMON_FRONT},
    budgets={**_COMMON_BUDGETS, "area": 1500},
    protocol="""\
1. Read this file and INDEX.md. Do not open other files yet.
2. Pick the one area, or the one index entry, that matches the question.
3. Open that area's README.md and pick its one matching topic.
4. Open that topic's README.md, then at most two of its leaves.
5. Open a payload file only when the exact equation, code or pseudocode is needed.
6. Follow See also, Related topics or Related areas only if the answer is still missing.
7. Open a section's text (text/<doc>/<order>.txt) only when the summary is not enough.
8. Open sources/<doc>.md only to cite or check where a statement comes from.""",
    line_forms=_forms("../../../../", _KB2_LEAF_DIR, {
        ("readme", "Areas"): rf"^- \[.+\]\(areas/{_NN}-{SLUG}/README\.md\) — .+$",
        ("area", "Topics"): rf"^- \[.+\]\(topics/{_NN}-{SLUG}/README\.md\) — .+$",
        ("area", "Related areas"): rf"^- \[.+\]\(\.\./{_NN}-{SLUG}/README\.md\)$",
        ("topic", "Related topics"):
            rf"^- \[.+\]\(\.\./(?:\.\./\.\./{_NN}-{SLUG}/topics/)?{_NN}-{SLUG}/README\.md\)$",
    }),
    leaf_dir=_KB2_LEAF_DIR,
    schema="kb-2.schema.json",
    max_areas=12,
    max_topics_per_area=12,
)

SPECS = {"kb/1": KB1, "kb/2": KB2}
#: What `kb build` writes.
CURRENT = KB2


def letter_title(kind: str, letter: str) -> str:
    word = "Index" if kind == "index_letter" else "Glossary"
    return f"# {word}: {'Other' if letter == 'other' else letter.upper()}"


def letter_of(text: str) -> str:
    """The letter file an entry belongs to: its first letter, case-folded."""
    for ch in unicodedata.normalize("NFKD", text).casefold():
        if "a" <= ch <= "z":
            return ch
        if ch.isalpha():
            return "other"
    return "other"


def slugify(text: str, limit: int = SLUG_MAX) -> str:
    """kb-1 section 1a's rule, for any name: ASCII, lower case, dashes."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")
    if len(slug) > limit:
        cut = slug[:limit]
        slug = cut.rsplit("-", 1)[0] if "-" in cut else cut
    return slug.strip("-")


def doc_slug(folder: str) -> str:
    """One document's slug, before collisions (kb-1 section 1a)."""
    return slugify(folder, DOC_SLUG_MAX) or "doc"


def doc_slugs(folders: Iterable[str]) -> dict[str, str]:
    """Every folder's slug, collisions numbered in code-point order of folders."""
    taken: set[str] = set()
    out: dict[str, str] = {}
    for folder in sorted(set(folders)):
        base = doc_slug(folder)
        slug, n = base, 1
        while slug in taken:
            n += 1
            slug = f"{base}-{n}"
        taken.add(slug)
        out[folder] = slug
    return out


def tokens(text: str) -> int:
    return round(len(text.split()) * TOKENS_PER_WORD)
