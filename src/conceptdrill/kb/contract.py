"""kb/1, the knowledge-base format, as constants (docs/kb-format/kb-1.md).

This module holds the contract's mechanical parts so the generator and the
checker read one copy. The spec stays the source: tests compare the heading
table and the reading protocol here against `kb-1.md`. Nothing here knows what
a knowledge base is about.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

FORMAT = "kb/1"

#: The 27 letter files under index/ and glossary/.
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["other"]

SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
SLUG_RE = re.compile(rf"^{SLUG}$")
SLUG_MAX = 60
DOC_SLUG_MAX = 56

#: The files at the root, and the folders allowed beside them.
ROOT_FILES = ["README.md", "INDEX.md", "GLOSSARY.md", "kb.json", "manifest.json"]
ROOT_DIRS = ["index", "glossary", "topics", "payload", "text", "sources"]
REQUIRED_DIRS = ["index", "glossary", "topics", "sources"]

#: `##` headings in order, per file kind (kb-1 section 4).
HEADINGS: dict[str, list[str]] = {
    "readme": ["Purpose", "Topics", "How to read this knowledge base", "Sources"],
    "index": ["Letters"],
    "glossary": ["Letters"],
    "index_letter": ["Index"],
    "glossary_letter": ["Glossary"],
    "topic": ["Purpose", "Leaves", "How the approaches differ", "Related topics"],
    "leaf": ["Use when", "Summary", "Key concepts", "Method", "Payload",
             "Pitfalls", "Sources", "See also"],
    "source": ["Sections", "Unused sections"],
}

#: Frontmatter keys, exactly, per file kind (kb-1 section 3).
FRONTMATTER: dict[str, list[str]] = {
    "readme": ["format", "kb", "title", "documents", "topics"],
    "index": ["format", "kb", "letters"],
    "glossary": ["format", "kb", "letters"],
    "index_letter": ["format", "kb", "letter", "entries"],
    "glossary_letter": ["format", "kb", "letter", "terms"],
    "topic": ["format", "id", "title", "leaves", "concepts", "related"],
    "leaf": ["format", "id", "title", "topic", "concepts", "sources", "payload",
             "related"],
    "source": ["format", "doc", "title", "sections"],
}

#: Token budgets per file kind (kb-1 section 7), at words x TOKENS_PER_WORD.
BUDGETS: dict[str, int] = {
    "readme": 1500, "topic": 1500, "leaf": 2500, "index_letter": 4000,
    "glossary_letter": 4000, "index": 500, "glossary": 500, "source": 4000,
}

#: The measured tokens-per-word of the project (summarize.TOKENS_PER_WORD).
TOKENS_PER_WORD = 1.441

PAYLOAD_KINDS = ["equation", "code", "pseudocode"]
PAYLOAD_LABELS = ["verified", "unverified", "line numbers not separated",
                  "detected pseudocode"]

#: Inline a payload file's content under its leaf line up to this many lines.
INLINE_MAX_LINES = 40

#: kb-1 section 8, verbatim.
PROTOCOL = """\
1. Read this file and INDEX.md. Do not open other files yet.
2. Pick the one topic, or the one index entry, that matches the question.
3. Open that topic's README.md, then at most two of its leaves.
4. Open a payload file only when the exact equation, code or pseudocode is needed.
5. Follow See also or Related topics only if the answer is still missing.
6. Open a section's text (text/<doc>/<order>.txt) only when the summary is not enough.
7. Open sources/<doc>.md only to cite or check where a statement comes from."""

_SID = rf"{SLUG}#[0-9]+"
_PID = rf"{SLUG}#[^#\s`]+"
_LABELS = "(?:" + "|".join(map(re.escape, PAYLOAD_LABELS)) + ")"
_LETTER_LINK = r"(?:[A-Z]|Other)"

#: Line forms (kb-1 section 5): (kind, heading) -> the one regex every
#: non-blank line there must match, unless the section is `None.`.
LINE_FORMS: dict[tuple[str, str], re.Pattern] = {k: re.compile(v) for k, v in {
    ("readme", "Topics"): rf"^- \[.+\]\(topics/[0-9]{{2}}-{SLUG}/README\.md\) — .+$",
    ("readme", "Sources"): rf"^- \[{SLUG}\]\(sources/{SLUG}\.md\) — .*$",
    ("index", "Letters"): rf"^- \[{_LETTER_LINK}\]\(index/[a-z]+\.md\) — [0-9]+ entries$",
    ("glossary", "Letters"): rf"^- \[{_LETTER_LINK}\]\(glossary/[a-z]+\.md\) — [0-9]+ terms$",
    ("index_letter", "Index"):
        rf"^- .+ → \[.+\]\(\.\./topics/[0-9]{{2}}-{SLUG}/{SLUG}\.md\)$",
    ("glossary_letter", "Glossary"):
        rf"^- \*\*.+\*\* — .* → \[.+\]\(\.\./topics/[0-9]{{2}}-{SLUG}/{SLUG}\.md\)$",
    ("topic", "Leaves"): rf"^- \[.+\]\({SLUG}\.md\) — .+$",
    ("topic", "Related topics"): rf"^- \[.+\]\(\.\./[0-9]{{2}}-{SLUG}/README\.md\)$",
    ("leaf", "Key concepts"): r"^- \S.*$",
    ("leaf", "Payload"):
        rf"^- `{_PID}` (?:equation|code|pseudocode) · {_LABELS}(?:, {_LABELS})* · "
        rf"from `{_SID}` → \[file\]\(\.\./\.\./payload/{SLUG}/[^)\s]+\)$",
    ("leaf", "Sources"):
        rf"^- `{_SID}` .* → \[text, [0-9]+ words\]\(\.\./\.\./text/{SLUG}/[0-9]+\.txt\)"
        rf" · \[{SLUG}\]\(\.\./\.\./sources/{SLUG}\.md\)$",
    ("leaf", "See also"): r"^- \[.+\]\([^)\s]+\.md\)$",
    ("source", "Sections"):
        rf"^- `{_SID}` p\.[0-9]+ l\.[0-9]+–[0-9]+ .* → "
        rf"\[text, [0-9]+ words\]\(\.\./text/{SLUG}/[0-9]+\.txt\) · .+$",
    ("source", "Unused sections"):
        rf"^- `{_SID}` p\.[0-9]+ l\.[0-9]+–[0-9]+ .* → "
        rf"\[text, [0-9]+ words\]\(\.\./text/{SLUG}/[0-9]+\.txt\) — .+$",
}.items()}

#: Where a fenced block may appear: only under a leaf's Payload heading.
FENCES_ALLOWED = {("leaf", "Payload")}


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
