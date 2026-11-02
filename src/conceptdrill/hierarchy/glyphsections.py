"""Sections from a `<doc>.glyphs.md`, summarised per section (T-0117).

The docmodel is the reliable input when its sections survive, but in the 21
CCL papers twelve docmodels had lost them (`2016icpr` keeps 2 sections,
`2019caip_labeling` 1) while the numbered headings still stand as lines of the
glyphs markdown. Two-column pages also put table cells and pseudocode steps on
lines of their own ("1 Pixels", "6 Block R must undergo Union"), so a numbered
line is accepted as a heading only when it continues the numbering of the last
heading accepted -- and a markdown `#` heading outranks a plain line that
carries the same number.

The rules are those of the scratch script that produced `labeling-21`
(2026-10-29), ported unchanged so the command reproduces its split.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
         "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
NUMBERED = re.compile(r"^(?P<num>[IVX]{1,4}|\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+"
                      r"(?P<title>[A-Z][^.!?]{2,90})$")
LETTERED = re.compile(r"^(?P<num>[A-H])\.\s+(?P<title>[A-Z][^.!?]{2,90})$")
ABSTRACT = re.compile(r"^(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\b[\s.:—–-]*"
                      r"(?P<rest>.*)$", re.IGNORECASE)
UNNUMBERED = re.compile(r"^(?:conclusions?|acknowledge?ments?|discussion)\s*$",
                        re.IGNORECASE)
STOP = re.compile(r"^(?:references|bibliography|literature cited)\s*$",
                  re.IGNORECASE)
NOISE = (re.compile(r"!\[[^\]]*\]\([^)]*\)"), re.compile(r"<!--.*?-->"))

#: Sections with less text than this are recorded as skipped, not summarised:
#: below it the text is a heading with a stray line, not a section.
MIN_WORDS = 40


def _clean(line: str) -> str:
    for pattern in NOISE:
        line = pattern.sub(" ", line)
    return re.sub(r"\s+", " ", line.lstrip("#").strip())


def _number(match, lettered: bool, scheme: str):
    n = match.group("num")
    if lettered:
        return None if scheme != "roman" else ("letter", ord(n) - 64)
    if n in ROMAN:
        return ("roman", ROMAN[n])
    if n.isdigit() or "." in n:
        return ("arabic",) + tuple(int(x) for x in n.split("."))
    return None


def follows(prev: Optional[tuple], cur: tuple, jump: int = 3) -> bool:
    """Does `cur` plausibly continue the numbering after `prev`?

    Garbled two-column pages lose headings, so a jump of up to `jump` at the
    top level (2 below it) is allowed, and so is entering a new section's
    children when the section's own heading was lost (`2.1` right after `1`).
    The first heading may carry a chapter prefix (`35.1`).
    """
    if prev is None:
        return cur[-1] == 1 or (len(cur) == 1 and cur[0] <= 2)
    if len(cur) == len(prev) + 1 and cur[:-1] == prev and cur[-1] == 1:
        return True
    for k in range(min(len(prev), len(cur))):
        if cur[:k] == prev[:k] and 1 <= cur[k] - prev[k] <= (jump if k == 0 else 2) \
                and all(x == 1 for x in cur[k + 1:]):
            return True
    return False


def split_glyphs(text: str) -> list[dict]:
    """The sections of one glyphs markdown, in document order.

    Each is `{"title", "number", "page", "line_start", "line_end", "text",
    "own_words", "text_words"}`. A parent's `text` includes its subsections, as
    `summarize_tree` feeds `subtree_text`. `page` counts the `---` page breaks
    of glyphs.md; lines are 1-based and inclusive, heading to last line. They
    are kept so a section can later be mapped to pdfdrill's Section ids by page
    and title (drillgroup T-0105 via T-0077) -- conceptdrill invents no
    pdfdrill ids.
    """
    raw = text.splitlines()
    marked = [l.lstrip().startswith("#") for l in raw]
    lines = [_clean(l) for l in raw]
    roman = sum(bool(re.match(r"^(I|II|III|IV|V)\.\s+[A-Z]{3}", l)) for l in lines)
    arabic = sum(bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][a-z]", l)) for l in lines)
    scheme = "roman" if roman >= 2 and roman >= arabic else "arabic"
    # Numbers carried by markdown headings: a plain line with the same number
    # is a table cell or a pseudocode step, never the heading itself.
    md_numbers = {m.group("num") for l, mk in zip(lines, marked) if mk
                  for m in [NUMBERED.match(l)] if m}
    numbered_doc = len(md_numbers) >= 2 or arabic + roman >= 3
    sections: list[dict] = []
    cur, prev, top, page = None, None, None, 0
    for lineno, (line, mk) in enumerate(zip(lines, marked), 1):
        if line == "---":
            page += 1
            continue
        if not line:
            continue
        bare = re.sub(r"^(?:[IVX]+|\d+)\.?\s+", "", re.sub(r"\s+\d+$", "", line))
        if STOP.match(bare) and sections:
            cur = None          # skip the references, keep looking for headings
            continue
        head = None
        m = ABSTRACT.match(line)
        if m and not sections:
            head = ("Abstract", (), m.group("rest"))
        elif UNNUMBERED.match(line) and sections:
            head = (line.title(), (), "")
        elif mk and not numbered_doc and sections and len(line.split()) <= 8 \
                and not re.match(r"^(?:fig|table|keywords|\d)", line, re.IGNORECASE):
            head = (line, (), "")
        else:
            m = NUMBERED.match(line) or LETTERED.match(line)
            title = m.group("title").strip() if m else ""
            if m and len(title.split()) <= 10 and sum(c.isdigit() for c in title) <= 2 \
                    and not re.search(r"[;=]", title) \
                    and (mk or m.group("num") not in md_numbers):
                n = _number(m, m.re is LETTERED, scheme)
                if n and (n[0] == scheme or n[0] == "letter"):
                    tup = (top, n[1]) if n[0] == "letter" else n[1:]
                    if n[0] == "letter" and top is None:
                        tup = None
                    if tup and follows(prev, tup, 5 if mk else 3):
                        prev = tup
                        if n[0] != "letter":
                            top = tup[0]
                        head = (f"{m.group('num')}. {title}", tup, "")
        if head:
            cur = {"title": head[0], "number": list(head[1]),
                   "page": max(page, 1), "line_start": lineno, "line_end": lineno,
                   "lines": [head[2]] if head[2] else []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
            cur["line_end"] = lineno
    split = "numbered"
    if not sections:
        sections = _markdown_split(lines, marked)
        split = "markdown"
        if not sections:
            sections, split = _page_split(lines), "page"
    for i, s in enumerate(sections):
        own = "\n".join(s["lines"])
        n = tuple(s["number"])
        kids = []
        if n:
            for t in sections[i + 1:]:
                tn = tuple(t["number"])
                if len(tn) > len(n) and tn[:len(n)] == n:
                    kids.append(f"{t['title']}\n" + "\n".join(t["lines"]))
                else:
                    break
        s["text"] = own + ("\n\n" + "\n\n".join(kids) if kids else "")
        s["own_words"] = len(own.split())
        s["text_words"] = len(s["text"].split())
        s["split"] = split     # which rule made it: numbered, markdown or page
        del s["lines"]
    return sections


#: A markdown heading that reads as words: letters, digits, spaces and plain
#: punctuation, at least one word of four letters, at most eight words.
_WORDY = re.compile(r"^[\w ,:;'’()&/.-]+$")


def _clean_heading(text: str) -> bool:
    words = text.split()
    return (1 <= len(words) <= 8 and bool(_WORDY.match(text))
            and any(sum(c.isalpha() for c in w) >= 4 for w in words))


def _markdown_split(lines: list[str], marked: list[bool]) -> list[dict]:
    """For documents that number nothing: their clean markdown headings.

    The first level-1 heading is the document title, not a section; a heading
    text that repeats is a running page header; math fragments, figure labels
    and author lines do not read as words. Fewer than two headings left is no
    structure at all, and the page split takes over -- as it does when the
    headings cover less than half of the document's non-blank lines.
    """
    heads = [(i, l) for i, (l, mk) in enumerate(zip(lines, marked)) if mk and l]
    counts: dict[str, int] = {}
    for _, l in heads:
        counts[l.casefold()] = counts.get(l.casefold(), 0) + 1
    title = next((l for i, l in heads), None)
    sections, cur, page = [], None, 0
    keep = {i for i, l in heads if l != title and counts[l.casefold()] == 1
            and _clean_heading(l)}
    if len([i for i in keep if not STOP.match(lines[i])]) < 2:
        return []
    for lineno, line in enumerate(lines, 1):
        if line == "---":
            page += 1
            continue
        if not line:
            continue
        i = lineno - 1
        if i in keep and STOP.match(line):
            cur = None
            continue
        if i in keep:
            cur = {"title": line, "number": [], "page": max(page, 1),
                   "line_start": lineno, "line_end": lineno, "lines": []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
            cur["line_end"] = lineno
    covered = sum(len(c["lines"]) + 1 for c in sections)
    if covered * 2 < sum(1 for l in lines if l and l != "---"):
        return []
    return sections


def _page_split(lines: list[str]) -> list[dict]:
    """The last resort: one section per page of glyphs.md, titled by number."""
    sections, cur, page = [], None, 0
    for lineno, line in enumerate(lines, 1):
        if line == "---":
            page += 1
            cur = None
            continue
        if not line:
            continue
        if cur is None:
            cur = {"title": f"Page {max(page, 1)}", "number": [], "page": max(page, 1),
                   "line_start": lineno, "line_end": lineno, "lines": [line]}
            sections.append(cur)
        else:
            cur["lines"].append(line)
            cur["line_end"] = lineno
    return sections


def summarise_sections(doc: str, sections: list[dict], summarizer, *,
                       min_words: int = MIN_WORDS,
                       log: Callable[[str], None] = print) -> dict:
    """Summary, abstraction and label for every section long enough to have them.

    Short sections are kept and marked `skipped`, never dropped: the record
    accounts for every section the split found.

    Every section gets `id` (`<doc>#<order>`, the span id the summariser sees)
    and `order`: the stable key a reading folder links to (T-0105). With the
    split unchanged, the same glyphs.md gives the same ids.
    """
    for i, s in enumerate(sections):
        s["id"], s["order"] = f"{doc}#{i}", i
        if s["text_words"] < min_words:
            s["skipped"] = f"only {s['text_words']} words of text"
            continue
        r = summarizer.summarize(s["id"], s["title"], s["text"])
        s.update(abstraction=r.abstraction, summary=r.summary, label=r.label,
                 model=r.model, error=r.error or None,
                 more_concepts=[{"label": c.label, "abstraction": c.abstraction,
                                 "summary": c.summary} for c in r.siblings])
        log(f"{doc:20s} {i + 1:2d}/{len(sections)} {s['title'][:40]:40s} "
            f"{'ERROR ' + r.error if r.error else 'ok'}")
    record = {"doc": doc, "sections": sections}
    usage = getattr(summarizer, "usage", None)
    if usage is not None:
        record["usage"] = dict(usage)
    return record


def render_markdown(record: dict) -> str:
    """The plain-reading form: per section its abstract, summary and label."""
    out = [f"# {record['doc']}", ""]
    for s in record["sections"]:
        out += [f"## {s['title']}", ""]
        if s.get("skipped"):
            out += [f"_Not summarised: {s['skipped']}._", ""]
            continue
        if s.get("error"):
            out += [f"_Error: {s['error']}_", ""]
            continue
        out += [f"**Abstract.** {s['abstraction']}", "",
                f"**Summary.** {s['summary']}", "",
                f"_Label:_ {s['label']}", ""]
        for c in s.get("more_concepts") or []:
            out += [f"- further concept — **Abstract.** {c['abstraction']} "
                    f"**Summary.** {c['summary']}", ""]
    return "\n".join(out)
