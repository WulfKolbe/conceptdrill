"""1.2 Glossary, definitions, and theorem environments.

Highest structural weight in the system: a term the author explicitly defined is
the strongest possible signal that it is a concept of the document.

Three sources, in descending reliability:

  1. Blocks the parser already typed `definition` / `theorem`.
  2. `TERM — definition` entries inside a section whose title names a term list
     (glossary, nomenclature, notation, acronyms, ...).
  3. Schwartz-Hearst acronym pairs in running prose — "Convolutional Neural
     Network (CNN)" defines CNN, and the long form becomes the description.

The Schwartz-Hearst implementation matches the one in the Semantic Compiler's
`semantic/concepts.py`, so both tools agree on what counts as a defined acronym.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import is_acceptable_phrase, normalise_phrase
from ..types import Candidate
from .base import BaseGenerator

# Section titles that hold a list of defined terms.
_TERM_SECTION = re.compile(
    r"(?i)\b(glossary|acronyms?|abbreviations?|nomenclature|notation|"
    r"list of symbols|symbol table|definitions?)\b")

# "Definition 3.1 (Concept Drift). ..." / "Theorem 2. ..."
_ENV_LEAD = re.compile(
    r"(?i)^\s*(definition|defn|theorem|thm|lemma|corollary|proposition|"
    r"remark|axiom|claim)\s*"
    r"(\d+(?:\.\d+)*)?\s*"
    r"(?:\(([^)]{2,80})\))?\s*[.:]?\s*")

# A glossary line: "TERM — meaning" / "TERM: meaning" / "TERM - meaning"
_ENTRY = re.compile(r"^\s*([^\s—–:.-][^—–:]{0,70}?)\s*[—–:]\s+(.{4,})$")

# Schwartz-Hearst helpers.
_PAREN = re.compile(r"\(([^)]{2,40})\)")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9.\-']*")


def _is_short_form(s: str) -> bool:
    """An acronym-shaped string: 2-10 chars, at most 2 tokens, >=2 capitals."""
    s = s.strip()
    if not (2 <= len(s) <= 10) or len(s.split()) > 2:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9.\-/]*$", s):
        return False
    return sum(c.isupper() for c in s) >= 2


# A sentence boundary: an expansion may not reach back across one.
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(])|\n\s*\n")


def find_long_form(short: str, preceding: str) -> Optional[str]:
    """Schwartz-Hearst: locate `short`'s expansion in the text before the paren.

    Every alphanumeric character of the short form must match a character in the
    long form scanning right to left, and the first short character must land on
    a word initial.

    The window is cut at the last sentence boundary. Without that, OCR'd prose —
    where an acronym often follows an unrelated sentence — lets the matcher
    stitch a "long form" out of words spanning two sentences, producing
    confident nonsense that then outranks real concepts on structural weight.
    """
    chars = [c.lower() for c in short if c.isalnum()]
    if not chars:
        return None

    # Only the current sentence is eligible.
    segments = _SENTENCE_END.split(preceding.rstrip())
    window = segments[-1] if segments else preceding
    words = _WORD.findall(window)
    if len(words) < len(chars):
        return None
    # Search windows of increasing size, shortest (tightest) match wins.
    for window in range(len(chars), min(len(chars) * 2 + 3, len(words)) + 1):
        candidate_words = words[-window:]
        candidate = " ".join(candidate_words)
        ci = len(candidate) - 1
        si = len(chars) - 1
        while si >= 0 and ci >= 0:
            if candidate[ci].lower() == chars[si]:
                si -= 1
            ci -= 1
        if si < 0:
            # The first short char must align to a word initial.
            first = chars[0]
            if candidate_words and candidate_words[0][:1].lower() == first:
                return candidate
    return None


# Words that cannot occur inside a noun phrase. Their presence means the match
# ran across a clause boundary — which happens constantly in OCR'd text, where
# the sentence-ending period is often simply missing.
_CONNECTIVES = frozenset("""
however therefore moreover furthermore thus hence whereas although though
because since unless until while and or but nor so yet if then when where
we our this that these those it there here also both either neither
""".split())


def _plausible_long_form(long_form: str, short: str) -> bool:
    """Reject expansions that are structurally implausible.

    Schwartz-Hearst is character matching with no notion of meaning, so on noisy
    input it happily returns word soup. An expansion should be about as long as
    its acronym and should read as a noun phrase, not a clause.
    """
    words = long_form.split()
    letters = sum(c.isalnum() for c in short)
    if not (letters <= len(words) <= letters * 2 + 2):
        return False
    # A mid-phrase sentence terminator means the window was still too wide.
    if re.search(r"[.!?;]", long_form[:-1]):
        return False
    if any(w.lower() in _CONNECTIVES for w in words):
        return False
    return True


class GlossaryGenerator(BaseGenerator):
    """Defined terms: environments, term-list sections, and acronym pairs."""

    source = "glossary"

    def __init__(self, *, include_acronyms: bool = True,
                 include_environments: bool = True,
                 include_term_sections: bool = True) -> None:
        self.include_acronyms = include_acronyms
        self.include_environments = include_environments
        self.include_term_sections = include_term_sections

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()

        def add(name: str, *, kind: str, description: str = "",
                section_id: Optional[str] = None, alias: str = "") -> None:
            name = normalise_phrase(name)
            if not name or not is_acceptable_phrase(name, max_tokens=6):
                return
            key = name.lower()
            if key in seen:
                return
            seen.add(key)
            meta: dict[str, object] = {"kind": kind}
            if description:
                meta["description"] = normalise_phrase(description)[:400]
            if alias:
                meta["aliases"] = [alias]
            out.append(Candidate(
                name=name, source=self.source, kind=kind,
                section_id=section_id, metadata=meta,
            ))

        if self.include_environments:
            self._from_environments(doc, add)
        if self.include_term_sections:
            self._from_term_sections(doc, add)
        if self.include_acronyms:
            self._from_acronyms(doc, add)

        return self._sorted(out)

    # ---- sources --------------------------------------------------------

    def _from_environments(self, doc: Document, add) -> None:
        """Blocks typed as definitions/theorems, or prose that leads with one."""
        typed = {"definition", "theorem", "lemma", "corollary", "proposition",
                 "axiom", "remark", "claim"}
        for block in doc.blocks:
            btype = block.type.lower()
            text = block.text.strip()
            if not text:
                continue

            named = block.props.get("name") or block.props.get("title")
            if btype in typed and isinstance(named, str) and named.strip():
                add(named, kind=btype, description=text, section_id=block.section_id)
                continue

            match = _ENV_LEAD.match(text)
            if not match:
                continue
            # Only trust the lead-in on a block the parser typed, or one short
            # enough that the environment marker is clearly its subject.
            if btype not in typed and len(text) > 1200:
                continue
            parenthetical = match.group(3)
            body = text[match.end():]
            if parenthetical:
                add(parenthetical, kind=match.group(1).lower(),
                    description=body, section_id=block.section_id)
            elif btype in typed:
                # An unnamed environment: take its opening subject phrase.
                lead = re.split(r"(?<=[.;])\s", body, maxsplit=1)[0]
                subject = re.match(r"(?:the|a|an)?\s*([A-Za-z][\w\s\-]{2,60}?)\s+"
                                   r"(?:is|are|denotes?|refers?|means)\b",
                                   lead, flags=re.IGNORECASE)
                if subject:
                    add(subject.group(1), kind=btype, description=body,
                        section_id=block.section_id)

    def _from_term_sections(self, doc: Document, add) -> None:
        """`TERM — meaning` lines inside a glossary-like section."""
        term_sections = {
            sid for sid, sec in doc.sections.items()
            if _TERM_SECTION.search(sec.title or "")
        }
        if not term_sections:
            return
        for block in doc.blocks:
            if block.section_id not in term_sections:
                continue
            for line in block.text.splitlines():
                match = _ENTRY.match(line)
                if not match:
                    continue
                term, meaning = match.group(1), match.group(2)
                add(term, kind="term", description=meaning,
                    section_id=block.section_id)

    def _from_acronyms(self, doc: Document, add) -> None:
        """Schwartz-Hearst over prose. The long form becomes the concept name and
        the acronym its alias — the expansion is the more embeddable string."""
        for block in doc.prose_blocks:
            text = block.text
            for match in _PAREN.finditer(text):
                inner = match.group(1).strip()
                if not _is_short_form(inner):
                    continue
                long_form = find_long_form(inner, text[:match.start()])
                if not long_form or not _plausible_long_form(long_form, inner):
                    continue
                add(long_form, kind="acronym", description=f"also written {inner}",
                    section_id=block.section_id, alias=inner)
