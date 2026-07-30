"""Checking what the model actually returned against what the prompt asked for.

A prompt is an instruction, not a guarantee. Every constraint in
`prompts/section-concept.md` that can be checked mechanically is checked here,
so a prompt revision is judged by measurement rather than by reading the model's
output and forming an impression.

Two defects motivate the specific checks:

* **Document-referential framing.** "The section outlines the approach to the
  TQIC task" describes the document instead of stating the content. Every
  banned construction below is wasted signal in a text whose only reader is an
  embedding model.

* **A label that is a sentence.** If every label follows `X is a Y that Zs`,
  the copula template becomes a constant substring across the corpus — the same
  class of failure as `\\section*{` in milder form, and one that `M7` would flag.

A document-local acronym is the third: `(TQIC)` is defined by one paper and
means nothing in another, so it cannot help a label match across documents.
"""
from __future__ import annotations

import re
from typing import Optional

#: Constructions that describe the document rather than its content. Enumerated
#: rather than described, because a rule the checker cannot apply is a rule the
#: model can be judged against only by opinion.
BANNED_CONSTRUCTIONS: tuple[str, ...] = (
    "this section", "the section", "this chapter", "the chapter",
    "this paper", "the paper", "this article", "the article",
    "this work", "the present work", "the authors",
    "we describe", "we present", "we propose", "we introduce", "we show",
    "here we", "is described", "is presented", "is discussed",
    "is introduced", "outlines", "discusses", "describes", "presents",
)

_BANNED = tuple(
    (phrase, re.compile(r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b",
                        re.IGNORECASE))
    for phrase in BANNED_CONSTRUCTIONS)

#: A parenthesised acronym: `(TQIC)`, `(BERT)`, `(DBMS)`. Two or more capitals
#: inside brackets is always a document-local shorthand in a label.
ACRONYM = re.compile(r"\(\s*[A-Z][A-Z0-9\-]{1,}\s*\)")

#: A bare acronym outside brackets, for reporting only. Three or more capitals
#: in a row is a strong signal; two is too noisy to act on.
BARE_ACRONYM = re.compile(r"\b[A-Z]{3,}[0-9]*\b")

#: Citation markers and float references that carry no cross-document meaning.
CITATION = re.compile(
    r"\[\s*\d+\s*(?:[,;-]\s*\d+\s*)*\]"           # [3]  [3, 4]  [3-5]
    r"|\[[^\]]*\bet al\.?[^\]]*\]"                 # [Martin et al., 1999]
    r"|\(\s*[A-Z][A-Za-z-]+\s+et\s+al\.?[^)]*\)")  # (Martin et al., 1999)

FLOAT_REFERENCE = re.compile(
    r"\b(?:figure|fig\.?|table|tab\.?|equation|eq\.?|section|sec\.?|"
    r"algorithm|alg\.?|listing|appendix)\s*\.?\s*\d+", re.IGNORECASE)

#: Copulas that make a label a sentence rather than a noun phrase. Anchored at
#: the start of a clause; a copula mid-phrase inside a relative clause is
#: caught by the same list because a noun phrase should not contain one either.
COPULA = re.compile(r"\b(?:is|are|was|were|be|been|being)\b", re.IGNORECASE)

#: Label word budget, from the prompt.
LABEL_WORDS = (30, 42)


def banned_constructions(text: str) -> list[str]:
    """Every banned phrase present, in the order declared."""
    return [phrase for phrase, pattern in _BANNED if pattern.search(text or "")]


def word_count(text: str) -> int:
    return len((text or "").split())


def check_label(text: str, *, words: tuple[int, int] = LABEL_WORDS) -> list[str]:
    """Every way a label breaks the prompt's contract for it."""
    problems: list[str] = []
    if not (text or "").strip():
        return ["label is empty"]

    n = word_count(text)
    lo, hi = words
    if n < lo:
        problems.append(f"label is {n} words, below {lo}")
    elif n > hi:
        problems.append(f"label is {n} words, above {hi}")

    for phrase in banned_constructions(text):
        problems.append(f"banned construction {phrase!r}")

    found = ACRONYM.search(text)
    if found:
        problems.append(f"parenthesised acronym {found.group(0)!r}")

    copula = COPULA.search(text)
    if copula:
        problems.append(f"copula {copula.group(0)!r}: label is a sentence, "
                        f"not a noun phrase")

    citation = CITATION.search(text)
    if citation:
        problems.append(f"citation marker {citation.group(0)!r}")

    float_ref = FLOAT_REFERENCE.search(text)
    if float_ref:
        problems.append(f"document-local reference {float_ref.group(0)!r}")
    return problems


def check_abstraction(text: str) -> list[str]:
    """Every way an abstraction breaks its contract.

    Only the banned constructions are mechanical. "No proper noun naming this
    paper's artifacts" needs to know which nouns are the paper's own, which the
    text alone does not say — `bare_acronyms` reports candidates for a human to
    look at rather than pretending to decide.
    """
    problems: list[str] = []
    if not (text or "").strip():
        return ["abstraction is empty"]
    for phrase in banned_constructions(text):
        problems.append(f"banned construction {phrase!r}")
    return problems


def check_summary(text: str) -> list[str]:
    problems: list[str] = []
    if not (text or "").strip():
        return ["summary is empty"]
    for phrase in banned_constructions(text):
        problems.append(f"banned construction {phrase!r}")
    return problems


def bare_acronyms(text: str) -> list[str]:
    """Uppercase runs, reported rather than gated. See `check_abstraction`."""
    return sorted(set(BARE_ACRONYM.findall(text or "")))


def check_tiers(label: Optional[str], abstraction: Optional[str],
                summary: Optional[str]) -> dict[str, list[str]]:
    """All three tiers at once. Absent tiers are not checked, only reported."""
    out: dict[str, list[str]] = {}
    out["label"] = check_label(label) if label else ["label is absent"]
    out["abstraction"] = (check_abstraction(abstraction) if abstraction
                          else ["abstraction is absent"])
    out["summary"] = check_summary(summary) if summary else ["summary is absent"]
    return out
