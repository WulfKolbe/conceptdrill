r"""The one function that produces `basis_text`, and its postconditions.

Everything that becomes a basis vector passes through `clean_basis_text`. The
postconditions below are asserted on its own output, so a rule that fails to
fire raises here rather than surfacing later as a corpus-wide constant
substring.

## Why this exists

The previous corpus build embedded `\span*{2 Related Work} Prominent
examples...` for every span of every document. `\span*{` was present in
every label. Two labels sharing a constant prefix are similar because of the
prefix, so the merge threshold was measuring boilerplate. Nothing in the
pipeline could have caught it: `captions.clean_caption` cleans titles, and
`captions.clean_body_text` deliberately leaves body prose alone.

## The title does not contribute

**Decided: `basis_text` never contains the marker title.** A title is
document-local navigation text — `1 Introduction`, `3. APPROACH`, `Team Name`.
Identical titles across papers describe unrelated content, so prepending one
manufactures cross-document similarity that is about span numbering rather
than concepts, which is precisely the failure this module exists to prevent.

`INCLUDE_TITLE` exists so the choice is visible and reversible, not so it can
be flipped casually: turning it on reintroduces the constant-substring class of
bug that `M7` was written to detect.

## No optional dependency

The stripper here is a deterministic scanner, not pylatexenc. `captions.py`
falls back to a regex cleaner through a swallowed exception, which means two
runs can clean the same caption differently and neither can say so. Text that
becomes a basis vector must not depend on which packages happen to be
installed, so this path has one implementation and no fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .captions import FORMATTING_MACROS, _SYMBOL_WORDS
from .sanitize import sanitize_text

#: Does the marker title contribute to `basis_text`? See the module docstring.
INCLUDE_TITLE = False

#: Characters that may never appear in a basis text. Every one of them is
#: markup that tokenizes as noise or evidences a command that survived.
FORBIDDEN_CHARS = frozenset("\\{}$%&#~^")

#: Commands whose argument is *not* prose: dropped with the argument.
#: `\span*{2 Related Work}` is the marker title again; `\cite{foo}` is a
#: reference marker; `\label{sec:x}` is an anchor. None of them describe a
#: concept.
DROP_WITH_ARGUMENT = frozenset({
    "section", "subsection", "subsubsection", "subsubsubsection", "paragraph",
    "subparagraph", "chapter", "part", "title", "author", "date", "thanks",
    "label", "ref", "eqref", "pageref", "autoref", "cref", "Cref",
    "cite", "citep", "citet", "citeauthor", "citeyear", "nocite", "bibitem",
    "index", "footnote", "footnotemark", "footnotetext", "caption",
    "includegraphics", "input", "include", "usepackage", "documentclass",
    "begin", "end", "url", "hspace", "vspace", "setlength", "newcommand",
})

#: Commands the postcondition names explicitly, for a clearer violation report.
NAMED_RESIDUE = ("\\span", "\\subsection", "\\label", "\\ref", "\\cite")

#: A leading numeric ordinal: `3`, `3.1`, `3.1.4`, with or without a dot.
_LEADING_NUMBER = re.compile(r"^\(?\d+(?:\.\d+)*\)?[.):]?\s+")

#: A leading alphabetic or roman ordinal. Punctuation is REQUIRED here, or
#: `A framework for...` would lose its article.
_LEADING_LETTER = re.compile(r"^\(?(?:[IVXLCDM]+|[A-Za-z])\)?[.):]\s+")

#: Everything that is not a letter or digit IN ANY SCRIPT. `[^a-z0-9]` threw
#: away Greek, Cyrillic, CJK and Coptic, so a title in one of those normalised
#: to the empty string -- and `text.startswith("")` is always true, so every
#: concept in such a document "began with its marker title" and the
#: postcondition raised. Two chunks of the overnight run died on it.
#: `sanitize.py` deliberately preserves those scripts as content; this has to
#: agree with it.
_WORD = re.compile(r"[^\w]+", re.UNICODE)


class BasisTextViolation(AssertionError):
    """The cleaner produced text that breaks its own postcondition."""


@dataclass(frozen=True)
class BasisText:
    """Cleaned text plus the names of the rules that fired producing it."""
    text: str
    rules_fired: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text)


# --------------------------------------------------------------------------
# The stripper
# --------------------------------------------------------------------------

def _read_group(source: str, i: int) -> tuple[Optional[str], int]:
    """Brace-matched group starting at `source[i]`, or `(None, i)`.

    Brace matching rather than a regex: `\\span*{The $f(x)$ case}` nests, and
    a non-greedy regex stops at the first `}` leaving the rest as residue.
    """
    if i >= len(source) or source[i] != "{":
        return None, i
    depth, j = 0, i
    while j < len(source):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[i + 1:j], j + 1
        j += 1
    return None, i          # unbalanced: leave it to the character sweep


def _strip_latex(source: str, rules: set[str], depth: int = 0) -> str:
    """Remove LaTeX structure, keeping the prose inside it.

    Three dispositions, and the difference between them is the whole point:
    a formatting command keeps its argument (`\\emph{Siblings}` -> `Siblings`),
    a structural command loses it (`\\cite{x}` -> nothing), and an unknown
    command drops its own name but keeps its argument, because an unknown
    command wrapping prose is far more common than one wrapping markup.
    """
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]

        if ch == "%":                       # comment to end of line
            rules.add("drop-comment")
            while i < n and source[i] != "\n":
                i += 1
            continue

        if ch != "\\":
            if ch in "{}":
                rules.add("drop-bare-brace")
                out.append(" ")
            elif ch == "$":
                rules.add("drop-math-delimiter")
                out.append(" ")
            elif ch in "&#~^":
                rules.add("drop-special-character")
                out.append(" ")
            else:
                out.append(ch)
            i += 1
            continue

        # A backslash: either an escaped character or a command.
        match = re.match(r"\\([A-Za-z@]+)(\*?)", source[i:])
        if not match:
            escaped = source[i + 1:i + 2]
            rules.add("unescape")
            out.append(escaped if escaped.isalnum() else " ")
            i += 2 if escaped else 1
            continue

        name, star = match.group(1), match.group(2)
        i += match.end()
        while i < n and source[i] == " ":
            i += 1
        # Optional argument, always discarded: it is a rendering hint.
        if i < n and source[i] == "[":
            close = source.find("]", i)
            if close != -1:
                i = close + 1
        group, i = _read_group(source, i)

        lowered = name.lower()
        if lowered in DROP_WITH_ARGUMENT:
            rules.add(f"drop-command:{lowered}{star}")
            out.append(" ")
        elif lowered in FORMATTING_MACROS:
            rules.add(f"unwrap-command:{lowered}")
            out.append(_strip_latex(group, rules, depth + 1) if group else " ")
        elif lowered in _SYMBOL_WORDS:
            rules.add(f"expand-symbol:{lowered}")
            out.append(f" {_SYMBOL_WORDS[lowered]} ")
            if group is not None:
                out.append(_strip_latex(group, rules, depth + 1))
        else:
            rules.add("drop-unknown-command")
            out.append(" ")
            if group is not None:
                out.append(_strip_latex(group, rules, depth + 1))
    return "".join(out)


def _normalise(text: str) -> str:
    """Lowercase, alphanumerics only. For comparing a title to a prefix."""
    return _WORD.sub(" ", (text or "").lower()).strip()


def _strip_leading_title(text: str, title: str, rules: set[str]) -> str:
    """Remove the title from the front, raw or normalized.

    Both forms are tried because the extractive tier prepends the *cleaned*
    title while a document's own body text repeats the *raw* one.
    """
    if not title:
        return text
    for candidate in (title, _strip_latex(title, set())):
        candidate = re.sub(r"\s+", " ", candidate or "").strip()
        if candidate and text.startswith(candidate):
            rules.add("strip-leading-title:raw")
            return text[len(candidate):].lstrip(" .:-—–")

    want = _normalise(title)
    if not want:
        return text
    have = _normalise(text)
    if have.startswith(want):
        # Walk the same number of normalised words off the front of the
        # original, so punctuation and case are preserved in what remains.
        wanted_words = len(want.split())
        words = text.split()
        for take in range(len(words), 0, -1):
            if len(_normalise(" ".join(words[:take])).split()) == wanted_words:
                rules.add("strip-leading-title:normalized")
                return " ".join(words[take:]).lstrip(" .:-—–")
    return text


def check_basis_text(text: str, title: str = "") -> list[str]:
    """Every postcondition clause `text` breaks. Empty means it is clean."""
    problems: list[str] = []
    if text is None:
        return ["text is None"]

    bad = sorted(set(text) & FORBIDDEN_CHARS)
    if bad:
        for ch in bad:
            where = text.find(ch)
            problems.append(
                f"forbidden character {ch!r} at {where}: "
                f"{text[max(0, where - 20):where + 20]!r}")

    for command in NAMED_RESIDUE:
        if command in text:
            problems.append(f"command residue {command!r}")
    if re.search(r"\\[A-Za-z@]+", text):
        problems.append("command residue: a backslash-command survived")

    if title:
        for candidate in (title.strip(), _normalise(title)):
            wanted = _normalise(candidate)
            # An empty normalisation matches everything. A title of pure
            # punctuation -- `\({` was one -- must not condemn every text.
            if not wanted:
                continue
            if _normalise(text).startswith(wanted):
                problems.append(f"begins with the marker title {candidate!r}")
                break

    if _LEADING_NUMBER.match(text) or _LEADING_LETTER.match(text):
        problems.append(f"begins with an ordinal: {text[:24]!r}")
    return problems


def clean_basis_text(text: str, *, title: str = "",
                     include_title: bool = INCLUDE_TITLE) -> BasisText:
    """Span content to embeddable prose. The only producer of `basis_text`.

    Raises `BasisTextViolation` when its own output breaks a postcondition —
    a cleaner that silently returns dirty text is worse than no cleaner, since
    everything downstream then trusts it.
    """
    rules: set[str] = set()
    raw = text or ""
    sanitized = sanitize_text(raw)
    if sanitized != raw:
        rules.add("sanitize-unicode")

    stripped = _strip_latex(sanitized, rules)
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    if collapsed != sanitized.strip():
        rules.add("collapse-whitespace")

    without_title = _strip_leading_title(collapsed, title, rules)

    ordinal = _LEADING_NUMBER.match(without_title) or \
        _LEADING_LETTER.match(without_title)
    if ordinal:
        rules.add("strip-leading-ordinal")
        without_title = without_title[ordinal.end():]

    # A title stripped from the front can expose the ordinal that preceded the
    # next clause, so one more pass. Bounded, not a loop: two is enough for
    # "3.1 Method. 3.1 Method text" and a third would start eating content.
    again = _LEADING_NUMBER.match(without_title) or \
        _LEADING_LETTER.match(without_title)
    if again:
        rules.add("strip-leading-ordinal")
        without_title = without_title[again.end():]

    result = re.sub(r"\s+", " ", without_title).strip()

    if include_title and title:
        clean_title = re.sub(r"\s+", " ", _strip_latex(title, rules)).strip()
        clean_title = _LEADING_NUMBER.sub("", clean_title)
        if clean_title:
            rules.add("prepend-title")
            result = f"{clean_title}. {result}".strip(". ").strip()

    problems = check_basis_text(result, title if not include_title else "")
    if problems:
        raise BasisTextViolation(
            "clean_basis_text produced text that breaks its own "
            f"postcondition: {problems}; text={result[:200]!r}")
    return BasisText(text=result, rules_fired=tuple(sorted(rules)))
