"""Sanitising text produced by a language model.

Models emit characters that are invisible or indistinguishable from ASCII on
screen but different to every downstream consumer. This is not hypothetical:
the improved `section-concept` prompt handed to this project arrived containing
nine U+2011 NON-BREAKING HYPHENs and four U+2013 EN DASHes, all of which look
exactly like `-`.

Why it matters here specifically:

  * **Tokenisation.** `embedding-ready` and `embedding‑ready` are different token
    sequences. A basis vector built from the second is not the vector you think
    you built.
  * **Matching.** Content-addressed caches, `in` tests and grep all miss.
  * **Silence.** Nothing errors. The text simply is not what it appears to be.

The rule applied: **normalise what is invisible or a lookalike; keep what is
content.** Greek letters, accented names and CJK are meaning and are preserved.
Zero-width joiners, bidi controls and typographic dashes are not, and are
folded to their ASCII intent.
"""
from __future__ import annotations

import re
import unicodedata

#: Characters with no visible width. They survive copy-paste, break string
#: equality, and are invisible in every editor.
ZERO_WIDTH = {
    "​": "",   # ZERO WIDTH SPACE
    "‌": "",   # ZERO WIDTH NON-JOINER
    "‍": "",   # ZERO WIDTH JOINER
    "⁠": "",   # WORD JOINER
    "﻿": "",   # ZERO WIDTH NO-BREAK SPACE / BOM
    "­": "",   # SOFT HYPHEN
}

#: Directional formatting. Invisible, and can reorder rendered text.
BIDI = {c: "" for c in (
    "‎", "‏",                                    # LRM, RLM
    "‪", "‫", "‬", "‭", "‮",      # embedding/override
    "⁦", "⁧", "⁨", "⁩",                # isolates
)}

#: Spaces that are not U+0020. NBSP is the common one; a model writing
#: "30 words" with U+00A0 breaks a naive split on ASCII space.
SPACES = {c: " " for c in (
    " ", " ", " ", " ", " ", " ", " ",
    " ", " ", " ", " ", " ", " ", " ",
    " ", "　",
)}

#: Dashes and hyphens that look like ASCII '-'.
DASHES = {c: "-" for c in (
    "‐", "‑", "‒", "–", "—", "―",
    "−", "﹘", "﹣", "－",
)}

#: Quotes and apostrophes that look like ASCII ' and ".
QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"', "«": '"', "»": '"',
    "‹": "'", "›": "'",
}

#: Miscellaneous typographic substitutions.
MISC = {
    "…": "...",    # HORIZONTAL ELLIPSIS
    "⁄": "/",      # FRACTION SLASH
    "·": "-",      # MIDDLE DOT used as a separator
    "•": "-",      # BULLET
    "©": "(c)", "®": "(r)", "™": "(tm)",
}

#: The full replacement table.
TRANSLATIONS: dict[int, str] = {
    ord(k): v for table in (ZERO_WIDTH, BIDI, SPACES, DASHES, QUOTES, MISC)
    for k, v in table.items()
}

#: Control characters that must never appear in prose. Tab, backspace,
#: formfeed and carriage return are how a legal JSON escape eats a LaTeX
#: command (`\tau` -> TAB + "au"), so they are stripped here as well as
#: reported by `replyparse.control_corruption`.
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

#: Categories worth deleting wholesale: Cf (format), Co (private use),
#: Cs (surrogate). Cn (unassigned) is left alone -- new codepoints are
#: legitimate content on a newer Unicode version.
_DROP_CATEGORIES = {"Cf", "Co", "Cs"}


def sanitize_text(text: str, *, collapse_whitespace: bool = True) -> str:
    """Fold invisible and lookalike characters to their ASCII intent.

    Preserves real content: Greek letters, accents and CJK all survive, because
    they carry meaning. Only characters that are invisible, or that impersonate
    ASCII punctuation, are changed.
    """
    if not text:
        return ""

    # Compatibility normalisation first: folds ligatures, fullwidth forms and
    # the many Unicode look-alikes into their canonical shape.
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(TRANSLATIONS)

    # Anything invisible that survived NFKC and the table.
    out = "".join(ch for ch in out
                  if unicodedata.category(ch) not in _DROP_CATEGORIES)

    out = _CONTROLS.sub(" ", out)

    if collapse_whitespace:
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def find_suspect_characters(text: str) -> tuple[str, ...]:
    """Names of invisible or lookalike characters present in `text`.

    Used to report what a model emitted, so a corpus build can say *why* a
    summary was altered rather than silently rewriting it.
    """
    seen: dict[str, None] = {}
    for ch in text or "":
        if ord(ch) < 128:
            continue
        if ord(ch) in TRANSLATIONS or unicodedata.category(ch) in _DROP_CATEGORIES:
            seen.setdefault(
                f"U+{ord(ch):04X} {unicodedata.name(ch, 'UNNAMED')}", None)
    return tuple(seen)


def sanitize_summary_fields(values: dict[str, str]) -> tuple[dict[str, str],
                                                             tuple[str, ...]]:
    """Sanitise every field of a model reply. Returns `(clean, warnings)`."""
    clean: dict[str, str] = {}
    warnings: list[str] = []
    for field, raw in values.items():
        suspects = find_suspect_characters(raw)
        clean[field] = sanitize_text(raw)
        if suspects:
            warnings.append(f"{field}: normalised {', '.join(suspects)}")
    return clean, tuple(warnings)
