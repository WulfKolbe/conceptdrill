r"""Dimension zero: telling document furniture from subject matter.

A reference list, a funding statement and a numbered box are all sections, and
none of them is a concept. Left in the basis they become rows — the previous
corpus build merged `4 URDF Framework` with `References` — and every row they
occupy is a CES coordinate that means nothing.

Sections classified here are **absorbed into the reserved structural row**,
bypassing the similarity threshold entirely. They never compete for a concept
row and never create one.

## Deterministic and rule-based, not model-driven

This is a closed vocabulary. A model asked "is this section structural?" would
be unauditable, non-reproducible, and would cost an API call per section to
answer a question a lookup table answers exactly. Every classification names
the rule that fired, so a disputed one can be argued about.

## Unmatched is content

The default is `None` — content. This layer never guesses. Over-absorption
costs a concept; under-absorption costs a junk row, and only one of those is
recoverable by looking at the record afterwards.

## Abstract is content

Deliberately absent from every list below. An abstract states what the document
is about, which is exactly what a concept space wants.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

#: Normalised title -> class. Exact match after `normalise_title`.
#:
#: German entries are not decoration: `~/pdfdrill-library` holds German theses,
#: and `Literaturverzeichnis` is as structural as `References`.
EXACT_RULES: dict[str, tuple[str, str]] = {}


def _register(structural_class: str, rule: str, *titles: str) -> None:
    for title in titles:
        EXACT_RULES[title] = (structural_class, rule)


_register("bibliography", "reference-list",
          "references", "reference", "bibliography", "works cited",
          "literature", "literature cited", "list of references",
          "literaturverzeichnis", "literatur", "quellen",
          "quellenverzeichnis", "bibliographie")

_register("acknowledgement", "acknowledgement",
          "acknowledgement", "acknowledgements", "acknowledgment",
          "acknowledgments", "danksagung", "danksagungen")

_register("funding", "funding",
          "funding", "funding information", "funding statement",
          "financial support", "foerderung", "finanzierung")

_register("competing-interests", "competing-interests",
          "conflict of interest", "conflicts of interest",
          "competing interests", "competing interest",
          "declaration of competing interest",
          "declaration of competing interests",
          "declaration of interest", "interessenkonflikt")

_register("author-contributions", "author-contributions",
          "author contributions", "author contribution",
          "authors contributions", "contributions",
          "credit authorship contribution statement",
          "authorship contribution statement", "autorenbeitraege")

_register("data-availability", "data-availability",
          "data availability", "data availability statement",
          "code availability", "availability of data and materials")

_register("ethics", "ethics-statement",
          "ethics statement", "ethical statement", "ethics approval",
          "ethics declaration", "ethical considerations", "ethikerklaerung")

_register("table-of-contents", "table-of-contents",
          "contents", "table of contents", "inhalt", "inhaltsverzeichnis")

_register("list-of-floats", "list-of-floats",
          "list of figures", "list of tables", "list of abbreviations",
          "list of symbols", "list of algorithms", "nomenclature",
          "abbreviations", "abbildungsverzeichnis", "tabellenverzeichnis",
          "abkuerzungsverzeichnis", "symbolverzeichnis")

_register("index", "index",
          "index", "subject index", "author index", "stichwortverzeichnis")

_register("supplementary", "supplementary",
          "supplementary material", "supplementary materials",
          "supplementary information", "supporting information",
          "electronic supplementary material")

_register("front-matter", "front-matter",
          "about the author", "about the authors", "copyright",
          "copyright notice", "imprint", "impressum", "dedication",
          "widmung", "preface", "vorwort", "foreword", "geleitwort",
          "colophon", "kolophon", "title page", "titelblatt")

#: Metadata blocks. Not in the brief's minimum list, added because this corpus
#: has them: NTCIR working notes open with `Team Name` / `Subtasks` /
#: `Keywords`, and Elsevier PDFs open with `ARTICLE INFO`.
_register("metadata-block", "metadata-block",
          "keywords", "key words", "keyword", "index terms", "schlagworte",
          "schlagwoerter", "stichworte", "article info", "article information",
          "team name", "teamname", "subtasks", "subtask",
          "author", "authors", "autoren", "affiliation", "affiliations",
          "correspondence", "corresponding author")

#: Prefix rules: the title *starts* with one of these. `Appendix. Additional
#: Details` and `Anhang B: Messwerte` are appendices whatever follows.
PREFIX_RULES: tuple[tuple[str, str, str], ...] = (
    ("appendix", "appendix", "appendix"),
    ("appendix", "appendix", "appendices"),
    ("appendix", "appendix", "anhang"),
    ("supplementary", "supplementary", "supplementary"),
)

#: A numbered float that the drill surfaced as a section: `Box 12`, `Figure 3`.
#: These carry a caption, not a topic.
FLOAT_PATTERN = re.compile(
    r"^(box|figure|fig|table|tab|listing|algorithm|alg|equation|eq|scheme|plate)"
    r"\s*\.?\s*\d+[a-z]?$")

#: Drill artifacts that are not sections at all: `#1` is a LaTeX macro
#: parameter that reached the section list, and a title of only digits or
#: punctuation names nothing.
ARTIFACT_PATTERN = re.compile(r"^(#\d+|\d+(\.\d+)*|[^\w]+)$")

#: A leading ordinal. Numeric ones may omit punctuation (`3 Methods`), but a
#: single letter or roman numeral must carry it: without that requirement
#: `A framework for search` loses its article and `I Introduction` is
#: indistinguishable from a title beginning with the pronoun.
_ORDINAL = re.compile(
    r"^(?:\(?\d+(?:\.\d+)*\)?[.):]?\s+|\(?(?:[ivxlcdm]+|[a-z])\)?[.):]\s+)")
_UMLAUT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                         "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def normalise_title(title: str) -> str:
    """Fold a section title to its comparison form.

    Lowercased, umlauts transliterated, accents stripped, leading ordinal and
    trailing punctuation removed, whitespace collapsed. `7. REFERENCES` and
    `Keywords:` both have to reach `references` and `keywords`, or the rules
    match nothing real.
    """
    text = (title or "").strip().translate(_UMLAUT).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text).strip()
    text = _ORDINAL.sub("", text)
    text = text.strip(" .:;,-–—_")
    return re.sub(r"\s+", " ", text).strip()


def classify_section(title: str, *, is_appendix: bool = False
                     ) -> tuple[Optional[str], Optional[str]]:
    """`(structural_class, rule_fired)`, or `(None, None)` for content.

    Rule order is deliberate. The artifact and float patterns run first because
    `#1` and `Box 12` normalise to something no word list should be consulted
    about. `is_appendix` runs last so an explicitly-flagged appendix is caught
    even when its title says nothing — measured on this corpus the flag is
    false everywhere, so it is a safety net rather than a workhorse.
    """
    normalised = normalise_title(title)

    if not normalised:
        return "untitled", "untitled-section"
    if ARTIFACT_PATTERN.match(normalised):
        return "artifact", "drill-artifact"
    if FLOAT_PATTERN.match(normalised):
        return "float", "float-container"
    if normalised in EXACT_RULES:
        structural_class, rule = EXACT_RULES[normalised]
        return structural_class, rule
    for structural_class, rule, prefix in PREFIX_RULES:
        # A word boundary, not a space: `Appendix. Additional Details` keeps
        # its period through normalisation, and `startswith(prefix + " ")`
        # silently missed it.
        if re.match(rf"{re.escape(prefix)}\b", normalised):
            return structural_class, rule
    if is_appendix:
        return "appendix", "docmodel-is-appendix"
    return None, None


def is_structural(title: str, *, is_appendix: bool = False) -> bool:
    return classify_section(title, is_appendix=is_appendix)[0] is not None
