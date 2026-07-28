"""1.3 Bibliography — cited paper titles as concepts.

The premise: a paper the author chose to cite names something the document is
about, and the citing community has already validated it. Titles are stable,
meaningful, and free of the vocabulary drift that plagues frequency-based
concepts.

Citation counts and years are carried into metadata, where the
`citation_importance` metric reads them. When the source bibliography has no
counts — which is the case for anything parsed straight out of a PDF — that
metric falls back to neutral rather than fabricating a number.

Long titles are shortened through the abstractor. With the default
`NullAbstractor` that is a clause-boundary truncation; the full title is always
kept as the tau text so the embedding sees everything.
"""
from __future__ import annotations

import re
from typing import Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import normalise_phrase
from ..types import Candidate
from .base import BaseGenerator

# Leading "[12]" / "12." / "(3)" markers left over from reference parsing.
_LEADING_MARKER = re.compile(r"^\s*(?:\[\d+\]|\(\d+\)|\d{1,3}[.)])\s*")

# An author run at the head of a raw reference string: "Smith, J., Doe, A.:"
_AUTHOR_RUN = re.compile(
    r"^(?:[A-Z][A-Za-z'\-]+,\s*(?:[A-Z]\.\s*)+(?:and\s+)?[,;]?\s*){1,8}[:.]\s*")

# Trailing venue/DOI tails.
_TRAILING_VENUE = re.compile(
    r"(?i)\s*(?:in:?\s+proc\w*|in:?\s+|proceedings\b|journal\b|arxiv[:\s]|"
    r"doi[:\s]|pp\.\s*\d+|vol\.\s*\d+|\(\d{4}\)).*$")

MIN_TITLE_WORDS = 3
MAX_TITLE_WORDS = 30


# Residue that means the venue tail was never successfully removed.
_UNCLEANED = re.compile(
    r"(?i)(proceedings\b|\bin:\s|\bpp\.|\bvol\.|\bdoi\b|arxiv|"
    r"\b\d+\(\d+\)|\b\d+\s*[-–]\s*\d+\b)")


def clean_title(raw: str) -> str:
    """Best-effort title isolation from a raw reference string.

    Bibliography text out of OCR is messy. Stripping is attempted, and if what
    survives still carries venue residue the entry is rejected outright by
    returning "". Keeping a half-cleaned string was worse: fragments like
    "survey. Proceedings of the IEEE 104(1), 34-57 (2015" entered the vocabulary
    at bibliography's 0.7 structural weight and outranked real concepts.
    """
    text = re.sub(r"\s+", " ", raw or "").strip()
    if not text:
        return ""
    text = _LEADING_MARKER.sub("", text)
    stripped = _AUTHOR_RUN.sub("", text)
    if len(stripped.split()) >= MIN_TITLE_WORDS:
        text = stripped
    trimmed = _TRAILING_VENUE.sub("", text)
    if len(trimmed.split()) >= MIN_TITLE_WORDS:
        text = trimmed
    text = text.strip(" .,;:")
    if _UNCLEANED.search(text):
        return ""
    return text


class BibliographyGenerator(BaseGenerator):
    """One candidate per bibliography entry, named by its title."""

    source = "bibliography"

    def __init__(self, *, shorten_long_titles: bool = True,
                 shorten_threshold: int = 12,
                 max_words: int = 8) -> None:
        self.shorten_long_titles = shorten_long_titles
        self.shorten_threshold = shorten_threshold
        self.max_words = max_words

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()

        for entry in doc.bibliography:
            title = clean_title(entry.title)
            if not title:
                continue
            words = title.split()
            if not (MIN_TITLE_WORDS <= len(words) <= MAX_TITLE_WORDS):
                continue

            name = title
            if self.shorten_long_titles and len(words) > self.shorten_threshold:
                shortened = abstractor.shorten_title(title, self.max_words)
                if shortened and len(shortened.split()) >= 2:
                    name = shortened

            name = normalise_phrase(name)
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)

            metadata: dict[str, object] = {
                "bib_id": entry.id,
                "full_title": title,
                "abstractor": abstractor.name,
            }
            if entry.label:
                metadata["label"] = entry.label
            if entry.year is not None:
                metadata["year"] = entry.year
            if entry.citations is not None:
                metadata["citations"] = entry.citations
            if entry.keywords:
                metadata["keywords"] = list(entry.keywords)

            out.append(Candidate(
                name=name,
                source=self.source,
                kind="paper_title",
                # tau is the *full* title even when the name was shortened: the
                # embedding should see every word the author published.
                tau=title,
                metadata=metadata,
            ))

        return self._sorted(out)
