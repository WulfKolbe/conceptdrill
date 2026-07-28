"""1.4 Frequent noun phrases.

Occurrence threshold and length cap come straight from the spec: at least 3
occurrences, at most 5 tokens. Stopword-only phrases are already excluded by
`nlp.is_acceptable_phrase`.

Two refinements beyond the literal spec, both aimed at vocabulary quality:

  * **TF-IDF-ish weighting across sections.** A phrase spread evenly over every
    section is usually domain filler ("the proposed method"); one concentrated
    in a few sections is usually a real concept. The score is carried in
    metadata for the scoring stage rather than used to filter here, so the
    generator stays a generator.
  * **Substring absorption.** When "deep hashing model" and "deep hashing" both
    clear the threshold and the longer adds no occurrences of its own, the
    shorter wins. This stops near-duplicate phrases from consuming the
    vocabulary budget before the diversity filter ever sees them.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import Phrase, analyse, backend_name, noun_phrases
from ..types import Candidate
from .base import BaseGenerator

DEFAULT_MIN_COUNT = 3
DEFAULT_MAX_TOKENS = 5


class NounPhraseGenerator(BaseGenerator):
    """Frequent noun phrases from the document's prose."""

    source = "nounphrase"

    def __init__(self, *, min_count: int = DEFAULT_MIN_COUNT,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 absorb_substrings: bool = True,
                 drop_person_names: bool = True,
                 max_candidates: int = 400) -> None:
        self.min_count = min_count
        self.max_tokens = max_tokens
        self.absorb_substrings = absorb_substrings
        self.drop_person_names = drop_person_names
        self.max_candidates = max_candidates

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        blocks = doc.prose_blocks
        if not blocks:
            return []

        texts = [b.text for b in blocks]
        phrases = noun_phrases(
            texts, min_count=self.min_count, max_tokens=self.max_tokens)
        if not phrases:
            return []

        if self.drop_person_names:
            # A proper-noun chunker cannot tell "Vaswani" from "Elasticsearch",
            # so citation authors arrive as high-frequency single-token phrases
            # and crowd out real concepts. NER already knows which are people;
            # `analyse` is cached, so consulting it costs nothing.
            people = {name.lower() for (name, label) in analyse(texts).entities
                      if label == "PERSON"}
            if people:
                phrases = [p for p in phrases if p.text.lower() not in people]

        if self.absorb_substrings:
            phrases = self._absorb(phrases)

        section_spread = self._section_spread(doc, phrases)
        n_sections = max(1, len({b.section_id for b in blocks}))
        backend = backend_name()

        out: list[Candidate] = []
        for phrase in phrases[:self.max_candidates]:
            sections = section_spread.get(phrase.text.lower(), set())
            # Inverse document frequency over sections, normalised to [0, 1].
            df = max(1, len(sections))
            idf = math.log(1 + n_sections / df) / math.log(1 + n_sections)
            out.append(Candidate(
                name=phrase.text,
                source=self.source,
                kind="noun_phrase",
                frequency=phrase.count,
                section_id=next(iter(sorted(s for s in sections if s)), None)
                if len(sections) == 1 else None,
                metadata={
                    "count": phrase.count,
                    "section_df": df,
                    "section_idf": round(idf, 6),
                    "nlp_backend": backend,
                },
            ))
        return self._sorted(out)

    # ---- helpers --------------------------------------------------------

    def _absorb(self, phrases: list[Phrase]) -> list[Phrase]:
        """Drop a longer phrase when a shorter one contains all its occurrences.

        "Longer adds nothing" means its count does not exceed the shorter's, so
        every time the long form appeared the short form did too.
        """
        by_len = sorted(phrases, key=lambda p: (len(p.text.split()), p.text))
        keep: list[Phrase] = []
        for phrase in by_len:
            lowered = f" {phrase.text.lower()} "
            redundant = any(
                f" {shorter.text.lower()} " in lowered
                and shorter.text.lower() != phrase.text.lower()
                and shorter.count >= phrase.count
                for shorter in keep
            )
            if not redundant:
                keep.append(phrase)
        return sorted(keep, key=lambda p: (-p.count, p.text))

    def _section_spread(self, doc: Document,
                        phrases: list[Phrase]) -> dict[str, set]:
        """Which sections each phrase occurs in. Substring containment is
        adequate here and far cheaper than re-running the tagger per section."""
        wanted = {p.text.lower() for p in phrases}
        spread: dict[str, set] = defaultdict(set)
        for block in doc.prose_blocks:
            lowered = " ".join(block.text.lower().split())
            for phrase in wanted:
                if phrase in lowered:
                    spread[phrase].add(block.section_id)
        return spread
