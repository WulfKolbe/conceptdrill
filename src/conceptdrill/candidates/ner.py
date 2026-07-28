"""1.5 Named entities.

Collects ORG, PRODUCT, PERSON, and GPE entities as the spec asks. Two caveats
worth stating rather than hiding:

  * PERSON entities in scientific prose are overwhelmingly citation authors
    ("as shown by Vaswani"), which are rarely concepts. They are collected as
    specified but default to *off*, controlled by `include_persons`.
  * Without a stanza or spaCy model installed, `nlp.named_entities` degrades to
    acronym detection labelled ORG. The active backend is recorded in each
    candidate's metadata so a downstream reader can see the tier that ran.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import backend_name, named_entities
from ..types import Candidate
from .base import BaseGenerator

#: Entity labels kept by default. PERSON is deliberately absent — see module docstring.
DEFAULT_LABELS = frozenset({"ORG", "PRODUCT", "GPE", "NORP", "EVENT", "LAW",
                            "WORK_OF_ART", "LANGUAGE", "FAC", "LOC"})

PERSON_LABELS = frozenset({"PERSON"})


class NERGenerator(BaseGenerator):
    """Named entities as concept candidates."""

    source = "ner"

    def __init__(self, *, min_count: int = 2,
                 include_persons: bool = False,
                 labels: Optional[frozenset[str]] = None,
                 max_candidates: int = 200) -> None:
        self.min_count = min_count
        self.include_persons = include_persons
        self.labels = frozenset(labels) if labels else DEFAULT_LABELS
        self.max_candidates = max_candidates

    @property
    def _wanted(self) -> frozenset[str]:
        return self.labels | PERSON_LABELS if self.include_persons else self.labels

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        blocks = doc.prose_blocks
        if not blocks:
            return []

        entities = named_entities(
            [b.text for b in blocks],
            min_count=self.min_count,
            wanted=self._wanted,
        )
        backend = backend_name()

        out: list[Candidate] = []
        seen: set[str] = set()
        for ent in entities[:self.max_candidates]:
            key = ent.text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(
                name=ent.text,
                source=self.source,
                kind=f"entity_{ent.label.lower()}",
                frequency=ent.count,
                metadata={
                    "entity_label": ent.label,
                    "count": ent.count,
                    "nlp_backend": backend,
                },
            ))
        return self._sorted(out)
