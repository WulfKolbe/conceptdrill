"""The `CandidateGenerator` protocol and the structural weight table.

Each generator is independent and stateless: it reads a `Document` and returns
`Candidate`s. Swapping one out means registering a different callable — no other
module changes.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..abstractor import Abstractor
from ..document import Document
from ..types import Candidate

#: Structural importance by generator, straight from the spec. This is metric
#: w1 and the only prior ConceptDrill has before it looks at any embedding.
STRUCTURAL_WEIGHTS: dict[str, float] = {
    "glossary": 1.0,        # explicitly defined by the author
    "heading": 1.0,         # explicitly marked as important
    "definition": 1.0,
    "theorem": 1.0,
    "bibliography": 0.7,
    "nounphrase": 0.5,
    "equation": 0.4,
    "ner": 0.3,
}

DEFAULT_STRUCTURAL_WEIGHT = 0.3


def structural_weight(source: str) -> float:
    return STRUCTURAL_WEIGHTS.get(source, DEFAULT_STRUCTURAL_WEIGHT)


@runtime_checkable
class CandidateGenerator(Protocol):
    """Mines one class of concept candidate out of a document."""

    #: Registry key, and the `source` stamped on every candidate produced.
    source: str

    def generate(self, doc: Document, *, abstractor: Abstractor) -> Sequence[Candidate]:
        ...


class BaseGenerator:
    """Small shared base: identity and a deterministic ordering helper."""

    source: str = "base"

    def generate(self, doc: Document, *, abstractor: Abstractor) -> Sequence[Candidate]:
        raise NotImplementedError

    @staticmethod
    def _sorted(candidates: list[Candidate]) -> list[Candidate]:
        """Order by (-frequency, name) so a run never depends on dict order."""
        return sorted(candidates, key=lambda c: (-c.frequency, c.name))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} source={self.source}>"
