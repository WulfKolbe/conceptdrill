"""ConceptDrill — document-tailored concept projection.

Builds a concept space out of a document's own structure (headings, definitions,
bibliography, noun phrases, entities, equations), scores the candidates for
quality, and projects arbitrary text into the result. The CES pipeline from
arXiv 2209.00445 with the document standing in for the external ontology.

    from conceptdrill import ConceptDrill

    drill = ConceptDrill.from_path("paper.json")
    drill.explain_text("Deep learning for graphs", top_k=5)

Projections are always additional views: the source document is opened
read-only and output goes to a sidecar.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .abstractor import Abstractor, CallableAbstractor, NullAbstractor
from .core import ConceptDrill
from .document import Document
from .embeddings import get_embedder
from .space import ConceptSpace, refine_space
from .types import (Block, BibEntry, Candidate, Concept, ConceptHit, Confidence,
                    Projection, Section, SkippedObject)

__all__ = [
    "Abstractor", "BibEntry", "Block", "CallableAbstractor", "Candidate",
    "Concept", "ConceptDrill", "ConceptHit", "ConceptSpace", "Confidence",
    "Document", "NullAbstractor", "Projection", "Section", "SkippedObject",
    "__version__", "get_embedder", "refine_space",
]
