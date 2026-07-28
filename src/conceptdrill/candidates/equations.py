"""1.6 Equation abstractions.

Each equation is turned into a short natural-language description, which becomes
the candidate. The description — not the LaTeX — is what gets embedded, because
a Sentence-BERT-class model reads "summation over a loss function" far better
than it reads `\\mathcal{L}_c = \\sum_i y_i \\log p_i`.

The description comes from the injected `Abstractor`. The default is
deterministic and structural (operator-derived), so this generator runs offline;
wiring in a real language model upgrades the descriptions without touching this
file.

Surrounding prose is passed as context. It is the strongest available hint at
what an equation *means*, and an LLM abstractor will use it.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from ..abstractor import Abstractor
from ..document import Document
from ..nlp import is_acceptable_phrase, normalise_phrase
from ..types import Candidate
from .base import BaseGenerator

CONTEXT_CHARS = 600


def _strip_latex(latex: str) -> str:
    """Readable remnant of a formula — used only for display, never embedded."""
    text = re.sub(r"\\(?:label|tag|nonumber|notag)\s*{[^}]*}", " ", latex)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}$&\\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class EquationGenerator(BaseGenerator):
    """Natural-language abstractions of the document's equations."""

    source = "equation"

    def __init__(self, *, min_length: int = 8, max_equations: int = 200,
                 merge_duplicates: bool = True) -> None:
        self.min_length = min_length
        self.max_equations = max_equations
        self.merge_duplicates = merge_duplicates

    def generate(self, doc: Document, *,
                 abstractor: Abstractor) -> Sequence[Candidate]:
        math_blocks = doc.math_blocks
        if not math_blocks:
            return []

        # Prose neighbours by document order, so an equation can be described
        # with the sentence that introduces it.
        order = {b.id: i for i, b in enumerate(doc.blocks)}
        prose = sorted(doc.prose_blocks, key=lambda b: order.get(b.id, 0))

        descriptions: Counter[str] = Counter()
        detail: dict[str, dict] = {}

        for block in math_blocks[:self.max_equations]:
            latex = block.text.strip()
            if len(latex) < self.min_length:
                continue
            context = self._context_for(block, prose, order)
            description = abstractor.describe_equation(latex, context)
            description = normalise_phrase(description)
            if not description or not is_acceptable_phrase(description, max_tokens=8):
                continue
            key = description.lower()
            descriptions[key] += 1
            # Keep the first equation seen for a description as its exemplar.
            detail.setdefault(key, {
                "name": description,
                "example_object_id": block.id,
                "example_latex": latex[:400],
                "example_readable": _strip_latex(latex)[:200],
                "section_id": block.section_id,
            })

        out: list[Candidate] = []
        for key, count in descriptions.items():
            info = detail[key]
            if not self.merge_duplicates:
                count = 1
            out.append(Candidate(
                name=info["name"],
                source=self.source,
                kind="equation_abstraction",
                frequency=count,
                section_id=info["section_id"],
                # tau is the description: that is the point of the abstraction.
                tau=info["name"],
                metadata={
                    "equation_count": count,
                    "example_object_id": info["example_object_id"],
                    "example_latex": info["example_latex"],
                    "example_readable": info["example_readable"],
                    "abstractor": abstractor.name,
                    "abstractor_deterministic": abstractor.is_deterministic,
                },
            ))
        return self._sorted(out)

    def _context_for(self, block, prose: list, order: dict) -> str:
        """Prose immediately before and after the equation."""
        pos = order.get(block.id)
        if pos is None or not prose:
            return ""
        before = [b for b in prose if order.get(b.id, 0) < pos]
        after = [b for b in prose if order.get(b.id, 0) > pos]
        parts: list[str] = []
        if before:
            parts.append(before[-1].text[-CONTEXT_CHARS // 2:])
        if after:
            parts.append(after[0].text[:CONTEXT_CHARS // 2])
        return " ".join(parts).strip()
