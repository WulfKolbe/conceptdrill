r"""Algorithm 2 of arXiv 2209.00445: hierarchical on-demand concept spaces.

The paper's refinement loop, implemented directly:

    C_curr <- C^1                                  start at the top level
    while |C_curr| < S:
        for each context text t:  w[c] += CES(f(t))[c]
        c_max <- argmax score[c]                   the space's dominant concept
        C_add <- top ceil(p*|children|) children of c_max, by siblings score
        C_curr <- (C_curr \ {c_max}) u C_add       if removeP
                  C_curr u C_add                   otherwise

The intuition, in the paper's words: the highest-weighted concept "represents a
main topic of the text, and will therefore benefit the most from a more refined
representation".

This is **different from `basis.py`**, which merges concepts across documents.
Refinement grows one space downward through a hierarchy; merging fuses spaces
sideways across a corpus. Both are useful; only this one is from the paper.

## The siblings score is degenerate on a marker tree

    sibscore(p, c) = mean over s in siblings(c,p) of
                     |parents(c) & parents(s)| / |parents(c)|

The paper needed this because Wikipedia's category edges are unlabelled and a
concept has **many** parents, so the overlap says how tightly a child belongs to
its sibling group. Measured across the 334 drilled documents: **0 of 8695
markers have more than one parent**. In a tree every sibling shares the single
parent, every term is |{p}|/|{p}| = 1, and the score is always exactly 1 — it
cannot rank anything.

So `sibscore` is implemented faithfully for the DAG case, and
`children_ranked` falls back to a **document-order** tie-break when the scores
are uniform. Ordering children by where the author put them is a real signal in
a document; pretending a constant is a ranking is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

#: Fraction of a concept's children admitted on expansion (the paper's `p`).
DEFAULT_CHILD_FRACTION = 0.5

#: Blend between accumulated weight and label purity (the paper's lambda).
DEFAULT_LAMBDA = 1.0


@dataclass
class ConceptGraph:
    """The ontology Algorithm 2 walks: `G = (V, E)` with depth.

    Deliberately not tied to a marker tree. The paper uses Wikipedia
    categories, this project uses span hierarchies, and a caller may bring
    anything with parents, children and a depth.
    """

    #: concept id -> its children, in a meaningful order (document order, for
    #: a marker tree). Order is preserved and used as the sibscore tie-break.
    children: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: concept id -> its parents. A SET in general: an ontology is a DAG.
    parents: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: concept id -> human label, for reporting.
    labels: dict[str, str] = field(default_factory=dict)
    #: concept id -> depth. Not assumed to start at 1.
    depth: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_section_tree(cls, tree) -> "ConceptGraph":
        """Build from a `MarkerTree`. Children keep document order."""
        children, parents, labels, depth = {}, {}, {}, {}
        for node in tree.iter_document_order():
            children[node.id] = tuple(node.children)
            parents[node.id] = (node.parent_id,) if node.parent_id else ()
            labels[node.id] = node.title
            depth[node.id] = node.level
        return cls(children=children, parents=parents, labels=labels, depth=depth)

    # ---- the set operations, spelled out ------------------------------

    def children_of(self, concept: str) -> tuple[str, ...]:
        return self.children.get(concept, ())

    def parents_of(self, concept: str) -> frozenset[str]:
        return frozenset(self.parents.get(concept, ()))

    def siblings(self, concept: str, parent: str) -> tuple[str, ...]:
        """`children(parent) \\ {concept}` — plain set difference, order kept."""
        return tuple(c for c in self.children_of(parent) if c != concept)

    def sibscore(self, parent: str, concept: str) -> float:
        """The paper's siblings score for the edge `(parent, concept)`.

        Mean over the concept's siblings of the fraction of the concept's
        parents that the sibling also has. Returns 1.0 for an only child: with
        no siblings to disagree with, the edge is maximally coherent.
        """
        sibs = self.siblings(concept, parent)
        if not sibs:
            return 1.0
        mine = self.parents_of(concept)
        if not mine:
            return 0.0
        total = sum(len(mine & self.parents_of(s)) / len(mine) for s in sibs)
        return total / len(sibs)

    def is_tree(self) -> bool:
        """True when no concept has more than one parent.

        When true, `sibscore` is constant and cannot rank children — see the
        module docstring.
        """
        return all(len(self.parents_of(c)) <= 1 for c in self.children)

    def roots(self) -> tuple[str, ...]:
        return tuple(c for c in self.children if not self.parents_of(c))

    def top_level(self) -> tuple[str, ...]:
        """The paper's `C^1`: every concept at the shallowest depth present.

        Not "depth == 1": real documents start at level 1 or 2 depending on the
        drill, so a hard-coded 1 would return nothing for a third of the corpus.
        """
        if not self.depth:
            return ()
        shallowest = min(self.depth.values())
        return tuple(c for c in self.children if self.depth.get(c) == shallowest)

    def children_ranked(self, concept: str) -> list[tuple[str, float]]:
        """Children with their siblings scores, best first.

        Ties break on document order, which is the position the author chose.
        On a tree every score is 1.0, so this is *entirely* a document-order
        ranking — stated plainly rather than dressed up as a scored one.
        """
        kids = self.children_of(concept)
        scored = [(kid, self.sibscore(concept, kid)) for kid in kids]
        order = {kid: i for i, kid in enumerate(kids)}
        scored.sort(key=lambda kv: (-kv[1], order[kv[0]]))
        return scored


@dataclass
class RefinementStep:
    """One iteration, recorded so the walk can be inspected afterwards."""
    expanded: str
    expanded_label: str
    score: float
    added: tuple[str, ...]
    removed: tuple[str, ...]
    size_after: int


@dataclass
class RefinementResult:
    """The tailored space `C*` and how it was reached."""
    concepts: tuple[str, ...] = ()
    steps: tuple[RefinementStep, ...] = ()
    stopped_because: str = ""
    sibscore_informative: bool = True

    def __len__(self) -> int:
        return len(self.concepts)

    def to_dict(self, graph: Optional[ConceptGraph] = None) -> dict[str, Any]:
        return {
            "size": len(self.concepts),
            "concepts": [
                {"id": c, "label": graph.labels.get(c, "") if graph else ""}
                for c in self.concepts
            ],
            "stopped_because": self.stopped_because,
            "sibscore_informative": self.sibscore_informative,
            "steps": [
                {"expanded": s.expanded, "label": s.expanded_label,
                 "score": round(s.score, 6), "added": list(s.added),
                 "removed": list(s.removed), "size_after": s.size_after}
                for s in self.steps
            ],
        }


def label_entropy(assignments: Sequence[Any]) -> float:
    """Shannon entropy in bits of a label distribution.

    Used by the paper's optional blend: a concept whose texts share one label
    is a purer distinction than one that collects a mixture.
    """
    if not assignments:
        return 0.0
    counts: dict[Any, int] = {}
    for a in assignments:
        counts[a] = counts.get(a, 0) + 1
    n = len(assignments)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def refine(graph: ConceptGraph, *,
           context_vectors: np.ndarray,
           concept_vectors: Callable[[Sequence[str]], np.ndarray],
           target_size: int,
           child_fraction: float = DEFAULT_CHILD_FRACTION,
           remove_parent: bool = False,
           labels: Optional[Sequence[Any]] = None,
           blend: float = DEFAULT_LAMBDA,
           start: Optional[Sequence[str]] = None,
           max_iterations: int = 1000) -> RefinementResult:
    """Algorithm 2: grow a concept space until it reaches `target_size`.

    `context_vectors` are the embedded contextual texts `T'`, one unit-norm row
    each. `concept_vectors(ids)` embeds concepts on demand — passed as a
    callable so this function never needs to know about embedders.

    Termination is handled explicitly, which the paper leaves implicit: a leaf
    concept cannot be expanded, so once every member of the space is a leaf the
    loop stops and says so. Without that it spins forever on any real
    hierarchy, where the deepest concepts always have no children.
    """
    current: list[str] = list(start) if start else list(graph.top_level())
    if not current:
        return RefinementResult(stopped_because="no starting concepts")

    contexts = np.asarray(context_vectors, dtype=np.float64)
    if contexts.ndim == 1:
        contexts = contexts.reshape(1, -1)

    steps: list[RefinementStep] = []
    exhausted: set[str] = set()
    reason = "reached target size"

    for _ in range(max_iterations):
        if len(current) >= target_size:
            break

        expandable = [c for c in current
                      if graph.children_of(c) and c not in exhausted]
        if not expandable:
            reason = "no expandable concept remains (all leaves)"
            break

        # --- score the current space by projecting the contexts into it ---
        matrix = np.asarray(concept_vectors(current), dtype=np.float64)
        if matrix.size == 0:
            reason = "concepts could not be embedded"
            break
        ces = contexts @ matrix.T                       # (n_texts, n_concepts)
        weight = ces.sum(axis=0)                        # w[c]

        score = weight
        if labels is not None and len(labels) == contexts.shape[0] and blend < 1.0:
            # Purity: entropy of the labels whose argmax concept is c.
            winners = np.argmax(ces, axis=1)
            purity = np.zeros(len(current), dtype=np.float64)
            for i in range(len(current)):
                owned = [labels[t] for t in range(len(winners)) if winners[t] == i]
                purity[i] = -label_entropy(owned)
            score = blend * weight + (1.0 - blend) * purity

        # --- pick the dominant expandable concept ---
        index = {c: i for i, c in enumerate(current)}
        best = max(expandable, key=lambda c: (float(score[index[c]]), c))
        ranked = graph.children_ranked(best)
        if not ranked:
            exhausted.add(best)
            continue

        take = max(1, math.ceil(child_fraction * len(ranked)))
        additions = [kid for kid, _ in ranked[:take] if kid not in current]
        if not additions:
            # Every child is already present; expanding again would not grow.
            exhausted.add(best)
            continue

        removed: tuple[str, ...] = ()
        if remove_parent:
            current = [c for c in current if c != best]
            removed = (best,)
        current.extend(additions)
        exhausted.add(best)

        steps.append(RefinementStep(
            expanded=best, expanded_label=graph.labels.get(best, ""),
            score=float(score[index[best]]), added=tuple(additions),
            removed=removed, size_after=len(current)))
    else:
        reason = "iteration limit reached"

    return RefinementResult(
        concepts=tuple(current), steps=tuple(steps), stopped_because=reason,
        sibscore_informative=not graph.is_tree())
