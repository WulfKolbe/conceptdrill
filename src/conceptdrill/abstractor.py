"""The `Abstractor` hook — where an LLM would summarise, and what happens without one.

Two places in the spec call for a language model: turning an equation into a
short description ("probability of emission"), and shortening an over-long paper
title. Both are optional enrichments of the tau mapping, not load-bearing steps.

The default `NullAbstractor` is deterministic and offline: it derives an
equation description from the LaTeX structure and truncates titles at a clause
boundary. Every abstractor reports `is_deterministic`, which is recorded in the
output — so a projection always states whether an LLM touched its vocabulary.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Abstractor(Protocol):
    """Turns raw source into a short natural-language concept name."""

    name: str
    is_deterministic: bool

    def describe_equation(self, latex: str, context: str = "") -> str:
        """A short description of what the equation represents, or "" to skip."""
        ...

    def shorten_title(self, title: str, max_words: int = 8) -> str:
        """A condensed form of a paper title, or the title unchanged."""
        ...


# LaTeX operators mapped to the mathematical idea they signal. Used to build a
# structural description when no LLM is available. Ordered most- to
# least-specific so the first hit is the most informative.
_MATH_SIGNALS: tuple[tuple[str, str], ...] = (
    (r"\\iint|\\iiint|\\oint", "multiple integral"),
    (r"\\int", "integral"),
    (r"\\sum", "summation"),
    (r"\\prod", "product"),
    (r"\\lim", "limit"),
    (r"\\frac\s*{\s*\\partial", "partial derivative"),
    (r"\\partial", "partial derivative"),
    (r"\\nabla\s*\\cdot", "divergence"),
    (r"\\nabla\s*\\times", "curl"),
    (r"\\nabla\^?2|\\Delta\b", "Laplacian"),
    (r"\\nabla", "gradient"),
    (r"\\mathbb{E}|\\operatorname{E}|\\mathbb\s*E", "expectation"),
    (r"\\Pr\b|\\mathbb{P}|\bp\s*\(", "probability"),
    (r"\\log|\\ln\b", "logarithm"),
    (r"\\exp\b|e\^", "exponential"),
    (r"\\argmin|\\arg\s*\\min", "minimisation objective"),
    (r"\\argmax|\\arg\s*\\max", "maximisation objective"),
    (r"\\min\b", "minimum"),
    (r"\\max\b", "maximum"),
    (r"\\mathcal{L}|\\mathcal\s*L", "loss function"),
    (r"\\hat{", "estimator"),
    (r"\\bar{", "mean"),
    (r"\\sigma|\\Sigma", "variance or covariance"),
    (r"\\mu\b", "mean parameter"),
    (r"\\theta\b", "parameter vector"),
    (r"\\otimes|\\oplus", "tensor operation"),
    (r"\\langle.*\\rangle", "inner product"),
    (r"\\|.*\\|", "norm"),
    (r"\\sqrt", "square root"),
    (r"\\cos|\\sin|\\tan", "trigonometric expression"),
    (r"\\matrix|\\begin{[bp]matrix}", "matrix expression"),
    (r"\\leq|\\geq|\\le\b|\\ge\b", "inequality"),
    (r"=", "equality"),
)

_CLAUSE_SPLIT = re.compile(r"\s*[:;—–]\s*|\s+-\s+")


class NullAbstractor:
    """Deterministic, offline, no model.

    Equation descriptions come from operator structure rather than meaning, so
    they are coarse — "summation over a loss function" rather than "cross-entropy
    of the class posterior". That is an honest floor, and it is reproducible.
    """

    name = "null"
    is_deterministic = True

    def describe_equation(self, latex: str, context: str = "") -> str:
        if not latex or not latex.strip():
            return ""
        signals: list[str] = []
        for pattern, label in _MATH_SIGNALS:
            if re.search(pattern, latex):
                if label not in signals:
                    signals.append(label)
            if len(signals) == 3:
                break
        if not signals:
            return ""
        # Free variables give the description something document-specific to
        # attach to without pretending to understand the equation.
        if len(signals) == 1:
            return signals[0]
        return f"{signals[0]} over {' and '.join(signals[1:])}"

    def shorten_title(self, title: str, max_words: int = 8) -> str:
        title = re.sub(r"\s+", " ", title or "").strip()
        if not title:
            return ""
        if len(title.split()) <= max_words:
            return title
        # Prefer cutting at a clause boundary: scientific titles put the
        # contribution before the colon and the qualification after it.
        head = _CLAUSE_SPLIT.split(title)[0].strip()
        if head and len(head.split()) <= max_words:
            return head
        return " ".join(title.split()[:max_words])


class CallableAbstractor:
    """Adapts an arbitrary callable into the `Abstractor` protocol.

    The escape hatch for wiring in a real model without subclassing::

        drill = ConceptDrill(doc, abstractor=CallableAbstractor(my_llm))

    `fn` receives a prompt string and returns the completion. Marked
    non-deterministic, which propagates into the output metadata.
    """

    is_deterministic = False

    def __init__(self, fn, *, name: str = "callable",
                 deterministic: bool = False) -> None:
        self._fn = fn
        self.name = name
        self.is_deterministic = deterministic
        self._fallback = NullAbstractor()

    def _call(self, prompt: str) -> str:
        try:
            out = self._fn(prompt)
        except Exception:
            return ""
        return re.sub(r"\s+", " ", str(out or "")).strip().strip('"')

    def describe_equation(self, latex: str, context: str = "") -> str:
        prompt = (
            "Describe in at most six words what this LaTeX equation represents. "
            "Answer with the noun phrase only.\n\n"
            f"Equation: {latex}\n"
        )
        if context:
            prompt += f"Surrounding text: {context[:500]}\n"
        return self._call(prompt) or self._fallback.describe_equation(latex, context)

    def shorten_title(self, title: str, max_words: int = 8) -> str:
        if len(title.split()) <= max_words:
            return title
        out = self._call(
            f"Condense this paper title to at most {max_words} words naming the "
            f"core concept. Answer with the phrase only.\n\nTitle: {title}\n"
        )
        return out or self._fallback.shorten_title(title, max_words)


def get_abstractor(spec: Optional[str] = None) -> Abstractor:
    """Resolve an abstractor by name. Only `null` exists without user wiring."""
    if spec in (None, "", "null", "none", "off"):
        return NullAbstractor()
    raise ValueError(
        f"unknown abstractor {spec!r}; pass a CallableAbstractor instance "
        f"programmatically to use a language model"
    )
