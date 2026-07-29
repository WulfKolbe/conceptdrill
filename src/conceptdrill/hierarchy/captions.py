"""Cleaning LaTeX residue out of DocModel section captions.

The Semantic Compiler ingests LaTeX and stores section titles under
`props.caption`, but it does not expand macros. Real examples from
`2209.00445/model.docmodel.json`:

    '\\ALG\\ Application'
    '\\emph{Siblings} score'
    'Testing the effect of the $removeP$ parameter'
    'Testing the effect of the $\\tau$ function'

These must not reach an embedding model unfiltered — backslashes and dollar
signs tokenize into noise and would pollute every basis vector derived from the
title.

`clean_caption` is pure text-in/text-out. It knows nothing about the DocModel,
sections, or files.
"""
from __future__ import annotations

import re

#: Macros whose *argument* is the visible text: keep the argument, drop the name.
#: Their name disappearing is correct behaviour, not data loss — see `lost_macros`.
FORMATTING_MACROS = frozenset({
    "emph", "text", "textit", "textbf", "textrm", "texttt", "textsc",
    "mathrm", "mathbf", "mathit", "textnormal", "mbox", "operatorname",
})

_TEXT_MACROS = re.compile(
    r"\\(?:" + "|".join(sorted(FORMATTING_MACROS, key=len, reverse=True)) +
    r")\s*\{([^{}]*)\}")

#: A bare macro with no argument: `\ALG`, `\tau`, `\LaTeX`.
_BARE_MACRO = re.compile(r"\\([A-Za-z]+)\s*")

#: Maths symbols worth spelling out rather than deleting. Deleting them turns
#: "the $\tau$ function" into "the function", which changes the meaning.
#:
#: Each maps to (word, unicode). Both forms count as "preserved": the regex
#: fallback emits the word, while pylatexenc emits the character — checking only
#: the word made `lost_macros` report a preserved τ as lost.
_SYMBOLS = {
    "alpha": ("alpha", "α"), "beta": ("beta", "β"), "gamma": ("gamma", "γ"),
    "delta": ("delta", "δ"), "epsilon": ("epsilon", "ε"), "zeta": ("zeta", "ζ"),
    "eta": ("eta", "η"), "theta": ("theta", "θ"), "kappa": ("kappa", "κ"),
    "lambda": ("lambda", "λ"), "mu": ("mu", "μ"), "pi": ("pi", "π"),
    "rho": ("rho", "ρ"), "sigma": ("sigma", "σ"), "tau": ("tau", "τ"),
    "phi": ("phi", "φ"), "psi": ("psi", "ψ"), "omega": ("omega", "ω"),
}

#: Name -> word, for the regex fallback.
_SYMBOL_WORDS = {name: word for name, (word, _) in _SYMBOLS.items()}


def _fallback_clean(raw: str) -> str:
    """Regex cleaner, used when pylatexenc is unavailable.

    Deterministic and dependency-free. Weaker than pylatexenc, but the pipeline
    must not depend on an optional package for something this load-bearing.
    """
    text = _TEXT_MACROS.sub(r"\1", raw)

    def _macro(match: re.Match) -> str:
        name = match.group(1)
        word = _SYMBOL_WORDS.get(name.lower())
        return f" {word} " if word else " "

    text = _BARE_MACRO.sub(_macro, text)
    text = text.replace("$", " ").replace("{", " ").replace("}", " ")
    text = text.replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_caption(raw: str) -> str:
    """Plain-text form of a LaTeX-ish caption. Never returns None.

    Tries pylatexenc first because it handles nesting and symbol mapping
    properly; falls back to a regex cleaner so the pipeline still runs without
    it. Both paths collapse whitespace, so a dropped macro cannot leave a
    ragged title like ' Application'.
    """
    if not raw or not raw.strip():
        return ""
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        text = LatexNodes2Text().latex_to_text(raw)
    except Exception:
        text = _fallback_clean(raw)
    text = re.sub(r"\s+", " ", text).strip()
    # A cleaner that eats everything is worse than one that does nothing.
    return text if text else _fallback_clean(raw)


def lost_macros(raw: str, cleaned: str) -> tuple[str, ...]:
    """Macro names present in `raw` whose text is absent from `cleaned`.

    `\\ALG\\ Application` cleans to `Application`: the macro held the paper's
    own name for its algorithm, and dropping it leaves a title that means
    nothing on its own. Callers keep the raw caption when this is non-empty, so
    the loss is recorded rather than silent.

    Formatting macros are **not** losses. `\\emph{Siblings}` is supposed to lose
    the name `emph` and keep the argument `Siblings`; reporting that as lost
    would flag every italicised title in the corpus.
    """
    lowered = cleaned.lower()
    lost = []
    for name in _BARE_MACRO.findall(raw or ""):
        key = name.lower()
        if key in FORMATTING_MACROS:
            continue
        # A symbol survives as either its word or its character, depending on
        # which cleaner ran. Both count as preserved.
        forms = _SYMBOLS.get(key, (name,))
        if not any(form.lower() in lowered for form in forms):
            lost.append(name)
    return tuple(dict.fromkeys(lost))
