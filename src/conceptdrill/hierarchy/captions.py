"""Cleaning LaTeX residue out of DocModel span captions.

The Semantic Compiler ingests LaTeX and stores marker titles under
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
markers, or files.
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


def clean_caption_traced(raw: str) -> tuple[str, tuple[str, ...]]:
    """`clean_caption`, plus the names of the rules that actually fired.

    Which cleaner ran is provenance, not a detail. The pylatexenc path and the
    regex fallback produce different text for the same input — `\\tau` becomes
    `τ` or `tau` — so a run that cannot say which one ran cannot explain its own
    basis vectors. The tier is decided per call because the fallback is reached
    by an exception, not by a configuration flag.
    """
    if not raw or not raw.strip():
        return "", ("empty",)

    rules: list[str] = []
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        text = LatexNodes2Text().latex_to_text(raw)
        rules.append("pylatexenc")
    except Exception as exc:
        text = _fallback_clean(raw)
        rules.append(f"regex-fallback({type(exc).__name__})")

    text = re.sub(r"\s+", " ", text).strip()
    # A cleaner that eats everything is worse than one that does nothing.
    if not text:
        text = _fallback_clean(raw)
        rules.append("regex-fallback(empty-result)")
    if text != raw.strip():
        rules.append("whitespace-collapsed" if text == re.sub(r"\s+", " ", raw).strip()
                     else "macros-expanded")
    return text, tuple(rules)


def clean_caption(raw: str) -> str:
    """Plain-text form of a LaTeX-ish caption. Never returns None.

    Tries pylatexenc first because it handles nesting and symbol mapping
    properly; falls back to a regex cleaner so the pipeline still runs without
    it. Both paths collapse whitespace, so a dropped macro cannot leave a
    ragged title like ' Application'.
    """
    return clean_caption_traced(raw)[0]


def caption_cleaner_tier() -> str:
    """Which cleaner this process will reach for: `pylatexenc` or `regex`.

    A manifest-level answer. Individual captions can still fall back on a
    parse error, which `clean_caption_traced` records per call.
    """
    try:
        import pylatexenc.latex2text  # noqa: F401
        return "pylatexenc"
    except Exception:
        return "regex"


#: A DocModel inline placeholder: `{{<bibkey>_<rest>||<KIND>}}`.
#: 186 of these appear across 47 of the reference paper's 85 paragraphs — 55%.
#: Left in place they would be tokenized as noise by every embedding model.
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}|]*?)\s*\|\|\s*([A-Z]+)\s*\}\}")

#: What each placeholder kind becomes. `FO` marks a formula slot: naming it
#: keeps the sentence grammatical where deleting it would leave "Let be a space
#: of textual objects". `CIT` is different — the citekey is real signal, since a
#: span citing roberta or glove is partly *about* those things.
_PLACEHOLDER_WORDS = {"FO": "formula", "TAB": "table", "FIG": "figure",
                      "ALG": "algorithm", "EQ": "equation"}


def _placeholder_text(match: re.Match) -> str:
    ident, kind = match.group(1), match.group(2)
    if kind == "CIT":
        # `2209.00445_REF_roberta` -> `roberta`
        key = ident.split("_REF_")[-1] if "_REF_" in ident else ident.split("_")[-1]
        key = re.sub(r"[^A-Za-z0-9]+", " ", key).strip()
        return f" {key} " if key else " "
    return f" {_PLACEHOLDER_WORDS.get(kind, kind.lower())} "


def clean_body_text(text: str) -> str:
    """Strip DocModel placeholders out of paragraph text.

    Formula and float references become a neutral word; citation references
    become their citekey, which carries meaning. Everything else in the text is
    left alone — this is body prose, not a caption, and aggressive LaTeX
    stripping here would do more harm than good.
    """
    if not text:
        return ""
    out = _PLACEHOLDER.sub(_placeholder_text, text)
    return re.sub(r"[ \t]+", " ", out).strip()


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
