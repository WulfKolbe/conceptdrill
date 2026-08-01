"""Getting embeddable text out of a formula.

The reference paper carries 73 `Formula` objects and one `Equation`, and until
now every one of them was excluded from summarisation: their only content is
`props.latex`, and raw LaTeX embeds as noise. That is 74 objects of the
document's actual mathematics contributing nothing to its concept vectors.

`~/la2speech` solves the hard part — LaTeX to spoken text via the Speech Rule
Engine — and pdfdrill will carry the result in the docmodel in a future update.
Three sources, in order of preference:

  1. **A spoken field already in the docmodel.** Preferred once it lands; the
     compiler has better context than this package does.
  2. **Generated on demand** through a `SpeechBackend` (la2speech's protocol:
     `speak_math(latex) -> str`). Optional, requires node + SRE.
  3. **A deterministic fallback** that names the operators. Coarse, offline,
     and always available.

## Speech text is not automatically embedding text

SRE spells multi-letter identifiers letter by letter, because that is how
mathematics is read aloud: `Obj(c)` becomes "O b j of open paren c". Correct
for a listener, useless for an embedding model, which sees three meaningless
tokens instead of one meaningful one. Measured on this paper:

    AVERAGE_{s \\in siblings(c,p)}
      raw        "A V E R A G of E sub s is a member of s of i b l i n g of s"
      protected  "AVERAGE sub s is a member of the siblings of"

`protect_identifiers` wraps runs of three or more letters in `\\text{}` before
speaking. Pure mathematics is unaffected, so it costs nothing where it does not
help.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

#: Props that may carry pre-computed spoken text. The docmodel does not have
#: this yet; these are the plausible names, tried in order, so the pipeline
#: picks it up the day it appears without a code change.
SPOKEN_PROP_CANDIDATES = (
    "spoken", "spoken_text", "speech", "spoken_math", "latex_spoken",
    "tts", "tts_text",
)

#: Props holding the LaTeX source, in preference order.
LATEX_PROP_CANDIDATES = ("latex", "latex_raw", "latex_original", "latex_code")

#: Three or more letters, not already part of a command. Two-letter runs are
#: left alone: they are far more often two variables than one identifier.
_MULTI_LETTER = re.compile(r"(?<![\\A-Za-z])([A-Za-z]{3,})(?![A-Za-z])")

#: LaTeX commands that are already words, and must not be double-wrapped.
_KNOWN_COMMANDS = re.compile(r"\\(?:text|operatorname|mathrm|mathbf|mathcal)\s*\{")


@runtime_checkable
class SpeechBackend(Protocol):
    """la2speech's shape: LaTeX in, spoken text out."""

    def speak_math(self, latex: str) -> str:
        ...


#: Structural arguments that must never be wrapped: `\begin{cases}` becoming
#: `\begin{\text{cases}}` is invalid LaTeX, SRE fails on it, and la2speech
#: returns "[unspoken math]". That is exactly what happened to the domain-size
#: definition in 1702.06385: a whole display equation reached the corpus as a
#: placeholder string.
_STRUCTURAL_ARG = re.compile(r"\\(?:begin|end|label|ref|eqref|cite)\s*\{[^{}]*\}")


def protect_identifiers(latex: str) -> str:
    r"""Wrap multi-letter identifiers so a speech engine reads them as words.

    Skips text that already contains `\text{}`-style wrappers, so applying
    this twice is harmless, and never touches the argument of a structural
    command -- see `_STRUCTURAL_ARG`.
    """
    if not latex:
        return ""
    if _KNOWN_COMMANDS.search(latex):
        return latex

    # Hold structural arguments aside, protect the rest, put them back.
    held: list[str] = []

    def _hold(match: re.Match) -> str:
        held.append(match.group(0))
        return f"\x00{len(held) - 1}\x00"

    guarded = _STRUCTURAL_ARG.sub(_hold, latex)
    out = _MULTI_LETTER.sub(lambda m: r"\text{" + m.group(1) + "}", guarded)
    for i, original in enumerate(held):
        out = out.replace(f"\x00{i}\x00", original)
    return out


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------

#: Operator -> the words a reader would use. Ordered most- to least-specific.
_OPERATORS: tuple[tuple[str, str], ...] = (
    (r"\\subseteq", " subset of or equal to "),
    (r"\\subset", " subset of "),
    (r"\\supseteq", " superset of or equal to "),
    (r"\\in\b", " in "),
    (r"\\notin\b", " not in "),
    (r"\\cap", " intersection "),
    (r"\\cup", " union "),
    (r"\\times", " times "),
    (r"\\otimes", " tensor product "),
    (r"\\leq|\\le\b", " less than or equal to "),
    (r"\\geq|\\ge\b", " greater than or equal to "),
    (r"\\neq|\\ne\b", " not equal to "),
    (r"\\approx", " approximately "),
    (r"\\sum", " the sum of "),
    (r"\\prod", " the product of "),
    (r"\\int", " the integral of "),
    (r"\\lim", " the limit of "),
    (r"\\frac", " the fraction "),
    (r"\\sqrt", " the square root of "),
    (r"\\log\b", " log "),
    (r"\\exp\b", " exponential "),
    (r"\\partial", " partial derivative "),
    (r"\\nabla", " gradient "),
    (r"\\infty", " infinity "),
    (r"\\forall", " for all "),
    (r"\\exists", " there exists "),
    (r"\\rightarrow|\\to\b", " maps to "),
    (r"\\ldots|\\cdots|\\dots", " and so on "),
    (r"\\langle", " left angle bracket "),
    (r"\\rangle", " right angle bracket "),
    (r"\\mathcal|\\mathbb|\\mathbf|\\mathrm|\\operatorname|\\text", " "),
    (r"\\left|\\right", " "),
    (r"=", " equals "),
)

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho", "sigma": "sigma",
    "tau": "tau", "phi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
}


def fallback_speak(latex: str) -> str:
    """Operator-level rendering, offline and deterministic.

    Far coarser than a speech engine: it names operators and keeps identifiers,
    producing "L equals the sum of y times log p" rather than a faithful
    reading. That is still enormously better than raw LaTeX for an embedder,
    and it always works.
    """
    if not latex or not latex.strip():
        return ""
    text = latex
    for name, word in _GREEK.items():
        text = re.sub(rf"\\{name}\b", f" {word} ", text)
    for pattern, word in _OPERATORS:
        text = re.sub(pattern, word, text)
    text = re.sub(r"[_^]", " ", text)
    text = re.sub(r"[{}$&\\|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------

def spoken_from_props(props: Mapping[str, Any]) -> str:
    """Pre-computed spoken text from the docmodel, or "" if absent."""
    for key in SPOKEN_PROP_CANDIDATES:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def latex_from_props(props: Mapping[str, Any]) -> str:
    for key in LATEX_PROP_CANDIDATES:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


#: Markers a speech backend emits when it could not read the input. They are
#: NOT speech, and accepting one puts the literal string "[unspoken math]"
#: into a span where an equation was -- which is what happened to the
#: `\begin{cases}` definition in 1702.06385.
SPEECH_FAILURE_MARKERS = ("[unspoken math]", "[unspoken", "[math]",
                          "[no speech]", "[error]")


def is_speech_failure(said: str) -> bool:
    """True when the backend returned a placeholder rather than speech."""
    text = (said or "").strip().lower()
    if not text:
        return True
    return any(text.startswith(m) or text == m.strip()
               for m in (m.lower() for m in SPEECH_FAILURE_MARKERS))


#: Macros that render as nothing and mean nothing, but which a speech engine
#: reads aloud. `\ensuremath{X \rightarrow Y}\xspace` came back from SRE as
#: "X right arrow Y backslash xspace": the spacing macro became two words in
#: the middle of the corpus's central claim.
#: Wrappers that carry no meaning and can be unwrapped. `\text` and its
#: relatives are NOT here on purpose: inside maths they are what tells a
#: speech engine "these are words". Removing `\text{if type(a) is nominal}`
#: made SRE spell it out letter by letter -- "i f o f t y p e" -- which is
#: worse than leaving it alone.
_NOOP_WRAPPERS = re.compile(r"\\(?:ensuremath|protect|mbox|hbox)\s*\{")

#: A text wrapper immediately inside another. SRE reads one level correctly
#: and fails on two: `\text{\textit{binary}}` came back as "backslash textit
#: open brace binary close brace". Collapsed to a single wrapper.
_NESTED_TEXT = re.compile(
    r"\\(text|textit|textbf|textrm|texttt|textsc|emph|mathit)\s*\{\s*"
    r"\\(?:text|textit|textbf|textrm|texttt|textsc|emph|mathit)\s*\{([^{}]*)\}\s*\}")
_NOOP_MACROS = re.compile(r"\\(?:xspace|,|;|!|quad|qquad|thinspace|/|@)"
                          r"(?![A-Za-z])")


def strip_noop_macros(tex: str) -> str:
    r"""Remove macros that occupy no meaning. `\ensuremath{..}` keeps its
    argument; `\xspace` and friends go entirely.
    """
    out = tex or ""
    for _ in range(4):                     # bounded: nesting is shallow
        match = _NOOP_WRAPPERS.search(out)
        if not match:
            break
        start = match.end() - 1            # at the opening brace
        depth, i = 0, start
        while i < len(out):
            if out[i] == "{":
                depth += 1
            elif out[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i >= len(out):
            break
        out = out[:match.start()] + out[start + 1:i] + out[i + 1:]
    for _ in range(4):
        collapsed = _NESTED_TEXT.sub(r"\\\1{\2}", out)
        if collapsed == out:
            break
        out = collapsed
    out = _NOOP_MACROS.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def math_text(props: Mapping[str, Any], *,
              speaker: Optional[SpeechBackend] = None,
              protect: bool = True,
              min_latex_chars: int = 3) -> tuple[str, str]:
    """Embeddable text for one math object, and where it came from.

    Returns `(text, source)` with source one of `docmodel` | `speech` |
    `fallback` | `none`, so a run can report how its mathematics was rendered
    rather than leaving it a mystery.

    Formulas shorter than `min_latex_chars` are dropped: this paper is full of
    single-symbol `Formula` objects like `T` and `k`, which as prose contribute
    a stray letter and no concept signal at all.
    """
    spoken = spoken_from_props(props)
    if spoken:
        return spoken, "docmodel"

    latex = latex_from_props(props)
    if len(latex) < min_latex_chars:
        return "", "none"

    if speaker is not None:
        try:
            source_tex = strip_noop_macros(latex)
            source_tex = protect_identifiers(source_tex) if protect else source_tex
            said = (speaker.speak_math(source_tex) or "").strip()
            if said and not is_speech_failure(said):
                return said, "speech"
        except Exception:
            # A speech backend failure must cost quality, not the run.
            pass

    text = fallback_speak(latex)
    return (text, "fallback") if text else ("", "none")
