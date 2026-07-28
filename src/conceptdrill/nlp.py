"""Noun-phrase and named-entity extraction, with graceful degradation.

Three tiers, tried in order: **stanza** (already a Semantic Compiler dependency),
**spaCy**, then a **regex** fallback. The fallback is not as good, but it is
deterministic, dependency-free, and keeps the pipeline working rather than
failing — a missing NLP model should cost quality, not the run.

Which tier ran is reported through `backend_name()` and lands in the output
metadata, so a projection never hides the fact that it used the weak tier.

**The tier is a reproducibility control, not just a performance one.** Different
tiers mine different noun phrases, so the same document yields a different
vocabulary under stanza than under regex. `CONCEPTDRILL_NLP_BACKEND` pins it:

    CONCEPTDRILL_NLP_BACKEND=regex    # deterministic, dependency-free, fast
    CONCEPTDRILL_NLP_BACKEND=stanza   # best quality, requires models
    CONCEPTDRILL_NLP_BACKEND=auto     # default: best available

Both neural tiers are invoked **once per batch**, not once per text. Stanza in
particular costs seconds of model overhead per call, so per-block invocation
turns a fast run into a slow one.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

# Words that must not start or end a noun phrase.
STOPWORDS = frozenset("""
a an the this that these those it its it's their there here we our us you your
i he she they them his her hers him who whom whose which what when where why how
and or but nor for so yet if then than as at by from in into of on onto to with
without within about above below over under again further once all any both each
few more most other some such no not only own same too very can will just don
should now is are was were be been being have has had having do does did doing
also thus hence however therefore moreover furthermore whereas while although
though because since unless until upon among between during before after above
one two three four five six seven eight nine ten first second third next last
et al fig figure table section eq equation ref cf e.g i.e etc vs
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

# A crude noun-phrase shape for the fallback tier: optional determiner/adjective
# run followed by capitalised or lowercase nominals. Deliberately conservative.
_FALLBACK_NP = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}"          # Title Case Runs
    r"|[a-z]+(?:\s+[a-z]+){0,3}(?=\s+(?:is|are|was|were|can|will|of|for|in)\b))"
)

# Acronyms: 2-6 capitals, optionally with digits.
_ACRONYM = re.compile(r"\b([A-Z]{2,6}[0-9]{0,2})\b")

# NER types the spec asks for.
WANTED_ENT_TYPES = frozenset({
    "ORG", "PRODUCT", "PERSON", "GPE", "LOC", "NORP", "FAC", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE",
})


@dataclass(frozen=True)
class Phrase:
    text: str
    count: int


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    count: int


# --------------------------------------------------------------------------
# Backend detection
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _stanza_pipeline():
    """A stanza pipeline, or None. Never downloads: a missing model means this
    tier is unavailable, not that the run should stall on a network fetch."""
    try:
        import stanza
    except ImportError:
        return None
    try:
        return stanza.Pipeline(
            lang="en", processors="tokenize,pos,lemma,ner",
            download_method=None, use_gpu=False, verbose=False,
        )
    except Exception:
        # Try again without NER — the POS tagger alone still gives noun phrases.
        try:
            return stanza.Pipeline(
                lang="en", processors="tokenize,pos,lemma",
                download_method=None, use_gpu=False, verbose=False,
            )
        except Exception:
            return None


@lru_cache(maxsize=1)
def _spacy_pipeline():
    try:
        import spacy
    except ImportError:
        return None
    for model in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model)
        except Exception:
            continue
    return None


#: Environment variable pinning the tier. See the module docstring.
BACKEND_ENV = "CONCEPTDRILL_NLP_BACKEND"

VALID_BACKENDS = frozenset({"auto", "stanza", "spacy", "regex"})


@lru_cache(maxsize=8)
def _resolve_backend(requested: str) -> str:
    if requested == "regex":
        return "regex"
    if requested == "stanza":
        return "stanza" if _stanza_pipeline() is not None else "regex"
    if requested == "spacy":
        return "spacy" if _spacy_pipeline() is not None else "regex"
    if _stanza_pipeline() is not None:
        return "stanza"
    if _spacy_pipeline() is not None:
        return "spacy"
    return "regex"


def backend_name() -> str:
    """Which tier is active: `stanza`, `spacy`, or `regex`.

    An explicitly requested tier that is unavailable falls back to `regex`
    rather than raising — the same graceful-degradation rule as everywhere else.
    """
    requested = (os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if requested not in VALID_BACKENDS:
        requested = "auto"
    return _resolve_backend(requested)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def normalise_phrase(text: str) -> str:
    """Collapse whitespace, strip punctuation and leading/trailing stopwords."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\n\r.,;:!?()[]{}\"'`-–—/\\")
    words = text.split()
    while words and words[0].lower() in STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in STOPWORDS:
        words.pop()
    return " ".join(words)


def is_acceptable_phrase(text: str, max_tokens: int = 5) -> bool:
    """Reject junk: too long, too short, all stopwords, mostly non-alphabetic."""
    if not text:
        return False
    words = text.split()
    if not (1 <= len(words) <= max_tokens):
        return False
    if all(w.lower() in STOPWORDS for w in words):
        return False
    letters = sum(c.isalpha() for c in text)
    if letters < max(2, len(text) * 0.5):
        return False
    # A single word must be substantial, unless it is an acronym.
    if len(words) == 1 and len(text) < 4 and not text.isupper():
        return False
    return True


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    """One pipeline pass over a batch of texts.

    Noun phrases and entities come out of the *same* run. Computing them
    separately would double the model cost for no benefit, since both consumers
    aggregate counts across the whole batch anyway.
    """
    phrases: Counter = field(default_factory=Counter)
    entities: Counter = field(default_factory=Counter)
    backend: str = "regex"


@lru_cache(maxsize=8)
def _analyse_cached(texts: tuple[str, ...], max_tokens: int,
                    backend: str) -> Analysis:
    """Memoised so a build's noun-phrase and NER passes share one model run."""
    if backend == "stanza":
        return _stanza_analyse(texts, max_tokens)
    if backend == "spacy":
        return _spacy_analyse(texts, max_tokens)
    return _regex_analyse(texts, max_tokens)


def analyse(texts: Iterable[str], *, max_tokens: int = 5) -> Analysis:
    """Run the active backend over `texts` once."""
    batch = tuple(t for t in texts if t and t.strip())
    if not batch:
        return Analysis(backend=backend_name())
    return _analyse_cached(batch, max_tokens, backend_name())


def noun_phrases(texts: Iterable[str], *, min_count: int = 3,
                 max_tokens: int = 5) -> list[Phrase]:
    """Frequent noun phrases across `texts`, sorted by (-count, text)."""
    counter = analyse(texts, max_tokens=max_tokens).phrases
    return [Phrase(text, count) for text, count in
            sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= min_count]


def named_entities(texts: Iterable[str], *, min_count: int = 1,
                   wanted: Optional[frozenset[str]] = None) -> list[Entity]:
    """Named entities, sorted by (-count, label, text)."""
    wanted = wanted or WANTED_ENT_TYPES
    counter = analyse(texts).entities
    return [Entity(text, label, count) for (text, label), count in
            sorted(counter.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))
            if count >= min_count and label in wanted]


_NOMINAL = frozenset({"NOUN", "PROPN"})
_CHUNK_POS = frozenset({"NOUN", "PROPN", "ADJ"})


def _chunks_from_pos(tokens: list[tuple[str, str]], max_tokens: int) -> list[str]:
    """Greedy noun-phrase chunker over (word, upos) pairs.

    A run of adjectives, nouns and proper nouns — the standard minimal-NP
    pattern. Two constraints keep it honest:

      * the run must contain a **noun**; a run of bare adjectives is not a noun
        phrase, and without this "fine", "necessary" and "total" all surface as
        concepts;
      * trailing adjectives are trimmed, so "hashing efficient" becomes "hashing".
    """
    out: list[str] = []
    run: list[tuple[str, str]] = []

    def flush() -> None:
        if not any(pos in _NOMINAL for _, pos in run):
            return
        trimmed = list(run)
        while trimmed and trimmed[-1][1] == "ADJ":
            trimmed.pop()
        if trimmed:
            out.append(" ".join(w for w, _ in trimmed[-max_tokens:]))

    for word, pos in tokens:
        if pos in _CHUNK_POS:
            run.append((word, pos))
        else:
            flush()
            run = []
    flush()
    return out


def _stanza_analyse(texts: tuple[str, ...], max_tokens: int) -> Analysis:
    """One `bulk_process` call for the whole batch.

    Stanza pays its model overhead per invocation, so calling it per block turns
    a two-second run into a two-minute one.
    """
    nlp = _stanza_pipeline()
    phrases: Counter = Counter()
    entities: Counter = Counter()
    if nlp is None:
        return _regex_analyse(texts, max_tokens)

    try:
        import stanza
        docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts])
    except Exception:
        # bulk_process is unavailable on older stanza; fall back to one call
        # over the joined batch rather than one call per text.
        try:
            docs = [nlp("\n\n".join(texts))]
        except Exception:
            return _regex_analyse(texts, max_tokens)

    for doc in docs:
        for sentence in getattr(doc, "sentences", ()) or ():
            pairs = [(w.text, w.upos or "X") for w in sentence.words]
            for chunk in _chunks_from_pos(pairs, max_tokens):
                phrase = normalise_phrase(chunk)
                if is_acceptable_phrase(phrase, max_tokens):
                    phrases[phrase] += 1
        for ent in getattr(doc, "ents", ()) or ():
            label = _map_label(getattr(ent, "type", ""))
            phrase = normalise_phrase(ent.text)
            if is_acceptable_phrase(phrase):
                entities[(phrase, label)] += 1

    return Analysis(phrases, entities, "stanza")


def _spacy_analyse(texts: tuple[str, ...], max_tokens: int) -> Analysis:
    """One `nlp.pipe` call for the whole batch."""
    nlp = _spacy_pipeline()
    phrases: Counter = Counter()
    entities: Counter = Counter()
    if nlp is None:
        return _regex_analyse(texts, max_tokens)

    try:
        docs = list(nlp.pipe(texts))
    except Exception:
        return _regex_analyse(texts, max_tokens)

    for doc in docs:
        for chunk in doc.noun_chunks:
            phrase = normalise_phrase(chunk.text)
            if is_acceptable_phrase(phrase, max_tokens):
                phrases[phrase] += 1
        for ent in doc.ents:
            phrase = normalise_phrase(ent.text)
            if is_acceptable_phrase(phrase):
                entities[(phrase, ent.label_)] += 1

    return Analysis(phrases, entities, "spacy")


def _regex_analyse(texts: tuple[str, ...], max_tokens: int) -> Analysis:
    """Fallback: title-case runs plus acronyms.

    Biased toward proper nouns and named methods, which in scientific prose is
    where most of the concept signal is anyway. No NER without a model, so
    acronyms stand in and are labelled ORG rather than inventing entity types.
    """
    phrases: Counter = Counter()
    entities: Counter = Counter()
    for text in texts:
        for match in _FALLBACK_NP.findall(text):
            phrase = normalise_phrase(match)
            if is_acceptable_phrase(phrase, max_tokens):
                phrases[phrase] += 1
        for acro in _ACRONYM.findall(text):
            if acro.lower() in STOPWORDS or len(acro) < 2:
                continue
            phrases[acro] += 1
            entities[(acro, "ORG")] += 1
    return Analysis(phrases, entities, "regex")


def _map_label(stanza_type: str) -> str:
    """stanza's OntoNotes labels already match spaCy's; normalise case."""
    return (stanza_type or "").upper()


def word_frequencies(texts: Iterable[str]) -> Counter[str]:
    """Lowercased word counts. Feeds the reusability heuristic."""
    counter: Counter[str] = Counter()
    for text in texts:
        for word in _WORD.findall((text or "").lower()):
            counter[word] += 1
    return counter
