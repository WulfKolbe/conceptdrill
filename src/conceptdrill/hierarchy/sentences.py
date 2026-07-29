"""Splitting document text into the units that get projected.

The sentence is the projection unit, so a bad split is not cosmetic: a fragment
embeds as a fragment, and that fragment is what the whole corpus is compared
against. Two failure modes matter and pull in opposite directions —
over-splitting on an abbreviation, and under-splitting a whole paragraph into
one vector.

**Rule-based, not stanza.** stanza costs about seven seconds of model load per
process and would be invoked over tens of thousands of sentences; it is also
neural, so its output shifts with the model version. This splitter is
deterministic, dependency-free and inspectable, which for a projection unit
matters more than the last percent of accuracy. `nlp.py`'s tiering is available
if a corpus ever needs it.

The abbreviation list is drawn from what the drilled corpus actually contains,
including `e.g.` (which otherwise leaves a stray `g.`) and rendered LaTeX macros
like `ALG.`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

#: Abbreviations that essentially NEVER end a sentence: each is followed by the
#: thing it introduces, so a period here is punctuation, not a boundary.
NEVER_FINAL = frozenset("""
e.g i.e cf viz vs fig figs eq eqs sec secs ch chs no nos vol vols
pp p resp approx ref refs tab tabs alg algs def defs
thm lem cor prop rem ex app appx w.r.t s.t i.i.d a.k.a
dr prof mr mrs ms st jr sr dept univ
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue wed thu fri sat sun
""".split())

#: Abbreviations that CAN end a sentence. "Shown by Vaswani et al." is a
#: complete sentence, and English writes one period for both jobs, so the
#: ambiguity is real and unresolvable from punctuation alone. These are allowed
#: to split: a following capital is better evidence of a new sentence than the
#: abbreviation is of continuation. Under-splitting fuses two ideas into one
#: vector, which is the worse error for a projection unit.
MAY_BE_FINAL = frozenset("""
al etc inc ltd co corp incl excl est max min avg std
""".split())

#: Everything recognised as an abbreviation, for callers that just want to ask.
ABBREVIATIONS = NEVER_FINAL | MAY_BE_FINAL

#: A candidate boundary: sentence punctuation, then space, then something that
#: can begin a sentence. Requiring the follower prevents splitting "3. Method"
#: mid-list and "0.85" mid-number.
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])')

#: The word immediately before a candidate boundary.
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.[\"')\]]*\s*$")

#: A single initial, as in "J. Smith" or "A. B. Author".
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.[\"')\]]*\s*$")

#: A decimal or version number straddling the candidate.
_DECIMAL = re.compile(r"\d\.$")


def _is_real_boundary(before: str) -> bool:
    """Does the text ending here actually end a sentence?"""
    if _DECIMAL.search(before):
        return False
    if _INITIAL.search(before):
        return False
    match = _LAST_WORD.search(before)
    if match:
        word = match.group(1).lower().rstrip(".")
        if word in NEVER_FINAL:
            return False
        # MAY_BE_FINAL deliberately falls through and splits.
        # A single letter before a period is an initial or an enumerator.
        if len(word) == 1:
            return False
    return True


def split_sentences(text: str, *, min_chars: int = 2) -> list[str]:
    """Sentences, in order. Never returns empty or whitespace-only strings.

    Paragraph breaks are hard boundaries: a blank line ends a sentence whether
    or not it carries punctuation, which matters because drilled text often
    loses the final period of a heading-like line.
    """
    if not text or not text.strip():
        return []

    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        start = 0
        for match in _BOUNDARY.finditer(block):
            if not _is_real_boundary(block[start:match.start()]):
                continue
            piece = block[start:match.start()].strip()
            if len(piece) >= min_chars:
                out.append(piece)
                start = match.end()
        tail = block[start:].strip()
        if len(tail) >= min_chars:
            out.append(tail)
    return out


@dataclass(frozen=True)
class Sentence:
    """One projectable unit, with enough provenance to trace it back."""
    id: str
    text: str
    section_id: Optional[str]
    #: Id of the paragraph or formula this came from.
    source_id: str = ""
    #: Position within the document, then within the source.
    flow_index: int = 0
    ordinal: int = 0

    @property
    def is_usable(self) -> bool:
        return bool(self.text.strip())


def sentences_from_tree(tree, *, levels: Optional[set[int]] = None,
                        include_orphans: bool = True,
                        min_chars: int = 20) -> list[Sentence]:
    """Every sentence in a document, in reading order.

    `min_chars` defaults to 20 rather than 0: a five-character fragment embeds
    to a near-arbitrary direction and pollutes any space it is projected into.
    Short fragments are dropped, not kept as noise.

    Orphan paragraphs are included by default. They are front matter — usually
    the abstract — which is some of the most concept-dense text in a paper, and
    excluding it because no section owns it would be a poor trade.
    """
    out: list[Sentence] = []
    seen_sections = set()

    for node in tree.iter_document_order():
        if levels is not None and node.level not in levels:
            continue
        seen_sections.add(node.id)
        for para in node.paragraphs:
            for i, piece in enumerate(split_sentences(para.text)):
                if len(piece) < min_chars:
                    continue
                out.append(Sentence(
                    id=f"{para.id}#s{i}", text=piece, section_id=node.id,
                    source_id=para.id, flow_index=para.flow_index, ordinal=i))

    if include_orphans and levels is None:
        for para in tree.orphans:
            for i, piece in enumerate(split_sentences(para.text)):
                if len(piece) < min_chars:
                    continue
                out.append(Sentence(
                    id=f"{para.id}#s{i}", text=piece, section_id=None,
                    source_id=para.id, flow_index=para.flow_index, ordinal=i))

    out.sort(key=lambda s: (s.flow_index, s.source_id, s.ordinal))
    return out


def sentence_stats(sentences: Sequence[Sentence]) -> dict[str, Any]:
    """Shape of the sentence set, for reporting rather than decoration."""
    if not sentences:
        return {"sentences": 0}
    lengths = [len(s.text.split()) for s in sentences]
    lengths.sort()
    return {
        "sentences": len(sentences),
        "sources": len({s.source_id for s in sentences}),
        "sections": len({s.section_id for s in sentences if s.section_id}),
        "orphaned": sum(1 for s in sentences if s.section_id is None),
        "words_min": lengths[0],
        "words_median": lengths[len(lengths) // 2],
        "words_max": lengths[-1],
    }
