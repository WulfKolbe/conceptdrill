# Steps 13-15: the semantic neighbourhood, measured

Produced by `experiments/math-layer/ces_neighbourhood.py`; raw numbers in
`ces-neighbourhood.json`. Encoder `modernbert`, revision
`d556a88e332558790b210f7bdbe87da2fa94a8d8` — **the same revision the corpus
basis was built with**, so the query vectors and the stored rows are in one
space.

Retrieval pool: **5,008 concept rows** from 16 chunk bases, concatenated. Not a
merge: each chunk was integrated independently, so one concept may hold several
rows. Adequate for "what does this corpus put near this text", which is the
only question asked here.

---

## Step 13. Ranked neighbourhood for "vector space"

| rank | cos | concept the corpus returns |
|---|---|---|
| 1 | 0.717 | concept vector constructed from activation deviation over baseline words |
| 2 | 0.703 | free vector space on a set constructed via the free functor from the category of sets |
| 3 | 0.702 | definition of workspace-like vector representations by five functional properties |
| 4 | 0.685 | mapping of any vector from the original latent embedding space to a human-understandable conceptual space |

**The top hit is the NLP sense.** `tests/fixtures/vector-space.md` §4 predicted
exactly this collision from string counts alone; here it is in vector space,
where it was supposed to be resolved. Rank 2 is the only genuinely algebraic
entry in the top four, and it is a *category-theoretic* construction rather
than the definition.

For the fixture's own first sentence — *V is an n-dimensional vector space over
K* — the top hit is "smooth projective variety whose points are in bijection
with the collection of all linear subspaces" at 0.668. Adjacent mathematics,
not the concept.

The `MathConcept` neighbourhood, by contrast, is entirely usable, because it
was not built by nearest-neighbour: see `mathconcepts.json`, where every entry
carries its spoken form, exact LaTeX, symbol status, a Loogle-verified anchor
and its evidence. That contrast is the finding.

---

## Step 14. Context dependence

Second statement, differing from the fixture's third sentence only in *linear*
versus *a linear transformation* — the mathematics is identical:

> Let T : V → W be a linear transformation.

| comparison | top-1 change | top-8 membership retained |
|---|---|---|
| sentence alone | **+0.0018** | 4 of 8 |
| whole three-sentence passage | **+0.0159** | 6 of 8 |

**Additions** (sentence alone): *quotient of a coalgebra by a congruence*,
*bijection between multisets and diagrams*, *the equivalence of injective and
surjective properties for functions mapping a finite set to itself*, *bounded
bilinear map from the general linear group to matrix space*.
**Removals**: *estimation of a linear order on a finite non-empty set*,
*triviality of the kernel via preservation of puncture-connecting lines*, and
two others.

Read carefully, this is one result and it is not "the neighbourhood is stable":

- The **score** barely moves — 0.26% on a 0.68 cosine.
- **Half the top-8 changes anyway.**

Both can be true because every candidate sits in a 0.65-0.72 band. When the
gaps between ranks 1 and 8 are smaller than the difference between two
paraphrases, the ordering among them is arbitrary, and the churn is not
evidence of semantic sensitivity. A layer that treated top-k membership as a
signal would read noise as meaning.

The honest summary of step 14: **two statements that mean the same thing
produce the same score and a different list, and the list is the unreliable
half.**

---

## Step 15. Prediction from semantic context

Candidate continuations, each spoken as SRE would render it, ranked by their
best match in the corpus. No OCR, no glyphs — semantic context only.

| continuation | best cos | what the corpus returned |
|---|---|---|
| rank-nullity | **0.762** | eigenvectors of a tensor as fixed and base points of its gradient map |
| matrix of T | **0.761** | eigenvectors of a tensor as fixed and base points of its gradient map |
| isomorphism | 0.712 | the equivalence of injective and surjective properties for functions on a finite set |
| linearity | 0.701 | eigenvectors of a tensor as fixed and base points of its gradient map |
| kernel | 0.691 | eigenvectors of a tensor as fixed and base points of its gradient map |
| basis | 0.651 | proof by contradiction that assumes the negation of the desired conclusion |

**One row is the top hit for four of six continuations**, and a second row —
*Reidemeister theorem for smooth link diagrams* — tops three of the remaining
probes. Two rows out of 5,008 win 7 of 13 probes in this run. That is the hub
behaviour `projection_stats` already watches for: *"a basis row that never wins
is dead weight; one that wins everything is a sink."*

The whole spread, over thirteen probes ranging from *vector space* to
*Reidemeister* to *rank-nullity*, is **0.651 to 0.762**. There is no threshold
in that band that admits the right answers and rejects the wrong ones, because
the wrong answers score higher than the right ones.

Only *isomorphism* retrieved something genuinely apt, and it is the one
continuation whose wording ("injective and surjective") is generic enough to
appear in any field.

---

## What this means for the math layer

**CES retrieval contributes nothing usable to mathematical prediction on this
corpus.** Measured, not suspected. The consequence for the design in
`docs/math-layer.md`:

- `MathConcept.evidence.ces_row_ids` stays, as **provenance** — the record of
  which span this came from.
- CES similarity must **not** be used to rank, gate, or select mathematical
  concepts. §7.3 of the design listed this as an open question. It is now
  answered for this corpus, in the negative.
- The ranking signal that *did* work is the one in `mathconcepts.json`:
  observed-versus-implied status, plus a Loogle lookup that resolves or does
  not. Both are discrete facts about the text and about Mathlib. Neither is a
  cosine.

**The scope of that conclusion, stated precisely.** This measures a corpus of
278 ML, NLP and topology papers that contains no definitional linear algebra.
It does **not** show that CES cannot represent mathematics. The distinguishing
experiment is cheap and is not run here: build a basis from a corpus of
mathematics textbooks and repeat these thirteen probes. If *vector space* still
returns activation-deviation concept vectors, the fault is in the
representation; if it returns the definition, the fault was corpus coverage.
Until that runs, the honest claim is the narrow one.

**A caveat that cuts the other way.** These probes are 3-12 words. The basis
rows are 22-28 word noun phrases produced by an LLM under a word budget. Short
queries against long labels is not the regime the basis was tuned for, and
`project_sentences` normally embeds full sentences. Some of the flatness in
that 0.65-0.72 band may be a length mismatch rather than a semantic one. The
`fixture-whole` and `variant-whole` probes are full passages and behave the
same way, which is evidence against that explanation but not a controlled test
of it.
