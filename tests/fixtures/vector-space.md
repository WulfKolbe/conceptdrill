# Fixture: two vector spaces and a linear map

The smallest piece of real mathematics that exercises every relation in
`docs/math-layer.md` and contains no proof, no numbers, and no domain
vocabulary. Three sentences, six symbols, and — deliberately — several concepts
that are *implied* rather than stated, so a math layer that cannot tell the
difference fails here rather than on a corpus document.

**This is expected output, hand-authored, except where marked GENERATED.**

---

## 1. Source text

> V is an n-dimensional vector space over K.
> W is an m-dimensional vector space over K.
> Let T : V → W be linear.

## 2. LaTeX

Preserved exactly. The colon is written two ways on purpose — see §5.

```latex
Let $V$ be an $n$-dimensional vector space over $K$.
Let $W$ be an $m$-dimensional vector space over $K$.
Let $T : V \to W$ be linear.
```

Alternate spelling of the third line, same mathematics:

```latex
Let $T \colon V \rightarrow W$ be linear.
```

## 3. Spoken

**GENERATED** by `speech.load_speaker()` (la2speech → speech-rule-engine,
clearspeak), 2026-08-16. Regenerate rather than trust: SRE output is a property
of the installed engine.

| LaTeX | spoken |
|---|---|
| `V` | `V` |
| `W` | `W` |
| `K` | `K` |
| `n` | `n` |
| `m` | `m` |
| `T : V \to W` | `T colon V right arrow W` |
| `T \colon V \rightarrow W` | `T colon V right arrow W` |
| `\dim V = n` | `dimension V equals n` |
| `\dim W = m` | `dimension W equals m` |
| `\dim_K V = n` | `the dimension sub K of V equals n` |
| `T(\alpha u + \beta v) = \alpha T(u) + \beta T(v)` | `T of open paren alpha u plus beta v close paren equals alpha T of u plus beta T of v` |

Rendered as this pipeline would see it, prose and spoken math joined the way
`MarkerNode.body_text` joins them:

> V is an n-dimensional vector space over K. W is an m-dimensional vector
> space over K. Let T colon V right arrow W be linear.

**Two observations that matter more than the table.**

`\to` and `\rightarrow` produce *identical* speech. The spoken form is lossy
with respect to the LaTeX spelling, which is why `MathConcept.latex` is a list
of exact strings and not a normalised one.

`\dim V` speaks as `dimension V` — the word *dimension* reaches the embedder
from the mathematics even though the prose never says it. Where the prose says
`n-dimensional`, the concept arrives twice by two routes.

## 4. Ranked mathematical concepts

Ranked by how directly the text supports them, not by importance. `basis` is
last and is the honest failure case: nothing in these three sentences mentions
one.

CES evidence is from the 278-document corpus run
(`~/conceptdrill-corpus-llm/full`), counting occurrences of the phrase in
concept labels.

| # | concept | support in this text | in CES today |
|---|---|---|---|
| 1 | vector space | **observed** — named twice, verbatim | 97 labels, all as a component of an applied concept, none definitional |
| 2 | dimension | **observed** — `n-dimensional`, `m-dimensional`; also via `\dim` | 3756 occurrences, overwhelmingly the informal "quality dimension" sense |
| 3 | linear map | **observed** — the adjective *linear* plus `T : V \to W` | `linear map` 74, `linear transformation` 32 |
| 4 | domain | **implied** — by the notation `V \to W`, never named | 3028, almost all the informal "problem domain" sense |
| 5 | codomain | **implied** — same notation, never named | 22 |
| 6 | field | **implied** — by the phrase *vector space over K*; the word *field* does not occur | 1460, dominated by "research field" |
| 7 | basis | **predicted** — not in the text at all; follows from *dimension* by definition | 388 |

**The CES column is the finding, not decoration.** Every one of these concepts
has a high raw count and almost no definitional presence: `dimension`,
`domain` and `field` are frequent because English reuses those words for
something else entirely. A math layer that matches concepts to CES rows by
string will collide with the informal sense at every step, and cosine in CES
space will not separate them either, because the informal sense is what the
rows were built from.

## 5. Relations

Closed vocabulary from `docs/math-layer.md` §4. `status` is `observed` (the
text says it), `implied` (follows from a phrase or notation in the text), or
`external` (true of the concept, absent from this text).

| subject | relation | object | status | warrant |
|---|---|---|---|---|
| `V` | is-a | vector space | observed | "V is an ... vector space" |
| `V` | has-dimension | `n` | observed | "n-dimensional" |
| `V` | over-field | `K` | observed | "over K" |
| `W` | is-a | vector space | observed | "W is an ... vector space" |
| `W` | has-dimension | `m` | observed | "m-dimensional" |
| `W` | over-field | `K` | observed | "over K" |
| `T` | is-a | linear map | observed | "Let T ... be linear" + function notation |
| `T` | maps-to | `V \to W` | observed | `T : V \to W` |
| `T` | domain | `V` | implied | left of the arrow; the text never says "domain" |
| `T` | codomain | `W` | implied | right of the arrow; the text never says "codomain" |
| `K` | is-a | field | implied | *vector space over K* presupposes it; not stated |
| dimension | related-to | basis | external | definitional, from outside this text |
| linear map | related-to | vector space | external | definitional, from outside this text |

**Facts this text does NOT license.** Each of these is a plausible reading and
none of them is written down; a layer that emits any of them has invented a
fact.

- That `n` and `m` are finite, or natural numbers. `n-dimensional` is
  compatible with an infinite cardinal.
- That `V \ne W`, or that `n \ne m`.
- That `T` is injective, surjective, or an isomorphism.
- That `T` has a matrix. That requires a chosen basis, which no sentence gives.
- Rank, nullity, kernel, or image. None appears.
- That `K` is commutative, infinite, or of characteristic zero — a field by the
  usual convention, but the text says only `K`.

## 6. Symbols

LaTeX preserved exactly. `observed` means the string occurs in §2;
`predicted` means the concept usually carries this symbol and it is **not**
present here.

### Observed

| latex | concept | role |
|---|---|---|
| `V` | vector space | instance |
| `W` | vector space | instance |
| `K` | field | instance |
| `n` | dimension | value |
| `m` | dimension | value |
| `T` | linear map | instance |
| `T : V \to W` | linear map | signature |
| `T \colon V \rightarrow W` | linear map | signature, alternate spelling |
| `\to` | maps-to | operator |
| `\rightarrow` | maps-to | operator, alternate spelling |

### Predicted

Not in this text. A continuation of a document that opened this way would very
likely introduce them, and that likelihood is exactly what must not be recorded
as observation.

| latex | concept | role |
|---|---|---|
| `\dim V` | dimension | derived |
| `\dim W` | dimension | derived |
| `\dim_K V` | dimension | derived, field made explicit |
| `\mathbb{K}` | field | instance, alternate typography |
| `\mathbb{F}` | field | instance, common alternative letter |
| `\ker T` | kernel | derived |
| `\operatorname{im} T` | image | derived |
| `\{v_1, \dots, v_n\}` | basis | instance |
| `\mathcal{B}` | basis | instance |
| `T(v)` | linear map | application |
| `\operatorname{Hom}_K(V, W)` | linear map | the space of them |
| `K^{m \times n}` | matrix | representation, requires a chosen basis |

## 7. Candidate formal anchors

**Every one is `unverified`**: proposed offline as a string, resolved against
no Mathlib revision. Recorded so a verification step has something to check,
never as a claim that the name exists.

| concept | Lean (unverified) | Loogle query |
|---|---|---|
| vector space | `Module` | `Module ?K ?V` |
| field | `Field` | `Field ?K` |
| dimension | `Module.rank`, `FiniteDimensional.finrank` | `Module.rank ?K ?V` |
| linear map | `LinearMap` | `\|- _ →ₗ[_] _` |
| basis | `Basis` | `Basis ?ι ?K ?V` |
| domain / codomain | — | — |

`domain` and `codomain` get no anchor on purpose. In Mathlib they are not
objects but positions in a type, so the honest entry is an empty `lean` list
with the Loogle query retained — a query with no result is a finding.

## 8. What this fixture is for

A math layer run against §1 should reproduce §4-§7 and, more importantly,
should reproduce the **status** column. Getting `vector space` right is easy.
The test is whether it reports `field` as implied rather than observed, `basis`
as predicted rather than present, and declines to emit any line from the
"does NOT license" list.
