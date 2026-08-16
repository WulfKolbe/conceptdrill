# The math layer at the OCR boundary

**Status: design only.** Steps 16-18 of `newtask.md`. Nothing here is
implemented, and one of the three sections exists mainly to say what the layer
must never be allowed to do.

Read `docs/math-layer.md` first for the layer itself, and
`experiments/math-layer/neighbourhood.md` for the measurement that constrains
this design more than any preference does.

---

## 16. The interface

PDFDRILL has a region of a page it cannot read confidently — a smudged
subscript, an `l`/`1`/`I` ambiguity, a symbol that could be `\in` or `\epsilon`.
It asks this layer what the mathematics would allow.

```
interpret_region(
    visual:  ranked glyph/LaTeX candidates from OCR, each with its own score
    spoken:  the spoken form of the surrounding mathematics, if any
    context: the MathConcepts already established for this span
    ces:     the span's CES concepts            ← provenance only, see below
) -> ranked LaTeX interpretations, each with a semantic justification
```

**The return is a re-ranking of the caller's candidates, never a new one.**

That is the whole shape of it, and it is deliberate. If the layer may propose
LaTeX that OCR did not see, it becomes a generator, and a generator with a
plausible prior will confidently invent a subscript that was never on the page.
The set of candidates is the caller's; the ordering may be argued with.

Two consequences follow immediately:

- **An empty visual candidate list returns an empty result.** Not a guess. A
  region OCR could not read at all is a region this layer has nothing to say
  about, and saying so is the correct output.
- **Every returned interpretation carries a justification naming the evidence**
  — which `MathConcept`, which relation, which Loogle-supported anchor. An
  unjustified re-rank is a silent overrule of the only party that actually
  looked at the page.

### What each input is worth

| input | weight in the decision | why |
|---|---|---|
| visual candidates | **primary** | the only evidence about what is on the page |
| spoken math | strong | independently derived from the same region by SRE; agreement between two readings is real corroboration |
| MathConcepts | moderate | constrains *type*: if `T` is a linear map with domain `V`, then `T(v)` is well-formed and `T + v` is not |
| Lean anchors | narrow but sharp | rules a form out entirely when it cannot typecheck in principle |
| CES concepts | **zero** | measured — see below |

**CES gets weight zero, and that is a measurement, not an opinion.** Step 15
found one basis row that is top-1 for four of six mathematical continuations,
a total spread of 0.651-0.762 across probes as unrelated as *vector space* and
*Reidemeister theorem*, and no threshold that admits the right answers. A
signal like that used as an OCR prior would not be neutral — it would actively
mislead, because it is confidently wrong rather than uninformative. CES stays
in the interface signature as `evidence.ces_row_ids`, for provenance, and
touches no score.

### Justification, concretely

For a region OCR reads as either `T(v)` or `T\{v\}`, with `MathConcept`
`linear map` established (`T`, domain `V`, codomain `W`, anchor `LinearMap`
supported):

```
1. T(v)     0.81   application of a linear map to a vector
                   evidence: mc_4d2e88af16c0 (linear map, observed),
                             relation domain -> V,
                             anchor LinearMap [supported, 11118 hits]
2. T\{v\}   0.19   no established relation licenses set-brace application
```

The second candidate is **kept, ranked, and explained** — not deleted. Deleting
it would destroy the caller's ability to overrule this layer with better visual
evidence.

---

## 17. The boundary

Three kinds of claim, and the layer must never let one masquerade as another.

| | question it answers | authority | may it change what the page says? |
|---|---|---|---|
| **Visual evidence** | what marks are on the page | OCR alone | yes — it is the source |
| **Semantic prediction** | what mathematics would be coherent here | this layer | **no** — priors and ordering only |
| **Formal validation** | what typechecks in Mathlib | Lean/Loogle | no — it may only *exclude* |

**The rule: the math layer never writes into the visual channel.** It returns a
ranking and a justification. PDFDRILL decides. If a math-layer preference and
the OCR's own top candidate disagree, both survive into the output with their
scores, and the disagreement is recorded rather than resolved silently.

This is the same rule ConceptDrill already applies to its inputs — *never write
to the input document; output goes to a sidecar* — applied one layer out.

**Formal validation excludes, it does not select.** A Loogle hit means a name
exists. It does not mean the paper used it, and a region that fails to
typecheck under this layer's guessed context may be perfectly correct
mathematics whose context was guessed wrongly. So a failed formal check lowers
a rank and never removes a candidate.

**The failure mode this section exists to prevent** is circularity: the layer
predicts `\epsilon`, PDFDRILL records `\epsilon`, a later drill reads
`\epsilon` as observed evidence, and a prediction has laundered itself into an
observation. The defence is the `observed | implied | predicted | external`
status that `MathConcept` already carries on every symbol and every relation.
**Anything this layer contributed must be re-readable as `predicted` forever.**
A pipeline that cannot distinguish what it read from what it guessed has no way
back.

---

## 18. The benchmark

**The quantity: does semantic context reduce the number of mathematical OCR
candidates that must be considered, without discarding the correct one?**

Two numbers, always together, because either alone is trivially gameable:

- **Candidate reduction** — how much of the candidate set is eliminated or
  demoted. Maximised by returning nothing.
- **Retention** — how often the ground-truth reading is still in the top *k*.
  Maximised by changing nothing.

A method is useful only if it moves both. Report them as a pair, never as a
single blended score.

### Arms

Ablated, because step 15 already showed that lumping the inputs together would
hide a component that contributes nothing — or worse than nothing.

| arm | inputs |
|---|---|
| A | visual only — the baseline |
| B | visual + spoken |
| C | visual + MathConcept relations |
| D | visual + Lean anchors |
| E | visual + CES concepts |
| F | everything |

**Arm E is the control, and it is expected to fail.** Including it is the point:
if E moves the numbers, step 15's conclusion is wrong on this task and should
be revisited; if E is flat, F's gains are attributable to B, C and D. An
experiment with no arm that can lose is not an experiment.

### Data

Regions where the answer is already known, so nothing has to be hand-labelled:
take drilled documents whose LaTeX source is available, degrade the rendered
region (blur, downsample, JPEG), run OCR, and keep the true reading. The
`props.latex_original` already in the docmodel is the ground truth.

**Stratify by ambiguity class** — `l/1/I`, `\in`/`\epsilon`, sub/superscript
position, bracket kind, single/multi-letter identifier — and report per class.
An aggregate will be dominated by whichever class is most common, and the
classes are where the mechanism either works or does not.

### What would make the result void

Stated in advance, so they cannot be decided after seeing the numbers:

1. **Ground truth derived from any model in the pipeline.** It comes from the
   LaTeX source or it does not exist.
2. **The same LLM proposing candidates and judging them.**
3. **A tuning pass on the test regions.** Split before looking.
4. **Reporting reduction without retention**, or one number combining them.
5. **A run whose noise floor was not measured first.** The identical-config
   repeat established ±3 and ±21 on the corpus gates; nothing smaller than a
   measured floor counts as a difference here either.

### The honest prior

Based on step 15: expect **B and C to help, D to help rarely but decisively
when it fires, and E to do nothing.** If the measured outcome is that F beats A
only by way of B, then the useful part of this whole layer is *spoken math
corroborating OCR* — which would be worth knowing, cheap to ship, and a much
smaller thing than the architecture above.

That outcome should be reported as the result, not treated as a
disappointment.
