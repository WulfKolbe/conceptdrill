# An experimental math-semantic layer above CES

**Status: design only. No code exists for this and none is proposed here.**

CES answers *what is this span about*. It does not answer *what mathematical
object is this span talking about*, and it cannot: a basis row is a 20-28 word
English noun phrase embedded by a sentence encoder, and `\dim V = n` is not a
noun phrase. This document describes a layer that sits **above** the finished
pipeline, reads its output, and emits a second kind of record — one that names
mathematical concepts and points at candidate formal anchors in a proof
assistant library.

Above, not inside. Nothing here changes a basis row, a projection, a
`basis_version`, or any file the current pipeline writes.

---

## 1. What already exists, and where

The four things the layer has to attach to, located exactly.

### 1.1 The CES concept data format

There is no single "concept object". There are **four representations of a
concept**, at four stages, and they carry different numbers. Conflating them is
the standing trap.

| stage | type | file | the vector it carries |
|---|---|---|---|
| generated | `SpanSummary` | `src/conceptdrill/hierarchy/summarize.py:110` | none — text only |
| in the basis | `BasisRow` | `src/conceptdrill/hierarchy/basis.py:110` | **encoder space**, 768-d, unit-norm float64 |
| projected onto | `ConceptHit` | `src/conceptdrill/hierarchy/project.py:46` | none — one coordinate |
| a sentence's CES vector | `SentenceProjection` | `src/conceptdrill/hierarchy/project.py:60` | **CES space**, one float per basis row |
| on disk, per run | `CONCEPT_FIELDS` | `src/conceptdrill/hierarchy/runlog.py:68` | none — the audit record |

`BasisRow` is the concept as the corpus holds it:

```python
row_id: str            # "row_" + sha256(level, canonical label)[:12] — content-addressed
level: int             # marker depth; levels never merge across each other
label: str             # the 22-28 word noun phrase; THE identity
vector: np.ndarray     # unit-norm float64 in ENCODER space (768-d for modernbert)
support: int           # how many candidates merged in
documents: tuple[str]  # bibkeys that contributed
merged_labels: tuple[str]   # every label folded into this row — the evidence
```

**There is no `score` field, and three different numbers get mistaken for one.**

- `BasisRow.support` is a **count**, not a score. High support means many spans
  said something similar, not that the concept is good.
- `merge_cosine` (`runlog.CONCEPT_FIELDS`, from `IntegrationResult.similarity`,
  `basis.py:164`) is the cosine at which this candidate met its nearest row.
  It is a fact about the *integration decision*, valid only at that moment.
- `ConceptHit.similarity` (`project.py:46`) is the **CES coordinate**: cosine
  between a sentence and one basis row. This is the only one that behaves like
  a relevance score.

Two spaces, and they are not the same space. `BasisRow.vector` lives in the
encoder's 768 dimensions; `SentenceProjection.vector` lives in *n_rows*
dimensions, one coordinate per concept. `project_vectors` (`project.py:107`)
is the map between them, and it is the whole of CES:

```
CES(s) = M @ f(s)      M = basis.matrix(), rows in canonical order
```

On disk (`src/conceptdrill/hierarchy/corpus.py:52`): `ces-basis.json` +
`ces-basis.npz` for the basis, `ces-index.json` + `ces-vectors.npz` for
sentences. Every one of them carries `basis_version` and refuses to load
against a different one.

### 1.2 The span → summary → basis pipeline

```
model.docmodel.json
  └─ build_tree()            docmodel_tree.py:800   objects → MarkerTree
       ├─ read_content()     docmodel_tree.py:428   prose → Paragraph units
       ├─ read_math()        docmodel_tree.py:339   math objects → Paragraph units,
       │                                            text via mathtext.math_text()
       └─ assign_by_extent() docmodel_tree.py:558   position decides ownership
  └─ MarkerNode              docmodel_tree.py:632
       .body_text            docmodel_tree.py:651   own extent, prose + spoken math JOINED
       .summarizer_title     docmodel_tree.py:656
  └─ summarize_tree()        summarize.py:521       ← the LLM boundary
       └─ Summarizer.summarize(span_id, title, body) → SpanSummary
            .concepts        summarize.py:135       dominant + siblings
            .label           the basis tier (BASIS_TIER = "label")
  └─ clean_basis_text()      basistext.py:271       the one producer of basis_text
  └─ embedder.encode()
  └─ basis.integrate()       basis.py:315           add or merge against TAU=0.75
```

Three tiers per concept — `summary`, `abstraction`, `label` — with independent
derivations enforced by `check_tier_independence` (`summarize.py:176`). Only
`label` becomes a basis vector.

### 1.3 Query and projection

- `project_sentences()` — `project.py:131`. Embeds, one matmul per batch,
  returns `SentenceProjection` with `top_concepts`.
- `QueryEngine` — `inference.py:116`. Embeds a query, computes its CES vector,
  ranks basis rows as `categories`, and finds stored sentences by **cosine in
  CES space** (`_cosine_rows`, `inference.py:103`). `Neighbour.shared_concepts`
  is why two things matched, which raw cosine cannot say.
- `refine()` — `refine.py:202`. The paper's Algorithm 2. Unrelated to merging.

### 1.4 Where the layer belongs

**A new package `src/conceptdrill/mathlayer/`, consuming run artefacts, writing
its own file.** Not inside `hierarchy/`.

The reason is the one already written into `corpus.py`: an artefact that
belongs to a different lifetime must not be stored in the other's directory. A
`MathConcept` is not versioned by `basis_version` — it survives a re-projection
that invalidates every CES vector, because `\dim V = n` does not stop being
true when TAU changes. Putting it under `hierarchy/` would tie it to a
versioning scheme it does not obey.

---

## 2. Inputs

Three, all of which already exist and none of which needs a new producer.

| input | source | present today |
|---|---|---|
| span text | `MarkerNode.body_text` | yes |
| spoken math | already interleaved into `body_text` as `Paragraph` units from `read_math` | yes — 34,395 objects rendered from `props.spoken` in the last full run, 2 by fallback, 28 with no text |
| CES concepts | `SpanSummary.concepts` for that span, plus the `BasisRow` each was integrated into | yes |

**A complication that must not be papered over.** `Paragraph`
(`docmodel_tree.py:265`) is `id, text, marker_id, flow_index` — it carries **no
flag saying whether it came from prose or from mathematics**. By the time a
span reaches `body_text` the spoken math has been joined into one string and is
indistinguishable from the prose around it.

It is recoverable, not lost: the math-derived paragraphs keep the docmodel
object's own id, and `MATH_TYPES` (`docmodel_tree.py:239`) says which object
types those were. A math layer can re-derive the set. But it costs a second
pass over the objects, and the alternative — adding a field to `Paragraph` —
is a change to the existing pipeline and therefore out of scope for a
non-invasive layer. **This is the one real friction point in the design.**

Formulas cited from inside a sentence are a different case again: `inline_resolver`
(`docmodel_tree.py:387`) splices them into the prose and records their ids in
`consumed`, so they are inside sentences rather than beside them.

---

## 3. Output

A list of `MathConcept` records per document, written as
`math-concepts.json` beside the run it was derived from. Two things per record:
**mathematical concepts** named canonically, and **candidate formal anchors**
— names in a proof-assistant library that plausibly denote the same object.

Candidate is the operative word. An anchor proposed here has not been checked
against any Mathlib version, and the schema is built so that it cannot pretend
otherwise.

### 3.1 The `MathConcept` object

```json
{
  "id": "mc_a3f19c22b70e",
  "concept": "linear map",
  "spoken": "a linear map from upper V to upper W",
  "latex": ["T : V \\to W", "T \\colon V \\rightarrow W"],
  "symbols": [
    {"latex": "T", "role": "instance", "status": "observed"},
    {"latex": "\\ker T", "role": "derived", "status": "predicted"}
  ],
  "related": [
    {"relation": "domain", "target": "vector space", "status": "observed"},
    {"relation": "over-field", "target": "field", "status": "implied"}
  ],
  "lean": [
    {"name": "LinearMap", "status": "unverified"}
  ],
  "loogle": ["|- _ →ₗ[_] _"],
  "evidence": {
    "doc_id": "1702.06385",
    "span_ids": ["SE0003"],
    "ces_row_ids": ["row_7c1a09ffb214"],
    "quote": "Let T : V \\to W be linear."
  },
  "confidence": 0.62
}
```

Ten keys. Nine are the ones this layer was asked to connect; `id` is the tenth
and is not optional — without a stable identity nothing can link two records,
and a positional index would transpose the moment a document is re-read. It is
content-addressed the same way `make_row_id` is (`basis.py:96`):
`"mc_" + sha256(canonical concept)[:12]`.

**Field rules, all of them load-bearing:**

- `concept` — the canonical English name, lowercase. This is the identity.
- `spoken` — how the concept is *read*, not how a formula is read. It is the
  bridge to the spoken math already in `body_text`.
- `latex` — LaTeX **exactly as it appeared**, never normalised. `T \colon V
  \rightarrow W` and `T : V \to W` are two entries, not one; which spelling an
  author used is evidence about the document.
- `symbols` — `status` is `observed` (this string is in the document) or
  `predicted` (this concept usually carries this symbol, and it is not here).
  The distinction is the difference between a reading and a guess, and merging
  the two is how an ontology fills with facts nobody wrote.
- `related` — `relation` from the closed vocabulary in §4; `status` is
  `observed`, `implied` (follows from a phrase in the text), or `external`
  (true of the concept, absent from this document).
- `lean` — objects, not strings, because `status: "unverified"` is the default
  and a bare list of names would read as confirmed. Verification means: this
  name resolves in a stated Mathlib revision. Until something checks, it does
  not.
- `loogle` — the query that would find the anchor. Kept even when `lean` is
  empty: a query with no result is a finding, not a blank.
- `evidence` — where this came from. `ces_row_ids` is the link back to CES and
  the reason this is a layer *above* it rather than a parallel pipeline.
- `confidence` — one float, computed from a declared function of the evidence
  (how many observed symbols, whether the relation was observed or implied,
  whether an anchor verified). **Never a model's self-report.** A number a
  language model chooses for its own output measures fluency.

---

## 4. Relation vocabulary

Closed set, checked on write, exactly as `MERGE_DECISIONS` and `DERIVATIONS`
are checked in `runlog.py`:

```
is-a  has-dimension  over-field  maps-to  domain  codomain  related-to
```

`related-to` is the escape hatch and should be rare. A layer that emits mostly
`related-to` has produced a similarity graph, which CES already is, and has
added nothing.

---

## 5. The hook

**The smallest place where span text, spoken math and CES concepts are all in
hand is the loop at `tools/hierarchy_run.py:205-214`**, immediately after
`summarize_tree` returns and before cleaning and embedding:

```python
run = summarize_tree(tree, summarizer, cache=cache, prompt=prompt_text)

for node in nodes:                       # ← here
    summary = run.summaries.get(node.id)
    ...
```

At that point `node` gives `body_text` (prose plus spoken math) and
`node.paragraphs` (the units, whose ids identify the math ones), and `summary`
gives every CES concept the span produced. Nothing downstream has happened yet.

`summarize.py` also already has a callback seam — `progress(node, summary)`,
`summarize.py:590` — receiving exactly the same pair. It cannot be used as-is:
its return value is discarded, so it can observe but not contribute. Widening
it would be a change to the pipeline.

**The non-invasive form is therefore neither of those: a separate pass over
the run's own artefacts.** `spans.jsonl` records `extent_object_ids`,
`input_text_sha256` and every concept per span; the docmodel is on disk and
read-only. A math layer can reconstruct all three inputs after the fact, with
one cost — it re-reads the docmodel to recover which units were mathematics —
and one benefit: it runs against completed runs, including the 278 documents
already built, without re-running anything and without a line changing in
`hierarchy/`.

Take the in-process hook only if measurement shows the second pass is the
bottleneck. It is not known to be.

---

## 6. What this must not do

- **Not write into the drill folder or the corpus store.** Its own file, its
  own directory.
- **Not participate in `basis_version`.** A `MathConcept` is not a basis row
  and must never be integrated into `M`. If it were, TAU would start deciding
  whether two theorems are the same theorem.
- **Not import Lean, Mathlib, or a network client at analysis time.** Anchors
  are proposed offline as strings; verification is a separate, explicitly
  invoked step, the way `speech.py` loads la2speech by path and probes it
  rather than assuming.
- **Not silently fall back.** If the concept extractor is unavailable, raise.
  The rule that voided the 416-document run applies here unchanged.
- **Not invent relations.** Every edge carries `observed | implied | external`,
  and a layer that cannot say which produced an edge produces none.

---

## 7. What is not decided

1. **Who extracts the mathematical concepts.** A second LLM prompt over the
   span is the obvious candidate and is not specified here. It is a separate
   decision with its own gate.
2. **How `confidence` is computed.** Named as a declared function of evidence;
   the function itself is unwritten.
3. **Whether CES concepts help at all.** The measurement in
   `tools/bullshit_basis.py` found that semantic validity is *not* recoverable
   from position in concept space — real and nonsense concepts interleave, with
   cross p50 0.492 against real-pairwise p50 0.550. Mathematical validity may
   be equally invisible to CES. If so, this layer's value is the anchors and
   the relations, and the `ces_row_ids` link is provenance rather than signal.
   That is worth knowing before building it.
4. **Whether `vector space` is a CES concept or CES vocabulary.** Measured on
   the 278-document corpus: `vector space` occurs in 97 concept labels, but in
   every sampled case as a *component* of an applied concept — "lowercase
   conversion of mathematical formulas to unify semantically equivalent
   expressions in vector space", "R-matrix on a tensor product of vector
   spaces". None is the definitional concept. A math layer looking for
   `vector space` in CES will find the phrase everywhere and the object
   nowhere, and that gap is precisely what it exists to fill.

The worked example against real text is `tests/fixtures/vector-space.md`.

---

## 8. Review, and the smallest production implementation

Written after steps 9-18 ran. Three findings from the experiment change the
design above, and they are the reason this section exists rather than a
straight build of §3.

**Finding 1 — CES contributes nothing to this, measured.** Step 15: one basis
row is top-1 for four of six mathematical continuations, the whole spread
across thirteen probes is 0.651-0.762, and *vector space* retrieves the NLP
sense at 0.717. **The layer therefore must not import the basis, the corpus
store, or the projection code at all.** §7.3 asked whether CES helps; the
answer is no, and the consequence is that the production layer is *smaller*
than designed — it needs span text, spoken math and the docmodel, and nothing
from `basis.py`, `project.py` or `corpus.py`. `ces_row_ids` survives as an
opaque provenance string that nothing reads.

**Finding 2 — verification is not optional and is not free.** Two of five
names claimed `exact` did not exist, both stale in the same direction. The
`unverified` default earns its place, and `loogle.py` has to be a real module
rather than a note.

**Finding 3 — the useful ranking signal is discrete.** What separated good
candidates from bad was `observed | implied | predicted` and whether a Loogle
query resolved. Neither is a cosine. So `confidence()` is arithmetic over
declared statuses, deterministic and unit-testable, with no embedding anywhere
near it.

### Files to add

| file | what it holds | tests |
|---|---|---|
| `src/conceptdrill/mathlayer/__init__.py` | the public surface, nothing else | — |
| `src/conceptdrill/mathlayer/schema.py` | `MATHCONCEPT_FIELDS`, the closed vocabularies, `mathconcept_record()`, `concept_id()`, `confidence()` | `tests/test_mathlayer_schema.py` |
| `src/conceptdrill/mathlayer/store.py` | atomic read/write of `math-concepts.json` | `tests/test_mathlayer_store.py` |
| `src/conceptdrill/mathlayer/loogle.py` | explicit verification; offline by default, raises rather than degrading | `tests/test_mathlayer_loogle.py` |
| `src/conceptdrill/mathlayer/extract.py` | the `Extractor` protocol and its implementations | `tests/test_mathlayer_extract.py` |
| `tools/mathlayer_show.py` | the CLI of step 21 | — |

`schema.py` follows `runlog.py` exactly: a field tuple that is the contract, a
record builder that **raises on an unknown key and fills a missing one with
`None`**, and frozen vocabularies. That module already proved the pattern —
`bool(strict_mode)` silently turning `None` into `False` is the bug it caught.

`concept_id()` is content-addressed the way `basis.make_row_id` is. A
positional id transposes the moment a document is re-read.

`extract.py` mirrors `summarize.Summarizer`: a protocol, plus implementations
carrying `measurement_safe`. `ExtractiveSummarizer` being usable as a stand-in
voided a 416-document corpus; the same trap is available here and gets the same
mechanical guard rather than a convention.

### Files to change

**None.**

That is the test of whether this is really a layer. `hierarchy/` keeps every
signature it has, `basis_version` is untouched, no existing test changes, and
the pipeline runs identically whether or not `mathlayer/` is installed. If a
change to an existing file turns out to be needed, the design is wrong and the
right response is to redesign rather than to make the change.

The one place a change is *tempting* is `Paragraph` (`docmodel_tree.py:265`),
which carries no flag saying a unit came from mathematics — so the math layer
must re-read the docmodel to recover which units were formulas. That cost is
accepted. Adding the field would be a one-line improvement to the pipeline and
would break the property this section is asserting.

### Deliberately not in the minimal layer

- **Any use of the basis, projections or the corpus store.** Finding 1.
- **Network access during extraction.** `loogle.py` is invoked explicitly, as a
  separate verification pass, the way `speech.load_speaker` probes and raises.
- **The OCR interface of `docs/math-layer-ocr.md`.** It has no consumer yet, and
  a benchmark that has not run cannot justify an interface.
- **Writing anywhere near a drill folder or `ces-basis.json`.**
