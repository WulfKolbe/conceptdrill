# ConceptDrill

Builds a concept space out of **a document's own structure** and projects text
into it.

This is the CES pipeline from [arXiv 2209.00445][ces] with the external ontology
replaced by the document: headings, definitions, bibliography titles, frequent
noun phrases, named entities, and equation abstractions become the concept
vocabulary. No Wikipedia, no fixed taxonomy — the space is tailored to whatever
you feed it.

ConceptDrill is a **projection** tool. It never reconstructs, rewrites, or
mutates the source document. Output is always an additional view stored beside
the input.

There are two pipelines:

| | scope | concepts mined from | output |
|---|---|---|---|
| **single-document** | one document | its own headings, definitions, bibliography, noun phrases, entities, equations | `<input>.conceptdrill.json` |
| **hierarchical CES** | a corpus | LLM summaries of each section, merged into a shared basis | `model.ces.json` in the drill folder |

Both pipelines are complete end to end: section tree, summarisation, adaptive
basis, sentence projection, 2-D layout, corpus storage, query and query log.

[ces]: https://arxiv.org/abs/2209.00445

## Install

```bash
pip install -e .                # core: numpy only
pip install -e '.[models]'      # + torch/transformers for real embeddings
pip install -e '.[nlp]'         # + stanza for better noun phrases and NER
pip install -e '.[llm]'         # + openai client for the section summariser
pip install -e '.[latex]'       # + pylatexenc for better caption cleaning
pip install -e '.[hierarchy]'   # everything the hierarchical pipeline can use
pip install -e '.[spacy]'       # + spaCy as an alternative NLP tier
pip install -e '.[viz]'         # + umap-learn/scikit-learn for 2-D layout
pip install -e '.[dev]'         # + pytest
```

Every extra degrades rather than fails. Without `[llm]` the summariser falls
back to a deterministic extractive floor; without `[latex]` caption cleaning
falls back to a regex; without `[models]` only the `hash` embedder exists.

The `hash` backend is deterministic and offline but lexical rather than
semantic — good for tests and reproducibility work, not for quality.

## Use

`examples/paper.json` ships with the repo, so every command below runs as
written from a clone.

```bash
# Top concepts for a text span
conceptdrill --input examples/paper.json --text "Deep learning for graphs" --top 5

# Inspect the vocabulary the document produced
conceptdrill concepts examples/paper.json --metrics

# Project every object and write a sidecar
conceptdrill project examples/paper.json --model sentencebert --top 10

# One independent projection set per model
conceptdrill project examples/paper.json --model sentencebert --model mathbert

# Confirm a stored projection is unmodified
conceptdrill verify examples/paper.conceptdrill.json
```

The first command prints:

```
'Deep learning for graphs'  [sentencebert, 27 concepts]
   1. +0.5004  Deep Residual Learning for Image Recognition
   2. +0.4481  Convolutional Neural Network
   3. +0.3990  CNN
   4. +0.3790  Support Vector Networks
   5. +0.3290  embedding
```

Note what that shows: the query is *off-topic* for this document, and the scores
say so. Nothing above 0.5, and the matches are the document's neural-network
citations rather than its subject matter. An on-topic query separates cleanly:

```python
from conceptdrill import ConceptDrill

drill = ConceptDrill.from_path("examples/paper.json")
drill.explain_text("mapping a paragraph into an interpretable concept vector", top_k=5)
# [('Conceptualizing Embedding Spaces for Large Language Model Interpretability', 0.582205),
#  ('Concept Projection', 0.497809),
#  ('Semantic Projection', 0.483661),
#  ('Distributed Representations of Words and Phrases and their Compositionality', 0.466503),
#  ('Method > Semantic Projection', 0.456117)]

drill.project_text("greedy selection under a diversity constraint")  # -> (27,) float32
drill.get_concept_space_info()   # size 27, dim 384, 6 candidate sources
```

Those figures come from an actual run with the default `sentencebert` backend
and `CONCEPTDRILL_NLP_BACKEND=stanza`. The last decimal can move — see
[Reproducibility](#reproducibility).

## Input

Two formats, sniffed automatically.

**Semantic Compiler DocModel** (`model.docmodel.json` from
[drillspace](https://github.com/WulfKolbe)) — read-only. Text lives under a
different prop per object type, and `docmodel.py` holds that mapping:

| Type | Text source |
|---|---|
| `Paragraph`, `Abstract`, `Footnote`, `Sidenote` | `props.text` |
| `Equation`, `Formula` | `props.latex` → `latex_raw` → `latex_original` |
| `ListItem` | `props.content` |
| `Table` | `props.caption` + `props.latex_code` |
| `Picture`, `Diagram` | `props.caption` |
| `Section` | `props.title` |

`Citation` carries only a citekey; `Page`, `Document`, `Toc` and `TableRow`
carry no content. These are **recorded as skips with a reason**, not silently
dropped, so a run accounts for every object in the input.

**Generic JSON** — for use without the Semantic Compiler:

```json
{"sections":     [{"id": "s1", "title": "Method", "level": 1}],
 "blocks":       [{"id": "b1", "type": "paragraph", "text": "...", "section": "s1"}],
 "bibliography": [{"title": "Attention Is All You Need", "year": 2017,
                   "citations": 100000}]}
```

## Pipeline

```
document → candidates → quality scoring → selection → embedding → projection
```

**Candidates** come from six generators (`src/conceptdrill/candidates/`), each
independently replaceable. They are merged by normalised name, keeping the
highest-structural-weight source and recording every contributor — so
multi-source agreement stays visible.

**Quality** is a weighted sum of seven metrics over a shared context:

```
Q(c) = 0.25·structural + 0.20·coverage + 0.10·purity + 0.15·information_gain
     + 0.10·embedding_variance + 0.10·citation_importance + 0.10·reusability
```

Each metric is a registered pure function; override weights with `--weight
coverage=0.3`, or swap an implementation by passing `metrics=` to
`QualityScorer`.

**Selection** is greedy best-first under a diversity constraint — a candidate
within 0.95 cosine of something already chosen is rejected, so paraphrases do
not consume the vocabulary budget.

**Projection** is the matrix form: with `M`'s rows being unit-norm concept
vectors and `l` a unit-norm span vector, `project(s) = M @ l` gives every cosine
in one call.

## Embedding models

| Name | Checkpoint | Routed to |
|---|---|---|
| `sentencebert` | `sentence-transformers/all-MiniLM-L6-v2` | prose, headings, bibliography |
| `mathbert` | `tbs17/MathBERT` | equations, formulae |
| `codebert` | `microsoft/codebert-base` | code, algorithms |
| `hash` | — | offline deterministic fallback |

Any `org/checkpoint` path also works. `conceptdrill routing` prints the
object-type → model table.

`code` and `algorithm` route to CodeBERT but are marked **dormant**: the
DocModel emits no such objects yet. They light up automatically if it starts to.

Device defaults to **CPU**. `torch.cuda.is_available()` returning True does not
mean the device survives a forward pass — on ROCm it can report a device and
then segfault. Opt in with `CONCEPTDRILL_DEVICE=cuda`.

## Hierarchical CES (multi-document)

A second pipeline builds a **shared basis across a corpus** from the section
hierarchy of drilled documents, rather than mining one document's own text.

```
drilled docs → section tree → LLM summaries → basis vectors
                                                    │
                              adaptive integration ─┴→ matrix M
                                                          │
              sentences → BERT → l → M·l → CES vectors ───┤
                                                          ├→ corpus store
                                        query → CES → ────┘   + query log
```

```python
from conceptdrill.hierarchy.docmodel_tree import load_tree
from conceptdrill.hierarchy.summarize import ExtractiveSummarizer, summarize_tree
from conceptdrill.hierarchy import store

tree = load_tree("~/pdfdrill-library/2209.00445/model.docmodel.json")
run  = summarize_tree(tree, ExtractiveSummarizer())
store.save(tree, run.summaries, summary_stats=run.stats())
```

### Section tree

`model.docmodel.json` is the only reliable input. Verified across all 334
drilled documents in the library — 334 parsed, 0 failures:

| source | hierarchy | verdict |
|---|---|---|
| `.md` | 2 headers total | destroyed |
| `.llm.txt` | 0 markers | destroyed |
| `texsrc/*.tex` | present, but `.sty` noise and competing `.tex` files | awkward |
| **`model.docmodel.json`** | levels, `is_appendix`, `flow_index`, paragraph links | **use this** |

Three properties of the DocModel drive `docmodel_tree.py`:

- Section titles are under **`props.caption`**, not `props.title`.
- **`parent` is `null` on every Section** — the tree is rebuilt from `level` +
  `flow_index`. 89 of 251 usable documents start at level 1 and 103 at level 2,
  so nothing may assume a root level.
- Captions carry unresolved LaTeX macros (`\ALG\ Application`), cleaned by
  `captions.py`, which records what it had to drop.

**Formula and Equation objects carry no `parent_section`** — zero of the
reference paper's 74 — so their owning section is inferred from `flow_index`.

### Summaries

Each section yields three tiers (`prompts/section-concept.md`). Measured at
1.604 BERT tokens/word on real prose:

| tier | words | tokens | role |
|---|---|---|---|
| `summary` | 80-150 | 128-241 | document-faithful |
| `abstraction` | ~70 | 96-128 | document-independent |
| **`label`** | **30-42** | **48-67** | **the basis tier** |

Only `label` fits a 50-70 token window. Two summarisers: `ExtractiveSummarizer`
(deterministic, offline, a genuine floor — it cannot abstract) and
`NovitaSummarizer` (`hierarchy/novita.py`, an OpenAI-compatible chat model;
the network call is injected, so it tests offline). All LLM output is passed
through `sanitize.py`, which folds invisible and lookalike characters to ASCII
while preserving Greek, accents and CJK.

### Spoken math

Formulas contribute nothing as raw LaTeX, so `mathtext.py` renders them to
prose, preferring a spoken field on the object once the DocModel carries one,
then a speech backend ([la2speech](https://github.com/WulfKolbe)), then a
deterministic operator-naming fallback. The source of every rendering is
tallied in `tree.stats()`.

Speech text is not automatically embedding text: SRE spells multi-letter
identifiers letter by letter, so `protect_identifiers` wraps them first.
`AVERAGE_{s \in siblings(c,p)}` reads as *"A V E R A G of E sub s ..."* raw and
*"AVERAGE sub s is a member of the siblings of"* protected.

### Adaptive basis

Per level, a candidate merges into its nearest row when cosine >= `TAU`, else
becomes a new row. Row order is `(level, -support, label)`; `row_id` is
content-addressed so identity survives reordering, and `basis_version` hashes
the ordered ids so a stored CES vector detects that positions moved.

**`TAU` defaults to 0.65, measured rather than guessed.** The design spec
proposed 0.85; across three topically related papers that produced *zero*
merges, because the highest cross-document similarity observed was **0.647** —
and that pair was genuinely related. 0.85 is a near-paraphrase threshold and
the wrong scale. Use `basis.calibrate()` on your own corpus; it reports
within- and cross-document distributions separately, because they answer
different questions.

The basis arithmetic is **float64** throughout: a merge decision is one
comparison against `TAU`, and a wrong cosine adds a row that should have
merged, unrecoverably.

### Modules

| module | responsibility |
|---|---|
| `docmodel_tree.py` | Sections, hierarchy, paragraphs, math attachment |
| `captions.py` | LaTeX caption cleaning, DocModel placeholder removal |
| `mathtext.py` | Formula to embeddable prose |
| `summarize.py` | Tiers, `ExtractiveSummarizer`, cache, `summarize_tree` |
| `novita.py` | Chat-backed summariser, throttle, credentials |
| `replyparse.py` | Tolerant JSON recovery from model replies |
| `sanitize.py` | Invisible/lookalike character folding |
| `basis.py` | Adaptive integration, row order, `calibrate` |
| `store.py` | `model.ces.json` in the drill folder |
| `sidecar.py` | `CES_BUILT` capability + content-hash proof |
| `sentences.py` | Sentence splitting, the projection unit |
| `project.py` | `CES(s) = M @ f(s)`, with basis-version provenance |
| `layout2d.py` | 2-D layout for inspection (PCA / UMAP / t-SNE) |
| `corpus.py` | Corpus-level basis + sentence index on disk |
| `inference.py` | Query to concepts and neighbours; the query log |
| `refine.py` | The paper's Algorithm 2: hierarchical on-demand refinement |

### Sentences, projection and layout

Sentences are the projection unit. The splitter is **rule-based, not stanza**:
stanza costs ~7s of model load per process and is neural, so its output moves
with the model version. Abbreviations are split into those that never end a
sentence (`e.g.`, `Fig.`) and those that can (`et al.`, `etc.`) — in "Vaswani
et al. The result holds" the period does both jobs and punctuation cannot
resolve it.

Every CES vector records its `basis_version` and embedding model, because a
bare vector is not safely interpretable: coordinate 4 means whatever row 4 was,
and rows reorder as support changes. `margin` (top-1 minus top-2) is stored
beside the top similarity, since a high top-1 with a near-zero margin is
ambiguity rather than confidence.

Measured on 1489 sentences from three papers against a 38-row basis:

| | p10 | median | p90 |
|---|---|---|---|
| top-1 similarity | 0.239 | 0.438 | 0.647 |
| margin | 0.0056 | 0.0369 | 0.1339 |

**For 60% of sentences the margin is below 0.05** — the best concept is barely
distinguishable from the second. The machinery is sound (all 38 rows win at
least once, the most frequent takes 12%), but a 38-row basis from three
documents, 30 rows of them singletons, is too thin to discriminate. Basis
quality is the limit, not projection.

`layout2d` reduces CES vectors to two dimensions. **PCA is the default, not a
fallback**: a CES coordinate already means something — the cosine to one named
concept — so a linear projection keeps the axes interpretable and the loadings
name the concept driving each axis. UMAP's axes mean nothing. On the run above
PCA carried **68% of variance in two dimensions** (57.1% + 10.9%).

Eigenvector signs are pinned, because SVD may return `v` or `-v` and two runs
would otherwise produce mirrored plots.

### Two different ways to change the concept space

The paper and this project grow a space in different directions, and both are
implemented:

| | direction | module | from |
|---|---|---|---|
| **refinement** | down a hierarchy, one document | `refine.py` | **the paper, Algorithm 2** |
| **merging** | sideways across a corpus | `basis.py` | this project's own design |

`refine` implements the paper directly: start at the top level, project the
contextual texts into the current space, expand the highest-weighted concept
into its best children, repeat until the target size. `removeP` chooses whether
the expanded parent is replaced by its children or kept alongside them.

On `2209.00445`, growing from its 14 top-level sections to 24:

```
expand The Conceptualization Algorithm  w=22.97  + Generating Conceptual Spaces, ...
expand Empirical Evaluation             w=13.44  + Qualitative Evaluation, ...
expand Evaluating Understandability     w=12.62  + Evaluation By Humans, ...
expand Application                      w= 5.88  + Using CES for Comparing Models, ...
```

The expansion order is the document's own emphasis, recovered from the text
rather than the outline.

**The siblings score is degenerate on a section tree.** The paper defines

```
sibscore(p, c) = mean over s in siblings(c,p) of |parents(c) ∩ parents(s)| / |parents(c)|
```

which measures how tightly a child is bound to its sibling group in a
**multi-parent** graph — Wikipedia categories, where the paper's edges are
unlabelled. Measured across the drilled library: **0 of 8695 sections have more
than one parent.** In a tree every sibling shares the single parent, so every
term is `|{p}|/|{p}| = 1` and the score is always exactly 1 — it cannot rank
anything.

It is implemented faithfully for the DAG case and verified against one, but on
a tree `children_ranked` falls back to **document order**, and
`RefinementResult.sibscore_informative` says so. Ordering children by where the
author put them is a real signal; presenting a constant as a ranking is not.

Two other places the paper leaves things implicit, handled explicitly here:
termination (a leaf cannot be expanded, so the loop stops when every member is
a leaf and reports why), and `C^1` (taken as the shallowest depth *present*,
because a third of the corpus starts at level 1 and the rest at level 2).

### Corpus store and querying

The corpus basis is the one artefact that deliberately does **not** live in a
drill folder — a shared basis belongs to the corpus, and writing it beside one
document would make that document silently authoritative for the rest.

```
<corpus_dir>/
    ces-basis.json     rows, document order, tau, basis_version
    ces-basis.npz      the matrix M, float64
    ces-index.json     one record per stored sentence
    ces-vectors.npz    sentence CES vectors, float64
    queries.jsonl      append-only query log
```

`basis_version` is written into every file and checked on load, so vectors left
behind by an earlier basis are **refused rather than silently misread**. That is
the entire reason the version exists.

```python
from conceptdrill.hierarchy.corpus import CorpusStore
from conceptdrill.hierarchy.inference import QueryEngine, QueryLog

store = CorpusStore("~/ces-corpus")
basis = store.load_basis()
records, vectors = store.load_index(basis_version=basis.basis_version())

engine = QueryEngine(basis, embedder, records=records, vectors=vectors)
result = engine.query("how are latent embeddings made interpretable")

result.annotated      # the query with its concepts injected
result.categories     # what it is about
result.neighbours     # where the corpus discusses it, with shared_concepts
QueryLog("~/ces-corpus/queries.jsonl").append(result, answer=result.annotated)
```

Search happens **in CES space, not embedding space**, which is the point: the
match is explainable, because the coordinates are named concepts and
`shared_concepts` says which ones the query and the sentence agree on. It uses
cosine rather than dot product, so a sentence matching every concept weakly
cannot outrank one matching a single concept strongly on magnitude alone.

On the three-paper corpus (38 rows, 1489 sentences) retrieval routes correctly:
*"how are latent embeddings made interpretable"* returns sentences from the CES
paper, *"estimating the completeness of a knowledge base"* from 2305.05403.

### Where output goes

Artefacts land in the **drilled document's own folder**, joining pdfdrill's
`model.<stage>.json` family, and register as a normal pdfdrill capability:

```
~/pdfdrill-library/2209.00445/
    model.docmodel.json      input, read-only, never modified
    model.ces.json           written here
    2209.00445.drill.json    gains fact CES_BUILT + a content-hash proof
```

`sidecar.capability_valid()` re-hashes the recorded inputs, so re-drilling a
document invalidates its CES output automatically. Writes are additive and
preserve every sidecar key they do not understand.

The **corpus basis is not per-document** and deliberately does not live in a
drill folder: writing it into one would make that folder silently authoritative
for every other document.

## Reproducibility

The pipeline is deterministic. The **transformer backends are not bit-exact**,
and that distinction is worth stating precisely rather than glossing.

**Exactly reproducible — `hash` backend.** Two runs produce an identical
`content_hash`, byte for byte. This is enforced by
`test_content_hash_is_reproducible`.

**Reproducible except for float noise — transformer backends.** Measured over
three cold runs of `examples/paper.json` with `sentencebert`:

| Property | Result |
|---|---|
| Concept vocabulary | identical |
| Concept quality scores | identical (drift 0.0) |
| Top-1 concept per object | identical |
| Rank slots that differ | 4 / 450 (0.9%), all near-ties deep in the list |
| Per-object similarities | drift up to ~5e-4 |
| `content_hash` | **not stable** |

**The cause is this machine's BLAS, not the pipeline — and not what I first
claimed.** I originally attributed the drift to torch thread scheduling and to
SDPA attention-kernel selection. Both attributions were wrong. Measured:

    float32 matmul is non-deterministic and numerically wrong on this host.
    Repeating `a @ b` on identical tensors, single-threaded, in one process
    returns different results in 0-100% of trials depending on the process,
    deviating by up to 4.25 (torch) and 6.92 (numpy) on values of magnitude 27.
    float64 matmul, sum(), and elementwise ops are clean; memory readback is
    clean. numpy and torch both link OpenBLAS 0.3.26 built for target CORE2
    with an empty DYNAMIC_ARCH.

Reproduce it:

```bash
python3 -c "import torch; a=torch.randn(512,768); w=torch.randn(768,768)
r=(a@w).clone(); print(sum(1 for _ in range(100) if not torch.equal(a@w, r)), '/100')"
```

The numbers in the table above were therefore measured on a host with a faulty
SGEMM kernel. They are **not** a property of ConceptDrill, and on a machine with
a correct BLAS the transformer backends may well be exactly reproducible. Two
settings are still applied by default — `torch.set_num_threads(1)` and
`attn_implementation="eager"` — because they reduce variance cheaply, **not**
because they fix this. Opt out with `CONCEPTDRILL_TORCH_THREADS=0`.

Everything outside the model is fully deterministic:

- `projection_id = sha256(object_id | model | revision | source | metric | k)` —
  no wall clock, no iteration order.
- Every projection records the **resolved** model revision, not the requested
  one (`sentencebert` resolves to commit `1110a243…`).
- Candidates sort by `(-score, name)`; ties break on name, never dict order.
- `CONCEPTDRILL_NLP_BACKEND=regex|stanza|spacy|auto` pins the NLP tier — tiers
  mine different noun phrases, so this is a correctness control, not just a
  performance one.
- Concept embeddings are cached content-addressed under `.conceptdrill_cache/`,
  keyed so an entry can never be served to a different model or revision.

`conceptdrill verify` recomputes a stored sidecar's hash. That detects
hand-editing of a file regardless of backend, since it rehashes what is on disk.

## Output layout

A sidecar stores **one concept space per embedding model**:

```json
{"concept_spaces": {"sentencebert": {"size": 27, "concepts": [...]},
                    "mathbert":     {"size": 28, "concepts": [...]}},
 "projections":    [{"object_id": "b1", "embedding_model": "sentencebert", ...}],
 "skipped":        [{"object_id": "...", "reason": "..."}],
 "content_hash":   "…"}
```

Each model scores and selects with its own embeddings, so two models rarely
agree on the same vocabulary — 27 vs 28 concepts above. Resolve a projection's
concepts against `concept_spaces[projection.embedding_model]`, or use
`storage.resolve_concept(payload, projection, concept_id)`.

## Agent interface

ConceptDrill does not index or schedule. `conceptdrill.api` exposes the
operations another agent drives, each returning a status envelope
(`completed` / `updated` / `failed`) rather than raising:

```python
request_concept_generation(path)   request_embedding(texts)
request_projection(path)           request_storage(path)
request_verification(sidecar)
```

## Tests

```bash
python -m pytest
```

634 tests, offline, ~9s. The suite pins the `hash` embedder and the `regex` NLP
tier so results do not depend on which optional models happen to be installed.
Tests touching `~/pdfdrill-library` skip when it is absent.

Numerics are repaired automatically at startup by `blasfix.py` — see
[Reproducibility](#reproducibility). `./setup.sh` checks the environment and
runs `embrun.py --selftest` as its acceptance test.
