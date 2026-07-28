# ConceptDrill — Design

Date: 2026-07-28

## Purpose

ConceptDrill builds a **concept space out of a document's own structure** and
projects arbitrary text spans into it. It implements the CES pipeline
(arXiv 2209.00445) with the external ontology replaced by a vocabulary mined
from the document: headings, definitions, bibliography titles, noun phrases,
named entities, and equation abstractions.

It is a *projection* tool. It never reconstructs, rewrites, or mutates the
source document. Output is always an additional view stored beside the input.

## Position relative to the Semantic Compiler

The Semantic Compiler (`~/drillspace`) emits `model.docmodel.json`. ConceptDrill
consumes that file read-only and writes a sidecar. There is no import
dependency in either direction, which makes "never modify the original object" a
structural property rather than a convention.

ConceptDrill also accepts a simpler generic document JSON, so it is usable
without the Semantic Compiler.

## Pipeline

```
document JSON
    │
    ▼
Document          sections, blocks (typed), bibliography
    │
    ▼
Candidates        6 generators, deduplicated, structurally weighted
    │
    ▼
Quality scoring   7 metrics -> Q(c), weighted sum
    │
    ▼
ConceptSpace      top-N selection + diversity filter + section hierarchy
    │
    ▼
Embeddings        v_c = f(tau(c)), cached, L2-normalised
    │
    ▼
Projection        project(s) = M @ f(s)      (cosine via normalised dot)
```

## Modules

| Module | Responsibility |
|---|---|
| `types.py` | Frozen dataclasses: `Block`, `Section`, `BibEntry`, `Candidate`, `Concept`, `ConceptHit`, `Projection` |
| `document.py` | `Document` — generic JSON parsing, section tree, block access |
| `docmodel.py` | Adapter: Semantic Compiler `model.docmodel.json` -> `Document` |
| `candidates/` | One generator per source (§1.1–1.6), each a `CandidateGenerator` |
| `scoring/` | `metrics.py` (7 registered pure functions) + `scorer.py` (weighted sum) |
| `embeddings/` | `Embedder` protocol, `hashing`, `transformer`, content-addressed `cache` |
| `space.py` | Selection, diversity, hierarchy, matrix `M`, `refine_space` |
| `projection.py` | `project`, `explain`, `Projection` record construction |
| `nlp.py` | Noun-phrase / NER shim: stanza -> spaCy -> regex fallback |
| `abstractor.py` | LLM hook for equation descriptions and title shortening |
| `storage.py` | Sidecar read/write |
| `api.py` | `request_projection` / `request_embedding` / `request_concept_generation` / `request_storage` |
| `core.py` | The `ConceptDrill` facade class |
| `cli.py` | `project`, `concepts`, `explain` subcommands |

## Object type handling

The Semantic Compiler emits `Abstract, Citation, Diagram, Document, Equation,
Footnote, Formula, ListItem, Page, Picture, Paragraph, Section, Sidenote, Table,
TableRow, Toc`. Text lives under a different prop per type, so extraction is a
per-type table:

| Type | Text source |
|---|---|
| `Paragraph`, `Abstract`, `Footnote`, `Sidenote` | `props.text` |
| `Equation`, `Formula` | `props.latex` -> `latex_raw` -> `latex_original` |
| `ListItem` | `props.content` |
| `Table` | `props.caption` + `props.latex_code` |
| `Picture`, `Diagram` | `props.caption` |
| `Section` | `props.title` / heading text |

`Citation` carries only a `citekey` with no text; `Page`, `Document`, `Toc`, and
`TableRow` carry no projectable content. These are skipped with a recorded
reason rather than silently dropped, so a projection run accounts for every
object in the model.

`Code` and `Algorithm` are absent from the DocModel today. They stay registered
in the type -> model routing table pointing at CodeBERT, dormant until the
DocModel emits them. No speculative extraction is performed.

## Embedding backends

One protocol, two implementations:

- **`hash`** — signed hashing trick over word and character n-grams. Fully
  deterministic, no network, no model download. Because it is lexical rather
  than random-per-text, similar texts receive similar vectors, so the coverage,
  purity, and variance metrics produce meaningful numbers offline. This is the
  backend the test suite uses.
- **`transformer`** — HuggingFace `AutoModel` with attention-masked mean pooling
  and L2 normalisation. Covers all three named models through one code path:

  | Name | Checkpoint |
  |---|---|
  | `sentencebert` | `sentence-transformers/all-MiniLM-L6-v2` |
  | `mathbert` | `tbs17/MathBERT` |
  | `codebert` | `microsoft/codebert-base` |

Mean pooling reproduces `all-MiniLM-L6-v2`'s native sentence-transformers
behaviour, so a single implementation serves sentence and non-sentence models
alike and drops the `sentence-transformers` dependency.

## Determinism

This is a hard requirement, so it is designed for rather than hoped for:

- `projection_id = sha256(object_id | model | revision | concept_source | metric | k)`.
  No wall clock, no iteration order.
- Every projection records the resolved model revision.
- A `content_hash` covers the whole output *except* the timestamp, so two runs
  are byte-comparable.
- Candidate sets are sorted by `(-score, name)` before selection; ties break on
  name, never on dict order.
- Stored similarities are rounded to a fixed precision to survive
  platform float jitter.
- Concept-vocabulary embeddings are cached content-addressed under
  `.conceptdrill_cache/` keyed by `sha256(text | model | revision)`.

## Scoring

`Q(c) = 0.25·structural + 0.20·coverage + 0.10·purity + 0.15·information_gain
+ 0.10·embedding_variance + 0.10·citation_importance + 0.10·reusability`

Weights are overridable. Each metric is a registered pure function over a
shared `ScoringContext` (paragraph embeddings, section labels, concept
embedding), so any single metric can be swapped without touching the others.

Coverage uses the specified piecewise-linear penalty: fractions at or below
0.01 map to 0.1, at or above 0.9 map to 0.3, with the maximum on the 0.2–0.4
plateau.

## Confidence

Stored as both top-1 similarity and the top-1 minus top-2 margin. Similarity
alone says how strong the best match is; the margin says whether the mapping is
ambiguous, which is what multi-model disagreement analysis reads.

## Testing

pytest against the `hash` backend so the suite is offline and fast. A synthetic
mock document fixture plus a real `model.docmodel.json` fixture copied from
drillspace. Determinism is asserted by running the pipeline twice and comparing
`content_hash`.

## Out of scope

No indexing, no storage backend beyond the sidecar, no document
reconstruction. The `api.py` request functions expose the operations another CLI
agent would drive; ConceptDrill does not call out to one.
