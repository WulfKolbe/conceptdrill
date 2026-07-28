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

[ces]: https://arxiv.org/abs/2209.00445

## Install

```bash
pip install -e .              # core: numpy only
pip install -e '.[models]'    # + torch/transformers for real embeddings
pip install -e '.[nlp]'       # + stanza for better noun phrases and NER
```

Without `[models]` only the `hash` embedding backend is available. It is
deterministic and offline but lexical rather than semantic — good for tests and
reproducibility work, not for quality.

## Use

```bash
# Top concepts for a text span
conceptdrill --input paper.json --text "Deep learning for graphs" --top 5

# Inspect the vocabulary the document produced
conceptdrill concepts paper.json --metrics

# Project every object and write a sidecar
conceptdrill project paper.json --model sentencebert --top 10

# One independent projection set per model
conceptdrill project paper.json --model sentencebert --model mathbert

# Confirm a stored projection is unmodified
conceptdrill verify paper.conceptdrill.json
```

```python
from conceptdrill import ConceptDrill

drill = ConceptDrill.from_path("paper.json")
drill.explain_text("efficient search over binary codes", top_k=5)
# [('bit codes', 0.4683), ('hashing', 0.4595), ('bit subcode', 0.4), ...]

drill.project_text("...")          # one similarity per concept
drill.get_concept_space_info()     # sizes, levels, sources, parameters
```

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

## Reproducibility

- `projection_id = sha256(object_id | model | revision | source | metric | k)` —
  no wall clock, no iteration order.
- Every projection records the **resolved** model revision, not the requested one.
- A `content_hash` covers the output except the timestamp, so two runs are
  byte-comparable. `conceptdrill verify` recomputes it.
- Candidates are sorted by `(-score, name)`; ties break on name, never dict order.
- `CONCEPTDRILL_NLP_BACKEND=regex|stanza|spacy|auto` pins the NLP tier — tiers
  mine different noun phrases, so this is a correctness control, not just a
  performance one.
- Concept embeddings are cached content-addressed under `.conceptdrill_cache/`,
  keyed so a cache entry can never be served to a different model or revision.

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

190 tests, offline, ~10s. The suite pins the `hash` embedder and the `regex` NLP
tier so results do not depend on which optional models happen to be installed.
