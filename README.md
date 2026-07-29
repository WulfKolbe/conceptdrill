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

191 tests, offline, ~9s. The suite pins the `hash` embedder and the `regex` NLP
tier so results do not depend on which optional models happen to be installed.
