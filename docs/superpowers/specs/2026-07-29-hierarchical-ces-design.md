# CES Hierarchical Self-Adapting Embedding Structure — Design

Date: 2026-07-29
Status: **definition — not implemented**

## Purpose

Build a *shared, multi-document* conceptual basis from the section hierarchy of
drilled documents, and project sentences into it. This is CES
(arXiv 2209.00445, drilled at `~/pdfdrill-library/2209.00445/`) with the concept
space grown from document structure and **adapted across a corpus** rather than
mined per document.

It extends `conceptdrill`, reusing its `Embedder` protocol, `ConceptSpace`,
matrix form and sidecar storage. It adds what conceptdrill deliberately lacks:
hierarchy-derived basis texts, an LLM summarisation stage, and cross-document
basis adaptation.

## Decisions taken

| Question | Decision |
|---|---|
| Home | Extend `conceptdrill` |
| Input | `model.docmodel.json` only |
| Summariser | Novita API, reusing `~/Gemma4/section_concepts.py` + its prompt |
| Basis merge | Cosine threshold, then running average |

## Input: what the docmodel actually gives us

Verified against `2209.00445/model.docmodel.json`. **This is the reliable
source; the other formats are not:**

| source | hierarchy | verdict |
|---|---|---|
| `.md` | 2 headers total | destroyed |
| `.llm.txt` | 0 markers, paragraph-chunked | destroyed |
| `texsrc/*.tex` | 33/18/5 but `.sty` noise and two competing `.tex` files | usable, awkward |
| **`model.docmodel.json`** | 24 Sections, L2–L4, `is_appendix`, `flow_index`, 84/85 paragraphs linked | **use this** |

Three traps, each of which must be handled explicitly:

1. **Titles are under `props.caption`**, not `props.title`.
2. **`parent` is `null` on every Section.** The tree must be rebuilt from
   `level` + `flow_index`: walking sections in `flow_index` order, a section's
   parent is the nearest preceding section of lower `level`.
3. **Captions contain unresolved LaTeX macros** — `\ALG\ Application`,
   `\emph{Siblings} score`, `$removeP$`, `$\tau$`. These must be cleaned before
   they reach either the summariser or the embedder, or they poison the basis.
   `section_concepts.clean_title` (pylatexenc with a regex fallback) already
   does this.

Levels start at **2**, not 1. Paragraph→section linkage is `props.parent_section`
(present on 84/85; the gap is `flow_index=1`, pre-first-section matter).

A document is only usable if its `drill.json` `facts` include `MODEL_BUILT`;
`LATEX_INGESTED` indicates the good path. Documents lacking Sections are
skipped and recorded, never silently dropped.

## Summary tier sizing — measured, not assumed

Median **1.604 BERT tokens per word** on this paper's prose (75 paragraphs).
The prepared prompt emits three tiers; against the 50–70 token target:

| tier | words | tokens | role |
|---|---|---|---|
| `summary` | 80–150 | 128–241 | document-faithful; **too long** for the basis |
| `abstraction` | ~70 | 96–128 | document-independent; **too long** |
| **`label`** | 20–40 | **32–64** | **the basis tier** |

`label` is already specified as "a canonical concept definition suitable for
reuse across documents" — exactly the cross-document basis role. To centre it in
50–70 tokens rather than straddle the lower bound, its target should move to
**30–42 words**. That is a one-line prompt change, and the built pipeline must
measure realised token lengths and report them rather than trusting the target.

All three tiers are stored. `label` builds the shared basis; `abstraction` and
`summary` are retained for per-document views and for later comparison of which
tier projects better.

## Pipeline

```
drilled docs ──► section tree ──► LLM summaries ──► basis vectors
                                                          │
                                   ┌──────────────────────┘
                                   ▼
                        adaptive integration  ──►  matrix M
                                   │
   sentences ──► BERT ──► l ──► M·l ──► CES vectors ──► UMAP / storage
                                                   └──► inference (query)
```

### 1. Section tree (step 1, structural)
`hierarchy/docmodel_tree.py` — read Sections, clean captions, rebuild
parent/child from `level`+`flow_index`, attach paragraphs via `parent_section`.
Emits a `SectionNode` tree with concatenated body text per node.

### 2. Summarisation (step 1A/1B)
`hierarchy/summarize.py` — adapts `~/Gemma4/section_concepts.py`. Its
`concept_of()` and `prompts/section-concept.md` are reused as-is; only the input
changes from `extract_sections(tex)` to the docmodel tree.

Carries over from the prepared script: Novita OpenAI-compatible client,
`temperature=0.2`, the ~30 req/min throttle (`_MIN_INTERVAL` 2.2s), thread-pool
concurrency, `gemmatester.parse_objects` for tolerant JSON recovery, and
per-section error capture so one failure does not sink the run.

**Summaries are cached content-addressed** on `sha256(prompt|model|section
text)`. The LLM is the slowest and least reproducible stage; caching makes
re-runs cheap and keeps a corpus build comparable across runs.

### 3. Basis vectors (step 1B)
Embed each `label` through the existing `Embedder`. Reuses conceptdrill's
`CachedEmbedder` and its L2-normalisation guarantee.

### 4. Adaptive integration (step 2)
`hierarchy/basis.py` — the self-adapting core, per level:

```
for each candidate vector v at level L:
    j = argmax cosine(v, M_L)
    if cosine(v, M_L[j]) < TAU:   append v as a new basis row (support = 1)
    else:                          M_L[j] = normalise((M_L[j]*n + v) / (n+1)); n += 1
```

`TAU` default 0.85, configurable. Each row records support count, contributing
documents, and the labels merged into it, so a basis row is explainable.

Order-dependent by construction. Documents are therefore processed in a
**deterministic order** (sorted bibkey), and that order is recorded in the store
so a rebuild reproduces the same basis.

### 5. Matrix M — row order

**Decided: level-major, then support count descending, then label
alphabetically.** As a sort key:

```python
sort_key = (level, -support_count, label)   # level ascending: L2, L3, L4
```

The label tie-break is what makes it total: two rows at the same level with
equal support would otherwise be ordered by dict insertion, which varies.

#### The consequence this creates, and how it is handled

Support counts change as the corpus grows, so **row positions move**. A document
added today can push a basis row from index 4 to index 2. Any CES vector stored
before that reads its coordinates against the wrong concepts — and nothing about
the vector itself reveals the mismatch. Concretely:

```
build 1:  rows [A(support 5), B(support 3)]   CES = [sim_A, sim_B]
add doc:  B merges, support 6
build 2:  rows [B(support 6), A(support 5)]   CES = [sim_B, sim_A]
          -> the stored build-1 vector is now silently transposed
```

Three mechanisms keep the ordering as specified while making this detectable
rather than silent:

1. **`row_id` is content-addressed, not positional.**
   `row_id = sha256(level | canonical_label)[:12]`. It survives reordering,
   merges and corpus growth. Consumers that need to name a coordinate use
   `row_id`; position is only ever a rendering detail.

2. **`basis_version` = `sha256` over the ordered `row_id` list.** It changes
   whenever rows are added, merged, or reordered.

3. **Every stored CES vector records the `basis_version` it was computed
   against.** Comparing two CES vectors, or a query against a stored corpus,
   must check the versions match; on mismatch the correct action is to
   re-project, not to compare. This turns a silent wrong answer into a loud
   refusal.

Merging updates a row's *support count and vector* but never its `row_id`,
because the canonical label is what identifies the concept. A merge that would
change the label (it does not, in the running-average rule) would create a new
identity, and that is the correct semantics.

### 6. Sentence CES vectors (steps 3–4)
Sentence-split paragraphs, embed, `CES = M @ l`. This is conceptdrill's existing
matrix path unchanged.

### 7. Projection to 2-D (step 5)
Separate tool, as you suggested. UMAP/t-SNE are non-deterministic without a
fixed seed and add heavy dependencies; keeping them out of the core keeps the
core reproducible.

### 8. Storage (step 6) and inference (steps 7–8)
Sidecar per document plus a corpus-level basis store holding M, row metadata and
the document order. Inference: embed query → `M @ l` → similarity search in CES
space → return the query annotated with its top basis categories. Query vectors
and answers are appended to a query log for later analysis.

## Structural level (deferred, as you noted)

A later level covering title, abstract, results/conclusion, appendix,
bibliography. Two of its inputs are already available: `is_appendix` on Sections,
and object counts in `drill.json` (`Paragraph: 85, Formula: 73, Citation: 60,
Table: 16, Reference: 46`). This level likely wants *statistical* descriptors
rather than LLM summaries, so it needs a different generator — not just a
different prompt.

## Constraints inherited from conceptdrill

- Never write to the input document; output is a sidecar.
- All embedders return L2-normalised rows, so `M @ l` is cosine.
- No import dependency on drillspace. Reading `docmodel.json` is fine; importing
  `pdfdrill` is not.
- Skipped objects and documents are recorded with a reason.

## Reproducibility

Two non-deterministic stages, both must be pinned and recorded:

- **The LLM.** `temperature=0.2` is not deterministic. Summaries are cached and
  the cache is the reproducibility boundary; the store records model id and
  prompt hash.
- **This machine's float32 GEMM is wrong** (see `setup.sh`, `gemm_check.py`).
  Every basis vector, cosine and CES coordinate is affected. `setup.sh` must pass
  before any corpus build is trusted.

## Open questions

1. `TAU` = 0.85 is a guess. Needs calibration against a labelled pair of
   documents known to share concepts.
2. Do L3/L4 basis vectors live in one matrix with L2, or one matrix per level
   with the CES vector being the concatenation? Affects M's shape and the
   meaning of a CES coordinate. Note the decided row order is level-major, which
   groups levels contiguously either way — so a per-level split is a slicing
   decision, not a reordering one.
3. Sentence splitter: stanza is available but slow; a regex splitter is faster
   and deterministic. Which, given sentences are the projection unit?
