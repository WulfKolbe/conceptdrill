# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m pytest                          # full suite, offline, ~10s
python -m pytest tests/test_scoring.py     # one file
python -m pytest -k coverage_curve         # one test
pip install -e '.[models,nlp,dev]'         # everything

PYTHONPATH=src python -m conceptdrill.cli …   # run without installing
```

There is no linter or type-checker configured.

## What this is

A concept-projection tool implementing CES (arXiv 2209.00445) with the document
standing in for the external ontology. **Two pipelines:**

- `conceptdrill/` — single-document. Mines a vocabulary from one document's own
  structure. Complete.
- `conceptdrill/hierarchy/` — multi-document. Builds a shared basis from LLM
  summaries of section hierarchies across a corpus, projects sentences into it,
  stores both, and answers queries. Complete end to end.

See `README.md` for usage and the two specs under
`docs/superpowers/specs/` for design rationale.

## Architecture

The pipeline is `document → candidates → scoring → selection → embedding →
projection`, and each stage is injectable through `ConceptDrill.__init__`.

Things that are not obvious from any single file:

- **`ScoringContext` is computed once per build and shared by all seven
  metrics.** It holds block embeddings, candidate embeddings, and a cached
  similarity row per candidate. Four metrics read the same row — computing them
  independently would multiply the embedding cost. `core.build()` then reuses
  those same candidate embeddings for selection via `_align_embeddings`, so the
  whole build is one embedding pass.

- **Everything is ordered deterministically on purpose.** Candidates sort by
  `(-structural_weight, -frequency, name)`, scored items by `(-score, name)`,
  metrics iterate `sorted(self.metrics)`. Changing any of these changes output
  hashes and breaks `test_content_hash_is_reproducible`.

- **`nlp.analyse()` is `lru_cache`d and returns noun phrases *and* entities from
  one pipeline run.** Both consumers read it. Calling stanza per text block
  costs ~22s per pass versus ~2s batched — this was a real bug, not a
  hypothetical.

- **The DocModel keeps text under a different prop per object type.**
  `docmodel.TEXT_EXTRACTORS` is that mapping and the only place a new object
  type needs handling.

- **`bibitem` is excluded from `PROSE_TYPES` deliberately.** Reference lists are
  projectable but mining them for noun phrases yields author surnames, which
  then outrank real concepts.

## Hierarchy package

`hierarchy/` reads drilled documents from `~/pdfdrill-library/` and writes back
into the drill folder. Things not obvious from any single file:

- **`model.docmodel.json` is the only reliable input.** The `.md` retains 2
  headers and `.llm.txt` none. Three traps: titles live under `props.caption`,
  `parent` is null on every Section (rebuild from `level` + `flow_index`), and
  captions carry unresolved LaTeX macros.

- **Levels do not start at 1.** Across 334 library documents, 89 start at level
  1 and 103 at level 2. Nothing may assume a root level.

- **`Formula`/`Equation` carry no `parent_section`** — zero of 74 in the
  reference paper — so `assign_by_flow` infers the owner from `flow_index`.
  Without it every formula becomes an orphan.

- **`summarize_tree` feeds `subtree_text`, not `body_text`.** Summarising a
  level-2 section from its own paragraphs while ignoring its subsections
  describes almost nothing (1350 vs 11141 characters for one real section).

- **Only the `label` tier fits the BERT window.** Measured 1.604 tokens/word:
  `label` 30-42 words lands at 48-67 tokens; the other two overshoot.

- **All LLM output goes through `sanitize.py`.** Models emit invisible and
  lookalike characters that change tokenisation. It preserves Greek, accents
  and CJK — those are content.

- **`replyparse.control_corruption` runs BEFORE sanitising.** Sanitising strips
  the control characters that evidence a LaTeX command eaten by a legal JSON
  escape (`\t`, `\b`, `\f`, `\r`).

- **`refine.py` is the paper's Algorithm 2; `basis.py` is not.** Refinement
  grows one space *down* a hierarchy (from the paper); merging fuses spaces
  *sideways* across a corpus (this project's design). Do not conflate them.

- **The paper's siblings score is degenerate here.** It ranks children by
  shared-parent overlap, which needs a DAG. 0 of 8695 sections have more than
  one parent, so it is always 1.0 and `children_ranked` falls back to document
  order. `sibscore_informative` reports which happened.

- **`basis_version` is checked on every load.** `corpus.py` refuses vectors
  belonging to a different basis rather than misreading them; a CES coordinate
  means whatever row it indexed at projection time.

- **Search is in CES space, by cosine.** Dot product would let a sentence
  matching everything weakly outrank one matching one concept strongly.

- **`basis.py` is float64 throughout.** A merge decision is one comparison
  against TAU; a wrong cosine adds a row that should have merged. `DEFAULT_TAU`
  is 0.65, measured — 0.85 produced zero merges across three related papers.

## Integration with pdfdrill

`~/MX/PDFDRILL` is where projections are collected. Follow its conventions
rather than inventing parallel ones:

- Output goes in the **drill folder** as `model.ces.json`, joining the
  `model.<stage>.json` family.
- Register `CES_BUILT` in the sidecar via `hierarchy/sidecar.py`, which
  reproduces pdfdrill's proof format byte-compatibly **without importing it**.
  `capability_valid()` re-hashes inputs — do not reinvent staleness.
- Sidecar writes are additive; preserve keys you do not understand.
- The corpus basis is **not** per-document and must not live in a drill folder.

## Constraints that must hold

- **Never write to the input document.** Output goes to a sidecar. `api.py` and
  `storage.py` open inputs read-only; keep it that way.
- **Skipped objects are recorded, never dropped.** A run must account for every
  object in the input — `test_every_object_is_accounted_for` enforces this.
- **All embedders return L2-normalised rows.** The matrix shortcut `M @ l` is
  only cosine similarity because of this.
- **Timestamps stay out of ids and `content_hash`.** `storage.VOLATILE_KEYS`
  lists what is excluded.
- **A sidecar stores one concept space per model** under `concept_spaces`.
  Each model selects its own vocabulary, so resolving a projection against
  another model's space would dangle. `storage.resolve_concept` does it right.

## Environment notes

- **`CONCEPTDRILL_DEVICE` defaults to CPU, intentionally.** This machine's ROCm
  stack reports `cuda.is_available() == True` and then **segfaults** on the
  forward pass. Do not "fix" the device default back to autodetection.
- **`CONCEPTDRILL_NLP_BACKEND`** pins the NLP tier. Tests pin it to `regex`;
  stanza is installed here and would otherwise change the mined vocabulary.
- `np.savez` appends `.npz` to a path that lacks it — `cache.flush()` writes
  through a file object to keep the atomic rename working.
- **This host's BLAS is broken; float32 matmul is non-deterministic and wrong.**
  Repeating `a @ b` on identical tensors, single-threaded, differs in up to
  100% of trials with deviations up to 4.25. float64, `sum()`, elementwise ops
  and memory readback are all clean. numpy and torch both link OpenBLAS 0.3.26
  built for `CORE2` with an empty `DYNAMIC_ARCH`. Verify before trusting ANY
  float32 numerical result measured here:
      python3 -c "import torch;a=torch.randn(512,768);w=torch.randn(768,768);\
      r=(a@w).clone();print(sum(1 for _ in range(100) if not torch.equal(a@w,r)))"
- **Only the `hash` backend is bit-reproducible here.** That is a consequence of
  the BLAS fault above, not of the pipeline. `torch.set_num_threads(1)` and
  `attn_implementation="eager"` are set by default because they reduce variance
  cheaply — they do NOT fix it, and earlier notes claiming thread scheduling and
  SDPA as the cause were wrong. Re-measure on a host with a correct BLAS before
  drawing conclusions about ConceptDrill's reproducibility.

## Related

| path | what it is |
|---|---|
| `~/MX/PDFDRILL` | current pdfdrill; sidecar/proofs/repo conventions; where projections are collected |
| `~/pdfdrill-library/` | 3300 drilled documents, 334 with a docmodel, 251 usable |
| `~/la2speech/` | LaTeX to spoken text (SRE); source of formula prose |
| `~/Gemma4/` | the original `section_concepts.py` and prompt this package adapts |
| `~/drillspace` | older Semantic Compiler checkout |

ConceptDrill has **no import dependency** on any of them and must not acquire
one. Formats are reproduced, not imported.

Credentials live in a gitignored `.env` (see `.env.example`), read only from
this project. Never read a neighbouring project's key file.
