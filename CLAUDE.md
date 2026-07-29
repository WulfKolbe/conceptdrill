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

A concept-projection tool: it mines a concept vocabulary from a document's own
structure and projects text into it (CES, arXiv 2209.00445, with the document
standing in for the external ontology). See `README.md` for usage and
`docs/superpowers/specs/2026-07-28-conceptdrill-design.md` for the design
rationale.

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

The Semantic Compiler that produces `model.docmodel.json` lives at
`~/drillspace` (package `pdfdrill`). ConceptDrill has no import dependency on
it and must not acquire one.
