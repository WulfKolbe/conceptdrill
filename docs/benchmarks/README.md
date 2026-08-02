# Benchmarks vendored into this project

## `bullshitbench-v2.json`

**Source.** Copied verbatim from `/home/wkolbe/Downloads/questions.v2.json` on
2026-08-02, 99,549 bytes, sha256 recorded below. The file identifies itself as
`{"benchmark": "bullshit-detection", "version": "v2.0-draft", "source":
"drafts/new-questions.md"}` — so the upstream original is a markdown draft this
JSON was generated from. It is **not** authored here and is not modified here;
if it needs correcting, correct it upstream and re-copy.

**What it is.** 100 questions that sound professionally competent and are
incoherent. Each carries the domain it imitates, the technique used to build
it, and a `nonsensical_element` saying exactly what is wrong.

    domains     finance 15, legal 15, medical 15, physics 15, software 40
    techniques  13, from plausible_nonexistent_framework (16) to
                reified_metaphor (3)
    scoring     0 full engagement, 1 partial recognition, 2 clear identification

**Why it is here.** The 13 techniques are an ontology of *semantic failure
modes*, and the questions are labelled instances of each class. That makes it
the one thing this project has been missing: text whose meaning is known to be
absent, against which a concept space can be measured.

Everything ConceptDrill measures so far assumes the source says something.
Every gate asks whether a concept was faithfully extracted, never whether there
was anything there to extract. A basis built from these questions spans the
region of concept space occupied by fluent nonsense — and the question worth
answering is whether that region is *separable* from the region occupied by
real papers, or whether the two are interleaved.

**The metric this is for**, in the words of the person who set the task: CES
could introduce an orthogonal pair, *semantic validity* and *context
recoverability* — mathematics scores 1/1, a creative metaphor 0.4/1, this
benchmark roughly 0.1/0.2, a random word salad 0/0. Perplexity and plain
embedding similarity cannot separate a novel scientific hypothesis from fluent
nonsense; the claim is that a conceptual space can.

**Proposed extension, not yet in the file.** Six structural defects judged
detectable on the CES graph rather than on surface text: circular
justification, ontology mismatch, broken provenance, constraint violation,
semantic drift, pseudo-formalism. They are recorded here as the intended
direction; the vendored file does not contain them.

**sha256** `43f43d14bd20ddf17a29fafad2c3e9862e06c2e9c4a5bc94cf08d6ee2a6b1ece`
