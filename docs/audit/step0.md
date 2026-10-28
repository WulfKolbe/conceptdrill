# Audit step 0

Five questions, one answer line each. Every citation is code opened and read
during this audit; nothing is cited from memory, from a traceback, or from a
grep hit alone. Line numbers verified against the working tree at `929275d`.

---

**1. Class holding a CES basis concept**
`BasisRow` — `src/conceptdrill/hierarchy/basis.py:110` — frozen dataclass of
`row_id, level, label, vector, support, documents, merged_labels`; held by
`ConceptBasis` at `basis.py:185`.

**2. Class holding a per-sentence concept score**
`ConceptHit` — `src/conceptdrill/hierarchy/project.py:46` — its `similarity`
field is one sentence's cosine against one basis row; the whole vector lives on
`SentenceProjection` at `project.py:60`.

**3. Modules of the span → label → embedding chain, in order**
`docmodel_tree.py:795` (`build_tree` → `MarkerNode:633`, `body_text:651`) →
`summarize.py:521` (`summarize_tree` → `SpanSummary:111`, `BASIS_TIER:89`) →
`basistext.py:271` (`clean_basis_text`) → `embeddings/__init__.py:42`
(`get_embedder`) → `basis.py:315` (`ConceptBasis.integrate`).

**4. Module that already produces spoken math**
`src/conceptdrill/hierarchy/mathtext.py` — `math_text:261` resolves it from
three sources, but the only producer *inside this repository* is
`fallback_speak:150`; the preferred source is read, not produced
(`spoken_from_props:175` over `SPOKEN_PROP_CANDIDATES:42`).

**5. Entry point of the hierarchical pipeline**
`tools/hierarchy_run.py:94` (`main`, argparse at `:95`) — **not** the installed
console script, which is `conceptdrill.cli:main` (`pyproject.toml:58`) and
contains no reference to the hierarchy package.

---

## Where a one-line answer would mislead

**On 3.** The order above is the call order at the one site that runs the
chain, `tools/hierarchy_run.py`: `summarize_tree` at `:197`, `clean_basis_text`
at `:212`, `embedder.encode` at `:248`, `basis.integrate` at `:255`. The chain
is not encoded anywhere as a declared pipeline — it exists only as that
sequence of calls, so reading it off any single module is not possible.

**On 4.** "Produces" splits three ways and the distinction is load-bearing,
since `math_text` returns a `source` tag precisely so a run can say which
happened:

- `docmodel` — `props.spoken`, written by pdfdrill outside this repository and
  merely read here (`mathtext.py:175`).
- `speech` — la2speech/SRE, loaded **by path and never imported**
  (`speech.py:74`, `load_speaker`), so it is also outside this repository.
- `fallback` — `mathtext.py:150`, the one implementation this repository owns.

Answering "mathtext.py" without that split would credit this repository with
speech synthesis it does not contain.

**On 5.** There are two candidate readings of "entry point" and only one is
true. The packaged entry point (`pyproject.toml:58`) reaches the
single-document pipeline only: `cli.py`'s subcommands are `project`
(`cli.py:84`), `concepts` (`:102`), `explain` (`:113`), `verify` (`:124`) and
`routing` (`:127`), and the string `hierarchy` does not occur in that file. The
hierarchical pipeline is reachable only by running `tools/hierarchy_run.py`
directly.

## NOT FOUND

Nothing in questions 1-5 was absent. Two adjacent things were looked for and
are recorded here so a later reader does not repeat the search:

- **A declared span → label → embedding pipeline object.** NOT FOUND. See the
  note on 3; the chain is a call sequence in a script, not a structure.
- **A hierarchy subcommand on the installed CLI.** NOT FOUND
  (`src/conceptdrill/cli.py`, zero occurrences of `hierarchy`).
