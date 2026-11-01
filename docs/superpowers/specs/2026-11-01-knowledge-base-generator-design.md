# Knowledge-base generator — design

Date: 2026-11-01. Status: draft for the owner's approval. Task: drillgroup
T-0119 (`tasks/open/T-0119-conceptdrill-knowledge-base-generator.md`).
Branch: `t-0119-kb-generator`.

## 1. Intent

The owner, 2026-11-01:

> "The conceptdrill goal is to read a set of drilled documents and generate a
> knowledge base that a small LLM can search without swallowing the complete
> content. [...] your generator would produce the content a chat bot would read
> like a set of skills and tools to apply the found knowledge. To achieve for
> example that a CLI can generate code from a set of documents we need
> structures that respect math formulas, LaTeX algorithm definition and code
> snippets."

> "I need a generator for knowledge bases not an example of a knowledge."
> "The output must have always the same structure, but highly concentrated on
> the content."

So the product is a **generator**: `drilled documents in → knowledge-base folder
out`. The folder's *form* is identical for every input set; only its content
changes. A Grok-written example over conceptdrill's own summaries (pasted by the
owner, 2026-11-01) shows the intended shape: a root TOC, an index, a glossary,
content-named topic folders with a README each, short leaf files with
frontmatter and fixed headings, thin pointers to the source papers. It is an
illustration of the target, not a template to copy.

The test case the owner named: **"generate a block-based labeling
implementation in C++ from the papers"**, over the 21 connected-components
labeling (CCL) papers already in the library. A CLI given only the knowledge
base must be able to find the method, its exact definition, and the verbatim
algorithm/formula/code material it needs, in a few small reads.

## 2. Goals

1. **One command** builds a knowledge base from a list of library documents:
   `conceptdrill kb build DOC... --name NAME` (or `--docs-file FILE`), writing
   `output_root()/kb/<NAME>/`.
2. **A fixed skeleton**, the same for every input (section 4), checked by
   `conceptdrill kb check KB_DIR`, which fails on any deviation.
3. **Content-organised, not document-organised.** Topics group concepts across
   papers; papers appear only as sources.
4. **Leaves are skills.** Each leaf tells a reader how to *apply* a method or
   use a fact, not only what a paper says (section 5).
5. **Precise material survives verbatim.** Display equations (LaTeX),
   algorithm/pseudocode blocks and code listings are carried exactly as drilled,
   each with a stable id and a pointer to its source section. The LLM never
   rewrites them.
6. **Small reads.** An LLM reaches a leaf in at most three reads (root README or
   INDEX → topic README → leaf), and every navigation file and leaf has a size
   budget.
7. **Every claim points to a source.** Every leaf names the source sections it
   was built from; nothing in the KB has no source.
8. **Accounted for.** Every section of every input document is either used by
   some leaf or listed as unused with a reason (the project's rule: skipped
   objects are recorded, never dropped).
9. **Repeatable and costed.** A rebuild over unchanged inputs reuses cached LLM
   results; every build records its token usage and commit (as `sections` does
   since T-0117).

## 3. Non-goals and anti-goals

Non-goals (v1):
- Answering questions. The KB is read by another LLM or CLI; the generator does
  not run a chat.
- Proposing new ideas from the KB. The owner's bootstrapping idea ("use a large
  number of blob related OCR documents as an example and then ask a chat to
  propose new ideas for an implementation") is a *use* of a KB. v1 must make it
  possible (cross-paper comparison in topic READMEs, verbatim payload), not do it.
- Fixing pdfdrill's extraction. Garbled two-column text, OCR'd equations and
  line numbers inside code are pdfdrill's; the generator filters and flags them
  and reports what it could not use.
- The per-document LLM reading folder (drillgroup T-0105, parked). That is
  pdfdrill's per-document layer; this KB is the cross-document layer above it,
  and its source pointers are designed to retarget to T-0105's files later.

Anti-goals (would be a failure even if it runs):
- An LLM-paraphrased algorithm or formula presented as the paper's.
- A leaf with a claim that no cited source section supports.
- A structure that changes with the content (a heading that exists in one KB and
  not another, a folder level that appears only sometimes).
- Files a small model must read whole that exceed their budget.
- Writing into the library outside `<doc>/conceptdrill/`, or anywhere outside
  the project folder (house rules).

## 4. The fixed skeleton

```
kb/<name>/
├── README.md            # entry point: purpose, topic TOC, how to read this KB
├── INDEX.md             # letter TOC → index/<letter>.md (always all 27)
├── GLOSSARY.md          # letter TOC → glossary/<letter>.md (always all 27)
├── index/a.md … z.md, other.md     # concept → leaf path; "None." when empty
├── glossary/a.md … z.md, other.md  # term → one-line definition → leaf path
├── kb.json              # machine-readable: every node, id, path, source, hash
├── manifest.json        # command, commit, inputs + sha256, token usage, written_at
├── NN-<topic-slug>/     # 01.., one per topic, ordered by size then name
│   ├── README.md        # topic purpose, leaves with one line each, comparison, related topics
│   └── <leaf-slug>.md   # one skill card per concept group
├── payload/
│   └── <doc>/<payload-id>.<ext>   # verbatim equation (.tex), algorithm (.txt), code (.c/.cpp/.txt)
└── sources/
    └── <doc>.md         # one per input document: title, its sections with ids, which leaves use each
```

Every `.md` file starts with YAML frontmatter and has fixed headings, in fixed
order, always present (an empty section says `None.`, it is never omitted):

| file | frontmatter keys | headings |
|---|---|---|
| root `README.md` | `kb, title, documents, topics, built_with` | `## Purpose`, `## Topics`, `## How to read this knowledge base`, `## Sources` |
| `INDEX.md`, `GLOSSARY.md` | `kb, letters` | `## Letters` (one line per letter file, with its entry count) |
| `index/<l>.md` | `kb, letter, entries` | `## Index` |
| `glossary/<l>.md` | `kb, letter, terms` | `## Glossary` |
| topic `README.md` | `id, title, leaves, concepts, related` | `## Purpose`, `## Leaves`, `## How the approaches differ`, `## Related topics` |
| leaf | `id, title, topic, concepts, sources, payload, related` | `## Use when`, `## Summary`, `## Key concepts`, `## Method`, `## Payload`, `## Pitfalls`, `## Sources`, `## See also` |
| `sources/<doc>.md` | `doc, title, sections` | `## Sections`, `## Unused sections` |

`## How to read this knowledge base` is fixed text, the same in every KB: the
reading protocol for a small model (read README/INDEX, open one topic README,
open at most two leaves, open a payload file only when exact material is
needed, follow See also only if necessary).

Ids are stable and derived, never random: a leaf id is its topic slug plus leaf
slug; a payload id is `<doc>#<first docmodel object id>` (section 6.2); a
section id is `<doc>#<order>` over the glyphs.md split, as in `sections`
(T-0117). Each section also records `page`, `line_start`, `line_end` in
glyphs.md (`t-0117-cache-ids`, cc03796), so it can later be mapped to
pdfdrill's Section ids by page and title for T-0105 (via T-0077). conceptdrill
invents no pdfdrill ids (coordinator, Q5).

## 5. The leaf as a skill card

Prose headings are written by the LLM from the cited sections only: their
tiers (summary, abstraction, label, further concepts) **and their text**
(capped per section), plus the payload list. The ~25-word tiers alone cannot
give `## Method` steps or `## Pitfalls`; the text can, and stays within the
cited sources. It is told to use nothing else and to write `None.` where the
sources say nothing. (Designer's call; input cost for the 21 CCL papers is
about 0.5 M prompt tokens, ~$0.03 at ling-3.0-flash prices.)

- `## Use when` — the problem this solves, in one or two sentences.
- `## Summary` — at most 5 sentences.
- `## Key concepts` — bullets, each a concept that also appears in INDEX.md.
- `## Method` — numbered steps a reader can follow, as far as the sources state them.
- `## Payload` — **inserted by code, not the LLM**: one entry per payload item:
  kind, id, source section, and either the verbatim content inline (when small)
  or a link to `payload/<doc>/<id>.<ext>`.
- `## Pitfalls` — what the sources warn about; `None.` otherwise.
- `## Sources` — inserted by code: the cited section ids with titles, linking
  to `sources/<doc>.md`.
- `## See also` — inserted by code from the topic graph.

## 6. Pipeline

Each stage writes its result into the build directory, so a failed build can be
inspected and resumed; LLM calls go through `SummaryCache`.

1. **Sections.** Always the glyphs.md split of `sections` (T-0117) — one id
   scheme for every document; docmodel sections are not used (in
   `2009tiplabeling` and `2019tip_FoSNNcZ` they are body lines). Per section:
   the tiers, reused from a `sections` run given with `--from-run RUN` only if
   that run's manifest hashes match today's glyphs.md; otherwise `kb build`
   refuses and says to re-run `sections`. (labeling-21 has no input hashes and
   its inputs were re-drilled since, so acceptance uses a fresh `sections` run.)
2. **Payload.** From `model.docmodel.json`, by type. A payload item is placed
   in the glyphs.md section that contains it: first by finding its text
   (a code line, the equation's LaTeX) inside the section's line range, then by
   page; every item records how it was placed (`text`, `page`), and items that
   cannot be placed are listed in `kb.json`, never dropped.
   - `Equation` → display equations, LaTeX from `props.latex` exactly as
     stored. Verification (coordinator, Q4; `confidence` is NOT a signal):
     `verified` when the equation is in drillgroup's
     `results/2026-10-31/equations/inkdrill-confirmed.csv` (inkdrill distance 0)
     or matches the author's LaTeX for texsrc documents
     (`results/2026-10-31/equations`); otherwise `unverified`. Unverified
     equations go into the KB labelled so, not filtered out.
   - `CodeListing` → code (`props.code`), **verbatim**, marked "line numbers not
     separated" (coordinator, Q3: the fused numbers such as `{ 1112` are
     pdfdrill's, fixed in pdfdrill T-0120; do not strip by guessing).
     Consecutive `CodeListing` objects (adjacent in `flow_index`, same page or
     next page) are one listing: Spaghetti's code is 7 objects, DRAG's 10. The
     listing's file is the objects' `props.code` joined by `\n` in flow order;
     its id is the first object's, and `kb.json` lists all member ids.
   - Algorithm blocks: there is no algorithm object type in the docmodel. v1
     detects pseudocode among `CodeListing`, `ListItem` and `Paragraph` runs by
     markers (numbered steps, `for/while/if … then`, `←`, `:=`, `procedure`,
     `Algorithm N`) and keeps the run verbatim as `.txt`.
   - Inline `Formula` objects are not payload (2484 in the 21 CCL papers, mostly
     single symbols); they stay inside the text they belong to.
   Placeholders pdfdrill left (`<!-- DEFERRED … -->`) are not payload; they are
   counted in `kb.json`.
3. **Concepts.** Every section concept (dominant and further ones) becomes a
   unit with its tiers, source section id and the section's payload.
4. **Topics and leaves (structure decided by code).** Embed concept labels
   (default encoder `modernbert`, cached), merge near-duplicates across
   documents with the existing `ConceptBasis` merge at TAU, then group the
   merged concepts into topics by agglomerative clustering on cosine in float64.
   Each topic has 1..K leaves; a leaf is one merged concept or a small group of
   them. Topic count and leaf sizes are bounded (defaults: 3–12 topics,
   ≤ 8 leaves per topic, ≤ 6 source sections per leaf).
5. **Naming and prose (LLM).** One call per topic names it and writes the topic
   README prose ("How the approaches differ"); one call per leaf writes the
   leaf's prose headings. Slugs come from the names, made unique by code.
6. **Navigation (code only).** Root README, INDEX, GLOSSARY (terms from leaf
   key concepts, definitions from the abstraction tier), `sources/<doc>.md`,
   `kb.json`, `manifest.json`.
7. **Check.** `kb check` runs as the last stage of `build` and stands alone.

## 7. `kb check` — what "same structure" means, testably

Fails (exit 1, every violation listed) unless:
- the tree matches section 4 exactly: no extra or missing files or folders;
- every `.md` has its frontmatter keys and exactly its headings, in order;
- every relative link resolves; every payload file is referenced by a leaf;
- every INDEX concept points to a leaf that lists it under Key concepts;
- every leaf cites at least one source section, and every input section is
  either cited or listed under `## Unused sections` with a reason;
- every payload file's bytes equal the docmodel's content for its ids
  (`props.latex`; `props.code` joined as stated);
- size budgets hold (defaults, in tokens at the measured 1.441 tokens/word:
  root README ≤ 1500, topic README ≤ 1500, leaf ≤ 2500 excluding linked
  payload files, each letter file ≤ 4000; over budget fails the build, it
  never changes the skeleton).

## 8. Constraints

- Output under `output_root()` (T-0117); caches under `cache_dir()`; nothing
  outside the project folder, no hidden files, no temp files.
- No import of pdfdrill or any neighbour project; formats are reproduced.
- The library is read only; the KB is not copied into document folders.
- LLM: NovitaSummarizer, default model `inclusionai/ling-3.0-flash`, key
  `NOVITA_API_KEY`; any run that spends it is named and costed beforehand.
- float64 for every similarity decision (this host's float32 BLAS is wrong).
- Deterministic ordering everywhere; given the same cached LLM results, a
  rebuild is byte-identical except `written_at`.

## 9. Acceptance

On the 21 CCL papers (labeling-21 set), through the installed command:
1. `conceptdrill sections <21 docs> --run ccl-21` (fresh, hashed), then
   `conceptdrill kb build <21 docs> --name ccl-21 --from-run ccl-21` writes
   `work/output/kb/ccl-21/`; `kb check` passes; manifests record usage.
2. The owner's test case, walked by hand and recorded: from README/INDEX to a
   block-based labeling leaf in ≤ 3 reads; that leaf lists the decision-tree /
   block-scan payload of the BBDT, Spaghetti and DRAG papers, and the payload
   files equal the docmodel's content.
3. A second, unrelated document set builds a KB with the identical skeleton
   (`kb check` passes; the two trees differ only in topic/leaf names and
   content).
4. Comparison with the owner's chat that read 20 drilled CCL documents by hand:
   the method's needed facts (to be listed from that chat) are reachable in
   the KB.

## 10. Calls made by the designer (for the owner to check)

- Topic structure by embeddings + clustering in code, the LLM only names and
  writes prose — so the skeleton and its bounds are code's, not the model's.
- Payload is copied by code and never passes through the LLM.
- Algorithm detection by text markers, because the docmodel has no algorithm
  type and will not get one (coordinator, Q3: pseudocode handling is pdfdrill
  T-0120). Detected runs are verbatim and labelled "detected pseudocode".
- One section scheme (glyphs.md) for all documents; payload placed into it by
  text, then page.
- Leaf prose sees the cited sections' text, not only the tiers.
- INDEX and GLOSSARY always split into 27 letter files.
- Consecutive CodeListing objects are stitched into one listing.
- Default budgets and bounds in sections 6–7.
- The KB lives under `work/output/kb/`; nothing is copied into the library.

## 11. Open questions

For the owner:
- Q1. The comparison chat that read 20 drilled CCL documents: where is its
  output, so acceptance 4 can use it?
- Q2. Is `ling-3.0-flash` right for leaf prose, or should leaves use the model
  that met the word bands best (`ling-3.1-flash`, ~5× the cost)?

Answered by the drillgroup coordinator, 2026-11-01 (now in sections 4 and 6):
- Q3: no algorithm object in pdfdrill, by decision; pseudocode and the
  CodeListing line numbers are pdfdrill T-0120. Keep listings verbatim, marked
  "line numbers not separated".
- Q4: `confidence` is not a signal. Verified = inkdrill distance 0 or the
  author's LaTeX; all else labelled "unverified", not filtered.
- Q5: T-0105 will key on pdfdrill's Section id via T-0077; keep page + line
  range per section for a later mapping by page and title.
