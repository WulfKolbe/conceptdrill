# Owner's example of a knowledge-base structure (verbatim)

Received 2026-11-01 from the owner in the conceptdrill session, with the words:
"please check example structure given below from actual data. Your task is to
develope a general structure that generate such output independent from the
actual content. The output must have always the same structure, but highly
concentrated on the content." and later: "The example was generated based on
your concepts summaries by Grok to give you a better understanding of the goal."

This is reading material for the kb/1 format (`docs/kb-format/kb-1.md`). It is
not a template, not a fixture, and nothing in conceptdrill reads it. Below the
line it is unchanged.

---

**Yes.** Here is a practical, content-driven navigation design so a simple LLM (or a human) can find a topic by reading only a few small files instead of the full concept dump or original papers.

### 1. Folder tree (content-based, not arXiv-ID-based)

```
knowledge-base/
├── README.md                          # Global TOC + concept index + how to search
├── INDEX.md                           # Flat alphabetical concept → path map
├── GLOSSARY.md                        # Short definitions of recurring terms
│
├── 01-probabilistic-models-optimization/
│   ├── README.md                      # Local TOC + key concepts
│   ├── probability-measures-on-relations.md
│   ├── maps-classification.md
│   ├── equivalence-relations-clustering.md
│   ├── linear-orders-ranking.md
│   └── minlp-semi-supervised.md
│
├── 02-graph-communities/
│   ├── README.md
│   ├── hyperbolic-communities.md
│   ├── core-tail-decomposition.md
│   └── community-detection-models.md
│
├── 03-sequential-patterns-mdl/
│   ├── README.md
│   ├── squish-event-sequences.md
│   ├── code-tables-covers.md
│   └── choicisodes-interleaving.md
│
├── 04-knowledge-bases-extraction/
│   ├── README.md
│   ├── commonsense-knowledge-ascent.md
│   ├── facets-and-consolidation.md
│   └── open-information-extraction.md
│
├── 05-conversational-qa/
│   ├── README.md
│   ├── crown-passage-retrieval.md
│   ├── word-proximity-networks.md
│   └── conversational-query-expansion.md
│
├── 06-recommenders-explainability/
│   ├── README.md
│   ├── elixir-actionable-explanations.md
│   ├── counterfactual-explanations-acc.md
│   ├── critique-based-and-set-preference.md
│   └── chat-based-personalization.md
│
├── 07-applied-domains/
│   ├── README.md
│   └── mental-health-forums-triage.md
│
└── _sources/                          # Optional: one thin pointer file per original paper
    ├── 1408.0838.md
    ├── 1602.04650.md
    └── ...
```

Each leaf `.md` file is short (ideally < 2–4 k tokens): abstract-level summary + bullet concepts + “see also” links. Original long extracts stay out of the main tree or live only under `_sources/` as optional deep dives.

### 2. Markdown syntax / navigation rules

Use these conventions everywhere so an LLM can follow a predictable path:

- **YAML frontmatter** (first lines of every file):
  ```yaml
  ---
  title: Hyperbolic Community Models
  topics: [graphs, communities, hyperbolic, density]
  concepts: [core-tail, gamma-H, quasi-clique, nested matrices]
  sources: [1602.04650]
  related: 
    - ../01-probabilistic-models-optimization/equivalence-relations-clustering.md
    - ./community-detection-models.md
  ---
  ```

- **Every folder has a `README.md`** that starts with:
  - One-sentence purpose
  - Bullet list of child files (with 1-line descriptions)
  - “Key concepts in this folder”
  - “Related folders”

- **Global `README.md`** (the entry point) contains:
  - High-level TOC (the 7 folders above)
  - “How to search”: “Start here → open the relevant folder README → follow `related:` links or the concept list”
  - Pointer to `INDEX.md`

- **`INDEX.md`** format (machine- and LLM-friendly):
  ```markdown
  ## Concept Index
  - core-tail decomposition → 02-graph-communities/hyperbolic-communities.md
  - Minimum Description Length (MDL) → 03-sequential-patterns-mdl/README.md
  - counterfactual explanation → 06-recommenders-explainability/counterfactual-explanations-acc.md
  - word proximity network → 05-conversational-qa/word-proximity-networks.md
  ...
  ```

- **Heading discipline**:
  - `# Title`
  - `## Summary` (3–6 sentences max)
  - `## Key Concepts` (bullets)
  - `## See Also`
  - Optional `## Details` or `## Source Pointers` only if needed

- **Link style**: relative Markdown links only. Prefer “See also” sections over burying links in prose.

- **Length rule**: any file an LLM is expected to read first must fit in a single context window with room to spare. Deep extracts go under `_sources/` or are linked externally.

- **Cross-linking**: every concept that appears in more than one place gets an entry in `INDEX.md` and at least one `related:` link.

### 3. General prompts that can generate this structure

**Prompt A – Propose the tree from a concept dump**
```
You are given a concept extraction dump of research papers. 
Propose a content-driven folder tree (not organized by paper ID) that groups related concepts. 
Output only:
1. The folder tree in indented Markdown
2. For each leaf file: a one-line description and the main concepts it should contain
3. A short list of high-level topics that should appear in the root README
Do not copy long text. Aim for navigation, not storage of the full extracts.
```

**Prompt B – Generate a folder README + concept index entries**
```
Given the list of concepts that belong to the folder “{folder_name}”, write:
1. A README.md with YAML frontmatter, one-sentence purpose, bullet list of child files, key concepts, and related folders
2. The corresponding lines that should be added to the global INDEX.md
Keep everything short. Use relative links.
```

**Prompt C – Generate a leaf concept file**
```
Write a short Markdown file for the topic “{topic}”.
Include:
- YAML frontmatter (title, topics, concepts, sources, related)
- ## Summary (max 5 sentences)
- ## Key Concepts (bullets only)
- ## See Also (relative links)
Base the content only on the supplied concept summaries. Do not invent details.
```

**Prompt D – Build / update the global INDEX**
```
From the following list of (concept, file_path) pairs, produce a clean alphabetical INDEX.md 
in the format:
- concept name → path/to/file.md
Merge near-duplicates and prefer the most specific path.
```

**Prompt E – Navigation instruction for a simple LLM**
```
You have a knowledge base organized as described in the root README.md.
To answer a question about a topic:
1. Read only the root README.md and INDEX.md first
2. Open the single most relevant folder README.md
3. Open at most 1–2 leaf files that match the concepts
4. Follow “See Also” or related: links only if necessary
Never read an entire paper extract unless the user explicitly asks for source-level detail.
```

This design lets an LLM start at the root TOC / INDEX, drop into one folder, and reach a precise concept file in 2–3 small reads. The prompts above are reusable whenever you add new concept dumps. 

I can refine the tree, draft the actual root README + INDEX skeleton, or generate the prompts in a ready-to-copy form if you want.
