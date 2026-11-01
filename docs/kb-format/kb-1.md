# kb/1 — the knowledge-base format

Version `kb/1`, 2026-11-01. Drillgroup T-0119.

This is the contract between the generator (`conceptdrill kb build`) and every
reader of a knowledge base (a small LLM, a CLI, a person). It fixes the *form*
only. It names no topic, field or document; two knowledge bases built from
unrelated document sets differ only in names and content, never in form.
`conceptdrill kb check KB_DIR` decides conformance mechanically: exit 0 when
the tree conforms, exit 1 with every violation listed otherwise.

The machine-readable parts are JSON Schemas shipped with the package:
`src/conceptdrill/kb/schemas/kb.schema.json` and `manifest.schema.json`.

## 1. Words used here

- **document** — one input from the library. Its `folder` is the library
  folder name as it is (any characters); its `doc` is a slug derived from it
  (section 1a) and is what every path and id in the knowledge base uses.
- **section** — one section of a document's `glyphs.md` split, id
  `<doc>#<order>`, `order` counting from 0. Its **text** is its glyphs.md lines
  `line_start`..`line_end`, verbatim.
- **payload item** — a verbatim piece of a document: the display equations of
  one section (in flow order), one code listing (consecutive listing objects
  joined), or one detected pseudocode block. Id `<doc>#<first object id>`.
- **leaf** — one skill card: how to apply one concept or a small group of
  concepts, with its sources and payload.
- **topic** — a group of leaves.
- **slug** — `[a-z0-9]+(-[a-z0-9]+)*`, at most 60 characters.

## 1a. Document slugs

`doc` = the folder name, NFKD-normalised, accents dropped, lower-cased, every
run of characters outside `[a-z0-9]` replaced by one `-`, leading and trailing
`-` removed, cut to 56 characters at a `-` boundary where possible (else hard).
An empty result becomes `doc`. Documents are slugged in order of their folder
names (code-point order); a slug already taken gets `-2`, `-3`, … appended.
`kb.json` keeps both, so the folder can always be found again.

## 2. The tree

Exactly these entries, nothing else:

```
<kb>/
├── README.md
├── INDEX.md
├── GLOSSARY.md
├── kb.json
├── manifest.json
├── index/
│   └── a.md … z.md, other.md         27 files, always all of them
├── glossary/
│   └── a.md … z.md, other.md         27 files, always all of them
├── topics/
│   └── NN-<slug>/                    NN = 01, 02, … contiguous; at least one
│       ├── README.md
│       └── <slug>.md                 one per leaf; at least one per topic
├── payload/
│   └── <doc>/                        one folder per document that has payload
│       └── <file>                    <slug-of-id>.tex | <slug-of-id>.txt
├── text/
│   └── <doc>/                        one folder per input document
│       └── <order>.txt               one per section: its text, verbatim
└── sources/
    └── <doc>.md                      one per input document
```

- No hidden entry (a name starting with `.`) anywhere.
- No empty folder. A `payload/<doc>/` exists only if it holds a file.
- `text/<doc>/<order>.txt` holds no frontmatter and no added line: its bytes
  are the section's glyphs.md lines joined by `\n`, with a final `\n`. It is
  the full-text level, opened only when a summary is not enough; it has no
  size budget, and its word count is stated wherever it is linked. NUL bytes
  in the input stay verbatim; `kb.json` counts them per section (`nul`) so a
  reader is told.
- A letter file holds the entries whose first letter (case-folded, after
  stripping non-letters) is its letter; everything else goes to `other.md`.
- Every name is lower case, except `README.md`, `INDEX.md` and `GLOSSARY.md`.

## 3. Frontmatter

Every `.md` file starts with a YAML block between two `---` lines, holding
exactly the keys below for its file kind, no more. Lists may be empty, never
absent.

| file | keys |
|---|---|
| `README.md` | `format: kb/1`, `kb`, `title`, `documents`, `topics` |
| `INDEX.md` | `format`, `kb`, `letters` |
| `GLOSSARY.md` | `format`, `kb`, `letters` |
| `index/<l>.md` | `format`, `kb`, `letter`, `entries` (int) |
| `glossary/<l>.md` | `format`, `kb`, `letter`, `terms` (int) |
| `topics/NN-<s>/README.md` | `format`, `id`, `title`, `leaves`, `concepts`, `related` |
| `topics/NN-<s>/<s>.md` (leaf) | `format`, `id`, `title`, `topic`, `concepts`, `sources`, `payload`, `related` |
| `sources/<doc>.md` | `format`, `doc`, `title`, `sections` (int) |

- `documents`, `sources`: document names / section ids that exist in `kb.json`.
- `topics`, `leaves`, `related`, `topic`: topic or leaf ids that exist.
- `payload`: payload ids that exist.
- A topic id is its slug; a leaf id is `<topic slug>/<leaf slug>`.

## 4. Headings

After the frontmatter: one `# <title>` line, equal to the frontmatter
`title` (`INDEX.md`: `# Index`; `GLOSSARY.md`: `# Glossary`; letter files:
`# Index: <L>` / `# Glossary: <L>`, with `other` written `Other`), then exactly these `##` headings, in this order, each
present once. A heading with nothing to say holds the single line `None.`
No other `#` or `##` heading appears; `###` and below are free inside a section.

| file | `##` headings in order |
|---|---|
| `README.md` | Purpose · Topics · How to read this knowledge base · Sources |
| `INDEX.md` | Letters |
| `GLOSSARY.md` | Letters |
| `index/<l>.md` | Index |
| `glossary/<l>.md` | Glossary |
| topic `README.md` | Purpose · Leaves · How the approaches differ · Related topics |
| leaf | Use when · Summary · Key concepts · Method · Payload · Pitfalls · Sources · See also |
| `sources/<doc>.md` | Sections · Unused sections |

`## How to read this knowledge base` holds exactly the text of section 8 of
this document.

## 5. Line forms

These sections hold only lines of the given form (plus blank lines), or `None.`:

| where | line form |
|---|---|
| README · Topics | `- [<topic title>](topics/NN-<slug>/README.md) — <one line>` |
| README · Sources | `- [<doc>](sources/<doc>.md) — <document title>` |
| INDEX · Letters, GLOSSARY · Letters | `- [<L>](index/<l>.md) — <n> entries` (glossary: `glossary/<l>.md`, `terms`) |
| index/`<l>` · Index | `- <concept> → [<leaf title>](../topics/NN-<s>/<s>.md)` |
| glossary/`<l>` · Glossary | `- **<term>** — <definition> → [<leaf title>](../topics/NN-<s>/<s>.md)` |
| topic README · Leaves | `- [<leaf title>](<s>.md) — <one line>` |
| topic README · Related topics | `- [<topic title>](../NN-<s>/README.md)` |
| leaf · Key concepts | `- <concept>` (each also in the index, pointing at this leaf) |
| leaf · Payload | `- \`<payload id>\` <kind> · <label>[, <label>] · from \`<section id>\` → [file](../../payload/<doc>/<file>)` |
| leaf · Sources | `- \`<section id>\` <section title> → [text, <n> words](../../text/<doc>/<order>.txt) · [<doc>](../../sources/<doc>.md)` |
| leaf · See also | `- [<leaf title>](<relative path to the leaf>)` |
| sources · Sections | `- \`<section id>\` p.<page> l.<start>–<end> <title> → [text, <n> words](../text/<doc>/<order>.txt) · <leaf links, comma separated>` |
| sources · Unused sections | `- \`<section id>\` p.<page> l.<start>–<end> <title> → [text, <n> words](../text/<doc>/<order>.txt) — <reason>` |

`<kind>` is `equation`, `code` or `pseudocode`. `<label>` is one of
`verified`, `unverified`, `line numbers not separated`, `detected pseudocode`.
Leaf `## Payload` lines may be followed by a fenced block holding the file's
exact content, when the file is at most 40 lines.

## 6. Links

- Every link is a relative Markdown link to a file inside the knowledge base,
  and resolves. No absolute paths, no URLs, no links out of the tree.
- Every leaf is linked from its topic README and from at least one index entry.
- Every payload file is linked from at least one leaf.
- Every topic README is linked from the root README.
- Every `text/` file is linked from its `sources/<doc>.md`; a cited section's
  text is also linked from every leaf citing it.

## 7. Accounting, budgets, provenance

- **Accounting.** `kb.json` lists every section of every input document, each
  either `cited` (with the leaf ids citing it) or `unused` (with a reason).
  Each appears in `sources/<doc>.md` under the matching heading. Likewise every
  payload item found is `placed` (with its file) or `unplaced` (with a reason).
  Nothing found is left out.
- **Verbatim.** A payload file's bytes are the source's bytes as stated in
  `kb.json` (`members`, `field`, joined by `\n` in `kb.json` order); its sha256
  is recorded there.
- **Budgets**, in tokens counted as words × 1.441: root README ≤ 1500, topic
  README ≤ 1500, leaf ≤ 2500 (fenced payload blocks counted), each letter file
  ≤ 4000, `INDEX.md`/`GLOSSARY.md` ≤ 500, `sources/<doc>.md` ≤ 4000. A build
  over budget fails; it never changes the tree.
- **Provenance.** `manifest.json` names the command, the conceptdrill commit,
  every input path with its sha256, every requested document it skipped with
  the reason, the embeddings the structure was clustered from (`embedding`:
  model, sha256 of the float64 vectors, their count), the LLM model and token
  usage (or that a stub stood in for it), and `written_at`. Two builds with
  equal input and embedding hashes have the same structure.

## 7a. Rules beyond the schemas

The schemas fix shapes; `kb check` also enforces, each a violation when broken:

1. `status: cited` ⇒ `cited_by` non-empty and `reason` null; `status: unused`
   ⇒ `cited_by` empty and `reason` a non-empty string.
2. Each leaf in `cited_by` lists that section in its `sources`, and each
   section a leaf lists in `sources` names that leaf in `cited_by`.
3. `status: placed` ⇒ `file`, `sha256`, `section` and `placed_by` set;
   `status: unplaced` ⇒ all four null and `reason` a non-empty string.
4. Payload files, text files and leaf slugs are collision-free: no two items
   share a path.
5. Every id referenced anywhere (frontmatter, `kb.json` cross-references, line
   forms) exists in `kb.json`.
6. Every frontmatter equals `kb.json`: a leaf's `title`, `topic`, `concepts`,
   `sources`, `payload`, `related`; a topic's `title`, `leaves`, `concepts`
   (the union of its leaves'), `related`; the README's `documents` and
   `topics`; a source file's `title` and `sections` (its section count).
7. Each document's sections have orders 0..n-1 without a gap, and every
   section has its text file, whose sha256, word count and NUL count `kb.json`
   records.
8. Document slugs follow section 1a and are unique; `folder` is non-empty.
9. Letter-file entry counts equal their `INDEX.md`/`GLOSSARY.md` lines and
   frontmatter.

When an input (`glyphs`, `docmodel`) still exists with its recorded sha256,
`kb check` also re-derives from it: the section split (ids and text bytes) and
every payload file's bytes. When it is gone or changed, that is reported as a
note, not a violation: the knowledge base stays valid as built.

## 8. The reading protocol (fixed text)

The root README's `## How to read this knowledge base` is exactly:

```
1. Read this file and INDEX.md. Do not open other files yet.
2. Pick the one topic, or the one index entry, that matches the question.
3. Open that topic's README.md, then at most two of its leaves.
4. Open a payload file only when the exact equation, code or pseudocode is needed.
5. Follow See also or Related topics only if the answer is still missing.
6. Open a section's text (text/<doc>/<order>.txt) only when the summary is not enough.
7. Open sources/<doc>.md only to cite or check where a statement comes from.
```
