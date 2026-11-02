# kb/2 — draft: an area level above topics

Status: DRAFT for review, 2026-11-02. Not implemented. The owner chose it on
2026-11-02 ("Add an area level"). It is a change to `kb-1.md`; everything
there not named here stays as it is.

## Why

kb/1 has two levels, topics then leaves, with topic folders numbered `01`–`99`
and at most 8 leaves per topic. The 87-paper OCR set (1,276 sections) gives
1,060 leaves with the `hash` embedder, so 144 topics, and the build fails at
`topics/100-…`. Even with `modernbert` merging more, 50 or more topics would
not fit a root README a small model reads first: 1,500 tokens is about 40
topic lines. A form that holds only for small sets is not a fixed form.

## The change

One more level, **always present**, so the tree has the same shape for every
set: a small set gets exactly one area.

```
<kb>/
├── README.md                 # Purpose · Areas · How to read · Sources
├── INDEX.md, GLOSSARY.md, index/, glossary/, kb.json, manifest.json   (as kb/1)
├── areas/
│   └── NN-<slug>/            # 01.. contiguous, 1 to 12 areas
│       ├── README.md         # Purpose · Topics · How the topics differ · Related areas
│       └── topics/
│           └── NN-<slug>/    # 01.. contiguous within the area, 1 to 12 topics
│               ├── README.md # as kb/1's topic README
│               └── <slug>.md # leaves, 1 to 8 per topic
├── payload/, text/, sources/ (as kb/1)
```

- Bounds: at most 12 areas × 12 topics × 8 leaves, which is 1,152 leaves. A
  build that needs more fails, naming the count. It never changes the tree.
- Ids: an area id is its slug; a topic id becomes `<area>/<topic>`; a leaf id
  becomes `<area>/<topic>/<leaf>`.
- Area `README.md` frontmatter: `format`, `id`, `title`, `topics`, `concepts`
  (union of its topics'), `related` (areas). Headings: Purpose · Topics · How
  the topics differ · Related areas. Line forms mirror the topic README's.
- The root README's `## Topics` becomes `## Areas`, one line per area.
- Reading protocol, step 3 becomes two: open the area's README, then the
  topic's README, then at most two leaves. The index still points straight at
  leaves, so INDEX → letter → leaf → payload is unchanged.
- `format: kb/2` everywhere. `kb check` checks kb/2. kb/1 trees remain
  checkable as kb/1, by their `format` field.

## Generator

- Leaves and topics are clustered as now, with topics no longer capped at 12
  in total. Areas are the same average-linkage clustering over topic centroids,
  12 topics at most per area, with the area count the smallest that fits
  (ceil(topics / 12)), at least 1.
- The writer gains one call per area (title, purpose, how the topics differ),
  from its topics' titles and purposes. The kb call reads areas, not topics.
- Cost: the leaf and topic prompts are unchanged, so the cached replies still
  serve. New calls: one per area plus the kb call, a few per set. Even so,
  rebuilding ccl-21 and physics-5 under kb/2 is NOT free under --cache-only:
  the area and kb prompts are new and will miss. The estimate goes to the
  owner first.

## Open

- Whether the 1,152-leaf bound should instead grow leaves per topic.
- Whether `kb reach` should also count area READMEs as a read (it measures
  index → leaf, which this change does not lengthen).
