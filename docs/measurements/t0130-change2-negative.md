# T-0130 change (2): a measured negative, reverted

Change (2) gave every payload item an index entry. The name came from the
item's caption, else the nearest preceding numbered subsection of its
chapter, else the chapter title -- never the item's own section title.

Measured on --cache-only rebuilds (t0130-reach-c2-*.json):

| | by section title | by paper title |
|---|---|---|
| ccl-21 after (1) | 22/59 | 4/18 |
| ccl-21 after (1)+(2) | 27/59 | 4/18 |
| physics-5 | 0/6 | 0/4 (both) |

All 5 gains on the section-title measure were reached only through generated
entries. Each such entry, made for one item from a preceding subsection,
equals ANOTHER item's own section title: the rule barred naming an item's own
section, not a neighbour's. The independent paper-title measure did not move.
The gain is circular, so the change was reverted (coordinator decision,
2026-11-02). Commits reverted: 9e260f0, d29a8f0, a4f1c44, a116073.
