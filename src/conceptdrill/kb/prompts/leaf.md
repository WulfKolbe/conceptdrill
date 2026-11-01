KIND: leaf

You write one entry of a knowledge base. Its reader is a small language model
that must apply the knowledge -- answer a question, follow a method, write
code -- after reading only a few short files.

You receive source sections (id, title, text) and a list of payload items by
id and kind: the verbatim equations, code and pseudocode those sections hold.
The payload is shown to the reader exactly as the source has it; do not copy,
rewrite or paraphrase it. Refer to an item by its id when a step needs it.

Use only the given sections. Do not add knowledge they do not contain. Do not
describe the documents ("this paper", "the authors", "the section"): state the
content itself.

Reply with exactly one JSON object and nothing else:

{"title": "a noun phrase of at most 10 words naming the method or concept",
 "use_when": "one or two sentences: the problem this solves",
 "summary": "at most five sentences",
 "concepts": [{"name": "a term of one to five words", "definition": "one sentence"}],
 "method": ["one step a reader can follow, in order, as the sources state it"],
 "pitfalls": ["a warning or limitation the sources give"]}

Give one to six concepts. "method" and "pitfalls" are empty lists when the
sources state no steps or no warnings.
