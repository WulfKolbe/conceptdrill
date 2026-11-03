"""Knowledge bases in the kb/2 format (docs/kb-format/kb-2.md); kb/1 trees
(docs/kb-format/kb-1.md) are still checked.

`contract` holds the formats' mechanical parts, `check` decides conformance.
The generator never learns what a knowledge base is about: no code here names
a field, a topic or a document (tests/test_kb_generality.py).
"""
