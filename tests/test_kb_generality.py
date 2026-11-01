"""The generator is general: no code or prompt of it knows what it is about.

The owner, 2026-11-01: "I need a generator for knowledge bases not an example
of a knowledge." A field term or a paper title in `conceptdrill/kb/` or its
prompts is that failure. The list names the first test corpus's terms; a
second corpus adds its own.
"""
from __future__ import annotations

import re

from conceptdrill.output import REPO_ROOT

#: Terms of the corpora the generator is tested on. Never allowed in its code.
CORPUS_TERMS = [
    r"label(l)?ing", r"connected.component", r"\bccl\b", r"\bbbdt\b", r"spaghetti",
    r"yacclab", r"union.find", r"\bdrag\b", r"grana", r"bolelli", r"mirflickr",
    r"\boasis\b", r"2016icpr", r"2019tip", r"sensors-15",
]


def _kb_sources():
    root = REPO_ROOT / "src" / "conceptdrill" / "kb"
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".json", ".txt"}]


def test_the_kb_package_exists():
    assert _kb_sources()


def test_no_kb_code_or_prompt_names_a_corpus_term():
    hits = [(str(p.relative_to(REPO_ROOT)), t) for p in _kb_sources()
            for t in CORPUS_TERMS if re.search(t, p.read_text(), re.IGNORECASE)]
    assert hits == []
