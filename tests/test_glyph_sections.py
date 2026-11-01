"""Sections of a `<doc>.glyphs.md`, and the `conceptdrill sections` command.

Twelve of the 21 docmodels of the CCL papers (T-0117) have lost their
sections -- `2016icpr` keeps 2, `2019caip_labeling` 1 -- while the numbered
headings survive as lines of the glyphs markdown. Two-column pages put table
cells and pseudocode steps there too ("1 Pixels", "6 Block R must undergo
Union"), so a numbered line is a heading only if it continues the numbering.
"""
from __future__ import annotations

import json

from conceptdrill import cli
from conceptdrill.hierarchy.glyphsections import split_glyphs
from conceptdrill.hierarchy.summarize import SpanSummary

IEEE = """# A Paper Title

Abstract—We study labeling of connected components in binary images.

I. INTRODUCTION

Labeling assigns one label to every connected region of foreground pixels.

II. RELATED WORK

Two-scan algorithms resolve label equivalences with union-find.

A. Decision trees

Decision trees reduce the number of neighbour reads per pixel.

V. RESULTS

Run time is measured on four datasets.

1 Pixels

REFERENCES

[1] A. Rosenfeld. Sequential operations. 1966.

VI. CONCLUSION

Block-based labeling is fastest.
"""


def _titles(text):
    return [s["title"] for s in split_glyphs(text)]


def test_numbered_headings_become_sections_in_order():
    assert _titles(IEEE) == ["Abstract", "I. INTRODUCTION", "II. RELATED WORK",
                             "A. Decision trees", "V. RESULTS", "VI. CONCLUSION"]


def test_a_numbered_line_that_breaks_the_numbering_is_text():
    sections = split_glyphs(IEEE)
    results = next(s for s in sections if s["title"] == "V. RESULTS")
    assert "1 Pixels" in results["text"]


def test_the_references_are_skipped_not_summarised():
    assert not any("Rosenfeld" in s["text"] for s in split_glyphs(IEEE))


def test_a_parent_section_carries_its_subsections():
    related = next(s for s in split_glyphs(IEEE) if s["title"] == "II. RELATED WORK")
    assert "neighbour reads" in related["text"]


def test_a_markdown_heading_outranks_a_plain_line_with_its_number():
    text = ("Abstract\n\nShort abstract text.\n\n### 1 INTRODUCTION\n\nIntro.\n\n"
            "5 PROPOSED\n\nMethod.\n\n6 Block R must undergo Union\n\n"
            "### 6 YACCLAB\n\nBenchmark.\n")
    assert _titles(text)[-1] == "6. YACCLAB"
    assert "6. Block R must undergo Union" not in _titles(text)


class FakeSummarizer:
    model = "fake-model"

    def summarize(self, span_id, title, body):
        return SpanSummary(span_id=span_id, title=title, summary=f"S {title}",
                           abstraction=f"A {title}", label=f"L {title}",
                           model=self.model)


def test_the_sections_command_writes_the_run_and_the_library_copy(
        tmp_path, monkeypatch):
    library = tmp_path / "library"
    (library / "paperA").mkdir(parents=True)
    (library / "paperA" / "paperA.glyphs.md").write_text(IEEE * 3)
    monkeypatch.setattr(cli, "make_section_summarizer", lambda: FakeSummarizer())
    out = tmp_path / "output"
    code = cli.main(["sections", "paperA", "missing", "--run", "r1",
                     "--library", str(library), "--output-root", str(out)])
    assert code == 0
    run = out / "r1"
    record = json.loads((run / "paperA.json").read_text())
    assert record["sections"][0]["abstraction"] == "A Abstract"
    assert "**Abstract.** A Abstract" in (run / "paperA.md").read_text()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["run_id"] == "r1" and manifest["produced_by"] == "conceptdrill"
    assert manifest["skipped"] == [{"doc": "missing",
                                    "reason": "no glyphs.md in the library"}]
    dest = library / "paperA" / "conceptdrill" / "r1"
    assert (dest / "paperA.json").read_bytes() == (run / "paperA.json").read_bytes()
    assert not (library / "missing").exists()
