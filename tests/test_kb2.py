"""kb/2: an area level above topics (docs/kb-format/kb-2.md).

The owner chose it on 2026-11-02 ("Add an area level"): kb/1's one topic
level overflows at the 87-paper OCR set (144 topics, `topics/100-…`). The
area level is always present, so every set has the same tree. kb/1 trees
already built stay checkable as kb/1, by their `format`.

Nothing here spends a key: the writer is stubbed or recorded, the embedder is
`hash`. Names are nonsense on purpose.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from conceptdrill import cli
from conceptdrill.kb import build as B
from conceptdrill.kb.check import check
from conceptdrill.kb.contract import KB1, KB2, SPECS
from conceptdrill.output import REPO_ROOT

LIB = REPO_ROOT / "tests" / "fixtures" / "kb-synthetic"
KB1_TREE = REPO_ROOT / "tests" / "fixtures" / "kb1-synthetic"
SPEC2 = REPO_ROOT / "docs" / "kb-format" / "kb-2.md"
FLORP, ZIB = "Florp Dynamics (v2)", "zib_zab"


def build(tmp_path, *docs, name="s", extra=()):
    code = cli.main(["kb", "build", *docs, "--name", name, "--library", str(LIB),
                     "--output-root", str(tmp_path), "--llm", "stub",
                     "--embedder", "hash", *extra])
    return code, tmp_path / "kb" / name


def kbjson(kb):
    return json.loads((kb / "kb.json").read_text())


def front(path: Path) -> dict:
    import yaml
    text = path.read_text()
    return yaml.safe_load(text[4:text.index("\n---\n", 3) + 1])


def headings(path: Path) -> list[str]:
    return [l[3:] for l in path.read_text().splitlines() if l.startswith("## ")]


@pytest.fixture
def built(tmp_path):
    code, kb = build(tmp_path, FLORP, ZIB)
    assert code == 0
    return kb


@pytest.fixture
def three_areas(tmp_path, monkeypatch):
    """One topic per area: the synthetic set's three topics become three areas."""
    monkeypatch.setattr(B, "MAX_TOPICS_PER_AREA", 1)
    code, kb = build(tmp_path, FLORP, ZIB, name="three")
    assert code == 0
    return kb


# --------------------------------------------------------------------------
# The contract's constants are kb-2.md's
# --------------------------------------------------------------------------

def test_both_formats_are_known():
    assert SPECS == {"kb/1": KB1, "kb/2": KB2}
    assert (KB1.format, KB2.format) == ("kb/1", "kb/2")


def test_the_kb2_reading_protocol_is_the_spec_text():
    text = SPEC2.read_text()
    block = text[text.index("## 8. The reading protocol"):]
    assert KB2.protocol == block.split("```\n")[1].rstrip("\n")


@pytest.mark.parametrize("row,kind", [("| `README.md` |", "readme"),
                                      ("| area `README.md` |", "area"),
                                      ("| topic `README.md` |", "topic"),
                                      ("| leaf |", "leaf")])
def test_the_kb2_heading_table_is_the_spec_table(row, kind):
    table = SPEC2.read_text().split("## 4. Headings")[1].split("## 5.")[0]
    line = next(l for l in table.splitlines() if l.startswith(row))
    assert [h.strip() for h in line.split("|")[2].split("·")] == KB2.headings[kind]


def test_kb2_bounds():
    assert (KB2.max_areas, KB2.max_topics_per_area, KB2.max_leaves_per_topic) == (12, 12, 8)


# --------------------------------------------------------------------------
# kb/1 trees stay checkable
# --------------------------------------------------------------------------

def test_a_kb1_tree_still_conforms_as_kb1():
    report = check(KB1_TREE)
    assert report.violations == []
    assert report.format == "kb/1"


def test_the_cli_names_the_format_it_checked(capsys):
    assert cli.main(["kb", "check", str(KB1_TREE)]) == 0
    assert re.search(r"^kb/1: conforms", capsys.readouterr().out, re.M)


# --------------------------------------------------------------------------
# The generator writes kb/2
# --------------------------------------------------------------------------

def test_the_stubbed_build_conforms_to_kb2(built, capsys):
    report = check(built)
    assert report.violations == [] and report.format == "kb/2"
    assert kbjson(built)["format"] == "kb/2"


def test_there_is_no_root_topics_folder_and_always_an_area(built):
    assert not (built / "topics").exists()
    areas = sorted(p.name for p in (built / "areas").iterdir())
    assert len(areas) == 1 and re.fullmatch(r"01-[a-z0-9-]+", areas[0])
    area = built / "areas" / areas[0]
    assert (area / "README.md").is_file()
    topics = sorted(p.name for p in (area / "topics").iterdir())
    assert topics and all(re.fullmatch(r"\d\d-[a-z0-9-]+", t) for t in topics)
    assert [t[:2] for t in topics] == [f"{i:02d}" for i in range(1, len(topics) + 1)]


def test_ids_nest_area_topic_leaf(built):
    kb = kbjson(built)
    [area] = kb["areas"]
    assert area["folder"] == "areas/01-" + area["id"]
    for t in kb["topics"]:
        assert t["area"] == area["id"] and t["id"].startswith(area["id"] + "/")
        assert t["id"].count("/") == 1
        assert t["folder"].startswith(area["folder"] + "/topics/")
    assert area["topics"] == [t["id"] for t in kb["topics"]]
    for l in kb["leaves"]:
        assert l["id"].count("/") == 2 and l["id"].rsplit("/", 1)[0] == l["topic"]
        assert l["path"].startswith("areas/")


def test_the_root_readme_lists_areas(built):
    f = front(built / "README.md")
    assert f["format"] == "kb/2" and "topics" not in f
    assert f["areas"] == [a["id"] for a in kbjson(built)["areas"]]
    assert headings(built / "README.md") == ["Purpose", "Areas",
                                             "How to read this knowledge base", "Sources"]
    assert re.search(r"^- \[.+\]\(areas/01-[a-z0-9-]+/README\.md\) — ",
                     (built / "README.md").read_text(), re.M)


def test_an_area_readme_has_its_fixed_form(built):
    [area] = kbjson(built)["areas"]
    path = built / area["folder"] / "README.md"
    assert headings(path) == ["Purpose", "Topics", "How the topics differ", "Related areas"]
    f = front(path)
    assert list(f) == ["format", "id", "title", "topics", "related"]
    assert f["topics"] == area["topics"]


def test_an_area_names_no_concepts_in_its_frontmatter():
    """Measured on ccl-21 (66 leaves, one area): the union of an area's
    concepts was 275 frontmatter lines, 948 words, and the README 1836 tokens
    over its 1500 budget. Concepts stay with topics, leaves and the index."""
    assert "concepts" not in KB2.frontmatter["area"]
    row = next(l for l in SPEC2.read_text().splitlines()
               if l.startswith("| `areas/NN-<a>/README.md` |"))
    keys = [k.strip(" `") for k in row.split("|")[2].split(",")]
    assert keys == KB2.frontmatter["area"]


def test_index_entries_point_into_the_area_tree(built):
    text = "".join(p.read_text() for p in (built / "index").glob("*.md"))
    assert re.search(r"\]\(\.\./areas/01-[a-z0-9-]+/topics/\d\d-[a-z0-9-]+/[a-z0-9-]+\.md\)",
                     text)


def test_leaf_links_reach_payload_text_and_sources_from_four_levels_down(built):
    leaf = next(p for p in (built / "areas").rglob("*.md")
                if p.name != "README.md" and "../../../../payload/" in p.read_text())
    text = leaf.read_text()
    assert "../../../../text/" in text and "../../../../sources/" in text


# --------------------------------------------------------------------------
# Several areas
# --------------------------------------------------------------------------

def test_several_areas_conform_and_are_numbered(three_areas):
    assert check(three_areas).violations == []
    names = sorted(p.name for p in (three_areas / "areas").iterdir())
    assert [n[:2] for n in names] == ["01", "02", "03"]
    assert len(kbjson(three_areas)["areas"]) == 3


def test_the_area_count_is_the_smallest_that_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "MAX_TOPICS_PER_AREA", 2)
    code, kb = build(tmp_path, FLORP, ZIB, name="two")
    assert code == 0
    topics = len(kbjson(kb)["topics"])
    assert len(kbjson(kb)["areas"]) == -(-topics // 2)


def test_a_set_needing_too_many_areas_fails_and_names_the_count(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "MAX_TOPICS_PER_AREA", 1)
    monkeypatch.setattr(B, "MAX_AREAS", 2)
    with pytest.raises(B.BuildError, match=r"3 areas.*at most 2"):
        B.build_kb([FLORP, ZIB], name="over", library=LIB, out_root=tmp_path,
                   writer=B.StubWriter(), embedder="hash", command="", commit=None)
    assert not (tmp_path / "kb" / "over").exists()


class RecordingWriter(B.StubWriter):
    def __init__(self):
        super().__init__()
        self.areas, self.kb_input = [], None

    def area(self, topics):
        self.areas.append(topics)
        return super().area(topics)

    def kb(self, areas):
        self.kb_input = areas
        return super().kb(areas)


def test_the_writer_names_each_area_from_its_topics_and_the_kb_from_its_areas(
        tmp_path, monkeypatch):
    monkeypatch.setattr(B, "MAX_TOPICS_PER_AREA", 1)
    w = RecordingWriter()
    B.build_kb([FLORP, ZIB], name="rec", library=LIB, out_root=tmp_path, writer=w,
               embedder="hash", command="", commit=None)
    assert len(w.areas) == 3
    assert all(len(a) == 1 and set(a[0]) == {"title", "purpose"} for a in w.areas)
    assert len(w.kb_input) == 3 and all(set(a) == {"title", "purpose"} for a in w.kb_input)


def test_a_topic_related_across_areas_links_to_the_other_area(three_areas):
    for readme in (three_areas / "areas").glob("*/topics/*/README.md"):
        for target in re.findall(r"\]\(([^)]+)\)", readme.read_text()):
            assert (readme.parent / target).resolve().is_file()


# --------------------------------------------------------------------------
# The checker on broken kb/2 trees
# --------------------------------------------------------------------------

def _copy(built, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(built, dst)
    return dst


def test_area_folders_must_be_numbered_contiguously(three_areas, tmp_path):
    kb = _copy(three_areas, tmp_path)
    third = sorted((kb / "areas").iterdir())[2]
    third.rename(third.with_name("05" + third.name[2:]))
    assert any(re.search(r"^areas/05-[a-z0-9-]+: expected 03-", v)
               for v in check(kb).violations)


def test_a_root_topics_folder_is_not_in_the_kb2_tree(built, tmp_path):
    kb = _copy(built, tmp_path)
    (kb / "topics" / "01-x").mkdir(parents=True)
    (kb / "topics" / "01-x" / "README.md").write_text("x")
    assert any(re.search(r"^topics: not in the kb/2 tree", v) for v in check(kb).violations)


def test_an_area_with_too_many_topics_is_named(built, tmp_path):
    kb = _copy(built, tmp_path)
    data = kbjson(kb)
    data["areas"][0]["topics"] = data["areas"][0]["topics"] * 13
    (kb / "kb.json").write_text(json.dumps(data))
    assert any(re.search(r"kb\.json: schema: areas/0/topics", v) for v in check(kb).violations)


def test_an_area_readme_must_list_its_topics(built, tmp_path):
    kb = _copy(built, tmp_path)
    [readme] = list((kb / "areas").glob("*/README.md"))
    text = readme.read_text()
    readme.write_text(re.sub(r"(## Topics\n\n)- .*\n", r"\1", text, count=1))
    assert any(re.search(r"not linked from areas/01-[a-z0-9-]+/README\.md", v)
               for v in check(kb).violations)


# --------------------------------------------------------------------------
# SOURCES.md: the document list leaves the root README (owner, 2026-11-02)
# --------------------------------------------------------------------------
# The 87-paper set's root README was 1533 tokens over its 1500 budget: its
# ## Sources list (733 words) and `documents` frontmatter (192 words), not
# the areas. The list moves to a root file of its own, as INDEX.md does.

def test_the_root_readme_points_at_sources_md_and_lists_no_documents(built):
    f = front(built / "README.md")
    assert list(f) == ["format", "kb", "title", "areas"]
    text = (built / "README.md").read_text()
    sources = text.split("## Sources\n\n", 1)[1].strip().splitlines()
    assert sources == ["- [Sources](SOURCES.md) — 2 documents"]


def test_sources_md_lists_every_document(built):
    path = built / "SOURCES.md"
    f = front(path)
    assert f == {"format": "kb/2", "kb": "s", "documents": 2}
    assert headings(path) == ["Documents"]
    text = path.read_text()
    assert "# Sources\n" in text
    for d in kbjson(built)["documents"]:
        assert f"- [{d['doc']}](sources/{d['doc']}.md) — {d['title']}" in text


def test_a_source_file_not_listed_in_sources_md_is_named(built, tmp_path):
    kb = _copy(built, tmp_path)
    path = kb / "SOURCES.md"
    path.write_text(re.sub(r"- \[zib-zab\].*\n", "", path.read_text())
                    .replace("documents: 2", "documents: 1"))
    found = check(kb).violations
    assert any(re.search(r"sources/zib-zab\.md: not linked from SOURCES\.md", v) for v in found)
    assert any(re.search(r"SOURCES\.md: .*documents", v) for v in found)


def test_sources_md_is_in_the_spec_tables():
    text = SPEC2.read_text()
    table = text.split("## 4. Headings")[1].split("## 5.")[0]
    row = next(l for l in table.splitlines() if l.startswith("| `SOURCES.md` |"))
    assert [h.strip() for h in row.split("|")[2].split("·")] == KB2.headings["sources_list"]
    front_row = next(l for l in text.splitlines() if l.startswith("| `README.md` | `format: kb/2`"))
    assert [k.strip(" `").split(":")[0] for k in front_row.split("|")[2].split(",")] == \
        KB2.frontmatter["readme"]
    assert "SOURCES.md" in KB2.root_files and "SOURCES.md" not in KB1.root_files
