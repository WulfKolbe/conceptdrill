"""`kb build` with its LLM stubbed: the deterministic generator (T-0119, step B).

The input is tests/fixtures/kb-synthetic, two tiny documents with nonsense
topics laid out like the library. Nothing here spends an API key: the writer
is `StubWriter`, which returns fixed strings, and the embedder is `hash`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptdrill import cli
from conceptdrill.kb.check import check, form_of
from conceptdrill.output import REPO_ROOT

LIB = REPO_ROOT / "tests" / "fixtures" / "kb-synthetic"
FLORP, ZIB = "Florp Dynamics (v2)", "zib_zab"


def build(tmp_path, *docs, name="synthetic", extra=()):
    code = cli.main(["kb", "build", *docs, "--name", name, "--library", str(LIB),
                     "--output-root", str(tmp_path), "--llm", "stub",
                     "--embedder", "hash", *extra])
    return code, tmp_path / "kb" / name


@pytest.fixture
def built(tmp_path):
    code, kb = build(tmp_path, FLORP, ZIB)
    assert code == 0
    return kb


def kbjson(kb):
    return json.loads((kb / "kb.json").read_text())


def test_the_stubbed_build_conforms(built):
    report = check(built)
    assert report.violations == []


def test_document_slugs_and_folders(built):
    docs = {d["doc"]: d["folder"] for d in kbjson(built)["documents"]}
    assert docs == {"florp-dynamics-v2": FLORP, "zib-zab": ZIB}


def test_every_section_is_accounted_for(built):
    sections = kbjson(built)["sections"]
    assert len(sections) == 9
    tiny = next(s for s in sections if s["id"] == "zib-zab#3")
    assert tiny["status"] == "unused" and "words" in tiny["reason"]
    assert all(s["status"] == "cited" for s in sections if s["id"] != "zib-zab#3")


def test_text_files_are_the_sections_lines_verbatim(built):
    glyphs = (LIB / ZIB / f"{ZIB}.glyphs.md").read_text().splitlines()
    s = next(s for s in kbjson(built)["sections"] if s["id"] == "zib-zab#1")
    want = "\n".join(glyphs[s["line_start"] - 1:s["line_end"]]) + "\n"
    assert (built / s["text_file"]).read_text() == want


def _payload(kb, kind):
    return [p for p in kbjson(kb)["payload"] if p["kind"] == kind]


def test_consecutive_code_listings_are_one_verbatim_item(built):
    [code] = _payload(built, "code")
    assert code["members"] == ["obj_cl1", "obj_cl2"]
    assert code["labels"] == ["line numbers not separated"]
    assert (built / code["file"]).read_text() == "int florp(int q) { 1\nreturn q * 2; 2"


def test_a_sections_display_equations_are_one_item_placed_by_text(built):
    [eq] = _payload(built, "equation")
    assert eq["members"] == ["obj_eq1"] and eq["field"] == "latex"
    assert eq["placed_by"] == "text" and eq["section"] == "florp-dynamics-v2#2"
    assert eq["labels"] == ["unverified"]
    assert (built / eq["file"]).read_text() == "q = z^{2} + 1"


def test_a_pseudocode_run_is_detected_and_kept_verbatim(built):
    [ps] = _payload(built, "pseudocode")
    assert ps["members"] == ["obj_li1", "obj_li2", "obj_li3"]
    assert ps["labels"] == ["detected pseudocode"]
    assert ps["section"] == "florp-dynamics-v2#3"


def test_deferred_placeholders_are_not_payload(built):
    assert not any("obj_eq9" in p["members"] for p in kbjson(built)["payload"])


def test_a_document_without_glyphs_is_listed_and_skipped(tmp_path):
    code, kb = build(tmp_path, FLORP, "nowhere")
    assert code == 0
    manifest = json.loads((kb / "manifest.json").read_text())
    assert manifest["skipped"] == [{"folder": "nowhere",
                                    "reason": "no nowhere.glyphs.md or nowhere.md in the library"}]
    assert check(kb).violations == []


def test_the_stub_spends_nothing_and_says_so(built):
    llm = json.loads((built / "manifest.json").read_text())["llm"]
    assert llm == {"stub": True, "model": None,
                   "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}


def test_a_rebuild_is_byte_identical_except_written_at(tmp_path):
    _, kb = build(tmp_path, FLORP, ZIB)
    first = {p.relative_to(kb).as_posix(): p.read_bytes()
             for p in kb.rglob("*") if p.is_file()}
    code, kb = build(tmp_path, FLORP, ZIB, extra=["--replace"])
    assert code == 0
    second = {p.relative_to(kb).as_posix(): p.read_bytes()
              for p in kb.rglob("*") if p.is_file()}
    assert first.keys() == second.keys()
    differ = [k for k in first if first[k] != second[k]]
    assert differ == ["manifest.json"]


def test_an_existing_kb_is_not_overwritten_without_replace(tmp_path):
    build(tmp_path, FLORP)
    code, _ = build(tmp_path, FLORP)
    assert code == 1


def test_unrelated_sets_give_the_same_form(tmp_path):
    _, a = build(tmp_path, FLORP, name="a")
    _, b = build(tmp_path, ZIB, name="b")
    assert check(a).violations == [] and check(b).violations == []
    assert form_of(a) == form_of(b)


def test_nul_bytes_are_kept_and_counted(tmp_path):
    """pdfdrill leaves NUL bytes in some glyphs.md (a pdfdrill defect). They
    stay verbatim in the text file, and kb.json tells the reader how many."""
    import shutil
    lib = tmp_path / "lib"
    shutil.copytree(LIB / ZIB, lib / ZIB)
    g = lib / ZIB / f"{ZIB}.glyphs.md"
    g.write_text(g.read_text().replace("1 Zib Basics\n\n", "1 Zib Basics\n\n\x00\x00 "))
    code = cli.main(["kb", "build", ZIB, "--name", "nul", "--library", str(lib),
                     "--output-root", str(tmp_path), "--llm", "stub", "--embedder", "hash"])
    assert code == 0
    kb = tmp_path / "kb" / "nul"
    s = next(s for s in kbjson(kb)["sections"] if s["id"] == "zib-zab#1")
    assert s["nul"] == 2 and (kb / s["text_file"]).read_text().count("\x00") == 2
    assert all(x["nul"] == 0 for x in kbjson(kb)["sections"] if x["id"] != "zib-zab#1")


def test_the_manifest_records_the_embeddings_used(built):
    emb = json.loads((built / "manifest.json").read_text())["embedding"]
    assert emb["model"] == "hash" and len(emb["sha256"]) == 64 and emb["vectors"] == 8


def test_the_embeddings_are_cached_on_disk(tmp_path, monkeypatch):
    """A 16-minute modernbert pass must not be repeated on the next build."""
    monkeypatch.setenv("CONCEPTDRILL_CACHE_DIR", str(tmp_path / "cache"))
    code, _ = build(tmp_path, ZIB)
    assert code == 0
    assert list((tmp_path / "cache").glob("emb-hash-*.npz"))


# --------------------------------------------------------------------------
# T-0130 change (1): payload travels with its chapter
# --------------------------------------------------------------------------

def _sec(order, number):
    from conceptdrill.kb.build import Section
    return Section(doc="d", order=order, title=f"S{order}", number=number, page=1,
                   line_start=1, line_end=1, text="x\n", embed_text="x", status="cited")


def _item(section, member):
    from conceptdrill.kb.build import Payload
    return Payload("d", "code", [member], "code", "c", 1, "c", [], section, "text")


def test_a_leaf_citing_a_chapter_carries_its_subsections_payload():
    """Same paper, same top-level section: chapter 2's leaf carries the
    payload of 2.1, even in a different leaf; chapter 3's does not travel."""
    import numpy as np
    from conceptdrill.kb.build import Leaf, attach_payload
    s2, s21, s3 = _sec(2, [2]), _sec(3, [2, 1]), _sec(4, [3])
    p21, p3 = _item("d#3", "obj_a"), _item("d#4", "obj_b")
    leaves = [Leaf([s2], [], np.zeros(2)), Leaf([s21], [], np.zeros(2)),
              Leaf([s3], [], np.zeros(2))]
    attach_payload(leaves, [p21, p3])
    assert [p.id for p in leaves[0].payload] == ["d#obj_a"]
    assert [p.id for p in leaves[1].payload] == ["d#obj_a"]
    assert [p.id for p in leaves[2].payload] == ["d#obj_b"]


def test_unnumbered_sections_keep_only_their_own_payload():
    import numpy as np
    from conceptdrill.kb.build import Leaf, attach_payload
    a, b = _sec(0, []), _sec(1, [])
    leaves = [Leaf([a], [], np.zeros(2)), Leaf([b], [], np.zeros(2))]
    attach_payload(leaves, [_item("d#1", "obj_c")])
    assert [p.id for p in leaves[0].payload] == []
    assert [p.id for p in leaves[1].payload] == ["d#obj_c"]


def test_a_chapter_number_from_another_paper_is_another_chapter():
    import numpy as np
    from conceptdrill.kb.build import Leaf, attach_payload
    mine, other = _sec(2, [2]), _sec(2, [2])
    other.doc = "e"
    leaves = [Leaf([mine], [], np.zeros(2)), Leaf([other], [], np.zeros(2))]
    attach_payload(leaves, [_item("d#2", "obj_d")])
    assert [p.id for p in leaves[1].payload] == []


# --------------------------------------------------------------------------
# A document's Markdown projection as input, when it has no glyphs.md
# --------------------------------------------------------------------------

def _md_library(tmp_path, with_glyphs=False):
    import shutil
    lib = tmp_path / "lib"
    (lib / ZIB).mkdir(parents=True)
    text = (LIB / ZIB / f"{ZIB}.glyphs.md").read_text()
    (lib / ZIB / f"{ZIB}.md").write_text(text)
    if with_glyphs:
        shutil.copy(LIB / ZIB / f"{ZIB}.glyphs.md", lib / ZIB / f"{ZIB}.glyphs.md")
    return lib


def _build_lib(tmp_path, lib, *docs, name="m"):
    return cli.main(["kb", "build", *docs, "--name", name, "--library", str(lib),
                     "--output-root", str(tmp_path), "--llm", "stub", "--embedder", "hash"])


def test_a_document_without_glyphs_is_read_from_its_markdown(tmp_path):
    lib = _md_library(tmp_path)
    assert _build_lib(tmp_path, lib, ZIB) == 0
    kb = tmp_path / "kb" / "m"
    doc = kbjson(kb)["documents"][0]
    assert doc["input"] == "md" and doc["split"] == "numbered"
    assert doc["glyphs"]["path"].endswith(f"{ZIB}.md")
    assert check(kb).violations == []


def test_glyphs_is_preferred_when_both_exist(tmp_path):
    lib = _md_library(tmp_path, with_glyphs=True)
    assert _build_lib(tmp_path, lib, ZIB) == 0
    assert kbjson(tmp_path / "kb" / "m")["documents"][0]["input"] == "glyphs"


def test_a_document_with_neither_is_skipped_and_says_so(tmp_path):
    lib = _md_library(tmp_path)
    (lib / "bare").mkdir()
    assert _build_lib(tmp_path, lib, ZIB, "bare") == 0
    manifest = json.loads((tmp_path / "kb" / "m" / "manifest.json").read_text())
    assert manifest["skipped"] == [{"folder": "bare",
                                    "reason": "no bare.glyphs.md or bare.md in the library"}]


def test_docs_file_skips_comments_and_blank_lines(tmp_path):
    lib = _md_library(tmp_path)
    f = tmp_path / "set.txt"
    f.write_text(f"# a comment line\n\n{ZIB}\n  # indented comment\n")
    assert cli.main(["kb", "build", "--docs-file", str(f), "--name", "c", "--library",
                     str(lib), "--output-root", str(tmp_path), "--llm", "stub",
                     "--embedder", "hash"]) == 0
    assert json.loads((tmp_path / "kb" / "c" / "manifest.json").read_text())["skipped"] == []
