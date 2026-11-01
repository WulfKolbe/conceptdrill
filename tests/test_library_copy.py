"""A copy of each document's plain results goes into its library folder.

Owner, 2026-11-01 (T-0117): "we need at least a copy of the plain results in
the folder of each document in pdfdrill-library". The contract:

* only `<library>/<doc>/conceptdrill/<run>/` is written: `<doc>.json`,
  `<doc>.md` and `provenance.json`, nothing else in the document folder;
* a document without a library folder is listed and skipped, never created;
* copying the same run twice gives byte-identical files;
* `provenance.json` says what made the files: command, commit, run id, the
  inputs with their sha256, and when -- including, plainly, when the producer
  was not conceptdrill at all.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from conceptdrill.hierarchy.libcopy import (MANIFEST, copy_to_library,
                                            write_manifest)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    library = tmp_path / "library"
    for doc in ("paperA", "paperB"):
        (library / doc).mkdir(parents=True)
        (library / doc / f"{doc}.glyphs.md").write_text(f"# {doc}\n")
        (library / doc / "model.docmodel.json").write_text("{}")
    run = tmp_path / "output" / "run-1"
    run.mkdir(parents=True)
    for doc in ("paperA", "paperB", "nowhere"):
        (run / f"{doc}.json").write_text(json.dumps({"doc": doc}))
        (run / f"{doc}.md").write_text(f"# {doc}\n")
    write_manifest(run, run_id="run-1", command="conceptdrill sections paperA",
                   commit="abc1234",
                   documents={d: [library / d / f"{d}.glyphs.md"]
                              for d in ("paperA", "paperB")}
                   | {"nowhere": []})
    return library, run


def test_only_the_run_subfolder_is_written(setup):
    library, run = setup
    before = {p for p in library.rglob("*")}
    report = copy_to_library(run, library)
    assert sorted(report["copied"]) == ["paperA", "paperB"]
    added = {p for p in library.rglob("*")} - before
    for doc in ("paperA", "paperB"):
        dest = library / doc / "conceptdrill" / "run-1"
        assert {p for p in added if p.parent == dest} == {
            dest / f"{doc}.json", dest / f"{doc}.md", dest / "provenance.json"}
        assert (dest / f"{doc}.json").read_bytes() == (run / f"{doc}.json").read_bytes()
        assert (dest / f"{doc}.md").read_bytes() == (run / f"{doc}.md").read_bytes()
    assert all(p.is_dir() or p.parent.name == "run-1" for p in added)


def test_a_document_without_a_library_folder_is_listed_and_skipped(setup):
    library, run = setup
    report = copy_to_library(run, library)
    assert [s["doc"] for s in report["skipped"]] == ["nowhere"]
    assert not (library / "nowhere").exists()


def test_copying_twice_gives_identical_files(setup):
    library, run = setup
    copy_to_library(run, library)
    dest = library / "paperA" / "conceptdrill" / "run-1"
    first = {p.name: p.read_bytes() for p in dest.iterdir()}
    copy_to_library(run, library)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == first


def test_provenance_names_the_run_its_command_commit_and_inputs(setup):
    library, run = setup
    copy_to_library(run, library)
    dest = library / "paperA" / "conceptdrill" / "run-1"
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov["run_id"] == "run-1"
    assert prov["command"] == "conceptdrill sections paperA"
    assert prov["conceptdrill_commit"] == "abc1234"
    assert prov["produced_by"] == "conceptdrill"
    glyphs = library / "paperA" / "paperA.glyphs.md"
    assert prov["inputs"] == [{"path": str(glyphs), "sha256": _sha(glyphs)}]
    assert {f["name"]: f["sha256"] for f in prov["files"]} == {
        "paperA.json": _sha(dest / "paperA.json"),
        "paperA.md": _sha(dest / "paperA.md")}
    assert prov["written_at"]


def test_a_run_not_made_by_conceptdrill_says_so(tmp_path):
    library = tmp_path / "library"
    (library / "paperA").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    (run / "paperA.json").write_text("{}")
    (run / "paperA.md").write_text("x")
    script = tmp_path / "scratch.py"
    script.write_text("print(1)")
    write_manifest(run, run_id="labeling-21", command="python3 scratch.py run",
                   commit=None, documents={"paperA": []},
                   produced_by="a scratch script, not conceptdrill",
                   producer={"path": str(script), "sha256": _sha(script)})
    copy_to_library(run, library)
    prov = json.loads((library / "paperA" / "conceptdrill" / "labeling-21"
                       / "provenance.json").read_text())
    assert prov["produced_by"] == "a scratch script, not conceptdrill"
    assert prov["producer"] == {"path": str(script), "sha256": _sha(script)}
    assert prov["conceptdrill_commit"] is None


def test_a_run_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match=MANIFEST):
        copy_to_library(tmp_path / "run", tmp_path)
