"""`kb reach`: can the reading protocol get to each payload item? (T-0130)

Deterministic, no model. A payload item's method name is the title of the
section it was placed in, without its numbering. The item is reachable when
an index entry names that method -- the entry holds at least half of the
title's content words -- and points at a leaf that carries the item: INDEX ->
letter file -> leaf -> payload, inside the protocol's reads.
"""
from __future__ import annotations

import json

from conceptdrill import cli
from conceptdrill.kb.reach import content_words, method_name, names, reach
from conceptdrill.output import REPO_ROOT

LIB = REPO_ROOT / "tests" / "fixtures" / "kb-synthetic"


def test_the_method_name_is_the_section_title_without_its_number():
    assert method_name("2.1. Zent Steps") == "Zent Steps"
    assert method_name("III. PROPOSED ALGORITHM") == "PROPOSED ALGORITHM"
    assert method_name("Abstract") == "Abstract"


def test_generic_titles_name_no_method():
    for title in ("Page 3", "Results", "Experimental Results", "Introduction",
                  "Abstract", "Conclusion", "Method"):
        assert content_words(method_name(title)) == set(), title


def test_an_entry_names_a_method_with_half_its_content_words():
    assert names("zent steps of the florp", "Zent Steps")
    assert names("Zent quibbling", "Zent Steps")          # 1 of 2: half
    assert not names("florp rate", "Zent Steps")
    assert not names("anything", "Page 3")                # nothing to name


def _kb(concepts, leaf_payload):
    return {"sections": [{"id": "d#1", "title": "2. Zent Steps"}],
            "payload": [{"id": "d#obj_1", "kind": "pseudocode", "section": "d#1",
                         "status": "placed"}],
            "leaves": [{"id": "t/l", "payload": leaf_payload}],
            "concepts": [{"name": c, "leaf": "t/l"} for c in concepts]}


def test_reachable_when_an_entry_names_it_and_points_at_its_leaf():
    r = reach(_kb(["Zent steps"], ["d#obj_1"]))
    assert r["reachable"] == 1 and r["items"] == 1 and r["share"] == 1.0


def test_unreachable_when_the_naming_entry_points_at_a_leaf_without_it():
    r = reach(_kb(["Zent steps"], []))
    assert r["reachable"] == 0
    assert r["rows"][0]["reason"] == "no leaf carrying it is indexed under its method"


def test_unreachable_when_no_entry_names_the_method():
    r = reach(_kb(["florp rate"], ["d#obj_1"]))
    assert r["rows"][0]["reason"] == "no index entry names its method"


def test_the_stubbed_synthetic_build_reaches_one_item_by_a_generated_entry(tmp_path, capsys):
    """Stub concept names are CONCEPT-n and name nothing. The one item reached
    is the equation of chapter 2, through the entry generated for the
    pseudocode of 2.1 ("Quibbling of Florps - pseudocode"): it names the
    equation's own section -- the circular kind of reach that T-0130's second
    measure exists to expose."""
    assert cli.main(["kb", "build", "Florp Dynamics (v2)", "--name", "s", "--library",
                     str(LIB), "--output-root", str(tmp_path), "--llm", "stub",
                     "--embedder", "hash"]) == 0
    capsys.readouterr()
    assert cli.main(["kb", "reach", str(tmp_path / "kb" / "s")]) == 0
    out = capsys.readouterr().out
    assert "reach: 1 of 3 payload items" in out


# --------------------------------------------------------------------------
# A second, independent query source: the paper's title (coordinator, T-0130)
# --------------------------------------------------------------------------

def _kb2(concepts, leaf_payload, doc_title="Quorp Flux in Dense Marls"):
    kb = _kb(concepts, leaf_payload)
    kb["documents"] = [{"doc": "d", "title": doc_title}]
    kb["payload"][0]["doc"] = "d"
    return kb


def test_a_paper_is_reached_when_an_entry_naming_its_title_carries_its_payload():
    from conceptdrill.kb.reach import reach_by_title
    r = reach_by_title(_kb2(["quorp flux"], ["d#obj_1"]))
    assert (r["documents"], r["reachable"]) == (1, 1)


def test_a_paper_is_not_reached_by_an_entry_that_names_something_else():
    from conceptdrill.kb.reach import reach_by_title
    r = reach_by_title(_kb2(["Zent steps"], ["d#obj_1"]))
    assert r["reachable"] == 0 and r["rows"][0]["reason"] == "no index entry names the paper"


def test_both_measures_are_reported(tmp_path, capsys):
    assert cli.main(["kb", "build", "Florp Dynamics (v2)", "--name", "s", "--library",
                     str(LIB), "--output-root", str(tmp_path), "--llm", "stub",
                     "--embedder", "hash"]) == 0
    capsys.readouterr()
    cli.main(["kb", "reach", str(tmp_path / "kb" / "s")])
    out = capsys.readouterr().out
    assert "reach: 1 of 3 payload items" in out
    # "Florp Dynamics" has two content words; the chapter-named entry holds
    # one of them, half: the paper-title measure is weak for short titles.
    assert "reach by paper title: 1 of 1 papers with payload" in out
