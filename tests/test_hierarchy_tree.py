"""Unit tests for the docmodel section tree, one unit at a time.

`REAL_DOCMODEL` is the actual drilled CES paper. Tests that use it are marked
so the suite still runs on a machine without the library.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptdrill.hierarchy.docmodel_tree import (Paragraph, RawSection,
                                                  SectionNode,
                                                  attach_paragraphs,
                                                  link_parents,
                                                  read_paragraphs,
                                                  read_sections,
                                                  build_tree, load_tree,
                                                  is_latex_artifact)

REAL_DOCMODEL = Path(
    "/home/wkolbe/pdfdrill-library/2209.00445/model.docmodel.json")

needs_real = pytest.mark.skipif(
    not REAL_DOCMODEL.exists(), reason="drilled library not present")


def section(oid: str, caption: str, level: int, flow: int, **props) -> dict:
    """A minimal DocModel Section object, shaped like the real thing."""
    return {"id": oid, "type": "Section",
            "props": {"caption": caption, "level": level,
                      "flow_index": flow, **props},
            "realizations": [], "children": [], "parent": None}


# --------------------------------------------------------------------------
# read_sections
# --------------------------------------------------------------------------

def test_no_objects_yields_no_sections():
    assert read_sections([]) == []


def test_non_section_objects_are_ignored():
    objs = [{"id": "p1", "type": "Paragraph", "props": {"text": "hi"}},
            section("s1", "Method", 2, 5)]
    assert [s.id for s in read_sections(objs)] == ["s1"]


def test_caption_is_used_as_the_title():
    """The DocModel stores titles under props.caption, not props.title."""
    assert read_sections([section("s1", "Introduction", 2, 3)])[0].title == \
        "Introduction"


def test_title_prop_is_accepted_as_a_fallback():
    objs = [{"id": "s1", "type": "Section",
             "props": {"title": "Method", "level": 2, "flow_index": 1}}]
    assert read_sections(objs)[0].title == "Method"


def test_sections_come_back_in_flow_index_order():
    objs = [section("s3", "Third", 2, 90),
            section("s1", "First", 2, 3),
            section("s2", "Second", 2, 11)]
    assert [s.id for s in read_sections(objs)] == ["s1", "s2", "s3"]


def test_ordering_does_not_depend_on_input_order():
    a = [section("s1", "A", 2, 3), section("s2", "B", 2, 11)]
    assert [s.id for s in read_sections(a)] == \
           [s.id for s in read_sections(list(reversed(a)))]


def test_flow_index_ties_break_on_id():
    objs = [section("sb", "B", 2, 7), section("sa", "A", 2, 7)]
    assert [s.id for s in read_sections(objs)] == ["sa", "sb"]


def test_missing_flow_index_falls_back_to_position():
    objs = [{"id": "s1", "type": "Section", "props": {"caption": "A", "level": 2}},
            {"id": "s2", "type": "Section", "props": {"caption": "B", "level": 2}}]
    assert [s.id for s in read_sections(objs)] == ["s1", "s2"]


def test_object_without_an_id_is_skipped():
    """An unidentifiable section cannot be linked to paragraphs later."""
    objs = [{"id": "", "type": "Section", "props": {"caption": "X", "level": 2}},
            section("s1", "Real", 2, 1)]
    assert [s.id for s in read_sections(objs)] == ["s1"]


def test_appendix_flag_is_carried():
    objs = [section("s1", "Body", 2, 1),
            section("s2", "Extra", 2, 9, is_appendix=True)]
    got = {s.id: s.is_appendix for s in read_sections(objs)}
    assert got == {"s1": False, "s2": True}


def test_level_is_preserved_and_may_start_at_two():
    """Real documents start at level 2; nothing may assume 1."""
    assert read_sections([section("s1", "Top", 2, 1)])[0].level == 2


def test_malformed_level_degrades_rather_than_raising():
    objs = [{"id": "s1", "type": "Section",
             "props": {"caption": "X", "level": "not-a-number", "flow_index": 1}}]
    assert read_sections(objs)[0].level == 1


def test_malformed_props_do_not_raise():
    objs = [{"id": "s1", "type": "Section", "props": "not-a-dict"}]
    assert read_sections(objs)[0].title == ""


def test_non_dict_objects_are_skipped():
    assert read_sections(["nonsense", None, section("s1", "A", 2, 1)])[0].id == "s1"


# --------------------------------------------------------------------------
# Caption cleaning is applied, and loss is recorded
# --------------------------------------------------------------------------

def test_latex_is_cleaned_out_of_titles():
    got = read_sections([section("s1", "\\emph{Siblings} score", 2, 1)])[0]
    assert got.title == "Siblings score"
    assert got.title_raw == "\\emph{Siblings} score"


def test_raw_title_is_always_retained():
    """The summariser needs the raw form when cleaning lost information."""
    got = read_sections([section("s1", "\\ALG\\ Application", 2, 1)])[0]
    assert got.title == "Application"
    assert got.title_raw == "\\ALG\\ Application"


def test_degraded_title_is_flagged():
    got = read_sections([section("s1", "\\ALG\\ Application", 2, 1)])[0]
    assert got.title_is_degraded
    assert "ALG" in got.lost_macros


def test_clean_title_is_not_flagged_as_degraded():
    assert not read_sections([section("s1", "Introduction", 2, 1)])[0].title_is_degraded


# --------------------------------------------------------------------------
# link_parents
# --------------------------------------------------------------------------

def _linked(*specs):
    """specs: (id, level) or (id, level, is_appendix). Flow order = given order."""
    secs = read_sections([
        section(sid, sid.upper(), lvl, i,
                **({"is_appendix": True} if len(spec) > 2 and spec[2] else {}))
        for i, spec in enumerate(specs)
        for sid, lvl in [(spec[0], spec[1])]
    ])
    return link_parents(secs)


def test_no_sections_yields_no_links():
    assert link_parents([]) == {}


def test_a_lone_section_is_a_root():
    assert _linked(("a", 2)) == {"a": None}


def test_subsection_hangs_off_its_section():
    assert _linked(("a", 2), ("b", 3)) == {"a": None, "b": "a"}


def test_siblings_share_a_parent():
    got = _linked(("a", 2), ("b", 3), ("c", 3))
    assert got["b"] == "a" and got["c"] == "a"


def test_three_levels_nest():
    got = _linked(("a", 2), ("b", 3), ("c", 4))
    assert got == {"a": None, "b": "a", "c": "b"}


def test_returning_to_a_higher_level_starts_a_new_branch():
    got = _linked(("a", 2), ("b", 3), ("c", 4), ("d", 2), ("e", 3))
    assert got["d"] is None
    assert got["e"] == "d"


def test_level_jump_attaches_to_the_nearest_lower_level():
    """L2 -> L4 with no L3: the L4 belongs under the L2, not at the root."""
    assert _linked(("a", 2), ("b", 4)) == {"a": None, "b": "a"}


def test_a_deeper_first_section_is_still_a_root():
    """A document opening at L3 has no L2 to hang from."""
    got = _linked(("a", 3), ("b", 2))
    assert got == {"a": None, "b": None}


def test_equal_levels_never_nest():
    got = _linked(("a", 2), ("b", 2), ("c", 2))
    assert all(v is None for v in got.values())


def test_appendix_does_not_hang_off_the_last_body_section():
    """Otherwise an appendix subsection grafts onto 'Ethics Statement'."""
    got = _linked(("body", 2), ("app", 3, True))
    assert got["app"] is None


def test_appendix_nests_within_itself():
    got = _linked(("body", 2), ("app", 2, True), ("appsub", 3, True))
    assert got["app"] is None
    assert got["appsub"] == "app"


def test_every_section_gets_an_entry():
    got = _linked(("a", 2), ("b", 3), ("c", 4), ("d", 2))
    assert set(got) == {"a", "b", "c", "d"}


def test_no_section_is_its_own_parent():
    got = _linked(("a", 2), ("b", 3), ("c", 3), ("d", 2))
    assert all(k != v for k, v in got.items())


def test_links_form_an_acyclic_tree():
    got = _linked(("a", 2), ("b", 3), ("c", 4), ("d", 3), ("e", 2))
    for start in got:
        seen, cur = {start}, got[start]
        while cur is not None:
            assert cur not in seen, "cycle"
            seen.add(cur)
            cur = got[cur]


# --------------------------------------------------------------------------
# read_paragraphs / attach_paragraphs
# --------------------------------------------------------------------------

def para(pid: str, text: str, flow: int, parent=None) -> dict:
    props = {"text": text, "flow_index": flow}
    if parent:
        props["parent_section"] = parent
    return {"id": pid, "type": "Paragraph", "props": props}


def test_no_objects_yields_no_paragraphs():
    assert read_paragraphs([]) == []


def test_paragraph_text_and_owner_are_read():
    got = read_paragraphs([para("p1", "hello", 4, "s1")])[0]
    assert got.text == "hello" and got.section_id == "s1"


def test_paragraphs_come_back_in_flow_order():
    objs = [para("p2", "b", 9, "s1"), para("p1", "a", 4, "s1")]
    assert [p.id for p in read_paragraphs(objs)] == ["p1", "p2"]


def test_empty_paragraphs_are_dropped():
    """Whitespace-only text contributes nothing to a summary."""
    assert read_paragraphs([para("p1", "   ", 1, "s1")]) == []


def test_paragraph_without_a_parent_section_keeps_none():
    assert read_paragraphs([para("p1", "front matter", 1)])[0].section_id is None


def test_listitem_and_abstract_count_as_body_text():
    objs = [{"id": "a1", "type": "Abstract",
             "props": {"text": "we present", "flow_index": 0}},
            {"id": "l1", "type": "ListItem",
             "props": {"content": "an item", "flow_index": 2}}]
    assert {p.id for p in read_paragraphs(objs)} == {"a1", "l1"}


def test_sections_are_not_read_as_paragraphs():
    assert read_paragraphs([section("s1", "Method", 2, 1)]) == []


def test_attach_groups_by_section():
    paras = read_paragraphs([para("p1", "a", 1, "s1"), para("p2", "b", 2, "s2"),
                             para("p3", "c", 3, "s1")])
    by, orphans = attach_paragraphs(paras, ["s1", "s2"])
    assert [p.id for p in by["s1"]] == ["p1", "p3"]
    assert [p.id for p in by["s2"]] == ["p2"]
    assert orphans == []


def test_attach_preserves_flow_order_within_a_section():
    paras = read_paragraphs([para("p2", "b", 9, "s1"), para("p1", "a", 4, "s1")])
    by, _ = attach_paragraphs(paras, ["s1"])
    assert [p.id for p in by["s1"]] == ["p1", "p2"]


def test_every_known_section_gets_a_key_even_when_empty():
    by, _ = attach_paragraphs([], ["s1", "s2"])
    assert by == {"s1": [], "s2": []}


def test_parentless_paragraph_becomes_an_orphan_not_a_loss():
    """Front matter precedes the first section; it must be accounted for."""
    paras = read_paragraphs([para("p1", "title block", 1)])
    by, orphans = attach_paragraphs(paras, ["s1"])
    assert [p.id for p in orphans] == ["p1"]


def test_paragraph_naming_an_unknown_section_is_an_orphan():
    paras = read_paragraphs([para("p1", "x", 1, "ghost")])
    _, orphans = attach_paragraphs(paras, ["s1"])
    assert [p.id for p in orphans] == ["p1"]


def test_no_paragraph_is_ever_dropped():
    paras = read_paragraphs([para("p1", "a", 1, "s1"), para("p2", "b", 2),
                             para("p3", "c", 3, "ghost")])
    by, orphans = attach_paragraphs(paras, ["s1"])
    assert sum(len(v) for v in by.values()) + len(orphans) == len(paras)


# --------------------------------------------------------------------------
# Against the real drilled document
# --------------------------------------------------------------------------

@needs_real
def test_real_document_yields_the_expected_sections():
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    secs = read_sections(objs)
    assert len(secs) == 24
    assert secs[0].title == "Introduction"
    assert secs[0].level == 2


@needs_real
def test_real_document_levels_and_appendix_counts():
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    secs = read_sections(objs)
    levels = {}
    for s in secs:
        levels[s.level] = levels.get(s.level, 0) + 1
    # 14 = 8 body sections + 6 appendix sections, all at level 2.
    assert levels == {2: 14, 3: 8, 4: 2}
    assert sum(levels.values()) == 24
    assert sum(1 for s in secs if s.is_appendix) == 6


@needs_real
def test_no_real_title_retains_latex_markup():
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    for s in read_sections(objs):
        assert "\\" not in s.title, s.title_raw
        assert "$" not in s.title, s.title_raw


@needs_real
def test_exactly_one_real_title_is_degraded():
    """Only \\ALG\\ Application loses meaning; if this count moves, the
    cleaner's behaviour changed."""
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    degraded = [s for s in read_sections(objs) if s.title_is_degraded]
    assert [s.title_raw for s in degraded] == ["\\ALG\\ Application"]


# --------------------------------------------------------------------------
# build_tree — integration
# --------------------------------------------------------------------------

def _doc(*objects, bibkey="test"):
    return {"meta": {"bibkey": bibkey}, "objects": list(objects)}


def test_empty_document_yields_an_empty_tree():
    tree = build_tree(_doc())
    assert len(tree) == 0 and tree.roots == () and tree.orphans == ()


def test_tree_links_parents_and_children():
    tree = build_tree(_doc(section("s1", "Method", 2, 1),
                           section("s2", "Sub", 3, 2)))
    assert tree.nodes["s2"].parent_id == "s1"
    assert tree.nodes["s1"].children == ("s2",)
    assert tree.roots == ("s1",)


def test_body_text_holds_only_a_section_own_paragraphs():
    tree = build_tree(_doc(section("s1", "Method", 2, 1),
                           para("p1", "own text", 2, "s1"),
                           section("s2", "Sub", 3, 3),
                           para("p2", "sub text", 4, "s2")))
    assert tree.nodes["s1"].body_text == "own text"
    assert "sub text" not in tree.nodes["s1"].body_text


def test_subtree_text_includes_descendants():
    """A level-2 summary needs its subsections, or it describes nothing."""
    tree = build_tree(_doc(section("s1", "Method", 2, 1),
                           para("p1", "own text", 2, "s1"),
                           section("s2", "Sub", 3, 3),
                           para("p2", "sub text", 4, "s2")))
    got = tree.subtree_text("s1")
    assert "own text" in got and "sub text" in got


def test_subtree_text_of_a_leaf_is_its_body():
    tree = build_tree(_doc(section("s1", "Only", 2, 1),
                           para("p1", "text", 2, "s1")))
    assert tree.subtree_text("s1") == "text"


def test_depth_reflects_nesting():
    tree = build_tree(_doc(section("s1", "A", 2, 1), section("s2", "B", 3, 2),
                           section("s3", "C", 4, 3)))
    assert (tree.depth("s1"), tree.depth("s2"), tree.depth("s3")) == (0, 1, 2)


def test_descendants_are_transitive():
    tree = build_tree(_doc(section("s1", "A", 2, 1), section("s2", "B", 3, 2),
                           section("s3", "C", 4, 3)))
    assert [n.id for n in tree.descendants("s1")] == ["s2", "s3"]


def test_by_level_filters_in_document_order():
    tree = build_tree(_doc(section("s1", "A", 2, 1), section("s2", "B", 3, 2),
                           section("s3", "C", 2, 3)))
    assert [n.id for n in tree.by_level(2)] == ["s1", "s3"]


def test_orphan_paragraphs_are_kept_on_the_tree():
    tree = build_tree(_doc(section("s1", "A", 2, 5),
                           para("p0", "front matter", 1)))
    assert [p.id for p in tree.orphans] == ["p0"]


def test_every_paragraph_is_accounted_for():
    tree = build_tree(_doc(section("s1", "A", 2, 2),
                           para("p0", "front", 1),
                           para("p1", "body", 3, "s1")))
    placed = sum(len(n.paragraphs) for n in tree.nodes.values())
    assert placed + len(tree.orphans) == 2


def test_summarizer_title_restores_a_degraded_title():
    tree = build_tree(_doc(section("s1", "\\ALG\\ Application", 2, 1)))
    assert tree.nodes["s1"].summarizer_title == "Application (\\ALG\\ Application)"


def test_summarizer_title_is_plain_when_nothing_was_lost():
    tree = build_tree(_doc(section("s1", "Introduction", 2, 1)))
    assert tree.nodes["s1"].summarizer_title == "Introduction"


def test_bibkey_is_carried_from_meta():
    assert build_tree(_doc(bibkey="2209.00445")).bibkey == "2209.00445"


def test_malformed_objects_list_does_not_raise():
    assert len(build_tree({"objects": "nonsense"})) == 0


def test_cyclic_parents_do_not_hang_depth():
    """Defensive: a malformed tree must not spin forever."""
    tree = build_tree(_doc(section("s1", "A", 2, 1)))
    tree.nodes["s1"] = SectionNode(
        id="s1", title="A", title_raw="A", level=2, flow_index=1,
        is_appendix=False, lost_macros=(), parent_id="s1", children=(),
        paragraphs=())
    assert tree.depth("s1") >= 0


# --------------------------------------------------------------------------
# build_tree against the real document
# --------------------------------------------------------------------------

@needs_real
def test_real_tree_matches_the_printed_outline():
    tree = load_tree(REAL_DOCMODEL)
    stats = tree.stats()
    assert stats["sections"] == 24
    assert stats["levels"] == {2: 14, 3: 8, 4: 2}
    assert stats["roots"] == 14
    assert stats["appendix_sections"] == 6


@needs_real
def test_real_tree_nests_the_known_subsections():
    tree = load_tree(REAL_DOCMODEL)
    by_title = {n.title: n for n in tree.nodes.values()}
    assert by_title["Generating Conceptual Spaces"].parent_id == \
        by_title["The Conceptualization Algorithm"].id
    assert by_title["Evaluation By Humans"].parent_id == \
        by_title["Evaluating Understandability"].id


@needs_real
def test_real_tree_accounts_for_every_paragraph():
    """84 of 85 paragraphs carry parent_section; the 85th is front matter."""
    tree = load_tree(REAL_DOCMODEL)
    stats = tree.stats()
    assert stats["paragraphs"] == 84
    # One orphan remains: the Abstract, which legitimately precedes the first
    # section. The other candidate was a Paragraph whose entire text is
    # `\\maketitle`, dropped as a LaTeX artifact.
    assert stats["orphan_paragraphs"] == 1
    assert "computational interpretation" in tree.orphans[0].text


@needs_real
def test_real_subtree_text_is_larger_than_body_text():
    tree = load_tree(REAL_DOCMODEL)
    by_title = {n.title: n for n in tree.nodes.values()}
    node = by_title["The Conceptualization Algorithm"]
    assert len(tree.subtree_text(node.id)) > len(node.body_text)


@needs_real
def test_real_tree_reports_one_degraded_title():
    assert load_tree(REAL_DOCMODEL).stats()["degraded_titles"] == 1


# --------------------------------------------------------------------------
# is_latex_artifact
# --------------------------------------------------------------------------

def test_maketitle_is_an_artifact():
    """The real document has a Paragraph whose whole text is \\maketitle."""
    assert is_latex_artifact("\\maketitle")


def test_argumentless_commands_are_artifacts():
    assert is_latex_artifact("  \\clearpage  ")
    assert is_latex_artifact("\\maketitle \\clearpage")


def test_a_command_whose_argument_holds_words_is_kept():
    """Deliberately conservative. `\\input{preamble}` is arguably an artifact,
    but the argument could equally be real text, and dropping content is worse
    than keeping a little noise. Only wordless markup is discarded."""
    assert not is_latex_artifact("\\input{preamble}")


def test_empty_text_is_an_artifact():
    assert is_latex_artifact("") and is_latex_artifact("   ")


def test_real_prose_is_not_an_artifact():
    assert not is_latex_artifact("We present a method for semantic projection.")


def test_prose_containing_a_command_is_kept():
    """Only pure markup is dropped; content with markup in it is content."""
    assert not is_latex_artifact(
        "The \\emph{siblings} score measures conceptual distance.")


def test_artifact_paragraphs_never_reach_the_tree():
    tree = build_tree(_doc(section("s1", "A", 2, 1),
                           para("p1", "\\maketitle", 2, "s1"),
                           para("p2", "real content here", 3, "s1")))
    assert [p.id for p in tree.nodes["s1"].paragraphs] == ["p2"]
