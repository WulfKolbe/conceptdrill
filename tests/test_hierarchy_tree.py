"""Unit tests for the docmodel span tree, one unit at a time.

`REAL_DOCMODEL` is the actual drilled CES paper. Tests that use it are marked
so the suite still runs on a machine without the library.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptdrill.hierarchy.docmodel_tree import (Paragraph, SectionMarker,
                                                  MarkerNode,
                                                  attach_paragraphs,
                                                  link_parents,
                                                  read_paragraphs,
                                                  read_markers,
                                                  build_tree, load_tree,
                                                  is_latex_artifact)

REAL_DOCMODEL = Path(
    "/home/wkolbe/pdfdrill-library/2209.00445/model.docmodel.json")

needs_real = pytest.mark.skipif(
    not REAL_DOCMODEL.exists(), reason="drilled library not present")


def span(oid: str, caption: str, level: int, flow: int, **props) -> dict:
    """A minimal DocModel Section object, shaped like the real thing."""
    return {"id": oid, "type": "Section",
            "props": {"caption": caption, "level": level,
                      "flow_index": flow, **props},
            "realizations": [], "children": [], "parent": None}


# --------------------------------------------------------------------------
# read_markers
# --------------------------------------------------------------------------

def test_no_objects_yields_no_sections():
    assert read_markers([]) == []


def test_non_section_objects_are_ignored():
    objs = [{"id": "p1", "type": "Paragraph", "props": {"text": "hi"}},
            span("s1", "Method", 2, 5)]
    assert [s.id for s in read_markers(objs)] == ["s1"]


def test_caption_is_used_as_the_title():
    """The DocModel stores titles under props.caption, not props.title."""
    assert read_markers([span("s1", "Introduction", 2, 3)])[0].title == \
        "Introduction"


def test_title_prop_is_accepted_as_a_fallback():
    objs = [{"id": "s1", "type": "Section",
             "props": {"title": "Method", "level": 2, "flow_index": 1}}]
    assert read_markers(objs)[0].title == "Method"


def test_sections_come_back_in_flow_index_order():
    objs = [span("s3", "Third", 2, 90),
            span("s1", "First", 2, 3),
            span("s2", "Second", 2, 11)]
    assert [s.id for s in read_markers(objs)] == ["s1", "s2", "s3"]


def test_ordering_does_not_depend_on_input_order():
    a = [span("s1", "A", 2, 3), span("s2", "B", 2, 11)]
    assert [s.id for s in read_markers(a)] == \
           [s.id for s in read_markers(list(reversed(a)))]


def test_flow_index_ties_break_on_id():
    objs = [span("sb", "B", 2, 7), span("sa", "A", 2, 7)]
    assert [s.id for s in read_markers(objs)] == ["sa", "sb"]


def test_missing_flow_index_falls_back_to_position():
    objs = [{"id": "s1", "type": "Section", "props": {"caption": "A", "level": 2}},
            {"id": "s2", "type": "Section", "props": {"caption": "B", "level": 2}}]
    assert [s.id for s in read_markers(objs)] == ["s1", "s2"]


def test_object_without_an_id_is_skipped():
    """An unidentifiable span cannot be linked to paragraphs later."""
    objs = [{"id": "", "type": "Section", "props": {"caption": "X", "level": 2}},
            span("s1", "Real", 2, 1)]
    assert [s.id for s in read_markers(objs)] == ["s1"]


def test_appendix_flag_is_carried():
    objs = [span("s1", "Body", 2, 1),
            span("s2", "Extra", 2, 9, is_appendix=True)]
    got = {s.id: s.is_appendix for s in read_markers(objs)}
    assert got == {"s1": False, "s2": True}


def test_level_is_preserved_and_may_start_at_two():
    """Real documents start at level 2; nothing may assume 1."""
    assert read_markers([span("s1", "Top", 2, 1)])[0].level == 2


def test_malformed_level_degrades_rather_than_raising():
    objs = [{"id": "s1", "type": "Section",
             "props": {"caption": "X", "level": "not-a-number", "flow_index": 1}}]
    assert read_markers(objs)[0].level == 1


def test_malformed_props_do_not_raise():
    objs = [{"id": "s1", "type": "Section", "props": "not-a-dict"}]
    assert read_markers(objs)[0].title == ""


def test_non_dict_objects_are_skipped():
    assert read_markers(["nonsense", None, span("s1", "A", 2, 1)])[0].id == "s1"


# --------------------------------------------------------------------------
# Caption cleaning is applied, and loss is recorded
# --------------------------------------------------------------------------

def test_latex_is_cleaned_out_of_titles():
    got = read_markers([span("s1", "\\emph{Siblings} score", 2, 1)])[0]
    assert got.title == "Siblings score"
    assert got.title_raw == "\\emph{Siblings} score"


def test_raw_title_is_always_retained():
    """The summariser needs the raw form when cleaning lost information."""
    got = read_markers([span("s1", "\\ALG\\ Application", 2, 1)])[0]
    assert got.title == "Application"
    assert got.title_raw == "\\ALG\\ Application"


def test_degraded_title_is_flagged():
    got = read_markers([span("s1", "\\ALG\\ Application", 2, 1)])[0]
    assert got.title_is_degraded
    assert "ALG" in got.lost_macros


def test_clean_title_is_not_flagged_as_degraded():
    assert not read_markers([span("s1", "Introduction", 2, 1)])[0].title_is_degraded


# --------------------------------------------------------------------------
# link_parents
# --------------------------------------------------------------------------

def _linked(*specs):
    """specs: (id, level) or (id, level, is_appendix). Flow order = given order."""
    secs = read_markers([
        span(sid, sid.upper(), lvl, i,
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
    assert got.text == "hello" and got.marker_id == "s1"


def test_paragraphs_come_back_in_flow_order():
    objs = [para("p2", "b", 9, "s1"), para("p1", "a", 4, "s1")]
    assert [p.id for p in read_paragraphs(objs)] == ["p1", "p2"]


def test_empty_paragraphs_are_dropped():
    """Whitespace-only text contributes nothing to a summary."""
    assert read_paragraphs([para("p1", "   ", 1, "s1")]) == []


def test_paragraph_without_a_parent_section_keeps_none():
    assert read_paragraphs([para("p1", "front matter", 1)])[0].marker_id is None


def test_listitem_and_abstract_count_as_body_text():
    objs = [{"id": "a1", "type": "Abstract",
             "props": {"text": "we present", "flow_index": 0}},
            {"id": "l1", "type": "ListItem",
             "props": {"content": "an item", "flow_index": 2}}]
    assert {p.id for p in read_paragraphs(objs)} == {"a1", "l1"}


def test_sections_are_not_read_as_paragraphs():
    assert read_paragraphs([span("s1", "Method", 2, 1)]) == []


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
    """Front matter precedes the first span; it must be accounted for."""
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
    secs = read_markers(objs)
    assert len(secs) == 24
    assert secs[0].title == "Introduction"
    assert secs[0].level == 2


@needs_real
def test_real_document_levels_and_appendix_counts():
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    secs = read_markers(objs)
    levels = {}
    for s in secs:
        levels[s.level] = levels.get(s.level, 0) + 1
    # 14 = 8 body spans + 6 appendix spans, all at level 2.
    assert levels == {2: 14, 3: 8, 4: 2}
    assert sum(levels.values()) == 24
    assert sum(1 for s in secs if s.is_appendix) == 6


@needs_real
def test_no_real_title_retains_latex_markup():
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    for s in read_markers(objs):
        assert "\\" not in s.title, s.title_raw
        assert "$" not in s.title, s.title_raw


@needs_real
def test_exactly_one_real_title_is_degraded():
    """Only \\ALG\\ Application loses meaning; if this count moves, the
    cleaner's behaviour changed."""
    objs = json.loads(REAL_DOCMODEL.read_text())["objects"]
    degraded = [s for s in read_markers(objs) if s.title_is_degraded]
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
    tree = build_tree(_doc(span("s1", "Method", 2, 1),
                           span("s2", "Sub", 3, 2)))
    assert tree.nodes["s2"].parent_id == "s1"
    assert tree.nodes["s1"].children == ("s2",)
    assert tree.roots == ("s1",)


def test_body_text_holds_only_a_section_own_paragraphs():
    tree = build_tree(_doc(span("s1", "Method", 2, 1),
                           para("p1", "own text", 2, "s1"),
                           span("s2", "Sub", 3, 3),
                           para("p2", "sub text", 4, "s2")))
    assert tree.nodes["s1"].body_text == "own text"
    assert "sub text" not in tree.nodes["s1"].body_text


def test_subtree_text_includes_descendants():
    """A level-2 summary needs its subsections, or it describes nothing."""
    tree = build_tree(_doc(span("s1", "Method", 2, 1),
                           para("p1", "own text", 2, "s1"),
                           span("s2", "Sub", 3, 3),
                           para("p2", "sub text", 4, "s2")))
    got = tree.subtree_text("s1")
    assert "own text" in got and "sub text" in got


def test_subtree_text_of_a_leaf_is_its_body():
    tree = build_tree(_doc(span("s1", "Only", 2, 1),
                           para("p1", "text", 2, "s1")))
    assert tree.subtree_text("s1") == "text"


def test_depth_reflects_nesting():
    tree = build_tree(_doc(span("s1", "A", 2, 1), span("s2", "B", 3, 2),
                           span("s3", "C", 4, 3)))
    assert (tree.depth("s1"), tree.depth("s2"), tree.depth("s3")) == (0, 1, 2)


def test_descendants_are_transitive():
    tree = build_tree(_doc(span("s1", "A", 2, 1), span("s2", "B", 3, 2),
                           span("s3", "C", 4, 3)))
    assert [n.id for n in tree.descendants("s1")] == ["s2", "s3"]


def test_by_level_filters_in_document_order():
    tree = build_tree(_doc(span("s1", "A", 2, 1), span("s2", "B", 3, 2),
                           span("s3", "C", 2, 3)))
    assert [n.id for n in tree.by_level(2)] == ["s1", "s3"]


def test_orphan_paragraphs_are_kept_on_the_tree():
    tree = build_tree(_doc(span("s1", "A", 2, 5),
                           para("p0", "front matter", 1)))
    assert [p.id for p in tree.orphans] == ["p0"]


def test_every_paragraph_is_accounted_for():
    tree = build_tree(_doc(span("s1", "A", 2, 2),
                           para("p0", "front", 1),
                           para("p1", "body", 3, "s1")))
    placed = sum(len(n.paragraphs) for n in tree.nodes.values())
    assert placed + len(tree.orphans) == 2


def test_summarizer_title_restores_a_degraded_title():
    tree = build_tree(_doc(span("s1", "\\ALG\\ Application", 2, 1)))
    assert tree.nodes["s1"].summarizer_title == "Application (\\ALG\\ Application)"


def test_summarizer_title_is_plain_when_nothing_was_lost():
    tree = build_tree(_doc(span("s1", "Introduction", 2, 1)))
    assert tree.nodes["s1"].summarizer_title == "Introduction"


def test_bibkey_is_carried_from_meta():
    assert build_tree(_doc(bibkey="2209.00445")).bibkey == "2209.00445"


def test_malformed_objects_list_does_not_raise():
    assert len(build_tree({"objects": "nonsense"})) == 0


def test_cyclic_parents_do_not_hang_depth():
    """Defensive: a malformed tree must not spin forever."""
    tree = build_tree(_doc(span("s1", "A", 2, 1)))
    tree.nodes["s1"] = MarkerNode(
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
    assert stats["markers"] == 24
    assert stats["levels"] == {2: 14, 3: 8, 4: 2}
    assert stats["roots"] == 14
    assert stats["appendix_markers"] == 6


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
    # 84 prose units + 60 math units rendered to text. The other 14 math
    # objects are single symbols like T and k, dropped as signal-free.
    assert stats["paragraphs"] == 144
    assert stats["math_sources"] == {"fallback": 60, "none": 14}
    # One orphan remains: the Abstract, which legitimately precedes the first
    # span. The other candidate was a Paragraph whose entire text is
    # `\\maketitle`, dropped as a LaTeX artifact.
    assert stats["orphan_paragraphs"] == 1
    assert "computational interpretation" in tree.orphans[0].text


@needs_real
def test_real_math_reaches_the_sections_that_contain_it():
    """74 math objects used to be excluded from every summary. Formula objects
    carry no parent_section, so this only works via flow-index assignment."""
    tree = load_tree(REAL_DOCMODEL)
    by_title = {n.title: n for n in tree.nodes.values()}
    algo = by_title["The Conceptualization Algorithm"]
    assert any(p.id.startswith("obj_") and "sum" in p.text or "equals" in p.text
               for p in algo.paragraphs)


@needs_real
def test_real_math_enlarges_the_body_it_belongs_to():
    import json as _json
    from conceptdrill.hierarchy.docmodel_tree import build_tree as _bt
    raw = _json.loads(REAL_DOCMODEL.read_text())
    with_math = load_tree(REAL_DOCMODEL)
    without = _bt(raw, str(REAL_DOCMODEL), include_math=False)
    a = [n for n in with_math.nodes.values()
         if n.title == "The Conceptualization Algorithm"][0]
    b = [n for n in without.nodes.values()
         if n.title == "The Conceptualization Algorithm"][0]
    assert len(a.body_text) > len(b.body_text)


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
    tree = build_tree(_doc(span("s1", "A", 2, 1),
                           para("p1", "\\maketitle", 2, "s1"),
                           para("p2", "real content here", 3, "s1")))
    assert [p.id for p in tree.nodes["s1"].paragraphs] == ["p2"]


# --------------------------------------------------------------------------
# assign_by_flow — math has no parent_section
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.docmodel_tree import assign_by_flow, read_math  # noqa: E402


def _unit(uid, flow, parent=None):
    return Paragraph(id=uid, text="t", marker_id=parent, flow_index=flow)


def _secs(*pairs):
    return read_markers([span(sid, sid, 2, flow) for sid, flow in pairs])


def test_unit_is_assigned_to_the_section_it_follows():
    got = assign_by_flow([_unit("m1", 5)], _secs(("s1", 1), ("s2", 9)))
    assert got[0].marker_id == "s1"


def test_unit_after_the_last_section_belongs_to_it():
    got = assign_by_flow([_unit("m1", 99)], _secs(("s1", 1), ("s2", 9)))
    assert got[0].marker_id == "s2"


def test_unit_before_the_first_section_stays_unowned():
    """Front matter has no span, and inventing one would be wrong."""
    got = assign_by_flow([_unit("m1", 0)], _secs(("s1", 5)))
    assert got[0].marker_id is None


def test_an_explicit_parent_always_wins():
    got = assign_by_flow([_unit("m1", 99, parent="s1")], _secs(("s1", 1), ("s2", 9)))
    assert got[0].marker_id == "s1"


def test_exact_boundary_belongs_to_that_section():
    got = assign_by_flow([_unit("m1", 9)], _secs(("s1", 1), ("s2", 9)))
    assert got[0].marker_id == "s2"


def test_no_sections_leaves_units_untouched():
    got = assign_by_flow([_unit("m1", 5)], [])
    assert got[0].marker_id is None


# --------------------------------------------------------------------------
# Math in the tree
# --------------------------------------------------------------------------

def _formula(fid, latex, flow):
    return {"id": fid, "type": "Formula",
            "props": {"latex": latex, "flow_index": flow}}


def test_math_reaches_the_section_body():
    tree = build_tree({"objects": [
        span("s1", "Method", 2, 1),
        para("p1", "prose here", 2, "s1"),
        _formula("f1", r"L = \sum_i y_i \log p_i", 3)]})
    assert "the sum of" in tree.nodes["s1"].body_text


def test_math_can_be_excluded():
    tree = build_tree({"objects": [
        span("s1", "Method", 2, 1),
        _formula("f1", r"L = \sum_i y_i", 2)]}, include_math=False)
    assert tree.nodes["s1"].body_text == ""


def test_trivial_formulas_do_not_reach_the_body():
    tree = build_tree({"objects": [
        span("s1", "Method", 2, 1), _formula("f1", "T", 2)]})
    assert tree.nodes["s1"].body_text == ""


def test_math_source_tally_is_recorded():
    tree = build_tree({"objects": [
        span("s1", "M", 2, 1), _formula("f1", r"a \subseteq b", 2)]})
    assert tree.stats()["math_sources"].get("fallback") == 1


def test_docmodel_spoken_field_is_preferred_when_present():
    """The coming docmodel update will carry spoken math; it must win."""
    tree = build_tree({"objects": [
        span("s1", "M", 2, 1),
        {"id": "f1", "type": "Formula",
         "props": {"latex": r"a \subseteq b", "spoken": "a is contained in b",
                   "flow_index": 2}}]})
    assert "a is contained in b" in tree.nodes["s1"].body_text
    assert tree.stats()["math_sources"].get("docmodel") == 1


# --------------------------------------------------------------------------
# Content-type routing and the skip ledger
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.docmodel_tree import (BODY_PROPS,  # noqa: E402
                                                  SKIPPED_TYPES, SkippedObject,
                                                  read_content, reference_tail)


def obj(oid, otype, flow=1, **props):
    return {"id": oid, "type": otype, "props": {"flow_index": flow, **props}}


@pytest.mark.parametrize("otype,prop", [
    ("Paragraph", "text"), ("Abstract", "text"), ("ListItem", "content"),
    ("Sidenote", "content"), ("Footnote", "content"),
    ("Picture", "caption"), ("Diagram", "caption"),
])
def test_every_prose_type_is_read(otype, prop):
    """Five of these were consulted by nothing at all, discarding 148,772
    characters of Sidenote text across the 10-document set."""
    units, skips = read_content([obj("o1", otype, **{prop: "real prose here"})])
    assert [u.id for u in units] == ["o1"], (otype, skips)


def test_a_picture_url_is_never_read_as_text():
    """`url` and `cdn_url` hold mathpix links, longer than the caption."""
    units, skips = read_content([
        obj("o1", "Picture", url="https://cdn.mathpix.com/cropped/x.jpg")])
    assert units == []
    assert skips[0].reason.startswith("no text under")


def test_a_caption_that_is_only_a_url_is_skipped_with_a_reason():
    units, skips = read_content([
        obj("o1", "Diagram", caption="https://cdn.mathpix.com/cropped/x.jpg")])
    assert units == []
    assert skips[0].reason == "caption is only a URL"


@pytest.mark.parametrize("otype", sorted(SKIPPED_TYPES))
def test_every_skipped_type_names_its_reason(otype):
    units, skips = read_content([obj("o1", otype, text="anything", raw_text="x")])
    assert units == []
    assert skips[0].object_id == "o1" and skips[0].reason == SKIPPED_TYPES[otype]


def test_an_unknown_type_is_recorded_rather_than_ignored():
    """This is how LtxCommand was found: 40 objects, previously invisible."""
    units, skips = read_content([obj("o1", "SomethingNew", text="hello")])
    assert units == []
    assert "unhandled object type" in skips[0].reason


def test_a_table_is_skipped_because_it_is_numbers():
    units, skips = read_content([
        obj("o1", "Table", raw_text="NaiveBayes\nJ48\n56.30\n42.20")])
    assert units == []
    assert "numeric results" in skips[0].reason


def test_sections_and_math_are_not_reported_as_skips():
    """A span is accounted for as a span; math goes through read_math."""
    _, skips = read_content([obj("s1", "Section", caption="Method", level=1),
                             obj("f1", "Formula", latex="x = y")])
    assert skips == []


# ---- reference tails ------------------------------------------------------

def test_prose_citing_works_is_not_a_reference_list():
    """`model [7, 10, 16, 18] or inform` is prose. Marker density does not
    separate these: real prose reaches 8.6 markers per 1000 characters."""
    text = ("We select a temporal retrieval model [7, 10, 16, 18] or inform it "
            "about the period of interest [3] and then rank [9] accordingly.")
    assert reference_tail(text) == len(text)


def test_a_reference_list_is_found_from_its_start():
    text = ("[Allen et al., 2000] Allen, J., Byron, D. Toward conversational "
            "agents. Cognitive Science. "
            "[Fensel et al., 2003] Fensel, D., Hendler, J. Spinning the web. "
            "[Martin et al., 1999] Martin, D., Cheyer, A. The open agent.")
    assert reference_tail(text) == 0


def test_a_mixed_object_keeps_its_prose_and_loses_the_entries():
    """One Sidenote holds a paper's conclusion followed by eight references.
    Dropping it loses the conclusion; keeping it puts surnames in a label."""
    prose = ("In this work we laid out an approach to temporal query intent "
             "classification and found simple n-gram features effective. ")
    tail = ("[1] Stanford CoreNLP. Toolkit. "
            "[2] O. Alonso, M. Gertz. Value of temporal information. "
            "[3] K. Berberich, S. Bedathur. A language modeling approach. ")
    cut = reference_tail(prose + tail)
    assert 0 < cut <= len(prose) + 2
    assert "n-gram features" in (prose + tail)[:cut]


def test_two_entries_are_not_enough_to_be_a_list():
    text = "Prose here. [A] Author, B. Title. [C] Another, D. Title."
    assert reference_tail(text) == len(text)


def test_an_object_that_is_entirely_references_is_skipped_with_a_reason():
    text = ("[Alcala-Fdez et al., 2011] Jesus Alcala-Fdez, Alberto Fernandez. "
            "[Brown et al., 2012] Gavin Brown, Adam Pocock. "
            "[Reimherr and Nicolae, 2013] Matthew Reimherr, Dan Nicolae. ")
    units, skips = read_content([obj("o1", "Sidenote", content=text)])
    assert units == []
    assert skips[0].reason == "reference list: no prose before the entries"


# ---- conservation ---------------------------------------------------------

def test_the_tree_accounts_for_every_object():
    """GATE 5a.2 in miniature: nothing unaccounted, nothing counted twice."""
    objects = [obj("s1", "Section", flow=1, caption="Method", level=1),
               obj("p1", "Paragraph", flow=2, text="Real prose about a method."),
               obj("t1", "Table", flow=3, raw_text="1\n2\n3"),
               obj("x1", "MysteryType", flow=4, text="who knows"),
               obj("f1", "Formula", flow=5, latex="x = y + z")]
    tree = build_tree({"objects": objects})
    account = tree.accounting(objects)
    assert account["unaccounted"] == []
    assert account["double_counted"] == []
    assert account["accounted_for"] == len(objects)


def test_a_skipped_object_reaches_the_tree_with_its_reason():
    tree = build_tree({"objects": [
        obj("s1", "Section", flow=1, caption="Method", level=1),
        obj("t1", "Table", flow=2, raw_text="1\n2")]})
    assert any(s.object_id == "t1" and "numeric" in s.reason for s in tree.skipped)


def test_skipped_objects_serialise():
    s = SkippedObject("o1", "Table", "because")
    assert s.to_dict() == {"object_id": "o1", "object_type": "Table",
                           "reason": "because"}
