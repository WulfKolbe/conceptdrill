"""Unit tests for Algorithm 2 (hierarchical refinement).

Includes the DAG case the siblings score was designed for, which a section
tree never exercises.
"""
from __future__ import annotations

import numpy as np
import pytest

from conceptdrill.hierarchy.docmodel_tree import build_tree
from conceptdrill.hierarchy.refine import (ConceptGraph, label_entropy, refine)


def tree_graph() -> ConceptGraph:
    """root -> a, b ; a -> a1, a2 ; b -> b1"""
    return ConceptGraph(
        children={"root": ("a", "b"), "a": ("a1", "a2"), "b": ("b1",),
                  "a1": (), "a2": (), "b1": ()},
        parents={"root": (), "a": ("root",), "b": ("root",),
                 "a1": ("a",), "a2": ("a",), "b1": ("b",)},
        labels={c: c.upper() for c in ("root", "a", "b", "a1", "a2", "b1")},
        depth={"root": 1, "a": 2, "b": 2, "a1": 3, "a2": 3, "b1": 3})


def dag_graph() -> ConceptGraph:
    """A DAG: x has two parents, y has one. This is what sibscore is for."""
    return ConceptGraph(
        children={"p": ("x", "y"), "q": ("x",), "x": (), "y": ()},
        parents={"p": (), "q": (), "x": ("p", "q"), "y": ("p",)},
        labels={"p": "P", "q": "Q", "x": "X", "y": "Y"},
        depth={"p": 1, "q": 1, "x": 2, "y": 2})


def vectors_for(ids, table):
    return np.vstack([table[i] for i in ids])


# --------------------------------------------------------------------------
# The set operations
# --------------------------------------------------------------------------

def test_siblings_is_children_minus_self():
    g = tree_graph()
    assert g.siblings("a1", "a") == ("a2",)
    assert g.siblings("a2", "a") == ("a1",)


def test_an_only_child_has_no_siblings():
    assert tree_graph().siblings("b1", "b") == ()


def test_parents_is_a_set():
    assert dag_graph().parents_of("x") == frozenset({"p", "q"})


def test_a_root_has_no_parents():
    assert tree_graph().parents_of("root") == frozenset()


def test_sibscore_of_an_only_child_is_one():
    """No sibling can disagree, so the edge is maximally coherent."""
    assert tree_graph().sibscore("b", "b1") == 1.0


def test_sibscore_is_always_one_on_a_tree():
    """Every sibling shares the single parent: |{p}|/|{p}| = 1 for every term.
    Measured: 0 of 8695 real sections have more than one parent."""
    g = tree_graph()
    assert g.sibscore("a", "a1") == 1.0
    assert g.sibscore("a", "a2") == 1.0
    assert g.sibscore("root", "a") == 1.0


def test_a_tree_is_detected_as_such():
    assert tree_graph().is_tree()
    assert not dag_graph().is_tree()


def test_sibscore_discriminates_on_a_dag():
    """x has parents {p,q}; its sibling y has only {p}. The overlap is 1 of x's
    2 parents, so the edge scores 0.5 -- x is less bound to this sibling group
    than an exclusively-p child would be."""
    g = dag_graph()
    assert g.sibscore("p", "x") == pytest.approx(0.5)
    assert g.sibscore("p", "y") == pytest.approx(1.0)


def test_children_ranked_puts_the_higher_score_first():
    ranked = dag_graph().children_ranked("p")
    assert [c for c, _ in ranked] == ["y", "x"]


def test_children_ranked_falls_back_to_document_order_on_a_tree():
    """Every score is 1.0, so this is entirely a document-order ranking."""
    ranked = tree_graph().children_ranked("a")
    assert [c for c, _ in ranked] == ["a1", "a2"]
    assert {s for _, s in ranked} == {1.0}


# --------------------------------------------------------------------------
# Starting set
# --------------------------------------------------------------------------

def test_top_level_is_the_shallowest_depth_present():
    assert tree_graph().top_level() == ("root",)


def test_top_level_does_not_assume_depth_one():
    """Real documents start at level 1 or 2; hard-coding 1 would return nothing
    for a third of the corpus."""
    g = ConceptGraph(children={"a": (), "b": ()}, parents={"a": (), "b": ()},
                     depth={"a": 2, "b": 2})
    assert set(g.top_level()) == {"a", "b"}


def test_graph_from_a_section_tree():
    t = build_tree({"objects": [
        {"id": "s1", "type": "Section",
         "props": {"caption": "Method", "level": 2, "flow_index": 1}},
        {"id": "s2", "type": "Section",
         "props": {"caption": "Sub", "level": 3, "flow_index": 2}}]})
    g = ConceptGraph.from_section_tree(t)
    assert g.children_of("s1") == ("s2",)
    assert g.parents_of("s2") == frozenset({"s1"})
    assert g.labels["s1"] == "Method"
    assert g.top_level() == ("s1",)


# --------------------------------------------------------------------------
# The refinement loop
# --------------------------------------------------------------------------

VECS = {"root": np.array([1.0, 0, 0]), "a": np.array([0.9, 0.4, 0]),
        "b": np.array([0, 1.0, 0]), "a1": np.array([0.8, 0.6, 0]),
        "a2": np.array([0.7, 0, 0.7]), "b1": np.array([0, 0.9, 0.4])}


def run(target, **kw):
    g = tree_graph()
    return g, refine(g, context_vectors=np.array([[1.0, 0.0, 0.0]]),
                     concept_vectors=lambda ids: vectors_for(ids, VECS),
                     target_size=target, **kw)


def test_it_starts_at_the_top_level():
    _, res = run(target=1)
    assert res.concepts == ("root",)


def test_it_grows_to_the_target():
    _, res = run(target=3)
    assert len(res) >= 3


def test_expansion_adds_children_of_the_dominant_concept():
    _, res = run(target=3)
    assert res.steps[0].expanded == "root"
    assert set(res.steps[0].added) <= {"a", "b"}


def test_child_fraction_limits_how_many_are_taken():
    _, res = run(target=99, child_fraction=0.5)
    first = res.steps[0]
    assert len(first.added) == 1, "half of two children, rounded up"


def test_full_child_fraction_takes_all():
    _, res = run(target=99, child_fraction=1.0)
    assert len(res.steps[0].added) == 2


def test_remove_parent_replaces_rather_than_adds():
    _, res = run(target=99, child_fraction=1.0, remove_parent=True)
    assert res.steps[0].removed == ("root",)
    assert "root" not in res.concepts


def test_keeping_the_parent_is_the_default():
    _, res = run(target=3)
    assert res.steps[0].removed == ()
    assert "root" in res.concepts


def test_it_stops_when_everything_is_a_leaf():
    """The paper leaves termination implicit; without this it spins forever,
    because the deepest concepts always have no children."""
    _, res = run(target=999)
    assert "leaves" in res.stopped_because
    assert len(res) < 999


def test_it_stops_at_the_target_size():
    _, res = run(target=3)
    assert res.stopped_because == "reached target size"


def test_no_starting_concepts_is_reported():
    g = ConceptGraph()
    res = refine(g, context_vectors=np.array([[1.0, 0.0]]),
                 concept_vectors=lambda ids: np.zeros((len(ids), 2)),
                 target_size=5)
    assert res.stopped_because == "no starting concepts"


def test_steps_record_the_walk():
    _, res = run(target=4)
    assert all(s.size_after > 0 for s in res.steps)
    assert [s.expanded for s in res.steps][0] == "root"


def test_a_concept_is_never_added_twice():
    _, res = run(target=99, child_fraction=1.0)
    assert len(set(res.concepts)) == len(res.concepts)


def test_the_result_flags_a_degenerate_sibscore():
    """A caller should know the child ranking was document order, not scoring."""
    _, res = run(target=3)
    assert res.sibscore_informative is False


def test_a_dag_reports_an_informative_sibscore():
    g = dag_graph()
    res = refine(g, context_vectors=np.array([[1.0, 0.0]]),
                 concept_vectors=lambda ids: np.ones((len(ids), 2)),
                 target_size=3)
    assert res.sibscore_informative is True


def test_an_explicit_start_is_honoured():
    g = tree_graph()
    res = refine(g, context_vectors=np.array([[1.0, 0.0, 0.0]]),
                 concept_vectors=lambda ids: vectors_for(ids, VECS),
                 target_size=3, start=["a"])
    assert res.concepts[0] == "a"


def test_serialisation_carries_labels():
    g, res = run(target=3)
    d = res.to_dict(g)
    assert d["size"] == len(res)
    assert any(c["label"] for c in d["concepts"])


# --------------------------------------------------------------------------
# The optional label blend
# --------------------------------------------------------------------------

def test_entropy_of_a_pure_set_is_zero():
    assert label_entropy(["a", "a", "a"]) == 0.0


def test_entropy_of_a_balanced_pair_is_one_bit():
    assert label_entropy(["a", "b"]) == pytest.approx(1.0)


def test_entropy_of_nothing_is_zero():
    assert label_entropy([]) == 0.0


def test_labels_are_accepted_without_error():
    g = tree_graph()
    res = refine(g, context_vectors=np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                 concept_vectors=lambda ids: vectors_for(ids, VECS),
                 target_size=3, labels=["x", "y"], blend=0.5)
    assert len(res) >= 3
