"""Document parsing and the Semantic Compiler adapter."""
from __future__ import annotations

from conceptdrill.document import Document, is_docmodel
from conceptdrill.docmodel import (NON_PROJECTABLE, extract_text,
                                   _looks_like_bibliography, skip_reason)


# --------------------------------------------------------------------------
# Generic schema
# --------------------------------------------------------------------------

def test_generic_parsing(mock_document):
    assert len(mock_document.blocks) == 10
    assert len(mock_document.sections) == 6
    assert len(mock_document.bibliography) == 4


def test_prose_excludes_equations_and_tables(mock_document):
    types = {b.type for b in mock_document.prose_blocks}
    assert "equation" not in types
    assert "table" not in types
    assert "paragraph" in types


def test_math_blocks_found(mock_document):
    assert len(mock_document.math_blocks) == 2


def test_code_blocks_empty_without_code_objects(mock_document):
    """The DocModel emits no code objects; the accessor must simply return
    nothing rather than failing."""
    assert mock_document.code_blocks == []


def test_section_path_builds_breadcrumb(mock_document):
    path = mock_document.section_path("s2a")
    assert path == ["2 Method", "2.1 Semantic Projection"]


def test_top_level_section_walks_to_root(mock_document):
    assert mock_document.top_level_section("s2a") == "s2"
    assert mock_document.top_level_section("s2") == "s2"
    assert mock_document.top_level_section(None) is None


def test_bibliography_accepts_bare_strings():
    doc = Document.from_generic({"bibliography": ["A Bare Title String"]})
    assert doc.bibliography[0].title == "A Bare Title String"


def test_section_iteration_is_deterministic(mock_document):
    a = [s.id for s in mock_document.iter_sections_sorted()]
    b = [s.id for s in mock_document.iter_sections_sorted()]
    assert a == b


# --------------------------------------------------------------------------
# DocModel adapter
# --------------------------------------------------------------------------

def test_docmodel_is_detected(docmodel_json, mock_document_json):
    assert is_docmodel(docmodel_json)
    assert not is_docmodel(mock_document_json)


def test_docmodel_loads_from_path(docmodel_path):
    doc = Document.load(docmodel_path)
    assert doc.meta.get("bibkey") == "dm2026"
    assert len(doc.blocks) > 0


def test_text_extraction_per_type():
    """Each type keeps its text under a different prop — the crux of the adapter."""
    assert extract_text({"type": "Paragraph", "props": {"text": "hello"}}) == "hello"
    assert extract_text({"type": "Equation", "props": {"latex": "x=1"}}) == "x=1"
    assert extract_text({"type": "ListItem", "props": {"content": "item"}}) == "item"
    assert extract_text({"type": "Picture", "props": {"caption": "a figure"}}) == "a figure"
    assert extract_text({"type": "Section", "props": {"title": "Method"}}) == "Method"


def test_equation_falls_back_through_latex_variants():
    assert extract_text({"type": "Equation",
                         "props": {"latex": "", "latex_raw": "y=2"}}) == "y=2"
    assert extract_text({"type": "Equation",
                         "props": {"latex_original": "z=3"}}) == "z=3"


def test_table_combines_caption_and_source():
    text = extract_text({"type": "Table", "props": {"caption": "Latencies",
                                                    "latex_code": "\\begin{tabular}"}})
    assert "Latencies" in text and "tabular" in text


def test_unknown_type_still_yields_text():
    """A new DocModel object type must degrade to a best-effort lookup rather
    than silently producing nothing."""
    assert extract_text({"type": "BrandNewThing",
                         "props": {"text": "content"}}) == "content"


def test_non_projectable_types_are_skipped():
    for otype in ("Page", "Document", "Toc", "TableRow", "Citation"):
        reason = skip_reason({"type": otype, "props": {"citekey": "X"}})
        assert reason is not None
    assert "citekey" in NON_PROJECTABLE["citation"]


def test_empty_text_is_skipped_with_a_reason():
    reason = skip_reason({"type": "Paragraph", "props": {"text": "  "}})
    assert reason and "no text" in reason


def test_skipped_objects_are_recorded(docmodel_path):
    doc = Document.load(docmodel_path)
    skipped = doc.meta.get("skipped")
    assert skipped, "skips must be recorded, not silently dropped"
    types = {s["object_type"] for s in skipped}
    assert {"Page", "Document", "Citation"} <= types


def test_every_object_is_accounted_for(docmodel_json, docmodel_path):
    """No object may vanish: blocks + skips must equal the input count."""
    doc = Document.load(docmodel_path)
    n_in = len(docmodel_json["objects"])
    assert len(doc.blocks) + len(doc.meta["skipped"]) == n_in


def test_reference_list_items_become_bibliography(docmodel_path):
    doc = Document.load(docmodel_path)
    assert len(doc.bibliography) == 1
    assert "Hashnet" in doc.bibliography[0].title
    assert doc.bibliography[0].year == 2017


def test_bibliography_heuristic_rejects_plain_prose():
    assert not _looks_like_bibliography("First we consider the simple case.")
    assert not _looks_like_bibliography("There are 2017 samples in the set.")


def test_bibliography_heuristic_accepts_a_reference():
    assert _looks_like_bibliography(
        "Cao, Z., Long, M.: Hashnet: Deep learning to hash. In: Proc. ICCV (2017)")


def test_section_parent_outside_tree_is_dropped():
    """A Section whose parent is the Document root must become a root itself,
    or `top_level_section` would walk off the tree."""
    doc = Document.from_docmodel({
        "objects": [
            {"id": "doc", "type": "Document", "props": {}},
            {"id": "s1", "type": "Section", "props": {"title": "A"},
             "parent": "doc"},
            {"id": "p1", "type": "Paragraph", "props": {"text": "text here"},
             "parent": "s1"},
        ]
    })
    assert doc.sections["s1"].parent_id is None
    assert doc.top_level_section("s1") == "s1"
