"""`kb check`: the kb/1 contract, decided mechanically (T-0119, step A).

These trees are kb/1 (a root `topics/`); kb/1 stays checkable under kb/2,
which tests/test_kb2.py covers.

Every test here builds a small BROKEN tree and asserts that the check names
the break. A valid tree is never written by hand: the one valid tree comes
from the generator, with its LLM stubbed (step B). Names here are nonsense on
purpose -- the check must not care what a knowledge base is about.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conceptdrill.kb import check as kbcheck
from conceptdrill.kb.contract import KB1, LETTERS
from conceptdrill.output import REPO_ROOT

HEADINGS, PROTOCOL = KB1.headings, KB1.protocol
SPEC = REPO_ROOT / "docs" / "kb-format" / "kb-1.md"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def md(front: dict, body: str) -> str:
    import yaml
    return "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n" + body


LEAF_FRONT = {"format": "kb/1", "id": "zorp/blix", "title": "Blix", "topic": "zorp",
              "concepts": [], "sources": ["d1#0"], "payload": [], "related": []}


def leaf_body(skip=None, title="Blix"):
    heads = [h for h in HEADINGS["leaf"] if h != skip]
    return f"# {title}\n\n" + "".join(f"## {h}\n\nNone.\n\n" for h in heads)


def problems(kb: Path) -> list[str]:
    return kbcheck.check(kb).violations


def any_match(found: list[str], pattern: str) -> bool:
    return any(re.search(pattern, v) for v in found)


# --------------------------------------------------------------------------
# The contract's constants are the document's, not a second copy
# --------------------------------------------------------------------------

def test_the_reading_protocol_is_the_spec_text():
    text = SPEC.read_text()
    block = text[text.index("## 8. The reading protocol"):]
    fenced = block.split("```\n")[1]
    assert PROTOCOL == fenced.rstrip("\n")


def test_the_heading_table_is_the_spec_table():
    text = SPEC.read_text()
    row = next(l for l in text.splitlines() if l.startswith("| leaf |"))
    assert [h.strip() for h in row.split("|")[2].split("·")] == HEADINGS["leaf"]


def test_there_are_27_letters():
    assert LETTERS == [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["other"]


# --------------------------------------------------------------------------
# Broken trees
# --------------------------------------------------------------------------

def test_an_empty_folder_is_not_a_knowledge_base(tmp_path):
    found = problems(tmp_path)
    assert any_match(found, r"README\.md: missing")
    assert any_match(found, r"kb\.json: missing")


def test_an_extra_folder_is_named(tmp_path):
    (tmp_path / "notes").mkdir()
    assert any_match(problems(tmp_path), r"notes: not in the kb/\d tree")


def test_a_hidden_entry_is_named(tmp_path):
    write(tmp_path / "topics" / "01-zorp" / ".draft.md", "x")
    assert any_match(problems(tmp_path), r"\.draft\.md: hidden")


def test_a_missing_letter_file_is_named(tmp_path):
    for letter in LETTERS[:-1]:
        write(tmp_path / "index" / f"{letter}.md", "x")
    assert any_match(problems(tmp_path), r"index/other\.md: missing")


def test_topic_folders_must_be_numbered_contiguously(tmp_path):
    write(tmp_path / "topics" / "01-zorp" / "README.md", "x")
    write(tmp_path / "topics" / "03-quax" / "README.md", "x")
    assert any_match(problems(tmp_path), r"topics/03-quax: expected 02-")


def test_a_missing_heading_is_named(tmp_path):
    write(tmp_path / "topics" / "01-zorp" / "blix.md",
          md(LEAF_FRONT, leaf_body(skip="Pitfalls")))
    assert any_match(problems(tmp_path),
                     r"topics/01-zorp/blix\.md: headings .*Pitfalls")


def test_an_empty_heading_must_say_none(tmp_path):
    body = leaf_body().replace("## Summary\n\nNone.\n\n", "## Summary\n\n")
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(LEAF_FRONT, body))
    assert any_match(problems(tmp_path), r"blix\.md: ## Summary is empty")


def test_a_missing_frontmatter_key_is_named(tmp_path):
    front = {k: v for k, v in LEAF_FRONT.items() if k != "payload"}
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(front, leaf_body()))
    assert any_match(problems(tmp_path), r"blix\.md: frontmatter .*payload")


def test_the_title_must_match_the_frontmatter(tmp_path):
    write(tmp_path / "topics" / "01-zorp" / "blix.md",
          md(LEAF_FRONT, leaf_body(title="Something Else")))
    assert any_match(problems(tmp_path), r"blix\.md: title")


def test_a_dead_link_is_named(tmp_path):
    body = leaf_body().replace("## See also\n\nNone.",
                               "## See also\n\n- [Gone](gone.md)")
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(LEAF_FRONT, body))
    assert any_match(problems(tmp_path), r"blix\.md: dead link gone\.md")


@pytest.mark.parametrize("target", ["/etc/passwd", "https://example.org/x",
                                    "../../../outside.md"])
def test_a_link_out_of_the_tree_is_named(tmp_path, target):
    body = leaf_body().replace("## See also\n\nNone.",
                               f"## See also\n\n- [Out]({target})")
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(LEAF_FRONT, body))
    assert any_match(problems(tmp_path), r"blix\.md: link .* (absolute|URL|outside)")


def test_a_line_of_the_wrong_form_is_named(tmp_path):
    body = leaf_body().replace("## Key concepts\n\nNone.",
                               "## Key concepts\n\nsome prose, not a bullet")
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(LEAF_FRONT, body))
    assert any_match(problems(tmp_path), r"blix\.md: ## Key concepts line 1")


def test_an_over_budget_leaf_is_named(tmp_path):
    body = leaf_body().replace("## Summary\n\nNone.",
                               "## Summary\n\n" + "word " * 2000)
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(LEAF_FRONT, body))
    assert any_match(problems(tmp_path), r"blix\.md: \d+ tokens, budget 2500")


def test_a_changed_reading_protocol_is_named(tmp_path):
    front = {"format": "kb/1", "kb": "zz", "title": "Zz", "documents": [], "topics": []}
    body = ("# Zz\n\n## Purpose\n\nNone.\n\n## Topics\n\nNone.\n\n"
            "## How to read this knowledge base\n\n1. Read everything.\n\n"
            "## Sources\n\nNone.\n")
    write(tmp_path / "README.md", md(front, body))
    assert any_match(problems(tmp_path), r"README\.md: .*reading protocol")


def test_kb_json_must_follow_its_schema(tmp_path):
    write(tmp_path / "kb.json", json.dumps({"format": "kb/0"}))
    assert any_match(problems(tmp_path), r"kb\.json: schema")


def _kb_json(sections, leaves=None):
    leaves = leaves or [{"id": "zorp/blix", "title": "Blix", "topic": "zorp",
                         "path": "topics/01-zorp/blix.md", "concepts": [],
                         "sources": ["d1#0"], "payload": [], "related": []}]
    return {"format": "kb/1", "kb": "zz",
            "documents": [{"doc": "d1", "folder": "D 1 (x)", "title": "D1",
                           "glyphs": {"path": "/nowhere/d1.glyphs.md", "sha256": "0" * 64},
                           "docmodel": None}],
            "sections": sections, "payload": [],
            "topics": [{"id": "zorp", "title": "Zorp", "folder": "topics/01-zorp",
                        "leaves": ["zorp/blix"]}],
            "leaves": leaves, "concepts": [], "terms": []}


def _section(order, status="cited", cited_by=("zorp/blix",), reason=None):
    return {"id": f"d1#{order}", "doc": "d1", "order": order, "title": f"S{order}",
            "number": [], "page": 1, "line_start": 1, "line_end": 2,
            "status": status, "cited_by": list(cited_by), "reason": reason,
            "text_file": f"text/d1/{order}.txt", "text_sha256": "0" * 64, "words": 2,
            "nul": 0}


def test_a_section_left_out_is_named(tmp_path):
    write(tmp_path / "kb.json", json.dumps(_kb_json([_section(0), _section(2)])))
    assert any_match(problems(tmp_path), r"kb\.json: d1 sections .*d1#1")


def test_an_unused_section_needs_a_reason(tmp_path):
    write(tmp_path / "kb.json", json.dumps(_kb_json(
        [_section(0), _section(1, status="unused", cited_by=(), reason=None)])))
    assert any_match(problems(tmp_path), r"kb\.json: d1#1 unused without a reason")


def test_a_cited_section_must_be_cited_by_its_leaf(tmp_path):
    write(tmp_path / "kb.json", json.dumps(_kb_json([_section(0), _section(1)])))
    assert any_match(problems(tmp_path), r"kb\.json: d1#1 .*zorp/blix does not cite it")


def test_a_payload_file_must_hold_its_recorded_bytes(tmp_path):
    kb = _kb_json([_section(0)])
    kb["payload"] = [{"id": "d1#obj_1", "doc": "d1", "kind": "code", "members": ["obj_1"],
                      "field": "code", "section": "d1#0", "placed_by": "text",
                      "status": "placed", "labels": ["line numbers not separated"],
                      "file": "payload/d1/d1-obj-1.txt", "sha256": "0" * 64, "reason": None}]
    write(tmp_path / "kb.json", json.dumps(kb))
    write(tmp_path / "payload" / "d1" / "d1-obj-1.txt", "int x;\n")
    assert any_match(problems(tmp_path), r"payload/d1/d1-obj-1\.txt: sha256")


def test_the_report_is_complete_not_first_error_only(tmp_path):
    (tmp_path / "notes").mkdir()
    found = problems(tmp_path)
    assert len(found) > 5


def test_the_cli_exits_1_and_lists_every_violation(tmp_path, capsys):
    from conceptdrill import cli
    (tmp_path / "notes").mkdir()
    assert cli.main(["kb", "check", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert re.search(r"notes: not in the kb/\d tree", out)
    assert re.search(r"kb/\d: \d+ violations", out)


# --------------------------------------------------------------------------
# Section 7a: the rules beyond the schemas, one broken tree each
# --------------------------------------------------------------------------

def _write_kb(tmp_path, kb):
    write(tmp_path / "kb.json", json.dumps(kb))
    return problems(tmp_path)


def test_rule1_a_cited_section_has_no_reason(tmp_path):
    found = _write_kb(tmp_path, _kb_json([_section(0, reason="why")]))
    assert any_match(found, r"kb\.json: d1#0 cited but has a reason")


def test_rule1_an_unused_section_is_cited_by_nobody(tmp_path):
    found = _write_kb(tmp_path, _kb_json(
        [_section(0), _section(1, status="unused", reason="short")]))
    assert any_match(found, r"kb\.json: d1#1 unused but cited by")


def test_rule2_a_leaf_source_must_name_the_leaf_back(tmp_path):
    found = _write_kb(tmp_path, _kb_json(
        [_section(0, status="unused", cited_by=(), reason="short")]))
    assert any_match(found, r"kb\.json: zorp/blix cites d1#0, which does not name it")


def _payload(**kw):
    item = {"id": "d1#obj_1", "doc": "d1", "kind": "code", "members": ["obj_1"],
            "field": "code", "section": "d1#0", "placed_by": "text",
            "status": "placed", "labels": [], "file": "payload/d1/d1-obj-1.txt",
            "sha256": "0" * 64, "reason": None}
    item.update(kw)
    return item


def test_rule3_a_placed_item_needs_its_file(tmp_path):
    kb = _kb_json([_section(0)])
    kb["payload"] = [_payload(file=None)]
    assert any_match(_write_kb(tmp_path, kb), r"kb\.json: d1#obj_1 placed without file")


def test_rule3_an_unplaced_item_has_no_file_and_a_reason(tmp_path):
    kb = _kb_json([_section(0)])
    kb["payload"] = [_payload(status="unplaced", reason=None)]
    found = _write_kb(tmp_path, kb)
    assert any_match(found, r"kb\.json: d1#obj_1 unplaced but has file")
    assert any_match(found, r"kb\.json: d1#obj_1 unplaced without a reason")


def test_rule4_two_items_may_not_share_a_file(tmp_path):
    kb = _kb_json([_section(0)])
    kb["payload"] = [_payload(), _payload(id="d1#obj_2", members=["obj_2"])]
    assert any_match(_write_kb(tmp_path, kb),
                     r"kb\.json: payload/d1/d1-obj-1\.txt used by d1#obj_1 and d1#obj_2")


def test_rule5_a_referenced_id_must_exist(tmp_path):
    kb = _kb_json([_section(0)])
    kb["leaves"][0]["related"] = ["zorp/nope"]
    assert any_match(_write_kb(tmp_path, kb), r"kb\.json: zorp/blix related zorp/nope does not exist")


def test_rule6_frontmatter_must_equal_kb_json(tmp_path):
    kb = _kb_json([_section(0)])
    write(tmp_path / "kb.json", json.dumps(kb))
    front = dict(LEAF_FRONT, title="Other Title")
    write(tmp_path / "topics" / "01-zorp" / "blix.md", md(front, leaf_body(title="Other Title")))
    assert any_match(problems(tmp_path), r"blix\.md: frontmatter title differs from kb\.json")


def test_rule7_every_section_has_its_text_file(tmp_path):
    found = _write_kb(tmp_path, _kb_json([_section(0)]))
    assert any_match(found, r"text/d1/0\.txt: missing")


def test_rule8_a_doc_slug_must_be_derived_from_its_folder(tmp_path):
    kb = _kb_json([_section(0)])
    kb["documents"][0]["folder"] = "Something Else"
    assert any_match(_write_kb(tmp_path, kb), r"kb\.json: doc d1 is not the slug of 'Something Else'")


def test_rule9_letter_counts_must_agree(tmp_path):
    front = {"format": "kb/1", "kb": "zz", "letter": "q", "entries": 3}
    write(tmp_path / "index" / "q.md", md(front, "# Index: Q\n\n## Index\n\nNone.\n"))
    assert any_match(problems(tmp_path), r"index/q\.md: entries 3 but 0 lines")


@pytest.mark.parametrize("folder,slug", [
    ("0111016 (1)", "0111016-1"),
    ("020-Cation Distribution in Eu-Sc-Fe Garnets", "020-cation-distribution-in-eu-sc-fe-garnets"),
    ("IPTA_79", "ipta-79"),
    ("Café   Größe", "cafe-groe"),
    ("___", "doc"),
])
def test_doc_slugs_follow_section_1a(folder, slug):
    from conceptdrill.kb.contract import doc_slug
    assert doc_slug(folder) == slug


def test_long_folder_names_are_cut_at_a_dash_and_collisions_numbered():
    from conceptdrill.kb.contract import doc_slugs
    long = ("001 Handbook of Categorical Algebra Volume 1, Basic Category Theory "
            "(Francis Borceux) (Z-Library)")
    got = doc_slugs([long, "a b", "a-b", "A B"])
    assert len(got[long]) <= 56 and not got[long].endswith("-")
    assert got[long] == "001-handbook-of-categorical-algebra-volume-1-basic"
    assert [got["A B"], got["a b"], got["a-b"]] == ["a-b", "a-b-2", "a-b-3"]


def test_rule7_the_nul_count_must_match_the_text_file(tmp_path):
    sec = dict(_section(0), nul=3)
    kb = _kb_json([sec])
    write(tmp_path / "text" / "d1" / "0.txt", "a b\n")
    assert any_match(_write_kb(tmp_path, kb), r"text/d1/0\.txt: 0 NUL bytes, kb\.json says 3")
