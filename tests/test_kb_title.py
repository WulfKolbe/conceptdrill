"""A document's title: its first `# ` heading unless that is a generic section
name, else the docmodel's meta.title unless it carries LaTeX, else the first
plain line before any heading, else the folder name."""
import json

from conceptdrill.kb.build import _title_of


def test_a_real_first_heading_is_the_title():
    assert _title_of("# Zent Steps of the Florp\n\nbody\n", "f") == "Zent Steps of the Florp"


def test_generic_section_headings_are_not_titles():
    for head in ("REFERENCES", "1 Introduction", "I. Introduction", "1. Introduction",
                 "Contents", "Bibliography", "Acknowledgments", "Abstract", "Introduction"):
        assert _title_of(f"# {head}\n\nbody\n", "fold") == "fold", head


def test_the_docmodel_title_comes_before_a_generic_heading(tmp_path):
    dm = tmp_path / "model.docmodel.json"
    dm.write_text(json.dumps({"meta": {"title": "Zent:  A Florp\nStudy"}}))
    assert _title_of("# 1 Introduction\n", "fold", dm) == "Zent: A Florp Study"


def test_a_docmodel_title_with_latex_is_skipped(tmp_path):
    dm = tmp_path / "model.docmodel.json"
    dm.write_text(json.dumps({"meta": {"title": r"\raisebox{-0.2\height}{x} Zent"}}))
    assert _title_of("Zent Florp Study\n\n## Abstract\n", "fold", dm) == "Zent Florp Study"


def test_the_first_plain_line_before_any_heading_is_the_title():
    assert _title_of("PubLayNet: largest dataset\nXu Zhong\n## Abstract\n", "f") \
        == "PubLayNet: largest dataset"


def test_latex_preamble_lines_are_not_titles():
    assert _title_of("\\par\n\\begingroup\n## Introduction\n", "fold") == "fold"


def test_a_heading_wins_over_the_docmodel_title(tmp_path):
    dm = tmp_path / "model.docmodel.json"
    dm.write_text(json.dumps({"meta": {"title": "Other"}}))
    assert _title_of("# Zent Steps\n", "f", dm) == "Zent Steps"
