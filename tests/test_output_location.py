"""Every run writes inside the project, under ONE setting.

Owner, 2026-11-01 (T-0117): runs land in `<repo>/work/output/<run>/`, not in
`~/conceptdrill-corpus-llm`, which sits outside the project folder. The final
location will be given later, so the root is one setting -- `output_root()`,
overridable by `CONCEPTDRILL_OUTPUT_ROOT` or a CLI option -- and moving it is a
one-line change to `DEFAULT_OUTPUT_ROOT`.

The path had been written out by hand in twelve places. A thirteenth would
silently write outside the project again, so the last test refuses the literal.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conceptdrill.output import (DEFAULT_OUTPUT_ROOT, OUTPUT_ROOT_ENV,
                                 REPO_ROOT, output_root)

OLD = "conceptdrill-corpus-llm"


def test_the_default_root_is_work_output_inside_the_project(monkeypatch):
    monkeypatch.delenv(OUTPUT_ROOT_ENV, raising=False)
    assert output_root() == DEFAULT_OUTPUT_ROOT == REPO_ROOT / "work" / "output"
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_the_environment_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "elsewhere"))
    assert output_root() == tmp_path / "elsewhere"


def test_an_explicit_root_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "env"))
    assert output_root(tmp_path / "cli") == tmp_path / "cli"


def test_nothing_resolves_outside_the_project(monkeypatch):
    monkeypatch.delenv(OUTPUT_ROOT_ENV, raising=False)
    assert OLD not in str(output_root())


@pytest.mark.parametrize("glob", ["src/**/*.py", "tools/*.py", "tools/*.sh",
                                  "experiments/**/*.py"])
def test_no_code_names_the_old_output_folder(glob):
    offenders = [str(p.relative_to(REPO_ROOT))
                 for p in REPO_ROOT.glob(glob) if OLD in p.read_text()]
    assert offenders == []
