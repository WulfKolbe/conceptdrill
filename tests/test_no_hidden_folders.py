"""A test run leaves no hidden folder in the project.

House rule (owner, 2026-10-30): no hidden files. pytest's cache provider wrote
`.pytest_cache/` into the checkout on every run, so it is switched off in
`pyproject.toml`; this runs a real pytest and looks at what appeared.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib

from conceptdrill.output import REPO_ROOT

ALLOWED = {".git", ".claude"}


def _dot_folders():
    return {p.name for p in REPO_ROOT.iterdir() if p.is_dir() and p.name.startswith(".")}


def test_the_cache_provider_is_switched_off_by_config():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "no:cacheprovider" in config["tool"]["pytest"]["ini_options"]["addopts"]


def test_a_test_run_creates_no_hidden_folder():
    before = _dot_folders()
    run = subprocess.run([sys.executable, "-m", "pytest",
                          "tests/test_output_location.py"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    appeared = _dot_folders() - before
    assert appeared - ALLOWED == set()
    assert not (REPO_ROOT / ".pytest_cache").exists()
