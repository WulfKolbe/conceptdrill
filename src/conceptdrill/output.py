"""Where runs write: one setting, inside the project.

House rule (owner, 2026-10-30): no files outside the project folder. Runs had
defaulted to a folder in the home directory, written out by hand in twelve tools.
They now all ask `output_root()`, so the final location the owner gives later
is a change to `DEFAULT_OUTPUT_ROOT` and nothing else (T-0117).

Precedence: an explicit path (a tool's CLI option), then the
`CONCEPTDRILL_OUTPUT_ROOT` environment variable, then the default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

#: The checkout this module was imported from. Editable installs only: a copy
#: installed into site-packages has no project folder to write into.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The one config value. Visible and git-ignored (`work/` in .gitignore).
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "output"

OUTPUT_ROOT_ENV = "CONCEPTDRILL_OUTPUT_ROOT"


def output_root(explicit: Optional[Union[str, Path]] = None) -> Path:
    """The directory run directories are created under."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(OUTPUT_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_OUTPUT_ROOT
