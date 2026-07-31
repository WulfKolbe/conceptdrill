r"""Loading a spoken-maths backend without depending on one.

`la2speech` drives SRE (`speech-rule-engine`, the engine inside MathJax) and
turns LaTeX into text that says what the maths *means*:

    \hat{F}_{0}(X ; Y)
      fallback  ->  hat F 0 (X ; Y)
      SRE       ->  F hat sub 0 of open paren X semicolon Y close paren

    \sum_{y \in Y} p(y) \log p(y)
      fallback  ->  the sum of y in Y p(y) log p(y)
      SRE       ->  the sum over y is a member of Y of p of y log p of y

The fallback drops structure: `|p|` becomes `p`, `\tau_{k}` becomes `tau k`
which reads as a product, and set membership disappears. Those distinctions
are the content of the formula.

**Loaded by path, never imported.** ConceptDrill must not acquire an import
dependency on a neighbouring project — formats are reproduced, not imported.
`load_speaker` reads a module from a filesystem path given by the caller or by
`CONCEPTDRILL_LA2SPEECH`, and returns something satisfying the `SpeechBackend`
protocol `mathtext` already expects. If it cannot, it raises: a run that
silently fell back would report `fallback` counts indistinguishable from a run
that never asked for speech.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

#: Where la2speech lives, unless told otherwise.
DEFAULT_PATH = str(Path.home() / "la2speech")

#: Subdirectory holding `sre_bridge.js` and `node_modules/`.
BRIDGE_SUBDIR = "sre"


class SpeechUnavailable(RuntimeError):
    """The requested spoken-maths backend could not be constructed."""


def _load_module(project: Path):
    entry = project / "latex2speech.py"
    if not entry.exists():
        raise SpeechUnavailable(f"no latex2speech.py under {project}")
    spec = importlib.util.spec_from_file_location("la2speech_latex2speech", entry)
    if spec is None or spec.loader is None:
        raise SpeechUnavailable(f"cannot load {entry}")
    module = importlib.util.module_from_spec(spec)
    # The module imports its own siblings, so its directory must be importable
    # for the duration of the load.
    added = str(project) not in sys.path
    if added:
        sys.path.insert(0, str(project))
    # Registered BEFORE exec: @dataclass resolves its own module through
    # sys.modules, and a module absent from it fails with a bare AttributeError
    # on NoneType that says nothing about the cause.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise SpeechUnavailable(f"{entry} failed to import: "
                                f"{type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            sys.path.remove(str(project))
    return module


def load_speaker(path: Optional[str] = None, *,
                 node_dir: Optional[str] = None) -> Any:
    """A `SpeechBackend` backed by SRE, or raise.

    Never returns a degraded object. `mathtext` already falls back to its own
    renderer when no speaker is supplied, so a speaker that quietly did
    nothing would be indistinguishable from no speaker at all in
    `mathtext_source_counts` — the one place the difference is visible.
    """
    project = Path(path or os.environ.get("CONCEPTDRILL_LA2SPEECH")
                   or DEFAULT_PATH).expanduser()
    if not project.is_dir():
        raise SpeechUnavailable(f"no la2speech project at {project}")

    module = _load_module(project)
    for name in ("SRESpeechBackend", "LatexSpeaker"):
        if not hasattr(module, name):
            raise SpeechUnavailable(f"{name} missing from {project}")

    bridge_dir = Path(node_dir) if node_dir else project / BRIDGE_SUBDIR
    if not (bridge_dir / "sre_bridge.js").exists():
        raise SpeechUnavailable(
            f"sre_bridge.js not found under {bridge_dir}; SRE needs the "
            f"directory holding the bridge and node_modules/")

    try:
        backend = module.SRESpeechBackend(node_dir=str(bridge_dir))
        speaker = module.LatexSpeaker(backend=backend)
    except Exception as exc:
        raise SpeechUnavailable(f"could not construct the SRE backend: "
                                f"{type(exc).__name__}: {exc}") from exc

    probe = speaker.speak_math(r"\hat{F}_{0}(X)")
    if not probe or "hat" not in probe.lower():
        raise SpeechUnavailable(
            f"the backend loaded but produced {probe!r} for a known input; "
            f"refusing to run rather than fill a corpus with it")
    return speaker


def describe(speaker: Any) -> dict[str, Any]:
    """What went into a run, for the manifest."""
    backend = getattr(speaker, "backend", None)
    return {
        "speaker": type(speaker).__name__,
        "backend": type(backend).__name__ if backend is not None else None,
        "node_dir": getattr(backend, "node_dir", None),
        "domain": getattr(backend, "domain", None),
    }
