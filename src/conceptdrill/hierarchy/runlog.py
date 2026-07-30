"""The audit record for one hierarchy run.

A run that reports only aggregates cannot be checked afterwards. Row counts,
merge rates and basis sizes are all derived numbers; when one of them looks
wrong there is no way back to the text that produced it. This module writes the
inputs and the per-section decisions alongside the totals, so every number in a
report has a line of JSON behind it.

    run-<timestamp>-<git-sha>/
        manifest.json    what produced this run: code, models, environment
        sections.jsonl   one line per section IN THE INPUT TREE, skips included
        basis.json       the rows, each naming the sections that built it

Two properties are enforced in code rather than left to the caller:

* **Every field is present on every record.** `null` is a measurement (the
  section had no label); a missing key is an unanswered question. `section_record`
  builds each line from `SECTION_FIELDS` exactly, so a typo raises instead of
  silently producing a record with a hole in it.

* **Every section in the tree gets a line.** Skipped, failed and absorbed
  sections are recorded with a reason. `RunLog.finish` refuses to write when the
  line count and the tree's section count disagree, because a run that cannot
  account for its input is not evidence.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

FORMAT = "conceptdrill.hierarchy.run"
FORMAT_VERSION = 1

#: Every key on every `sections.jsonl` line, in write order. The contract.
SECTION_FIELDS: tuple[str, ...] = (
    "doc_id", "section_id", "level", "flow_index", "is_appendix",
    "title_raw", "title_cleaned", "cleaning_rules_fired",
    "structural_class", "structural_rule_fired",
    "tier_label", "tier_abstraction", "tier_summary", "basis_text",
    "embedding_model", "embedding_revision",
    "row_id_assigned", "merge_decision", "merge_cosine", "merge_target_row_id",
    "warnings", "error",
)

#: Every key in `manifest.json`.
MANIFEST_FIELDS: tuple[str, ...] = (
    "format", "format_version", "run_id", "started_at", "finished_at",
    "summarizer_class", "embedder_backend", "embedder_resolved_revision",
    "nlp_backend", "caption_cleaner_tier", "mathtext_source_counts",
    "tau", "git_sha", "git_dirty", "gemm_check_result", "torch_threads",
    "blas_build", "strict_mode", "corpus_paths", "section_count", "doc_count",
)

#: Manifest keys a run may never leave unanswered. A null here means the run
#: does not know what produced it, which makes its numbers uninterpretable.
MANIFEST_REQUIRED: tuple[str, ...] = (
    "summarizer_class", "embedder_backend", "gemm_check_result", "strict_mode",
)

#: `merge_decision` vocabulary.
#:
#: `absorbed` is distinct from `not_integrated` on purpose: a section absorbed
#: into the reserved structural row *was* processed and *did* reach the basis,
#: it simply did not become a concept. Collapsing the two would make it
#: impossible to tell dimension zero from a pipeline failure.
MERGE_DECISIONS = frozenset({"added", "merged", "absorbed", "skipped",
                             "not_integrated"})


class IncompleteRun(RuntimeError):
    """Raised when a run cannot account for its own input."""


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------

def git_state(repo: Optional[str | Path] = None) -> tuple[str, Optional[bool]]:
    """`(short sha, dirty)`. `("unknown", None)` outside a work tree.

    The dirty flag matters as much as the sha: a run from a modified tree is
    not reproducible from its commit, and saying so is cheaper than discovering
    it later.
    """
    cwd = str(repo) if repo else str(Path(__file__).resolve().parents[3])
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=cwd, capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"],
                                cwd=cwd, capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
        return (sha or "unknown"), bool(status)
    except Exception:
        return "unknown", None


def gemm_state() -> dict[str, Any]:
    """Measured float32 GEMM error per library, and a verdict.

    Measured here rather than trusted from `setup.sh`: the fault this guards
    against is intermittent, so a check that ran in a different process at a
    different time says nothing about this one.
    """
    out: dict[str, Any] = {"errors": {}, "tolerance": None, "verdict": "unknown"}
    try:
        import blasfix
    except Exception as exc:
        out["verdict"] = f"unavailable: {type(exc).__name__}"
        return out

    out["tolerance"] = getattr(blasfix, "TOLERANCE", None)
    measured = []
    for lib in ("numpy", "torch"):
        try:
            err = blasfix.gemm_error(lib)
        except Exception as exc:
            err = None
            out["errors"][f"{lib}_error"] = f"{type(exc).__name__}: {exc}"
        out["errors"][lib] = err
        if err is not None:
            measured.append(err)

    if not measured:
        out["verdict"] = "unknown"
    elif out["tolerance"] is not None and max(measured) <= out["tolerance"]:
        out["verdict"] = "green"
    else:
        out["verdict"] = "red"
    out["worst"] = max(measured) if measured else None
    return out


def torch_state() -> dict[str, Any]:
    out: dict[str, Any] = {"threads": None, "version": None, "available": False}
    try:
        import torch
        out.update(available=True, threads=torch.get_num_threads(),
                   version=torch.__version__)
    except Exception:
        pass
    return out


def blas_build() -> dict[str, Any]:
    """Which BLAS numpy actually linked, plus the active `LD_PRELOAD`.

    `LD_PRELOAD` is part of the answer on this host: the mitigation for the
    AVX2 fault is a preloaded non-AVX2 kernel, so a run's numerics depend on an
    environment variable that nothing else records.
    """
    out: dict[str, Any] = {
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "platform": platform.platform(),
        "numpy_version": None, "libraries": [],
    }
    try:
        import numpy as np
        out["numpy_version"] = np.__version__
        info = getattr(np, "__config__", None)
        cfg = info.show(mode="dicts") if info and hasattr(info, "show") else None
        if isinstance(cfg, dict):
            deps = cfg.get("Build Dependencies", {})
            for name, detail in (deps or {}).items():
                if isinstance(detail, dict):
                    out["libraries"].append(
                        {"name": name, "found": detail.get("name"),
                         "version": detail.get("version")})
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

def section_record(**values: Any) -> dict[str, Any]:
    """One `sections.jsonl` line: exactly `SECTION_FIELDS`, nothing else.

    Unknown keys raise. Missing keys become `None`. This is what makes "null is
    allowed, absent is not" a property of the code rather than a convention the
    caller is asked to remember.
    """
    unknown = set(values) - set(SECTION_FIELDS)
    if unknown:
        raise KeyError(f"not part of the section record contract: "
                       f"{sorted(unknown)}")
    decision = values.get("merge_decision")
    if decision is not None and decision not in MERGE_DECISIONS:
        raise ValueError(f"merge_decision {decision!r} not in {sorted(MERGE_DECISIONS)}")
    return {name: values.get(name) for name in SECTION_FIELDS}


@dataclass
class RunLog:
    """Accumulates a run's records and writes the directory at the end."""

    root: Path
    run_id: str
    started_at: str
    _sections: list[dict[str, Any]] = field(default_factory=list)
    _expected: int = 0

    @classmethod
    def open(cls, parent: str | Path, *, timestamp: Optional[str] = None,
             repo: Optional[str | Path] = None) -> "RunLog":
        stamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sha, _ = git_state(repo)
        run_id = f"run-{stamp}-{sha}"
        root = Path(parent) / run_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, run_id=run_id,
                   started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ---- accumulation ---------------------------------------------------

    def expect(self, n: int) -> None:
        """Declare how many sections the input tree holds. Checked at write."""
        self._expected += int(n)

    def add_section(self, **values: Any) -> dict[str, Any]:
        record = section_record(**values)
        self._sections.append(record)
        return record

    def __len__(self) -> int:
        return len(self._sections)

    # ---- writing --------------------------------------------------------

    def finish(self, *, summarizer_class: str, embedder_backend: str,
               embedder_resolved_revision: Optional[str],
               nlp_backend: Optional[str], tau: Optional[float],
               strict_mode: bool, corpus_paths: Sequence[str],
               doc_count: int, basis_rows: Iterable[Mapping[str, Any]],
               mathtext_source_counts: Optional[Mapping[str, int]] = None,
               repo: Optional[str | Path] = None,
               extra: Optional[Mapping[str, Any]] = None) -> Path:
        """Write the three files. Raises when the run cannot account for itself."""
        from .captions import caption_cleaner_tier

        if self._expected and len(self._sections) != self._expected:
            raise IncompleteRun(
                f"{len(self._sections)} section records for {self._expected} "
                f"sections in the input trees; a run that drops sections is "
                f"not auditable")

        seen: dict[tuple[Any, Any], int] = {}
        for rec in self._sections:
            key = (rec["doc_id"], rec["section_id"])
            seen[key] = seen.get(key, 0) + 1
        duplicates = sorted(k for k, n in seen.items() if n > 1)
        if duplicates:
            raise IncompleteRun(f"duplicate section records: {duplicates[:5]}")

        sha, dirty = git_state(repo)
        gemm = gemm_state()
        torch_info = torch_state()

        manifest = {
            "format": FORMAT, "format_version": FORMAT_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summarizer_class": summarizer_class,
            "embedder_backend": embedder_backend,
            "embedder_resolved_revision": embedder_resolved_revision,
            "nlp_backend": nlp_backend,
            "caption_cleaner_tier": caption_cleaner_tier(),
            "mathtext_source_counts": dict(mathtext_source_counts or {}),
            "tau": tau,
            "git_sha": sha, "git_dirty": dirty,
            "gemm_check_result": gemm,
            "torch_threads": torch_info.get("threads"),
            "blas_build": blas_build(),
            # Not `bool(strict_mode)`: coercing None to False would answer
            # "were fallbacks allowed?" with a guess, in the one field whose
            # job is to say that no guessing happened.
            "strict_mode": strict_mode if isinstance(strict_mode, bool) else None,
            "corpus_paths": list(corpus_paths),
            "section_count": len(self._sections),
            "doc_count": int(doc_count),
        }
        if extra:
            manifest.update({k: v for k, v in extra.items()
                             if k not in MANIFEST_FIELDS})

        missing = [k for k in MANIFEST_REQUIRED if manifest.get(k) is None]
        if missing:
            raise IncompleteRun(f"manifest fields must not be null: {missing}")

        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")

        with (self.root / "sections.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self._sections:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        (self.root / "basis.json").write_text(
            json.dumps({"format": FORMAT, "format_version": FORMAT_VERSION,
                        "run_id": self.run_id,
                        "rows": [dict(r) for r in basis_rows]},
                       indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")
        return self.root
