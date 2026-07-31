"""The audit record for one hierarchy run.

A run that reports only aggregates cannot be checked afterwards. Row counts,
merge rates and basis sizes are all derived numbers; when one of them looks
wrong there is no way back to the text that produced it. This module writes the
inputs and the per-span decisions alongside the totals, so every number in a
report has a line of JSON behind it.

    run-<timestamp>-<git-sha>/
        manifest.json    what produced this run: code, models, environment
        spans.jsonl   one line per span IN THE INPUT TREE, skips included
        basis.json       the rows, each naming the markers that built it

Two properties are enforced in code rather than left to the caller:

* **Every field is present on every record.** `null` is a measurement (the
  span had no label); a missing key is an unanswered question. `span_record`
  builds each line from `SPAN_FIELDS` exactly, so a typo raises instead of
  silently producing a record with a hole in it.

* **Every span in the tree gets a line.** Skipped, failed and absorbed
  markers are recorded with a reason. `RunLog.finish` refuses to write when the
  line count and the tree's span count disagree, because a run that cannot
  account for its input is not evidence.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

FORMAT = "conceptdrill.hierarchy.run"
FORMAT_VERSION = 1

#: Every key on every `spans.jsonl` line, in write order. The contract.
#:
#: One line per SPAN. A span is the content between one section marker and the
#: next marker at any level -- the unit that is summarised and embedded. A
#: marker never carries text; it opens a span. But the CONCEPT, not the span, is the unit that
#: becomes a basis row: a span defining three ideas yields three vectors.
#: Everything that is a property of a concept lives in `concepts`, and
#: everything that is a property of the span stays here.
SPAN_FIELDS: tuple[str, ...] = (
    "doc_id", "span_id", "marker_id", "level", "flow_index", "is_appendix",
    "title_raw", "title_cleaned", "cleaning_rules_fired",
    "structural_class", "structural_rule_fired",
    "derivation", "own_text_chars",
    "concept_count", "concepts",
    "warnings", "error",
)

#: Every key on every entry of a span's `concepts` list.
#:
#: Same rule as `SPAN_FIELDS`: unknown keys raise, missing keys become null.
#: A concept is what gets embedded, integrated and counted, so this is where
#: the embedding and merge decisions live.
CONCEPT_FIELDS: tuple[str, ...] = (
    "concept_index",
    "tier_label", "tier_abstraction", "tier_summary", "basis_text",
    "cleaning_rules_fired",
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
    "blas_build", "strict_mode", "corpus_paths", "span_count", "doc_count",
)

#: Manifest keys a run may never leave unanswered. A null here means the run
#: does not know what produced it, which makes its numbers uninterpretable.
MANIFEST_REQUIRED: tuple[str, ...] = (
    "summarizer_class", "embedder_backend", "gemm_check_result", "strict_mode",
)

#: `merge_decision` vocabulary.
#:
#: `absorbed` is distinct from `not_integrated` on purpose: a span absorbed
#: into the reserved structural row *was* processed and *did* reach the basis,
#: it simply did not become a concept. Collapsing the two would make it
#: impossible to tell dimension zero from a pipeline failure.
MERGE_DECISIONS = frozenset({"added", "merged", "absorbed", "skipped",
                             "not_integrated"})

#: Where a span's summarised text came from.
#:
#: `own_text` -- the paragraphs belonging to this heading and no deeper one.
#: That is the unit: given `\span{A} P0 \subsection{B} P1 \subsection{C} P2`
#: there are three units, A's own P0 plus B plus C, and A is not P0+P1+P2.
#:
#: `empty` -- a heading running straight into its first subsection, with no
#: paragraphs of its own. Nothing to summarise; a title is not a concept. It is
#: recorded rather than dropped, because a run must account for every span.
DERIVATIONS = frozenset({"own_text", "empty"})


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

def concept_record(**values: Any) -> dict[str, Any]:
    """One entry of a span's `concepts` list: exactly `CONCEPT_FIELDS`."""
    unknown = set(values) - set(CONCEPT_FIELDS)
    if unknown:
        raise KeyError(f"not part of the concept record contract: "
                       f"{sorted(unknown)}")
    decision = values.get("merge_decision")
    if decision is not None and decision not in MERGE_DECISIONS:
        raise ValueError(f"merge_decision {decision!r} not in "
                         f"{sorted(MERGE_DECISIONS)}")
    return {name: values.get(name) for name in CONCEPT_FIELDS}


def span_record(**values: Any) -> dict[str, Any]:
    """One `spans.jsonl` line: exactly `SPAN_FIELDS`, nothing else.

    Unknown keys raise. Missing keys become `None`. This is what makes "null is
    allowed, absent is not" a property of the code rather than a convention the
    caller is asked to remember.
    """
    unknown = set(values) - set(SPAN_FIELDS)
    if unknown:
        raise KeyError(f"not part of the span record contract: "
                       f"{sorted(unknown)}")

    derivation = values.get("derivation")
    if derivation is not None and derivation not in DERIVATIONS:
        raise ValueError(f"derivation {derivation!r} not in {sorted(DERIVATIONS)}")

    concepts = values.get("concepts")
    if concepts is not None:
        if not isinstance(concepts, (list, tuple)):
            raise TypeError("concepts must be a list of concept records")
        # Rebuild each entry through the concept contract, so a hole or a typo
        # in a nested record raises here rather than reaching the artefact.
        concepts = [concept_record(**dict(c)) for c in concepts]
        values = {**values, "concepts": concepts,
                  "concept_count": values.get("concept_count", len(concepts))}
    return {name: values.get(name) for name in SPAN_FIELDS}


@dataclass
class RunLog:
    """Accumulates a run's records and writes the directory at the end."""

    root: Path
    run_id: str
    started_at: str
    #: Where `root` is renamed to on a successful finish. None means write in
    #: place. See `open` for why this exists.
    final_root: Optional[Path] = None
    _spans: list[dict[str, Any]] = field(default_factory=list)
    _expected: int = 0

    @classmethod
    def open(cls, parent: str | Path, *, timestamp: Optional[str] = None,
             repo: Optional[str | Path] = None,
             name: Optional[str] = None) -> "RunLog":
        """Create the run directory.

        `name` pins the directory to a fixed path so the current run is always
        findable without knowing a timestamp. `run_id` inside `manifest.json`
        still carries the stamp and sha, so provenance is not lost by giving
        the directory a stable name.
        """
        stamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sha, _ = git_state(repo)
        run_id = f"run-{stamp}-{sha}"

        final = Path(parent) / (name or run_id)
        # Build in a sibling directory and swap at the end. Writing in place
        # would empty the output for the whole length of a run -- forty
        # minutes of an LLM corpus build during which the documented path
        # holds nothing, so anyone looking finds an empty directory and
        # concludes the program produced nothing. The previous complete run
        # stays readable until a new complete one replaces it.
        root = final.with_name(final.name + ".partial")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, run_id=run_id,
                   started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   final_root=final)

    # ---- accumulation ---------------------------------------------------

    def expect(self, n: int) -> None:
        """Declare how many markers the input tree holds. Checked at write."""
        self._expected += int(n)

    def add_span(self, **values: Any) -> dict[str, Any]:
        record = span_record(**values)
        self._spans.append(record)
        return record

    def __len__(self) -> int:
        return len(self._spans)

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

        if self._expected and len(self._spans) != self._expected:
            raise IncompleteRun(
                f"{len(self._spans)} span records for {self._expected} "
                f"markers in the input trees; a run that drops markers is "
                f"not auditable")

        seen: dict[tuple[Any, Any], int] = {}
        for rec in self._spans:
            key = (rec["doc_id"], rec["span_id"])
            seen[key] = seen.get(key, 0) + 1
        duplicates = sorted(k for k, n in seen.items() if n > 1)
        if duplicates:
            raise IncompleteRun(f"duplicate span records: {duplicates[:5]}")

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
            "span_count": len(self._spans),
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

        with (self.root / "spans.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self._spans:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        (self.root / "basis.json").write_text(
            json.dumps({"format": FORMAT, "format_version": FORMAT_VERSION,
                        "run_id": self.run_id,
                        "rows": [dict(r) for r in basis_rows]},
                       indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")

        # Swap only now that every file is written. An interrupted run leaves
        # <name>.partial behind and the last good <name> untouched.
        if self.final_root is not None:
            if self.final_root.exists():
                shutil.rmtree(self.final_root)
            self.root.rename(self.final_root)
            self.root = self.final_root
        return self.root
