#!/usr/bin/env python3
"""Make float32 matrix multiplication trustworthy, automatically, everywhere.

Some machines compute float32 GEMM **wrongly** — not merely with different
rounding. On the development host, `a @ b` lands ~2.0 away from the float64
answer where correct float32 rounding is ~1e-4, in both OpenBLAS and Intel MKL,
and only on their AVX2 kernels. Everything downstream (embeddings, cosines,
rankings) is then quietly wrong.

This module fixes it without the caller doing anything:

    import blasfix; blasfix.ensure_sane_blas()   # BEFORE importing numpy/torch
    import numpy, torch

Escalating, cheapest first, stopping as soon as the arithmetic is correct:

    0. nothing            most machines are fine; this costs one small matmul
    1. env mitigations    MKL_CBWR / thread caps, set before any BLAS loads
    2. re-exec once       with LD_PRELOAD pointing at a non-AVX2 BLAS
    3. warn and continue  never refuse to run

Design rules, learned the hard way:

  * **It never exits.** An earlier version refused to run and told the user to
    set environment variables. That is not a program that runs everywhere.
  * **It never raises.** Any failure inside the fixer degrades to "unfixed",
    because a broken fixer must not break the program.
  * **It is a no-op on healthy machines.** One 256x768 matmul, then done.
  * **numpy and torch are checked separately.** They can link different BLAS
    backends with different faults: MKL_CBWR fixes torch and does nothing for
    numpy's OpenBLAS.
"""
from __future__ import annotations

import glob
import os
import sys

#: Correct float32 GEMM at the probe size errs ~1e-4. This is two orders above
#: that — comfortably clear of rounding, far below a real fault (~1.0).
TOLERANCE = 1e-2

#: Guards against an infinite re-exec loop.
REEXEC_FLAG = "BLASFIX_REEXECED"

#: Set only if absent, so an explicit choice by the caller always wins.
ENV_MITIGATIONS = (
    ("MKL_CBWR", "COMPATIBLE"),      # Intel MKL: force a reproducible SSE path
)

#: Globs for a BLAS built without AVX2. On the affected host the AVX2 kernel is
#: the broken one and these older builds are correct.
FALLBACK_BLAS_GLOBS = (
    "/usr/lib64/libopenblas_core2*.so*",
    "/usr/lib/x86_64-linux-gnu/libopenblas_core2*.so*",
    "/usr/lib64/libopenblas_nehalem*.so*",
    "/usr/lib/x86_64-linux-gnu/libopenblas_nehalem*.so*",
)


def apply_env_mitigations() -> list[str]:
    """Set variables that only bite before the BLAS is loaded. Returns applied."""
    applied = []
    for key, value in ENV_MITIGATIONS:
        if not os.environ.get(key):
            os.environ[key] = value
            applied.append(f"{key}={value}")
    return applied


def gemm_error(lib: str = "numpy", n: int = 256, k: int = 768,
               trials: int = 6) -> float | None:
    """Worst deviation of a float32 product from the float64 answer.

    None if the library is unavailable. Repeated because the fault is
    intermittent — one clean product proves nothing.
    """
    # One try around everything, including data setup and the import. A fixer
    # that raises is worse than no fixer, and a broken numpy must read as
    # "unknown" rather than take the program down.
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        truth = a.astype(np.float64) @ b.astype(np.float64)

        if lib == "torch":
            import torch
            ta, tb = torch.from_numpy(a), torch.from_numpy(b)
            product = lambda: (ta @ tb).numpy().astype(np.float64)   # noqa: E731
        else:
            product = lambda: (a @ b).astype(np.float64)             # noqa: E731

        worst = 0.0
        for _ in range(max(1, trials)):
            worst = max(worst, float(abs(product() - truth).max()))
            if worst > TOLERANCE:
                break
        return worst
    except Exception:
        return None


def find_fallback_blas() -> str | None:
    """A non-AVX2 BLAS to LD_PRELOAD, or None if this system has none."""
    for pattern in FALLBACK_BLAS_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                return path
    return None


def _reexec_with(preload: str) -> None:
    """Restart this process with LD_PRELOAD set. Does not return on success."""
    env = dict(os.environ)
    existing = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    env[REEXEC_FLAG] = "1"
    try:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    except Exception:
        # Could not re-exec (frozen binary, restricted platform, ...).
        # Fall through; the caller will warn and continue.
        pass


def ensure_sane_blas(*, allow_reexec: bool = True, verbose: bool = True,
                     stream=None) -> dict:
    """Make float32 GEMM correct if it is not. Never exits, never raises.

    Call BEFORE importing numpy or torch — the env mitigations and LD_PRELOAD
    only take effect while the BLAS is still unloaded.

    Returns a report: ``{status, checked, applied, preload, errors}`` where
    status is ``ok`` | ``fixed`` | ``unfixable`` | ``unknown``.
    """
    stream = stream or sys.stderr
    report = {"status": "unknown", "checked": [], "applied": [],
              "preload": os.environ.get("LD_PRELOAD", ""), "errors": {}}

    reexeced = bool(os.environ.get(REEXEC_FLAG))
    report["applied"] = apply_env_mitigations()

    errors = {lib: gemm_error(lib) for lib in ("numpy", "torch")}
    report["errors"] = errors
    report["checked"] = [lib for lib, err in errors.items() if err is not None]

    measured = [err for err in errors.values() if err is not None]
    if not measured:
        report["status"] = "unknown"          # no numpy: nothing to check yet
        return report

    if max(measured) <= TOLERANCE:
        report["status"] = "fixed" if (reexeced or report["applied"]) else "ok"
        return report

    # Still wrong. One re-exec with a non-AVX2 BLAS is the remaining lever.
    if allow_reexec and not reexeced:
        fallback = find_fallback_blas()
        if fallback:
            if verbose:
                print(f"blasfix: float32 matmul is wrong "
                      f"(err {max(measured):.2e}); retrying with {fallback}",
                      file=stream)
            _reexec_with(fallback)            # normally does not return

    report["status"] = "unfixable"
    if verbose:
        worst = max(measured)
        print(f"blasfix: WARNING float32 matmul on this machine is off by "
              f"{worst:.2e} versus a float64 reference (correct is ~1e-4).",
              file=stream)
        for lib, err in sorted(errors.items()):
            if err is not None and err > TOLERANCE:
                print(f"         {lib}: {err:.2e}", file=stream)
        print("         Numerical results will be inaccurate. Continuing anyway.",
              file=stream)
        print("         Likely a CPU fault under sustained AVX2/FMA load; "
              "check BIOS defaults, cooling, memtest86+.", file=stream)
    return report


def main(argv=None) -> int:
    """`python3 blasfix.py` — report this machine's status."""
    report = ensure_sane_blas(allow_reexec=False, verbose=False)
    print(f"status : {report['status']}")
    print(f"applied: {', '.join(report['applied']) or 'nothing'}")
    print(f"preload: {report['preload'] or 'none'}")
    for lib, err in sorted(report["errors"].items()):
        shown = "unavailable" if err is None else f"{err:.3e}"
        mark = "" if err is None or err <= TOLERANCE else "   <-- WRONG"
        print(f"  {lib:6s} {shown}{mark}")
    fallback = find_fallback_blas()
    print(f"fallback BLAS available: {fallback or 'none found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
