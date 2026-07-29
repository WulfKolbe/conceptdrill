#!/usr/bin/env python3
"""Print the worst float32 GEMM error of one library against a float64 reference.

    python3 gemm_check.py numpy
    python3 gemm_check.py torch

Correct float32 rounding at this size is ~1e-4. Anything near 1.0 means the
library is returning wrong answers, not rounding differently.

Kept as a standalone file rather than a heredoc inside setup.sh so that both the
setup script and the test suite exercise exactly the same code, and so it can be
run directly when diagnosing a machine.
"""
import sys

import numpy as np

TOLERANCE = 1e-2
TRIALS = 8


def worst_error(lib: str, n: int = 256, k: int = 768, trials: int = TRIALS) -> float:
    """Worst deviation from the float64 product over `trials` repeats.

    Repeated because the fault is intermittent: a single product can come back
    clean on a machine that is wrong a third of the time.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    truth = a.astype(np.float64) @ b.astype(np.float64)

    if lib == "torch":
        import torch
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        product = lambda: (ta @ tb).numpy().astype(np.float64)  # noqa: E731
    elif lib == "numpy":
        product = lambda: (a @ b).astype(np.float64)            # noqa: E731
    else:
        raise ValueError(f"unknown library {lib!r}; expected numpy or torch")

    worst = 0.0
    for _ in range(max(1, trials)):
        worst = max(worst, float(np.abs(product() - truth).max()))
        if worst > TOLERANCE:
            break
    return worst


def main(argv):
    lib = argv[1] if len(argv) > 1 else "numpy"
    try:
        print(f"{worst_error(lib):.3e}")
    except ImportError:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
