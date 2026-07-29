"""Unit tests for blasfix.py.

The contract that matters most is negative: this module must never exit, never
raise, and never block the program. It is a repair, not a gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("blasfix", REPO / "blasfix.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["blasfix"] = mod
    spec.loader.exec_module(mod)
    return mod


blasfix = _load()


# --------------------------------------------------------------------------
# apply_env_mitigations
# --------------------------------------------------------------------------

def test_sets_mkl_cbwr_when_absent(monkeypatch):
    monkeypatch.delenv("MKL_CBWR", raising=False)
    assert "MKL_CBWR=COMPATIBLE" in blasfix.apply_env_mitigations()


def test_never_overrides_an_explicit_choice(monkeypatch):
    """A caller who set it deliberately must win."""
    monkeypatch.setenv("MKL_CBWR", "AUTO")
    assert blasfix.apply_env_mitigations() == []
    import os
    assert os.environ["MKL_CBWR"] == "AUTO"


# --------------------------------------------------------------------------
# gemm_error
# --------------------------------------------------------------------------

def test_gemm_error_is_a_nonnegative_float():
    err = blasfix.gemm_error("numpy", n=64, k=128, trials=2)
    assert err is None or err >= 0.0


def test_gemm_error_returns_none_for_a_missing_library(monkeypatch):
    """An absent torch must read as 'unknown', never as 'zero error'."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert blasfix.gemm_error("torch", n=32, k=64, trials=1) is None


def test_gemm_error_survives_a_broken_backend(monkeypatch):
    """A fixer that raises is worse than no fixer."""
    import numpy as np
    monkeypatch.setattr(np.random, "default_rng",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert blasfix.gemm_error("numpy") is None


# --------------------------------------------------------------------------
# find_fallback_blas
# --------------------------------------------------------------------------

def test_find_fallback_returns_a_real_file_or_none():
    found = blasfix.find_fallback_blas()
    assert found is None or Path(found).is_file()


def test_find_fallback_is_deterministic():
    assert blasfix.find_fallback_blas() == blasfix.find_fallback_blas()


def test_find_fallback_returns_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(blasfix, "FALLBACK_BLAS_GLOBS", ("/nonexistent/*.so",))
    assert blasfix.find_fallback_blas() is None


# --------------------------------------------------------------------------
# ensure_sane_blas — the contract
# --------------------------------------------------------------------------

def test_never_exits_even_when_unfixable(monkeypatch, capsys):
    """The whole point: an unrepairable machine must still run the program."""
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 99.0)
    monkeypatch.setattr(blasfix, "find_fallback_blas", lambda: None)
    report = blasfix.ensure_sane_blas(allow_reexec=False)
    assert report["status"] == "unfixable"
    assert "Continuing anyway" in capsys.readouterr().err


def test_reports_ok_on_a_sound_machine(monkeypatch):
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 1e-5)
    monkeypatch.setattr(blasfix, "apply_env_mitigations", lambda: [])
    assert blasfix.ensure_sane_blas(allow_reexec=False)["status"] == "ok"


def test_reports_fixed_when_a_mitigation_was_applied(monkeypatch):
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 1e-5)
    monkeypatch.setattr(blasfix, "apply_env_mitigations", lambda: ["MKL_CBWR=COMPATIBLE"])
    assert blasfix.ensure_sane_blas(allow_reexec=False)["status"] == "fixed"


def test_unknown_when_numpy_is_unavailable(monkeypatch):
    """No measurement is not the same as a good measurement."""
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: None)
    assert blasfix.ensure_sane_blas(allow_reexec=False)["status"] == "unknown"


def test_does_not_reexec_twice(monkeypatch):
    """The guard flag must stop an infinite restart loop."""
    calls = []
    monkeypatch.setenv(blasfix.REEXEC_FLAG, "1")
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 99.0)
    monkeypatch.setattr(blasfix, "find_fallback_blas", lambda: "/tmp/fake.so")
    monkeypatch.setattr(blasfix, "_reexec_with", lambda p: calls.append(p))
    blasfix.ensure_sane_blas(allow_reexec=True, verbose=False)
    assert calls == []


def test_reexecs_once_when_repairable(monkeypatch):
    calls = []
    monkeypatch.delenv(blasfix.REEXEC_FLAG, raising=False)
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 99.0)
    monkeypatch.setattr(blasfix, "find_fallback_blas", lambda: "/tmp/fake.so")
    monkeypatch.setattr(blasfix, "_reexec_with", lambda p: calls.append(p))
    blasfix.ensure_sane_blas(allow_reexec=True, verbose=False)
    assert calls == ["/tmp/fake.so"]


def test_reexec_failure_degrades_to_a_warning(monkeypatch, capsys):
    """os.execve can fail (frozen binary, restricted platform). The program
    must carry on rather than die inside the repair."""
    monkeypatch.delenv(blasfix.REEXEC_FLAG, raising=False)
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 99.0)
    monkeypatch.setattr(blasfix, "find_fallback_blas", lambda: "/tmp/fake.so")
    monkeypatch.setattr(blasfix, "_reexec_with", lambda p: None)  # returns = failed
    report = blasfix.ensure_sane_blas(allow_reexec=True)
    assert report["status"] == "unfixable"
    assert "WARNING" in capsys.readouterr().err


def test_quiet_mode_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 99.0)
    monkeypatch.setattr(blasfix, "find_fallback_blas", lambda: None)
    blasfix.ensure_sane_blas(allow_reexec=False, verbose=False)
    assert capsys.readouterr().err == ""


def test_report_shape_is_stable(monkeypatch):
    monkeypatch.setattr(blasfix, "gemm_error", lambda *a, **k: 1e-5)
    report = blasfix.ensure_sane_blas(allow_reexec=False, verbose=False)
    assert set(report) == {"status", "checked", "applied", "preload", "errors"}


def test_cli_reports_without_reexecuting(capsys):
    assert blasfix.main([]) == 0
    out = capsys.readouterr().out
    assert "status" in out and "fallback BLAS" in out
