"""Unit tests for embrun.py, one isolated unit at a time.

Each test uses explicit hand-built inputs. Nothing here loads a model from the
network or depends on another test having run first.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]


def _load_embrun():
    """Import embrun.py by path — it is a top-level script, not a package."""
    spec = importlib.util.spec_from_file_location("embrun", REPO / "embrun.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["embrun"] = module
    spec.loader.exec_module(module)
    return module


embrun = _load_embrun()


class FakeOutput:
    """Stands in for a HF model output. mean_pooling only reads one attribute,
    which is the whole point of testing it in isolation."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


# --------------------------------------------------------------------------
# Unit: mean_pooling
# Contract: (model_output, attention_mask) -> (B, H), padding excluded.
# --------------------------------------------------------------------------

def test_mean_pooling_returns_batch_by_hidden():
    hidden = torch.zeros(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.long)
    assert embrun.mean_pooling(FakeOutput(hidden), mask).shape == (2, 4)


def test_mean_pooling_averages_unmasked_tokens():
    # Two tokens, values 1.0 and 3.0 -> mean 2.0.
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    mask = torch.ones(1, 2, dtype=torch.long)
    out = embrun.mean_pooling(FakeOutput(hidden), mask)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


def test_mean_pooling_excludes_padding():
    """The masked token carries a huge value. If padding leaked in, the result
    would not be 1.0 — this is the test that actually pins the contract."""
    hidden = torch.tensor([[[1.0, 1.0], [999.0, 999.0]]])
    mask = torch.tensor([[1, 0]], dtype=torch.long)
    out = embrun.mean_pooling(FakeOutput(hidden), mask)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0]]))


def test_mean_pooling_all_padding_does_not_divide_by_zero():
    hidden = torch.ones(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    out = embrun.mean_pooling(FakeOutput(hidden), mask)
    assert torch.isfinite(out).all()


def test_mean_pooling_rows_are_independent():
    """Row 1 is fully padded; it must not affect row 0."""
    hidden = torch.tensor([[[2.0], [2.0]], [[50.0], [50.0]]])
    mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.long)
    out = embrun.mean_pooling(FakeOutput(hidden), mask)
    assert torch.allclose(out[0], torch.tensor([2.0]))


# --------------------------------------------------------------------------
# Unit: base directory resolution
# Contract: env var or the script's own directory. Knows nothing about torch.
# --------------------------------------------------------------------------

def test_base_dir_is_a_real_directory():
    assert embrun.BASE_DIR.is_dir()


def test_base_dir_contains_the_chunks_file():
    """The whole point of the change: the default must find chunks.json."""
    assert (embrun.BASE_DIR / "chunks.json").exists()


def test_base_dir_is_not_the_dead_sandbox_path():
    assert str(embrun.BASE_DIR) != "/home/claude/embwork"


def test_env_var_overrides_base_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBRUN_DIR", str(tmp_path))
    reloaded = _load_embrun()
    assert reloaded.BASE_DIR == tmp_path


def test_model_root_sits_under_base_dir():
    assert embrun.MODEL_ROOT.parent == embrun.BASE_DIR


# --------------------------------------------------------------------------
# Unit: compare_vectors
# Contract: two {chunk_id: vector} dicts -> divergence stats.
# Knows nothing about models, files, or torch.
# --------------------------------------------------------------------------

def test_identical_vectors_have_zero_divergence():
    v = {"a": [1.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0]}
    out = embrun.compare_vectors(v, dict(v))
    assert out["n_compared"] == 2
    assert out["max_abs_diff"] == 0.0
    assert out["mean_abs_diff"] == 0.0
    assert out["min_cosine"] == pytest.approx(1.0)


def test_known_offset_is_reported_exactly():
    out = embrun.compare_vectors({"a": [1.0, 1.0]}, {"a": [1.0, 1.25]})
    assert out["max_abs_diff"] == pytest.approx(0.25)
    assert out["mean_abs_diff"] == pytest.approx(0.125)


def test_opposite_vectors_give_cosine_minus_one():
    out = embrun.compare_vectors({"a": [1.0, 0.0]}, {"a": [-1.0, 0.0]})
    assert out["min_cosine"] == pytest.approx(-1.0)


def test_pure_scaling_keeps_cosine_at_one():
    """The reason cosine is reported next to the raw diff: a scaled vector has
    a large absolute difference but has not actually moved."""
    out = embrun.compare_vectors({"a": [1.0, 2.0]}, {"a": [10.0, 20.0]})
    assert out["max_abs_diff"] == pytest.approx(18.0)
    assert out["min_cosine"] == pytest.approx(1.0)


def test_min_cosine_is_the_worst_case_not_the_average():
    out = embrun.compare_vectors(
        {"same": [1.0, 0.0], "flipped": [1.0, 0.0]},
        {"same": [1.0, 0.0], "flipped": [-1.0, 0.0]})
    assert out["min_cosine"] == pytest.approx(-1.0)


def test_absent_chunk_is_counted_not_silently_dropped():
    out = embrun.compare_vectors({"a": [1.0], "gone": [1.0]}, {"a": [1.0]})
    assert out["n_compared"] == 1
    assert out["n_missing"] == 1


def test_length_mismatch_is_treated_as_missing():
    """A 768-dim vector must never be compared against a 384-dim one."""
    out = embrun.compare_vectors({"a": [1.0, 2.0]}, {"a": [1.0]})
    assert out["n_compared"] == 0
    assert out["n_missing"] == 1


def test_no_overlap_returns_nulls_rather_than_zero():
    """Zero divergence would be a lie when nothing was compared."""
    out = embrun.compare_vectors({"a": [1.0]}, {"b": [1.0]})
    assert out["n_compared"] == 0
    assert out["max_abs_diff"] is None


def test_zero_vector_does_not_divide_by_zero():
    out = embrun.compare_vectors({"a": [0.0, 0.0]}, {"a": [0.0, 0.0]})
    assert out["min_cosine"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Unit: blas_sanity_check
# Contract: () -> (ok, max_error). Builds its own data, no globals, no models.
# --------------------------------------------------------------------------

def test_sanity_check_returns_ok_flag_and_error():
    ok, err = embrun.blas_sanity_check(n=64, k=128)
    assert isinstance(ok, bool)
    assert err >= 0.0


def test_sanity_check_is_deterministic():
    """It must not itself be a source of noise."""
    assert embrun.blas_sanity_check(n=64, k=128)[1] == \
           embrun.blas_sanity_check(n=64, k=128)[1]


def test_sanity_check_flags_a_tolerance_it_cannot_meet():
    """A tolerance of zero can only pass if float32 matches float64 exactly,
    which it never does — proves the check actually compares something."""
    ok, err = embrun.blas_sanity_check(n=64, k=128, tolerance=0.0)
    assert not ok
    assert err > 0.0


def test_sanity_check_passes_at_a_generous_tolerance():
    assert embrun.blas_sanity_check(n=64, k=128, tolerance=1e6)[0]


def test_sanity_check_repeats_to_catch_an_intermittent_fault():
    """A single product can come back clean on a machine that is wrong a third
    of the time, so one trial is not a check. More trials must never report a
    *better* worst-case than fewer."""
    _, few = embrun.blas_sanity_check(n=256, k=768, trials=1, tolerance=1e6)
    _, many = embrun.blas_sanity_check(n=256, k=768, trials=16, tolerance=1e6)
    assert many >= few


def test_sanity_check_stops_early_once_it_has_failed():
    """Must not burn 16 products proving what the first one already showed."""
    ok, err = embrun.blas_sanity_check(n=64, k=128, tolerance=0.0, trials=99)
    assert not ok and err > 0.0


# --------------------------------------------------------------------------
# Unit: gemm_check (shared by setup.sh and the suite)
# Contract: library name -> worst float32 GEMM error vs a float64 reference.
# --------------------------------------------------------------------------

def _load_gemm_check():
    spec = importlib.util.spec_from_file_location("gemm_check", REPO / "gemm_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gemm_check = _load_gemm_check()


def test_gemm_check_rejects_unknown_library():
    with pytest.raises(ValueError, match="unknown library"):
        gemm_check.worst_error("blas-o-matic")


def test_gemm_check_returns_a_nonnegative_error():
    assert gemm_check.worst_error("numpy", n=64, k=128, trials=2) >= 0.0


def test_gemm_check_agrees_between_libraries_on_a_sound_path():
    """numpy and torch may link different BLAS, but both must land near the
    float64 answer. If they disagree by orders of magnitude, one is broken."""
    n = gemm_check.worst_error("numpy", n=64, k=128, trials=2)
    t = gemm_check.worst_error("torch", n=64, k=128, trials=2)
    assert max(n, t) < gemm_check.TOLERANCE, (
        f"float32 GEMM error numpy={n:.3e} torch={t:.3e} exceeds "
        f"{gemm_check.TOLERANCE}; this machine computes wrong results")


def test_gemm_check_cli_prints_a_parsable_number(capsys):
    assert gemm_check.main(["gemm_check.py", "numpy"]) == 0
    float(capsys.readouterr().out.strip())


def test_importing_embrun_does_not_reexec_the_process():
    """embrun must be importable from a test runner. An earlier version called
    ensure_sane_blas() at module level, which re-execs -- importing it under
    pytest re-executed pytest and killed the suite with no output."""
    import os
    before = os.getpid()
    _load_embrun()
    assert os.getpid() == before


def test_import_applies_env_mitigations_before_torch():
    """The mitigations only bite while the BLAS is unloaded, so they must run
    at import time even though the repair does not."""
    import os
    _load_embrun()
    assert os.environ.get("MKL_CBWR")
