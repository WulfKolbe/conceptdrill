#!/usr/bin/env python3
"""Adapted from the user's example script.

Contract:
  load_local_model(dir)            -> (tokenizer, model) in eval mode
  mean_pooling(out, attn_mask)     -> (B, H) float32, pad tokens excluded
  embed_chunks(tok, model, texts)  -> (N, H) float32 tensor
  inspect_model(name, spec)        -> dict report entry (no side effects on other models)

Deviations from the original, forced by this sandbox:
  BASE_DIR is /home/claude/embwork, not /home/oai/share
  disk budget is ~2.9 GB, not 16 GB -> HF cache entry purged after each model
"""
import json
import os
import shutil
import sys
from pathlib import Path

# CHANGE 6 — self-healing numerics. MUST come before numpy/torch are imported:
# the mitigations it applies (MKL_CBWR, and a re-exec with LD_PRELOAD) only take
# effect while the BLAS is still unloaded. On a healthy machine this costs one
# small matmul and does nothing else. It never exits and never raises, so the
# script runs everywhere -- see the CHANGE 3/4 notes for what it is guarding.
# Split deliberately. The env mitigations MUST run before torch is imported, but
# the repair must NOT run at import time: `ensure_sane_blas` may re-exec the
# process, and importing this module from a test runner would then re-exec
# pytest itself. (It did. The suite died with no output.) The re-exec therefore
# happens in main(), where re-entering the process is safe and expected.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blasfix                                                     # noqa: E402

blasfix.apply_env_mitigations()

import torch                                                       # noqa: E402
from transformers import AutoTokenizer, AutoModel                  # noqa: E402

# CHANGE 3 — pins threads, but READ THIS: it does not make the run reproducible.
#
# This script does not reproduce itself on this machine. Two runs diverge by
# ~1.8e-3, which is larger than the divergence from the reference result.json
# produced under a different torch version. My first two explanations were both
# wrong: it is neither thread-scheduling reduction order nor SDPA kernel
# selection. Measured cause:
#
#   float32 matmul on this machine is non-deterministic AND numerically wrong.
#   Repeating `a @ b` on identical tensors in one process, single-threaded,
#   returns different results in 0-100% of trials depending on the process,
#   with deviations up to 4.25 (torch) and 6.92 (numpy) on values of magnitude
#   ~27. float64 matmul, sum(), and elementwise ops are all clean, and memory
#   readback is clean, so this is not bad RAM.
#
#   Both numpy and torch link OpenBLAS 0.3.26, built with target CORE2 and an
#   empty DYNAMIC_ARCH. That is a Core2-era SGEMM kernel running on a modern
#   AMD CPU. Reproduce with:
#       python3 -c "import torch;a=torch.randn(512,768);w=torch.randn(768,768);\
#       r=(a@w).clone();print(sum(1 for _ in range(100) if not torch.equal(a@w,r)))"
#
# Until the BLAS is replaced, every float32 embedding produced here carries
# that error. Pinning threads is kept because it reduces variance and costs
# little, NOT because it fixes the problem. Set EMBRUN_THREADS=0 to opt out.
_threads = int(os.environ.get("EMBRUN_THREADS", "1"))
if _threads > 0:
    torch.set_num_threads(_threads)

# CHANGE 1 — the only thing that stopped this running here.
# Was: Path("/home/claude/embwork"), a sandbox path absent on this machine.
# Now: this script's own directory, which is where chunks.json already lives.
# Override with EMBRUN_DIR to point it elsewhere.
BASE_DIR = Path(os.environ.get("EMBRUN_DIR") or Path(__file__).resolve().parent)
MODEL_ROOT = BASE_DIR / "models"
HF_CACHE = BASE_DIR / "hf-cache"

MODEL_SPECS = {
    "bert": {
        "repo": "bert-base-uncased",
        "local_dir": MODEL_ROOT / "bert-base-uncased",
    },
    "roberta": {
        "repo": "roberta-base",
        "local_dir": MODEL_ROOT / "roberta-base",
    },
    "codebert": {
        "repo": "microsoft/codebert-base",
        "local_dir": MODEL_ROOT / "microsoft-codebert-base",
    },
    "mathbert": {
        "repo": "tbs17/MathBERT",
        "local_dir": MODEL_ROOT / "tbs17-MathBERT",
    },
}


def ensure_dirs():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)


def bytes_to_mb(n):
    return n / (1024 * 1024)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled


def approx_model_ram_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total


def download_and_save_model(repo_id: str, local_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(HF_CACHE))
    model = AutoModel.from_pretrained(repo_id, cache_dir=str(HF_CACHE))
    local_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(local_dir))
    model.save_pretrained(str(local_dir))
    del tokenizer, model


def load_local_model(local_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(local_dir), local_files_only=True)
    model.eval()
    return tokenizer, model


def embed_chunks(tokenizer, model, texts, batch_size=8):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=512)
        with torch.no_grad():
            res = model(**enc)
        out.append(mean_pooling(res, enc["attention_mask"]))
    return torch.cat(out, dim=0)


def purge_cache():
    if HF_CACHE.exists():
        shutil.rmtree(HF_CACHE)
    HF_CACHE.mkdir(parents=True, exist_ok=True)


# CHANGE 4 — refuse to produce embeddings silently on a machine that computes
# them wrongly. See the CHANGE 3 note: float32 GEMM here can be off by ~2.0
# against a float64 reference, where correct float32 rounding is ~1e-4.
GEMM_TOLERANCE = 1e-2


def blas_sanity_check(n: int = 256, k: int = 768, tolerance: float = GEMM_TOLERANCE,
                      trials: int = 8):
    """Is float32 matmul on this machine numerically trustworthy?

    Pure and self-contained: builds its own data, compares float32 products
    against the same product computed in float64. Returns (ok, worst_error).

    float64 is the reference because it exercises a different kernel path; on a
    sound machine the float32 result lands within ~1e-4 of it at this scale.
    Anything near 1.0 means the kernel is returning wrong answers, not merely
    rounding differently.

    **Repeated `trials` times, worst result wins — and it is still only a
    partial screen.** Measured on this host: 10 fresh processes with no
    workaround, the check flagged 3. The fault appears to be decided per
    process, and a clean process reports exactly 6.409e-05 every time while a
    bad one reports 2.0-6.1. So a failure here is conclusive proof the machine
    is wrong; a pass is weak evidence it is right. Do not read a pass as a
    guarantee.

    Checks **torch**, because that is the library that computes the embeddings.
    numpy and torch can link different BLAS backends with different faults and
    different workarounds, so testing numpy here would validate the wrong path.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    truth = a.astype(np.float64) @ b.astype(np.float64)

    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    worst = 0.0
    for _ in range(max(1, trials)):
        got = (ta @ tb).numpy().astype(np.float64)
        worst = max(worst, float(np.abs(got - truth).max()))
        if worst > tolerance:
            break
    return worst <= tolerance, worst


# CHANGE 2 — added so a run can be checked against a previous one.
def compare_vectors(mine: dict, reference: dict) -> dict:
    """Divergence between two {chunk_id: vector} maps.

    Pure: no torch model, no disk, no globals. Returns
    {n_compared, n_missing, max_abs_diff, mean_abs_diff, min_cosine}.

    Cosine is reported alongside the raw difference because they answer
    different questions: a large `max_abs_diff` with `min_cosine` at 1.0 means
    the vectors were merely scaled, whereas a dropped cosine means they point
    somewhere else and the embeddings genuinely disagree.
    """
    shared = [cid for cid in mine if cid in reference
              and len(mine[cid]) == len(reference[cid])]
    missing = [cid for cid in mine if cid not in shared]
    if not shared:
        return {"n_compared": 0, "n_missing": len(missing),
                "max_abs_diff": None, "mean_abs_diff": None, "min_cosine": None}

    max_abs, total_abs, count, min_cos = 0.0, 0.0, 0, 1.0
    for cid in shared:
        a, b = mine[cid], reference[cid]
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na > 0 and nb > 0:
            min_cos = min(min_cos, dot / (na * nb))
        for x, y in zip(a, b):
            d = abs(x - y)
            max_abs = max(max_abs, d)
            total_abs += d
            count += 1

    return {
        "n_compared": len(shared),
        "n_missing": len(missing),
        "max_abs_diff": max_abs,
        "mean_abs_diff": total_abs / count,
        "min_cosine": min_cos,
    }


def inspect_model(name: str, spec: dict, chunks):
    repo = spec["repo"]
    local_dir = spec["local_dir"]

    if not local_dir.exists() or not any(local_dir.iterdir()):
        download_and_save_model(repo, local_dir)
        purge_cache()

    tokenizer, model = load_local_model(local_dir)

    disk_bytes = dir_size_bytes(local_dir)
    ram_bytes = approx_model_ram_bytes(model)
    param_count = sum(p.nelement() for p in model.parameters())
    hidden_size = getattr(model.config, "hidden_size", None)
    max_pos = getattr(model.config, "max_position_embeddings", None)
    vocab_size = getattr(model.config, "vocab_size", None)

    texts = [c["text"] for c in chunks]
    emb = embed_chunks(tokenizer, model, texts)

    tok_lens = [len(tokenizer(t)["input_ids"]) for t in texts]

    vectors = {}
    for c, row in zip(chunks, emb):
        vectors[c["id"]] = [round(float(v), 6) for v in row]

    entry = {
        "name": name,
        "repo": repo,
        "local_path": str(local_dir),
        "embedding_shape": list(emb.shape),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "max_positions": max_pos,
        "parameter_count": param_count,
        "approx_model_ram_mb": round(bytes_to_mb(ram_bytes), 2),
        "on_disk_mb": round(bytes_to_mb(disk_bytes), 2),
        "token_lengths": tok_lens,
        "vectors": vectors,
    }
    del tokenizer, model
    return entry


# CHANGE 5 — acceptance test for the setup script.
# Cosine, not raw difference: mean-pooled embeddings from a correct machine
# agree with the reference to ~1e-9 in cosine, while absolute differences move
# with torch version. A cosine below this means the environment is wrong, not
# merely different.
SELFTEST_MIN_COSINE = 0.9999
SELFTEST_MODEL = "mathbert"


def selftest(chunks, reference_path: Path, model: str = SELFTEST_MODEL,
             min_cosine: float = SELFTEST_MIN_COSINE) -> int:
    """Run one model and check it against the reference. 0 = pass.

    This is the setup script's acceptance criterion: it exercises the whole
    chain — model download, local save, load, tokenize, forward, mean-pool —
    and compares the result to a known-good baseline.
    """
    if not reference_path.exists():
        print(f"selftest: no reference at {reference_path}", file=sys.stderr)
        return 2
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    if model not in (ref.get("vectors") or {}):
        print(f"selftest: reference has no vectors for {model!r}", file=sys.stderr)
        return 2

    entry = inspect_model(model, MODEL_SPECS[model], chunks)
    stats = compare_vectors(entry["vectors"], ref["vectors"][model])
    ref_meta = ref["models"][model]

    shape_ok = entry["embedding_shape"] == ref_meta["embedding_shape"]
    param_ok = entry["parameter_count"] == ref_meta["parameter_count"]
    cos = stats["min_cosine"]
    cos_ok = cos is not None and cos >= min_cosine

    print(f"  shape        {entry['embedding_shape']} vs {ref_meta['embedding_shape']}"
          f"   {'ok' if shape_ok else 'MISMATCH'}")
    print(f"  parameters   {entry['parameter_count']:,}"
          f"   {'ok' if param_ok else 'MISMATCH'}")
    print(f"  chunks       {stats['n_compared']} compared, {stats['n_missing']} missing")
    print(f"  min cosine   {cos:.9f} (need >= {min_cosine})"
          f"   {'ok' if cos_ok else 'TOO LOW'}")
    print(f"  max abs diff {stats['max_abs_diff']:.3e}")

    if shape_ok and param_ok and cos_ok:
        print("selftest PASSED")
        return 0
    print("selftest FAILED", file=sys.stderr)
    return 1


def main():
    ensure_dirs()

    # Repair the numerics here rather than at import time -- this may re-exec
    # the process, which is safe from main() and catastrophic from an import.
    # It never exits: an unrepairable machine still runs, with a warning.
    # `--selftest` is the gate that actually decides pass/fail.
    report = blasfix.ensure_sane_blas()
    if report.get("status") == "fixed":
        print(f"note: float32 arithmetic repaired "
              f"({', '.join(report['applied']) or 'LD_PRELOAD fallback'})",
              file=sys.stderr)

    chunks = json.load(open(BASE_DIR / "chunks.json", encoding="utf-8"))
    args = sys.argv[1:]

    if "--selftest" in args:
        return selftest(chunks, BASE_DIR / "result.json")

    which = [a for a in args if not a.startswith("-")] or list(MODEL_SPECS)
    unknown = [w for w in which if w not in MODEL_SPECS]
    if unknown:
        print(f"embrun: unknown model(s) {', '.join(unknown)}; "
              f"expected {', '.join(MODEL_SPECS)}", file=sys.stderr)
        return 2

    for name in which:
        entry = inspect_model(name, MODEL_SPECS[name], chunks)
        outp = BASE_DIR / f"report_{name}.json"
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        print(f"{name}: shape={entry['embedding_shape']} "
              f"params={entry['parameter_count']:,} "
              f"disk={entry['on_disk_mb']} MB -> {outp}")


if __name__ == "__main__":
    main()
