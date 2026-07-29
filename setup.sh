#!/usr/bin/env bash
# Environment doctor + setup for conceptdrill / embrun.
#
#   ./setup.sh            check everything, then run the acceptance test
#   ./setup.sh --check    check only, install nothing, run nothing
#   ./setup.sh --install  install missing Python dependencies first
#
# Every check below exists because it actually failed during development. The
# WHAT-CAN-GO-WRONG note on each is the observed symptom, not a hypothetical.
set -uo pipefail
cd "$(dirname "$0")"

PASS=0; WARN=0; FAIL=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
hint() { printf '        ↳ %s\n' "$1"; }

MODE="${1:-}"

echo "== 1. Python =="
# WHAT CAN GO WRONG: the project needs >=3.10 for `X | Y` type syntax and
# dataclass slots. Solus ships 3.14; a venv on an older interpreter fails at
# import time with a SyntaxError that looks unrelated to the real cause.
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
if python3 -c 'import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
  ok "python $PYV"
else
  bad "python $PYV is older than 3.10"
  hint "install a newer interpreter, or create a venv on one"
fi

echo "== 2. Python packages =="
# WHAT CAN GO WRONG: torch and transformers are optional for conceptdrill (the
# `hash` backend works without them) but mandatory for embrun. Missing
# transformers surfaces as a RuntimeError only when the first model loads,
# i.e. after a long download has already run.
need_install=()
for mod in numpy torch transformers; do
  if V=$(python3 -c "import $mod;print(getattr($mod,'__version__','?'))" 2>/dev/null); then
    ok "$mod $V"
  else
    bad "$mod missing"; need_install+=("$mod")
  fi
done
for mod in stanza pytest; do
  if V=$(python3 -c "import $mod;print(getattr($mod,'__version__','?'))" 2>/dev/null); then
    ok "$mod $V (optional)"
  else
    warn "$mod missing (optional)"
    hint "stanza improves noun-phrase quality; pytest runs the suite"
  fi
done
if [ "$MODE" = "--install" ] && [ ${#need_install[@]} -gt 0 ]; then
  echo "  installing: ${need_install[*]}"
  python3 -m pip install --user "${need_install[@]}" || bad "pip install failed"
fi

echo "== 3. float32 arithmetic =="
# WHAT CAN GO WRONG: **the big one.** On this host, AVX2 float32 GEMM returns
# wrong answers -- ~1.9 off a float64 reference where correct float32 rounding
# is ~1e-4. OpenBLAS and Intel MKL both fail; both are correct on SSE. Symptom
# if unnoticed: embeddings differ run to run, similarity rankings shuffle, and
# every downstream number is quietly wrong. The fault is decided per process,
# so this check catches roughly a third of bad processes -- a failure is proof,
# a pass is not a guarantee.
# numpy and torch may link DIFFERENT BLAS backends, with different faults and
# different workarounds. Here numpy uses the system OpenBLAS while a pypi torch
# uses Intel MKL, so `MKL_CBWR=COMPATIBLE` fixes one and not the other. Each is
# therefore checked against the path it actually uses: embrun computes with
# torch, conceptdrill's matrix operations use numpy.
for LIB in numpy torch; do
  ERR=$(python3 gemm_check.py "$LIB" 2>/dev/null)
  if [ -z "$ERR" ]; then
    warn "$LIB unavailable for the float32 check"
  elif python3 -c "import sys;sys.exit(0 if float('$ERR')<=1e-2 else 1)"; then
    ok "$LIB float32 matmul error $ERR (correct is ~1e-4)"
  else
    bad "$LIB float32 matmul error $ERR -- computes WRONG results"
    if [ "$LIB" = "numpy" ]; then
      hint "numpy here links the system OpenBLAS:"
      hint "  export LD_PRELOAD=/usr/lib64/libopenblas_core2p-r0.3.26.so"
    else
      hint "pypi torch links Intel MKL:   export MKL_CBWR=COMPATIBLE"
      hint "distro torch links OpenBLAS:  export LD_PRELOAD=.../libopenblas_core2p-*.so"
    fi
    hint "root cause looks like hardware under sustained AVX2/FMA load;"
    hint "check BIOS defaults (PBO/undervolt off), cooling, memtest86+, mprime"
  fi
done

echo "== 4. compute device =="
# WHAT CAN GO WRONG: torch.cuda.is_available() returns True on this ROCm stack
# and then SEGFAULTS the interpreter on the forward pass -- the process dies
# with no Python traceback, which reads like a hang or an OOM. Both conceptdrill
# and embrun therefore default to CPU and never autodetect.
DEV=$(python3 -c "
import torch
print('cuda-reported' if torch.cuda.is_available() else 'cpu-only')" 2>/dev/null)
case "$DEV" in
  cpu-only)      ok "cpu (no accelerator reported)" ;;
  cuda-reported) warn "torch reports an accelerator -- NOT used by default"
                 hint "on this box the ROCm forward pass segfaults; CPU is deliberate"
                 hint "override only if you have verified it: CONCEPTDRILL_DEVICE=cuda" ;;
  *)             warn "could not query torch device" ;;
esac

echo "== 5. disk and inputs =="
# WHAT CAN GO WRONG: embrun saves four models locally (~1.8 GB) plus an HF
# download cache (~0.5 GB). The original script purged the cache after every
# model because its sandbox had a 2.9 GB budget; that purge just forces
# re-downloads here.
AVAIL=$(df -Pm . | awk 'NR==2{print $4}')
if [ "${AVAIL:-0}" -gt 4096 ]; then ok "disk free ${AVAIL} MB"
else bad "only ${AVAIL} MB free; models need ~2.5 GB"; fi
for f in chunks.json result.json; do
  [ -f "$f" ] && ok "$f present" || bad "$f missing (required by embrun)"
done

echo "== 6. network (only needed on first run) =="
# WHAT CAN GO WRONG: models download from huggingface.co. Offline, the first
# run fails partway through; already-downloaded models under models/ still work.
if curl -sS -m 8 -o /dev/null -w '' https://huggingface.co 2>/dev/null; then
  ok "huggingface.co reachable"
else
  warn "huggingface.co unreachable"
  hint "fine if models/ is already populated; otherwise the first run will fail"
fi
if [ -d models ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
  ok "models/ already populated ($(du -sh models 2>/dev/null | cut -f1))"
else
  warn "models/ empty -- first run downloads ~1.4 GB"
fi

echo
echo "== summary: $PASS ok, $WARN warn, $FAIL fail =="
[ "$FAIL" -gt 0 ] && { echo "fix the failures above before trusting any output"; exit 1; }
[ "$MODE" = "--check" ] && exit 0

echo
echo "== 7. acceptance test: embrun vs result.json =="
# WHAT CAN GO WRONG: this is the end-to-end criterion. It exercises download,
# local save, load, tokenize, forward and mean-pooling, then compares against a
# baseline produced on a machine with correct arithmetic. A low cosine means
# the environment is wrong, not merely a different library version -- absolute
# differences move with torch version, cosine does not.
python3 embrun.py --selftest
exit $?
