#!/usr/bin/env bash
# Install conceptdrill into venv/ (visible, git-ignored) and put `conceptdrill` on PATH.
#
# The venv sees the system site-packages (numpy, torch, openai, stanza are installed
# there) and gets conceptdrill itself editable, so the command runs this checkout.
# `conceptdrill` is linked into $CONCEPTDRILL_BIN (default ~/.local/bin) only when
# that link is missing or already this checkout's; a link to another install (the
# owner's main, a worktree, an audit clone) is left as it is, reported, and the
# install exits 3 -- a worktree install never takes over the owner's command.
#
#   scripts/install.sh              # install and link
#   scripts/install.sh --check      # only report what is installed
#   scripts/install.sh --link-only  # only the guarded link step
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${CONCEPTDRILL_BIN:-$HOME/.local/bin}"
cd "$ROOT"

check() {
    [ -x "$ROOT/venv/bin/conceptdrill" ] || { echo "venv/bin/conceptdrill missing" >&2; return 1; }
    (cd / && "$ROOT/venv/bin/python" -c '
import os, conceptdrill
print("conceptdrill", conceptdrill.__version__, "from",
      os.path.dirname(os.path.realpath(conceptdrill.__file__)))')
    [ -x "$BIN/conceptdrill" ] || { echo "conceptdrill is not in $BIN" >&2; return 1; }
    echo "$BIN/conceptdrill -> $(readlink -f "$BIN/conceptdrill")"
}

link_command() {
    local target="$ROOT/venv/bin/conceptdrill" link="$BIN/conceptdrill" now
    mkdir -p "$BIN"
    if [ -e "$link" ] || [ -L "$link" ]; then
        now="$(readlink "$link" || true)"
        case "$now" in
            "$ROOT/venv/bin/"*) ;;
            *) echo "$link is not this install's (it is ${now:-a file, not a link}); left as it is" >&2
               return 3 ;;
        esac
    fi
    ln -sfn "$target" "$link"
    echo "$link -> $target"
}

if [ "${1:-}" = "--check" ]; then check; exit $?; fi
if [ "${1:-}" = "--link-only" ]; then link_command; exit $?; fi

if [ ! -x venv/bin/python ]; then
    python3 -m venv --system-site-packages venv
fi
venv/bin/python -m pip install -q --no-build-isolation --no-deps -e .

GOT="$(cd / && "$ROOT/venv/bin/python" -c 'import os, conceptdrill; print(os.path.realpath(conceptdrill.__file__))')"
case "$GOT" in
    "$ROOT/src/conceptdrill"/*) ;;
    *) echo "conceptdrill imports from $GOT, not from $ROOT/src" >&2; exit 4 ;;
esac

link_command || { echo "installed in $ROOT/venv; the command was not linked (see above)" >&2; exit 3; }
check
