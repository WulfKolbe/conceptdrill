#!/usr/bin/env bash
# One shard of the full-corpus run, in chunks so partial progress survives.
#
# hierarchy_run writes every file at the end. A 10-hour single run that is
# interrupted leaves nothing, so each chunk of documents is its own complete
# run directory: whatever finished is usable, whatever did not can be resumed
# by re-running with the same chunk list.
set -u
cd /home/wkolbe/conceptdrill
export CONCEPTDRILL_STRICT=1 CONCEPTDRILL_NLP_BACKEND=regex
SHARD="$1"; CHUNK="${2:-20}"
OUT="$HOME/conceptdrill-corpus-llm/full"
mkdir -p "$OUT"
mapfile -t DOCS < <(python3 -c "
import json,sys
d=json.load(open('docs/measurements/corpus-full-shards.json'))
print('\n'.join(d['shards'][int('$SHARD')]['docs']))")
total=${#DOCS[@]}
for ((i=0; i<total; i+=CHUNK)); do
  part=("${DOCS[@]:i:CHUNK}")
  name="s${SHARD}-c$((i/CHUNK))"
  if [ -f "$OUT/$name/manifest.json" ]; then
    echo "### SKIP $name (already complete)"; continue
  fi
  echo "### START $name  ${#part[@]} documents"
  python3 tools/hierarchy_run.py --summarizer novita --temperature 0.2 \
      --token-ceiling 50 --out "$OUT" --name "$name" --docs "${part[@]}" \
      2>&1 | grep -vE "Loading|HF_TOKEN|Warning|it/s|%\|" | tail -2
  echo "### DONE $name"
done
echo "### SHARD $SHARD COMPLETE"
