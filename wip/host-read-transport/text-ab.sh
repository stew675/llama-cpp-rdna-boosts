#!/usr/bin/env bash
# Text-identity A/B for the per-expert host read transport (TODO #306 widened).
#
#   ./text-ab.sh            # all four arms, greedy, hashed
#
# WHY: `file` and `map` read the SAME page-cache pages, so the pool contents (and every logit) must be
# byte-identical.  This proves it on the arm that matters -- the instrument build with both knobs -- before any
# number is trusted.  It runs WITHOUT the 56 GiB rig and without dropping the page cache: the transport is a
# code path, not a warm-vs-cold question, so a warm box exercises exactly the same branches.
#
# No `-lv` at all on any arm: this is a TEXT arm (AGENTS.md, the `-lv` table).  The census is a separate
# `-lv 4` run (the server arms in `run-arms.sh` carry it).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
DIR=${DIR:-$PWD/runs/text}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
PROMPT=${PROMPT:-$REPO/wip/bench-56gb/prompt_short.txt}
NLIB=${NLIB:-8}          # -lme: a bounded pool, so BOTH the fill and the demand paths are reached
NCONTEXT=${NCONTEXT:-4096}
NGEN=${NGEN:-24}
mkdir -p "$DIR"

run_arm() {   # tag fillsrc demandsrc
    local tag=$1
    local fs=$2
    local ds=$3
    local log=$DIR/$tag.log
    local e=()
    [ "$fs" = map  ] || e+=("MOE_HOST_POOL_FILL_SRC=$fs")
    [ "$ds" = map  ] || e+=("MOE_HOST_POOL_DEMAND_SRC=$ds")
    echo "--- $tag (FILL_SRC=${fs} DEMAND_SRC=${ds}) ---"
    timeout 900 env "${e[@]}" MOE_HOST_POOL_DIO=0 \
        "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
        -c "$NCONTEXT" -b 2048 -ub 2048 --load-mode mmap --lazy-mode on \
        --spec-type none --seed 42 --temp 0 -n "$NGEN" -p "$(cat "$PROMPT")" \
        --no-display-prompt --single-turn > "$log" 2>&1
    local rc=$?
    printf '%-10s rc=%d  %s\n' "$tag" "$rc" "$(python3 "$REPO/scripts/extract-generated.py" "$log")"
}

echo "=== host-read-transport text A/B  $(date -Is)  bin=$BIN ==="
run_arm mapp  map  map
run_arm fillf file map
run_arm demf  map  file
run_arm bothf file file
echo "=== the four hashes must be IDENTICAL; anything else is a transport bug, not a tuning result ==="
