#!/usr/bin/env bash
# Adaptive-transport tests (DESIGN.md §4).  Needs root-free `sudo -n` for the memory scope.
#
#   ./test-adaptive.sh text       the four arms + `auto`: text must be identical
#   ./test-adaptive.sh escalate   `auto` forced to start on `map` inside a 32G scope: must escalate, text same
#   ./test-adaptive.sh slowguard  the slow-read guard (TR_SLOW_MS=1) must fire
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
PROMPT=${PROMPT:-$REPO/wip/bench-56gb/prompt_short.txt}
DIR=${DIR:-$PWD/runs/adaptive}
mkdir -p "$DIR"
# NOTE: NO `--host-experts pool` -- that flag is deprecated and makes the loader keep a full PINNED host
# master (~46 GiB of anonymous memory, which is also what OOM-killed the first version of this test inside a
# scope).  `-ncmoe 48 -lme 8` is the delivery's own placement: the masters stay in the model mapping and the
# pool is filled from it, which is the shape the transport question is about.
COMMON=(-m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts 8
        -c 4096 -b 2048 -ub 2048 --load-mode mmap --lazy-mode on
        --spec-type none --seed 42 --temp 0 --no-display-prompt --single-turn)

case "${1:-text}" in
text)
    for tr in map file dio auto; do
        timeout 900 env MOE_HOST_POOL_TRANSPORT=$tr MOE_HOST_POOL_DIO=0 \
            "$BIN" "${COMMON[@]}" -n 24 -p "$(cat "$PROMPT")" > "$DIR/text-$tr.log" 2>&1
        printf '%-6s %s\n' "$tr" "$(python3 "$REPO/scripts/extract-generated.py" "$DIR/text-$tr.log")"
    done
    echo "all four must match"
    ;;
escalate)
    # A 32G scope cannot cache the 46.84 GiB host set, so the mapping arm is slow from the first read.
    # `TR_START=map` forces the start (calibration would otherwise pick the winner and never switch).
    for v in start census; do
        extra=()
        [ "$v" = census ] && extra=(-lv 4)
        [ "$v" = start  ] && extra=()
        # 44G: a 32G scope OOM-killed llama-cli on this shape (the mapped set has nowhere to be reclaimed
        # to), while the rig's server survived it.  The rule is what is under test, not the exact pressure,
        # so the test tightens the ratio instead: on THIS box 44G gives 1 599-1 838 MiB/s of mapping fills
        # against 4 656-9 512 warm, which is a real degradation but not a 0.6x collapse.
        timeout 1200 sudo -n systemd-run --scope -p MemoryMax=44G -p MemorySwapMax=0 --quiet \
            env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
                MOE_HOST_POOL_TRANSPORT=auto MOE_HOST_POOL_TR_START=map MOE_HOST_POOL_DIO=0 \
                MOE_HOST_POOL_TR_EPOCH_MIB=16 MOE_HOST_POOL_TR_RATIO=85 \
            "$BIN" "${COMMON[@]}" "${extra[@]}" -n 24 -p "$(cat "$PROMPT")" > "$DIR/esc-$v.log" 2>&1
    done
    echo "--- text (no -lv) ---"
    python3 "$REPO/scripts/extract-generated.py" "$DIR/esc-start.log"
    echo "--- transitions (-lv 4) ---"
    grep -E "host read transport" "$DIR/esc-census.log"
    ;;
slowguard)
    timeout 1200 sudo -n systemd-run --scope -p MemoryMax=44G -p MemorySwapMax=0 --quiet \
        env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
            MOE_HOST_POOL_TRANSPORT=auto MOE_HOST_POOL_TR_START=map MOE_HOST_POOL_DIO=0 \
            MOE_HOST_POOL_TR_SLOW_MS=1 MOE_HOST_POOL_TR_SLOW_N=4 \
        "$BIN" "${COMMON[@]}" -lv 4 -n 24 -p "$(cat "$PROMPT")" > "$DIR/slow.log" 2>&1
    grep -E "host read transport" "$DIR/slow.log"
    ;;
*) echo "usage: $0 [text|escalate|slowguard]"; exit 2 ;;
esac
