#!/usr/bin/env bash
# The interleaved (ABBA) transport A/B that the probe knobs are set from.
#
#   ./abba-transport.sh 44G     # one shape per invocation (they are long; never run two in parallel)
#   ./abba-transport.sh 32G
#
# WHY ABBA: the 44G band was measured at +/-25 % run-to-run (RESULTS.md §2), so a single pass per arm cannot
# separate them.  Each shape runs map / file / dio / auto, then auto / dio / file / map, and the summary
# averages the two passes.  Never parallel (AGENTS.md).
#
# WHY llama-cli and not the rig's server: the question is which TRANSPORT served which read, and the cli's exit
# census carries the per-path rates (`pool fills:` and `per-expert ... reads (bounce)`) that the decision is
# about.  The server arms (throughput at scale) are the separate decode ladder.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
M=${M:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
DIR=${DIR:-$PWD/runs/abba}
SCOPE=${1:?usage: $0 <44G|32G>}
NLIB=${NLIB:-8}
NCONTEXT=${NCONTEXT:-32768}
NB=${NB:-6144}
NPASS=${NPASS:-1}          # 50k x 1: the first pass is the cold one, the second is the cache's
mkdir -p "$DIR"
PROMPT=${PROMPT:-$DIR/prompt_20k.txt}
if [ ! -f "$PROMPT" ]; then
    head -c 100000 "$REPO/wip/bench-56gb/prompt_50k.txt" > "$PROMPT"   # ~20k tokens on this model
fi

run_arm() { # <tag> <transport|-> <pass>
    local tag=$1 tr=$2 pass=$3
    local envs=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES")
    [ "$tr" != "-" ] && envs+=("MOE_HOST_POOL_TRANSPORT=$tr")
    # drop the page cache per arm (the rig's own hygiene rule): without it a warm cache left by the previous
    # arm hides the transport entirely, which is how the first cut of this matrix measured 32G as if it were
    # unpressured
    [ "${CACHE_DROP:-1}" = "1" ] && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    printf '%-16s ' "$tag"
    timeout 1200 sudo -n systemd-run --scope -p "MemoryMax=$SCOPE" -p MemorySwapMax=0 --quiet \
        env "${envs[@]}" "$BIN" -m "$M" -ngl all --n-cpu-moe 48 --limit-host-experts "$NLIB" \
        -c "$NCONTEXT" -b "$NB" -ub "$NB" --load-mode auto --lazy-mode on --spec-type none \
        --seed 42 --temp 0 -n 32 -p "$(cat "$PROMPT")" --no-display-prompt --single-turn -lv 4 \
        > "$DIR/$tag.log" 2>&1
    local rc=$?
    local tr_now
    tr_now=$(grep -oE "host read transport -> [a-z]+" "$DIR/$tag.log" | tail -1 | awk '{print $NF}')
    local fill bounce gen pp
    fill=$(grep -oE "\([0-9.]+ MiB/fill, [0-9]+ MiB/s" "$DIR/$tag.log" | tail -1 | grep -oE "[0-9]+ MiB/s" | awk '{print $1}')
    probes=$(grep -oE "probe [0-9]+ calls" "$DIR/$tag.log" | tail -1 | awk '{print $2}')
    bounce=$(grep -E "per-expert (storage|mapping) reads \(bounce\)" "$DIR/$tag.log" | tail -1 | grep -oE "[0-9.]+ MiB/s," | head -1 | tr -d ',' | sed 's/ MiB\/s//')
    pp=$(grep -oE "prompt eval time = *[0-9.]+ ms / *[0-9]+ tokens \( *[0-9.]+ ms per token, *[0-9.]+ tokens per second\)" "$DIR/$tag.log" | tail -1 | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
    gen=$(grep -oE "eval time = *[0-9.]+ ms / *[0-9]+ tokens \( *[0-9.]+ ms per token, *[0-9.]+ tokens per second\)" "$DIR/$tag.log" | tail -1 | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}')
    printf 'rc=%s transport=%-5s fill=%-7s bounce=%-9s pp=%-8s gen=%-7s probes=%s\n' \
        "$rc" "${tr_now:-none}" "${fill:-?}" "${bounce:-?}" "${pp:-?}" "${gen:-?}" "${probes:-0}"
    printf '%s %s %s %s %s %s %s\n' "$SCOPE" "${tag%%.*}" "${tr_now:-none}" "${fill:-0}" "${bounce:-0}" "${pp:-0}" "${gen:-0}" >> "$DIR/summary.txt"
}

echo "=== transport A/B  scope=$SCOPE  lme=$NLIB  prompt=$(wc -c < "$PROMPT") B  $(date -Is)"
# ABBA: the second half repeats the arms in reverse, so a monotone drift cancels in the average
for pass in a b; do
    arms=(map file dio auto)
    [ "$pass" = b ] && arms=(auto dio file map)
    for a in "${arms[@]}"; do
        run_arm "${a}.${pass}" "$a" "$pass"
    done
done
echo "=== done; rows in $DIR/summary.txt"
