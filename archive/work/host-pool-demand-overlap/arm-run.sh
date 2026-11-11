#!/usr/bin/env bash
# Arm runner for the TODO #132 demand-fill / overlap campaign (2026-11-11).  NOT part of the delivery.
#
# usage: OUT=<dir> [VIS=0,2] [SM=tensor] [NGEN=2000] [LV=4] [MODEL=...] bash arm-run.sh TAG [VAR=VAL ...] ::: [llama-cli args]
#
# Same contract as archive/work/host-pool-speed/tools/arm-run-261103.sh (which this is derived from): box
# hygiene check, /proc/diskstats + swap accounting, census scraped at -lv 4, generated-text sha.
set -uo pipefail
REPO=/home/stew675/llama-cpp-rdna-boosts
FORK=${FORK:-/home/stew675/llama.cpp}
MODEL=${MODEL:-/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf}
PROMPT=${PROMPT:-$REPO/prompts/prose-rdna-boosts.txt}
DISK=${DISK:-nvme2n1}
VIS=${VIS:-0,2}
SM=${SM:-tensor}
NGEN=${NGEN:-2000}
LV=${LV:-4}
MEM_FLOOR_GB=${MEM_FLOOR_GB:-55}
OUT=${OUT:?set OUT to a disk output directory}
TAG=$1; shift
mkdir -p "$OUT"

mem() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }
swapused_gb() { awk '/SwapFree|SwapTotal/{v[$1]=$2} END{printf "%.1f", (v["SwapTotal:"]-v["SwapFree:"])/1048576}' /proc/meminfo; }
pswpout() { awk '/pswpout/{print $2}' /proc/vmstat; }
diskr() { awk -v d="$DISK" '$3==d {print $6}' /proc/diskstats; }
num() { grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' | head -1; }

avail=$(mem); sw=$(swapused_gb)
if [ "$(printf '%.0f' "$avail")" -lt "$MEM_FLOOR_GB" ]; then
    echo "REFUSED $TAG: box not clean (MemAvailable ${avail} GiB < ${MEM_FLOOR_GB})" >&2
    exit 3
fi
# A desktop session keeps a few GiB in swap; that is a state to report, not a refusal.  What is refused is a
# run that SWAPS (the pswpout delta is recorded below) — the 2026-11-03 session's impossible numbers came
# from a box that was paging, so the delta, not the standing figure, is the gate.
echo "box: MemAvailable ${avail} GiB, swap in use ${sw} GiB" >&2
if pgrep -f "[l]lama-(cli|server|bench|perplexity)" >/dev/null; then
    echo "REFUSED $TAG: another llama process is running" >&2
    exit 3
fi

envs=()
while [ $# -gt 0 ] && [ "$1" != ":::" ] && [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && [ "$1" = ":::" ] && shift
extra=("$@")

log=$OUT/$TAG.log
d0=$(diskr); p0=$(pswpout); t0=$(date +%s)
common=(-m "$MODEL" -ncmoe 48 -ngl 99 -sm "$SM" -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048
        --spec-type none --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
        -n "$NGEN" -p "$(cat "$PROMPT")")
[ "$LV" != "0" ] && common+=(-lv "$LV")
env HIP_VISIBLE_DEVICES="$VIS" LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib "${envs[@]}" \
    timeout 3600 "$FORK/build-rocm/bin/llama-cli" "${common[@]}" "${extra[@]}" > "$log" 2>&1
rc=$?
t1=$(date +%s); d1=$(diskr); p1=$(pswpout); avail1=$(mem); sw1=$(swapused_gb)

prompt=$(grep -a 'prompt eval time' "$log" | num)
[ -z "$prompt" ] && prompt=$(grep -a '\[ Prompt:' "$log" | tail -1 | sed -E 's/.*Prompt: ([0-9.]+) t\/s.*/\1/')
decode=$(grep -a 'eval time =' "$log" | grep -av prompt | num)
[ -z "$decode" ] && decode=$(grep -a '\[ Prompt:' "$log" | tail -1 | sed -E 's/.*Generation: ([0-9.]+) t\/s.*/\1/')
gen_n=$(grep -a '\[ Prompt:' "$log" | tail -1 | sed -E 's/.*Generation: +([0-9]+) tokens.*/\1/' )
arena=$(grep -aoE 'MoE arena = [0-9.]+' "$log" | tail -1 | awk '{print $4}')
psw=$(( p1 - p0 )); disk=$(awk -v a="$d0" -v b="$d1" 'BEGIN{printf "%.1f", (b-a)*512/1048576}')
txt=$(python3 "$REPO/scripts/extract-generated.py" "$log" 2>/dev/null || echo "-")
fills=$(grep -a 'pool fills:' "$log" | tail -1 | sed -E 's/.*pool fills: ([0-9]+) storage reads ([0-9]+) ms.*/\1 \2/')
poolh=$(grep -a 'host pool (L2):' "$log" | tail -1 | sed -E 's/.*h=([0-9.]+).*fills=([0-9]+).*/\1 \2/')

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TAG" "${decode:-?}" "${prompt:-?}" "${gen_n:-?}" "${arena:-?}" "${disk}" "$((t1-t0))" "$rc" \
    "${avail}" "${avail1}" "${psw}" "${sw1}" "${txt}" "${fills:-?}" "${poolh:-?}" "${*:-—}" >> "$OUT/summary.tsv"

printf '%-26s dec=%-7s pre=%-8s(n=%s) arenaH=%-6s disk=%-8sMiB %ss rc=%s swap=%s pswpout=%s\n   fills=[%s] pool=[%s] text=%s\n' \
    "$TAG" "${decode:-?}" "${prompt:-?}" "${gen_n:-?}" "${arena:-?}" "$disk" "$((t1-t0))" "$rc" "$sw1" "$psw" \
    "${fills:-?}" "${poolh:-?}" "$txt"
if pgrep -f "[l]lama-(cli|server|bench|perplexity)" >/dev/null; then
    echo "WARNING: a llama process survived $TAG" >&2
fi
