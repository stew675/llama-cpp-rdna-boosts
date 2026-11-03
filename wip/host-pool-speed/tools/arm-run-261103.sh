#!/usr/bin/env bash
# Revalidation runner (2026-11-03, after the box restart).  NOT part of the delivery.
#
# WHY: every number measured before the restart is suspect.  On `soar` /tmp is **tmpfs**, so run artefacts
# (logs, rocprof CSVs, a llama.cpp worktree, the prefill-logit gate's base) silently consumed RAM, an aborted
# run left a llama-cli holding 70 GiB RSS, and the box ran with 12 GiB of swap in use -- which produced
# impossible numbers (a 75 t/s pinned prefill, 1.02 t/s pool at npl=1, a llama-bench core dump).
#
# This runner enforces the hygiene that failure taught:
#   * writes ONLY to $OUT (a disk path inside this repo, never /tmp);
#   * refuses to start an arm unless the box is clean (MemAvailable floor, swap ~0, no other llama process);
#   * records MemAvailable / Cached / SwapFree / pswpout / disk-read before AND after each arm, into the TSV;
#   * never leaves a process behind (`timeout`, then a pid check).
#
# usage:  OUT=/path/on/disk VIS=0,1,2 SM=tensor NGEN=2000 bash arm-run-261103.sh TAG FRAC [llama-cli args...]
#         FRAC=- means "no MOE_HOST_POOL_FRAC" (the delivery default).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
FORK=${FORK:-/home/stew675/llama.cpp}
MODEL=${MODEL:-/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf}
PROMPT=${PROMPT:-$REPO/prompts/prose-rdna-boosts.txt}
DISK=${DISK:-nvme2n1}
VIS=${VIS:-0,1,2}
SM=${SM:-tensor}
NGEN=${NGEN:-2000}
MEM_FLOOR_GB=${MEM_FLOOR_GB:-60}
OUT=${OUT:?set OUT to a disk output directory}
TAG=$1; FRAC=$2; shift 2
mkdir -p "$OUT"

mem() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }
swapused_gb() { awk '/SwapFree|SwapTotal/{v[$1]=$2} END{printf "%.1f", (v["SwapTotal:"]-v["SwapFree:"])/1048576}' /proc/meminfo; }
pswpout() { awk '/pswpout/{print $2}' /proc/vmstat; }
diskr() { awk -v d="$DISK" '$3==d {print $6}' /proc/diskstats; }
num() { grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' | head -1; }

avail=$(mem); sw=$(swapused_gb)
if [ "$(printf '%.0f' "$avail")" -lt "$MEM_FLOOR_GB" ] || [ "$(printf '%.0f' "$sw")" -gt 1 ]; then
    echo "REFUSED $TAG: box not clean (MemAvailable ${avail} GiB < ${MEM_FLOOR_GB}, swap ${sw} GiB)" >&2
    exit 3
fi
if pgrep -f "[l]lama-(cli|server|bench|perplexity)" >/dev/null; then
    echo "REFUSED $TAG: another llama process is running" >&2
    exit 3
fi

log=$OUT/$TAG.log
d0=$(diskr); p0=$(pswpout); t0=$(date +%s)
envs=(HIP_VISIBLE_DEVICES="$VIS" LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib)
[ "$FRAC" != "-" ] && envs+=(MOE_HOST_POOL_FRAC="$FRAC")
common=(-m "$MODEL" -ncmoe 48 -ngl 99 -sm "$SM" -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048
        --spec-type none --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
        -n "$NGEN" -p "$(cat "$PROMPT")")
env "${envs[@]}" timeout 3600 "$FORK/build-rocm/bin/llama-cli" "${common[@]}" "$@" > "$log" 2>&1
rc=$?
t1=$(date +%s); d1=$(diskr); p1=$(pswpout); avail1=$(mem); sw1=$(swapused_gb)

# the footer is always printed; the per-phase lines only at -lv >= 3
prompt=$(grep -a 'prompt eval time' "$log" | num)
[ -z "$prompt" ] && prompt=$(grep -a '\[ Prompt:' "$log" | tail -1 | sed -E 's/.*Prompt: ([0-9.]+) t\/s.*/\1/')
decode=$(grep -a 'eval time =' "$log" | grep -av prompt | num)
[ -z "$decode" ] && decode=$(grep -a '\[ Prompt:' "$log" | tail -1 | sed -E 's/.*Generation: ([0-9.]+) t\/s.*/\1/')
prompt_n=$(grep -a 'prompt eval time' "$log" | sed -E 's/.*ms \/ +([0-9]+) tokens.*/\1/' | tr -d '\r')
arena=$(grep -aoE 'MoE arena = [0-9.]+' "$log" | tail -1 | awk '{print $4}')
arena_all=$(grep -a 'alloc_all_locked: MoE' "$log" | sed -E 's/.*arena ([0-9.]+) MiB of ([0-9.]+) MiB.*/\1 of \2/' | tail -1)
psw=$(( p1 - p0 )); disk=$(awk -v a="$d0" -v b="$d1" 'BEGIN{printf "%.1f", (b-a)*512/1048576}')
txt=$(python3 "$REPO/scripts/extract-generated.py" "$log" 2>/dev/null || echo "-")

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TAG" "${decode:-?}" "${prompt:-?}" "${prompt_n:-?}" "${arena:-?}" "${arena_all:-?}" \
    "${disk}" "$((t1-t0))" "$rc" "${avail}" "${avail1}" "${psw}" "${sw1}" "${txt}" >> "$OUT/summary.tsv"

printf '%-20s dec=%-6s pre=%-8s(n=%s) arenaH=%-6s arena=%s disk=%-8sMiB %ss rc=%s avail %s->%sGiB pswpout=%s swap=%s\n   text=%s\n' \
    "$TAG" "${decode:-?}" "${prompt:-?}" "${prompt_n:-?}" "${arena:-?}" "${arena_all:-?}" "$disk" "$((t1-t0))" "$rc" \
    "$avail" "$avail1" "$psw" "$sw1" "$txt"
if pgrep -f "[l]lama-(cli|server|bench|perplexity)" >/dev/null; then
    echo "WARNING: a llama process survived $TAG" >&2
fi
