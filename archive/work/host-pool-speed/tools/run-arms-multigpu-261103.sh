#!/usr/bin/env bash
# W1/W4 arm runner for the MULTI-GPU box (soar).  NOT part of the delivery.
#
# Same contract as lib-arms.sh / lib-arms-pair.sh:  run_arm <name> [VAR=VAL ...] ::: [extra llama-cli args]
#
# Differences from lib-arms-pair.sh (which hardcodes the x8 pair):
#   * VIS   = HIP_VISIBLE_DEVICES (default 0,2 = the two x8 GPUs; 0,1,2 = all three; 0 = 1-GPU control)
#   * SM    = split mode (default tensor); extra args go after `:::`
#   * DISK  = diskstats device (default nvme2n1)
#   * it also records the arena's per-device sizing line and the teardown report, because on a tensor
#     split the arena is PER DEVICE and the "no arena at all" failure mode (F7) is silent at -lv 3.
set -uo pipefail
REPO=/home/stew675/llama-cpp-rdna-boosts
FORK=/home/stew675/llama.cpp
MODEL=/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
PROMPT=$REPO/prompts/prose-rdna-boosts.txt
DISK=${DISK:-nvme2n1}
VIS=${VIS:-0,2}
SM=${SM:-tensor}
OUT=${OUT:?set OUT to the output directory}
mkdir -p "$OUT"
COMMON=(-m "$MODEL" -ncmoe "${NCMOE:-48}" -ngl 99 -sm "$SM" -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048
        --spec-type none --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
        -n "${NGEN:-2000}" -p "$(cat "$PROMPT")")
[ "${T:-0}" != "0" ] && COMMON+=(-t "$T")
[ "${LV:-4}" != "0" ] && COMMON+=(-lv "${LV:-4}")
disk_read_sectors() { awk -v d="$DISK" '$3==d {print $6}' /proc/diskstats; }
mem_state() { awk '/MemAvailable|^Cached|^Buffers|SwapTotal|SwapFree|SwapCached/ {printf "%s=%s ", $1, $2}' /proc/meminfo; }
num() { grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' | head -1; }
run_arm() {
    local name=$1; shift
    local envs=() extra=()
    while [ $# -gt 0 ] && [ "$1" != ":::" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    extra=("$@")
    local log=$OUT/$name.log
    local d0 d1 m0 m1 rc t0 t1
    d0=$(disk_read_sectors); m0=$(mem_state); t0=$(date +%s)
    env HIP_VISIBLE_DEVICES="$VIS" LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib "${envs[@]}" \
        timeout 3600 "$FORK/build-rocm/bin/llama-cli" "${COMMON[@]}" "${extra[@]}" > "$log" 2>&1
    rc=$?; t1=$(date +%s); d1=$(disk_read_sectors); m1=$(mem_state)
    local decode prompt diskmib arena poolmode dio poolslot pool sizing slots_txt
    prompt=$(grep 'prompt eval time' "$log" | num)
    decode=$(grep 'eval time =' "$log" | grep -v prompt | num)
    diskmib=$(awk -v a="$d0" -v b="$d1" 'BEGIN{ if (b>=a) printf "%.1f", (b-a)*512/1048576; else printf "na" }')
    arena=$(grep -oE 'MoE arena = [0-9.]+' "$log" | tail -1 | awk '{print $4}')
    poolmode=$(grep -cE 'host_experts_mode=POOL' "$log")
    dio=$(grep -E 'pool_decide_dio_locked' "$log" | tail -1 | sed -E 's/.*fill mode: ([A-Za-z_\\-]+).*/\1/')
    # the per-table slot count: report the MAXIMUM over tables (the last line can be a table the
    # RAM ceiling starved to the one-slot floor -- see W3; a mean/last-line read hides that).
    poolslot=$(grep -E 'table_pool_locked' "$log" | sed -E 's/.* -> ([0-9]+)\/[0-9]+ slots.*/\1/' | sort -n | tail -1)
    poolslot="$poolslot/$(grep -E 'table_pool_locked' "$log" | sed -E 's/.* -> [0-9]+\/([0-9]+) slots.*/\1/' | sort -n | tail -1)"
    slots_txt=$(grep -E 'per-device sizing:' "$log" | tail -1 | sed -E 's/.*arena total ([0-9.]+) MiB.*/\1/')
    sizing=$(grep -cE 'cov90|concentration' "$log")
    pool=$(grep -E 'host pool \(L2\):' "$log" | tail -1 | sed 's/.*host pool (L2): //')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "$poolmode" "${dio:-none}" \
        "${poolslot:-none}" "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}" "${slots_txt:-?}" "${sizing:-0}" \
        >> "$OUT/summary.tsv"
    printf '%-24s decode=%-7s prompt=%-8s arenaH=%-6s POOL=%s dio=%-9s slots=%-8s arenaMiB=%-9s disk=%-8sMiB %ss rc=%s\n   pool=%s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "$poolmode" "${dio:-none}" "${poolslot:-none}" \
        "${slots_txt:-?}" "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}"
}
