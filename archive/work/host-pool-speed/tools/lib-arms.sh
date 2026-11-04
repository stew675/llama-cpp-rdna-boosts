#!/usr/bin/env bash
# Shared arm runner for the host-pool-speed campaign: sourced by run-arms-*.sh.  NOT part of the delivery.
# Exports: run_arm <name> [VAR=VAL ...] ::: [llama-cli args], and $OUT / $OUT/summary.tsv.
# wip/host-pool-speed: the 122B pool-vs-pinned arm matrix (1 GPU, -ncmoe 48, 2000-token decodes).
# Alternates the pinned reference with every pool arm in one run, and records the host memory state
# plus the bytes read from disk per arm.  NOT part of the delivery.  See ../HANDOVER.md §0/§6.
#
set -uo pipefail

REPO=/home/stew675/llama-cpp-rdna-boosts
FORK=/home/stew675/llama.cpp
MODEL=/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
PROMPT=$REPO/prompts/prose-rdna-boosts.txt
DISK=nvme2n1
OUT=${OUT:?set OUT to the output directory}
mkdir -p "$OUT"

# ---- the campaign's reference arm (README §9): 1 GPU, -ncmoe 48, 5258-token prose, 2000 generated ---
# `LV=0` drops -lv entirely (needed when hashing the generated text: a verbose log interleaves statistics
# into it, `scripts/extract-generated.py`); `NGEN` overrides -n (short runs are correctness smoke only).
COMMON=(-m "$MODEL" -ncmoe 48 -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048
        --spec-type none --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
        -n "${NGEN:-2000}" -p "$(cat "$PROMPT")")
[ "${LV:-4}" != "0" ] && COMMON+=(-lv "${LV:-4}")

disk_read_sectors() { awk -v d="$DISK" '$3==d {print $6}' /proc/diskstats; }
mem_state() { awk '/MemAvailable|^Cached|^Buffers|SwapTotal|SwapFree|SwapCached/ {printf "%s=%s ", $1, $2}' /proc/meminfo; }
num() { grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' | head -1; }

# run_arm <name> [VAR=VAL ...] ::: [extra llama-cli args ...]
run_arm() {
    local name=$1; shift
    local envs=() extra=()
    while [ $# -gt 0 ] && [ "$1" != ":::" ]; do envs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    extra=("$@")

    local log=$OUT/$name.log
    local d0 d1 m0 m1 rc t0 t1
    d0=$(disk_read_sectors); m0=$(mem_state); t0=$(date +%s)
    env HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib "${envs[@]}" \
        timeout 1800 "$FORK/build-rocm/bin/llama-cli" "${COMMON[@]}" "${extra[@]}" > "$log" 2>&1
    rc=$?; t1=$(date +%s); d1=$(disk_read_sectors); m1=$(mem_state)

    local decode prompt diskmib arena dio poolslot poolmode
    prompt=$(grep 'prompt eval time' "$log" | num)
    decode=$(grep 'eval time =' "$log" | grep -v prompt | num)
    diskmib=$(awk -v a="$d0" -v b="$d1" 'BEGIN{ if (b>=a) printf "%.1f", (b-a)*512/1048576; else printf "na" }')
    arena=$(grep -oE 'MoE arena = [0-9.]+' "$log" | tail -1 | awk '{print $4}')
    poolmode=$(grep -cE 'host_experts_mode=POOL' "$log")
    dio=$(grep -E 'pool_decide_dio_locked' "$log" | tail -1 | sed -E 's/.*fill mode: ([A-Za-z_\-]+).*/\1/')
    poolslot=$(grep -E 'table_pool_locked' "$log" | tail -1 | sed -E 's/.* -> ([0-9]+\/[0-9]+) slots.*/\1/')
    pool=$(grep -E 'host pool \(L2\):' "$log" | tail -1 | sed 's/.*host pool (L2): //')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "${poolmode}" "${dio:-none}" \
        "${poolslot:-none}" "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}" >> "$OUT/summary.tsv"
    printf '%-16s decode=%-7s prompt=%-7s arenaH=%-6s POOL=%s dio=%-9s slots=%-8s disk=%-8sMiB %ss rc=%s\n   pool_report=%s\n   before: %s\n   after : %s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "$poolmode" "${dio:-none}" "${poolslot:-none}" \
        "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}" "$m0" "$m1"
}

rm -f "$OUT/summary.tsv"
echo -e "arm\tdecode_tps\tprompt_tps\tarena_h\tpool_active\tdio\tpool_slots\tdisk_MiB\tsecs\trc\tpool_report" > "$OUT/summary.tsv"

