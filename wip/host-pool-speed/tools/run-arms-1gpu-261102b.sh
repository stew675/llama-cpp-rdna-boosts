#!/usr/bin/env bash
# Session 3 (2026-11-02, 1-GPU host, re-based r206 tip efaa60db0 + campaign patches 0001/0002):
# the 122B-A10B reference arm, pinned vs pool, decode >= 2000 tokens (README §9.4/§9.5).
#
#   OUT=<dir> bash tools/run-arms-1gpu-261102b.sh
#
# Deliberately NOT `lib-arms.sh`: that one points at ~/llama.cpp, and this session's build is
# ~/llama-scratch/build-rocm (the re-based tree).  Everything else follows lib-arms.sh (same COMMON
# arm, same /proc/diskstats + MemAvailable accounting, same alternating order).
set -uo pipefail

REPO=/home/stew675/llama-cpp-rdna-boosts
FORK=${FORK:-/home/stew675/llama-scratch}
MODEL=/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
PROMPT=$REPO/prompts/prose-rdna-boosts.txt
DISK=${DISK:-nvme0n1}
OUT=${OUT:?set OUT to the output directory}
mkdir -p "$OUT"

COMMON=(-m "$MODEL" -ncmoe 48 -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048
        --spec-type none --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
        -n "${NGEN:-2000}" -p "$(cat "$PROMPT")")
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
    env HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib "${envs[@]}" \
        timeout 1800 "$FORK/build-rocm/bin/llama-cli" "${COMMON[@]}" "${extra[@]}" > "$log" 2>&1
    rc=$?; t1=$(date +%s); d1=$(disk_read_sectors); m1=$(mem_state)

    local decode prompt diskmib arena poolmode dio poolslot pool
    prompt=$(grep 'prompt eval time' "$log" | num)
    decode=$(grep 'eval time =' "$log" | grep -v prompt | num)
    diskmib=$(awk -v a="$d0" -v b="$d1" 'BEGIN{ if (b>=a) printf "%.1f", (b-a)*512/1048576; else printf "na" }')
    arena=$(grep -oE 'MoE arena = [0-9.]+' "$log" | tail -1 | awk '{print $4}')
    poolmode=$(grep -cE 'host_experts_mode=POOL' "$log")
    dio=$(grep -E 'pool_decide_dio_locked' "$log" | tail -1 | sed -E 's/.*fill mode: ([A-Za-z_\-]+).*/\1/')
    poolslot=$(grep -E 'table_pool_locked' "$log" | tail -1 | sed -E 's/.* -> ([0-9]+\/[0-9]+) slots.*/\1/')
    pool=$(grep -E 'host pool \(L2\):' "$log" | tail -1 | sed 's/.*host pool (L2): //')

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "$poolmode" "${dio:-none}" \
        "${poolslot:-none}" "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}" >> "$OUT/summary.tsv"
    printf '%-22s decode=%-7s prompt=%-7s arenaH=%-6s POOL=%s dio=%-9s slots=%-8s disk=%-8sMiB %ss rc=%s\n   pool_report=%s\n' \
        "$name" "${decode:-?}" "${prompt:-?}" "${arena:-?}" "$poolmode" "${dio:-none}" "${poolslot:-none}" \
        "${diskmib}" "$((t1-t0))" "$rc" "${pool:-none}"
}

rm -f "$OUT/summary.tsv"
echo -e "arm\tdecode_tps\tprompt_tps\tarena_h\tpool_active\tdio\tpool_slots\tdisk_MiB\tsecs\trc\tpool_report" > "$OUT/summary.tsv"
echo "before run: $(mem_state)"

# Alternating arms, never parallel (AGENTS.md).  `pool` is measured FIRST in its group so a preceding
# `pinned` allocation (62 GiB of hipHostMalloc) cannot bias its page cache; an interleaved pool arm
# later in the run is the control for that ordering effect.
case "${ARMS:-all}" in
all)
    run_arm pinned-1        :::
    run_arm pool-frac80-1   MOE_HOST_POOL_FRAC=80  ::: --host-experts pool
    run_arm pool-frac100-1  MOE_HOST_POOL_FRAC=100 ::: --host-experts pool
    run_arm pool-cold0-1    MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_COLD=0 ::: --host-experts pool
    run_arm pinned-2        :::
    run_arm pool-frac80-2   MOE_HOST_POOL_FRAC=80  ::: --host-experts pool
    ;;
decode)
    run_arm pinned-1        :::
    run_arm pool-frac80-1   MOE_HOST_POOL_FRAC=80  ::: --host-experts pool
    run_arm pool-cold0-1    MOE_HOST_POOL_FRAC=80 MOE_HOST_POOL_COLD=0 ::: --host-experts pool
    run_arm pinned-2        :::
    ;;
# Step 0 of the decode-gap decomposition: how much of the pool FRAC 100 deficit is the missing
# GGML_MOE_DEVREMAP (r203 measured +8.1 % on soar), and how much is the copy kind / pool path overhead?
# Needs the working tree carrying patches/h2d-experiment.patch (MOE_HOST_POOL_H2D).
step0)
    run_arm pinned            :::
    run_arm pinned-noremap    GGML_MOE_DEVREMAP=0 :::
    run_arm pool-fr100        MOE_HOST_POOL_FRAC=100 ::: --host-experts pool
    run_arm pool-fr100-h2d    MOE_HOST_POOL_FRAC=100,MOE_HOST_POOL_H2D=1 ::: --host-experts pool
    run_arm pinned-noremap-2  GGML_MOE_DEVREMAP=0 :::
    run_arm pool-fr100-2      MOE_HOST_POOL_FRAC=100 ::: --host-experts pool
    run_arm pool-fr100-h2d-2  MOE_HOST_POOL_FRAC=100,MOE_HOST_POOL_H2D=1 ::: --host-experts pool
    ;;
# one pool arm, for the mechanism (a) exposure counters (the pageable-master fallback path).
pool)
    run_arm pool-frac80     MOE_HOST_POOL_FRAC=80  ::: --host-experts pool
    run_arm pool-frac80-g   MOE_HOST_POOL_FRAC=80,GGML_SCHED_DEVGATHER=1 ::: --host-experts pool
    ;;
# wip/host-pool-speed mechanism (b): the pool-aware device gather (`GGML_SCHED_DEVGATHER=1`).  One
# control arm pair per group: `pinned` with and without the gather, then the pool arms.
gather)
    run_arm pinned-gather        GGML_SCHED_DEVGATHER=1 :::
    run_arm pool-fr80-gather     MOE_HOST_POOL_FRAC=80,GGML_SCHED_DEVGATHER=1 ::: --host-experts pool
    run_arm pool-fr100-gather    MOE_HOST_POOL_FRAC=100,GGML_SCHED_DEVGATHER=1 ::: --host-experts pool
    run_arm pool-fr80-gath-cold0 MOE_HOST_POOL_FRAC=80,GGML_SCHED_DEVGATHER=1,MOE_HOST_POOL_COLD=0 ::: --host-experts pool
    run_arm pinned-gather-2      GGML_SCHED_DEVGATHER=1 :::
    run_arm pool-fr80-gather-2   MOE_HOST_POOL_FRAC=80,GGML_SCHED_DEVGATHER=1 ::: --host-experts pool
    ;;
esac

echo "after run: $(mem_state)"
column -t -s$'\t' "$OUT/summary.tsv"
