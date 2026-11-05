#!/usr/bin/env bash
# One mirror-prefill arm for wip/host-expert-mirror step 1 (the spike).
#   usage: ./arm.sh <tag> <ub> [ENV=V ...]
# The config is wip/bench-56gb/runme-56gb-mirror.sh translated to llama-cli: 1 x R9700,
# -lme 47 (a 46.84 GiB host expert set) with MOE_HOST_POOL_RAM_PCT=0 so the ceiling cannot
# shave the mirror, KV attendance pool off.  No cgroup: this box has no passwordless sudo for
# systemd-run, so MemAvailable is ~78 GiB rather than the rig's 57 GiB.  The tier is still a
# MIRROR (checked from the plan line on every run) -- what this spike needs is a RATIO (the
# copy's share of the prefill), not the rig's absolute t/s.
set -uo pipefail
TAG=${1:?usage: arm.sh TAG UB [ENV=V ...]}
UB=${2:?}; shift 2
DIR="$PWD/runs/$TAG"; mkdir -p "$DIR"
MODEL=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=${DEV:-0}
export MOE_HOST_POOL_RAM_PCT=0
export LLAMA_KV_REGION_POOL=0
export GGML_CUDA_SLAB_HEADROOM_MIB=1536
if [ $# -gt 0 ]; then export "$@"; fi
# HOST_ARGS: the tier spelling.  Default = the mirror's -lme 47; PINNED=1 pins the whole host set
# (a duplicate in ROCm_Host), which is the "everything pinned" arm this spike needs.
if [ "${PINNED:-0}" = "1" ]; then HOST_ARGS=(--host-experts pinned); else HOST_ARGS=(--limit-host-experts 47); fi
grep -E "MemAvailable|AnonHugePages" /proc/meminfo > "$DIR/mem_before.txt"
# swap/paging deltas: on this box a 46.84 GiB pinned tier plus an 83.6 GB model is close enough to the
# 91 GiB ceiling that a run can be disk-bound (INVARIANTS r215 item 8's pressure trap, one tier over).
awk '/^pswpin|^pswpout|^pgmajfault/{printf "%s=%s ",$1,$2} END{print ""}' /proc/vmstat > "$DIR/vm_before.txt"
( while true; do grep MemAvailable /proc/meminfo >> "$DIR/mem.txt"; sleep 5; done ) &
MON=$!
"${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}" \
  -m "$MODEL" \
  --fit off -ngl 99 --n-cpu-moe 48 "${HOST_ARGS[@]}" \
  -c "${CTX:-204800}" -fa on -ctk q8_0 -ctv q8_0 -sm "${SM:-layer}" ${SM:+$([ "$SM" = tensor ] && echo "-mg 0")} \
  --load-mode mmap --lazy-mode auto -t 7 --cache-ram 1024 \
  -b "$UB" -ub "$UB" \
  -f prompt_52k.txt -n 1 --temp 0 --seed 42 --no-display-prompt --single-turn -lv 4 2>&1 | tee "$DIR/run.log"
RC=${PIPESTATUS[0]}
kill $MON 2>/dev/null
grep -E "MemAvailable|AnonHugePages" /proc/meminfo > "$DIR/mem_after.txt"
awk '/^pswpin|^pswpout|^pgmajfault/{printf "%s=%s ",$1,$2} END{print ""}' /proc/vmstat > "$DIR/vm_after.txt"
echo "exit=$RC"
