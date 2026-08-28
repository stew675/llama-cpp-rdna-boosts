#!/bin/bash
# run-vulkan-all.sh - Vulkan leg of the v2 benchmark: PPL (2 corners x 3
# models) then the 8-row llama-benchy Vulkan throughput suite. Logs with
# timestamps to benchmarks/results/run-vulkan.log.
#
# usage: run-vulkan-all.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
HERE="$ROOT/benchmarks/scripts"
RES="$ROOT/benchmarks/results"
mkdir -p "$RES/benchy" "$RES/ppl"

MASTER_BIN="$HOME/llama.cpp/build-rocm/bin"
BOOSTS_BIN="$HOME/llama.cpp/build-rdna-boosts/bin"
WIKI="/llm/models/wikitext-2-raw/wiki.test.raw"
Q8="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
Q6="/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf"
Q4="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

LOG="$RES/run-vulkan.log"
: > "$LOG"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "=== run-vulkan start ==="
say "GPU check: $(rocm-smi --showmeminfo vram 2>/dev/null | grep Used | tr '\n' ' ')"

# ---------- Phase 1: Vulkan PPL (3 models x 2 corners: f16, bf16) ----------
for spec in "Q8_0:$Q8" "Q6_K:$Q6" "Q4_K_XL:$Q4"; do
    TAG="${spec%%:*}"; MODEL="${spec#*:}"
    say "Vulkan PPL [$TAG] starting"
    GGML_VK_VISIBLE_DEVICES=1,2 "$HOME/llama.cpp/build-vulkan/bin/llama-perplexity" \
        -m "$MODEL" -f "$WIKI" -c 2048 --chunks 128 -fa on -ctk f16 -ctv f16 \
        -sm layer -ngl 99 2>&1 | grep -E "Final estimate" | tee -a "$LOG" \
        | tee "$RES/ppl/ppl-vulkan-f16-$TAG.txt" > /dev/null
    say "Vulkan PPL [$TAG] f16 done"
    GGML_VK_VISIBLE_DEVICES=1,2 "$HOME/llama.cpp/build-vulkan/bin/llama-perplexity" \
        -m "$MODEL" -f "$WIKI" -c 2048 --chunks 128 -fa on -ctk bf16 -ctv bf16 \
        -sm layer -ngl 99 2>&1 | grep -E "Final estimate" | tee -a "$LOG" \
        | tee "$RES/ppl/ppl-vulkan-bf16-$TAG.txt" > /dev/null
    say "Vulkan PPL [$TAG] bf16 done"
done

# ---------- Phase 2: Vulkan throughput suite (8 rows) ----------
say "Vulkan throughput suite starting (8 rows)"
"$HERE/run-benchy-vulkan-suite.sh" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
say "Vulkan throughput suite done rc=$RC"

say "=== run-vulkan complete ==="
say "results: $(ls $RES/benchy/V-*.json 2>/dev/null | wc -l) json"
exit 0
