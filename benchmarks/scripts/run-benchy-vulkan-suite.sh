#!/bin/bash
# run-benchy-vulkan-suite.sh - Vulkan leg of the llama-benchy matrix.
#
# Same master tip (fe235f434) as the ROCm baseline, but on the Vulkan
# backend: 2 KV types x 4 model/card sets = 8 rows.
#   sets: 1Q6 = 1 card Q6_K, 1Q4 = 1 card Q4_K_XL, 2 = 2 cards Q8_0,
#         3 = 3 cards Q8_0
# Vulkan device indices: 0 = iGPU, 1/2/3 = the three R9700s (so the GPU
# sets below are 1 / 1,2 / 1,2,3). Layer split (Vulkan has no tensor
# split). Native BF16 KV is genuinely supported on Vulkan (unlike ROCm
# master, which silently stores f16).
#
# Row label format: V-<set>-<kv>   e.g. V-1Q6-bf16
#
# usage: run-benchy-vulkan-suite.sh [--only <row>] [--rows "V-1Q6-bf16 ..."]

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
HERE="$ROOT/benchmarks/scripts"
VULKAN_BIN="$HOME/llama.cpp/build-vulkan/bin"
M_Q8="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
M_Q6="/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf"
M_Q4="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

ALL_ROWS="V-1Q6-f16 V-1Q6-bf16 V-1Q4-f16 V-1Q4-bf16 V-2-f16 V-2-bf16 V-3-f16 V-3-bf16"

ROWS="$ALL_ROWS"
while (( $# )); do
    case "$1" in
        --only) shift; ROWS="$1" ;;
        --rows) shift; ROWS="$1" ;;
        *) break ;;
    esac
    shift
done

declare -A MODEL=([1Q6]="$M_Q6" [1Q4]="$M_Q4" [2]="$M_Q8" [3]="$M_Q8")
declare -A ALIAS=([1Q6]=Qwen3.8-27B-Q6_K [1Q4]=Qwen3.8-27B-Q4_K_XL [2]=Qwen3.8-27B-Q8_0 [3]=Qwen3.8-27B-Q8_0)
declare -A GPUS=([1Q6]=1 [1Q4]=1 [2]=1,2 [3]=1,2,3)   # Vulkan indices: R9700s are 1-3
declare -A CTX=([1Q6]=140000 [1Q4]=140000 [2]=262144 [3]=262144)

for ROW in $ROWS; do
    REST="${ROW#*-}"; CFG="${REST%%-*}"; KV="${REST#*-}"
    if [[ -z "${MODEL[$CFG]+x}" || ( "$KV" != f16 && "$KV" != bf16 ) ]]; then
        echo "ERROR: unknown row '$ROW' (cfg='$CFG' kv='$KV')" >&2
        echo "valid: $ALL_ROWS" >&2
        exit 1
    fi
    echo "==================== $ROW ===================="
    "$HERE/benchy-run.sh" "$ROW" "$VULKAN_BIN" "${MODEL[$CFG]}" "${ALIAS[$CFG]}" "${GPUS[$CFG]}" "${CTX[$CFG]}" "$KV" "$KV" vulkan
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "!! row $ROW failed rc=$RC — aborting suite" >&2
        exit $RC
    fi
    sleep 5
done
echo "===== vulkan suite complete ====="
ls -la "$ROOT/benchmarks/results/benchy/" 2>/dev/null | grep V-
