#!/bin/bash
# run-benchy-suite.sh - run the full 8-row llama-benchy matrix.
#
# Builds:
#   M = master            ~/llama.cpp/build-rocm/bin
#   B = rdna-boosts       ~/llama.cpp/build-rdna-boosts/bin
# Models (1 card): Q6_K, Q4_K_XL ; (2/3 card): Q8_0
#
# Each row: start server -> llama-benchy -> teardown. Results land in
# benchmarks/results/benchy/<ROW>.json / .md
#
# usage: run-benchy-suite.sh [--only <row>] [--rows "M-1Q6 B-2 ..."] [-v]
#   (default: all 8 rows in the order below)

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
HERE="$ROOT/benchmarks/scripts"
MASTER_BIN="$HOME/llama.cpp/build-rocm/bin"
BOOSTS_BIN="$HOME/llama.cpp/build-rdna-boosts/bin"
M_Q8="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
M_Q6="/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf"
M_Q4="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

ALL_ROWS="M-1Q6 B-1Q6 M-1Q4 B-1Q4 M-2 B-2 M-3 B-3"

ROWS="$ALL_ROWS"
while (( $# )); do
    case "$1" in
        --only) shift; ROWS="$1" ;;
        --rows) shift; ROWS="$1" ;;
        *) break ;;
    esac
    shift
done

declare -A BIN=([M]=$MASTER_BIN [B]=$BOOSTS_BIN)
declare -A MODEL=([1Q6]="$M_Q6" [1Q4]="$M_Q4" [2]="$M_Q8" [3]="$M_Q8")
declare -A ALIAS=([1Q6]=Qwen3.8-27B-Q6_K [1Q4]=Qwen3.8-27B-Q4_K_XL [2]=Qwen3.8-27B-Q8_0 [3]=Qwen3.8-27B-Q8_0)
declare -A GPUS=([1Q6]=0 [1Q4]=0 [2]=0,2 [3]=0,1,2)
declare -A CTX=([1Q6]=140000 [1Q4]=140000 [2]=262144 [3]=262144)

for ROW in $ROWS; do
    BLD="${ROW%%-*}"; CFG="${ROW#*-}"
    if [[ -z "${BIN[$BLD]+x}" || -z "${MODEL[$CFG]+x}" ]]; then
        echo "ERROR: unknown row '$ROW' (build='$BLD' cfg='$CFG')" >&2
        echo "valid: $ALL_ROWS" >&2
        exit 1
    fi
    echo "==================== $ROW ===================="
    "$HERE/benchy-run.sh" "$ROW" "${BIN[$BLD]}" "${MODEL[$CFG]}" "${ALIAS[$CFG]}" "${GPUS[$CFG]}" "${CTX[$CFG]}"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "!! row $ROW failed rc=$RC — aborting suite" >&2
        exit $RC
    fi
    sleep 5
done
echo "===== suite complete ====="
ls -la "$ROOT/benchmarks/results/benchy/" 2>/dev/null
