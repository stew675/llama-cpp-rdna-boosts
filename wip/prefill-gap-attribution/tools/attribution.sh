#!/usr/bin/env bash
# Prefill kernel-time attribution on gfx1201.
#
#   ./attribution.sh <model.gguf> [pp] [device] [reps]
#
# PMC counters return 0 on gfx1201, so this uses rocprofv3 --kernel-trace durations only.
# Output: the folded per-class / per-kernel split.
set -euo pipefail

LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
BUILD_DIR=${BUILD_DIR:-$LLAMA_DIR/build-rocm}
ROCM_LIB=${ROCM_LIB:-/opt/rocm-7.14-gfx1201/lib}
FOLD=${FOLD:-$(dirname "$0")/fold-kernel-trace.py}

model=${1:?usage: attribution.sh <model.gguf> [pp] [device] [reps]}
pp=${2:-8192}
dev=${3:-0}
reps=${4:-1}

export LD_LIBRARY_PATH="$ROCM_LIB:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=$dev

out=/tmp/pfa_attn_$(basename "$model" .gguf)_pp$pp
rm -rf "$out"

echo "### $model  pp$pp  device $dev" >&2
rocprofv3 --kernel-trace -f csv -o "$out" -- \
  "$BUILD_DIR/bin/llama-bench" -m "$model" -p "$pp" -n 0 -r "$reps" 2>&1 \
  | grep -E "pp$pp|error" >&2 || true

python3 "$FOLD" "${out}_kernel_trace.csv" --top 12
