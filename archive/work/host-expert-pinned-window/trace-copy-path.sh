#!/usr/bin/env bash
# Trace one H2D copy per source kind (registered file mapping / pinned / pageable) and dump both the
# copy and the kernel stream, to answer "direct DMA or host-side bounce?".
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
mkdir -p runs
export LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=0
"$ROCM/bin/rocprofv3" --memory-copy-trace --kernel-trace --hip-trace -f csv -o runs/copy-path -- \
    ./build/probe-copy-path "${1:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}" \
    2>&1 | tail -20
echo "--- copies (kind, bytes, ns) ---"
ls runs/copy-path*
