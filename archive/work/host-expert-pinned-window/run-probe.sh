#!/usr/bin/env bash
# Run the pinned-window probe.  Pin the device: the 610M iGPU is enumerated as a HIP device and the
# probe would otherwise be able to pick it (AGENTS.md rule 3).
#   ./run-probe.sh [GGUF_PATH] [ITERS]
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${ROCM:-/opt/rocm-10.1.0-gfx120X}
mkdir -p runs
OUT=runs/probe-$(date +%Y%m%d-%H%M%S).log
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ./build/probe-pinned-window "${1:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}" "${2:-40}" \
    2>&1 | tee "$OUT"
echo "log: $OUT"
