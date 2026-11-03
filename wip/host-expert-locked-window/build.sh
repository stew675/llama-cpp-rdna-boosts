#!/usr/bin/env bash
# Build the locked-window probe against the delivery's canonical ROCm line.
#   ./build.sh [ROCM_PREFIX]        default: /opt/rocm-10.1.0-gfx120X
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${1:-/opt/rocm-10.1.0-gfx120X}
mkdir -p build
"$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 \
    -o build/probe-locked-window probe-locked-window.cpp
echo "built: $(pwd)/build/probe-locked-window  (ROCm $ROCM)"
