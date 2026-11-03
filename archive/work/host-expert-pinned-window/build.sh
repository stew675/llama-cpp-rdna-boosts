#!/usr/bin/env bash
# Build the pinned-window probe against the delivery's canonical ROCm line.
#   ./build.sh [ROCM_PREFIX]        default: /opt/rocm-10.1.0-gfx120X
set -euo pipefail
cd "$(dirname "$0")"
ROCM=${1:-/opt/rocm-10.1.0-gfx120X}
mkdir -p build
"$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 \
    -o build/probe-pinned-window probe-pinned-window.cpp
echo "built: $(pwd)/build/probe-pinned-window  (ROCm $ROCM)"
"$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 -o build/probe-copy-path probe-copy-path.cpp
  "$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 -o build/probe-churn probe-churn.cpp
  "$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 -o build/probe-cold-read probe-cold-read.cpp
  "$ROCM/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 -o build/probe-alias-d2d probe-alias-d2d.cpp
