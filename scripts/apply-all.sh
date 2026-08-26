#!/usr/bin/env bash
# Apply all block patches in manifest order to a fresh branch of a llama.cpp
# checkout at the recorded baseline.
#
# Usage: ./apply-all.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   where to apply (default: current directory)
#   rdna-boosts-repo     path to THIS repo (default: parent of scripts/)
#
# Requires a clean llama.cpp working tree checked out at the baseline SHA
# recorded in MANIFESTS.md. Creates a branch `rdna-boosts` and applies the
# patches in manifest order, falling back to 3-way merge per patch.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches"
ORDER="01-adaptive-mtp.patch 02-chunked-gdn.patch 03-bf16-kv-cache.patch 04-wmma-flash-attn.patch 05-bit-identical-decode-cpu.patch 06-gfx1151-mmvq-table.patch 07-host-buffer-revert.patch 08-meta-device-wrapper-skip.patch 09-q6k-mmvq-vdr2.patch 10-fused-core.patch"

cd "$LLAMA"

if [ ! -f CMakeLists.txt ] || [ ! -d ggml ]; then
    echo "ERROR: $LLAMA does not look like a llama.cpp checkout" >&2; exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2; exit 1
fi

BRANCH="rdna-boosts"
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: branch $BRANCH already exists; delete it first (git branch -D $BRANCH)" >&2
    exit 1
fi
git checkout -q -b "$BRANCH"

for p in $ORDER; do
    echo "== $p"
    if ! git apply "$PATCHES/$p" 2>/tmp/apply-err.txt; then
        echo "   apply failed ($(head -1 /tmp/apply-err.txt)); trying 3-way..."
        if ! git apply -3 "$PATCHES/$p" 2>>/tmp/apply-err.txt; then
            echo "ERROR: $p failed with 3-way too; rebase the hunks manually (see MANIFESTS.md)" >&2
            exit 1
        fi
    fi
done

echo
echo "All patches applied on branch $BRANCH."
echo "Next steps (per-block verification from MANIFESTS.md):"
echo "  cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGGML_HIP_ROCBLAS=ON"
echo "  cmake --build build -j"
echo "  ./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # expect 46/46"
