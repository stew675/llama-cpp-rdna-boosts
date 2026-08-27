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
# patches in manifest order, falling back to 3-way merge per patch; each
# patch is committed individually with its block label as the message.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches"
ORDER="01-adaptive-mtp.patch 02-chunked-gdn.patch 03-bf16-kv-cache.patch 04-wmma-flash-attn.patch 05-bit-identical-decode-cpu.patch 06-host-buffer-revert.patch 07-meta-device-wrapper-skip.patch 08-fused-core.patch 09-meta-headroom.patch 10-k-quant-boosts.patch"

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
    git add -A
    msg="rdna-boosts: apply $p"
    subj="$(sed -n 's/^Subject: \[PATCH [0-9]*\/[0-9]*\] //p' "$PATCHES/$p" | head -1)"
    if [ -n "$subj" ]; then
        msg="rdna-boosts: $subj"
    fi
    git commit -q -m "$msg"
    echo "   committed"
 done

echo
N_BLOCKS="$(echo $ORDER | wc -w)"
if [ "$(git rev-list --count HEAD ^HEAD~$N_BLOCKS)" -eq "$N_BLOCKS" ]; then
    echo "All $N_BLOCKS patches applied and committed on branch $BRANCH, one commit each:"
    git log --oneline -$N_BLOCKS
else
    echo "All patches applied on branch $BRANCH (check git status for uncommitted leftovers)."
fi
echo "Next steps (per-block verification from MANIFESTS.md):"
echo "  cmake -B build -DGGML_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGGML_HIP_ROCBLAS=ON"
echo "  cmake --build build -j"
echo "  ./build/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET   # expect 46/46"
