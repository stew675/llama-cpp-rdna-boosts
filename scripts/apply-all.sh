#!/usr/bin/env bash
# Apply the full rdna-boosts patch set (blocks 01-11 + the hybrid block 12)
# to a clean llama.cpp checkout at the recorded baseline.
#
# Usage: ./apply-all.sh [llama.cpp-checkout] [rdna-boosts-repo]
#   llama.cpp-checkout   where to apply (default: current directory)
#   rdna-boosts-repo     path to THIS repo (default: parent of scripts/)
#
# Requires a clean llama.cpp working tree checked out at the baseline SHA
# recorded in MANIFESTS.md (currently a7cc83bba).  Blocks 01-11 are applied
# with `git am` (plain `git apply` of the concatenated series silently drops
# hunks -- verified 2026-08-29), one commit each with the block subject.
# Block 12 (the hybrid HIP all-reduce) is applied with `git apply` and
# committed as "rdna-boosts: block 12: hybrid HIP all-reduce".
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${1:-$(pwd)}"
RDNA="${2:-$REPO_DIR}"
PATCHES="$RDNA/patches"
BLOCKS="0001-rdna-boosts-block-01-adaptive-MTP-draft-depth.patch \
0002-rdna-boosts-block-02-fused-chunked-gated-delta-net-p.patch \
0003-rdna-boosts-block-03-BF16-KV-cache-and-native-BF16-f.patch \
0004-rdna-boosts-block-04-RDNA4-WMMA-flash-attn-Q6_K-mmq-.patch \
0005-rdna-boosts-block-05-CPU-bit-identical-decode-verify.patch \
0006-rdna-boosts-block-06-host-buffer-revert-for-discrete.patch \
0007-rdna-boosts-block-07-meta-device-wrapper-skip.patch \
0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch \
0009-rdna-boosts-block-09-meta-buffer-compute-container-h.patch \
0010-rdna-boosts-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch \
0011-rdna-boosts-block-11-skip-CUDA-graphs-for-multi-toke.patch"
BLOCK12="12-hybrid-allreduce-hip.patch"

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

# Blocks 01-11: git am (commits each with the original subject).
git am "$PATCHES"/000[1-9]-*.patch "$PATCHES"/001[01]-*.patch

# Block 12: the hybrid all-reduce (WIP patch; apply + commit).
echo "== $BLOCK12"
git apply "$PATCHES/$BLOCK12"
git add -A
git commit -q -m "rdna-boosts: block 12: hybrid HIP all-reduce (RDNA4-gated)"
echo "   committed"

echo
N_BLOCKS=12
echo "All $N_BLOCKS patches applied and committed on branch $BRANCH:"
git log --oneline -$N_BLOCKS
echo
echo "Build (uses your system ROCm install; see patches/README.md for the full env-knob list):"
echo "  cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS=\"${GPU_TARGETS:-gfx1100;gfx1151;gfx1201}\" -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j"
echo
echo "  GPU_TARGETS is read from the environment if set; the default builds for all three"
echo "  RDNA families the set supports (one binary runs on any of them; trim to your GPU"
echo "  for a faster build):"
echo "    gfx1100  RDNA 3   (RX 7900 XTX/XT, RX 7800 XT, ...)"
echo "    gfx1151  RDNA 3.5 (Strix Point / Strix Halo APUs)"
echo "    gfx1201  RDNA 4   (RX 9070 XT / 9070; gfx1200 = RX 9060 XT)"
echo "  All three compile with ROCm >= 6.3 (RDNA4 codegen officially from 6.4.x; this set"
echo "  was verified on ROCm 7.14).  Keep GGML_HIP_RCCL=1: block 12's hybrid all-reduce"
echo "  uses RCCL for the large-tensor path, and on non-RDNA4 (gfx1100/gfx1151) the"
echo "  internal AR is gated off and everything falls back to RCCL."
echo "  Optional (both ON by default upstream; keep for parity with the tuned reference build):"
echo "    -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=1"
echo
echo "Verify:"
echo "  ./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p \"The capital of France is\" -n 20 --seed 42 --temp 0 --single-turn"
echo "  (compare same-seed output against a known-good build; coherence gate)"
