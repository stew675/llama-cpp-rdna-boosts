#!/usr/bin/env bash
# One warm tree for every release arm: switch states inside ONE checkout instead of making a new one.
#
# WHY THIS EXISTS (measured 2026-10-25, gfx1201, ROCm 10.1.0, ccache 30 GiB):
#   * the SAME content rebuilt in a NEW source path : 5m56s wall / 85 CPU-min, 722 ccache MISSES
#     (an r132 worktree at /tmp/... and one at ~/ccache-test, both fully cold)
#   * the SAME content rebuilt at the SAME source path in a FRESH build dir : 8.6s, 0 misses
#     (/tmp/mb-all, 343 TUs, all direct hits)
#   ccache's key includes the absolute source paths that reach the compiler command line, so a new
#   worktree path is a full cold rebuild no matter how identical the tree is.  A release that assessed
#   two PRs and folded them used FOUR paths (two arm worktrees, the folded tree, and the canonical
#   checkout) and therefore paid four cold builds: ~1-2 h of compiling for ~20 min of real work.
#   Build directories are free to create and delete; source paths are not.  Hence: one path, forever.
#
# The per-release cost with this script is the genuine content delta only: one .cu ~15-30 s, a widely
# included header (common.cuh, ggml-impl.h) ~3-5 min for its dependents, an all-new tree ~5-6 min.
#
# Usage: ./scratch-tree.sh [options] <subcommand> [args]
#
# Subcommands:
#   init                 create the scratch worktree (detached at release.json.base) if absent, build it
#   state REV            check out REV in the scratch tree and rebuild incrementally
#   blocks [BASE]        reset to BASE (default release.json.base), apply the delivery set with
#                        scripts/apply-all.sh (release.json.n_default commits on branch `scratch`), rebuild
#   arm PATCH...         reset to the release tip, `git am -3` the given patch files on top, rebuild
#                        (this is the PR-assessment / beta flow: one arm at a time, each a delta)
#   tip                  reset to release.json.tip (the canonical release tree), rebuild
#   keep TAG             copy <build>/bin to <build>/bin-TAG, so a later state can be A/B'd against it
#   gate [args...]       run scripts/gate-per-block-build.sh with --repo <scratch path> (extra args pass
#                        through), i.e. the per-block gate reuses this path's warm cache
#   show                 print the path, the checked-out revision and the build dir
#
# Options:
#   --path DIR       SCRATCH_PATH  scratch worktree (default: $HOME/llama-scratch)
#   --build-dir NAME BUILD_DIR     build directory inside it (default: build-rocm; must match the
#                                  canonical build's name so the compiler command lines match)
#   --jobs N         JOBS          build parallelism (default: nproc)
#   --rocm DIR       ROCM_DIR      ROCm install (default: resolved from the build, scripts/rocmlib.sh)
#   --gpu-target T   GPU_TARGETS   default gfx1201
#   --fast                         build llama-cli/server/bench/perplexity + test-backend-ops only
#                                  (the delta does not change: `all` costs ~0 extra on a warm ccache,
#                                  so this is only for a tight edit loop)
#   --repo DIR       REPO          the fork checkout the worktree is created from (default: the
#                                  delivery repo's ../llama.cpp)
#
# Everything is measured: each build prints its wall time, so a release's compile cost is visible.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
MANIFEST="$REPO_DIR/release.json"
FORK="${SCRATCH_FORK:-$REPO_DIR/../llama.cpp}"
SCRATCH="${SCRATCH_PATH:-$HOME/llama-scratch}"
BUILD_DIR="${BUILD_DIR:-build-rocm}"
JOBS="${JOBS:-$(nproc)}"
ROCM_DIR="${ROCM_DIR:-${ROCM_714:-$(rocm_dir)}}"
GPU_TARGETS="${GPU_TARGETS:-gfx1201}"
FAST=0
CCACHE_ON="${CCACHE:-1}"

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --path)      SCRATCH="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        --rocm)      ROCM_DIR="$2"; shift 2 ;;
        --gpu-target) GPU_TARGETS="$2"; shift 2 ;;
        --repo)      FORK="$2"; shift 2 ;;
        --fast)      FAST=1; shift ;;
        -h|--help)   usage 0 ;;
        -*)          echo "unknown option: $1" >&2; usage 2 ;;
        *)           break ;;
    esac
done
[ $# -ge 1 ] || usage 2
SUB="$1"; shift

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found" >&2; exit 2; }
BASE="$(jq -r '.base' "$MANIFEST")"
TIP="$(jq -r '.tip'  "$MANIFEST")"
case "$BUILD_DIR" in /*|..*|*..) echo "ERROR: --build-dir must be a plain relative path" >&2; exit 2 ;; esac
[ -d "$ROCM_DIR" ] || { echo "ERROR: ROCm install not found: $ROCM_DIR" >&2; exit 2; }
git -C "$FORK" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: $FORK is not a git checkout" >&2; exit 2; }

ISYSTEM="-isystem$ROCM_DIR/include"
CCACHE_LAUNCHERS=()
if [ "$CCACHE_ON" != 0 ] && command -v ccache >/dev/null 2>&1; then
    CCACHE_LAUNCHERS=(-DCMAKE_HIP_COMPILER_LAUNCHER=ccache
                      -DCMAKE_C_COMPILER_LAUNCHER=ccache
                      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
TARGETS_ALL=(all)
TARGETS_FAST=(llama-cli llama-server llama-bench llama-perplexity test-backend-ops)

ensure_worktree() {
    if [ -d "$SCRATCH/.git" ] || git -C "$SCRATCH" rev-parse --git-dir >/dev/null 2>&1; then
        return 0
    fi
    if [ -e "$SCRATCH" ] && [ -n "$(ls -A "$SCRATCH" 2>/dev/null)" ]; then
        echo "ERROR: $SCRATCH exists and is not a git worktree; move it aside" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$SCRATCH")"
    echo "==> creating worktree $SCRATCH at $BASE (from $FORK)"
    git -C "$FORK" worktree add --detach "$SCRATCH" "$BASE" || exit 2
}

configure_and_build() {
    local t0 t1
    local -a tgts
    if [ "$FAST" = 1 ]; then tgts=("${TARGETS_FAST[@]}"); else tgts=("${TARGETS_ALL[@]}"); fi
    if [ ! -f "$SCRATCH/$BUILD_DIR/CMakeCache.txt" ]; then
        # The embedded web UI is OFF (see the configure below): `ui-assets.cmake` then fetches nothing from
        # HuggingFace and runs no npm, so a fresh build dir is hermetic.  Nothing is provisioned (the
        # retired `ui-dist-cache.sh` mirror is recorded in WORKLOG.md, TODO #342).
        echo "==> configure $SCRATCH/$BUILD_DIR"
        HIPCXX="$ROCM_DIR/lib/llvm/bin/clang" HIP_PATH="$ROCM_DIR" \
        cmake -S "$SCRATCH" -B "$SCRATCH/$BUILD_DIR" \
                -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 \
                -DHIP_PLATFORM=amd -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS="$GPU_TARGETS" \
                -DCMAKE_C_COMPILER="$ROCM_DIR/lib/llvm/bin/clang" \
                -DAMDGPU_TARGETS="$GPU_TARGETS" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_COMPILER="$ROCM_DIR/lib/llvm/bin/clang++" \
                -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_DIR/lib" \
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_HIP_FLAGS="$ISYSTEM" \
                -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
                "${CCACHE_LAUNCHERS[@]}" > "$SCRATCH/$BUILD_DIR.configure.log" 2>&1 \
            || { echo "    CONFIGURE FAILED (log: $SCRATCH/$BUILD_DIR.configure.log)"; tail -5 "$SCRATCH/$BUILD_DIR.configure.log"; exit 1; }
    fi
    t0="$(date +%s)"
    echo "==> build $(git -C "$SCRATCH" rev-parse --short HEAD) -> ${tgts[*]} (-j $JOBS)"
    if ! cmake --build "$SCRATCH/$BUILD_DIR" --target "${tgts[@]}" -j "$JOBS" -- -k; then
        echo "    BUILD FAILED"
        exit 1
    fi
    t1="$(date +%s)"
    echo "    built in $((t1 - t0))s"
}

reset_to() {
    local rev="$1"
    git -C "$SCRATCH" rev-parse --verify "$rev^{commit}" >/dev/null 2>&1 \
        || { echo "ERROR: $rev not in $SCRATCH (fetch it into $FORK first)" >&2; exit 2; }
    git -C "$SCRATCH" rev-parse --verify "$rev^{commit}" >/dev/null 2>&1 || true
    git -C "$SCRATCH" checkout -q --detach "$rev" || exit 2
}

case "$SUB" in
    init)
        ensure_worktree
        configure_and_build
        ;;
    state)
        [ $# -ge 1 ] || { echo "ERROR: state needs REV" >&2; exit 2; }
        ensure_worktree
        reset_to "$1"
        [ -z "$(git -C "$SCRATCH" status --porcelain)" ] || { echo "ERROR: $SCRATCH is dirty; reset it first" >&2; exit 2; }
        configure_and_build
        ;;
    blocks)
        want="${1:-$BASE}"
        ensure_worktree
        git -C "$SCRATCH" checkout -q --detach "$want" || exit 2
        git -C "$SCRATCH" reset -q --hard
        git -C "$SCRATCH" branch -D scratch >/dev/null 2>&1 || true
        echo "==> applying the delivery set to $SCRATCH at $want"
        RDNA_BRANCH=scratch bash "$REPO_DIR/scripts/apply-all.sh" "$SCRATCH" || exit 1
        configure_and_build
        ;;
    arm)
        [ $# -ge 1 ] || { echo "ERROR: arm needs one or more patch files" >&2; exit 2; }
        ensure_worktree
        # A dirty worktree makes the checkout BELOW fail, and that used to be reported as "release tip not in
        # $SCRATCH" -- a misleading message, because the tip is reachable through the shared object store and
        # the actual cause was two modified files left behind by an earlier session.  Name the real cause.
        if [ -n "$(git -C "$SCRATCH" status --porcelain)" ]; then
            echo "ERROR: $SCRATCH has uncommitted changes; 'arm' checks out $TIP and would discard them." >&2
            git -C "$SCRATCH" status --short | head -10 >&2
            echo "ERROR: commit/stash them there, or run this arm with another --path" >&2
            exit 2
        fi
        git -C "$SCRATCH" checkout -q --detach "$TIP" 2>/dev/null \
            || { echo "ERROR: release tip $TIP is not a commit in $SCRATCH (run 'blocks' first, or fetch it into $FORK)" >&2; exit 2; }
        for p in "$@"; do
            [ -f "$p" ] || { echo "ERROR: no such patch: $p" >&2; exit 2; }
            echo "==> git am $p"
            git -C "$SCRATCH" am -3 "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")" \
                || { echo "    git am failed; resolve or 'git am --abort' in $SCRATCH" >&2; exit 1; }
        done
        configure_and_build
        ;;
    tip)
        ensure_worktree
        reset_to "$TIP"
        configure_and_build
        ;;
    keep)
        [ $# -ge 1 ] || { echo "ERROR: keep needs TAG" >&2; exit 2; }
        [ -d "$SCRATCH/$BUILD_DIR/bin" ] || { echo "ERROR: no build in $SCRATCH/$BUILD_DIR" >&2; exit 2; }
        rm -rf "$SCRATCH/$BUILD_DIR/bin-$1"
        cp -a "$SCRATCH/$BUILD_DIR/bin" "$SCRATCH/$BUILD_DIR/bin-$1"
        echo "==> kept $SCRATCH/$BUILD_DIR/bin-$1 (run it with LD_LIBRARY_PATH=$SCRATCH/$BUILD_DIR/bin-$1)"
        ;;
    gate)
        ensure_worktree
        bash "$REPO_DIR/scripts/gate-per-block-build.sh" --repo "$SCRATCH" "$@"
        ;;
    show)
        echo "path      : $SCRATCH"
        echo "rev       : $(git -C "$SCRATCH" rev-parse --short HEAD 2>/dev/null || echo '<none>') $(git -C "$SCRATCH" symbolic-ref -q --short HEAD 2>/dev/null || echo '(detached)')"
        echo "build dir : $SCRATCH/$BUILD_DIR"
        echo "base/tip  : $BASE / $TIP"
        ;;
    *) usage 2 ;;
esac
