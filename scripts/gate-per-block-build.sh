#!/usr/bin/env bash
# Per-block build gate -- the r2 "every one of the 16 commits builds the `all`
# target" standard.
#
# WHY THIS EXISTS:
#   The 16-block delivery is bisected and reviewed block by block, so every block
#   commit must build on its own.  Historically this regressed silently (the r2
#   fix landed 2026-10-05; by r49 blocks 06-14 no longer built) because only the
#   *tip* was ever built.  Fixes kept being folded into the last block instead of
#   the block that owns the broken code.  This gate is the instrument that makes
#   the regression impossible to miss.
#
# METHOD (do NOT use an incremental/reused build dir):
#   A reused build cache hides configure/header errors and reports a *wrong*
#   per-block list (the r2 lesson: the reused cache kept the tip's
#   GGML_CUDA_FA_QUANTS and made blocks 00-07 look like configure failures).  A
#   fresh `rm -rf "$BUILD_DIR"` + configure + `all` build per commit is the only
#
# COST, AND THE ONE RULE THAT DECIDES IT (measured 2026-10-25, gfx1201, ROCm 10.1.0, ccache 30 GiB):
#   The wipe itself is free.  The SAME content rebuilt at the SAME source path in a fresh build dir
#   takes 8.6s (343 TUs, all ccache direct hits); the same content in a NEW source path takes 5m56s
#   and 85 CPU-min (722 misses), because ccache's key includes the absolute source paths that reach
#   the compiler command line.  So the gate's cost is (a) the genuine content delta of each selected
#   block, plus (b) a FULL COLD BUILD OF THE FIRST BLOCK whenever --repo points at a path ccache has
#   never seen.  A block state is mostly a previous release's content (the files later blocks own are
#   unchanged in it), so the same --repo path across releases turns (b) into hits and leaves only (a):
#   one .cu ~15-30s, a widely included header (common.cuh, ggml-impl.h) ~3-5 min, an all-new
#   tree ~5-6 min.  Keep --repo on the long-lived checkout (the default ~/llama.cpp, or the one scratch
#   tree scripts/scratch-tree.sh maintains) and never give a release arm its own worktree path.
#   reliable method.  ccache makes it fast (only the per-block delta recompiles).
#
#   WHAT COSTS THE 35 MINUTES (measured 2026-10-25, r138 re-base, gfx1201, -j20):
#   the default target set is `all`, and one of its TUs dominates everything else:
#     tests/test-backend-ops.cpp  5m13s  (14.4k lines, one TU, compiles serially)
#     src/llama-context.cpp          5s
#   A re-base changes that test TU through the base (and blocks 04/05/07 edit it), so the ladder
#   sees ~4 *distinct* contents of it, i.e. ~21 of the ~35 minutes are spent compiling a test
#   binary that says nothing about whether a block's delivery code is sound.  Measured full
#   ladder: 2101s cold, 95s once every block tree is in ccache.
#
#   Two profiles, both honest:
#     * GATE_TARGETS_FAST (release flow): the libraries + the shipped binaries.  The op tests are
#       run separately at the tip (`test-backend-ops -o ...`), which is where they can actually
#       fail, so nothing is lost except "the test TU compiles in every intermediate tree".
#     * `all` (the r2 standard, the default here): use it when a commit touches tests/ or a
#       shared header, or once for a re-base when you want the strongest statement.
#   Do NOT read a fast-profile pass as "every target builds in every block" — it is not that claim.
#
# Usage:
#   ./gate-per-block-build.sh [options]
#
# Options / environment:
#   --repo DIR       REPO        llama.cpp checkout to walk (default: $LLAMA_REPO
#                                or ~/llama.cpp).  Must contain release.json's
#                                base..tip commits.
#   --base REV       BASE        base commit (default: release.json .base)
#   --tip REV        TIP         tip commit  (default: release.json .tip)
#   --build-dir NAME BUILD_DIR   build directory, relative to REPO (default
#                                build-per-block)
#   --jobs N         JOBS        build parallelism (default 16)
#   --rocm DIR       ROCM_DIR    ROCm install (default /opt/rocm-10.1.0-gfx120X;
#                                ROCM_714 is still honoured as an env alias)
#   --gpu-target T   GPU_TARGETS (default gfx1201)
#   --keep-going                 continue past the first failure (default: stop)
#   --changed-from REV           only build commits in BASE..TIP whose own delta
#                                (patch-id) differs from the commit with the same
#                                subject in BASE..REV (release speedup: pass the
#                                previous release tip); the tip is always built so
#                                the final combination stays compiler-verified
#   --fast                       build the release profile per commit instead of
#                                `all`: ggml-hip, llama, llama-cli, llama-server,
#                                llama-bench, llama-perplexity.  Drops the test and
#                                example/tool binaries; their coverage is the tip's
#                                own gates (the release flow builds the tip with
#                                `all` once and runs `test-backend-ops -o ...`
#                                there).  See the COST note for the measured reason
#                                (one test TU is 5m13s cold, and a re-base gives it
#                                ~4 distinct contents, i.e. ~21 of ~35 minutes)
#   --targets "T1 T2 ..."        an explicit target list (default: `all`)
#   --list-only                  print the commits that would be built, then exit
#
# Exit: 0 = every commit built, 1 = at least one failed, 2 = usage/prereq error.
#
# ccache is used when present (disable with CCACHE=0).  This is a local,
# compiler-only gate: it does not run the model or touch a GPU.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"

REPO="${LLAMA_REPO:-$HOME/llama.cpp}"
BUILD_DIR="build-per-block"
JOBS="${JOBS:-16}"
ROCM_DIR="${ROCM_DIR:-${ROCM_714:-/opt/rocm-10.1.0-gfx120X}}"
GPU_TARGETS="${GPU_TARGETS:-gfx1201}"
KEEP_GOING=0
LIST_ONLY=0
CHANGED_FROM=""
# The release profile: the libraries plus the binaries the delivery actually ships/uses.  It drops the
# test and example/tool binaries, whose coverage is the tip's own gates (the release flow builds the
# tip with `all` once, and `test-backend-ops -o ...` runs there).
GATE_TARGETS_FAST="ggml-hip llama llama-cli llama-server llama-bench llama-perplexity"
TARGETS="${GATE_TARGETS:-all}"
TARGETS_GIVEN=0
BASE=""
TIP=""
TIP_GIVEN=0

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)       REPO="$2"; shift 2 ;;
        --base)       BASE="$2"; shift 2 ;;
        --tip)        TIP="$2"; TIP_GIVEN=1; shift 2 ;;
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --rocm)       ROCM_DIR="$2"; shift 2 ;;
        --gpu-target) GPU_TARGETS="$2"; shift 2 ;;
        --keep-going) KEEP_GOING=1; shift ;;
        --changed-from) CHANGED_FROM="$2"; shift 2 ;;
        --targets)    TARGETS="$2"; TARGETS_GIVEN=1; shift 2 ;;
        --fast)       TARGETS="$GATE_TARGETS_FAST"; TARGETS_GIVEN=1; shift ;;
        --list-only)  LIST_ONLY=1; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

if [ -z "$BASE" ] || [ -z "$TIP" ]; then
    command -v jq >/dev/null || { echo "ERROR: jq is required to read release.json" >&2; exit 2; }
    [ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found" >&2; exit 2; }
    [ -n "$BASE" ] || BASE="$(jq -r '.base' "$MANIFEST")"
    [ -n "$TIP" ]  || TIP="$(jq -r '.tip' "$MANIFEST")"
fi

# A defaulted TIP names release.json's field, i.e. the LAST RELEASED chain -- not the chain in the
# checkout.  Run before the release commit and that silently re-verifies the previous release: with
# `--changed-from <prev tip>` every block's delta is unchanged by construction, so the gate prints
# "[skip] ... (delta unchanged)" for all 15 and a single tip build, i.e. a vacuous PASS (r205 hit
# exactly this).  The chain being released is a branch head, so say so and name the invocation.
if [ "$TIP_GIVEN" = 0 ] && command -v git >/dev/null 2>&1 && [ -d "$REPO/.git" ]; then
    head_rev="$(git -C "$REPO" rev-parse --verify -q HEAD || true)"
    tip_rev_c="$(git -C "$REPO" rev-parse --verify -q "$TIP" || true)"
    if [ -n "$head_rev" ] && [ -n "$tip_rev_c" ] && [ "$head_rev" != "$tip_rev_c" ] &&
       git -C "$REPO" merge-base --is-ancestor "$tip_rev_c" "$head_rev" 2>/dev/null; then
        echo "WARNING: --tip defaulted to release.json's tip ($(git -C "$REPO" rev-parse --short "$tip_rev_c"))," >&2
        echo "         but $REPO has $(git -C "$REPO" log -1 --format='%h %s' "$head_rev")" >&2
        echo "         ahead of it.  This gate would verify the PREVIOUS release's chain (every block then" >&2
        echo "         reads 'delta unchanged' and only the tip builds).  Pass" >&2
        echo "           --tip $head_rev" >&2
        echo "         to gate the chain you are about to release." >&2
    fi
fi
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERROR: $REPO is not a git checkout" >&2; exit 2; }
case "$BUILD_DIR" in /*|..*|*..) echo "ERROR: --build-dir must be a plain relative path" >&2; exit 2 ;; esac
[ -d "$ROCM_DIR" ] || { echo "ERROR: ROCm install not found: $ROCM_DIR" >&2; exit 2; }

# Put THIS tree's headers in front of /usr/include, where Fedora keeps a complete ROCm 7.1.x install.
# clang injects its own --hip-path/include AFTER /usr/include, so without this <hip/...>,
# <hipblas/...> and <rocblas/...> resolve to the distro headers while we link this tree's libraries.
# On ROCm 10.1 that is fatal (clang 24 no longer declares the __ocml_* intrinsics 7.1's
# amd_hip_bf16.h calls, so every TU including <hip/hip_bf16.h> fails); on 7.14 it silently compiled
# the distro headers and produced a binary no gate could tell apart.  A joined -isystem<path> takes
# no separate argument, so CMake >= 4.3 cannot swallow it when it appends --cuda-host-only.
# (Same rule and same guard as ~/bin/build-llama-rocm; do not pass a bare -mllvm here.)
ISYSTEM="-isystem$ROCM_DIR/include"

command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
git -C "$REPO" cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "ERROR: base $BASE not in $REPO" >&2; exit 2; }
git -C "$REPO" cat-file -e "$TIP^{commit}"  2>/dev/null || { echo "ERROR: tip $TIP not in $REPO" >&2; exit 2; }
if [ -n "$CHANGED_FROM" ]; then
    git -C "$REPO" cat-file -e "$CHANGED_FROM^{commit}" 2>/dev/null \
        || { echo "ERROR: --changed-from $CHANGED_FROM not in $REPO" >&2; exit 2; }
fi

CCACHE_LAUNCHERS=()
if [ "${CCACHE:-1}" != 0 ] && command -v ccache >/dev/null 2>&1; then
    CCACHE_LAUNCHERS=(-DCMAKE_HIP_COMPILER_LAUNCHER=ccache
                      -DCMAKE_C_COMPILER_LAUNCHER=ccache
                      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rboosts-per-block.XXXXXX")"
ORIG_REF="$(git -C "$REPO" symbolic-ref -q --short HEAD || git -C "$REPO" rev-parse HEAD)"
echo "==> per-block build gate: $BASE..$TIP in $REPO (targets: $TARGETS)"
echo "    logs: $LOG_DIR   build dir: $REPO/$BUILD_DIR   jobs: $JOBS"
[ -n "$CHANGED_FROM" ] && echo "    changed-from: $CHANGED_FROM (blocks whose own delta is unchanged are skipped; the tip is always built)"

# --changed-from: map subject -> patch-id for the reference chain.  Block subjects are
# stable, so a commit is "changed" when its own delta (patch-id, i.e. the diff to its
# parent) differs from the commit with the same subject in the reference (normally the
# previous release tip).  Comparing the tree instead would select every descendant of
# the first edited block; comparing the delta selects only the edited blocks.  The tip
# is always built so the final combination is always compiler-verified.
patch_id() { git -C "$REPO" show "$1" | git patch-id --stable | awk '{print $1}' | head -1; }
declare -A REF_PID=()
if [ -n "$CHANGED_FROM" ]; then
    while IFS= read -r rc; do
        REF_PID["$(git -C "$REPO" log -1 --format=%s "$rc")"]="$(patch_id "$rc")"
    done < <(git -C "$REPO" rev-list --reverse "$BASE..$CHANGED_FROM")
fi

TIP_REV="$(git -C "$REPO" rev-parse "$TIP")"
selected=()
for c in $(git -C "$REPO" rev-list --reverse "$BASE..$TIP"); do
    subj="$(git -C "$REPO" log -1 --format=%s "$c")"
    is_tip=0
    [ "$(git -C "$REPO" rev-parse "$c")" = "$TIP_REV" ] && is_tip=1
    if [ -n "$CHANGED_FROM" ] && [ "$is_tip" = 0 ]; then
        ref_pid="${REF_PID[$subj]:-}"
        if [ -n "$ref_pid" ] && [ "$ref_pid" = "$(patch_id "$c")" ]; then
            echo "==> [skip] ${c:0:9} $subj (delta unchanged from $CHANGED_FROM)"
            continue
        fi
    fi
    selected+=("$c")
done

if [ "$LIST_ONLY" = 1 ]; then
    echo "==> selected ${#selected[@]} commit(s):"
    for c in "${selected[@]}"; do
        printf '    %s %s\n' "${c:0:9}" "$(git -C "$REPO" log -1 --format=%s "$c")"
    done
    rm -rf "$LOG_DIR"
    exit 0
fi

total=0; ok=0; failed=0; t_gate0="$(date +%s)"; declare -a fail_list=()
for c in "${selected[@]}"; do
    total=$((total+1)); t_blk0="$(date +%s)"
    subj="$(git -C "$REPO" log -1 --format=%s "$c")"
    printf '==> [%2d] %s %s\n' "$total" "${c:0:9}" "$subj"
    git -C "$REPO" checkout -q "$c" || { echo "    CHECKOUT FAILED"; failed=$((failed+1)); fail_list+=("$c checkout"); [ "$KEEP_GOING" = 0 ] && break; continue; }

    rm -rf "$REPO/$BUILD_DIR"
    if ! HIPCXX="$ROCM_DIR/lib/llvm/bin/clang" HIP_PATH="$ROCM_DIR" \
        cmake -S "$REPO" -B "$REPO/$BUILD_DIR" \
            -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 \
            -DHIP_PLATFORM=amd -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS="$GPU_TARGETS" \
            -DCMAKE_C_COMPILER="$ROCM_DIR/lib/llvm/bin/clang" \
            -DAMDGPU_TARGETS="$GPU_TARGETS" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER="$ROCM_DIR/lib/llvm/bin/clang++" \
            -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_DIR/lib" \
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_HIP_FLAGS="$ISYSTEM" \
            "${CCACHE_LAUNCHERS[@]}" \
            > "$LOG_DIR/${total}.configure.log" 2>&1; then
        echo "    CONFIGURE FAILED (log: $LOG_DIR/${total}.configure.log)"
        tail -5 "$LOG_DIR/${total}.configure.log" | sed 's/^/      /'
        failed=$((failed+1)); fail_list+=("$c configure")
        [ "$KEEP_GOING" = 0 ] && break
        continue
    fi
    # Include-order guard: the flag must reach the HIP compile line (a re-edited CMAKE_HIP_FLAGS, a
    # CMake change appending after it, or a build dir reused from another ROCm tree all break it).
    if ! grep -qF -- "$ISYSTEM" "$REPO/$BUILD_DIR/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/flags.make" 2>/dev/null; then
        echo "    CONFIGURE LOST $ISYSTEM (HIP headers would resolve from /usr/include)"
        failed=$((failed+1)); fail_list+=("$c include-order")
        [ "$KEEP_GOING" = 0 ] && break
        continue
    fi
    if cmake --build "$REPO/$BUILD_DIR" --target $TARGETS -j "$JOBS" -- -k \
            > "$LOG_DIR/${total}.build.log" 2>&1; then
        echo "    BUILD OK ($(( $(date +%s) - t_blk0 ))s)"
        ok=$((ok+1))
    else
        echo "    BUILD FAILED (log: $LOG_DIR/${total}.build.log)"
        grep -E "error:" "$LOG_DIR/${total}.build.log" \
            | sed -E 's/^.*(error:)/      \1/' | sort -u | head -12
        failed=$((failed+1)); fail_list+=("$c build")
        [ "$KEEP_GOING" = 0 ] && break
    fi
done

git -C "$REPO" checkout -q "$ORIG_REF" 2>/dev/null || true

echo
if [ "$TARGETS" = "all" ]; then
    echo "==> result: $ok/$total built, $failed failed in $(( $(date +%s) - t_gate0 ))s (build dir wiped per commit)"
else
    echo "==> result: $ok/$total built, $failed failed in $(( $(date +%s) - t_gate0 ))s (build dir wiped per commit)"
    echo "    profile: --fast (NOT the r2 'all' standard): tests/ and the example/tool binaries are"
    echo "    not built here -- the tip's own release gates cover them (build the tip with 'all' once"
    echo "    and run test-backend-ops there)"
fi
echo "    (a first block that took minutes on an otherwise unchanged tree means ccache has never seen"
echo "     this --repo path: rerun on the long-lived checkout, see the COST note in this script)"
if [ "$failed" -ne 0 ]; then
    printf '    failed: %s\n' "${fail_list[*]}"
    echo "    full logs kept in $LOG_DIR"
    exit 1
fi
rm -rf "$LOG_DIR"
if [ "$TARGETS" = "all" ]; then
    echo "    all $total block commits build the all target"
else
    echo "    all $total block commits build the release target profile"
fi
exit 0
