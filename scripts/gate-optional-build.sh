#!/usr/bin/env bash
# Optional-artifact build gate -- compile the OPTIONAL-applied tree, which nothing else does.
#
# WHY THIS EXISTS (issue #371, r367)
#   `scripts/validate-set.sh` check 8 proves every `optional/*/01xx-*.patch` still applies STRICTLY on the
#   default tree and produces its recorded tree -- but a tree is not a build.  Nothing in the delivery ever
#   compiled an optional-applied checkout, so when the r360 re-cut rebased `0101-mixed-kv` onto block 10's
#   new `ggml_cuda_fattn_kv_pair_native_host` it kept both functions and dropped the outer one's closing
#   brace.  The patch applied, the tree re-verified, every release gate stayed green -- and the tree did not
#   compile (a nested function definition, illegal in C++), for seven releases (r360..r366), until the
#   reporter hit it on Windows.  This gate is the missing step: apply the default set at `release.json.base`,
#   apply EVERY optional artifact in numeric order (the `--include-optional-all` consumer path), and build
#   the `all` target once.
#
# WHAT IT CHECKS
#   HARD  * the applied default tree is `release.json.tree` (the apply is not what is under test, but a
#           mis-cut chain must fail here rather than as a confusing compile error).
#   HARD  * every optional patch applies on top of it (numeric order; a later artifact may build on an
#           earlier one -- that is the contract apply-all.sh gives a consumer).
#   HARD  * the cumulative tree equals `release.json.optional_tree`.  scratch-tree applies with `git am -3`,
#           so this is what proves the merge produced the delivery's tree and not a plausible other one.
#   HARD  * the `all` target builds at that tree (the whole point: an optional tree is a tree a consumer
#           compiles, so it owes the same build the default set owes).
#   INFO  * wall time, the tree and the artifact count.
#
#   WHAT IT CANNOT FORCE.  (a) An artifact's own *claims*: `scripts/gate-mixed-kv.sh` is `0101`'s gate and is
#   run separately on this build (BUILD=<path>/bin).  (b) A build of each artifact ALONE -- the union is what
#   `--include-optional-all` hands a consumer, and validate-set check 8 already proves each one's own tree;
#   two artifacts that define the same symbol differently would still show up here, one that only breaks in
#   its own tree would not.  (c) A non-AMD backend: the gate configures HIP only, like the delivery.
#
# USAGE
#   ./gate-optional-build.sh [options]
#     --tip SHA          default-tree tip to apply the artifacts on (default: release.json.tip)
#     --expect-tree SHA  cumulative optional tree to assert (default: release.json.optional_tree)
#     --path DIR         scratch worktree            (default: $HOME/llama-scratch; one path forever --
#                        ccache keys on the absolute source path, see scripts/scratch-tree.sh)
#     --build-dir NAME   build directory inside it   (default: build-rocm-optional, so the canonical
#                        `build-rocm` is never disturbed)
#     --jobs N / --gpu-target T / --rocm DIR         passed to scripts/scratch-tree.sh
#   It reads release.json, so in a release run it after scripts/make-release.sh (which records the tip and
#   the optional trees); before that cut, pass --tip/--expect-tree explicitly.
#   Exit: 0 pass, 1 a check/build failed, 2 usage or prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/release.json"
TIP=""
EXPECT_TREE=""
SCRATCH="${SCRATCH_PATH:-$HOME/llama-scratch}"
BUILD_DIR="${BUILD_DIR:-build-rocm-optional}"
PASSTHRU=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tip)          TIP="$2"; shift 2 ;;
        --expect-tree)  EXPECT_TREE="$2"; shift 2 ;;
        --path)         SCRATCH="$2"; shift 2 ;;
        --build-dir)    BUILD_DIR="$2"; shift 2 ;;
        --jobs)         PASSTHRU+=(--jobs "$2"); shift 2 ;;
        --gpu-target)   PASSTHRU+=(--gpu-target "$2"); shift 2 ;;
        --rocm)         PASSTHRU+=(--rocm "$2"); shift 2 ;;
        -h|--help)      sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST not found" >&2; exit 2; }
TIP="${TIP:-$(jq -r '.tip // empty' "$MANIFEST")}"
EXPECT_TREE="${EXPECT_TREE:-$(jq -r '.optional_tree // empty' "$MANIFEST")}"
[ -n "$TIP" ] || { echo "ERROR: no default tip (release.json.tip is empty; pass --tip)" >&2; exit 2; }

# numeric order, exactly what apply-all.sh --include-optional-all does
OPT_PATCHES=()
while IFS= read -r _p; do OPT_PATCHES+=("$_p"); done < <(
    find "$REPO_DIR/optional" -mindepth 2 -maxdepth 2 -name '01[0-9][0-9]-*.patch' \
        | while IFS= read -r _q; do printf '%s\t%s\n' "$(basename "$_q" | cut -c1-4)" "$_q"; done \
        | LC_ALL=C sort -n | cut -f2-)
N_OPT="${#OPT_PATCHES[@]}"
if [ "$N_OPT" -eq 0 ]; then
    echo "ERROR: no optional artifacts under optional/*/01xx-*.patch -- nothing to build" >&2
    exit 2
fi
[ -n "$EXPECT_TREE" ] || { echo "ERROR: no cumulative optional tree (release.json.optional_tree is empty; pass --expect-tree)" >&2; exit 2; }

echo "==> optional-artifact build gate"
echo "    default tip : $TIP"
echo "    artifacts   : $N_OPT ($(for p in "${OPT_PATCHES[@]}"; do basename "$p" | cut -d- -f1; done | paste -sd, -))"
echo "    scratch     : $SCRATCH ($BUILD_DIR)"
echo "    expect tree : $EXPECT_TREE"

# scratch-tree.sh `arm` checks the release tip out in the scratch worktree and `git am -3`s each patch,
# then builds `all`.  Run it in NUMERIC order, one invocation with every patch.
ARGS=(--path "$SCRATCH" --build-dir "$BUILD_DIR")
ARGS+=("${PASSTHRU[@]}")

bash "$REPO_DIR/scripts/scratch-tree.sh" "${ARGS[@]}" arm "${OPT_PATCHES[@]}" || exit 1

GOT_TREE="$(git -C "$SCRATCH" rev-parse HEAD^{tree} 2>/dev/null || true)"
if [ -z "$GOT_TREE" ]; then
    echo "FAIL: cannot read the applied tree in $SCRATCH" >&2
    exit 1
fi
if [ "$GOT_TREE" != "$EXPECT_TREE" ]; then
    echo "FAIL: cumulative optional tree $GOT_TREE != expected $EXPECT_TREE" >&2
    echo "      the artifacts did not reproduce the delivery's tree (a stale/mis-based cut, or a merge" >&2
    echo "      that resolved to different content); re-cut them against the default tree." >&2
    exit 1
fi
echo "    applied tree OK ($GOT_TREE)"
[ -x "$SCRATCH/$BUILD_DIR/bin/llama-cli" ] || { echo "FAIL: no llama-cli after the build" >&2; exit 1; }
echo "    artifacts' own gates may now run against BUILD=$SCRATCH/$BUILD_DIR"
echo "OPTIONAL_BUILD=PASS ($N_OPT artifacts, tree $GOT_TREE, all target)"
