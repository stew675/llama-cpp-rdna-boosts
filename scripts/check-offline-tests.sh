#!/usr/bin/env bash
# Offline-ctest check -- the r374 lesson (GitHub issue #382).
#
# WHY THIS EXISTS
#   The delivery changes llama.cpp's RUNTIME policy in places (block 10 refuses a mixed K/V pair for every
#   model, with its own diagnostic; blocks add options to `common/arg.cpp`; block 12/14 change internal
#   contracts), and upstream's own offline ctests encode the rules that were replaced.  Nothing in the release
#   flow could see that: a build compiles, the ROCm gates run models, and `validate-set.sh` checks patch
#   integrity -- none of them runs `ctest`.  So r368 shipped with `test-kv-cache` and `test-fp8-kv` failing on
#   every CPU-only tree while upstream b12597 was green, and the sweep that fixed it (#382) found two MORE:
#   `test-arg-parser` (an abbreviation pair ordered longest-first) and `test-llama-context` (a device-state
#   restore copying a zero-length piece).  Four failures, three blocks, one release -- all of them visible to a
#   single command.
#
# WHAT IT CHECKS
#   The default block set applied at `release.json.base`, configured CPU-only with the test suite enabled
#   (`-DLLAMA_BUILD_TESTS=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF`), built once, then `ctest` over every test.
#   It FAILS on any test failure except the ones below, which need something this check cannot require of a
#   box (and which are not the delivery's):
#       test-tokenizers-ggml-vocabs   clones ggml-vocabs and needs git-lfs (a pointer file is not a GGUF)
#       test-jinja-py                 needs the python `jinja2` module
#       test-conversion-fp8           needs `torch`
#       test-kv-cache-scales          needs `torch`
#   A skipped-by-environment test is REPORTED, never silently dropped, and `--strict` requires them to pass too
#   (use it on a box that has git-lfs + jinja2 + torch).  The delivery's own four tests -- `test-kv-cache`,
#   `test-fp8-kv`, `test-arg-parser`, `test-llama-context` -- are never exempt: they are the reason the check
#   exists.
#
# USAGE
#   ./scripts/check-offline-tests.sh [--build-dir DIR] [--fork PATH] [--jobs N] [--keep] [--strict]
#   --build-dir DIR   reuse an existing CPU-only test build (configured from the release tree) and only run
#                     ctest.  The fast path once you have a build; DIR must contain `CTestTestfile.cmake`.
#   --fork PATH       the llama.cpp checkout to clone the base from (default: ~/llama.cpp)
#   --jobs N          build parallelism (default: nproc)
#   --keep            keep the temporary tree and its build (printed), for a repro
#   --strict          no environment exemptions (fail on any test)
#
# COST: ~6-10 min cold (one CPU-only build of the whole tree with tests), ~1 min with --build-dir.
# Exit: 0 = every test the delivery can run here passes; 1 = a failure; 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${HOME}/llama.cpp"
BUILD_DIR=""
JOBS="$(nproc 2>/dev/null || echo 8)"
KEEP=0
STRICT=0

# tests that need a package / a git extension this check cannot require a box to have; they are NOT the
# delivery's, and the four tests that ARE the delivery's must never appear here
ALLOW=(
    test-tokenizers-ggml-vocabs   # git-lfs
    test-jinja-py                 # python jinja2
    test-conversion-fp8           # python torch
    test-kv-cache-scales          # python torch
)
REQUIRED=(
    test-kv-cache
    test-fp8-kv
    test-arg-parser
    test-llama-context
)

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --build-dir) [ $# -ge 2 ] || { echo "ERROR: --build-dir needs a path" >&2; exit 2; }; BUILD_DIR="$2"; shift 2 ;;
        --fork)      [ $# -ge 2 ] || { echo "ERROR: --fork needs a path" >&2; exit 2; };      FORK="$2"; shift 2 ;;
        --jobs)      [ $# -ge 2 ] || { echo "ERROR: --jobs needs a number" >&2; exit 2; };    JOBS="$2"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        --strict)    STRICT=1; shift ;;
        *) echo "ERROR: unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

[ -f "$REPO_DIR/release.json" ] || { echo "ERROR: $REPO_DIR/release.json is missing" >&2; exit 2; }
BASE="$(jq -r '.base' "$REPO_DIR/release.json" 2>/dev/null || true)"
[ -n "$BASE" ] && [ "$BASE" != "null" ] || { echo "ERROR: release.json has no .base" >&2; exit 2; }

WORK=""
cleanup() {
    if [ -n "$WORK" ] && [ "$KEEP" -eq 0 ]; then
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

if [ -n "$BUILD_DIR" ]; then
    [ -f "$BUILD_DIR/CTestTestfile.cmake" ] || {
        echo "ERROR: --build-dir $BUILD_DIR is not a configured test build (no CTestTestfile.cmake)" >&2
        echo "ERROR: configure one with -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF" >&2
        exit 2
    }
    BUILD="$BUILD_DIR"
    echo "==> ctest in $BUILD (the tree it was built from is not re-checked here)"
else
    [ -e "$FORK/.git" ] || { echo "ERROR: no llama.cpp checkout at $FORK (pass --fork)" >&2; exit 2; }
    git -C "$FORK" rev-parse --verify "$BASE^{commit}" >/dev/null 2>&1 || {
        echo "ERROR: base $BASE is not in $FORK" >&2; exit 2; }

    WORK="$(mktemp -d "${TMPDIR:-/tmp}/rdna-offline-tests.XXXXXX")" || exit 2
    echo "==> materialising the default set at $BASE in $WORK/llama.cpp"
    # a shared clone of the fork carries the base commit without a network fetch; the patches are applied with
    # the tested flow (scripts/apply-all.sh, strict git am), never by hand
    git clone -q --shared --no-hardlinks "$FORK" "$WORK/llama.cpp" || { echo "ERROR: clone failed" >&2; exit 2; }
    (
        cd "$WORK/llama.cpp" || exit 1
        git checkout -q "$BASE" || exit 1
        git branch -D rdna-boosts >/dev/null 2>&1 || true
        # the tree check inside apply-all.sh compares against release.json, which is the record we are
        # checking -- so a stale record fails HERE, at apply time, with apply-all's own message
        bash "$REPO_DIR/scripts/apply-all.sh" . > "$WORK/apply.log" 2>&1
    ) || { echo "ERROR: applying the set at $BASE failed -- see $WORK/apply.log" >&2; cat "$WORK/apply.log" >&2; exit 1; }
    grep -E "^All [0-9]+ patches applied cleanly" "$WORK/apply.log" | sed 's/^/    /'

    BUILD="$WORK/build-tests-cpu"
    echo "==> configuring a CPU-only test build ($BUILD)"
    cmake -S "$WORK/llama.cpp" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
        > "$WORK/configure.log" 2>&1 || {
        echo "ERROR: configure failed -- see $WORK/configure.log" >&2; tail -20 "$WORK/configure.log" >&2; exit 1; }

    echo "==> building (this is the slow part: one CPU build with the test suite)"
    cmake --build "$BUILD" -j "$JOBS" > "$WORK/build.log" 2>&1 || {
        echo "ERROR: the CPU-only test build FAILED -- see $WORK/build.log" >&2
        grep -E "error:|FAILED" "$WORK/build.log" | head -20 >&2
        exit 1; }
    echo "    built"
fi

LOG="$(mktemp "${TMPDIR:-/tmp}/rdna-offline-ctest.XXXXXX")"
echo "==> ctest --output-on-failure"
( cd "$BUILD" && ctest --output-on-failure --timeout 900 ) > "$LOG" 2>&1
CTEST_RC=$?

mapfile -t failed < <(grep -E "^[[:space:]]+[0-9]+ - " "$LOG" | sed -E 's/^[[:space:]]+[0-9]+ - ([^ ]+).*/\1/' | sort -u)

if [ ${#failed[@]} -eq 0 ]; then
    grep -E "tests passed" "$LOG" | sed 's/^/    /'
    echo "OFFLINE_TESTS=PASS"
    rm -f "$LOG"
    exit 0
fi

exempt=()
blocking=()
for t in "${failed[@]}"; do
    hit=0
    for a in "${ALLOW[@]}"; do
        [ "$t" = "$a" ] && hit=1
    done
    if [ "$hit" -eq 1 ] && [ "$STRICT" -eq 0 ]; then
        exempt+=("$t")
    else
        blocking+=("$t")
    fi
done

if [ ${#exempt[@]} -gt 0 ]; then
    echo "INFO  exempt (a package / git-lfs this box lacks; --strict requires them too):"
    printf '        %s\n' "${exempt[@]}"
fi

if [ ${#blocking[@]} -eq 0 ]; then
    echo "OFFLINE_TESTS=PASS (${#exempt[@]} environment-exempt test(s))"
    rm -f "$LOG"
    exit 0
fi

echo "FAIL  ${#blocking[@]} offline test(s) fail on the delivery tree:" >&2
printf '        %s\n' "${blocking[@]}" >&2
echo "      these are upstream's own tests over the delivery's policy: fix the TEST or the policy" >&2
echo "      (a policy change owes the test alignment in the same block; see AGENTS.md, 'Report findings')," >&2
echo "      then re-run.  Full ctest log:" >&2
echo "        $LOG" >&2
timeout 5 tail -60 "$LOG" >&2 || true
exit 1
