#!/bin/bash
# gate-rebase-release.sh -- the whole pre-release verdict for a RE-BASE, in one command.
#
# WHY: a re-base has to answer four questions at once, and doing them by hand across a long session is how
# things get half-checked (this campaign shipped a wrong launch count, a single-run census and an unpinned
# attribution base that way).  This runs the existing, already-trusted gates in the right order, scopes every
# GPU-fault check to its own window, and prints ONE verdict.  It deliberately contains no threshold logic of
# its own beyond the per-arm performance tolerance; each step delegates to the gate that owns the rule.
#
# WHAT IT CHECKS (each step must pass; nothing is skipped silently -- a step that cannot run FAILS):
#   1. state      both repos clean, the fork tip/tree, `master` pinned at the fork point, the builds present
#   2. build      a clean build per block commit            -> scripts/gate-per-block-build.sh --fast
#   3. numerics   the prefill-logit KLD gate                -> scripts/gate-prefill-logits.sh
#   4. ops        tests/test-backend-ops: MUL_MAT_ID (incl. the duplicate-id oracle), MUL_MAT
#   5. text       same-seed greedy text on the coherence arms, against recorded hashes
#   6. perf       interleaved wall clock of every arm against the SHIPPED build, per-arm ratio
#   7. faults     scripts/check-gpu-faults.sh, scoped to this run's start timestamp
#
# USAGE
#   scripts/gate-rebase-release.sh                       # everything
#   scripts/gate-rebase-release.sh --quick               # skip steps 2 and 3 (the slow ones)
#   scripts/gate-rebase-release.sh --perf-only           # just step 6 (an A/B while iterating)
#   scripts/gate-rebase-release.sh --only text,faults    # one or more steps by name (build|logits|ops|text|perf)
#                                                       # `state` is included only when named
#   scripts/gate-rebase-release.sh --tolerance 1.0       # allowed regression per arm, percent (default 1.0)
#
# ENV (all optional)
#   FORK          fork checkout                 (default: ~/llama.cpp)
#   NEW_BUILD     the re-based build            (default: $FORK/build-rocm)
#   BASE_BUILD    the SHIPPED build to beat     (default: $FORK/build-rocm-r235)
#   BASE_COMMIT   the fork point `master` pins  (default: a691b2345)
#   ROUNDS        A/B rounds                    (default: 3)
#   REPS          llama-bench -r                (default: 5)
#   ARMS          "name|model|extra llama-bench args" lines (defaults below)
#   LOGDIR        output dir                    (default: mktemp -d)
#
# Exit: 0 = every step passed, 1 = at least one FAILED, 2 = usage/startup error.

set -uo pipefail

FORK="${FORK:-$HOME/llama.cpp}"
NEW_BUILD="${NEW_BUILD:-$FORK/build-rocm}"
BASE_BUILD="${BASE_BUILD:-$FORK/build-rocm-r235}"
BASE_COMMIT="${BASE_COMMIT:-a691b2345}"
ROUNDS="${ROUNDS:-3}"
REPS="${REPS:-5}"
TOLERANCE="${TOLERANCE:-1.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUICK=0 PERF_ONLY=0 ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)     QUICK=1 ;;
        --perf-only) PERF_ONLY=1 ;;
        --only)      ONLY="$2"; shift ;;
        --tolerance) TOLERANCE="$2"; shift ;;
        --rounds)    ROUNDS="$2"; shift ;;
        --reps)      REPS="$2"; shift ;;
        -h|--help)   sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "gate-rebase-release: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
START_EPOCH="$(date +%s)"
START_HUMAN="$(date)"

# The performance arms: name | model | llama-bench args (device pin included -- every multi-device arm MUST pin
# HIP_VISIBLE_DEVICES or the 9950X3D's iGPU (gfx1036) is enumerated and the load aborts with "invalid kernel file").
DEFAULT_ARMS="\
dense-27B-prefill|/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf|HIP_VISIBLE_DEVICES=0 -p 512 -n 0 -ngl 99
moe-35B-prefill|/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf|HIP_VISIBLE_DEVICES=0,1,2 -sm tensor -p 512 -n 0 -ngl 99
moe-35B-decode|/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf|HIP_VISIBLE_DEVICES=0,1,2 -sm tensor -p 0 -n 128 -ngl 99
"
ARMS="${ARMS:-$DEFAULT_ARMS}"

# The coherence arms: name | model | extra args | expected hash.  A hash is only valid against the prompt and
# args it was recorded with, so they live together here.  Update deliberately, never to make a run pass.
DEFAULT_TEXT_ARMS="\
qwen35-4B|/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf|HIP_VISIBLE_DEVICES=0 -ngl 99|1c5d32ac537d
"
TEXT_ARMS="${TEXT_ARMS:-$DEFAULT_TEXT_ARMS}"

results=()
pass_n=0 fail_n=0

record() { # record <status> <step> <detail>   (PASS|FAIL|WARN|INFO; only FAIL fails the gate)
    results+=("$1|$2|$3")
    case "$1" in
        PASS) pass_n=$((pass_n+1)) ;;
        FAIL) fail_n=$((fail_n+1)) ;;
    esac
    printf '  [%s] %-22s %s\n' "$1" "$2" "$3"
}

need_file() { [ -e "$1" ] || { record FAIL "$2" "missing $1"; return 1; }; return 0; }

# split_arm <tokens...> -- an arm line mixes environment assignments with binary flags; they must be separated
# (`env VAR=x -ngl 99` makes env parse -ngl as its own option).  Anything containing '=' is an assignment.
split_arm() {
    ENVS=(); FLAGS=()
    local t
    for t in "$@"; do
        case "$t" in *=*) ENVS+=("$t") ;; *) FLAGS+=("$t") ;; esac
    done
}

# ---------------------------------------------------------------------------------------------------------------
# 1. state
# ---------------------------------------------------------------------------------------------------------------
step_state() {
    echo "== 1. state =============================================================="
    local ok=1
    need_file "$FORK/.git" state || ok=0
    need_file "$NEW_BUILD/bin/llama-bench" state || ok=0
    need_file "$BASE_BUILD/bin/llama-bench" state || ok=0

    if [ -d "$FORK/.git" ]; then
        local dirty; dirty="$(git -C "$FORK" status --porcelain | head -5)"
        [ -z "$dirty" ] && record PASS tree-clean "fork working tree clean" \
                        || { record FAIL tree-clean "fork tree is dirty:"$'\n'"$dirty"; ok=0; }

        local master; master="$(git -C "$FORK" log --oneline -1 "$BASE_COMMIT" 2>/dev/null)"
        [ -n "$master" ] && record PASS base-commit "fork point $BASE_COMMIT resolves: ${master:0:60}" \
                         || { record FAIL base-commit "$BASE_COMMIT does not resolve in $FORK"; ok=0; }

        # The attribution base must be a SHA, not a branch: a local commit on `master` silently moves what every
        # "unpatched upstream" arm measures (this bit the campaign on 2026-11-08, TODO #303).
        local mhead; mhead="$(git -C "$FORK" rev-parse --short=9 master 2>/dev/null)"
        [ "$mhead" = "${BASE_COMMIT:0:9}" ] && record PASS master-pinned "master == $BASE_COMMIT" \
            || record WARN master-pinned "master is $mhead, expected $BASE_COMMIT -- build the attribution base from 'git checkout --detach $BASE_COMMIT', never from master"

        record INFO fork-tip "$(git -C "$FORK" rev-parse --short HEAD) tree $(git -C "$FORK" rev-parse --short 'HEAD^{tree}')"
    fi
    return $(( ok == 1 ? 0 : 1 ))
}

# ---------------------------------------------------------------------------------------------------------------
# 2. per-block build
# ---------------------------------------------------------------------------------------------------------------
step_build() {
    echo "== 2. build (clean, per block commit) ==================================="
    local log="$LOGDIR/gate-per-block-build.log"
    if bash "$REPO/scripts/gate-per-block-build.sh" --fast > "$log" 2>&1; then
        record PASS per-block-build "clean build per block commit (--fast)"
    else
        record FAIL per-block-build "see $log"
    fi
}

# ---------------------------------------------------------------------------------------------------------------
# 3. prefill-logit KLD
# ---------------------------------------------------------------------------------------------------------------
step_logits() {
    echo "== 3. numerics: prefill-logit KLD ======================================="
    local log="$LOGDIR/gate-prefill-logits.log"
    if bash "$REPO/scripts/gate-prefill-logits.sh" > "$log" 2>&1; then
        record PASS prefill-logits "$(grep -iE 'mean KLD|PASS' "$log" | tail -1 | cut -c1-90)"
    else
        record FAIL prefill-logits "see $log"
    fi
}

# ---------------------------------------------------------------------------------------------------------------
# 4. ops suites
# ---------------------------------------------------------------------------------------------------------------
step_ops() {
    echo "== 4. ops: test-backend-ops ============================================="
    local test_bin="$NEW_BUILD/bin/test-backend-ops" log line
    if ! need_file "$test_bin" ops; then return 1; fi
    for op in MUL_MAT_ID MUL_MAT; do
        log="$LOGDIR/ops-$op.log"
        if HIP_VISIBLE_DEVICES=0 timeout 3600 "$test_bin" -o "$op" > "$log" 2>&1; then
            line="$(grep -E '[0-9]+/[0-9]+ tests passed' "$log" | tail -1 | tr -s ' ')"
            record PASS "ops-$op" "$line"
        else
            record FAIL "ops-$op" "$(grep -E '[0-9]+/[0-9]+ tests passed|Failing' "$log" | tail -2 | tr '\n' ' ')"
        fi
    done
}

# ---------------------------------------------------------------------------------------------------------------
# 5. greedy text (the purity/coherence guarantee)
# ---------------------------------------------------------------------------------------------------------------
step_text() {
    echo "== 5. text: same-seed greedy ============================================"
    local cli="$NEW_BUILD/bin/llama-cli" ex="$REPO/scripts/extract-generated.py"
    if ! need_file "$cli" text || ! need_file "$ex" text; then return 1; fi
    local name model args expect log got
    while IFS='|' read -r name model args expect; do
        [ -z "${name:-}" ] && continue
        case "$name" in \#*) continue ;; esac
        if ! need_file "$model" "text-$name"; then continue; fi
        log="$LOGDIR/text-$name.log"
        # No -lv on a text arm: the same stream carries the generated text (see AGENTS.md).
        # shellcheck disable=SC2086
        split_arm $args
        if ! env "${ENVS[@]}" timeout 1800 "$cli" -m "$model" "${FLAGS[@]}" \
                -p "The capital of France is" -n 20 --seed 42 --temp 0 \
                --no-display-prompt --single-turn > "$log" 2>&1; then
            record FAIL "text-$name" "llama-cli failed, see $log"; continue
        fi
        got="$(python3 "$ex" "$log" 2>/dev/null | tail -1 | sed -E 's/.*sha=//' | tr -d ' \n')"
        if [ "$expect" = "UNSET" ]; then
            record WARN "text-$name" "hash $got -- no recorded hash for this arm; record it in the gate's TEXT_ARMS"
        elif [ "$got" = "$expect" ]; then
            record PASS "text-$name" "$got"
        else
            record FAIL "text-$name" "got $got, expected $expect (see $log)"
        fi
    done <<< "$TEXT_ARMS"
}

# ---------------------------------------------------------------------------------------------------------------
# 6. performance A/B against the shipped build
# ---------------------------------------------------------------------------------------------------------------
bench_tps() { # bench_tps <binary> <model> <arm args...> -> t/s on stdout
    local bin="$1" model="$2"; shift 2
    local out
    split_arm "$@"
    out="$(env "${ENVS[@]}" timeout 1800 "$bin/bin/llama-bench" -m "$model" -r "$REPS" "${FLAGS[@]}" \
            2>>"$LOGDIR/bench.err")"
    printf '%s' "$(printf '%s\n' "$out" | grep -oE '\|\s+[0-9]+\.[0-9]+ ±' | grep -oE '[0-9]+\.[0-9]+' | tail -1)"
}

step_perf() {
    echo "== 6. perf: new build vs the SHIPPED build (interleaved, -r $REPS, $ROUNDS rounds) =="
    local name model args n b r ok=1
    while IFS='|' read -r name model args; do
        [ -z "${name:-}" ] && continue
        if ! need_file "$model" "perf-$name"; then ok=0; continue; fi
        local ratios=()
        for r in $(seq 1 "$ROUNDS"); do
            n="$(bench_tps "$NEW_BUILD"  "$model" $args)"
            b="$(bench_tps "$BASE_BUILD" "$model" $args)"
            if [ -z "$n" ] || [ -z "$b" ] || [ "$b" = "0" ]; then
                record FAIL "perf-$name" "round $r produced no measurement (new='$n' base='$b')"; ok=0; break
            fi
            ratios+=("$(python3 -c "print('%+.3f' % (100*($n/$b-1)))")")
        done
        [ ${#ratios[@]} -eq 0 ] && continue
        local verdict
        verdict="$(python3 - "$TOLERANCE" "${ratios[@]}" <<'PY'
import sys, statistics
tol = float(sys.argv[1]); vals = [float(x) for x in sys.argv[2:]]
mean = statistics.mean(vals)
spread = max(vals) - min(vals)
status = "PASS" if mean >= -tol else "FAIL"
print(f"{status} {mean:+.2f}% (rounds: {' '.join(f'{v:+.2f}' for v in vals)}; spread {spread:.2f}; tol {-tol:+.1f})")
PY
)"
        case "$verdict" in
            PASS*) record PASS "perf-$name" "${verdict#PASS }" ;;
            *)     record FAIL "perf-$name" "${verdict#FAIL }"; ok=0 ;;
        esac
    done <<< "$ARMS"
    return $(( ok == 1 ? 0 : 1 ))
}

# ---------------------------------------------------------------------------------------------------------------
# 7. GPU faults, scoped to this run
# ---------------------------------------------------------------------------------------------------------------
step_faults() {
    echo "== 7. GPU faults (scoped to this run) =================================="
    local log="$LOGDIR/gate-faults.log"
    local args=() f
    for f in "$LOGDIR"/*.log; do [ -e "$f" ] && args+=(--log "$f"); done
    if bash "$REPO/scripts/check-gpu-faults.sh" --since "@$START_EPOCH" "${args[@]}" > "$log" 2>&1; then
        record PASS gpu-faults "no real GPU fault in this run's logs"
    else
        record FAIL gpu-faults "$(grep -iE 'fault|slab|page' "$log" | head -3 | tr '\n' ' ')"
    fi
}

# ---------------------------------------------------------------------------------------------------------------
echo "gate-rebase-release: new=$NEW_BUILD base=$BASE_BUILD fork=$FORK base-commit=$BASE_COMMIT"
echo "  started $START_HUMAN, logs in $LOGDIR"
echo

wanted() { # wanted <step> -- --only selects a comma-separated subset, everything when unset
    [ -z "$ONLY" ] && return 0
    case ",$ONLY," in *",$1,"*) return 0 ;; esac
    return 1
}

if [ "$PERF_ONLY" = 1 ]; then
    step_perf
else
    wanted state  && step_state
    if [ "$QUICK" = 1 ]; then record WARN per-block-build "skipped (--quick)"; else wanted build  && step_build;  fi
    if [ "$QUICK" = 1 ]; then record WARN prefill-logits "skipped (--quick)";  else wanted logits && step_logits; fi
    wanted ops    && step_ops
    wanted text   && step_text
    wanted perf   && step_perf
    wanted faults && step_faults
fi

echo
echo "== verdict =============================================================="
printf '%s\n' "${results[@]}" | awk -F'|' '{printf "  %-5s %-22s %s\n", $1, $2, $3}'
echo
printf '  %d passed, %d failed  (%s)\n' "$pass_n" "$fail_n" "$LOGDIR"
if [ "$fail_n" -gt 0 ]; then
    echo "  VERDICT: NOT RELEASABLE"
    exit 1
fi
echo "  VERDICT: RELEASABLE"
exit 0
