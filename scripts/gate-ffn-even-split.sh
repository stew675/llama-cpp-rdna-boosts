#!/usr/bin/env bash
# The OPT-IN `-of tp-ffn-even` gate (r338, block 09, contributor PR #342).
#
# WHY THIS EXISTS
#   `-sm tensor` splits an FFN tensor with granularity `lcm(blck_size, 128)`, so a 640-wide expert set on two
#   devices floors the per-device share to 256 and splits **256 / 384** -- the light GPU waits in every MoE
#   layer's all-reduce.  `-of tp-ffn-even` (`LLAMA_TP_FFN_EVEN=1`) lowers the granularity toward
#   `lcm(blck_size, 64)` when the size divides evenly, so the same shape splits **320 / 320**; measured
#   +7 % `draft-mtp` decode and +4 % prefill there (`benchmarks/2026-11-21-ffn-even-split.md`).
#
#   The granularity IS the reduction's grouping, so enabling it moves the logits of every shape it reaches:
#   mean KLD 0.0140 / same-top-p 96.47 % on the 2-device Flash-Next point, against a `-ub 4096` vs `-ub 3840`
#   reordering control of 0.0152 / 96.52 % on the same point -- a reordering, not a defect, but far outside the
#   prefill-logit gate's rounding-level limit (0.005 / 98 %).  That is why it is **OFF by default** and why this
#   gate exists: the DELIVERY default must stay byte-identical, the opt-in must reach the mechanism, and every
#   spelling of the switch must mean the same thing.
#
# WHAT IT CHECKS
#   A. THE DEFAULT IS INERT.  No `-of`, no `LLAMA_TP_FFN_EVEN`, no `LLAMA_ARG_OPTIONAL_FEATURES`: no
#      `FFN granularity` line is printed and the greedy text is the recorded default reference.  A default that
#      carried the reordering is the promotion bug this gate exists to catch.
#   B. THE TAG REACHES THE MECHANISM.  `-of tp-ffn-even` prints exactly one
#      `tensor split: FFN granularity 128 -> 64 so <tensor> splits 320 / device (opt-in: -of tp-ffn-even)` WARN,
#      and the greedy text is the recorded OPT-IN reference -- i.e. the tag changes the split, and the text it
#      produces is the one the record above measured.  The arm also asserts the two texts DIFFER, so a tag that
#      silently did nothing cannot pass.
#   C. ONE MECHANISM, FOUR SPELLINGS.  `-of tp-ffn-even`, `LLAMA_TP_FFN_EVEN=1`,
#      `LLAMA_ARG_OPTIONAL_FEATURES=tp-ffn-even` and `-of "tp-ffn-even,"` (a comma list with an empty entry) all
#      produce the OPT-IN reference.  A user asking for the feature must get it under every spelling.
#   D. REFUSAL AND DISCOVERY.  `-of nope` exits non-zero, names `nope` and lists `tp-ffn-even`; `-of list` exits
#      0 and lists it.  An unknown tag must never be a silent no-op.
#   E. THE TAG IS SCOPED.  On 35B-A3B Q8_0 at THREE devices (512-wide experts: no equal split at 64 exists) the
#      default and `-of tp-ffn-even` arms print identical text and neither prints the granularity line.  This is
#      the arm that keeps the feature from reaching the shapes the delivery validates at 3 GPUs.
#
# `-lv 2` is passed ONLY on the census arms: every level that prints statistics interleaves log lines into the
# generated text, so a hash arm runs with the CLI's own default (errors only).  The census arms generate one
# token -- the granularity line is printed at model load.
#
# NOT IN SCOPE: whether the opt-in is *faster* (throughput is not a correctness signal, and the record above
# names what it measured).  `--perf` re-measures it on this box and prints both arms; nothing fails on it.
#
# Usage:
#   ./gate-ffn-even-split.sh [--perf] [--no-faults]
#
# Options / environment:
#   BUILD_DIR     the canonical tree (default ~/llama.cpp/build-rocm)
#   LLAMA_CLI     binary under test (default $BUILD_DIR/bin/llama-cli)
#   MODEL         the 2-device arm's gguf (default the Flash-Next GSQ-RCO IQ3_S shard 1)
#   INERT_MODEL   the 3-device (inert) arm's gguf (default Qwen3.6-35B-A3B Q8_0)
#   PROMPT        prompt file (default prompts/prose-rdna-boosts.txt)
#   GPU           HIP_VISIBLE_DEVICES for the 2-device arm (default 0,2)
#   INERT_GPU     HIP_VISIBLE_DEVICES for the 3-device arm (default 0,1,2)
#   NGEN          generated tokens for a text arm (default 128)
#   EXTRA_ARGS    the 2-device arm's model args (default -ngl 99 -sm tensor -ncmoe 47 -lme 47 -ub 4096
#                 --load-mode dio; a host-expert arm needs all of them)
#   INERT_EXTRA   the 3-device arm's model args (default -ngl 99 -sm tensor -ncmoe 40)
#   EXPECT_DEFAULT / EXPECT_TAG  the two reference hashes.  They are a PER-DEVICE-COUNT reference
#                 (GREEDY-PURITY.md §3): recorded at r338 on 2 devices with the prose prompt, -n 128, seed 42,
#                 temp 0, EXTRA_ARGS as above.  A deliberate change to the 2-device Flash-Next text owes a
#                 re-record here.
#   TIME_LIMIT    per-run seconds (default 1800)
#   KEEPOUT       log dir (default a mktemp -d)
#
# Exit: 0 = PASS, 1 = FAIL, 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}"
INERT_MODEL="${INERT_MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
GPU="${GPU:-0,2}"
INERT_GPU="${INERT_GPU:-0,1,2}"
NGEN="${NGEN:-128}"
EXTRA_ARGS="${EXTRA_ARGS:--ngl 99 -sm tensor -ncmoe 47 -lme 47 -ub 4096 --load-mode dio}"
INERT_EXTRA="${INERT_EXTRA:--ngl 99 -sm tensor -ncmoe 40}"
EXPECT_DEFAULT="${EXPECT_DEFAULT:-c420dfe04c42}"
EXPECT_TAG="${EXPECT_TAG:-cbdb63954bbb}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"

# the optional-features layer is a `common/` facility; keep the caller's own environment out of the arms
unset LLAMA_TP_FFN_EVEN LLAMA_ARG_OPTIONAL_FEATURES || true

PERF=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --perf)      PERF=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help)   sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--perf] [--no-faults]" >&2; exit 2 ;;
    esac
done

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }
ok()   { echo "    ok: $*"; }

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
mkdir -p "$KEEPOUT"

echo "==> -of tp-ffn-even gate (r338, block 09)"
# `llama-cli --version` prints on stderr, so a gate that pins the binary (INVARIANTS r329) must capture both.
echo "    binary=$LLAMA_CLI ($("$LLAMA_CLI" --version 2>&1 | tr '\n' ' ' | cut -c1-90))"
echo "    logs=$KEEPOUT"

[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "FAIL: model not found: $MODEL (set MODEL)" >&2; exit 2; }
[ -f "$PROMPT" ]    || { echo "FAIL: prompt not found: $PROMPT (set PROMPT)" >&2; exit 2; }

# One llama-cli run.
#   cli <name> <gpu> <model> <extra> <ngen> <lv|-> [argv...]
# `[argv...]` are appended to the command (e.g. `-of tp-ffn-even`); the three environment spellings are driven
# by `envi` below instead.
cli() {
    local name="$1" gpu="$2" model="$3" extra="$4" ngen="$5" lv="$6"; shift 6
    local lv_args=()
    [ "$lv" != "-" ] && lv_args=(-lv "$lv")
    # shellcheck disable=SC2086  # $extra is an intended word-split argument list
    HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" \
        -m "$model" $extra -f "$PROMPT" -n "$ngen" --seed 42 --temp 0 --no-display-prompt --single-turn \
        "${lv_args[@]}" "$@" > "$KEEPOUT/$name.log" 2>&1
    local rc=$?
    [ "$rc" -eq 0 ] || fail "arm '$name' exited $rc (see $KEEPOUT/$name.log)"
    return 0
}

# the same run with an environment assignment in front (the env-var spellings of the switch)
#   envi <name> <gpu> <model> <extra> <ngen> <lv|-> <K=V> [argv...]
envi() {
    local name="$1" gpu="$2" model="$3" extra="$4" ngen="$5" lv="$6" kv="$7"; shift 7
    local lv_args=()
    [ "$lv" != "-" ] && lv_args=(-lv "$lv")
    # shellcheck disable=SC2086
    env "$kv" HIP_VISIBLE_DEVICES="$gpu" timeout "$TIME_LIMIT" "$LLAMA_CLI" \
        -m "$model" $extra -f "$PROMPT" -n "$ngen" --seed 42 --temp 0 --no-display-prompt --single-turn \
        "${lv_args[@]}" "$@" > "$KEEPOUT/$name.log" 2>&1
    local rc=$?
    [ "$rc" -eq 0 ] || fail "arm '$name' exited $rc (see $KEEPOUT/$name.log)"
    return 0
}

# `<sha>` of a text arm, via the delivery's own extractor (a naive sed slice does not reproduce the hashes)
extract() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\2/p' | head -1
}

# how many `FFN granularity` lines a run printed (0, 1, ...)
gran_lines() {
    grep -c 'tensor split: FFN granularity' "$KEEPOUT/$1.log" 2>/dev/null || true
}

echo
echo "==> A. the delivery DEFAULT is inert"
cli a_default_census "$GPU" "$MODEL" "$EXTRA_ARGS" 1 2
if [ "$(gran_lines a_default_census)" != "0" ]; then
    fail "the default arm printed a 'FFN granularity' line: the opt-in is on by default"
else
    ok "no granularity line without the tag"
fi
cli a_default_text "$GPU" "$MODEL" "$EXTRA_ARGS" "$NGEN" -
h_default="$(extract a_default_text)"
if [ "$h_default" != "$EXPECT_DEFAULT" ]; then
    fail "default text '$h_default' != recorded default '$EXPECT_DEFAULT' (the default path moved)"
else
    ok "default text $h_default == recorded reference"
fi

echo
echo "==> B. the tag reaches the mechanism"
cli b_tag_census "$GPU" "$MODEL" "$EXTRA_ARGS" 1 2 -of tp-ffn-even
if [ "$(gran_lines b_tag_census)" -lt 1 ]; then
    fail "the tag printed no 'FFN granularity' line: the split did not change"
elif ! grep -q 'FFN granularity 128 -> 64 .* splits 320 / device .*opt-in: -of tp-ffn-even' \
        "$KEEPOUT/b_tag_census.log"; then
    fail "the tag's census line is not '128 -> 64 ... 320 / device (opt-in: -of tp-ffn-even)'"
else
    ok "128 -> 64, 320 / device, one line"
fi
cli b_tag_text "$GPU" "$MODEL" "$EXTRA_ARGS" "$NGEN" - -of tp-ffn-even
h_tag="$(extract b_tag_text)"
if [ "$h_tag" != "$EXPECT_TAG" ]; then
    fail "opt-in text '$h_tag' != recorded opt-in reference '$EXPECT_TAG'"
else
    ok "opt-in text $h_tag == recorded reference"
fi
if [ -n "$h_tag" ] && [ "$h_tag" = "$h_default" ]; then
    fail "the opt-in produced the DEFAULT text: the tag changed nothing (this gate would be vacuous)"
else
    ok "the opt-in text differs from the default ($h_default -> $h_tag)"
fi

echo
echo "==> C. one mechanism, four spellings (each must equal the opt-in reference)"
envi c_env    "$GPU" "$MODEL" "$EXTRA_ARGS" "$NGEN" - LLAMA_TP_FFN_EVEN=1
envi c_argenv "$GPU" "$MODEL" "$EXTRA_ARGS" "$NGEN" - LLAMA_ARG_OPTIONAL_FEATURES=tp-ffn-even
cli  c_csv    "$GPU" "$MODEL" "$EXTRA_ARGS" "$NGEN" - -of "tp-ffn-even,"
for arm in c_env c_argenv c_csv; do
    h="$(extract "$arm")"
    if [ -z "$h" ] || [ "$h" != "$h_tag" ]; then
        fail "arm '$arm' text '$h' != the tag's '$h_tag' (the spellings are not one mechanism)"
    else
        ok "$arm == $h_tag"
    fi
done

echo
echo "==> D. refusal and discovery"
if HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -of nope \
        > "$KEEPOUT/d_bad.log" 2>&1; then
    fail "'-of nope' was accepted (an unknown tag must be refused)"
elif ! grep -q "unknown --optional-features tag 'nope'" "$KEEPOUT/d_bad.log"; then
    fail "'-of nope' was refused without naming the tag"
elif ! grep -q 'tp-ffn-even' "$KEEPOUT/d_bad.log"; then
    fail "'-of nope' was refused without listing the known tags"
else
    ok "'-of nope' refused, named, and the tags listed"
fi
if ! timeout "$TIME_LIMIT" "$LLAMA_CLI" -of list > "$KEEPOUT/d_list.log" 2>&1; then
    fail "'-of list' exited non-zero"
elif ! grep -q 'tp-ffn-even' "$KEEPOUT/d_list.log"; then
    fail "'-of list' does not list tp-ffn-even"
else
    ok "'-of list' lists tp-ffn-even"
fi

echo
echo "==> E. the tag is SCOPED (35B-A3B Q8_0, 3 devices: no even split at 64 exists)"
if [ ! -f "$INERT_MODEL" ]; then
    fail "inert model not found: $INERT_MODEL (set INERT_MODEL)"
else
    cli e_default_census "$INERT_GPU" "$INERT_MODEL" "$INERT_EXTRA" 1 2
    cli e_tag_census     "$INERT_GPU" "$INERT_MODEL" "$INERT_EXTRA" 1 2 -of tp-ffn-even
    if [ "$(gran_lines e_default_census)" != "0" ] || [ "$(gran_lines e_tag_census)" != "0" ]; then
        fail "a granularity line appeared on the 3-device 512-wide arm: the rule reached a shape it must not"
    else
        ok "no granularity line in either arm"
    fi
    cli e_default_text "$INERT_GPU" "$INERT_MODEL" "$INERT_EXTRA" "$NGEN" -
    cli e_tag_text     "$INERT_GPU" "$INERT_MODEL" "$INERT_EXTRA" "$NGEN" - -of tp-ffn-even
    h_def="$(extract e_default_text)"
    h_oft="$(extract e_tag_text)"
    if [ -z "$h_def" ] || [ "$h_def" != "$h_oft" ]; then
        fail "3-device text differs with the tag ('$h_def' vs '$h_oft'): the tag is not scoped"
    else
        ok "3-device text identical with and without the tag ($h_def)"
    fi
fi

if [ "$PERF" -eq 1 ]; then
    echo
    echo "==> F. (measurement, no pass/fail) draft-mtp decode on the 2-device arm"
    MTP="${MTP:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf}"
    if [ ! -f "$MTP" ]; then
        note "no spec model at $MTP: skipped"
    else
        cli perf_default "$GPU" "$MODEL" "$EXTRA_ARGS --spec-draft-model $MTP --spec-type draft-mtp --spec-draft-n-max 3" 256 -
        cli perf_tag     "$GPU" "$MODEL" "$EXTRA_ARGS --spec-draft-model $MTP --spec-type draft-mtp --spec-draft-n-max 3" 256 - -of tp-ffn-even
        note "default: $(grep -oE 'Generation: *[0-9.]+ t/s' "$KEEPOUT/perf_default.log" | tail -1)"
        note "opt-in : $(grep -oE 'Generation: *[0-9.]+ t/s' "$KEEPOUT/perf_tag.log" | tail -1)"
    fi
fi

if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (kernel journal since '$GATE_SINCE_TS', and every arm's log)"
    n_fault=0
    "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$GATE_SINCE_TS" --quiet || n_fault=$?
    for lg in "$KEEPOUT"/*.log; do
        "$REPO_DIR/scripts/check-gpu-faults.sh" --log "$lg" --quiet || n_fault=$((n_fault + 1))
    done
    if [ "$n_fault" -ne 0 ]; then
        fail "a GPU fault was reported (see above)"
    else
        ok "no GPU faults in the journal window or in any arm's log"
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the default is inert, the tag reaches the split (128 -> 64, 320/device), every spelling agrees,"
    echo "      an unknown tag is refused, and the 3-device arm is untouched."
    exit 0
fi
echo "FAIL: $fails check(s) failed -- logs in $KEEPOUT"
exit 1
