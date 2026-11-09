#!/usr/bin/env bash
# The fusion-window leaf-copy aliasing gate (issue #287, r239).
#
# WHY THIS EXISTS
#   `ggml_cuda_check_fusion_memory_ranges()` decides whether a fusion may RUN by comparing the byte
#   ranges of its output nodes against every src of the nodes in its window, and it used to skip every
#   `src->op == GGML_OP_NONE`, i.e. every "leaf", on the assumption that a leaf is an immutable weight.
#   That assumption breaks on a HOST-RESIDENT expert table: with `--ncmoe`/`--cpu-moe` the scheduler
#   copies the routed expert tables into a COMPUTE buffer every ubatch (`ROCm0#blk.N.ffn_gate_exps.weight#0`),
#   and a scheduler copy is a leaf too.  The allocator's last-use rule is computed on the UNFUSED graph, so
#   it frees that copy after its last unfused consumer (the gate `MUL_MAT_ID`) and hands the block to a
#   LATER node, e.g. the GLU output -- which is exactly what the 3-op `MUL_MAT_ID(gate) MUL_MAT_ID(up) GLU`
#   fusion writes, from inside the same launch that still reads the gate copy.  The result is garbage
#   logits: the reporter's arm prints a single repeated character for 400 tokens.
#
#   The gate + up + GLU MMVQ fusion is only eligible for a routed MoE when the ubatch is inside the
#   extended MMVQ MoE band (9..16 columns on RDNA4, block 07) AND the batch is offloaded
#   (`GGML_OP_OFFLOAD_MIN_BATCH`), so the arm below reproduces the report exactly: `-ub 14`,
#   `GGML_OP_OFFLOAD_MIN_BATCH=9`, `MOE_EXPERT_CACHE_MIB=0` (the expert cache disabled, which is what
#   makes the GPU read the scheduler's copy instead of the arena).
#
# WHAT IT CHECKS
#   A. THE REPRO ARM.  Qwen3.6-35B-A3B Q4_K_M, 1 GPU, `--n-cpu-moe 99 -fa on -ub 14`, expert cache off,
#      op-offload min batch 9.  The gate fails unless:
#        - the text is byte-identical to the same arm with `GGML_CUDA_DISABLE_FUSION=1` (the unfused
#          chain is the oracle: at this shape the fused gate+up+GLU kernel is bit-identical to it);
#        - the text is not a degenerate single-character run (the pre-fix signature: `3333...`/`!!!!`.
#          NEGATIVE-TESTED (r239): the pre-fix build yields 58 chars `793107d73635` against 193
#          `3fab4da87774` for every other arm);
#        - with `GGML_CUDA_FUSE_TRACE=1` the run logged >= 1 `FUSE_TRACE ALIAS .../copy@` line, i.e. the
#          arm really reached a compute-buffer leaf overlap and the check is what declined the fusion.
#          Without that line the arm would be vacuous (nothing to catch).
#      The reporter's other control (`GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1`, which makes the 14-token batch
#      ineligible for the fused MMVQ) must produce the same text.
#   B. THE FAILED-FILL ARM.  The same arm with `MOE_HOST_POOL_FILL_FAULT=200`: a declined/failed host fill
#      is served from the scheduler's copy, so it must reproduce the same text.  This is the second way
#      the reporter reached the bug.
#   C. THE PRODUCTION ARM.  The expert cache ON (default) at the same ubatch: the check now declines the
#      fusion on the delivery's own config too (238 declines on a 40-token 1-GPU run), and it must NOT
#      change the output: the text must equal the recorded anchor `3fab4da87774`.  A moved anchor here
#      means a delivery output change, not a gate failure to "fix" by re-recording.
#
#   A GPU-fault scan over this gate's own window is the last step (`scripts/check-gpu-faults.sh --since`),
#   so a corruption that turns into a real fault fails the gate even if the text happens to survive.
#
# Usage:
#   ./gate-fusion-alias-copy.sh
#   ./gate-fusion-alias-copy.sh --no-faults
# Environment:
#   BUILD_DIR    llama.cpp build tree            (default: ~/llama.cpp/build-rocm)
#   LLAMA_CLI    llama-cli                       (default: $BUILD_DIR/bin/llama-cli)
#   MODEL        a host-resident-expert MoE GGUF (default: Qwen3.6-35B-A3B Q4_K_M)
#   NCMOE        --n-cpu-moe value               (default: 99, i.e. every expert on the host)
#   UB           ubatch size                     (default: 14, inside the extended MMVQ MoE band)
#   NGEN         generated tokens                (default: 40)
#   CTX          context size                    (default: 4096)
#   PROMPT       the prompt string               (default: the issue's)
#   GPU          HIP_VISIBLE_DEVICES to pin      (default: 0)
#   ANCHOR       section C's expected sha        (default: 3fab4da87774)
#   KEEPOUT      dir for the logs                (default: mktemp -d)
#   TIME_LIMIT   per llama-cli timeout           (default: 1200 s)
#   A_ARM/B_ARM/C_ARM=0 skips that section.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/LMStudio-Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
NCMOE="${NCMOE:-99}"
UB="${UB:-14}"
NGEN="${NGEN:-40}"
CTX="${CTX:-4096}"
PROMPT="${PROMPT:-Write a detailed technical explanation of how a hash table handles collisions, with examples in C.}"
GPU="${GPU:-0}"
ANCHOR="${ANCHOR:-3fab4da87774}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1200}"
A_ARM="${A_ARM:-1}"
B_ARM="${B_ARM:-1}"
C_ARM="${C_ARM:-1}"

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }

GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

FAULTS=1
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--no-faults]" >&2; exit 2 ;;
    esac
done

echo "==> fusion-window leaf-copy aliasing gate (issue #287)"
echo "    build=$BUILD_DIR"
echo "    model=$MODEL"
echo "    logs=$KEEPOUT"
mkdir -p "$KEEPOUT"

[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "FAIL: model not found: $MODEL (set MODEL)" >&2; exit 2; }

# run_arm <name> <extra llama-cli args> [env assignments...] -- one armed llama-cli run.  A TEXT arm
# passes no -lv at all (the generated text is the signal, and `-lv 3` both corrupts it and hides the
# ggml lines); a CENSUS arm passes `-lv 2`, because llama-cli starts at errors-only and the gate's
# decision line is a WARN (AGENTS.md, the -lv table).
run_arm() {
    local name="$1" cli_args="$2"; shift 2
    env "$@" HIP_VISIBLE_DEVICES="$GPU" \
        timeout "$TIME_LIMIT" "$LLAMA_CLI" -m "$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -fa on \
        -c "$CTX" -ub "$UB" -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
        --no-display-prompt --single-turn $cli_args > "$KEEPOUT/$name.log" 2>&1
    local rc=$?
    local got
    got="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$name.log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1)"
    echo "    $name: exit $rc, ${got:-no text extracted}"
}

# `<chars> <sha>` of an arm's generated text (via the delivery's own extractor: a naive sed slice does
# not reproduce the hashes).
extract() {
    python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log" 2>/dev/null |
        sed -n 's/^\([0-9]*\) chars sha=\([0-9a-f]*\)$/\1 \2/p' | head -1
}

# A degenerate output (the corruption's signature): fewer than 4 distinct non-space characters.
degenerate() { # <name>
    local text
    text="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/$1.log" --text 2>/dev/null)"
    local distinct
    distinct="$(printf '%s' "$text" | tr -d ' \t\n' | fold -w1 | sort -u | wc -l)"
    [ "$distinct" -lt 4 ]
}

# text_arm <name> <oracle-name> <env...> -- run it, then require a non-degenerate text that is
# byte-identical to <oracle-name>'s.
text_arm() {
    local name="$1" oracle="$2"; shift 2
    run_arm "$name" "" "$@"
    local n_chars n_sha o_chars o_sha
    read -r n_chars n_sha < <(extract "$name")
    read -r o_chars o_sha < <(extract "$oracle")
    if [ -z "${n_sha:-}" ]; then
        fail "$name produced no text (log: $KEEPOUT/$name.log)"
        return 1
    fi
    if degenerate "$name"; then
        fail "$name produced a degenerate text (${n_chars} chars sha=${n_sha}): the fusion overwrote the host-expert copy it was still reading (issue #287)"
        return 1
    fi
    if [ "${n_sha:-}" != "${o_sha:-}" ] || [ "${n_chars:-0}" != "${o_chars:-0}" ]; then
        fail "$name differs from $oracle (${n_chars} chars ${n_sha} vs ${o_chars} chars ${o_sha})"
        return 1
    fi
    note "$name == $oracle (${n_chars} chars ${n_sha})"
}

if [ "$A_ARM" -eq 1 ]; then
    echo
    echo "==> A. the repro arm: expert cache off, op-offload min batch 9, a 14-token GPU ubatch"
    run_arm a_chain "" MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_DISABLE_FUSION=1
    read -r a_chars a_sha < <(extract a_chain)
    if [ -z "${a_sha:-}" ]; then
        fail "the unfused oracle produced no text (log: $KEEPOUT/a_chain.log)"
    else
        text_arm a_fused        a_chain MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9
        text_arm a_no_moe_band  a_chain MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1

        # The mechanism: the run must have DECLINED a fusion over a compute-buffer leaf copy.  Without
        # this line the arm is vacuous (the fused window never saw the copy, so nothing could be caught).
        run_arm a_trace "-lv 2" MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_FUSE_TRACE=1
        if grep -qE 'FUSE_TRACE ALIAS [A-Z_]+<-[A-Z_]+/copy@' "$KEEPOUT/a_trace.log"; then
            note "trace: $(sed -n 's/.*FUSE_TRACE ALIAS \(.*\)/\1/p' "$KEEPOUT/a_trace.log" | sort -u | tr '\n' ' ')"
        else
            fail "the repro arm declined no fusion over a compute-buffer leaf copy (no 'FUSE_TRACE ALIAS .../copy@' line): MMVQ_MOE_BAND, GGML_OP_OFFLOAD_MIN_BATCH, UB or the model no longer reach the shape (log: $KEEPOUT/a_trace.log)"
        fi
    fi
fi

if [ "$B_ARM" -eq 1 ] && [ "${a_sha:-}" != "" ]; then
    echo
    echo "==> B. the failed-host-fill arm: a declined fill is served from the scheduler's copy"
    text_arm b_fill_fault a_chain MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9 MOE_HOST_POOL_FILL_FAULT=200
fi

if [ "$C_ARM" -eq 1 ]; then
    echo
    echo "==> C. the production arm: expert cache ON, same ubatch (output must not move)"
    run_arm c_cache_on "" GGML_OP_OFFLOAD_MIN_BATCH=9
    read -r c_chars c_sha < <(extract c_cache_on)
    if [ -z "${c_sha:-}" ]; then
        fail "the cache-on arm produced no text (log: $KEEPOUT/c_cache_on.log)"
    elif degenerate c_cache_on; then
        fail "the cache-on arm produced a degenerate text (${c_chars} chars sha=${c_sha})"
    elif [ "${c_sha:-}" != "$ANCHOR" ]; then
        fail "the cache-on arm moved: ${c_chars} chars sha=${c_sha}, anchor ${ANCHOR} (a delivery output change, not a gate number to update)"
    else
        note "c_cache_on: ${c_chars} chars ${c_sha} == anchor ${ANCHOR}"
    fi
    # ... and the check really does bite on the delivery's own config, i.e. it is not inert there.
    run_arm c_trace "-lv 2" GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_FUSE_TRACE=1
    c_declines="$(grep -c 'FUSE_TRACE ALIAS' "$KEEPOUT/c_trace.log")"
    note "c_trace: ${c_declines:-0} distinct (output <- src) alias classes declined"
    [ "${c_declines:-0}" -ge 1 ] ||
        fail "the cache-on arm declined nothing at all: the leaf-copy check is inert on the production path"
fi

if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    echo
    echo "==> GPU faults (since ${GATE_SINCE_TS:-the journal start})"
    since_args=()
    [ -n "${GATE_SINCE_TS:-}" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log; do [ -f "$f" ] && log_args+=(--log "$f"); done
    "$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" |
        sed 's/^/    /' || fail "a GPU fault was recorded during this gate's arms"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the leaf-copy check declines the overlapping fusion and the text is unchanged"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
