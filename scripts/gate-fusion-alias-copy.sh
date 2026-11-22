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
#        - the fused arm's text equals the recorded ANCHOR (the fused launch's own output; with section C
#          this is the exact-bytes check);
#        - the text is not a degenerate single-character run (the pre-fix signature: `3333...`/`!!!!`.
#          NEGATIVE-TESTED (r239): the pre-fix build yields 58 chars `793107d73635` against 193
#          `3fab4da87774` for every other arm);
#        - the LOGITS, not the text, put the fused arm and the `GGML_CUDA_DISABLE_FUSION=1` chain in one
#          class: `llama-perplexity --kl-divergence` at the same shape must read mean KLD <= KLD_MAX and
#          same-top-p >= KLD_MIN_TOPP (the oracle re-derived at r340 -- see the TEXT ORACLE note below);
#        - with `GGML_CUDA_FUSE_TRACE=1` the run logged >= 1 `FUSE_TRACE ALIAS .../copy@` line, i.e. the
#          arm really reached a compute-buffer leaf overlap and the check is what declined the fusion.
#          Without that line the arm would be vacuous (nothing to catch).  **Since r323 that line comes from
#          the `GGML_GLU_INPLACE=0` arm** (see the GLU note below): with the rule ON the fused `GLU` output
#          takes its gate's memory, so it can no longer land on the dead scheduler-copy block and there is
#          nothing to decline.  The default-env arm's decline count is reported, not asserted -- a decline
#          is the protection firing, so requiring zero would punish it.
#
#   TEXT ORACLE RETIRED (r340, `TODO` #359).  Until r340 arm A also required the fused text to be
#   BYTE-IDENTICAL to the unfused chain, on the r239/r323 claim that the fused gate+up+GLU launch is
#   bit-identical to it at this shape.  That claim stopped holding at the r328 re-base, and not because of
#   the fusion: the UNFUSED chain moved (194 chars `dfe59c9c5257`) while the fused arm still reads the r239
#   anchor (193 chars `3fab4da87774`), which is what r329 and r331 both read too.  Measured cause: the
#   chain's per-op `MUL_MAT_ID` is placed and partitioned under block 09's host-offload relaxation
#   (issue #323, `GGML_MOE_HOST_OFFLOAD`, ON since r307 and shipped in the r328 base), which changes the
#   graph's split count at this shape (958 with it, 1443 with `GGML_MOE_HOST_OFFLOAD=0`) and with it the
#   per-op kernel path -- the `GGML_MOE_HOST_OFFLOAD=0` control reproduces the anchor in the chain 3/3.
#   So the divergence is the documented "placement selects the MoE kernel set, and the sets are not
#   bit-identical" class (`GREEDY-PURITY.md` §89, `INVARIANTS.md` r305, the r328 campaign's R09.3), not
#   corruption, and it is ROUNDING level: mean KLD **0.000002** / same-top-p **100.000 %** between the two
#   arms at that shape (256 scored tokens, `-ub 14`, hipBLASLt enabled); 0.000008 with the pins the gate's
#   KLD legs add (`ROCBLAS_USE_HIPBLASLT=0` + `MOE_EXPERT_HEATMAP=load`).  One near-tie at the end of a
#   40-token run was the whole 193-vs-194 difference, i.e. a text hash is the wrong instrument here.  The
#   oracle is therefore the logit compare, and the exact-bytes claim moved to the ANCHOR (the fused launch's
#   own text, asserted in A and in C).  The limits are not vacuous: the SAME-KERNEL reordering control at
#   this shape (the chain recorded at `-ub 14`, compared against a `-ub 8` run, 256 tokens) reads
#   **0.003746 / 93.701 %**, ~470x the measured pair -- so KLD_MAX 0.001 / top-p 99.9 % separates "two MoE
#   kernel sets at one shape" from "a materially different prefill", and a corrupted window would fail it.
#   See `WORKLOG.md` 261121.020.
#
#   GLU IN PLACE (r323, block 00).  `ggml_op_can_inplace(GGML_OP_GLU)` hands an FFN's GLU the memory of its
#   gate, which is exactly the block the allocator used to give the fused window's output here.  The two
#   trace arms therefore pin `GGML_GLU_INPLACE=0` to keep the hazard reachable: the protection is still
#   needed whenever the rule does not apply (a second reader of the gate, the split form, the switch off).
#      The reporter's other control (`GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1`, which makes the 14-token batch
#      ineligible for the fused MMVQ) must produce the same text as the unfused chain (it is the chain).
#   B. THE FAILED-FILL ARM.  The same arm with `MOE_HOST_POOL_FILL_FAULT=200`: a declined/failed host fill
#      is served from the scheduler's copy, so it must reproduce the same text -- compared against the FUSED
#      arm (this arm runs the default, fused path), i.e. against the anchor's text.  This is the second way
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
#   ANCHOR       expected fused-arm sha, A+C (default: 3fab4da87774)
#   KEEPOUT      dir for the logs                (default: mktemp -d)
#   TIME_LIMIT   per llama-cli timeout           (default: 1200 s)
#   --- the arm-A logit oracle (r340) ---
#   LLAMA_PPL    llama-perplexity                (default: $BUILD_DIR/bin/llama-perplexity)
#   KLD_CORPUS   scored text                     (default: prompts/prose-rdna-boosts.txt)
#   KLD_CTX      context / tokens per chunk      (default: 256)
#   KLD_CHUNKS   chunks                          (default: 1)
#   KLD_MAX      mean KLD limit                  (default: 0.001; measured 0.000002)
#   KLD_MIN_TOPP same-top-p floor, percent       (default: 99.9; measured 100.000)
#   KLD_BASE_DIR where the in-run base is written
#                (default: ~/.cache/rdna-boosts/gates/fusion-alias -- never /tmp, which is tmpfs)
#   KLD_THREADS  llama-perplexity -t             (default: 8)
#   KEEP_KLD=1   keep the base file               (default: removed after the compare)
#   A_ARM/B_ARM/C_ARM/KLD_ARM=0 skips that section (KLD_ARM=0 leaves arm A without its oracle).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
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
KLD_ARM="${KLD_ARM:-1}"
# The arm-A logit oracle (r340).
LLAMA_PPL="${LLAMA_PPL:-$BUILD_DIR/bin/llama-perplexity}"
KLD_CORPUS="${KLD_CORPUS:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
KLD_CTX="${KLD_CTX:-256}"
KLD_CHUNKS="${KLD_CHUNKS:-1}"
KLD_MAX="${KLD_MAX:-0.001}"
KLD_MIN_TOPP="${KLD_MIN_TOPP:-99.9}"
KLD_BASE_DIR="${KLD_BASE_DIR:-$HOME/.cache/rdna-boosts/gates/fusion-alias}"
KLD_THREADS="${KLD_THREADS:-8}"
KEEP_KLD="${KEEP_KLD:-0}"
# The arm-A shape, as env for both the text arms and the KLD legs.  The KLD legs additionally pin the
# external solution selection (`ROCBLAS_USE_HIPBLASLT=0`, GREEDY-PURITY.md §44 -- a base recorded in one
# process is compared in another) and the expert-cache residency prior (§89: residency selects the MoE
# kernel set), so a library lottery cannot be read as a kernel difference.
A_ENVS="MOE_EXPERT_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=9"
KLD_ENVS="$A_ENVS ROCBLAS_USE_HIPBLASLT=0 MOE_EXPERT_HEATMAP=load"

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

# anchor_arm <name> <env...> -- run it, then require a non-degenerate text that is byte-identical to
# the recorded ANCHOR (the fused launch's own output).
anchor_arm() {
    local name="$1"; shift
    run_arm "$name" "" "$@"
    local n_chars n_sha
    read -r n_chars n_sha < <(extract "$name")
    if [ -z "${n_sha:-}" ]; then
        fail "$name produced no text (log: $KEEPOUT/$name.log)"
        return 1
    fi
    if degenerate "$name"; then
        fail "$name produced a degenerate text (${n_chars} chars sha=${n_sha}): the fusion overwrote the host-expert copy it was still reading (issue #287)"
        return 1
    fi
    if [ "${n_sha:-}" != "$ANCHOR" ]; then
        fail "$name moved: ${n_chars} chars sha=${n_sha}, anchor ${ANCHOR} (a delivery output change, not a gate number to update)"
        return 1
    fi
    note "$name: ${n_chars} chars ${n_sha} == anchor ${ANCHOR}"
}

# --- the arm-A logit oracle (r340) ----------------------------------------------------------------
# A fresh-process text hash is not a valid oracle for the two arms at this shape (a rounding-level
# kernel-set difference flips one near-tie; TEXT ORACLE RETIRED in the header).  These legs record a KLD
# base from the UNFUSED chain and compare the FUSED arm against it at the gate's own ubatch.
le() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 <= b+0) }'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 >= b+0) }'; }

# kld_arm <name> <env assignments> [extra llama-perplexity args...]
kld_arm() {
    local name="$1" envs="$2"; shift 2
    # shellcheck disable=SC2086  # $envs is an intended word-split K=V list
    env $envs HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$LLAMA_PPL" \
        -m "$MODEL" -f "$KLD_CORPUS" -c "$KLD_CTX" --chunks "$KLD_CHUNKS" \
        -ngl 99 --n-cpu-moe "$NCMOE" -fa on -ub "$UB" -b "$UB" -t "$KLD_THREADS" \
        --seed 42 "$@" > "$KEEPOUT/$name.log" 2>&1
}

kld_self_test() {
    mkdir -p "$KLD_BASE_DIR"
    local base="$KLD_BASE_DIR/chain-$(basename "${MODEL%.gguf}")-ub$UB.kld"
    echo
    echo "==> A.2 the logit oracle: the fused arm against the unfused chain, at -ub $UB"
    echo "    ppl=$LLAMA_PPL  corpus=$KLD_CORPUS  ctx=$KLD_CTX chunks=$KLD_CHUNKS  base=$base"
    echo "    limits: mean KLD <= $KLD_MAX, same-top-p >= ${KLD_MIN_TOPP}%"
    # record the base from the chain (the reference), then compare the fused arm:
    if ! kld_arm a_kld_chain "$KLD_ENVS GGML_CUDA_DISABLE_FUSION=1" --kl-divergence-base "$base"; then
        fail "the chain KLD base run failed (log: $KEEPOUT/a_kld_chain.log)"
        return 1
    fi
    [ -s "$base" ] || { fail "the chain KLD base was not written: $base"; return 1; }
    if ! kld_arm a_kld_fused "$KLD_ENVS" --kl-divergence-base "$base" --kl-divergence; then
        fail "the fused KLD compare run failed (log: $KEEPOUT/a_kld_fused.log)"
        return 1
    fi
    local kld topp
    kld="$(sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([-0-9.][0-9.]*\).*/\1/p' "$KEEPOUT/a_kld_fused.log" | head -1)"
    topp="$(sed -n 's/^Same top p:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' "$KEEPOUT/a_kld_fused.log" | head -1)"
    [ "$KEEP_KLD" -eq 1 ] || rm -f "$base"
    if [ -z "${kld:-}" ] || [ -z "${topp:-}" ]; then
        fail "could not parse the fused-vs-chain KLD (see $KEEPOUT/a_kld_fused.log)"
        return 1
    fi
    note "fused vs chain: mean KLD = $kld   same-top-p = $topp %"
    if ! le "$kld" "$KLD_MAX"; then
        fail "the fused arm diverges from the unfused chain: mean KLD $kld > $KLD_MAX (not rounding level)"
        return 1
    fi
    if ! ge "$topp" "$KLD_MIN_TOPP"; then
        fail "the fused arm's top-1 token moved against the unfused chain: same-top-p $topp % < $KLD_MIN_TOPP %"
        return 1
    fi
    note "the two chains are one class at this shape (a text hash is not: the 193/194 char difference is one near-tie)"
}

if [ "$A_ARM" -eq 1 ]; then
    echo
    echo "==> A. the repro arm: expert cache off, op-offload min batch 9, a 14-token GPU ubatch"
    # The unfused chain: no longer the text oracle (see the header), but still the KLD reference and the
    # control that the band switch must reproduce.  It must at least produce a sane text.
    run_arm a_chain "" $A_ENVS GGML_CUDA_DISABLE_FUSION=1
    read -r a_chars a_sha < <(extract a_chain)
    if [ -z "${a_sha:-}" ]; then
        fail "the unfused chain produced no text (log: $KEEPOUT/a_chain.log)"
    elif degenerate a_chain; then
        fail "the unfused chain produced a degenerate text (${a_chars} chars sha=${a_sha})"
    else
        # The fused arm carries the EXACT-BYTES assertion (the anchor); the chain equivalence is the KLD
        # below.  Reporting both texts keeps the 193-vs-194 near-tie visible in the log.
        anchor_arm a_fused $A_ENVS
        note "a_chain: ${a_chars} chars ${a_sha} (the chain is judged by the KLD below, not by this hash)"
        text_arm a_no_moe_band a_chain $A_ENVS GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1

        # The mechanism: the run must have DECLINED a fusion over a compute-buffer leaf copy.  Without
        # this line the arm is vacuous (the fused window never saw the copy, so nothing could be caught).
        # `GGML_GLU_INPLACE=0` restores the overlap r323 removed: by default the GLU output takes the gate's
        # memory and cannot land on the dead scheduler copy, so the default arm has nothing to decline.
        run_arm a_trace "-lv 2" $A_ENVS GGML_CUDA_FUSE_TRACE=1
        run_arm a_trace_gluoff "-lv 2" $A_ENVS GGML_CUDA_FUSE_TRACE=1 GGML_GLU_INPLACE=0
        note "trace: default env declines $(grep -c 'FUSE_TRACE ALIAS' "$KEEPOUT/a_trace.log") class(es), GGML_GLU_INPLACE=0 declines $(grep -c 'FUSE_TRACE ALIAS' "$KEEPOUT/a_trace_gluoff.log")"
        if grep -qE 'FUSE_TRACE ALIAS [A-Z_]+<-[A-Z_]+/copy@' "$KEEPOUT/a_trace_gluoff.log"; then
            note "trace: $(sed -n 's/.*FUSE_TRACE ALIAS \(.*\)/\1/p' "$KEEPOUT/a_trace_gluoff.log" | sort -u | tr '\n' ' ')"
        else
            fail "the repro arm declined no fusion over a compute-buffer leaf copy (no 'FUSE_TRACE ALIAS .../copy@' line in the GGML_GLU_INPLACE=0 arm): MMVQ_MOE_BAND, GGML_OP_OFFLOAD_MIN_BATCH, UB or the model no longer reach the shape (log: $KEEPOUT/a_trace_gluoff.log)"
        fi

        if [ "$KLD_ARM" -eq 1 ]; then
            [ -x "$LLAMA_PPL" ] || fail "llama-perplexity not executable: $LLAMA_PPL (set LLAMA_PPL, or KLD_ARM=0 to run arm A WITHOUT its oracle)"
            if [ -x "$LLAMA_PPL" ]; then
                kld_self_test
            fi
        else
            note "KLD_ARM=0: arm A ran WITHOUT its logit oracle (a text hash alone is not the oracle since r340)"
        fi
    fi
fi

if [ "$B_ARM" -eq 1 ] && [ "${a_sha:-}" != "" ]; then
    echo
    echo "==> B. the failed-host-fill arm: a declined fill is served from the scheduler's copy"
    # This arm runs the default (FUSED) path, so its oracle is the fused arm's text, i.e. the anchor -- not
    # the unfused chain's (a chain hash was the pre-r340 shape of this check and is a near-tie away).
    anchor_arm b_fill_fault $A_ENVS MOE_HOST_POOL_FILL_FAULT=200
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
    # ... and the check really does bite when the hazard exists: pin `GGML_GLU_INPLACE=0`, because with the
    # rule ON the GLU output takes the gate's memory and never lands on the dead copy (r323, see the header).
    run_arm c_trace "-lv 2" GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_FUSE_TRACE=1
    run_arm c_trace_gluoff "-lv 2" GGML_OP_OFFLOAD_MIN_BATCH=9 GGML_CUDA_FUSE_TRACE=1 GGML_GLU_INPLACE=0
    c_declines="$(grep -c 'FUSE_TRACE ALIAS' "$KEEPOUT/c_trace.log")"
    c_declines_gluoff="$(grep -c 'FUSE_TRACE ALIAS' "$KEEPOUT/c_trace_gluoff.log")"
    note "c_trace: ${c_declines:-0} class(es) declined on the default env, ${c_declines_gluoff:-0} with GGML_GLU_INPLACE=0"
    [ "${c_declines_gluoff:-0}" -ge 1 ] ||
        fail "the GGML_GLU_INPLACE=0 arm declined nothing at all: the leaf-copy check is inert on the production path"
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
    echo "PASS: the leaf-copy check declines the overlapping fusion, the fused output is unmoved, and the fused and unfused chains are one logit class"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
