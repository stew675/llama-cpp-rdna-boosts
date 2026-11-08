#!/usr/bin/env bash
# Quant-coverage regression gate for the host-resident-expert device gather.
#
# WHY THIS EXISTS (r30, 2026-10-02):
#   The pruned expert gather copies only the routed experts, so the quantized
#   `MUL_MAT_ID` MMQ's speculative read past a routed expert can land in stale
#   (NaN) bytes and poison the tile -> the model emits a repeated `/` (the
#   "good old ////////" corruption).  The guard is a one-time zero of each
#   expert slot's head; the host-resident upload path (`copy_experts`) guards
#   with `min(expert_size, 512)`, but the gather hard-coded 64 bytes.
#   64 is enough for IQ4_NL only - IQ4_XS (and any quant whose MMQ over-read is
#   wider) still corrupts.  The r29 beta5 gate set only exercised IQ4_NL / Q8_0
#   / Q4_K_M, so the regression shipped.  This gate closes the hole by checking
#   every qwen4exp quant present, not just the one that happened to define the
#   64-byte pad.
#
# WHAT IT CHECKS (per quant, gather ON and OFF):
#   * HARD: neither run may contain a `////` run (the corruption signature) and
#     both must produce an extractable generation.  This is what the r29
#     regression fails.
#   * SOFT: the gather is expected to be bit-transparent (gather ON == OFF
#     hash).  qwen4exp at `--temp 0` is nonetheless run-to-run nondeterministic
#     under `-sm tensor` (a rare near-tie flip, presumably the hybrid
#     all-reduce), so a hash mismatch is reported as a WARN by default; set
#     STRICT=1 (recommended with `-sm layer`) to make it a hard failure.
#
#
# WHAT IT CHECKS TOO (`TODO` #297): the SPARSE QSA decode arm, which had no gate until now.
#   The sparse arm is env-forced (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) and only engages above the
#   indexer selection width (`indexer_top_k + r - 1`, ~2051 tokens of context), so every coherence
#   arm and the harness-wide qwen4exp arms ran the DENSE arm -- and a delivery re-derivation broke
#   the sparse one for five re-base rungs (R12-R16: `indexer-score.cu` called a ROPE-only accessor
#   on its fused node and every sparse run aborted) with nothing to notice.  Section E therefore
#   runs three arms on a prompt above that width and asserts, in this order:
#     * `sparse` (fused node ON, the shipped default) logs the `indexer-score: fused-score launched`
#       census -- reachability, which a text comparison cannot see;
#     * `chain`   (`GGML_CUDA_QSA_INDEXER_SCORE=0`) logs NO census line and produces the SAME text
#       -- the fused node is the op, and it is byte-identical to the per-op chain it replaces;
#     * `dense`   (the shipped default policy) logs NO census line and produces DIFFERENT text --
#       i.e. the sparse arm is a genuinely different path, so "sparse == chain" cannot be satisfied
#       by never taking the sparse arm at all.
#   The census comes from `GGML_CUDA_QSA_SCORE_LOG=1` (a stderr line, so it can never perturb the
#   generated text); its sensitivity is proved by the other two arms on every run.
#
# Usage:
#   ./gate-qwen4exp-quant-coherence.sh [--no-faults] [llama-cli]
# Environment:
#   LLAMA_CLI    path to llama-cli      (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   MODEL_ROOT   qwen4exp model root    (default: /llm/models/Qwen3.8/Flash-Next)
#   QUANTS       space-separated subset (default: "IQ3_XXS IQ4_NL IQ4_XS Q4_K_M")
#   SM           split mode             (default: tensor)
#   NCMOE        n-cpu-moe              (default: 48)
#   CTX          context size           (default: 8192)
#   NGEN         tokens                 (default: 32)
#   NGPU         GPUs                   (default: 2 -> HIP_VISIBLE_DEVICES=0,1)
#   STRICT       1 = a hash mismatch is a hard failure (default: 0 = warn)
#   KEEPOUT      dir for the per-run logs (default: mktemp -d)
#   --no-faults  skip the kernel-journal fault check (a box with no readable kernel log)
#   QSA_ARM      1 = run the sparse-arm section E (default 1)
#   QSA_QUANT    the quant directory section E uses (default GSQ-IQ3_XXS, 1 GPU, -sm layer)
#   QSA_MODEL    override the model path outright
#   QSA_PROMPT   the section's prompt (default prompts/prose-rdna-boosts.txt: ~5246 tokens, i.e.
#                above the ~2051-token selection width the sparse arm needs)
#   QSA_NGEN     generated tokens for the three arms (default 32)
#   QSA_CTX      context size (default 8192)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL_ROOT="${MODEL_ROOT:-/llm/models/Qwen3.8/Flash-Next}"
QUANTS="${QUANTS:-IQ3_XXS IQ4_NL IQ4_XS Q4_K_M}"
SM="${SM:-tensor}"
NCMOE="${NCMOE:-48}"
CTX="${CTX:-8192}"
NGEN="${NGEN:-32}"
NGPU="${NGPU:-2}"
STRICT="${STRICT:-0}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
QSA_ARM="${QSA_ARM:-1}"
QSA_QUANT="${QSA_QUANT:-GSQ-IQ3_XXS}"
QSA_MODEL="${QSA_MODEL:-}"
QSA_PROMPT="${QSA_PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
QSA_NGEN="${QSA_NGEN:-32}"
QSA_CTX="${QSA_CTX:-8192}"
QSA_GPU="${QSA_GPU:-0}"
QSA_NCMOE="${QSA_NCMOE:-48}"
mkdir -p "$KEEPOUT"

# the positional llama-cli the usage line has always documented was silently ignored; parse it (#294)
FAULTS=1
for a in "$@"; do
    case "$a" in
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        -*) echo "usage: $0 [--no-faults] [llama-cli]" >&2; exit 2 ;;
        *) LLAMA_CLI="$a" ;;
    esac
done
# The kernel journal is shared between runs, so the fault window starts here, before the first arm (#294).
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

case "$NGPU" in
  1) DEVS=0 ;;
  2) DEVS=0,1 ;;
  3) DEVS=0,1,2 ;;
  *) DEVS="$NGPU" ;;
esac

[ -x "$LLAMA_CLI" ] || { echo "FAIL: llama-cli not executable: $LLAMA_CLI" >&2; exit 2; }
[ -d "$MODEL_ROOT" ] || { echo "FAIL: model root not found: $MODEL_ROOT" >&2; exit 2; }
command -v python3 >/dev/null || { echo "FAIL: python3 required" >&2; exit 2; }
[ -f "$REPO_DIR/scripts/extract-generated.py" ] || { echo "FAIL: extractor not found" >&2; exit 2; }

PROMPT='The capital of France is'
PROMPT_SHA="$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"

extract_hash() { python3 "$REPO_DIR/scripts/extract-generated.py" "$1" 2>/dev/null | sed -n 's/.*sha=\([0-9a-f]*\).*/\1/p' | head -1; }

echo "==> qwen4exp quant-coherence gate"
echo "    cli=$LLAMA_CLI"
echo "    sm=$SM ncmoe=$NCMOE ctx=$CTX ngen=$NGEN gpus=$DEVS strict=$STRICT"
echo "    prompt sha256=$PROMPT_SHA  (extractor: scripts/extract-generated.py)"
echo "    logs: $KEEPOUT"

fails=0
warns=0
for q in $QUANTS; do
    # the quant directory also holds draft assets that are NOT the model under test: mmproj,
    # the mtp-* draft head and the DFlash2 drafter (which is itself a dflash-arch model and needs
    # ctx_other, so loading it standalone fails).  Pick the quant's own shard 1.
    model="$(ls "$MODEL_ROOT/$q"/*.gguf 2>/dev/null | grep -v -e mmproj -e /mtp- -e DFlash -e dflash | head -1 || true)"
    if [ -z "$model" ]; then
        echo "SKIP  $q: no model under $MODEL_ROOT/$q"
        continue
    fi
    declare -A h sl rc
    for G in 1 0; do
        out="$KEEPOUT/${q}_devgather$G.out"; err="$KEEPOUT/${q}_devgather$G.err"
        timeout 1800 env HIP_VISIBLE_DEVICES="$DEVS" \
            MOE_EXPERT_CACHE_MIB="${MOE_EXPERT_CACHE_MIB:-}" MOE_EXPERT_CACHE_DEVMAP=1 \
            GGML_SCHED_DEVGATHER="$G" \
            "$LLAMA_CLI" -m "$model" -ngl 99 -sm "$SM" -ncmoe "$NCMOE" -fa 1 \
            -c "$CTX" -t 8 -lzm off -lm none \
            -p "$PROMPT" -n "$NGEN" --seed 42 --temp 0 \
            --no-display-prompt --single-turn >"$out" 2>"$err"
        rc[$G]=$?
        sl[$G]="$(grep -o '////' "$out" 2>/dev/null | wc -l)"
        h[$G]="$(extract_hash "$out")"
        echo "      $q gather=${G} rc=${rc[$G]} slashruns=${sl[$G]} hash=${h[$G]:-<none>}"
    done

    q_fail=0
    for G in 1 0; do
        if [ "${rc[$G]}" -ne 0 ]; then
            echo "FAIL  $q: gather=${G} run exited ${rc[$G]} (see $KEEPOUT)"; q_fail=1
        fi
        if [ "${sl[$G]}" -ne 0 ]; then
            echo "FAIL  $q: gather=${G} emitted ${sl[$G]} '////' run(s) - corrupted tile"; q_fail=1
        fi
        if [ -z "${h[$G]}" ]; then
            echo "FAIL  $q: gather=${G} produced no extractable generation (see $KEEPOUT)"; q_fail=1
        fi
    done

    if [ "$q_fail" -ne 0 ]; then
        fails=$((fails + 1))
        continue
    fi

    if [ "${h[1]}" = "${h[0]}" ]; then
        echo "PASS  $q: gather transparent (on == off == ${h[1]})"
    elif [ "$STRICT" = "1" ]; then
        echo "FAIL  $q: gather is NOT transparent: on=${h[1]} off=${h[0]} (STRICT=1)"
        fails=$((fails + 1))
    else
        echo "WARN  $q: gather on=$h[1] != off=${h[0]} - qwen4exp is run-to-run nondeterministic at"
        echo "      temp 0 here; both outputs are coherent (0 '////').  Re-run or use STRICT=1 for a hard gate."
        warns=$((warns + 1))
    fi
done

# ---------------------------------------------------------------------------------------------------
# E. the SPARSE QSA decode arm (TODO #297) -- reachability + fused/chain purity
#
# The three arms are deliberately 1 GPU `-sm layer`: qwen4exp at temp 0 is run-to-run
# nondeterministic under `-sm tensor` (see WHAT IT CHECKS), and this section asserts byte-identity.
# ---------------------------------------------------------------------------------------------------
if [ "$QSA_ARM" = "1" ]; then
    echo
    echo "==> E. the sparse QSA decode arm (LLAMA_QSA_DENSE_DECODE_UNTIL=0)"
    qsa_model="${QSA_MODEL:-}"
    if [ -z "$qsa_model" ]; then
        qsa_model="$(ls "$MODEL_ROOT/$QSA_QUANT"/*.gguf 2>/dev/null |
            grep -v -e mmproj -e /mtp- -e DFlash -e dflash | head -1 || true)"
    fi
    if [ -z "$qsa_model" ] || [ ! -f "$qsa_model" ]; then
        echo "SKIP  E: no model for QSA_QUANT=$QSA_QUANT under $MODEL_ROOT (set QSA_MODEL=)"
    elif [ ! -f "$QSA_PROMPT" ]; then
        echo "SKIP  E: no prompt at $QSA_PROMPT (set QSA_PROMPT=)"
    else
        qsa_run() { # <arm> <env assignment...>
            local arm="$1"; shift
            env "$@" GGML_CUDA_QSA_SCORE_LOG=1 HIP_VISIBLE_DEVICES="$QSA_GPU" \
                timeout 1800 "$LLAMA_CLI" -m "$qsa_model" -ngl 99 -ncmoe "$QSA_NCMOE" -sm layer \
                -fa on -c "$QSA_CTX" -n "$QSA_NGEN" -f "$QSA_PROMPT" -t 8 \
                --seed 42 --temp 0 --no-display-prompt --single-turn \
                > "$KEEPOUT/qsa_$arm.out" 2> "$KEEPOUT/qsa_$arm.err"
            local rc=$?
            local cen sl
            cen="$(grep -c 'indexer-score: fused-score launched' "$KEEPOUT/qsa_$arm.err" 2>/dev/null || true)"
            sl="$(grep -o '////' "$KEEPOUT/qsa_$arm.out" 2>/dev/null | wc -l)"
            echo "      $arm rc=$rc fused-census=${cen:-0} slashruns=$sl hash=$(extract_hash "$KEEPOUT/qsa_$arm.out")"
            return $rc
        }

        qsa_ok=1
        qsa_run sparse LLAMA_QSA_DENSE_DECODE_UNTIL=0                                    || qsa_ok=0
        qsa_run chain  LLAMA_QSA_DENSE_DECODE_UNTIL=0 GGML_CUDA_QSA_INDEXER_SCORE=0      || qsa_ok=0
        qsa_run dense  HIP_VISIBLE_DEVICES="$QSA_GPU"                                   || qsa_ok=0

        if [ "$qsa_ok" -eq 0 ]; then
            echo "FAIL  E: an arm failed to run (see $KEEPOUT/qsa_*.err)"
            fails=$((fails + 1))
        else
            qsa_cen() { grep -c 'indexer-score: fused-score launched' "$KEEPOUT/qsa_$1.err" 2>/dev/null || true; }
            cs="$(qsa_cen sparse)"; cc="$(qsa_cen chain)"; cd_="$(qsa_cen dense)"
            hs="$(extract_hash "$KEEPOUT/qsa_sparse.out")"
            hc="$(extract_hash "$KEEPOUT/qsa_chain.out")"
            hd="$(extract_hash "$KEEPOUT/qsa_dense.out")"
            echo "      census: sparse=$cs chain=$cc dense=$cd_"
            grep -o 'indexer-score: fused-score launched.*' "$KEEPOUT/qsa_sparse.err" | head -2 | sed 's/^/        /'

            # 1. the fused indexer-score op really launched (reachability: an op that never runs
            #    cannot fail a comparison against itself)
            if [ "${cs:-0}" -ge 1 ]; then
                echo "PASS  E: the sparse arm launched the fused indexer-score op ($cs census line(s))"
            else
                echo "FAIL  E: the sparse arm launched NO fused indexer-score op: the arm is unreachable,"
                echo "      so the hash comparison below would be vacuous (GGML_CUDA_QSA_SCORE_LOG=1 was set)"
                fails=$((fails + 1))
            fi
            # 2. ... and the census is sensitive: the per-op chain (and the dense default) must log none
            if [ "${cc:-1}" -eq 0 ] && [ "${cd_:-1}" -eq 0 ]; then
                echo "PASS  E: the census is sensitive - 0 lines with the fused op off and on the dense arm"
            else
                echo "FAIL  E: the census fired without the fused op (chain=$cc dense=$cd_): it does not"
                echo "      measure what the assertions above assume"
                fails=$((fails + 1))
            fi
            # 3. the fused node is byte-identical to the per-op chain it replaces (the purity contract)
            if [ -n "$hs" ] && [ "$hs" = "$hc" ]; then
                echo "PASS  E: fused == chain on the sparse arm ($hs)"
            else
                echo "FAIL  E: the fused indexer score is not pure: fused=${hs:-<none>} chain=${hc:-<none>}"
                fails=$((fails + 1))
            fi
            # 4. non-vacuity: the sparse arm is a different path from the shipped dense default
            if [ -n "$hs" ] && [ "$hs" != "$hd" ]; then
                echo "PASS  E: the sparse arm differs from the dense arm ($hs vs ${hd:-<none>}), so it is not vacuous"
            else
                echo "FAIL  E: the sparse arm produced the dense arm's text (${hs:-<none>}): either the sparse"
                echo "      path no longer changes the numerics or the arm never engaged - investigate before"
                echo "      relaxing this (the prompt must be above the ~2051-token selection width)"
                fails=$((fails + 1))
            fi
            for arm in sparse chain dense; do
                if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$KEEPOUT/qsa_$arm.err"; then
                    echo "FAIL  E: $arm aborted:"
                    grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault|what\(\):" "$KEEPOUT/qsa_$arm.err" | sed 's/^/      /'
                    fails=$((fails + 1))
                fi
            done
        fi
    fi
fi

echo
# the kernel journal, scoped to this gate's own window (#294)
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    log_args=()
    for f in "$KEEPOUT"/*.log "$KEEPOUT"/*.err; do [ -f "$f" ] && log_args+=(--log "$f"); done
    if out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" "${log_args[@]}" 2>&1)"; then
        echo "PASS  GPU faults: kernel journal (since ${GATE_SINCE_TS:-start}) and every arm log clean"
    else
        echo "FAIL  GPU faults (since ${GATE_SINCE_TS:-the journal start}):"
        echo "$out" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

if [ "$fails" -eq 0 ]; then
    echo "==> PASS: every measured qwen4exp quant produced a coherent (0 '////') gather-ON output"
    [ "$QSA_ARM" = "1" ] && echo "    and the sparse QSA decode arm launched the fused indexer score, byte-identical to the chain"
    [ "$QSA_ARM" != "1" ] && echo "    (section E was SKIPPED with QSA_ARM=0 - the sparse arm was NOT checked)"
    [ "$warns" -gt 0 ] && echo "    ($warns quant(s) only WARNed on transparency - see above)"
    echo "    logs kept in $KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed - the per-quant coherence arms and/or the sparse QSA section E (logs in $KEEPOUT)" >&2
exit 1
