#!/usr/bin/env bash
# gate-arms-spec -- the SPECULATOR on the two-arm server (STEP 10): one cold greedy TEXT, four arms, plus the
# four-axis MTP protocol on the pair and off.
#
# WHY.  STEP 10 re-admits a drafter onto the pair: the DECODE arm drives the draft/verify/accept round for the
# slots it owns, on `ctx_dec`, which borrows the prefill arm's memory.  Three things can go wrong and each needs
# its own instrument:
#
#   * the drafter reads the WRONG TARGET ROWS -- its target-side reads (`llama_get_embeddings_nextn`) must follow
#     the context that RAN the batch (`common_batch::ctx`).  A drafter reading ctx_tgt on the decode arm sees the
#     prefill arm's last pass, i.e. stale rows: the drafts are then garbage, every one is rejected, and the TEXT
#     is unchanged.  A text claim alone is VACUOUS here, so this gate also asserts that the drafts were ACCEPTED
#     (a zero-acceptance arm is a FAIL, not a pass);
#   * the shared speculator object races between the two arms (DESIGN V4) -- a wrong/garbled text, or a crash;
#   * a slot's speculative state survives a HANDOVER (`spec_draft`/`spec_i_batch`/`spec_ckpt` name a batch the
#     other arm ran) -- a wrong text, or an abort in the accept round.
#
# THE CLAIM: `pair+draft == pair+none == single+draft == single+none`, one fresh process per arm, one request in
# flight, greedy and seed-pinned.  The `pair+none` arm is the control that says the PAIR alone moves no text; the
# `single+draft` arm is the delivery's own drafter behaviour on the same shape.  A shape is therefore four arms,
# and all four must agree.
#
# SHAPES: the campaign's MTP shape (27B Q8_0, embedded MTP head, `draft-mtp-adaptive` at the delivery ceiling 7),
# on the three the STEP 10 oracle names: 3 devices f16 / 3 devices q8_0 / 2 devices f16.
#
# NON-VACUITY, per arm (a gate that could pass on the single-context server is defective):
#   * every `pair` arm must print `the decode arm is generating: N step(s), M handover(s)` with N >= 2 and M >= 1;
#   * every `draft` arm must show `#gen drafts > 0` AND `#acc drafts > 0` in the `-lv 4` statistics line;
#   * the `pair+draft` arm additionally must show the drafts accepted WHILE the decode arm owned the slot -- the
#     census and the acceptance come from the same request, and the gate prints both.
#
# PINS (each is what makes an arm a point, not a run): HIP_VISIBLE_DEVICES per shape, ROCBLAS_USE_HIPBLASLT=0 on
# every arm (benchmarks/prefill-logit-methodology.md: a fresh-process text comparison), MOE_EXPERT_HEATMAP=load
# where a host tier is involved, LLAMA_SERVER_PARK_IDLE_SLOTS=0 (server-arm.sh pins it -- parking moves cells and
# can flip a near-tie), one request in flight, and one GPU job at a time via an exclusive flock for the whole gate.
#
# USAGE: gate-arms-spec.sh [--shapes LIST] [--axes LIST|none] [--n-axis N] [--keep]
#   --shapes LIST   comma list of 27b-3gpu-f16,27b-3gpu-q8,27b-2gpu-f16 (default: all three)
#   --axes LIST     comma list of the four-axis protocol's axes for phase B (default R,K,P,C; `none` skips it)
#   --n-axis N      generated tokens per four-axis arm (default 600 -- the claim is byte identity plus a non-zero
#                   acceptance, so a shorter stream than the protocol's 3000 is enough and keeps it affordable)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
BASE="$V2/baseline"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-spec}"
SHAPES="27b-3gpu-f16,27b-3gpu-q8,27b-2gpu-f16"
AXES="R,K,P,C"
N_AXIS=600
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --shapes)  SHAPES="$2"; shift 2 ;;
        --axes)    AXES="$2"; shift 2 ;;
        --n-axis)  N_AXIS="$2"; shift 2 ;;
        --keep)    KEEP=1; shift ;;
        -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

MODEL=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
# the delivery's adaptive ceiling for this model (block 02: n_min_adaptive is 3, so a lower ceiling is an invalid
# pair and the drafter would not initialise -- the acceptance check below is what makes that a FAIL, not a pass)
NMAX=7

shape_var() {  # $1 = shape, $2 = field (devices | extra)
    case "$1:$2" in
        27b-3gpu-f16:devices) echo 0,1,2 ;;
        27b-3gpu-f16:extra)   echo "-sm tensor -mg 0 -ctk f16 -ctv f16" ;;
        27b-3gpu-q8:devices)  echo 0,1,2 ;;
        27b-3gpu-q8:extra)    echo "-sm tensor -mg 0 -ctk q8_0 -ctv q8_0" ;;
        27b-2gpu-f16:devices) echo 0,2 ;;
        27b-2gpu-f16:extra)   echo "-sm tensor -mg 0 -ctk f16 -ctv f16" ;;
        *) echo "" ;;
    esac
}

lock="${MTP_LOCK:-$HOME/.cache/rdna-boosts/gpu.lock}"
mkdir -p "$(dirname "$lock")"
exec 9>"$lock"
if ! flock -w "${LOCK_WAIT:-1800}" 9; then
    echo "FAIL: another GPU job holds $lock (waited ${LOCK_WAIT:-1800}s)" >&2
    exit 1
fi

fails=0
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }
LOGS=()

# one cold server arm: $1 = tag, $2 = devices, $3 = extra args, $4 = n_predict, $5 = nreq, $6 = prompt file
run_arm() {
    local tag="$1" dev="$2" extra="$3" npred="$4" nreq="$5" prompt="$6"
    HIP_VISIBLE_DEVICES="$dev" timeout 3600 "$BASE/server-arm.sh" --out "$OUT" --tag "$tag" \
        --model "$MODEL" --np 2 --ctx 16384 --n-predict "$npred" --nreq "$nreq" \
        --prompt-file "$prompt" --prompt-file2 "$prompt" \
        --ready 600 --req-timeout 1800 --timeout 3000 \
        --env "ROCBLAS_USE_HIPBLASLT=0" \
        --extra "$extra" > "$OUT/$tag.out" 2>&1
    LOGS+=("$OUT/$tag.log")
}

# --- the four arms of the claim, per shape ---------------------------------------------------------------
# $1 = shape, $2 = "off|on" the pair, $3 = spec type ("none" | "adaptive")
arm_args() {
    local pair="$2" spec="$3"
    local extra
    extra="$(shape_var "$1" extra)"
    if [ "$pair" = on ]; then
        extra="$extra --prefill-decode-arms --prefill-decode-arms-mode on"
    else
        extra="$extra --prefill-decode-arms-mode off"
    fi
    if [ "$spec" = adaptive ]; then
        extra="$extra --spec-type draft-mtp-adaptive --spec-draft-n-max $NMAX"
    else
        extra="$extra --spec-type none"
    fi
    # -lv 4: a CENSUS arm (the acceptance statistics, the decode-arm census).  The response text comes back over
    # HTTP, so -lv 4 cannot pollute it -- unlike a llama-cli hash arm, which takes no -lv at all.
    echo "$extra -lv 4"
}

# the statistics of the LAST request in an arm's log: "#gen drafts" / "#acc drafts" (0 when absent)
stat() {  # $1 = log, $2 = field  (#gen drafts | #acc drafts)
    local line
    line="$(grep -a 'statistics' "$1" 2>/dev/null | tail -1)"
    sed -n "s/.*$2 = *\([0-9]*\).*/\1/p" <<<"$line"
}

echo "==> arms-spec: the speculator on the pair -- shapes [$SHAPES]; axes [$AXES] n_axis=$N_AXIS"
IFS=',' read -r -a want <<<"$SHAPES"
for s in "${want[@]}"; do
    [ -n "$(shape_var "$s" devices)" ] || { echo "FAIL  unknown shape '$s'"; fails=$((fails+1)); continue; }
    echo
    echo "-- $s  (devices $(shape_var "$s" devices) '$(shape_var "$s" extra)')"

    declare -A sha=() chars=() sgen=() sacc=() steps=() hand=()
    for arm in single-none single-mtp pair-none pair-mtp; do
        case "$arm" in
            single-none) pair=off; spec=none ;;
            single-mtp)  pair=off; spec=adaptive ;;
            pair-none)   pair=on;  spec=none ;;
            pair-mtp)    pair=on;  spec=adaptive ;;
        esac
        tag="$s-$arm"
        echo "   [$tag] pair=$pair spec=$spec"
        if ! run_arm "$tag" "$(shape_var "$s" devices)" "$(arm_args "$s" "$pair" "$spec")" \
                48 2 "$REPO/prompts/prose-rdna-boosts.txt"; then
            echo "FAIL  $tag did not complete (see $OUT/$tag.out)"
            tail -3 "$OUT/$tag.out" 2>/dev/null | sed 's/^/      /'
            fails=$((fails+1))
            continue
        fi
        s0="$(sed -n 's/^slot0: [0-9]* chars sha=\([0-9a-f]*\).*/\1/p' "$OUT/$tag.hashes" 2>/dev/null)"
        s1="$(sed -n 's/^slot1: [0-9]* chars sha=\([0-9a-f]*\).*/\1/p' "$OUT/$tag.hashes" 2>/dev/null)"
        sha[$arm]="$s0/$s1"
        sgen[$arm]="$(stat "$OUT/$tag.log" '#gen drafts')"
        sacc[$arm]="$(stat "$OUT/$tag.log" '#acc drafts')"
        census="$(grep -a 'decode arm is generating' "$OUT/$tag.log" | tail -1)"
        steps[$arm]="$(sed -n 's/.*generating: *\([0-9]*\) step(s).*/\1/p' <<<"$census")"
        hand[$arm]="$(grep -oE '[0-9]+ handover' <<<"$census" 2>/dev/null | grep -oE '^[0-9]+' | tail -1)"
        echo "      text: ${sha[$arm]:-<absent>}  drafts=${sgen[$arm]:-0} accepted=${sacc[$arm]:-0}" \
             "decode-arm steps=${steps[$arm]:--} handovers=${hand[$arm]:--}"
    done

    # ---- 1. the claim: all four arms read ONE text -------------------------------------------------------
    ref="${sha[single-none]:-}"
    hard "$([ -n "$ref" ] && echo 1 || echo 0)" \
        "$s: the single-context control arm produced a text (${ref:-<absent>})"
    for arm in single-mtp pair-none pair-mtp; do
        hard "$([ -n "$ref" ] && [ "${sha[$arm]:-x}" = "$ref" ] && echo 1 || echo 0)" \
            "$s: $arm reads the SAME text as single-none ($ref vs ${sha[$arm]:-<absent>})"
    done

    # ---- 2. the non-vacuity arms: the decode arm ran, the drafter drafted and was ACCEPTED ---------------
    for arm in pair-none pair-mtp; do
        hard "$([ "${steps[$arm]:-0}" -ge 2 ] && [ "${hand[$arm]:-0}" -ge 1 ] && echo 1 || echo 0)" \
            "$s: $arm really ran on the decode arm (${steps[$arm]:-0} step(s), ${hand[$arm]:-0} handover(s))"
    done
    for arm in single-mtp pair-mtp; do
        hard "$([ "${sgen[$arm]:-0}" -gt 0 ] && [ "${sacc[$arm]:-0}" -gt 0 ] && echo 1 || echo 0)" \
            "$s: $arm really drafted and really accepted (${sgen[$arm]:-0} drafts, ${sacc[$arm]:-0} accepted)"
    done
    # the specific failure mode that makes the acceptance check vacuous rather than merely empty
    for arm in single-mtp pair-mtp; do
        if grep -qa "failed to initialize speculative decoding context" "$OUT/$s-$arm.log" 2>/dev/null; then
            echo "FAIL  $s: $arm's drafter never INITIALISED --" \
                 "$(grep -a 'failed to initialize speculative' "$OUT/$s-$arm.log" | tail -1 | sed 's/.*: //')"
            fails=$((fails+1))
        fi
    done
    unset sha chars sgen sacc steps hand
done

# ---- 3. phase B: the four-axis protocol, ON the pair and OFF ----------------------------------------------
# The pair is a SERVER feature, so these arms are server arms: one axis prompt, one cold request, `-n N_AXIS`,
# `-lv 4` for the acceptance statistics (the response comes over HTTP, so the log stays clean).
if [ "$AXES" != none ] && [ -n "$AXES" ]; then
    s=27b-3gpu-f16
    IFS=',' read -r -a axes <<<"$AXES"
    for a in "${axes[@]}"; do
        case "$a" in
            R) P="$REPO/prompts/reasoning.txt" ;;
            K) P="$REPO/prompts/recall.txt" ;;
            P) P="$REPO/prompts/prose-rdna-boosts.txt" ;;
            C) P="$REPO/prompts/code-python.txt" ;;
            *) echo "FAIL  unknown axis '$a'"; fails=$((fails+1)); continue ;;
        esac
        echo
        echo "-- four-axis $a on $s (n=$N_AXIS, one request per arm)"
        declare -A ax_sha=() ax_gen=() ax_acc=() ax_steps=() ax_hand=()
        for arm in pair-none pair-mtp; do
            spec=none; [ "$arm" = pair-mtp ] && spec=adaptive
            tag="$s-axis-$a-$arm"
            if ! run_arm "$tag" "$(shape_var "$s" devices)" "$(arm_args "$s" on "$spec")" \
                    "$N_AXIS" 1 "$P"; then
                echo "FAIL  $tag did not complete (see $OUT/$tag.out)"
                tail -3 "$OUT/$tag.out" 2>/dev/null | sed 's/^/      /'
                fails=$((fails+1)); continue
            fi
            ax_sha[$arm]="$(sed -n 's/^slot0: [0-9]* chars sha=\([0-9a-f]*\).*/\1/p' "$OUT/$tag.hashes" 2>/dev/null)"
            ax_gen[$arm]="$(stat "$OUT/$tag.log" '#gen drafts')"
            ax_acc[$arm]="$(stat "$OUT/$tag.log" '#acc drafts')"
            census="$(grep -a 'decode arm is generating' "$OUT/$tag.log" | tail -1)"
            ax_steps[$arm]="$(sed -n 's/.*generating: *\([0-9]*\) step(s).*/\1/p' <<<"$census")"
            ax_hand[$arm]="$(grep -oE '[0-9]+ handover' <<<"$census" 2>/dev/null | grep -oE '^[0-9]+' | tail -1)"
            echo "      $arm: text=${ax_sha[$arm]:-<absent>} drafts=${ax_gen[$arm]:-0} accepted=${ax_acc[$arm]:-0}" \
                 "decode-arm steps=${ax_steps[$arm]:--} handovers=${ax_hand[$arm]:--}"
        done
        hard "$([ -n "${ax_sha[pair-none]:-}" ] && [ "${ax_sha[pair-none]:-x}" = "${ax_sha[pair-mtp]:-y}" ] && echo 1 || echo 0)" \
            "axis $a: pair+none and pair+draft read ONE text (${ax_sha[pair-none]:-<absent>})"
        hard "$([ "${ax_gen[pair-mtp]:-0}" -gt 0 ] && [ "${ax_acc[pair-mtp]:-0}" -gt 0 ] && echo 1 || echo 0)" \
            "axis $a: the drafter really drafted and accepted ON the pair (${ax_gen[pair-mtp]:-0} drafts, ${ax_acc[pair-mtp]:-0} accepted)"
        hard "$([ "${ax_steps[pair-mtp]:-0}" -ge 2 ] && [ "${ax_hand[pair-mtp]:-0}" -ge 1 ] && echo 1 || echo 0)" \
            "axis $a: the decode arm really ran (${ax_steps[pair-mtp]:-0} step(s), ${ax_hand[pair-mtp]:-0} handover(s))"
        unset ax_sha ax_gen ax_acc ax_steps ax_hand
    done
fi

echo
echo "--> GPU faults"
if [ "${#LOGS[@]}" -gt 0 ] && [ -x "$REPO/scripts/check-gpu-faults.sh" ]; then
    for l in "${LOGS[@]}"; do
        [ -f "$l" ] || continue
        "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" --log "$l" >/dev/null 2>&1 \
            || { echo "FAIL  a GPU fault appeared during $l"; fails=$((fails+1)); }
    done
    echo "    journal scanned for ${#LOGS[@]} log(s)"
fi

echo
if [ "$fails" = 0 ]; then
    echo "==> PASS  the speculator runs ON the pair: one text across four arms, drafts accepted on the decode arm"
else
    echo "==> FAIL  $fails check(s) failed; logs in $OUT"
fi
[ "$KEEP" = 1 ] && echo "    logs kept in $OUT"
exit $(( fails > 0 ? 1 : 0 ))
