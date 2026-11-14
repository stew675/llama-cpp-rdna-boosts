#!/usr/bin/env bash
# Speculator GDN-split gate (issue #309).
#
# WHY THIS EXISTS
#   A hybrid/recurrent target (qwen35/qwen3.8, qwen4exp, ...) splits its chunked GDN prefill at
#   `GDN_CHUNKED_MIN_TOKENS` and finishes it with a sequential tail of that many tokens
#   (ggml/src/ggml-cuda/gated_delta_net.cu).  The tail length is a *numeric* property of the graph: move it
#   and every state the prefill hands to the decode changes, so the model's own greedy output changes.
#
#   For one release cycle the threshold was floored with `n_rs_batch`, i.e. with the WIDEST draft any enabled
#   speculator can produce (`common_speculative_n_max() + 1`).  The draft-based family is clamped to 15
#   (`--spec-draft-n-max`), so its bound was 16 and the gate below would have passed; the DRAFTLESS n-gram
#   family is not clamped (`--spec-ngram-mod-n-max` defaults to 64), so merely ADDING
#   `--spec-type ...,ngram-mod` to a command moved the tail from 16 to 65 and changed the model's text -
#   with no draft even being used.  That is issue #309, and no gate in this directory could see it: the
#   coherence gate, the prefill-logit gate and the MTP four-axis protocol all run ONE speculator set, so a
#   speculator-set-dependent prefill split has nothing to be compared against.
#
#   The invariant this gate enforces is therefore: THE SPECULATOR SET MUST NOT CHANGE THE MODEL'S OWN GREEDY
#   TEXT, and the mechanism must be visible in the census (the same prefill tail in every arm, whatever
#   `n_rs_batch` is).
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (GGML_ASSERT / terminate / segfault) and scripts/check-gpu-faults.sh reads the
#           kernel journal clean over this gate's own window (skipped with --no-faults)
#   HARD  * every arm really ran a chunked GDN prefill: the `chunked GDN prefill ... sequential tail N` INFO
#           line appears in every arm's log (with -lv 4).  A model/config that never chunks (a short prompt, a
#           non-recurrent arch, `GGML_CUDA_GDN_CHUNKED=0`) SKIPs instead of passing vacuously.
#   HARD  * the tail length N is the SAME in every arm, and `n_rs_batch` is reported per arm so a reader can
#           see that the bound DID differ between them.  This is the mechanism check: an arm whose tail
#           differs has moved the prefill split, which is the defect.
#   HARD  * every arm's greedy text is byte-identical to the `plain` arm (`--spec-type none`), per prompt
#   HARD  * the wide n-gram arm actually drafted deep (its ngram-mod mean accepted length is above the
#           `narrow` arm's), so a "pass" cannot come from the speculator being inert
#   INFO  generation t/s per arm, so the wide-draft win is visible in the record (it is NOT gated on)
#
#   WHAT IT CANNOT FORCE.  A greedy near-tie flip is data-dependent: a model/prompt where a moved boundary
#   never changes the argmax passes this gate while being wrong.  The mechanism check (identical tail, per
#   arm) is the part that cannot be defeated that way, which is why the two are both HARD.  The gate also
#   cannot see a speculator that changes the text through the verify WIDTH instead of the prefill split; that
#   is the documented 8-row trade (`GREEDY-PURITY.md` 11/19) and is measured by `gate-spec-align.sh` /
#   `gate-spec-width-band.sh`.
#
# Usage:
#   ./gate-spec-gdn-split.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_CLI     llama-cli binary      (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   MODEL         hybrid target GGUF    (default: Qwen3.8-27B Q8_0, Unsloth)
#   MODEL_ALT     optional second file  (default: empty; e.g. the Bartowski Q8_0, whose ssm_alpha/beta are
#                                       F32 instead of Q8_0, which is the report that opened issue #309)
#   PROMPT_A      ordinary prose        (default: prompts/prose-rdna-boosts.txt, sha256-pinned there)
#   PROMPT_B      verbatim-copy prompt  (default: prompts/recall.txt, sha256-pinned there)
#   NGEN          generated tokens      (default: 600)
#   CTX           -c                    (default: 32768)
#   NGPU          GPUs                  (default: 3 -> `-sm tensor -ts 1,2,2`; 1 = the default split)
#   GDN_ARM       optional extra env    (default: empty; e.g. "GGML_CUDA_GDN_CHUNKED_BF16=0")
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}"
MODEL_ALT="${MODEL_ALT:-}"
PROMPT_A="${PROMPT_A:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
PROMPT_B="${PROMPT_B:-$REPO_DIR/prompts/recall.txt}"
NGEN="${NGEN:-600}"
CTX="${CTX:-32768}"
NGPU="${NGPU:-3}"
GDN_ARM="${GDN_ARM:-}"
KEEP=0
FAULTS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --keep)      KEEP=1; shift ;;
        --no-faults) FAULTS=0; shift ;;
        -h|--help)   sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "ERROR: no model at $MODEL" >&2; exit 2; }

TMP="$(mktemp -d)"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$TMP"; }
trap cleanup EXIT

DEV="0"
SM=()
if [ "$NGPU" -gt 1 ]; then
    DEV="$(seq -s, 0 $((NGPU - 1)))"
    SM=(-sm tensor)
    case "$NGPU" in
        2) SM+=(-ts 1,1) ;;
        3) SM+=(-ts 1,2,2) ;;
        *) : ;;
    esac
fi

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

# The arms.  `plain` is the reference; `mtp` and `mtp+ngram-narrow` are the bounded-width controls; the
# `mtp+ngram` arm is issue #309's set (ngram-mod at its DEFAULT width, which is what a user gets).
ARM_NAMES=(plain mtp ngram-narrow ngram-default)
arm_args() {
    case "$1" in
        plain)         echo "" ;;
        mtp)           echo "--spec-type draft-mtp --spec-draft-n-max 4" ;;
        ngram-narrow)  echo "--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 40 --spec-ngram-mod-n-max 15 --spec-draft-n-max 4 --spec-draft-n-start 4" ;;
        ngram-default) echo "--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 40 --spec-draft-n-max 4 --spec-draft-n-start 4" ;;
    esac
}

FAIL=0
fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }

echo "==> speculator GDN-split gate (issue #309)"
echo "    cli=$LLAMA_CLI"
echo "    model=$MODEL"
echo "    devices=$DEV ${SM[*]:-}  ctx=$CTX  ngen=$NGEN  prompts=$(basename "$PROMPT_A"),$(basename "$PROMPT_B")"
[ -n "$GDN_ARM" ] && echo "    extra env: $GDN_ARM"

# text runs carry NO -lv at all (llama-cli streams the generated text on stdout and every log level that
# adds a census interleaves into it - AGENTS.md, the -lv table), so the hash and the census are two runs.
run_arm() { # model prompt arm tag n lv
    local m="$1" p="$2" arm="$3" tag="$4" n="$5" lv="$6"
    local log="$TMP/${TAG_MODEL}/$tag.log"
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEV" $GDN_ARM timeout 3600 "$LLAMA_CLI" \
        -m "$m" -ngl 99 "${SM[@]}" -c "$CTX" -f "$p" -n "$n" \
        --seed 42 --temp 0 --reasoning off --no-display-prompt --single-turn -lm dio \
        $(arm_args "$arm") $lv > "$log" 2>&1
    echo "$log"
}

check_model() { # model tag
    local m="$1"
    TAG_MODEL="$2"
    local out="$TMP/$TAG_MODEL"
    mkdir -p "$out"
    local texts=() tails=() rsb=() tps=()

    for arm in "${ARM_NAMES[@]}"; do
        # (a) the text: no -lv, so the stream carries only the generated span
        local tlog
        tlog="$(run_arm "$m" "$PROMPT_A" "$arm" "text-$arm" "$NGEN" "")"
        if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|Aborted" "$tlog"; then
            fail "$TAG_MODEL/$arm aborted (see $tlog)"
            continue
        fi
        local h
        if ! h="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$tlog" 2>/dev/null)"; then
            fail "$TAG_MODEL/$arm produced no extractable text (see $tlog)"
            continue
        fi
        texts+=("$arm=${h##*sha=}")
        tps+=("$arm=$(grep -oE "Generation: [0-9.]+ t/s" "$tlog" | tail -1 | grep -oE "[0-9.]+")")

        # (b) the mechanism: -lv 4, one token generated (the census prints during the prompt prefill)
        local clog tail rs
        clog="$(run_arm "$m" "$PROMPT_A" "$arm" "census-$arm" 1 "-lv 4")"
        tail="$(grep -oE "sequential tail [0-9]+" "$clog" | grep -oE "[0-9]+" | head -1)"
        rs="$(grep -oE "rollback bound n_rs_batch=[0-9]+" "$clog" | grep -oE "[0-9]+$" | head -1)"
        tails+=("$arm=${tail:-none}")
        rsb+=("$arm=n_rs_batch=${rs:-?}")
    done

    echo "    --- $TAG_MODEL ---"
    echo "    text         : ${texts[*]}"
    echo "    prefill tail : ${tails[*]}"
    echo "    rollback bnd : ${rsb[*]}"
    echo "    generation   : ${tps[*]} t/s"

    local vac=0 t
    for t in "${tails[@]}"; do
        case "${t#*=}" in none|"") vac=$((vac+1)) ;; esac
    done
    if [ "$vac" -gt 0 ]; then
        echo "SKIP: $TAG_MODEL: no chunked GDN prefill in $vac arm(s) (short prompt, non-recurrent arch or"
        echo "      GGML_CUDA_GDN_CHUNKED=0) - there is no split to compare"
        return 2
    fi

    # HARD: one tail for every arm (the mechanism check - cannot be defeated by a near-tie that did not flip)
    local uniq
    uniq="$(printf '%s\n' "${tails[@]#*=}" | sort -u | wc -l)"
    if [ "$uniq" -ne 1 ]; then
        fail "$TAG_MODEL: the speculator set moved the GDN prefill tail: ${tails[*]}"
    else
        pass "$TAG_MODEL: one GDN prefill tail for every speculator set (${tails[0]#*=}), n_rs_batch ${rsb[*]}"
    fi

    # HARD: every arm reproduces `plain` byte for byte
    local ref="${texts[0]#*=}" bad=0
    for t in "${texts[@]}"; do
        [ "${t#*=}" != "$ref" ] && bad=$((bad+1))
    done
    if [ "$bad" -ne 0 ]; then
        fail "$TAG_MODEL: a speculator set changed the model's own greedy text: ${texts[*]}"
    else
        pass "$TAG_MODEL: every speculator set reproduces the plain greedy text ($ref)"
    fi

    # HARD: the wide arm really drafted deeper than the narrow one on the copy prompt, so a PASS cannot come
    # from an inert speculator (the census run is -lv 4 and hashes nothing)
    local acc_wide acc_narrow
    acc_wide="$(run_arm "$m" "$PROMPT_B" ngram-default "wide-$TAG_MODEL" 200 "-lv 4")"
    acc_narrow="$(run_arm "$m" "$PROMPT_B" ngram-narrow "narrow-$TAG_MODEL" 200 "-lv 4")"
    local aw an
    aw="$(grep -oE "ngram-mod:.*#mean acc len = *[0-9.]+" "$acc_wide" | grep -oE "[0-9.]+$" | tail -1)"
    an="$(grep -oE "ngram-mod:.*#mean acc len = *[0-9.]+" "$acc_narrow" | grep -oE "[0-9.]+$" | tail -1)"
    echo "    copy-prompt ngram mean accepted length: default=${aw:-?} narrow=${an:-?}"
    if [ -n "${aw:-}" ] && [ -n "${an:-}" ]; then
        if awk "BEGIN{exit !($aw > $an)}"; then
            pass "$TAG_MODEL: the default n-gram width still drafts deeper than the bounded one (${aw} > ${an})"
        else
            fail "$TAG_MODEL: the default n-gram width did not out-draft the bounded one (${aw} vs ${an}) - the speculator is inert and this gate proves nothing"
        fi
    else
        echo "SKIP: no ngram-mod census on $(basename "$PROMPT_B") - the deep-draft arm cannot be checked"
    fi
    return 0
}

SKIPPED=0
for m in "$MODEL" ${MODEL_ALT:+"$MODEL_ALT"}; do
    check_model "$m" "$(basename "$(dirname "$m")")"
    [ "$?" = 2 ] && SKIPPED=$((SKIPPED+1))
done

if [ "$FAULTS" = 1 ]; then
    fault_logs=()
    for f in "$TMP"/*/*.log; do
        [ -f "$f" ] && fault_logs+=(--log "$f")
    done
    if "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$SINCE" "${fault_logs[@]}" --quiet; then
        pass "no GPU fault in the kernel journal or any arm's log"
    else
        fail "GPU fault reported - rerun with --keep and inspect $TMP"
        KEEP=1
    fi
fi

if [ "$FAIL" != 0 ]; then
    echo "==> FAIL (logs in $TMP, kept)" >&2
    exit 1
fi
if [ "$SKIPPED" != 0 ]; then
    echo "==> SKIP: no arm chunked a GDN prefill ($SKIPPED model(s)) - pick a hybrid model and a prompt long"
    echo "    enough to chunk, or unset GGML_CUDA_GDN_CHUNKED=0"
    exit 0
fi
echo "==> PASS: the speculator set does not change the model's own greedy text, and every arm splits the"
echo "    GDN prefill with the same sequential tail."
exit 0
