#!/usr/bin/env bash
# The padded GQA-6 tile fold gate (block 05, contributor PR #362).
#
# WHY THIS EXISTS
#   The fold replaces the tile kernel's decode/verify kernel for ONE geometry -- head 256/256, GQA ratio 6,
#   a symmetric NATIVE q8_0 K/V pair, query width 1..8, RDNA3_0 -- with a dedicated kernel that places all
#   six real query heads in one padded eight-column block (`ncols1 1, ncols2 8`), so a query row crosses
#   the K/V cache once instead of three times.  It is DEFAULT ON, and three things about that are invisible
#   to every other gate:
#     1. it is a DIFFERENT kernel (a copy of flash_attn_tile, isolated there so the ordinary instantiations
#        do not move), so the arm can only be trusted while the arm is provably the one that ran: the
#        one-shot INFO census at `-lv 4` is the proof, and a census that fires on a build where the fold
#        never launches (or never fires where it should) makes the whole thing unreadable;
#     2. the geometry test is a runtime decision (K/V operand types, GQA ratio, width), not a template
#        parameter, so the MIXED K/V pair of optional 0101 must NOT take it -- the fold's kernel has one
#        K/V type and reads both operands as q8_0;
#     3. the arch gate is RDNA3_0.  On RDNA4 the band owns the decode depths and the tile window below it
#        is not attention-bound, so a fold there is a regression risk with no win; the gate proves the
#        refusal (no census on gfx1201 even with the kill switch unset).
#
#   The numerics point is the DECODE WIDTH, not the prefill gate's default: the fold needs a native q8_0
#   K/V pair (the prefill gate defaults to f16, where the fold cannot run at all) and it is width-gated
#   (`Q->ne[1] <= 8`), so a ubatch-wide prefill scores `Q->ne[1] > 8` and is blind to it by construction.
#   `gate-prefill-logits.sh --kv q8_0 --ub 1` is the point that can see it, and it is run here against a
#   base recorded from the previous release (or from this one's kill-switch arm on a fresh box):
#     * the kill-switch arm must read 0.000000 / 100.000 % against that base -- it IS the ordinary kernel,
#       so anything else means the isolation leaked;
#     * the default (fold) arm must read the same, i.e. the fold is a rounding-level re-grouping at this
#       point.  If it does not, the artifact is not shippable as a default-on feature.
#
# Usage:
#   ./gate-gqa6-fold.sh [--build DIR] [--model FILE] [--base-dir DIR] [--record-base]
#                       [--device N] [--ctx N] [--chunks N] [--no-faults]
#
# Environment:
#   BUILD_DIR   llama.cpp build tree             (default: ~/llama.cpp/build-rocm)
#   TBO         test-backend-ops                 (default: $BUILD_DIR/bin/test-backend-ops)
#   LLAMA_CLI   llama-cli                        (default: $BUILD_DIR/bin/llama-cli)
#   MODEL       the head-256/GQA-6 model         (default: the first /llm/models/Qwen3.8/27B/*/*.gguf)
#   KV          the K/V cache pair               (default: q8_0, the only pair the fold accepts)
#   GPU         HIP_VISIBLE_DEVICES              (default: 0)
#   BASE_DIR    prefill-logit base dir           (default: ~/.cache/rdna-boosts/gates)
#   PROMPT      text-arm prompt                  (default: prompts/prose-rdna-boosts.txt)
#   NGEN        text-arm generated tokens         (default: 128)
#   A_ARM/B_ARM/C_ARM  0 skips a section          (default: 1)
#
# Exit: 0 PASS, 1 FAIL, 2 usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
TBO="${TBO:-$BUILD_DIR/bin/test-backend-ops}"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
MODEL="${MODEL:-}"
KV="${KV:-q8_0}"
GPU="${GPU:-0}"
BASE_DIR="${BASE_DIR:-$HOME/.cache/rdna-boosts/gates}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
NGEN="${NGEN:-128}"
A_ARM="${A_ARM:-1}"
B_ARM="${B_ARM:-1}"
C_ARM="${C_ARM:-1}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:-2}"
DEVICE="${DEVICE:-0}"
FAULTS="${FAULTS:-1}"
RECORD_BASE=0
KILL=GGML_HIP_FA_TILE_GQA6_FOLD
TEXT_OPTS='hsk=256.*nr23=\[6,1\]'
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
SINCE=""
fail=0
note() { echo "  $*"; }
bad()  { echo "  FAIL: $*" >&2; fail=1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --build)      BUILD_DIR="$2"; TBO="$BUILD_DIR/bin/test-backend-ops"; LLAMA_CLI="$BUILD_DIR/bin/llama-cli"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --base-dir)   BASE_DIR="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --ctx)        CTX="$2"; shift 2 ;;
        --chunks)     CHUNKS="$2"; shift 2 ;;
        --record-base) RECORD_BASE=1; shift ;;
        --no-faults)  FAULTS=0; shift ;;
        -h|--help)    sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_CLI" ] || { echo "ERROR: no llama-cli at $LLAMA_CLI" >&2; exit 2; }
[ -x "$TBO" ]       || { echo "ERROR: no test-backend-ops at $TBO" >&2; exit 2; }
[ -f "$PROMPT" ]    || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }

if [ -z "$MODEL" ]; then
    MODEL="$(ls -1 /llm/models/Qwen3.8/27B/*/*.gguf 2>/dev/null | head -1)"
fi
[ -n "$MODEL" ] && [ -f "$MODEL" ] || {
    echo "ERROR: no model.  The fold only engages on head-256/256 with a GQA ratio of 6 (qwen35 27B," >&2
    echo "       DeepSeek-Ling, ...); pass --model." >&2; exit 2; }

echo "==> padded GQA-6 tile fold gate (block 05, PR #362)"
echo "    build=$BUILD_DIR  model=$(basename "$MODEL")  kv=$KV  device=$DEVICE"
echo "    arm A (census + text) / B (op oracle) / C (decode-width KLD)  keep=$KEEPOUT"

# the device's gfx target, from llama.cpp's own device banner (the gate runs on one pinned device)
CC="$(rocminfo 2>/dev/null | sed -n 's/^ *Name: *\(gfx[0-9a-f]*\).*/\1/p' | head -1)"
[ -n "$CC" ] || CC="$(HIP_VISIBLE_DEVICES="$GPU" "$LLAMA_CLI" -m "$MODEL" -ngl 0 -p x -n 0 \
    --no-display-prompt --single-turn 2>&1 | grep -oE 'gfx[0-9a-f]{3,4}' | head -1)"
echo "    cc=${CC:-unknown}"

if [ "$FAULTS" = 1 ] && command -v journalctl >/dev/null; then SINCE="$(date '+%Y-%m-%d %H:%M:%S')"; fi

# ---------------------------------------------------------------------------
# A. the census + the text: the fold must be the arm that ran, and the kill switch must not be
# ---------------------------------------------------------------------------
TEXTLOG=""
run_text() {  # run_text TAG [ENV=VAL ...]  -> prints ONE "N chars sha=..." line
    local tag="$1"; shift
    TEXTLOG="$KEEPOUT/text-$tag.log"
    env "$@" "$LLAMA_CLI" -m "$MODEL" -ngl 99 -ctk "$KV" -ctv "$KV" -fa auto \
        -f "$PROMPT" -n "$NGEN" --seed 42 --temp 0 --reasoning off \
        --no-display-prompt --single-turn > "$TEXTLOG" 2>&1
    python3 "$REPO_DIR/scripts/extract-generated.py" "$TEXTLOG" 2>/dev/null | tail -1
}
run_census() {  # run_census TAG [ENV=VAL ...]  -> prints the census hit count for a 1-token probe
    local tag="$1"; shift
    env "$@" "$LLAMA_CLI" -m "$MODEL" -ngl 99 -ctk "$KV" -ctv "$KV" -fa auto \
        -p "The capital of France is" -n 32 --seed 42 --temp 0 --no-display-prompt --single-turn -lv 4 \
        > "$KEEPOUT/census-$tag.log" 2>&1
    grep -c "padded GQA-6 tile fold ON" "$KEEPOUT/census-$tag.log"
}

if [ "$A_ARM" = 1 ]; then
    echo "--> arm A: census non-vacuity + same-seed text (default vs $KILL=0)"
    on_census="$(run_census on)"
    off_census="$(run_census off "$KILL=0")"
    note "census lines: default=$on_census  kill-switch=$off_census"
    if [ "$on_census" -ge 1 ] && [ "$off_census" -eq 0 ]; then
        note "OK: the fold launched by default and the kill switch really stands it down"
    else
        if [ "$on_census" -eq 0 ] && [ "${CC:-}" = "gfx1201" ]; then
            note "OK (RDNA4): the arch gate refuses the fold by construction ($CC), census 0 in both arms"
        else
            bad "the census does not prove the arm: default=$on_census kill-switch=$off_census (a numeric arm that"
            bad "    cannot show it ran is not reviewable; check the model's geometry: head 256/256, GQA ratio 6,"
            bad "    a native q8_0 K/V pair, RDNA3_0)"
        fi
    fi
    on_sha="$(run_text on)"
    off_sha="$(run_text off "$KILL=0")"
    note "text default:     $on_sha"
    note "text kill-switch: $off_sha"
    [ -n "$on_sha" ] && [ -n "$off_sha" ] || bad "a text arm produced no extractable output (see $KEEPOUT)"
    if [ "$on_sha" = "$off_sha" ]; then
        note "OK: same-seed greedy text identical in both arms"
    else
        # A near-tie inside the rounding-level re-grouping is ALLOWED (the fold is not bit-identical by
        # construction) -- but then the KLD point must be quoted with it, so this is reported, not failed.
        note "WARN: the text moved between the arms -- quote the arm C KLD point with it (a near-tie inside"
        note "      the re-grouping is expected and allowed; a moved TEXT with a clean KLD is not)"
    fi
fi

# ---------------------------------------------------------------------------
# B. the op oracle: the geometry's coverage must pass in both arms
# ---------------------------------------------------------------------------
if [ "$B_ARM" = 1 ]; then
    echo "--> arm B: test-backend-ops FLASH_ATTN_EXT at head-256/GQA-6, both arms"
    for arm in "on:" "off:$KILL=0"; do
        tag="${arm%%:*}"; envs="${arm#*:}"
        # shellcheck disable=SC2086
        out="$(env $envs "$TBO" -o FLASH_ATTN_EXT -b ROCm"$DEVICE" -p "$TEXT_OPTS" 2>&1 | tail -3)"
        printf '%s\n' "$out" | sed 's/^/  /'
        if printf '%s\n' "$out" | grep -q "backends passed"; then
            note "OK ($tag arm)"
        else
            bad "the head-256/GQA-6 coverage failed in the $tag arm"
        fi
    done
fi

# ---------------------------------------------------------------------------
# C. the decode-width numerics point (the ONLY point that can see the fold)
# ---------------------------------------------------------------------------
if [ "$C_ARM" = 1 ]; then
    echo "--> arm C: decode-width KLD (--kv $KV --ub 1) -- the fold's own point"
    GL=( --base-dir "$BASE_DIR" --model "$MODEL" --kv "$KV" --ub 1 --ctx "$CTX" --chunks "$CHUNKS"
         --device "$DEVICE" )
    [ "$FAULTS" = 0 ] && GL+=( --no-faults )
    if [ "$RECORD_BASE" = 1 ]; then
        echo "    recording the base from the KILL-SWITCH arm (the ordinary kernel) -- pass a base from the"
        echo "    PREVIOUS release instead when one exists: this records the build under test as its own base"
        echo "    only for the fold's internal off/on comparison."
        LLAMA_PPL="${LLAMA_PPL:-$BUILD_DIR/bin/llama-perplexity}" \
            env "$KILL=0" "$REPO_DIR/scripts/gate-prefill-logits.sh" --record "${GL[@]}" 2>&1 |
            grep -E "base PPL|mean KLD|same top|PASS|FAIL" | sed 's/^/  /'
    else
        kld_arm() {  # kld_arm LABEL [--candidate-env S]
            local label="$1"; shift
            local out rc
            out="$("$REPO_DIR/scripts/gate-prefill-logits.sh" "${GL[@]}" "$@" 2>&1)"; rc=$?
            printf '%s\n' "$out" | grep -E "candidate:|mean KLD|same top|PASS|FAIL|ERROR: base" | sed 's/^/  /'
            if printf '%s' "$out" | grep -q "ERROR: base not found"; then
                bad "the $label arm has no base to compare against.  Record one from the PREVIOUS release's"
                bad "    build (not from this one), then re-run: "
                bad "    LLAMA_PPL=<prev-release>/bin/llama-perplexity $REPO_DIR/scripts/gate-prefill-logits.sh --record \\"
                bad "      --base-dir $BASE_DIR --model $MODEL --kv $KV --ub 1 --ctx $CTX --chunks $CHUNKS --device $DEVICE"
            elif [ $rc -ne 0 ]; then
                bad "the $label arm's KLD point FAILED (see above)"
            fi
        }
        echo "    off arm (must be the baseline: mean KLD 0.000000 / same-top-p 100.000 %):"
        kld_arm "off" --candidate-env "$KILL=0"
        echo "    fold arm (must read the same numbers -- the fold is a rounding-level re-grouping here):"
        kld_arm "fold"
    fi
fi

# ---------------------------------------------------------------------------
# D. the fault scan
# ---------------------------------------------------------------------------
if [ -n "$SINCE" ]; then
    echo "--> fault scan (kernel journal since $SINCE)"
    if bash "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$SINCE" 2>&1 | tail -3 | sed 's/^/  /'; then
        note "OK: no GPU fault in this gate's window"
    else
        bad "a GPU fault was recorded in this gate's window (check-gpu-faults.sh above)"
    fi
fi

echo
if [ "$fail" -eq 0 ]; then echo "==> PASS: gate-gqa6-fold"; else echo "==> FAIL: gate-gqa6-fold (see above)" >&2; fi
exit "$fail"
