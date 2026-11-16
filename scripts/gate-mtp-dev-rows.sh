#!/usr/bin/env bash
# MTP device h-rows gate (`LLAMA_MTP_DEV_ROWS`, issue #318).
#
# WHY THIS EXISTS
#   Upstream #31927 keeps the MTP target's hidden rows (`h_nextn`) in a device buffer and lets the drafter read
#   them in place instead of copying each target batch's rows back to the host.  The delivery shipped that path
#   on (r281, when the base gained #31927) until r293, when a Windows/HIP user measured it as a -1.3 to -1.6 %
#   DECODE LOSS at every depth from 63k to 161k (issue #318) with identical text and identical acceptance, and
#   this box then measured it NEUTRAL: the ordering handshake (one host-blocking event wait plus two blocking
#   row copies per decode round) replaces the single async readback the host path had already paid for.  So the
#   delivery DEFAULTS TO THE HOST ROWS (the pre-#31927 behaviour) and keeps the device path behind
#   `LLAMA_MTP_DEV_ROWS=1`.
#
#   What must stay true, and what nothing else in this directory checks:
#     * the DEFAULT run is the host-row arm (a release that silently turns the device path back on is exactly
#       the state issue #318 was filed against);
#     * the opt-in arm really takes the device path (otherwise the opt-in is dead code);
#     * the two arms produce the SAME greedy text and the SAME draft acceptance, because the arms only move
#       where the target's hidden rows are read from.  A drafter fed different rows can only lose acceptance
#       (the target verifies every draft), but a wrong row is invisible to a text-only check, so the acceptance
#       counts are gated too.
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (GGML_ASSERT / terminate / segfault) and scripts/check-gpu-faults.sh reads the kernel
#           journal clean over this gate's own window (skipped with --no-faults)
#   HARD  * the default arm's census is `target h rows on the device: no` AND its log has no `h_nextn buffer
#           size` line: the device buffer must not even be allocated on a default run
#   HARD  * the `device` arm's census is `target h rows on the device: yes` and its log DOES carry the
#           `h_nextn buffer size` line (the buffer was allocated, so the arm is not vacuous)
#   HARD  * the two arms' greedy text is byte-identical (no `-lv` on a text arm, see AGENTS.md's `-lv` table)
#   HARD  * the two arms' draft acceptance counts (accepted / generated) are identical
#   INFO  generation t/s per arm, so the arm's cost on this platform is visible in every record (NOT gated: on
#         this box the two are within noise, see ENVIRONMENT.md §3)
#
#   WHAT IT CANNOT FORCE.  A model whose MTP head shares the target's memory (a gemma4 assistant) or a
#   multi-head chained MTP (step35) never takes the device path at all (`is_mem_shared` / `chain_heads` keep the
#   host rows), so the gate SKIPs on such a model instead of passing vacuously: point MODEL at a single-head
#   non-shared MTP model.  A multi-sequence server arm is NOT covered (the device path keeps one carry row per
#   sequence, `n_batch + seq_id`); the arm is opt-in, off by default, and the single-sequence arms above are
#   what makes the default safe.
#
# Usage:
#   ./gate-mtp-dev-rows.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_CLI     llama-cli binary    (default: ~/llama.cpp/build-rocm/bin/llama-cli)
#   MODEL         single-head MTP GGUF (default: Qwen3.5-4B Q8_0, which carries a `nextn` head)
#   PROMPT        prose prompt        (default: prompts/prose-rdna-boosts.txt, sha256-pinned there)
#   NGEN          generated tokens    (default: 128)
#   CTX           -c                  (default: 8192)
#   NGPU          GPUs                (default: 1; >1 uses `-sm layer`, which keeps the target's output device
#                                     the device the draft reads, i.e. the arm the device path actually reaches)
#   KV            KV cache type       (default: q8_0)
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
PROMPT="${PROMPT:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
NGEN="${NGEN:-128}"
CTX="${CTX:-8192}"
NGPU="${NGPU:-1}"
KV="${KV:-q8_0}"
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
[ -f "$PROMPT" ]    || { echo "ERROR: no prompt at $PROMPT" >&2; exit 2; }

TMP="$(mktemp -d)"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$TMP"; }
trap cleanup EXIT

DEV="0"
SM=()
if [ "$NGPU" -gt 1 ]; then
    DEV="$(seq -s, 0 $((NGPU - 1)))"
    SM=(-sm layer)
fi

SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

# arms: label -> LLAMA_MTP_DEV_ROWS value (empty = the shipped default, i.e. no env at all)
ARM_NAMES=(default device)

FAIL=0
fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }

echo "==> MTP device h-rows gate (LLAMA_MTP_DEV_ROWS, issue #318)"
echo "    cli=$LLAMA_CLI"
echo "    model=$MODEL"
echo "    devices=$DEV ${SM[*]:-}  ctx=$CTX  ngen=$NGEN  kv=$KV  prompt=$(basename "$PROMPT")"

# text runs carry NO -lv: llama-cli streams the generated text on stdout and every level that adds a census
# interleaves into it (AGENTS.md, the -lv table), so the hash and the census are two runs per arm.
run_arm() { # arm tag n lv
    local arm="$1" tag="$2" n="$3" lv="$4"
    local log="$TMP/$tag.log"
    local env_val=""
    [ "$arm" = device ] && env_val=1
    # shellcheck disable=SC2086
    env HIP_VISIBLE_DEVICES="$DEV" LLAMA_MTP_DEV_ROWS="$env_val" timeout 3600 "$LLAMA_CLI" \
        -m "$MODEL" -ngl 99 "${SM[@]}" -c "$CTX" -f "$PROMPT" -n "$n" -ctk "$KV" -ctv "$KV" \
        --seed 42 --temp 0 --no-display-prompt --single-turn \
        --spec-type draft-mtp --spec-draft-n-max 3 $lv > "$log" 2>&1
    echo "$log"
}

texts=() acc=() tps=() census=() bufs=()

for arm in "${ARM_NAMES[@]}"; do
    # (a) the text: no -lv, so the stream carries only the generated span
    tlog="$(run_arm "$arm" "text-$arm" "$NGEN" "")"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault|Aborted" "$tlog"; then
        fail "$arm aborted (see $tlog)"
        continue
    fi
    if ! h="$(python3 "$REPO_DIR/scripts/extract-generated.py" "$tlog" 2>/dev/null)"; then
        fail "$arm produced no extractable text (see $tlog)"
        continue
    fi
    texts+=("$arm=${h##*sha=}")
    tps+=("$arm=$(grep -oE "Generation: [0-9.]+ t/s" "$tlog" | tail -1 | grep -oE "[0-9.]+")")

    # (b) the census and the acceptance counts: -lv 4 (the census is a library INFO line, the acceptance
    #     summary is printed at the end of the generation, so this run generates the same number of tokens)
    clog="$(run_arm "$arm" "census-$arm" "$NGEN" "-lv 4")"
    census+=("$arm=$(sed -nE 's/.*- target h rows on the device: (yes|no).*/\1/p' "$clog" | head -1)")
    if grep -q "h_nextn buffer size" "$clog"; then
        bufs+=("$arm=allocated")
    else
        bufs+=("$arm=none")
    fi
    # draft acceptance: identical counts are the purity property the two arms share
    a="$(sed -nE 's/.*draft acceptance = ([0-9.]+) \( *([0-9]+) accepted \/ *([0-9]+) generated\).*/\2\/\3/p' "$clog" | head -1)"
    acc+=("$arm=${a:-none}")
done

echo "    text       : ${texts[*]}"
echo "    acceptance : ${acc[*]}"
echo "    h_nextn    : ${census[*]}  (buffer: ${bufs[*]})"
echo "    generation : ${tps[*]} t/s"

census_of() { local kv; for kv in "${census[@]}"; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo "missing"; }
bufs_of()   { local kv; for kv in "${bufs[@]}";   do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo "missing"; }
acc_of()    { local kv; for kv in "${acc[@]}";    do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo "missing"; }
text_of()   { local kv; for kv in "${texts[@]}";  do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo "missing"; }

# the default arm: host rows, and the device buffer is not even allocated
if [ "$(census_of default)" = "no" ] && [ "$(bufs_of default)" = "none" ]; then
    pass "default: host rows (no h_nextn device buffer allocated)"
elif [ "$(census_of default)" = "missing" ]; then
    fail "default: no census line (the MTP drafter never started)"
else
    fail "default is the DEVICE path (census '$(census_of default)', buffer '$(bufs_of default)'): the shipped default must be the host rows (issue #318)"
fi

# the opt-in arm: the device path, with the buffer really allocated
if [ "$(census_of device)" = "yes" ] && [ "$(bufs_of device)" = "allocated" ]; then
    pass "LLAMA_MTP_DEV_ROWS=1: device rows (h_nextn device buffer allocated)"
elif [ "$(census_of device)" = "missing" ]; then
    fail "device arm: no census line (the MTP drafter never started)"
else
    fail "LLAMA_MTP_DEV_ROWS=1 did not reach the device path (census '$(census_of device)', buffer '$(bufs_of device)') -- this model keeps the host rows by design (shared memory / chained heads); point MODEL at a single-head non-shared MTP model"
fi

# the two arms must agree on the text and on the acceptance
if [ "$(text_of default)" = "missing" ] || [ "$(text_of device)" = "missing" ]; then
    fail "one of the arms produced no text hash"
elif [ "$(text_of default)" = "$(text_of device)" ]; then
    pass "greedy text byte-identical in both arms (sha=$(text_of default))"
else
    fail "greedy text differs: default=$(text_of default) device=$(text_of device)"
fi

if [ "$(acc_of default)" = "missing" ] || [ "$(acc_of device)" = "missing" ]; then
    fail "one of the arms reported no draft acceptance"
elif [ "$(acc_of default)" = "$(acc_of device)" ]; then
    pass "draft acceptance identical in both arms ($(acc_of default) accepted/generated)"
else
    fail "draft acceptance differs: default=$(acc_of default) device=$(acc_of device) -- the drafter is not being fed the same hidden rows"
fi

# real GPU faults only (the echoed prompt lines are filtered by that script, never grep the log by hand)
if [ "$FAULTS" = 1 ]; then
    if [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
        if "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$SINCE" --log "$TMP/text-default.log" --log "$TMP/text-device.log" > "$TMP/faults.log" 2>&1; then
            pass "no GPU faults in the kernel journal over this gate's window"
        else
            fail "GPU faults reported (see $TMP/faults.log)"
            sed -n '1,20p' "$TMP/faults.log" >&2
        fi
    else
        echo "note: scripts/check-gpu-faults.sh not executable, fault check skipped"
    fi
fi

echo
if [ "$FAIL" = 0 ]; then
    echo "RESULT: PASS"
    exit 0
fi
echo "RESULT: FAIL"
[ "$KEEP" = 1 ] && echo "logs kept in $TMP"
exit 1
