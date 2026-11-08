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
    [ "$warns" -gt 0 ] && echo "    ($warns quant(s) only WARNed on transparency - see above)"
    echo "    logs kept in $KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails quant(s) regressed (logs in $KEEPOUT)" >&2
exit 1
