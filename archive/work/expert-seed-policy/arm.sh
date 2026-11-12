#!/usr/bin/env bash
# One arm of the expert-seed measurement (`TODO` #318, `MOE_EXPERT_SEED`).
#
# Measures the four numbers the item asks for -- time-to-first-token, prompt eval, first-N-token decode and
# the bytes read -- per arm, in THREE runs, because one llama-cli run cannot give all of them honestly:
#
#   ttft   `-n 1`   no `-lv`  wall clock from process start to exit (the first token IS the run)
#   wall   `-n N`   no `-lv`  wall clock + the generated-text hash + /proc io (read_bytes/rchar)
#   census `-n N`   `-lv 4`   the timing footer (prompt eval / eval) and the cache census
#
# Why the split: llama.cpp's own timings are `LLAMA_LOG_INFO` (so they need `-lv 4`) and `-lv 4` must never
# touch a text/hash arm (`AGENTS.md`), while a wall clock is what the user actually feels.  run 2 is the text
# arm, run 3 the census arm.
#
# Usage: ./arm.sh NAME [env assignments...] -- [llama-cli args...]
#   e.g. ./arm.sh eager MOE_EXPERT_SEED=eager -- -n 500
# Env: LABEL (subdir), OUT (default: ./runs), plus every variable arm.sh reads below.
set -uo pipefail

NAME="${1:?usage: arm.sh NAME [env...] -- [args...]}"; shift
declare -a ENVS=() ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; ARGS=("$@"); break; fi
    ENVS+=("$1"); shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CLI="${CLI:-$HOME/llama.cpp/build-rocm/bin/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"
GPU="${GPU:-0}"
CTX="${CTX:-4096}"
UB="${UB:-1024}"
NCMOE="${NCMOE:-47}"
PROMPT="${PROMPT:-The capital of France is}"
PROMPT_FILE="${PROMPT_FILE:-}"
NGEN="${NGEN:-500}"
LABEL="${LABEL:-$NAME}"
OUT="${OUT:-$HERE/runs}"
DROP="${DROP:-1}"
MOE_ARGS=()
[ "$NCMOE" != "none" ] && MOE_ARGS=(--n-cpu-moe "$NCMOE")
BASE_ARGS=(-m "$MODEL" --fit off -ngl all "${MOE_ARGS[@]}" --load-mode mmap --lazy-mode on
           -c "$CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -t 8 --jinja --spec-type none
           --temp 0 --seed 42 --single-turn --no-display-prompt)
# `--ignore-eos` so `-n N` is really N tokens (this model answers "Paris" and stops after ~51); without it the
# decode metric is whatever the model chose to emit, which is not comparable across arms.
[ "${IGNORE_EOS:-1}" = "1" ] && BASE_ARGS+=(--ignore-eos)
[ -n "$PROMPT_FILE" ] && PROMPT="$(cat "$PROMPT_FILE")"

DIR="$OUT/$LABEL/$NAME"
mkdir -p "$DIR"
echo "==> $LABEL/$NAME ${ENVS[*]:-} ${ARGS[*]:-}" | tee "$DIR/arm.txt"

drop_caches() {
    [ "$DROP" = "1" ] || return 0
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
}

# /proc/<pid>/io is gone the moment the process exits, so sample it while it runs and keep the LAST value
# (every field is monotonic).  `read_bytes` = storage reads; `rchar` = everything the process read.
run_one() {
    local tag="$1"; shift
    local extra="$1"; shift
    local -a cmd=(env "${ENVS[@]}" HIP_VISIBLE_DEVICES="$GPU" ROCBLAS_USE_HIPBLASLT=0
                  "$CLI" "${BASE_ARGS[@]}" $extra -p "$PROMPT" "${ARGS[@]}" "$@")
    drop_caches
    local t0 t1
    t0=$(date +%s.%N)
    "${cmd[@]}" > "$DIR/$tag.out" 2> "$DIR/$tag.err" &
    local pid=$!
    local rb=0 rc=0 last_rb=0 last_rc=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ -r "/proc/$pid/io" ]; then
            last_rb=$(awk '/^read_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null)
            last_rc=$(awk '/^rchar:/{print $2}' "/proc/$pid/io" 2>/dev/null)
        fi
        sleep 0.2
    done
    wait "$pid"; rc=$?
    t1=$(date +%s.%N)
    printf '%s %s %s %s %s\n' "$(echo "$t1 - $t0" | bc)" "$rc" "${last_rb:-0}" "${last_rc:-0}" "$tag" >> "$DIR/times.raw"
    echo "    $tag: wall $(printf '%.2f' "$(echo "$t1 - $t0" | bc)")s rc=$rc read=$(numfmt --to=iec "${last_rb:-0}") rchar=$(numfmt --to=iec "${last_rc:-0}")"
}

rm -f "$DIR/times.raw"
run_one ttft "" -n 1
run_one wall "" -n "$NGEN"
run_one census "-v" -n "$NGEN"
echo "$DIR" > "$OUT/$LABEL/.lastdir"
