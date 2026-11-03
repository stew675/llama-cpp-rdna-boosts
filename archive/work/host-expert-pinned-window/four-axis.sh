#!/usr/bin/env bash
# Four-axis adaptive-MTP gate (`benchmarks/mtp-adaptive-methodology.md`, Protocol A) for ONE arm.
#
#   ./four-axis.sh <label>                  # the arm is the caller's environment + HOSTEXP
#   HOSTEXP=window ./four-axis.sh window
#
# Axes: R reasoning, P prose, C code, K recall.  `-n 3000` (rule 0), `--reasoning on` for R and
# `off` for P/C/K, seed 42, temp 0, one run per cell, the MTP head built into the GGUF (no `-md`).
# `--spec-draft-n-max 3` is the fixed-depth arm rule 2 compares against; `none` is the plain arm.
#
# Output: one TSV line per (axis, spec) with t/s, acceptance and mean accepted length, into
# $OUT/<label>.tsv, and the generated text hash for the `none`/`draft-mtp` purity rule (4).
set -u

BIN=${BIN:-$HOME/llama-scratch/build-rocm/bin/llama-cli}
MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
PROMPTS=${PROMPTS:-$HOME/llama-cpp-rdna-boosts/prompts}
OUT=${OUT:-$HOME/llama-cpp-rdna-boosts/wip/host-expert-pinned-window/runs/four-axis}
LABEL=${1:-arm}
N=${N:-3000}
NMAX=${NMAX:-3}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# AUTO_ARENA=1 / AUTO_POOL=1 leave the arena / the pool at the delivery's own AUTO sizing (the real
# scenario); the campaign's pinned 4096/2048 exist for A/B comparability.
if [ -z "${AUTO_ARENA:-}" ]; then export MOE_EXPERT_CACHE_MIB=${MOE_EXPERT_CACHE_MIB:-4096}; fi
if [ -z "${AUTO_POOL:-}"  ]; then export MOE_HOST_POOL_MIB=${MOE_HOST_POOL_MIB:-2048}; fi

mkdir -p "$OUT"
TSV="$OUT/$LABEL.tsv"
: > "$TSV"

for axis in ${AXES:-reasoning prose-rdna-boosts code-python recall}; do
    case "$axis" in reasoning) REA=on ;; *) REA=off ;; esac
    for spec in none draft-mtp; do
        key="$LABEL-${axis%%-*}-$spec"
        extra=()
        [ "$spec" = draft-mtp ] && extra=(--spec-type draft-mtp --spec-draft-n-max "$NMAX")
        timeout 3600 "$BIN" -m "$MODEL" -f "$PROMPTS/$axis.txt" \
            -ngl 99 -ncmoe 40 --host-experts "${HOSTEXP:-pool}" \
            --reasoning "$REA" -n "$N" --seed 42 --temp 0 --single-turn --no-display-prompt \
            -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -lv 4 \
            "${extra[@]}" > "$OUT/$key.out" 2> "$OUT/$key.err"
        tg=$(tr -d '\b\r' < "$OUT/$key.out" | grep -oE "Generation: *[0-9.]+ t/s" | tail -1 | grep -oE "[0-9.]+")
        acc=$(grep -oE "draft acceptance = [0-9.]+" "$OUT/$key.err" | tail -1 | awk '{print $4}')
        ml=$(grep -oE "mean len = *[0-9.]+" "$OUT/$key.err" | tail -1 | grep -oE "[0-9.]+")
        txt=$(python3 "$HOME/llama-cpp-rdna-boosts/scripts/extract-generated.py" "$OUT/$key.out" 2>/dev/null | tr -d '\n')
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${axis%%-*}" "$spec" "${tg:--}" "${acc:--}" "${ml:--}" "${txt:-EXTRACT-FAIL}" \
            | tee -a "$TSV"
    done
done

echo "--- $LABEL (n=$N, nmax=$NMAX, host-experts=${HOSTEXP:-pool})"
awk -F'\t' '{printf "%-10s %-10s %8s t/s  acc=%-9s meanlen=%-6s %s\n",$1,$2,$3,$4,$5,$6}' "$TSV"
