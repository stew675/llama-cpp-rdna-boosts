#!/usr/bin/env bash
# Interleaved ABBA/rotation harness for the flash-next prefill campaign (HANDOVER-COLD-START.md §3A).
#
# The box is period-2 bimodal on this shape (INVARIANTS.md r234) and this campaign's arms are bimodal by
# 7-8 %, so a single run per arm is NOT a result.  This runs every arm once per CYCLE, rotating the order
# each cycle, so a slow-phase drift hits every arm equally and the per-arm medians are comparable.
#
#   usage: ./abba.sh SUITE CYCLES ARM[:ENV=V;ENV=V ...] [ARM ...]
#   e.g.:  ./abba.sh s3-h48 3 \
#            gather \
#            bulk1:GGML_SCHED_DEVGATHER=0;MOE_HOST_POOL_STAGE_BULK=1 \
#            hybrid:GGML_SCHED_DEVGATHER=0 \
#            hcopy:GGML_SCHED_DEVGATHER=0;GGML_SCHED_STAGE_REDIRECT=0
#
# Everything in arm.sh's fixed env applies (heat prior is NOT pinned there -- pass MOE_EXPERT_HEATMAP=load
# in the arm's env or in EXTRAENV below).  Base env for every arm of this campaign:
#   MOE_EXPERT_HEATMAP=load   residency is an input to greedy text (GREEDY_PURITY §89), so an unpinned
#                             A/B compares two different configs.
#   NCMOE=<48>                the campaign's shape (override with NCMOE=47 outside).
# EXTRAENV="A=1 B=2" prefixes extra env for every arm (e.g. EXTRAENV="NCMOE=47").
#
# out: runs/abba-SUITE/<arm>-c<N>/run.log  +  runs/abba-SUITE/summary.tsv
set -uo pipefail
SUITE=${1:?usage: abba.sh SUITE CYCLES ARM[:ENV=V;...] ...}; CYCLES=${2:?cycles}; shift 2
[ $# -gt 0 ] || { echo "no arms given" >&2; exit 2; }
SELF=$(dirname "$(readlink -f "$0")")
ARMS=("$@")
BASE=( MOE_EXPERT_HEATMAP=load NCMOE=${NCMOE:-48} )
if [ -n "${EXTRAENV:-}" ]; then
  # shellcheck disable=SC2206
  BASE+=( ${EXTRAENV} )
fi
OUT="$SELF/runs/abba-$SUITE"
mkdir -p "$OUT"
SUM="$OUT/summary.tsv"
[ -s "$SUM" ] || printf 'suite\tarm\tcycle\tprefill_tps\tgen_tps\twall_s\th2d\torder\n' > "$SUM"

rate() { # file key
  grep -hoE "Prompt: [0-9.]+ t/s \| Generation: [0-9.]+ t/s" "$1" | tail -1 |
    sed -E "s/Prompt: ([0-9.]+) t\/s .* Generation: ([0-9.]+) t\/s/\1 \2/"
}

for c in $(seq 1 "$CYCLES"); do
  n=${#ARMS[@]}
  for i in $(seq 0 $((n - 1))); do
    # rotate the order every cycle so the phase drift is not confounded with arm position
    idx=$(( (i + c - 1) % n ))
    spec=${ARMS[$idx]}
    name=${spec%%:*}
    envs=""
    [ "$spec" != "$name" ] && envs=${spec#*:}
    # shellcheck disable=SC2206
    arm_env=( "${BASE[@]}" )
    if [ -n "$envs" ]; then IFS=';' read -r -a extra <<< "$envs"; arm_env+=( "${extra[@]}" ); fi
    tag="abba-$SUITE/$name-c$c"
    rm -rf "$SELF/runs/$tag"
    t0=$(date +%s)
    RUNS="$SELF/runs/abba-$SUITE" "$SELF/arm.sh" "$name-c$c" "${arm_env[@]}" >/dev/null 2>&1
    t1=$(date +%s)
    log="$SELF/runs/$tag/run.log"
    read -r pf gen < <(rate "$log" 2>/dev/null || echo "0 0")
    h2d=$(grep -hoE "H2D bandwidth calibration: [0-9.]+ GB/s" "$log" | head -1 | grep -oE "[0-9.]+")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n' "$SUITE" "$name" "$c" "${pf:-0}" "${gen:-0}" \
      "$((t1 - t0))" "${h2d:-}" "$i" >> "$SUM"
    printf '  c%s %-14s prefill %8s  gen %5s  wall %3ss  %s\n' "$c" "$name" "${pf:-?}" "${gen:-?}" "$((t1 - t0))" "${envs}"
  done
done

echo
echo "--- medians (n per arm) ---"
awk -F'\t' 'NR>1 { v[$2] = v[$2] " " $4; n[$2]++ }
END {
  for (a in v) {
    m = split(substr(v[a], 2), x, " ");
    for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (x[j] + 0 < x[i] + 0) { t = x[i]; x[i] = x[j]; x[j] = t }
    med = (m % 2) ? x[(m + 1) / 2] : (x[m / 2] + x[m / 2 + 1]) / 2;
    printf "%-14s n=%d  median %8.1f  min %8.1f  max %8.1f  spread %5.1f %%  vals:%s\n",
      a, m, med, x[1], x[m], 100 * (x[m] - x[1]) / med, v[a];
  }
}' "$SUM" | sort -k3 -nr
echo "summary: $SUM"
