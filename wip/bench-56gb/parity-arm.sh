#!/usr/bin/env bash
# One arm of `wip/host-expert-parity/` PLAN.md P0.2/P0.3 on the 56 GiB rig.
#
#   LME=auto SINK=1 DIO=auto ./parity-arm.sh <tag>
#
# env in:
#   LME=47|auto|N      -lme (default 47 = a mirror)
#   SINK=0|1           MOE_HOST_POOL_SINK (unset = the binary default)
#   DIO=auto|0|1       MOE_HOST_POOL_DIO
#   SPAN=0|1           MOE_HOST_POOL_STAGE_SPAN
#   RANK=...           MOE_HOST_POOL_RANK
#   PORT (8033), MEM (56G), CACHE_DROP (1: drop the page cache first)
#
# What it does differently from `measure.sh`: it also runs a SECOND 50k pass, because the first pass pays the
# PLE fault-in (26.8 GiB, `--lazy-mode auto`) and the tier's own miss path at the same time -- the pair is the
# only way to separate them -- and it samples the server's `/proc/<pid>/io read_bytes` after each stage, which
# is the "storage MiB" of the crossover table (a page-cache read is not a storage read).
set -uo pipefail
TAG=${1:?usage: LME=... ./parity-arm.sh <tag>}
PORT=${PORT:-8033}
MEM=${MEM:-56G}
DIR=${DIR:-$PWD/runs/$TAG}
OUT=$DIR/parity.txt
UNIT=bench-$TAG
CG=/sys/fs/cgroup/system.slice/$UNIT.scope
mkdir -p "$DIR"
export MEM

rbytes() { timeout 5 sudo -n awk '/^read_bytes/{print $2}' "/proc/$1/io" 2>/dev/null || echo 0; }
# stdin -> one tagged row.  NOT a heredoc: `python3 - <<PY` REPLACES stdin, which silently ate every t/s line
# (the arm still produced its storage/census columns, so the loss looked like a quiet benchmark).
TAGROW='import json,sys
tag=sys.argv[1]
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("{"): continue
    d=json.loads(line); t=d["timings"] if "timings" in d else d
    print("%s prompt_n=%s prompt_tps=%s pred_tps=%s wall_s=%s" % (tag, t.get("prompt_n"), t.get("prompt_tps", t.get("prompt_per_second")), t.get("pred_tps", t.get("predicted_per_second")), d.get("wall_s")))'
tps() { python3 -c "$TAGROW" "$@"; }

echo "=== $TAG  LME=${LME:-47} SINK=${SINK:-default} DIO=${DIO:-auto} SPAN=${SPAN:-default} RANK=${RANK:-default} FILLSRC=${FILLSRC:-default} DEMANDSRC=${DEMANDSRC:-default}  $(date -Is) ===" | tee -a "$OUT"
if [ "${CACHE_DROP:-1}" = "1" ]; then
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
fi

# NOT piped: run.sh leaves a background memory-sampling subshell holding stdout, so a pipe never closes.
# The knobs go as ARGUMENTS: sudo resets the environment inside the scope, so an exported LME never arrives.
awk '/MemAvailable/{printf "MemAvailable at arm start: %.2f GiB\n", $2/1048576}' /proc/meminfo | tee -a "$OUT"
./run.sh "$TAG" ./runme-56gb-tier.sh "${LME:-}" "${SINK:-}" "${DIO:-}" "${SPAN:-}" "${RANK:-}" "${HEATMAP:-load}" "${SERVER:-}" "${FILLSRC:-}" "${DEMANDSRC:-}" >> "$OUT" 2>&1
# bounded: a launch script that fails (a missing binary in the root scope, a refused -lme) would otherwise
# block the health poll for an hour and look like a hang.
if ! timeout "${HEALTH_TIMEOUT:-900}" python3 bench.py "$PORT" health | tee -a "$OUT"; then
  echo "!! $TAG: server never became healthy -- tail of server.log:" | tee -a "$OUT"
  tail -5 "$DIR/server.log" | tee -a "$OUT"
  timeout 20 sudo systemctl stop "$UNIT.scope" >/dev/null 2>&1
  exit 1
fi
# a scope's MainPID is not readable by the user (and `systemctl show` prints nothing for it here): the
# server IS the scope's only process, because the launch script `exec`s it.
PID=$(timeout 5 sudo -n head -1 "$CG/cgroup.procs" 2>/dev/null)
[ -z "$PID" ] && PID=$(pgrep -f "llama-server .*--port $PORT" | head -1)
echo "pid=$PID" | tee -a "$OUT"
[ -z "$PID" ] && { echo "!! $TAG: no server pid -- the io/storage column is unavailable" | tee -a "$OUT"; PID=0; }

# Verification that the arm is the arm: read the plan back out of the server's own log.  A knob that did not
# arrive (the sudo/environment trap above) shows up here as the default tier instead of passing silently.
COV=$(grep -oE "role=blk\.[0-9]+\.ffn_gate_exps\.weight -> [0-9]+/[0-9]+ slots" "$DIR/server.log" | head -1 | grep -oE "[0-9]+/[0-9]+")
echo "applied tier: slots=${COV:-UNKNOWN} (expect 512/512 mirror, ~500/512 auto, 480/512 at -lme 44, 393/512 at -lme 36)" | tee -a "$OUT"

NGEN=${NGEN:-600}
NPASS=${NPASS:-2}
io_prev=$(rbytes "$PID")
for i in 1 2 3; do
  python3 bench.py "$PORT" gen prompt_short.txt "$NGEN" | tps "gen$i" | tee -a "$OUT"
done
io_now=$(rbytes "$PID")
echo "storage_reads_MiB: gen=$(( (io_now - io_prev) / 1048576 ))" | tee -a "$OUT"
io_prev=$io_now
for p in $(seq 1 "$NPASS"); do
  lbl=pp50k_pass$p
  python3 bench.py "$PORT" gen prompt_50k.txt 1 | tps "$lbl" | tee -a "$OUT"
  io_now=$(rbytes "$PID")
  echo "storage_reads_MiB: $lbl=$(( (io_now - io_prev) / 1048576 ))" | tee -a "$OUT"
  io_prev=$io_now
done

echo "peak_bytes=$(timeout 5 sudo -n cat "$CG/memory.peak" 2>/dev/null)" | tee -a "$OUT"
timeout 5 sudo -n grep -E "^(anon|file) " "$CG/memory.stat" 2>/dev/null | tee -a "$OUT"
timeout 20 sudo systemctl stop "$UNIT.scope" >/dev/null 2>&1
sleep 1

{
  echo "-- census --"
  grep -E "host pool \(L2\):|sink \(MOE_HOST_POOL_SINK\):|sink reuse|sink headroom|sink reclaimed|pool fills:|coalesced prefill|per-expert storage|per-expert mapping|device gather|scheduler pageable" "$DIR/server.log" | tail -20
  echo "-- timings (from server.log, the redundant copy) --"
  grep -E "print_timing:.*(prompt eval time|eval time)" "$DIR/server.log" | tail -12 | sed -E 's/^[0-9.]+ I slot //'
  echo "-- placement --"
  grep -E "host pool \(MOE_HOST_POOL" "$DIR/server.log" | tail -6
  grep -E "host pool .* -> " "$DIR/server.log" | tail -4
} | tee -a "$OUT"
echo "=== $TAG done $(date -Is) ===" | tee -a "$OUT"
