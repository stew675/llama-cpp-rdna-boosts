#!/usr/bin/env bash
# storage-oracle.sh -- point it at a model file, get back a markdown report that says which llama.cpp
# load/residency configuration to use on THAT file on THAT device, and -- more usefully -- which
# configurations to avoid, each with the measurement that condemns it.
#
#   ./storage-oracle.sh --report out.md /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
#   ./storage-oracle.sh --verify --verify-args "-ngl 99 -sm tensor" MODEL.gguf
#
# Design rules:
#   * MEASURE, NEVER ASSUME.  Every verdict cell names the probe and the number behind it.  A probe that
#     could not be run says UNKNOWN; it never guesses.
#   * COLD BY CONSTRUCTION, NO ROOT, NO drop_caches (unless --cold is passed explicitly for --verify).
#     O_DIRECT is cold by nature; the page-cache arms evict and VERIFY with mincore (see storage-probe.c).
#   * BOUNDED I/O.  The run prints its I/O budget first and never exceeds it.
#   * STORAGE ONLY.  It knows nothing about VRAM fit, tensor splits or arch; it says so in the report.
#
# Requires: bash, cc, findmnt/lsblk (util-linux), filefrag (e2fsprogs), awk, lspci (optional).
# Linux only (POSIX I/O paths; this repo is RDNA/ROCm).

set -uo pipefail

VERSION=1

# ---- decision thresholds (the whole decision layer; each is justified in README.md) --------------
T_DIO_BEATS=1.30     # O_DIRECT must beat the mapped fault path by this factor to be the cold load mode
T_SAME=1.15          # a mode within this factor of the best is ACCEPTABLE, not recommended
T_SPREAD=0.50        # worst band below this fraction of the best band => placement warning
T_WAKE_US=1000       # idle-gap wake above this => on-demand reads are unsafe
T_STREAM_P90_US=500  # 4 KiB p90 under a stream above this => on-demand reads are unsafe
T_STREAM_P99_US=1000 # ... and the same for p99
T_CACHE_FIT=0.80     # a resident plan must need <= this fraction of MemAvailable
T_VERIFY_MATCH=0.20  # prediction "held" if measured is within this fraction of predicted
T_FRAG_EXTENTS=256   # extent count above which a file is called fragmented
T_WARM_GAIN=2.0      # warm-cache rate / best cold rate above which dio is a tradeoff, not a free win

say()  { printf '  %s\n' "$*" >&2; }
die()  { printf 'storage-oracle: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<'EOF'
usage: storage-oracle.sh [options] MODEL [MODEL...]

Writes a markdown report: which load/residency configuration to use for these model files on the
device(s) they live on, and which configurations to avoid.  PROBES ONLY -- it changes nothing.

options:
  --report PATH        report path (default: ./storage-oracle-<host>-<UTCSTAMP>.md)
  --resident-gib N     RAM this model needs BESIDES its files (KV cache, pinned pools, compute
                       buffers) -- used only to decide whether "keep it resident" plans are viable  [8]
  --budget-gib N       maximum I/O the probes may read, in GiB                                 [24]
  --quick              smaller probe sizes (faster, coarser); implied when --budget-gib < 12
  --no-io              facts and decision rules only; every measurement becomes UNKNOWN
  --verify             also run llama-cli head-to-head (current default vs the recommendation)
  --cold               (with --verify) drop the page cache before each arm -- needs root
  --binary PATH        llama-cli to use for --verify
  --verify-args "..."  extra llama-cli args for --verify, e.g. "-ngl 99 -sm tensor"   [default "-ngl 0"]
  --verify-n N         tokens to generate per verify arm (load time is what is measured)        [1]
  --probe PATH         use an already-built storage-probe
  -h, --help           this text

environment:
  TMPDIR               where the probe is built (default /tmp)
  PROBE_TIMEOUT        per-probe timeout in seconds (default 600)
EOF
}

# ---- small helpers -------------------------------------------------------------------------------
# pv LINE KEY -> the token after KEY= in a "PROBE ... key=value ..." line
pv() { grep -oE "(^|[[:space:]])$2=[^[:space:]]+" <<<"$1" | head -1 | sed -E "s/.*[[:space:]]?$2=//"; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 > b+0) }'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 >= b+0) }'; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 < b+0) }'; }
fm() { awk -v v="$1" -v f="${2:-2}" 'BEGIN{ printf "%.*f", f, v }'; }
gib() { awk -v b="$1" 'BEGIN{ printf "%.1f", b/1073741824 }'; }

# ---- options -------------------------------------------------------------------------------------
REPORT=""
RESIDENT_GIB=8
BUDGET_GIB=24
QUICK=0
NO_IO=0
VERIFY=0
COLD=0
BINARY=""
VERIFY_ARGS="-ngl 0"
VERIFY_N=1
PROBE_BIN=""
MODELS=()

while [ $# -gt 0 ]; do
    case $1 in
        --report)        REPORT=${2:?--report needs a path}; shift 2 ;;
        --resident-gib)  RESIDENT_GIB=${2:?}; shift 2 ;;
        --budget-gib)    BUDGET_GIB=${2:?}; shift 2 ;;
        --binary)        BINARY=${2:?}; shift 2 ;;
        --verify-args)   VERIFY_ARGS=${2-}; shift 2 ;;
        --verify-n)      VERIFY_N=${2:?}; shift 2 ;;
        --probe)         PROBE_BIN=${2:?}; shift 2 ;;
        --quick)         QUICK=1; shift ;;
        --no-io)         NO_IO=1; shift ;;
        --verify)        VERIFY=1; shift ;;
        --cold)          COLD=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; while [ $# -gt 0 ]; do MODELS+=("$1"); shift; done ;;
        -*)              die "unknown option '$1' (--help)" ;;
        *)               MODELS+=("$1"); shift ;;
    esac
done

[ ${#MODELS[@]} -gt 0 ] || { usage; exit 2; }
case " $VERIFY_ARGS " in
    *" -c "*|*" --ctx-size "*) ;;
    *) VERIFY_ARGS="$VERIFY_ARGS -c 4096" ;;   # the default context allocates a KV cache that dwarfs the load
esac
[ "$BUDGET_GIB" -lt 12 ] && QUICK=1

for m in "${MODELS[@]}"; do
    [ -f "$m" ] || die "no such file: $m"
    [ -r "$m" ] || die "cannot read: $m"
done

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST=$(hostname -s 2>/dev/null || echo unknown)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT=${REPORT:-./storage-oracle-${HOST}-${STAMP}.md}

# ---- build the probe -----------------------------------------------------------------------------
build_probe() {
    [ -n "$PROBE_BIN" ] && { [ -x "$PROBE_BIN" ] || die "--probe '$PROBE_BIN' is not executable"; return; }
    have cc || die "cc not found (needed to build storage-probe.c)"
    local src=$SELF_DIR/storage-probe.c
    [ -f "$src" ] || die "storage-probe.c not found next to this script"
    PROBE_DIR=${TMPDIR:-/tmp}/storage-oracle.$(id -u)
    mkdir -p "$PROBE_DIR" || die "cannot create $PROBE_DIR"
    PROBE_BIN=$PROBE_DIR/storage-probe
    if [ ! -x "$PROBE_BIN" ] || [ "$src" -nt "$PROBE_BIN" ]; then
        say "building probe -> $PROBE_BIN"
        cc -O2 -o "$PROBE_BIN" "$src" 2>"$PROBE_DIR/build.log" || {
            sed 's/^/    /' "$PROBE_DIR/build.log" >&2; die "probe build failed"; }
    fi
}
probe() { timeout "${PROBE_TIMEOUT:-600}" "$PROBE_BIN" "$@"; }

# ---- facts ---------------------------------------------------------------------------------------
# findmnt FIELD FILE -> one field
mnt() { findmnt -no "$1" -T "$2" 2>/dev/null | head -1; }

sum_bytes=0
primary=""
MODEL_FILES=()
for m in "${MODELS[@]}"; do
    sz=$(stat -c %s "$m")
    sum_bytes=$((sum_bytes + sz))
    [ -z "$primary" ] && primary=$m
done

# expand split shards: probing all of them is pointless (they are the same file on the same device),
# but the resident-plan arithmetic needs their total size
for m in "${MODELS[@]}"; do
    base=$(basename "$m")
    if [[ $base =~ ^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        dir=$(dirname "$m"); stem=${BASH_REMATCH[1]}; pat="$dir/$stem-[0-9][0-9][0-9][0-9][0-9]-of-${BASH_REMATCH[3]}.gguf"
        n=0; sz=0
        for s in $pat; do [ -f "$s" ] || continue; n=$((n+1)); sz=$((sz + $(stat -c %s "$s"))); done
        [ "$n" -gt 1 ] && say "split model: $n shards, $(gib "$sz") GiB total" && SHARD_COUNT=$n
        sum_bytes=$sz
    fi
done

[ ${#MODEL_FILES[@]} -eq 0 ] && MODEL_FILES=("${MODELS[@]}")   # not a split set: evict the given files
PSIZE=$(getconf PAGESIZE 2>/dev/null || echo 4096)
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
MEMLOCK_KB=$(ulimit -l 2>/dev/null || echo 0)
[ "$MEMLOCK_KB" = "unlimited" ] && MEMLOCK_KB=-1
NEED_MIB=$(awk -v f="$sum_bytes" -v r="$RESIDENT_GIB" 'BEGIN{ printf "%.0f", f/1048576 + r*1024 }')
MEM_AVAIL_MIB=$(awk -v k="$MEM_AVAIL_KB" 'BEGIN{ printf "%.0f", k/1024 }')
MEMLOCK_MIB=$(awk -v k="$MEMLOCK_KB" 'BEGIN{ if (k < 0) print -1; else printf "%.0f", k/1024 }')

FP=$primary
F_SIZE=$(stat -c %s "$FP")
F_GIB=$(gib "$F_SIZE")
FS_TYPE=$(mnt FSTYPE "$FP"); FS_TYPE=${FS_TYPE:-unknown}
FS_SRC=$(mnt SOURCE "$FP");  FS_SRC=${FS_SRC:-?}
FS_TARGET=$(mnt TARGET "$FP"); FS_TARGET=${FS_TARGET:-?}
FS_OPTS=$(mnt OPTIONS "$FP");  FS_OPTS=${FS_OPTS:-?}
FS_FREE=$(df -P -B1 "$(dirname "$FP")" 2>/dev/null | awk 'NR==2{print $4}')
FS_FREE_GIB=$( [ -n "${FS_FREE:-}" ] && gib "$FS_FREE" || echo "?")
BLK=""
if [ -b "$FS_SRC" ] 2>/dev/null; then
    BLK=$(lsblk -no PKNAME "$FS_SRC" 2>/dev/null | head -1)
    [ -z "$BLK" ] && BLK=${FS_SRC##*/}
elif [[ $FS_SRC == /dev/* ]]; then
    BLK=$(basename "$FS_SRC")
fi
BLK=${BLK:-unknown}
BLK_DIR=/sys/block/$BLK
if [ -d "$BLK_DIR" ]; then
    DEV_MODEL=$(cat "$BLK_DIR/device/model" 2>/dev/null | tr -s ' ' | sed 's/ *$//')
    DEV_FW=$(cat "$BLK_DIR/device/firmware_rev" 2>/dev/null)
    Q_SCHED=$(cat "$BLK_DIR/queue/scheduler" 2>/dev/null | sed -E 's/.*\[([^]]+)\].*/\1/')
    Q_ROTA=$(cat "$BLK_DIR/queue/rotational" 2>/dev/null)
    Q_MAXKB=$(cat "$BLK_DIR/queue/max_sectors_kb" 2>/dev/null)
    Q_RA=$(cat "$BLK_DIR/queue/read_ahead_kb" 2>/dev/null)
    PCI=$(basename "$(readlink -f "$BLK_DIR/device/device")" 2>/dev/null)
else
    DEV_MODEL=""; DEV_FW=""; Q_SCHED=""; Q_ROTA=""; Q_MAXKB=""; Q_RA=""; PCI=""
fi
PCIE_LINK=""
PCIE_PORT=""
if [ -n "${PCI:-}" ] && have lspci; then
    spd=$(lspci -vv -s "$PCI" 2>/dev/null | sed -n 's/.*LnkSta:[[:space:]]*Speed \([^,]*\), Width \(x[0-9]*\).*/\1 \2/p' | head -1)
    if [ -n "$spd" ]; then PCIE_LINK="$spd"
    else PCIE_LINK="unknown (lspci -vv needs root for LnkSta)"; fi
    PCIE_PORT=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$PCI" 2>/dev/null)")" 2>/dev/null)
fi
# who else hangs off the same upstream port (a chipset-hung drive shares the uplink)
PCI_SHARED=""
if [ -n "${PCIE_PORT:-}" ] && have lspci; then
    PCI_SHARED=$(lspci -s "$PCIE_PORT" -t 2>/dev/null | tr '|' '\n' | grep -c -- '--')
fi

# The idle-gap ladder below is a property of the drive AND of the power policy in force: APST on this
# box costs 16 ms per cold read, off it costs ~5 ms.  Record what produced the numbers.
NVME_CTRL=$(basename "$(readlink -f "/sys/block/${BLK}/device" 2>/dev/null)" 2>/dev/null)
APST_PARAM=$(cat /sys/module/nvme_core/parameters/default_ps_max_latency_us 2>/dev/null || echo "")
APST_LINE=""
if [ -n "${NVME_CTRL:-}" ] && have nvme; then
    apst_out=$(nvme get-feature "/dev/$NVME_CTRL" -f 0x0c -H 2>/dev/null || true)
    case $apst_out in
        *"(APSTE): Disabled"*) APST_LINE="disabled on $NVME_CTRL" ;;
        *"(APSTE): Enabled"*)
            itpt=$(printf '%s\n' "$apst_out" | sed -nE 's/.*\(ITPT\): *([0-9]+) ms.*/\1/p' | head -1)
            itps=$(printf '%s\n' "$apst_out" | sed -nE 's/.*\(ITPS\): *([0-9]+).*/\1/p' | head -1)
            APST_LINE="enabled on $NVME_CTRL: idle ${itpt:-?} ms -> PS${itps:-?}" ;;
    esac
fi
if [ -n "$APST_LINE" ]; then
    APST_NOTE="$APST_LINE (nvme_core.default_ps_max_latency_us=${APST_PARAM:-?})"
elif [ -n "${APST_PARAM:-}" ]; then
    APST_NOTE="nvme_core.default_ps_max_latency_us=$APST_PARAM (applied at controller creation; the live table needs root to read)"
else
    APST_NOTE="unknown"
fi

if have filefrag; then
    F_EXTENTS=$(filefrag "$FP" 2>/dev/null | sed -nE 's/.*: ([0-9]+) extents? found.*/\1/p' | head -1)
else
    F_EXTENTS=""
fi
F_EXTENTS=${F_EXTENTS:-?}
F_BLKSIZE=$(stat -c %o "$FP")

# ---- per-file probe plan -------------------------------------------------------------------------
if [ "$QUICK" = 1 ]; then
    MAP_CHUNK=256; MAP_STEPS=6; R_DIO_N=1500; R_BUF_N=600; R_MMAP_N=400; R_1M_N=120
    BULK_COUNT=0; POP_MIB=256; STREAM_COUNT=256
else
    MAP_CHUNK=1024; MAP_STEPS=8; R_DIO_N=3000; R_BUF_N=1000; R_MMAP_N=600; R_1M_N=200
    BULK_COUNT=0; POP_MIB=1024; STREAM_COUNT=768
fi
# never map more than the file holds, never use a chunk below 64 MiB, and shrink the step count
# rather than the chunk when the file is small
size_mib=$(( F_SIZE / 1048576 ))
[ "$MAP_STEPS" -gt $(( size_mib / 64 )) ] && MAP_STEPS=$(( size_mib / 64 ))
[ "$MAP_STEPS" -lt 1 ] && MAP_STEPS=1
map_max=$(( size_mib / MAP_STEPS ))
[ "$map_max" -lt "$MAP_CHUNK" ] && MAP_CHUNK=$map_max
[ "$MAP_CHUNK" -lt 1 ] && MAP_CHUNK=1

# probe sizes are clamped to the file, then to the budget: never read more than the operator allowed
compute_plan() {
    size_mib=$(( F_SIZE / 1048576 ))
    [ "$MAP_STEPS" -gt $(( size_mib / 64 )) ] && MAP_STEPS=$(( size_mib / 64 ))
    [ "$MAP_STEPS" -lt 1 ] && MAP_STEPS=1
    map_max=$(( size_mib / MAP_STEPS ))
    [ "$map_max" -lt "$MAP_CHUNK" ] && MAP_CHUNK=$map_max
    [ "$MAP_CHUNK" -lt 1 ] && MAP_CHUNK=1
    # the sequential arms read one band, so the rate they report belongs to the region the map
    # measured -- otherwise a "best band" verdict and a sequential number come from different places
    BULK_COUNT=$(( MAP_CHUNK / 4 ))
    [ "$BULK_COUNT" -gt 1024 ] && BULK_COUNT=1024
    [ "$BULK_COUNT" -lt 16 ] && BULK_COUNT=16
    PLAN_GIB=$(( (MAP_CHUNK * MAP_STEPS + 3 * BULK_COUNT * 4 + POP_MIB + STREAM_COUNT * 4 + 256) / 1024 ))
}
compute_plan
[ "$NO_IO" = 0 ] && say "I/O plan: about ${PLAN_GIB} GiB of reads (budget ${BUDGET_GIB} GiB)"
if [ "$PLAN_GIB" -gt "$BUDGET_GIB" ] && [ "$QUICK" = 0 ]; then
    say "over budget -> falling back to --quick parameters"
    QUICK=1; MAP_CHUNK=256; MAP_STEPS=6; R_DIO_N=1500; R_BUF_N=600; R_MMAP_N=400; R_1M_N=120
    POP_MIB=256; STREAM_COUNT=256
    compute_plan
fi
while [ "$PLAN_GIB" -gt "$BUDGET_GIB" ] && [ "$MAP_STEPS" -gt 2 ]; do
    MAP_STEPS=$(( MAP_STEPS / 2 ))
    compute_plan
done
[ "$NO_IO" = 0 ] && [ "$PLAN_GIB" -gt "$BUDGET_GIB" ] && die "the smallest useful probe plan is ${PLAN_GIB} GiB; raise --budget-gib or use --no-io"
[ "$NO_IO" = 0 ] && say "plan: ${MAP_STEPS} x ${MAP_CHUNK} MiB bands, ${BULK_COUNT} x 4 MiB sequential, ${POP_MIB} MiB fault-in, ${STREAM_COUNT} x 4 MiB stream"


# ---- run the probes -------------------------------------------------------------------------------
probe_all() {
    local f=$1
    local maxoff_mib=$(( F_SIZE / 1048576 ))

    CACHED_BEFORE=$(probe cached "$f" 2>/dev/null) || CACHED_BEFORE=""
    say "band map (${MAP_STEPS} x ${MAP_CHUNK} MiB, O_DIRECT)"
    while IFS= read -r line; do
        [ -n "$line" ] && MAP_LINES+=("$line")
    done < <(probe map "$f" "$MAP_CHUNK" "$MAP_STEPS" 0 2>/dev/null)

    # best/worst band, ignoring chunks that are unwritten (zero-filled: the controller answers those
    # from its zero-fill path at bus speed and they say nothing about the drive)
    local l l_off l_gbs l_zero
    for l in ${MAP_LINES[@]+"${MAP_LINES[@]}"}; do
        [ -n "$l" ] || continue
        l_zero=$(pv "$l" zeros); l_off=$(pv "$l" off_gib); l_gbs=$(pv "$l" gbs)
        [ -z "$l_gbs" ] && continue
        if [ "${l_zero:-0}" = 1 ]; then MAP_ZERO=$((MAP_ZERO + 1)); continue; fi
        if gt "$l_gbs" "$MAP_BEST_GBS"; then
            MAP_BEST_GBS=$l_gbs
            MAP_BEST_OFF=$(awk -v g="$l_off" 'BEGIN{printf "%d", g*1024}')
        fi
        if [ "$MAP_WORST_GBS" = 0 ] || lt "$l_gbs" "$MAP_WORST_GBS"; then
            MAP_WORST_GBS=$l_gbs
            MAP_WORST_OFF=$(awk -v g="$l_off" 'BEGIN{printf "%d", g*1024}')
        fi
    done
    say "best band: ${MAP_BEST_GBS:-?} GB/s at ${MAP_BEST_OFF} MiB | worst: ${MAP_WORST_GBS:-?} GB/s at ${MAP_WORST_OFF} MiB"

    local base=$MAP_BEST_OFF
    local region=1024
    if [ $((base + region)) -gt "$maxoff_mib" ]; then region=$((maxoff_mib - base)); fi
    [ "$region" -lt 64 ] && { base=0; region=$maxoff_mib; }
    [ "$region" -gt 1024 ] && region=1024
    PROBE_BASE=$base
    PROBE_REGION=$region

    say "cold 4 KiB latency, three mechanisms at the best band"
    L_DIO=$(probe rand "$f" dio  4096 "$R_DIO_N"  "$region" "$base" 2>/dev/null) || L_DIO=""
    L_BUF=$(probe rand "$f" buf  4096 "$R_BUF_N"  "$region" "$base" 2>/dev/null) || L_BUF=""
    L_MMAP=$(probe rand "$f" mmap 4096 "$R_MMAP_N" "$region" "$base" 2>/dev/null) || L_MMAP=""
    say "cold 1 MiB latency"
    L_1M=$(probe rand "$f" dio 1048576 "$R_1M_N" "$region" "$base" 2>/dev/null) || L_1M=""

    say "sequential throughput (O_DIRECT, buffered after eviction, buffered warm, mmap fault-in)"
    B_DIO=$(probe bulk  "$f" dio 4 "$BULK_COUNT" "$base" 2>/dev/null) || B_DIO=""
    EVICT=$(probe evict "$f" "$BULK_COUNT" "$base" 2>/dev/null) || EVICT=""
    CACHED_ARM=$(probe cached "$f" 2>/dev/null) || CACHED_ARM=""
    B_BUF=$(probe bulk  "$f" buf 1 "$((BULK_COUNT * 4))" "$base" 2>/dev/null) || B_BUF=""
    B_WARM=$(probe bulk "$f" buf 1 "$((BULK_COUNT * 4))" "$base" 2>/dev/null) || B_WARM=""
    POP=$(probe pop "$f" "$POP_MIB" "$base" 2>/dev/null) || POP=""

    say "idle-gap ladder (how expensive is a read after a pause)"
    G5=$(probe rand    "$f" dio 4096 20 512 "$base" 5    2>/dev/null) || G5=""
    G50=$(probe rand   "$f" dio 4096 20 512 "$base" 50   2>/dev/null) || G50=""
    G250=$(probe rand  "$f" dio 4096 20 512 "$base" 250  2>/dev/null) || G250=""
    G1000=$(probe rand "$f" dio 4096 20 512 "$base" 1000 2>/dev/null) || G1000=""

    say "cold 4 KiB latency WHILE the same file is being streamed"
    local sbase=$MAP_WORST_OFF sregion=256
    [ $((sbase + sregion)) -gt "$maxoff_mib" ] && sregion=$((maxoff_mib - sbase))
    [ "$sregion" -lt 64 ] && { sbase=$base; sregion=$region; }
    probe bulk "$f" dio 4 "$STREAM_COUNT" "$sbase" >/dev/null 2>&1 &
    local spid=$!
    sleep 0.3
    STR=$(probe rand "$f" dio 4096 300 256 "$base" 2>/dev/null) || STR=""
    wait "$spid" 2>/dev/null

    CACHED_AFTER=$(probe cached "$f" 2>/dev/null) || CACHED_AFTER=""

    # a sample the probe could not make cold was skipped, never timed; surface it
    for l in "$L_DIO" "$L_BUF" "$L_MMAP" "$L_1M"; do
        [ -n "$l" ] || continue
        sw=$(pv "$l" skipped_warm); [ -n "$sw" ] && SKIPPED=$((SKIPPED + sw))
    done
}


# ---- verification --------------------------------------------------------------------------------
# bytes this device has read so far (MiB): the objective half of the verification
dev_read_mib() {
    local f=/sys/block/${BLK}/stat
    [ -r "$f" ] || { echo ""; return; }
    awk '{printf "%.0f", $3*512/1048576}' "$f"
}

V_ROWS=(); V_NOTE=""; V_BIN=""; V_FAILED=0; V_ERRS=(); LAZY_TABLES=""; LAZY_PIN=""

find_binary() {
    [ -n "$BINARY" ] && { [ -x "$BINARY" ] && { V_BIN=$BINARY; return; }; V_NOTE="--binary '$BINARY' is not executable"; return; }
    local c
    for c in ./build/bin/llama-cli "$HOME/llama.cpp/build-rocm/bin/llama-cli" \
             "$HOME/llama.cpp/build/bin/llama-cli"; do
        [ -x "$c" ] && { V_BIN=$c; return; }
    done
    c=$(command -v llama-cli 2>/dev/null) && [ -n "$c" ] && { V_BIN=$c; return; }
    V_NOTE="no llama-cli found (use --binary)"
}

# run one arm: label cold|keep, extra args...
verify_arm() {
    local label=$1 arm_mode=$2; shift 2
    local cached_pre wall t0 t1 out mib0 mib1 mib eff rc=0 f
    if [ "$arm_mode" = keep ]; then
        # deliberately reuse the cache: this is the restart arm, and the number it reports is the
        # page cache's, not the drive's
        cached_pre=$(probe cached "$FP" 2>/dev/null); cached_pre=$(pv "$cached_pre" pct); cached_pre=${cached_pre:-?}
    elif [ "$COLD" = 1 ]; then
        if [ "$(id -u)" = 0 ]; then
            sync; echo 1 > /proc/sys/vm/drop_caches; cached_pre=0
        else
            say "--cold needs root; skipping the cache drop"
            cached_pre=$(probe cached "$FP" 2>/dev/null); cached_pre=$(pv "$cached_pre" pct); cached_pre=${cached_pre:-?}
        fi
    else
        # both arms must start from the same cache state, or the first arm silently benchmarks the
        # second one.  Evict the whole model set, shard by shard.
        for f in ${MODEL_FILES[@]+"${MODEL_FILES[@]}"}; do
            probe evict "$f" $(( $(stat -c %s "$f") / 1048576 )) >/dev/null 2>&1
        done
        cached_pre=$(probe cached "$FP" 2>/dev/null); cached_pre=$(pv "$cached_pre" pct); cached_pre=${cached_pre:-?}
    fi
    mib0=$(dev_read_mib)
    say "verify: $label"
    t0=$(date +%s.%N)
    # --single-turn is mandatory: without it llama-cli drops into chat mode and never exits
    # shellcheck disable=SC2086 -- VERIFY_ARGS is a deliberate word-split flag list
    out=$(timeout 1800 "$V_BIN" -m "$FP" -p hi -n "$VERIFY_N" --single-turn --no-display-prompt \
              --no-warmup -lv 4 $VERIFY_ARGS "$@" 2>&1) || rc=$?
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    mib1=$(dev_read_mib)

    # what the lazy pin actually did (r230: `--lazy-mode mlock` pins ONLY the TENSOR_READ_LAZY tables,
    # so on a model without one the mode is inert and the restart win is the ordinary page cache)
    local pin="" is_mlock=0
    case " $* " in *" mlock "*) is_mlock=1 ;; esac
    pin=$(printf '%s\n' "$out" | grep -oE 'pinned [0-9]+ table\(s\), [0-9]+ MiB' | head -1)
    if [ -n "$pin" ]; then
        LAZY_PIN=$pin
        LAZY_TABLES=$(awk '{print $2}' <<<"$pin")
    elif [ "$is_mlock" = 1 ] && [ -z "${LAZY_TABLES:-}" ]; then
        LAZY_TABLES=0
    fi

    # An arm that did not load and generate is NOT a fast arm: a failed load reads ~10 MiB in under a
    # second, which looks like a spectacular result.  That mistake was made once, in this tool's own
    # example report, before this check existed.
    if [ "$rc" -ne 0 ] || ! printf '%s\n' "$out" | grep -q 'prompt eval time'; then
        local err
        err=$(printf '%s\n' "$out" | grep -E '^[0-9.]+ E ' | head -1 | cut -c1-150)
        [ -z "$err" ] && err=$(printf '%s\n' "$out" | grep -iE 'error|failed' | tail -1 | cut -c1-150)
        [ -z "$err" ] && err="exit status $rc, no prompt-eval line"
        V_FAILED=1
        V_ROWS+=("$label${US}FAILED${US}$wall${US}n/a${US}n/a${US}$cached_pre${US}$*")
        V_ERRS+=("$label: $err")
        say "  arm FAILED -- $err"
        return
    fi

    if [ -n "$mib0" ] && [ -n "$mib1" ]; then
        mib=$(awk -v a="$mib0" -v b="$mib1" 'BEGIN{printf "%.0f", b-a}')
        eff=$(awk -v m="$mib" -v w="$wall" 'BEGIN{ if (w+0<=0||m+0<=0) print "n/a"; else printf "%.2f", (m/1024)/w }')
    else
        mib="n/a"; eff="n/a"
    fi
    V_ROWS+=("$label${US}OK${US}$wall${US}$mib${US}$eff${US}$cached_pre${US}$*")
}

run_verify() {
    find_binary
    [ -z "$V_BIN" ] && { V_NOTE=${V_NOTE:-"no llama-cli"}; return; }
    say "verify with $V_BIN"
    verify_arm "shipped default (no --load-mode)" cold
    if [ -n "$REC_LOAD_MODE" ] && [ "$REC_LOAD_MODE" != "auto" ]; then
        verify_arm "recommended (--load-mode $REC_LOAD_MODE --lazy-mode $REC_LAZY_MODE), cold" cold \
            "--load-mode" "$REC_LOAD_MODE" "--lazy-mode" "$REC_LAZY_MODE"
        verify_arm "recommended, WARM page cache (a restart)" keep \
            "--load-mode" "$REC_LOAD_MODE" "--lazy-mode" "$REC_LAZY_MODE"
    else
        V_NOTE="the recommendation is the shipped default; there is nothing to compare"
    fi
}

# ---- state the probes fill in (declared here so --no-io and a failed probe still produce a report) --
MAP_LINES=()
MAP_BEST_OFF=0; MAP_BEST_GBS=0; MAP_WORST_GBS=0; MAP_WORST_OFF=0; MAP_ZERO=0
PROBE_BASE=0;  PROBE_REGION=1024
CACHED_BEFORE=""; CACHED_AFTER=""; CACHED_ARM=""; EVICT=""
L_DIO=""; L_BUF=""; L_MMAP=""; L_1M=""
B_DIO=""; B_BUF=""; B_WARM=""; POP=""
G5=""; G50=""; G250=""; G1000=""; STR=""
SKIPPED=0

# ---- run them ------------------------------------------------------------------------------------
build_probe
if [ "$NO_IO" = 0 ]; then
    probe_all "$FP"
fi

# ---- decisions -----------------------------------------------------------------------------------
# numbers
n_dio_p50=$(pv "${L_DIO:-}" p50_us);  n_dio_p50=${n_dio_p50:-}
n_dio_p90=$(pv "${L_DIO:-}" p90_us);  n_dio_p99=$(pv "${L_DIO:-}" p99_us)
n_dio_max=$(pv "${L_DIO:-}" max_us);  n_dio_gt1=$(pv "${L_DIO:-}" gt1ms)
n_buf_p50=$(pv "${L_BUF:-}" p50_us);  n_mmap_p50=$(pv "${L_MMAP:-}" p50_us)
n_1m_p50=$(pv "${L_1M:-}" p50_us);    n_1m_p99=$(pv "${L_1M:-}" p99_us)
n_bulk_dio=$(pv "${B_DIO:-}" gbs);    n_bulk_buf=$(pv "${B_BUF:-}" gbs)
n_bulk_warm=$(pv "${B_WARM:-}" gbs);  n_pop=$(pv "${POP:-}" gbs)
n_cached=$(pv "${CACHED_BEFORE:-}" pct); n_cached_after=$(pv "${CACHED_AFTER:-}" pct)
n_pop_resid=$(pv "${POP:-}" resident_before_pct)
n_g5=$(pv "${G5:-}" p50_us); n_g50=$(pv "${G50:-}" p50_us)
n_g250=$(pv "${G250:-}" p50_us); n_g1000=$(pv "${G1000:-}" p50_us)
n_stream_p50=$(pv "${STR:-}" p50_us); n_stream_p90=$(pv "${STR:-}" p90_us)
n_stream_p99=$(pv "${STR:-}" p99_us); n_stream_gt1=$(pv "${STR:-}" gt1ms)

for v in n_dio_p50 n_dio_p90 n_dio_p99 n_buf_p50 n_mmap_p50 n_1m_p50 n_bulk_dio n_bulk_buf n_pop; do
    [ -z "${!v}" ] && NO_IO=1
done

# rates per candidate load mode
rate_of() {   # $1 = mode
    case $1 in
        dio)      printf '%s' "$n_bulk_dio" ;;
        none)     printf '%s' "$n_bulk_buf" ;;
        mmap|auto|mlock) printf '%s' "$n_pop" ;;
        *)        printf '' ;;
    esac
}
est_secs() { awk -v b="$F_SIZE" -v g="$1" 'BEGIN{ if (g+0 <= 0) print "n/a"; else printf "%.1f", (b/1073741824)/g }'; }

REC_LOAD_MODE="auto"; REC_LAZY_MODE="off"; REC_REASON=""

US=$'\x1f'
MATRIX=()      # config,first_load,restart,residency,ondemand,verdict,evidence  (US-joined)
AVOID=()       # severity,thing,evidence,action                        (US-joined)

add_matrix() { MATRIX+=("$1${US}$2${US}$3${US}$4${US}$5${US}$6${US}$7"); }
add_avoid()  { AVOID+=("$1${US}$2${US}$3${US}$4"); }

# resident / cache feasibility
FITS=0
if [ "$MEM_AVAIL_MIB" -gt 0 ] && awk -v n="$NEED_MIB" -v a="$MEM_AVAIL_MIB" -v t="$T_CACHE_FIT" 'BEGIN{exit !(n <= a*t)}'; then
    FITS=1
fi
MLOCK_OK=0
if [ "$MEMLOCK_MIB" = -1 ]; then MLOCK_OK=1
elif [ "$MEMLOCK_MIB" -gt 0 ] && awk -v n="$NEED_MIB" -v a="$MEMLOCK_MIB" 'BEGIN{exit !(n <= a)}'; then MLOCK_OK=1; fi

RESTART_MODE="mmap"; RESTART_LAZY="off"
ONDEMAND_SAFE=1
if [ -n "$n_g250" ] && ge "$n_g250" "$T_WAKE_US"; then ONDEMAND_SAFE=0; fi
if [ -n "$n_g1000" ] && ge "$n_g1000" "$T_WAKE_US"; then ONDEMAND_SAFE=0; fi
if [ -n "$n_stream_p90" ] && ge "$n_stream_p90" "$T_STREAM_P90_US"; then ONDEMAND_SAFE=0; fi
if [ -n "$n_stream_p99" ] && ge "$n_stream_p99" "$T_STREAM_P99_US"; then ONDEMAND_SAFE=0; fi

if [ "$NO_IO" = 0 ]; then
    # ---- the two axes that decide this -------------------------------------------------------------
    # 1. FIRST COLD LOAD : size / the mode's measured cold rate.
    # 2. EVERY RESTART   : size / the warm page-cache rate if the bytes survive the process, otherwise
    #                      size / its cold rate again.  A pinned mapping (`--lazy-mode mlock`) is the
    #                      only option that GUARANTEES the bytes survive; a buffered load gets the page
    #                      cache on a best-effort basis; `dio` gets nothing.
    # A one-time cold-load penalty is worth paying for a per-restart saving: that is the whole rule.
    best_mode="" best_rate=0
    for cand in dio none; do
        r=$(rate_of "$cand"); [ -z "$r" ] && continue
        if gt "$r" "$best_rate"; then best_rate=$r; best_mode=$cand; fi
    done
    r_mmap=$(rate_of mmap)
    r_warm=${n_bulk_warm:-}

    r_best_bulk=${n_bulk_dio:-}
    [ "$best_mode" = none ] && r_best_bulk=${n_bulk_buf:-}
    if [ "$FITS" = 1 ] && [ "$MLOCK_OK" = 1 ]; then
        # ---- the pinning plan wins, and says what the first load costs ------------------------------
        penalty=$(awk -v b="$F_SIZE" -v x="$best_rate" -v y="$r_mmap" \
                     'BEGIN{ if (x+0<=0||y+0<=0) print "n/a"; else printf "%.1f", (b/1073741824)*(1/y - 1/x) }')
        # r230 measured the pin itself: `off`'s copy dies with the process and is swappable, the pin is
        # a page-table walk to re-establish (0.19 s for 27 GiB) and the pages survive the exit.  The two
        # mapped forms differ only in how the *bulk* is read, so offer the faster one.
        # Both pinning forms keep the lazy data path and pay ONE populate; they differ only in how the
        # non-lazy bulk is read (`mmap`: fault-in + copy, `none`: buffered pread).  Recommend the faster.
        v_mmap=ALTERNATIVE; v_none=ALTERNATIVE
        if ge "${n_bulk_buf:-0}" "$r_mmap"; then v_none=RECOMMENDED; else v_mmap=RECOMMENDED; fi
        add_matrix "--load-mode mmap --lazy-mode mlock" "$(est_secs "$r_mmap")" \
            "$(est_secs "$r_warm")" "lazy tables pinned, survive the process" "safe (pinned, never reclaimed)" \
            "$v_mmap" \
            "non-lazy bulk at the mapped fault-in rate ${r_mmap} GB/s (${penalty} s more than ${best_mode} on the first load); the pin re-establishes in a page-table walk once the pages are warm (r230: 0.19 s / 27 GiB), so a restart reads nothing"
        add_matrix "--load-mode none --lazy-mode mlock" "$(est_secs "${n_bulk_buf:-0}")" \
            "$(est_secs "$r_warm")" "lazy tables pinned, survive the process" "safe (pinned, never reclaimed)" \
            "$v_none" \
            "non-lazy bulk at the buffered rate ${n_bulk_buf} GB/s plus the pin's one-time populate; a restart reads nothing from the device (warm ${r_warm} GB/s) -- the form r230's fastest restart used (8.26 s ready-to-serve, 0.19 s to re-pin 27 GiB)"
        if [ "$v_none" = RECOMMENDED ]; then REC_LOAD_MODE=none; else REC_LOAD_MODE=mmap; fi
        REC_LAZY_MODE=mlock
        REC_REASON="the working set fits in RAM (${NEED_MIB} MiB) and RLIMIT_MEMLOCK allows pinning it, so the first load's populate is paid once and every restart is served from pinned page cache"
    elif [ "$FITS" = 1 ]; then
        add_matrix "--load-mode mmap --lazy-mode mlock" "n/a" "n/a" "cannot be pinned" "n/a" "BROKEN" \
            "RLIMIT_MEMLOCK is $( [ "$MEMLOCK_MIB" = -1 ] && echo unlimited || echo "${MEMLOCK_MIB} MiB" )"
        add_avoid "BROKEN" "--lazy-mode mlock" \
            "needs ${NEED_MIB} MiB pinned, RLIMIT_MEMLOCK is $( [ "$MEMLOCK_MIB" = -1 ] && echo unlimited || echo "${MEMLOCK_MIB} MiB" )" \
            "raise it: ulimit -l $(( NEED_MIB / 1024 )) (or systemd LimitMEMLOCK) -- this is the single biggest restart win on this box"
        REC_LOAD_MODE=$best_mode; REC_LAZY_MODE=off
        REC_REASON="cold load ${best_rate} GB/s; the working set fits, but mlock is refused at the current RLIMIT, so restarts fall back to the page cache"
    else
        REC_LOAD_MODE=$best_mode; REC_LAZY_MODE=off
        REC_REASON="cold load ${best_rate} GB/s; the working set ${NEED_MIB} MiB does not fit beside MemAvailable ${MEM_AVAIL_MIB} MiB, so nothing can be held and streaming is the only plan"
    fi

    # the streaming modes, on both axes
    for cand in none dio; do
        r=$(rate_of "$cand"); [ -z "$r" ] && continue
        ratio=$(awk -v a="$best_rate" -v b="$r" 'BEGIN{printf "%.2f", (b+0<=0?0:a/b)}')
        verdict=AVOID
        [ "$cand" = "$best_mode" ] && [ "$REC_LAZY_MODE" = mlock ] && verdict=ALTERNATIVE
        [ "$cand" = "$best_mode" ] && [ "$REC_LAZY_MODE" != mlock ] && verdict=RECOMMENDED
        [ "$cand" != "$best_mode" ] && lt "$ratio" "$T_SAME" && verdict=ACCEPTABLE
        case $cand in
            none) add_matrix "--load-mode none --lazy-mode off" "$(est_secs "$r")" \
                      "$(est_secs "$r_warm")" "page cache only (reclaimable)" "safe while streaming" "$verdict" \
                      "buffered pread ${r} GB/s cold, ${ratio}x ${best_mode}; fills the cache, so a restart is warm while the pages live" ;;
            dio)  add_matrix "--load-mode dio --lazy-mode off" "$(est_secs "$r")" \
                      "$(est_secs "$r")" "none (bypasses the cache)" "safe (streams ahead, never faults mid-run)" "$verdict" \
                      "O_DIRECT ${r} GB/s, ${ratio}x ${best_mode}; every restart re-reads the file, and alignment can read more bytes than the file holds" ;;
        esac
    done
    if [ -n "$r_mmap" ]; then
        add_matrix "--load-mode mmap --lazy-mode off" "$(est_secs "$r_mmap")" \
            "$(est_secs "$r_warm")" "page cache only (reclaimable)" "UNSAFE if tables stay on demand (see AVOID below)" \
            "AVOID" \
            "the fault-in path runs at ${r_mmap} GB/s -- pin it (mlock) to pay that once, or stream it; leaving it on demand pays it on every access"
    fi

    # ---- AVOID: the mapped cold load, when it measurably loses
    if [ -n "$n_pop" ] && [ -n "$n_bulk_dio" ] && gt "$(awk -v a="$n_bulk_dio" -v b="$n_pop" 'BEGIN{print a/b}')" "$T_DIO_BEATS"; then
        extra=$(awk -v b="$F_SIZE" -v x="$n_bulk_dio" -v y="$n_pop" 'BEGIN{printf "%.1f", (b/1073741824)*(1/y - 1/x)}')
        add_avoid "AVOID" "--load-mode mmap --lazy-mode off (mapped, never pinned, never reused)" \
            "fault-in ${n_pop} GB/s vs O_DIRECT ${n_bulk_dio} GB/s, i.e. ${extra} s more on this ${F_GIB} GiB file -- and it pays that again whenever the pages are reclaimed" \
            "either pin it (--lazy-mode mlock: one populate, then every restart is free) or stream it (--load-mode dio/none); what is wrong is the middle ground"
    fi

    # ---- the one rule worth stating on its own: a resident copy beats an in-process copy ----------
    if [ "$FITS" = 1 ] && [ "$MLOCK_OK" = 1 ] && [ -n "$r_warm" ]; then
        add_avoid "AVOID" "--lazy-mode off when the working set fits (an in-process copy is not a cache)" \
            "${NEED_MIB} MiB fits beside MemAvailable ${MEM_AVAIL_MIB} MiB and RLIMIT_MEMLOCK allows pinning; the warm page cache serves ${r_warm} GB/s, and a 'none'/off load reads the file into a buffer that dies with the process" \
            "use --lazy-mode mlock: the pinned pages survive the process, so the next start reads nothing from the device"
    fi

    # ---- AVOID: residency when it cannot be held
    if [ "$FITS" = 0 ]; then
        add_avoid "AVOID" "--lazy-mode mlock / any 'keep it all resident' plan" \
            "working set ${NEED_MIB} MiB > $T_CACHE_FIT x MemAvailable ${MEM_AVAIL_MIB} MiB" \
            "stream instead (--lazy-mode off); pinned tables would be reclaimed and re-faulted"
    fi

    # ---- AVOID: on-demand reads during decode
    if [ "$ONDEMAND_SAFE" = 0 ]; then
        ev=""
        [ -n "$n_g250" ] && ev="4 KiB read after a 250 ms pause: ${n_g250} us"
        [ -n "$n_stream_p99" ] && ev="$ev; 4 KiB p99 while streaming: ${n_stream_p99} us"
        add_avoid "AVOID" "any on-demand disk tier (--lazy-mode on|auto on large tables, MOE_HOST_POOL_DIO=1, a KV spill/window with a host tail)" \
            "$ev" \
            "everything must be resident before serving, or read ahead of the stall; there is no cheap late read here"
    fi

    # ---- AVOID: trusting the file's streaming speed
    if [ -n "$MAP_WORST_GBS" ] && [ -n "$MAP_BEST_GBS" ] && lt "$(awk -v a="$MAP_WORST_GBS" -v b="$MAP_BEST_GBS" 'BEGIN{print a/b}')" "$T_SPREAD"; then
        add_avoid "AVOID" "assuming this file streams at its best rate" \
            "bands range ${MAP_WORST_GBS} GB/s (at ${MAP_WORST_OFF} MiB) to ${MAP_BEST_GBS} GB/s (at ${MAP_BEST_OFF} MiB)" \
            "re-write or move the file to relocate it; probe the band map again after"
    fi

    # ---- tradeoff: dio throws the warm cache away
    if [ -n "$n_bulk_warm" ] && [ -n "$n_bulk_dio" ] && gt "$(awk -v a="$n_bulk_warm" -v b="$n_bulk_dio" 'BEGIN{print a/b}')" "$T_WARM_GAIN" && [ "$FITS" = 1 ]; then
        add_avoid "TRADEOFF" "--load-mode dio on every start" \
            "warm page cache serves ${n_bulk_warm} GB/s against ${n_bulk_dio} GB/s from the device" \
            "dio for the cold load only; let the page cache serve the restarts (or pin them with --lazy-mode mlock)"
    fi

    # ---- transparency items
    if [ -n "$SKIPPED" ] && [ "$SKIPPED" -gt 0 ]; then
        add_avoid "UNKNOWN" "the page-cache arms (buffered pread / mmap fault)" \
            "${SKIPPED} sample(s) could not be made cold and were skipped" \
            "re-run when the file is less hot, or with --cold (root) for --verify"
    fi
else
    add_matrix "--load-mode dio --lazy-mode off" "n/a" "n/a" "none" "unknown" "UNKNOWN" "--no-io"
    add_matrix "--load-mode mmap --lazy-mode mlock" "n/a" "n/a" "unknown" "unknown" "UNKNOWN" "--no-io"
    REC_REASON="no probes were run (--no-io)"
fi

# file-layout facts that change the recommendation
if [ "$F_EXTENTS" != "?" ] && [ "$F_EXTENTS" -gt "$T_FRAG_EXTENTS" ] 2>/dev/null; then
    add_avoid "ADVISORY" "this file's on-disk layout" "${F_EXTENTS} extents (fragmented for a large model)" \
        "re-write the file contiguously; the drive then sees one run of reads"
fi
case "$FS_OPTS" in
    *relatime*|*strictatime*) add_avoid "ADVISORY" "atime updates on this filesystem" "mount options: $FS_OPTS" \
        "mount with noatime: an atime update is a write hidden inside a read workload" ;;
esac
if [ -n "${Q_ROTA:-}" ] && [ "$Q_ROTA" = 1 ]; then
    add_avoid "ADVISORY" "treating this device as fast random storage" "rotational=1" "use large sequential reads only"
fi

# ---- verification (needs the recommendation, so it runs after the decisions) ---------------------
if [ "$VERIFY" = 1 ]; then
    if [ "$NO_IO" = 1 ]; then V_NOTE="--verify needs probes; --no-io was given"
    else run_verify; fi
else
    V_NOTE="not requested (--verify)"
fi

# The pin only covers TENSOR_READ_LAZY tables (the PLE/engram kind).  On a model without one,
# `--lazy-mode mlock` is inert -- measured here: a 4B model with no lazy table ran `mmap --lazy-mode
# mlock` and `mmap --lazy-mode off` to 5.17 s and 5.18 s cold, i.e. the mode did nothing, and only the
# ordinary page cache gave the 1.00 s restart.  So confirm it before standing behind the recommendation.
avoid_drop() {           # $1 = substring of the "thing" field
    local i kept=()
    for i in ${AVOID[@]+"${AVOID[@]}"}; do
        case $i in *"$1"*) ;; *) kept+=("$i") ;; esac
    done
    AVOID=(${kept[@]+"${kept[@]}"})
}
matrix_fix_verdict() {   # $1 = config substring, $2 = new verdict
    local i row
    [ ${#MATRIX[@]} -gt 0 ] || return 0
    for i in "${!MATRIX[@]}"; do
        row=${MATRIX[$i]}
        case ${row%%${US}*} in
            *"$1"*) MATRIX[$i]=$(printf '%s' "$row" | awk -F"${US}" -v v="$2" 'BEGIN{OFS=FS}{$6=v; print}') ;;
        esac
    done
}
if [ "${LAZY_TABLES:-}" = 0 ]; then
    add_avoid "ADVISORY" "--lazy-mode mlock on this model" \
        "the loader pinned 0 tables: it has no TENSOR_READ_LAZY table, so the mode is inert and the restart win is the ordinary page cache" \
        "drop the flag (it costs nothing but buys nothing); the streaming recommendation below is what actually applies"
    # the pin-specific advice does not apply to a model with no lazy table
    avoid_drop "when the working set fits (an in-process copy is not a cache)"
    avoid_drop "mapped, never pinned, never reused"
    extra2=$(awk -v b="$F_SIZE" -v x="${n_bulk_dio:-0}" -v y="${n_pop:-0}" \
                 'BEGIN{ if (x+0<=0||y+0<=0) print "n/a"; else printf "%.1f", (b/1073741824)*(1/y - 1/x) }')
    add_avoid "AVOID" "--load-mode mmap --lazy-mode off (mapped, never pinned)" \
        "fault-in ${n_pop:-?} GB/s against O_DIRECT ${n_bulk_dio:-?} GB/s: ${extra2} s more on this file, and nothing is pinned" \
        "stream it (--load-mode dio for one cold start, --load-mode none if you restart often); this model has no TENSOR_READ_LAZY table, so there is nothing for a pin to hold"
    if [ "$REC_LAZY_MODE" = mlock ]; then
        matrix_fix_verdict "mlock" "INERT"
        matrix_fix_verdict "--load-mode $best_mode --lazy-mode off" "RECOMMENDED"
        REC_LOAD_MODE=$best_mode; REC_LAZY_MODE=off
        REC_REASON="cold load ${best_rate} GB/s; --lazy-mode mlock measured inert on this model (0 tables pinned), so a restart rides the ordinary page cache -- if you restart often and the set fits, \`--load-mode none\` keeps those pages warm, at ${n_bulk_buf} GB/s against ${best_rate} cold"
    fi
elif [ -z "${LAZY_TABLES:-}" ] && [ "$REC_LAZY_MODE" = mlock ]; then
    # The pin covers only TENSOR_READ_LAZY tables and nothing on disk says whether this model has one,
    # so an unverified pin is UNCONFIRMED, not recommended: a model without such a table makes the flag
    # inert, and recommending it would be a guess dressed up as a measurement.
    matrix_fix_verdict "mlock" "UNCONFIRMED"
    matrix_fix_verdict "--load-mode $best_mode --lazy-mode off" "RECOMMENDED"
    REC_LOAD_MODE=$best_mode; REC_LAZY_MODE=off
    REC_REASON="cold load ${best_rate} GB/s. The pinning plans are UNCONFIRMED: whether --lazy-mode mlock pins anything depends on the model having a TENSOR_READ_LAZY table, and this run did not load it. Run with \`--verify\` once -- it reports \`pinned N table(s), X MiB\` -- and the pin moves to the top if it does"
    add_avoid "UNCONFIRMED" "--lazy-mode mlock (until --verify has loaded the model once)" \
        "the pin covers only TENSOR_READ_LAZY tables (per_layer_token_embd and friends); nothing measurable on disk says whether this model has one, and 0 pinned tables makes the mode inert (measured: 5.17 s vs 5.18 s cold on a 4B model with no lazy table)" \
        "run \`--verify\`; the third arm then reports the pin and the restart it buys"
fi

# ---- report --------------------------------------------------------------------------------------
tmp=$REPORT.$$.tmp
trap 'rm -f "$tmp"' EXIT
{
cat <<EOF
# Storage oracle report

Generated $(date -u '+%Y-%m-%d %H:%M:%S UTC') on \`$HOST\` by \`storage-oracle.sh\` v$VERSION.
**Scope: storage and load mechanism only.** It says nothing about VRAM fit, tensor splits, quantization
or architecture -- check those separately.

## 1. Recommendation

### First (cold) load

**\`--load-mode $REC_LOAD_MODE --lazy-mode $REC_LAZY_MODE\`** — $REC_REASON.

### Restarts

$( if [ "$REC_LAZY_MODE" = mlock ]; then
     echo "**\`--load-mode $REC_LOAD_MODE --lazy-mode mlock\`** — unchanged. The pin survives the process (r230 measured a page-table walk of **0.19 s for 27 GiB** to re-establish it on a warm cache), so a restart reads nothing from the device."
   elif [ "$RESTART_LAZY" = mlock ]; then
     echo "**\`--load-mode $RESTART_MODE --lazy-mode mlock\`** — the working set fits in RAM and RLIMIT_MEMLOCK allows pinning it, so a restart re-pins instead of re-reading."
   else
     echo "**\`--load-mode $RESTART_MODE --lazy-mode $RESTART_LAZY\`** — the page cache is the only reuse available; a cold start repeats the first-load path."
   fi )
Warm-cache rate measured: ${n_bulk_warm:-n/a} GB/s against ${n_bulk_dio:-n/a} GB/s from the device.

EOF
if [ ${#MATRIX[@]} -gt 0 ]; then
cat <<'EOF'
| configuration | first cold load (s) | restart (s) | residency | on-demand safety | verdict | evidence |
|---|---|---|---|---|---|
EOF
for row in "${MATRIX[@]}"; do
    IFS=${US} read -r c1 c2 c3 c4 c5 c6 c7 <<<"$row"
    printf '| `%s` | %s | %s | %s | %s | **%s** | %s |\n' "$c1" "$c2" "$c3" "$c4" "$c5" "$c6" "$c7"
done
fi
cat <<'EOF'

## 2. Avoid

EOF
if [ ${#AVOID[@]} -eq 0 ]; then
    echo "Nothing measured on this file was bad enough to forbid a configuration."
else
    for row in "${AVOID[@]}"; do
        IFS=${US} read -r sev what ev act <<<"$row"
        printf -- '- **%s** `%s` — %s. _Do instead:_ %s\n' "$sev" "$what" "$ev" "$act"
    done
fi
cat <<EOF

## 3. Facts

| | |
|---|---|
| model file | \`$FP\` |
| model set | ${SHARD_COUNT:-${#MODELS[@]}} file(s), $(gib "$sum_bytes") GiB |
| filesystem | $FS_TYPE on \`$FS_TARGET\` from \`$FS_SRC\` ($FS_OPTS) |
| free space | ${FS_FREE_GIB} GiB |
| block device | \`/dev/${BLK}\` ${DEV_MODEL}${DEV_FW:+ (fw $DEV_FW)} |
| PCIe link | ${PCIE_LINK:-unknown}${PCIE_PORT:+ at \`$PCIE_PORT\`} |
| file layout | ${F_EXTENTS} extent(s), st_blksize ${F_BLKSIZE} |
| queue | scheduler ${Q_SCHED:-?}, rotational ${Q_ROTA:-?}, max_sectors ${Q_MAXKB:-?} KiB, read_ahead ${Q_RA:-?} KiB |
| memory | MemTotal $(awk -v k="$MEM_TOTAL_KB" 'BEGIN{printf "%.1f", k/1048576}') GiB, MemAvailable $(gib $((MEM_AVAIL_KB * 1024))) GiB |
| working set | ${NEED_MIB} MiB = model files + ${RESIDENT_GIB} GiB (KV/pools/compute, \`--resident-gib\`) |
| RLIMIT_MEMLOCK | $( [ "$MEMLOCK_MIB" = -1 ] && echo unlimited || echo "${MEMLOCK_MIB} MiB" ) |
| power policy (APST) | $APST_NOTE |
| lazy tables (all --lazy-mode mlock can pin) | $( if [ "${LAZY_TABLES:-}" = 0 ]; then echo "**none** - the mode is inert on this model"; elif [ -n "${LAZY_TABLES:-}" ]; then echo "${LAZY_TABLES} table(s) -- ${LAZY_PIN}"; else echo "unknown (run --verify) - the pin only covers TENSOR_READ_LAZY tables"; fi ) |
| kernel / cpu | $(uname -r) / $(nproc) threads, page size ${PSIZE} |
| probe cost | about ${PLAN_GIB} GiB read |

## 4. Measurements

EOF
if [ "$NO_IO" = 1 ]; then
    echo "No probes were run (\`--no-io\`, or a probe failed)."
else
cat <<EOF
### 4.1 Band map — is this file uniformly fast on the device?

${MAP_CHUNK} MiB \`O_DIRECT\` reads, ${MAP_STEPS} evenly spaced offsets. A band that is much slower than
the best one is a placement problem, not a filesystem problem, and the fix is to re-write the file.
Note these chunks are one \`pread\` each, which the block layer splits into 512 KiB commands and
pipelines deeply, so they read *higher* than the 4 MiB sequential arm in 4.3 (which is closer to how
the loader issues reads). Read this table for the shape, 4.3 for the rate to predict from.

| offset (GiB) | GB/s |
|---|---|
EOF
    for l in "${MAP_LINES[@]:-}"; do
        [ -n "$l" ] || continue
        z=$(pv "$l" zeros)
        printf '| %s | %s%s |\n' "$(pv "$l" off_gib)" "$(pv "$l" gbs)" "$( [ "$z" = 1 ] && echo ' _(unwritten, ignored)_' )"
    done
cat <<EOF

### 4.2 Cold per-read latency

Every sample evicted and verified cold (\`mincore\`); skipped-warm samples are counted, never timed.

| mechanism | p50 | p90 | p99 | max | >1 ms | skipped |
|---|---|---|---|---|---|---|
| \`O_DIRECT\` 4 KiB | ${n_dio_p50:-n/a} us | ${n_dio_p90:-n/a} us | ${n_dio_p99:-n/a} us | ${n_dio_max:-n/a} us | ${n_dio_gt1:-?} | $(pv "${L_DIO:-}" skipped_warm | grep . || echo "?") |
| buffered pread 4 KiB | ${n_buf_p50:-n/a} us | $(pv "${L_BUF:-}" p90_us) us | $(pv "${L_BUF:-}" p99_us) us | $(pv "${L_BUF:-}" max_us) us | $(pv "${L_BUF:-}" gt1ms) | $(pv "${L_BUF:-}" skipped_warm) |
| mmap fault 4 KiB | ${n_mmap_p50:-n/a} us | $(pv "${L_MMAP:-}" p90_us) us | $(pv "${L_MMAP:-}" p99_us) us | $(pv "${L_MMAP:-}" max_us) us | $(pv "${L_MMAP:-}" gt1ms) | $(pv "${L_MMAP:-}" skipped_warm) |
| \`O_DIRECT\` 1 MiB | ${n_1m_p50:-n/a} us | $(pv "${L_1M:-}" p90_us) us | $(pv "${L_1M:-}" p99_us) us | $(pv "${L_1M:-}" max_us) us | $(pv "${L_1M:-}" gt1ms) | $(pv "${L_1M:-}" skipped_warm) |

### 4.3 Sequential throughput

| path | GB/s |
|---|---|
| \`O_DIRECT\`, 4 MiB blocks | ${n_bulk_dio:-n/a} |
| buffered pread, 1 MiB (after eviction) | ${n_bulk_buf:-n/a} |
| buffered pread, 1 MiB (warm page cache) | ${n_bulk_warm:-n/a} |
| \`mmap\` + \`MAP_POPULATE\` (the fault-in path) | ${n_pop:-n/a} (${n_pop_resid:-?}% was still cached) |
| page-cache residency at the start / before the buffered arm | ${n_cached:-?}% / $(pv "${CACHED_ARM:-}" pct)% |
| page-cache residency after the probes | ${n_cached_after:-?}% |

### 4.4 Idle-gap ladder — cost of a read after a pause

This is the number no specification publishes, and it decides whether a read *during* decode is safe.
**It is also a property of the power policy, not just the drive:** APST transitions can multiply it
several-fold (measured on this box: 16.3 ms with APST at its default against 4.7 ms with APST off).
APST here: $APST_NOTE.

| idle before the read | p50 latency |
|---|---|
| 5 ms | ${n_g5:-n/a} us |
| 50 ms | ${n_g50:-n/a} us |
| 250 ms | ${n_g250:-n/a} us |
| 1000 ms | ${n_g1000:-n/a} us |

### 4.5 Cold 4 KiB while the same file streams

| p50 | p90 | p99 | samples > 1 ms |
|---|---|---|---|
| ${n_stream_p50:-n/a} us | ${n_stream_p90:-n/a} us | ${n_stream_p99:-n/a} us | ${n_stream_gt1:-n/a} |
EOF
fi
cat <<EOF

## 5. Verification

EOF
if [ "$VERIFY" = 0 ]; then
    echo "Not requested (\`--verify\`)."
else
if [ -n "$V_NOTE" ]; then echo "**$V_NOTE**"; fi
cat <<EOF
Binary: \`${V_BIN:-n/a}\` — args: \`$VERIFY_ARGS\` — \`--single-turn --no-warmup -lv 4\`, ${VERIFY_N} token(s).

| arm | result | wall clock (s) | device read (MiB) | effective (GB/s) | page cache before | extra args |
|---|---|---|---|---|
EOF
    for row in "${V_ROWS[@]:-}"; do
        [ -n "$row" ] || continue
        IFS=${US} read -r a1 a2 a3 a4 a5 a6 a7 <<<"$row"
        printf '| %s | %s | %s | %s | %s | %s%% | `%s` |\n' "$a1" "$a2" "$a3" "$a4" "$a5" "$a6" "$a7"
    done
    for e in ${V_ERRS[@]+"${V_ERRS[@]}"}; do printf -- '- %s\n' "$e"; done
    if [ "$V_FAILED" = 1 ]; then
        echo
        echo "**Verification did not run.** At least one arm failed, so nothing here confirms or refutes the recommendation. The error is above — usually a flag problem (VRAM fit, a tensor split that puts the whole model on one device), not a storage problem. Fix \`--verify-args\` and re-run."
    elif [ ${#V_ROWS[@]} -ge 2 ]; then
        w1=$(printf '%s' "${V_ROWS[0]}" | awk -F"${US}" '{print $3}')
        w2=$(printf '%s' "${V_ROWS[1]}" | awk -F"${US}" '{print $3}')
        d=$(awk -v a="$w1" -v b="$w2" 'BEGIN{ if (a+0<=0) {print "0 0"; exit} printf "%.3f %.1f", ((a-b)/a < 0 ? (b-a)/a : (a-b)/a), (a-b)/a*100 }')
        dfrac=${d% *}; dpct=${d#* }
        thr=$(awk -v t="$T_VERIFY_MATCH" 'BEGIN{printf "%d", t*100}')
        echo
        if lt "$dfrac" "$T_VERIFY_MATCH"; then
            echo "**Prediction inconclusive.** The two arms differ by ${dpct}% of wall clock (threshold +/-${thr}%), which is noise for a single run. Interleave the arms, or use \`--cold\` for a stricter comparison."
        elif ge "$dpct" 0; then
            echo "**Prediction held.** The recommended arm finished ${dpct}% faster than the shipped default on the same bytes (\`T_VERIFY_MATCH\` = +/-${thr}%)."
        else
            echo "**Prediction did NOT hold.** The recommended arm finished ${dpct}% *slower* than the shipped default. Do not adopt the recommendation on this evidence -- re-run interleaved (A/B/A/B) and with \`--cold\`, and if it repeats, treat the probe-based ranking as refuted for this file."
        fi
    fi
    if [ "$V_FAILED" = 0 ] && [ ${#V_ROWS[@]} -ge 3 ]; then
        cw=""; ww=""
        cw=$(printf '%s' "${V_ROWS[1]}" | awk -F"${US}" '{print $3}')
        ww=$(printf '%s' "${V_ROWS[2]}" | awk -F"${US}" '{print $3}')
        if [ -n "$cw" ] && [ -n "$ww" ]; then
            echo
            echo "**Restart effect.** Cold ${cw} s, warm restart ${ww} s -- the second start is $(awk -v a="$cw" -v b="$ww" 'BEGIN{printf "%.0f", 100*b/a}')% of the first. That is the page cache doing the work; a pin decides whether those pages are still there."
        fi
    fi
    if [ "$V_FAILED" = 0 ] && [ ${#V_ROWS[@]} -ge 2 ]; then
        cat <<'EOF'

**How to read this.** Both arms load the same bytes from the same file and generate the same one
token, so the difference is the mechanism. *Device read* is the byte count the kernel attributes to
this block device (`/sys/block/.../stat`) across the arm — it proves the arm really read the file and
how much of it — and *effective* is that divided by the wall clock. The page cache is evicted before
each arm so the first one does not race the second one's cache. Compare **ranks**, not absolute seconds — device upload and graph setup are in
both columns and this report does not model them. One run per arm: interleave (`A/B/A/B`) manually
if you are deciding on a noisy box.
EOF
    fi
fi
cat <<EOF

## 6. Decision rules used

| threshold | value | meaning |
|---|---|---|
| \`T_DIO_BEATS\` | $T_DIO_BEATS | O_DIRECT must beat the mapped fault path by this factor to be the cold load mode |
| \`T_SAME\` | $T_SAME | a mode within this factor of the best is ACCEPTABLE rather than RECOMMENDED |
| \`T_SPREAD\` | $T_SPREAD | worst band below this fraction of the best => placement warning |
| \`T_WAKE_US\` | $T_WAKE_US | idle-gap wake above this forbids on-demand reads |
| \`T_STREAM_P90_US\` / \`T_STREAM_P99_US\` | $T_STREAM_P90_US / $T_STREAM_P99_US | 4 KiB p90/p99 under a stream above this forbids on-demand reads |
| \`T_CACHE_FIT\` | $T_CACHE_FIT | a resident plan must need no more than this fraction of MemAvailable |
| \`T_WARM_GAIN\` | $T_WARM_GAIN | warm/cold rate above this makes \`dio\` a tradeoff, not a free win |

## 7. What this did not test

- VRAM fit, tensor splits (\`-sm\`), \`-ncmoe\`, quantization, architecture: out of scope by design.
- Multi-process interference: probes run one at a time, and never in parallel with a benchmark.
- Throughput under *multiple* concurrent readers (one stream only).
- Rotational or network storage beyond what the facts table shows.

Reproduce any arm with:

\`\`\`sh
$PROBE_BIN map  "$FP" $MAP_CHUNK $MAP_STEPS 0
$PROBE_BIN rand "$FP" dio 4096 $R_DIO_N $PROBE_REGION $PROBE_BASE
$PROBE_BIN bulk "$FP" dio 4 $BULK_COUNT $PROBE_BASE
$PROBE_BIN pop  "$FP" $POP_MIB $PROBE_BASE
\`\`\`
EOF
} > "$tmp" || die "cannot write $tmp"
mv "$tmp" "$REPORT" || die "cannot write $REPORT"

printf 'report: %s\n' "$REPORT"
grep -F '**`--load-mode' "$REPORT" | head -1 || true
