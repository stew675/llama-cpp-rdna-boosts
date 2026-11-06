#!/usr/bin/env bash
# gpu-watch.sh - a 2 s sampler for GPU use / VRAM / host RAM / locked memory, so a wedge leaves a record.
#
# WHY THIS EXISTS
#   2026-11-05: this box locked up with a R9700 pinned at 100 % and had to be power-cycled, and the kernel
#   journal recorded NOTHING (the previous boot's last line was ~70 min before the end: no ring timeout, no
#   `GPU reset begin!`, no OOM, no hung-task trace).  A hard wedge that kills the journal is exactly the case
#   where the only surviving evidence is what an out-of-band sampler already wrote to disk.  `rocm-smi` run
#   by hand after the reboot tells you nothing about before it.
#
#   This is NOT a GPU-fault check - that is `check-gpu-faults.sh` (kernel journal + a llama log).  This is the
#   flight recorder that makes the NEXT incident diagnosable.
#
# Usage:
#   ./gpu-watch.sh --cmd '<command>' [options]     # run COMMAND under timeout, sample until it exits
#   ./gpu-watch.sh --dur 600 [options]             # sample for 600 s (watch a run you started yourself)
#   ./gpu-watch.sh --summary FILE                  # print the per-GPU peak/mean of an existing CSV
#
# Options:
#   --out FILE        CSV to append to (default: /tmp/gpu-watch-<date>-<pid>.csv)
#   --interval SEC    sampling period (default 2)
#   --timeout SEC     with --cmd, kill it after SEC (default 1800); SIGTERM then SIGKILL 10 s later
#   --rocm-every N    also call `rocm-smi --showuse` every N samples (default 10; 0 = never)
#   --quiet           no per-sample echo on the terminal
#   -h, --help        this help
#
# Environment:
#   SAMPLE_NOTE       free-form text prefixed to every line (the arm name; do set it)
#
# Columns: ts, core_s, gpu_busy_pct[-1 = unknown], vram_used_mib, vram_total_mib, mem_avail_mib, mem_free_mib,
#          shmem_mib, mlocked_mib, llama_rss_mib, llama_vmlck_mib, llama_pids, load1, rocm_busy, per_card_busy, note
#
# `shmem_mib` is the counter a pinned host pool shows up in on Linux (hipHostMalloc yields shmem, NOT VmLck --
# see PR #267's note), and `mlocked_mib` is /proc/meminfo Mlocked (a file-alias mlock shows here, not in shmem).
#
# Reading it: a wedge looks like gpu_busy_pct climbing to 100 and STAYING there while core_s (seconds since the
# sampler started) keeps advancing and shmem/rss do not move; a memory death looks like mem_avail_mib collapsing
# first; a host-side stall looks like gpu_busy_pct at 0 with the process still resident.
#
# TWO ARTEFACTS to know before quoting a number: (a) the FIRST sample of a run can read 99-100 % while the device
# is being woken from runtime suspend -- ignore a lone peak at core_s 0; (b) the R9700s report EBUSY on
# gpu_busy_percent while they are runtime-suspended, so -1 means "unknown", never "idle" (rocm_busy is the
# cross-check and is the only source that answers while suspended).
set -uo pipefail

OUT=""
INTERVAL=2
DUR=""
CMD=""
CMD_TIMEOUT=1800
ROCM_EVERY=10
QUIET=0
SUMMARY=""
NOTE="${SAMPLE_NOTE:-}"

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --out)        OUT="$2"; shift 2 ;;
        --interval)   INTERVAL="$2"; shift 2 ;;
        --dur)        DUR="$2"; shift 2 ;;
        --cmd)        CMD="$2"; shift 2 ;;
        --timeout)    CMD_TIMEOUT="$2"; shift 2 ;;
        --rocm-every) ROCM_EVERY="$2"; shift 2 ;;
        --summary)    SUMMARY="$2"; shift 2 ;;
        --quiet)      QUIET=1; shift ;;
        -h|--help)    usage 0 ;;
        *)            echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

# ---- summary mode: summarise an existing CSV ------------------------------------------------------------
if [ -n "$SUMMARY" ]; then
    [ -f "$SUMMARY" ] || { echo "no such file: $SUMMARY" >&2; exit 2; }
    awk -F, 'NR == 1 { next }
        ($3 ~ /^[0-9]+$/) {
            g++
            b = $3 + 0
            if (b > peak) { peak = b; peak_t = $1; peak_core = $2 }
            sum += b
            if (b >= 99) pinned++
            if (g == 1 || $6 + 0 < min_avail) { min_avail = $6 + 0 }
            if ($8 + 0 > shmem_peak) { shmem_peak = $8 + 0 }
            if ($9 + 0 > mlocked_peak) { mlocked_peak = $9 + 0 }
        }
        END {
            if (!g) { print "no numeric samples"; exit }
            printf "samples            : %d\n", g
            printf "gpu_busy_pct       : mean %.1f  peak %d  at %s (t+%ss)\n", sum / g, peak, peak_t, peak_core
            printf "samples at >=99 %%  : %d\n", pinned + 0
            printf "mem_avail_mib min  : %d\n", min_avail
            printf "shmem_mib peak     : %d\n", shmem_peak
            printf "mlocked_mib peak   : %d\n", mlocked_peak
        }' "$SUMMARY"
    exit 0
fi

if [ -z "$CMD" ] && [ -z "$DUR" ]; then
    echo "need --cmd '<command>' or --dur <seconds> (see --help)" >&2
    exit 2
fi
if [ -n "$OUT" ]; then
    mkdir -p "$(dirname "$OUT")" 2>/dev/null
else
    OUT="/tmp/gpu-watch-$(date +%Y%m%d-%H%M%S)-$$.csv"
fi

# ---- the sample (cheap: sysfs + /proc; rocm-smi only every N ticks, it opens the driver) -------------------
# The R9700s report EBUSY on gpu_busy_percent while the device is runtime-suspended (their normal idle state),
# so -1 means "unknown", never "idle".
gpu_dirs() {
    local d
    for d in /sys/class/drm/card*/device; do
        [ -f "$d/mem_info_vram_total" ] || continue
        # skip the iGPU?  No: it is a real device here (gfx1036) and its VRAM is the carveout.  Keep every card.
        printf '%s\n' "$d"
    done
}

sample() {
    local core_s="$1" tick="$2" out="$3"
    local ts busy vram_u vram_t rocm=""
    ts="$(date +%H:%M:%S)"

    # Collect every card into parallel arrays (cheap: 4 cards, two sysfs reads each).
    local i=0 bs=() us=() ts_=()
    while read -r d; do
        [ -n "$d" ] || continue
        busy="$(cat "$d/gpu_busy_percent" 2>/dev/null)"
        case "$busy" in ''|*[!0-9]*) busy=-1 ;; esac
        vram_u="$(cat "$d/mem_info_vram_used" 2>/dev/null)"; [ -n "$vram_u" ] || vram_u=0
        vram_t="$(cat "$d/mem_info_vram_total" 2>/dev/null)"; [ -n "$vram_t" ] || vram_t=0
        bs[$i]="$busy"; us[$i]=$(( vram_u / 1048576 )); ts_[$i]=$(( vram_t / 1048576 )); i=$(( i + 1 ))
    done < <(gpu_dirs)
    # the reported gpu_busy_pct is the MAX over the cards (the wedge signature), -1 only if every card is unknown
    local maxb=-1 vram_used=0 vram_total=0 k
    for k in "${!bs[@]}"; do
        if [ "${bs[$k]}" -ge 0 ] && { [ "$maxb" -lt 0 ] || [ "${bs[$k]}" -gt "$maxb" ]; }; then maxb="${bs[$k]}"; fi
        vram_used=$(( vram_used + ${us[$k]} )); vram_total=$(( vram_total + ${ts_[$k]} ))
    done
    [ "$maxb" -ge 0 ] || maxb=-1

    local avail free shmem mlocked rss vmlck pids load rocm_busy
    meminfo() { awk -v k="$1:" '$1 == k { printf "%d", $2 / 1024 }' /proc/meminfo; }
    avail=$(meminfo MemAvailable)
    free=$(meminfo MemFree)
    shmem=$(meminfo Shmem)
    mlocked=$(meminfo Mlocked)
    rss=0; vmlck=0; pids=""
    for p in $(pgrep -f 'llama-(cli|server|bench)' 2>/dev/null | head -8); do
        r=$(awk '/^VmRSS:/ { printf "%d", $2 / 1024 }' "/proc/$p/status" 2>/dev/null)
        l=$(awk '/^VmLck:/ { printf "%d", $2 / 1024 }' "/proc/$p/status" 2>/dev/null)
        rss=$(( rss + ${r:-0} ))
        vmlck=$(( vmlck + ${l:-0} ))
        pids="${pids:+$pids+}$p"
    done
    load=$(awk '{ print $1 }' /proc/loadavg)

    # rocm-smi every N ticks only: it opens the driver, sysfs does not.  It is also the only source that works
    # while the R9700s are runtime-suspended, which is why the summary cross-checks against it.
    if [ "$ROCM_EVERY" -gt 0 ] && [ $(( tick % ROCM_EVERY )) -eq 0 ]; then
        rocm_busy=$(timeout 10 rocm-smi --showuse --csv 2>/dev/null | awk -F, '$1 ~ /^card[0-9]+$/ { print $2 }' | paste -sd'/' -)
        [ -n "$rocm_busy" ] || rocm_busy="?"
    else
        rocm_busy="-"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$ts" "$core_s" "$maxb" "$vram_used" "$vram_total" "$avail" "$free" "$shmem" "$mlocked" \
        "$rss" "$vmlck" "${pids:--}" "$load" "$rocm_busy" "${bs[*]}" \
        "$(printf '%s' "$NOTE" | tr ',' ';')" >> "$out"
}

HEADER='ts,core_s,gpu_busy_pct,vram_used_mib,vram_total_mib,mem_avail_mib,mem_free_mib,shmem_mib,mlocked_mib,llama_rss_mib,llama_vmlck_mib,llama_pids,load1,rocm_busy,per_card_busy,note'

# ---- run ------------------------------------------------------------------------------------------------
: > "$OUT"
printf '%s\n' "$HEADER" >> "$OUT"
echo "gpu-watch: writing $OUT (interval ${INTERVAL}s, rocm every ${ROCM_EVERY})" >&2

T0=$(date +%s)
TICK=0
CHILD=""

stop_child() {
    [ -n "$CHILD" ] || return 0
    kill -TERM "$CHILD" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$CHILD" 2>/dev/null || { CHILD=""; return 0; }; sleep 0.5; done
    kill -KILL "$CHILD" 2>/dev/null
    CHILD=""
}

on_signal() {
    echo "gpu-watch: signal received, stopping the workload" >&2
    stop_child
    sample "$(( $(date +%s) - T0 ))" "$TICK" "$OUT"
    sync -d "$OUT" 2>/dev/null
    echo "gpu-watch: $(wc -l < "$OUT") lines in $OUT" >&2
    exit 130
}
trap on_signal INT TERM

if [ -n "$CMD" ]; then
    # One workload, bounded.  SIGKILL 10 s after SIGTERM so a driver wait can never hold the whole arm.
    ( eval "exec timeout --kill-after=10 --signal=TERM $CMD_TIMEOUT $CMD" ) &
    CHILD=$!
fi

DEADLINE=""
[ -n "$DUR" ] && DEADLINE=$(( T0 + DUR ))

while :; do
    sample "$(( $(date +%s) - T0 ))" "$TICK" "$OUT"
    if [ $(( TICK % 5 )) -eq 0 ]; then sync -d "$OUT" 2>/dev/null; fi
    TICK=$(( TICK + 1 ))

    if [ -n "$CHILD" ]; then
        kill -0 "$CHILD" 2>/dev/null || break
    elif [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
        break
    fi
    sleep "$INTERVAL"
done

if [ -n "$CHILD" ]; then
    wait "$CHILD"
    RC=$?
    echo "gpu-watch: workload exited rc=$RC after $(( $(date +%s) - T0 ))s" >&2
else
    RC=0
fi

sample "$(( $(date +%s) - T0 ))" "$TICK" "$OUT"
sync -d "$OUT" 2>/dev/null
# (the post-loop sample above captures the exit state; with --dur it can repeat the last one, which is harmless)
echo "gpu-watch: $(wc -l < "$OUT") lines in $OUT" >&2
"$(dirname "${BASH_SOURCE[0]}")/gpu-watch.sh" --summary "$OUT" >&2

# Kernel-log snapshot for the run's window: the cheap half of a fault check, so a wedge's last seconds are on
# disk even when journald never flushed.  The authoritative check is still `check-gpu-faults.sh --since`.
LOG="${OUT%.csv}.dmesg"
timeout 60 journalctl -k --since "@$(( T0 - 5 ))" --no-pager > "$LOG" 2>/dev/null
echo "gpu-watch: kernel log for the run window -> $LOG ($(wc -l < "$LOG" 2>/dev/null || echo 0) lines)" >&2
exit "$RC"
