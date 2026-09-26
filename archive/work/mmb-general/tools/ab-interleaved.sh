#!/bin/bash
# Interleaved delivery-vs-WIP A/B for llama.cpp on this box (gfx1201, 3x R9700).
#
# Usage:
#   ab-interleaved.sh <label> <gpulist> <model> <rounds> [--wip-env ENV=V] -- <llama-bench args...>
#
# Examples:
#   # qwen4exp (the campaign's headline model), deep prefill, 3-GPU tensor
#   ab-interleaved.sh "qwen4exp deep" 0,1,2 \
#       /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf 2 \
#       -- -sm tensor -ctk q8_0 -ctv q8_0 -fa auto -p 32768,65536,98304 -n 0 -r 2
#
#   # dense 27B with the WIP master switch on
#   ab-interleaved.sh "27B IQ3_S" 0 /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf 3 \
#       --wip-env GGML_CUDA_MMB=1 -- -p 8192,32768 -n 0 -r 3
#
#   # decode at depth (B4-style): "-d 0,16384" gives tg128 and "tg128 @ d16384"
#   ab-interleaved.sh "decode" 0 <model> 3 -- -p 0 -n 128 -d 0,16384 -r 3
#
# Why interleaved, and why in ONE warm session: the first prefill test of an invocation is
# cold-start-limited (~-9 %) and the clock ramps on the first compute-dense run.  On 3-GPU
# qwen4exp the same config varies +/-3 % at pp8192 but only +/-0.1-0.4 % at pp65536/98304,
# so **prefer depth for a verdict** and treat a shallow-only delta as +/-2 %.  NEVER run
# two benches at once.
#
# Binaries default to the delivery (`~/llama-base`) and the WIP (`~/llama.cpp`); override
# with BASE_BIN=... WIP_BIN=...  The WIP side is the one that gets --wip-env.
set -u

HERE="$(dirname "$(readlink -f "$0")")"

if [ $# -lt 5 ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

LABEL="$1"; GPUS="$2"; MODEL="$3"; ROUNDS="$4"; shift 4
WIPENV=""
while [ $# -gt 0 ]; do
    case "$1" in
        --wip-env) WIPENV="$2"; shift 2 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done

BASE_BIN="${BASE_BIN:-$HOME/llama-base/build-rocm/bin/llama-bench}"
WIP_BIN="${WIP_BIN:-$HOME/llama.cpp/build-rocm/bin/llama-bench}"
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx102X/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES="$GPUS"

for b in "$BASE_BIN" "$WIP_BIN"; do
    [ -x "$b" ] || { echo "missing binary: $b" >&2; exit 1; }
done

echo "### $LABEL"
echo "#   delivery: $BASE_BIN"
echo "#   WIP     : $WIP_BIN ${WIPENV:+[$WIPENV]}"
echo "#   args    : $*"

ROWS=$(mktemp)
trap 'rm -f "$ROWS"' EXIT

for r in $(seq 1 "$ROUNDS"); do
    off=$(timeout 7200 "$BASE_BIN" -m "$MODEL" -ngl 99 -b 2048 -ub 2048 "$@" 2>/dev/null | python3 "$HERE/lbparse.py")
    on=$(env GGML_CUDA_MMB=1 $WIPENV timeout 7200 "$WIP_BIN" -m "$MODEL" -ngl 99 -b 2048 -ub 2048 "$@" 2>/dev/null | python3 "$HERE/lbparse.py")
    echo "  r$r delivery: $off"
    echo "  r$r WIP     : $on"
    for kv in $off; do echo "B ${kv%%=*} ${kv#*=}" >> "$ROWS"; done
    for kv in $on;  do echo "W ${kv%%=*} ${kv#*=}" >> "$ROWS"; done
done

echo "  --- mean over $ROUNDS round(s) ---"
python3 - "$ROWS" <<'PY'
import collections, sys
d = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(sys.argv[1]):
    arm, key, val = line.split()
    d[key][arm].append(float(val))
for key in sorted(d, key=lambda k: (k[:2], int(''.join(c for c in k if c.isdigit()) or 0))):
    b, w = d[key].get('B'), d[key].get('W')
    if not b or not w:
        print(f"  {key:>15}  incomplete (delivery={b}, WIP={w})")
        continue
    mb, mw = sum(b)/len(b), sum(w)/len(w)
    spread_b = (max(b)-min(b))/mb*100 if mb else 0.0
    print(f"  {key:>15}  delivery={mb:9.2f} (+/-{spread_b:4.1f} %)  WIP={mw:9.2f}  delta={(mw/mb-1)*100:+6.2f} %")
PY
