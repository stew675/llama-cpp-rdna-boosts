#!/bin/bash
# Depth-trajectory helper for the adaptive-MTP controller (issue #35 WIP).
#
# Runs ONE cell and prints the tg footer, the acceptance / mean-len summary, and a
# TIME-WEIGHTED histogram of the adaptive draft depth (which depth the controller
# actually sits at, and for how long).  The transition line is temporarily logged
# at TRC (level 4) while investigating, so -lv 4 shows it -- with the shipped DBG
# line it needs -lv 5 and floods.
#
# Usage: ./depth-trace.sh <tag> <n_max> [extra llama-cli flags...]
#   env: LLAMA_BIN, Q8, PROMPTS, OUT, DEV2, LV
set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-7.14-gfx1201/lib}
LLAMA_BIN=${LLAMA_BIN:-/home/stew675/llama-cpp-rebase/build-rocm/bin/llama-cli}
Q8=${Q8:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}
PROMPTS=${PROMPTS:-/home/stew675/llama-cpp-rdna-boosts/prompts}
OUT=${OUT:-/tmp/depth-trace}
DEV2=${DEV2:-1,2}
FLOOR=${FLOOR:-3}
LV=${LV:-4}

tag=$1; n=$2; shift 2
mkdir -p "$OUT"
HIP_VISIBLE_DEVICES="$DEV2" timeout 1800 "$LLAMA_BIN" -m "$Q8" -n 3000 \
  --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
  -p "$(cat "$PROMPTS/code-python.txt")" --spec-type draft-mtp-adaptive --spec-draft-n-max "$n" \
  -c 32768 -b 2048 -ub 2048 -fa auto -ngl 99 -lv "$LV" "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"

tg=$(tr -d '\b\r' < "$OUT/$tag.out"  | grep -oE "Generation: *[0-9.]+ t/s" | tail -1 | grep -oE "[0-9.]+")
pp=$(tr -d '\b\r' < "$OUT/$tag.out"  | grep -oE "Prompt: *[0-9.]+ t/s"     | tail -1 | grep -oE "[0-9.]+")
acc=$(grep -oE "draft acceptance = [0-9.]+" "$OUT/$tag.err" | tail -1 | awk '{print $4}')
ml=$(grep -oE "mean len = *[0-9.]+"         "$OUT/$tag.err" | tail -1 | grep -oE "[0-9.]+")
printf '[%s] cap=%s  tg=%s t/s  pp=%s  acc=%s  meanlen=%s\n' "$tag" "$n" "${tg:-NA}" "${pp:-NA}" "${acc:-none}" "${ml:-0}"

python3 - "$OUT/$tag.err" "$FLOOR" <<'PY'
import re, sys
path, floor = sys.argv[1], int(sys.argv[2])
ts_re = re.compile(r'^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ')
tr_re = re.compile(r'adaptive draft depth seq \d+: (\d+) -> (\d+)')
def sec(m):
    return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3))/1e3 + int(m.group(4))/1e6
cur = floor; last = None; end = None
hist = {}
with open(path, errors='replace') as f:
    for line in f:
        m = ts_re.match(line)
        if not m:
            continue
        end = sec(m)
        if 'adaptive draft depth enabled' in line:
            cur, last = floor, end
            continue
        t = tr_re.search(line)
        if t:
            if last is None:
                last, cur = end, floor
            hist[cur] = hist.get(cur, 0.0) + max(0.0, end - last)
            cur, last = int(t.group(2)), end
if last is not None:
    hist[cur] = hist.get(cur, 0.0) + max(0.0, (end or last) - last)
tot = sum(hist.values()) or 1.0
print(f'  depth histogram (time-weighted, {tot:.1f}s of verify rounds):')
for d in sorted(hist):
    print(f'    depth {d:>2}: {hist[d]/tot*100:5.1f}%  ({hist[d]:6.1f}s)')
if not hist:
    print('    (no transition lines found -- controller never changed depth)')
PY
