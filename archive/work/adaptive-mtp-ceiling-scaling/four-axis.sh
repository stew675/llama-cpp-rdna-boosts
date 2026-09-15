#!/bin/bash
# Four-axis adaptive-MTP gate (-n 3000) for issue #35 table tuning.
#
# Runs R/P/C/K x {configs} on one model/layout and prints t/s + acceptance +
# mean accepted length + the time-weighted adaptive depth, then the constraint
# check vs the fixed n3 column:
#   R <= 1.03 x fixed3 (reasoning no more than 3% worse)
#   P >= fixed3
#   C >= 1.10 x fixed3 (code ideally >= 10% better)
#   K: must be able to climb to a depth of 12 quickly
#
# Usage: ./four-axis.sh <tag> [extra llama-cli flags...]
#   env: LLAMA_BIN MODEL PROMPTS DEV OUT SPECS LV
#   SPECS default: "draft-mtp:3 draft-mtp-adaptive:7 draft-mtp-adaptive:12"
set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-7.14-gfx1201/lib}
LLAMA_BIN=${LLAMA_BIN:-/home/stew675/llama-cpp-rebase/build-rocm/bin/llama-cli}
MODEL=${MODEL:-/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf}
PROMPTS=${PROMPTS:-/home/stew675/llama-cpp-rdna-boosts/prompts}
DEV=${DEV:-1,2}
OUT=${OUT:-/tmp/four-axis}
SPECS=${SPECS:-"draft-mtp:3 draft-mtp-adaptive:7 draft-mtp-adaptive:12"}
LV=${LV:-4}

tag=${1:-run}; shift || true
extra=("$@")
mkdir -p "$OUT"

for axis in reasoning prose-rdna-boosts code-python recall; do
  case "$axis" in reasoning) REA=on;; *) REA=off;; esac
  for spec in $SPECS; do
    type=${spec%%:*}; n=${spec##*:}
    key="$tag-${axis%%-*}-${type#draft-mtp}-$n"
    HIP_VISIBLE_DEVICES="$DEV" timeout 1800 "$LLAMA_BIN" -m "$MODEL" -f "$PROMPTS/$axis.txt" \
      --reasoning "$REA" -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv "$LV" \
      --spec-type "$type" --spec-draft-n-max "$n" "${extra[@]}" \
      > "$OUT/$key.out" 2> "$OUT/$key.err"
    tg=$(tr -d '\b\r' < "$OUT/$key.out" | grep -oE "Generation: *[0-9.]+ t/s" | tail -1 | grep -oE "[0-9.]+")
    acc=$(grep -oE "draft acceptance = [0-9.]+" "$OUT/$key.err" | tail -1 | awk '{print $4}')
    ml=$(grep -oE "mean len = *[0-9.]+" "$OUT/$key.err" | tail -1 | grep -oE "[0-9.]+")
    md=$(python3 - "$OUT/$key.err" <<'PY'
import re,sys
tr=re.compile(r'adaptive draft depth seq \d+: (\d+) -> (\d+)'); ts=re.compile(r'^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ')
s=re.compile(r'(?:^| )(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ')
def sec(m): return int(m.group(1))*60+int(m.group(2))+int(m.group(3))/1e3+int(m.group(4))/1e6
cur=3; last=None; end=None; w=0.0; wt=0.0
for line in open(sys.argv[1],errors='replace'):
    m=ts.match(line)
    if not m: continue
    end=sec(m)
    if 'adaptive draft depth enabled' in line: cur,last=3,end; continue
    t=tr.search(line)
    if t:
        if last is None: last,cur=end,3
        dt=max(0.0,end-last); w+=dt*cur; wt+=dt; cur,last=int(t.group(2)),end
if last is not None:
    dt=max(0.0,(end or last)-last); w+=dt*cur; wt+=dt
print(f'{w/wt:.2f}' if wt else '-')
PY
)
    printf '%-34s tg=%-6s acc=%-9s meanlen=%-6s meandepth=%s\n' "$key" "${tg:-NA}" "${acc:-none}" "${ml:-0}" "$md"
  done
done
echo "ALLDONE"
