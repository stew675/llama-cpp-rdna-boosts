#!/bin/bash
# run-all.sh - full v2 benchmark run: PPL accuracy leg, then the 16-row
# llama-benchy throughput suite. Logs with timestamps to
# benchmarks/results/run-all.log (launch with nohup; safe to check in on).
#
# usage: run-all.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root
HERE="$ROOT/benchmarks/scripts"
RES="$ROOT/benchmarks/results"
mkdir -p "$RES/benchy" "$RES/ppl"

MASTER_BIN="$HOME/llama.cpp/build-rocm/bin"
BOOSTS_BIN="$HOME/llama.cpp/build-rdna-boosts/bin"
WIKI="/llm/models/wikitext-2-raw/wiki.test.raw"
Q8="/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf"
Q6="/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf"
Q4="/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"

LOG="$RES/run-all.log"
: > "$LOG"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "=== run-all start ==="
say "GPU check: $(rocm-smi --showmeminfo vram 2>/dev/null | grep Used | tr '\n' ' ')"

# ---------- Phase 1: PPL (3 models x 4 corners) ----------
for spec in "Q8_0:$Q8" "Q6_K:$Q6" "Q4_K_XL:$Q4"; do
    TAG="${spec%%:*}"; MODEL="${spec#*:}"
    say "PPL [$TAG] starting"
    "$HERE/run-ppl.sh" "$MASTER_BIN" "$BOOSTS_BIN" "$MODEL" "$WIKI" 2>&1 | tee -a "$LOG" \
        | tee "$RES/ppl/ppl-$TAG.txt" > /dev/null
    RC=${PIPESTATUS[0]}
    say "PPL [$TAG] done rc=$RC -> $RES/ppl/ppl-$TAG.txt"
    if [ $RC -ne 0 ]; then say "!! PPL [$TAG] failed — continuing anyway"; fi
done

# ---------- Phase 2: llama-benchy throughput suite (16 rows) ----------
say "throughput suite starting (16 rows)"
"$HERE/run-benchy-suite.sh" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
say "throughput suite done rc=$RC"

say "=== run-all complete ==="
say "results: $(ls $RES/benchy/*.json 2>/dev/null | wc -l) json, $(ls $RES/benchy/*.md 2>/dev/null | wc -l) md"
exit 0
