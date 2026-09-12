#!/usr/bin/env bash
# qwen4exp n_max sweep: plain vs draft-mtp n_max 1/2/3/5/7, same-seed text hashes + first divergence.
#   KV=<type> [TXT=<prompt>] [BIN=...] [OUT=...] bash nmax.sh
set -u
RDNA=${RDNA:-/home/stew675/llama-cpp-rdna-boosts}
BIN=${BIN:-/home/stew675/ll25/verify/build-rocm/bin}
TG=${TG:-$RDNA/wip/kv-quant-purity-followups/tools/textgen.py}
M=${M:-/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
D=${D:-/llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf}
TXT=${TXT:-$RDNA/wip/strix-halo/qsa-item4/p5000.txt}
KV=${KV:-q8_0}
OUT=${OUT:-/tmp/qsa-item4}; mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:${LD_LIBRARY_PATH:-}
run() { local tag=$1; shift
  timeout 2400 "$BIN/llama-cli" -m "$M" "$@" -f "$TXT" -n 128 --seed 42 --temp 0 \
    --single-turn --no-display-prompt -c 8192 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa auto \
    -ngl all -sm layer -mg 0 > "$OUT/n-$tag.log" 2>&1
  printf '  %-8s %s\n' "$tag" "$(python3 $TG "$OUT/n-$tag.log" | tail -1)"
}
run plain --spec-type none
for nm in 1 2 3 5 7; do run n$nm --spec-type draft-mtp -md "$D" --spec-draft-n-max $nm; done
echo "== first divergence (chars) vs plain =="
python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
p = open(f'{out}/n-plain.log.txt').read()
for f in ['n-n1','n-n2','n-n3','n-n5','n-n7']:
    t = open(f'{out}/{f}.log.txt').read()
    n = min(len(p), len(t)); i = 0
    while i < n and p[i] == t[i]: i += 1
    print(f"  {f[2:]:4s} len={len(t):4d} firstdiff={i if i < n else 'none (pure)'}")
PY
