#!/usr/bin/env bash
# qwen4exp plain-vs-MTP text gate (TODO item 4 + the QSA width work).
#   KV=<type> [TAG=<tag>] [OUT=<dir>] [TXT=<prompt>] [BIN=<built bin dir>] [NG=<tokens>] bash gate.sh
# The *sparse* regime can be forced with LLAMA_QSA_DENSE_DECODE_UNTIL=0 (item 4's repro config).
set -u
RDNA=${RDNA:-/home/stew675/llama-cpp-rdna-boosts}
BIN=${BIN:-/home/stew675/ll25/verify/build-rocm/bin}
TG=${TG:-$RDNA/wip/kv-quant-purity-followups/tools/textgen.py}
M=${M:-/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
D=${D:-/llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf}
TXT=${TXT:-$RDNA/wip/strix-halo/qsa-item4/p5000.txt}
NG=${NG:-128}
KV=${KV:-f16}
CTX=${CTX:-8192}
FA=${FA:-auto}
OUT=${OUT:-/tmp/qsa-item4}
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:${LD_LIBRARY_PATH:-}

tex() { # tag spec...
  local tag=$1; shift
  timeout 2400 "$BIN/llama-cli" -m "$M" "$@" -f "$TXT" -n "$NG" --seed 42 --temp 0 \
    --single-turn --no-display-prompt -c "$CTX" -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa "$FA" \
    -ngl all -sm layer -mg 0 > "$OUT/t-$tag.log" 2>&1
  printf '  %-28s %s\n' "$tag" "$(python3 $TG "$OUT/t-$tag.log" | tail -1)"
}

TAG=${TAG:-def}
echo "== config: $TAG (KV=$KV CTX=$CTX DENSE_DECODE_UNTIL=${LLAMA_QSA_DENSE_DECODE_UNTIL:-unset}) =="
tex "$TAG-plain" --spec-type none
tex "$TAG-n3"    --spec-type draft-mtp -md "$D" --spec-draft-n-max 3
