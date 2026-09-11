#!/usr/bin/env bash
# F3 driver: KV-type-parameterized purity/perf gates for the newly enabled FA types.
#   f3run.sh text  <tag> <mk> <kv> <none|mtp> <nmax> [ngen]
#   f3run.sh mtp   <tag> <mk> <kv> <nmax> [ngen]        (acceptance, --log-verbosity 4)
#   f3run.sh width <mk> <split> <devices> <kv> <wlist> [rs]
set -u
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/canon-llama/build-base/bin}
PROBE=${PROBE:-/tmp/lw-f2}
TG=/home/stew675/llama-cpp-rdna-boosts/wip/kv-quant-purity-followups/tools/textgen.py
TXT=/home/stew675/llama-cpp-rdna-boosts/wip/sm-tensor-plain-vs-spec/p0long.txt
mpath() { case $1 in
    4b)  echo /home/stew675/Qwen3.5-4B-Q8_0.gguf ;;
    g4e) echo /llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf ;;
    27b) echo /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf ;;
    q4)  echo /models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf ;;
    moe) echo /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ;;
    *) echo "unknown model $1" >&2; return 2 ;;
esac; }
mdev() { case $1 in 4b|moe|g4e) echo 0 ;; *) echo 0,1,2 ;; esac; }
msplit() { case $1 in 4b|g4e) echo layer ;; *) echo tensor ;; esac; }
mdraft() { case $1 in
    q4) echo "-md /models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" ;;
    *)  echo "" ;;
esac; }

cmd=$1; shift
case $cmd in
text)
    TAG=$1; MK=$2; KV=$3; SPEC=$4; NMAX=${5:-3}; NGEN=${6:-128}
    M=$(mpath "$MK"); DEV=$(mdev "$MK"); SM=$(msplit "$MK")
    SPECARG="--spec-type none"; [ "$SPEC" = mtp ] && SPECARG="--spec-type draft-mtp $(mdraft $MK) --spec-draft-n-max $NMAX"
    log=/tmp/f3/text-$TAG.log
    HIP_VISIBLE_DEVICES=$DEV timeout 3600 "$BIN/llama-cli" -m "$M" $SPECARG \
      -f /tmp/prompt3k.txt -n "$NGEN" --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm "$SM" -mg 0 > "$log" 2>&1
    printf '%-28s kv=%-5s spec=%-4s nmax=%s -> %s  [%s]\n' "$TAG" "$KV" "$SPEC" "$NMAX" "$(python3 $TG $log)" \
      "$(grep -ao 'Generation: *[0-9.]* t/s' "$log" | tail -1)"; ;;
mtp)
    TAG=$1; MK=$2; KV=$3; NMAX=${4:-3}; NGEN=${5:-96}
    M=$(mpath "$MK"); DEV=$(mdev "$MK"); SM=$(msplit "$MK")
    log=/tmp/f3/mtp-$TAG.log
    HIP_VISIBLE_DEVICES=$DEV timeout 3600 "$BIN/llama-cli" -m "$M" $(mdraft $MK) --spec-type draft-mtp \
      --spec-draft-n-max "$NMAX" -f /tmp/prompt3k.txt -n "$NGEN" --seed 42 --temp 0 --single-turn \
      --no-display-prompt -c 32768 -b 2048 -ub 2048 -ctk "$KV" -ctv "$KV" -fa auto -ngl all -sm "$SM" \
      -mg 0 --log-verbosity 4 > "$log" 2>&1
    printf '%-16s kv=%-5s nmax=%s n=%s | %s | %s\n' "$TAG" "$KV" "$NMAX" "$NGEN" \
      "$(grep -a 'draft acceptance' "$log" | tail -1 | sed 's/.*draft acceptance/draft acceptance/')" \
      "$(grep -a 'Generation:' "$log" | tail -1 | sed 's/.*Generation: *//')"; ;;
width)
    MK=$1; SPLIT=$2; DEVICES=$3; KV=$4; WLIST=$5; RS=${6:-0}
    M=$(mpath "$MK")
    out=""
    for w in $WLIST; do
        h=$(HIP_VISIBLE_DEVICES=$DEVICES W="$w" NGL=99 SPLIT="$SPLIT" RS="$RS" CB=0 CTK="$KV" CTV="$KV" \
            "$PROBE" "$M" "$TXT" 256 2>/dev/null | grep '^\[L\]' | sed 's/.*logits0_hash=\([0-9a-f]*\).*/\1/')
        out="$out $w:${h:-FAILED}"
    done
    printf '%-4s %-6s rs=%-7s %-5s: %s\n' "$MK" "$SPLIT" "$RS" "$KV" "$out"; ;;
bench)
    MK=$1; KV=$2; NPL=$3; NPP=$4; NTG=$5; REP=${6:-1}
    M=$(mpath "$MK"); DEV=$(mdev "$MK"); SM=$(msplit "$MK")
    for i in $(seq "$REP"); do
        out=$(HIP_VISIBLE_DEVICES=$DEV timeout 3000 "$BIN/llama-batched-bench" -m "$M" -c 32768 -b 2048 -ub 512 \
          -npp "$NPP" -ntg "$NTG" -npl "$NPL" -ctk "$KV" -ctv "$KV" -fa auto -ngl 99 -sm "$SM" -mg 0 \
          --output-format jsonl 2>/dev/null | python3 -c '
import sys, json
rows=[]
for line in sys.stdin:
    line=line.strip()
    if not line or not line.startswith("{"): continue
    d=json.loads(line)
    rows.append((d["pl"], d["speed_pp"], d["speed_tg"]))
print(" ".join("%d:pp%7.1f/tg%7.2f" % r for r in rows))')
        printf '%-4s kv=%-5s (%d) %s\n' "$MK" "$KV" "$i" "$out"
    done; ;;
*) echo "unknown subcommand $cmd" >&2; exit 2 ;;
esac
