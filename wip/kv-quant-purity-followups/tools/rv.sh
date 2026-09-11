#!/usr/bin/env bash
# Block-15 revalidation driver (2026-09-11: re-cut against the 15-block delivery).
#   BIN   = binary dir under test (default /tmp/bin-blk15 = block 15)
#   PROBE = width-probe binary   (default /tmp/lw-blk15)
# Extra args of the form KEY=VAL (no leading dash) are exported for that run;
# anything else is passed through to the binary.  So:
#   rv.sh res b15-v3off 27b 2048 LLAMA_KQ_MASK_DERIVED=0
#
# Subcommands:
#   res   <tag> <model> <ub> [KEY=VAL...] [cli args...]   reserve (device compute / host / KV)
#   kv    <tag> <model> <ub> <ctk> <ctv> [KEY=VAL...]     reserve with explicit KV types
#   coh   <outfile> <model> <ngen> [prompt] [KEY=VAL...]  coherence text (same-seed)
#   mtp   <tag> <27bm|q4m|moem> [ngen] [KEY=VAL...]       adaptive-MTP gate
#   bench <model> <ub> <p> <n> <r> <Aenv> <Benv>          interleaved llama-bench A/B
#   width <model> <split> <devices> <wlist> [rs] [fa] [KEY=VAL...]
set -u
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=${BIN:-/tmp/bin-blk15}
PROBE=${PROBE:-/tmp/lw-blk15}

mpath() { case $1 in
    4b)  echo /home/stew675/Qwen3.5-4B-Q8_0.gguf ;;
    27b) echo /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf ;;
    g4e) echo /llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf ;;
    g31) echo /llm/models/Gemma4/31B-QAT/Q4_K_XL/gemma-4-31B-it-qat-Q4_K_XL.gguf ;;
    q4)  echo /models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf ;;
    moe) echo /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ;;
    *)   echo "unknown model $1" >&2; return 2 ;;
esac; }
# device sets mirror the 2026-09-10 validation
mdev() { case $1 in 4b|g4e) echo 0 ;; *) echo 0,1,2 ;; esac; }

# split remaining args into ENVS (KEY=VAL) and REST
split_args() { ENVS=(); REST=(); for a in "$@"; do case $a in [A-Za-z_]*=*) ENVS+=("$a");; *) REST+=("$a");; esac; done; }

parse_res() { # <log> -> device / host / model / ctx / kvbuf
    local log=$1 dev host mb kvb
    dev=$(grep -oE '^.*(ROCm[0-9]|Meta\(\)) compute buffer size is +[0-9.]+ MiB' "$log" | tail -1 | grep -oE '[0-9.]+ MiB' | grep -oE '[0-9.]+')
    host=$(grep -oE 'ROCm_Host compute buffer size is +[0-9.]+ MiB' "$log" | tail -1 | grep -oE '[0-9.]+ MiB' | grep -oE '[0-9.]+')
    mb=$(sed -nE 's/.*\((ROCm[0-9]|Meta\(\)) \(.*\) *\|[^=]*= *[0-9]+ \+ +\(([0-9]+) = +([0-9]+) \+ +([0-9]+) \+ +([0-9]+)\).*/model=\3 ctx=\4 compute=\5/p' "$log" | tail -1)
    kvb=$(grep -oE 'KV buffer size = +[0-9.]+ MiB' "$log" | tail -1 | grep -oE '[0-9.]+')
    echo "dev=${dev:-?} host=${host:-?} kvbuf=${kvb:-?} $mb"
}

cmd=${1:?}; shift
case $cmd in
res|kv)
    TAG=$1; MK=$2; UB=$3; shift 3
    M=$(mpath "$MK") || exit 2
    DEV=$(mdev "$MK")
    CTK=q8_0; CTV=q8_0
    if [ "$cmd" = kv ]; then CTK=$1; CTV=$2; shift 2; fi
    split_args "$@"
    log=/tmp/rv-res-$TAG-$MK-$UB.log
    export HIP_VISIBLE_DEVICES=$DEV
    env "${ENVS[@]+"${ENVS[@]}"}" timeout 3600 "$BIN/llama-cli" -m "$M" -f /tmp/tiny.txt \
      -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub "$UB" -fa auto -ctk "$CTK" -ctv "$CTV" \
      -n 1 --seed 42 --temp 0 -v --single-turn --no-display-prompt "${REST[@]+"${REST[@]}"}" > "$log" 2>&1
    printf '[res %-22s %-4s ub=%-5s k=%-5s v=%-5s dev=%s] ' "$TAG" "$MK" "$UB" "$CTK" "$CTV" "$DEV"
    parse_res "$log"; ;;
coh)
    OUT=$1; MK=$2; NGEN=$3; shift 3
    PROMPT=/tmp/v4p.txt
    if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then PROMPT=$1; shift; fi
    M=$(mpath "$MK") || exit 2
    DEV=$(mdev "$MK")
    split_args "$@"
    export HIP_VISIBLE_DEVICES=$DEV
    env "${ENVS[@]+"${ENVS[@]}"}" timeout 3600 "$BIN/llama-cli" -m "$M" -f "$PROMPT" \
      -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub 2048 -fa auto -ctk q8_0 -ctv q8_0 \
      -n "$NGEN" --seed 42 --temp 0 --single-turn --no-display-prompt 2>/dev/null \
      | tr -d '\r' | grep -v '^Loading model' | grep -v '^Prompt:' > "$OUT" || true
    printf '[coh %-4s %-22s] %s bytes sha=%s\n' "$MK" "$(basename "$OUT")" "$(wc -c < "$OUT")" "$(sha256sum "$OUT" | cut -c1-12)"; ;;
mtp)
    TAG=$1; MK=$2; NGEN=${3:-96}; shift 2; [ $# -gt 0 ] && shift
    DEV=0,1,2; SPEC="--spec-type draft-mtp"
    case $MK in
      27bm) M=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf; D="" ;;
      q4m)  M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
            D="-md /models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" ;;
      moem) M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; DEV=0 ;;
    esac
    split_args "$@"
    export HIP_VISIBLE_DEVICES=$DEV
    log=/tmp/rv-mtp-$TAG.log
    if [ "$MK" = moem ]; then
        # Protocol A (mtp-adaptive-methodology.md), identical to the 2026-09-10/11 MoE gate
        PP="Write a detailed technical explanation of how a modern GPU memory hierarchy works, covering caches, coalescing, and bandwidth, then discuss how tensor cores change the arithmetic intensity of matrix multiplication."
        env "${ENVS[@]+"${ENVS[@]}"}" timeout 3600 "$BIN/llama-cli" --model "$M" --top-k 20 --threads 8 --parallel 1 \
          --top-p 0.95 --min-p 0.001 --predict "$NGEN" --cache-ram 16384 --ctx-size 8192 --flash-attn auto \
          --temperature 0.0 --batch-size 512 --ubatch-size 512 --n-gpu-layers all --cache-type-k f16 \
          --cache-type-v f16 --repeat-penalty 1.0 --presence-penalty 1.5 --seed 42 --single-turn $SPEC \
          --spec-draft-n-max 3 --log-verbosity 4 --prompt "$PP" > "$log" 2>&1
    else
        env "${ENVS[@]+"${ENVS[@]}"}" timeout 3600 "$BIN/llama-cli" -m "$M" $D $SPEC \
          -f /tmp/prompt3k.txt -n "$NGEN" --seed 42 --temp 0 --single-turn --no-display-prompt \
          -c 32768 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto -ngl all -sm tensor -mg 0 \
          --log-verbosity 4 > "$log" 2>&1
    fi
    printf '[mtp %-12s %-5s n=%-4s] %s\n' "$TAG" "$MK" "$NGEN" "$(grep -a 'draft acceptance' "$log" | tail -1 | sed 's/.*draft acceptance/draft acceptance/')"; ;;
bench)
    MK=$1; UB=$2; P=$3; N=$4; R=$5; A=$6; B=$7
    M=$(mpath "$MK") || exit 2
    export HIP_VISIBLE_DEVICES=${RV_DEV:-$(mdev "$MK")}
    run() { env $1 "$BIN/llama-bench" -m "$M" -p "$P" -n "$N" -r "$R" -b 2048 -ub "$UB" \
        -ctk q8_0 -ctv q8_0 -fa on -ngl 99 -sm tensor --output csv 2>/dev/null | python3 /tmp/b15parse.py; }
    for i in $(seq "$R"); do
        printf 'A%s(%s) ' "$i" "$A"; run "$A"
        printf 'B%s(%s) ' "$i" "$B"; run "$B"
    done; ;;
width)
    MK=$1; SPLIT=$2; DEVICES=$3; WLIST=$4; shift 4
    RS=0; FA=auto
    if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then RS=$1; shift; fi
    if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then FA=$1; shift; fi
    M=$(mpath "$MK") || exit 2
    split_args "$@"
    TXT=/home/stew675/llama-cpp-rdna-boosts/wip/sm-tensor-plain-vs-spec/p0long.txt
    for w in $WLIST; do
        out=$(env "${ENVS[@]+"${ENVS[@]}"}" W="$w" NGL=99 SPLIT="$SPLIT" RS="$RS" FA="$FA" CB=0 \
              HIP_VISIBLE_DEVICES="$DEVICES" "$PROBE" "$M" "$TXT" 256 512 2>/dev/null \
              | grep '^\[L\]' | sed 's/.*hash=\([0-9a-f]*\).*/\1/')
        printf '  W=%-3s %s\n' "$w" "${out:-FAILED}"
    done; ;;
*) echo "unknown subcommand $cmd" >&2; exit 2 ;;
esac
