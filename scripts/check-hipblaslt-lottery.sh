#!/usr/bin/env bash
# Is the per-start hipBLASLt SOLUTION LOTTERY present on this ROCm line?  (GREEDY-PURITY.md §44, TODO #362)
#
# WHY THIS EXISTS
#   §44 tells every tester to pin `ROCBLAS_USE_HIPBLASLT=0` before comparing fresh processes, and `README.md`
#   says the pin is only *needed* below the `rocm-libraries` fix `c50ea60130f`, which ships as hipBLASLt 1.5 =
#   **ROCm 10.2**.  That mapping was never measured here, and a one-off arm cannot express it: on this box the
#   two recorded call sites (`ffn_gate_inp` F32 router, GDN F32 `ssm_alpha`/`ssm_beta`) read ONE logits class
#   over dozens of fresh starts on BOTH lines, i.e. the arms are stable and therefore say nothing about the
#   mechanism (that is §44's own rule: stability is config-dependent).
#
# WHAT IT MEASURES (two arms, N fresh starts each, same build, same model)
#   A. THE DELIVERY DEFAULT.  A prefill-logits fingerprint per fresh start (`llama-perplexity`'s
#      `--kl-divergence-base` dump is a function of the logits only, so its sha256 is a logits hash) plus the
#      hipBLASLt dispatch census.  On the delivery this arm is expected to send **nothing** to hipBLASLt: block
#      06's `mmf32` claims the F32 shapes (r336), and block 10's GDN gate/beta epilogue fusion removes the
#      per-layer `ssm` GEMMs.  A class split HERE is a real delivery defect (the default path is not
#      reproducible across starts) and fails the check.
#   B. THE LIBRARY ARM.  The same arm with `GGML_CUDA_MMF32=0`, which puts the F32 GEMMs back on the pre-block-06
#      rocBLAS route so hipBLASLt is actually entered.  `HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1` +
#      `HIPBLASLT_LOG_FILE` records, per call, the `getAllSolutions()` candidate list.  That list is the
#      mechanism `c50ea60130f` stably sorts, and its *cardinality* is the verdict:
#        NO-CHOICE    every call offered exactly ONE candidate -> the library has no per-start choice to make
#                     (this is what hipBLASLt 1.5 / ROCm 10.2 does for these shapes; the pin is not needed
#                     on this line for them).
#        EXPOSED      several candidates and the order (or the logits class) varies across starts -> the
#                     lottery is live here: pin `ROCBLAS_USE_HIPBLASLT=0` before comparing fresh processes.
#        STABLE-LIST  several candidates, one order, one class in every start -> the arm did NOT witness the
#                     lottery.  Per §44 that is no evidence about the mechanism, not a pass.
#   B does not FAIL on EXPOSED (a ROCm <= 10.1 box with a live lottery is a valid state; the doctrine handles
#   it).  It FAILS if it never reached hipBLASLt at all (VACUOUS -- an arm that did not exercise the library
#   must not report a verdict) or if a run failed.  Its verdict is the datum the `README.md` / §44 paragraph
#   and the "is 10.2 a validated line" question are decided on.
#
# NOT a release gate: it measures a toolchain, it changes nothing, and its verdict is expected to differ
# between ROCm lines.  Run it once per line into a release record (or when a new ROCm install appears).
#
# Usage:
#   ./scripts/check-hipblaslt-lottery.sh                     # 8 starts per arm, 35B-A3B Q8_0 GDN
#   ./scripts/check-hipblaslt-lottery.sh --starts 18
#   ./scripts/check-hipblaslt-lottery.sh --arm a             # the delivery-default census only
#   ./scripts/check-hipblaslt-lottery.sh --model M.gguf --extra "-ncmoe 40"
# Environment:
#   BUILD_DIR    llama.cpp build tree        (default: ~/llama.cpp/build-rocm -- the canonical 10.2 tree)
#   LLAMA_PPL    llama-perplexity            (default: $BUILD_DIR/bin/llama-perplexity)
#   MODEL        a model with the F32 GDN/ router GEMMs (default: Qwen3.6-35B-A3B Q8_0)
#   CORPUS       scored text                 (default: prompts/prose-rdna-boosts.txt)
#   START_UB     prefill ubatch              (default: 256, the width the recorded arms used)
#   CTX          context / tokens per chunk  (default: 256, i.e. one ubatch per start)
#   CHUNKS       chunks                      (default: 1)
#   EXTRA        extra llama-perplexity args (default: "-ncmoe 20 -fa on"; word-split)
#   GPU          HIP_VISIBLE_DEVICES         (default: 0)
#   STARTS       fresh starts per arm        (default: 8)
#   BASE_DIR     where the KLD dumps go      (default: ~/.cache/rdna-boosts/gates/hipblaslt -- never /tmp)
#   KEEPOUT      dir for the logs            (default: mktemp -d)
#   TIME_LIMIT   per run                     (default: 1800 s)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$HOME/llama.cpp/build-rocm}"
LLAMA_PPL="${LLAMA_PPL:-$BUILD_DIR/bin/llama-perplexity}"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
CORPUS="${CORPUS:-$REPO_DIR/prompts/prose-rdna-boosts.txt}"
START_UB="${START_UB:-256}"
CTX="${CTX:-256}"
CHUNKS="${CHUNKS:-1}"
EXTRA="${EXTRA:--ncmoe 20 -fa on}"
GPU="${GPU:-0}"
STARTS="${STARTS:-8}"
BASE_DIR="${BASE_DIR:-$HOME/.cache/rdna-boosts/gates/hipblaslt}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
ARMS="ab"

while [ $# -gt 0 ]; do
    case "$1" in
        --starts)   STARTS="${2:-8}"; shift 2 ;;
        --starts=*) STARTS="${1#*=}"; shift ;;
        --arm)      ARMS="${2:-ab}"; shift 2 ;;
        --arm=*)    ARMS="${1#*=}"; shift ;;
        --model)    MODEL="${2:?--model needs a path}"; shift 2 ;;
        --model=*)  MODEL="${1#*=}"; shift ;;
        -h|--help)  sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//; /^set -uo/d'; exit 0 ;;
        *) echo "usage: $0 [--starts N] [--arm a|b|ab] [--model FILE]" >&2; exit 2 ;;
    esac
done
case "$STARTS" in ''|*[!0-9]*) echo "usage: --starts must be a number (got '$STARTS')" >&2; exit 2 ;; esac
case "$ARMS" in a|b|ab) ;; *) echo "usage: --arm must be a, b or ab (got '$ARMS')" >&2; exit 2 ;; esac

fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
note() { echo "    $*"; }
mkdir -p "$KEEPOUT" "$BASE_DIR"

[ -x "$LLAMA_PPL" ] || { echo "FAIL: llama-perplexity not executable: $LLAMA_PPL (set LLAMA_PPL/BUILD_DIR)" >&2; exit 2; }
[ -f "$MODEL" ]     || { echo "FAIL: model not found: $MODEL (set MODEL)" >&2; exit 2; }
[ -f "$CORPUS" ]    || { echo "FAIL: corpus not found: $CORPUS (set CORPUS)" >&2; exit 2; }

# the ROCm line this build links, and the hipBLASLt it will load (reported, because that is what differs
# between the lines this check is run on)
ROCM_LINE="$(ldd "$LLAMA_PPL" 2>/dev/null | sed -n 's#.*=> \(/opt/rocm[^/]*\)/.*#\1#p' | head -1)"
HBLT_HDR="${ROCM_LINE:-/opt/rocm}/include/hipblaslt/hipblaslt-version.h"
HBLT_VER="-"
if [ -f "$HBLT_HDR" ]; then
    HBLT_VER="$(sed -n 's/^#define HIPBLASLT_VERSION_\(MAJOR\|MINOR\|PATCH\) *\([0-9]*\).*/\2/p' "$HBLT_HDR" | paste -sd. -)"
fi

echo "==> the per-start hipBLASLt lottery check (GREEDY-PURITY.md §44, TODO #362)"
echo "    ppl=$LLAMA_PPL"
echo "    rocm line=${ROCM_LINE:-unknown}  hipBLASLt=${HBLT_VER}"
echo "    model=$(basename "$MODEL")  ub=$START_UB ctx=$CTX chunks=$CHUNKS  extra='$EXTRA'  starts=$STARTS"
echo "    logs=$KEEPOUT  bases=$BASE_DIR"

# one run: run_ppl <tag> <extra env...> [ppl args...] -- prints the log's path
run_ppl() {
    local tag="$1" log="$2" envs="$3"; shift 3
    # shellcheck disable=SC2086  # $envs / $EXTRA are intended word-split lists
    env $envs HIP_VISIBLE_DEVICES="$GPU" timeout "$TIME_LIMIT" "$LLAMA_PPL" \
        -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" -ngl 99 $EXTRA \
        -ub "$START_UB" -b "$START_UB" -t 8 "$@" > "$log" 2>&1
}

# the candidate list of every `getAllSolutions()` call, grouped per call, from a hipBLASLt trace log
# -> "calls=<n> lens=<l0,l1,..> orders=<distinct sequences>"
list_census() { # <hbl-log>
    python3 - "$1" <<'PY'
import re, sys
blocks, cur = [], None
try:
    f = open(sys.argv[1], errors="replace")
except OSError:
    print("calls=0 lens=- orders=-")
    sys.exit(0)
# TWO LOG FORMATS, and both must be handled: hipBLASLt 1.4 (ROCm <= 10.1) prints a bare
# `[Info][getBestSolutions]` header and then one `[Info][getAllSolutions] sol-idx = N` line per candidate,
# while 1.5 (ROCm 10.2) prints the pick ON the header line (`getBestSolutions(): sol-idx = N, ...`) and no
# candidate list at all -- which is the fix's own signature.
for line in f:
    if "rocblaslt_matmul" in line:
        cur = None
        continue
    m = re.search(r"sol-idx = (\d+)", line)
    if "getBestSolutions" in line:
        cur = []
        blocks.append(cur)
        if m:
            cur.append(int(m.group(1)))
        continue
    if m and cur is not None:
        cur.append(int(m.group(1)))
lens = sorted({len(b) for b in blocks})
orders = len({tuple(b) for b in blocks})
print(f"calls={len(blocks)} lens={','.join(str(x) for x in lens) or '-'} orders={orders}")
PY
}

# The logits fingerprint of one start: the sha256 of llama-perplexity's logits dump.  The dump is a function
# of the logits only (no timestamps), and its header is fixed by the model's vocabulary, so equal logits give
# an equal hash.  The dump is deleted after it is hashed unless KEEP_BASE=1.
start_logits() { # <arm> <n> <envs> <hbl-log-or-empty> -> prints "<base-class> <census>"
    local arm="$1" n="$2" envs="$3" hbl="$4"
    local base="$BASE_DIR/$arm-$n.kld"
    local log="$KEEPOUT/$arm-$n.log"
    local hbl_envs=""
    rm -f "$base" "$hbl"
    if [ -n "$hbl" ]; then
        hbl_envs="HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1 HIPBLASLT_LOG_FILE=$hbl"
        : > "$hbl"
    fi
    if ! run_ppl "$arm-$n" "$log" "$envs $hbl_envs" --kl-divergence-base "$base"; then
        echo "RUNFAIL -"
        return
    fi
    if [ ! -s "$base" ]; then
        echo "NOBASE -"
        return
    fi
    local cls
    cls="$(sha256sum "$base" | cut -c1-16)"
    [ "${KEEP_BASE:-0}" = 1 ] || rm -f "$base"
    local cen="-"
    [ -n "$hbl" ] && cen="$(list_census "$hbl")"
    echo "$cls $cen"
}

# ---------------------------------------------------------------------------------------------------
# A. the delivery default: is it reproducible across starts, and does it reach hipBLASLt at all?
# ---------------------------------------------------------------------------------------------------
if [[ "$ARMS" == *a* ]]; then
    echo
    echo "==> A. the delivery default ($STARTS fresh starts)"
    a_classes=""
    a_disp=0
    for n in $(seq 1 "$STARTS"); do
        hbl="$KEEPOUT/hbl-a-$n.log"
        read -r cls cen < <(start_logits a-default "$n" "" "$hbl")
        d="$(grep -c "rocblaslt_matmul" "$hbl" 2>/dev/null || true)"
        d="${d:-0}"
        a_disp=$((a_disp + d))
        a_classes="$a_classes$cls\n"
        note "start $n: logits=$cls  hipBLASLt dispatches=$d  ${cen}"
    done
    uniq_a="$(printf "%b" "$a_classes" | sort | uniq -c | sort -rn)"
    note "class census:"; printf '%s\n' "$uniq_a" | sed 's/^/      /'
    if printf "%b" "$a_classes" | grep -q "RUNFAIL\|NOBASE"; then
        fail "a delivery-default start did not complete (see $KEEPOUT)"
    elif [ "$(printf "%b" "$a_classes" | sort -u | wc -l)" -gt 1 ]; then
        fail "the DELIVERY DEFAULT is not reproducible across fresh starts: $(printf "%b" "$a_classes" | sort -u | tr '\n' ' ')— a delivery defect, not a library lottery (the delivery never pins the library)"
    fi
    note "hipBLASLt dispatches over $STARTS starts: $a_disp$([ "$a_disp" -eq 0 ] && echo ' (the delivery default sends these F32 GEMMs to mmf32/GDN fusions, so the §44 pin is inert on it)')"
fi

# ---------------------------------------------------------------------------------------------------
# B. the library arm: MMF32 off, so the F32 GEMMs take the pre-block-06 rocBLAS/hipBLASLt route.
# ---------------------------------------------------------------------------------------------------
if [[ "$ARMS" == *b* ]]; then
    echo
    echo "==> B. the library arm (GGML_CUDA_MMF32=0, $STARTS fresh starts): is a per-start choice OFFERED?"
    b_classes=""
    b_lens=""
    b_orders=""
    b_calls=0
    for n in $(seq 1 "$STARTS"); do
        hbl="$KEEPOUT/hbl-b-$n.log"
        read -r cls cen < <(start_logits b-library "$n" "GGML_CUDA_MMF32=0" "$hbl")
        b_classes="$b_classes$cls\n"
        calls="$(printf '%s' "$cen" | sed -n 's/^calls=\([0-9]*\).*/\1/p')"
        lens="$(printf '%s' "$cen" | sed -n 's/.*lens=\([^ ]*\).*/\1/p')"
        orders="$(printf '%s' "$cen" | sed -n 's/.*orders=\([^ ]*\).*/\1/p')"
        b_calls=$((b_calls + ${calls:-0}))
        b_lens="$b_lens$lens\n"
        b_orders="$b_orders$orders\n"
        note "start $n: logits=$cls  $cen"
    done
    uniq_b="$(printf "%b" "$b_classes" | sort | uniq -c | sort -rn)"
    note "class census:"; printf '%s\n' "$uniq_b" | sed 's/^/      /'
    note "candidate-list lengths seen: $(printf "%b" "$b_lens" | sort -u | tr '\n' ' ')"
    note "distinct candidate orders per start: $(printf "%b" "$b_orders" | sort -u | tr '\n' ' ')"

    if printf "%b" "$b_classes" | grep -q "RUNFAIL\|NOBASE"; then
        fail "a library-arm start did not complete (see $KEEPOUT)"
    elif [ "$b_calls" -eq 0 ]; then
        fail "the library arm never reached hipBLASLt (0 dispatches in $STARTS starts): the arm is VACUOUS and must not report a verdict -- check that mmf32 is the only thing claiming these shapes on this build"
    else
        maxlen="$(printf "%b" "$b_lens" | tr ' ' '\n' | sort -n | tail -1)"
        nord="$(printf "%b" "$b_orders" | sort -u | wc -l)"
        ncls="$(printf "%b" "$b_classes" | sort -u | wc -l)"
        echo
        if [ "${maxlen:-0}" -le 1 ]; then
            note "VERDICT: NO-CHOICE -- every one of the $b_calls hipBLASLt calls was offered a SINGLE solution, so"
            note "         there is no per-start solution pick on this line: the §44 pin is not needed here for these"
            note "         shapes (hipBLASLt 1.5 / ROCm 10.2 carries the fix that collapses the candidate container)."
        elif [ "$nord" -gt 1 ] || [ "$ncls" -gt 1 ]; then
            note "VERDICT: EXPOSED -- $maxlen candidates per call, $nord distinct order(s) and $ncls logits class(es)"
            note "         over $STARTS starts: the lottery is LIVE on this line.  Pin ROCBLAS_USE_HIPBLASLT=0 before"
            note "         comparing fresh processes here (GREEDY-PURITY.md §44)."
        else
            note "VERDICT: STABLE-LIST -- $maxlen candidates per call but one order and one logits class in every start:"
            note "         the arm did NOT witness the lottery.  By §44 that is no evidence about the mechanism (and no"
            note "         pass): quote it as an unexercised arm, never as 'the library is fine here'."
        fi
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "PASS: the check ran (a library verdict above is the datum; it is not a delivery gate)"
    exit 0
fi
echo "FAILED: $fails check(s) -- see $KEEPOUT"
exit 1
