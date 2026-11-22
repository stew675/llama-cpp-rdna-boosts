#!/usr/bin/env bash
# gate-server-arms.sh -- the T2 TEXT ORACLE (tier 1 of the two-tier gate the maintainer approved).
#
# The two-arm server's reply must be BYTE-IDENTICAL to the single-context server's reply, for BOTH slots, with the
# shape and purity pins fixed.  That is only meaningful because T0 (enabling-0003) made the KV cell layout a
# function of the request sequence alone: measured outside the server, the two arms now read the reference's exact
# logits rows (480/480) and cells (729/729 census lines), so a server that shares one allocator between a prefill
# arm and a decode arm must reproduce the same text.
#
#   ./gate-server-arms.sh --mode ref                     # the reference: today's single-context server
#   ./gate-server-arms.sh --mode arms                    # the two-arm server (needs --prefill-decode-arms)
#   ./gate-server-arms.sh --mode ref --control npad      # NEGATIVE CONTROL: the gate must FAIL (npad moves text)
#   ./gate-server-arms.sh --compare A B                  # compare two result files (exit 1 on a mismatch)
#
# Rules it enforces (each one has cost this campaign a session):
#   * one server at a time -- never two arms in parallel on the 32 GiB card;
#   * HIP_VISIBLE_DEVICES=0 (the iGPU is enumerated otherwise and the load aborts);
#   * a TEXT arm takes no --single-turn/-lv games: the server returns text over HTTP, so -lv 4 is SAFE here, and
#     it is what we want, because the arm census (MoE residency, handshake lines) must be visible;
#   * the purity pins: ROCBLAS_USE_HIPBLASLT=0 (the hipBLASLt choice can move text MID-PROCESS -- pass 2) and
#     MOE_EXPERT_HEATMAP=load (residency selects the MoE kernel set, GREEDY-PURITY §89);
#   * the shape pins: -c, -ub, the arena, -np -- a shape change moves the text by itself (that is the --control).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/llama.cpp}
BUILD=${BUILD:-$SRC/build-2ctx}
SERVER=$BUILD/bin/llama-server
OUT=${OUT:-$HERE/../runs/server-oracle}

MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
PROMPT_FILE=${PROMPT_FILE:-$HERE/../../../prompts/kld-corpus-35b-in-domain.txt}

# shape pins -- the product shape of RESULTS-2026-11-21.md §2.1 (arena 5000, the ubatch the pair fits at).
# -ub 2048 does NOT fit two contexts on a 32 GiB card (RESULTS-2026-11-21.md §3.2).
NGL=${NGL:-99}
NCMOE=${NCMOE:-40}
ARENA=${ARENA:-5000}
CTX=${CTX:-8192}
UB=${UB:-512}
B=${B:-2048}
NP=${NP:-2}
NPREDICT=${NPREDICT:-32}
PORT=${PORT:-8123}
PROMPT_CHARS=${PROMPT_CHARS:-2000}      # ~500 tokens per slot; enough to be deterministic, quick to run
REPS=${REPS:-1}                         # request PAIRS per server process; >1 = the warm reps (labels r<rep>.<slot>)

mode=ref
control=()
compare=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)    mode=$2; shift 2 ;;
        --control) control+=("$2"); shift 2 ;;
        --compare) compare=("$2" "$3"); shift 2; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---- the compare mode: two result files ------------------------------------------------
if [[ ${#compare[@]} -gt 0 ]]; then
    a=${compare[0]} b=${compare[1]}
    # TIER 1 IS THE TEXT.  The CENSUS/HANDSHAKE lines are tier 2: they are reported so a difference has a
    # mechanism attached, and they are never asserted here -- a residency-census difference with identical text is
    # a known, legal state (measured: LLAMA_KV_N_PAD_MIN=8192 changes the tier-2 census on this server shape and
    # does NOT move the text).
    tmp=$(mktemp -d)
    grep '^SLOT ' "$a" > "$tmp/a" || true
    grep '^SLOT ' "$b" > "$tmp/b" || true
    if [[ ! -s $tmp/a || ! -s $tmp/b ]]; then
        echo "ORACLE ERROR: no SLOT lines in a result file (a run failed silently?)"
        rm -rf "$tmp"; exit 2
    fi
    # vacuity guard: a reply of zero tokens would make any two runs "equal"
    if grep -q 'ntok=0' "$tmp/a" "$tmp/b"; then
        echo "ORACLE ERROR: a slot produced 0 tokens (the comparison would be vacuous)"
        rm -rf "$tmp"; exit 2
    fi
    a=$tmp/a; b=$tmp/b
    if diff -q "$a" "$b" > /dev/null; then
        echo "ORACLE PASS: $(basename "$a") == $(basename "$b")"
        exit 0
    fi
    echo "ORACLE FAIL: $(basename "$a") != $(basename "$b")"
    diff "$a" "$b" | sed 's/^/  /'
    exit 1
fi

# ---- the run mode ----------------------------------------------------------------------
[[ -x $SERVER ]] || { echo "no llama-server in $BUILD (build it: BUILD_DIR=$(basename "$BUILD") ~/bin/build-llama-rocm-101)" >&2; exit 1; }
mkdir -p "$OUT"

if [[ -n ${UB_OVERRIDE:-} ]]; then UB=$UB_OVERRIDE; fi   # before the flag array is built

# NO_HOST_TIER=1: a fully resident shape (every weight in VRAM, no expert arena, no gathers) -- the common
# single-GPU end-user case, and the shape a dense MoE-free model has to use because -ncmoe does not apply
tier_args=()
if [[ ${NO_HOST_TIER:-0} == 1 ]]; then
    unset MOE_EXPERT_CACHE_MIB MOE_EXPERT_HEATMAP
    tier_args=(-ngl "$NGL")
else
    tier_args=(-ngl "$NGL" -ncmoe "$NCMOE" -lme auto)
fi

common=(
    -m "$MODEL"
    "${tier_args[@]}"
    -c "$CTX" -b "$B" -ub "$UB" -np "$NP"
    -t 8 --host 127.0.0.1 --port "$PORT"
    -lv 4
)
case $mode in
    ref)  ;;
    arms) common+=(--prefill-decode-arms) ;;
    *)    echo "--mode must be ref|arms" >&2; exit 2 ;;
esac

for c in "${control[@]}"; do
    case $c in
        # the negative control: a shape pin that is KNOWN to move the single-context text (pass 1 §3.3), so a gate
        # that cannot see it is not an oracle.  It must never be combined with a real run.
        npad) export LLAMA_KV_N_PAD_MIN=8192 ;;
        # a SHAPE change (the prefill ubatch): must move the text, else the oracle cannot see a shape effect
        ub256) export UB_OVERRIDE=256 ;;
        # the residency prior deliberately NOT pinned: the §89 hazard, tier 2's reason for existing
        noheat) unset MOE_EXPERT_HEATMAP ;;
        *)    echo "unknown control: $c" >&2; exit 2 ;;
    esac
done

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0
export MOE_EXPERT_CACHE_MIB="$ARENA"

tag="$mode"; for c in "${control[@]}"; do tag+="-$c"; done
log="$OUT/$tag.log"
res="$OUT/$tag.txt"

# two DIFFERENT prompts so the two slots are distinguishable, each a fixed slice of the campaign prompt
p0=$(head -c "$PROMPT_CHARS" "$PROMPT_FILE")
p1=$(tail -c "$PROMPT_CHARS" "$PROMPT_FILE")

echo "== $tag: starting the server ($mode, arena $ARENA, -c $CTX -ub $UB -np $NP)" >&2
timeout 900 "$SERVER" "${common[@]}" > "$log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT

# wait for the health endpoint (model load + expert-cache init included)
for _ in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $srv 2>/dev/null; then
        echo "server died during start-up; see $log" >&2
        tail -5 "$log" >&2
        exit 1
    fi
    sleep 1
done

req() { # $1 = prompt, $2 = outfile
    curl -sf "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg p "$1" --argjson n "$NPREDICT" \
                '{prompt:$p, n_predict:$n, temperature:0, top_k:1, seed:42, cache_prompt:false, stream:false}')" \
        > "$2"
}

# BOTH slots at once: the arms' decode step is one token per active slot in one batch, so the reference has to be
# the same concurrent 2-slot shape, not two sequential requests.
#
# REPS > 1 sends that pair again, REPS times, in ONE server process: the WARM reps are the interesting ones -- the
# campaign's text carrier was a WARM-rep effect (the two allocation cursors, then a take racing the next prompt), so
# a single cold pair can pass while a warm one moves.  The labels stay `SLOT <i>` at REPS=1 (the acceptance values
# of RESULTS-2026-11-22.md pass 4 are those) and become `SLOT r<rep>.<i>` beyond it.
: > "$res"
for rep in $(seq 0 $((REPS - 1))); do
    req "$p0" "$OUT/.r0.json" &
    r0=$!
    req "$p1" "$OUT/.r1.json" &
    r1=$!
    wait $r0 $r1 || { echo "a request failed (rep $rep); see $log" >&2; exit 1; }

    for i in 0 1; do
        f="$OUT/.r$i.json"
        content=$(jq -r '.content' "$f")
        ntok=$(jq -r '.tokens_predicted' "$f")
        sha=$(printf '%s' "$content" | sha256sum | cut -c1-16)
        if [[ $REPS -gt 1 ]]; then
            printf 'SLOT r%d.%d sha=%s ntok=%s content=%s\n' "$rep" "$i" "$sha" "$ntok" "$content" >> "$res"
            printf 'SLOT r%d.%d sha=%s ntok=%s\n' "$rep" "$i" "$sha" "$ntok"
        else
            printf 'SLOT %d sha=%s ntok=%s content=%s\n' "$i" "$sha" "$ntok" "$content" >> "$res"
            printf 'SLOT %d sha=%s ntok=%s\n' "$i" "$sha" "$ntok"
        fi
    done
    rm -f "$OUT/.r0.json" "$OUT/.r1.json"
done

kill $srv 2>/dev/null || true
wait $srv 2>/dev/null || true
trap - EXIT

# the arm census, recorded so a difference has a mechanism attached (never asserted here -- tier 2 does that)
if grep -q "MoE expert cache:" "$log"; then
    grep -o "MoE expert cache: h=[0-9.]* ([0-9]*/[0-9]* reaches)" "$log" | tail -1 | sed 's/^/CENSUS /' | tee -a "$res" >&2
fi
if grep -q "the co-owner pair shares ONE KV allocation owner" "$log"; then
    echo "HANDSHAKE the pair shares one allocation owner" | tee -a "$res" >&2
else
    echo "HANDSHAKE single-context (no pair)" | tee -a "$res" >&2
fi
echo "== $tag: $res" >&2
