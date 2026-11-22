#!/usr/bin/env bash
# gate-server-reuse.sh -- the REVERSE-HANDOVER oracle: a reused prefix must move NO token.
#
# WHY THIS EXISTS
#   On the two-arm path a finished conversation can only stay resident if its RECURRENT state moves back from the
#   decode arm's memory to the prefill arm's at the request boundary (the reverse half of the handover).  When it
#   does, the next request on that slot reuses its prefix instead of re-prefilling it -- measured on the delivery's
#   own `gate-ckpt-reuse.sh`: a turn appending ~9 tokens to 5200 cost 685 ms without the reverse take and 33 ms
#   with it, against the single-context server's 33 ms.
#
#   The risk the take carries is a WRONG STATE, not a crash: a recurrent half that does not match the attention
#   cells at the reuse point produces fluent, wrong text on the NEXT request (the campaign's §89/§44 class of
#   carrier).  So the oracle is the same one the delivery uses for every cache change: **a reused prefix must
#   produce byte-identical text**, checked against an arm that never reuses.
#
# WHAT IT CHECKS  (two servers, one at a time, same shape + purity pins, three requests on ONE slot)
#   arm `cache`    cache_prompt true   -- request 2 EXTENDS the prompt (the reuse case, which consumes the state the
#                                        reverse take put back), request 3 forks BELOW the shared prefix (the
#                                        rollback case: on a hybrid the reference re-prefills there for want of a
#                                        checkpoint, so it is a text comparison only)
#   arm `nocache`  cache_prompt false  -- every request re-prefills: the reference for the same three replies
#   HARD  * all three replies are byte-identical between the arms
#   HARD  * the cached arm really reused: `cache_n > 0` on request 2 (otherwise the comparison is vacuous)
#   HARD  * no abort in either log, and `scripts/check-gpu-faults.sh` clean over both
#   INFO  prompt_ms and cache_n per request per arm (that is the feature's own number)
#
# Usage:
#   ./gate-server-reuse.sh                    # both arms + the comparison (the gate)
#   ./gate-server-reuse.sh --mode cache       # one arm only (writes $OUT/cache.txt)
#   ./gate-server-reuse.sh --mode nocache
#   ./gate-server-reuse.sh --compare CACHED REFERENCE   # the FIRST file is the cached arm
#
# It needs `--prefill-decode-arms` in the build for the cached arm to be the two-arm path (SERVER_ARGS overrides).
#
# Rules (each has cost this campaign a session):
#   * one server at a time -- never two arms in parallel on the 32 GiB card;
#   * HIP_VISIBLE_DEVICES=0;
#   * the purity pins, on EVERY arm: ROCBLAS_USE_HIPBLASLT=0 (the library choice can move mid-process, pass 2) and
#     MOE_EXPERT_HEATMAP=load (residency selects the MoE kernel set, GREEDY-PURITY §89);
#   * the shape pins: -c, -b/-ub, the arena, -np -- a shape change moves the text by itself;
#   * the server returns text over HTTP, so `-lv 4` is SAFE here and it is what makes the census visible.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/llama.cpp}
BUILD=${BUILD:-$SRC/build-2ctx}
SERVER=$BUILD/bin/llama-server
OUT=${OUT:-$HERE/../runs/server-reuse}

MODEL=${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
PROMPT_FILE=${PROMPT_FILE:-$HERE/../../../prompts/kld-corpus-35b-in-domain.txt}

NGL=${NGL:-99}
NCMOE=${NCMOE:-40}
ARENA=${ARENA:-5000}
CTX=${CTX:-8192}
UB=${UB:-512}
NP=${NP:-2}
NPREDICT=${NPREDICT:-32}
PORT=${PORT:-8124}
PROMPT_CHARS=${PROMPT_CHARS:-2000}   # ~500 tokens: the shared prefix
SHRINK_CHARS=${SHRINK_CHARS:-1200}   # request 3 forks BELOW the prefix (the rollback case)
SERVER_ARGS=${SERVER_ARGS:---prefill-decode-arms}

mode=both
compare=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)    mode=$2; shift 2 ;;
        --compare) compare+=("$2" "$3"); shift 2; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---- the compare mode -------------------------------------------------------------------
if [[ ${#compare[@]} -gt 0 ]]; then
    a=${compare[0]} b=${compare[1]}
    tmp=$(mktemp -d)
    grep '^REPLY ' "$a" > "$tmp/a" || true
    grep '^REPLY ' "$b" > "$tmp/b" || true
    if [[ ! -s $tmp/a || ! -s $tmp/b ]]; then
        echo "ORACLE ERROR: no REPLY lines in a result file (a run failed silently?)"; rm -rf "$tmp"; exit 2
    fi
    # vacuity: the cached arm must have reused the GROWN prefix (request B).  Request C is a fork BELOW the shared
    # prefix and legitimately re-prefills on a hybrid model (the reference does the same for want of a checkpoint),
    # so it is a TEXT comparison only -- do not require a cache hit there.
    if ! awk '$1 == "INFO" && ($2 == "B" || $2 == "r2s0") { split($0, f, "cache_n="); split(f[2], g, " "); if (g[1]+0 > 0) { found = 1 } }
              END { exit !found }' "$a"; then
        echo "ORACLE ERROR: the cached arm reused nothing on a grown prompt (cache_n = 0) -- vacuous"
        rm -rf "$tmp"; exit 2
    fi
    if diff -q "$tmp/a" "$tmp/b" > /dev/null; then
        echo "ORACLE PASS: a reused prefix (grow + shrink) moved no token"
        rm -rf "$tmp"; exit 0
    fi
    echo "ORACLE FAIL: the reused prefix moved the text"
    diff "$tmp/a" "$tmp/b" | sed 's/^/  /'
    exit 1
fi

# ---- one arm ----------------------------------------------------------------------------
[[ -x $SERVER ]] || { echo "no llama-server in $BUILD" >&2; exit 1; }
mkdir -p "$OUT"

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
export MOE_EXPERT_CACHE_MIB="$ARENA" MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0

run_arm() { # $1 = cache|nocache
    local arm=$1 cache_prompt
    case $arm in
        cache)   cache_prompt=true  ;;
        nocache) cache_prompt=false ;;
        *) echo "--mode must be cache|nocache" >&2; exit 2 ;;
    esac
    local log="$OUT/$arm.log" res="$OUT/$arm.txt"

    # three requests on ONE slot: A -> B extends the prompt (the reuse case), B -> C forks BELOW the shared
    # prefix (the rollback case: on a hybrid the reference re-prefills here for want of a checkpoint, so C is a
    # text comparison, not a reuse count)
    local pA pB pC
    pA=$(head -c "$PROMPT_CHARS" "$PROMPT_FILE")
    # an extension in the prompt's OWN style: a chat instruction here makes the model emit EOG immediately (empty
    # reply, sha of the empty string) on both arms, which would make the comparison vacuous
    pB="$pA$(tail -c $((PROMPT_CHARS / 4)) "$PROMPT_FILE")"
    pC=$(head -c "$SHRINK_CHARS" "$PROMPT_FILE")

    echo "== $arm: starting the server ($SERVER_ARGS, arena $ARENA, -c $CTX -ub $UB -np $NP)" >&2
    timeout 1800 "$SERVER" -m "$MODEL" -ngl "$NGL" -ncmoe "$NCMOE" -lme auto \
        -c "$CTX" -b 2048 -ub "$UB" -np "$NP" -t 8 --host 127.0.0.1 --port "$PORT" -lv 4 \
        $SERVER_ARGS > "$log" 2>&1 &
    local srv=$!
    trap 'kill $srv 2>/dev/null || true' EXIT

    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
        kill -0 $srv 2>/dev/null || { echo "server died during start-up; see $log" >&2; tail -5 "$log" >&2; exit 1; }
        sleep 1
    done

    : > "$res"

    # CONCURRENT=1: the MULTI-AGENT case -- two conversations, one per slot, each in two rounds (the second round
    # extends the first), the two slots driven in PARALLEL in each round.  That is the shape that interleaves a
    # forward take with a reverse one, so it is the test the request-boundary handshake has to survive.
    if [[ ${CONCURRENT:-0} == 1 ]]; then
        local base0 base1 r
        base0=$(head -c "$PROMPT_CHARS" "$PROMPT_FILE")
        base1=$(tail -c "$PROMPT_CHARS" "$PROMPT_FILE")
        for r in 1 2; do
            local s0 s1
            if [[ $r == 1 ]]; then s0=$base0; s1=$base1; else s0="$base0$(tail -c 400 "$PROMPT_FILE")"; s1="$base1$(head -c 400 "$PROMPT_FILE")"; fi
            local id0 id1
            curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
                -d "$(jq -nc --arg p "$s0" --argjson n "$NPREDICT" --argjson c "$cache_prompt" \
                        '{prompt:$p, n_predict:$n, temperature:0, top_k:1, seed:42, cache_prompt:$c,
                          id_slot:0, stream:false}')" > "$OUT/.r${r}s0.json" &
            id0=$!
            curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
                -d "$(jq -nc --arg p "$s1" --argjson n "$NPREDICT" --argjson c "$cache_prompt" \
                        '{prompt:$p, n_predict:$n, temperature:0, top_k:1, seed:42, cache_prompt:$c,
                          id_slot:1, stream:false}')" > "$OUT/.r${r}s1.json" &
            id1=$!
            wait "$id0" "$id1" || { echo "a concurrent request failed (see $log)" >&2; exit 1; }
            for sl in 0 1; do
                local tag="r${r}s${sl}" f="$OUT/.r${r}s${sl}.json" content ntok sha pms cn
                content=$(jq -r '.content' "$f")
                ntok=$(jq -r '.tokens_predicted' "$f")
                pms=$(jq -r '.timings.prompt_ms // 0' "$f")
                cn=$(jq -r '.timings.cache_n // 0' "$f")
                sha=$(printf '%s' "$content" | sha256sum | cut -c1-16)
                printf 'REPLY %s sha=%s ntok=%s content=%s\n' "$tag" "$sha" "$ntok" \
                    "$(printf '%s' "$content" | tr '\n' ' ')" | tee -a "$res"
                printf 'INFO %s prompt_ms=%.1f cache_n=%s\n' "$tag" "$pms" "$cn" | tee -a "$res"
                rm -f "$f"
            done
        done
        kill $srv 2>/dev/null || true
        wait $srv 2>/dev/null || true
        trap - EXIT
        if grep -q "GGML_ASSERT\|terminate called\|Segmentation fault" "$log"; then
            echo "FAIL $arm: an abort in the log (see $log)" >&2; return 1
        fi
        echo "== $arm: $res" >&2
        return 0
    fi

    local tag prompt
    for tag in A B C; do
        case $tag in A) prompt=$pA ;; B) prompt=$pB ;; C) prompt=$pC ;; esac
        curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
            -d "$(jq -nc --arg p "$prompt" --argjson n "$NPREDICT" --argjson c "$cache_prompt" \
                    '{prompt:$p, n_predict:$n, temperature:0, top_k:1, seed:42, cache_prompt:$c,
                      id_slot:0, stream:false}')" > "$OUT/.$tag.json"
        local content ntok sha pms cn
        content=$(jq -r '.content' "$OUT/.$tag.json")
        ntok=$(jq -r '.tokens_predicted' "$OUT/.$tag.json")
        pms=$(jq -r '.timings.prompt_ms // 0' "$OUT/.$tag.json")
        cn=$(jq -r '.timings.cache_n // 0' "$OUT/.$tag.json")
        sha=$(printf '%s' "$content" | sha256sum | cut -c1-16)
        # the compared line is the TEXT (and its length); the timings are the record, never asserted
        printf 'REPLY %s sha=%s ntok=%s content=%s\n' "$tag" "$sha" "$ntok" \
            "$(printf '%s' "$content" | tr '\n' ' ')" | tee -a "$res"
        printf 'INFO %s prompt_ms=%.1f cache_n=%s\n' "$tag" "$pms" "$cn" | tee -a "$res"
        rm -f "$OUT/.$tag.json"
    done

    kill $srv 2>/dev/null || true
    wait $srv 2>/dev/null || true
    trap - EXIT

    # a rejected request or an abort is a gate error, not a pass
    if grep -q "GGML_ASSERT\|terminate called\|Segmentation fault" "$log"; then
        echo "FAIL $arm: an abort in the log (see $log)" >&2; return 1
    fi
    echo "== $arm: $res" >&2
}

if [[ $mode == both ]]; then
    run_arm cache
    run_arm nocache
    "$0" --compare "$OUT/cache.txt" "$OUT/nocache.txt"
else
    run_arm "$mode"
fi
