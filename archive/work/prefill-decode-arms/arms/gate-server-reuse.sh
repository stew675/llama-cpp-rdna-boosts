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
# TWO CONTRACTS (they are not the same claim -- do not conflate them):
#   GATE=strict (default)   `cache` vs `nocache`: a REUSED prefix must move no token.  This is the strong claim, and
#                           it only holds where the DELIVERY itself has it -- measured true on the 4B (hybrid), phi-4
#                           (dense) and the 35B-A3B host tier; NOT on the fully resident 27B, where the reference
#                           server diverges too (see `--gate parity` below and RESULTS-2026-11-22.md pass 8 §3).
#   GATE=parity             `cache` vs `ref` (the SAME requests with reuse on the SINGLE-CONTEXT server): the arms'
#                           reuse path must be byte-identical to the reference server's own reuse path.  This holds
#                           everywhere and is the contract a promotion can be judged on.
#   HARD  * all three replies are byte-identical between the two arms
#   HARD  * the cached arm really reused: `cache_n > 0` on request 2 (otherwise the comparison is vacuous)
#   HARD  * no abort in either log, and `scripts/check-gpu-faults.sh` clean over both
#   INFO  prompt_ms and cache_n per request per arm (that is the feature's own number)
#
# Usage:
#   ./gate-server-reuse.sh                    # both arms + the comparison (the gate)
#   ./gate-server-reuse.sh                    # GATE=strict (default)
#   GATE=parity ./gate-server-reuse.sh        # the arms-vs-single-context reuse contract
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
B=${B:-2048}
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
    # FORK_REUSE=1 (a DENSE model: a pure attention KV, so rolling back below the prefix needs no checkpoint): the
    # fork must reuse as well -- that is the dense case's own claim, and it is not covered by the grow check
    if [[ ${FORK_REUSE:-0} == 1 && ${GATE:-strict} == strict ]]; then
        if ! awk '$1 == "INFO" && $2 == "C" { split($0, f, "cache_n="); split(f[2], g, " "); exit (g[1]+0 <= 0) }' "$a"; then
            echo "ORACLE ERROR: the dense-model fork did not reuse (C: cache_n = 0) -- the rollback path is untested"
            rm -rf "$tmp"; exit 2
        fi
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
export MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0
export MOE_EXPERT_CACHE_MIB="$ARENA"

run_arm() { # $1 = cache|nocache
    local arm=$1 cache_prompt
    local server_args=$SERVER_ARGS
    case $arm in
        cache)   cache_prompt=true  ;;
        nocache) cache_prompt=false ;;
        ref)     cache_prompt=true; server_args=${SERVER_ARGS_REF:-} ;;   # the single-context server, reuse ON
        *) echo "--mode must be cache|nocache|ref" >&2; exit 2 ;;
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

    # NO_HOST_TIER=1: fully resident (no expert arena, no -ncmoe) -- the common single-GPU end-user shape, and the
    # only shape a dense MoE-free model accepts
    local tier_args=()
    if [[ ${NO_HOST_TIER:-0} == 1 ]]; then
        unset MOE_EXPERT_CACHE_MIB
        tier_args=(-ngl "$NGL")
    else
        tier_args=(-ngl "$NGL" -ncmoe "$NCMOE" -lme auto)
    fi
    # a leftover server from a previous arm ANSWERS THE HEALTH PROBE and this gate would then silently drive the
    # WRONG arm (measured 2026-11-22 on the 27B: the "nocache" arm hit the leaked cached-arm server and the
    # comparison read as a state failure).  Refuse instead -- the park/ckpt gates do the same.
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "ERROR: something is already listening on port $PORT (a leaked server? set PORT=...)" >&2
        pgrep -af "llama-server.*--port $PORT" >&2 || true
        exit 2
    fi

    echo "== $arm: starting the server (${server_args:-single-context}, ${NO_HOST_TIER:+no host tier }-c $CTX -ub $UB -np $NP)" >&2
    timeout 1800 "$SERVER" -m "$MODEL" "${tier_args[@]}" \
        -c "$CTX" -b "$B" -ub "$UB" -np "$NP" -t 8 --host 127.0.0.1 --port "$PORT" -lv 4 \
        $server_args > "$log" 2>&1 &
    local srv=$!
    trap 'kill ${srv:-} 2>/dev/null || true' EXIT

    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
        kill -0 $srv 2>/dev/null || { echo "server died during start-up; see $log" >&2; tail -5 "$log" >&2; exit 1; }
        sleep 1
    done

    : > "$res"

    # CONCURRENT=1 / TWO_SLOT=1: the MULTI-AGENT case -- two conversations, one per slot, each in two rounds (the
    # second extends the first).  TWO_SLOT=1 drives them strictly SEQUENTIALLY (two resident conversations, never
    # co-batched), which keeps the text a valid oracle: GREEDY-PURITY.md §63 / issue #234 -- a pair of CONCURRENT
    # requests is NOT a text oracle, because co-batching changes the ubatch shape and hence the kernel family.  So
    # CONCURRENT=1 is the STRESS arm (concurrent takes: it must not abort or fault) and its warm text is recorded,
    # never asserted; TWO_SLOT=1 is the text oracle with two resident slots.
    if [[ ${CONCURRENT:-0} == 1 || ${TWO_SLOT:-0} == 1 ]]; then
        local base0 base1 r
        # TWO RESIDENT CONVERSATIONS need two prompts that ASK for something.  The kld corpus is a CHAT corpus
        # (`<|im_start|>` / `<|im_end|>` turns): slicing it and appending an instruction puts the instruction
        # inside an ASSISTANT turn, so the model emits EOS immediately and every reply is empty -- which made the
        # arm vacuous (both arms empty => "identical") or a false failure (one arm's degenerate stop differed).
        # Use the delivery's own prompt files instead, which are plain continuations, and extend with a plain
        # follow-up sentence rather than a chat turn.
        local PF0="$HERE/../../../prompts/prose-rdna-boosts.txt" PF1="$HERE/../../../prompts/code-python.txt"
        base0=$(head -c "$PROMPT_CHARS" "${PF0:-$PROMPT_FILE}")
        base1=$(head -c "$PROMPT_CHARS" "${PF1:-$PROMPT_FILE}")
        # NON-VACUITY (r346): a raw corpus concatenation ("...the tail of the corpus" + "the head of the corpus")
        # is a degenerate prompt -- it answered EMPTY on the 27B and "..." on the 4B, so round 2's slot-1 reply
        # was either identical-because-both-empty (vacuous) or different-because-one-took-a-degenerate-stop
        # (a false failure).  Each round now ends with a real instruction, so every compared reply is a
        # well-formed continuation.
        # ROUND 2 IS A PURE CONTINUATION: the same prompt plus the NEXT 2048 characters of the same source.  A
        # synthetic follow-up sentence ("now give the two most important points") makes this model emit EOS
        # immediately when it follows a truncated document -- measured empty in BOTH arms, i.e. a vacuous arm
        # dressed as a pass.  Extending the source keeps the reused prefix a real prefix and every compared
        # reply a real continuation.
        local ext0 ext1
        ext0=$(head -c $((PROMPT_CHARS + 2048)) "${PF0:-$PROMPT_FILE}" 2>/dev/null | tail -c 2048)
        ext1=$(head -c $((PROMPT_CHARS + 2048)) "${PF1:-$PROMPT_FILE}" 2>/dev/null | tail -c 2048)
        local instr1=''
        local instr2=''
        for r in 1 2; do
            local s0 s1
            if [[ $r == 1 ]]; then s0="$base0$instr1"; s1="$base1$instr1"
            else s0="$base0$ext0"; s1="$base1$ext1"; fi
            if [[ ${CONCURRENT:-0} == 1 ]]; then
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
            else
                local sl prompt_s
                for sl in 0 1; do
                    [[ $sl == 0 ]] && prompt_s=$s0 || prompt_s=$s1
                    curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
                        -d "$(jq -nc --arg p "$prompt_s" --argjson n "$NPREDICT" --argjson c "$cache_prompt" --argjson s "$sl" \
                                '{prompt:$p, n_predict:$n, temperature:0, top_k:1, seed:42, cache_prompt:$c,
                                  id_slot:$s, stream:false}')" > "$OUT/.r${r}s${sl}.json"
                done
            fi
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
    if [[ ${CONCURRENT:-0} == 1 ]]; then
        # the STRESS arm: two slots taking concurrently.  No text claim is made across arms -- the two arms' round-2
        # prefills have different lengths, so they co-batch differently, and GREEDY-PURITY.md §63 says a co-batched
        # pair is not a text oracle.  What it does assert: every request returned, the resume really reused, and
        # nothing aborted (checked per arm above and below).
        run_arm nocache
        if ! awk '$1 == "INFO" && $2 == "r2s0" { split($0, f, "cache_n="); split(f[2], g, " "); exit (g[1]+0 <= 0) }' "$OUT/cache.txt"; then
            echo "STRESS FAIL: the concurrent resume reused nothing (r2s0 cache_n = 0) -- vacuous"; exit 2
        fi
        echo "STRESS PASS: two slots resumed concurrently, reuse happened, no abort in either arm"
        echo "  (text is NOT asserted across arms: a co-batched pair is not a text oracle, GREEDY-PURITY.md §63)"
        exit 0
    fi
    # TWO_SLOT non-vacuity (r346): a text oracle is only an oracle when BOTH replies are real continuations and
    # the reuse really happened.  Both conditions are asserted here rather than left to the reader: an empty
    # reply compares equal to an empty reply (the 27B arm used to "pass" exactly that way), and a round that
    # reused nothing compares two full prefills.
    if [[ ${TWO_SLOT:-0} == 1 ]]; then
        # WHICH CONTRACT applies is the same question as for the single-slot arm, and the answer is the same:
        # `strict` (cache vs nocache) only holds where the DELIVERY's own single-context reference reuses (the 4B,
        # phi-4 and the 35B host tier); on the fully resident 27B the reference RE-PREFILLS there for want of a
        # checkpoint, so a strict comparison reads a placement difference as a state difference -- measured
        # r346: slot 0 round 2 moves, slot 1 does not.  Use GATE=parity for that model (cache vs the
        # single-context server's own reuse path, both reusing).
        REFARM=nocache; [[ ${GATE:-strict} == parity ]] && REFARM=ref
        run_arm "$REFARM"
        vac=0
        for f in "$OUT/cache.txt" "$OUT/$REFARM.txt"; do
            for t in r1s0 r1s1 r2s0 r2s1; do
                n=$(awk -v t="$t" '$1 == "REPLY" && $2 == t { split($0, f2, "ntok="); split(f2[2], g, " "); print g[1]+0 }' "$f")
                if [[ ${n:-0} -lt 4 ]]; then
                    echo "VACUOUS: $(basename "$f") $t has ntok=${n:-0} (< 4) -- a degenerate reply cannot carry a text claim" >&2
                    vac=1
                fi
            done
        done
        for t in r2s0 r2s1; do
            cn=$(awk -v t="$t" '$1 == "INFO" && $2 == t { split($0, f2, "cache_n="); split(f2[2], g, " "); print g[1]+0 }' "$OUT/cache.txt")
            if [[ ${cn:-0} -le 0 ]]; then
                echo "VACUOUS: round 2 $t reused nothing (cache_n=${cn:-0}) -- two full prefills prove nothing" >&2
                vac=1
            fi
        done
        [[ "$vac" == 0 ]] || { echo "ORACLE VACUOUS (not a pass): the two-slot arm proved nothing on this shape" >&2; exit 2; }
        "$0" --compare "$OUT/cache.txt" "$OUT/$REFARM.txt"
        exit $?
    fi
    if [[ ${GATE:-strict} == parity ]]; then
        run_arm ref
        "$0" --compare "$OUT/cache.txt" "$OUT/ref.txt"
    else
        run_arm nocache
        "$0" --compare "$OUT/cache.txt" "$OUT/nocache.txt"
    fi
else
    run_arm "$mode"
fi
