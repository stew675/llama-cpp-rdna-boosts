#!/usr/bin/env bash
# MoE expert-cache take-over PROMISE lifetime gate (issue #327).
#
# WHY THIS EXISTS
#   A take-over hook tells the scheduler "do not fill this `input_cpy`, I serve the table from the arena", and the
#   consumer that then cannot serve it re-fills the skipped copy from the host master (`moe_cache_repair_copy`,
#   `TODO` #335/#336).  The promise used to be identified by the buffer ADDRESS alone (`taken_cpy`), and the
#   scheduler recycles those addresses across graphs and across the decode/prefill layouts, so a promise left over
#   from an earlier pass matched a buffer a LATER pass filled itself.  On a partial (non-identity) arena the
#   cheapest such pass is a wide one that `pf_in_place_locked` declines, so every routed table was re-read from the
#   host master: measured on the issue's config, 46 layers x 3 roles = 138 whole-table H2Ds (206-450 MiB each,
#   ~27.8 GiB) on EVERY 30-34 token prompt, +10.4 s of prefill (2.07 s -> 0.17 s once scoped).  A promise is now
#   scoped to ONE staging pass (`moe_cache_pass_advance`, advanced by the CUDA backend at the end of every
#   `graph_compute` call), and a stale one is DROPPED instead of repaired (`stale=` in the exit census).
#
#   The trap the gate exists for: the repair must not be disabled, only its vacuous invocations removed.  A captured
#   decode graph replays its launches without dispatching, so it never consumes its promise; that is what leaves the
#   stale one.  The `fault` arm is the other half -- `MOE_EXPERT_TAKEOVER_FAULT=1` breaks a promise INSIDE the pass
#   that made it, which must still repair, byte for byte.
#
# WHAT IT CHECKS
#   `seq`   the issue's shape: one server, 1 GPU, Flash-Next `-ncmoe 47 --host-experts pool -b/-ub 2048`, one slot,
#           requests in order (a ~5k-token prompt first, so the short ones follow a decode pass):
#     HARD  * NO `moe_cache_repair_copy` line in the whole run (the regression: 138 per short prompt, 276 in the
#             reporter's sequence)
#     HARD  * the exit census reads `take-over promises broken=0 repaired=0` AND `stale>0` -- a run whose stale
#             count is 0 never reached the code, so a green result there is vacuous
#     HARD  * the arm is not vacuous: `stale>0` in the exit census, which only a pass whose promise went unhonoured
#             produces -- an identity (fully resident) arena takes the take-over path and is SERVED, so it never
#             reaches the repair at all and would pass this gate without exercising anything
#     HARD  * the short requests really are in the reporter's band (`prompt_n` in 17..64, not cached), and the
#             repeated request's text is byte-identical to its first occurrence
#     INFO  prompt ms per request (the fix's ~12x on the short prompts).  Measured r307: 0 repairs, `stale=8178`,
#           LONG 4277 ms, the 29-token request 283 ms -> 160 ms on the repeat (r306: 283 -> 2077)
#   `fault` the genuine-repair control (3 devices `-sm layer -ncmoe $FAULT_NCMOE`, `-lme $FAULT_LME` -c 8192 -ub 512`, identity arena,
#           `llama-cli` on the KLD corpus):
#     HARD  * `MOE_EXPERT_TAKEOVER_FAULT=1` still repairs: `broken>0` and `repaired == broken`, `stale == 0`
#     HARD  * its greedy text is byte-identical to the no-fault arm (r298: `595 chars sha=c0e7e91d84c4`)
#
# Usage:
#   ./gate-moe-promise.sh [--keep] [--no-faults] [--seq-only|--fault-only]
#
# Environment:
#   LLAMA_SERVER   llama-server binary    (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   LLAMA_CLI      llama-cli binary       (default: next to LLAMA_SERVER; the `fault` arm)
#   MODEL          Flash-Next GGUF shard 1 (default: Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS)
#   SEQ_GPU        device for `seq`       (default: 0 -- the arm needs ONE device; `HIP_VISIBLE_DEVICES=$SEQ_GPU`)
#   SEQ_CTX        seq context            (default: 65536, the issue's)
#   SEQ_UB         seq `-b`/`-ub`         (default: 2048, the issue's: the wide ubatch is what makes the stale
#                                          promise's buffer reappear in a declining pass)
#   SEQ_NCMOE      seq `-ncmoe`           (default: 47, the issue's)
#   SEQ_LONG       long-prompt tokens     (default: 5000, approximate: the arm only needs a decode pass before the
#                                          short ones)
#   SEQ_SHORT      short-prompt chars     (default: 150, i.e. ~34 tokens on this tokenizer)
#   SEQ_NGEN       tokens per request     (default: 32; the longest decode pass anywhere in the gate)
#   FAULT_DEVS     devices for `fault`    (default: 0,2 -- TWO devices are what makes the arena IDENTITY here)
#   FAULT_CTX      fault context          (default: 8192, MANIFESTS gate 28's recipe)
#   FAULT_PROMPT   fault prompt           (default: prompts/kld-corpus-35b-in-domain.txt)
#   FAULT_NGEN     fault tokens           (default: 32)
#   PORT           server port            (default: 18099)
#   KEEPOUT        per-arm log dir        (default: mktemp -d)
#   HEATMAP        MOE_EXPERT_HEATMAP     (default: load -- the prior is an INPUT; a writable one is rewritten per
#                                          run, and the `fault` arm's text oracle is a specific prior's)
#
# Exit: 0 = pass, 1 = fail, 2 = usage/startup error.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/rocmlib.sh"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
LLAMA_CLI="${LLAMA_CLI:-$(dirname "$LLAMA_SERVER")/llama-cli}"
MODEL="${MODEL:-/llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL/Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf}"
SEQ_GPU="${SEQ_GPU:-0}"
SEQ_CTX="${SEQ_CTX:-65536}"
SEQ_UB="${SEQ_UB:-2048}"
SEQ_NCMOE="${SEQ_NCMOE:-48}"
SEQ_LONG="${SEQ_LONG:-5000}"
SEQ_SHORT="${SEQ_SHORT:-150}"
SEQ_NGEN="${SEQ_NGEN:-32}"
FAULT_DEVS="${FAULT_DEVS:-0,1,2}"   # r346: the repointed 94 GB default model needs a third card for this arm
FAULT_CTX="${FAULT_CTX:-8192}"
# FAULT_NCMOE (r346): the fault arm is the ISSUE's shape at -ncmoe 48, which the repointed 94 GB default model
# needs raised on this box (two 32 GiB cards at 48 did not fit the context; 56 does).  It is a knob, not a
# different claim: the arm still needs TWO devices and an IDENTITY arena to break a promise mid-pass.
FAULT_NCMOE="${FAULT_NCMOE:-40}"
# FAULT_LME (r346): the fault arm needs a pool SMALLER than the host expert set, or nothing ever evicts and
# MOE_EXPERT_TAKEOVER_FAULT=1 breaks nothing.  Measured on the repointed 94 GB model: an auto budget sizes the
# WHOLE set and breaks 0 promises; 8 GiB on 3 devices at -ncmoe 40 breaks 238 and repairs 238.
FAULT_LME="${FAULT_LME:-8}"    # r346: measured -- auto (whole set) broke NOTHING; 8 GiB evicts 238
FAULT_PROMPT="${FAULT_PROMPT:-$REPO_DIR/prompts/kld-corpus-35b-in-domain.txt}"
FAULT_NGEN="${FAULT_NGEN:-32}"
PORT="${PORT:-18099}"
HEATMAP="${HEATMAP:-load}"
KEEP=0; NOFAULTS=0; SEQ_ONLY=0; FAULT_ONLY=0

for a in "$@"; do
    case "$a" in
        --keep)       KEEP=1 ;;
        --no-faults)  NOFAULTS=1 ;;
        --seq-only)   SEQ_ONLY=1 ;;
        --fault-only) FAULT_ONLY=1 ;;
        -h|--help)    sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
DEVS=""
SRV_PID=""

cleanup() {
    if [ "$KEEP" -eq 1 ]; then
        echo "==> logs kept in $KEEPOUT"
    else
        rm -rf "$KEEPOUT"
    fi
}
trap cleanup EXIT

stop_srv() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
    fi
    SRV_PID=""
}

# the exit census, once per run
census() { # $1 log -> "broken repaired stale" on stdout ("? ? 0" if the line is absent)
    # The fields are read independently: a build before issue #327 prints no `stale=` at all, and that must read
    # as a number (0), not as an unparsed line.
    local line b r st
    line=$(grep -E "take-over promises broken=" "$1" | tail -1)
    b=$(printf '%s' "$line" | sed -nE 's/.*broken=([0-9]+).*/\1/p')
    r=$(printf '%s' "$line" | sed -nE 's/.*repaired=([0-9]+).*/\1/p')
    st=$(printf '%s' "$line" | sed -nE 's/.*stale=([0-9]+).*/\1/p')
    echo "${b:-?} ${r:-?} ${st:-0}"
}

# ---------------------------------------------------------------- seq: the issue's shape
run_seq() {
    local arm=seq dir="$KEEPOUT/seq"
    mkdir -p "$dir"
    DEVS="$SEQ_GPU"
    env LD_LIBRARY_PATH="$(rocm_ld_library_path)" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        MOE_EXPERT_HEATMAP="$HEATMAP" \
        MOE_HOST_POOL_RAM_PCT="${MOE_HOST_POOL_RAM_PCT:-75}" \
        GGML_CUDA_SLAB_HEADROOM_MIB="${GGML_CUDA_SLAB_HEADROOM_MIB:-1536}" \
        ROCBLAS_USE_HIPBLASLT=0 \
        "$LLAMA_SERVER" -m "$MODEL" \
        -ngl all -ncmoe "$SEQ_NCMOE" --host-experts pool \
        --load-mode auto --lazy-mode on \
        -c "$SEQ_CTX" -np 1 -fa on -ctk q8_0 -ctv q8_0 \
        -b "$SEQ_UB" -ub "$SEQ_UB" -t 8 \
        --cache-ram 0 --spec-type none \
        --override-tensor 'per_layer_token_embd.*=CPU' \
        --temp 0 --seed 42 \
        --host 127.0.0.1 --port "$PORT" -lv 4 > "$dir/log.txt" 2>&1 &
    SRV_PID=$!

    local up=0
    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        echo "FAIL  $arm: the server did not come up (see $dir/log.txt)"; tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_srv; return 1
    fi

    python3 - "$PORT" "$SEQ_LONG" "$SEQ_SHORT" "$SEQ_NGEN" "$dir/req.json" <<'PY'
import json, sys, urllib.request
port, nlong, nshort, ngen, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
# deterministic prompts: the long one only has to precede the short ones with a decode pass
base   = "The quick brown fox jumps over the lazy dog. "
long_p = base * max(1, nlong // 9)
short  = ("The quick brown fox jumps over the lazy dog and then it walks slowly back to the old barn "
          "where the farmer keeps his tools and the horses rest")[:nshort]
# the short prompt is sent twice: the repeat is the text oracle (same process, same residency state)
steps = [("LONG", long_p), ("S1", short), ("S2", short)]
res = {}
for name, text in steps:
    body = json.dumps({"prompt": text, "n_predict": ngen, "temperature": 0, "seed": 42,
                       "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        t = d.get("timings", {}) or {}
        res[name] = {"content": d.get("content") or "", "prompt_n": t.get("prompt_n"),
                     "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n")}
    except Exception as e:
        res[name] = {"error": str(e)}
json.dump(res, open(out, "w"), indent=1)
PY
    local rc=$?
    stop_srv
    sleep 2
    return $rc
}

# ---------------------------------------------------------------- fault: the genuine repair
run_fault() {
    local arm="$1" dir="$KEEPOUT/$1"
    mkdir -p "$dir"
    DEVS="$FAULT_DEVS"
    local envs=()
    [ "$arm" = "fault" ] && envs+=(MOE_EXPERT_TAKEOVER_FAULT=1)
    env LD_LIBRARY_PATH="$(rocm_ld_library_path)" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        MOE_EXPERT_HEATMAP="$HEATMAP" \
        "${envs[@]}" \
        timeout 3600 "$LLAMA_CLI" -m "$MODEL" \
        -ngl 99 -sm layer -ncmoe "$FAULT_NCMOE" --host-experts pool ${FAULT_LME:+-lme $FAULT_LME} \
        --load-mode mmap --lazy-mode auto \
        -c "$FAULT_CTX" -fa on -ctk q8_0 -ctv q8_0 \
        -ub 512 -b 512 -t 8 --temp 0 --seed 42 -n "$FAULT_NGEN" \
        --no-display-prompt --single-turn -f "$FAULT_PROMPT" -lv 4 > "$dir/log.txt" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL  $arm: llama-cli exited $rc (see $dir/log.txt)"; tail -5 "$dir/log.txt" | sed 's/^/      /'
        return 1
    fi
    return 0
}

if ! [ -x "$LLAMA_SERVER" ]; then echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; fi
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2; exit 2
fi

GATE_T0="$(date '+%Y-%m-%d %H:%M:%S')"
echo "==> MoE take-over promise gate (issue #327)"
echo "    model $MODEL"

fails=0
fail() { echo "FAIL  $*"; fails=$((fails + 1)); }
pass() { echo "PASS  $*"; }

if [ "$FAULT_ONLY" -eq 0 ]; then
    echo "-- seq: ${SEQ_UB}-wide ubatch after a decode pass, 1 GPU, -ncmoe $SEQ_NCMOE, partial arena"
    if run_seq; then
        log="$KEEPOUT/seq/log.txt"
        req="$KEEPOUT/seq/req.json"
        n_rep=$(grep -cE "moe_cache_repair_copy" "$log")
        read -r broken repaired stale <<<"$(census "$log")"
        broken=${broken:-?}; repaired=${repaired:-?}; stale=${stale:-0}

        if [ "$n_rep" -eq 0 ]; then
            pass "seq: no moe_cache_repair_copy line in the whole run"
        else
            fail "seq: $n_rep moe_cache_repair_copy line(s) -- a stale promise was repaired (issue #327):"
            grep -E "moe_cache_repair_copy" "$log" | head -3 | sed 's/^/      /'
        fi
        if [ "$broken" = "0" ] && [ "$repaired" = "0" ]; then
            pass "seq: census broken=0 repaired=0"
        else
            fail "seq: census reads broken=$broken repaired=$repaired (expected 0/0)"
        fi
        if [ "$stale" != "?" ] && [ "${stale:-0}" -gt 0 ] 2>/dev/null; then
            pass "seq: census stale=$stale (the arm reached the code -- not vacuous)"
        else
            fail "seq: census stale=${stale} -- the stale-promise path was never reached, so this arm proves nothing"
        fi

        if python3 - "$req" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
band = [k for k in ("S1", "S2") if k in r and r[k].get("error") is None]
if len(band) != 2:
    print("      a short request failed: %r" % r); sys.exit(1)
for k in band:
    n = r[k].get("prompt_n") or 0
    if not (17 <= n <= 64):
        print("      %s processed %s tokens: outside the 17..64 band this gate is about" % (k, n)); sys.exit(1)
    if r[k].get("cache_n") not in (None, 0):
        print("      %s reused %s cached tokens: a cached prompt does not prefill" % (k, r[k]["cache_n"])); sys.exit(1)
if r["S1"]["content"] != r["S2"]["content"]:
    print("      the repeated short request changed its text"); sys.exit(1)
if not r["S1"]["content"]:
    print("      the short request generated nothing"); sys.exit(1)
print("      S1/S2 prompt_n=%s/%s text=%d chars  LONG prompt_n=%s" %
      (r["S1"].get("prompt_n"), r["S2"].get("prompt_n"), len(r["S1"]["content"]), r["LONG"].get("prompt_n")))
sys.exit(0)
PY
        then
            pass "seq: both short requests prefilled in band and repeated byte-identically"
        else
            fail "seq: the short-request checks failed (see above)"
        fi
        if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
            fail "seq: the server aborted"
        fi
        if [ "$NOFAULTS" -eq 0 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
            if ! "$REPO_DIR/scripts/check-gpu-faults.sh" --since "$GATE_T0" --log "$log" >/dev/null 2>&1; then
                fail "seq: check-gpu-faults.sh is not clean (run it by hand on $log)"
            fi
        fi
        # INFO: the timings this gate exists to protect
        python3 - "$req" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for k in ("LONG", "S1", "S2"):
    v = r.get(k) or {}
    if v.get("prompt_ms"):
        print("INFO  %s: prompt_n=%s prompt_ms=%.0f" % (k, v.get("prompt_n"), v["prompt_ms"]))
PY
    else
        fail "seq: the arm did not run"
    fi
fi

if [ "$SEQ_ONLY" -eq 0 ]; then
    echo "-- fault: MOE_EXPERT_TAKEOVER_FAULT=1 on an IDENTITY arena ($FAULT_DEVS, -sm layer, -lme $FAULT_LME), vs no-fault"
    if run_fault ref && run_fault fault; then
        read -r rb rr rs <<<"$(census "$KEEPOUT/ref/log.txt")"
        read -r fb fr fs <<<"$(census "$KEEPOUT/fault/log.txt")"
        rb=${rb:-0}; rr=${rr:-0}; rs=${rs:-0}; fb=${fb:-0}; fr=${fr:-0}; fs=${fs:-0}
        href=$(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/ref/log.txt" 2>/dev/null | tail -1)
        hfault=$(python3 "$REPO_DIR/scripts/extract-generated.py" "$KEEPOUT/fault/log.txt" 2>/dev/null | tail -1)

        if [ "$fb" -gt 0 ] 2>/dev/null; then
            pass "fault: promises broken=$fb on a real mid-pass eviction (the arm is not vacuous)"
        else
            fail "fault: broken=$fb -- MOE_EXPERT_TAKEOVER_FAULT=1 broke nothing, so the repair is untested"
        fi
        if [ "$fr" = "$fb" ] && [ "$fb" -gt 0 ] 2>/dev/null; then
            pass "fault: repaired=$fr == broken=$fb"
        else
            fail "fault: broken=$fb repaired=$fr -- a broken promise must be repaired exactly once"
        fi
        if [ "$fs" = "0" ] && [ "$rs" = "0" ]; then
            pass "fault: stale=0 in both arms (a same-pass break is never stale)"
        else
            fail "fault: stale=ref/$rs fault/$fs -- a same-pass break was classified stale"
        fi
        if [ -n "$href" ] && [ "$href" = "$hfault" ]; then
            pass "fault: text byte-identical to the no-fault arm ($href)"
        else
            fail "fault: text differs -- ref [$href] fault [$hfault]"
        fi
    else
        fail "fault: the arm(s) did not run"
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS (MoE take-over promise scoped to its staging pass; the broken-promise repair still works)"
    exit 0
fi
echo "==> FAIL ($fails check(s))"
exit 1
