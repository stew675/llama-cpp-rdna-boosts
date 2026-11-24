#!/usr/bin/env bash
# gate-arms-multislot -- the two-arm server under load: 4 slots, 2 prefilling + 2 decoding, 20 rounds.
#
# THE POINT: the two-thread design must survive what the single-context server cannot even express -- two
# prompts being processed on one arm while two decoders keep emitting on the other, round after round, with
# fresh prompts each time (so no round is served out of a previous round's prompt cache).
#
# WHAT IT CHECKS (HARD)
#   * the server SURVIVES every round and its log has no GGML_ASSERT / abort / fatal error
#   * every request got a response (a stalled decode arm shows up here as a round that never completes)
#   * the decode arm really carried slots: the census reports steps and handovers > 0, and its widest batch is
#     >= 2 slots (with 2 decoders there must be a round where they shared one decode batch)
#   * the reservation census ended with zero cells reserved (the I4 bitmap under real load)
#   * no GPU fault in the journal or the arm's log
#   INFO: VRAM/RSS before and after, and the wall time of the whole soak.
#
# NO TEXT CLAIM, deliberately: with several requests in flight the batch composition varies with arrival order
# and the composition picks the kernels, so even the UNMODIFIED server reads a different answer run to run on
# some shapes (INVARIANTS r362-wip item 5).  The byte claims live in gate-arms-n1-memory.sh (one request in
# flight) and gate-arms-itl.sh (a fixed pair of requests).
#
# USAGE: gate-arms-multislot.sh [--model FILE] [--rounds N] [--np N] [--extra "ARGS"] [--keep]
#
# ARMS selects the arm under test (default `--prefill-decode-arms`).  `ARMS="--prefill-decode-arms-mode off"`
# runs the SAME soak as the pair-OFF control: the delivery's single-context server on the same shape, where
# the two decode-arm census checks are SKIPped (there is no decode arm to carry a slot) and every other
# check still applies.  A soak that only ever runs one arm cannot tell "the pair broke it" from "the shape
# does that".
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$V2/../.." && pwd)"
MODEL="${MODEL:-/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf}"
EXTRA="${EXTRA--ncmoe 40}"   # `EXTRA=` (empty) means "no extra args"
ROUNDS="${ROUNDS:-20}"
NP="${NP:-4}"
CTX="${CTX:-16384}"
NPRED="${NPRED:-16}"
PORT="${PORT:-19700}"
ARMS="${ARMS:---prefill-decode-arms}"

# A DRAFTER arm needs at least 20 rounds for this gate's own overlap check ("two decoders shared ONE decode
# batch at least once").  A verify step emits up to `n_max + 1` tokens, so a 16-token request finishes in ~6
# decode-arm steps and two slots rarely overlap: at 10 rounds one run in eight saw `widest batch 1` for the whole
# soak -- no fault, no error, the CHECK failed (`INVARIANTS.md` r378-wip item 2, `HANDOVER.md` §7.10).  Raised
# rather than lowered: the check is the non-vacuity proof, so the soak is made long enough to earn it.
case "$ARMS $EXTRA" in
    *--spec-type*)
        # `--spec-type none` is not a drafter
        if ! grep -q -- "--spec-type none" <<<"$ARMS $EXTRA" && [ "${ROUNDS:-0}" -lt 20 ]; then
            echo "note: a speculator arm needs >= 20 rounds for the overlap check; raising ROUNDS $ROUNDS -> 20" >&2
            ROUNDS=20
        fi
        ;;
esac
# PIN THE RESIDENCY PRIOR (AGENTS.md rule 5, GREEDY-PURITY.md §89): on a host-expert shape the expert-cache
# RESIDENCY picks the MoE kernel set, and `MOE_EXPERT_HEATMAP` DEFAULT WRITES the persisted map, so two soaks
# in a row start from different priors -- which flipped this gate's outcome (measured 2026-11-23: the same
# binary passed 3/3, then failed 3/3 once the map had been rewritten by its own runs).  `load` = load+seed,
# never write.
export MOE_EXPERT_HEATMAP="${MOE_EXPERT_HEATMAP:-load}"
OUT="${OUT:-$HOME/.cache/rdna-boosts/gates/arms-multislot}"
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build-rocm/bin}"
mkdir -p "$OUT"
fails=0
hard() { if [ "$1" = 1 ]; then echo "PASS  $2"; else echo "FAIL  $2"; fails=$((fails+1)); fi; }

cd "$HOME/llama.cpp" || exit 1
SINCE="$(date '+%Y-%m-%d %H:%M:%S')"
LOG="$OUT/soak.log"
rss() { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }
vram() { rocm-smi --showmeminfo vram 2>/dev/null | sed -n "s/.*GPU\\[0\\].*Used Memory.*: \\([0-9]*\\).*/\\1/p" | head -1; }

# shellcheck disable=SC2086
setsid "$BIN/llama-server" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -np "$NP" -c "$CTX" \
    --kv-unified -ngl 99 -t 16 --no-webui -lv 4 $EXTRA $ARMS > "$LOG" 2>&1 &
SPID=$!
# Resolve the SERVER pid itself (not just the job pid) and give it a real chance to run its teardown: the
# reservation census and the decode-arm counters print from DESTRUCTORS, so a SIGKILL loses them.
SRV=""
cleanup() {
    [ -n "$SRV" ] || SRV="$(pgrep -n -f "llama-server.*--port $PORT" 2>/dev/null)"
    [ -n "$SRV" ] && { kill -TERM "$SRV" 2>/dev/null; }
    for _ in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    [ -n "$SRV" ] && kill -KILL "$SRV" 2>/dev/null
    [ -n "${SPID:-}" ] && kill -KILL -"$SPID" 2>/dev/null
    true
}
SRV="$(pgrep -n -f "llama-server.*--port $PORT" 2>/dev/null)"
trap cleanup EXIT

for _ in $(seq 1 240); do
    kill -0 "$SPID" 2>/dev/null || { echo "FAIL  the server died during load"; tail -5 "$LOG"; exit 1; }
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL  never healthy"; tail -5 "$LOG"; exit 1; }
[ -n "$SRV" ] || SRV="$(pgrep -n -f "llama-server.*--port $PORT" 2>/dev/null)"
echo "==> multislot soak: $(basename "$MODEL") np=$NP ctx=$CTX rounds=$ROUNDS '${EXTRA}' arm='${ARMS}'"
echo "    vram before $(vram) | rss before $(rss "$SPID") KiB"

t0=$(date +%s)
python3 - "$PORT" "$ROUNDS" "$NPRED" "$REPO" "$OUT/soak.json" <<'PY'
import concurrent.futures, json, sys, urllib.request

port, rounds, npred, repo, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
long_a = open(f"{repo}/prompts/reasoning.txt", encoding="utf-8").read()
long_b = open(f"{repo}/prompts/code-python.txt", encoding="utf-8").read()

def ask(args):
    rnd, slot, text = args
    body = json.dumps({"prompt": text, "n_predict": npred, "temperature": 0.0, "seed": 42,
                       "cache_prompt": False, "id_slot": slot, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1200))
        return (rnd, slot, len(d.get("content", "")), None)
    except Exception as e:
        return (rnd, slot, 0, str(e))

res = []
for rnd in range(rounds):
    # 2 prefills + 2 decodes per round; the prompts alternate so nothing is served from a previous round's cache
    jobs = [(rnd, 0, long_a), (rnd, 1, long_b), (rnd, 2, "The capital of France is"),
            (rnd, 3, "Write one line about rivers.")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        res.extend(ex.map(ask, jobs))

bad = [r for r in res if r[3] is not None]
short = [r for r in res if r[3] is None and r[2] == 0]
json.dump({"rounds": rounds, "n": len(res), "errors": bad[:5], "empty": short[:5]}, open(out, "w"), indent=1)
# an EMPTY completion is legitimate (a reasoning prompt can stop on its first token); a request that raised is
# the failure this gate is looking for
print(f"    {len(res)} requests, {len(bad)} errors, {len(short)} empty (a stop on the first token is valid)")
sys.exit(0 if not bad else 1)
PY
soak_rc=$?
t1=$(date +%s)
echo "    vram after $(vram) | rss after $(rss "$SPID") KiB | wall $((t1 - t0)) s"

hard "$([ "$soak_rc" = 0 ] && echo 1 || echo 0)" "every request in every round completed (no error, no stall)"
hard "$(grep -qaE 'GGML_ASSERT|GGML_ABORT|fatal error' "$LOG" && echo 0 || echo 1)" "the server log has no assertion/abort"
census="$(grep -oE 'the decode arm is generating: [0-9]+ step\(s\), [0-9]+ handover\(s\), widest batch [0-9]+' "$LOG" | tail -1)"
echo "      census: ${census:-<absent>}"
steps="$(sed -n 's/.*generating: \([0-9]*\) step.*/\1/p' <<<"$census")"
hand="$(sed -n 's/.*, \([0-9]*\) handover.*/\1/p' <<<"$census")"
wide="$(sed -n 's/.*widest batch \([0-9]*\).*/\1/p' <<<"$census")"
# the pair-OFF control arm has no decode arm: the two census checks do not apply to it (SKIP, not PASS)
PAIR=0
case " $ARMS " in *" --prefill-decode-arms "*) PAIR=1 ;; esac
if [ "$PAIR" = 1 ]; then
    hard "$([ -n "${steps:-}" ] && [ "${steps:-0}" -gt 0 ] && [ "${hand:-0}" -gt 0 ] && echo 1 || echo 0)" \
        "the decode arm carried slots (steps=$steps handovers=$hand)"
    hard "$([ -n "${wide:-}" ] && [ "${wide:-0}" -ge 2 ] && echo 1 || echo 0)" \
        "two decoders shared ONE decode batch at least once (widest batch=$wide)"
else
    echo "SKIP  decode-arm census: this is the pair-OFF control (arm='${ARMS}')"
fi
# The reservation census is printed from the CACHE DESTRUCTOR, so a soak whose server is killed before its
# teardown finishes the cache would read "absent" and say nothing about the bitmap.  It is a HARD check where the
# teardown is deterministic (gate-arms-n1-memory.sh, one short arm); here it is asserted whenever the destructor
# ran and reported otherwise, so a leaked reservation is still caught on the arm where it can be.
# NON-VACUITY for the slab-armed (host-expert) shapes: when the slab is live, the pair's decode arm must have
# been given a pinned AUXILIARY narrow compute region of its own.  Without one its compute buffer aliased the
# prefill arm's at base 0, which is the defect this gate caught on 2026-11-23 (two live 1280 MiB views at
# offset 0 -> HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION in the K rope's SET_ROWS).  A GPU-resident shape
# never arms the slab, so the check does not apply there.
if [ "$PAIR" = 1 ] && grep -qaE 'ggml_cuda_slab_init_locked: device [0-9]+: slab ' "$LOG"; then
    # It must NOT be the work pool's shared narrow base.  A PINNED region is the only accepted answer: an
    # anonymous allocation outside the slab was measured (STEP 7b, 2026-11-23) and REJECTED -- the outside
    # cushion is the slab's growth fuel (`slab_extend` maps physical from it), so taking a 1280 MiB compute
    # buffer from it starved the arena (42 `arena alloc failed` lines) and aborted the run.
    aux="$(grep -oE 'aux narrow-[0-9]+ \(borrowing-context compute\) pinned at' "$LOG" | tail -1)"
    hard "$([ -n "$aux" ] && echo 1 || echo 0)" \
        "the slab is armed and the decode arm's compute buffer got its OWN pinned region (${aux:-<absent>})"
else
    echo "SKIP  aux-region pin: the slab is not armed on this shape (or this is the pair-OFF control)"
fi
resv="$(grep -oE 'shared KV cache teardown: [0-9]+ of [0-9]+ cell\(s\) still reserved' "$LOG" | tail -1)"
leaked="$(sed -n 's/.*teardown: \([0-9]*\) of .*/\1/p' <<<"$resv")"
if [ -n "$resv" ]; then
    hard "$([ "${leaked:-1}" -eq 0 ] && echo 1 || echo 0)" "the shared cache ended with ZERO cells reserved ($resv)"
else
    echo "SKIP  reservation census: the teardown did not reach the cache destructor (see n1-memory, where it is HARD)"
fi
if "$REPO/scripts/check-gpu-faults.sh" --since "$SINCE" --log "$LOG" > "$OUT/faults.txt" 2>&1; then
    hard 1 "kernel journal and the server log clean"
else
    hard 0 "GPU fault(s) -- see $OUT/faults.txt"; tail -3 "$OUT/faults.txt" | sed 's/^/      /'
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: $ROUNDS rounds x $NP slots on one memory, the decode arm sharing batches, no fault"
    exit 0
fi
echo "==> FAIL ($fails check(s))  (logs: $OUT)"
exit 1
