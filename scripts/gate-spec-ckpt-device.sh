#!/usr/bin/env bash
# Speculative-checkpoint location gate (LLAMA_SERVER_SPEC_CKPT_DEVICE, PR #254, r184).
#
# WHY THIS EXISTS
#   On a recurrent or hybrid target the server saves a checkpoint of the target's recurrent state before a
#   verify whose draft is longer than n_rs_seq, and restores it when only part of the draft is accepted.  It
#   used to go to host memory (149.6 MiB for Qwen3.8-27B, ~4 ms per save on this box, ~16 ms on a slower
#   CPU).  r184 passes LLAMA_STATE_SEQ_FLAGS_ON_DEVICE for that checkpoint, so save and restore are device
#   copies into one staging buffer per slot that the context keeps.
#
#   That is a *state copy*, and the device arm is not a byte image: `data_tgt` holds only the metadata (the io
#   header and the cell ranges) while the payload lives in the context's per-seq staging buffer
#   (llama_context::mem_storage), which the next save overwrites and the restore copies back into the live
#   cells.  A checkpoint restored after a save the stage no longer holds, or with flags that disagree with the
#   ones it was saved with, restores the wrong bytes SILENTLY: no crash, no assert, a wrong answer a few
#   tokens later.  tests/test-llama-context's cp_d cases cover the library io class, NOT the server's use of
#   it, and neither the coherence gate nor gate-ckpt-reuse.sh (a context checkpoint, a different object)
#   reaches this code.
#
# WHAT IT CHECKS
#   HARD  * no arm aborts (GGML_ASSERT / terminate / segfault) and no GPU fault
#   HARD  * the device arm really took the device path on its slot (the "in device memory" decision line) and
#           really created speculative checkpoints - otherwise the arm tests nothing and this SKIPs
#   HARD  * the --no-kv-offload arm takes the HOST path with its own reason ("state is not offloaded"): with
#           -nkvo the state is in a host buffer, so a "device" decision there would be about a GPU the context
#           does not use (the pass that found this logged "device memory ... 13502.5 MiB free on ROCm0" for a
#           host-resident state).  That arm is NOT part of the text comparison: moving the KV cache off the GPU
#           selects different kernels on its own, so its text legitimately differs
#   HARD  * every request's greedy text, drafted tokens and accepted tokens are byte-identical between the arms
#   INFO  per request t/s per arm, the checkpoint counts (created / restored) and the device-memory delta
#
# The request set is deliberately n-gram heavy: the first request rewrites a repository file (a one-line comment
# above every function), so the n-gram drafter proposes long drafts, `draft.size() > n_rs_seq` holds on many
# steps, and the draft is NOT fully accepted - which is what makes the server save a checkpoint AND restore one
# (a verbatim copy task is accepted 100 % and never restores).  A plain prose completion rarely takes a
# checkpoint at all, and that is the second request: the feature must be inert there.  The rewritten file's
# sha256 is printed so a record can name the exact input; nothing under prompts/ is used, so no shipped prompt
# is edited in place.
#
# Usage:
#   ./gate-spec-ckpt-device.sh [--keep] [--no-faults]
#
# Environment:
#   LLAMA_SERVER   llama-server binary  (default: ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL          target GGUF, hybrid  (default: /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf)
#   DRAFT          optional draft GGUF  (default: empty -> --spec-type ngram-mod only, which on a recurrent
#                                        target is the FULL seq_rm type and takes a checkpoint on any draft)
#   SPEC_TYPE      speculative type     (default: ngram-mod, used only when DRAFT is set)
#   COPY_FILE      the file the copy request reproduces (default: scripts/cleanup-ghcr.py)
#   CTX            context              (default: 16384)
#   NGEN_COPY      tokens for the copy request   (default: 1200)
#   NGEN_PROSE     tokens for the prose request  (default: 256)
#   NGPU           GPUs (default: 1)    -> HIP_VISIBLE_DEVICES=0; the device path is single-GPU by design
#   NCMOE          n-cpu-moe            (default: 0)
#   EXTRA_ARGS     extra llama-server args (default: -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048)
#   PORT           server port          (default: 18099)
#   KEEPOUT        dir for the per-arm logs (default: mktemp -d)
#
# Exit: 0 = pass (or SKIP: the device path was not reachable, e.g. the state did not fit), 1 = fail, 2 = usage.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
DRAFT="${DRAFT:-}"
SPEC_TYPE="${SPEC_TYPE:-ngram-mod}"
COPY_FILE="${COPY_FILE:-$REPO_DIR/scripts/cleanup-ghcr.py}"
CTX="${CTX:-16384}"
NGEN_COPY="${NGEN_COPY:-1200}"
NGEN_PROSE="${NGEN_PROSE:-256}"
NGPU="${NGPU:-1}"
NCMOE="${NCMOE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:--fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048}"
PORT="${PORT:-18099}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
mkdir -p "$KEEPOUT"
# the fault window starts here, before the first server (#294)
GATE_SINCE_TS="$(date -d '-1 second' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"

case "$NGPU" in
  1) DEVS=0 ;;
  2) DEVS=0,1 ;;
  3) DEVS=0,1,2 ;;
  4) DEVS=0,1,2,3 ;;
  *) DEVS="$NGPU" ;;
esac

KEEP=0
FAULTS=1
for a in "$@"; do
    case "$a" in
        --keep) KEEP=1 ;;
        --no-faults) FAULTS=0 ;;
        -h|--help) sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "usage: $0 [--keep] [--no-faults]" >&2; exit 2 ;;
    esac
done

[ -x "$LLAMA_SERVER" ] || { echo "ERROR: no llama-server at $LLAMA_SERVER" >&2; exit 2; }
[ -f "$MODEL" ]        || { echo "ERROR: no model at $MODEL" >&2; exit 2; }
[ -f "$COPY_FILE" ]    || { echo "ERROR: no COPY_FILE at $COPY_FILE" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 2; }

SRV_PID=""
stop_arm() {
    if [ -n "${SRV_PID:-}" ]; then
        kill "$SRV_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null
        SRV_PID=""
    fi
}

# drive one arm: start the server, run the two requests, stop it.  Writes $arm/{log.txt,steps.json}.
run_arm() {
    local arm="$1"; shift
    local dir="$KEEPOUT/$arm"
    mkdir -p "$dir"
    local pid
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx120X/lib}" \
        HIP_VISIBLE_DEVICES="$DEVS" \
        "$@" \
        "$LLAMA_SERVER" -m "$MODEL" \
        ${DRAFT:+-md "$DRAFT"} --spec-type "$SPEC_TYPE" \
        -sm layer -ngl 99 -ncmoe "$NCMOE" -c "$CTX" -t 8 -lv 5 \
        --host 127.0.0.1 --port "$PORT" $EXTRA_ARGS ${ARM_EXTRA:-} > "$dir/log.txt" 2>&1 &
    pid=$!
    SRV_PID="$pid"

    local up=0
    for _ in $(seq 1 900); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" -ne 1 ]; then
        echo "FAIL  $arm: the server did not come up (see $dir/log.txt)"
        tail -5 "$dir/log.txt" | sed 's/^/      /'
        stop_arm
        return 1
    fi

    python3 - "$PORT" "$NGEN_COPY" "$NGEN_PROSE" "$COPY_FILE" "$dir/steps.json" "$pid" <<'PY'
import json, os, sys, urllib.request

port, ngen_copy, ngen_prose, copy_file, out_file, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
src = open(copy_file, encoding="utf-8", errors="replace").read()

# R1 is n-gram heavy on purpose: a verbatim copy makes the drafter propose long drafts, so the target's
# speculative checkpoint is taken on many steps.  R2 is plain prose with no draft to copy: the feature must be
# inert there (no checkpoint at all, or far fewer), and it is the second half of the text comparison.
reqs = {
    "copy":  {"messages": [{"role": "user", "content":
              "Add a one-line comment above every function definition in this Python file saying what the function\n"
              "does. Change nothing else and output the whole modified file only.\n\n```python\n" + src + "\n```"}],
              "max_tokens": ngen_copy},
    "prose": {"messages": [{"role": "user", "content":
              "Write about six short paragraphs on why small, boring, reproducible builds are worth the effort."}],
              "max_tokens": ngen_prose},
}
for r in reqs.values():
    r.update(temperature=0, chat_template_kwargs={"enable_thinking": False})

def rss_mib(p):
    try:
        for line in open("/proc/%d/status" % p):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    except Exception:
        pass
    return None

res = {}
dead = False
for name, body in reqs.items():
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        t = d.get("timings", {}) or {}
        res[name] = {"content": d["choices"][0]["message"].get("content") or "",
                     "tps": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                     "draft": t.get("draft_n"), "acc": t.get("draft_n_accepted"), "error": None}
    except Exception as e:  # a dead server shows up here (connection reset) or in the abort scan
        res[name] = {"content": "", "tps": None, "n": None, "draft": None, "acc": None, "error": str(e)}
        dead = True
        break

json.dump({"steps": res, "dead": dead, "rss_mib": rss_mib(pid)}, open(out_file, "w"), indent=1)
PY
    local rc=$?
    stop_arm
    return $rc
}

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: something is already listening on port $PORT (set PORT=...)" >&2
    exit 2
fi

echo "==> speculative-checkpoint location gate (layer, $NGPU GPU, -ncmoe $NCMOE, ctx $CTX)"
echo "    model $MODEL${DRAFT:+ + draft $DRAFT ($SPEC_TYPE)}"
echo "    copy request input: $COPY_FILE ($(sha256sum "$COPY_FILE" | cut -c1-16)...)"

# Three arms, one binary; they differ only in where the target's speculative checkpoint lives.
run_arm host env LLAMA_SERVER_SPEC_CKPT_DEVICE=0 || { echo "==> FAIL: the host arm did not run"; exit 1; }
run_arm dev  env                                   || { echo "==> FAIL: the device arm did not run"; exit 1; }
ARM_EXTRA="-nkvo"
run_arm nkvo env LLAMA_SERVER_SPEC_CKPT_DEVICE=1 || { echo "==> FAIL: the -nkvo arm did not run"; exit 1; }
ARM_EXTRA=""

fails=0

for arm in host dev nkvo; do
    log="$KEEPOUT/$arm/log.txt"
    if grep -qE "GGML_ASSERT|terminate called|Segmentation fault" "$log"; then
        echo "FAIL  $arm: the server aborted:"
        grep -m1 -E "GGML_ASSERT|terminate called|Segmentation fault" "$log" | sed 's/^/      /'
        fails=$((fails + 1))
    fi
done

# the device arm must really be on the device path, and really take checkpoints, or the comparison is empty
n_dev=$(grep -c "speculative checkpoints of the target in device memory" "$KEEPOUT/dev/log.txt" || true)
n_host=$(grep -c "speculative checkpoints of the target in host memory" "$KEEPOUT/host/log.txt" || true)
n_ckpt=$(grep -c "created speculative checkpoint" "$KEEPOUT/dev/log.txt" || true)
n_ckpt_host=$(grep -c "created speculative checkpoint" "$KEEPOUT/host/log.txt" || true)
n_restore=$(grep -c "restoring speculative checkpoint" "$KEEPOUT/dev/log.txt" || true)
n_restore_host=$(grep -c "restoring speculative checkpoint" "$KEEPOUT/host/log.txt" || true)
n_nkvo_dev=$(grep -c "speculative checkpoints of the target in device memory" "$KEEPOUT/nkvo/log.txt" || true)
n_nkvo_reason=$(grep -c "the context's state is not offloaded" "$KEEPOUT/nkvo/log.txt" || true)

echo
if [ "${n_dev:-0}" -eq 0 ]; then
    echo "SKIP  the device arm did not take the device path (no 'in device memory' decision):"
    grep -m2 "speculative checkpoints of the target" "$KEEPOUT/dev/log.txt" | sed 's/.*| /      /'
    echo "      the decision needs one GPU, an offloaded state and state + 512 MiB free; lower CTX / NCMOE,"
    echo "      or use a smaller state.  Nothing was compared.  logs kept in $KEEPOUT"
    exit 0
fi
if [ "${n_ckpt:-0}" -eq 0 ]; then
    echo "SKIP  the device arm created no speculative checkpoint, so the device save never ran"
    echo "      (needs a recurrent/hybrid target and a draft longer than n_rs_seq: keep SPEC_TYPE=ngram-mod)."
    echo "      logs kept in $KEEPOUT"
    exit 0
fi
if [ "${n_host:-0}" -gt 0 ]; then
    echo "FAIL  the kill-switch arm took the device path ($n_host 'in device memory' line(s)) - the switch is incomplete"
    fails=$((fails + 1))
else
    echo "PASS  kill switch: host arm 100 % host memory, device arm on the device path ($n_dev slot decision(s))"
fi
if [ "${n_nkvo_dev:-0}" -gt 0 ] || [ "${n_nkvo_reason:-0}" -lt 1 ]; then
    echo "FAIL  -nkvo: expected the host path with the 'state is not offloaded' reason, got:"
    grep -m2 "speculative checkpoints of the target" "$KEEPOUT/nkvo/log.txt" | sed 's/.*| /      /'
    fails=$((fails + 1))
else
    echo "PASS  -nkvo: host path, and it says why (the state is not offloaded)"
fi
echo "INFO  speculative checkpoints: device arm $n_ckpt created / $n_restore restored," \
     "host arm $n_ckpt_host / $n_restore_host"

python3 - "$KEEPOUT" <<'PY'
import json, sys

root = sys.argv[1]
arms = {}
for a in ("host", "dev", "nkvo"):
    try:
        arms[a] = json.load(open("%s/%s/steps.json" % (root, a)))
    except Exception as e:
        print("FAIL  %s: no results (%s)" % (a, e))
        sys.exit(1)

fails = 0
print()
print("  request   host t/s   device t/s   delta      draft/acc (host vs device)   text")
for name in ("copy", "prose"):
    h = arms["host"]["steps"].get(name) or {}
    d = arms["dev"]["steps"].get(name) or {}
    k = arms["nkvo"]["steps"].get(name) or {}
    if h.get("tps") is None or d.get("tps") is None or k.get("tps") is None:
        print("FAIL  %s: no completion (host %s / device %s / nkvo %s)" % (
            name, h.get("error"), d.get("error"), k.get("error")))
        fails += 1
        continue
    # host vs device only: the -nkvo arm moves the KV cache off the GPU, which changes the kernels and therefore
    # the text by itself, so it is checked for its decision (and for not aborting), never for text identity
    same = h.get("content") == d.get("content")
    same_da = (h.get("draft"), h.get("acc")) == (d.get("draft"), d.get("acc"))
    print("  %-9s %8.1f %12.1f %8s  %s/%s vs %s/%s   %s" % (
        name, h["tps"], d["tps"], "%+.1f%%" % ((d["tps"] / h["tps"] - 1) * 100 if h["tps"] else 0),
        h.get("draft"), h.get("acc"), d.get("draft"), d.get("acc"),
        "== host/device" if same else "DIFFERS"))
    if not same:
        print("        the greedy text is not identical across arms: a device checkpoint restored bytes the")
        print("        host path does not (the device arm is not a byte image - see the script header)")
        fails += 1
    if not same_da:
        print("        drafted/accepted tokens differ between host and device arms")
        fails += 1

rh = arms["host"].get("rss_mib")
rd = arms["dev"].get("rss_mib")
if rh and rd:
    print()
    print("  resident host memory at the end: host %d MiB, device %d MiB (%+d MiB: the state is in VRAM, not RSS)"
          % (rh, rd, rd - rh))

sys.exit(1 if fails else 0)
PY
rc=$?
[ "$rc" -ne 0 ] && fails=$((fails + 1))

# the only valid GPU-fault check: the kernel journal plus the arm logs, with the echoed prompts stripped.
# The window is this gate's own start timestamp, not a fixed "N min ago" guess (#294).
if [ "$FAULTS" -eq 1 ] && [ -x "$REPO_DIR/scripts/check-gpu-faults.sh" ]; then
    since_args=()
    [ -n "$GATE_SINCE_TS" ] && since_args=(--since "$GATE_SINCE_TS")
    fault_out="$("$REPO_DIR/scripts/check-gpu-faults.sh" "${since_args[@]}" --log "$KEEPOUT/dev/log.txt" 2>&1)"
    fault_rc=$?
    if [ "$fault_rc" -eq 0 ]; then
        echo "PASS  no GPU fault in the kernel journal or the device arm's log"
    else
        echo "FAIL  check-gpu-faults.sh reports a fault (exit $fault_rc):"
        echo "$fault_out" | tail -5 | sed 's/^/      /'
        fails=$((fails + 1))
    fi
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "==> PASS: the device checkpoint restores the same bytes the host path does (text, drafts and accepted"
    echo "    tokens identical), the kill switch keeps the host path, and -nkvo stays on it with its own reason."
    echo "    logs kept in $KEEPOUT"
    [ "${KEEP:-0}" = "1" ] || rm -rf "$KEEPOUT"
    exit 0
fi
echo "==> FAIL: $fails check(s) failed (logs in $KEEPOUT)" >&2
exit 1
