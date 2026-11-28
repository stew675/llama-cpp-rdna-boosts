#!/usr/bin/env bash
# KV admission gate (LLAMA_SERVER_KV_ADMIT, LLAMA_SERVER_KV_ADMIT_RESERVE -- block 16 amendment, r365)
# and the KV full stop (LLAMA_SERVER_KV_FULL_STOP -- block 16 amendment, r377, contributor PR #372).
#
# WHY THIS EXISTS
#   With a unified KV cache the server used to start every request that found a free slot, whether or not its
#   prompt fit beside what the other slots already held (the delivery's DEFAULT server is `-np -1` => 4 slots,
#   `kv_unified = true`).  Two prompts whose sum exceeds the context both prefilled until the cache was full, and
#   then decode failed: the retry loop can only clear IDLE slots (`try_clear_idle_slots` skips a processing one),
#   so it halved `n_batch` down to 1 and answered HTTP 500 "Context size has been exceeded." -- or, with a
#   speculator, "speculative batch index N is not inside the current sub-batch".  `server_slot_sched::admit` now
#   asks the server context for a whole-request go/no-go before a slot is chosen, and a refusal is deferred through
#   the stock "no slot is available" path.  Every other gate in this repo runs ONE request at a time, which is
#   exactly why this class had no coverage; a concurrent-server TEXT comparison is forbidden as an oracle
#   (GREEDY-PURITY.md §38) except in the one shape used below.
#
#   Admission is a PRE-LAUNCH veto, so it cannot stop a reply that grows PAST its reserve (r377, contributor PR
#   #372).  Before each step the generating slots' next tokens must fit in the free cells: idle slots are saved and
#   cleared first, and if that is not enough the reply that has generated the most ends with a length stop
#   (`truncated=true`, `stop_type=limit`, `finish_reason: "length"`) instead of the cache filling and the decode
#   retry splitting the batch, which is what answered HTTP 500.  The `full`/`full0`/`fullfit`/`fullpure` arms below
#   are that path and nothing else reaches it.
#
# WHAT IT CHECKS  (each arm a fresh server; `-c 8192 -np 4 --kv-unified` unless named otherwise)
#   HARD  fit      two prompts that fit side by side: both answered, and NOT ONE deferral (the gate must not see
#                  the feature firing where it is not needed)
#   HARD  fit0     the same pair on `LLAMA_SERVER_KV_ADMIT=0`: answered, and each text the same as `fit`'s
#                  (unchanged when it fits -- the amendment moves nothing here)
#   HARD  over     two copies of `prompts/prose-rdna-boosts.txt` (5246 tokens), stateless (`cache_prompt: false`):
#                  both answered, >= 1 deferral, and the second request starts only after the first finishes
#                  (`wall2 > wall1`) -- the shape that used to answer 500
#   HARD  over0    the same on the kill switch: both **HTTP 500** (NON-VACUITY -- a gate that cannot reproduce the
#                  pre-change failure is not a gate)
#   HARD  purity   each answer of `over` is BYTE-IDENTICAL to that same prompt run alone on a fresh server (the
#                  deferred request re-prefills all 5246 tokens and lands on the same text -- the admission
#                  decision must not move a token)
#   HARD  drain    four overlapping prompts: all answered, >= 3 deferrals, clean shutdown
#   HARD  reserve  the generation reserve really is `min(n_ctx / n_parallel, n_ctx_slot())`: the cells charged over
#                  the prompt read 2048 at `-c 8192`/4 slots and 4096 at `-c 16384`/4 slots, `256` with
#                  `LLAMA_SERVER_KV_ADMIT_RESERVE=256`, and `n_predict + prompt` when the request caps it
#   HARD  nouni    `-np 4` WITHOUT `--kv-unified`: zero deferrals and both answered (a per-slot cache cannot have
#                  slots take each other's cells, so the feature must be inert)
#   HARD  full     (r377, contributor PR #372) a reply that OUTGROWS its reserve: `-c 16384`, reserve 512, both whole
#                  prose prompts admitted and 5000-token replies, so the cache fills mid-reply -- both answered, ONE
#                  ends `stop_type=limit` with `truncated=true` and the other completes all 5000 tokens, the full-stop
#                  line fires exactly once, the startup census states the arm (`KV full stop: on`), and the decode
#                  retry NEVER runs (no "failed to find free space": the split is what aborted the slot before)
#   HARD  full0    the same on `LLAMA_SERVER_KV_FULL_STOP=0`: both **HTTP 500** (NON-VACUITY, and the pre-r377 build)
#   HARD  fullfit  the same pair whose replies FIT (500 each): both answered, nothing truncated, text byte-identical
#                  to the kill switch's (a feature that fires where it is not needed would move tokens)
#   HARD  fullstream the SSE arm: both streams complete, one ends `truncated=true`/`stop_type=limit` with text already
#                  delivered, the other completes, and the streamed token count is the count the cut logged (the
#                  full-stop path sets `has_next_token` and calls `send_final_response` itself, so a duplicated or
#                  lost terminal chunk can only show up here)
#   NOTE  these arms carry **no text oracle on purpose**: two co-batched requests on this model are not reproducible
#                  across server runs (measured over a stagger sweep and with a first-token barrier: a FIXED timing
#                  repeats exactly, the text moves when the timing moves), and a lone-run comparison is forbidden
#                  (GREEDY-PURITY.md §38/§63).  Purity is asserted in run (the cut's token count == the client's) and
#                  the byte-exact-prefix property was measured by hand at the same -c (WORKLOG.md 261128.006).
#   HARD  shut     SIGTERM with a deferred queue pending: the process exits on its own (a task that was never
#                  launched is answered at termination -- without that the exit hangs in HIP's teardown, measured)
#   HARD  spin     a waiting (deferred) task costs no CPU over a decode, and the server idles at ~0 ticks after the
#                  queue drains (a wake path that re-offers a task it cannot launch would spin at 100 %)
#   HARD  the per-arm logs carry no `GGML_ASSERT` / abort, and `scripts/check-gpu-faults.sh` is clean over the run
#   INFO  the prompt-cache state the deferred request restored from (reported, not asserted)
#
# Usage:  ./gate-kv-admit.sh [--keep] [--no-faults] [--self-test]
#   --self-test runs every arm with LLAMA_SERVER_KV_ADMIT=0 AND LLAMA_SERVER_KV_FULL_STOP=0 (both amendments'
#   pre-change behaviour) and requires the gate to FAIL: a gate that cannot reject the unfixed server is not a gate.
#   Run it once before trusting a green run.  Both switches must be stood down: the `full` arm still ends a reply
#   with admission off, so KV_ADMIT alone would leave it passing on the pre-r377 build.
# Exit:   0 = pass, 1 = fail, 2 = usage/prerequisite error.
#
# Environment:
#   LLAMA_SERVER  llama-server binary (default ~/llama.cpp/build-rocm/bin/llama-server)
#   MODEL         target GGUF (default the 4B Qwen3.5 Q8_0 -- dense enough to be fast, and a GDN hybrid, so the
#                 accounting is exercised against a recurrent+attention cache rather than a plain one)
#   CTX           context (default 8192 -- small on purpose: the overlap shape needs a refusal, not a big model)
#   NPAR          slots   (default 4, the value `-np -1` resolves to on the delivery's default server)
#   GPU           HIP_VISIBLE_DEVICES (default 0)
#   KEEPOUT       dir for the per-arm logs (default mktemp -d)
#   PORT          base port (each arm takes the next one)
#
# The prompts are the versioned ones (`prompts/README.md` carries their sha256); this gate generates no text of
# its own, so a pass is comparable across releases.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO_DIR/scripts/rocmlib.sh"

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}"
MODEL="${MODEL:-/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf}"
CTX="${CTX:-8192}"
NPAR="${NPAR:-4}"
GPU="${GPU:-0}"
PORT="${PORT:-18300}"
KEEPOUT="${KEEPOUT:-$(mktemp -d)}"
export LLAMA_SERVER MODEL CTX NPAR GPU PORT KEEPOUT REPO_DIR
export ROCM_LIB_DIR="$(rocm_ld_library_path 2>/dev/null || true)"

for f in "$LLAMA_SERVER" "$MODEL" "$REPO_DIR/prompts/prose-rdna-boosts.txt"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done
[ -x "$LLAMA_SERVER" ] || { echo "ERROR: $LLAMA_SERVER is not executable" >&2; exit 2; }

exec python3 - "$@" <<'PYEOF'
import hashlib, json, os, random, signal, subprocess, sys, threading, time, urllib.request, urllib.error

args     = sys.argv[1:]
KEEP     = "--keep" in args
NOFAULTS = "--no-faults" in args
SELFTEST = "--self-test" in args
BIN   = os.environ["LLAMA_SERVER"]
MODEL = os.environ["MODEL"]
CTX   = int(os.environ["CTX"])
NPAR  = int(os.environ["NPAR"])
GPU   = os.environ["GPU"]
PORT  = int(os.environ["PORT"])
DIR   = os.environ["KEEPOUT"]
REPO  = os.environ["REPO_DIR"]
LIB   = os.environ.get("ROCM_LIB_DIR") or None

PROSE = open(os.path.join(REPO, "prompts", "prose-rdna-boosts.txt"), encoding="utf-8").read()
P_A   = open(os.path.join(REPO, "prompts", "code-python.txt"), encoding="utf-8").read()
P_B   = open(os.path.join(REPO, "prompts", "recall.txt"), encoding="utf-8").read()

FAILS, INFOS, LOGS = [], [], []
_port = [PORT]

class Server:
    """One arm's server: fresh process, own port, log under $KEEPOUT."""
    def __init__(self, tag, ctx=None, npar=None, extra=None, env=None, unified=True):
        _port[0] += 1
        self.tag, self.port = tag, _port[0]
        cmd = [BIN, "-m", MODEL, "-ngl", "99", "-c", str(ctx or CTX),
               "--host", "127.0.0.1", "--port", str(self.port), "-lv", "4"]
        if npar is not None:
            cmd += ["-np", str(npar)]
        if unified:
            cmd += ["-kvu"]
        cmd += list(extra or [])
        e = dict(os.environ)
        e["HIP_VISIBLE_DEVICES"] = GPU
        if SELFTEST:
            # --self-test: make every arm behave like the pre-amendment server.  Both amendments of this gate
            # have their own kill switch and both must be stood down, or `full` passes on the pre-r377 build.
            e["LLAMA_SERVER_KV_ADMIT"]      = "0"
            e["LLAMA_SERVER_KV_FULL_STOP"]  = "0"
        if LIB:
            e["LD_LIBRARY_PATH"] = LIB + (":" + e["LD_LIBRARY_PATH"] if e.get("LD_LIBRARY_PATH") else "")
        for k, v in (env or {}).items():
            e[k] = v
        self.log = os.path.join(DIR, "arm-%s.log" % tag)
        self.f = open(self.log, "w")
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self.f,
                                  stderr=subprocess.STDOUT, env=e)
        self._wait_up()
        LOGS.append(self.log)

    def _wait_up(self, n=240):
        for _ in range(n):
            if self.p.poll() is not None:
                self.f.close()
                sys.exit("ERROR: %s did not load (see %s)" % (self.tag, self.log))
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=1).read()
                return
            except Exception:
                time.sleep(0.5)
        self.f.close()
        sys.exit("ERROR: %s did not come up (see %s)" % (self.tag, self.log))

    # ---- server state -------------------------------------------------------------------------------------------
    def lines(self):
        return open(self.log, errors="replace").read().splitlines()

    def admits(self):
        return [l for l in self.lines() if "KV admission: task" in l]

    def need_cells(self):
        a = self.admits()
        if not a:
            return None
        try:
            return int(a[-1].split("needs ")[1].split(" cells")[0])
        except Exception:
            return None

    def ticks(self):
        st = open("/proc/%d/stat" % self.p.pid).read().rsplit(")", 1)[1].split()
        return int(st[11]) + int(st[12])

    def stop(self, label="stop"):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(timeout=30); rc = self.p.returncode
        except subprocess.TimeoutExpired:
            self.p.kill(); self.p.wait(); rc = "HANG"
        self.f.close()
        return rc

    def abort(self):
        try:
            self.p.kill(); self.p.wait()
        except Exception:
            pass
        self.f.close()

    # ---- requests -----------------------------------------------------------------------------------------------
    def ask(self, prompt, n_predict=64, timeout=900, cache=True, slot=None, ignore_eos=False):
        body = {"prompt": prompt, "temperature": 0, "top_k": 1, "cache_prompt": cache}
        if n_predict is not None:
            body["n_predict"] = n_predict
        if slot is not None:
            body["id_slot"] = slot
        if ignore_eos:
            body["ignore_eos"] = True
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:%d/completion" % self.port, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=timeout))
            return dict(rc=200, wall=time.time()-t0, pn=r["timings"]["prompt_n"],
                        gen=r["timings"]["predicted_n"], stop=r.get("stop_type"), trunc=r.get("truncated"),
                        content=r["content"],
                        sha=hashlib.sha256(r["content"].encode()).hexdigest()[:12])
        except urllib.error.HTTPError as e:
            return dict(rc=e.code, wall=time.time()-t0, body=e.read()[:90].decode(errors="replace"))
        except Exception as e:
            return dict(rc="ERR", wall=time.time()-t0, body=type(e).__name__)

    def ask_stream(self, prompt, n_predict, slot, ignore_eos=True, timeout=1800):
        """Read one SSE /completion stream: the concatenated content deltas and the terminal chunk."""
        body = {"prompt": prompt, "temperature": 0, "top_k": 1, "cache_prompt": False, "n_predict": n_predict,
                "id_slot": slot, "stream": True}
        if ignore_eos:
            body["ignore_eos"] = True
        parts, final = [], None
        try:
            for raw in urllib.request.urlopen(urllib.request.Request(
                    "http://127.0.0.1:%d/completion" % self.port, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}), timeout=timeout):
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                if d.get("stop"):
                    final = d
                    break
                parts.append(d.get("content", ""))
            return dict(rc=200, text="".join(parts), final=final)
        except urllib.error.HTTPError as e:
            return dict(rc=e.code, body=e.read()[:90].decode(errors="replace"))
        except Exception as e:
            return dict(rc="ERR", body=type(e).__name__)

    def together_stream(self, prompts, n_predict, cache=False, stagger=0.2, ignore_eos=True, slots=None):
        out = [None] * len(prompts)
        def run(i):
            out[i] = self.ask_stream(prompts[i], n_predict, None if slots is None else slots[i], ignore_eos)
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for t in th:
            t.start(); time.sleep(stagger)
        for t in th:
            t.join()
        return out

    def together(self, prompts, n_predict=64, cache=True, stagger=0.2, timeout=900, ignore_eos=False, slots=None):
        out = [None] * len(prompts)
        def run(i):
            out[i] = self.ask(prompts[i], n_predict, timeout, cache, None if slots is None else slots[i], ignore_eos)
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
        for t in th:
            t.start(); time.sleep(stagger)
        for t in th:
            t.join()
        return out

def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   [%s]" % detail) if detail else ""))
    if not cond:
        FAILS.append(name)

def info(msg):
    print("  info  " + msg)

def rcs(res):
    return [r["rc"] for r in res]

def shas(res):
    return [r.get("sha") for r in res]

def no_abort(tag):
    bad = [l for l in open(os.path.join(DIR, "arm-%s.log" % tag), errors="replace")
           if "GGML_ASSERT" in l or "Aborted" in l or "GGML_ABORT" in l]
    check("%s: no assert / abort in the log" % tag, not bad, (bad[0][:120] if bad else ""))

print("== KV admission gate (%s, -c %d, %d slots, GPU %s) ==" % (os.path.basename(MODEL), CTX, NPAR, GPU))

# ----------------------------------------------------------------------------------------------------------------
print("== fit: a pair that fits is not deferred, and its text is the kill switch's ==")
s = Server("fit"); fit = s.together([P_A, P_B]); n_fit = len(s.admits()); rc = s.stop()
s = Server("fit0", env={"LLAMA_SERVER_KV_ADMIT": "0"}); fit0 = s.together([P_A, P_B]); s.stop()
check("fit: both answered", all(r["rc"] == 200 for r in fit), str(rcs(fit)))
check("fit: no deferral", n_fit == 0, "deferrals=%d" % n_fit)
check("fit: text identical to LLAMA_SERVER_KV_ADMIT=0", shas(fit) == shas(fit0), "%s vs %s" % (shas(fit), shas(fit0)))
check("fit: clean shutdown", rc == 0, str(rc))
no_abort("fit")

# ----------------------------------------------------------------------------------------------------------------
print("== lone: the reference text for one prose prompt on a fresh server ==")
s = Server("lone"); lone = s.ask(PROSE, 64, cache=False); s.stop()
check("lone: answered", lone["rc"] == 200, str(lone.get("body", "")))
LONE_SHA = lone.get("sha")

# ----------------------------------------------------------------------------------------------------------------
print("== over / over0 / purity: the overlap that used to answer 500 ==")
s = Server("over")
ov = s.together([PROSE, PROSE], 64, cache=False)
n_over = len(s.admits()); rc = s.stop()
check("over: both answered", all(r["rc"] == 200 for r in ov), str(rcs(ov)))
check("over: the admission fired", n_over >= 1, "deferrals=%d" % n_over)
check("over: the second request waited for the first", len(ov) == 2 and ov[1]["wall"] > ov[0]["wall"],
      "walls=%.1f/%.1f" % (ov[0]["wall"], ov[1]["wall"]))
check("over: each answer == the lone run's text (a deferred request moves no token)",
      LONE_SHA is not None and shas(ov) == [LONE_SHA, LONE_SHA],
      "%s vs lone %s" % (shas(ov), LONE_SHA))
check("over: clean shutdown", rc == 0, str(rc))
no_abort("over")

s = Server("over0", env={"LLAMA_SERVER_KV_ADMIT": "0"})
ov0 = s.together([PROSE, PROSE], 64, cache=False)
n_over0 = len(s.admits()); s.stop()
check("over0: the kill switch reproduces the 500 (non-vacuity)", rcs(ov0) == [500, 500], str(rcs(ov0)))
check("over0: no deferral on the kill switch", n_over0 == 0, "deferrals=%d" % n_over0)
no_abort("over0")

# ----------------------------------------------------------------------------------------------------------------
print("== full / full0: a reply that OUTGROWS its reserve ends with a length stop instead of HTTP 500 ==")
# WHY this shape (r377, contributor PR #372): admission is a PRE-LAUNCH veto, so it cannot stop a reply that grows
# past its reserve.  -c 16384 / 4 slots with reserve 512 admits both whole prose prompts (2 x 5758 <= 16384) and
# leaves 5892 cells, so 5000-token replies fill the cache mid-reply -- the case the decode retry cannot recover
# (it can only clear IDLE slots, then it halves n_batch and the split aborts a running slot, or raises
# "speculative batch index N is not inside the current sub-batch").
#
# WHAT THESE ARMS DO NOT CHECK, and why: a TEXT oracle.  Two co-batched requests on this model are NOT
# reproducible across server runs -- measured over a stagger sweep (0.05/0.2/0.6/1.5 s) and with a first-token
# barrier, the pair's text changes discretely with arrival timing while each FIXED timing repeats exactly.  So a
# cross-run comparison would be flaky (a gate that passes for the wrong reason), and a lone-run comparison is
# forbidden outright (GREEDY-PURITY.md §38/§63: co-batching changes the ubatch shape).  The purity of the cut is
# asserted IN RUN instead -- the token count the code says it ended, the token count the client received, and (in
# `fullstream`) that every text delta arrived before the terminal chunk -- and the byte-exact-prefix property was
# measured by hand at the same -c and recorded in WORKLOG.md / INVARIANTS.md.
FULL_ENV = {"LLAMA_SERVER_KV_ADMIT_RESERVE": "512"}
PINS = [0, 1]   # id_slot pinned, so the two requests do not land on a third slot if one finishes early
s = Server("full", ctx=CTX * 2, env=FULL_ENV)
fu = s.together([PROSE, PROSE], 5000, cache=False, ignore_eos=True, slots=PINS)
lines = s.lines()
n_full  = len([l for l in lines if "KV cache full" in l])
n_retry = len([l for l in lines if "failed to find free space" in l])
full_cen = [l for l in lines if "KV full stop:" in l]
cut_log = [l for l in lines if "ending the reply with a length stop after" in l]
rc = s.stop()
victim = [r for r in fu if r.get("trunc") is True]
check("full: both answered, neither HTTP 500", all(r["rc"] == 200 for r in fu), str(rcs(fu)))
check("full: exactly one reply ends with a length stop (truncated)",
      len(victim) == 1 and all(r.get("stop") == "limit" for r in fu),
      str([(r.get("gen"), r.get("stop"), r.get("trunc")) for r in fu]))
check("full: the other reply completes all 5000 tokens", sorted(r.get("gen") or 0 for r in fu)[-1] == 5000,
      str([r.get("gen") for r in fu]))
check("full: the full-stop fired exactly once", n_full == 1, "census lines=%d" % n_full)
check("full: the startup line states the arm", len(full_cen) == 1, (full_cen[0][:130] if full_cen else "none"))
check("full: the batch was never split (no decode retry)", n_retry == 0, "%d retry line(s)" % n_retry)
# IN-RUN purity: the count the code decided on is the count the client was given (nothing dropped, nothing invented)
n_cut = None
if cut_log:
    try:
        n_cut = int(cut_log[0].rsplit("after ", 1)[1].split(" generated")[0])
    except Exception:
        n_cut = None
check("full: the reply the code ended is the reply the client received (%s tokens)" % n_cut,
      n_cut is not None and len(victim) == 1 and victim[0].get("gen") == n_cut,
      "log=%s client=%s" % (n_cut, victim[0].get("gen") if victim else None))
check("full: clean shutdown", rc == 0, str(rc))
no_abort("full")

s = Server("full0", ctx=CTX * 2, env=dict(FULL_ENV, LLAMA_SERVER_KV_FULL_STOP="0"))
fu0 = s.together([PROSE, PROSE], 5000, cache=False, ignore_eos=True, slots=PINS)
n0   = len([l for l in s.lines() if "KV cache full" in l])
n0c  = len([l for l in s.lines() if "KV full stop:" in l])
s.stop()
check("full0: the kill switch reproduces the 500 (NON-VACUITY)", rcs(fu0) == [500, 500], str(rcs(fu0)))
check("full0: no full-stop line and no startup census", n0 == 0 and n0c == 0,
      "fired=%d census=%d" % (n0, n0c))
no_abort("full0")

# ----------------------------------------------------------------------------------------------------------------
print("== fullfit: where the replies FIT the feature must not fire at all (behaviour, no text oracle) ==")
s = Server("fullfit", ctx=CTX * 2, env=FULL_ENV)
ff = s.together([PROSE, PROSE], 500, cache=False, ignore_eos=True, slots=PINS)
lff = s.lines()
nf  = len([l for l in lff if "KV cache full" in l])
nfr = len([l for l in lff if "failed to find free space" in l])
nfc = len([l for l in lff if "KV full stop:" in l])
rc  = s.stop()
check("fullfit: both answered with all 500 tokens and nothing truncated",
      all(r["rc"] == 200 and r.get("gen") == 500 and r.get("trunc") is False for r in ff),
      str([(r["rc"], r.get("gen"), r.get("trunc")) for r in ff]))
check("fullfit: the full stop did not fire (no census line beyond the startup one)",
      nf == 0 and len([l for l in lff if "KV cache full" in l]) == 0, "lines=%d" % nf)
check("fullfit: no decode retry (the cache had room)", nfr == 0, "%d retry line(s)" % nfr)
check("fullfit: the startup line is there (the arm is not vacuous)", nfc == 1, "census=%d" % nfc)
check("fullfit: clean shutdown", rc == 0, str(rc))
no_abort("fullfit")

# ----------------------------------------------------------------------------------------------------------------
print("== fullstream: the SSE path ends cleanly, with every delta delivered before the terminal chunk ==")
s = Server("fullstream", ctx=CTX * 2, env=FULL_ENV)
st = s.together_stream([PROSE, PROSE], 5000, cache=False, ignore_eos=True, slots=PINS)
lst = s.lines()
cut_stream = [l for l in lst if "ending the reply with a length stop after" in l]
rc = s.stop()
sv = [r for r in st if (r.get("final") or {}).get("truncated") is True]
check("fullstream: both streams complete (no broken pipe / hang)",
      all(r["rc"] == 200 for r in st), str([r["rc"] for r in st]))
check("fullstream: one stream ends truncated, the other complete",
      len(sv) == 1 and len([r for r in st if (r.get("final") or {}).get("truncated") is False]) == 1,
      str([((r.get("final") or {}).get("truncated"), (r.get("final") or {}).get("stop_type")) for r in st]))
check("fullstream: the truncated stream carried text before its terminal chunk",
      len(sv) == 1 and len(sv[0].get("text", "")) > 0 and sv[0].get("final", {}).get("stop_type") == "limit",
      "chars=%d" % (len(sv[0].get("text", "")) if sv else -1))
n_cut_s = None
if cut_stream:
    try:
        n_cut_s = int(cut_stream[0].rsplit("after ", 1)[1].split(" generated")[0])
    except Exception:
        n_cut_s = None
check("fullstream: the streamed token count is the ended reply's (%s)" % n_cut_s,
      n_cut_s is not None and len(sv) == 1 and
      (sv[0]["final"].get("timings") or {}).get("predicted_n") == n_cut_s,
      "log=%s stream=%s" % (n_cut_s, (sv[0]["final"].get("timings") or {}).get("predicted_n") if sv else None))
check("fullstream: clean shutdown", rc == 0, str(rc))
no_abort("fullstream")

# ----------------------------------------------------------------------------------------------------------------
print("== drain: four overlapping prompts, one at a time ==")
s = Server("drain"); dr = s.together([PROSE] * 4, 64, cache=False, stagger=0.15)
n_dr = len(s.admits()); rc = s.stop()
check("drain: all four answered", all(r["rc"] == 200 for r in dr), str(rcs(dr)))
check("drain: at least three deferrals", n_dr >= 3, "deferrals=%d" % n_dr)
check("drain: serialised", all(dr[i+1]["wall"] > dr[i]["wall"] for i in range(3)),
      str([round(r["wall"], 1) for r in dr]))
check("drain: clean shutdown", rc == 0, str(rc))
no_abort("drain")

# ----------------------------------------------------------------------------------------------------------------
print("== reserve: min(n_ctx / n_parallel, n_ctx_slot()) cells over the prompt ==")
def census(tag, ctx, env=None, n_predict=None):
    """Fire two prose requests at a server and read the cells the refusal charged for the second."""
    s = Server(tag, ctx=ctx, env=env)
    th = [threading.Thread(target=s.ask, args=(PROSE, n_predict, 3.0, False)) for _ in range(2)]
    for t in th:
        t.start(); time.sleep(0.2)
    time.sleep(2.0)
    n = s.need_cells()
    s.stop()
    return n

base  = census("res-8192",  CTX)
big   = census("res-16384", CTX * 2)
small = census("res-256",   CTX, env={"LLAMA_SERVER_KV_ADMIT_RESERVE": "256"})
capped = census("res-np32", CTX, n_predict=32)
share = CTX // NPAR
check("reserve: default == n_ctx / n_parallel (%d) over the prompt" % share,
      base is not None and big is not None and big - base == share, "ctx=%s ctx*2=%s (delta %s)" %
      (base, big, None if (base is None or big is None) else big - base))
check("reserve: LLAMA_SERVER_KV_ADMIT_RESERVE=256 overrides it",
      base is not None and small is not None and base - small == share - 256, "%s vs %s" % (base, small))
check("reserve: a request's n_predict caps it (32)",
      base is not None and capped is not None and base - capped == share - 32, "%s vs %s" % (base, capped))

# ----------------------------------------------------------------------------------------------------------------
print("== nouni: without a unified KV cache the feature is inert ==")
s = Server("nouni", unified=False); nu = s.together([P_A, P_B]); n_nu = len(s.admits()); rc = s.stop()
check("nouni: both answered", all(r["rc"] == 200 for r in nu), str(rcs(nu)))
check("nouni: no deferral", n_nu == 0, "deferrals=%d" % n_nu)
check("nouni: clean shutdown", rc == 0, str(rc))
no_abort("nouni")

# ----------------------------------------------------------------------------------------------------------------
print("== shut: SIGTERM with a deferred queue pending ==")
s = Server("shut")
th = [threading.Thread(target=s.ask, args=(PROSE, 400, 900, False)) for _ in range(4)]
for t in th:
    t.start(); time.sleep(0.2)
time.sleep(1.5)
n_shut = len(s.admits())
t0 = time.time(); rc = s.stop()
for t in th:
    t.join()
check("shut: the queue really held deferred tasks", n_shut >= 1, "deferrals=%d" % n_shut)
check("shut: the process exited on its own", rc == 0 and time.time() - t0 < 20,
      "rc=%s in %.1fs" % (rc, time.time() - t0))

# ----------------------------------------------------------------------------------------------------------------
print("== spin: a waiting task is offered ONCE, and the server idles after the drain ==")
# WHY this shape: the wake this amendment adds re-offers a deferred task when no slot is processing.  A wake that
# re-offers one the admission still refuses would loop (a re-offer logs its refusal), so the detector is the LOG and
# the refusal COUNT over a window in which nothing can legitimately change -- not CPU ticks, which are dominated by
# the box's own busy-poll decode (measured: 190-225 ticks per 2 s in every arm, deferred task or not).
s = Server("spin")
t1 = threading.Thread(target=s.ask, args=(PROSE, 400, 900, False)); t1.start(); time.sleep(0.4)
t2 = threading.Thread(target=s.ask, args=(PROSE, 400, 900, False)); t2.start()
time.sleep(1.0)
ad_a, ln_a = len(s.admits()), len(s.lines())
time.sleep(2.5)
ad_b, ln_b = len(s.admits()), len(s.lines())
t1.join(); t2.join()
i0 = s.ticks(); time.sleep(2.0); i1 = s.ticks()
rc = s.stop()
check("spin: the second request really was deferred", ad_a >= 1, "deferrals=%d" % ad_a)
check("spin: it is offered ONCE while it waits (no wake loop)", ad_b == ad_a,
      "refusals %d -> %d over 2.5 s" % (ad_a, ad_b))
check("spin: the log does not storm while it waits", ln_b - ln_a < 200, "%d -> %d lines" % (ln_a, ln_b))
check("spin: the server idles at ~0 ticks once the queue drains", i1 - i0 < 10, "idle ticks=%d" % (i1 - i0))
check("spin: clean shutdown", rc == 0, str(rc))

# ----------------------------------------------------------------------------------------------------------------
if not NOFAULTS:
    print("== faults: scripts/check-gpu-faults.sh over this run ==")
    faults = os.path.join(REPO, "scripts", "check-gpu-faults.sh")
    if os.path.exists(faults):
        cmd = ["bash", faults, "--since", "30 min ago"]
        for lg in LOGS:
            cmd += ["--log", lg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
        check("faults: no real GPU fault in the run's logs", r.returncode == 0, " | ".join(tail))
    else:
        info("check-gpu-faults.sh not found - skipped")

# ----------------------------------------------------------------------------------------------------------------
cache_state = [l for s in ("over",) for l in open(os.path.join(DIR, "arm-%s.log" % s), errors="replace")
               if "cached n_tokens" in l]
info("deferred request prompt-cache restores: %d line(s) in the over arm" % len(cache_state))
if not KEEP:
    info("logs: %s (kept: pass --keep to keep every arm's)" % DIR)

print()
if SELFTEST:
    # negative control: the arms MUST fail when the feature is off
    if FAILS:
        print("SELF-TEST PASS: the gate rejected the pre-amendment server (%d failing checks)" % len(FAILS))
        sys.exit(0)
    print("SELF-TEST FAIL: the gate passed with LLAMA_SERVER_KV_ADMIT=0 -- it cannot see the defect")
    sys.exit(1)
if FAILS:
    print("FAILED (%d): %s" % (len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("PASS")
PYEOF
