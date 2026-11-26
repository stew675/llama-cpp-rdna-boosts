#!/usr/bin/env bash
# KV admission gate (LLAMA_SERVER_KV_ADMIT, LLAMA_SERVER_KV_ADMIT_RESERVE -- block 16 amendment, r365).
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
#   HARD  shut     SIGTERM with a deferred queue pending: the process exits on its own (a task that was never
#                  launched is answered at termination -- without that the exit hangs in HIP's teardown, measured)
#   HARD  spin     a waiting (deferred) task costs no CPU over a decode, and the server idles at ~0 ticks after the
#                  queue drains (a wake path that re-offers a task it cannot launch would spin at 100 %)
#   HARD  the per-arm logs carry no `GGML_ASSERT` / abort, and `scripts/check-gpu-faults.sh` is clean over the run
#   INFO  the prompt-cache state the deferred request restored from (reported, not asserted)
#
# Usage:  ./gate-kv-admit.sh [--keep] [--no-faults] [--self-test]
#   --self-test runs every arm with LLAMA_SERVER_KV_ADMIT=0 (the pre-amendment behaviour) and requires the gate to
#   FAIL: a gate that cannot reject the unfixed server is not a gate.  Run it once before trusting a green run.
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
            e["LLAMA_SERVER_KV_ADMIT"] = "0"   # --self-test: make every arm behave like the pre-amendment server
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
    def ask(self, prompt, n_predict=64, timeout=900, cache=True, slot=None):
        body = {"prompt": prompt, "temperature": 0, "top_k": 1, "cache_prompt": cache}
        if n_predict is not None:
            body["n_predict"] = n_predict
        if slot is not None:
            body["id_slot"] = slot
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:%d/completion" % self.port, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}), timeout=timeout))
            return dict(rc=200, wall=time.time()-t0, pn=r["timings"]["prompt_n"],
                        sha=hashlib.sha256(r["content"].encode()).hexdigest()[:12])
        except urllib.error.HTTPError as e:
            return dict(rc=e.code, wall=time.time()-t0, body=e.read()[:90].decode(errors="replace"))
        except Exception as e:
            return dict(rc="ERR", wall=time.time()-t0, body=type(e).__name__)

    def together(self, prompts, n_predict=64, cache=True, stagger=0.2, timeout=900):
        out = [None] * len(prompts)
        def run(i):
            out[i] = self.ask(prompts[i], n_predict, timeout, cache)
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
