#!/usr/bin/env python3
"""interactive.py -- what a USER sees with two arms, on a real llama-server (WIP; NOT part of the delivery).

The campaign's metric of record (`smooth.py`) reports the decoder's ITL percentiles inside a prefill's window.
This tool answers the neighbouring question, the one a person actually feels:

    "my reply is streaming, and then I paste a long document into the other chat -- what happens?"

Scenario, per arm:

    t0        slot 0: a chat reply starts streaming (one token per line, so every token is a visible arrival)
    t0+delay  slot 1: a LONG prompt arrives (a ~2-4k-token document) -- the "paste a document" case
              both streams are read concurrently; every arrival is timestamped

Reported:

  * the long prompt's **TTFT** and its prefill rate (how long that user waited);
  * the chat stream's token count and its **gaps inside the prefill window** (p50/p95/max), which is the stall;
  * a **timeline**: one character per bucket, so a freeze and a burst are visible at a glance;
  * the chat stream's worst single gap, and the number of tokens it got through the window.

Arms are server configurations exactly like `smooth.py`'s: NAME or NAME:ENV=VAL[,ENV=VAL...].

Usage:
  ./interactive.py --server ~/llama.cpp/build-2ctx/bin/llama-server --model M.gguf \
      --server-args "-ngl 99 -ncmoe 40 -lme auto -c 16384 -b 2048 -ub 512" \
      --arm default --arm arms:LLAMA_ARG_PREFILL_DECODE_ARMS=1
"""
import argparse, json, os, statistics, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))


def document(reps, tag):
    return f"[{tag}] The history of computing is a long and tangled story. " * reps + \
           "\n\nSummarise the text above in one sentence."


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


class Server:
    def __init__(self, a, env_extra):
        self.a = a
        self.base = f"http://127.0.0.1:{a.port}"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-10.1.0-gfx120X/lib"
        env["HIP_VISIBLE_DEVICES"] = str(a.gpu)
        env.update(env_extra)
        cmd = [a.server, "-m", a.model, "-np", str(a.np), "--port", str(a.port), "-t", str(a.threads),
               "--no-warmup", "-lv", "4"] + a.server_args.split()
        self.log_path = os.path.join(a.outdir, f"{a.arm}.server.log")
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        import requests
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; see {self.log_path}")
            try:
                if requests.get(self.base + "/health", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("server not ready")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
        self.log.close()


def stream(base, prompt, n_predict, out, key, delay=0.0, id_slot=None):
    """one streaming request; records (arrival time, kind) -- 'tok' for a content update, 'meta' for progress"""
    import requests
    time.sleep(delay)
    t0 = time.time()
    arrivals, progress, timings = [], [], None
    body = {"prompt": prompt, "n_predict": n_predict, "stream": True, "temperature": 0, "seed": 42,
            "cache_prompt": False, "return_progress": True}
    if id_slot is not None:
        body["id_slot"] = id_slot
    with requests.post(base + "/completion", json=body, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            d = json.loads(line[6:])
            now = time.time()
            if d.get("prompt_progress") is not None:
                progress.append((now, d["prompt_progress"].get("processed"), d["prompt_progress"].get("total")))
                continue
            if d.get("content") or d.get("reasoning_content") or d.get("tokens"):
                arrivals.append(now)
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = {"t0": t0, "arrivals": arrivals, "progress": progress, "timings": timings,
                "ttft": arrivals[0] - t0 if arrivals else None}


def per_second(r, window):
    """tokens delivered in each whole second from the reply's start, and a mask of the prefill window.
    This is the shape a person perceives: a freeze shows up as a low or zero column, a burst as a high one."""
    t_end = max(r["arrivals"][-1] if r["arrivals"] else r["t0"], (window[1] if window else 0))
    n = int(t_end - r["t0"]) + 1
    cols = [0] * n
    for a in r["arrivals"]:
        k = int(a - r["t0"])
        if 0 <= k < n:
            cols[k] += 1
    mask = ["|" if window is not None and window[0] <= r["t0"] + k <= window[1] else "."
            for k in range(n)]
    return cols, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.expanduser("~/llama.cpp/build-2ctx/bin/llama-server"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--server-args", default="-c 16384 -b 2048 -ub 512")
    ap.add_argument("--gpu", default=0)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=250, help="document repetitions (~14.5 tokens each)")
    ap.add_argument("--chat-n", type=int, default=600, help="tokens requested on the streaming reply")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between the reply starting and the document")
    ap.add_argument("--warm", type=int, default=1, help="throwaway long-prompt requests before the measured pair")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--outdir", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    arms = []
    for spec in a.arm:
        if ":" in spec:
            name, envs = spec.split(":", 1)
            ev = dict(kv.split("=", 1) for kv in envs.split(",") if kv)
        else:
            name, ev = spec, {}
        arms.append((name, ev))
    if not arms:
        arms = [("default", {}), ("arms", {"LLAMA_ARG_PREFILL_DECODE_ARMS": "1"})]

    rows = []
    for name, env in arms:
        a.arm = name
        srv = Server(a, env)
        try:
            for _ in range(a.warm):
                w = {}
                t = threading.Thread(target=stream, args=(srv.base, document(a.reps, "WARM"), 2, w, "pf", 0.0, 1))
                t.start(); t.join()
                print(f"[{name}] warm: document prefilled in "
                      f"{(w['pf']['timings'] or {}).get('prompt_ms', 0) / 1000.0:.2f} s", flush=True)

            out = {}
            t_chat = threading.Thread(target=stream,
                                      args=(srv.base, "Count slowly from one to five hundred, one number per line.",
                                            a.chat_n, out, "chat", 0.0, 0))
            t_doc = threading.Thread(target=stream,
                                     args=(srv.base, document(a.reps, "DOC"), 24, out, "doc", a.delay, 1))
            t_chat.start(); t_doc.start()
            t_chat.join(); t_doc.join()

            chat, doc = out["chat"], out["doc"]
            doc_t = doc["timings"] or {}
            pf_s = (doc_t.get("prompt_ms") or 0) / 1000.0
            win = (doc["t0"], doc["t0"] + pf_s)

            itl = [b - x for x, b in zip(chat["arrivals"], chat["arrivals"][1:])]
            idx = [i for i, arrival in enumerate(chat["arrivals"]) if win[0] <= arrival <= win[1]]
            during = [itl[i] for i in idx if i < len(itl)]
            n_during = len(idx)
            worst = max(during) * 1000 if during else float("nan")
            row = {
                "arm": name,
                "doc_ttft": doc["ttft"] * 1000 if doc["ttft"] else float("nan"),
                "doc_pf_s": pf_s,
                "doc_tps": 1000.0 * doc_t["prompt_n"] / doc_t["prompt_ms"] if doc_t.get("prompt_ms") else float("nan"),
                "chat_n": len(chat["arrivals"]),
                "chat_during": n_during,
                "itl_p50": pct(itl, 50) * 1000 if itl else float("nan"),
                "itl_p95": pct(itl, 95) * 1000 if itl else float("nan"),
                "gap_p50": pct(during, 50) * 1000 if during else float("nan"),
                "gap_p95": pct(during, 95) * 1000 if during else float("nan"),
                "gap_max": worst,
            }
            rows.append(row)
            open(os.path.join(a.outdir, f"{name}.json"), "w").write(json.dumps(out, indent=1))

            cols, mask = per_second(chat, win)
            print(f"\n[{name}] the document: TTFT {row['doc_ttft']:.0f} ms, prefill {pf_s:.2f} s "
                  f"at {row['doc_tps']:.0f} t/s")
            print(f"[{name}] the streaming reply, while the document was being prefilled:")
            print(f"        {row['chat_n']} tokens total, {n_during} of them inside the {pf_s:.1f} s window; "
                  f"gaps in that window p50 {row['gap_p50']:.0f} ms / p95 {row['gap_p95']:.0f} ms / "
                  f"worst {row['gap_max']:.0f} ms")
            print("        tokens/s from the reply's start (one column per second):")
            print("            " + "".join(f"{k:>4}" for k in range(len(cols))))
            print("            " + "".join(f"{v:>4}" for v in cols))
            print("            " + "".join(f"{m:>4}" for m in mask) + "   '|' = the document is prefilling")
            print(flush=True)
        finally:
            srv.stop()

    hdr = ("arm", "doc_ttft", "doc_pf_s", "doc_tps", "chat_n", "chat_during",
           "itl_p50", "itl_p95", "gap_p50", "gap_p95", "gap_max")
    print("\n" + "\t".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print("\t".join(f"{r[h]:>10.1f}" if isinstance(r[h], float) else f"{r[h]:>10}" for h in hdr))
    open(os.path.join(a.outdir, "interactive.tsv"), "w").write(
        "\t".join(hdr) + "\n" + "".join("\t".join(str(r[h]) for h in hdr) + "\n" for r in rows))


if __name__ == "__main__":
    main()
