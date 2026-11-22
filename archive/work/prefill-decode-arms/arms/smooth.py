#!/usr/bin/env python3
"""Smoothness harness for the prefill-transfer-overlap campaign (WIP; NOT part of the delivery).

COPIED VERBATIM from archive/work/prefill-transfer-overlap/arms/smooth.py (2026-11-22, archive/work/prefill-decode-arms)
so the two-arm server can be measured with the campaign's metric of record and against its baseline table.  The
ONLY addition is `--warm N`: N throwaway long-prompt requests before the measured pair, because every arm here
starts a FRESH server and the first pass fills the expert arena (~3x slower, a cold-fill number, never quoted).
`--warm 0` is the archived tool's behaviour exactly.

The campaign's metric of record is NOT throughput, it is **smoothness**: a decoder and a prefill sharing a
server should both make steady progress.  So this measures BOTH sides on one run:

  * decoder: inter-token latency p50 / p95 / max, and the worst single gap (the stall a user feels)
  * prefill: tokens/second, AND its own step-time distribution from the `prompt_progress` events
             (a prefill that alternates 500 ms stalls with bursts is not "smooth" even at a good mean)

Arms are server configurations; the prompt pair is identical across arms and slots are PINNED, because an
unpinned two-slot arm measures a scheduling race (see scripts/gate-prefill-duty.py).

Usage:
  ./smooth.py --server ~/llama.cpp/build-rocm/bin/llama-server --model M.gguf \
              --server-args "-ncmoe 40 -lme auto -c 16384 -b 2048 -ub 2048" \
              --arm lone --arm "duty0:LLAMA_SERVER_PREFILL_DUTY=0" --arm "duty16:LLAMA_SERVER_PREFILL_DUTY=16"
  (an arm is NAME or NAME:ENV=VAL[,ENV=VAL...]; the first arm named `lone` runs the prompt with no decoder)
"""
import argparse, json, os, statistics, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))


def sentence(reps, tag):
    return f"[{tag}] The history of computing is a long and tangled story. " * reps


class Server:
    def __init__(self, a, env_extra):
        self.a = a
        port = a.port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/opt/rocm-10.1.0-gfx120X/lib"
        env["HIP_VISIBLE_DEVICES"] = str(a.gpu)
        env.update(env_extra)
        cmd = [a.server, "-m", a.model, "-np", str(a.np), "--port", str(port), "-t", str(a.threads),
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
    import requests
    time.sleep(delay)
    t0 = time.time()
    arrivals, steps = [], []          # steps: (t, processed, total) from prompt_progress
    timings = None
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
            pp = d.get("prompt_progress")
            if pp is not None:
                steps.append((now, pp.get("processed"), pp.get("total")))
                continue
            if d.get("content") or d.get("reasoning_content") or d.get("tokens"):
                arrivals.append(now)
            if d.get("stop"):
                timings = d.get("timings")
                break
    out[key] = {"t0": t0, "arrivals": arrivals, "steps": steps, "timings": timings,
                "ttft": arrivals[0] - t0 if arrivals else None}


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


def report(name, r, prefill_window):
    """prefill_window = (t_start, t_end) of the prompt's own processing, for 'during' counts."""
    itl = [b - a for a, b in zip(r["arrivals"], r["arrivals"][1:])]
    # the gaps that land INSIDE the prefill window: the stall a user feels while a prompt is being processed
    idx = [i for i, a in enumerate(r["arrivals"])
           if prefill_window is not None and prefill_window[0] <= a <= prefill_window[1]]
    during = len(idx)
    itl_during = [itl[i] for i in idx if i < len(itl)]
    steps = r["steps"]
    step_dt = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    t = r["timings"] or {}
    pf_ms = t.get("prompt_ms")
    pf_tps = 1000.0 * t["prompt_n"] / pf_ms if pf_ms and t.get("prompt_n") else float("nan")
    return {
        "arm": name,
        "pf_tps": pf_tps,
        "pf_wall": (pf_ms or 0) / 1000.0,
        "pf_steps": len(steps),
        "pf_step_p50": pct(step_dt, 50) * 1000 if step_dt else float("nan"),
        "pf_step_max": max(step_dt) * 1000 if step_dt else float("nan"),
        "dec_n": len(r["arrivals"]),
        "dec_during": during,
        "itl_p50": pct(itl, 50) * 1000 if itl else float("nan"),
        "itl_p95": pct(itl, 95) * 1000 if itl else float("nan"),
        "itl_max": max(itl) * 1000 if itl else float("nan"),
        "itl_d_p50": pct(itl_during, 50) * 1000 if itl_during else float("nan"),
        "itl_d_p95": pct(itl_during, 95) * 1000 if itl_during else float("nan"),
        "itl_d_max": max(itl_during) * 1000 if itl_during else float("nan"),
        "ttft": (r["ttft"] or 0) * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.expanduser("~/llama.cpp/build-rocm/bin/llama-server"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--server-args", default="-c 16384 -b 2048 -ub 2048")
    ap.add_argument("--gpu", default=0)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=400, help="long-prompt repetitions (~15 tokens each)")
    ap.add_argument("--dec-n", type=int, default=400, help="decode tokens requested")
    ap.add_argument("--arm", action="append", default=[], help="NAME[:ENV=VAL[,ENV=VAL]]")
    ap.add_argument("--warm", type=int, default=0,
                    help="throwaway long-prompt requests before the measured pair (0 = the archived behaviour; the "
                         "first pass fills the expert arena and is ~3x slower, so a warm number needs >= 1)")
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
        arms = [("lone", {}), ("duty0", {"LLAMA_SERVER_PREFILL_DUTY": "0"}),
                ("duty16", {"LLAMA_SERVER_PREFILL_DUTY": "16"})]

    rows = []
    for name, env in arms:
        a.arm = name
        srv = Server(a, env)
        try:
            lone = name.startswith("lone")

            # --warm: the same prompt shape, thrown away, so the measured pair is not a cold-arena fill
            for _w in range(a.warm):
                wout = {}
                wt = threading.Thread(target=stream,
                                      args=(srv.base, sentence(a.reps, "WARM"), 2, wout, "pf", 0.0, 1))
                wt.start()
                wt.join()
                print(f"[{name}] warm {_w + 1}/{a.warm}: "
                      f"prompt {(wout['pf']['timings'] or {}).get('prompt_ms', 0) / 1000.0:.2f} s", flush=True)

            out = {}
            if lone:
                th = [threading.Thread(target=stream,
                                       args=(srv.base, sentence(a.reps, "LONE"), 2, out, "pf", 0.0, 0))]
            else:
                th = [threading.Thread(target=stream,
                                       args=(srv.base, "[DEC] Count slowly from one to five hundred, one number per line.",
                                             a.dec_n, out, "dec", 0.0, 0)),
                      threading.Thread(target=stream,
                                       args=(srv.base, sentence(a.reps, "MIX"), 2, out, "pf", 0.4, 1))]
            for t in th:
                t.start()
            for t in th:
                t.join()
            pf = out["pf"]
            win = (pf["t0"], pf["t0"] + (pf["timings"] or {}).get("prompt_ms", 0) / 1000.0)
            if not lone:
                rows.append(report(name + "/dec", out["dec"], win))
            rows.append(report(name + "/pf", pf, None))
            open(os.path.join(a.outdir, f"{name}.json"), "w").write(json.dumps(out, indent=1))
        finally:
            srv.stop()

    hdr = ("arm", "pf_tps", "pf_wall", "pf_steps", "pf_step_p50", "pf_step_max",
           "dec_n", "dec_during", "itl_p50", "itl_p95", "itl_max", "itl_d_p50", "itl_d_p95", "itl_d_max", "ttft")
    print("\n" + "\t".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print("\t".join(f"{r[h]:>12.1f}" if isinstance(r[h], float) else f"{r[h]:>12}" for h in hdr))
    open(os.path.join(a.outdir, "smooth.tsv"), "w").write(
        "\t".join(hdr) + "\n" + "".join("\t".join(str(r[h]) for h in hdr) + "\n" for r in rows))


if __name__ == "__main__":
    main()
