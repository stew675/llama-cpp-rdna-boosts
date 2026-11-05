import json, sys, time, urllib.request

def post(url, payload, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0

def health(port, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("server never became healthy")

if __name__ == "__main__":
    port = int(sys.argv[1]); mode = sys.argv[2]
    base = "http://127.0.0.1:%d/completion" % port
    if mode == "health":
        print("healthy after %.1f s" % health(port)); raise SystemExit
    prompt = open(sys.argv[3]).read()
    n_predict = int(sys.argv[4])
    body, wall = post(base, {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                             "cache_prompt": False, "stream": False})
    t = body.get("timings", {})
    print(json.dumps({
        "wall_s": round(wall, 2),
        "prompt_n": t.get("prompt_n"), "prompt_ms": round(t.get("prompt_ms", 0), 1),
        "prompt_tps": round(t.get("prompt_per_second", 0), 1),
        "pred_n": t.get("predicted_n"), "pred_ms": round(t.get("predicted_ms", 0), 1),
        "pred_tps": round(t.get("predicted_per_second", 0), 2),
    }))
