# BETA-TESTING — `beta/prefill-decode-arms` (the two-arm server)

Run the ladder **in order**; each arm is one GPU job, never two at once.  Expect ~20 minutes for A1-A6 on a
2-GPU box, plus the build.  Report back the block at the end of §5 — the numbers are what decides whether this is
folded into the delivery as block 17.

Every command below assumes:

```bash
DELIVERY=~/llama-cpp-rdna-boosts          # this repo
LLAMA=~/llama.cpp                          # the checkout you applied the patch to
BIN=$LLAMA/build/bin                       # your build dir
PROMPT=$DELIVERY/prompts/prose-rdna-boosts.txt   # 5246 tokens, sha256 below
MODEL=/path/to/your/model.gguf             # a model you can run with -ngl 99
NP=2                                       # >= 2 slots: the pair needs two
HASH() { sha256sum | cut -d' ' -f1 | cut -c1-12; }   # the short hashes below are sha256[:12]
```

`prompts/prose-rdna-boosts.txt` is a **shipped, hash-stable** prompt — do not edit it; a different prompt gives a
different (still comparable) hash, just say which one you used.

## 1. Apply (5 min)

```bash
git clone https://github.com/ggml-org/llama.cpp "$LLAMA" && cd "$LLAMA"
git checkout 919794500
bash "$DELIVERY/scripts/apply-all.sh" --include-beta .
```

It must print `Applied tip: 13df5e2c7 (...)`, then apply `beta/prefill-decode-arms/prefill-decode-arms.patch`, then:

```
      release.json tree: 27db9bfd4ef7f41c7df60f704410de28863beac3
      applied tree:      dc9c657e6e9d0cd3af485153e69a7cbfaf5ef941   (THREE beta patches: block 17 + STEP 15 + STEP 16)
```

**If the applied tree is different, stop and report it** — do not proceed (the patch was cut against exactly that
release tree, and a different tree means a different starting point).

## 2. Build (the `all` profile, not just `llama-server`)

Build every target, not only the server: a stale `libllama-<tool>-impl.so` beside a fresh `libllama` **segfaults**
(a real trap this campaign hit — `llama-perplexity` and `llama-gen-docs` were 8 hours older than the library and
died on start).

```bash
cd "$LLAMA"
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="$(your arch, e.g. gfx1201)" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"            # the WHOLE build, not a single target
$BIN/llama-server --version                 # must print the commit you just built
```

Report: **your ROCm line** (`$BIN/llama-server --version` does not print it — report the ROCm you built against,
e.g. `cat /opt/rocm/.info/version`), your GPU model(s), the arch, and whether the build was clean.

## 3. The arms

### A1/A2 — the core claim: the pair does not move the TEXT (the gate that matters)

Pair **OFF** first, one request in flight, fresh process:

```bash
cd "$LLAMA"
export HIP_VISIBLE_DEVICES=0        # pin it; never let the iGPU be enumerated
# the delivery DEFAULT pin: without it a fresh-process text claim is a coin flip
ROCBLAS_USE_HIPBLASLT=0 LLAMA_SERVER_PARK_IDLE_SLOTS=0 \
setsid "$BIN/llama-server" -m "$MODEL" -ngl 99 -np $NP -c 16384 \
        --port 18080 --seed 42 --temp 0 -n 128 \
        -p "$(cat "$PROMPT")" > /tmp/pairoff.log 2>&1 &
SRV=$!
for i in $(seq 120); do curl -sf localhost:18080/health >/dev/null && break; sleep 1; done
curl -s localhost:18080/completion -H 'content-type: application/json' \
     -d "{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()))' "$PROMPT"),\"n_predict\":64,\"temperature\":0,\"seed\":42}" \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["content"],end="")' > /tmp/ans-off.txt
HASH < /tmp/ans-off.txt                     # <- record this
kill -TERM -$SRV; wait $SRV 2>/dev/null     # ONE TERM, then wait (see §4)
```

Then the **same** command with the pair ON: add `--prefill-decode-arms-mode on` (the explicit form — it refuses to
start rather than silently run one arm, which is what you want while testing).

**PASS = the same hash as A1, and `grep -c "the decode arm is generating" /tmp/pairon.log` ≥ 1.**

The census line is the **non-vacuity** check — it proves the decode arm really ran the request:

```
the decode arm is generating: N step(s), M handover(s), widest batch W slot(s)
```

It needs the server at `-lv 4`; the pair-ON arm often reads differently from pair-OFF in ways that are **not**
defects (step counts, handovers), so hash and census are two separate runs if you want both from one binary.

### A3 — a drafter ON the pair (only if you have an MTP/`draft-mtp` capable model)

```bash
ROCBLAS_USE_HIPBLASLT=0 LLAMA_SERVER_PARK_IDLE_SLOTS=0 \
setsid "$BIN/llama-server" -m "$MODEL" -ngl 99 -np $NP -c 16384 --port 18081 \
        --prefill-decode-arms-mode on \
        --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
        ... > /tmp/pairdraft.log 2>&1 &
```

**PASS = the same answer hash as A1/A2** (one cold greedy text across all arms is the campaign's own claim), and
the log shows `draft acceptance = <n>/<m>` with **n > 0** (a drafter that drafted nothing is a FAIL, not a pass).

With an **external** head add `-md /path/to/mtp-model.gguf`.  If you get
`cannot run the two-arm server: ...` it names the refusal — that is by design, report which one you hit.

### A4 — the trade it is for: a decoder that keeps decoding while another slot prefills

Two requests, overlapping: slot A decodes a long answer while slot B prefills a long prompt.

```bash
# B first, into the background, then A immediately after; run each arm 3 times and take the median
time curl -s localhost:18080/completion -d @bigprompt.json ...
```

The number to report: **ms/token for the decoder while the other slot prefills**, pair OFF vs ON, with the `-ub`
you used (`-ub 2048` is where the win lives on a 1-GPU box; `-ub 4096` on a tensor split), plus the prefill t/s.
The campaign measured **−22.8 %** ITL and **+24.1 %** prefill at `-ub 2048` (4B, 1 GPU) and **−56.9 %** ITL at
`-ub 4096` (Flash-Next, 3 GPU tensor).  **A single run is not a measurement here** — repeats, median.

### A5 — a host-expert tier (only if you run `-ncmoe` with an arena)

```bash
MOE_EXPERT_HEATMAP=load ... -ncmoe 40 --prefill-decode-arms-mode on
```

Two things to report, both from the log of the first run:

1. any `arena alloc failed` line, and any
   `... had to TAKE arena tables ... Set GGML_CUDA_SLAB_RESERVE_MIB=<n>` line — **set that number and re-run**;
2. the calibration line (once per slot): `the aux narrow-N region (…) came from the unmapped VA spare …
   X MiB would host every auxiliary region (headroom … + aux …)`.

**PASS = 0 `arena alloc failed`, the aux region pinned (not `cudaMalloc`), and the text matching A1.**  A 4-slot
soak (20 rounds, 4 concurrent requests) is the strongest form if you have the time: **0 errors and a decode batch
wider than 1** ("widest batch W slot(s)" with W ≥ 2 proves two decoders really shared a step).

### A6 — faults, and the negative control

```bash
bash "$DELIVERY/scripts/check-gpu-faults.sh" --since "<when you started>" --log /tmp/pairon.log
```

Then flip the mode to `auto` on a shape the pair refuses (e.g. `-np 1`) and confirm the server **starts** on one
arm and warns — `auto` must never refuse, and `on` must never silently continue.

## 4. Hygiene (these have each burned someone)

* **Pin `HIP_VISIBLE_DEVICES`.**  A box whose iGPU is enumerated aborts at load with `invalid kernel file` and it
  looks like this patch's fault.
* **`setsid` + ONE `TERM` + `wait`**, then kill the whole process group.  `timeout` forwarding a second signal
  makes the server skip its destructors, and the teardown censuses vanish silently (never `kill -9` a server whose
  teardown numbers you want).
* **`ROCBLAS_USE_HIPBLASLT=0` on every fresh-process TEXT comparison** — the library's per-start autotuning makes
  a one-run text difference a lottery (`GREEDY-PURITY.md` §44).
* **`LLAMA_SERVER_PARK_IDLE_SLOTS=0` pinning** for a byte text claim on a multi-slot server: parking moves the KV
  and the placement of the next sequence, and a near-tie then flips (`GREEDY-PURITY.md` §107).
* **`MOE_EXPERT_HEATMAP=load`** on every host-expert arm — otherwise the persisted residency prior drifts between
  runs and the same binary changes its text/kernels (`GREEDY-PURITY.md` §89).
* **`-lv` is per arm:** none for a text/hash arm, **`-lv 4`** for a census arm, **never `-lv 3`** (it pollutes the
  text and still hides the censuses).  On the server the text arrives over HTTP, so `-lv 4` is safe there.
* **Never two GPU jobs at once.**  If you benchmark, size `-t` and leave the IRQ cores free.
* **`rm -f gpucore.*.gpu` after any fault** (they are ~32 GB each) — and on a box where `/tmp` is tmpfs, watch it.
* Nothing here goes to `/tmp` if it is tmpfs and larger than a log.

## 5. What to report

1. **The environment:** ROCm line + version, GPU model(s) and count, arch, the checkout you built
   (`git -C "$LLAMA" rev-parse --short HEAD`), the **applied tree** `git -C "$LLAMA" rev-parse HEAD^{tree}`, and
   whether the build was clean.
2. **A1/A2/A3:** the answer hashes (or the text itself if you have no hash to compare against), and for the pair
   arms the census line `the decode arm is generating: …` (this is the proof it was not vacuous), plus the draft
   acceptance counts for A3.
3. **A4:** the ITL pair and the prefill pair, with `-ub`, the shape, and **the number of repeats** — one run is not
   a result.
4. **A5 (if run):** `arena alloc failed` count, the calibration line, the reserve you set, the aux region size.
5. **Anything that refused or aborted**, verbatim — the refusal text names the predicate's reason, and that is the
   most useful bug report this candidate can get.
6. **Any GPU fault**, with the `check-gpu-faults.sh` output (not a hand-grep of the log).

Then decide with the maintainer: this patch is a **promotion candidate only**.  Confirming it does not ship it —
promotion is **block 17**, a block-structure decision made by the maintainer, with a validated window and a full
gate ladder first.
