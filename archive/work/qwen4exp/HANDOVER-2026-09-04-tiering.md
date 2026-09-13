# Session handover: expert-tiering feasibility - where we are, what to measure next

Date: 2026-09-04.  This session burned ~800K tokens exploring whether a
dense-on-GPU / tiered-expert design is viable.  This doc exists so a fresh
session can pick up WITHOUT the full context.  Read LRU_EXPERTS.md and
PHASE0_ROUTING.md for the deeper record; this is the distilled "what is
decided, what is NOT, what to measure, and the traps".

## 1. The goal (unchanged)

Run Qwen3.8-Flash-Next in a configuration that does NOT fit all weights in
VRAM (ultimately Q8_0 = ~142 GiB vs 97.8 GiB on 3x R9700; today's test
vehicle = Q4_K_XL = 103.69 GiB).  Idea: keep the DENSE work on the GPU,
tier the ROUTED EXPERTS (a GPU-resident hot set + the rest host-resident),
and move experts between tiers based on usage frequency.

## 2. Model geometry (measured, exact)

Q4_K_XL file (103.69 GiB total, 48 layers, 512 experts/layer, top-10):

| component | size | note |
|---|---|---|
| routed experts (ffn_gate_exps + ffn_up_exps + ffn_down_exps) | 71.73 GiB | 3.13 MB per (layer, expert) |
| per_layer_token_embd (PLE n-gram) | 26.82 GiB | managed INDEPENDENTLY (user); not part of the expert tier question |
| true dense (attn qkv/q/k/v/out/gate, ssm/GDN, hc_attn/ffn, shared experts, router gate_inp, indexer, norms, embd, output head) | 5.13 GiB | the "heavy every-token" work |
| metadata shard | 0.01 GiB | |

So the dense side is TINY (5.13 GiB).  The expert side dominates (71.7 GiB).
Per-expert Q4_K_XL = 3.13 MB (gate_exps Q4_K ~0.92 MB + up_exps Q4_K ~0.92 MB
+ down_exps Q5_1 ~1.23 MB).

2x R9700 = 63.7 GiB (68.4 GB).  The proposed experiment layout:
- dense ~5.5 GB + PLE headroom on GPU (user: "~9 GB dense")
- ~40 GB of experts on GPU  -> S = 266 of 512 per layer (52% resident)
- ~37 GB of experts on CPU  -> 246 of 512 per layer host-resident
- ~14 GB KV
- total ~63 GB of the 68.4 GB available

## 3. What is DECIDED (measured, trustworthy)

1. Routing is NOT heavy-tailed.  2000 decode tokens touch 355-487 of 512
   experts per layer; windows drift with no settling core.  The
   GPT-OSS-120B "15-20% of experts handle 80%" evidence does NOT transfer.
   (LLAMA_ROUTE_DUMP instrumentation, validated deterministic; router
   ffn_gate_inp is F32 so not a quant artifact.)
2. Per-WINDOW working set is small: a 100-token window uses 100-190 distinct
   experts per layer, 60-75% overlap with the previous window.  This is what
   makes a GPU expert CACHE viable at S=200-266: 85-95% of selections are
   covered from GPU memory.
3. Full-GPU decode = ~20 ms/token (all resident).  Full-CPU decode =
   397 ms/token (everything on CPU, M=1) and it is KV-INDEPENDENT (64 vs 512
   KV identical) => the CPU cost is the M=1 dense+expert work/dequant, NOT
   attention-over-cache.
4. CPU mmid at M=1 is NOT fast: the full-CPU anchor implies ~18 GB/s
   effective for the ~7 GB/token stream, ~5x worse than the 90 GB/s DRAM
   peak.  (See the L3 trap below - a microbenchmark wrongly suggested
   1.2 TFLOPS / 5 ms/token; that was cache-warm and is RETRACTED.)
5. The "dense on GPU / ALL experts on CPU, no movement" split is NOT viable
   at 20 ms/token.  The M=1 CPU expert cost is hundreds of ms, not 5 ms.
6. PCIe H2D bandwidth on this machine ~= 14 GB/s aggregate (user-provided).
   At that rate, moving large expert volumes per token is impossible; only a
   small per-token migration (K swaps/layer per 10-20 tokens, spread async)
   is affordable.

## 4. The proposed experiment (what the fresh session should set up)

User's proposed configuration to TEST viability of the shuffling policy:
  2 GPUs.  dense ~9 GB + ~40 GB experts (S=266/layer) + ~14 GB KV on GPU;
  ~32-37 GB experts host-resident on CPU.  Then MEASURE whether the
  LFRU-style shuffling policy keeps decode at GPU-bound speed (~20-24 ms/tok)
  or whether the CPU cold tail + bus traffic push it over.

This is the right experiment because S=266 (52%) is the natural midpoint:
- high enough that the GPU hot cache covers the per-window working set
  (the sims say ~85-95% of selections hit at S=200-266)
- low enough that the CPU tail (5-15% cold) is a real test of the CPU path

## 5. What MUST be measured before the final decision (in priority order)

### M1. The TRUE M=1 CPU expert cost (DRAM-cold).  THE critical unknown.
Why it matters: it decides whether a 5-15% cold tail (24-72 pairs/token on
CPU) is 1 ms or 50 ms.  Everything else hinges on this.
How to measure it RIGHT:
  - The trap: any benchmark that loops the same small tensor is L3-warm.
    9950X3D has 96 MB V-cache; the test tensors were 18-28 MB.  MUST defeat
    cache: either (a) cycle the mmid ids across many distinct experts so each
    iteration reads fresh DRAM, or (b) interleave the mmid with a large
    memory scrub, or (c) measure inside the real decode with only experts on
    CPU (see M3).
  - Acceptable proxies: a standalone probe that reads the FULL 512-expert
    tensor per iteration (tensor > L3), timing only the 10 used slices, cold.
  - Bounds to sanity-check against: 1.5 GB/token expert stream at 90 GB/s
    ideal = 16 ms; the full-CPU 397 ms anchor says the real number is much
    higher than 16 ms (M=1 dequant + strided K-major reads are inefficient).
    The truth is between; we need the actual number, not a bound.

### M2. Can CPU cold-expert work OVERLAP the GPU decode?
The graph is per-layer serialized: layer i+1 needs layer i's FULL output
(dense + experts summed).  If cold experts for layer i run on the CPU while
the GPU is still doing layer i's dense, and the SUM waits for both, then the
per-layer time = max(GPU_dense_i, CPU_cold_i).  If instead the CPU work sits
on the critical path (GPU waits for CPU), the token time = sum of both.
This is a SCHEDULING question (how llama.cpp's backend scheduler splits a
mixed GPU/CPU layer graph and whether the async copies hide).  Measure with
a 2-backend graph: dense ops pinned to GPU, expert mmid pinned to CPU, with
the real tensor identities, and time a full 48-layer decode.

### M3. The n-cpu-moe mechanism is the WRONG tool - confirm and bypass.
llama.cpp's --n-cpu-moe (host-resident expert tensors) H2D-copies the USED
experts to the GPU every pass (~1.5 GB/token at 14 GB/s = ~105 ms/token).
It also CRASHES under the Meta-TP wrapper (tested: ggml-cuda abort with
tensor-split + ncmoe).  The tiered design needs a NEW mechanism: a persistent
GPU slot cache + compute the cold tail on the CPU IN PLACE (only the tiny
[2560] result crosses).  This is the actual implementation work and does not
exist yet.

### M4. The shuffling-policy parameters at S=266 on the REAL trace.
The sims (lfru_clean.py / lfru_swap.py) at S=104..256 gave:
  - cold rate 13-25% at S=208-256 with K=4/layer per 20 tokens
  - migration traffic 12-44 MB/token at those settings
  Re-run at S=266 with the full policy sweep (K in {1,2,4}, interval in
  {10,20,40}, decay half-life in {200, 2000, 10000}) and report cold% AND
  MB/token AND the steady-state-vs-convergence split.  Use the 2000-token
  html trace (route_html2000.txt) AND a second workload for stability.

### M5. Bus traffic reality at 14 GB/s.
Migration bursts (K x 48 experts per event) must be SPREAD async across the
interval, or they stall (1/layer/20tok = 236 MB burst = 17 ms at 14 GB/s).
Measure the actual achievable sustained H2D with async copies overlapping
decode, not the peak.

## 6. Pitfalls (learned the hard way this session)

1. L3 CACHE WARMING IN MICROBENCHMARKS.  This burned the session: an
   op-level perf number looked great (1.2 TFLOPS, "5 ms/token") because the
   harness looped one 18 MB tensor 18k times inside the 96 MB V-cache.  Any
   CPU timing claim MUST be validated against the full-CPU decode anchor
   (397 ms/token) or measured with cache-defeated tensors.
2. The full-CPU 397 ms anchor is the ONLY clean CPU number we have.  Do not
   decompose it by weight-share (dense 5.1 GiB vs expert 1.5 GB/token stream)
   and assume proportionality - the M=1 inefficiencies are not proportional.
3. Do not assume DRAM peak (90 GB/s) for M=1 quantized mmid.  Strided
   K-major expert reads + dequant are ~5x worse (18 GB/s effective per the
   anchor).
4. The routing DRIFTS: no static resident set is optimal, and a plant from
   the prompt is 3-6x worse than the oracle.  But per-window locality is
   strong.  Policy sims MUST separate convergence (first ~200 tokens) from
   steady state.
5. test-backend-ops perf and llama-bench ngl=0 are useful but have the above
   traps; llama-bench -ncmoe crashes on the Meta-TP path - do not rely on it
   for the tiered measurement.

## 7. Decision criteria (what "viable" means)

The experiment passes if, at S=266 with the chosen policy:
  - decode stays GPU-bound: token time within ~10-20% of the all-resident
    ~20 ms (i.e. <= 24 ms), measured end-to-end on the 2-GPU box
  - the CPU cold tail + bus migration + the [2560]-result crossings all fit
    under the GPU time (not on the critical path)
  - the policy beats the static S=266 split by a measurable margin at the
    Q8_0 geometry (where experts are 4.9 MB and S is tighter) - that is the
    real target; Q4_K_XL is the test vehicle
If the CPU cold cost at 5-15% cold is already >20 ms/token, the tiered idea
is dead and the answer is "accept layer-granular offload or a smaller model".
If it is 1-5 ms, the LFRU policy is the right architecture and the next step
is the loader/slot-cache implementation (Phase 1 in LRU_EXPERTS.md).

## 8. Tools / data left for the fresh session

- /tmp/prof/route_html600.txt, route_code600.txt, route_math600.txt,
  route_html2000.txt  (routing traces; LLAMA_ROUTE_DUMP output)
- /tmp/route_analysis.py, /tmp/cache_sim.py, /tmp/policy_sim.py,
  /tmp/lfru_clean.py, /tmp/lfru_swap.py, /tmp/lfru_final.py, /tmp/route_merge.py
- /tmp/feasibility_summary.txt (earlier, now partially retracted - see the
  correction in PHASE0_ROUTING.md)
- Scratch tree: /home/stew675/llama.cpp @ 2aad6e58d (route-dump instrumented);
  boost repo /home/stew675/llama-cpp-rdna-boosts (canonical): wip patches
  0021 (phase timers) and 0022 (route dump); LRU_EXPERTS.md + PHASE0_ROUTING.md
- Model: /models/Qwen3.8/Flash-Next/Q4_K_XL/... (Q4_K_XL, 103.69 GiB)
- Env for runs: HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0; warm the
  page cache first; leave GPUs idle between runs.

## 9. Suggested first actions for the fresh session

1. M1: build the cache-defeating CPU mmid probe (rotate ids across all 512
   experts, tensor >> 96 MB L3) and get the true per-pair M=1 CPU cost.
2. M2: prototype the mixed GPU-dense / CPU-expert layer graph (pin ops to
   backends) and measure per-layer overlap vs serialization.
3. M4: re-run the policy sweep at S=266 on both traces with the corrected
   CPU-cost model plugged in.
4. Then decide: LFRU implementation (Phase 1) or abandon.
