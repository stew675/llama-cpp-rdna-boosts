# Phase 0 MEASUREMENT RESULTS (2026-09-04): expert routing profile of Qwen3.8-Flash-Next-UD

Instrumentation (scratch tree `7c7d9eeff` + route dump, env `LLAMA_ROUTE_DUMP`):
captures the per-layer `ffn_moe_topk` ids + `ffn_moe_weights` for every ubatch
eval (prefill and decode) by keeping the argsort parent alive (ggml_set_output)
and reading it back per step.  Validated: ids in [0,512), weights descending,
deterministic run-to-run (only ~2-3% near-tie boundary flips from allreduce FP),
diverse generated content (123/200 distinct tokens).  Router tensor (ffn_gate_inp)
is F32 in the Q4_K_XL file - the measured routing is NOT a quantization artifact.

Workloads (all at real KV, bf16 KV, tensor split 3x R9700):
- HTML/canvas generation: 2125-token prompt + 600 and + 2000 decode tokens
- coding task: 2743-token prompt + 600 decode
- math/reasoning: 2923-token prompt + 600 decode

## 1. THE HEADLINE: the routing is NOT heavy-tailed (contrary to the premise)

The design doc assumed the GPT-OSS-120B evidence (15-20% of experts handle ~80%
of tokens).  Qwen3.8-Flash-Next-UD does NOT behave like that:

- decode working set per layer over 600 tokens: 227-430 distinct of 512 experts.
  over 2000 tokens: 355-487 distinct (layer 0 reaches 487/512!).  still growing
  at +5-15 new experts per 250-token window.
- per-100-token windows use 100-190 distinct experts, 60-75% overlap with the
  previous window (steady drift, no settling core).
- firing coverage by the top-S experts of the DECODE trace itself (oracle):
      S=64:   67%      S=128:  86%      S=192:  95%      S=256:  99%
  (S is slots per layer of 512; identical shape on all three workloads)
- the router within a token is only mildly peaked: top-1 expert carries
  19-27% of the renormalized selection mass; sum of the raw top-10 softmax
  probs = 0.13-0.18 over all 512 experts.

Interpretation: this model was (almost certainly) trained with an expert
load-balancing aux loss, which intentionally spreads routing across all
experts.  There is no "core set of always-hot experts" of practical size.

## 2. Policy simulation (miss/layer/token after a 64-token warmup, 2000-token trace)

    S     oracle   reactive-LRU   prompt-plant   naive(0..S-1)
   128     2.16       1.54          5.71           7.47
   192     1.12       0.89          4.80           6.26
   220     0.82       0.67          4.44           5.71
   256     0.54       0.49          3.97           4.93
   288     0.35       0.39          3.57           4.41
   320     0.22       0.32          3.22           3.79
   384     0.06       0.24          2.63           2.52
   448     0.005      0.22          2.33           1.26

- oracle (resident = top-S by the full decode trace) needs S>=320 (62% of
  experts!) to get under ~0.25 miss/layer/token.
- the PROMPT prior is far worse than oracle (4-6x): generated HTML/JS code
  routes differently from the English spec that prompted it.  A prompt-derived
  resident set barely helps.
- reactive LRU floors at ~0.2-0.5 miss/layer/token even at S=448: the tail of
  one-shot experts keeps arriving (the new-expert rate IS the LRU floor).
- naive 0..S-1 (bring-up plant) is useless until S ~ 448.
- first-half->second-half calibration (the honest "workload prior" test):
  S=320 gives 2.9 miss/layer/token - the routing drifts mid-session.

Cross-workload resident agreement: Jaccard of the top-64/128 sets between the
three workloads is only 0.15-0.3.  There is no workload-stable resident set.

## 3. H2D byte budget (Q8_0 target, 4.92 MB per (layer,expert) miss)

Q8_0 deficit is ~44 GiB of 142 GiB; experts are 112 GiB of that, so with the
~30 GiB non-expert always resident, feasible resident expert budget is
roughly S = 240-260 of 512 (47-51% of experts resident).

- reactive LRU at S=256: 0.49 miss/layer/token x 48 layers x 4.92 MB
  ~= 115 MB/token of H2D.  At 50 GB/s = 2.3 ms/token if serialized, but spread
  per layer (2.4 MB/layer) it fits inside each layer's ~0.4 ms window (0.05 ms
  copy/layer) -> ~5.8 GB/s sustained, potentially fully overlapped with the
  ~20 ms decode.
- per-miss LATENCY: a cold expert needs a 4.9 MB copy (~0.1 ms) between the
  layer's router output and its mmid; up to ~0.5 misses/layer can be absorbed
  if copies are async and overlap the hit experts' GEMMs, but this is the
  make-or-break detail for a stall-free decode.
- the PR's CPU-cold fallback is NOT attractive here: ~0.5 cold expert/layer x
  48 layers = ~24 cold expert-GEMMs/token on the CPU would add far more than
  the H2D path at this miss rate.

## 4. What this means for the design

1. The "resident core + dynamic margin" (Amendment 3, 80/20) premise FAILS on
   this model: the resident tier is not better than pure LRU at the feasible S
   (oracle is only reachable with future knowledge; prompt/workload priors are
   weak because routing drifts with content).
2. The useful design is closer to a LARGE adaptive cache: as many slots as the
   VRAM budget allows (S ~ 250-260/layer at Q8_0), LRU/windowed dynamics, NO
   fixed resident tier (or a tiny one), accepting ~0.5 miss/layer/token.
3. Whether that is worth building hinges on the per-miss copy hiding: ~115
   MB/token at ~6 GB/s sustained is fine IF the decode never stalls on a cold
   expert.  The hot path (hits) is unaffected; only the cold path latency
   matters.  With S=250-260 the hit rate is ~95% of expert-selections.
4. Options to consider next:
   a. VERIFY with the Q8_0 model itself (does Q8_0 route identically? the
      router is F32 in both, expert weights differ only in quant error, so
      routing should be ~identical; confirm once a Q8_0 file exists).
   b. Prototype the decode stall question in the harness: emulate "slot cache
      + async H2D fill of the cold experts for THIS step" and measure whether
      the per-layer copy can hide (this is the only real unknown left).
   c. Reconsider scope: the value is "run Q8_0 at ~Q4_K_XL speed minus a few %"
      rather than "1.7-2.1x".  If the goal is quality-per-VRAM, Q4_K_XL already
      runs at full speed; the tier only buys the Q8_0 quality delta at ~95%
      speed.  Re-derive the expected quality/perf trade before building.
   d. A fundamentally different angle if more VRAM headroom existed (larger S)
      or if routing were more concentrated (smaller n_expert): not this model.

Data files: /tmp/prof/route_html600.txt, route_code600.txt, route_math600.txt,
route_html2000.txt.  Tools: /tmp/route_analysis.py, /tmp/cache_sim.py,
/tmp/policy_sim.py, /tmp/route_merge.py.

---

## FEASIBILITY SUPPLEMENT (2026-09-04): dense-on-GPU / experts-on-CPU split

Clarified design (user): GPU does the DENSE work (heavy, serial, every token),
CPU does the ROUTED EXPERTS (light, parallel, spillable).  Measured on the
9950X3D + R9700s, Q4_K_XL:

- CPU M=1 mmid at the TRUE qwen4exp shapes (test-backend-ops):
  gate_up [K=2560, out=1280, 10-of-512] Q4_K = 55.7 us/layer @ 1.18 TFLOPS
  down    [K=2560, out=640,  10-of-512] Q5_1 = 48.2 us/layer @ 680 GFLOPS
  => all 10 experts of one layer ~= 104 us; ALL 48 layers ~= 5 ms/token worst case
- full-CPU decode = 397 ms/token and KV-INDEPENDENT (64 vs 512 KV identical)
  => the dense work (attention/GDN/norms, sequential + recurrent state) is
     ~392 ms of it.  The experts are only ~5 ms even 100% on the CPU.
- => the CPU is FAST at expert mmid but SLOW at dense.  The split direction is
     unambiguous: dense belongs on the GPU, experts can spill to the CPU.

2-GPU Q4_K_XL frame (50 GB weights + 14 GB KV, model 111.3 GB):
- dense 34.3 GB always on GPU; expert budget 15.7 GB -> S = 104..208 of 512/layer
- cold experts on CPU: static S=104 (59% cold) = 2.9 ms CPU; LFRU S=104 (25%) = 1.2 ms
- both hide under the GPU's dense-bound ~20 ms/token => ~50 t/s either way
- llama.cpp TODAY at 50 GB (layer-granular: 27 whole layers on CPU incl. dense)
  = ~232 ms/token = 4.3 t/s.  The tiered split is an ~11x win over that.

Policy verdict for THIS frame: static expert split (dense + top-S by profile on
GPU) == LFRU in token rate; the movement policy adds only bus traffic here.
The movement policy becomes load-bearing when the CPU expert budget tightens
(Q8_0: ~4.9 MB/experts, smaller S, 5 ms worst-case CPU vs a smaller GPU budget).

Implementation note: llama.cpp's existing --n-cpu-moe path (dense GPU + expert
tensors host-resident) is the WRONG mechanism for this - it H2D-copies the used
experts to the GPU every pass (~1.5 GB/token at 14 GB/s PCIe = ~105 ms/token).
The design here computes cold experts ON THE CPU IN PLACE (only the tiny
[2560] result crosses) - that is the novel part and is what makes the split win.
