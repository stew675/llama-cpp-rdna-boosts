# HANDOVER 2026-09-03: LLAMA_QSA_SPARSE_FA decode regression - sparse slower + faster decay than dense

## STATUS / THE PROBLEM (user report, 2026-09-03, current HEAD f8cce6e68)

llama-server decode on the 3x R9700 tensor-split, short prompt + generation:

- LLAMA_QSA_SPARSE_FA unset/1 (SPARSE path, now the DEFAULT): decode starts at
  ~39 t/s and falls quickly - BELOW 30 t/s after only ~1000 generated tokens.
- LLAMA_QSA_SPARSE_FA=0 (DENSE masked FA path): decode starts at ~46 t/s and
  after 12,000 tokens has only fallen to ~43 t/s.

Expected: the QSA sparse path (attend only the indexer-selected top-k cells)
should be FASTER than the dense path AND decay more slowly (its work should be
~constant in the real KV length). It is currently slower at the start AND
decays much faster. Something is wrong with the sparse path.

## LIKELY ROOT CAUSE (identified, needs confirmation + fix)

The sparse budget is NOT small. From the model GGUF metadata + qwen4exp.cpp:

- `qwen4exp.attention.indexer.top_k = 2048`; the QSA layers (every 4th of 48)
  have `attention.compress_ratios = 4`.
- qwen4exp.cpp build_qsa_top_k (line ~729):
  `width = min(n_kv, indexer_top_k + r - 1)` = min(n_kv, 2048 + 4 - 1) = up to
  **2051 CELLS**, passed as the k of the fused `ggml_indexer_top_k` op; the
  flash_attn_qsa kernel then attends those up-to-2051 cells.
- CONSEQUENCE: for any real KV < ~2051 tokens the sparse kernel attends ALL
  cells (zero sparsity) - and it is slower per attended cell than the dense
  flash_attn_ext (start 39 vs 46 at equal cell counts). The decode time then
  grows with real KV until the 2051 cap, exactly the observed 39 -> <30 by
  1000 tokens. Beyond ~2051 the sparse work should flatten (verify!).
- The dense path's slow decay (46 -> 43 over 12K) says the shared O(n_kv)
  indexer (block score mm over n_kv/r blocks + pooling GET_ROWS over n_kv) is
  cheap; the regression is specific to the sparse attention kernel/geometry.

TWO possible fixes (decide after measuring):
1. The sparse kernel is far less efficient per cell than the dense FA kernel
   (~6x estimated from the numbers above). Make flash_attn_qsa as efficient as
   the RDNA4-tuned dense FA (streaming/vectorized K/V reads, occupancy,
   tile geometry) - the kernel's random-cell smem staging + L2 V reads may
   also degrade as the cache spreads with n_kv.
2. If the per-cell cost cannot reach dense-FA parity, the sparse path only
   wins at very large real KV (crossover ~ 2051 x per-cell-ratio); that is an
   architecture property, not a bug - but the kernel still must not LOSE at
   short context as badly as it does.

NOTE: the decode campaign's 45.6-45.7 t/s tg128 numbers NEVER measured this -
llama-bench's tg128 test runs from an EMPTY context (n_prompt=0, real KV grows
0->128 in the test). Real-KV decode must be measured with `-pg N,128`.

## ENVIRONMENT / FACTS (verified this session)

- ~/llama.cpp branch qwen4exp @ f8cce6e68 (flattened history: master 9cffdcc80
  + 13 rdna-boosts blocks + 2 beta-patch commits 1d35cd17e + f8cce6e68).
  Clean. NOT pushed. Old fine-grained history = branch qwen4exp-full-2026-09-03.
  The two commits == beta/qwen4exp/{managed-ngrams,qwen4exp-support}.patch.
- Model: /models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  (4 shards, 103.68 GiB). 48 layers: 12 QSA (attn) + 36 GDN. MoE 512 experts
  x 640, top-10. hc = 4 streams. indexer: 4 heads, key 128, top_k 2048,
  ratios: 4 on the QSA layers.
- Machine: 3x R9700 gfx1201 (32.6 GiB each), 184 GB RAM, 9950X3D2 (16c/32t),
  ROCm /opt/rocm-7.14-gfx1201. Manual DPM applied (tune_r9700.sh, -85 mV/265 W).
- Build: build-rocm (Unix Makefiles). Rebuild ggml-hip after kernel changes
  (~1-2 min for one .cu; full ~5-10 min). `make -C build-rocm -j16 <target>`.
- Gates use env: HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 (sparse FA
  is DEFAULT now; LLAMA_QSA_SPARSE_FA=0 = dense; -fa off = manual attention).
- COLD-CACHE LESSON (must warm before ANY perf run):
  `for f in /models/Qwen3.8/Flash-Next/Q4_K_XL/*.gguf; do dd if=$f of=/dev/null bs=4M; done`
  (~20 s). A cold cache makes the first evals pay disk page-ins (fake ~4x
  prefill slowdown, deterministic tight error bars).
- Kill servers with `pkill -9 -f "llama-server --model"` in a SEPARATE tool
  call (the pattern kills any shell whose cmdline contains it). Wait ~5 s and
  verify rocm-smi VRAM ~ 60 MiB on GPUs 0-2 before the next load.

## SERVER LAUNCH (the user's exact config - reproduced working 2026-09-03)

```
HIP_VISIBLE_DEVICES=0,1,2 NCCL_PROXY_CPUSET=8..15 GGML_CUDA_FA_WMMA_256=0 \
./build-rocm/bin/llama-server \
  --model /models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --alias test --fit false --top-k 20 --port 8040 --threads 8 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --host 127.0.0.1 --threads-http 4 \
  --ctx-size 102400 --flash-attn auto --temperature 1.0 --batch-size 2048 \
  --ubatch-size 2048 --n-gpu-layers all --no-kv-unified --cache-type-k bf16 \
  --cache-type-v bf16 --ctx-checkpoints 64 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
  --split-mode tensor
```
(setsid ... < /dev/null > log 2>&1 &; log to /tmp/srv.log). Completion:
`curl -s -X POST localhost:PORT/completion -d '{"prompt":"...","n_predict":N,
"temperature":0,"stream":false}'` - response timings.predicted_n/ms = decode
t/s. To measure the decay curve: generate long (n_predict 20000+) and read the
server's per-window timing lines, or use the completion timings per request.

READY-MADE probes in /tmp (written this session, survive reboot only):
- /tmp/server_test.sh CTX [extra flags]  - launch + short completion + timings
- /tmp/srv_sustain.sh  - 150-token chat completion, prints content + timing
- /tmp/tdec.cpp + /tmp/tdec - decode token ids to text (llama_vocab)
Rebuild any /tmp harness against build-rocm when llama_model_params changes.

## BENCH-LEVEL REPRO (the clean real-KV discriminator - DO THIS FIRST)

llama-bench -pg N,128 measures decode over REAL KV = N (prefill N, then 128
tokens). Warm the cache first. Sparse (default) vs dense (LLAMA_QSA_SPARSE_FA=0):

```
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 ./build-rocm/bin/llama-bench \
  -m <model> -pg 512,128 -pg 2048,128 -pg 4096,128 -pg 8192,128 -pg 16384,128 \
  -ub 2048 -r 3 -t 16 -ngl 99 --split-mode tensor
```
(re-run with LLAMA_QSA_SPARSE_FA=0 for the dense curve; note -pg runs prefill
N then decode 128 in ONE context = the real-KV decode rate). Expected if the
root-cause theory holds: sparse decays from ~46 (KV 512) toward a floor by
KV ~2051 and stays flat; dense decays slowly. Confirm the sparse floor and the
crossover vs dense. Also grab the ppN numbers to confirm prefill parity.

## INVESTIGATION PLAN (next session, in order)

1. -pg sweep above (sparse vs dense). This settles the curves at the bench
   level and whether the server matches (server ~39 start at KV ~500-1000).
2. Dump the ACTUAL n_top_k at decode: the top_k tensor ne[0] (or idx->ne[0])
   for a few real KV values - verify the min(n_kv, 2051) cap is what the graph
   really produces (no off-by-one turning it into n_kv). Quick in-compute read
   or log the fused op's k param (op_params).
3. If sparse flattens at ~2051 cells but is still at/below the dense rate:
   the kernel per-cell efficiency is the problem. Op-profile the decode at
   real KV 1024 vs 8192 sparse vs dense (GGML_CUDA_OP_TIMING=1; llama-bench
   swallows logs unless --verbose; the decode graph may not hang like the
   prefill one did - try it). Compare flash_attn_qsa vs flash_attn_ext real
   times at equal cell counts.
4. Kernel-level: micro-bench flash_attn_qsa at fixed top-k vs growing n_kv
   (does the constant-2051-cell kernel slow down as the cache spreads? the
   smem tile gather + the per-cell L2 V reads are the suspects). Compare
   against the dense FA kernel's per-cell cost. Check occupancy/vectorization
   (the kernel uses 8 lanes/cell, half2 math; dense FA on RDNA4 has WMMA +
   wide loads).
5. THEN fix: either (a) rework the sparse kernel for dense-FA-level
   efficiency (bigger job), or (b) reduce the per-cell overhead + confirm the
   flat floor is at least competitive; if the budget (2051) itself is the
   issue vs the reference implementation semantics, verify against the model
   (the +r-1 "whole blocks plus tail" comment - do NOT change semantics
   without a quality gate).
6. Re-gate after any kernel change: single-seq coherence (bseq_val streams
   vs the known-good 198 42750 367...), llama-bench pp/tg numbers, and the
   server full-config sanity.

## GATES (unchanged; run after ANY decode-affecting change)

- bseq_val K=1/K=2 streams == known-good: "Quantum" -> 198 42750 367 367...,
  "The answer" -> 369 9542 11 694... (/tmp/bseq_val -p PROMPT [-p PROMPT...]
  -n 10 -c 4096 -ngl 99 --split-mode 3, same env).
- llama-bench (warm cache): tg128 ~45.7, pp512 ~1538, pp8192 ~2024 (empty-ctx
  tg128 = the OLD reference - NOT a real-KV number; keep it only as the
  regression baseline for the decode kernel work).
- test-llama-archs -a qwen4exp (MoE GPU ~1e-13, CPU 0.00).
- Server full config at ctx 102400: coherent text, ~39 t/s (sparse) - this
  session's number is the pre-fix baseline.

## REPO / DOC STATE

- llama.cpp qwen4exp @ f8cce6e68 (flattened, clean). Beta baseline:
  ~/llama-cpp-rdna-boosts/beta/qwen4exp/ (managed-ngrams.patch +
  qwen4exp-support.patch + README.md with the open-items list).
- WIP archive: ~/llama-cpp-rdna-boosts/wip/qwen4exp/archive/ (per-commit
  patches 0005-0020 with notes, all handovers incl. this session's record,
  plans, tools). Handover for the crash/BF16-fix sessions:
  archive/handover-2026-09-03-server-crash-investigation.md.
- Boost repo main @ fc6aa91 (pushed by user).
- The decode-campaign handover (archive/handover-2026-09-04-decode.md) has the
  op-profile notes and kernel-count wall analysis - the sparse kernel's cost
  model should be read before touching it.

## OPEN (unchanged backlog)

- "The answer" 3-token K=2 multi-seq decode drift at step 3.
- Mixed K/V cache types (k=bf16/v=f16) crash at model init (meta:537).
- FA-off + tensor-split unsupported (upstream Meta-backend constraint).
- Decode levers: server batch decode (M > 1), decode-expert mmvq config, GDN
  state fold, body-op fusion; prefill thread (pp8192 at ~2024); ML-Kernel/gpudh
  review vs TP-V1; v_shifted probe; splitter->F32; meta import gates;
  multi-device sync in meta_tp_test.
- PARKED this session: "is MTP (block 01 adaptive draft depth) enabled on the
  qwen4exp paths?" - dropped mid-check; qwen4exp has no graph_mtp and the GGUF
  scan was not completed (no mtp tensors seen in a quick grep of shards 2-4).

---

## RESOLVED 2026-09-03 (session addendum): the sparse decode regression is fixed

Three kernel commits on ~/llama.cpp qwen4exp (behind the two beta-patch
commits; apply in order, all coherent + gates green):

1. 419370021: stage V in smem for the BF16 QSA FA path. The BF16 branch
   never wrote V_smem; the VKQ pass re-read every cell's V from L2 once per
   head-warp. Mirrors the F16/Q8_0 gather. -1.4 ms/step at low ctx, -6 ms at
   depth (bit-neutral by construction; verified).
2. ffb9c56f4: slice the QSA top-k walk across gridDim.y at decode. Decode
   launched ONE block for the whole top-k list = one CU; now slices of 256
   cells per block with per-slice online softmax + the existing
   flash_attn_combine_results fixup (exact copy of the dense FA's KV-slice
   pattern). Bit-identical to the single-block path (verified 0.0 logit
   delta vs GGML_CUDA_QSA_SLICES=1). Only engages when tokens x streams
   under-use the GPU (prefill untouched).
3. 340f4621a: shrink the decode slices to 64 cells (33 blocks at the 2051
   budget) - dense parity at low ctx, lead past ~8K.

Measured (bseq_pp, bf16 KV, tensor split, steady-state ms/step):
  real KV    dense    sparse(before)   sparse(after)
    512       20.2        23.5             20.2
   2048       20.5        31.9             20.6
   8192       21.3        32.5             21.1
  32768       24.3        34.9             23.3

The root-cause note above (2051-cell budget not sparse below ~2K) still
stands as the architectural reason the OLD kernel decayed; the decay was
per-cell cost (V re-reads) plus single-CU serialization, both now fixed.
Debug envs added: GGML_CUDA_QSA_DEBUG (launch geometry), GGML_CUDA_QSA_SLICES
(override slice count).

Server (user config, ctx 102400, bf16 KV, HTML prompt, temp 0):
  before: 38.4 t/s start, decaying to 29.2 by 1500 gen (tg_3s), avg ~34.
  after:  43.2 (warmup) -> 47.5-48, flat ~46.5-47 through 4000 gen, avg
  46.7. No decay. (Dense server was 46 start -> 43 by 12K; sparse now
  equals dense at low ctx and beats it at every depth past ~8K real KV.)

REMAINING to 50 t/s server-side (~1.9 ms/step): the decode floor is ~19.5
ms of MoE/GDN/elementwise + launch tail (bench sparse at short ctx = 20.2
ms = 49.5 t/s); the server adds ~0.8 ms host overhead (not threads, not
ctx size, not checkpoint flags). Ranked levers (decode-campaign evidence):
mmvq decode-expert config (down K=640), kernel-count-reducing fusions,
server host overhead hunt. The QSA path itself is at parity and no longer
the constraint.

---

## PRE-COMPACTION NOTES 2026-09-03 (session end)

### Git state
- llama.cpp qwen4exp tip = 6c820fd79 (4 new commits above the 2 beta
  commits: 419370021 V-smem, ffb9c56f4 slicing, 340f4621a 64-cell slices,
  6c820fd79 trailing-newline cleanup). Flattened history, NOT pushed.
  Beta patches regenerate to this tip tree-identically (committed to the
  boost repo as c60c548; user pushed).
- The 3 kernel fixes + cleanup live ONLY on the llama.cpp qwen4exp branch
  and in the beta patch - nothing pushed to qwen4exp-src (the canonical
  repo at /home/stew675/qwen4exp.cpp) yet; the user pushes.

### Reproduce commands (the canonical real-KV decode harness)
- /tmp/bseq_pp.cpp (+ binary): prefill P tokens cyclically, then decode
  single-token steps at real KV = P. Usage:
  `bseq_pp -m <model> -pp 512,2048,8192,32768 -n 24 -ngl 99 --split-mode 3 -kv bf16`
  (-kv bf16 matters: bf16 is the production path; -c N overrides ctx).
  Steady state = steps 3..N-1 (steps 0-2 are graph warmup).
  Rebuild: g++ -std=c++17 -O2 -I include -I ggml/include /tmp/bseq_pp.cpp
  -L build-rocm/bin -lllama -Wl,-rpath,<abs build-rocm/bin> -o /tmp/bseq_pp
  (link against build-rocm/bin/libllama.so; rebuild when the API changes).
- Env debug toggles in fattn-qsa.cu: GGML_CUDA_QSA_DEBUG (prints launch
  geometry per QSA op), GGML_CUDA_QSA_SLICES=N (override slice count),
  GGML_CUDA_QSA_IDENTITY=1 (idx = sequential, dense-equivalent ordering).
- Dense comparison: LLAMA_QSA_SPARSE_FA=0. Always warm the page cache
  first and leave the GPUs idle (a leftover server holds 31 GiB/GPU and
  any harness then OOMs at model load).

### Gate nuance: "The answer" baseline is STALE on this tree
- The recorded reference stream "The answer" = 369 9542 11 694... does
  NOT reproduce on the current tree under ANY kernel state - the pristine
  pre-change fattn-qsa.cu also yields 369 9542 13 198 760 4087...
  (prompt documented as drift-prone: 11 = ' ', 13 = newline at step 3).
- The reliable single-seq gates = "Quantum" (198 42750 367 367 318
  210452 13 29144) and "What is the capital of France?" (271 760 6511
  314 9338 369 2972 57590 ...) - both pass bit-exact. Sliced vs
  single-block numerics verified identical (0.0 logit delta over a 4-token
  dump at /tmp/ld_sliced.bin vs /tmp/ld_unsliced.bin).

### Where the session stopped (next steps for 50 t/s server-side)
- Server at ~46.5-48 t/s flat (21.4 ms/step avg 46.7 over 4000 gen; was
  38.4 -> 29.2 decaying). Bench at short ctx = 20.2 ms (49.5 t/s).
- Remaining to 50 t/s server: decode floor ~19.5 ms (MoE/GDN/elementwise
  + launch tail) + ~0.8 ms server host overhead. Ruled out: threads
  (8 vs 16 identical), ctx size (102400 vs 600 identical), checkpoint
  flags (on/off identical). Ranked levers: (1) decode-expert MMVQ config
  (down K=640 @ ~166 GB/s), (2) kernel-count-reducing fusions, (3) server
  host overhead (sampler cost / per-token slot bookkeeping).

---

## UPDATE 2026-09-04 (fusion session): MoE weighted-reduction fusion unblocked on Meta

llama.cpp qwen4exp commit d9b1ef288 (on 6c820fd79), NOT pushed:
- PROBLEM: the decode's expert aggregation (weighted MUL + 10 views + 9
  ADDs/layer) only fused for 2/48 layers on the tensor-split decode. The
  CUDA fusion pass (ggml_cuda_try_fuse -> moe_weighted_reduction) refuses
  to fuse when dst may alias the experts input; the alloc deps that prevent
  the aliasing are registered by the CUDA backend's graph_optimize, which
  NEVER runs for graphs owned by the Meta-TP wrapper (meta had
  .graph_optimize = nullptr, and the scheduler allocates the whole graph
  into Meta buffers before the wrapper dispatches the per-device subgraphs).
- FIX: moved the structural matcher (struct + ggml_match_moe_weighted_reduction)
  from ggml-cuda.cu into a portable header
  ggml/src/ggml-moe-weighted-reduction.h (pure ggml types; ggml_can_fuse_subgraph
  is in ggml-impl.h = base lib), added ggml_backend_meta_graph_optimize to
  ggml-backend-meta.cpp registering the same alloc deps against the
  scheduler's allocator, and wired .graph_optimize into ggml_backend_meta_i.
  Result: 429/429 fusion fires (all 48 layers x every fragment).
- VERIFICATION: fused-vs-unfused decode logitdump (GGML_CUDA_DISABLE_FUSION=1
  A/B, same prompt) BIT-IDENTICAL; "Quantum" gate passes; pp512 1557 (no
  prefill regression from the extra alloc deps).
- PERF: bseq_pp real-KV 2048 steady state 20.60 -> 20.33 ms/step (-0.27 ms).
  Server (user config): 46.7 -> 47.4 t/s flat (21.08 ms/tok). Still ~1.1 ms
  from 50 t/s server-side.
- Debug env kept: GGML_CUDA_MWR_DEBUG=1 logs "MWR FUSED <name>" per fusion.

REMAINING fusion map (from the post-fix op profile, per 48-layer decode):
- ffn tail per layer: shared_expert_gate MM + UNARY sigmoid + MUL + ADD
  (4 kernels; 3 tiny tails foldable into the hc_combine op contract - the
  combine consumes ffn_out; passing moe_out/shexp_down/shared_gate instead
  would save ~144 kernels, est ~0.2-0.5 ms; moderately invasive: combine
  src contract 3->5, CPU ref, meta mirror).
- GDN state ops (GET_ROWS/CPY per recr layer) est ~0.3-0.6 ms, invasive.
- mmvq decode-expert config (item a of the user plan): MEASURED DEAD END
  in the decode campaign (session 7: default rpb formula wins/tie every
  shape; only ~0.3-1% single-shape wins; VDR 4->8 no change).
- Server host overhead ~0.8 ms/step (server 21.08 vs bseq 20.33 at equal
  real KV; threads/ctx/checkpoints ruled out earlier; sampler + per-token
  slot bookkeeping = the suspects). NEVER HUNTED - the biggest remaining
  single lever toward 50 t/s server-side.

---

## UPDATE 2026-09-03 (host-overhead hunt): the server gap is NOT the decode

llama.cpp commits d9b1ef288 (moe fusion) + 7c7d9eeff (phase instrumentation).
All measurements on the 3x R9700 tensor-split, user config, bf16 KV.

FINDINGS (each verified by measurement):
1. The sampler chain is FREE: bseq_pp + the exact server chain (top_k 20,
   top_p .95, min_p .001, greedy) = 20.33 ms/step = plain bseq. The server's
   own DEBUG_TIMINGS agree: t_sampl = 0.107 ms, t_post = 0.108-0.148 ms,
   t_pre = 0.036 ms.
2. ctx size (102400 vs small) and threads (8 vs 16) = no effect at the
   bseq level. The prompt CONTENT (cyclic vs real HTML) = no effect (19.6
   both). llama-cli == llama-server (cli wraps cli_server).
3. llama-bench tg128 (21.9 ms/tok at KV 0-128) is NOT comparable: it never
   reaches the steady graph; bseq at the same regime = 20.0.
4. THE DECODE IS AT PARITY: llama_context::decode in the SERVER = 20.05
   ms/step steady (last-150 avg, 3000-token run) + sched sync 0.121 =
   20.16 ms/token - vs bseq 19.6-20.1 at the same real KV. The graph-reuse
   machinery works (1991/2000 reused; the 12 rebuilds/token-run amortize
   to nothing). The decode-side fusion + QSA work has reached the floor.
5. THE REMAINING ~1.0 ms/token (tg 21.15 vs decode 20.16) = the llama-server
   update_slots LOOP overhead: pre 0.036 + post 0.148 + sampl 0.147 = 0.33
   measured + ~0.67 ms/token of untimed queue/event-loop churn (the
   NEXT_RESPONSE task round-trip + the update_slots wakeups + metrics).

PATH TO 50 t/s (20.0 ms/token): the decode is already ~20.0 at low KV
(19.6-19.9). Options: (a) reduce the server loop overhead - the ~0.67 ms
untimed churn per token (single-slot fast path / fewer queue round-trips;
the llama-server hot loop = invasive-ish, upstream-adjacent); (b) squeeze
the decode below 20 via the remaining small fusions (ffn tail ~0.2-0.4 ms,
GDN state ops ~0.3 ms) - only meaningful if the loop overhead also drops.
The mmvq decode-expert config (item a) was measured dead in the decode
campaign (session 7).

INSTRUMENTATION: LLAMA_DECODE_PHASE_DEBUG=1 prints decode phases
(balloc_init/mem_update/mctx_ready/ubatch_compute_done/sched_sync) per
decode from llama-context.cpp; DEBUG_TIMINGS in server-context.cpp (the
t_pre/t_decode/t_post/t_sampl 5s averages) is compile-time - enable the
define + rebuild llama-server to use it. Both were reverted to zero-cost.
Tools: /tmp/bseq_pf = bseq_pp with BSEQ_PROMPT_FILE= real-prompt prefill.

---

## NEW WORKSTREAM 2026-09-03: LRU / tiered expert weights (LRU_EXPERTS.md)

Design doc: wip/qwen4exp/LRU_EXPERTS.md. Goal: run qwen4exp Q8_0 (~142 GiB
projected vs 97.8 GiB VRAM) with per-layer hot expert slots on the GPUs +
the rest host-resident, loading on demand. Based on upstream PR 26563
(expert caching, never merged) - reviewed, its failure modes mapped onto
the rdna-boosts tree (Meta-TP split, lazy reader, mmvq decode, CUDA-graph
replay). Key conclusions: (1) pure LRU streaming is impossible (the router
runs inside the decode graph - the needed experts are unknown before the
step); the base = PR-style dual path (hot GPU slots + always-available
host tensor, cold computed on the CPU with the LRU refresh on cadence);
(2) all graph shapes must stay fixed per step (replay) - only the slot
CONTENT + LUT change (H2D); (3) no bit-exact reference exists for Q8_0
all-GPU - gates = determinism at a fixed cache state + coherence + content
quality; (4) model facts measured: Q4_K_XL = 103.69 GiB, routed experts
~74 GiB (71%), down [640,2560,512] Q5_1 = 600 MiB/layer, fused gate_up
[1280,2560,512] Q4_K ~943 MiB/layer; Q8_0 expert = ~4.9 MB/(layer,expert);
(5) phased plan: measure the routing profile first (hit-rate-vs-S curves
decide S and the CPU cold budget), then loader placement (S=512 full
residency as a plumbing gate), then the dual-path decode graph, then the
live LRU policy. Suggested first steps + open questions in the doc.
