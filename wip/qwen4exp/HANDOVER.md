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
