# HANDOFF: qwen4exp session 2 - 2026-08-31 (before context compaction)

Status: MAJOR WIN. The real CPU-burn root cause was found and fixed.
Context ~600K, compacting. This doc supersedes qwen4exp-handoff-2026-08-31.md
(update 1 and 2 are still valid history; update 3 below is the new headline).

## TL;DR - what happened this session

1. **The 16-thread CPU burn was NOT the ngram gather.** It was the QSA
   indexer's top-2048-of-16384 partial sort running on the CPU backend.
   FIXED by backporting upstream's HIP radix top-k kernel.
2. **pp16384 went 784 -> 1307 t/s (ub512, +67%) and 1083 -> 1716 t/s
   (ub2048, +58%).** CPU burn dropped 1260% -> 140% (16 cores -> 1.4).
   Verified BIT-IDENTICAL via logitdump A/B (GPU radix vs CPU heap_select).
3. The PLE batch cache (user-requested prototype) is implemented, verified
   bit-exact on prefill + continuation batches, but performance-NEUTRAL
   (1722.8 vs 1722.2). The ngram gather was never the bottleneck.
4. **Depth scaling measured out to 65536**: pp16384 1734, pp32768 1381,
   pp65536 995 (ub2048). VRAM at 65536 = 31.645/31.86 GiB per GPU (99.3%).
   Model n_ctx_train = 262144, so 200K is within native range but needs
   KV/offload strategy. GPU0 VRAM climbs ~1 GiB then plateaus over a run
   (30.5 -> 31.65 GiB, stable after warmup) - normal graph/KV growth,
   NOT a leak (checked over 240s, plateaued at 31.645 and stayed).

## Current best numbers (all fixes, tensor split, 3x R9700)

Env: GGML_CUDA_FA_WMMA_256=0 (WMMA FA hurts head_dim 256), -t 16,
--split-mode tensor, HIP_VISIBLE_DEVICES=0,1,2, ub2048:

| test | t/s |
|---|---|
| pp512 | 1492 |
| pp2048 | 2028 |
| pp4096 | 2031 |
| pp8192 | 1935 |
| pp16384 | 1716 |
| pp32768 | 1381 |
| pp65536 | 995 |
| tg128 | 40.1 |

Prefill target >1000 t/s at 16K: exceeded 1.7x. Depth curve still falls
(attention + top-k + mask scale with n_kv); 200K needs work (see below).

## THE FIX (this session's headline)

**Root cause**: QSA sparse-attention indexer (12 layers, compress_ratio=4,
indexer.top_k=2048) does top-2048-of-16384 per token. The CUDA TOP_K op was
gated at `ne[0] <= 1024` without CUB, so the scheduler put it on CPU:
264 TOP_K nodes at 16K, 16 OpenMP threads in `std::__heap_select`
(gdb-verified: ggml_compute_forward_top_k -> __heap_select).

**Fix (2 files, pure backport from upstream master)**:
- `ggml/src/ggml-cuda/top-k.cu`: added upstream `top_k_radix_cuda` -
  4-pass 8-bit radix-select, per-row histograms, for ncols > 1024 on HIP.
  Our tree (rdna-boosts 4fa92f0ae) predates it.
- `ggml/src/ggml-cuda/ggml-cuda.cu` supp(): TOP_K now returns true on HIP
  for any ne[0] (matches master). ARGSORT stays <=1024 (bitonic-only on
  HIP; the MoE router argsorts are small so they were already fine).
- Added env A/B switch: `GGML_CUDA_TOP_K_CPU=1` forces the old CPU path.
- Backup patch: wip/qwen35moe-prefill/patches/0006-qsa-topk-radix-gpu.patch

**Correctness**: logitdump (same build, GPU radix vs CPU heap_select,
32 tokens, seed 42) = byte-identical output. The radix kernel returns the
same index SET in unsorted order; the consumer scatters into a mask so
order does not matter. Also matches the reference worktree build.

## PLE batch cache prototype (user-requested; NEUTRAL, keep/drop decision)

Implemented the "one sorted batch gather per prefill" idea:
- New hook: `llama_model::prepare_batch(batch, memory)` called once per
  decode() before the ubatch loop (llama-model.h, llama-context.cpp).
- qwen4exp override computes ALL PLE row indices for the whole logical
  batch, gathers+dequantizes in sorted order (sequential mmap reads),
  caches F32. Per-ubatch set_input just memcpys the slice - no gather,
  no hash, no OpenMP spin in the hot path.
- Eligibility: tokens present, single seq, contiguous positions; boundary
  predecessors resolved from attn cells (seq_pos_token_le).
- Verify env: LLAMA_PLE_CACHE_VERIFY=1 (recomputes per-ubatch and compares;
  PASSED on prefill pos 0/512/1024/1536 AND continuation batches).
- Disable env: LLAMA_PLE_CACHE_DISABLE=1.
- Result: neutral at all depths (1722.8 vs 1722.2 at pp16384/ub2048)
  because the gather was never the bottleneck. DECISION NEEDED: keep
  (it removes per-ubatch gather, would matter if table were bigger or on
  slower storage; adds ~200 lines + a virtual hook) or drop (lean tree).
  My recommendation: drop unless a future config shows gather-bound
  behavior; the top-k backport is the real win and is tiny.

## Why the PLE cache would matter (answer to "when does it become useful")

The ngram gather is per-token CONSTANT (16 rows/token, 90B each) - it does
NOT scale with context depth. The things that DO scale with n_kv:
- QSA indexer top-k (now GPU; still O(n_kv) per token per QSA layer)
- KQ mask build (n_kv x n_tokens, CPU set_input per ubatch)
- FLASH_ATTN over n_kv
- get_prev_tokens (now O(log n) via PR #27992 index - DONE this session)

So the PLE cache's usefulness is independent of depth: it only wins when
the gather itself is slow (managed/lazy reader on slow storage, or huge
table). At 200K the depth-scaled costs above dominate, NOT the gather.
The path to 200K is: KV memory strategy + the depth-scaled ops, not the
ngram table.

## Depth scaling analysis for the 200K target

- n_ctx_train = 262144 (native). 200K is within range.
- VRAM at 65536: 99.3% full. The KV cache for 12 full-attn layers +
  36 GDN/recurrent layers at 65K ctx is the pressure. Model weights use
  ~78-80 GiB; total VRAM 97.8 GiB (3x 32.6).
- pp t/s falloff: 1734 (16K) -> 995 (65K) = -43%. Per-token cost grows
  ~linearly in n_kv from attention/mask/top-k.
- 200K will need: KV cache quantization or offload strategy (BF16 KV was
  already measured better than F16 on this model), possibly swa/sliding
  window on some layers, and the depth-scaled op costs addressed.

## Remaining work / next steps

1. **200K context**: investigate KV memory at 65K->200K. Check the KV
   buffer split (attn vs recurrent), consider -c 200000 --no-kv-offload
   tradeoffs, KV quant types. Measure pp at 96K/128K to map the curve.
2. **Decode 40 -> 50**: decode is memory-bandwidth bound (103.68 GiB
   weights); result_output [2560x82774x1] = 0.6ms alone. Not top-k bound
   (A/B confirmed 37.8 both ways). Candidate: expert MMQ width, speculative
   decode, or KV-cache-aware decode batching.
3. **Decide PLE cache**: keep or drop (see above).
4. Consider raising GGML_CUDA_TOP_K env handling: the A/B switch is useful,
   keep it; document GGML_CUDA_TOP_K_CPU=1 for regression testing.
5. Strix Halo (RDNA3.5) validation of the fused MoE MMQ (RDNA4-gated) and
   the radix top-k (should be arch-neutral, but verify) - parked.
6. 2-GPU / model-deploy follow-ups from qwen35moe plan - parked.

## Tree state (35 files modified, uncommitted by design)

~/llama.cpp rdna-boosts @ 4fa92f0ae, working tree has:
- block 11 revert (prefill CUDA graphs) - KEEP
- ngram patches 0001-0007 - KEEP
- MoE fusion work (mmq/mmvq) - KEEP
- PR #27992 kv-cache prev-token index (applied + unit-tested, 0 mismatches;
  flat at 16K, big win at 240K) - KEEP for the 200K target
- **NEW this session**: top-k.cu radix backport + supp change +
  GGML_CUDA_TOP_K_CPU env - KEEP
- **NEW**: PLE batch cache prototype (prepare_batch hook + set_input cache
  path + verify/disable envs) - DECISION NEEDED
- NOT committed to ~/llama.cpp (needs user approval; boosts-repo wip/
  commits are the norm)

Backup patches in ~/llama-cpp-rdna-boosts/wip/qwen35moe-prefill/patches/:
- 0005-kv-prev-tokens-index.patch (PR #27992)
- 0006-qsa-topk-radix-gpu.patch (this session's fix)

## Environment (unchanged, still valid)

- Runtime PM fixed (power/control=on on 0000:{03,06,09}:00.0); re-apply
  after reboot. journalctl "SMU is resuming" should be quiet.
- /tmp is tmpfs - keep logitdump bins out of it (16GB each at 16K).
- 3x R9700 need HIP_VISIBLE_DEVICES=0,1,2; NO HSA_OVERRIDE_GFX_VERSION.
- llama-cli needs --single-turn; -b n_batch, -ub n_ubatch, -n 0 pure
  prefill; logitdump --split-mode is INT (TENSOR=3).
- Reference worktree /home/stew675/q4exp-ref (a7cc83bba+ngram) - keep.
- Machine: Ryzen 9 9950X3D, 184GB RAM, ROCm 10.0.0.

## Key refs

- ggml/src/ggml-cuda/top-k.cu (radix kernel, the fix)
- ggml/src/ggml-cuda/ggml-cuda.cu supp() TOP_K (HIP allowed)
- src/models/qwen4exp.cpp: build_qsa_top_k (~525), llm_graph_input_ple
  set_input (~1043), prepare_batch (~1305)
- src/llama-model.h / llama-context.cpp: prepare_batch hook
- src/llama-kv-cache.cpp:1829 get_prev_tokens (+ index from PR #27992)
- PRs: #27992 (prev-token index), #27941 (qwen4exp fixes), #27977
- upstream top-k.cu = source of the radix backport (master)

## UPDATE 4: PLE batch cache ARCHIVED and dropped (branch qwen4exp)

- Decision: drop. Verified neutral at all depths (the ngram gather was never
  the bottleneck; the QSA top-k -> GPU fix was). Kept the tree lean.
- Forward patch archived: patches/0007-ple-batch-cache-archive.patch
  (applies cleanly to the PLE-free tree; round-trip verified).
- The managed-ngrams lazy_reader path (patches 0001-0007) is RETAINED.
- Work now lives on branch `qwen4exp` in ~/llama.cpp (checkpoint 23f006087,
  PLE-free d2548f9af). Build + pp16384 sanity re-verified: 1737 t/s.
- Original question "when would the PLE cache become useful": only when the
  gather itself is slow (bigger table, slower storage, or a gather-bound
  config). It does NOT help with depth - the depth-scaled costs (QSA top-k
  [now GPU], KQ mask build, attention) dominate at 200K, not the ngram table.

## UPDATE 5: prefill tuning - where we are, what remains (analysis)

**State (PLE-free, GPU radix top-k, branch qwen4exp @ d2548f9af):**
pp512 1492 | pp2048 2028 | pp4096 2031 | pp8192 1935 | pp16384 1716 |
pp32768 1381 | pp65536 995 | tg128 40.1. CPU burn at 16K = 140% (was 1260%).

**What was already captured (all the big CPU-side wins):**
- QSA top-k -> GPU radix (0.5-0.7 ms/node, was the 16-core burn)
- ub2048 amortization, prefill CUDA-graph revert, WMMA FA off, MoE fusion,
  managed-ngrams, PR #27992 (O(log n) prev tokens)
- Verified: GET_ROWS (ngram gather) is FLAT with depth (2.8ms both 16K/64K)

**The remaining depth-scaler: FLASH_ATTN over the FULL KV cache.**
build_attn_qsa (qwen4exp.cpp:668) builds a full [n_kv, n_batch] KQ mask
(-INFINITY, then unmasks the 2048 top-k rows via set_rows) and runs
ggml_flash_attn_ext(q, k, v, kq_mask) over ALL n_kv cells. The top-k
selection only zeroes scores - the kernel still touches every cell.
OP_TIMING: FLASH_ATTN_EXT 35.7ms @ 16K -> 158ms @ 64K (4.4x per 4x ctx,
linear in n_kv). At 64K that is ~92% of per-ubatch time; at 16K ~36%.
The mask FILL/SET_ROWS ops themselves are small (fused, <1.5ms).

**Anomaly to re-check**: MUL_MAT node_56 [2560x15x2048] shows 625ms flat
across depths with OP_TIMING (95% GPU) - a 78 MFLOP matmul cannot take
625ms; likely a graph-split/allreduce barrier sync point counted inside
the first op's window. Not a depth scaler, but worth a look later.

**The 200K-enabling fix (next big prefill project): sparse attention.**
QSA already computes top-2048 of n_kv per token - but the attention step
ignores that and reads all n_kv. A sparse path that gathers only the
top-k K/V cells (per-token gather, then flash-attend over 2048 cells
instead of n_kv) would:
- 16K: 35.7ms -> ~4-5ms per layer
- 64K: 158ms -> ~5ms (attention goes from 92% to ~5% of ubatch time)
- 200K: attention stays constant ~5ms instead of 12.5x the 16K cost
This is THE change that makes 200K prefill workable (attention is the
only remaining linear-in-n_kv op besides the top-k itself, which is
already GPU).
ggml has NO 2D/per-token gather op (get_rows is single-index-per-column)
and no sparse flash-attn - a custom op/kernel is required (user is
willing to roll own kernels on HIP).

Estimated impact at 64K if attention drops to ~5ms/layer: ubatch time
~2.06s -> ~0.3s (995 -> ~5000+ t/s) - but the node_56 barrier anomaly
and remaining fixed costs must be re-measured before trusting that.

## UPDATE 6: ACTION PLAN - two items to attend to (next session)

Model support is ~1 week old - plenty of green-field optimization left.
Two concrete work items, in priority order:

### ITEM A (quick, do first): node_56 625ms barrier anomaly
- Symptom: OP_TIMING shows MUL_MAT node_56 [2560x15x2048] at ~625ms, 95%
  GPU, FLAT at 16K/32K/64K (also node_570 [2560x128x2048] ~180ms flat).
  A 78 MFLOP matmul should be ~0.1ms. Suspect: graph-split/allreduce
  barrier sync counted in the first op's window, or a per-split sync
  serialization across the 3 GPUs (hybrid all-reduce, block 12).
- How to check: GGML_SCHED_DEBUG=2 node->backend assignment + split
  boundaries; compare GGML_CUDA_ALLREDUCE modes (1/2/3); OP_TIMING on
  1-GPU vs 3-GPU to see if the 625ms is a cross-GPU sync artifact.
- If it is sync overhead: reduce graph splits (fewer sync points), or
  check the hybrid allreduce launch pattern for per-split stalls.
- Impact: if real, ~625ms/ubatch at 16K = ~5% -> could be 5-15% total.

### ITEM B (the 200K enabler, main project): sparse attention
- build_attn_qsa (qwen4exp.cpp:668) masks scores but flash-attends over
  ALL n_kv cells. FLASH_ATTN_EXT: 35.7ms@16K -> 158ms@64K (linear in
  n_kv); ~36% of ubatch at 16K, ~92% at 64K. This is the ONLY remaining
  linear-in-n_kv op (top-k is already GPU radix).
- Goal: gather only the top-k K/V cells per token, flash-attend over
  ~2048 cells instead of n_kv. Keeps attention ~constant at any depth.
- Design options (to evaluate):
  1. Custom ggml op: sparse flash-attn taking q, top-k indices, k/v
     cache -> per-token gather of K/V, compute over gathered only.
     Fattn infrastructure exists: ggml/src/ggml-cuda/fattn-*.cuh
     (fattn-tile.cu is the tile-based FA). Would be a HIP/CUDA kernel
     addition, new GGML_OP_* or a fused-node variant.
  2. Graph-level: ggml_get_rows for a per-token gather needs a NEW 2D
     gather op (get_rows is single-index-per-column; top-k differs per
     token). Then dense flash-attn over [n_top_k, n_tokens] with a
     trivial all-ones mask. Simpler kernel but two passes + the gather
     memory traffic.
  3. Fused indexer+attn: keep top-k indices from build_qsa_top_k and
     feed directly into a sparse FA kernel that only loads selected KV
     rows (no gather materialization). Most efficient, most work.
- Correctness bar: bit-exact vs the mask path (the mask semantics:
  -INFINITY except top-k rows = 0, plus the base kq_mask ADD). Must
  preserve causality (positions > query masked) and the base mask.
- Measure before/after: pp16384, pp65536, pp131072 if VRAM allows.
- NOTE: verify the node_56 anomaly first (ITEM A) - if there's a hidden
  fixed per-ubatch sync cost, it caps the sparse-attn win at 64K.

### Model context (for both items)
- qwen4exp = Qwen3.8-Flash-Next UD-Q4_K_XL, 48 layers, 12 full-attn
  layers (3,7,...,47) + 36 GDN/recurrent. QSA indexer: head_count 4,
  key_length 128, top_k 2048, compress_ratios [0,0,0,4,...].
- n_ctx_train 262144; 200K target is native-range but attention scales
  linearly -> sparse attention is REQUIRED for 200K prefill.
- Best measured: pp16384 1716, pp65536 995 (ub2048, tensor, t16).

## UPDATE 7: ITEM A RESOLVED - the slow-node anomaly is one-time hipBLAS init

### Symptom recap
- node_56 (blk.0.ssm_alpha [2560x15] F32 -> hipBLAS): 630ms FIRST eval only
- node_570 (blk.3.indexer.k_proj [2560x128] BF16 -> hipBLAS): 180ms, looked
  like every eval, 92% of one graph split

### Root cause (GEMMTIMING instrumentation on the cublasGemmEx call)
- k_proj BF16 GEMM: first call per DEVICE = 174-189ms (3 GPUs = 3 slow
  prints), every subsequent identical call = 0.02ms. This is one-time
  hipBLAS/rocBLAS BF16-kernel setup/JIT for the [2560x128]x[2560xN] shape.
- ssm_alpha F32 GEMM: 630ms on first process eval only (same one-time cost).
- MMDISPATCH confirmed: indexer k_proj (type 30 = BF16) and ssm_alpha (F32)
  go through ggml_cuda_mul_mat_cublas fallback. Never MMQ (should_use_mmq
  requires quantized src0; BF16/F32 are excluded).
- ALLREDUCE mode (hybrid/internal/none) does not change the anomaly.
- NOT disk: weights are resident BF16/F32, tiny, and the cost never repeats
  after the first call per device.

### Why the op-timing made it look per-eval
The event bracket in GGML_CUDA_OP_TIMING wraps the whole compute_forward
including (a) the 3 GPUs' first-call init and (b) waiting for hc_mixed
(dependency). At pp256 the bracket (184ms) coincidentally matched the eval
wall time (1031 t/s = 248ms/eval), reinforcing the false "real cost" read.

### Steady-state truth
k_proj: 0.02ms x 12 indexer layers = negligible. ssm_alpha: F32 0.02ms x 48.
The BF16 indexer is NOT a depth problem. FLASH_ATTN over full KV remains
the only linear-in-n_kv cost (35.7ms@16K -> 158ms@64K, ~92% at 64K).

### Takeaway for ITEM B (sparse attention project)
The indexer side is cheap and correct (BF16, GPU top-k 0.5-0.7ms). The
sparse-FA kernel only needs to fix the dense attention pass over n_kv.
Instrumentation (GEMMTIMING/MMDISPATCH/MMQTIMING/SLOWNODE) was removed;
tree clean at d2548f9af on branch qwen4exp.

## UPDATE 8 (pre-ITEM-B COMPACTION HANDOFF): where we stand

### Git state (both repos clean, safe to resume after compaction)
- ~/llama.cpp branch `qwen4exp` @ d2548f9af (clean tree):
  - 23f006087 = full working-tree snapshot checkpoint (all session work)
  - d2548f9af = PLE cache dropped, instrumentation removed
  - Contains: block-11 revert, ngram patches 0001-0007, MoE fusion,
    PR #27992 prev-token index, HIP radix top-k backport, LLAMA_UBATCH_TIMING
- boosts repo wip/qwen35moe-prefill/ @ f003b6f:
  - handoff doc (this file) with full history
  - patches/: 0005-kv-prev-tokens-index.patch, 0006-qsa-topk-radix-gpu.patch,
    0007-ple-batch-cache-archive.patch
  - plan-qwen4exp.md, plan-decode.md, plan-fused-moe.md

### Measured state (best config, ub2048, tensor, t16, WMMA off)
pp512 1492 | pp2048 2028 | pp4096 2031 | pp8192 1935 | pp16384 1716 |
pp32768 1381 | pp65536 995 | tg128 40.1. CPU burn 16K: 140%.
Everything committed; no uncommitted work anywhere.

### ITEM B (next): sparse attention for 200K - the full design context
- The only linear-in-n_kv cost left: FLASH_ATTN_EXT over the ENTIRE KV
  cache in build_attn_qsa (qwen4exp.cpp:668-740). Top-k (2048) only masks
  scores; the FA kernel still touches all n_kv cells.
  35.7ms@16K -> 158ms@64K (4.4x per 4x ctx); ~36% of ubatch at 16K, ~92%
  at 64K. At 200K it would be 12.5x the 16K cost -> prefill unusable.
- The indexer side is cheap and correct: GPU radix top-k 0.5-0.7ms/node,
  BF16 k_proj 0.02ms steady (one-time hipBLAS init only). ITEM A closed.
- Goal: per-token gather of the top-k K/V cells, attend over ~2048 cells
  instead of n_kv. Keeps attention constant at any depth.
- ggml has NO 2D/per-token gather op and NO sparse flash-attn. fattn infra:
  ggml/src/ggml-cuda/fattn-*.cuh (fattn-tile.cu is the tile FA). Design
  options (from UPDATE 6): (1) custom sparse-FA kernel taking top-k idx,
  (2) new 2D gather op + dense FA over [n_top_k, n_tokens], (3) fused
  indexer+attn. Correctness bar: bit-exact vs the mask path (semantics:
  -INFINITY except top-k rows=0, plus base kq_mask ADD; preserve causality).
- The model: qwen4exp = Qwen3.8-Flash-Next UD-Q4_K_XL, 48 layers, 12
  full-attn layers (3,7,...,47), 36 GDN/recurrent. n_ctx_train 262144.
  QSA hparams: head_count 4, key_length 128, top_k 2048, ratios [0,0,0,4,..].

### How to resume (after compaction)
1. Re-read this file (UPDATE 6 = action plan, UPDATE 7 = ITEM A closure).
2. Re-apply runtime PM fix if rebooted: echo on | sudo tee
   /sys/bus/pci/devices/0000:{03,06,09}:00.0/power/control
3. Start ITEM B: sparse attention. Measure pp16384/65536 before/after;
   verify bit-exactness vs the mask path at 16K (logitdump A/B works).

## UPDATE 9: ITEM B DESIGN - fused sparse flash-attention op

### Multi-GPU split (verified via GGML_META_DEBUG=1 at pp256/ub256)
- Q [256, n_tps, 24, n_stream] split by q-head: 12/12/0 (GPU2: 0 heads)
- K/V [256, n_kv, 2, n_stream] split by kv-head: 1/1/0 (each active GPU has 1
  kv-head with the FULL n_kv positions; no cross-GPU softmax combine)
- top_k, mask: MIRRORED (full copy per GPU; indices valid everywhere)
- FLASH_ATTN_EXT result split by head: 12/12/0
- KV cache type: F16 (default cache_type_k/v); prefill uses the TILE kernel
  (WMMA off, D=256, no MFMA on RDNA4 -> vec needs Q->ne[1]==1)

### Cost model (per layer, both active GPUs, ub2048)
- dense: 412 GFLOP@16K / 1649 GFLOP@64K -> 35.7ms / 158ms (~5-11 TFLOP/s eff)
- sparse (n_top_k=2051): 52 GFLOP CONSTANT at any depth -> est 5-8ms/layer
- 64K win: 158ms -> ~6ms/layer (26x); attention becomes depth-constant

### Design: new op GGML_OP_FLASH_ATTN_QSA (fused, no materialized gather)
- src0 q   [256, n_tps, 24, n_stream] F32 (FA permuted layout)
- src1 k   [256, n_kv, 2, n_stream]   F16 cache view
- src2 v   [256, n_kv, 2, n_stream]   F16 cache view
- src3 idx [n_top_k, n_tps, 1, n_stream] I32 (top_k from build_qsa_top_k)
- src4 mask [n_kv, n_tps, 1, n_stream] F16 (base kq_mask, gathered in-kernel)
- params: scale (+ softcap if model uses it)
- dst [256, 24, n_tps, n_stream] (same as FA result)
- kernel: vec-style D=256; per (stream, head, token block): for i in 0..n_top_k:
  cell=idx[i,j,s]; score=dot(Q[j],K[cell])*scale+mask[cell,j,s]; online softmax;
  VKQ += V[cell]*exp(score-max). Reads only 2051 cells/token.
- split handler: mirror handle_flash_attn_ext (q axis2, kv axis2, idx/mask
  mirrored, result axis1)
- correctness: exp(-inf)=0 so gathered+mask == mask path mathematically; FP
  accumulation order differs (top_k order vs cell order) -> expect logitdump
  max-diff ~1e-5 relative, same class as vec-vs-tile FA diff. VERIFY + report.

### Files to touch
- ggml/include/ggml.h: op enum + constructor decl
- ggml/src/ggml.c: constructor + name/desc/params tables
- ggml/src/ggml-cuda/fattn-qsa.cuh: kernel (vec-style)
- ggml/src/ggml-cuda/ggml-cuda.cu: dispatch + supported + alloc size
- ggml/src/ggml-backend-meta.cpp: split handler
- src/models/qwen4exp.cpp: build_attn_qsa uses new op (env toggle A/B)
- verify: logitdump A/B sparse vs mask; bench pp16384/65536
