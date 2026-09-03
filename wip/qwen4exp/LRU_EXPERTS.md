# LRU / tiered expert weights for qwen4exp (design + implementation plan)

Status: exploration + design (2026-09-03). No code written yet.
Target: run Qwen3.8-Flash-Next with Q8_0 expert weights on the 3x R9700
(97.8 GiB VRAM) by keeping only the frequently-used experts resident on the
GPUs and the rest host-mapped, loading them on demand.

Reference: https://github.com/ggml-org/llama.cpp/pull/26563 ("expert
caching", miltos22) - never merged. This doc reviews it, maps its lessons
onto the rdna-boosts tree (Meta-TP tensor split, lazy reader, fused ops),
and plots a correct implementation.

---

## 1. The PR (26563): what it did

Architecture (all in llama.cpp stock, CUDA single-GPU):
- `llama_expert_heatmap`: decay-tracked per-(layer, expert) usage counters
  fed from the router's selected-expert tensors (GPU readback via
  ggml_set_output + synchronize). `decay_rate` 0.99 per token, top-S ranking.
- `llama_expert_hotstore`: per-layer GPU buffers holding the S hottest
  experts' weights as a compact tensor `dst_hot` = [ne0, ne1, hot_s + 1]
  (the +1 = a zero-filled sentinel slot). Per-layer LUTs: `hot_lut` =
  i32[n_experts] mapping expert id -> slot (or sentinel if cold) and
  `cold_mask` = f32[n_experts] (1.0 = cold). `copy_top_s` after the first
  ubatch, then `resync_top_s` on a token cadence, with hysteresis + dwell
  to stop slot thrashing, and multi-slot freezing.
- `llama_expert_tier`: a registry + drop-in hook at the top of
  `build_lora_mm_id` (the shared MoE builder). When a weight tensor `w` is
  registered, every expert GEMM becomes a DUAL path:
    - hot: real ids remapped through `hot_lut` -> slot ids (cold experts
      map to the sentinel slot = zero contribution), `ggml_mul_mat_id`
      against `dst_hot` on the GPU;
    - cold: a new op `ggml_mul_mat_id_cold` (CPU) computes ONLY the cold
      experts against the FULL tensor (which stays CPU/mmap resident);
    - result = hot + cold (per-expert scales applied to both).
- CLI: `--expert-hot-s N` (-ehs), `-1` = autofit from free VRAM,
  `--ecf/--expert-cache-force` for non-CUDA.

Author's measured claim: ~1.7-2.1x steady-state decode on
Qwen3.6-35B-A3B Q2_M/Q5_K_P on 8 GB VRAM (56 vs 33, 36 vs 17 tok/s).

## 2. Why it never merged (the issues)

From the PR body + comments + code review of the diff:
1. DECODE-ONLY: the tier bails at `cur->ne[2] > 1` (BUG-3E3 comment: the
   CUDA MMQ mul_mat_id compaction assumes distinct expert ids per token and
   the sentinel duplicates break it at prefill). Prefill fell back to the
   stock path - which needs the full tensor on the GPU... or ran the CPU.
   The author's perf comments ("Real performance upgrade coming soon",
   "+10-20% flat out generation") show the prefill/perf story was unsettled.
2. NO MULTI-GPU: "Not intended for multiple GPUs." The whole feature = one
   CUDA device. Ours = 3x R9700 tensor-split behind the Meta-TP wrapper.
3. NUMERICS: "Due to differences in how experts are processed on the CPU
   vs the GPU... when different experts are cached, the output may slightly
   vary." The hot/cold split is NOT bit-stable across resyncs. (In stock
   llama.cpp this was waved at with the --n-cpu-moe precedent.)
4. MODEL FRAGILITY: verified only on separate gate/up tensors (Qwen-style);
   fused `gate_up_exps` models reported broken (incoherent output). Also a
   Windows atomics build break, an OOM vs the fit target, and a global
   registry in `build_lora_mm_id` that touches every MoE model.
5. LIVE HEATMAP COMPLEXITY: the reviewer (iiLaurens) suggested offline
   calibration instead - live tracking "touches a lot of code and adds a
   lot of complexity, and it might not even add that much more performance
   over offline calibration."
6. LARGE-PR POLICY: 1405 insertions across ggml core (new op + op-count
   bump + ggml-cpu file), llama core, common, tools. The bot flagged it.

The author self-closed it ("plans to organize and re-open").

## 3. What transfers to our goal (and what does not)

The user goal: qwen4exp with Q8_0 weights (~142 GiB, see below) on
97.8 GiB VRAM. The concept transfers: per-layer hot expert cache + host
residency for the rest. What does NOT transfer as-is:
- the single-device assumption (we have the Meta-TP 3-device split);
- the stock CUDA mmid + MMQ path (we have the mmvq decode path + the fused
  gate/up + the q8_1 prequant cache);
- the "cold = CPU mmid in the middle of the graph" mechanics (we need to
  think about how a CPU node behaves in the Meta wrapper);
- bit-exactness: we lose the decode reference anyway (the full Q8_0 model
  cannot run on the GPUs, so there is no all-GPU reference to be exact
  against). Correctness bar = coherent output + determinism at a FIXED
  cache state (see 9).

## 4. Model facts (measured from the GGUF)

Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf (the Q4_K_XL today):
- total = 103.69 GiB over 4 shards; 48 layers; 512 experts x 640 hidden,
  top-10; routed experts are the bulk.
- blk.0.ffn_down_exps.weight = [640, 2560, 512] Q5_1 = 600 MiB/layer.
- blk.0.ffn_gate_up_exps.weight = [1280, 2560, 512] (fused gate+up) Q4_K
  = ~943 MiB/layer (computed from dims).
- per-layer routed experts ~1.54 GiB x 48 = ~74 GiB = ~71% of the file.
- the non-expert remainder (attn qkv/output Q8_0, GDN, norms, shared
  experts, indexer) ~30 GiB.

Q8_0 projection (target): experts 2.34 GiB/layer x 48 = ~112 GiB +
~30 GiB non-expert = ~142 GiB total. Over 97.8 GiB by ~44 GiB.
- one (layer, expert) Q8_0 = gate_up 3.28 MB + down 1.64 MB ~= 4.9 MB.
- a hot store of S experts/layer: S x 4.9 MB x 48 layers. S=32 ->
  7.5 GiB; S=64 -> 15 GiB.

## 5. The fundamental constraint (why naive LRU streaming is hard)

The router runs INSIDE the decode graph (logits mm -> top-k -> ids feed the
mmid in the SAME graph). The set of experts needed for step N+1 is not
known until step N+1's own forward pass. Therefore you cannot, between
steps, "load exactly the experts the next step will need". The PR solves
this by keeping the FULL tensor always available on the CPU and computing
cold experts there - the graph is self-contained, the hot set is just a
cache that makes most of the work fast.

A pure GPU-streaming variant (swap cold -> free slot via H2D before the
step) requires predicting the next step's routing. Options:
(a) router-only pre-pass: impossible cheaply - the routers are mid-graph
    (each layer's router consumes that layer's hidden state);
(b) predictive loading from locality: consecutive tokens in these MoE
    models share a large fraction of their expert sets, so the LRU-resident
    set from the last K tokens covers most of the next step. Misses then
    stream. This only works if the miss rate is low AND the streaming
    (H2D ~4.9 MB per expert at ~50 GB/s = ~0.1 ms) can be scheduled before
    the layer that needs it - which again requires knowing the routing
    before the graph.

Conclusion: the architecturally sound base = the PR's dual path (hot GPU
slots + always-available host tensor). The host path does not have to be
CPU COMPUTE though - see options in 7.

## 6. rdna-boosts specifics that shape the design

- META-TP TENSOR SPLIT: every expert tensor is K-split across the 3
  devices (e.g. {213,213,214}-style rows of the K dim; the mmid decode
  = per-device mmvq over the K-segment + allreduce of the partials). So "an
  expert" lives on ALL THREE devices (one K-segment each). A hot slot must
  exist on every device; the per-device stores are symmetric.
- THE DECODE GRAPH IS SHAPE-KEYED AND REPLAYED (CUDA graphs): anything
  that changes the graph's tensor shapes per step breaks replay (the
  top-k budget churn at sub-2051 KV already forces periodic re-captures;
  the current tree caps at 2051 cells so decode = stable). A tiered design
  MUST keep all shapes fixed per step: fixed S slots, fixed-shape LUTs,
  and only change buffer CONTENT (H2D before the launch). That is exactly
  the PR's structure and it is replay-compatible.
- MIRRORED vs PARTIAL: routing/heatmap tensors are mirrored (replicated on
  all devices); mmid outputs are partial (allreduced). The LUT and the
  hot-slot content must be kept in sync on all 3 devices.
- THE FUSED GATE_UP TENSOR: qwen4exp uses a single fused [1280, 2560, 512]
  gate_up tensor (the PR broke on these; the dual-path must handle the
  2x-K slice layout - which is fine, it is just a bigger K per slot).
- MMVQ DECODE PATH + Q8_1 CACHE: the decode mmid is mmvq with the y-input
  q8_1 prequantized into a per-graph cache. The hot slots' y = the cache
  only if the mmid still sees the same tensor identities (see 8.3).
- THE LAZY READER (this tree's own): big tensors already load as
  mmap + CPU_Mapped with on-demand pread. The "host resident" expert
  tensors should reuse this machinery rather than a fresh one.
- CPU BACKEND IN THE SCHED: the context sched = [Meta, CPU]. The graph can
  split ops to the CPU backend today (the qwen35-era n-cpu-moe precedent).

## 7. Architecture options

### Option A - PR port (hot GPU slots + cold computed on CPU)
Per expert GEMM at decode: remap ids via LUT, mmid against dst_hot (GPU,
slots incl. sentinel) + the cold experts against the full CPU tensor via a
CPU mmid/mul_mat_id_cold, sum. The CPU work overlaps the GPU (separate
sched split) and is free while it fits under the GPU's ~20 ms/step.
CPU budget at decode: cold experts/layer x 48 layers x ~3.3 M MACs each.
Even 5-10% cold = 24-48 cold GEMMs/token ~= 0.16-0.32 GFLOP -> a few ms on
the 9950X3D with good threading. Needs the tree's CPU mul_mat_id at Q8_0
M=1 throughput (measure! the qwen35-era CPU MoE data point exists).
Risk: the graph splits per layer (CPU <-> GPU data crossings of [2560]
vectors + one allreduce/sum per layer) - the PR's "cross contamination"
issue; the Meta wrapper must let a CPU-backend node see its GPU-produced
inputs (the scheduler handles this today for the -ngl / n-cpu-moe split).
Numerics: hot+cold sums in f32; stable per fixed cache state.

### Option B - host-streaming into GPU slots (pure GPU compute)
Cold experts streamed H2D into slots instead of CPU compute. Requires the
next step's routing ahead of time (section 5) - not available. Sub-variants
(b1) two-pass graph per token (router pass then full pass) - routers are
mid-graph, rejected; (b2) predictive prefetch by locality + stall-on-miss:
complex, correctness risk, and the miss handling = a CPU fallback anyway.
Verdict: NOT the base design. The LRU dynamics (which experts are hot) are
orthogonal to WHERE the cold compute happens.

### Option C - hybrid (recommended): A + LRU refresh on cadence
Base = Option A. The "LRU" lives in the heatmap/resync policy (which
experts occupy the S slots), refreshed on a cadence (every K tokens) or on
demand, with hysteresis + dwell. Later, if the CPU cannot keep up, add a
second GPU buffer tier ("swap area") and move the cold path to H2D +
GPU compute WITH the graph split at the router boundary... (research item,
not in v1).

Recommendation: Option C, phased (section 10), with the policy layer
(heatmap) deliberately decoupled so it can start offline-calibrated
(static plant from a profile run) before the live-tracking code exists.

## 8. Component design (rdna-boosts flavored)

### 8.1 Placement & sizing
- Keep everything the decode touches today EXCEPT the routed-expert
  tensors on the GPUs. The routed experts (gate_up + down per layer) move
  to host residency (mmap'd, page-cache resident, exactly like the lazy
  reader does today for the non-resident shards).
- Allocate dst_hot per device: S+1 slots x [K_seg x out] per expert tensor
  (gate_up: K_seg = 1280/3-ish split per device, out 640; down: K_seg =
  640/3-ish, out 2560). The +1 sentinel slot = zero-filled once.
- The load-time tensor layout must change: create the expert tensors in a
  host/mmap buffer (the existing lazy-read CPU_Mapped path) and the
  hot-store tensors as Meta() GPU buffers. The model loader needs a new
  flag (e.g. `--expert-tiers N` + `--expert-slots S`) that (a) routes the
  expert tensors to the host buft, (b) sizes + allocates the per-device
  slot buffers, (c) plants the initial hot set (offline-calibrated ids or
  a static 0..S-1 plant for bring-up).
- The mmid dispatch must be told the expert tensor now has this split
  residency (the mmid decode kernel reads the slot buffer - see 8.3).

### 8.2 The decode graph (dual path)
Per (layer, expert GEMM) at decode, replacing the stock mmid:
1. ids_hot = get_rows(LUT_hot, ids)      [i32 n_eu, fixed shape]
2. hot   = mul_mat_id(dst_hot, x, ids_hot)     [GPU, slots; sentinel=0]
3. cold  = mul_mat_id_cold(w_full, x, ids, cold_mask)   [CPU or GPU]
4. out   = hot * scale(ids) + cold * scale(ids)   (or scale folded)
Follow the PR's lesson list: fixed shapes (no views with weird strides),
mask BEFORE bias/activation, per-expert scale on BOTH paths, sentinel
duplication safe only at n_tokens == 1 for the mmvq path (verify - the
tree's mmvq decode = n_tokens == 1 by construction; prefill stays stock
with the full tensor = which is host-resident now, see 8.4).

The new op = either the PR's GGML_OP_MUL_MAT_ID_COLD or - better - reuse
the existing CPU mmid with the cold-only ids precomputed by a tiny mask op
in the graph. Prefer the latter in v1 (no ggml core op-count change): the
cold ids = get_rows over the selected ids filtered by cold_mask, computed
host-side or by a small graph op. If the per-step cold set is usually tiny
(<2 experts), the mask/filter machinery is cheap.

### 8.3 The mmid/slot subtlety (the one that will bite)
The tree's decode mmid = mmvq with the y (activation) q8_1 prequantized
into a per-graph cache keyed on the input tensor identity, and the weight
tensor read as [K x out x n_expert] with per-expert column indexing. Two
changes:
- The weight now = dst_hot [K x out x S+1] with the ids = slot indices.
  The mmid kernel treats it as a [K x out x n_slots] tensor - no kernel
  change needed if the ids are simply the slot numbers; verify the mmvq
  column walk handles S+1 <= 512 identically.
- The sentinel slot (index S) must be all-zero INCLUDING the dequant
  blocks (a zero q-block still dequantizes to 0 exactly if the scale is
  zero - Q8_0 block of zero d and zero q = 0; verify the mmvq does not
  special-case anything on the id value).
- The activation q8_1 cache: the SAME x feeds both the hot (GPU) and cold
  (CPU) mmids - the CPU mmid path needs its own quantize (the CPU does not
  share the GPU q8_1 cache); keep the x F32 for the CPU path or accept a
  second quantize.
- RESYNC = buffer content swap only: slot p of dst_hot changes from expert
  a to expert b via a H2D copy of b's K-segment rows. Shape unchanged ->
  CUDA-graph replay stays valid. The LUT content (hot_lut/cold_mask) =
  small H2D per resync (or per step if the mask must reflect the slots).

### 8.4 Prefill
The PR punted prefill to the stock path. In our layout the full expert
tensor = host-resident, so the stock path would run prefill experts on the
CPU (n-cpu-moe-style) = slow for a long prompt. Options:
- v1: prefill experts on the CPU (accept the one-time cost; a 2-8k token
  prompt at CPU MoE ~50-150 t/s-ish = seconds) - simplest, correctness
  first. The dense non-expert layers stay on the GPU; only the mmid nodes
  split to the CPU.
- v2: streaming prefill in ubatch-sized blocks, refreshing the hot store
  from the observed router usage mid-prefill (the usage during the prompt
  is a good prior for the generation!) - this is where the live heatmap
  first pays: prime the store from the prompt's own routing.
- measure which; design the heatmap update to run at any n_tokens.

### 8.5 The policy layer (what is "hot")
- v1 (bring-up): STATIC PLANT - slots 0..S-1 (or a JSON/profiled id list).
  Zero policy code. Proves the dual-path graph + the loader.
- v2: OFFLINE CALIBRATION - a profile run (or the prompt prefill) records
  per-layer expert usage; a sidecar file stores the top-S ids; applied at
  load. No live tracking. (The iiLaurens suggestion - right for v2.)
- v3: LIVE LRU - decay-tracked heatmap fed from the router tensors
  (the moe_sel graph outputs; the tree's decode already exposes the
  selected-expert tensors to host readback at the graph boundary - check
  the readback cost; the PR used ggml_set_output + synchronize = a sync
  per cadence, not per step, so it is cheap) + resync on a token cadence
  with hysteresis (only swap when the challenger's score >= hyst x the
  incumbent's) + dwell (minimum age before eviction). Cadence 128-1024
  tokens; each resync = up to S H2D copies of ~2.4-4.9 MB x 48 layers =
  bounded by the cadence budget (e.g. 8 swaps/layer max per resync).
- v4 (optional): prompt-aware priming (use the current prompt's routing
  history to re-rank before a long generation) - the server slot workload.

### 8.6 The Meta-TP wrapper
The hot-store tensors = MIRRORED... no - they are the weight tensors:
the existing expert tensors were axis-0 split (PARTIAL-weight). dst_hot =
per-device buffers holding each device's K-segment of the hot experts =
the same split geometry as today, just S+1 slots instead of 512. The LUTs/
masks = MIRRORED (all devices need the same mapping). The mmid = PARTIAL
output + allreduce as today. The wrapper's splitter must learn: expert
weight tensors with the new residency (host for the full + per-device
slot buffers for the hot) - the same SPLIT_STATE bookkeeping, one extra
case. The CPU-cold nodes: the meta wrapper delegates per-device subgraphs
to the GPU backends; a CPU-bound node (the cold mmid) must be a node the
META hands to the CPU backend of the sched - today the decode graph lives
entirely in the meta; mixing the CPU in = the scheduler splits at the op
level (feasible - the tree already runs the -ngl partial-offload graphs
through the sched with the CPU backend).

## 9. Correctness strategy

- NO bit-exact reference exists for Q8_0-all-GPU (cannot run). The gates:
  (a) determinism at a FIXED cache state (same slots + same content ->
    byte-identical streams across runs);
  (b) the Q4_K_XL-equivalent sanity: run the tiered path against the
    EXISTING Q4_K_XL model (host-resident experts + S=512 slots = full
    residency) and compare to the current all-GPU decode - at S=512 the
    dual path degenerates to the stock mmid on dst_hot and should be
    bit-identical or near (isolates the dual-path plumbing from the
    LRU/host residency);
  (c) coherence gates (Quantum / capital-of-France streams) at the
    candidate cache states;
  (d) quality spot-check: the HTML/car prompt + the reasoning traces.
- The PR's "output may vary when the cache changes" applies: expect token
  drift when the resync moves experts between the GPU and CPU paths
  (different accumulation). Deterministic per cache state, and monotone
  quality as the hit rate rises - the acceptance test = the generated
  content quality, not token equality.

## 10. Phased implementation plan

Phase 0 - MEASURE (the gate for everything):
- Profile the expert usage on the target workloads (the HTML/car prompt,
  the reasoning traces, a coding chat): per-layer entropy of the top-10
  routing, the overlap of consecutive tokens' expert sets, and the
  cumulative coverage vs S (the hit-rate curve). Tools: an env-gated dump
  of the moe_sel ids in the bseq/llama-server harnesses (host readback at
  the graph boundary - the decode already exposes the ids), or a tiny
  offline runner. Deliverable: hit-rate-vs-S curves per layer + the CPU
  cold-GEMM count at the chosen S. This decides S (32? 64?) and whether
  the CPU cold path can stay under the GPU's decode time.
- Measure the CPU mul_mat_id Q8_0 M=1 throughput on the 9950X3D (the
  qwen35-era CPU-MoE numbers may transfer; re-verify at the qwen4exp
  shapes: [640 x 2560] and [1280 x 2560] x M=1).

Phase 1 - LOADER + PLACEMENT (no behavior change):
- A `--expert-...` flag set that (a) places the routed-expert tensors in
  the host/mmap buffer, (b) allocates the per-device dst_hot slot buffers
  (S configurable, default = the number that fits), (c) plants a static
  slot layout (0..S-1) + LUTs. No graph change yet: the decode still runs
  the stock mmid against the full tensor - which now lives on the host =
  the mmid reads it over the PCIe (slow but correct = the "everything is
  host" baseline for later comparison). Gate: runs + coherent at any S.

Phase 2 - DUAL-PATH GRAPH (the core):
- The decode-only dual path (hot mmid on dst_hot + cold on the full
  tensor) with the ids remapped through the LUT. Start with cold = CPU
  (option A). The graph = fixed shapes, replay-compatible. Gate: S=512
  (full residency) == current decode bit-identical; S < 512 = coherent +
  the hit-rate instrumentation matches the Phase-0 prediction.

Phase 3 - POLICY:
- v3 live LRU (decay heatmap + cadence resync + hysteresis/dwell) on top
  of the Phase-2 graph. The resync = H2D content swaps + LUT refresh
  between steps (replay-safe). Gate: determinism at a fixed cache state;
  the resync cost budgeted (< a few % of the token time).

Phase 4 - PREFILL + TUNING:
- The prefill strategy (8.4), the slot count sweep, the cadence/hysteresis
  tuning against the hit-rate curves, and the Q8_0 model bring-up.

Phase 5 - POLISH + PATCH:
- Split into the promotable patch set: (1) loader/placement, (2) dual-path
  graph + mmid slot handling, (3) policy (heatmap/resync), (4) CPU cold
  path perf. Each = its own beta/wip patch with the gates.

## 11. Risks & open questions

- CPU cold-path throughput at low hit rates (the cliff). Mitigation:
  measure first (Phase 0); the hybrid swap-area idea (Option B later) if
  the CPU cannot keep up; accept a hit-rate floor where the feature still
  wins (vs the model not running at all).
- mmvq + sentinel slot correctness at the decode (the PR's BUG-3E3 analog
  lives at n_tokens > 1 - our decode = n_tokens == 1 always; prefill stays
  on the full tensor so the sentinel never sees multi-token mmid).
- The allreduce of the hot+cold sums: hot = GPU partial, cold = CPU result
  entering mid-layer - the Meta wrapper's reduce-step bookkeeping for a
  mixed-backend layer (the scheduler must own the join, not the meta).
- The fused gate_up 2x-K slice per slot (the PR broke on gate_up_exps; our
  slots hold [1280 x 640] slices - bigger H2D per swap ~3.3 MB).
- LUT consistency across 3 devices + the graph-replay content updates.
- Memory: the Q8_0 non-expert (~30 GiB) + the hot store (S=32 ~7.5 GiB) +
  the KV/compute buffers must fit 97.8 GiB with headroom - the fit
  machinery (this tree's fit.cpp) must count the new buffers.
- Whether the host residency hurts the prefill-dense layers (only the mmid
  nodes split; the dense parts stay GPU).

## 12. Suggested first concrete steps (next session)

1. Phase-0 routing-profile harness (env-gated dump of the per-layer
   selected-expert ids at decode; run the HTML/car + reasoning prompts;
   produce the hit-rate-vs-S curves). This is cheap and settles S + the
   CPU budget.
2. Verify the CPU mul_mat_id Q8_0 M=1 decode throughput at the qwen4exp
   shapes (a 20-line probe reusing /tmp/m1_probe).
3. Re-read the mmvq decode column walk to confirm the dst_hot
   [K x out x S+1] + slot-id form needs no kernel change.
4. Then Phase 1 (loader placement) with S=512 full-residency as the
   plumbing gate.
