# gated_delta_net fusion — design analysis (2026-08-28)

Baseline (from `gdn-decode-baseline.md`): GDN costs ~0.50 ms/token (0.376 ms
busy + ~0.17 ms barrier) across 48 dispatches. This note analyzes whether/how
the 48 per-layer GDN calls can be fused to pay the inter-kernel barrier fewer
times.

## Model geometry (Qwen3.8-27B Q6_K)

Read from the GGUF metadata:

| param | value | meaning |
|---|---|---|
| block_count | 65 | 64 model layers + 1 MTP block |
| full_attention_interval | 4 | every 4th layer is full attention |
| attention.head_count | 24 | full-attn heads |
| attention.head_count_kv | 4 | |
| ssm.state_size (S_v) | 128 | GDN state dim |
| ssm.inner_size | 6144 | |
| ssm.group_count (n_k_heads) | 16 | |
| ssm.time_step_rank (n_v_heads) | 48 | |
| ssm.conv_kernel | 4 | |
| nextn_predict_layers | 1 | MTP |

`is_recr_impl[i] = (i < n_layer) && ((i+1) % full_attn_interval != 0)`.
For 64 layers: **48 SSM layers** (blk where `(i+1) % 4 != 0`), 16 full-attn.

## The decode graph: 48 GDN calls per decode pass (confirmed, not 42)

**Phase 0 VERDICT (2026-08-28):** the decode path fires **48**
`gated_delta_net_cuda<128,false,false>` per decode graph, exactly matching
the 48 recurrent layers from the model source. The earlier "42" was an
arithmetic error: a `-n 8` run produced **7 decode-graph passes** (the 8th
generated token comes from the adaptive-MTP draft head, which is a full-attn
block that does NOT emit GDN), and **336 total GDN / 48 recurrent = 7.0**
graphs.

Authoritative source: `src/models/qwen35.cpp` layer classification
```cpp
uint32_t full_attn_interval = 4;
for (i in 0..n_layer_all) is_recr_impl[i] = (i < n_layer()) && ((i+1) % 4 != 0);
// n_layer_all=65, n_layer_nextn=1 -> n_layer()=64
// full-attn at i+1 divisible by 4 -> i=3,7,11,...,63 -> 16 layers
// recurrent = 64 - 16 = 48
```
Empirical: 336 GDN ÷ 7 graphs = 48/graph (verified: GDN busy 0.375-0.419 ms
per graph, median 7.6-7.9 us/call).

### Per-call geometry (uniform across all 48)

- grid `(1536, 4, 32)`, block `(32, 4, 1)` — 196,608 blocks, 128 threads
  each.
- Duration **6.5–17.8 us, median 7.6–7.9 us**; total 0.376 ms busy per
  decode graph (48 calls).
- **Launch/memory-latency bound, not compute-bound**: each block does tiny
  work (S_v=128 columns), the grid is enormous relative to the work.

## The decisive constraint: they are a recurrent chain

The 48 GDN calls are **NOT independent** — delta-net state flows through the
layers. Layer *i*'s GDN writes `ssm_states[i]`, which layer *i+1*'s conv/GDN
reads. This is a sequential dependency; **the calls cannot run concurrently**
and cannot be collapsed into one parallel grid in the naive sense.

The GDN calls are also **interleaved** with each layer's surrounding ops.
Between two consecutive GDN calls there are 3–115 other kernels (norm, qkv
projections, silu, l2_norm, gate, output proj, residual), avg ~24.6. Pattern
example:
```
gdn  -> rms_norm_f32 -> mul_mat_vec_q -> silu -> mul_mat_vec_q -> cpy_scalar
      -> mul_mat_vec_f -> mul_mat_vec_q -> l2_norm_pair -> ... -> rms_norm_q8_1
      -> (next layer) gdn
```

So the decode graph is: `[SSM layer 0: conv + projections + GDN + norm + FFN]`
then `[SSM layer 1: ... GDN ...]`, etc. The GDN is one kernel *within* each
layer's pipeline, not a contiguous run. **48 such layers per decode pass.**

## What fusion is actually possible

Three candidate strategies, in order of tractability:

### A. Fuse the GDN with its layer's immediate neighbors (marginal, risky)
Collapse the small kernels adjacent to the GDN (norm, silu, l2_norm) into the
GDN kernel. This saves a few launches per layer but the GDN's dominant cost
is the barrier and memory latency, and these neighbors are already folded
(quantize_q8_1 etc.). Low upside, high correctness risk (bit-exactness).

### B. Persistent/whole-layer mega-kernel (the real target, large)
Merge the ENTIRE SSM-layer recurrence (conv → projections → GDN → norm →
output proj → residual) for ALL 48 layers into ONE long-running kernel with a
device-side loop over layers and internal state carry. This is the "Path B
persistent mega-kernel" from `fused-matmuls.md` (~4.5 ms headroom). It:
- eliminates ~48 GDN barriers AND the ~17 surrounding kernels each,
- keeps all recurrent state on-chip across the layer chain,
- requires reimplementing the SSM layer math as a fused kernel (like the
  existing `build_delta_net_fused`/chunked work, but across layers).
This is the highest-ROI option but a large, invasive project. The prefill
side already has a fused chunked GDN kernel (`gated_delta_net_chunked*.cu`)
as precedent — the decode side has no cross-layer fusion.

### C. Process the state update for all heads in one kernel (already done)
The S_v=128 and all heads are already handled in a single GDN launch; the 48
are per-layer, not per-head. No head-level fusion to harvest.

## Recommendation

Don't pursue A (the surrounding kernels are already folded; the GDN is
launch/memory-bound, not neighbor-bound, so fusing neighbors buys little).

**The GDN barrier cost cannot be removed by fusing the GDN calls together** —
they are a sequential recurrent chain. The only way to eliminate the ~48
inter-kernel barriers is **B: a cross-layer persistent/fused SSM kernel**.
That's a large project (~"Path B" scale), but it is the single remaining
lever that attacks the device-side gap the env vars and ROCm 10 could not.

There are two remaining items before committing to B:
1. **[RESOLVED] 48 layers, not 42** — confirmed in **Phase 0**
   (`~/llama.cpp/phase0-gdn-layer-count.md`): the decode graph fires 48 GDN
   per pass (336 total over 7 decode graphs). The fused kernel must loop **48**
   recurrent layers. The MTP block (layer 64) is a full-attn draft head and is
   NOT in this loop.
2. **FP16 accumulation / bit-exactness** — does the fused cross-layer kernel
   change accumulation order (the fork convention is bit-exact PPL)? The
   prefill chunked kernel already accepted a near-lossless bf16 path
   (PPL +0.056%); the decode path must decide bit-exact vs near-lossless.

---

**User decision (2026-08-28):** commit to Strategy B (the cross-layer
persistent SSM kernel) as the primary effort, with A as a documented
fallback. The full engineering plan is in
**[gdn-strategy-B-plan.md](gdn-strategy-B-plan.md)** (phases, state layout,
risks, effort ~12-17 days, realistic 1.5-2.5 ms/token). A (fuse GDN with its
per-layer neighbors, ~2-3 days, 0.3-0.6 ms/token) is documented there as
the fallback. Recommended: do B Phase 1 first (shared prerequisite), then
continue to Phase 2 if tractable, else ship Phase 1 as A.
