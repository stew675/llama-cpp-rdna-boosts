# gated_delta_net fusion — design analysis (2026-08-28)

Baseline (from `gdn-decode-baseline.md`): GDN costs ~0.50 ms/token (0.334 ms
busy + 0.170 ms barrier) across 42 dispatches. This note analyzes whether/how
the 42 per-layer GDN calls can be fused to pay the inter-kernel barrier fewer
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

## The decode graph: 42 GDN calls, NOT 48

Kernel trace shows **42** `gated_delta_net_cuda<128,false,false>` dispatches
per token (not 48), and the surrounding SSM ops (`ssm_conv_f32`,
`mul_mat_vec_f`, `l2_norm_pair`) also fire 42/token. So 42 SSM-layer
sequences execute per decode token (the MTP block and some tail layers are
not in the single-token decode path, or share a reduced count — the 6-layer
gap is not yet fully explained and should be confirmed before assuming a
fixed loop count).

### Per-call geometry (uniform across all 42)

- grid `(1536, 4, 32)`, block `(32, 4, 1)` — 196,608 blocks, 128 threads
  each.
- Duration **6.8–17.8 us, median 7.9 us**; total 0.363 ms across 42.
- **Launch/memory-latency bound, not compute-bound**: each block does tiny
  work (S_v=128 columns), the grid is enormous relative to the work.

## The decisive constraint: they are a recurrent chain

The 42 GDN calls are **NOT independent** — delta-net state flows through the
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
layer's pipeline, not a contiguous run.

## What fusion is actually possible

Three candidate strategies, in order of tractability:

### A. Fuse the GDN with its layer's immediate neighbors (marginal, risky)
Collapse the small kernels adjacent to the GDN (norm, silu, l2_norm) into the
GDN kernel. This saves a few launches per layer but the GDN's dominant cost
is the barrier and memory latency, and these neighbors are already folded
(quantize_q8_1 etc.). Low upside, high correctness risk (bit-exactness).

### B. Persistent/whole-layer mega-kernel (the real target, large)
Merge the ENTIRE SSM-layer recurrence (conv → projections → GDN → norm →
output proj → residual) for ALL 42 layers into ONE long-running kernel with a
device-side loop over layers and internal state carry. This is the "Path B
persistent mega-kernel" from `fused-matmuls.md` (~4.5 ms headroom). It:
- eliminates ~42 GDN barriers AND the ~17 surrounding kernels each,
- keeps all recurrent state on-chip across the layer chain,
- requires reimplementing the SSM layer math as a fused kernel (like the
  existing `build_delta_net_fused`/chunked work, but across layers).
This is the highest-ROI option but a large, invasive project. The prefill
side already has a fused chunked GDN kernel (`gated_delta_net_chunked*.cu`)
as precedent — the decode side has no cross-layer fusion.

### C. Process the state update for all heads in one kernel (already done)
The S_v=128 and all heads are already handled in a single GDN launch; the 42
are per-layer, not per-head. No head-level fusion to harvest.

## Recommendation

**Don't pursue A** (the surrounding kernels are already folded; the GDN is
launch/memory-bound, not neighbor-bound, so fusing neighbors buys little).

**The GDN barrier cost cannot be removed by fusing the GDN calls together** —
they are a sequential recurrent chain. The only way to eliminate the ~42
inter-kernel barriers is **B: a cross-layer persistent/fused SSM kernel**.
That's a large project (~"Path B" scale), but it is the single remaining
lever that attacks the device-side gap the env vars and ROCm 10 could not.

Before committing to B, two things to verify:
1. **Why 42, not 48** — confirm the SSM-layer loop count in the decode path
   so the fused kernel processes the right number of layers.
2. **FP16 accumulation / bit-exactness** — does the fused cross-layer kernel
   change accumulation order (the fork convention is bit-exact PPL)? The
   prefill chunked kernel already accepted a near-lossless bf16 path
   (PPL +0.056%); the decode path must decide bit-exact vs near-lossless.
