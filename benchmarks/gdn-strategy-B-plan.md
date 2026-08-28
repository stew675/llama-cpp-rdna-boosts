# Strategy B: fused cross-layer SSM (DeltaNet) decode kernel — engineering plan

Goal: eliminate the per-layer inter-kernel barrier + launch tax of the
42 `gated_delta_net_cuda` calls (and their surrounding per-layer kernels) by
running the entire recurrent SSM decode as ONE persisted kernel that loops
over layers internally with on-chip state carry.

Priority: B is the primary effort. A (documented at the end) is the fallback.

## 0. Why this is bounded — read this first

Per-token decode (Qwen3.8-27B Q6_K, bf16 KV, 1 card, from
`gdn-decode-baseline.md`):

| component | ms/token | % of GPU busy | fusable? |
|---|---:|--:|---|
| mmvq weight stream (DRAM-bound at VRAM limit) | 36.39 | 92.4% | **NO — card saturated** |
| non-matmul fusable work (all B can touch) | 3.03 | 7.6% | yes |
| ├ gated_delta_net | 0.36 | | yes |
| ├ norms (rms_norm_q8_1 + rms_norm_f32) | 0.89 | | yes |
| ├ get_rows / set_rows / cpy / conv-state | 0.60 | | yes |
| ├ mul_mat_vec_f (SSM projections, small) | 0.44 | | yes |
| └ l2_norm / unary / quantize | 0.74 | | yes |
| **wall** | **44.98** | | |
| └ inter-kernel launch gap | 5.57 | | partially |

So B's **hard ceiling is ~3.03 ms/token** of GPU-busy, and realistically you
can recover part of the 5.57 ms gap too (the gap is mostly behind the mmvq
stream, but the SSM-region barriers are removable). A realistic target is
**1.5–2.5 ms/token** improvement. Do NOT expect B to make decode "10x faster"
— it cannot touch the 92.4% mmvq wall.

## 1. Model geometry (verified from GGUF)

| param | value | role |
|---|---|---|
| block_count | 65 | 64 trunk layers + 1 MTP |
| full_attention_interval | 4 | every 4th layer is full-attn |
| ssm.state_size (S_v) | 128 | GDN state dim |
| ssm.dt_rank (n_v_heads / num_v_heads) | 48 | value heads |
| ssm.n_group (num_k_heads) | 16 | key heads |
| ssm.inner_size (d_inner) | 6144 | |
| ssm.conv_kernel | 4 | |
| nextn_predict_layers | 1 | MTP (separate graph_mtp) |

`is_recr_impl[i] = (i < n_layer) && ((i+1) % 4 != 0)` → **48 SSM layers**
(blk 0,1,2,4,5,6,8,...), 16 full-attn (blk 3,7,11,...63).

**OPEN QUESTION (resolve first):** the decode kernel trace shows **42** GDN
dispatches/token, not 48 (matches 42 ssm_conv / mul_mat_vec_f / l2_norm_pair).
Determine why 42: likely the MTP block layer is not in the single-token decode
path, and/or the last few trunk layers or the MTP trunk layer are handled
elsewhere. **B's fused kernel must loop the exact runtime count** (42), so
resolve this count before writing the loop. (Trace method: count
`gated_delta_net_cuda<128>` dispatches in a decode phase; already 42.)

## 2. The op sequence B must fuse (one SSM layer)

From `src/models/qwen35.cpp` `build_layer_attn_linear` + `build_ssm` and the
kernel trace between consecutive GDN calls (avg ~24 kernels/layer):

```
per SSM layer il (in sequence):
  input cur (n_embd=5120 x 1 token)
  build_qkvz:       qkvz = qkv_mm(cur)      [weight matmul — DRAM bound, in BUT as small proj]
                    z    = qkvz split
  beta = sigmoid(ssm_beta_mm(cur))           [small proj + sigmoid]
  alpha = ssm_alpha_mm(cur); +ssm_dt; softplus; *ssm_a; gate
  conv_state:       conv_state build (get_rows/concat/cpy) [state shuffle]
                    conv_input = concat(prev_conv, qkv)
  conv = ssm_conv(conv_input, conv1d);       [SSM conv kernel]
  conv_silu = silu(conv)
  split q_conv / k_conv / v_conv
  l2_norm(q_conv), l2_norm(k_conv);          [l2_norm_pair]
  GDN: gdn = gated_delta_net(q,k,v,g,beta,state, K)  [THE op — recurrent]
  state update (gdn writes state_slot)
  z_2d = reshape(z)
  attn_out_norm = build_norm_gated(gdn, z, ssm_norm)  [norm + gate sigmoid]
  final_output = reshape
  out = ssm_out_mm(final_output)             [weight matmul]
  cur = reshape(out) + residual
```

The fused kernel needs to do the SMALL, on-chip parts contiguously and call
into the weight matmuls (mmvq) as a cooperative/subtile step. The heavy mmvq
weights CANNOT be fused (no reuse, DRAM bound) — but the recurrent
state-carry (GDN) and the small norms/projections/conv CAN.

## 3. Recurrent state layout (critical for design)

- Recurrent state is **per-layer 2D tensors**: `cache_s_l{il}` shape
  `[n_embd_s, mem_size * (1 + n_rs_seq)]` where `n_embd_s = 128*128*48 =
  786432`.
- Conv state is per-layer `cache_r_l{il}` shape
  `[n_embd_r, mem_size * (1 + n_rs_seq)]`.
- So the state is NOT one contiguous block across layers — each layer has
  its own tensor with its own `kv_head` slot offset.
- **Design consequence:** a single fused kernel cannot simply pointer-advance
  from layer to layer; it must be indexed by layer (either a per-layer state
  base-address array passed as pointers, or the state stored in one fused
  buffer at graph build time).

## 4. The precedent: fused chunked GDN (prefill)

`gated_delta_net_chunked.cu` (700 lines) + `_bf16` (962 lines) already fuse
the prefill recurrence into ONE kernel (two passes: state-scan + output),
with a `fused_cache` (the state written directly into the cache slot,
skipping the cpy). Key existing infrastructure to reuse:

- `llm_graph_fused_node` / `add_fused_node` / `get_fused_nodes` plumbing
  (`src/llama-graph.{h,cpp}`).
- `ggml_cuda_gated_delta_net_fused_cache` + the `_fused_cache` backend entry
  (`ggml/src/ggml-cuda/ggml-cuda.cu` ~2836, ~3413, ~3420).
- `ggml_cuda_op_gated_delta_net_fused_cache` (fused GDN->cache, skip cpy).
- `LLM_FUSED_OP_GDN_AR` / `LLM_FUSED_OP_GDN_CH` op tags.

The prefill fused kernel already proves the cross-op fusion pattern works in
this codebase. B extends it: instead of fusing the GDN with the state-cpy
(prefill), fuse the GDN with its per-layer norms/projections/conv AND carry
state across the layer loop (decode).

## 5. Implementation phases

### Phase 0 — Resolve the 42-vs-48 count + establish a fused-kernel A/B
- [ ] Confirm the exact recurrent-layer count in the decode path (42).
- [ ] Capture a clean decode baseline already available (`gdn-decode-baseline.md`).
- [ ] Freeze `GGML_CUDA_GDN_*_FUSED` env switches for A/B (reuse the chunked
      kernel's toggle pattern).

### Phase 1 — Single-layer fused kernel (prove the pattern on ONE layer)
Write one kernel that does, for a single SSM layer: conv → silu → split →
l2_norm → GDN → gated norm → output, all on-chip, using shared memory, and
keeps the state carry. This is the risk-reduction step: prove numerics +
performance on one layer before the cross-layer loop.
- [ ] New `.cu` (start ~400 lines), reuse `gated_delta_net_cuda` math.
- [ ] Add a fused-node op tag `LLM_FUSED_OP_GDN_SSM_LAYER` + backend dispatch.
- [ ] Validate: bit-exact or near-lossless (decide — see §6), test-backend-ops.

### Phase 2 — Cross-layer loop (the actual "persistent" kernel)
Extend to a persistent kernel that loops over the 42 SSM layers, carrying the
recurrent state on-chip (or in a modest scratch) and calling the small per-layer
ops inline. The heavy mmvq weight matmuls are done as cooperative subtiles
(they must round-trip to DRAM, but the launch overhead is amortized).
- [ ] Layer-loop in `blockIdx`/grid or a `for` over layers with on-chip state.
- [ ] Handle per-layer state base via fused state buffer or pointer array.
- [ ] Handle the full-attn (non-SSM) layers as a separate path in the graph
      (B only fuses the SSM/recursive portion; full-attn layers stay on the
      existing flash-attn path).

### Phase 3 — Integration + multi-card
- [ ] Wire into `llama_memory_recurrent` so the fused kernel writes all layer
      states in one pass (or the fused state buffer).
- [ ] Multi-card `-sm tensor` split support (the kernel must respect per-layer
      device assignment).
- [ ] Register the op for the HIP backend only (arch-gate, like the fork's
      existing kernels).

### Phase 4 — Tuning + validation
- [ ] Grid/block sizing, shared mem budget, occupancy.
- [ ] test-backend-ops (GDN, MUL_MAT, flash-attn) full suite.
- [ ] Bit-exact / PPL A/B (see §6).
- [ ] Benchmark `tg64 @ d256/d2048/d65536` vs the 42-kernel baseline.

## 6. Numerics decision (make EARLY, it shapes the kernel)

- The fork convention is **bit-identical PPL** for fusions.
- The prefill chunked kernel accepted a **near-lossless bf16/WMMA path**
  (PPL +0.056%, KL 0.0036) as the default for S_v==128, with an fp32
  bit-exact fallback.
- B must decide decode-side:
  - **Conservative (recommended for first cut):** fp32 accumulate, bit-exact
    single-layer first; then measure whether a bf16 tensor-core path is
    warranted (probably yes for the S_v=128 GDN matmuls).
  - The fused kernel must preserve the exact recurrence accumulation order
    across layers or PPL will shift (which the near-lossless precedent shows
    is acceptable to ~0.056% if documented).

## 7. Risks

1. **Recurrent dependency** (not parallelizable) — B ONLY works if the state
   carry is sequential; do not try to parallelize across layers.
2. **State layout** — per-layer tensors complicate a single-kernel loop; plan
   the fused state buffer carefully (this is the biggest design risk).
3. **mmvq integration** — the fused kernel must still call the DRAM-bound
   weight matmuls; getting that right (cooperative subtile) is the crux of
   whether the launch savings materialize.
4. **Bit-exactness** — likely falls to near-lossless (documented).
5. **Multi-card** — per-layer device split means the fused kernel may need to
   be single-device (ssm portion) with mmvq on the split, a real complexity.

## 8. Effort estimate (senior CUDA/HIP engineer, focused)

| phase | effort |
|---|---|
| 0 (verify + baseline) | 0.5 day |
| 1 (single-layer fused) | 2–3 days |
| 2 (cross-layer persistent) | 4–6 days |
| 3 (integration + multi-card) | 3–4 days |
| 4 (tune + validate) | 2–3 days |
| **total** | **~12–17 days (~2.5–3.5 weeks)** |

For a realistic **1.5–2.5 ms/token** decode improvement (the SSM region is
~3.03 ms; the mmvq 36.4 ms wall is untouchable).

---

# Strategy A (fallback): fuse GDN with its per-layer neighbors

Documented fallback if B stalls. Smaller, surgical, captures part of the same
gain.

### What A does
Since the 42 GDN calls are each interleaved with ~6 directly-adjacent kernels
in the SSM layer (norm, l2_norm, silu, quantize, state cpy), fuse ONLY those
immediate neighbors into the GDN kernel per layer — no cross-layer carry, no
persistent loop. This collapses the ~42 GDN barriers AND several adjacent
barriers without touching the mmvq or the recurrent dependency.

### Scope
- ~300–400 lines (extend `gated_delta_net_cuda` to also emit the fused
  norm/l2_norm/silu/quantize for its layer).
- Reuses the SAME fused-node plumbing as B (Phase 1) but stops there — no
  cross-layer loop, no state-carry redesign.
- **Bit-exact is feasible** (no accumulation-order change across layers).

### Expected gain
- Removes the ~42 GDN barriers (0.17 ms) + a fraction of the adjacent kernel
  barriers. Modest — roughly **0.3–0.6 ms/token**. May or may not fully close
  the ~0.45 ms target, but it's low-risk and could combine with B's Phase 1.

### Effort
~2–3 days. Much lower risk than B.

## Decision framing
- **B** = "knock it out of the park": ~12–17 days, 1.5–2.5 ms/token, high
  reward, high risk.
- **A** = safe floor: ~2–3 days, 0.3–0.6 ms/token, low risk.
- Recommended: try B's **Phase 1 first** (the single-layer fused kernel is a
  shared prerequisite for both A and B). If Phase 1 proves numerics/perf and
  the cross-layer loop (Phase 2) looks tractable, continue to B. If Phase 2
  hits the state-layout or mmvq-integration wall, ship the Phase 1 result as
  A and stop there.

## References
- Baseline: `gdn-decode-baseline.md`
- Fusion feasibility + 42-vs-48: `gdn-fusion-analysis.md`
- Precedent kernels: `ggml/src/ggml-cuda/gated_delta_net_chunked*.cu`
- Fused-node plumbing: `src/llama-graph.{h,cpp}`,
  `ggml/src/ggml-cuda/ggml-cuda.cu` (GDN fused-cache entries)
