# Strix Halo (gfx1151) — decode (tg) verification at tip 376f02aa0

Status: CLOSED — nothing remains at tg. Same-session A/B at depth-0 = parity; A's decode is
ahead at depth. The old TODO note "tg@0 decode gap (24.2 vs 26.0)" was pre-PLE/shortcut-era
and is now moot.

## Wall measurements (same-session, this session)

| row | A | B | A/B |
|---|---|---|---|
| tg128 @ 0 (r3, -p 128 -n 128) | 25.97 ± 0.00 | 26.00 ± 0.03 | 0.999 (parity) |
| tg128 @ d12288 (r1) | 23.13 | 22.02 | 1.051 |
| tg128 @ d32768 (r1) | 20.94 | 18.56 | 1.128 |

## Per-op kernel mix (fresh same-session rocprof captures, -p 128 -n 128 -r 1)

/tmp/prof/tg-{A,B}.db_results.db. A 5.022s / 266,894 kernels vs B 4.940s / 305,680 kernels:
A's GPU-busy +1.7% but wall parity and ~300 FEWER kernels/step -> A is LESS launch-bound; B
hides more behind launch gaps. Same-name same-count per-call (decode steady-state >=1000):

- A faster: mul_mat_id_iq4_nl_weighted_rdna3_5 0.921 (5547 calls, 250.8 vs 272.3ms),
  gated_delta_net_cuda 0.974 (4644, 48.0 vs 49.2ms), k_get_rows_float_vec 0.945,
  k_get_rows_float 0.785, k_set_rows 0.853, mul_mat_q_routed_compact ~0.97, flash combine 0.929,
  quantize_mmq_q8_1_swiglu ~parity, topk fused 50.3/44.7ms (A's unfused full-sort vs B fused -
  the prefill numerics-fork family, sub-0.2% in decode).
- A slower (all tiny totals): ssm_conv_f32 1.167 (+1.7ms/capture = ~13us/step), rms_norm<256,true>
  1.114 but A fires 2.5x more (61 vs 24.6/step, different fusion boundaries), k_bin_bcast-add
  1.215 on 6x FEWER calls (55 vs 337/step - B does many more tiny adds).

## Structural decode split (entirely different per tree, nets to parity)

- A's hc decode = fused kernels: hc_mix_up_silu_dot 98/step, hc_mix_down_dots 98/step,
  hc_mix_rms_gamma_quant 98/step, hc_mix_collapse_inject 98/step, hc_combine_kernel 97/step;
  quantize_q8_1 244/step (one per standalone mmvq input prep) + mul_mat_vec_q<8,1,*> ~244/step
  (true/false/kksplit variants) + mul_mat_vec_q<21,1,true> 47/step (IQ4_NL router).
- B's hc decode = mul_mat_vec_q_fq<8,{false,true},16,2> 257/step + mul_mat_vec_fq_group 36/step
  (inline-quantize mmvq: no separate quantize_q8_1 for those) + mul_mat_vec_f 218/step +
  hc_combine_norm_f32 96/step + shared_gate_mul_add 48/step + mul_mat_vec_iq3_s_grid 47/step.
- B does ~600 small per-step kernels vs A's ~490; net identical wall.

## Verdict

Decode ledger item CLOSED. Depth-0 tg parity (0.999), at-depth tg ahead (1.05-1.13x). The
candidate deltas (quantize structure, ssm_conv per-call, rms count) are each <0.2%/step and
net zero against B's launch overhead; no lever with expected payoff. If future decode pushes
are wanted they are new work (e.g., porting B's fq-inline-quantize mmvq variant to cut A's
244/step standalone quantize - but that REDUCES A's launch-count advantage and is expected
wash-to-negative per the parity evidence).
