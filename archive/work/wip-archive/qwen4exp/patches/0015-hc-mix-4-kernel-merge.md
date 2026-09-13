# 0015 - hc_mix decode: 6 internal kernels per call -> 4 (bit-exact)

On top of 0014 (bd25e63eb -> af3640f67). The op now runs:
  1. rms_gamma_quant: the grouped RMSNorm + gamma (xn) with the xn Q8_1
     quantize as a phase-3 of the same kernel (one 1024-thread block per
     stream; each block quantizes its own 80 q8_1 groups with the
     quantize_q8_1 warp pattern - amax reduction, d = amax/127,
     roundf(x/d), sum - identical to the removed generic
     quantize_row_q8_1_cuda launch).
  2. down dots (Q8_0 x Q8_1, rpb=1, unchanged)
  3. up_silu dots (silu+quant prologue, rpb override, unchanged)
  4. collapse_inject: the stream-collapse blocks (n_embd/256) and the F32
     inject mmvf-replica rows (hc blocks) share one grid, branch by
     blockIdx - each path bit-unchanged.
Gate: decode logitdump byte-identical vs /home/ld_decode_ref.bin.
Perf: tg128 45.57 vs 44.70 (5b) - +2.0%, direction consistent with the
-190 kernel launches/device/token but the -r 3 means overlap at the noise
edge (+/-2.2 t/s). Mechanism-sound + bit-exact, kept.

Decode cost model established this session (micro-probes): M=1 mmvq GEMMs
pay a ~15-20us kernel floor in-chain (chained-graph probe: 17.8us/GEMM
back-to-back for the hc shapes) regardless of weight size (0.74MB router =
same ballpark as 16.6MB dense); decode = ~440 GEMM-class evals +
~1400 small evals per device per token; the wall is the sum of small-kernel
times (latency-bound, NOT queueing - queueing is only ~8us per per-call
graph dispatch; back-to-back kernels run at their true speed).
