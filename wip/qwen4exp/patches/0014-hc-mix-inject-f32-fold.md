# 0014 - hc_mix decode: fold the F32 inject product into the fused op

Applies: ggml (builder), ggml-cpu (ref + wdata), ggml-cuda (hc-mix.cu +
supports_op), ggml-backend-meta (MIRRORED), qwen4exp.cpp.

What it does: the inject weights are F32 in the GGUF (NOT Q8_0 - the
original Q8_0 assumption was wrong), so the reference inject MUL_MAT runs
the mul_mat_vec_f kernel. The op now computes inject internally with an
mmvf replica (float2 pairs, per-thread acc += v*u in the tid-stride
order over col2, 256-thread blocks = the dispatch's block_size for K =
10240, warp_reduce then the zero-padded warp-sums butterfly) and the dst
tail carries inject instead of the persisted xn; the combine consumes the
inject view. Decode logitdump A/B byte-identical (the mmvf replica was
exact first try). The hc_combine op's inject nb[1] contiguity assert was
relaxed (the inject view carries the mix dst's row stride; nt == 1 never
uses it).

Result: tg128 ~44.7 - FLAT vs the xn-tail form. The inject was already a
single kernel, so folding it changed the kernel count zero (5 + 1 graph
-> 6 internal); only the host dispatch shrank (noise at -t 16). The real
wall is per-kernel queueing (~8-16us on the saturated 3-GPU decode):
every kernel-count-preserving move is noise; only kernel-count-REDUCING
moves (rms+mul fold, silu-quant merge) or batching (M > 1) win.

Commit bd25e63eb (parent 50ae9d134). Apply chain 0013+0014 verified on
50ae9d134.
