Re-cut of #260 on r195, after the gfx1201 MTP regression you found (thank you for catching it). The row concat itself is unchanged. What changed is how the ksplit body is shared: the original cut turned `mul_mat_vec_q_ksplit` into an inlined `__device__` body, and that recompiled the multi-column ksplit kernels and cost ~7 % of MTP decode on the R9700. The cause and the measurements are in the PR thread. One patch on r195 (it also applies to r191), RDNA4 only; setting `GGML_CUDA_MMVQ_ROW_CONCAT_MIXED=0` turns it off.

In the K-quant M files `attn_v` is Q6_K in about half the layers while Q / K are Q4_K, Q5_K or IQ4_XS. Examples: 24 of 48 layers in our Qwen3-30B-A3B Q4_K_M (12 per step in your Qwen3-Coder-30B count), 6 of Qwen3.6-35B-A3B's 10 attention layers, all 16 of Llama-3.2-1B UD-Q4_K_XL. In those layers the RDNA4 1-token Q / K / V row concat used to leave V as its own launch. This patch lets the concat take such a pair.

**0001: row concat across a K-quant / Q6_K pair** (`common.cuh`, `mmvq.cu`, new `mmvq-ksplit-body.inc`).
- The body of `mul_mat_vec_q_ksplit` moves verbatim into `mmvq-ksplit-body.inc`. That file is `#include`d into the kernel, which is otherwise unchanged, and into a `__device__ __forceinline__` `mul_mat_vec_q_ksplit_body` that only the new kernel calls. Most of the diff is this move.
- A new `mul_mat_vec_q_ksplit_cat2<type, type_b>` runs the Q6_K body, with that segment's own row stride, for the Q6_K segment's row blocks, and the launch type's body for the rest.
- Both types take the same nwarps on RDNA4, and each row keeps its own one-warp dot product, so every row computes what its standalone launch computes.
- The type is chosen once per block, so the decode grid loop is off for these launches. The matcher takes either graph order and requires the launch tensor to be the largest segment, since the row guards use its row count.
- Pairs {Q4_K, Q5_K, IQ4_XS} x Q6_K. On other architectures the cat2 body is compiled out and the stock path runs.

**Gates**
- Device code, ROCm 10.1, r191 vs the patch: every kernel r191 already has compiles to r191's code on gfx1201 and gfx1100. The only differences are PC-relative relocation constants, also present with a plain duplicated-body control.
- R9700 (ROCm 10.0), your MTP arm (Qwen3.8-27B UD-Q4_K_XL, bf16 KV, `--spec-type draft-mtp`, 1 GPU), five interleaved rounds: r191 72.5 73.8 73.7 73.7 73.7; patch 73.7 x5.
- Greedy text byte-identical to r191 on the R9700: Qwen3-30B-A3B plain decode and the 27B MTP arm, 300 tokens each. The first cut's gates (server replies on five models, decode-path KLD 0) still apply to the concat itself.
- 0 GPU faults.

**R9700, Qwen3-30B-A3B Q4_K_M**, F16 KV, tg128 `-r 20`, r191 / patch ABBA x3: 168.9 -> 170.0 over all six runs (+0.6 %), 170.2 -> 172.2 without the first round, which both builds ran cold (+1.2 %). That's smaller than the +1.5 % in the first cut; your reproduction measured +1.4 %.

Notes:
- Untested here: multi-GPU and `-sm tensor` (your 3-GPU run of the first cut measured +1.2 %).

Files: `0001-*.patch` (`git format-patch` against r195; apply with `git am`) and this README. No rush; happy to run anything else on either card.

(Claude-assisted, as before.)
