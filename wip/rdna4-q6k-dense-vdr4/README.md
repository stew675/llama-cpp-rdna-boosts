# wip/rdna4-q6k-dense-vdr4: dense Q6_K mmvq VDR-4 on RDNA4 (opt-in)

Written with Claude (AI).

**Where it comes from:** this patch follows the dense Q5_K VDR-4 path that r368 added for
[#374](https://github.com/stew675/llama-cpp-rdna-boosts/issues/374). @mjohanning99 reported the missing path and you
implemented it, with `-DGGML_HIP_RDNA4_Q5K_DENSE_VDR4` as the RDNA4 opt-in. When we measured that flag on gfx12, Q6_K
was the next-largest dense type in our model and still took the 16-element chunk. This patch therefore applies the same
recipe to Q6_K:

- one 32-element call;
- one integer dp4a chain per sub-block before its single scale;
- an opt-in flag that matches the Q5_K one.

The Q6_K-specific part is the −32 offset, folded in with the `__builtin_amdgcn_perm` lookup that the existing VDR-2
kernel already uses. The gfx12 numbers and the first version of this patch were posted in
[#374](https://github.com/stew675/llama-cpp-rdna-boosts/issues/374#issuecomment-7222381702).

- **Patch:** one `git am` file made on `v18-919794500-r368`, the tree `scripts/apply-all.sh` makes (tip `af7a57ba8`,
  tree `bc7898a2c`). Once applied, the tree is `208b3cc28`.
- **Size:** `0001-cuda-dense-Q6_K-mmvq-VDR-4-on-RDNA4-opt-in-DGGML_HIP.patch`, +68/−1 lines in `vecdotq.cuh`, `mmvq.cu` and
  `mmvq-vecdot.cuh`.

## The change

- **New kernel:** `vec_dot_q6_K_q8_1_vdr4_dense` takes 32 elements per call.
  - The 4 `ql` ints it reads are 16 consecutive bytes. Their low nibbles are one 16-value sub-block and their high
    nibbles are another, so the two sub-blocks share `bq8_offset`, `scale_offset`, `vh_shift` and the q8_1 int group.
  - Each sub-block's 16 products are summed in one integer dp4a chain before its single scale multiply and float FMA.
    The sum is exact: |dot| ≤ 16·32·127 = 65024, and times |scale| ≤ 128 it stays below 2^24.
  - The −32 offset uses the same one-byte-table `__builtin_amdgcn_perm` lookup as `vec_dot_q6_K_q8_1_impl_mmvq_vdr2`.
- **Host side:** `get_vdr_mmvq_dense()` returns 4 for Q6_K on the RDNA4 parameter table, under the same macro.
- **MoE:** the MoE path (`moe == true`) keeps `vec_dot_q6_K_q8_1_vdr2`.
- **Off by default:** without `-DGGML_HIP_RDNA4_Q6K_DENSE_VDR4`, nothing changes. `GGML_HIP_Q6K_DENSE_VDR1` still takes
  precedence over the new flag. Other architectures are untouched.
- **Greedy purity:** the K partition differs from VDR 2, so greedy text moves once. Widths 1..8 use the same kernel, so
  the verify band stays pure. At `-np 2` the text was byte-identical between the arms.

## Measurements

The setup was one R9700 (gfx1201, 32 GB) on Windows 11 with the ROCm 10.2.0 runtime and the ROCm 10.0 compiler. The
build was r368 plus our 11 carry patches. None of them touches the HIP Q6_K path: the one carry in `vecdotq.cuh` is the
non-HIP `__vsub4` VDR=2 kernel. The arms differed only in their flags:

- **base:** no flags.
- **Q5K:** `-DGGML_HIP_RDNA4_Q5K_DENSE_VDR4`.
- **Q5K+Q6K:** Q5K plus this patch with `-DGGML_HIP_RDNA4_Q6K_DENSE_VDR4`.

### Kernel: `test-backend-ops perf -o MUL_MAT`

The shapes come from Qwen3.8-27B (m×k): 1024×5120, 6144×5120, 10240×5120, 12288×5120, 17408×5120, 5120×6144 and
5120×17408. Each comparison ran in A F F A order. The table shows the mean time change across the shapes, Q5K+Q6K
against Q5K.

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Q6_K | −20 % | −13 % | −22 % | −26 % | −24 % | −32 % | −28 % | −32 % |

- **Controls:** Q4_K and IQ4_XS moved +0 to +1.7 %. The output tensor (vocab 248320) did not move because it is DRAM-bound.
- **Correctness:** `test-backend-ops -o MUL_MAT` passed 163/163 for q5_K and q6_K.

### End to end: Qwen3.8-27B UD-Q5_K_XL

This model is dense with an MTP head and has 174 Q5_K, 184 Q6_K and 130 Q8_0 tensors. The server ran with
`-np 2 --kv-unified`, q8_0 KV, ctx 262144, an MTP draft and HIP graphs on. The runs used A F F A A F F A order,
A = Q5K and F = Q5K+Q6K.

| case | step time F vs A |
|---|---|
| short | −1.1 % |
| 5k | −1.1 % |
| 22k | −0.3 % |
| deep | −0.6 % |
| `-np 2` (two streams) | −2.4 % / −1.3 % (text byte-identical, tg +3 %) |
| 151k prompt (A B C C B A) | base 69.24 ms; Q5K −0.1 %; Q5K+Q6K −0.8 % |

Prefill at 151k was 623–636 t/s in every arm, so neither flag changes it. The end-to-end gain is small (≤ ~1 % with
one stream) because Q6_K is only part of each step.

## Not tested

- RDNA3 (gfx11): the new branch is compiled only with `RDNA4` defined.
- Linux.
- MoE models: the MoE path is unchanged by design.
- Perplexity and KLD: the correctness check was `test-backend-ops` only.

Whether this becomes an optional flag next to the Q5K/Q4K ones, or something else, is your call.
