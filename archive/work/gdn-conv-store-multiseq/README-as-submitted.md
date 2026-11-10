# wip/gdn-conv-store-multiseq — the one-kernel GDN conv step for a multi-sequence ubatch

Written by Claude (AI).  One `git am` file, one commit, made on `v16-a691b2345-r244` (canonical tip `443104a13`).  It is the
follow-up r244 names under "Not taken": with `-np 2 --kv-unified`, a verify ubatch that carries two sequences still runs the whole
GDN conv chain.  Text, draft counts and top-5 logprobs are identical to r244 with it on and off.

| file | touches | sha256 |
|---|---|---|
| `0001-gdn-conv-store-the-one-kernel-GDN-conv-step-also-tak.patch` | `ggml/src/ggml-cuda/gdn-conv-store.cu` / `.cuh` (+54 −35) | `eb963047cb7f842fd28575e85523328eafe527fe8df83e206b3fd75e42a8976b` |

## What it takes

With `--kv-unified` and two slots verifying together, the ubatch holds S sequences of T tokens:
- the state `[3, C, S]`;
- x^T `[T, C, S]`, a view of the in-projection's output `[C, T*S]`;
- the window view `[3, Kw, C, S]`;
- the slots `[3, K, C, S]`, the sequences' cells `nb[3]` apart.

r244's store takes S = 1 only, so such a step runs the chain on every GDN layer: `GET_ROWS`, `CONCAT`, `SET_ROWS`, `SSM_CONV`,
`SILU` and the norm pair.

## Change

- 1..8 sequences, one block row a sequence (grid `C/256 × S`).  The arithmetic is the single-sequence step's, per sequence.
- T, C and S are read from the transposed view, not from its root: the root is the 2-D MUL_MAT `[C, T*S]`.  The strides are
  checked, and the root must be contiguous with exactly C·T·S elements.
- With S > 1 the state stays the `GET_ROWS` copy.  The gather is deferred for one sequence only, since one sequence's cache rows
  may be another's source.
- `GGML_CUDA_CONV_DEBUG=1`: the `GDN_CONV_STORE` line gains `S=` and is printed once for each S, so a multi-sequence step shows
  up next to the single-sequence ones.
- Kill switches unchanged: `GGML_CUDA_FUSE_GDN_CONV_STORE=0`, or `GGML_CUDA_DISABLE_CONV_FUSION=1` for the whole conv family.

## Results

Linux, Ryzen 9 3900X, one R9700 (gfx1201), ROCm 10.0.0.  `ROCBLAS_USE_HIPBLASLT=0` (GREEDY-PURITY §44) in every arm.  r244 =
stock `v16-a691b2345-r244`; PR = r244 + this commit.  Every arm is a freshly started server with a few seconds of warm-up.

**Same output.**  Each arm against stock r244: text and draft counts of every request, and the top-5 logprobs of 64 tokens after
2K / 8K / 30K prompts (drafted, so verify steps included).

| config | arms against r244 | requests | differing |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft 5, `-np 2 --kv-unified -c 131072 -ub 2048`, q8_0 K/V | on; `GGML_CUDA_FUSE_GDN_CONV_STORE=0` | 25: tool-calling ERP ×3, an 8K copy, 7 prompt lengths, two prompts in one request, prompt-cache continuation and round trips, 110K, two requests at once, logprobs ×3 | none |
| same, `--spec-type draft-mtp --spec-draft-n-max 3` | on | 7: ERP ×3, the copy, logprobs ×3 | none |
| Qwen3.6-35B-A3B UD-Q4_K_M + its DFlash draft | on | 6: ERP ×3, logprobs ×3 | none |
| dense 27B, no draft | on | logprobs ×3 | none |

- `GGML_CUDA_CONV_DEBUG=1` printed `GDN_CONV_STORE C=10240 T=2 S=1 …` and `… T=4 S=2 …` in the DFlash2 set above and in the
  two-sequence runs below (DFlash2 and MTP), so the multi-sequence step ran.
- VRAM after the run is the same as r244's in every config (DFlash2 25562, MTP 23926, 35B-A3B 24223, no draft 22031 MiB).

**Two sequences a step** (dense 27B XL, `-np 2 --kv-unified -c 131072`, q8_0 K/V; two 2048-token prompts in one request decoding
together, 256 greedy tokens each; ms per step of each sequence, `predicted_ms / (predicted_n − accepted)`, medians; r244 / PR /
PR / r244).  A lower step time is faster.

| draft | r244 | PR | PR vs r244 |
|---|---|---|---|
| DFlash2, 5 | 56.86 ms | 55.96 ms | −1.59 % (faster) |
| MTP, 3 | 55.52 ms | 54.52 ms | −1.82 % (faster) |

- Texts and draft counts are identical in every run (DFlash2 accepted 180 / 212, MTP 167 / 191 for the two sequences).
- On r241 the same change measured −1.25 % (DFlash2) and −1.52 % (MTP), in a session of its own.

## Not validated

- 3+ slots: up to 8 sequences are taken; only 2 were run.
- A multi-sequence prefill ubatch (more than 16 tokens a sequence): the direct conv still declines there (`st->ne[2] == 1`), and
  `ggml_cuda_ssm_conv_in_try_fuse` still does not take the store graph.  This commit does not touch either.
- Several GPUs, `-sm tensor`, `-ncmoe` / host-resident experts.
- Windows, `GGML_CUDA_DISABLE_GRAPHS=1`, RDNA3, other GDN architectures (qwen3next, qwen4exp).
- Speed on 35B-A3B.
