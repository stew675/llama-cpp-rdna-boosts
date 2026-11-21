Thank you for taking #335 in r324 and for the two corrections. The point that `-ub 1` KLD cannot see the decode path was new to us, and we now check the text instead. Thanks also for the #333 fix in r322: our production compose no longer needs the split-32 override.

This follows from §44 and §47 of your GREEDY-PURITY.md. Turning hipBLASLt off (`ROCBLAS_USE_HIPBLASLT=0`) gives one output per input, but on a GSQ Flash-Next it costs 13-23 % of the prefill in our runs. The cost comes from the BF16 dense GEMMs falling back to rocBLAS's own Tensile kernels. These patches let MMB take those GEMMs on RDNA4, **only while hipBLASLt is off**, so the pinned mode costs 1-4 % against hipBLASLt instead. With hipBLASLt on, outputs and speed are unchanged. We are sharing it in case it is useful for your determinism runs. Nothing is urgent on our side.

**What it does** (all `mmb.cu`, RDNA4 only)

- **0001** BF16 dense weights take the MMB bf16-WMMA tile from 33 tokens (above the MMVF band) while hipBLASLt is off. The new `mmb_hipblaslt_off()` reads `ROCBLAS_USE_HIPBLASLT`. `mmb_dense_flag()` stays off on RDNA4, so BF16 gets its own switch: `GGML_CUDA_MMB_BF16_DENSE=0` / `1` forces off / on, and `GGML_CUDA_MMB_BF16_MIN_T` sets the floor.
- **0002** RDNA4 defaults `hc16` off. The hc gate's BF16 side copy (slot 1) is only read where the graph marks the output BF16-only, and those marking passes are RDNA3_5-only, so on RDNA4 it was written and never read. This patch also applies with hipBLASLt on, but changes no output. `GGML_CUDA_MMB_HC16=1` restores it.
- **0003** The warp-per-token tiny-M kernel gets a BF16-weight variant, used for BF16 weights with M <= 8 instead of padding the dense tile (the shared-expert gate M=1 and the hc inject M=4). `GGML_CUDA_MMB_TINY_M_BF16=0`: dense tile.
- **0004** That kernel reads the cached bf16 copy of the activation. It is bandwidth-bound, and the copy is usually already there from the pair's other GEMM. `GGML_CUDA_MMB_TINY_BF16_XH=0`: F32 read.
- **0005** For these BF16 GEMMs, MMB takes the RNE copy that the qwen4exp prefill producers already write into `bf16_src1_cache` for the cuBLAS path (`hc_combine_norm`'s xn, `dsv4_hc_pre`, the hc_post_rms norm), under the same key, instead of converting again. Same rounding, so bit-identical. Every other MMB GEMM still inserts its activation into the MMB cache, where the hc gate-mix fusion looks it up. `GGML_CUDA_MMB_PRODUCER_BF16=0`: convert.
- **0006** A BF16 weight whose 256x128 grid is below 64 workgroups (the hc inject and router at small prompt widths: 2x3 = 6 workgroups on 64 CUs) takes a 128x64 or 64x32 tile instead. Each output still sums K in the same order, so this is bit-identical. `GGML_CUDA_MMB_SMALL_TILE=0`: off; `GGML_CUDA_MMB_SMALL_GRID` sets the threshold.

**Speed and output.** r325, ROCm 10.1, 2 x R9700, `-sm tensor`, GSQ IQ3_S, your PR #304 pool settings (`--n-cpu-moe 47 -lme 47 -ub 6144 --load-mode dio`, `MOE_HOST_POOL_RAM_PCT=0`, `LLAMA_KV_REGION_POOL=0`, headroom 1536, `GGML_CUDA_ALLREDUCE=ce`), MTP expQ4K head. One run per arm, in t/s:

| | stock, hipBLASLt on | stock, `=0` | patched, `=0` | patched, hipBLASLt on |
|---|---|---|---|---|
| 8 short prompts (3092 tok) | 514 | 445 | 494 | 510 |
| 31.5k prefill | 2457 / 2439 | 1880 / 1882 | 2428 / 2404 | 2453 / 2437 |
| 4.4k prompts prefill / decode | 2105 / 109.0 | 1678 / 110.8 | 2072 / 109.7 | 2107 / 108.2 |
| 259.6k needle prefill / answer | 2331 / 6,358 | 1822 / 6,358 | 2308 / 6,358 | 2327 / 6,358 |
| greedy decode | 114.2 | 107.8 | 112.8 | 114.0 |

- **hipBLASLt on:** the patched build gives stock's outputs exactly (same greedy, prompt-set and 4.4k hashes) at the same speed.
- **Pinned (`=0`):** five fresh starts of the patched build gave the same greedy, prompt-set and 4.4k hashes every time, 0 faults.
- **KLD** (wikitext-2, `-c 8192`, 3 chunks, `-ub 4096`, same 2-GPU setup), base = stock with `=0`:

  | arm | mean KLD | same top | PPL ratio |
  |---|---|---|---|
  | patched, `=0` | 0.013868 | 96.30 % | 0.9938 ± 0.0018 |
  | stock, hipBLASLt on (one start) | 0.013985 | 96.57 % | 0.9996 ± 0.0018 |

  In this run, MMB's distance from rocBLAS-Tensile is about the same as hipBLASLt's.
- **test-backend-ops** `-o MUL_MAT` with `=0`: 2679/2680 on both trees. All 239 BF16 cases pass. The one failure is the same f16 case on both (`n=2105344`, rocBLAS's large-N path).

**Base:** formatted against and measured on r325 (`v17-4f9dfa0e3-r325`, tree `5959be96f8`).

**In use:** we validated the series on ROCm 10.1 with hipBLASLt off (2026-11-19, our production config, KLD against the rocBLAS path 0.0100 on Flash-Next UD-IQ4_XS). It did not run in production in that mode, because we moved to ROCm 10.2 the next morning. There we keep hipBLASLt on, since its 1.5.0 carries the stable solution order (`c50ea60130f`), so the series is inactive. If a version-gated default ever interests you (hipBLASLt on for >= 1.5.0, `=0` below), we have a small `ggml_cuda_init` probe for that and are happy to share it. We left it out of this series because of §44.

**Limits / not tested:**
- RDNA4 (gfx1201) only. The new defaults are RDNA4 config fields, and RDNA3_5 / RDNA3_0 are unchanged, though we did not run them.
- NVIDIA, Windows and 1-GPU runs: not tested.
- Of the models we run, Flash-Next GSQ IQ3_S has by far the most of these BF16 dense weights (484 tensors). Our UD-IQ4_XS production model has fewer (router, hc, indexer); there the series gave about +1 % prefill with hipBLASLt off. With hipBLASLt on it changes nothing on either.

We are happy to run any further checks on these cards if they would help.

Files:
- `0001`-`0006` `*.patch`: `git format-patch` against r325; apply with `git am`.
- This README.
- `logs/ab-r325.log`: the A/B, multi-start, KLD and test-backend-ops runs above.

(Claude-assisted, as before.)
