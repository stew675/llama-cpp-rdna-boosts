Thanks for r268, and for the copy-engine all-reduce chain. Its staged peer copies and per-arm logging made this a small change that was easy to measure. Below is a 43-line change that speeds up that arm with 2 GPUs, offered in case it's useful. The output is bit-identical. Nothing is urgent on our side; we already run it.

**0001: copy-engine all-reduce, one exchange for 2 ranks** (`ggml-cuda.cu`, +43)

For 2 ranks, `ggml_backend_cuda_comm_allreduce_ce` currently runs a reduce-scatter (each rank sends half its bf16 partial, the owner adds) and then an all-gather of the reduced halves. With two GPUs, each rank can instead send its whole bf16 partial to the peer once. Both ranks then form every element themselves as `fp32(bf16(fp32(own) + fp32(peer)))` in one kernel, written straight into the output.

- **Same result.** The rounding is the same as `ggml_cuda_ce_add_bf16` on the owner, followed by the all-gather and `to_fp32`, and fp32 addition is commutative, so the output bits do not change.
- **Same link traffic, less waiting.** The same bytes cross the link in each direction. There is one cross-device dependency instead of two, and no separate add or convert passes.
- **Synchronization.** The receive buffer (`tmp1`, already at least `ne` elements) is reused across calls. Each send waits on the peer's `ce_ev_done`, which was recorded in the previous call, so the wait does not stall.
- **Scope.** It only applies when `n == 2`; other rank counts, the q8 wire, the RCCL arm and the internal pipeline are untouched. `GGML_CUDA_AR_CE_SINGLE=0` restores the two-phase form.

**Measurements** (2x Radeon AI PRO R9700, gfx1201, PCIe 5.0 x8 each, ROCm 10.1; stock r268 `a92cc0eec`, tree `ce7e306fdaa0`; patched tree `9b02880045b2`). Servers ran in the order stock, patch, stock, patch; t/s, round 1 / round 2:

Qwen3.8-Flash-Next GSQ-RCO IQ3_S with your host-pool setup plus the copy-engine arm: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_EXPERT_HEATMAP=load GGML_CUDA_ALLREDUCE=ce`, MTP head, `-c 262144`:

| | stock | patch | |
|---|---|---|---|
| 31.5k-token prompt (2nd prefill) | 2230 / 2230 | 2285 / 2279 | +2.2-2.5 % |
| 4.4k-token prompt set | 1998 / 1999 | 2107 / 2103 | +5.2-5.5 % |
| 101k-token prompt | 2192 / 2192 | 2254 / 2246 | +2.5-2.8 % |
| decode, 256 tokens | 112.3 / 113.3 | 113.2 / 113.0 | unchanged |

Flash-Next GSQ-RCO IQ3_XXS, all experts in VRAM, default environment (your chain picks the copy-engine arm for dense models): `-sm tensor --n-cpu-moe 0 -ot per_layer_token_embd.*=CPU -lm none -lzm off -ub 2048 -b 2048`, `GGML_CUDA_SLAB_HEADROOM_MIB=2048`, MTP head, `-c 262144`:

| | stock | patch | |
|---|---|---|---|
| 31.5k-token prompt (2nd prefill) | 2748 / 2748 | 2762 / 2756 | +0.3-0.5 % |
| 4.4k-token prompt set | 2388 / 2394 | 2400 / 2403 | +0.4-0.5 % |
| 101k-token prompt | 2463 / 2470 | 2487 / 2486 | +0.6-1.0 % |
| decode, 256 tokens | 159.4 / 159.8 | 159.8 / 159.9 | unchanged |

- **Identical output.** In every arm the replies are byte-identical to stock: greedy decode, the 4.4k prompt set, the 101k prompt and a 256-token generation. Two concurrent streams were also run, and none of the 8 servers had a GPU fault.
- **Against the RCCL default** (one run each, r265, on a build that also carried other patches, all switched off for that run). With host experts and no override, your chain uses the RCCL ring. The patched copy-engine arm measured 2278 / 2104 / 2242 t/s on the three prompts, against 2270 / 2049 / 2227 for that default. Offered in case it is useful for the host-expert arm choice; it is a single run.

Untested: more than 2 GPUs (that path is unchanged), NVIDIA / CUDA, RDNA3, gfx1151, Windows.

Files: `0001-*.patch` (`git format-patch` against r268, apply with `git am`) and this README. Happy to run anything else on these cards.

(Claude-assisted, as before.)
