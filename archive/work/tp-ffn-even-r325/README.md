# tp-ffn-even-r325: an even FFN / shared-expert `-sm tensor` split when the size divides evenly

**Status: PROMOTED at r338 (PR #342), as an OPT-IN feature.**  Folded into **block 09** as
`-of tp-ffn-even` (`LLAMA_TP_FFN_EVEN=1`), **default OFF**, released as `v17-919794500-r338`
(tip `2f0d05010`, tree `3d7017fbb`).  The `0001-*.patch` below is the contributor's original, cut against r325:
it is **not applied** by this repo (the shipped code is the folded-and-adapted version in `patches/`), and it no
longer applies to the current release tree.  Kept as the campaign's original brief (verbatim from PR #342 below
the status header); it is not work in flight.  See [`PROMOTION.md`](PROMOTION.md) for what shipped, what it
measured here, and the three claims that were corrected.

Thank you for the quick look at #340, and for the r325 gather change. Here is a smaller one, from a decode trace of our 2-GPU setup.

With `-sm tensor`, FFN and shared-expert tensors split with granularity `lcm(blck_size, 128)` (the "round up granularity to 128" rule from upstream). Flash-Next's experts and shared expert are 640 wide. With a 32-wide down type (IQ4_NL in your GSQ IQ3_S; also in our UD-IQ4_XS, which mixes Q5_0 / IQ4_NL / Q8_0 downs), each GPU's 320-row share rounds to 256, so every MoE layer splits 256 / 384, with the heavier GPU alternating by layer. The lighter GPU then waits in each MoE all-reduce.

**What it does** (`src/llama-model.cpp` `get_split_granularity`, FFN / shared-expert branch only)

- With no `-ts` (or an equal one), when the size divides by the device count and each share is a multiple of some granularity between `lcm(blck_size, 64)` and `lcm(blck_size, 128)`, the granularity halves from `lcm(blck_size, 128)` to the largest such value. For Flash-Next on 2 GPUs that is 64, so the split is 320 / 320.
- The floor of 64 keeps MMB on the down projection (it needs K % 64). Gate, up and down take `blck_size` from the down, so their boundaries stay identical.
- Anything else (uneven `-ts`, no equal split at 64 or above, 256-wide types) keeps `lcm(blck_size, 128)`.
- One WARN line per model load: `tensor split: FFN granularity 128 -> 64 so blk.0.ffn_down_exps.weight splits 320 / device`.
- `LLAMA_TP_FFN_EVEN=0`: the old rule; outputs identical to stock.

**Trade-off.** A 320-row gate / up slice is not a multiple of 128, so in prefill those GEMMs take MMQ's bounds-checked variant and lose the fused gate/up kernel (`ggml_cuda_mmq_needs_fallback`, already guarded before the fusion is chosen). Per kernel that is somewhat slower. In the runs below the balance still comes out ahead on decode and short prompts, and about even on long prefill; greedy decode is 1 % lower (114.6 -> 113.3), within our run-to-run spread.

**Speed and output.** r325, ROCm 10.1, 2 x R9700, `-sm tensor`, GSQ IQ3_S, your PR #304 pool settings (`--n-cpu-moe 47 -lme 47 -ub 6144 --load-mode dio`, `MOE_HOST_POOL_RAM_PCT=0`, `LLAMA_KV_REGION_POOL=0`, headroom 1536, `GGML_CUDA_ALLREDUCE=ce`), stock defaults (hipBLASLt on), MTP expQ4K head. One run per arm, in t/s:

| | stock | patched | patched, `LLAMA_TP_FFN_EVEN=0` |
|---|---|---|---|
| 8 short prompts prefill (3092 tok) | 514 | 636 (+24 %) | 499 |
| 31.5k prefill (first / second run) | 2458 / 2437 | 2459 / 2446 | 2433 / 2422 |
| 4.4k prompts prefill / decode | 2113 / 109.0 | 2178 / 116.9 | 2105 / 109.0 |
| 512-token prompt set decode | 127.9 | 133.0 | 127.3 |
| greedy decode | 114.6 | 113.3 | 114.5 |
| 259.6k needle prefill / decode / answer (building count) | 2329 / 88.8 / 6,358 | 2328 / 91.5 / 6,358 | 2326 / 90.9 / 6,358 |

- `LLAMA_TP_FFN_EVEN=0` gives stock's greedy, prompt-set and 4.4k hashes exactly. The patched build changes them (the partial sums are grouped differently); the 256-token answer to a 30.3k-token summary prompt is unchanged. 0 faults in every arm.
- **KLD** (wikitext-2, `-c 8192`, 3 chunks, same 2-GPU setup), base = stock:

  | `-ub` | mean KLD | same top | PPL ratio |
  |---|---|---|---|
  | 4096 | 0.013897 | 96.57 % | 1.0059 ± 0.0019 |
  | 256 | 0.013137 | 96.76 % | 1.0032 ± 0.0018 |

  About the same distance as hipBLASLt on vs off in #340 (0.013985).
- On our own tree with our production model (Flash-Next UD-IQ4_XS, 48 CPU-MoE layers, ROCm 10.2, one run): decode +3-8 % (prompt set 129.7 -> 140.6), 4.4k prefill +3 %, short prompts +11 %, 31.5k prefill +2 %; KLD vs the same build with `LLAMA_TP_FFN_EVEN=0`: 0.0114 (`-ub 4096`) / 0.0119 (`-ub 256`).

**Base:** formatted against and measured on r325 (`v17-4f9dfa0e3-r325`). r326 and r327 do not touch `src/llama-model.cpp` / `.h`, and the patch applies cleanly to r327.

**Limits / not tested:**
- RDNA4 (gfx1201), 2 GPUs only. 3- or 4-GPU splits (for example 768-wide experts on 4 GPUs: 192 each instead of 128 / 256) should balance the same way, but we cannot run them.
- NVIDIA and Vulkan: not tested; their kernels may weigh the non-128 slice differently.
- It only acts when equal shares are possible at granularity 64 or above. A broader rule (64 whenever it balances better than 128, e.g. 960 wide on 4 GPUs) would cover more shapes; we left it out because we can only measure the 2-GPU case.
- A fallback-capable fused gate/up MMQ kernel would recover the fusion on these slices; not attempted.

Nothing is urgent on our side. We are happy to run any further checks on these cards if they would help.

Files:
- `0001-*.patch`: `git format-patch` against r325; apply with `git am`.
- This README.
- `logs/ab-r325.log`: the A/B and KLD runs above.

(Claude-assisted, as before.)
