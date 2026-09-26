# gfx1201 — do the gfx1151 lossy-prefill wins transfer? (`0010` DOWN16, `0011` blk16/res16)

**Date:** 2026-09-23.  **Box:** 3× Radeon AI PRO R9700 (gfx1201), ROCm 7.14.  **Tree:** closing
`closing-gfx1201`, tip `1be654fa71167e470ffcce70456bef7a23e6de25`.
**Question (maintainer):** the gfx1151 campaign set `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1
GGML_CUDA_MMB_DOWN16=1` for a *lossy* high-speed prefill (0010 + 0011, reported ~+10-15 % combined,
0011 alone +4.9/+4.8 % at pp8192/32768).  Does that win apply on gfx1201?

**Answer: no measurable transfer — and under this box's required `-sm tensor` mode the three
markings do not even execute.**  Two independent findings:

---

## Finding 1 — under `-sm tensor` the markings never run (a real portability gap)

HC16 (`0023`), DOWN16 (`0010`) and blk16/res16 (`0011`) are all implemented as **pre-allocation
graph markings** in the CUDA backend's `ggml_backend_cuda_graph_optimize`.  Under `-sm tensor`,
llama.cpp wraps the GPUs in the **meta backend**, and `ggml/src/ggml-backend-meta.cpp` says so
explicitly (its own `ggml_backend_meta_graph_optimize` comment):

> graph_optimize … **never runs for graphs owned by this backend** … Walk the graph for the same
> structural patterns here so the gallocr keeps the sources alive …

Measured with `GGML_CUDA_MMB_MARK_LOG=2` (the same binary, same model):

| split | `MMB_OPT` prints (CUDA `graph_optimize` calls) | `HC_BLK16 comb=` prints |
|---|---:|---:|
| `-sm tensor` | **0** | **0** |
| `-sm layer`  | 60 | 704 (comb 30-33 on most splits) |

So on the gfx1201 box's qwen4exp mode (`-sm tensor`), `0010`/`0011` are **inert** — that, not the
hardware, is why the first `-sm tensor` A/B read neutral (the text was also byte-identical:
`e440ed48b2f4` for default / blkres / down16 / all3 at 8K).  (HC16 is moot here — its marking is
`GGML_CUDA_CC_IS_RDNA3_5`-gated, so it never runs on RDNA4 by design.)

The gfx1151/gfx1100 boxes are single-GPU, so they run the CUDA `graph_optimize` directly and the
markings engage.  **The markings are therefore a per-device (single-backend) feature that the
tensor-split meta path bypasses.**

## Finding 2 — where the markings do run (`-sm layer`), the win is not there

Model Flash-Next **IQ4_XS** (this box; the gfx1151 measurement used IQ4_NL), f16 KV, `-b/-ub 4096`,
`llama-bench -r 3`, 3-GPU `-sm layer`:

| arm | pp8192 | pp32768 |
|---|---:|---:|
| default | 3444.07 | 4963.16 |
| `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1` (`0011`) | 3427.35 (−0.5 %) | 4960.35 (−0.06 %) |
| `GGML_CUDA_MMB_DOWN16=1` (`0010`) | 3485.97 (+1.2 %) | 4946.74 (−0.3 %) |
| all three | 3443.66 (−0.01 %) | 4952.29 (−0.2 %) |

**Flat, within run-to-run noise.**  (An earlier 2-round `-r 1` run showed +2.8 % at pp8192; the
`-r 3` run shows it was noise.  pp65536 did not complete for `default`/`down16` — a transient
`failed to decode prompt batch, res = -2` under memory pressure; irrelevant to the verdict since
the effect is already flat at 8192/32768.)

Firing was confirmed by the lossy text: `0011` moves the same-seed greedy text
`551425b9758e` -> `07f7d16a144a` (so the BF16 streams engage under `-sm layer`); `0010` leaves the
text at `551425b9758e`, so its firing is **unconfirmed** (its routed-down experts are IQ4_NL — 13+30
of them in this UD-IQ4_XS model — so the type gate passes; a mark log / kernel-timing check is
still owed).

Interpretation: the BF16-stream win is an **APU / unified-memory bandwidth** effect.  On gfx1151
(one device, ~256 GB/s) removing half the activation traffic is worth +5 % prefill; on three
discrete R9700s the activation traffic per device is 1/3 and the bandwidth is far higher, so it is
not the bottleneck.

## The port (if the markings are wanted under tensor split)

To make `0010`/`0011` (and any future `graph_optimize` marking) execute under `-sm tensor`, the
**meta backend's `graph_optimize` must run the CUDA markings on the per-device shard graphs, before
allocation**.  Two hard constraints:

* **The marks are keyed by `const ggml_tensor *`** (`mmb_state().bf16_only`,
  `ggml_cuda_mmb_is_bf16_only()`), so they must be applied to the exact shard tensor objects the
  kernels see at compute time — not to the meta-graph tensors.
* **The residual/block_out marks add gallocr alloc deps**, so this must happen in the scheduler's
  optimize phase (the meta backend currently builds `step_cgraphs[]` in `graph_compute`, which is
  after allocation).

That is a `ggml-backend-meta.cpp` change with allocator/alloc-dep risk (needs meta-split + CUDA-graph
re-validation).  The meta path already has a small precedent (`ggml_backend_meta_graph_optimize`
handles the `moe_weighted_reduction` alloc deps) to extend.

**But Finding 2 says the reward is likely zero on this hardware**, so the port should be
justified by something other than these two features (correctness, or enabling a future marking
that *is* bandwidth-bound here).

## Verdict

* `0010` / `0011`: **do not transfer** to gfx1201; leave default-OFF.  (`0010`'s firing also
  deserves a mark-log confirmation.)
* The meta-backend `graph_optimize` gap is the **real gfx1201 portability finding** — record it so
  the next "gfx1151 feature doesn't do anything on RDNA4" is diagnosed in minutes rather than
  re-measured blind.
