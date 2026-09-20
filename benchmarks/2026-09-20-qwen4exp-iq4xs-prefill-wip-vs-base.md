# 2026-09-20 — Qwen3.8-Flash-Next IQ4_XS prefill: delivery base vs the `mmb` WIP

**Scope:** prefill throughput (`-n 0`) of the WIP `~/llama-wip-mmb` tree (`wip-mmb-general`) against
the true applied delivery base, on the same model and the same parameters.  This is the first
end-to-end prefill record since the bf16-producer port; it exists because the earlier "MMB off" A/B
arm was **not** a pristine base — it was the WIP tree with only the MMB/HC16 gates off, so it already
carried the WIP's always-on wins (qsa3 attention, the always-QSA default flip, the non-temporal
memory hints, and the sessions 16–18 indexer work).  The true base is measured separately here.

## Environment

| | |
|---|---|
| host | `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5) |
| runtime | ROCm 7.14 (`/opt/rocm-7.14-gfx1151`) |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (87.24 GiB, qwen4exp, HC + QSA) |
| build | `Release`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GPU_TARGETS=gfx1151`, RCCL on, ccache |
| params | `-ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -n 0 -r 3` |

Page cache warmed (`dd` over all shards) before every run; no parallel benches.  `-r 3`, the ± is the
bench's own std-dev across the three reps (**within-run**; between runs the same point moved ~±3 %).

## The three arms

| arm | tree | build | env |
|---|---|---|---|
| **A — true delivery base** | `/tmp/llama-base` worktree @ `8a2567e1e` | `8a2567e1e` | (none) |
| **B — WIP, MMB/HC16 off** | `~/llama-wip-mmb` @ `5d55da3e9` | `5d55da3e9`¹ | (none) |
| **C — WIP, all gates on** | `~/llama-wip-mmb` @ `5d55da3e9` | `5d55da3e9`¹ | `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1` |

¹ `llama-bench` prints `build: b41f338a7` for arm B/C — the version string is baked at cmake
*configure* time and the build was incremental.  The binary is the current tip `5d55da3e9`; confirmed
by the `indexer_topk_histogram_pass1` / `indexer_topk_histogram_blocks` kernels in the trace.

**Arm A** is the applied 16-block delivery tree (blocks 00–15): it already contains the delivery's own
quant/MMQ work (block 10 k-quant boosts, block 13 fused MoE MMQ/mmvq) and the base flash-attention
path.  It does **not** contain any WIP change.  **Arm B** adds only the WIP's always-on wins —
`qsa3` packed-block attention (compile-time default on), the `always-QSA` prefill flip (the dense
shortcut dropped), the non-temporal load hints in `dsv4_hc`/`concat_transposed`/`moe_weighted_reduction`/
the fused unary producer, and the sessions 16–18 indexer work — while leaving the MMB dense GEMM and
the HC16 bf16-producer port off.  **Arm C** adds those two as well.

## Results — prefill t/s (higher is better)

| pp | A true base | B WIP always-on | C WIP all gates |
|---:|---:|---:|---:|
| 512 | 645.83 ± 4.67 | 667.15 ± 2.22 | 795.27 ± 1.85 |
| 1024 | 751.77 ± 2.04 | 764.14 ± 5.65 | 972.55 ± 0.89 |
| 2048 | 808.02 ± 0.83 | 823.54 ± 1.03 | **1062.48 ± 6.41** |
| 4096 | 726.44 ± 20.68 | 803.09 ± 18.23 | 1038.71 ± 12.39 |
| 8192 | 707.26 ± 8.08 | 804.54 ± 6.82 | **1037.50 ± 7.55** |
| 16384 | 687.58 ± 0.17 | 790.27 ± 0.09 | **1011.50 ± 1.26** |
| 32768 | 659.17 ± 0.96 | 764.18 ± 0.20 | **972.35 ± 0.25** |

| pp | A → C (total WIP) | A → B (always-on only) | B → C (MMB/HC16) |
|---:|---:|---:|---:|
| 512 | +23.1 % | +3.3 % | +19.2 % |
| 1024 | +29.4 % | +1.6 % | +27.3 % |
| 2048 | +31.5 % | +1.9 % | +29.0 % |
| 4096 | +43.0 % | +10.6 % | +29.3 % |
| 8192 | **+46.7 %** | +13.8 % | +29.0 % |
| 16384 | **+47.1 %** | +14.9 % | +28.0 % |
| 32768 | **+47.5 %** | +15.9 % | +27.2 % |

## Reading

* **The total WIP prefill gain against the true delivery base is ~+32 % at pp2048 and ~+43–48 % from
  pp4096 to pp32768** — not the +27–34 % the earlier MMB-off arm implied.  The difference is exactly
  the always-on WIP work in arm B (+14–16 % at 8K–32K), on top of the delivery's own quant/attention
  work that sits in *both* A and C.
* **The delivery base has a prefill cliff at pp4096** (808 → 726 t/s, then 707 at 8K): the base's
  `LLAMA_QSA_DENSE_SHORTCUT` still routes the startup/short-context regime to the dense masked FA arm.
  The WIP's `always-QSA` flip (arm B/C) removes it — arm B jumps to 803 at pp4096 while arm A drops.
  This is the single largest *shape* difference in the table.
* **MMB/HC16 (arm B → C)** is a flat ~+27–29 % over the whole range, starting already at pp512.
* **The indexer work (sessions 16–18)** is inside arm B/C: it took the 32K indexer family from 2.94 %
  to 1.47 % of the run (~+1.3 % end to end at 32K); that is why the prefill curve is now flatter.

## Caveats

* `-n 0` llama-bench prefill only — this says nothing about decode or MTP.  The WIP's decode line is
  the earlier MMB/non-temporal work; the MTP gate (`plain == draft-mtp`) is unchanged.
* Arm C is **bit-identical** to arm B and to the delivery on the model's output: PPL c2048 10.6015
  (`GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1`), greedy `9c281c415082`, and the WIP-vs-fallback A/B at ~16k
  context agrees (`7d2e5b3e46dd`).  The prefill gain is therefore not a numerics trade.
* Arm A already contains the delivery's own blocks 00–15 (its quant/MMQ and attention work), so the
  table isolates the **WIP** delta.  The upstream fork point `ebbb18522` (clean llama.cpp) is lower
  than arm A and is not measured here.

## Raw

The three arms were each measured in one `llama-bench` invocation:
`-m $M -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -p 512,1024,2048,4096,8192,16384,32768 -n 0 -r 3`
(from `/tmp/llama-base` for arm A, `~/llama-wip-mmb` for B/C with the listed env).  The exact command
form is `HANDOVER.md` §4.
