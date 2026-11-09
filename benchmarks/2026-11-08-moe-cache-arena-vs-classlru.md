# The MoE expert cache: delivery arena vs upstream's ClassLRU at equal MiB (`TODO` #291, `D1`)

**Date:** 2026-11-08 · **Box:** primary, 1 × R9700 (gfx1201, 32 GiB), Ryzen 9950X3D, ROCm 10.1.0, VMM build (`GGML_HIP_NO_VMM=OFF`, which the arena requires and `~/bin/build-llama-rocm` sets)
**Model:** Qwen3.6-35B-A3B Q8_0, `-ngl 99 -ncmoe 20` (16 320 MiB host experts) and `-ncmoe 40`, `-sm layer`, `-c 8192`
**Prompt:** the shipped 5 246-token prose prompt, greedy, 128 tokens — **real text, not `llama-bench`'s prompt** (`TODO` #296: `llama-bench` builds it from `std::rand() % n_vocab`, so every ubatch routes over every expert and the arena has no locality to exploit; the same arm reads 838 t/s there against 1 442 t/s here)
**Arms:** `build-rocm` = delivery `8715156da`; `build-rocm-base` = unpatched `a691b2345` (its ClassLRU cache is `--moe-cache-mib`).  Two interleaved rounds; every number below is the mean of two runs that differed by < 1 %.

## `-ncmoe 20` (16.3 GiB host experts) — the D1 shape

Delivery arena: `MoE expert cache (auto): arena 11411.2 MiB of 16320.0 MiB host experts (69.9 % residency)`, settling at **10 850.2 MiB** after the compute-buffer drops; the decode turn reads `MoE arena = 0.8787 (27289 hit / 31056 reaches this turn)`.

| arm | prefill t/s | decode t/s | text |
|---|---:|---:|---|
| **delivery arena (default)** | **1441.6** | **72.0** | `504 a2cdef23588f` |
| delivery, arena disabled (`MOE_EXPERT_CACHE_MIB=0`) | 757.1 | 48.7 | same |
| delivery arena **+** `--moe-cache-mib 4096` (coexist) | 808.4 | 63.9 | same |
| upstream ClassLRU, `--moe-cache-mib 4096` | 782.3 | 53.3 | same |
| upstream ClassLRU + `--moe-cache-cpu-share 0.5` | 778.3 | 55.7 | same |
| upstream ClassLRU + `--moe-cache-cpu-share 0.95` | 778.4 | 58.3 | same |
| upstream, no cache | 747.5 | 42.5 | same |

## `-ncmoe 40`

| arm | prefill t/s | decode t/s |
|---|---:|---:|
| **delivery arena (default)** | **1394.2** | **69.2** |
| delivery, arena disabled | 371.1 | 31.8 |
| delivery arena + upstream cache 4096 | 399.0 | 34.5 |
| upstream ClassLRU 4096 | 455.8 | 32.6 |
| upstream, no cache | 431.3 | 30.8 |

## What the numbers say

1. **At equal MiB the delivery's arena wins outright**: **+84 % prefill / +35 % decode** at `-ncmoe 20`, **+206 % / +112 %** at `-ncmoe 40`. Upstream's cache does help *its own* baseline (prefill 747.5 → 782.3, decode 42.5 → 53.3), and its `cpu-share` recovers part of the decode gap (53.3 → 58.3 at 0.95) — but neither comes near the arena.
2. **Coexistence is a loss, not a complement.** Enabling `--moe-cache-mib 4096` beside the arena costs **44 % prefill and 11 % decode** (1441.6/72.0 → 808.4/63.9 at `-ncmoe 20`; 1394.2/69.2 → 399.0/34.5 at `-ncmoe 40`). The ClassLRU cache takes VRAM the arena would otherwise size itself against, so two tiers at one budget is strictly worse than one. "Keep both" therefore means *either/or*, which is already the shipped state: `--moe-cache-mib` defaults to 0 and our arena is the default tier.
3. **The text is identical in every arm** — all five tiers, both `-ncmoe` levels, no cache, both caches, and both `cpu-share` values produce the same 504-character greedy generation. Whatever the tier, the numerics do not move.
4. **The D1 premise no longer reproduces, and part of it was a `#296`-class artifact.** D1 recorded upstream's cache as **+17.6 % decode** over ours (delivery 547.6/53.5 vs upstream 544.2/62.9 at 4080 MiB). Re-measured on today's tree at the same `-ncmoe 20`: ours **1441.6/72.0** against upstream **782.3/53.3** on real text, and ours **838/83.9** against upstream **857/48.0** under `llama-bench`. So (a) the arena's decode has improved substantially since r210 (53.5 → 83.9 on the `llama-bench` arm) and the inversion is simply stale, and (b) the arena's *prefill* is regime-dependent in exactly the way `#296` established for the host-expert transport (838 vs 1 442 t/s on the same arm, random tokens vs real text).

## Decision (`D1`, maintainer, 2026-11-08)

**Keep both, ours default-on, upstream's opt-in only (for comparison).** Concretely: the delivery's arena is the default tier and stays so; upstream's `--moe-cache-mib` path stays shipped and dormant at its `0` default, reachable for an A/B or for a platform where the arena cannot arm (`GGML_HIP_NO_VMM=ON` is upstream's default — the arena declines there and upstream's cache is then the only device tier); **enabling both at one budget is recorded as a slowdown** (44 % prefill / 11 % decode at `-ncmoe 20`), so coexistence stays either/or rather than "both on"; and neither **ClassLRU** nor **`--moe-cache-cpu-share`** is adopted, since at equal MiB neither closes the gap (0.95 share reaches 58.3 t/s against the arena's 72.0).

Records: `WORKLOG.md` 261108.015, `DECISIONS.md` `D1`, `RUNGS.md` `P13`.
