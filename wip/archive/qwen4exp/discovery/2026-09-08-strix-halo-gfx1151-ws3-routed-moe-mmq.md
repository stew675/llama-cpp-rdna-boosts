# Strix Halo (RDNA3.5 / gfx1151) — WS3 #3: routed-compact MoE MMQ for the i-quants

Date: 2026-09-08 session (handoff from SESSION-BRIEF-2026-09-08.md). Machine/build/model/protocol
identical to the 2026-09-06 records (same APU, same warm page cache, canonical
`~/bin/build-llama-rocm-714` build of ~/llama.cpp `qwen4exp`, llama-bench `-ngl 99 -t 15 -r 3
-b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none`, prompts DESCENDING in one process
= warm clock). A = ~/llama.cpp qwen4exp `a1121cf2d` (WS4 fusion DEFAULT ON + WS3 #3 below);
B = ~/strix-llama.cpp `c7af5c6c2` (halo-box, unchanged). IQ4_XS UD 87.24 GiB, non-MTP.

## What landed (A commit `a1121cf2d`, DEFAULT ON)

Port of halo-box's RDNA3.5 MoE MMQ family into A's mmq path (B source-of-record; A/B mmq-config
tables were already identical — the entire i-quant expert gap was the dispatch structure):
- `mul_mat_q_routed_compact`: one descriptor per real (expert, J-tile) pair (descriptor list built
  by `build_mmq_routed_descriptors`, single block, one thread per expert), replacing the plain
  (x-tile, expert) block grid whose J-tiling was sized to the flattened row count (every per-expert
  tile mostly empty). Bit-identical by construction: same `mul_mat_q_process_tile`, same per-tile
  accumulation order.
- Per-expert J selection in `mul_mat_q_switch_J` (`mmq_rdna3_5_id_get_J`: 16/48/64/128 by
  rows-per-expert range, gfx1151-measured; Q4_K/Q5_K/Q6_K/Q8_0 ranges + `GGML_Q6_COMPACT_J` env
  included; IQ types IQ3_S/IQ4_NL/IQ4_XS → J=64 at ub 2048 for this model's 512 experts / 10
  active = 40 rows per expert).
- Gate decision (resolved the open question): RDNA3_5-only, exactly as B gates it — this box is
  gfx1151 and B never enables it on RDNA4; gfx1201 stays off until validated on the gfx1201 box
  through the normal delivery flow. `GGML_CUDA_DISABLE_MMQ_ROUTED=1` disables only the compact
  dispatch (J selection stays) for same-build A/B.

## Numerics gate (PASSED)

- By construction compact == plain mmq (same kernel inner loop); verified: llama-cli seed-42
  temp-0 output text IDENTICAL across {compact-default, GGML_CUDA_DISABLE_MMQ_ROUTED=1,
  pre-change known-good} at the 7-token probe AND at a 4572-token pp + 40 decode (covers J=64/J=128
  chunks + 3 ubatches). The only difference in the long run's output was the printed pp t/s
  (compact 358.5 vs plain 328.2 — the perf win, same text).
- rocprof pp2048 (compact-on): `mul_mat_q_routed_compact` fires on the IQ expert GEMMs —
  IQ3_S J64 ×184, IQ4_NL J64 ×86, IQ4_XS J64 ×4, Q8_0 J48 ×8 (564 compact dispatches + 274
  descriptor builds in ~8873 total) — exactly the WS1-attributed i-quant expert piece. The Q8_0
  J128 plain mul_mat_q calls (non-compact, ~978) are the non-MoE/other-expert GEMMs, unchanged.

## Perf — depth-0 pp t/s (mean ± stdev, r3, warm clock, same session)

| pp | A compact ON (default) | A compact OFF (=1 env) | Δ compact | B (c7af5c6c2) | gap B/A | pre-change gap* |
|----|-----------------------:|------------------------:|----------:|--------------:|--------:|----------------:|
| 512 | 401.21 ± 0.54 | 380.95 ± 1.99 | +5.3% | 644.98 ± 21.3 | 1.61x | 2.05x |
| 1024 | 407.78 ± 8.62 | 398.27 ± 9.18 | +2.4% | 725.30 ± 13.1 | 1.78x | 2.04x |
| 2048 | 382.24 ± 0.50 | 364.44 ± 18.2 | +4.9% | 767.94 ± 12.5 | 2.01x | 2.21x |
| 4096 | 473.67 ± 1.45 | 462.78 ± 1.16 | +2.4% | 726.16 ± 1.52 | 1.53x | 1.68x |
| 8192 | 534.99 ± 1.57 | 521.08 ± 0.59 | +2.7% | 676.23 ± 1.12 | 1.26x | 1.37x |
| 16384 | 567.64 ± 2.41 | 548.90 ± 5.53 | +3.4% | 598.71 ± 1.58 | 1.06x | 1.14x |
| tg128 @0 | 24.24 ± 0.01 | 24.26 ± 0.00 | flat | 26.02 ± 0.03 | — | — |

*pre-change = the 2026-09-06 post-fusion record (cross-session — see caveat below).

Depth: pp2048 @ d12288 compact 329.88 vs 324.15 (+1.8%, no at-depth regression); tg128 @ d12288
flat (decode is the mmvq path, untouched by design). Memory stable across the r3 ladders at
~116 GB VRAM.

## Reading

- B is a ~1% stable reference across sessions (598.7 vs 600.9 @16k today vs 09-06; 645.0 vs 647.4
  @512) — same-session A-vs-B ratios are clean. The compact port moved EVERY gap row: pp16384
  1.14x -> 1.06x, pp8192 1.37x -> 1.26x, pp4096 1.68x -> 1.53x, pp2048 2.21x -> 2.01x, pp1024
  2.04x -> 1.78x, pp512 2.05x -> 1.61x. The pp16384 row is now within 6% of B.
- The same-session compact-vs-env delta (+2.4-5.3%, largest at small pp where rows-per-expert is
  smallest and the old grid waste was biggest) is the compact kernel alone, at the SAME J. The
  per-expert J selection contributes on top of that (its own same-session effect is visible in the
  env-off rows vs the 2026-09-06 absolutes, e.g. pp512 380.95 vs 315.79 — cross-session, treat as
  indicative only). CAVEAT: this box's absolutes drift +5-20% between sessions (clock/boost); the
  2026-09-06 pre-change column is context, not a same-session baseline. All conclusions that matter
  are same-session (this table's middle columns + the A-vs-B rows).
- Remaining gap at pp512-4096 is still dominated by the per-ubatch elementwise/routing/reduction
  tail (B's hc_* + weighted-expert-sum/graph fusions; A's tail is unfused outside WS4's hyperconn
  coverage), the shallow QSA-vs-dense chunk (WS3 #2 opt-in exists but is default-off pending its
  llama-bench artifact), and the fused-Q8_0-up/gate residue. tg@0 decode (A 24.2 vs B 26.0) is the
  later generation-phase item.

## Status

- WS3 #3 (routed-compact MoE mmq, i-quants) DONE: implemented, gated (RDNA3.5, B-parity), numerics
  + depth + memory gates PASSED, DEFAULT ON, env opt-out retained. RDNA4/gfx1201 enablement awaits
  the gfx1201 box in the delivery flow.
- Still open: WS3 #2 llama-bench-only multi-ubatch artifact (root cause not found; shortcut stays
  opt-in default OFF); the routing/reduction tail workstream (B's weighted-expert-sum + concat
  fusions) is NOT yet requested. WS6 re-base still NOT indicated.
