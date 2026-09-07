# 2026-09-07 investigative drops — archived from the qwen4exp branch

These four commits were part of the QSA decode campaign but provide NO functional or
performance effect on either architecture under the shipped arch policy (2026-09-07
crossover tables).  They were reverted/removed from the `~/llama.cpp` qwen4exp branch
(cleanup commit `8994dd490`) so the `qwen4exp-support.patch` deliverable carries only
what reproduces the result tables.  Each is preserved here for future re-evaluation.

| commit | what it was | why archived |
|---|---|---|
| `e6b7ae6f0` | env-gated `GGML_CUDA_QSA_DECODE_SKIP=N` layer-skip probe (measurement tool) | investigative only; its own comment said "remove after measurement" |
| `fde1f2def` | fused `INDEXER_POOL` op (pool+norm) | partial-fusion stepping stone; the fused `INDEXER_SCORE` subsumes it and shares no code; the op was only reachable via an env-gated config that is neither default nor in any table |
| `d6164ad6a` | topk radix_init fold (11 -> 10 launches/op) | measured wall-flat at 32K (interleaved: 40.27 vs 40.28 t/s) |
| `4ff65247c` | two-round 16-bit topk select (decode/small-row path) | measured wall-flat at 32K both without cache1 (40.18 vs 40.20) and WITH cache1 (37.69 vs 37.69, re-tested 2026-09-07 after the cleanup) |

The topk reverted to the pre-fold 8-bit radix with init (its state at `6703ad09f`).

Supporting records: `wip/archive/qwen4exp/discovery/2026-09-07-sparse-decode-cost-isolation-rocprof.md`
and the worklog entries in `wip/qwen4exp/gfx1201-porting.md` (the launch-cut A/Bs, the
round2 build + parity + flat measurement, and the cache1 re-test).
