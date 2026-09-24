# Phase-1 item 4 — the narrow-row RMS norm fusion (`rms_rows`) is DONE (2026-09-22)

**Status:** DONE (WIP), default **ON**.  Fork branch `gap-closing` @ **`59bfcd719`**, exported as
[`patches/0006`](patches/0006-gap-closing-WIP-port-the-narrow-row-RMS-norm-fusion-.patch).  Bit-identical
and gated, per the default-on policy.

## What was ported

The other solution's `ggml/src/ggml-cuda/norm-gated.{cu,cuh}`:
`rms_rows_f32<GATE>` is a **wave-per-row** RMS norm for narrow rows (`ncols <= 256`, `>= 4096` rows) that
processes **8 rows per 256-thread block** instead of the one-block-per-row `rms_norm_f32<256,...>`.  The
model's per-head norms are ~786k rows of <= 256 columns, so the upstream launch is block-scheduling
bound.  Two matchers, wired in `ggml_cuda_try_fuse` **before** the generic `RMS_NORM + MUL` fusion (and
the gated variant's alloc-dep pass added):

* `ggml_cuda_norm_gated_match_at` — `RMS_NORM -> MUL(w) -> [VIEW/RESHAPE/PERMUTE/TRANSPOSE | MUL_MAT]* ->
  SIGMOID -> MUL`; the gate `MUL_MAT` (if any) is pre-dispatched, then one kernel computes norm+weight+gate.
* `ggml_cuda_norm_rows_match_at` — plain `RMS_NORM -> MUL(w)`.

`idx-relu-sum` was **already banked** on our tree (the fused `GGML_CUDA_QSA_INDEXER_SCORE` computes
`bias + sum_h relu(dot_h)`), so only the norm half of item 4 was ported.

## Why it is bit-identical

The reduction replays `norm.cu`'s `rms_norm_f32<256,...>` + `block_reduce<SUM,256>` exactly: per-warp
`__shfl_xor` trees over 32 consecutive columns (`offsets 16,8,4,2,1`), then a xor tree over the 8 warp
partials padded with zeros.  The gate `1/(1+expf(-z))` is our `op_sigmoid`, and the product order
(`(scale*x)*w`, then `*s`) matches the separate `RMS_NORM -> MUL -> SIGMOID -> MUL` chain.

## Validation (gfx1151)

* `test-logits-width-probe` qwen4exp IQ4_NL, P=2048: **`width_purity=PASS (worst maxdiff 0)`**.
* Row-0 logits hash **identical on vs off** (`GGML_CUDA_DISABLE_NORM_ROWS=1`):
  `row0 = 8949d53f635c18c3` and every width hash equal.
* Same-seed greedy text **identical** on/off: `765 chars sha=f61199ba5644`
  (`prompts/code-python.txt`, `-n 48 --seed 42 --temp 0`).
* Kernel profile (qwen4exp IQ4_NL, pp8192, `-ub 8192`, r=1): `rms_norm_f32<256,true>` 282.8 + the gated
  `unary_gated_op_kernel<op_sigmoid>`/mul chain -> `rms_rows_f32<true>` **210.5** + `rms_rows_f32<false>`
  **77.4** ms; `rms_norm_f32<256,true>` drops to 0.3 ms.  (The bare `rms_norm_f32<256,false>` 133.5 ms has
  no `w`, so `norm_rows` cannot cover it.)

## A/B — and the `-ub 8192` memory caveat

Measured with the **memory-safe `-b/-ub 4096` protocol** (see
[`2026-09-22-ubatch-8192-memory-confound.md`](2026-09-22-ubatch-8192-memory-confound.md)):

| pp | disabled | default | delta |
|---:|---:|---:|---:|
| 8192 | 1222.74 ±13.37 | **1226.09** ±16.18 | +0.27 % |
| 32768 | 1166.13 ±0.77 | **1169.37** ±6.81 | +0.28 % |

A small but **sign-consistent** win (~+0.3 %).  At `-ub 8192` it read +0.5–1.1 %, i.e. the memory-pressure
confound flatters the change (the fusion removes nodes, so the default arm runs in a *looser* pressure
regime).  The clean number is the ub4096 one.  The kernel-level saving (~275 ms of GPU time) does not
translate 1:1 end-to-end because the prefill is partly memory/IO-bound.

**Kill switch:** `GGML_CUDA_DISABLE_NORM_ROWS=1`.

## Next

**Item 6** (`qsa3_attn` body, 817 vs the reference's 618 ms), then item 7 (tall tile) / item 8 (QSA graph
flags).  Any further A/B must use the `-ub 4096` protocol (or record the min free memory alongside).
