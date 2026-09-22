# Phase-1 item 1 follow-up — the `hc_combine_norm_f32` `_b256` swap: ported, measured, REJECTED (2026-09-21)

**Status:** closed **negative**.  The reference's `hc_combine_norm_f32_b256` was ported behind the
existing matcher, verified to fire, gated, and then **reverted**.  It is neither bit-identical to the
delivery's kernel nor faster on our tree.  The fork working tree is clean at `1004c65db`; nothing was
committed.

## What was tried

The reference's `hc-cn.cu::hc_combine_norm_f32_b256` (256 threads, two elements per thread, `float2`
/ packed accesses, `__launch_bounds__(256, 4)`) was added to `hyperconn.cu` and dispatched from
`ggml_cuda_op_hc_combine_norm` for the non-single-block, even-`n_embd` path, with
`GGML_CUDA_HC_CN_B256=0` as the A/B kill switch.  The `bo_hc` block_out variant was kept
(`block_out + (bo_hc ? row : t) * n_embd`); the combine/norm arithmetic was kept identical
(`hc_mul_rn`/`hc_add_rn`).

It fires: qwen4exp IQ4_NL width-probe row-0 hash moves
`268e0673300b7a33` (1024-thread) → `3a4b1b9264e7f60e` (b256), both `width_purity=PASS (worst maxdiff 0)`.

## Why it is rejected

### 1. It is not bit-identical — the RMS reduction tree differs

The delivery's `hc_combine_norm_f32` replays `rms_norm_f32<1024, true>`: 1024 threads, each
accumulating the up-to-3 columns `{tid, tid+1024, tid+2048}`, then
`block_reduce<block_reduce_method::SUM, 1024>`.  The `_b256` kernel accumulates **pairs**
(`col = (tid + k*256)*2`) into `block_reduce<SUM, 256>`.  The partition of the 2560 squares is
different, so the `tmp` sum — and therefore `scale` and every `xn` — differs by an ULP.  This is
intrinsic to the thread-count change, not a port mistake.

### 2. That ULP changes the greedy text (deterministically)

qwen4exp IQ4_NL, `prompts/code-python.txt`, `-n 32 --seed 42 --temp 0`, `-b/-ub 2048`:

| arm | generated text |
|---|---|
| 1024-thread (`GGML_CUDA_HC_CN_B256=0`) | `1b59d651f2c3` (673 chars) |
| `_b256` (default) | `fc7c8a10ea45` (673 chars) |

Both arms reproduce their hash across two runs each; the first text difference is line 10 (byte 599).
So the divergence is the kernel, not run-to-run variance.  A text-level change is not acceptable under
the delivery's purity rule (the gatemix ULP shift was accepted only because the greedy text stayed
identical).

### 3. It is slower here anyway

gfx1151, qwen4exp IQ4_NL, `-b/-ub 8192`, `-ctk/-ctv f16`, `-n 0 -r 2`:

| pp | 1024-thread | `_b256` | delta |
|---:|---:|---:|---:|
| 8192 | 1267.0 | 1258.2 | **−0.7 %** |
| 32768 | 1200.6 | 1190.7 | **−0.8 %** |

(The run-to-run noise is ~±1 %, but the sign was consistent across both points; the text
divergence disqualifies it regardless.)

## Conclusion and what it implies

* The doc's "the `_b256` is the obvious kernel swap behind the same matcher" is a **trap on our
  tree**: the reference tuned it against *its* HC norm stream, while the delivery deliberately
  replays `rms_norm_f32<1024, true>`.  On gfx1151 the delivery's 1024-thread kernel is the better
  kernel (and the bit-identical one), so there is nothing to recover.
* A bit-identical `_b256` would have to reproduce the 1024-thread reduction tree, which means
  discarding the pair-wise packed layout that is the only reason `_b256` is fast.  Given the base is
  already faster, that work has no payoff.
* **Do not re-attempt this swap**; if a future session wants HC-combine speed on a different arch,
  measure the base first (it may already win there too), and keep the bit-identity gate.

## Next item after this

With the `_b256` swap closed, the next Phase-1 candidates are **item 4** (`norm-gated.cu`/`rms_rows`,
~1.2 % on our tree — and the same "does it replay our `rms_norm` order?" question applies) and
**item 5** (MoE bf16 epilogue + drop `concat_transposed`, ~3–4 %, beta's `MMB_DOWN16` is gated off).
Item 3.5 (the three QSA correctness fixes) stays an audit.
