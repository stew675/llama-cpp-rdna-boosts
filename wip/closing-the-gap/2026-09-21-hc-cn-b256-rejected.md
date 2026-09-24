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
* **Do not re-attempt the pure-F32 swap**; if a future session wants HC-combine speed on a different
  arch, measure the base first (it may already win there too), and keep the bit-identity gate.

## Correction (2026-09-21, session 3, from the ubatch-8192 profile)

The 2026-09-20 profile's `hc_combine_norm_f32_b256` **554 ms / 190** vs our `hc_combine_norm_f32`
**974 ms / 188** is **not** a thread-count difference and does not contradict the rejection above.
The reference's `ggml_cuda_mmb_hc16()` / `mmb_down16()` / `ggml_cuda_mmb_blk16()` /
`ggml_cuda_mmb_res16()` are **compiled in as `true`** (`ac1ebb4e0`), so its `_b256` kernel reads the
`residual` and `block_out` as **BF16** and writes `out_res` as BF16 (`hc-cn.cu`'s `res_in_bf16` /
`blk_in_bf16` / `res_out_bf16` arms) - roughly half the memory traffic.  Our delivery has `hc16=1`
but `blk16=0` and `res16=0`, and `hc_combine_norm_f32` has no BF16 path at all.

Re-profiled on gfx1151, qwen4exp IQ4_NL, `-b/-ub 8192`, pp8192, r=1:

| kernel | ours | note |
|---|---:|---|
| `hc_combine_norm_f32` (1024, F32) | **973.8 ms / 188** | the delivery default |
| `hc_combine_norm_f32_b256` (256, F32) | **1002.0 ms / 188** | the port - slower |

The real 974 -> ~554 gap is therefore the **HC BF16 traffic** (`blk16`/`res16`), which is a
numerically lossy change (BF16 rounding of the residual/block_out/xn stream), not a kernel-shape
swap.  That is a separate, larger decision for the maintainer; it is not item 1's follow-up.

## Item 5 note (same profile)

At `-b/-ub 8192` (the adopted target) the profile shows **no `concat_transposed_src1_dim0` kernel at
all** - only `gdn_concat_tail` (0.4 ms) / `ple_concat_tail` (0.0 ms).  So item 5's "drop the
`concat_transposed` materialisation" is **already realised** at the target ubatch (the 375 ms figure
was the 2026-09-20 `-ub 16384`/QSA-score-assembly profile).  What remains of item 5 is the BF16 MoE
epilogue: `moe_weighted_reduction_f32_vec4` 369.5 ms / 96 (2.8 % of the profiled total), which the
reference's BF16 variant would roughly halve - but that is the same lossy BF16-intermediate change,
and the reference's own `NO MMB_DOWN16` ablation was **nil (+0.1 %)**.  Treat item 5 as a
memory/lossy-epilogue candidate, not a 3-4 % prefill win.

## Next item after this

With the `_b256` swap closed, the next Phase-1 candidates are **item 4** (`norm-gated.cu`/`rms_rows`,
~1.2 % on our tree — and the same "does it replay our `rms_norm` order?" question applies) and
**item 5** (MoE bf16 epilogue + drop `concat_transposed`, ~3–4 %, beta's `MMB_DOWN16` is gated off).
Item 3.5 (the three QSA correctness fixes) stays an audit.
