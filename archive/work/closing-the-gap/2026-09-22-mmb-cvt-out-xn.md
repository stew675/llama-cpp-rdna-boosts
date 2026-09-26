# `mmb_cvt_f32_bf16` gap closed — the fused combine now emits `out_xn` as BF16 (2026-09-22)

**Status:** done, **default ON**, bit-identical.  Fork branch `gap-closing` @ **`de5689d94`**, exported as
[`patches/0012`](patches/0012-gap-closing-WIP-produce-the-BF16-out_xn-copy-in-the-.patch).

## The gap

Session 5's kernel profile put `mmb_cvt_f32_bf16` at **2088.7 ms (ours) vs 610.3 ms (reference)** —
identical kernel and launch, but ours converted **1040 calls vs its 1544 with ~5× the elements per
call**.  The handover filed it as "non-lossy; our calls convert far larger tensors (activation cache /
`mmb_root` keying)".

## Root cause

`LLAMA_MMB_CVT_LOG` (now with `data=`/`root=` and a value-driven cap) on the pp8192 prefill showed the
traffic is **81 % `hc_norm`**: 372 conversions / **1.56e10 elements** of the `[10240, 4096]` HC
normalized stream, versus 3.6e9 for `final_output` and 4.2e7 for `ple_embd`.

Two facts made the activation cache useless here:

1. **The graph allocator reuses one `hc_norm` buffer across layers** — every `hc_norm-N` conversion
   logged the same `data=0x…41100`.  Keying the cache on `data` alone would hand layer *N+1* layer
   *N*'s stale BF16 copy, so the per-layer `root` in the key is load-bearing; the cache cannot dedupe.
2. **Each layer's `hc_norm` is reconverted once per consumer** (two MMB GEMMs, presented through
   different reshape/mul roots), and `cache_max` is 4 entries, so nothing survives.

The real defect is upstream of the cache: the graph optimizer **already marks `out_xn` BF16-only**
(`all_bf16_consumers`: every consumer reads a BF16 copy — the MMB GEMMs via `mmb_bf16_activation`,
`dsv4_hc_pre` via its `x16` arm), but the **fused `hc_combine_norm` was the one producer that did not
emit the copy**.  Every consumer therefore fell back to converting the F32 on the fly.

## Fix

`ggml_cuda_hc_combine_norm_set_bf16` now emits the copy whenever the mark is present, independent of
the `blk16`/`res16` gates:

```c
if (ggml_cuda_mmb_active() && ggml_cuda_mmb_is_bf16_only(args.out_xn)) {
    args.out_xn_bf16  = ggml_cuda_mmb_reserve_auto(ctx, args.out_xn, ggml_nelements(args.out_xn));
    args.store_xn_f32 = args.out_xn_bf16 == nullptr;
}
```

* `ggml_cuda_mmb_reserve_auto` honours the slot the graph assigned (`all_bf16_consumers` pins slot 4
  for the `dsv4_hc_pre` consumer), so the copy is not clobbered by the next generic producer.
* `hc_f2bf32` is the exact RNE formula as `mmb_f2bf`, so the MMB GEMM consumers are **provably
  bit-identical**.
* The **unfused** path already did this (`norm.cu::rms_norm_mul` honours `is_bf16_only` and drops the
  F32), so the change makes the fused path consistent with it rather than inventing new numerics.

## Gates (qwen4exp IQ4_NL, gfx1151)

| gate | before | after |
|---|---|---|
| width probe W=1..8 + `row0_row1_hashes` | `1:268e0673300b7a33/…`, PASS | **identical**, PASS |
| same-seed 128-token text (`prose-rdna-boosts.txt`) | `b639e324753d` (617 chars) | **identical** |
| unfused combine (`GGML_CUDA_DISABLE_HC_COMB=1`) | `b639e324753d` | **identical** |
| `plain == draft-mtp n3` (434 chars) | `984263fb8e0f` | **identical** |
| `mmb_cvt` calls / elements (pp8192) | 520 / 1.93e10 | **148 / 3.67e9** |

## Performance — `-b/-ub 4096`, `-p 8192,32768 -n 0 -r 3`

| | pp8192 | pp32768 |
|---|---:|---:|
| before | 1252.4 | 1206.5 |
| after | **1294.1** | **1244.7** |
| Δ | **+3.3 %** | **+3.2 %** |

This is on top of `patches/0011` (`blk16`/`res16`) and is independent of it (both `out_xn` and the
residual/`block_out` streams now stay in BF16 end to end).

## Remaining `mmb_cvt` traffic

`final_output` (3.6e9 elements, 4 conversions per name) is the bulk of what is left.  It is not the
`hc_norm` class: its data pointers are distinct per conversion, so it is a genuine per-consumer
conversion.  Skipping it would need a producer-side BF16 mark on the MoE/FFN output, which is a
separate (lossy) decision — left as a follow-up.
