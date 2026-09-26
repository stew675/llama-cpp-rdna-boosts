# Phase-1 item 7 — keep the M=4 HC inject out of the 384-row tall MMB tile (2026-09-22)

**Status:** DONE (WIP), default **ON**.  Fork branch `gap-closing` @ **`6e5f34ebf`**, exported as
[`patches/0008`](patches/0008-gap-closing-WIP-keep-the-M-4-HC-inject-out-of-the-38.patch).  Bit-identical
and worth **+0.8 % pp8192 / +1.1 % pp32768** at the clean `-b/-ub 4096`.

## The finding (re-profiled `b0f31f587`, as the plan said)

The item was scoped from the profile as "the tall `384x64` tile launches **2×** as often as the
reference's".  Re-profiled at the same config (qwen4exp IQ4_NL, pp8192, `-ub 8192`):

| `mmb_dense_kernel` tile | ours | reference |
|---|---:|---:|
| `<384, 64, 96, 32, 0>` (tall) | 1030 ms / **380** | 577 ms / **190** |
| `<128, 128, 32, 64, 0>` | 1182 / 404 | 809 / 498 |
| `<128, 256, 64, 64, 0>` | 1666 / 168 | 1739 / 168 |

The tall path's gate is **identical** in both trees (`mmb_tall() && IQ4_NL && M <= 384 && K >= 4096 &&
T >= 2048`, `tall_mode == 2`, `grid(1, (T+63)/64)`) and the grids match, so the 2× is not geometry — it
is that the gate admits **two** shapes.  Logging every tall launch (temporarily) showed exactly that:

```
190  M=4   K=10240 T=2048     <- the HC inject
190  M=320 K=10240 T=2048     <- the HC down (the tile's intended user)
```

`M=4` is the hyper-connection **inject** (`[10240 -> 4]`).  The tile is a **384-row** panel (`BM=384`)
with an N of 64, so for `M=4` it (a) computes 4 useful rows out of 384 and (b) launches **twice the
blocks** of the dense 128-row tile for the same `T` (N=64 vs the dense tile's N=128).  The reference
keeps that shape out of the tall path, which is why its tall dispatch count is 190 to our 380.

## The fix

Add a lower bound to the tall gate — `M >= tall_min_m`, default **16**, `GGML_CUDA_MMB_TALL_MIN_M`
overrides (`=0` restores the old behaviour) — so the `M=4` inject falls through to the normal dense
`<128,128>` tile.  16 excludes the inject (M=4) and keeps the real tall-M shape (M=320); the model has
nothing in between.

**Why it is bit-identical:** every MMB WMMA geometry accumulates over K in the same chunk order, so a
different BM/BN tile reproduces the same FP result — the delivery already relies on this ("every valid
geometry reproduces the same same-seed text hash").  Confirmed below.

## Numbers

Kernel profile (pp8192, `-ub 8192`):

| tile | before | after |
|---|---:|---:|
| tall `<384,64,96,32,0>` | 1030 / 380 | **540 / 190** |
| dense `<128,128,32,64,0>` | 1182 / 404 | 1448 / 594 |
| net dense+small family | ~4196 | ~4330 |

The tall half drops 490 ms and the dense `<128,128>` grows 266 ms (it takes the 190 M=4 dispatches at
half the block count), a net **−224 ms** on the M=4 work.  (The `<128,256>` tile reads +360 ms in the
same pair of profiles with an unchanged dispatch count; that is profiler/thermal variance — the
end-to-end A/B below is the ground truth, and it is a clean win.)

End-to-end, `-b/-ub 4096`, r=6 (the memory-safe protocol):

| pp | old (env=0) | new default | delta |
|---:|---:|---:|---:|
| 8192 | 1246.72 ±13.71 | **1256.56** ±12.09 | **+0.8 %** |
| 32768 | 1192.07 ±1.27 | **1205.03** ±2.21 | **+1.1 %** |

## Gates

* Same-seed greedy text **unchanged**: `765 chars sha=f61199ba5644` on both arms.
* `test-logits-width-probe` qwen4exp IQ4_NL, P=2048: **`width_purity=PASS (worst maxdiff 0)`**,
  row0 `8949d53f635c18c3` unchanged.
* The change is qwen4exp-only (the tall gate is IQ4_NL + HC-shape specific) and only affects the
  prefill `T >= 2048` band.

## Next

**Item 8** — audit the 9 QSA graph-side flags vs block-14/15 (small / likely redundant).  Any A/B at
`-b/-ub 4096`.
