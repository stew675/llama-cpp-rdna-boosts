# OP-3 — per-type MoE MMVQ band on gfx1201 (RDNA4): **no per-type band**

**Date:** 2026-09-24 · **Box:** soar (3× Radeon AI PRO R9700, gfx1201, Ryzen 9 9950X3D2) · **Build:**
`~/llama.cpp` `closing-gfx1201` tree `99b429a60d441f814c84737cfa57803bc15a2f6d` (r13 + beta +
closing `0001..0014`/`0016..0030`) · `-sm tensor`, `GGML_CUDA_ALLREDUCE=hybrid`.

**Verdict.**  The unconditional routed-expert MMVQ band floor (`mmvq_mmid_max_batch_band()` clamping
every per-type cap up to `MMVQ_MOE_MAX_BATCH_SIZE = 16`, patch `0028`) is a **net win for every expert
type in the relevant verify range** (`n_max <= 12` → `B <= 13`).  Some i-quants lose at `B >= 13`, but
that is the top of the range the maintainer deems irrelevant and the price of the decode/verify purity
the band exists to provide.  **No per-type band is warranted; close OP-3 as a no-op.**  The
`__launch_bounds__` widening trap is also cleared: it does not regress the `B <= 8` decode band.

---

## 1. What is actually being compared

`0028` changed `mmvq_mmid_max_batch_band()` from `max(cap, MMVQ_MAX_BATCH_SIZE = 8)` to
`max(cap, MMVQ_MOE_MAX_BATCH_SIZE = 16)`.  Because the floor is applied *after*
`get_mmvq_mmid_max_batch_rdna4(type)` (whose caps are all `<= 8`), the per-type RDNA4 table is
**entirely dead** for the MUL_MAT_ID mmvq-vs-MMQ decision — every type gets 16.  So "per-type band"
would mean making the floor type-aware again, i.e. returning the *measured* optimum per type instead
of a flat 16.

The only A/B the code offers is the global kill-switch `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1`, which
clamps every type to the dense band 8.  Comparing "band ON (floor 16)" vs "band OFF (floor 8)"
per model therefore reports, for that model's dominant expert types, whether mmvq `@9..16` beats MMQ
above 8 columns.  That is the decision the per-type question turns on.

**Harness** (`llama-batched-bench`, `S_TG t/s`; 2 interleaved reps per arm, `on/off/on/off`):

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
cd ~/llama.cpp
M=<model>.gguf
for arm in on off on off; do
  [ $arm = off ] && export GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1 || unset GGML_CUDA_DISABLE_MMVQ_MOE_BAND
  timeout 900 build-rocm/bin/llama-batched-bench -m "$M" -ngl 99 -sm tensor -c 8192 -b 2048 -ub 2048 \
    -npp 16 -ntg 32 -npl 8,12,13,14,15,16 > /tmp/bb_${arm}.out 2>/dev/null
done
```

Raw logs: `tools/runs/op3-band/*.out` (git-ignored).

## 2. Per-type result (band ON vs OFF, `S_TG t/s`)

Model -> its **routed-expert** types (from the GGUF, `ffn_*_exps` only):

| model | routed-expert types (count) |
|---|---|
| 35B True-Q3_K_M | Q3_K (100) + Q4_K (20) |
| 35B UD-Q3_K_M | IQ3_XXS (78) + IQ4_XS (39) |
| 35B UD-Q4_K_M | Q4_K (82) + Q5_K (38) |
| 35B UD-Q5_K_M | Q5_K (82) + Q6_K (38) |
| qwen4exp IQ4_XS | IQ3_S (94) + IQ4_NL (43) + IQ4_XS (2) |
| qwen4exp IQ4_NL | IQ4_NL (144) |

`delta % = (band ON - band OFF) / band OFF`:

| model (types) | B=8 | B=9 | B=10 | B=12 | B=13 | B=14 | B=15 | B=16 |
|---|---|---|---|---|---|---|---|---|
| Q4_K+Q5_K | -0.2 | **+26.2** | +26.1 | +21.3 | | | | +10.4 |
| Q5_K+Q6_K | -0.3 | | | +15.6 | +11.7 | +9.8 | +6.9 | +4.3 |
| IQ4_NL (qwen4exp, ctk/ctv q8_0) | -0.2 | +13.4 | +9.5 | +6.2 | | +2.2 | | +0.4 |
| Q3_K+Q4_K | -0.2 | | | +2.5 | +0.2 | -1.1 | -1.9 | -4.0 |
| IQ3_XXS+IQ4_XS | +0.2 | +2.8 | +1.2 | -0.2 | -2.2 | -3.5 | -4.6 | -6.3 |

`B = 8` is the control (both arms use mmvq there — identical, as expected).

**Reading it per type:**

* **k-quants `Q4_K`/`Q5_K`/`Q6_K`** — the band is a large win over the whole range (`+26%` at `B=9`,
  `+4..+11%` at `B=16`).  Keeping the floor at 16 is clearly right.
* **`IQ4_NL`** (the model `0028` was cut on) — a win at every width (`+13%` at `B=9` tapering to
  `+0.4%` at `B=16`).  Right.
* **`IQ3_XXS`/`IQ4_XS`** — a small win at `B=9..10` (`+2.8%`/`+1.2%`), neutral at `B=12`, then a
  monotonic **loss** from `B=13` (`-2.2%`) to `B=16` (`-6.3%`).  This is the only regression.
* **`Q3_K`** — mixed (`+2.5%` at `B=12`, `-4.0%` at `B=16`), consistent with the i-quant shape.

The maintainer scopes **`n_max <= 12` (i.e. `B <= 13`)** as relevant.  In that range the ONLY negative
is IQ3_XXS/IQ4_XS at `B=13` (`-2.2%`); everything else is a win or neutral.  A per-type cap at 12
would recover that 2.2% but would send `B=13` to MMQ, breaking the decode (`B=1`, mmvq) vs verify
(`B=13`, MMQ) reduction-order match that the band exists to guarantee.  Not a good trade.

## 3. Kernel-level confirmation (why the i-quants regress)

`rocprofv3 --kernel-trace` on 35B UD-Q3 (IQ3_XXS + IQ4_XS) at `B=16`, `-npp 16 -ntg 16 -npl 16`,
totals by kernel (`tools/runs/op3-band/trace_udq3_b16_*`):

| arm | routed-expert kernel time |
|---|---|
| band ON | `mul_mat_vec_q_moe<18,2>` 0.2081 s + `<23,8>` 0.0697 s + `<23,2>` 0.0021 s + `<14,8>` 0.0088 s = **0.2887 s** |
| band OFF | `mul_mat_q<18,16>` 0.1918 s + `mul_mat_q_routed_compact<23,16>` 0.0644 s + `<14,16>` 0.0076 s = **0.2638 s** |

The mmvq path is ~9.4% slower than MMQ at 16 columns for these types — the aggregate `-6.3%` is
exactly the routed-expert kernel, not a side effect.  The kernel is `block_dims = (warp_size,
ncols_dst)` — one warp per token, so at `ncols_dst=16` the block is 16 warps and the widened
`__launch_bounds__(512,1)` halves the per-thread register budget; i-quant dequant is the
register-heaviest, which is why only the i-quants pay.  **If the loss ever needs recovering** (e.g. the
maintainer raises the relevant ceiling past 12), the purity-preserving fix is a kernel change: split
the token dimension into two 8-warp blocks (`block_dims.y = min(ncols_dst, 8)`, token offset from
`blockIdx.z`) so the 8-warp launch bound applies again.  Each warp still computes one token with the
same per-token reduction order, so `W = 1..16` stays bit-identical.  A per-type *cap* is NOT that
fix — it changes the reduction order at the boundary.

## 4. Trap cleared: the `__launch_bounds__` widening does not hurt the `<= 8` decode band

The `0028` launch bound went from `MMVQ_MAX_BATCH_SIZE*32` (256) to `MMVQ_MOE_MAX_BATCH_SIZE*32`
(512).  Because the kernel is compiled with the worst case, the smaller budget could in principle
have cost occupancy at `B <= 8` too.  Measured two ways:

**(a) Static (register allocation).**  Extracted both variants' `gfx1201` code objects (`roc-obj
extract` + `llvm-readelf --notes` on `.hip_fatbin`): all **138** `mul_mat_vec_q_moe` instantiations
have **identical `vgpr_count`, `sgpr_count` and `private_segment_fixed_size`** in both builds.  The
only kernel-descriptor difference is `max_flat_workgroup_size` (`512` vs `256`), i.e. the launch-bound
metadata itself.  (A handful of instruction immediates in the unrolled item loop differ; they are
immaterial — see (b).)

**(b) Runtime.**  Two `libggml-hip.so` variants, `.so` swapped with `LD_PRELOAD`; the narrow build is
run with `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1 GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1` so its 16-token
warmup does not launch a 9..16-warp `mul_mat_vec_q_moe` (which would otherwise fail, as designed).
At `B <= 8` both arms dispatch mmvq, so this isolates the launch bound.  `-npp 16 -ntg 32
-npl 1,2,4,8`, 4 interleaved reps:

| model | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|
| UD-Q3 (i-quants), LB16 vs LB8 | -0.22% | +0.61% | +0.45% | -0.24% |
| UD-Q4 (k-quants), LB16 vs LB8 | -0.57% | +1.05% | +0.06% | -0.29% |

All within ±0.6% with no systematic trend (the same-build bands-ON vs bands-OFF control also matches
in that band), so **the launch-bound widening is a no-op for the `<= 8` decode/verify path**.  The
build was restored to the delivered `MMVQ_MOE_MAX_BATCH_SIZE` launch bound and the tree is clean
(`git status` empty, `libggml-hip.so` md5 matches the pre-experiment variant).

## 5. Follow-ups

* **None required.**  `0028` stands as delivered; keep the unconditional floor.
* If the relevant `n_max` ceiling is ever raised: implement the two-8-warp-block kernel split, not a
  per-type cap (see §3).  Gate it with the `W = 1..16` width-purity probe and the rule-5 batched gate.
* The dead per-type RDNA4 table is harmless but note it in any future cleanup of
  `get_mmvq_mmid_max_batch_rdna4` (it no longer influences the MUL_MAT_ID dispatch on RDNA4).
