# PR #344 — the hc mixer's Q8_0 band kernels (CLOSED: folded into block 11 at r339)

Contributor: `@briansp2020`.  Submission: `wip/hc-mix-q8-r325/` on the delivery repo's PR #344
(`0001-hc-mix-down-dots-serve-every-token-of-the-band-in-on.patch`,
`0002-hc-mix-up-collapse-computes-each-row-gate-directly-t.patch`, `README.md`, `logs/ab-r325.log` — all kept
**verbatim** here).  Cut against r325 (`v17-4f9dfa0e3-r325`); it still applies to the r338 tree (`git am -3`, one
offset in 0002) because the Q8_0 sections of `ggml/src/ggml-cuda/hc-mix.cu` are byte-identical across that range
(the file moved only in its BF16 section).

**Promoted at r339** into **block 11** (the same fold shape as PR #326 at r313, PR #340 into block 06 and PR #342
into block 09): default-on, with `GGML_CUDA_HC_DOWN_MT=0` / `GGML_CUDA_HC_UP_ROWS=0` as the kill switches.

## What was verified rather than assumed (all reproduced here)

* **Per op** (`HC_MIX`, `nt` 1/2/3/4/5/8): 20.20/22.37/27.04/33.13/37.67/51.33 -> 16.65/18.80/19.59/**21.62**/
  24.34/32.67 us/run; `bf16` unchanged; `test-backend-ops -o HC_MIX` **30/30**; the kill-switch arm reproduces
  the base column.
* **Bit-identical output** on every arm that reaches the op: plain greedy, the MTP verify band, a 5246-token
  prompt, **Q4_K_M's Q8_0 inject rows** (which `test-backend-ops` never covers), `HC_MIX_PREQ=0`,
  `HC_UP_FUSE=3`, both kill switches; all ten `gate-hc-fusions.sh` arms read the *same sha* on both builds.
* **The verify band is numerically the same band**: the four-axis `draft-mtp` acceptance census is identical to
  the digit (0.96652 / 0.36678 / 0.63463 / 0.78675, equal accepted/generated totals and mean lengths).
* **Rate**: `tg128` +2.3 % (host-expert shape, diluted), four-axis generation +3.7 to +6.3 %.

Full record: `benchmarks/2026-11-21-hc-mix-band-kernels.md`; `WORKLOG.md` 261121.018; `INVARIANTS.md` r339.

## The one defect: the submission does not build (fixed in the fold)

Five calls passed a **32-bit** mask literal to HIP's warp-shuffle intrinsics —
`__shfl_sync(0xFFFFFFFF, prev, 0, 32)` and four `__shfl_xor_sync(0xFFFFFFFF, …)`.  `hip/amd_detail/
amd_warp_sync_functions.h` rejects that with

```
static_assert(sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly promoting a smaller integer is
almost always an error.")
```

— a **hard compile error** on ROCm **10.1.0**, **10.2.0** and **7.14.1** (each reproduced by compiling the patched
`hc-mix.cu` alone against that line's headers), so the submission's "measured on ROCm 10.1" line cannot describe
this code as submitted.  The fold uses the delivery's wrappers `ggml_cuda_shfl()` / `ggml_cuda_shfl_xor()`, which
pass the 64-bit `GGML_CUDA_FULL_THREAD_MASK` and move the same lanes as the DPP-XOR path `warp_reduce_sum()`
takes, so the values are unchanged — every measurement above was taken with the fixed form.

Three further notes, recorded rather than filed: the module-local `static_assert(vdr == 2 && QI8_0/vdr == 4)` makes
the file uncompilable for any future dense `VDR != 2` (today it is an unconditional `2`); with `HC_UP_ROWS` on
(the default) `GGML_CUDA_HC_UP_FUSE` no longer selects the sub-block count (documented in `ENVIRONMENT.md` §7.7);
and the rows kernel has no `nrows` bound guard, which is safe only under the existing `fuse` predicate
(`n_embd % 32 == 0`, `hc == 4`, `blocks_up == 10`) — the PR states that predicate, so nothing was added.

## Not claimed (same limits as the submission)

NVIDIA/CUDA, Windows, gfx1151/gfx1100 untested; the merged `hc_*_down_inject` (post-#31028) layout with Q8_0 hc
weights has no file here; the BF16 hc path (`hc_mix_bf16_*`) is untouched, so nothing is claimed for it beyond
"its per-op number did not move".
