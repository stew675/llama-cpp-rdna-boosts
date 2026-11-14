# Upstream-PR proposal — ggml-cuda: do not pay the padded-head clamp on an unpadded FA head

> **Re-base status (2026-11-14):** cut against the current delivery base **`4f9dfa0e3`**; `git apply --check`
> clean there and nothing else in the file moved.  Not part of the delivery (the delivery carries it inside
> **block 05**).

> **Upstream candidate** · status: measured 2026-11-14 on gfx1201 (1 × R9700) and mirrored in block 05 ·
> not filed.
> Delivery: **block 05** (r281, GitHub issue #311) ·
> patch: [`UPSTREAM-PR-fa-tile-loader-min.patch`](UPSTREAM-PR-fa-tile-loader-min.patch) (1 file,
> `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, +7/−2).

**Value: 12.4 % of the FA prefill kernel back on every unpadded head.**  `#30285` ("cuda: fattn mma head sizes 40
72") added padded-head support and wrote the K/V tile loader's bounds as `min(k0_stop, DKQ/2)` (`k0_diff`) and
`min(i0_stop, DV)` (`i0_diff`) so a padded head (40, 72) never reads past its real head size.  For a head whose
size is a multiple of 16 the clamp cannot bind, but it still changes instruction selection for the whole
`flash_attn_ext_f16` instantiation: on RDNA4 head 256 (`ncols1=32, ncols2=2`, the prefill instance) the kernel
compiles to **108 759** instructions instead of **107 342** and takes **5 108 ms instead of 4 544 ms** for the same
992 dispatches.  End to end that is `llama-bench -p 4096 -d 32768`, dense Qwen3.8-27B Q5_K_XL, `q8_0` K/V,
`-ub 512 -b 1024`: **839.4 -> 876.0 t/s (+4.4 %)**, `pp512 @ d0` 1217.6 -> 1254.4, and `pp512 @ d32768`
842.0 -> 881.6.  The f16-KV arm moves by the same 4 % (`pp4096 @ d32768` 846.2 -> 881.1), so this is not the
quantized-cache path.

## Root cause

```cuda
const int k0_stop = k0_start + nbatch_K2 < DKQp/2 ? k0_start + nbatch_K2 : DKQp/2;
...
- const int k0_diff = min(k0_stop, DKQ/2) - k0_start;   // = k0_stop - k0_start when DKQp == DKQ
+ const int k0_diff = DKQp == DKQ ? k0_stop - k0_start : min(k0_stop, DKQ/2) - k0_start;
```

`DKQp`/`DVp` are `GGML_PAD(head, 16)`, so for head 256 the two spellings compute the identical value on every
iteration; what changes is what the optimizer does around the loop that loads a K/V tile (`load_tile_rows`
compiles to 4 755 instructions instead of 4 594, the loop loses a packed-16-bit form for a 32-bit one and gains
139 `s_delay_alu` scheduling waits).  The patch keeps upstream's expression verbatim for the padded arm and spells
the unpadded arm as a constant-folded plain subtraction, which restores the pre-`#30285` code exactly (verified:
ISA instruction counts equal to the parent of `#30285`, symbol by symbol across the whole TU).

## Validation

* Device ISA, per symbol, from the compiled instance TU (`template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2-dkq256-dv256.cu.o`,
  `gfx1201` code object, `llvm-objdump -d`): patched file **107 342** instructions in
  `flash_attn_ext_f16<256,256,32,2,false,…>` and **4 594** in `flash_attn_ext_f16_load_tile_rows<132,…>`, i.e.
  identical to the same TU compiled from the tree before `#30285`.  Unpatched: 108 759 / 4 755.
* Header-hunk bisect (34 hunks, `#30285`'s parent -> `4f9dfa0e3`) against those counts: only this hunk moves them.
  The heads-40/72 config rows, the `mma_tile_sizes<40|72>` specializations, the `DECL_…` lines and the
  `DKQp/DVp` padding each removed in isolation leave the counts unchanged.
* Launch geometry and resources are identical patched vs unpatched (`nthreads 256`, `nbatch_fa 64`,
  `nbatch_K2/V2 128`, `nbatch_combine 64`, `nstages 0`, shmem 38 400, grid 6 144 × 8, VGPR 256 / SGPR 128), so the
  win is instruction selection, not a configuration change.
* Numerics: **bit-identical**.  `test-backend-ops -o FLASH_ATTN_EXT` **6997/6997** (the padded 40/72 instances are
  the reason the clamp exists and they are covered by that run); prefill-logit gate mean KLD **0.000697** /
  same-top-p **98.784 %** (f16 K/V) and **0.000580 / 98.882 %** (`q8_0` K/V), the same numbers the unpatched tree
  reads against the same bases; `test-logits-width-probe` (`P=3584`) PASS with worst maxdiff 0; same-seed greedy
  text unchanged (`196 chars sha=83eec5e9b4f0`, Qwen3.5-4B Q8_0, 1 GPU and 3 GPU `-sm tensor`).
* Decode is unaffected: `llama-bench tg128 @ d32768` 25.22 -> 25.21 t/s, and the decode/verify band kernel
  (`flash_attn_ext_f16<256,256,4,8,…>`) is not perturbed by the change (320.6 vs 321.9 ms over 2 064 dispatches).
* Record: `benchmarks/2026-11-14-fa-prefill-min-codegen.md` (delivery repo).

## What was NOT validated

* Non-AMD backends (out of scope for this repo): the change is pure C++/CUDA-source and reaches every backend
  that compiles the MMA FA path, but only gfx1201/ROCm 10.1.0 was measured.
* gfx1100/gfx1151: expected to behave the same (the same instantiation shape), not measured.
* Head sizes 40/72 keep upstream's expression character for character, but their own codegen was not
  instruction-counted before/after (they compile with `DKQp != DKQ`, so the `min()` arm is the one taken).
