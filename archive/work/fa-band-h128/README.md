# `fa-band-h128` — contributor PR #329 (@overdoingism): the FA band for 128-dim heads with an odd GQA ratio

**Status: PROMOTED into official block 05 at r314** (`v17-4f9dfa0e3-r314`), **with two changes**, both gated.
Closed: the submission is preserved verbatim under [`pr329/`](pr329/) and nothing here is applied by any
script.  The engineering record is `INVARIANTS.md` r314, `GREEDY-PURITY.md` 96,
`benchmarks/2026-11-18-fa-band-h128.md` and `WORKLOG.md` 261118.009.

## What the submission was

`wip/fa-band-h128/` in the PR: one `git am` file against `v17-4f9dfa0e3-r305` (sha256
`639dcb16aca5dbd83986fa43ddad63d7406799d7dfe3ea0ab98dcfed85873063`, kept here verbatim) plus a README.  It
lets 128-dim heads into the RDNA4 decode/verify band when the GQA ratio is not a multiple of 8 (5, 6, 7,
9..12), in the band's existing 128-dim `ncols1 <= 4` instances and its 32-column block for every query width.
`GGML_HIP_FA_BAND_WMMA_H128=0` was its kill switch, default on.

## Assessment: the kernel work is right

Applies on the r313 tree with a clean `git am` and zero conflicts (`fattn-rdna.cuh` is byte-identical between
r305 and r313).  `test-backend-ops -o FLASH_ATTN_EXT` 7333/7333 in three arms (band, `H128=0`, `WMMA=2`).
Every quoted number reproduced on the primary box within arm-to-arm spread: ratio 7 kv 4096 nb 1 68.1 -> 49.6
us (reviewer: 69.9 -> 50.4), ratio 5 kv 45056 nb 8 1789.5 -> 356.9 (reviewer: 1794 -> 378).  Its KLD claim --
"the band's own difference from the tile kernel is smaller than stock's 16-row kernel against its 1-row
kernel" -- holds under the gate's own family-switch protocol: 0.001013 / 99.412 % against a same-kernel
`-ub 256` control of 0.002222 / 98.676 %.

## The two changes made at review

1. **It is on where it pays, not unconditionally.**  The submission documented the cost as "a loss below 4-8 K
   cells, -5 % at depth 0".  The cost is per *layer*: the band adds a fixed per-launch cost and its
   `P = nsm` split leaves most blocks idle on a short cache, so the crossover is `cells x passes_saved ~ 16-25k`
   -- and a sliding-window/hybrid model holds `n_swa + n_ubatch` cells for its windowed layers for the WHOLE
   run.  ricursive 65B (GQA 10, 64 of 80 layers windowed at 1024) lost **5.8 / 5.1 / 4.9 / 4.5 %** of tg64 at
   d0 / 4K / 16K / 32K, i.e. at every depth, not at depth 0.  With
   `K->ne[1] * (ceil(ratio/ncols2_tile) - ceil(ratio/8)) >= 20000` it reads **0.0 / 0.0 / -0.9 / -0.5 %**
   (`GGML_HIP_FA_BAND_H128_MIN_WORK`).  The attribute is the layer's KV **cache size** and never the used
   length: a verify batch moves the used length by `W-1` rows, so a length-keyed test could put a decode step
   and its own verify on different kernel families (`GREEDY-PURITY.md` 96).
2. **The 2-D launch is guarded on the compiled instance** (`Q->ne[0] == 256 || ncols1 <= 4`): the kernel's band
   fast path is compiled only for the 128-dim `ncols1 <= 4` instances, and a launch with `ncols1 = 8` would have
   fallen through to the ordinary stream-K path with `gridDim.y = P > 1`.  The dispatch can only pick 2 or 4
   today, so this is a structural guard rather than a bug fix.

Also folded in: 48 new `test-backend-ops` cases (ratios 5/6/7/12 x q8_0/f16/**bf16** at kv 32768) so the band
has an oracle in a *default* run -- under `auto` the submission's short-cache cases take the tile kernel -- and
the PR README's "masked 256-cell group at kv 512/1024/4096" corrected to 4096 only (the code was right).

## Not validated here

A non-SWA 128-dim odd-ratio model at a real context (Nanbeige4.3-33B / Qwen2.5-7B / Qwen3-14B): no such file
on this box, so the +16 % at 16K / +36 % at 32K is the contributor's measurement, corroborated only at the op
level (0.48-0.65 for ratio 5/7 from 8192 cells).  gfx1100 / gfx1151 and Windows are untested for this arm, as
they were in the submission, and `--swa-full` is untested (it would raise a windowed layer's cache to the full
context and re-enable the band on mostly-masked rows).
