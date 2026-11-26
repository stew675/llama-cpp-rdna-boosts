# WORKLOG — the dated delivery record

**Today's entries are here, newest first.  Every earlier day is in [`archive/worklog/`](archive/worklog/), one
file per day (`WORKLOG-YYYYMMDD.md`, newest first inside the file), and the table at the end of this file indexes
them.**  That keeps the current day in front of you without a second place to look for it, and the record still
has exactly one entry point.

**Recording work.**  Add the entry at the top of this file with the id `YYMMDD.VVV`: `YYMMDD` is the real local
calendar date, and `VVV` is a per-date sequence that increases in file order, so the topmost entry of a day
carries the highest number.  Never invent or extrapolate a future date, and never reuse an id.

**At the first entry of a new day**, move the previous day's entries into
`archive/worklog/WORKLOG-<that day>.md` (file header in the same form as its neighbours) and add its row to the
index below.  That is the whole roll-over: one move, one row.

A reference written `WORKLOG.md` 261124.012 means entry `261124.012` — it is in this file if it is today's,
otherwise look its day up in the index.

## 261126.002 (r366) -- confirmed on the x16 box, the canonical tip reproduced on a second machine, and the #367 slab experiment backed up before the tree was re-cut there

**The measurement.**  `radix` (1 x R9700, uncontended PCIe 5.0 **x16**, its own H2D probe **55.0 GB/s** against this
box's 28.7) at the reporter's shape, three server visits per arm alternating with a cold heat prior:
**r366 L1 2262.9 t/s / 44.78 tg, S2 60.10, L2 44.84, stand-downs `L47x3`, arena 20 224 MiB**; the
`GGML_CUDA_SLAB_RING_MIB=6144` layout 2217.5 / 44.11 / 55.89 / 45.89, stand-downs `L31..L47x3` = 51 tables, arena
12 288 MiB.  That is **+2.0 % prefill / +1.5 % decode / +7.5 % S2 decode / -2.3 % L2 decode** (the L2 arm has one
43.23 outlier; its other two visits read 45.64/45.66 against 45.89, i.e. -0.5 %).  cli arms on the same box agree
and are wider apart: **2359.3 / 2341.6** t/s and 42.0 / 42.2 tg against the hole's 2264.0 / 2259.5 and 37.3 / 37.2
(**+4.1 % / +13 %**).  Texts identical in both layouts (`L1 98c4f67d6ed9`, `L2 4b2af40814aa`), residency
`L1..L48 285/512` in both, `aux_refused=0` and `work=3904 MiB (narrow=1792)` against the hole's 11 840.  Prefill's
gain is roughly half the x8 box's (+2.0 % against +3.7 %), the direction the 2026-11-25 record established: the
narrower the link, the more a whole-table stand-down costs.  The flip is still not reproduced there either.

**The tip, on a second machine.**  `radix`'s tree was re-cut from `patches/` with `scripts/apply-all.sh` (a clean
checkout of `919794500`) and read tip **`489810652`** / tree **`82afebd10`** -- the third independent reproduction of
`release.json.tip` and the first on another box, which is what the `.tip` contract buys.

**The #367 collision (a finding for the maintainer).**  `radix`'s `~/llama.cpp` carried an **uncommitted** in-flight
slab experiment (issue #367's `GGML_CUDA_SLAB_ARENA_CAP_MIB`, 62 lines in `ggml-cuda-slab.cu` -- the same file r366
changes).  The maintainer confirmed it is backed up and released the tree; it was copied to
`~/scratch/issue367-slab-experiment-backup.diff` on that box before the checkout, and that campaign must now rebase
onto r366.  Nothing on `radix` was left dirty: checkout `919794500`, `apply-all.sh`, built, measured.

**Record.**  `benchmarks/2026-11-26-issue-368-wide-band-coupling.md` section 4.

## 261126.001 (r366) -- the slab's WIDE band is armed by the PREFILL PASS, not by the carved ring hole (GitHub issue #368, the r362 follow-up): r365's pool-tier default silently disabled every prefill transient's home in the slab

**The report.**  `@DanoPTT` re-ran issue #368 on r365 with the `MOE_EXPERT_CACHE_TABLES=1` instrument and 16
alternating visits (their `N` = r365 default, their `R` = the same binary with `GGML_CUDA_SLAB_RING_MIB=6144`).  `R`
reproduces both of their states exactly as on r345 (good 2921-2953 t/s prefill / 51.7-52.8 tg, slow 1613-1631 /
31.7-32.2, two texts), and the stand-down/residency maps show the cache's resident set is *not* the carrier (the
residency line is byte-identical in both states).  Their `N` is 72-80 % slower at prefill (**821-825 and 577 t/s**),
its decode is fine or better, and its ROCm0 compute buffer is **reallocated 1536 -> 2875 MiB** while `R` never
reallocates.  A second comment rules the MEC-firmware workaround out (`GPU_MAX_HW_QUEUES=2` leaves the flip
intact) and they ask (2) what the wide prefill view does on the pool tier without the hole and (3) which per-start
choice could be printed.

**What they found for us.**  r362's hole removal is one line, and it was right about the hole, but the hole was also
the *definition* of the slab's wide band: `ring_armed` -- which `ggml_cuda_slab_wide_available()` and
`ggml_cuda_slab_wide_aux_alloc()` gate on -- is `ring_region > 0 && ...`, so zeroing the region made the band
structurally impossible, and the prefill view was then classified *narrow* (the size/mark test also required a
carved region) and learned `narrow_floor` as its own size.  Measured on their exact shape here (1 x R9700, the
28.7 GB/s x8 box, `-ncmoe 47 --host-experts pool -c 131072 -b 8192 -ub 8192`, an 89 182-token L1): **924
`SLAB_WIDE_AUX` refusals in one prefill** and `work=3328 MiB (narrow=3328)` at the drop, against 0 refusals and
`narrow=1792` with a ring.  The FA staging arena, the CUDA pool's large prefill allocations, the MMB activation
cache and the chunked-GDN/QSA3 wide scratch all left the slab for the ~1536 MiB outside cushion.  Their
"1536 -> 2875 MiB" is the second half of that (the band being learned from the prefill view).

**The fix, folded into block 10** (the block that owns the slab -- a bug fix inside the owning block, so no new
block and no renumbering; the maintainer approved the fold and asked for the r362 headline to be re-measured).
`ring_armed` becomes `wide_armed` and is armed by `ring_requested` + the boundary covering the LIVE view, wherever
that view was placed; `ggml_cuda_slab_wide_place()` is the single placement rule (above a carved ring, the narrow
base when none is) and `ggml_cuda_slab_seed_floor` uses it too, so a no-ring slab neither moves the wide view nor
lets a prefill pass learn the narrow floor.  The ring hole is untouched: `ring_slot_capacity()` and
`arena_alloc_transient()` still require `ring_region > 0`, so a pool tier keeps staging nothing and r362's win
stands.

**Measured, after.**  cli arms on their shape (one process each, the model directory's heat prior loaded):
r365 default 2073.3 / 2076.8 t/s prefill and 30.8 / 30.7 tg; the hole layout 1983.2 / 1981.5 / 1991.9 and 27.5 /
27.1 / 27.1; **r366 2106.6 / 2107.4 / 2117.9 and 32.2 / 32.8 / 32.7**.  That is **+1.8 % prefill / +6 % decode**
against r365 and **+6.4 % / +19 %** against the hole.  Server visits in the 2026-11-25 record's shape (S1 -> L1 ->
S2 -> L2, three per arm alternating, a cold heat prior): r366 **1959.5 t/s L1 prefill / 34.37 tg**, S2 46.33, L2
34.61, stand-downs `L47x2`, arena 20 352 MiB; the hole 1888.9 / 33.74, S2 40.15, L2 35.02, stand-downs
`L31..L47x3` = 51 whole tables, arena 12 416 MiB -- **+3.7 % / +1.9 % / +15.4 % / -1.2 %**.  Text identical at
every visit point in both layouts (`L1 98c4f67d6ed9`, `L2 4b2af40814aa`), and `MOE_EXPERT_CACHE_TABLES=1` reads
`L1..L48 261/512` in both.  The reporter's flip is still not reproduced here (three fresh visits per arm).

**x16 box:** measured afterwards -- `WORKLOG.md` 261126.002, `benchmarks/…wide-band-coupling.md` section 4
(+2.0 % prefill / +1.5 % decode / +7.5 % S2 there, and the canonical tip reproduced on that machine).

**Gates.**  `validate-set.sh` green (base `919794500`, applied tree
`82afebd108d63b6220884d41080009561f0821e2`, both optional artifacts strict on the new tree, rung reconciliation
`0 unexplained` -- the change adds no file, define, env var or option); `gate-host-pool.sh` **59/59**;
`gate-kv-residency.sh` PASS; `gate-prefill-logits.sh` **0.000000 / 100.000 %**; `gate-host-expert-text.sh` 3/3
`7c5a512a8447`; `check-shlib-symbols.sh` 16/16; `check-gen-docs.sh` ok; `check-gpu-faults.sh` clean; the `all`
build green at the chain tip.

**The issue.**  Answered with the workaround (`GGML_CUDA_SLAB_RING_MIB=6144` until the release), the mechanism in
their own log's terms (the compute-buffer 1536 -> 2875 line is the band being learned), and a request to retest and
send the good/slow pair with `GGML_CUDA_SLAB_STATS=1 -lv 4` if anything remains; the same comment names the
`H2D bandwidth calibration` line (a measured per-start input, and on Windows a 512 MiB probe that is not returned,
issue #97) as the next per-start suspect for the flip itself.

**Record.**  `benchmarks/2026-11-26-issue-368-wide-band-coupling.md`; `INVARIANTS.md` r366; `patches/README.md`
block 10; `DIAGNOSTIC.md` section 7's `GGML_CUDA_SLAB_STATS` row.

## Archive index — days before today

| day | entries | ids | file |
|---:|---:|---|---|
| 2026-11-25 | 16 | `261125.001-261125.016` | [`WORKLOG-20261125.md`](archive/worklog/WORKLOG-20261125.md) |
| 2026-11-24 | 17 | `261124.001-261124.017` | [`WORKLOG-20261124.md`](archive/worklog/WORKLOG-20261124.md) |
| 2026-11-23 | 27 | `261123.001-261123.027` | [`WORKLOG-20261123.md`](archive/worklog/WORKLOG-20261123.md) |
| 2026-11-22 | 40 | `261122.001-261122.040` | [`WORKLOG-20261122.md`](archive/worklog/WORKLOG-20261122.md) |
| 2026-11-21 | 22 | `261121.001-261121.022` | [`WORKLOG-20261121.md`](archive/worklog/WORKLOG-20261121.md) |
| 2026-11-20 | 7 | `261120.001-261120.007` | [`WORKLOG-20261120.md`](archive/worklog/WORKLOG-20261120.md) |
| 2026-11-19 | 8 | `261119.001-261119.008` | [`WORKLOG-20261119.md`](archive/worklog/WORKLOG-20261119.md) |
| 2026-11-18 | 12 | `261118.001-261118.012` | [`WORKLOG-20261118.md`](archive/worklog/WORKLOG-20261118.md) |
| 2026-11-17 | 9 | `261117.001-261117.009` | [`WORKLOG-20261117.md`](archive/worklog/WORKLOG-20261117.md) |
| 2026-11-16 | 8 | `261116.001-261116.008` | [`WORKLOG-20261116.md`](archive/worklog/WORKLOG-20261116.md) |
| 2026-11-15 | 9 | `261115.001-261115.010 (9; .003 absent)` | [`WORKLOG-20261115.md`](archive/worklog/WORKLOG-20261115.md) |
| 2026-11-14 | 10 | `261114.001-261114.010` | [`WORKLOG-20261114.md`](archive/worklog/WORKLOG-20261114.md) |
| 2026-11-13 | 16 | `261113.001-261113.016` | [`WORKLOG-20261113.md`](archive/worklog/WORKLOG-20261113.md) |
| 2026-11-12 | 10 | `261112.001-261112.010` | [`WORKLOG-20261112.md`](archive/worklog/WORKLOG-20261112.md) |
| 2026-11-11 | 37 | `261111.001-261111.037` | [`WORKLOG-20261111.md`](archive/worklog/WORKLOG-20261111.md) |
| 2026-11-10 | 14 | `261110.001-261110.014` | [`WORKLOG-20261110.md`](archive/worklog/WORKLOG-20261110.md) |
| 2026-11-09 | 12 | `261109.001-261109.012` | [`WORKLOG-20261109.md`](archive/worklog/WORKLOG-20261109.md) |
| 2026-11-08 | 17 | `261108.001-261108.017` | [`WORKLOG-20261108.md`](archive/worklog/WORKLOG-20261108.md) |
| 2026-11-07 | 14 | `261107.001-261107.014` | [`WORKLOG-20261107.md`](archive/worklog/WORKLOG-20261107.md) |
| 2026-11-06 | 22 | `261106.001-261106.022` | [`WORKLOG-20261106.md`](archive/worklog/WORKLOG-20261106.md) |
| 2026-11-05 | 20 | `261105.001-261105.020` | [`WORKLOG-20261105.md`](archive/worklog/WORKLOG-20261105.md) |
| 2026-11-04 | 8 | `261104.001-261104.008` | [`WORKLOG-20261104.md`](archive/worklog/WORKLOG-20261104.md) |
| 2026-11-03 | 15 | `261103.001-261103.015` | [`WORKLOG-20261103.md`](archive/worklog/WORKLOG-20261103.md) |
| 2026-11-02 | 22 | `261102.001-261102.020` | [`WORKLOG-20261102.md`](archive/worklog/WORKLOG-20261102.md) |
| 2026-11-01 | 20 | `261101.001-261101.021 (20; .006 absent)` | [`WORKLOG-20261101.md`](archive/worklog/WORKLOG-20261101.md) |
| 2026-10-31 | 16 | `261031.001-261031.016` | [`WORKLOG-20261031.md`](archive/worklog/WORKLOG-20261031.md) |
| 2026-10-30 | 9 | `261030.001-261030.009` | [`WORKLOG-20261030.md`](archive/worklog/WORKLOG-20261030.md) |
| 2026-10-29 | 8 | `261029.001-261029.008` | [`WORKLOG-20261029.md`](archive/worklog/WORKLOG-20261029.md) |
| 2026-10-28 | 14 | `261028.001-261028.014` | [`WORKLOG-20261028.md`](archive/worklog/WORKLOG-20261028.md) |
| 2026-10-27 | 11 | `261027.001-261027.011` | [`WORKLOG-20261027.md`](archive/worklog/WORKLOG-20261027.md) |
| 2026-10-26 | 8 | `261026.001-261026.008` | [`WORKLOG-20261026.md`](archive/worklog/WORKLOG-20261026.md) |
| 2026-10-25 | 17 | `261025.001-261025.017` | [`WORKLOG-20261025.md`](archive/worklog/WORKLOG-20261025.md) |
| 2026-10-24 | 12 | `261024.001-261024.012` | [`WORKLOG-20261024.md`](archive/worklog/WORKLOG-20261024.md) |
| 2026-10-23 | 6 | `261023.001-261023.006` | [`WORKLOG-20261023.md`](archive/worklog/WORKLOG-20261023.md) |
| 2026-10-22 | 12 | `261022.001-261022.012` | [`WORKLOG-20261022.md`](archive/worklog/WORKLOG-20261022.md) |
| 2026-10-21 | 16 | `261021.001-261021.016` | [`WORKLOG-20261021.md`](archive/worklog/WORKLOG-20261021.md) |
| 2026-10-20 | 15 | `261020.001-261020.015` | [`WORKLOG-20261020.md`](archive/worklog/WORKLOG-20261020.md) |
| 2026-10-19 | 15 | `261019.001-261019.015` | [`WORKLOG-20261019.md`](archive/worklog/WORKLOG-20261019.md) |
| 2026-10-18 | 10 | `261018.001-261018.010` | [`WORKLOG-20261018.md`](archive/worklog/WORKLOG-20261018.md) |
| 2026-10-17 | 8 | `261017.001-261017.008` | [`WORKLOG-20261017.md`](archive/worklog/WORKLOG-20261017.md) |
| 2026-10-16 | 12 | `261016.001-261016.012` | [`WORKLOG-20261016.md`](archive/worklog/WORKLOG-20261016.md) |
| 2026-10-15 | 6 | `261015.001-261015.006` | [`WORKLOG-20261015.md`](archive/worklog/WORKLOG-20261015.md) |
| 2026-10-14 | 8 | `261014.001-261014.008` | [`WORKLOG-20261014.md`](archive/worklog/WORKLOG-20261014.md) |
| 2026-10-13 | 7 | `261013.001-261013.007` | [`WORKLOG-20261013.md`](archive/worklog/WORKLOG-20261013.md) |
| 2026-10-12 | 7 | `261012.001-261012.007` | [`WORKLOG-20261012.md`](archive/worklog/WORKLOG-20261012.md) |
| 2026-10-11 | 18 | `261011.001-261011.018` | [`WORKLOG-20261011.md`](archive/worklog/WORKLOG-20261011.md) |
| 2026-10-10 | 4 | `261010.001-261010.004` | [`WORKLOG-20261010.md`](archive/worklog/WORKLOG-20261010.md) |
| 2026-10-09 | 3 | `261009.001-261009.003` | [`WORKLOG-20261009.md`](archive/worklog/WORKLOG-20261009.md) |
| 2026-10-08 | 10 | `261008.001-261008.010` | [`WORKLOG-20261008.md`](archive/worklog/WORKLOG-20261008.md) |
| 2026-10-07 | 9 | `261007.001-261007.009` | [`WORKLOG-20261007.md`](archive/worklog/WORKLOG-20261007.md) |
| 2026-10-06 | 7 | `261006.001-261006.007` | [`WORKLOG-20261006.md`](archive/worklog/WORKLOG-20261006.md) |
| 2026-10-05 | 16 | `261005.001-261005.016` | [`WORKLOG-20261005.md`](archive/worklog/WORKLOG-20261005.md) |
| 2026-10-04 | 8 | `261004.001-261004.008` | [`WORKLOG-20261004.md`](archive/worklog/WORKLOG-20261004.md) |
| 2026-10-03 | 2 | `261003.001-261003.002` | [`WORKLOG-20261003.md`](archive/worklog/WORKLOG-20261003.md) |
| 2026-10-02 | 8 | `261002.001-261002.008` | [`WORKLOG-20261002.md`](archive/worklog/WORKLOG-20261002.md) |
| 2026-10-01 | 1 | `261001.001-261001.001` | [`WORKLOG-20261001.md`](archive/worklog/WORKLOG-20261001.md) |
| 2026-09-30 | 3 | `260930.001-260930.003` | [`WORKLOG-20260930.md`](archive/worklog/WORKLOG-20260930.md) |
| 2026-09-29 | 5 | `260929.001-260929.005` | [`WORKLOG-20260929.md`](archive/worklog/WORKLOG-20260929.md) |
| 2026-09-28 | 6 | `260928.001-260928.006` | [`WORKLOG-20260928.md`](archive/worklog/WORKLOG-20260928.md) |
| 2026-09-27 | 6 | `260927.001-260927.006` | [`WORKLOG-20260927.md`](archive/worklog/WORKLOG-20260927.md) |
| 2026-09-26 | 4 | `260926.001-260926.004` | [`WORKLOG-20260926.md`](archive/worklog/WORKLOG-20260926.md) |
| 2026-09-25 | 9 | `260925.001-260925.009` | [`WORKLOG-20260925.md`](archive/worklog/WORKLOG-20260925.md) |
| 2026-09-24 | 1 | `260924.001-260924.001` | [`WORKLOG-20260924.md`](archive/worklog/WORKLOG-20260924.md) |
| 2026-09-22 | 3 | `260922.001-260922.003` | [`WORKLOG-20260922.md`](archive/worklog/WORKLOG-20260922.md) |
| 2026-09-21 | 12 | `260921.001-260921.012` | [`WORKLOG-20260921.md`](archive/worklog/WORKLOG-20260921.md) |
| 2026-09-20 | 2 | `260920.001-260920.002` | [`WORKLOG-20260920.md`](archive/worklog/WORKLOG-20260920.md) |
| 2026-09-19 | 3 | `260919.001-260919.003` | [`WORKLOG-20260919.md`](archive/worklog/WORKLOG-20260919.md) |
| 2026-09-18 | 7 | `260918.001-260918.007` | [`WORKLOG-20260918.md`](archive/worklog/WORKLOG-20260918.md) |
| 2026-09-17 | 2 | `260917.001-260917.002` | [`WORKLOG-20260917.md`](archive/worklog/WORKLOG-20260917.md) |
| 2026-09-16 | 2 | `260916.001-260916.002` | [`WORKLOG-20260916.md`](archive/worklog/WORKLOG-20260916.md) |
| 2026-09-15 | 7 | `260915.001-260915.007` | [`WORKLOG-20260915.md`](archive/worklog/WORKLOG-20260915.md) |
| 2026-09-14 | 2 | `260914.001-260914.002` | [`WORKLOG-20260914.md`](archive/worklog/WORKLOG-20260914.md) |
| 2026-09-13 | 7 | `260913.001-260913.007` | [`WORKLOG-20260913.md`](archive/worklog/WORKLOG-20260913.md) |
| 2026-09-12 | 17 | `260912.001-260912.017` | [`WORKLOG-20260912.md`](archive/worklog/WORKLOG-20260912.md) |
| 2026-09-11 | 11 | `260911.001-260911.011` | [`WORKLOG-20260911.md`](archive/worklog/WORKLOG-20260911.md) |
