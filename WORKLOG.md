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

## 261125.008 (r362) -- the pool tier no longer pays the slab's ring hole out of the expert arena (GitHub issue #368), and a residency map that names which tables stand down

**The report.**  GitHub issue #368 (`@DanoPTT`, 1 x R9700, Windows, ROCm 10.2, Flash-Next IQ3_XXS
`-ncmoe 47 --host-experts pool -c 131072 -np 2 --kv-unified -b 8192 -ub 8192`, a 90k-token prompt): every fresh
server lands in one of **two deterministic states**, good L1 prefill 2960-3000 / tg 52 against slow 1615-1620 /
tg 30-32, two byte-stable but different texts, and `-ub 4096` never switches.  Startup, pool fill, arena size and
eviction *sizes* are the same in both states; the slow one re-evicts once more at the L2 revisit.

**Reproduced the shape, not the bimodality.**  Nine fresh starts at their flags read ONE state here (L1 1890-1900,
tg 34.06-34.24, text `98c4f67d6ed9`), with their eviction pattern to within a few percent (6602 MiB / 40 tables
then 8500 / 51, re-arm 42 then 54, against their 6811 / 41 and 8546 / 51).  Eleven one-variable arms (arena 0 / 512 /
12000 / 26000 MiB, the gather off, staging off, serial, `GGML_CUDA_MMB=0`, slab off, headroom, `MOE_GATHER_NSPLIT`
8/32, the PLE resident, `--host-experts pinned`/`-lme mirror`) all read the **same text**, so on this box the
expert cache's residency is not a text input on this shape — their two texts need another carrier.

**What did explain the cost of the state (the fix, block 10).**  Two delivery statements contradicted each other on
a pool-backed tier: `sched_input_gatherable` mode 1 gathers a pool table at EVERY prefill width, yet
`h2d_stage_region_bytes()` still carved a whole-table H2D ring hole — **6656 MiB at `-ub 8192`** (1800 at 4096, so
4096 is exactly the width that never shows the state).  The hole sits above the narrow floor, so the wide prefill
view can only be placed behind it: the work boundary moved 3264 -> 11776 MiB and `moe_cache_evict_slab_range` took
**8500 MiB / 51 whole tables** at every prefill, plus their 54-table re-arm at the drop — residency the prefill then
streams back from the host on every ubatch.  `slab_ring_region_bytes()` now returns 0 when the loader's tier is the
pool (`moe_cache_host_pool_requested()`, a read-only query added to block 09's cache), still deferring to
`GGML_CUDA_SLAB_RING_MIB` and to an explicit `GGML_SCHED_DEVGATHER=0`.

**Measured, after.**  Four fresh visits, the same flags and a cold heat-map directory: L1 **1895 -> 1958.5 t/s**
(+3.4 %), tg **34.18 -> 34.94** (+2.2 %), the cache-hit revisit **728 -> 902.5 t/s** (+24 %, it no longer re-evicts
6778 MiB / 41 tables), S2 tg 46.1 -> 47.1, arena 20 397 MiB held through the whole run, `slab-reclaim stand-downs
this run: none | re-arms: none`, text byte-identical (`98c4f67d6ed9`).

**The instrument the issue asks for (question 3, new).**  `MOE_EXPERT_CACHE_TABLES=1` prints, at exit, the per-layer
residency map (`L1..L48 261/512`, `I` for an identity table) and this run's per-layer slab-reclaim stand-down /
re-arm tally (`L30x3 L31x3`).  The eviction and re-arm messages carry counts and bytes only, so two runs that agree
on `h` and `evictions` can still hold different layers — which is the thing a "same eviction sizes" report cannot
show.

**Gates.**  `gate-prefill-logits.sh` **0.000000 / 100.000 %**, `gate-host-expert-text.sh --quick` 2/2 identical
(and the flipped-library arm still moves, so the pin is live), `check-shlib-symbols.sh` 16/16, `check-gen-docs.sh`
ok, `check-gpu-faults.sh` clean, `validate-set.sh` green with `0 unexplained` in the rung reconciliation (the two new
env keys are additions).  Owned gap: the host-expert KLD point still has no base recorded on this box, so its `--kld`
leg cannot run (the r361-vs-r362 text identity on this shape is the neutrality evidence instead).

**The cut.**  `v18-919794500-r362` = `ca149fbe6` (fork re-cut tip), base `919794500`, tree
`45beb90c0ea0126d454753f4276bd06541934ba2`; canonical re-cut tree matches, 18 official + 1 temporary = 19 default
blocks; optional `0100` tree `0a189a09b`, `0101` tree `1a0ee3d93`, cumulative `95d40ba7d` (both still apply strictly
on the new tree).  Record: `benchmarks/2026-11-25-issue-368-pool-tier-ring-hole.md`, `INVARIANTS.md` r362,
`DIAGNOSTIC.md` §28, `ENVIRONMENT.md` §28.

## 261125.007 (r361, release cut) -- tagged, pushed, the fork branch refreshed to the re-cut tip, the images verified, issue #366 answered and closed

**The cut.**  `v18-919794500-r361` = `41b54afa`, base `919794500`, tree
`8962b7a40ea7e509a60d763b96a2ceeb0902521a`, 18 official + 1 temporary = 19 default blocks, 2 optional (`0100` tree
`35ab35cf5`, `0101` tree `30a57b9a2`, cumulative `ae4981eac`; both re-applied STRICTLY on the new default tree, so
only their recorded trees were re-derived, not the patches).  `scripts/validate-set.sh` green on the committed set,
the rung reconciliation at `0 unexplained` with the two new variables counted as additions, tier-1 CI green on both
`main` and the tag, and the GitHub Release carries its 4 assets.

**The fork.**  `scripts/fork-refresh.sh --build --gates --push` re-cut the branch deterministically (two re-cuts read
the same tip), asserted tree `8962b7a40e…`, built it, ran `check-shlib-symbols.sh` (**16/16**), the coherence gate and
`gate-prefill-logits.sh` (**0.000000 / 100.000 %**, no GPU faults), and pushed:
`gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads **`c7a4f0b6c`**, which is what
`release.json.tip` records.  The patches were re-exported FROM that re-cut chain, so their `From` lines name the
commits the fork carries.

**The images.**  The worker (`radiance`) picked up the tag and published both ROCm lines off-host;
`scripts/build-containers.sh --verify-only` reads **all 23 tags resolve and every moving tag carries this release**
(the check that issue #364 exists for: a push's exit status is not evidence).

**The issue.**  GitHub issue #366 answered with the reproduction, the mechanism, the exactness evidence and the two
caveats (a >256-token penalty window still falls back; an adaptive speculator cannot be gated for text), and closed as
completed, asking the reporter to re-run their mixed gfx1100 + gfx1201 pair with `presence_penalty 1.5` if they get a
chance.

## 261125.006 (r361) -- a penalty no longer costs decode: the exact top-k fast path reproduces the penalties sampler (issue #366)

**The report.**  GitHub issue #366 (`@mjohanning99`): a presence penalty of `0.15` cost **5.7 %** of MTP decode on
their gfx1201 + gfx1100 pair (69.7 vs 73.7 t/s, acceptance unchanged), and their agent had traced it to the S2 fast
path standing down for any non-trivial penalties sampler.  Reproduced on this box first, before touching anything:
3x R9700, `-sm tensor`, Qwen3.8-27B UD-Q4_K_M, `--spec-type draft-mtp`, `top-k 20`, `--presence-penalty` the only
difference, 300 tokens, 4 arms in one loop — **79.4 / 71.5 / 80.1 / 71.8 t/s**, i.e. -10 % and perfectly alternating.

**The fix (block 12 amendment).**  A penalty is the logit bias's shape, not a reason to give up: it moves only the
tokens in its last-n window, so the affected set is the bias set plus that window, the raw top `k + 1 + |affected|`
still holds the top `k + 1` of the untouched tokens, and each affected token's adjusted logit is recomputed from its
RAW logit with `llama_sampler_penalties_apply()`'s own arithmetic **in the chain's order** (the bias sampler runs
first, so the penalty applies to the biased value).  The `k` tokens handed to the chain still carry their raw logits,
so the chain re-applies the bias and the penalty itself and the values stay bit-identical — only the candidate array
is smaller.  Two read-only accessors (`llama_sampler_penalties_get_params()`, `..._get_window()`) expose the
sampler's parameters and its window, so the fast path reads the chain's own state instead of keeping a second copy
of the map; the window is re-read once per sample.  Everything else still stands the path down (DRY, top-n-sigma,
an unknown sampler, two penalties samplers), a penalty **after** top-k is deliberately not reproduced, and a window
past 256 distinct tokens falls back per sample.

**Measured, after.**  5 alternating rounds, same text in both arms: **78.26 t/s fast against 75.00 t/s with
`GGML_LF_FAST_TOPK=0`** (+4.3 %), 3x R9700 `-sm tensor` MTP.  Exactness is the claim, so it was checked 7 ways
against the full-vocabulary arm (presence / repeat / frequency / all three / a penalty combined with a logit bias /
no penalty / a 4B model on one GPU) — every pair byte-identical, plus a 700-token `--top-k 60 --repeat-last-n 8192`
run that grew the window past the cap (472 fast / 228 fallbacks) and still read the full arm's text.  Two traps hit
on the way, both worth recording: hashing a `-lv 4` log (the census needs `-lv 4`, the hash needs none) invented a
"mismatch" that the self-check then disproved, and comparing two *different* sampling profiles for rate is
meaningless with MTP because the acceptance profile moves with the text.

**Instruments (new).**  `GGML_LF_FAST_TOPK_CENSUS=1` — which path each sample took, how many carried a penalty
window, the largest affected set and the self-check's mismatches; without it a text arm cannot tell a fast path
from a quiet fallback.  `GGML_LF_FAST_TOPK_SELFCHECK=1` — the whole vocabulary scored the way the chain would and
compared with the fast path's `k`, per sample, 0 mismatches over 400 samples; `=2` is its **control** (it swaps the
two best-ranked tokens: 128/128 mismatches and the text moves), which is what makes "the text is identical" a
sensitive claim.

**Gates.**  `scripts/gate-fast-topk-penalties.sh` (`MANIFESTS.md` gate 45): 7/7 text pairs identical against the
full-vocabulary arm, census 128 fast / 0 full with a penalty window on the penalised arm and 0 fast / 128 full on the
kill switch, self-check 0 mismatches with its rank-swap control at 128 and a moved text, the 700-token cap arm
472 fast / 228 full with identical text, no GPU faults.  Cross-build neutrality against the ACTUAL r360 binary
(checked out `3be8b9bd1`, rebuilt, and back): 4B greedy `7e6229b260f6` on 1 GPU and on 3-GPU `-sm tensor`, identical
on both builds, and — the one that matters — **27B 3-GPU `-sm tensor` + MTP + `--presence-penalty 0.15` reads
`3651d4d825b9` on r360 (its full path) and r361 (its fast path)**, i.e. the whole fix in one line.  MTP Protocol A
(dense 27B UD-Q4_K_XL, 1 GPU, prose prompt, `-n 2000`, `--presence-penalty 1.5`): plain **31.3 t/s**, `draft-mtp`
**52.7 t/s**, acceptance **0.49092** (> the ~0.45 floor).  `check-gen-docs.sh` ok (no option is added),
`check-shlib-symbols.sh` 16/16, `gate-prefill-logits.sh` **0.000000 / 100.000 %**, `validate-set.sh` green.

**The cut.**  `v18-919794500-r361` = `c7a4f0b6c`, base `919794500`, tree
`8962b7a40ea7e509a60d763b96a2ceeb0902521a`, 18 official + 1 temporary = 19 default blocks; optional `0100` tree
`35ab35cf5`, `0101` tree `30a57b9a2`, cumulative `ae4981eac` (both re-applied strictly on the new default tree).
`validate-set.sh` green, `0 unexplained` in the rung reconciliation with the two new variables counted as additions.
Record: `archive/work/fast-topk-penalties/`, `GREEDY-PURITY.md` §110, `INVARIANTS.md` r361, `DIAGNOSTIC.md` §27.

## 261125.005 (r360, release cut) -- tagged, pushed, the fork branch refreshed to the re-cut tip, the images are building on the worker

**The cut.**  `v18-919794500-r360` = `07611562`, base `919794500`, tree `4674dd9f2e83894895d71b7e987a06d80fa70813`, 18
official + 1 temporary = 19 default blocks, 2 optional (`0100` tree `831bc116a…`, `0101` tree `f455c3121…`, cumulative
`8bd172df5…`).  `scripts/validate-set.sh` green on the committed set; the GitHub Release carries its 4 assets.

**The fork.**  `scripts/fork-refresh.sh --build --gates --push` re-cut the branch deterministically, asserted the
applied tree against `release.json`, built it, ran `check-shlib-symbols.sh` + coherence + `gate-prefill-logits.sh`
(**0.000000 / 100.000 %**, no GPU faults) and pushed:
`gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` reads **`3be8b9bd1`**, which is what
`release.json.tip` records.  The patches were then re-exported FROM that re-cut chain (so their `From` lines name the
commits the fork carries, and the 19 hashes in `release.json` are the re-exported set), and the optional `0100`/`0101`
were re-cut on it: their produced trees are byte-identical to the pre-re-cut values, only the export ids moved.  A
rebase onto `761fcb43` (`wip/pr362`) was needed first: the tag had been pushed from a tree that predated it.

**One open item, by policy.**  The images are built off-host and were still building when this entry was written
(the worker picked `r360` up at 14:00 worker time and was at the HIP compile stage).  `scripts/build-containers.sh
--verify-only` on this tree is the check that finishes the release: a missing tag or a moving tag still carrying
`r359`'s digest exits 1, and that is the only honest proof (`GitHub issue #364`).


## 261125.004 (r360) -- `--fit` charged 1024 MiB of FA staging reserve on a q8_0 K/V cache the kernels read natively, and on a 12 GiB card that was eight layers of offload (GitHub issue #367, RDNA2)

**The report.**  `@pkrasicki` runs an RX 6700 XT (`gfx1031`, 12 GiB) on a DDR3 platform and measured the r354/r355 tree
(`91a8150`) **slower than upstream**, not faster, on two shapes: a Gemma-4 26A4B MoE with `-cmoe --moe-cache-mib 7000
-lm none --spec-type draft-mtp` (**0.5 t/s prompt / 14.4 t/s tg** against upstream's 15.8 / 52.5) and a Qwen3.8 27B
IQ3_S with `-sm tensor -mg 0 -c 16384 -fa on -ctk q8_0 -ctv q8_0` (**19.0 / 4.0** against 40.7 / 15.5).  No RDNA2
device is in the fleet, and a 4x decode loss is not a tuning difference, so this was reproduced by MEASUREMENT on the
shapes this box can reach rather than by inspection.

**Root cause: `--fit`, not a kernel.**  `llama_context::memory_breakdown()`'s no_alloc (`--fit`) path charges a
per-device compute reserve for the FA **prefill staging arena** (a raw device allocation outside the compute-graph
reserve), and `ggml_backend_cuda_fattn_stage_bound()` answered with a FLAT `2 * GGML_CUDA_FA_STAGE_MAX_MB` = **1024
MiB** for every device except RDNA3_5, gated only on `-fa` plus a non-F16 K cache.  That covers the delivery's own
default `-ctk q8_0 -ctv q8_0`, where `ggml_cuda_fattn_tile_type()` resolves the pair **natively** (the launcher's
`need_f16_K/V` are false) and the arena never grows at all: the reserve bought nothing and the fit paid for it in
layers.  Measured with `llama-fit-params` on a 27B UD-IQ3_S at `-c 16384 -fa on -ctk q8_0 -ctv q8_0`, an ~11 GiB
budget derived from the card's free VRAM:

| arm | model | context | compute | `-ngl` |
|---|---|---|---|---|
| r359, f16 KV (control) | 10616 | 1173 | **515** | 59 |
| r359, q8_0/q8_0 | 10616 | 697 | **1539** | **55** |
| r360, q8_0/q8_0 (native) | 10616 | 697 | **515** | **62** |
| r360, `GGML_CUDA_FA_KV_NATIVE=0` (staged) | 10616 | 697 | 1539 | 55 |
| r360, staged + `GGML_CUDA_FA_STAGE_MAX_MB=0` | 10616 | 697 | 4675 | 34 |
| same-tree **upstream** build, q8_0/q8_0 | 10616 | 697 | 541 | 59 |

So the delivery offloaded **4 fewer layers than its own base** on the same flags and 8 fewer with the GSQ cut, and on
a card whose model barely fits that is exactly the reported shape: the layers left on the host are streamed per token
(11.28 GiB / 62 layers ≈ 186 MiB each), so ~750 MiB/token over DDR3 lands at 4 t/s while upstream's 59/62 reads 15.5.
The fit's reserve is an ESTIMATE of an allocation the compute buffer does not contain -- a live run at `-c 16384
-ub 512` prints a **118.89 MiB** compute buffer, and it grows linearly at 0.232 MiB per ubatch token (237.72 at
`-ub 1024`, 475.37 at `-ub 2048`).

**The fix (blocks 09 + 10).**  The bound now takes the context's K/V types and returns 0 for a pair the launcher does
not convert, using the launcher's OWN predicate: `ggml_cuda_fattn_kv_pair_native_host(cc, type_k, type_v)` is
`ggml_cuda_fattn_tile_type` applied to a probe tensor, i.e. literally the `need_f16_K/V` test in
`ggml_cuda_flash_attn_ext_get_f16_extra_data`.  A policy switch (`GGML_CUDA_FA_KV_NATIVE=0`), a mixed pair, bf16
without bf16 MMA, or `GGML_CUDA_FA_STAGE_MAX_MB=0` all keep the reserve exactly as before, and the RDNA3_5 special
case is left alone.  Nothing about a kernel selection, a launch or a numerics path changed: this is a number the fit
subtracts from a budget.

**Two things checked because they are the obvious wrong turns.**  (1) *Is the reserve needed for a deep prefill?*  A
16K-context q8_0 KV prefill (1315 tokens) reads `990793afb918` on both the r359 and the r360 build, and the same
text on a `GGML_CUDA_FA_STAGE_MAX_MB=1` arm where the arena can never be charged for anything.  (2) *Is this
RDNA2-only?*  It is not: the accounting is in `llama-context`, and `gfx1036` (a real RDNA2 device on this box) reads
495 MiB of compute for BOTH its f16 and its q8_0 arm and 1519 for the staged arm, i.e. the same native answer as
gfx1201 -- the old "only RDNA3_5 reads natively" special case was a leftover from when the native q8_0 arm was
gfx1151-only, which under-charged nobody and over-charged everyone else.

**Gates.**  New `scripts/gate-fit-staging.sh` (`MANIFESTS.md` gate 44, four `llama-fit-params` arms, no compute,
~10 s) -- **NEGATIVE-TESTED**: on the same r359 build it FAILS A1 (+1024 MiB), A2 (`-ngl 55` < 59) and B1 (0 MiB over
the native arm), and passes 6/6 on r360.  `gate-prefill-logits.sh` **0.000000 / 100.000 %** (no GPU faults).
Byte-identical greedy text against the r359 build on 1 GPU (`1c5d32ac537d`), 3-GPU `-sm tensor` (`1c5d32ac537d`) and
the 16K q8_0 prefill (`990793afb918`); a real q8_0-KV FA run on the RDNA2 `gfx1036` iGPU generates (`f33aec64cbab`,
14.0/3.2 t/s on 2 CUs).  `validate-set.sh` green (19 default + 2 optional; release.json now records the r360 **re-cut** tip
`3be8b9bd1`, tree `4674dd9f2e…`, i.e. what `fork-refresh.sh` printed and pushed -- the patches were re-exported from
that chain so their `From` lines name the commits the fork carries), the optional
`0100`/`0101` re-cut onto the r360 tree (`831bc116a…` / `f455c3121…`, cumulative `8bd172df5…`; 0101 needed one
textual conflict resolved because block 10 owns `fattn-rdna.cuh`), the file-set reconciliation reads 0 unexplained.

**What the report's FIRST shape still needs, and what it is not.**  Test 1's command is not comparable across the two
builds: `-cmoe`, `--moe-cache-mib`, `-lm` and `-sm relaxed-tensor` do not exist upstream, so the upstream arm could not
have run it -- its `-ngl 99` on a 16 GiB model in 12 GiB means something else entirely (page-cache-resident experts
computed on the CPU, which is exactly what the delivery's `--moe-cache-mib 7000` replaces with a 7 GiB device cache
and a pinned host master).  That arm is a *placement* question on a DDR3 box with no RDNA2 device here to measure it
on, so it is not "fixed" by this entry: the reporter is asked for the exact upstream command, and `TODO` #361's
V620 validation still covers the RDNA2-only block-05/08/10 items.

**Where the RDNA2 story goes from here.**  The two block-05/08/10 items r334 folded in remain externally
validated only (`TODO` #361, waiting on `@jstamagal`'s V620), and this fix is orthogonal to them: it removes a
placement loss the fit was causing, not a kernel regression.  It also means the r360 tree is the first one where an
RDNA2 owner can see the delivery's own `-ctk q8_0` default without paying a 1 GiB fit reserve for it.


## 261125.003 (no release; record correction) -- the r359 entry's Flash-Next tg numbers were 32-token cold-start numbers: at 2000 tokens the model reads 39.24 t/s and the rows form's LOSS is bigger, not smaller

**Why this exists.**  The r359 measurements above used a 32-token greedy tail on a fresh `--no-warmup` server.  That is
fine for a dense model and wrong for a host-expert MoE: the first few dozen decode steps after a 94428-token prefill are
still settling the expert arena's resident set, so the mean over 32 tokens is not the model's decode rate.  Re-measured at
**2000 generated tokens** (same box, same shapes, same arms):

| arm | 27B UD-Q5_K_XL L1 / S2 tg | Flash-Next GSQ-RCO-IQ3_XXS L1 / S2 tg |
|---|---|---|
| 32 tokens, default (dense) | 23.41 / 27.24 | 18.20 / 34.34 |
| 32 tokens, rows form (`LLAMA_KV_ROWS_DISABLE=0`) | 14.73 / 25.95 | 14.80 / 31.98 |
| **2000 tokens, default (dense)** | **23.41 / 27.25** | **39.24 / 48.15** |
| **2000 tokens, rows form** | **14.76 / 25.99** | **28.40 / 44.65** |

So the dense model's numbers stand exactly (nothing to settle), and Flash-Next's were ~2x low on tg and understated the
rows form's loss (23 % -> **38 %** at L1, 7 % -> 8 % at S2).  The conclusion does not move, it gets stronger, and the
reporter's own L1 readings (48.3 rows-on, 51.8 r345) are in the corrected band rather than the 32-token one.

**One claim corrected with it.**  The r359 entry says the same-seed greedy text is identical in every arm and shape.  True
for every 32-token arm and for the 27B at 2000 tokens (`fa78a9826423`, `383dd3732894`), but at 2000 tokens Flash-Next's
compacted shape reads `dc97a530611c` dense and `6379caff5414` rows: over a long tail a near-tie reaches the kernel-family
switch between the two forms.  A text comparison ACROSS the rows and dense forms is not a purity claim; within one form
it still is (`INVARIANTS.md` r359 item 3).

**Rule for future arms:** a host-expert MoE decode is measured at >= 1000 (preferably 2000) generated tokens; a dense
model may use short tails.  The 32-token table is kept in the r359 entry above as the dated record it was.

**Still open:** the block 05 source comment (`src/llama-context.cpp`) quotes the 32-token Flash-Next pair (14.80 /
18.20).  It is a comment, so the code and every gate are unaffected, but it is stale and this repo does not hand-edit
`patches/`; it goes into the next block 05 amendment/re-cut.

## 261125.002 (r359, release) -- the `--kv-unified` K/V-rows form is OPT-IN: it cost 17-59 % of decode at depth and silently disabled the derived (mask-free) kq mask (GitHub issue #365); `v18-919794500-r359`

**Release v18-919794500-r359** -- base `919794500`, tip `003c88e1f`, default tree
`3910c43846612385fbf120d6b8ae0f1b4ecf867a` (was `84d50a416...`), 19 default patches (18 official + `0300`),
`rdna-boosts-all.patch`; both optional artifacts re-cut against the new tree (`0100` alone `0bf09806b`, `0101` alone
`51de75b7b`, cumulative `707eb474c`) and both still apply strictly.  **One block changed: 05.**

**The report (issue #365, @DanoPTT).**  After r358 their `--kv-unified -np 2` server stopped warning and the rows form
began running on the GPU, and on a single R9700 it made deep decode **17 % slower** than the r345 dense form (Qwen3.8-27B
UD-Q5_K_XL, q8_0 KV, 90k, MTP `draft-mtp n3`, acceptance unchanged at 0.590 vs 0.600): "the RDNA4 band and its q8_0
fetch-ahead decline the rows form, so these head-256 q8_0 layers fall to the stream-k MMA path".  Asked whether the rows
form could be declined when the band applies, and why `LLAMA_KV_ROWS_DISABLE=1` also changed prefill (-45 %) and the text
on Flash-Next.  Context worth recording: the r358 release was cut by an agent after a request that had only asked for a
reply to issue #363.

**Reproduced on gfx1201** (ONE R9700, `--kv-unified -np 2`, `-ctk/-ctv q8_0`, a 94428-token prompt, greedy
`--temp 0 --top-k 1`, each visit a fresh server).  Same-seed **text is identical in every arm and every shape**, so this
is a form/placement decision, not a numerics one:

| arm (rows on = r358 default) | 27B UD-Q5_K_XL `-c 131072`, prompt on slot 0 | Flash-Next GSQ-RCO-IQ3_XXS (host experts) |
|---|---|---|
| rows on (r358) | pp 833.6, tg 14.74 | pp 1998.9, tg 14.80 |
| rows off (r345 default) | pp 847.1, **tg 23.41** (+59 %) | pp 1994.9, **tg 18.20** (+23 %) |
| rows on, 5246-token prompt on slot 1 next to the resident 90k (row list 18x shorter) | pp 1219.3, tg 25.96 | pp 1654.4, tg 31.98 |
| rows off, same shape | pp 1223.2, **tg 27.23** (+5 %) | pp 1664.2, **tg 34.34** (+7 %) |
| rows on vs off with the band disabled (`GGML_HIP_FA_BAND_WMMA=0`), 27B | tg 14.74 vs **21.78** | -- |

So the rows form loses even where it compacts an 18x-shorter row list, and inside the MMA family alone the row list
itself costs ~32 % (its native q8_0 loader gathers row by row where the dense arm reads tile rows contiguously).

**The second mechanism, and the answer to their question 2: enabling the rows form DISABLES the derived kq mask.**  The
two are alternatives by construction: a rows graph builds `ggml_flash_attn_ext_rows` and no plain `ggml_flash_attn_ext`
node, so the derived-mask probe's graph never contains its op, `require_observed` refuses it, and the mask is then
MATERIALISED (`n_kv_rows x n_tokens` f16 per ubatch, host-filled) where the dense form allocates no mask at all and the
kernel derives causality from positions.  Measured (`-lv 4`, 27B Q8_0 `-kvu -np 2`): rows on ->
`derived kq mask ... was not used in the probe graph, set to disabled`; rows off -> `derived kq mask ... enabled`.  That
is also the mechanism behind their Flash-Next `LLAMA_KV_ROWS_DISABLE=1` arm losing ~7 GiB of expert arena (a -45 %
prefill / -40 % decode follow from the evictions), i.e. the switch was never isolating the FA kernel.  On our box the two
arms' startup memory breakdowns were identical and both shapes read one text, so their absolute Flash-Next deltas are a
VRAM-tight (Windows, 32 GB) effect, not a second defect.

**The fix (block 05): the form is opt-in.**  `cparams.auto_fkvr` now requires `LLAMA_KV_ROWS_DISABLE=0` ("do not
disable", so the variable means the same thing on both sides of the flip) plus `--kv-unified`; unset or `1` is the dense
form, i.e. the pre-r308 behaviour, which is what a delivery default must be when the alternative is a measured loss.  r358's
three correctness fixes stay -- they are what makes the opt-in arm usable, not a way to make it the default.  The probe
still runs on request and still grades the hardware capability.  Verified: the default is byte-identical to a **pre-r358**
build (`-kvu` 27B Q8_0, 20 greedy tokens, `42 chars sha=bdebe52759ef` on 1 GPU and on 3 GPUs `-sm tensor`, equal to an
r356 build), the opt-in reproduces the r358 numbers and text, and the KLD point is unaffected.

**Gates.**  `scripts/gate-fused-probe-splits.sh` (updated: a new `default` arm -- no env, must log
`K/V rows attention: dense form`, must never claim the rows form, and must read the same text as `rows-off`; every arm
that exercises the form now sets `LLAMA_KV_ROWS_DISABLE=0`) **PASS, 7 arms**, gfx1201 3 GPUs `-sm layer`, Flash-Next
GSQ-IQ3_XXS, bf16 KV, `--kv-unified -c 32768 -np 2`, 5246-token prefill + 48 greedy tokens: `default` `decline=none` and
`rows-off` both `f6f0a8fd998b` (the r311/r358 pinned sha for that shape), the four opt-in arms `enabled`, `sparse` by
observation, every graph `splits = 4` (= the layout bound), no CPU placement, `check-gpu-faults.sh` clean.
`scripts/gate-spec-purity.sh` **PASS** (manifest A `c1f9d0b3f5b6` / B `2fbc183348a8`, the eight axes `000b437b135b`,
`c1f9d0b3f5b6`, `eb89102cb7d8`, `a87c4318b649`, `2fbc183348a8`, `4d87b7b221f7`, `eb89102cb7d8`, `a87c4318b649` -- all
the shas r358 recorded -- and the width probe `worst maxdiff 0` on both cache types).  `scripts/gate-prefill-logits.sh`
**mean KLD 0.000000** against the recorded 10.2 base.  `scripts/validate-set.sh` **OK** on the final tree (fresh-tarball
strict apply, 19 + 2 checksums, both optional artifacts alone and cumulative, Windows portability 519 files, file-set
reconciliation 0 unexplained, the header-closure check clean).  `check-shlib-symbols.sh` 16/16 libraries,
`check-gen-docs.sh` clean (no option added), whole-set `all` build at the tip.  Coherence on the final build: `-kvu` 27B
Q8_0 20 greedy tokens reads `42 chars sha=bdebe52759ef` on 1 GPU and on 3 GPUs `-sm tensor` -- the pre-r358 (and r356
binary) value.  The last source edit after the ladder was a two-line COMMENT fix in `src/llama-context.cpp`; the probe
gate was re-run on that final build (PASS) and the purity/KLD arms do not reach the changed lines.

The frozen beta candidate (`beta/prefill-decode-arms/`, three patches) was re-applied on the new default tree and still
applies strictly, producing tree `a8f070fb2`; it is not re-cut (it is frozen as a code reproduction, `AGENTS.md`) and
its README's recorded tree predates this release.

**Containers.**  The `radiance` watcher built and pushed the six images for `7.14.1` and `10.2`;
`scripts/build-containers.sh --verify-only` on this tree reads all **23 tags** PRESENT, with every moving tag (`latest`,
`rocm-<line>`, `server-rocm-<line>`, `full-`/`light-rocm-<line>`) carrying this release's digest -- issue #364's check
doing exactly its job.  One wrinkle worth recording: the watcher logged a single `ERROR: git fetch --tags --force origin
failed` (`cannot lock ref 'refs/remotes/origin/main'`) on the poll right after the push, then fetched cleanly on the next
poll and built on attempt 1/3; the r358 entry's fix (failures are filed, not just logged) is what made that visible.

**Docs.**  `DIAGNOSTIC.md` §8 (the repointed opt-in, its default and why), `INVARIANTS.md` **r359** (four facts: the
derived-mask coupling, the seven-arm form table, the family-switch caveat for text, and "the default is pinned by the
gate"), `patches/README.md` block 05, `MANIFESTS.md` gate 05, `AGENTS.md`'s gate table (do not re-enable it by default),
`scripts/README.md`.  The reporter's two questions are answered in the issue with the measurements.

## 261125.001 (no release; tooling) -- the r358 container images never existed: the worker's stale watcher pinned the RETIRED ROCm 10.1, and a push's exit status was treated as proof of publication (GitHub issue #364)

Reported by **@tazmaniax**: `server-rocm-10.2-v18-919794500-r358` and `server-rocm-7.14.1-v18-919794500-r358` both
answer `manifest unknown`, and the moving `latest` / `server-rocm-7.14.1` tags still report **r353**.  Not "still
pending": **no release from r354 to r358 ever published an image**, and the only trace was the worker's own
`watch.log`.

### 1. The cause: a stale COPY of the watcher pinned a retired ROCm line

The worker (`radiance`) runs `~/bin/rdna-container-watch.sh` — an installed **copy** of
`scripts/container-watch.sh`, last installed 2026-11-06.  That copy defaulted `ROCM_LINES="7.14.1 10.1"` and
always passed it explicitly (`--rocm "$ROCM_LINES"`).  On 2026-11-23 `3785377d` ("drop ROCm 10.1 everywhere")
changed the repo copy to `10.2` **and made `build-containers.sh` refuse a 10.1 request**, so every build after
it died ~4 s in, before touching a single source file:

```
==> release v18-919794500-r358 (fork point 919794500 -> tip dcee0fc34, ... tree 84d50a416...)
==> reusing prepared source at /home/stew675/llama-src (tree 84d50a416...)
ERROR: unsupported ROCm release line: 10.1 (7.2, 7.14.1, 10.2; 10.1 was retired 2026-11-23 ...)
```

The tag being built was correct all along — r358's own `scripts/build-containers.sh` defaults to
`7.14.1 10.2` and the `10.2.0-full` base image pulls on the worker (verified, 23.1 GB).  The stale *wrapper*
shadowed it.  Six such failures were on disk (`failed-*` for r354, r355, r386, r356, r357, r358), `last-built`
was `v17-919794500-r353`, and GHCR's newest version was r353 (2026-11-23T16:58Z) — exactly the reporter's read.

**Why it was invisible for a day.**  `MAX_ATTEMPTS` parks a failing tag after 3 attempts and the park was
logged `ERROR: … has failed 3 times; not retrying` **to `watch.log` only** — which nobody reads — and cron then
repeated that one line every 10 minutes.  A user report was the alarm.  `CONTAINERS.md`'s claim that "the
watcher runs the script as it exists in the TAG it is building" was true of the build script and false of the
wrapper, and nothing detected the drift.

### 2. The fix (three parts, each verified)

1. **The line set is no longer pinned by the watcher** (`scripts/container-watch.sh`): `ROCM_LINES` has no
   default and `--rocm` is passed **only** when the operator sets it, so the ROCm lines always come from the
   tag's `build-containers.sh` — the single source of truth the release already carries.  This kills the class,
   not just the instance.  Tested both arms with a stub build script: unset -> `--push --prune --force` (no
   `--rocm`), set -> `--rocm 7.14.1 10.2` first.
2. **A push's exit status is no longer treated as proof** (`scripts/build-containers.sh`): after the push loop it
   now probes the REGISTRY for **every** tag the release owes (`--push` fails the step if one does not resolve),
   and two new modes are the same code without a build: **`--verify-only`** (present / missing / unverifiable,
   exit 1 unless every tag resolves **and** every MOVING tag — `latest`, `rocm-<line>`, `server-rocm-<line>`,
   `full-`/`light-rocm-<line>` — carries this release's **digest**, not the previous one) and **`--list-tags`**
   (the tag set, so nothing hand-maintains it).  `skopeo inspect` is used when present; otherwise ghcr.io is
   probed over its v2 API with an anonymous pull token (curl + jq).  ABSENT is reported only on a definite
   `manifest unknown`; a timeout/auth failure reads **UNKNOWN** and still fails (a check must not pass on
   ignorance).  Both paths were run against the real registry and agree (`latest` = `sha256:404e02dd9ca0…`);
   `--verify-only` also works on a box with **no** container engine (it is a check, not a build).
3. **A failure is filed, not logged** (`scripts/container-watch.sh`): one GitHub issue per tag, `Release
   container images not published: <tag>`, created on the first failure, commented **only on a state change**
   (attempt N -> N+1, then the park: 4 notifications, then silence — a 10-minute comment storm would be worse
   than the bug), and commented + closed on the next successful build.  The body carries host, attempt, log path
   and the log tail.  `NOTIFY=0` / `NOTIFY_REPO=` are the knobs; a missing `gh` or a failed API call is a
   WARNING, never a build failure.  Unit-tested end to end with a `gh` shim: create -> silent on a repeat ->
   comment per new attempt -> comment on park -> silent -> comment+close on recovery.

### 3. What the check says, today

`--verify-only` reproduces the reporter's report exactly (7.14.1 arm): the four r358 tags `MISSING`, while
`full/light/server/rocm-7.14.1` are `PRESENT` carrying **r353**'s digests — the stale moving tags the report
also named.  The worker is fixed (watcher re-installed from this tree, the six failed markers removed) and the
r358 build is running there.

**No backfill (maintainer's decision):** r358 supersedes r354-r357 and those releases will never carry an
immutable `-<release>` image tag.

Files: `scripts/container-watch.sh`, `scripts/build-containers.sh`, `scripts/README.md`, `CONTAINERS.md`
(the worker contract, the failure signal, the release checklist's `--verify-only` step, and the
"re-install the watcher copy" line), `AGENTS.md` (a release is not finished until its images resolve).

---

## Archive index — days before today

| day | entries | ids | file |
|---:|---:|---|---|
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
