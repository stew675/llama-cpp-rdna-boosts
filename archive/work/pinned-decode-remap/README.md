# pinned-decode-remap — host-expert decode without the per-op routing sync (PR #266)

**Status: PROMOTED in r203 into blocks 09 / 10 / 12.**  This record is kept because the design has a
history (a much larger device-remap family was removed in r40) and because the invariants it introduced are
the ones a future change has to keep.  Everything here is reference; nothing in `patches/` is applied from
this directory.

## What it was

PR #266 (`briansp2020`) against r196, two commits, each with its own kill switch:

1. **`GGML_MOE_DEVREMAP`** (default on) — the decode/verify band's routing no longer has to be read back to
   the host.  A new `moe_cache_remap` backend iface member (appended last; NULL in the meta/cpu/rpc backends)
   lets the backend map the device routing to slot ids in a kernel, writing `remap_dev` (`slots + e` = cold)
   and copying the raw routing into mapped pinned memory; the residency policy consumes that routing **one
   call late** and re-uploads the expert -> slot map only after a residency change.  One launch serves a
   layer's gate/up/down tables.
2. **`MOE_HOST_POOL_COLD`** (default on) — a `--host-experts pool` miss whose expert is already resident in
   the pinned pool is read in place through the pool's device alias (cold id `n_res + pool slot`) instead of
   being filled into an arena slot first.

## Why it was accepted (the r40 question)

r40 removed a *device-remap family* (`MOE_EXPERT_CACHE_DEVMAP`, `DEVPOLICY`, `KSLOT`, `DEV_EAGER`, the device
policy kernel, the device mirrors, deferred promotion, the device prefill tally, the in-place prefill read;
2684 deletions) and `ENVIRONMENT.md` §1.2 said "do not reintroduce".  This candidate is not that machinery:
the **policy stays on the host**, there is one map kernel and one per-table map, no device policy state, no
mirrors, no KSLOT, no promotion, no prefill path, and it declines under `-sm tensor`, on a pool-backed table,
on a sliced table and above the band.  The r40 case against it was also technical and is now stale: the
divergence it was blamed for was the per-start hipBLASLt solution pick (issue #67), prefill is untouched
because the path is band-only, and the decode win is positive and repeatable.  The maintainer's decision
(r203) was to accept and to amend the §1.2 note into "removed in r40, narrowly re-introduced in r203" — the
note now spells out both halves.

## Validation

`benchmarks/2026-11-01-host-expert-device-remap.md` (full tables).  In one line: byte-identical greedy text on
8 deterministic arms (pinned/pool × 128/1000 tokens, a 2048 MiB thrashing arena, gpt-oss-20b MXFP4, a 2-slot
server over slot switching), **identical MTP acceptance** (0.76089 = 1397/1836 and 0.83340 = 2141/2569),
prefill KLD **0.000000 / 100.000 %**, **+8.4 %** TG pinned / **+4.8 %** pool / +14.1 % and +9.2 % at 4 and
8 slots / +2.4…+3.3 % MTP, inert under `-sm tensor`, GPU faults clean.

## Invariants (the reason this record exists)

1. **`slot_dirty` is the map's freshness signal**, and every slot mutation must set it: fill, eviction,
   stand-down, `moe_cache_evict_slab_range`, the shrink path, device migration, re-arm.  A stale map reads a
   slot that now holds another expert (silent wrong text); the `slots + e` cold encoding also depends on the
   slot COUNT, so a count change without a rebuild indexes the master at the wrong offset or past it.
2. **The lag is policy-only.**  A stale lag can admit/evict the wrong expert for one step; it can never move
   a read address.  Keep the policy host-side (a device policy is the r40 family and stays removed).
3. **The pool-cold read is guarded by the eager path's synchronize.**  The pool's demand path never evicts a
   resident and retiring slots are excluded, but `pool_rerank_from_heat_locked` rewrites slots with plain
   host copies from `moe_cache_update_host`, which runs after `ggml_backend_synchronize(input_backend)` and
   the ids readback's sync.  A change that removes that sync must gate `MOE_HOST_POOL_COLD` (and the
   pool-sourced fills) too.
4. **Judge it with deterministic oracles.**  A concurrent two-slot text diff is not one (`GREEDY-PURITY.md`
   §38; the unchanged eager build reads two text basins run to run here) — use single-sequence arms, a
   sequentially driven 2-slot server, the MTP acceptance counts and the prefill-logit gate.

## Scope limits (accepted)

* `MOE_HOST_POOL_COLD` keys on `ggml_cuda_info().device_count == 1` (process-wide): on a multi-GPU box
  running a one-GPU arm it is silently off.  Pin `HIP_VISIBLE_DEVICES`.
* The remap declines on a pool-backed table (it needs a pinned master) and under `-sm tensor`, so the two
  switches never apply to one table together.
* Not extended to `-sm tensor` (that is where the r40 design lived and where its cost was), nor to prefill.

## Pointers

* `submission/` — the PR as filed (README + both patches).
* `GREEDY-PURITY.md` §82 — the rule and the numbers.
* `patches/README.md` (block 09) — where the code lives and what each block carries.
* `ENVIRONMENT.md` §1.2 — the r40 removal note, amended with the narrow re-introduction.
* `archive/work/moe-verify-fusions/FINDINGS-devmap-removal.md` — what r40 removed and why.
