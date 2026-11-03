# P1 RESULTS — the bypass switch, and what it found (2026-11-03)

**Verdict so far: the correctness leg is GREEN on ROCm 10.1, and the performance leg produced two
findings that matter more than the A/B it was meant to run.**  P1 is not finished: the DIO confound below
must be fixed before any throughput number means anything.

## What the two arms ACTUALLY were (read this before quoting any number below)

The switch lives inside `--host-experts pool`, and **the pool was never built in either arm** — both logs
report `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned`.  So `MOE_HOST_POOL_FRAC=50` did nothing, and the
control arm was not the pool at all:

| env value | what the fill actually did | is that "the pool"? |
|---|---|---|
`MOE_HOST_PUSH_SRC=pinned` | `expert_host_src()`'s **per-table pinned bounce** branch: `memcpy` the expert out of the page cache into a pinned ring slot (or `pread` it with O_DIRECT), then H2D | **no** |
`MOE_HOST_PUSH_SRC=mapping` | **direct pageable `cudaMemcpyAsync` from the model mapping** (33 393 fills on the 35B arm, 47 427 on the 122B arm), no staging at all | no |

So this was **push vs a staged copy through a pinned bounce ring**, which is a valid transport comparison —
and it is *not* "push vs pool".  `MOE_HOST_POOL_DIO` governs **both** the pool fill and the bounce staging
(shared `g_pool_dio`), which is why flipping it changed the control arm's source, not the push arm's.

**Consequently: the push TRANSPORT was tested; a push MODE was not.**  There is no `--host-experts push`, no
pool-free configuration, no page-cache-residency policy and no eagerness in what ran.

## What was built

`patches/0001-p1-moe-host-push-src-mapping.patch` (69 lines, `ggml/src/ggml-cuda/moe-expert-cache.cu`):
an env switch `MOE_HOST_PUSH_SRC=pinned|mapping` (default `pinned` = delivered behaviour).  When `mapping`
and the master is a pool-backed `ROCm_HostMap` with 1-D geometry, `expert_host_src()` returns the mapping
itself as the copy-engine source — no pool slot, no bounce ring, no host `memcpy`.  Strided/sliced tables and
2-D copies keep the pool (the 2-D pageable path is 4-5x slower and is where the documented fault lives).
Diagnostics: a `stderr` engagement line (ggml's WARN is filtered below llama.cpp's default level) and a copy
counter in `moe_cache_report`.

Applied on the fork at branch **`p1-push-src`** (commit `be83da8f0`, on top of `rdna-boosts` `efaa60db0`),
built with `~/bin/build-llama-rocm` (ROCm 10.1.0, gfx1201).

## Correctness — clean

Arm: 35B-A3B Q4_K_M, 1 GPU, `-sm tensor -mg 0`, `-ncmoe 40 --host-experts pool`, forced misses
(`MOE_EXPERT_CACHE_MIB=2048`, `MOE_HOST_POOL_FRAC=50`), same seed, greedy:

| | text | fills | direct-from-mapping copies |
|---|---|---|---|
`pinned` | 570 chars `sha=0cf4a62aa63f` | 30 033 | 0 |
`mapping` | 570 chars `sha=0cf4a62aa63f` | 30 033 | **33 393** |

- **Byte-identical text** with 33 393 direct pageable DMA fills (and 47 427 on the 122B arm below).
- **`check-gpu-faults.sh --since 16 min`: kernel clean, log clean, "no GPU faults detected"** (the
  kernel-journal faults inside the wider window are the probes' deliberate ones — `arm pageable` faults by
  design; they are outside this window).
- The access census is identical between arms (`hits=64947 fills=30033 colds=39420`, `h=48.32 %`), so the
  admission policy is untouched — exactly what the switch is supposed to do.

**So the "tears under back-to-back use" claim is refuted again, this time in the real path**, at a fill rate
of ~75 experts/token.  It is refuted on **10.1** only; 7.14 is still owed (and is a separate build).

## Performance — the A/B does not say what it was meant to say yet

**1. Where the set fits and the page cache is hot, the memcpy is already free.**  35B-A3B, arena 2 GiB,
pool FRAC 50, interleaved (P/M/M/P, 200 tokens): `pinned` 91.7 / 102.9, `mapping` 102.5 / 102.7 t/s.  Warm
vs warm: **102.9 vs 102.6 — no difference.**  The pool's `memcpy` is issued before the async DMA and the next
iteration's `memcpy` runs while that DMA is in flight, so removing it removes work that was already hidden.
(The 66-93 µs vs 137 µs microbenchmark gap from the archived campaign is a *serialized* per-copy measurement;
under pipelining it does not translate.)

**2. ⚠ A direct mapping push silently bypasses the pool's DIO decision, so it is not an apples-to-apples
arm.**  122B-A10B `-ncmoe 48`, pool FRAC 50, arena 4 GiB, interleaved (M/P/P/M, 64 tokens, all warm):
`mapping` **18.0 / 19.0 t/s** vs `pinned` **3.3 / 3.2 t/s**.  That 5.7x is **not** a copy win: with the set
larger than the page cache the staging path decides `MOE_HOST_POOL_DIO=auto → O_DIRECT`, i.e. the staged arm
reads from disk on purpose, while the mapping arm reads through the page cache and never enters that branch at
all.  The difference is the SOURCE (disk vs page cache), not the transport.  Forced buffered (`MOE_HOST_POOL_DIO=0`, both arms): `pinned` 4.9 / 4.9 vs `mapping`
4.4 / 11.3 — **inconclusive** (variance is large at 64 tokens; a warm-up protocol is needed).

## What this means for the design (the useful part)

A pageable push **requires the bytes to come through the page cache**: it either finds them there or faults
them in.  That is incompatible with the streaming mode whose entire purpose is *not* to fault the set into
the page cache.  So the mode ladder is a **memory ladder**, and the campaign's shape falls out of it:

| mode | the expert bytes live in | best for | cost |
|---|---|---|---|
`pinned` | a `ROCm_Host` copy (non-reclaimable) | the set fits in RAM — still the fastest, unchanged | whole set non-swappable |
**`push`** | **the page cache (reclaimable, shared with the GGUF itself)** | **the set fits in the page cache but should not be pinned** — a mode that does not exist today | a disk read if reclaimed |
`pool` (DIO) | a bounded pinned pool, streamed from disk | the set does not fit at all | every miss is an O_DIRECT read |

`push` is therefore **not** a replacement for `pool`.  It is the missing middle rung: same speed class as
`pinned` (measured: equal on the hot 35B arm), one less copy, and the host RAM it uses is reclaimable page
cache rather than a pinned duplicate — which is the RAM win the closed campaign was after, arriving from the
other direction.

P2 must therefore start by making the mapping source **honour the DIO decision** (if the pool would stream,
the mapping push is not available for that table — fall back per table, and log it), and the A/B protocol must
include a warm-up run, interleaving and a longer decode.  Until then, throughput claims from this switch are
confounded.

## Next

0. **Fix the test before fixing the code.**  The pool must actually be built for the A/B that matters
   (`MOE_HOST_POOL_MIB` explicit, and the teardown report must show tensors/slots/MiB — `0 tensors` means the
   arm is the bounce path, not the pool).  Then the two comparisons are: **push vs pool** and **push vs
   pinned**, each with the same source policy.
1. Fix the DIO interaction (per-table availability of the mapping source; the arm must not exist where the
   staging path would stream), then re-run the 122B A/B with a warm-up + interleave + ≥500 tokens.
2. 7.14 build of the same patch: the same correctness gate (the refutation is 10.1-only so far).
3. Then P2 (eagerness): the switch currently pushes at *demand* time; the win, if any, is in issuing the
   next op's fills early.
