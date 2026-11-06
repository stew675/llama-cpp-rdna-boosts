# W1 — the swap's marginal cost (`MOE_HOST_POOL_SINK_TAX`), measured 2026-11-06

**Verdict: GO.**  A 1-D D2H of one expert costs far less than the storage read it replaces — **9.6 µs** per
call on this box against a read that must cost at least ~0.3 ms — so the demote pays for itself if a demoted
victim is re-read even ~3-7 % of the time (and well under 1 % on slow storage).  Build W2.

## What was measured and how

`PLAN.md` §2 said the campaign turns on one constant, `c_pcie` for our own 1-D swap, and that it could be
measured without implementing the policy: the demote adds **exactly one D2H per admitted miss** (the arena
fill's H2D happens anyway, and the slot is the one the promoted expert vacated).  So
`MOE_HOST_POOL_SINK_TAX=1` issues exactly that copy — into a scratch pinned buffer, at exactly the point the
demote would issue it (`fill_slot_from_host`, on the victim's bytes, on the same stream, before the fill
overwrites them) — and changes nothing else.  It is a diagnostic and is never enabled in a shipped run.

Harness: `tools/arm-decay.sh` with `SINK_TAX=`, `LME=36|44`, `MOE_HOST_POOL_DIO=1`, `-lzm off`,
`MOE_HOST_POOL_RANK=combined`, `-ncmoe 48`, arena 4 GiB, one R9700, single device, every arm under
`scripts/gpu-watch.sh` with a bounded timeout.  The prefill/decode split is `-n 0` against `-n 64` on the same
config, so the per-*decode*-token rate is a subtraction, not an assumption.

## The numbers

| `-lme` | arm | gen t/s (reps) | D2H copies | of which decode | copies/token | MiB/token |
|---|---|---|---|---|---|---|
| 36 | `SINK_TAX=0` | 22.6 / 22.5 / 22.7 | — | — | — | — |
| 36 | `SINK_TAX=1` | 22.0 / 22.1 / 22.1 | 11 685 | 7 098 | **111** | 75.4 |
| 44 | `SINK_TAX=0` | 27.8 / 27.7 | — | — | — | — |
| 44 | `SINK_TAX=1` | 26.3 / 26.2 | 10 572 | 5 988 | 94 | 63.6 |

*Prefill share, measured with `-n 0`: 4 587 copies at `-lme 36`, 4 584 at `-lme 44` — i.e. ~40 % of the tax
lands before generation starts, and only the remainder is charged to decode tokens.*

| | `-lme 36` | `-lme 44` |
|---|---|---|
| generation cost | **-2.3 %** | -5.4 % |
| ms/token | **1.06** | 2.06 |
| effective `c_pcie` | **0.0140 ms/MiB** | 0.0324 ms/MiB |
| per copy (679 762 B) | **9.6 µs** | 22.0 µs |
| implied bandwidth | ~71 GB/s | ~31 GB/s |

Naive transport would be 26 µs/copy (~26 GB/s).  The measured 9.6 µs means roughly **two thirds of the copy
is hidden behind compute**: a MoE decode token is not PCIe-saturated, so an async D2H on the stream costs its
queueing and its residual bandwidth, not its full transfer time.  That is the finding that makes the demote
affordable — and it is the opposite direction from the 0.04 ms/MiB assumption `PLAN.md` §2 carried.

## What it buys, and why the exchange rate is so favourable

The write-back's *benefit* is that a victim stays reachable at 9.6 µs instead of a storage read.  Per
operation, on this box's NVMe (fills completed at ≳2 GB/s aggregate): a 0.68 MiB read is ~0.3 ms, so the copy
is **~30× cheaper than the read it replaces**.  On a 125 MB/s device it is ~5.4 ms — **~570×**.

Break-even reuse rate = copy cost / read cost:

| storage | read | copy | break-even reuse |
|---|---|---|---|
| this box (NVMe, ≳2 GB/s) | ~0.3 ms | 9.6 µs | **~3 %** |
| a 125 MB/s device (the constrained case) | ~5.4 ms | 9.6 µs | **~0.2 %** |

So the demote does not need to fix the ranking or predict anything: if even a few percent of demoted victims
are wanted again, it wins.  This is a *much* cheaper decision than the `+12.8 GiB of pinned RAM` alternative
the offline model priced — that alternative buys the same effect by pinning the whole complement.

## One inversion worth carrying into W3

The cost is **roughly constant in pool size** — it is driven by *arena* admissions, not by the pool
(111 copies/token at `-lme 36`, 94 at `-lme 44`, only 16 % apart), while the *benefit declines* as the pool
grows, because a bigger pool already covers the misses (and is provably inert at a mirror, S3).  Therefore the
demote's best configuration is the **below-complement** one, which is exactly where our holes are worst — the
same place the exclusive-tier design being compared against is at its best.  W3's sweep is now well posed.

## Caveats

* The tax pays the **transport only**.  The real demote also pays bookkeeping (slot retire/publish state, one
  event, a map update) — small next to 9.6 µs of DMA, but not zero, so W2 must report its own copy count and
  re-measure rather than inheriting this number as gospel.
* One device, one model, one prompt, one storage device.  The `-lme 44` cell costs twice the `-lme 36` cell in
  ms/token *because* its decode is 23 % faster (less compute to hide the copies) — so never compare the cost
  across configs by percentage; compare µs/copy and copies/token.
* `SINK_TAX` must never ship: it is a diagnostic that deliberately wastes bandwidth.  It lives on the fork
  branch only, behind an env var that defaults off.
* While reading this path: `fill_slot_from_host` calls `expert_src_done(..., retire=false)` **twice** on the
  success path (once inside the success branch, once unconditionally after the failure block).  Not touched
  here — an instrument must not fix what it measures — but it is a latent double-retire worth a separate look.
