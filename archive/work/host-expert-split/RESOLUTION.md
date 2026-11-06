# RESOLUTION — `host-expert-split` (promoted in r224, 2026-11-06)

**Outcome: promoted.**  The campaign's work is in the delivery set: **block 12** (the pool's policy and the
prefill read path — the cache-side hooks it uses are block 09).  Record: `WORKLOG.md` 261106.012 and the r224
entry; `patches/README.md` block 12; `ENVIRONMENT.md` (eight variables); `INVARIANTS.md` (the r113 entry
corrected in place + four new facts).

## What shipped

| change | knob | measured |
|---|---|---|
| the pool's content is ranked by the **DECODE** counter | `MOE_HOST_POOL_RANK=decode` (`combined` = parallel-slot workaround, `TODO` #275) | +1.2 % single-stream on a frozen real-world prior |
| an admitted miss **demotes** its arena victim into in-budget pool headroom | `MOE_HOST_POOL_SINK=1`, `_SPARE_PCT=4` | **+5.3 %** alone, **+7.8 %** with the ranking |
| the prefill's per-expert `O_DIRECT` reads are **coalesced** | `MOE_HOST_POOL_STAGE_SPAN=1`, `_SPAN_EXPERTS=16` | prefill **+81 % at `-lme` 12, +57 % at 18, +14 % at 36** |
| the exit-time heat flush no longer truncates a curated map to a 32-byte stub | — (bug fix) | a 45 s run writes 601 376 B / 144 tables |

## What it cost the record (read these before re-deriving anything)

* **The campaign's original "+21.7 % headline was prior leakage."**  `<model>.moe-heatmap` is rewritten by every
  run and it decides the pool's layout, so a benchmark that lets it write measures a prior learning the
  benchmark prompt.  On the frozen real-world map the honest decomposition at `-lme 36` is control 24.3 /
  ranking 24.6 / demote 25.6 / both 26.2.  Freeze = `chmod 444` on the sidecar (the code honours it at open and
  at every save).
* **Every ggml `INFO` counter needs `-lv 4`.**  At `-lv 3` every counter this campaign reasoned about was
  invisible, which is how a 2.3 GB/s ceiling was attributed to `moe_cache_stage_table` — a function that never
  runs on the field config.  The reads were `pool_prepopulate_locked` and the `moe_cache_gather_host` pre-pass.
* **A 20-token generation under-reports decode by 13-15 %.**  Quote gen at `-n 64` and say which harness.

## What stayed open

`TODO` #275 (the `-npl 4`/`-npl 8` cost of the decode ranking, accepted, with the `combined` workaround named).
The `moe_cache_stage_table` arm of the coalesced read is unexercised on the field model — the 35B-A3B
prefill-logit gate is its first proof.

## Where the evidence is

[`HANDOVER.md`](HANDOVER.md) (state at the end of the campaign), [`W4-SPAN-READS.md`](W4-SPAN-READS.md) (the
disk-I/O record), [`W1-SINK-TAX.md`](W1-SINK-TAX.md) (the copy cost), [`P1-POOL-RANK.md`](P1-POOL-RANK.md) (the
ranking), [`STEP1-REPLAY.md`](STEP1-REPLAY.md) (the offline evidence), [`DESIGN.md`](DESIGN.md) (the PR #267
review), [`PLAN.md`](PLAN.md) (the standing plan as it was), [`tools/`](tools/README.md) (the arm harnesses).
