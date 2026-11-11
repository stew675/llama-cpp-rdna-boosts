# host-pool-demand-overlap — `TODO` #132, the pool's demand read on a tensor split (CLOSED, r261)

**Status: CLOSED AND ARCHIVED (2026-11-11, r261).**  The item's stated mechanism — a demand-fill *overlap
worker* — was measured and **refuted** (there is no in-flight device work to overlap with: the host
synchronizes on the routing readback before every routed op, and the demand service that follows is strictly
additive).  What shipped instead is the byte half of the same read: on a `-sm tensor` split the fallback fill
read the **whole** host expert when it needed only this device's contiguous slice, so every cold expert was
read at double size.  `MOE_HOST_POOL_SLICE_READ` (block 12, default on, `=0` restores the old read):
**+6.0 % decode / +8.6 % prefill on the 2-GPU x8 pair, +24.8 % prefill at 3 devices, −10/−16 % disk, inert at
1 GPU**, with byte-identical greedy text and prefill KLD −0.000000.  The measurement record is
[`benchmarks/2026-11-11-host-pool-slice-read.md`](../../benchmarks/2026-11-11-host-pool-slice-read.md); the
delivery record is `WORKLOG.md` 261111.003 and `INVARIANTS.md` r261.

This directory is the closed campaign's evidence.  Nothing here is part of the delivery, and nothing here may
be applied to a checkout (`AGENTS.md`, WIP and promotion rules).

## What is in here

| file | what |
|---|---|
| `arm-run.sh` | the arm runner (box hygiene floor, `/proc/diskstats` + `pswpout` accounting, `-lv` split: census at `-lv 4`, text with **no** `-lv`), superseded in spirit by nothing — it is the record's instrument |
| `ab-261111.sh` | the whole A/B matrix (pair / 1 GPU / 3 GPUs x slice-read on|off, plus text arms), sequential |
| `probe-132.patch` | the **temporary** decomposition probe (**not shipped**): per-call timers around the ids readback, `moe_cache_update_host`, and the pool/bounce fill in the decode band, printed at `exit` under `MOE_PROBE_132` |
| `readprobe.py` | the queue-depth probe: 64 random 1.77 MiB `O_DIRECT` reads of the real shard at depth 1/2/4/8 |
| `runs/` | every raw log + `summary.tsv` per verdict (`261111-probe` = the decomposition, `261111-ring` = the refuted shared ring, `261111-ab` = the shipped A/B, `261111-coh` = coherence) |

## What was tried and dropped (so it is not re-proposed)

* **A shared, reusable demand ring across the two devices** (one per host tensor, `slots + i` in the pool's own
  allocation, per-(entry, device) drain events, 1 188 MiB pinned).  Hypothesis: on a tensor split the two
  devices miss the same cold experts microseconds apart, so one read could serve both.  Measured: **20 860
  ring reads against 96 hits (0.5 %)**, decode a tie (16.95 against 17.06 t/s), disk unchanged.  The cold tail
  is genuinely cold and the devices do not duplicate.  Reverted; nothing shipped.
* **A demand-fill worker / overlap.**  Blocked by the structure, not by effort: 48 full meta-backend
  synchronizes per token (one per routed op, for the ids readback) put the host exactly where the device has
  nothing queued, so the 33.2 ms of demand service per step overlaps with nothing; and the hook holds one
  global mutex, so a worker would have to read outside the lock to run at all.  See §1 of the record.
* **Parallel fills (depth 4)** are worth ~1.8x on the read wall (2.9-6.0 GiB/s measured) but need several
  misses known at once; the only exact lookahead is *within* a layer (gate/up/down share the routing), which
  bounds it at ~2.  Recorded as the next step, not shipped: `TODO` #141 (a sub-complement budget's static
  bound) and the devremap's cold source attack the same bytes with less machinery.
