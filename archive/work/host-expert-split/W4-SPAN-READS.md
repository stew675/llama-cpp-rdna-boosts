# W4 — prefill read granularity (the disk I/O ceiling)

**Status: implemented, measured, default-on (`MOE_HOST_POOL_STAGE_SPAN=0` is the kill switch). Recorded
2026-11-06.**  This closes the item HANDOVER §7 carried as "the next thing", and it corrects where that item
said the reads came from.

Read with [`HANDOVER.md`](HANDOVER.md) (current state), [`PLAN.md`](PLAN.md) (the standing plan) and
[`W1-SINK-TAX.md`](W1-SINK-TAX.md) (the copy cost).

---

## 1. What was actually wrong (correction to HANDOVER §7)

HANDOVER §7 blamed `moe_cache_stage_table`: "walks `e = 0..n_experts` and calls `expert_host_src(...)` + one
`cudaMemcpyAsync` per expert".  **That function never runs on this configuration.**  With the counters visible
(see §5) it reports 0 calls, 0 declines, no staging line at all — the scheduler does not stage these host
tensors through that path, so the whole per-expert loop there is dead on the field config.

The 2.3 GB/s was real, but it comes from two other places:

| site | when | measured |
|---|---|---|
| `pool_prepopulate_locked` (the ranked one-shot pool fill at sizing) | once, at the first band pass | 60 624 reads, 38.5 GiB, 18.1 s → **2191 MiB/s** |
| the gather pre-pass in `moe_cache_gather_host` (`-lme` below the mirror) | every prefill ubatch | at `-lme 12`: 198 723 reads, **128 GiB**, **56.3 s** of a 73 s run |

Both issue **one O_DIRECT read per expert** (~0.65 MiB).  `/proc/diskstats` sampled at 2 Hz during a run shows
the drive pinned at **~2.1 GiB/s** for 40 s — while the same drive does 4.2 GiB/s on a 16 MiB `dd iflag=direct`
read and 6.7 GB/s on four concurrent streams.  So the ceiling was read SIZE (latency-bound at depth 1), not the
drive, exactly as §7's `dd` table said — only the code that issues them was different.

The gather loop visits experts in ascending `e` and the experts of a table are contiguous in the file
(`src_offs + e*host_bytes`), so the missing ones form runs that can be served by one read each.

## 2. The fix — `stage_span_fill()`

For every maximal run of **≥ 2** consecutive experts that are in NEITHER tier:

* one `pread` of up to `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` (16, ~10.9 MiB on the field config) into a **pinned
  span buffer**, `align`-aligned with the O_DIRECT offset rounded down a block (`host + skip` is the payload);
* one H2D per expert out of that buffer into the same destination the per-expert path would use;
* the pool admission is **unchanged**: an expert read here is still published into an EMPTY pool slot (that is
  how a prefill-only pass ever reaches the pool budget), except the bytes are already in hand, so it is a host
  memcpy instead of a second read;
* a short/failed span read is left **unserved and counted** — the per-expert path handles those experts, so a
  read failure is a slowdown and never a correctness event (the mapping is always a valid source).

The ring is **grown on demand** (up to 32 entries per device, pinned, leaked like the pool and the bounce ring)
for a measured reason: the H2D copies are queued on the compute stream, so they only drain as that layer's
kernel queue reaches them.  A fixed 2- and then 4-deep ring spent **5.0-5.6 s of a 16 s prefill waiting on its
own copies**; with the ring growing to the backlog the wait at `-lme 36` is 1.5 ms and at `-lme 12` 0.34 s.

It is wired into **both** prefill read paths:

| path | runs on | why |
|---|---|---|
| `moe_cache_gather_host` (the gather pre-pass) | this model / `--host-experts pool` + `-ncmoe` | the per-ubatch prefill read |
| `moe_cache_stage_table` (the `!host_pinned` branch) | a pinned-master / scheduler-staging config | same shape, gated on `MOE_HOST_POOL_STAGE_BATCH`, **unexercised on this model** — see §6 |

### Knobs (all default-on; `ENVIRONMENT.md` entries are owed at promotion)

| var | default | class |
|---|---|---|
| `MOE_HOST_POOL_STAGE_SPAN` | on | kill switch (`0` = the pre-W4 per-expert read) |
| `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` | 16 | tuning (chunk size; clamped 1-64) |

`SPAN_MIN` is deliberately **not** a knob: it is load-bearing (measured at `-lme 36`: 1 → 362 t/s, **2 → 379**,
4 → 331) and a deployment that got it wrong would be silently 12 % slower with no symptom to look at.

## 3. Measured (canonical frozen prior, `-ub 6144`, `-n 64`, 1 GPU, `MOE_HOST_POOL_DIO=1`)

| `-lme` | prefill SPAN=0 | prefill SPAN=1 | Δ prefill | gen SPAN=0 | gen SPAN=1 |
|---|---|---|---|---|---|
| 12 | 94.2 | **170.7** | **+81 %** | 9.9 | 9.9 |
| 18 | 115.6 | **181.6** | **+57 %** | 12.8 | 12.9 |
| 36 | 330.9 / 330.7 (×3) | **375.5** / 378.8 (×3) | **+13.5 … +14.6 %** | 26.2 | 26.2 |

Reads, from the counters (not from a wall clock):

| `-lme` | span reads | bytes | rate | per-expert reads left | rate |
|---|---|---|---|---|---|
| 12 | 43 172 (2.52 MiB/read) | 108.9 GiB | **4828 MiB/s** | 40 036 (25.9 GiB) | 2201 MiB/s |
| 18 | 38 180 (2.04 MiB/read) | 77.7 GiB | — | 46 275 (29.8 GiB) | 2273 MiB/s |
| 36 | 7 248 (1.45 MiB/read) | 10.5 GiB | 3729 MiB/s | 35 011 (22.6 GiB) | 2332 MiB/s |

A read-rate ratio of **2.1-2.2×** per byte plus fewer reads is the whole mechanism.  At `-lme 12` the run
went 81 s → 55 s wall, and 198 723 → 83 208 reads.

**Why the runs are short.**  The mean chunk at `-lme 12` is 2.5 MiB, not 10.9: the pool holds 131 of 512
experts per table and the arena 43, so the 338 missing ones are interleaved with residents and the runs are
~4 experts.  That is also why the `-lme 36` win is the smallest (119 missing of 512 → runs of 1.2, i.e. most
of the traffic is still per-expert) — and why a *bigger* chunk, or bridging short resident islands, is not a
win: at `-lme 36` reading every 16-expert window would read 4.3× the bytes to save 1.5× on the rate.

## 4. Gates

* **Text identity: PASS.**  `sha=880c932888f8` (126 chars), the campaign reference, on SPAN=0 and SPAN=1 at
  `-lme 12` — the budget where the span path serves 168 536 experts / 108.9 GiB.  Pool engagement asserted from
  the same run (see §5), not assumed.
* **GPU faults: clean** (`scripts/check-gpu-faults.sh` over the kernel journal window + every arm log).
* **Fallbacks: 0** in every arm (`fallbacks=` counter), i.e. no chunk ever had to fall back.
* **Generation: unchanged** (26.2 vs 26.2 at `-lme 36`, 1 rep each at `-n 64`; 9.9/9.9 and 12.8/12.9).
  The span path only runs during a prefill; the decode path does not reach it.

## 5. Two traps this cost, both now fixed in the tooling

1. **ggml `INFO` needs `-lv 4`.**  `common_log_get_verbosity()` maps `GGML_LOG_LEVEL_INFO` to
   `LOG_LEVEL_TRACE` (4), and the arms ran `-lv 3`.  So every counter line this campaign reasons about
   (`prefill cache staging`, `pool fills`, `scheduler pageable-master fallback`, and now the span/bounce
   lines) was invisible, and the 2.3 GB/s figure was attributed by reading the wall clock.  That is how §7
   ended up naming the wrong function.  **The arms now honour `LV=`; use `LV=4` for any arm whose counters
   you intend to quote.**  It costs nothing measurable (generation is identical at `-lv 3` and `-lv 4`).
2. **`arm-coherence.sh` reported VACUOUS on every run, while the pool was engaged.**  This CLI logs *nothing*
   at its default threshold (not even ggml `WARN`), and the arm grepped the pool report out of a `2>&1` log
   written at that default.  Fixed by running at `-lv 3` and **splitting the streams** (`> OUT 2> LOG`): the
   logger writes every level to stderr and the generated text is written raw to stdout, so the hashed text
   stays byte-clean at any verbosity while the pool `WARN` lands in the log.  One run now proves both.

## 6. What is NOT proven

* The `moe_cache_stage_table` arm is **unexercised** on this model (that path declines here).  It is the same
  helper behind the same geometry guard (`host_bytes == expert_bytes`, `src_off == 0`, no `do2d`, no fused
  slices, a registered source, O_DIRECT), and its bytes are read-only-mapped, but "compiles" is all it has.
  **First proof owed: the 35B-A3B prefill-logit gate** (that config reaches the scheduler staging path).
* One model, one prompt, one device.  `-sm tensor` declines the whole pool path as before.
* Windows: the path is `pread`-only and POSIX-gated (`pool_file_io_available()` is false there), so it is
  inert, not ported.
* The absolute rates are this box's (`/llm` on `/dev/nvme1n1p2`).  The *ratio* is the claim, not the MiB/s.

## 7. Files touched

| file | change |
|---|---|
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `span_buf_t`/`span_buf_ensure`/`span_buf_acquire`/`stage_span_fill`; the gather pre-pass call; the `stage_table` call; `g_stage_span_*` + `g_bounce_read_*` counters; `MOE_STAGE_SPAN_MIN`; two report lines |
| `archive/work/host-expert-split/tools/*.sh` | `LV=`, `NGEN=`, `SINK=`, `HEATMAP=` honouring; coherence arm stream split + vacuity fix; batched arm `-lzm auto` |
| fork branch | `p1-pool-rank-r223` + the W4 commit(s) on top |
