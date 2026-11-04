# RESOLUTION — `no-xnack-and-avalon` retired (2026-11-04, r213 delivery session)

**Status: CLOSED AND ARCHIVED — an analysis note whose question is answered and whose recommended techniques are
either shipped or superseded.**  The directory moved here from `wip/no-xnack-and-avalon/` and no stub was left
behind (`AGENTS.md`, WIP and promotion rules).  It was never in the `wip/CAMPAIGNS.md` table: it was a reading
note, not a campaign (no code, no patches, nothing applied).

## What it asked, and the answer

*Can host-expert reads be done at speed on a GPU with no XNACK?*  Its 2026-11-02 conclusion, unchanged:

> A GPU without XNACK cannot take a page fault and retry, so a device access must target memory that is **present**.
> `hipHostRegister` does not provide presence (it installs page tables over reclaimable page-cache pages).  Two
> mechanisms exist — hold the pages (`mlock`, which erases the RAM saving) or **never let an SM read them**.  Avalon
> takes the second: every expert compute read is VRAM, host -> VRAM moves are copy-engine copies, the mapping is
> registered only for those copies, and the only host memory an SM touches is pinned.

That answer is what the delivery's shipped host tier already embodies, which is why the note closes: the r210/r211
tier (`-lme`, the pinned per-expert pool, the arena's complement, in-place pooled reads resolved through a device
map) never makes an SM read a reclaimable page, and `INVARIANTS.md` already carries the rule in its own words
(device-visible host memory must be resident; the mapping is read by the **copy engine**, not by an SM).

## Its two "worth taking" items

| item | disposition |
|---|---|
| registered-mapping alias fill (115 -> 61 us/fill) | **shipped** — the beta's `MOE_EXPERT_WINDOW_ALIAS_FILL` line of work became the tier's fill path (r210/r211); the note's independently measured 115 -> 61 us corroborates the direction |
| lend slab space to a prefetch ring and take it back | **already had** — r168's wide-aux growth plus r87's forced prefill -> decode drop (block 10) |
| **copy-engine PREGATE prefetch of the next layer's experts during prefill** | **not shipped, and not carried into `TODO.md`** — it is the same class of idea as `TODO` #140's host-side read-ahead, which was then **built and measured as a loss in both bands** (`archive/work/host-expert-eager-push/P6-POOL-REGION.md` §9): an eager copy bets against the signal that chose the victim, and it competes with the demand path for the same resource (on disk, prefill 24.1 -> 16.3 t/s).  A future attempt needs a *predictor* for re-admission, scored offline with `scripts/moe-trace-replay.py` — see `TODO` #140's refutation in `archive/work/host-pool-speed/RESOLUTION.md` |
| CPU-compute the missing experts concurrently | a different campaign (`-ncmoe`'s own CPU path); not this note's |

Its **"NOT worth taking"** finding — Avalon's tail-expert skipping (`GGML_CUDA_MOE_TAIL_SKIP`) deletes a term from
the layer's arithmetic for +5.3 % / +11.9 % — is a delivery rule, not a TODO: `GREEDY-PURITY.md` **§83**, including
the rule for benchmarking against an engine that ships it.

## Why it retired now

The mode the note was written to criticise (`--host-experts window`, the SM-in-place-read bet, which is why it
needed `mlock` and therefore stopped being a RAM saving — `WORKLOG.md` 261102.012/.013) is **not in the delivery**:
`--host-experts` ships `pinned` (default), `pool`, and the r210/r211 `-lme` budget over the pinned per-expert pool
(`mmap` was removed in r33, `window` never shipped).  With the question answered, the techniques dispositioned and
the one refused item recorded as a permanent rule, leaving the note in `wip/` claimed work that is not in flight.

**Evidence:** this directory's `README.md` (the code-cited mechanism table against `~/Avalon` at `c24c03d84`, the
caveats, and the two measurement notes it quoted), `WORKLOG.md` 261102.012/.013, `GREEDY-PURITY.md` §83.
