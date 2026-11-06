# HANDOVER — `host-expert-split`, state at 2026-11-06 (end of session)

**Read this first, then [`PLAN.md`](PLAN.md) for the standing plan and [`W1-SINK-TAX.md`](W1-SINK-TAX.md) for the
cost measurement.**  This file is the session handover: what is decided, what is built, what is proven, what is
owed, and the traps that cost time so they are not paid for twice.

Nothing from this campaign is in `patches/`.  The delivery repo is untouched except for `wip/` docs and
`scripts/` tooling; `release.json` is unchanged.

---

## 1. The campaign in one paragraph

The r223 pool (`--host-experts` / `-lme`) holds a **pinned per-expert cache** of the host MoE set and serves a miss
from that tier (in place, or a D2D fill) instead of from the model mapping.  It had two weaknesses: its content
was chosen by a heat prior that is *worse than chance* for decode routing, and it had **no write-back**, so an
arena eviction simply discarded a recently-hot expert.  This campaign fixed both, measured each, and found that
they compose: **+21.7 % single-user generation at `-lme 36`** with zero configuration, text byte-identical, and
`npl 4` as an accepted trade (-8 % on the batched path, decided by the maintainer as acceptable because a
single-user stream is the common case).  The one remaining frontier is **prefill streaming granularity** (§7),
which is worth more than everything above at small budgets.

## 2. Decisions taken (do not relitigate without new data)

| # | decision | basis |
|---|---|---|
| D1 | **Ranking ships ON** (`MOE_HOST_POOL_RANK=decode`, `combined` = kill switch) | +11.1 % single-stream at `-lme 36`; -5.1 %/‑13.2 % at npl 1/4 |
| D2 | **Demote ships ON** (`MOE_HOST_POOL_SINK=1`, `0` = kill switch) | +7.6 % on top of D1; never harmful in any cell measured (+1.8/-0.9/+0.3 % batched) |
| D3 | **`npl 4` is an accepted limitation**, not a blocker | maintainer's call; +21.7 % single-stream outweighs -8 % at 4 slots |
| D4 | **Headroom `K` is a fraction of slots** (`_PCT=4`, floor 2, cap slots/8), not a flat count | at `-lme 36` a table has ~393 slots, so the measured optimum K=16 is 4.1 % of *slots*; a flat 16 would be 6.25 % on a 256-expert table |
| D5 | **LFRU period stays 32** | plateau 32–64 with 32 at its top edge; below 32 the gate rejects more and cold re-reads dominate |
| D6 | **Never default a workaround for a bad prior**: the demote is the fix, the ranking is a hint | the demote measures, the ranking predicts |
| D7 | **`invert` mode must not ship** (diagnostic; remove before promotion) | it exists to prove the counter direction and was never a product option |
| D8 | **Prefill read granularity is the next target**, ahead of every gate | the drive does 6.7 GB/s; we stream at ~2.3 GB/s (see §7) |

## 3. Code state

* Fork: `~/llama.cpp`, branch **`p1-pool-rank-r223`**, tip **`bfb9b0b22`**.  Working tree clean.
* Build: `~/bin/build-llama-rocm` → `build-rocm` (ROCm 10.1, gfx1201).  The built binary is this branch.
* Branch commits, oldest first:
  | sha | what |
  |---|---|
  | `b2a8d2e76` | `MOE_HOST_POOL_RANK=combined|decode|floor|invert` + `_FLOOR` (the ranking switch) |
  | `edb13bb35` | `MOE_HOST_POOL_SINK_TAX` — the W1 diagnostic (one extra D2H per admitted miss) |
  | `cd2317032` | `expert_src_done` was called twice per fill — removed the duplicate |
  | `2b63fdd18` | `MOE_HOST_POOL_SINK` — the demote, via an in-pool exchange |
  | `a24d442c3` | a real reserve (`keep_free`) instead of shrinking `slots` |
  | `06d05092e` | the reserve check must not cost a full slot scan per fill |
  | `2f12008ac` | **defaults flipped** (D1/D2) + the reserve as a fraction (D4) |
  | `b70ef9434` | the exchange is **inert against a full mirror** |
  | `bfb9b0b22` | reclaim **duplicate-only** pool slots when a demote finds the pool saturated |
* Environment variables in this build (all read in `moe-expert-cache.cu`):

  | var | default | class |
  |---|---|---|
  | `MOE_HOST_POOL_RANK` | `decode` | kill switch (`combined` restores r223 behaviour) |
  | `MOE_HOST_POOL_RANK_FLOOR` | 50 | tuning (only for `floor`) |
  | `MOE_HOST_POOL_SINK` | on | kill switch (`0`) |
  | `MOE_HOST_POOL_SINK_SPARE` | -1 (auto) | absolute override |
  | `MOE_HOST_POOL_SINK_SPARE_PCT` | 4 | tuning (% of a table's slots) |
  | `MOE_HOST_POOL_SINK_SPARE_MIN` | 2 | tuning |
  | `MOE_HOST_POOL_SINK_TAX` | 0 | **diagnostic — must not ship** |
  | `MOE_EXPERT_CACHE_PERIOD` | 32 (pre-existing) | tuning, D5 |

## 4. What is proven, and on what

All single-user numbers: `llama-cli`, `prompts/prose-rdna-boosts.txt` (5 298 tok), `-n 64`, `--seed 42 --temp 0`,
`-ncmoe 48 -ngl 99 -ot 'per_layer_token_embd.*=CPU' -lm mmap -lzm off -fa on -c 8192 -ub 4096`,
`MOE_HOST_POOL_DIO=1`, `MOE_EXPERT_CACHE_MIB=4096`, `HIP_VISIBLE_DEVICES=0`, one R9700, `gpu-watch.sh` around
every run.  Model: `Qwen3.8-Flash-Next/GSQ-RCO-IQ3_S` (144 tables × 512 experts, 46.84 GiB host set).

**4.1 The headline (`-lme 36`, 3 reps where noted, all reproducible)**

| arm | gen t/s | vs r223 (22.6) |
|---|---|---|
| `combined`, no demote (= r223) | 22.6 | — |
| `decode`, no demote | 25.1 | +11.1 % |
| `combined` + demote | 24.3 | +7.5 % |
| `decode` + demote (**shipping**) | **27.5–27.6** | **+21.7 %** |

**4.2 W3 budget curve, shipping configuration (prefill / gen t/s)**

| `-lme` | 12 | 15 | 18 | 21 | 24 | 27 | 30 | 33 | 36 | 39 | 42 | 44 | 46 | 48 (mirror) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 94 | 104 | 116 | 131 | 150 | 174 | 207 | 251 | 331 | 473 | 672 | 740 | 778 | — |
| gen | 10.4 | 12.0 | 14.1 | 16.2 | 19.0 | 21.3 | 24.4 | 25.8 | 27.8 | 28.3 | 28.7 | 29.4 | 29.1 | **31.2** |
| pool h | .045 | — | .064 | — | .094 | — | .141 | — | .231 | — | — | .780 | — | — |

The **mirror is 31.2 t/s** and the exchange is provably inert there (0 demoted, 0 retired, 144 mirror tables,
47 962.5 MiB pinned = the whole host set).  A mirror that fits beats everything, as it must.

**4.3 Batched gate (`-lme 36`, `-lzm auto`, npl 1/4/8)**

| arm | npl 1 | npl 4 | npl 8 |
|---|---|---|---|
| `combined` | 26.22 | **50.07** | 33.72 |
| `decode` | 24.89 | 43.46 | 33.64 |
| `decode` + demote | 25.21 | 45.90 | 33.53 |
| `combined` + demote | **26.70** | 49.60 | 33.81 |

**4.4 Economics**

* W1 (`W1-SINK-TAX.md`): a 1-D D2H swap costs **9.6 µs** per 679 762 B at `-lme 36` (`c_pcie` 0.0140 ms/MiB,
  ~2/3 hidden behind compute), 22 µs at `-lme 44`.  111 D2H copies per decode token at 36.
* Reuse: **7.9 %** of demoted slots are used again at period 32, against a **~3 % break-even** on this box's
  NVMe (0.2 % on a 125 MB/s device).  That is *why* the demote pays, not a coincidence.
* Text identity: `sha=880c932888f8` (126 chars) for `combined`/`decode` × `SINK 0/1` with the pool engaged, and
  for the shipping defaults.

## 5. What is NOT proven (do not quote these as settled)

* **One model, one prompt, one device, 64 generated tokens.**  Nothing here says how it behaves on another model,
  a different prompt population, or multi-device.  `-sm tensor` declines the demote by design (2-D strided).
* **The `-lme 12..34` cells have no control arm.**  The old-default (`combined`, no demote) curve was never
  measured across the sweep; only `-lme 36` has the full decomposition.  This matters for the release table.
* **`allow_empty_fill` A/B** — restored to `true` after the reserve fix, never A/B'd on its own.
* **`floor` mode** is measured only at `-lme 36` single-stream (24.4 t/s); it is not swept.
* **The demote's interaction with `-sm layer` multi-device** is untested.
* **Windows portability** of the new path is unverified (`cudaHostAlloc`/`cudaMemcpyAsync` only — the earlier
  code is portable, so this is expected to be fine but is not proven).

## 6. Gates owed before promotion (all mandatory per `MANIFESTS.md`)

| gate | state | note |
|---|---|---|
| same-seed greedy text | **PASS** `880c932888f8` | must be re-run per cell at promotion |
| `scripts/gate-prefill-logits.sh` | **BLOCKED** | needs a base; `~/.cache/rdna-boosts/gates/` has only gpt-oss (20B) and 35B-A3B entries.  **Record the base from the DELIVERY TIP build first** — recording it from this branch is vacuous |
| MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) | **NOT RUN** | the draft head exists: `/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` |
| `llama-batched-bench -npl 1,4,8` | run, gate only | D3 accepts the npl-4 cost; re-run at promotion on `-lzm auto` |
| `scripts/gate-per-block-build.sh` | **NOT RUN** | needs the change folded into blocks 09/12 first |
| `scripts/gate-host-pool.sh` | **NOT RUN / MUST EXTEND** | the demote adds a fill-like path; the "a failed demote must DECLINE, never take the op over" rule (#274) needs an arm |
| `scripts/check-gpu-faults.sh` | clean on every arm run here | |
| `invert` removal | **TODO** | D7 |

## 7. The open problem: prefill streaming granularity (the next thing)

**Observation (maintainer):** during prefill the box streams ~2.3 GB/s from a drive that sustains ~7 GB/s.

**Measured on this box (`dd iflag=direct`, the model shard):**

| shape | throughput |
|---|---|
| one 0.6 MB (627 200 B) read, depth 1 | **520 MiB/s** (1.1 ms) |
| 630 784 B (4 KiB-aligned) | 565 MiB/s — **alignment is not the issue** |
| 2 MiB, depth 1 | 1.5 GiB/s |
| 16 MiB, depth 1 | **4.2 GiB/s** |
| 0.6 MB at depth 16 / 32 | 3.0 / **4.5 GiB/s** |
| 4 concurrent sequential streams | **6.7 GB/s** |

A 0.6 MB read is **latency-dominated**, so throughput is set by *read size* and *outstanding depth*.

**The offending code:** `moe_cache_stage_table` (`moe-expert-cache.cu`, ~line 7305) walks `e = 0..n_experts` and
calls `expert_host_src(...)` + **one `cudaMemcpyAsync` per expert**.  Every expert that is not arena- or
pool-resident therefore becomes a separate 0.6 MB read, sequentially, **per ubatch**.  Experts `e..e+n` are
contiguous in the file (`file_offs + e*host_bytes`), so those reads could be one span.

**Fix spec:**
1. Restrict to the simple case: `!batch`, `!do2d`, `g_pool_dio`, `n_slices == 0`, `pool->host_bytes == t.expert_bytes`.
2. Pre-pass per table: walk `e`, find maximal runs of consecutive experts that are in **neither** tier.
3. For each run, take chunks of ~16 experts (~10 MB): one `pread` (`pool_dio_read`) into a reusable pinned
   buffer, then one `cudaMemcpyAsync` of the chunk into `dst + e0*expert_bytes`.  Fall back to the existing
   per-expert path for any chunk whose read fails (and never fail the stage on a short read: the mapping is
   still a valid source).
4. Counters: chunk reads, bytes, and a fallback count; report beside `pool fills`.
5. Acceptance: prefill t/s at `-lme 12/18/24/36` improves (expect 1.5–3× on the disk legs); decode and text
   unchanged; no GPU faults; decline count unchanged.

**Caveat that must be handled:** every experiment on this box ran against a **warm page cache** (18+ runs over
the same 54 GB shard; `drop_caches` needs root).  A cold-cache verification needs
`wip/bench-56gb/run.sh`, which is **password-blocked** for the agent — the maintainer must start it.  Until then,
"it did not help" on this box may mean "there was nothing to fetch".

## 8. Negative results — do not repeat these

1. **A concurrent 4-worker demand-fill pool changed nothing** — and dispatched **zero jobs**, because at
   sub-mirror budgets the pool is saturated so `pool_free_slot_locked` never returned a slot.  The concurrency
   was never exercised.  If retried, target the **prefill stage path** (§7), not the demand path.
2. **`madvise(MADV_WILLNEED)` hints on routed experts changed nothing** — warm page cache.  Unverifiable here;
   reverted rather than shipped.
3. **LRU eviction on a saturated demote is harmful**: at `-lme 24` it evicted 10 220 residents, doubled the
   demotions (8 019 → 17 367) and cost a third of the throughput (18.7 → 12.4 t/s).  Only **duplicate**
   reclamation is safe (an expert the arena also holds).
4. **Reserving headroom by shrinking `slots` reserves nothing** — `slots` is the capacity everywhere, so the
   fill paths just fill the smaller pool.  The reserve must be a `keep_free` argument that fills respect and only
   the demote may dip below.
5. **A periodic conditional swap** (N rounds, K per round) is 4–5× less copy-efficient than on-demand and is
   negative on a page-cached box (`STEP1-REPLAY.md` §6).
6. **The warmer is the wrong instrument for a constrained box** — it moves latency, not bytes, and competes with
   a bandwidth-bound fill path.

## 9. Methodology traps that cost time this session

1. **`llama-batched-bench` is a GATE, not a decider.**  Its `npp 512 -ntg 128` mix is prefill-dominated and the
   ranking's sign **inverts** between it and the single-user path.  Deciding a default from it produced a
   retracted +20.8 %.
2. **`-lzm off` for single-user decision arms; `-lzm auto` for the batched gate.**  With `off`, PLE is resident
   (~24 GiB); plus a 36.8 GiB pool and a 23 GiB arena the box over-subscribes and `npl 8` **collapses from 33.7
   to 8–18 t/s** (MemAvailable min 46 → 18 GiB).  That reads exactly like a policy regression.  `arm-batched.sh`
   therefore defaults to `auto`.
3. **Never hash an `-lv 2`/`-lv 3` log.**  Per-token timing interleaves into the generated text (1 691 chars vs
   126).  Hash at **default verbosity**; the pool report is a WARN and is invisible at `-lv 1`, so verify pool
   engagement out-of-band at `-lv 3` on the identical config.
4. **The pool is built at the first decode-band pass.**  A 9-token prompt never engages it — the original 9/9
   coherence claim was measured **with no pool at all**.
5. **`gpu-watch.sh` word-splits `--cmd`**: `FOO=1 cmd` becomes argv[0].  Put env prefixes on the wrapper
   invocation (they are inherited), not inside `--cmd`.
6. **Trap the model process or it orphans.**  A killed wrapper left `llama-cli` holding 32 GB of VRAM and OOM'd
   the next six arms.  `arm-decay.sh` now has a TERM trap.
7. **Check MemAvailable before trusting any `npl 8` number.**
8. **Counters, not hand-divided columns.**  Three separate wrong numbers this session came from arithmetic
   (pool_h read as "fraction answered", copy volume divided by the wrong token count, a `K` knob silently
   ignored).  Every surviving claim leans on a printed counter or a device run.

## 10. Harnesses and where the records live

| what | where |
|---|---|
| arm scripts (decay / prefill / batched / coherence) | `wip/host-expert-split/tools/` + its `README.md` |
| flight recorder + fault scan | `scripts/gpu-watch.sh`, `scripts/check-gpu-faults.sh` |
| offline policy model | `scripts/moe-trace-replay.py` (`--pool-layout`, `--pool-band`, `--rebalance*`, `--evict-warm`) |
| the evidence | `STEP1-REPLAY.md`, `W1-SINK-TAX.md`, `P1-POOL-RANK.md` |
| design + PR #267 review | `DESIGN.md` |
| the standing plan | `PLAN.md` |
| dated narrative | `WORKLOG.md` 261105.014–019, 261106.001–010 |
| campaign index | `wip/CAMPAIGNS.md` |

The trace is **not** committed (33.7 MB); regenerate it with `STEP1-REPLAY.md` §1.

## 11. Suggested order for the next session

1. **§7 span coalescing** — implement, measure at `-lme 12/18/24/36`, keep or revert on the numbers.  This is the
   largest remaining win and it is self-contained.
2. **Control curve** — `combined` + `SINK=0` across `-lme` 12→46, for the release table.
3. **Gates** — prefill-logit (base from the delivery tip), MTP four-axis, per-block build, `gate-host-pool.sh`
   extension, `invert` removal.
4. **Promotion** — fold into blocks 09 (cache) and 12 (loader/pool policy), regenerate `patches/` from a canonical
   tree, `validate-set.sh`, release ladder, `fork-refresh.sh`.
5. **Docs at promotion** — `ENVIRONMENT.md` (the six variables, classes as in §3; `SINK_TAX` excluded),
   `INVARIANTS.md` r113 "no write-back" **corrected in place** with the release's r-number, `TODO.md` entry for
   the accepted `npl 4` limitation, `patches/README.md` per-block notes.
