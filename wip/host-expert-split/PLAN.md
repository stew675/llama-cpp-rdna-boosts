# PLAN — the rest of the `host-expert-split` campaign (post-investigation, pre-implementation)

**Status: the plan of record from 2026-11-05 onward.**  The investigation is closed; this is what gets built,
in what order, with what gates, and what would make us stop.  Nothing here is in `patches/` yet.

Read with: [`README.md`](README.md) (the scoping note), [`DESIGN.md`](DESIGN.md) (the PR review + convergence
design), [`STEP1-REPLAY.md`](STEP1-REPLAY.md) (the evidence, incl. §7 the ranking finding and §8 the
write-back-vs-warmer pricing), [`P1-POOL-RANK.md`](P1-POOL-RANK.md) (the one arm already implemented and
measured).

---

## 1. Settled — do not re-derive these

| # | fact | value | where |
|---|---|---|---|
| S1 | The pool's content is ranked by `t.heat`, the combined prefill+decode counter, which is **worse than chance** for decode routing (top-64 overlap 10.5 % against 12.5 % chance); `heat_dec` is 24.2 % | — | STEP1 §7 |
| S2 | Ranking the pool by `heat_dec` instead is **+10.6 % generation**, +1.1–2.1 % prompt, 0 copies, 0 bytes | 22.6 → 25.0 t/s | P1 |
| S3 | The ranking is **provably inert at a mirror** (identical counters to the digit; 833.7/835.0/833.4 t/s) | — | P1 |
| S4 | Our deficit is **holes**, not transport: at a full-complement budget (94 % of the set) the pool answers 0.795 of arena misses and **23–30 MiB/token** still comes from storage | — | STEP1 §7, PLAN §5 |
| S5 | An exclusive tier (the PR's shape) has **no holes by construction**; that — not D2H economics — is its entire advantage | — | STEP1 §8 |
| S6 | r113's refusal of the literal swap was measured **with a covering pool** (0 MiB of L3 available to win) **and** 2-D strided `cudaMemcpy2DAsync` copies on a tensor split. It does not bound the complement-scarce, single-device case | — | STEP1 §8 |
| S7 | A write-back is **slot-neutral**: the promoted expert vacated the pool slot the victim takes, so it needs no extra memory | 1 D2H per admission | STEP1 §8 |
| S8 | A page-cache **warmer** is worth −15…−20 % of storage reads at **zero** copies, but **only in mmap/buffered mode** — with `MOE_HOST_POOL_DIO=1` the page cache is bypassed and it is a no-op | — | STEP1 §8 |
| S9 | A better prior and a demote are **not independent**: P1 already removed 12–19 % of the hole traffic for free | — | P1, STEP1 §7 |
| S10 | A periodic conditional swap (N rounds, K per round, hottest-pool vs coldest-arena) is **4–5× less copy-efficient** than an on-demand one and is negative on a page-cached box. **Not worth a knob.** | 0.12–0.18 vs 0.65 MiB saved/MiB copied | STEP1 §6 |
| S11 | The copy direction and vector order are **correct** (an `invert` probe collapses to h 0.114 / 16.7 t/s) | — | P1 |

## 2. The one number that decides the rest of the campaign

Everything now reduces to an exchange rate between **pinned RAM** and **PCIe copies**, and it hangs on a
single unmeasured constant: the marginal cost of our own 1-D swap.

* To match an exclusive tier's L3 on our no-copy layout we must pin **+9.2 … +17.7 GiB** (mean **+12.8 GiB**);
  to reach *zero* storage reads we need the full mirror (46.8 GiB) where the tier needs ~29 GiB — **17.8 GiB
  apart**.
* At **equal L3**, the demote costs those copies instead: **1.10 ms/token** at `c_pcie ≈ 0.04 ms/MiB` — about
  9 % of a 12.8 ms token — and frees the 12.8 GiB.
* At `c_pcie ≈ 0.4 ms/MiB` (the r113 2-D shape scaled to 1-D) the same trade costs **11 ms/token** and the
  demote is worthless.

**Therefore: measure `c_pcie` for a 1-D contiguous D2H on this tree BEFORE writing the policy.**  That is W1,
and it is ~30 lines rather than ~150.

## 3. Work items, in order

### W0 — close out P1 (cheap, and it may be the campaign's real deliverable)

| item | why | pass criterion |
|---|---|---|
| MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`) | MANIFESTS demands it for any decode-path change; **not run** | within noise of `combined` on R/P/C/K, and text byte-identical |
| `scripts/gate-prefill-logits.sh` | needs a base; none exists for this model | record the base **from the delivery tip** (one build of the unmodified tree) first, then run — recording from the modified build is vacuous |
| repeat the `-npl 8` batched-bench row | the +20.8 % was a single run | confirms or kills it |
| `ENVIRONMENT.md` | rule 9: `MOE_HOST_POOL_RANK`, `MOE_HOST_POOL_RANK_FLOOR`; `invert` must not ship | entry written, classes assigned |
| default-on decision | repo policy is "beneficial ⇒ on, env only disables" | maintainer's call; `floor` is the defensible flip value because of the staging duty |

### W1 — instrument the swap's marginal cost (`MOE_HOST_POOL_SINK_TAX=1`, diagnostic)

The demote adds **exactly one D2H per admission** (the H2D to the arena already happens on that miss, and the
slot is the one the promoted expert vacated).  So the marginal cost can be measured *without* implementing the
policy: on every admitted miss, also issue one 1-D D2H of `expert_bytes` into a scratch pinned buffer, and
change nothing else.

* Arms: `SINK_TAX=0` vs `1`, same binary, same config, `MOE_HOST_POOL_DIO=1` so misses are real.
* Report: decode t/s delta at two pool budgets (a below-complement one and a full-complement one), plus the
  observed admissions/token, so the cost is quoted **per copy** and **per token**.
* Measurements under `scripts/gpu-watch.sh --cmd`, 3 runs per arm in both orders (the P1 protocol).
* **Go/no-go:** cost ≤ ~1.5 ms/token at the full-complement cell ⇒ build W2.  ≥ ~3 ms/token ⇒ stop, close the
  campaign with "buy the RAM" and hand the analysis to the PR.

### W2 — implement the demote (`MOE_HOST_POOL_SINK`, default **0** until gated)

Design constraints, all of which come from something measured:

1. **Trigger** — in `access_locked`, on an admitted miss with a victim, when the table is pool-backed and a
   pool slot is available (the vacated one first, then a free slot, then nothing).
2. **Direction and ordering** — copy the victim D2H into the pool slot *before* the incoming H2D overwrites the
   arena slot, on the compute stream.  Use the module's kernel-copy form, not `hipMemcpyAsync` (the PR's
   finding: a small pinned copy can leave later queued work unsubmitted on ROCm, which stalls a peer past its
   all-reduce spin budget).
3. **Retire, then publish** — the slot is `slot_state == 2` (retiring) until the copy's event fires; only then
   is `expert_slot[victim]` published.  One ordering state, reused, not a second one.
4. **Decline, never lie** — no available slot, a failed record, or a geometry it cannot express ⇒ **decline the
   demote** (drop the victim as today).  A failed demote must not take the op over — the same rule
   `gate-host-pool.sh` already enforces for a failed fill (issue #274).
5. **1-D only.**  `split_axis == -1 && n_slices == 0`.  A `-sm tensor` axis-0 slice is the 2-D strided copy
   that measured 2× decode in r113; it declines and behaves exactly as today.
6. **No new memory.**  The demote may never grow the pool; if the bookkeeping cannot find a slot it declines.
7. **Counters** — sinks, declines (per reason), bytes; reported once at teardown beside the pool line.

### W3 — the three-arm split curve (the campaign's answer)

One binary, one model, `MOE_HOST_POOL_DIO=1`, stalls measured, `gpu-watch.sh` on every arm:

* **arm A** `MOE_HOST_POOL_RANK=combined` (today)
* **arm B** `MOE_HOST_POOL_RANK=decode` (P1)
* **arm C** arm B + `MOE_HOST_POOL_SINK=1`

swept over `-lme` **30 … 47** (below complement → mirror) at two arena shapes (a small one and the field one),
reporting **pinned GiB vs decode t/s vs storage MiB/token vs copy MiB/token**.  Plot the three curves with the
mirror as the asymptote.

**Accept for C to ship:** it beats B by ≥ ~12.8 GiB of equivalent pinned RAM (i.e. reaches B's L3 at a budget
≥ 12.8 GiB smaller), **or** it reaches zero storage reads within ~2 GiB of the tier's ~29 GiB point; and at the
mirror it is **inert or disabled** (S3: there is nothing to win and the copies are pure cost).
**Accept for B to ship:** it is ≥ 5 % better than A at the below-complement budgets and no worse anywhere —
which the P1 record already suggests.

### W4 — the mmap-only warmer (optional, only after W3)

`MOE_HOST_POOL_WARM`: on arena eviction, `posix_fadvise(POSIX_FADV_WILLNEED)` on the victim's file range.  No
thread (the tree already does this in `llama-lazy-reader.cpp`, `llama-mmap.cpp`, `qwen4exp.cpp`), **armed only
in buffered fill mode** (`pool_decide_dio_locked` resolves it), self-disabling on: warm-hit rate ≈ 0 over N
evictions, warmed pages found non-resident (`mincore`), or reclaim pressure (`/proc/vmstat` `pgscan`/`pgsteal`)
rising while armed.  Long window, one-way latch per session — the idiom `g_pinned_off` / `g_heat_dead` already
use.  **Never presented as a fix for the constrained case**: it moves latency, not bytes (S8).

### W5 — promotion path, if and only if one arm wins

Fold into the owning block (09 for the cache, 12 for the loader/pool policy), regenerate `patches/` from a
canonical tree via `scripts/make-patches.sh`, `validate-set.sh`, per-block gate, then the full release ladder.
New env vars → `ENVIRONMENT.md` with their class (`SINK`/`RANK` as kill switches once default-on). Every
invariant this campaign changed (the r113 "no write-back" entry) gets **corrected in place** in
`INVARIANTS.md`, with the r-number of the release that did it.

## 4. Honest caveats carried into every number above

* One trace (Flash-Next GSQ-RCO IQ3_S, prose prompt, 806 430 decode reaches) and one arena ratio (19 GiB ≈ 43 %).
  The 2-D grid says the tier's advantage **grows** as the arena shrinks, so +12.8 GiB is conservative for small
  arenas.
* `c_pcie = 0.04 ms/MiB` is an assumption until W1 measures it.  It is the whole decision.
* Every model number is a **hit/byte** model: the device measured P1's effect at +10.6 % while the model
  predicted 12–19 % of the *L3 traffic* — same direction, different currency.  Judge arms on device
  measurements; use the model only for ranking candidate policies.
* The persisted prior is **flat** for this model+workload (S1).  Results here may not transfer to a model whose
  heat map is well-conditioned — which would *shrink* the demote's case, since its advantage is foreknowledge.

## 5. Risks

* **A hard machine lockup (2026-11-05)**: a R9700 pinned at 100 %, power-cycled, **no kernel-journal record**.
  No benchmark from this session was running.  `HUNG_TASK` detection is not in this kernel
  (`# CONFIG_DETECT_HUNG_TASK is not set`), so every GPU arm runs under `scripts/gpu-watch.sh --cmd` with a
  bounded `timeout`.  Until the cause is known: **single-device arms only**.
* **`-sm tensor`**: any demote there is a 2-D strided copy — the r113 2×-decode mechanism.  Declines by design.
* **CUDA graph capture**: a demote is a stream write inside a captured graph unless it is only taken on the
  eager path.  Decide before coding: the sink is a decode-band decision, and the band is not captured for the
  eager path — verify, do not assume.
* **Windows**: the demote itself is portable (no `mmap` alias, no `mlock`); the warmer is POSIX-only and must be
  a no-op there, like the existing stubs.
* **The mistake pattern of this session, worth naming**: three times a plausible number turned out to be my
  own arithmetic (pool_h read as "fraction answered", the copy volume divided by the wrong token count, a K
  knob silently ignored by a failed patch).  Every claim in this plan that survives must lean on a printed
  counter or a device run, not on a hand-divided column.

## 6. Where the records live

| what | where |
|---|---|
| the campaign's index entry and status | `wip/CAMPAIGNS.md` |
| the scoping note (what was open before) | `wip/host-expert-split/README.md` |
| the PR review + convergence design | `wip/host-expert-split/DESIGN.md` |
| the evidence (trace, replay, surface, pricing) | `wip/host-expert-split/STEP1-REPLAY.md` |
| the one implemented arm + its gates | `wip/host-expert-split/P1-POOL-RANK.md` |
| the fork arm | `~/llama.cpp` branch `p1-pool-rank-r223` (`b2a8d2e76`) |
| the replayer's new knobs | `scripts/moe-trace-replay.py` (`pool-layout`, `--pool-band`, `--rebalance*`, `--evict-warm`) |
| the flight recorder | `scripts/gpu-watch.sh` |
