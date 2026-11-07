# PLAN — `host-expert-parity`: the measurement first, then (maybe) the code

**Status: OPEN.  P0 is measurement-only and needs no new code.  P1 items are candidates, each behind a kill
switch, each gated on its own.**  Read [`README.md`](README.md) first (why this campaign and what changed about
the box), then [`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) (the per-patch verdicts that produced P1).

> **First pass measured 2026-11-06 — [`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md); the fix shipped in r228.**
> P0.1/P0.2/P0.4/P0.5 are done and P1.0 is **done in r228** (the demote is a decode-class notion: headroom capped
> by the coverage gap, prefill fills exempt, near-mirror pools disengaged — the field cell is now the `SINK=0`
> path to the counter, and `-lme 44` keeps +19-21 %).
>
> **Second pass the same day, on `soar` (3 × R9700, canonical prior) —
> [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md): P1.1, P1.2 and P1.4 are SETTLED.**  `TODO` #276 and
> #277 close as **unreachable by construction** (the demote declines on every `n_slices != 0` table, so it never
> runs beside a paired all-reduce: 0 demoted / 14 480 structural declines on `-sm tensor`, 7 862 copies on
> `-sm layer` where no all-reduce exists) and `TODO` #278 closes as **measured, not worth a code change** (the
> drop is real — 12 tables / 1 062 MiB at the field ubatch — but the re-fill is bounded by the evicted bytes,
> ≤0.2-1.2 % of the prompt, invisible in paired rows, and absent on the 3-GPU tensor shape).
>
> **Third pass, 2026-11-07, on `radix` — [`RESULTS-2026-11-07-radix.md`](RESULTS-2026-11-07-radix.md):
> `TODO` #280 is CLOSED and P0.2/P0.3 are DONE.**  The crossover table (mirror / `auto` / `-lme 44` / `-lme 36`,
> shipping vs `SINK=0`, buffered vs `DIO`, 17 arms + a 7-arm old/new A/B), the parity sentence, the fill-mode
> answer and the span pair are all in that file.  Two things came out of it that matter beyond the table: (a) the
> field shape's non-mirror prefill is dominated by the **band-transition tail** — the pool's re-layout after the
> last prefill ubatch, 0.6 s on a mirror, 7.2 s at `auto`, 13.5-18.6 s at `-lme 44`, all before the first token —
> and its fill order is now **file order** (r231, block 12); (b) **decode is flat from 98 % to 77 % coverage**
> (−9 % for 10.8 GiB less pinned RAM) while prefill is not, and a sub-complement budget should set
> `MOE_HOST_POOL_DIO=1` (+43-81 % prefill, decode neutral with the demote).
>
> **Fourth pass, 2026-11-07 (`radix` + `soar`) — [`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md):
> `TODO` #279 is CLOSED as measured / do not implement and P1.5 is done.**  The narrow-2 eviction branch is
> unreachable on every shipping shape (four probes: the pin's VA spare *is* the draft's own reserve minus the
> headroom, because the preflight skips `slab_extend` at the priming sizing), and when forced
> (`GGML_CUDA_SLAB_RESERVE_MIB`) the eviction is re-absorbed — identical final arena MiB and decode hit rate in
> both arms.  The 0003a preference cannot avoid it either: its own fallback eats the free VRAM its guard needs,
> so the pin is retried 13 ms later with the same eviction; latching it would spend the evicted bytes of cushion
> permanently to save one ~20 ms re-fill.  No code ships; the reference diff is beside the record.
>
> **Fifth pass, 2026-11-07 (`radix`) — [`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md): P1.3 is
> done and the campaign has NO open measurement left.**  `MOE_EXPERT_CACHE_TOUCH=3` damps churn exactly as PR 0008
> claimed (-17.3 %/-18.3 % demotions, -27 % sink uses) and pays 0.78/0.81 pt of arena hit rate plus 35 % more
> cold-served misses for a decode wash (+1.3 to +2.2 % at 93.8 % coverage, inside this box's spread), so **the
> default stays 2**; text identical (`0e3ed3d279f6`).  PR 0002 (sparse registration) closes with it as *not
> needed*: the cold path it would have registered for is live and served 3.3-4.6 % of all reaches with zero
> declines.  Every mechanism in PR #267 now has a verdict ([`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) §3).

## 1. Settled — do not re-derive

| # | fact | where |
|---|---|---|
| S1 | r224 ships a demote (`MOE_HOST_POOL_SINK=1`) into exchange headroom below the budget, a DECODE-ranked pool and coalesced DIO prefill reads | `WORKLOG.md` 261106.013, `patches/README.md` block 12 |
| S2 | On r224's box the demote measured **+5.3 % alone / +7.8 % combined** single-stream at `-lme 36` on a frozen real-world prior; the campaign's own "+21.7 %" was prior leakage | `archive/work/host-expert-split/HANDOVER.md` §4.2 |
| S3 | The demote's copy was **9.6 µs per 679 762 B** (≈0.0140 ms/MiB, ~2/3 hidden), break-even reuse ~3 % on that NVMe, ~0.2 % on a 125 MB/s device — on a **~26 GB/s host link** | `archive/work/host-expert-split/W1-SINK-TAX.md` |
| S4 | A mirror is still the fastest configuration (31.2 t/s vs 27.5 shipping at `-lme` 36/48 on that box, drifted prior), and the demote is **inert** against one (`slots >= n_experts`) | `archive/work/host-expert-split/PLAN.md` §3 (W2) |
| S5 | On the 56 GiB rig a mirror is 78.5 t/s / 1941 prefill, `auto` (97.7 %) is 75.6 / 1443, 92.4 % is 76.6 / 1061 — measured **before** r224 | [`wip/bench-56gb/RESULTS.md`](../../../wip/bench-56gb/RESULTS.md) |
| S6 | The span read is **O_DIRECT-only** (`stage_span_fill` returns unless `t.src_fd >= 0`); the field `runme` mirrors `--load-mode mmap` | block 12 `stage_span_fill`, `pool_decide_dio_locked` |
| S7 | The pool's fill mode is a **policy** (mirror `--load-mode`, else a MemAvailable test) and its own comment calls the **middle band unmeasured** — the 56 GiB rig *is* that band | block 12 `pool_decide_dio_locked` |
| S8 | The bounce ring is deliberately **uncapped** (a cap costs 8-25 % prefill) and a whole-master declined read is answered by r224's span coalescing instead | `bounce_acquire`'s comment; PR #267 0010 |
| S9 | Sparse registration / `mlock` + `MAP_FIXED` is a **non-goal** unless a partial tier cannot be made cheap | `DESIGN.md` §4, `archive/work/host-pool-speed/README.md` §6 |
| S10 | The decision inputs are **single-user cells**; `llama-batched-bench` is a gate, never a tie-breaker (at `-lme` 36 the ranking was +10.7 % single-stream and −11.4 % at 4 slots) | `archive/work/host-expert-split/PLAN.md` §3 (W0), r224 |

## 2. P0 — re-measure on this box (no code, no delivery change)

### P0.0 — prepare (once)

```bash
cd ~/llama-cpp-rdna-boosts/wip/bench-56gb
gcc -O2 -o ballast ballast.c && python3 gen_prompt.py          # once
./ballast 31 &                                                 # ONCE for the whole matrix
grep MemAvailable /proc/meminfo                                 # expect ~57.3 GiB (a 59 GiB box)
```

* Write **`runme-56gb-tier.sh`** (a sibling of `runme-56gb-mirror.sh`) that reads `LME` (`auto|<N>`), `SINK`,
  `DIO` (`auto|0|1`) and `SPAN` from the environment and passes the matching flags/env.  Keep `-lv 4`: the
  census lines this campaign quotes (`pool fills: … storage reads … / mapping memcpys …`,
  `coalesced prefill reads`, `device gather`, `demoted/retired/declined`, `evicted … re-armed`) are `INFO` and
  invisible without it (`INVARIANTS.md`, r224).
* **Freeze the heat prior before any arm.**  The record's canonical map (`sha256 6195152358e1…`, 601 376 B, mode
  444) is **not on this box**; the model-dir sidecar is mode 644 and hashes `a07822d3…`.  Either restore the
  canonical file or take this one deliberately, `chmod 444` it, and **quote its sha256 in every table** — a
  number measured against a writable map measures the benchmark, not the delivery (`INVARIANTS.md`, r224).
* `echo 3 | sudo tee /proc/sys/vm/drop_caches` **before every arm** whose first-pass 50k prefill row is quoted.
* One arm at a time, everything under `scripts/gpu-watch.sh --cmd` with a bounded timeout, `HIP_VISIBLE_DEVICES=0`.

### P0.1 — the demote's exchange rate on a PCIe 5.0 x16 link

This replaces W1's instrument: **`MOE_HOST_POOL_SINK_TAX` was removed at r224**, so the cost is measured on the
real path instead, as the `SINK=0/1` delta at one fixed config.

* Config: the campaign's decision cell — `-lme 36`, `MOE_EXPERT_CACHE_MIB=4096`, `MOE_HOST_POOL_DIO=1`,
  `MOE_HOST_POOL_RAM_PCT=0`, `-lzm off`, prose prompt, `NGEN=64`, one R9700.
* Arms: `SINK=0` / `SINK=1`, three alternating reps each (never block-ordered: a 16 % monotonic drift was
  measured across block-ordered arms, `archive/work/host-expert-mirror/RESOLUTION.md` rule 4).
* Report: gen t/s, `demoted` / `sink_uses` / `retired` / `declined`, and the implied ms/token + µs/copy.
* **Accept:** a number for µs/copy and ms/token *on this link*, quoted beside W1's 9.6 µs.  If the copy gets
  cheaper by roughly the link ratio, the demote's break-even reuse falls below W1's ~3 % and P1.1/P1.2 lose
  their case; if the copy costs **more** than ~1 % of the step, P1.1 is justified.  Do **not** compare the
  absolute t/s with the record's 26.2 — the prior differs (see P0.0); only within-run deltas are comparable.

### P0.2 — the field shape: the middle band, the parity row and the crossover (the campaign's product)

The rig, one arm per run (`run.sh` + `measure.sh`, `-c 204800`, `-ub 6144`, `--parallel 1`, `-ncmoe 48`):

| tag | tier | `SINK` | `DIO` | question it answers |
|---|---|---|---|---|
| **M1** | `-lme 47` + `RAM_PCT=0` (mirror, 100 %) | — (inert) | auto | the asymptote: 0 misses, 0 copies |
| A0 | `auto` (97.7 %) | 0 | auto | r223 behaviour on this box — the pre-r224 field default |
| **A1** | `auto` (97.7 %) | 1 | auto | **r224 shipping on the field config: the parity row** |
| A2 | `auto` | 1 | 1 | the memory-tight arm: does O_DIRECT win the middle band (S7)? |
| A3 | `auto` | 0 | 1 | A2's control |
| B0 | `-lme 44` (≈94 %) | 0 | auto | a sub-complement budget, control |
| **B1** | `-lme 44` | 1 | auto | does the demote hold the 92-94 % cliff at bay? |
| C0 | `-lme 36` (76.9 %) | 0 | 1 | the campaign's cell, control (DIO) |
| **C1** | `-lme 36` | 1 | 1 | reproduce the +7.8 % on this link (DIO) |
| C2 | `-lme 36` | 1 | 0 | C1's **buffered twin** — the fill-source arm |

* **Accept (parity):** the smallest budget whose decode is within **3 %** of M1 **and** whose first-pass 50k
  prefill is within **5 %** of M1.  That budget is the campaign's headline ("a tier of N GiB behaves like a
  mirror on this box"); if none of 94 %/76.9 % gets there, the headline is the gap table instead.
* **Accept (fill mode):** state which of A1/A2 wins the middle band, and quote the census
  (`storage reads` vs `mapping memcpys`) as the mechanism.  If `auto` (buffered) loses, the recommendation is a
  **documentation** change first (`--load-mode` / `MOE_HOST_POOL_DIO` guidance) — a default flip needs a second
  box.
### P0.3 — where the coalesced prefill read actually runs

The r224 record's `+81 %` (at `-lme` 12) / `+57 %` (18) / `+14 %` (36) were all measured with
`MOE_HOST_POOL_DIO=1`, and `stage_span_fill()` returns unless `t.src_fd >= 0` — so they are DIO-arm,
cache-staging numbers (S6).  Two things to establish on this box, because the field `runme` mirrors
`--load-mode mmap`:

* **Does the span fire at all on the field path, and where?**  Read the `-lv 4` census: `coalesced prefill reads`
  vs `device gather` vs `prefill cache staging`.  A mirror routes the prefill through the device gather
  (`archive/work/host-expert-mirror/RESOLUTION.md`), and a pool-backed *partial* tier stages through the cache —
  the span lives in the latter's read paths, so "no gain" and "not engaged" must not be confused.
* **A clean pair:** the same config with `SPAN=0` vs `SPAN=1`, **both `DIO=1`**, 3 alternating reps.  Do **not**
  use a `DIO=auto` vs `DIO=1` pair for this: the fill *source* changes with the mode, so that pair is two
  changes at once.

*Accept:* the span's delta on this box at the field config **and** at `-lme 36`, with the census that shows where
it ran.  If it is inert under a buffered fill, `ENVIRONMENT.md`'s `MOE_HOST_POOL_STAGE_SPAN` row and the r224
`WORKLOG.md` entry must say the three percentages are DIO-arm cache-staging numbers — do not leave a reader to
infer otherwise.

### P0.4 — the two gates r224 owes

* **Prefill-logit, re-recorded here first.**  No base on this host matches the r224 tip (`build=37148d118` is not
  in `~/.cache/rdna-boosts/gates/`; the 27B and 35B-A3B bases are r191-era).  Record from `88e7d88a7` (the tree
  in `~/llama.cpp` today) on ROCm 10.1.0, then run `--ab "MOE_HOST_POOL_STAGE_SPAN=0"` on the **35B-A3B**
  (`/llm/models/Qwen3.6/35B-A3B/Q4_K_M/`), which is the config the record says reaches
  `moe_cache_stage_table` — the span arm's first proof.  This box is one GPU: **check at `-lv 4` whether the
  1-GPU 35B-A3B arm actually reaches `prefill cache staging:`**, and if it does not, record the arm as owed to a
  multi-GPU box instead of claiming it.
* **MTP four-axis** (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`) on the dense 27B — owed by r224 and
  runnable here.  The change is inert on a dense model (no host pool), which is the point of the gate.

### P0.5 — the batched gate, 3 reps (the D1 input)

`llama-batched-bench -npl 1,4,8`, `-lzm auto`, `-lme 36` and 44, shipping vs `MOE_HOST_POOL_RANK=combined`
(with `SINK=1` in both), three reps, interleaved.  This is a **gate** (S10): it can block a default, it cannot
choose one.  `TODO` #275 is the accepted-limitation row this re-quantifies.

## 3. P1 — code candidates, in dependency order

Every item: **kill switch, default unchanged until it has a device win here**, one item per change, its own gate.

| # | item | do it only if | gate |
|---|---|---|---|
| **P1.0** (TODO #281 — **DONE in r228**) | **the exchange headroom must not exceed the pool's coverage gap, and must not gate the prefill fill path** — at `auto` (97.7 %) `K = 4 %` of slots (≈19/table) is larger than the gap (12/table), so every prefill fill is refused and the missing experts are re-read through the bounce ring on every ubatch (measured: 1 837 → 867 t/s first-pass prefill, `SINK=0` → `SINK=1`, decode flat) | already measured — this is a defect, not a candidate | coherence, `gate-host-pool.sh` (the existing `sink`/`sinkfail` arms), the `-lme 44` cell must keep its +21 %, the 97.7 % cell must return to ≈1 840 prefill, prefill-logit |
| **P1.1** (TODO #276 — **CLOSED 2026-11-06, unreachable**) | the demote's transport: a copy-kernel form of the D2H + a drain on a failed event record (PR 0004/0006/0007's rationale: a small pinned copy can leave later queued work unsubmitted on ROCm) | **settled: the hazard needs a copy AND a paired all-reduce, and the delivery never has both.**  On `-sm tensor` (the RCCL banner is in the log) the demote declines **100 %** of offers — `0 demoted / 14 480 geometry (structural)` — because a split table has no contiguous pool image; on `-sm layer` (7 862 copies) there is no all-reduce.  The cost side is net-positive (+19-21 % at `-lme 44`).  Re-entry: only if the demote is extended to an `n_slices != 0` table (a strided 2-D D2H, r113 = 2× decode loss) |
| **P1.2** (TODO #277 — **RETIRED 2026-11-06**) | precopy: K spares + a side stream that copies the policy's likely victims ahead of time (PR 0006/0008), including 0007's drain semantics | its own condition was "P1.1 leaves a residual on the admission path", and P1.1 closed as unreachable — the shape was already decided to need a dedicated K-spare ring, never more headroom (`PR-267-REMAINDER.md` §2) |
| **P1.3** (— **CLOSED 2026-11-07, measured: default unchanged**) | `MOE_EXPERT_CACHE_TOUCH` 2 vs 3 **with the demote on** (a knob: no code) | done: two live-demote cells, 3+3 interleaved reps each; churn -18 %, residency -0.8 pt, decode a wash | coherence (text identical, `0e3ed3d279f6`) |
| **P1.4** (TODO #278 — **CLOSED 2026-11-06, measured**) | keep the arena across a decode→prefill transition (PR 0003b) | **settled negative: the drop is real, the cost is not.**  Flash-Next 1 GPU, field ubatch (`-c 204800 -ub 6144`): 3 `moe_cache_evict_slab_range` events, 1 062 MiB, one `re-armed 12 stood-down` — bounded by the evicted bytes to 37-56 ms, i.e. ≤0.2-1.2 % of the prompt, and the paired 4 096-token rows measure row 3 **+15.8 % over** row 2 across a transition.  **0 evictions** on the 3-GPU `-sm tensor` field shape.  Re-entry: an arena *exactly* slab-full with short (≤4k) prompts, or the 56 GiB rig measured directly |
| **P1.5** (TODO #279 — **CLOSED 2026-11-07, measured / do not implement**) | narrow-2: prefer the `cudaMalloc` fallback over evicting arena tables when the draft compute region would fit there (PR 0003a) | **settled negative**: the branch is unreachable on every shipping shape and, forced, its eviction is re-absorbed (identical arena MiB + hit rate); the preference cannot avoid it without a latch that costs the evicted bytes of cushion permanently.  [`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md) | — |

Not here: the tier's policy/visibility work (cgroup figure, authoritative `-lme`, verdict line) is
[`wip/host-expert-coverage/`](../../../wip/host-expert-coverage/README.md); sparse registration is a non-goal (S9); the
bounce-ring cap is rejected with a number (S8).

## 4. Gates for anything that becomes code

`MANIFESTS.md` is the contract; the quick answer for this family:

* coherence (same-seed greedy, pool **engaged**, prose prompt, `-lv 3` with **split** streams — never hash a
  merged log);
* `scripts/gate-host-pool.sh` (a failed demote must DECLINE, never publish; the `sink`/`sinkfail` arms);
* `scripts/gate-prefill-logits.sh` for any read/staging-source change (the fill-mode A/B is a good first use:
  buffered vs O_DIRECT should be KLD 0);
* MTP four-axis for any decode-path change (the demote's own transport is one);
* `scripts/gate-per-block-build.sh --changed-from <prev tip>` and `scripts/check-gpu-faults.sh` before a tag;
* `scripts/validate-set.sh` only when the delivery set changes.

## 5. Traps — inherited, and the new ones

Inherited (all measured; see `archive/work/host-expert-split/HANDOVER.md` §8/§9): a 20-token generation
under-reports gen by 13-15 %; `-lzm off` for single-user decision arms and `-lzm auto` for the batched gate;
`INFO` counters need `-lv 4`; the pool is built at the first decode-band pass, so a 9-token prompt never engages
it; `gpu-watch.sh` word-splits `--cmd`, so env prefixes go on the wrapper; trap the model process or it orphans;
check `MemAvailable` before trusting `npl 8`; `/tmp` is shared, so never glob a patch series into `git am`.

New for this campaign:

1. **The prior is not on this box.**  Freeze it (mode 444) and quote its sha256 or the number is not comparable
   — with anything, including the record's own 26.2.
2. **`MOE_HOST_POOL_SINK_TAX` no longer exists** (removed at r224).  Do not look for W1's instrument; measure the
   `SINK=0/1` delta.
3. **Drop the page cache before every first-pass arm** — the 50k row is the one that shows a non-mirror tier, and
   a warm page cache hides it (`bench-56gb/RESULTS.md`'s asterisk is exactly this).
4. **The ballast must stay up for the whole matrix** (one `./ballast 31 &`), or `MemAvailable` differs between
   arms and the sizing decision (`MOE_HOST_LIMIT_FRAC`) moves under you.
5. **`-sm layer` and multi-device arms were not runnable on `radix`** (one R9700 + the gfx1036 iGPU).  **That
does not generalise: `soar` has 3 × R9700 and a PCIe 5.0 x16 link each** — multi-device arms and the paired
all-reduce run there, and the canonical heat prior (`6195152358e1…`) is present on that box.  Say which box.
6. **A first-run penalty is real on this rig** (`bench-56gb`'s cold-start rows): interleave arms, never judge a
   pair from one block-ordered pass.
7. **35B-A3B Q4_K_M cannot reproduce the arena transition eviction**, so use **Q8_0** of the same model for it:
   Q4_K_M's whole host expert set is 18 662 MiB (`-ncmoe 40` already hosts every MoE layer) and
   `alloc_table_locked` clamps slots at `n_experts`, so no `MOE_EXPERT_CACHE_MIB` forces the arena past it — and
   its priming pass already sees the full `-ub` prefill, so the work region is reserved at the width the real
   prefills need and never outgrows it (**0 evictions in every arm**).  **Q8_0 35B-A3B does evict**
   (`-ncmoe 40 -lme 12 -c 16384 -ub 2048`: 2 events, 380 MiB, `re-armed 3 stood-down` twice) at 3.6 k t/s, which
   is the cheap reproduction; Flash-Next is only needed for the field ubatch.  Both commands are in
   `RESULTS-2026-11-06-soar.md` §3.
