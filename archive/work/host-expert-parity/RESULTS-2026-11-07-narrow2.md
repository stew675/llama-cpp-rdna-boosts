# RESULTS 2026-11-07 (`radix` + `soar`) — the narrow-2 pin's eviction branch: `TODO` #279 vs PR #267's 0003a

**Status: measured; `TODO` #279 closes as DO NOT IMPLEMENT.  Nothing ships** — no block amendment, no
`optional/` artifact, no tag, `release.json` untouched (the delivery is still `v16-6050b467c-r231`).  The
reference implementation was written, built and A/B'd (`narrow2-prefer-malloc.diff` beside this file) and is
kept only because PR #267's reply owes the question an answer; **nothing applies it**.

**The question.**  PR #267's 0003a: when the draft's compute region does not fit the slab's unmapped VA spare,
`ggml_cuda_slab_narrow2_pin_locked()` *evicts* the arena tables in `[off, mapped)` to place it.  The PR keeps
the draft on its `cudaMalloc` fallback "whenever the region fits outside the slab" instead, on the theory that
an evicted arena is a resident set the cache has to re-fill.  `PLAN.md` P1.5 named the test: *"the A/B against
`GGML_CUDA_SLAB_MIN_ARENA_MIB` shows the eviction costs a resident set."*

**The answer in one paragraph.**  It does not, and it cannot be made to: (1) the branch is **unreachable on
every shipping shape** — the pin's VA spare *is* the draft context's own reserve (kv + compute + slack) minus
the headroom, because the MoE preflight deliberately skips `ggml_cuda_slab_extend_all()` at the priming
sizing, so the spare exceeds the compute region by construction (four probes below, all `unmapped VA spare`);
(2) forcing it (`GGML_CUDA_SLAB_RESERVE_MIB` squeeze, six interleaved arms) the eviction is bounded by the
region and is **fully re-absorbed** by the next sizing/re-arm pass — the final arena MiB and the decode hit
rate come out *identical to four decimals* in both arms, and the whole cost is one re-fill of the evicted
bytes (~314-450 MiB at the box's ~22 GB/s ⇒ ~20 ms, once); (3) the preference cannot avoid it anyway: the
free VRAM it needs is by definition the cushion the eviction does *not* consume, so it can only fire when the
eviction is small, **and** the pin is retried — measured, the fallback declines at `0.19.448` s and the pin
with the *same* 314 MiB eviction lands 13 ms later, because the fallback itself ate the free VRAM its own
guard needs.  A version that *latched* the decision would trade the evicted bytes of cushion permanently for
that one 20 ms re-fill, which is the wrong trade.  `TODO` #278's verdict (measured, not worth a code change)
is the same shape and this one is weaker still.

---

## 1. Environment of record

* `radix` (this box): Ryzen 7 9800X3D, **one R9700** (gfx1201, 34.2 GB), 93 GiB RAM — the campaign's
  `wip/bench-56gb/` rig host (no ballast scope needed: the arms below are VRAM-shaped, not RAM-shaped).
* `soar`: 3 × R9700 (gfx1201), 91 GiB RAM, same tree (`a19aab55e`, `build-rocm`), used only for the
  multi-device reachability probe.
* `~/llama.cpp` @ **`a19aab55e`** = `release.json` `v16-6050b467c-r231`, tree `5aeb007deac`, `build-rocm`
  build `11903`; ROCm 10.1.0-gfx120X; `HIP_VISIBLE_DEVICES` pinned on every arm (`AGENTS.md` rule 3).
* Heat prior: the model-dir sidecar (`MOE_EXPERT_HEATMAP=load` on every arm — an arm must never learn its own
  prompt, `INVARIANTS.md` r224).  No absolute number below is comparable across boxes; only within-run
  deltas travel.
* Every arm: one model load, `scripts/check-gpu-faults.sh` clean over the session.

## 2. Reachability — four probes, all `unmapped VA spare`

The pin is attempted when the draft context first reserves, and again from `ggml_cuda_slab_narrow2_floor()`
at the end of that reserve.  At that moment:

| probe | box | shape | draft reserve estimate | slab | VA spare | compute region | branch |
|---|---|---|---|---|---|---|---|
| P1 | radix | Flash-Next GSQ-RCO-IQ3_S + `mtp-…-Q4_K_M`, 1 GPU, `-c 204800 -ub 6144 -ncmoe 48 -lme 47` | 2036 MiB (kv 880 + compute 1024 + slack 192) | 16.38 GiB | **2.06 GiB** | 0.75 GiB | unmapped VA spare |
| P2 | radix | 35B-A3B UD-Q4_K_M + DFlash2-Q8_0, 1 GPU, `-c 204800 -ub 6144 -ncmoe 40` | 6496 MiB (kv 5280 + compute 1024 + slack 192) | 17.50 GiB | **6.31 GiB** | 0.50 GiB | unmapped VA spare |
| P3 | soar | P2 with `-sm tensor` on 3 × R9700 | same (per device) | 21.31 / 21.12 / 21.12 GiB | **6.38 / 6.31 / 6.31 GiB** | 0.50 GiB ×3 | unmapped VA spare |
| P4 | radix | P2 at `-c 65536 -ub 12288` and `-ub 16384` | 2906 MiB | 23.62 GiB | **2.88 GiB** | 1.25 / 1.50 GiB | unmapped VA spare |

The mechanism, not luck: the spare at creation is `reserve − headroom`, the reserve is
`headroom + draft-context estimate` (`ggml_cuda_slab_reserve_bytes`), and the MoE preflight **skips**
`ggml_cuda_slab_extend_all()` at the priming sizing (`moe-expert-cache.cu`: the `wip/first-prefill-latency`
guard), so nothing has eaten the spare by the time the pin runs.  The estimate's own `compute` term is a
*floor* over the same knob the pin would need to beat, so the straddle needs the estimate to be **short** —
which is r106's deliberate fallback (the `-sm tensor` drafter that used to abort in `graph_reserve`), not a
configuration anyone ships.

P4 also shows the one lever a user has over it: the draft's compute buffer grows with `-ub` (0.50 GiB at
6144 → 1.25 at 12288 → 1.50 at 16384 per device) while the estimate's `compute` term does not — so an extreme
ubatch narrows the gap, and a 46k+ ubatch at a 64k context is the smallest *natural* shape that closes it
(not measured here: the wide work view then takes the room the arena floor needs and the pin refuses instead).

## 3. The forced arm — how the branch was reached, and what it costs

`GGML_CUDA_SLAB_RESERVE_MIB` is the documented operator override, and it sets the reserve **verbatim** (no
`+ headroom`), so `spare = RESERVE_MIB − headroom`.  Field config with the draft's KV small enough that the
squeezed reserve still holds it: Flash-Next + MTP, `-c 16384`, `GGML_CUDA_SLAB_HEADROOM_MIB=1536`,
`MOE_HOST_POOL_RAM_PCT=0`, `-ncmoe 48 --limit-host-experts 47`, greedy `-n 128` over
`prompts/prose-rdna-boosts.txt` (5 298 tokens), `-lv 3`, arms **interleaved** (never block-ordered — this rig
drifts, `archive/work/host-expert-mirror/RESOLUTION.md` rule 4).

Harness: [`narrow2-ab.sh`](narrow2-ab.sh) / [`narrow2-ab.py`](narrow2-ab.py)
(`./narrow2-ab.sh 3`; `OUT=… UB=… RESERVE_MIB=…` to re-shape).

### 3.1 `-c 16384 -ub 6144 -b 6144`, `RESERVE_MIB=1792` (spare 256 MiB < the 768 MiB region)

| arm | pin branch | pin eviction | arena at exit | decode hit rate | prefill t/s | decode t/s |
|---|---|---|---|---|---|---|
| `pref=0` r1/r2/r3 (pre-#279 rule) | evict | 314 MiB | 20 712.6 MiB | 0.9471 | 1 243 / 1 312 / 1 211 | 55.65 / 54.68 / 54.54 |
| `pref=1` r1/r2/r3 (0003a rule) | **declined → pinned 13 ms later** | 314 MiB | 20 712.6 MiB | 0.9471 | 1 382 / 1 240 / 1 233 | 54.62 / 54.34 / 54.61 |
| mean | | | **identical** | **identical** | 1 255 → 1 285 (+2.4 %) | 54.96 → 54.52 (−0.8 %) |

### 3.2 `-c 16384 -ub 1024 -b 1024`, `RESERVE_MIB=1600` (a smaller wide view, so the pin's eviction is the only one)

| arm | pin branch | pin eviction | arena at exit | decode hit rate | prefill t/s | decode t/s |
|---|---|---|---|---|---|---|
| `pref=0` | evict | 450 MiB | 21 418.6 MiB | 0.9239 | 794 / 893 / 874 | 59.04 / 59.33 / 59.03 |
| `pref=1` | declined → pinned 13 ms later | 450 MiB | 21 418.6 MiB | 0.9239 | 865 / 937 / 929 | 59.86 / 64.29 / 67.13 |

Read 3.2's decode column carefully before quoting it: the two arms end in the **same state** (same pin — see
§4 — same arena MiB, same hit rate to 4 dp, same draft acceptance `0.52027`, same 47 reused graphs), so the
+7.8 % is *not* the mechanism; it is this rig's short-generation variance (the values rise monotonically with
run index inside one arm while the other arm's are flat, which is the drift signature, not a lever).  The
honest reading of both tables: **no attributable difference**, and every mechanical difference the arms could
show (arena bytes, residency, re-fills) is nil.

## 4. Why the preference cannot help — the two numbers that decide it

**(a) The window is the cushion.**  The fallback lands the region in free VRAM *outside* the slab, so the
eviction it avoids is exactly the region minus the spare — and the free VRAM it needs to be allowed to fire is
the same pool.  Whatever guard the rule uses (the reference patch uses "half the headroom must survive"), it
can therefore only fire while the eviction is ≤ free − cushion, i.e. **a fraction of the slab's headroom**
(measured: 1.42 GiB free, 0.75 GiB cushion, 0.06 GiB spare ⇒ at most ~0.6 GiB).  A *large* eviction cannot be
avoided by definition: there is no free VRAM to fall back on, which is why the region did not fit in the first
place.  So the rule can only ever save the case that §3 shows costs nothing.

**(b) The pin is retried, so even that window is a churn.**  `narrow2_floor` latches only when a pin
*commits*, so a declined attempt is re-made on the next compute-buffer allocation — and the fallback itself
has meanwhile taken the free VRAM the guard looked at.  Measured in all six `pref=1` arms:

```
0.19.448 W ggml_cuda_slab_narrow2_pin_locked: … would evict 0.44 GiB of arena tables; 1.42 GiB is free outside the
           slab, so the draft compute buffer stays on the cudaMalloc path (a 0.75 GiB cushion is kept; …)
0.19.461 W moe_cache_evict_slab_range: evicted 314.0 MiB of arena tables from the taken slab chunks
0.19.465 W ggml_cuda_slab_narrow2_commit_locked: … narrow-2 (draft compute) pinned at [23.06, 23.56] GiB (0.50 GiB,
           taken from the arena (tables evicted) plus the VA above the mapped extent mapped)
```

13 ms later, same eviction, plus a 512 MiB `cudaMalloc` that is then released.  (In `-ub 1024` the same
sequence with 450 MiB; the numbers differ, the pattern does not.)

**(c) Latching fixes (b) and loses the trade.**  A `narrow2_declined` latch would keep the draft outside the
slab for the run: it gives up ~the eviction's bytes of cushion **permanently** to save **one** re-fill of
those bytes — 314-450 MiB at the box's ~22 GB/s is ~15-20 ms, on a run whose first prefill is 4-7 s.  The
cushion is the thing r206 raised to 1280 MiB/GPU because 768 MiB OOMed a Flash-Next prefill; spending it on
this is exactly the mistake that finding exists to prevent.  Rejected on the trade, not on the measurement.

## 5. Verdict, and what is (not) delivered

* **`TODO` #279 closes as measured / do not implement.**  The pre-#279 rule stays: prefer the unmapped spare,
  refuse below the arena floor, and evict arena tables when the region straddles `mapped`.  It is a
  *shortfall* fallback (r106's `graph_reserve` abort guard) and its cost is one bounded re-fill that the next
  sizing pass absorbs.
* **Nothing ships in `patches/`**, so there is no r232 and no tag; `release.json` stays at `r231`, the fork
  branch stays `a19aab55e`.  The reference implementation is
  [`narrow2-prefer-malloc.diff`](narrow2-prefer-malloc.diff) — a diff against the r231 tree, **not** an
  `optional/` artifact: `optional/` is for a finished feature with no validated win *on hardware this repo
  does not work on*, and this one is inert by arithmetic on any hardware, so advertising it as an artifact a
  consumer may apply would be selling a known non-fix.
* **For PR #267's reply** (`PR-267-REMAINDER.md` §4 item 4): 0003a is **answered** — the mechanism is
  unreachable on every shape we ship, the eviction it targets is re-absorbed (identical arena MiB, hit rate
  and decode), and a preference cannot avoid it without a latch that costs more than it saves.  That closes
  the last owed number; the PR can be closed as *integrated rather than merged*.
* **Re-entry conditions**, so a future session does not re-run this: a drafter whose measured compute region
  exceeds its own reserve estimate **and** a shape where `slab_extend` has already eaten the spare **and**
  free VRAM ≥ region + half the headroom — or evidence from a reporter's box that the eviction is large
  *and* sticky (a different arena that does not re-arm).  The reference patch is the starting point for the
  first; the second would need the latch, and the record above is the reason not to.
