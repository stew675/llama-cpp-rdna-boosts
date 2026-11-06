# wip/host-expert-parity — the tier measured on a fast host link, and PR #267's remainder

**Status: OPEN (2026-11-06) — scoping + measurement plan, and a first measurement pass.**  Nothing implemented,
nothing applied to the fork, nothing in `patches/`, `release.json` untouched.  **The first pass is in
[`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md)**: the demote reproduces at `-lme` 36 (+8 %), is worth **+21 %**
at 94 % coverage, and **costs 53 % of the field prefill at `auto` (97.7 %)** — a shipped-default defect that was
**fixed in r228** (`TODO` #281 closed: the demote is a decode-class notion now — headroom capped by the
coverage gap, prefill fills exempt, a near-mirror pool disengages it; see §4 of the results file).  Read [`PLAN.md`](PLAN.md) (the work and its accept criteria) and
[`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) (what is actually left of the PR).  The evidence base is
[`archive/work/host-expert-split/`](../../archive/work/host-expert-split/RESOLUTION.md) (the campaign that
shipped in r224) and the rig is [`wip/bench-56gb/`](../bench-56gb/README.md).

## 1. The two questions

**A. Are we at parity with a mirror on *this* host, and if not, where is the crossover?**  r224 shipped a
demote (`MOE_HOST_POOL_SINK=1`), a DECODE-ranked layout and coalesced prefill reads, and measured them on a box
whose host link the record's own calibration puts at **~26 GB/s** (`archive/work/host-expert-split/W1-SINK-TAX.md`:
9.6 µs per 679 762 B, "naive transport ~26 µs/copy (~26 GB/s)").  This host is `radix`: Ryzen 7 9800X3D,
**93 GiB RAM, one R9700 (gfx1201) on a PCIe 5.0 x16 link** (`current_link_speed 32.0 GT/s`,
`current_link_width 16`, max the same) — i.e. it is much closer to `briansp2020`'s rig (one R9700, 56 GB
container) than the box those exchange rates came from, and per-byte copies should be roughly twice as cheap.
The demote's economics, the break-even reuse rate and the mirror-vs-partial gap therefore have to be
**re-measured, not inherited**.

**B. What is left of PR #267** that we have neither integrated nor implemented in a different manner.  Ten
patches were reviewed in `archive/work/host-expert-split/DESIGN.md`; three of the mechanisms were adopted in a
different shape (the demote, the bounce ring, the one-budget placement), one was measured and rejected, and the
rest are still open or explicitly deferred.  [`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) is the per-patch
re-assessment with what each one still owes.

## 2. Why this is a new campaign (coverage check, 2026-11-06)

An existing `wip/` campaign was checked for each half before opening this one:

| campaign | what it owns | why it does not cover this |
|---|---|---|
| [`host-expert-coverage/`](../host-expert-coverage/README.md) | the tier's **policy and visibility**: cgroup-aware `MemAvailable`, an authoritative `-lme`, one MIRROR/PARTIAL/NONE verdict | policy only — it explicitly leaves the transport question here.  This campaign *adds* one measured instance to it (the fill-mode decision is INFO) |
| [`bench-56gb/`](../bench-56gb/README.md) | the **rig**: a 56 GiB cgroup plus mlocked ballast outside it, and its measured matrix | the instrument, not the question.  This campaign uses it and fills the matrix rows it is missing |
| [`moe-static-hot-tier/`](../moe-static-hot-tier/README.md) | a static **VRAM** tier (heat vs layer) | a different tier (VRAM, not host) |
| [`kv-residency-cleanup/`](../kv-residency-cleanup/README.md) | the KV window / attendance pool | a different pool |
| [`vram-resident-slots/`](../vram-resident-slots/README.md), [`fp8-support/`](../fp8-support/README.md), [`moe-cpu-overlap/`](../moe-cpu-overlap/README.md) | parked / retained records | unrelated |

Closed and archived, do not reopen as campaigns: `archive/work/host-expert-split/` (the demote's own campaign
— its scope closed with r224), `archive/work/host-expert-mirror/` (the transport question, capped),
`archive/work/host-expert-eager-push/`, `archive/work/host-pool-layout/`, `archive/work/host-pool-speed/`.
Two artefacts of those are inherited rather than re-derived:

* `archive/work/host-pool-speed/HEAD-TO-HEAD.md` — the protocol for pitting pool mode against the
  `--host-experts mmap` family, **prepared 2026-11-02 and never run**.  Its methodology (interleaved arms, same
  binary, ratios to a same-run `pinned` reference, never parallel) is adopted here; its 122B-A10B arm matrix is
  not (this box is one GPU and the field model is Flash-Next).
* `archive/work/host-pool-speed/README.md` §4.3/§4.4/§6 — the standing per-mechanism verdicts on PR #267's
  precopy, the bounce-ring cap and sparse registration.  They are correct and are carried forward.

`wip/host-pool-speed/` still contains a `data/` directory (the per-arm logs the archived campaign's
`HEAD-TO-HEAD.md` §8 tells you to keep there).  It is a residue, not a stub — do not create a `README.md` in it.

## 3. What is different about this box (measured 2026-11-06)

| | r224's measurement box | `radix` (here) | consequence |
|---|---|---|---|
| host link | ~26 GB/s (the record's own naive-transport figure) | **PCIe 5.0 x16** (32 GT/s × 16) | the demote's D2H and the fill's H2D get cheaper per byte; the *disk* read does not.  Re-measure the constant before quoting any policy conclusion that used it |
| RAM | the tier's field numbers came from the 56 GiB rig (a 59 GiB box) | 93 GiB, so **86 GiB available with no cgroup** | without a memory cap `-lme auto` plans 80 % of `MemAvailable` → the whole 46.84 GiB set → **MIRROR**.  Every parity question in this campaign therefore exists only inside the `bench-56gb` rig |
| GPUs | one R9700 in the field arms | one R9700 + the **gfx1036 iGPU as HIP device 1** | `HIP_VISIBLE_DEVICES=0` on every arm (the AGENTS trap).  No `-sm tensor`/`-sm layer` arm is runnable here, so the demote under `-sm layer` stays unmeasured |
| the frozen heat prior | `/llm/models/.../*.moe-heatmap`, 601 376 B, sha256 `6195152358e1…`, mode 444 | **not present**: `~/.cache/llama-moe-heatmap/` is empty and the model-dir sidecar is mode 644, sha256 `a07822d3…` | a number that depends on the prior is only comparable **within a run set**.  Restore or re-freeze the map and quote its sha256 with every arm (`INVARIANTS.md`, r224) |
| gate bases | a base recorded from the r223 delivery tree (`build=37148d118`) | **no base here matches the r224 tip**; the 27B and 35B-A3B bases are r191-era (`50d9e90ee`, `561696677`) | re-record the base from `88e7d88a7` (the tree in `~/llama.cpp` today) before using `gate-prefill-logits.sh` as an A/B |

## 4. What r224 changed that this campaign must cover

Both were part of the r224 fold and both are load-bearing for a box like this one:

1. **The pool's fill mode is now a policy.**  `pool_decide_dio_locked` (block 12) mirrors `--load-mode`: a
   mapped load fills through the page cache (`memcpy` out of the mapping), `--load-mode dio` gets O_DIRECT; the
   MemAvailable heuristic survives only for a consumer that reports no load mode.  The code's own comment says
   the **middle band — the set fits in `MemAvailable`, the set + the planned pool does not — is unmeasured**.
   The 56 GiB rig (46.84 GiB set against 57.3 GiB available) *is* that band, so this is not a corner case; it is
   the shipped shape of the box the campaign exists for.
2. **The coalesced prefill read is O_DIRECT-only.**  `stage_span_fill()` returns immediately unless
   `t.src_fd >= 0`, which is only opened in O_DIRECT mode.  The field `runme` is `--load-mode mmap`, so the
   +81 % / +57 % / +14 % prefill numbers are DIO-arm numbers and may be **inert on the shipped config**.  That
   is a measurement, not an argument: `PLAN.md` P0.3 isolates the span with a `SPAN=0/1` pair at fixed
   `DIO=1`, and P0.2 answers the separate fill-mode question with the `DIO=auto` vs `DIO=1` pair.

(The `-sm tensor` demote stays a decline by design — a strided 2-D copy is r113's 2×-decode mechanism — and the
mirror `slots >= n_experts` case stays inert, both as shipped.)

## 5. Non-goals

* **No `--host-experts mmap` mode and no sparse registration** (`mlock` + `hipHostRegister(ReadOnly)` + a
  `MAP_FIXED` remap of the non-resident expert ranges).  Reviewed and refused as a first move in
  `archive/work/host-expert-split/DESIGN.md` §4 and as a non-goal in `archive/work/host-pool-speed/README.md`
  §6.  Re-entry condition, stated once: **only if a partial tier cannot be made cheap by the tier itself** —
  which is exactly what P0.2/P0.3 test.
* No new mode, and no re-litigation of r113 (a *covering* pool, 2-D strided copies, 0 MiB of L3 available to
  win).  `INVARIANTS.md`'s r113 entry was corrected in place at r224 for the same reason.
* No default flip on the strength of a replay model.  The replay (`scripts/moe-trace-replay.py`) ranks
  candidates; only this box's device arms decide.
* Nothing here is a release gate.  If an item is promoted it leaves through `patches/` with the full ladder in
  `MANIFESTS.md`.

## 6. What "done" looks like

1. **A crossover table for the shipped tier on this box** — pinned GiB vs decode t/s vs first-pass 50k prefill
   vs storage MiB/token, at least at `-lme` {mirror 47, `auto`, 44, 36}, shipping vs `SINK=0`, with the mirror
   as the asymptote.  This is the table `W3` (`archive/work/host-expert-split/PLAN.md`) owed and never got.
2. **One sentence answering "are we at parity"**: the smallest budget within ~3 % of a mirror on both axes —
   or "not parity, gap = X % at Y GiB of pinned RAM saved".
3. **A verdict per PR-267 item**: adopted / deferred with a re-entry condition / rejected with the measurement.
   Then the PR can be closed as *integrated rather than merged*, with the numbers in the reply.
4. No delivery change from this campaign until the maintainer picks an item out of `PLAN.md` P1.

## 7. Where the records live

| what | where |
|---|---|
| the standing plan and the arm matrix | [`PLAN.md`](PLAN.md) |
| **the first measurement pass (2026-11-06)** | [`RESULTS-2026-11-06.md`](RESULTS-2026-11-06.md) |
| the per-patch re-assessment of PR #267 | [`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) |
| the campaign that shipped in r224 (do not re-derive) | [`archive/work/host-expert-split/`](../../archive/work/host-expert-split/RESOLUTION.md) |
| the rig this runs on | [`wip/bench-56gb/`](../bench-56gb/README.md) (`run.sh`, `measure.sh`, `bench.py`, `ballast.c`) |
| the arm harnesses r224 used | [`archive/work/host-expert-split/tools/`](../../archive/work/host-expert-split/tools/README.md) |
| flight recorder + fault scan | `scripts/gpu-watch.sh`, `scripts/check-gpu-faults.sh` |
| the offline policy model | `scripts/moe-trace-replay.py` |
| the policy/visibility half | [`wip/host-expert-coverage/`](../host-expert-coverage/README.md) |
| the inherited head-to-head protocol | [`archive/work/host-pool-speed/HEAD-TO-HEAD.md`](../../archive/work/host-pool-speed/HEAD-TO-HEAD.md) |
