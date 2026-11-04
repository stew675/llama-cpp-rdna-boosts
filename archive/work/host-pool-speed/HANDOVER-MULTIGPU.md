# HANDOVER — `wip/host-pool-speed` on the MULTI-GPU box (soar)

**Written 2026-11-02 on the 1-GPU host, for a cold session on the multi-GPU box.**  Read this, then
[`README.md`](README.md) §7 (the residency rule), §9 (arms/protocol) and **§14-§17** (every number measured on the
1-GPU host).  This file is the *state and the order of work*; the charter and the measured record are `README.md`.

Nothing here is in `patches/`; nothing is applied to the fork (`AGENTS.md`, WIP rules).  The 1-GPU host's work is
finished as far as it can go: what remains needs two or three devices.

> **Update 2026-11-03 (session 4, on `soar`): W1 and W3 are DONE; see [`README.md`](README.md) §18 for the
> measured record and §0 below for the revised order.**  Headlines: F7 was an **environment mistake**
> (`MOE_EXPERT_CACHE_MIB=0` in that arm), so `pinned` on a split has a full 49.8 GiB arena and a cached
> baseline; the gather **fails** W1's assertion on the pair (`pinned` −47 % prefill) and is **mandatory** for
> `pool` at three devices (prefill 16.85 → 1126.8 t/s); W3 shipped as `patches/0005` (+59…73 % decode at
> `RAM_PCT=70`); and the pair's decode deficit is **86 % one synchronous demand read per token** (29 % of the
> step), which re-opens the overlap candidate for the multi-device case.  W2 (the tensor-split-aware remap) is
> unchanged and still last.

---

## 0. Status in one paragraph, and what to do first

**2026-11-03 (current):** W1 and W3 are closed (§3 below); the pool is at **-1.8 %** decode against a *cached*
`pinned` on the 3-GPU `-sm tensor` shape (51.03 at `FRAC 95` vs 51.97) and **-34 %** on the pair, where the gap is
86 % synchronous demand reads.  The next two items, in order: **(1)** the pool's staging path under a ≥3-device
split (`TODO` #131 — prefill 16.85 vs `pinned` 564.65; the gather works around it), then **(2)** the demand-fill
overlap for the multi-device case (ceiling ~25-30 % there; re-screen first, `README.md` §9.6).  W2 (the remap,
+3.3 %) follows and still needs the `§7` decisions.

**2026-11-02 (the original brief, kept for the reasoning):**

The pool arm's **prefill** problem is solved (the pool-aware device gather, `patches/0003`: srv-cell prefill
641-647 → 833-878 t/s, ahead of `pinned`'s 739) and the campaign's P1 items are **closed on evidence** (§3.6:
coalescing refuted, overlap a ~6 % ceiling).  **Decode is the remaining gap** and it is small on paper: the device
remap is worth **+3.3 %** on the 122B arm, and the maintainer has decided it survives `#266` **provided it is
tensor-split aware** (`TODO` #269).  So the first thing to do here is not code — it is **to make the multi-device
arms mean something**: F7 (§2) says on the 2-GPU `-sm tensor` shape `pinned` gets **no expert cache at all** and
a full pool can stall its prepopulate, so there is currently no baseline to beat.  Fix that, then take the remap.

**Order:** W1 (§4) → W2 → W3 → W4.  Do not start W2 before W1 passes; its whole acceptance condition is the split.

---

## 1. What "done" means (the acceptance criterion — do not re-scope it)

Pool mode should reach the point where **PR #267's `--host-experts mmap` + sparse registration offers no
performance advantage**.  PR #267's own envelope (1 × R9700, Flash-Next GSQ IQ3_S, MTP, q8_0 KV, 256K ctx):
`pinned` 74.7 vs `mmap` 72.4 greedy t/s, prefill 1189 vs 1278.  So the bar is **decode at `pinned` parity** on the
campaign's arms; prefill is already there (§3.1).

Two things are explicitly **not** in scope: a direct GPU read of pageable memory (the closed window mode — a fatal
`Page not present` fault on this no-XNACK hardware, `INVARIANTS.md`) and tail-expert skipping (`GREEDY-PURITY.md`
§83).

---

## 2. State of the tree, and the one blocker you inherit

* **Campaign set = five patches** in [`patches/`](patches/README.md): `0001` diagnostics (teardown report + forced
  `DIO` logging), `0002` full-coverage default + `cov90` per-table sizing, `0003` **the pool-aware device gather**,
  `0004` bounce-ring accounting + single owner, `0005` (**new 2026-11-03, W3**) the `RAM_PCT` ceiling scales every
  table proportionally and the concentration fit spends the ceiling.  Applied on the delivery tip `efaa60db0` the
  first four give tree `ac580a8e3`; **all five give tree `5fb99c1c9e45a485fc8780898fecfcb77b0544af`**, verified
  2026-11-03 by `git am` of `000[12345]-*.patch` on a fresh `efaa60db0` worktree (the *tree* is the contract: an
  ad-hoc `git am` stamps its own commit ids; only `apply-all.sh` is deterministic).  Per-patch build clean; greedy text `sha=4ac3a523160f` on 1 GPU (1-GPU host) and `sha=5531ae33f0f7` on both `soar`
  device counts (§18.5).
* Apply to whatever checkout you use (`~/llama.cpp` on soar is fine — do **not** push it; the delivery branch is
  refreshed only by `scripts/fork-refresh.sh` at release time):
  ```bash
  git checkout <release.json .base> && bash <delivery>/scripts/apply-all.sh .
  git am <delivery>/wip/host-pool-speed/patches/000[1234]-*.patch
  ~/bin/build-llama-rocm                     # or the soar equivalent; then rebuild llama-cli/server/bench
  ```
* **`HANDOVER.md` is the 1-GPU host's session-2 record** and its §3 baseline numbers are `soar`'s own (22.33
  `pinned` / 17.76 `pool` on the 122B).  **This host's 1-GPU numbers are ~1.5x higher on `pinned`** (32.8-34.2)
  at the same `h=0.754`, so when you compare against §14-§17 use **ratios, never absolute t/s**.
* ~~**The blocker you inherit (F7, 2026-11-02, preliminary — short decodes):**~~ **F7 IS CLOSED (2026-11-03).**  It
  was an **environment artefact**: that arm ran with `MOE_EXPERT_CACHE_MIB=0`, which disables the cache, and with
  the cache off a 1-token decode leaves the routed `MUL_MAT_ID` on the CPU (98 decode splits instead of 50, no
  slab, no arena, no `alloc_all_locked`) — reproduced exactly (14.40 t/s, prefill 887.2).  With the shipped
  default the pair sizes a **49808.2 MiB per-device arena, 190 slots/table on each device, h=0.9477** and decodes
  at ~40 t/s (§18.1/§18.2).  The instrumentation also confirmed the split accounting is healthy
  (`moe_host_expert_dev_count=2`, 33555 MiB per device, `layer_dev=Meta()` → distributed to ROCm0/ROCm1).  The
  original note is kept below for the record.  **Trap for the future: to A/B "cache off" use `-ncmoe 0`, never
  `MOE_EXPERT_CACHE_MIB=0`, on a multi-device split.**
* **The original note (F7, 2026-11-02, preliminary — short decodes):** on `HIP_VISIBLE_DEVICES=0,2 -sm
  tensor -ncmoe 48` with the 122B, **`pinned` never sizes the expert cache** (no `alloc_all_locked` line, no
  `moe_cache_report`, `Meta() 65248 = 51424 free` — ~51 of 65 GiB of device memory untouched), so its 14.46 t/s was
  the *no-cache* rate; and **`pool` at `FRAC 100` stalled once** in the prepopulate under page-cache pressure
  (rc=137 after 457 s, sleeping in a nanosleep loop), completing with a clean page cache.  Both pair pool runs did
  get an arena (`MoE arena = 0.8323`).  **First thing to check:**
  `llama_model_moe_host_expert_dev_count(model)` under `-sm tensor` (the cache allocates only when it is non-zero),
  then the `MUL_MAT_ID` device list the registration sees.  Until that is understood, no pair number means
  anything.

---

## 3. What is established — do not re-derive (all measured; pointers are to `README.md`)

1. **The gather is the prefill answer and it works through the meta backend** (§14.2-§14.4).  srv cell
   (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, req3): `pinned` 739.2 / `pool` FRAC 80 staging 641-647 / **pool+gather
   833-878**, decode flat; measured identically through the meta backend with one device (`-sm tensor`, 4059
   pool-backed calls).  On the 122B it is **neutral** (its 180 MB tables are below the 224 MiB always-gather gate,
   so only sub-gate ubatches gather) and **−11 % prefill on `pinned`** — hence a *pool-only* default is the
   promotion shape, which needs the capability predicate described in §5/W1.
2. **`MOE_HOST_POOL_COLD` is subsumed** (§14.4, OPEN A): with the gather, `COLD=0` reads 19.57 vs `COLD=1`'s
   19.25-19.86 (without it, 15.54).  This is the piece of `#266` that dies.
3. **The pool's old srv-cell reference 285.3 / 48.8 is WITHDRAWN** (§14.2): it came from the window campaign's
   beta tree; the same arm on this tree reads 641-647.  Pool's prefill was never 2.6x behind.
4. **The device remap is worth +3.3 % here, not +8.1 %** (§15): `pinned` 32.77 vs `pinned` `DEVREMAP=0`
   31.73/31.63.  The copy kind is **nil** a second time (`MOE_HOST_POOL_H2D` 25.90/25.85 vs 26.12/25.73, now at
   full coverage).  ~18 % of the FRAC-100 decode gap is **not** traffic (identical arena and reaches, *more* arena
   hits, *fewer* bytes).
5. **That residual is GPU-side, not host-side** (§15.1/§16.1): `perf` shows the host is idle-ish (64 % of its CPU
   cycles are ROCr `AsyncEventsLoop`); `rocprofv3` shows the pool arm issues **+58 `copyBuffer` calls per decode
   step** (161 vs 103) and an extra 2 MB of `fillBufferUnAligned` — but that fill is **ROCclr-internal, not ggml**
   (instrumentation: 8 memsets logged vs 441 traced fills) and its "4.7x" is a per-stream duration artifact.  Do
   not chase it again without a per-stream busy-union analysis; `tools/rocprof-summarise.py` flags that row shape.
6. **P1-a is closed on evidence** (§16-§17): the synchronous fill is **3-4 ms per decode token** (27.3-34.5 s per
   2000-token run, 1.8 MiB per read, 2.4-2.6 GB/s — already above the `O_DIRECT` block-size ceiling of the mount);
   **coalescing was built, measured and refuted** (FRAC 80: prefill 590 vs 630, decode 16.24 vs 17.65) because the
   pool's residency is **heat-ranked**, so the miss set is a scattered file subset; and an overlap worker has a
   ~6 % ceiling at FRAC 80 and ~0 at FRAC 100 against two measured negatives.
7. **The pinned bounce ring is 2.2-3.1 GiB and is deliberately UNCAPPED** (§16.2): capping it (charter §4.4 /
   PR #267's 0010) is neutral at FRAC 100 but costs **8-25 % of prefill at FRAC 80**, because the memory is the
   prefill miss path legitimately holding one entry per expert in flight.  Do not re-implement the cap.
8. **`MOE_HOST_POOL_PREFETCH` is a dead parse** (set, never read) — `TODO` #268.  `MOE_EXPERT_CACHE_COLD=uva` is
   default-on and **20.5 % of reaches are served cold** in both modes (801 of 944 GiB of expert traffic): the cold
   path, not the fill path, carries the bytes.
9. **OPEN B (RAM_PCT)**: on the PLE-free control, `RAM_PCT=85` does **not** thrash (144/144 tables full, 26.29
   t/s, zero `VmSwap`, 0.5 MiB `pswpout`); `RAM_PCT=70` starves **14 tables to the one-slot floor** and costs
   **−40 % decode** for 10 % fewer slots.  The defect is the clamp's *per-table distribution*, not its level —
   see W3.

---

## 4. The work, in order

### W1 — Make the multi-device arms mean something (F7), then gate the gather on the split

**Why first:** the remap's acceptance condition is "tensor-split aware" (§5), and the promotion gate is a
multi-device arm; neither exists while `pinned` has no cache and a full pool can stall.

1. Diagnose F7: `llama_model_moe_host_expert_dev_count(model)` under `-sm tensor`, the `MUL_MAT_ID` device list,
   and why `alloc_all_locked` never runs for `pinned`.  Record it in `HANDOVER.md` §4/F7 (that file is the 1-GPU
   record; append a dated note rather than editing its numbers).
2. Use **`MOE_HOST_POOL_FRAC=80` on any multi-device arm** (a full pool over-commits: 67.1 GiB pool + a 73 GiB
   model on a 91 GiB box stalled once; even the 1-GPU srv cell shows FRAC 100 *worse* than FRAC 80, 624.5 vs 646.8
   prefill and 82.9 GB RSS).
3. **The gather on a real split** — the arm nobody has run.  Assertion: **gather ≥ staging per split mode**, and
   `scripts/check-gpu-faults.sh` clean, at `-p 2048+` (never a 32-token smoke: a short prompt cannot exercise
   eviction).  Two failure modes to watch, both documented: a device that declines makes the meta gather answer
   false **after** the split already skipped staging → the host path (49.6 t/s in the single-backend case), and
   `GGML_SCHED_DEVGATHER` is **default off** with **no capability predicate in this tree** (the `moe_cache_gather_ok`
   the earlier handover described was window-campaign code and does not exist at `efaa60db0`).
   *If you make the gather default-on for pool tables, that predicate is the prerequisite*: it must answer per
   device and truthfully.
4. Arms: 3-GPU `-sm tensor` (the box's own shape), the x8 pair `-sm tensor`, and `-sm layer` as the control.
   `-t` sized so the IRQ cores stay free; one benchmark at a time.

**Pass:** gather ≥ staging per split mode at `-p 2048+`, text identical, faults clean, and a `pinned` baseline
that actually has a cache.

**DONE 2026-11-03 — twice.**  First pass: the assertion failed on the pair and the pool's prefill was broken at 3
devices.  Second pass (same day): **fixed** — a pool-backed table now gathers in the prefill band by default
(`patches/0006`), so the gather is the default *for the mode that needs it* and `pinned` never takes it.  The table
below is the original measurement; the post-fix one is `README.md` §18.2's resolution block:

| arm (3 × R9700, `-sm tensor`, `-ncmoe 48`, 2000-token decode) | prefill t/s | decode t/s |
|---|---|---|
| `pinned` | 471.8 | **67.6** |
| `pool` FRAC 80 (gather default) | **1502.6** | 67.1 |
| `pool` FRAC 95 (gather default) | **1569.2** | 67.1 |
| `pool` FRAC 95, `GGML_SCHED_DEVGATHER=0` (kill switch) | 16.9 | 27.3 |

Text: 257 chars `sha=5531ae33f0f7` on all eight mode/device arms; 5835 chars `sha=dc178c29fe89` across
`pinned`/`pool` F80/F95 at 3 GPUs with the 2000-token decode behind a 5258-token gather-served prefill; faults
clean.  The original pass follows.

**Original result: the assertion FAILS on the pair, and that is the finding** (`README.md` §18.1/§18.2):

| arm | pair (`-sm tensor`) | 3 GPUs (`-sm tensor`) |
|---|---|---|
| `pinned` (cached baseline) | 39.72-39.87 decode / **1604-1648** prefill | 51.97-52.04 / **565-606** |
| `pinned` + gather | 39.81 / **832.5 (-47 %)** | 51.71 / 587.8 (+4 %) |
| `pool` FRAC 80, staging | 26.08 / 1024.1 | 49.99 / **16.85 (!)** |
| `pool` FRAC 80 + gather | 26.11 / 991.5 (-3 %) | **49.48 / 1126.8 (+66x)** |
| `pool` FRAC 80 + cov90 | **28.02** / 698.7 | 48.97 / 24.01 (staging) |
| `pool` FRAC 95, staging | — | **51.03** / 15.03 |
| `pool` FRAC 30, staging | — | 47.25 / 206.3 |

1. F7 closed (above) — the pair's `pinned` baseline has an arena (`49808.2 MiB`, `h=0.9477`).
2. **The gather is not a general prefill win**: it costs `pinned` **-47 %** on the pair (the 180 MB tables are
   under the 224 MiB always-gather gate, exactly the mechanism `README.md` §14.4 describes at -11 % on one GPU).
   Do not make it default-on for pinned tables.
3. **The gather is the ONLY workable pool prefill at three devices**: the pool's staging path collapses there
   (16.85 t/s against `pinned`'s 564.65) and the gather restores 1126.8 t/s (+100 % over `pinned`).  This is a
   **delivery defect in block 12's pool mode under a ≥3-device tensor split** (`TODO` #131), not a campaign
   patch's: it is the pre-existing staging path.  Measured everywhere: `FRAC` 30/80/95, `STAGE_BATCH=0` (31.99),
   `DIO=0`, 2000-token and 64-token runs, so it is not coverage, not the batch arm and not the fill mode; the
   `pinned` arm stages the *same* bytes through the *same* kernel at full speed.
4. Text identity and faults: `257 chars sha=5531ae33f0f7` on all eight arms (pair and 3-GPU, both gather states);
   `check-gpu-faults.sh --since 04:10` clean.
5. **Promotion shape**: keep `GGML_SCHED_DEVGATHER` opt-in, and document that `--host-experts pool` on 3+
   devices needs it on until #131 is fixed.  A pool-only default still needs the truthful per-device capability
   predicate — the decision is still the maintainer's (this file §7, item 2).

### W2 — P0-a: the device remap over pool slots, tensor-split aware (`TODO` #269)

Read `TODO.md` #269 first; it carries the maintainer's decision and the condition.  The mechanism is nearly
there — `pool_cold_slot_locked()` + `moe_cache_get_cold()` already produce `n_res + pool_slot` and the eager path
consumes it (`MOE_HOST_POOL_COLD`) — and what is missing is:

* `devremap_ok_locked()` is gated on `table_cold_ok()` (a pinned master).  A pool table needs its own cold
  *source*: the map's cold entries must encode the **pool slot**, a host-side fact at map-build time, so the map
  carries `slots + pool_slot` rather than `slots + e`;
* **the freshness invariant is the real work**: the pool re-ranks at exactly two call sites
  (`alloc_table_locked`, `pool_rank_from_prefill_locked`), and a remap that lags one call has **no** sync (r203's
  invariant 3).  Either carry a **layout epoch** in the cold id and rebuild the map on an epoch change, or
  **freeze** the layout for the decode band.  The two call sites are what makes a freeze plausible;
* **per device**: a pool slot is process-wide and its `cudaHostAllocMapped` alias (`pool->base_dev`) is valid on
  every device, but each device's **arena slots differ**, so the map is per (table, device) and the cold-id
  encoding is per-device-slice aware.  That is the condition the maintainer set — a single-device-only
  implementation is not acceptable.

**Gates (all of them, on 1 GPU *and* the multi-device `-sm tensor` arm):** same-seed greedy text byte-identical
across `pool` (`COLD` × `DEVREMAP` crossed), MTP four-axis acceptance identical, prefill-logit mean KLD ≤ 0.005 /
same-top-p ≥ 98 %, `check-gpu-faults.sh` clean, and the r203 invariant list **extended**, not weakened.

**Ceiling:** +3.3 % decode on this host's 122B arm (+8.1 % was r203's own cell on 35B-A3B).  Judge it against a
same-session `pinned` control, and expect it to be *worth doing for `#266`'s sake* rather than for the number.

### W3 — the `RAM_PCT` clamp-distribution defect (small, independent, real) — **DONE 2026-11-03** (`patches/0005`)

**Result** (`README.md` §18.3): 1 GPU, `-sm layer`, `FRAC 100`, 2000-token decode — `RAM_PCT=70`
**10.94 -> 17.44 t/s (+59 %)** and `+73 %` with `SIZING=1` (18.92), at identical pinned bytes (54.5 GiB), where the
old clamp had starved **26 of 144 tables to one slot**; `RAM_PCT=85` **19.52 -> 20.97 (+7 %)**, where the shave bit
from layer 18 on.  `pinned` control 22.36.  The fix lives in `patches/0005` (proportional scale + the cov fit
spends the ceiling), so `RAM_PCT` is a usable knob again.  The text below is the original brief, kept for the
reasoning.


`pool_clamp_slots_to_ram()` shaves the **registration-order tail** down to the one-slot floor instead of shaving
every table proportionally: at `RAM_PCT=70` that is 14 tables at 1 slot out of 144 and **−40 % decode** for 10 %
fewer slots (§3.9).  A proportional shave (or a byte-budget pass with a floor only when the budget cannot cover
one slot per table) makes `RAM_PCT` a usable knob again and should lift the sub-full arms.  It needs no GPU
feature, so it can be done on any box: gate it on the 122B at `RAM_PCT` 70/80/85 × `FRAC` 60/80, watch `VmSwap`
and `pswpout`, and require text + MTP acceptance unchanged.  It is also the honest answer to the maintainer's
"85 % looks too high" report: the level is fine, the distribution is not.

### W4 — the campaign's Phase-1 numbers on the multi-GPU box — **the arm table is measured; `HEAD-TO-HEAD.md` is not yet written**

The pair and 3-GPU arm tables are in `README.md` §18.2 (2000-token decodes, alternating, `-lv 4`, pool before
pinned in each group).  What is *missing* for the head-to-head: the 1-GPU control on this box (only the W3 arms
ran there: `pinned` 22.36 / 20.97 pool `FRAC 100` `RAM_PCT 85`), a `pool` arm **at decode parity** with the
gather on and `FRAC 95` (the closest so far is 51.03 vs 51.97 = -1.8 %, which needs the §18.2 prefill fix to be a
legitimate arm), and the MTP arm (`--spec-type draft-mtp`: every arm this session ran `--spec-type none`).  Write
`HEAD-TO-HEAD.md` only after §18.2's prefill defect has a decision — otherwise the 3-GPU prefill column compares
a broken pool arm with #267's numbers.

The protocol to follow: `pinned` / `pool` FRAC 80 / FRAC 100 / `pool`+gather,
1 GPU (a control, so the box is comparable to §14), the x8 pair, and 3 GPUs, at `-ncmoe 48`, ≥ 2000-token decodes,
`-lv 4`, arms alternating, `pool` measured **before** any `pinned` arm in a group (its 40-67 GiB pin evicts the
page cache pool reads from).  Then write `HEAD-TO-HEAD.md` against PR #267's published envelope (§1).

---

## 5. Traps — every one of these has cost a run

**Multi-device specific**
* **Pin `HIP_VISIBLE_DEVICES` on every arm.**  The 9950X3D's iGPU (gfx1036) is enumerated as a HIP device; the
  delivery is built for `gfx1201` only, so an unpinned `-sm tensor` arm aborts at load with `invalid kernel file`
  (`ggml-cuda.cu:130`).  A **1-GPU** arm without the pin happens to work, which is what makes the trap easy to hit
  only on the 3-GPU arm.
* **The pair is `HIP_VISIBLE_DEVICES=0,2`** (both x8), never `0,1` (x4).  Under the pin the in-process indices are
  `0,1`, so `--device 0,1`, `-ts` and `-mg` refer to those — a bare `--device 0,2` silently selects one device.
* **`-sm tensor` + the MoE cache is the F7 regime** (§2): check `dev_count` and the arena line before believing any
  number.
* Pool is process-wide and `cudaHostAllocPortable|Mapped`; the cross-device source is `pool->base_dev` over a
  **D2D** copy (a plain `cudaMallocHost` cross-device H2D falls back to a pageable staging memmove that crashes on a
  2-D copy).

**Host-tooling**
* `llama-cli` needs **`--single-turn`** (plus `--no-display-prompt` for scripted output) and anything that can block
  goes inside `timeout`.
* **`-lv 4`** for anything out of libllama — `LOG_LEVEL_INFO` is mapped to TRACE(4), so `-lv 3` hides every
  `host_experts_mode=POOL` / `table_pool_locked` / `pool_decide_dio_locked` / `moe_cache_report` line.
* `srv-test.sh`'s `SRVENV` splits on **commas**, not spaces (`SRVENV="A=1,B=2"`).
* `pkill -f llama-server` matches the launching shell; use `pkill -f "[l]lama-server"` or kill by pid.  A killed
  `sudo` wrapper leaves the server holding VRAM (the next load fails with `unable to allocate ROCm0 buffer`).
* A **below-floor auto arena silently disables the whole expert cache** (`auto arena 6996 < floor 7776 ->
  disabling`, then `tensors=0`, decode ~24.5 t/s).  Pass `MOE_EXPERT_CACHE_MIB`, shrink `-c`, or fewer layers.
  `MOE_EXPERT_CACHE_MIB` in (0, 2048) hard-aborts — hold it fixed across arms or fix neither.
* **Never two benchmarks at once**; size `-t`; `ROCBLAS_USE_HIPBLASLT=0` on a fresh-process comparison only.
* `perf record` works but `-g` is refused (`perf_event_paranoid`=2 — do not raise it silently).
  `rocprofv3 --kernel-trace -f csv` gives comparable **durations** but a meaningless rate (never quote a t/s from
  a traced run); its CSVs are 40-75 MB — **do not commit them**, use `tools/rocprof-summarise.py`.

**Interpretation**
* The teardown report is **not** a snapshot of the run unless `patches/0001` is applied; with it, read
  `takeover=`, `expert access:`, `expert H2D traffic:`, `pool fills:`, `bounce ring total`.
* Throughput is not a correctness signal, and the prefill-logit KLD gate is **blind to residency defects** (it
  passes on a deliberately broken arm): a residency claim needs a multi-ubatch prefill plus
  `scripts/check-gpu-faults.sh`.
* A gather "win" of ~1.7x its neighbours is a **corruption signature** (Hole B).  The control is
  `GGML_MOE_GATHER_ONCE=1`, which must reproduce garbage; if it does not, the instrument is not measuring what you
  think.

---

## 6. Gates (see `MANIFESTS.md` → "the gate for what you changed", plus the campaign's own)

| change | gate |
|---|---|
| the gather, or any decode-path change | srv-test at ≥ 2000 tokens + `four-axis.sh` (acceptance identical, `none` == `draft-mtp` text) + `llama-batched-bench -npl 1,4,8` |
| any prefill-path change | `scripts/gate-prefill-logits.sh` (mean KLD ≤ 0.005, same-top-p ≥ 98 %).  For a host-expert arm use `--extra "-ncmoe 40 --host-experts pool"` (added 2026-11-02; without it the gate cannot judge a non-default config).  Its base must be recorded on the same ROCm line |
| the remap (W2) | the above **plus** `COLD` × `DEVREMAP` crossed, on 1 GPU and the multi-device arm, and the r203 invariants extended |
| anything, before a tag | `scripts/gate-per-block-build.sh --changed-from <prev tip>` + `scripts/check-gpu-faults.sh` |

Text oracle: 150 chars `sha=4ac3a523160f` (Q4_K_M, `pool`, in-mode via `archive/work/host-expert-pinned-window/oracle.sh`).

---

## 7. Open decisions the maintainer owes (ask before W2 lands)

> **Status 2026-11-03:** the measurements that these decisions were waiting on now exist (`README.md` §18), so
> all four can be answered; none has been decided by this session.  A recommendation follows each.

1. **The `FRAC` default on a multi-device split**: `100` is the single-GPU default (`patches/0002`) but it
   over-commits on the pair (and is worse even on the 1-GPU srv cell).  Keep 80 for multi-device, or make the
   default depend on `dev_count`?  *Measured: 3-GPU `FRAC 100` clamps itself to 63832 MiB (the RAM ceiling now
   scales it proportionally, `patches/0005`) and decodes 45.44 against FRAC 80's 49.99 and FRAC 95's 51.03 —
   i.e. on a big-arena split MORE coverage is better on decode, up to where the pinned pool evicts the model's
   own page cache.  On the pair (W3 build): FRAC 80 26.08 < FRAC 100 29.08 < **FRAC 95 32.02**, and the F100 ->
   F95 step is 4 GiB less pinned for 42 GiB less disk traffic.  Recommendation: **one default (no `dev_count`
   branch), but the RAM ceiling must reserve the model's own pageable footprint** before `100` is a good
   multi-device default (`README.md` §3.2/§18.2 finding D); until then 95 is the better split arm on this box.*
2. **The gather's promotion shape**: pool-only default (needs the capability predicate), or keep
   `GGML_SCHED_DEVGATHER` opt-in forever?  *Measured: mandatory for `pool` at 3 devices (+66x prefill),
   neutral for `pool` at 2, and **-47 %** for `pinned` at 2.  Recommendation: pool-only default, with the
   predicate answering per (table, device) at the point where the gather would serve ("is this table
   pool-backed?"); `pinned` keeps the staging path.  Do not flip the default before that predicate exists.*
3. **`#267`'s reply**: hold it until the head-to-head is written (still the plan), or answer now that the prefill
   half is settled and only decode is open?
4. **`MOE_HOST_POOL_PREFETCH`** (`TODO` #268): remove it and correct `ENVIRONMENT.md`, or re-wire it as a
   *known-sequence* readahead?  §17 says the coalesced form is refuted; the honest default is **remove**.

---

## 8. Files, and what not to touch

* **Measured records:** `README.md` §14 (re-base + srv-cell baseline + the gather), §15 (Step 0: remap/copy-kind/
  residual + the per-kernel trace), §16 (the bounce ring + the synchronous fill), §17 (coalescing, refuted),
  **§18 (session 4 on `soar`: F7 resolved, the W1 multi-device arms, W3, the pair's demand-fill accounting)**.
  Data: `README.md` §18's header names every directory; runners `tools/run-arms-multigpu-261103.sh` and
  `tools/run-text-purity-multigpu-261103.sh` (both take `VIS`/`SM`/`NCMOE` from the environment).
  `HANDOVER.md` is the 1-GPU session-2 record (it also holds `soar`'s own 1-GPU baseline numbers).
  `HANDOVER-2026-11-02.md` is the window campaign's hand-over, with its corrections in §6a.
* **Data:** `data/20261102-session3-{srv,122b,step0,fillcost,ring,coalesce,profile,openb,four-axis,batched}`.
  Runners: `tools/run-arms-1gpu-261102b.sh` (`ARMS=decode|gather|step0|pool|all`), `tools/rocprof-summarise.py`,
  `tools/pool-sizing-sim.py`, `tools/replayer-calib.py`; instruments reused by path from
  `archive/work/host-expert-pinned-window/` (`srv-test.sh`, `oracle.sh`, `four-axis.sh`).
* **Do not** apply `archive/work/host-expert-pinned-window/reference-window-261102.diff`, resurrect
  `--host-experts window`, or register the model mapping for SM reads.  Do not touch `patches/`, `release.json`
  or `rdna-boosts-all.patch` — promotion is a separate, maintainer-authorised step.

---

## 9. Promotion checklist (when the acceptance criterion is met)

1. The acceptance criterion of §1 demonstrated on `soar` (3-GPU and pair) **and** a 1-GPU control, with the same
   binary and the ratios-to-`pinned` table.
2. All gates of §6 green on the **final** tree, including the multi-device remap arm.
3. `ENVIRONMENT.md` rows for every knob the set adds or repoints (`MOE_HOST_POOL_SIZING`, `_COV_PCT`, the changed
   `FRAC` default; drop the dead `_PREFETCH`), `patches/README.md` block 12 note, `WORKLOG.md` entry,
   `release.json` re-cut via `scripts/make-release.sh`, then `scripts/validate-set.sh`.
4. `scripts/fork-refresh.sh --build --gates` for the branch, and the container images off-host (see `CONTAINERS.md`).
