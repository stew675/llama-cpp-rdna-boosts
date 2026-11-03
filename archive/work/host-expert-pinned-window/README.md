# wip/host-expert-pinned-window — removing the `--host-experts pool` bounce buffer (OPEN)

**What this is.**  A feasibility campaign for replacing the delivery's **pool** tier — the pinned copy of the
hot host experts that exists only because the read-only GGUF model mapping is pageable and a no-XNACK GPU
cannot read it (issue #116, `INVARIANTS.md` §"`-ncmoe`/`-cmoe` host experts are pinned") — with a **registered
window over the mapping itself**, so the page cache is the only cache and the bytes the op needs are the bytes
the kernel already holds.  Nothing here is in `patches/` and nothing here is applied to the fork (`AGENTS.md`,
WIP and promotion rules).

**State: PROMISING — the one-residency-tier shape in `PROTOTYPE.md` (v4) measures +35-40 % decode and is
byte-identical, with the pool deleted.**  On 1x R9700, 35B-A3B Q4_K_M at `-ncmoe 40`, arena 4096, pool 2048:
`--host-experts pool` **tg128 52.3-56.9 t/s** vs the same arm with the host master registered once and read in
place **72.6-77.9 t/s** (`MOE_EXPERT_WINDOW=1 MOE_EXPERT_TENSOR_REG=1`), prefill unchanged, text identical on
four arms x three prompts and inside a 10 GiB cgroup.  The win is entirely the **in-place cold read** (switch
it off with `MOE_EXPERT_CACHE_COLD_UVA=0` and the arm lands back on 53.5 t/s); the registration also unlocks
the sync-free decode remap for a `--host-experts pool` master (+9 % of that total).  The two failures §5 used
to carry (`PROTOTYPE.md` §5.3) **do not reproduce** on the saved v3 tree.  Opt-in, not default: the gates and
the 3-GPU `-sm tensor` arm are still owed.  Nothing in `patches/`.

**The pin/demote contract the design needs (§5) is measured and achievable; one caveat on the budget
(RLIMIT_MEMLOCK) and one unresolved internal (DMA vs bounce, §5.4 — it does not change the decision).**  The
registration is a *mapping*, not a lock: it makes the bytes addressable, not resident, so the page cache
stays the only cache (§5.6).

---

## 1. The question, and why the pool costs what it does

`--host-experts pool` keeps a second, pinned copy of the hot experts.  A demand miss is therefore

```
file/page cache --memcpy (SYNCHRONOUS, host thread)--> pinned pool slot --CE H2D--> VRAM --> op
```

The `memcpy` is a plain `memcpy` out of the page cache when the buffered mode is on (`pool_fill_slot_locked`,
`moe-expert-cache.cu:1368`) and it runs **on the token's critical path** (`moe-expert-cache.cu:274`).  It buys
exactly one thing: a device-readable address for bytes the page cache already holds.

The candidate is to register the mapping range in place instead:

```
map range --hipHostRegister--> CE H2D straight from it --> VRAM --> op        (design B)
map range --hipHostRegister--> the kernel reads it over PCIe in place         (design A)
```

with the host gating the window (it already gates graph submission) and bounding how much is registered.

## 2. Method

`probe-pinned-window.cpp` (build/run: `./build.sh`, `./run-probe.sh`) — one GPU, one real GGUF mapping, one
expert blob of 1.66 MiB (35B-A3B Q4_K_M geometry), 40 runs per arm over consecutive offsets, medians.  Arms:
pool (today), B (register + CE H2D), B with a deferred unregister, A (SM reads the registered range), and VRAM
references.  Every arm is byte-checked against the file.  Log: `runs/probe-20261102-074142.log`.

Box: 1x R9700 (gfx1201) at **PCIe 5.0 x16** (`03:00.0`, `current_link_width=x16 current_link_speed=32.0 GT/s`),
ROCm 10.1.0, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

Measurement traps hit on the way (both are now handled in the probe, and both matter for anyone re-running it):

* `/tmp` is **tmpfs** here, so a "cold page" test written there measures RAM.  The cold test writes to `/llm`
  (the xfs volume the models live on) and proves the drop with `mincore` (0/2048 resident).
* `POSIX_FADV_DONTNEED` **skips pages that have PTEs**, so a mapped range always survives it; the reclaim test
  uses `madvise(MADV_PAGEOUT)` plus an unregistered control.
* **`MADV_DONTNEED` is not a reclaim test either**, and is deliberately not used as one: on a `MAP_SHARED` file
  mapping it drops only the process's PTEs (`VmRSS` fell exactly 8 MiB in the probe) while the page cache keeps
  the pages -- measured host re-read of the range afterwards: **95.7 µs** for 4 MiB, i.e. a cache hit (a cold
  read of that half is ~8.7 ms).  It frees no memory and performs no I/O, so it cannot show whether a range is
  protected.  It is also not a good *demote* primitive for the design: it unpins nothing (`mlock` accounting is
  untouched), returns no memory, and it makes the next **device** fill cost ~1.9 ms/4 MiB because the driver has
  to re-establish its mapping over the zapped PTEs.  `MADV_PAGEOUT` is the only user-space call that asks the
  kernel to reclaim the pages, which is what "under pressure" means.
* Host CPU time cannot distinguish a DMA from a bounce: a blocking `hipMemcpy` busy-waits (100 % of wall on
  every arm, VRAM→VRAM included).  Use `rocprofv3 --hip-trace` for that question.

## 3. Results (median, µs unless stated)

**Registration works on a read-only `MAP_SHARED` file mapping, with UVA (host pointer == device pointer), and
costs nothing:** 0.6-0.7 µs to register and 0.2 µs to unregister, at **every** size up to 256 MiB
(`us/MiB` → 0.00), with **no** `Mlocked`/`Unevictable` movement — the pin is *virtual*, deferred to the first
device access.  One-time cost: the first copy out of registered memory in a process pays **3.0 ms**; every
later fresh range is at full rate (`3f`: 8 MiB first copies 158-163 µs = 51-53 GB/s, the very first 3019 µs).
An unaligned range start (+64 B) is **accepted** — no page-alignment constraint.  `hostRegisterReadOnlySupported`
reports `0` yet the `ReadOnly` flag is accepted; use `default(0)`.

| arm (1.66 MiB expert) | host µs | total µs | device µs | GB/s |
|---|---|---|---|---|
| **pool: memcpy + CE H2D** (today) | 50.1 | **137.3** | 66.4 | 26.2 |
| **B: register + CE H2D** | 1.3 | **62.1** | 43.2 | 40.3 |
| B, unregister deferred (realistic) | 0.8 | **48.0** | — | — |
| A: SM read, 32 blocks | 0.0 | **58.9** | 41.2 | 42.3 |
| ref: CE D2D, VRAM (a "hit") | 0 | 10.9 | 10.9 | 160.0 |

**Pool vs B vs A across blob sizes (median total µs)** — the pool's `memcpy` scales, the register does not:

| blob | pool | B | A (32 blk) | B/pool |
|---|---|---|---|---|
| 256 KiB | 45.7 | 34.4 | 31.5 | 0.75x |
| 512 KiB | 58.4 | 39.2 | 36.2 | 0.67x |
| 1.66 MiB | 126.2 | 61.3 | 59.2 | **0.49x** |
| 4 MiB | 227.1 | 105.4 | 103.5 | 0.46x |
| 16 MiB | 781.0 | 327.0 | 324.9 | 0.42x |
| 32 MiB | 1569.6 | 628.2 | 624.4 | **0.40x** |

**Sustained rate from a registered window: 55.7 GB/s** (256 MiB, `8`), i.e. the PCIe 5.0 x16 ceiling — the
window is not the bottleneck.

**Cold (page-cache miss) fill, 8 MiB, real disk, 0/2048 pages resident:**
`memcpy` into the pool **17.0 ms (0.5 GB/s)** vs `hipHostRegister` **2.3 ms (3.6 GB/s)** — the pool's cold fill
is the slowest single thing in the design, and registering in place is **7-13x** faster (it batches the faults
through GUP instead of faulting page-by-page inside a `memcpy`).

**Correctness:** pool, B and A all byte-identical to the file.  Unregistering a range while a copy still reads
it returned `Success` and the bytes were identical — **not a safety guarantee**; the demote must be gated on a
stream event (`cudaEventQuery`, exactly the `dr_lag` handoff DEVREMAP already uses).

## 4. Verdict

* **B is viable and wins decisively on the miss path: 2.2x faster at the real expert size, up to 2.5x at
  32 MiB, and 7-13x on a cold fill (17.0 ms pool vs 2.3 ms register for 8 MiB).**  It deletes the pool's RAM duplication and the synchronous host `memcpy`
  from the token's critical path; registration is 0.6 µs and needs no budget of its own beyond the window.
* **A buys nothing over B on this link** — an SM read of a registered range (42.3 GB/s) is as fast as a CE copy
  of it (40.4 GB/s), and they are equal at every size tested.  So there is no reason to give up VRAM hits: keep
  the arena, use B for the fill.  This also retires the "the zero-copy arm is 35 % slower" worry for the *read*:
  the SM saturates the link here; the in-tree gather's loss must come from compute interference / the head-zero
  guard / multi-ubatch effects, not from raw read bandwidth.
* Where the win lands: a miss costs 137.3 → 62.1 µs.  On a miss-heavy config (`-ncmoe` with the pool sized
  below the host set) at 2 misses/layer over 48 layers that is ~7.9 ms/token → ~3.4 ms/token.  On a config that
  fits, nothing changes (hits are VRAM either way), so this is a *fit-the-box-better* win, not a universal one.

## 5. Pin, demote and re-promote — the contract the design needs (measured)

**5.1 `hipHostRegister` alone does not hold the pages.**  Timed, not inferred from `mincore` (which is
misleading here) and not with `MADV_DONTNEED` (which cannot evict a `MAP_SHARED` mapping -- see §2): an 8 MiB range copies at 191 µs warm, and **15.1 ms** after `madvise(MADV_PAGEOUT)` — a
fault-speed refill, i.e. the registration did not protect it.  Bytes were always **identical**, so the failure
mode is *slow*, not *wrong* (ROCm recovers the range on use; why, we have not established — do not silently
rely on it).

**5.2 `mlock` + `hipHostRegister` is a real hold, and the demote/re-promote cycle is cheap.**  4 MiB warm range,
medians of 5: baseline **92.8 µs**; `mlock` + `MADV_PAGEOUT` → **92.3 µs (held)**; `munlock` +
`hipHostUnregister` (demote) → **92.3 µs**, i.e. the page cache keeps the pages so a re-promote is immediate;
re-`mlock` + re-register → **91.4 µs, full rate**.  A 20x demote/re-promote loop on an 8 MiB range: register
**0.6 µs**, copy **166.7 µs** median, **no degradation** between the first and second half of the loop.  So the
intended policy is directly supported:

```
promote : mlock(range) + hipHostRegister(range)      # held while in the window
demote  : munlock(range) + hipHostUnregister(range)  # pages return to ordinary page-cache pages
re-promote while still cached: 0.6 us register + unchanged copy rate
```

**No `madvise` call belongs in the runtime path.**  Releasing the pin is `munlock`; from there the kernel
reclaims the pages by its own policy when memory is short and keeps them otherwise -- that is the default state
of a file page, so there is nothing to advise.  `MADV_PAGEOUT` appears in this campaign **only as the probe's
pressure source** (the test has to try to take the pages away to know whether `mlock` holds); it is an
instrument, like the `mincore`/contention experiments in §2, not a proposed behaviour.  The design consequence
is an SLO, not a call: a demoted range may be reclaimed at any moment, so a re-promote must be treated as
*cached if we are lucky* -- the fill path has to absorb a fault-speed refill (~13 ms/8 MiB) without stalling the
token more than the pool path would.

**5.3 The hold is a SOUNDNESS REQUIREMENT, not a budget caveat — CORRECTED 2026-11-02.**  `mlock` is capped by
**`RLIMIT_MEMLOCK` = 8 MiB soft/hard on this box**, so a multi-GiB window needs `LimitMEMLOCK=infinity`
(systemd) or `CAP_IPC_LOCK` — and the earlier text here called that a "deployment, not a blocker" caveat and
asserted the un-`mlock`ed window was *"cached but reclaimable: **correct**, just slower on a refill"*.  **That is
wrong.**  There is no host-side refill to be slow: on a no-XNACK GPU a kernel read of a page that is not present
is a fatal fault (issue #116), which on a model larger than RAM aborts the run (`Page not present`,
`WORKLOG.md` 261102.012) and, when the range is only partly lockable, hangs it (261102.013).  The promote
policy of §5.2 — `mlock` + register on promote, `munlock` + unregister on demote — is therefore the design, not
an option; the code now takes the hold or **declines the registration** (`bind_host_dev_locked`) and gates every
in-place kernel read on it.

> **Cross-reference (2026-11-02).**  This requirement was already written down, in the parallel campaign:
> `wip/host-pool-speed/README.md` §7 — *"the intermediary **guarantees** the requisite pages are pinned **and**
> resident before it hands the go-ahead back to the GPU ... a residency check before the read is released, and
> **no** fallback that copies from pageable memory."*  The two campaigns ran at the same time on different hosts
> and were never cross-referenced; that is how this document came to argue the opposite of the maintainer's
> brief.  The authoritative statement is that §7 plus `INVARIANTS.md` (device-visible host memory must be
> resident).

**5.4 Internal mechanism: unresolved, and it does not change the decision.**  `rocprofv3 --memory-copy-trace`
(`trace-copy-path.sh`, `runs/copy-path_memory_copy_trace.csv`) shows one H2D copy per source and **no kernel at
all**, with the registered source at **155.6 µs** and the pageable control at **155.9 µs**, while a
`hipHostMalloc` source is *slower* (241 µs).  A host-DRAM contention test could not separate the arms
(two runs, opposite verdicts — marked VOID in the probe).  So "direct DMA vs internal bounce" is not settled by
the available tooling; either way the *measured* miss path, the byte-exactness, and the memory saving stand.

**5.5 Pool stays for the bulk/prefill policy.**  The pool is not replaced everywhere: when the host expert set is
larger than RAM the pool's O_DIRECT/warm residency and its bulk prefill staging are still the right policy, and a
page-cache window cannot prefetch a set that does not fit.  The window is the **decode/verify fill** tier (and a
prefill *source* where the table is cached); `--host-experts pool` keeps owning the bulk path.

**5.6 The registration is a mapping, not a lock — and it is what the win comes from.**  `hipHostRegister`
makes a mapping range *device-addressable* (UVA alias, 0.6 µs at any size, no `Mlocked`/`Unevictable` cost);
it does not hold the pages.  So a **kernel** read of a registered range is legal (data-identical under 70 GiB
of real pressure, `probe-cold-read`) and is re-faulted by the kernel's own reclaim policy, which is what lets
"the page cache is the only cache" be true: measured in situ, the whole arm runs correctly inside a
`MemoryMax=10G` cgroup on a 22.6 GB mapping, with `Mlocked` 0.00 GB throughout.  The consequence is that
non-resident experts can be read **in place** instead of copied into a pinned buffer, which is worth
**+35-40 % decode** at a sub-residency arena (`PROTOTYPE.md` §5): the miss path stops being a synchronous
host `memcpy` + CE copy on the token's critical path and becomes PCIe reads overlapped with compute.  The
same alias makes `table_cold_ok()` true for a pool-backed master, which unlocks the sync-free decode remap
(`GGML_MOE_DEVREMAP`) that was gated to pinned masters (+9 % of the total on its own).  Both of the delivery's
host tiers measure the *same* ~55 t/s without it — a `hipHostMalloc` master is the slowest source for device
access (§5.4's copy-path probe: 241 µs vs 155 µs for a registered mapping).

## 6. The shape of the change — written, and staged (`--host-experts window`)

It is now a **mode**, not an env variable: `--host-experts window` (the loader's
`host_expert_buft`/`llama_host_experts_mode_apply`, a device entry point
`ggml_backend_dev_moe_cache_set_host_window`, and `g_win_on` owned by the mode).  The env surface is
only the A/B: `MOE_EXPERT_TENSOR_REG=0` declines the registration (window then behaves exactly as
`pool`), plus `MOE_EXPERT_WINDOW_MLOCK` / `_TRACE` / `_ALIAS_FILL`.  `MOE_EXPERT_WINDOW` and
`MOE_EXPERT_WINDOW_MIB` are deleted — nothing is pinned (`VmLock` 0.00 MB), so there is no budget to
set.  The candidate, its gates and the `-sm tensor` ask are in
[`beta/0016-host-experts-window/`](../../beta/0016-host-experts-window/README.md); the measurements are
`PROTOTYPE.md` §8.  The design notes below are the original plan, kept because they still describe the
contract (promote/demote on a stream event, refuse rather than rely on reclaim, keep `pool` for the bulk
policy and for a set larger than RAM).

* Replaces the **pool** for single-device, whole-expert tables; keeps the pinned `ROCm_Host` master as the
  fallback (and for anything the window declines: `-sm tensor` slices, fused multi-slice tables, Windows — the
  `mmap` + `cudaHostRegister` recipe is POSIX-only, the KV mirror already records that).
* Env kill-switch `--host-experts window` (or `MOE_HOST_EXPERT_WINDOW=1`), default **off until gated**; the
  default-on flip is a separate, gated decision per `AGENTS.md`.
* Demote on a stream event, never on a device sync; LFRU/window size from the existing heat prior.
* Promote = `mlock` + `hipHostRegister`; demote = `munlock` + `hipHostUnregister`, gated on a stream event
  (§5.2).  Refuse to promote past the window budget rather than relying on reclaim (§5.1).
* Keep `--host-experts pool` for the bulk/prefill policy and for a set that exceeds RAM (§5.5).
* Gates: same-seed greedy text vs the pool arm, the `-ncmoe` matrix (1/2/3 devices, arena on/off), the MTP
  four-axis protocol (decode path), `gate-prefill-logits.sh` (the fill serves prefill staging too), and the
  existing `MOE_HOST_POOL_COLD` A/B.

## 7. The prototype

[`PROTOTYPE.md`](PROTOTYPE.md) — the design (one tier: the host master registered once and read in place),
what v1 got wrong (four distinct causes, all recorded), the v4 results (§5: +35-40 % decode, the attribution
per switch, the costs, the correctness matrix, the ring sweep that turned out to be a clamp, and why bugs A
and B do not reproduce).  Patches: `prototype-v1-window-tier.patch`,
`prototype-v2-pool-window.patch`, `prototype-v3-pool-window-coldread.patch` and
`prototype-v4-onetier.patch` (all against the r206 tree, `~/llama-scratch`).

## 8. Files

* `PROTOTYPE.md` — the live design + results (start here).
* `oracle.sh` — the deterministic greedy-text oracle (one line per arm; the arm is the caller's environment).
* `bench.sh` — `llama-bench -p 256 -n 128 -r 2` for one arm (same convention).
* `four-axis.sh` — the adaptive-MTP four-axis gate for one arm (same env convention; `AXES=`, `N=`,
  `NMAX=`, `AUTO_ARENA=1`, `AUTO_POOL=1`).
* `probe-pinned-window.cpp` — the probe (§1-§11 as printed above), no llama.cpp dependency.
* `probe-copy-path.cpp` + `trace-copy-path.sh` — the `rocprofv3 --memory-copy-trace` probe for §5.4.
* `build.sh` / `run-probe.sh` — ROCm 10.1 prefix, `--offload-arch=gfx1201`, `HIP_VISIBLE_DEVICES=0` pinned.
* `runs/` — probe logs, one per run.  **Read `probe-20261102-080314.log`** (the final form of the probe; it has §9-§11).
  Earlier logs in that directory are the same probe mid-fix and contain the two invalid tests described in
  §2 (`/tmp` tmpfs cold read at `073811`/`073902`; the `POSIX_FADV_DONTNEED` "pin" test at `074007`/`074026`).
