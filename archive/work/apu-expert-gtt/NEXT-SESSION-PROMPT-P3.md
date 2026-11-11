Paste everything below the line into a fresh session.  It is self-contained: it assumes no memory of the session
that wrote it, and it names every file, number and command you need.  Do not look for a deleted predecessor
campaign (`wip/apu-expert-pool/`, `wip/uma-static-window/`): it was deleted on purpose (WORKLOG 261111.008) and its
findings must not be ported.

---

# 0. Where you are

Repo `~/llama-cpp-rdna-boosts` (the **delivery** repo), branch `apu-expert-gtt`, tracking `origin/apu-expert-gtt`.
Head at handover: `58e9adf3`.  Box **halo**: Ryzen AI MAX+ 395 / **gfx1151**, UMA, 123 GiB RAM, 175 GiB swap,
kernel 7.2.13, **ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`**, cmdline has `iommu=off`, `RLIMIT_MEMLOCK` unlimited
(`/etc/security/limits.conf`), THP `madvise`.  One GPU → `HIP_VISIBLE_DEVICES=0`.

Development happens in **one long-lived scratch tree**, never in `~/llama.cpp` (that is the delivery checkout and
must stay clean):

```bash
cd ~/llama-scratch                       # a git worktree of ~/llama.cpp, detached at the campaign tree
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
cmake --build build-rocm --target all -j 16 -- -k        # incremental; ~10 s for a .cpp/.cu, ~15 min after a header
~/llama-cpp-rdna-boosts/scripts/scratch-tree.sh --rocm /opt/rocm-10.1.0-gfx1151 --gpu-target gfx1151 show
```

`scratch-tree.sh tip|state` **checks out** and will refuse a dirty tree — for iterate-on-the-current-tree work use
`cmake --build build-rocm` directly, as above.

Read, in this order, and trust them over any summary here:

1. `AGENTS.md` — policy: apply/verify rules, the `-lv` table, "report findings before shipping".
2. `archive/work/apu-expert-gtt/README.md` — §0 the shape, §2/P1.1+P2 the measured foundation, **§5.2 the mlock/refuse
   policy**, §6 traps, §7 open questions.
3. `archive/work/apu-expert-gtt/IMPLEMENTATION.md` — **P3.0 "the seams"** (four findings that define the tier) and its
   **stage table, which contains an OPEN BUG box**.  That box is the state of the world you are inheriting.
4. `archive/work/apu-expert-gtt/MEASUREMENTS.md` **§14 and §15** (the store's numbers, probe 16, and §15.3 the lock policy).
5. `archive/work/apu-expert-gtt/WORKLOG`-equivalent: `WORKLOG.md` entries **261111.009 … 261111.013** (newest first).
   **261111.013 is the handover of the current state** — read it before you run anything.
6. `archive/work/apu-expert-gtt/arms.sh` — the re-runnable arms (`text`, `census`, `rate`, `limit`).

# 1. What already works (do not re-litigate, do not rebuild)

The delivery is unaffected: nothing in `patches/`, `release.json` untouched, nothing applied to the fork.  The
campaign's whole output is ONE cumulative patch:

* `archive/work/apu-expert-gtt/gtt-expert-store.patch` — sha256 starts **`e38a3f15f57a39c3…`**, tree `f0991921…`, cut
  against the `release.json` tree `86389e4a…`, applies with `git am`.  Re-cut it after every change
  (`git -C ~/llama-scratch commit --amend`, `format-patch -1`, copy over) and update the sha256 in the README
  status header + the new WORKLOG entry.

What that patch contains and has **measured** (all on `halo`, all in `MEASUREMENTS.md`):

| piece | state |
|---|---|
| the GTT store: ONE anonymous region (`mmap` + `MADV_HUGEPAGE`), registered once `cudaHostRegister(Mapped\|ReadOnly)` **and `mlock`ed**, sliced per (tensor, expert) | **works**: 26 ms/GiB to create+register, 100 % huge pages after the load, `mlock` free (26.3 vs 25.8-26.0 ms/GiB) |
| `-lme` sizes it (N GiB exactly, or `auto` = `MOE_HOST_LIMIT_FRAC` of `MemAvailable`, capped at the set) | **works**: `-lme 6` realises exactly 6 144.0 MiB; `auto` on a fitting set is the whole set, zero evictions |
| the store as a whole-tensor **placement** (`MUL_MAT_ID` reads the store in place) | **works**: text identical to `MOE_EXPERT_GTT=0` and to the pre-P2 build on 35B-A3B (`627b2ebd7792`) and Flash-Next (`f586785d7183`); pp512 within noise of the plain device buffer for the READ-ONLY arm, −5.9/−6.2 % registered writable |
| the **lock policy**: extent clamped to `RLIMIT_MEMLOCK`; a fitting model **declines** with the fix; a model that cannot fit **refuses** (0.76 s on the 122B) | **works**, arms `./archive/work/apu-expert-gtt/arms.sh limit` |
| kill switches `MOE_EXPERT_GTT=0`, `MOE_EXPERT_GTT_RO=0`, `MOE_EXPERT_GTT_MLOCK=0` | **work**; `GTT=0` reproduces the pre-P2 log byte for byte |
| the store as a real **allocator** (first-fit, coalesced free list, `reserve`/`release`/`largest_free`) | **works**, verified as a no-op for the placement mode |
| the store as the cache's **ARENA** (P3) | **WIRED BUT NOT WORKING — this is your job** (`MOE_EXPERT_GTT_ARENA=1`, default OFF, only for a set that does not fit) |

# 2. The two blockers you inherit

**(B1) The arena tier registers NO tables.**  Census of an arena arm:

```
MoE expert cache: h=0.0000 (0/0 reaches), fills=0 evictions=0, tables=0, slots=0,
arena=0.0 MiB total (budget 4096 MiB/device)
```

With `tables=0` the ops read their **own buffer**, the arena is never touched, and **a byte-identical text arm
proves nothing**.  So: do not accept any evidence for this tier that does not include `tables=` > 0 and
`arena=` ≈ the resident bytes.

**(B2) Arena mode can allocate without bound.**  It took this 123 GiB box to full memory + swap **twice**.  Case 1
was a full-set arm mirroring 19 GiB through the pageable mapping; that is now prevented by construction (the
identity-mirror guard in `alloc_table_locked`).  Case 2 was a **`-lv 4` census with a PARTIAL arena (`-lme 4`),
inside a `MemoryMax=48G` cgroup** — so it is not simply cgroup-charged anonymous memory.

**The maintainer's hypothesis to test first** (stated at handover, not yet tested): something is sizing the arena
**and** the fractional percentage **and** the normal full weight load at the same time — possibly `--fit`.  Treat
that as the leading theory.  `--fit` is ON by default and runs a `no_alloc` probe load; the store plan is skipped
for `no_alloc` (`if (no_alloc) return;` in `plan_gtt_expert_store`), so check what the **real** load then does with
the *context* sizing, and whether the probe's numbers are being used to size something.

# 3. Safety protocol — MANDATORY, in this order, and it is the experiment

Do **not** run an arena arm unguarded.  Each barrier tells you a different thing; hit them one at a time and note
**which** one fires:

```bash
# barrier 1: address space.  ROCm maps large VA reservations, so do NOT go low; 96 GiB is the maintainer's figure.
bash -c 'ulimit -v 100663296; ulimit -l 8388608; exec <command>'   # 96 GiB AS, 8 GiB mlock
#   ulimit -v -> malloc/mmap failures (anonymous growth)      -> anon allocation, stack a backtrace or print sizes
#   ulimit -l -> the store's mlock is clamped (8 GiB)         -> less unevictable pressure, and the census says so

# barrier 2: cgroup (charge-based; known NOT to catch case 2 alone, so use it WITH barrier 1)
systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=4G --quiet <command>

# watch, in a second shell (1 s sampling; stop the run the moment SwapFree starts dropping)
while :; do free -g | awk 'NR==2||NR==3'; pid=$(pgrep -n llama-cli); [ -n "$pid" ] && \
  awk '/VmPeak|VmRSS|VmSwap|Threads/{printf "  %s", $0}' /proc/$pid/status; echo; sleep 1; done
# and after a kill: dmesg | tail  (which barrier the kernel reported), /proc/<pid>/smaps_rollup
```

`ulimit -v` will also break `cudaMalloc`/`hipMalloc` if ROCm needs more VA than the cap; if the run dies at *load*
with a HIP allocation error, raise the cap to 110 GiB and note that the *device* side is what needs the room.

A cheap, zero-risk first experiment: **`--fit off`** (skips the fit probe entirely) with the arena arm, guarded as
above.  If the explosion disappears, the hypothesis is confirmed and the bug is in the fit/store interaction.

# 4. Order of work (each step is cheap before the next is expensive)

1. **B1, no arena memory at stake.**  Find why `tables=0`.  Places to look, in this order:
   * does `ggml_backend_dev_moe_cache_register` (the scheduler's hook) get called for a host-**mapped** expert
     weight on an APU?  `moe_cache_pending()` gates the whole registration pre-pass
     (`ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda-moe-cache-hooks.cu`).
   * `moe_cache_register_host()` returns −1 on: `!g_enabled`, `weight == nullptr`, `ne[2] < 1`, `name_layer() < 0`.
     Instrument those four with a one-shot `fprintf(stderr, …)` under an env gate (there is precedent:
     `GTT_DEBUG_BUFT` was used and removed — do the same and **remove it before committing**).
   * is the weight's buft the `ROCm_HostMap` one (`moe_cache_gtt_arena()` true, `moe_host_master_buft` answering)?
     Is the tensor actually in the mapping (`llama-model.cpp`'s `host_master_mapped` branch)?
   * `moe_cache_tables_complete_locked()` / `g_preflight_device_bytes[device]`: with the tensors as host buft,
     `llama_model_moe_cache_preflight()` now runs on the APU and sets that byte count, so the completeness gate may
     be waiting for a sum that never arrives (the tables register with a *device slice* size, not the host size).
     **This is the most likely single cause — check it first.**
2. **B2, with the barriers.**  Once a table registers, the allocation source will be obvious: the arena reserve is
   exact and cannot exceed the region, so anything above it is the fill path (pageable H2D staging), the prefill
   staging ring (`span_buf_ensure`), the heat/ghost mirrors, or `--fit`.
3. **P3b, only then**: `-lme` smaller than the set, real evictions.  Accept criteria: `evictions > 0`,
   `unregisters == 0`, and same-seed greedy text identical to the `MOE_EXPERT_GTT=0` arm (hashed with **no `-lv`**).
4. **P3c**: the nextn/MTP head's 3 tables pinned in the store and **excluded from the `-lme` arithmetic**
   (decided by the maintainer: the head is metadata overhead, read every draft step).

# 5. Facts and conventions that will bite if you improvise

* **`-lv` is chosen by the arm**: text/hash arms take **no `-lv` at all** (hash with `scripts/extract-generated.py`);
  census/mechanism arms take **`-lv 4`**.  `-lv 3` is the trap (pollutes the text *and* hides the censuses).
  **Two runs per arm, never one.**
* **Text is the oracle for *correctness only*.**  B1 is the proof: identical text with an inert mechanism.  Always
  pair a text arm with the census that shows the mechanism moved.
* **Naming**: never write a bare "store" — it is always a **GTT store**.  And the GTT store *is* the expert **arena**,
  but it is **explicitly not** the slab's "arena" (the discrete boxes' VRAM slab).  Same role, different
  implementation; never interchange the words.
* **Never apply `weight_buft_supported()` to a host master.**  A host weight reaches the GPU through the scheduler's
  OFFLOAD path (`ggml_backend_sched_backend_id_from_cur`), not `supports_buft`.  With that check in place the expert
  tensors silently stayed on the device (20 583 MiB) and the run *looked* perfect.
* **`moe_cache_preflight()`'s integrated guard now yields to the arena tier** (it returns early when
  `moe_cache_gtt_arena()`).  That is the "select the GTT store instead of `g_enabled = false`" hookup; do not revert it.
* The tier sets `g_cold_uva = false` **by construction**: on an APU the "cold" source would be the pageable model
  mapping, and a GPU read of that is a *fatal* fault (no `PageableMemoryAccess`, no XNACK).  With no cold path every
  miss fills, so `MOE_EXPERT_CACHE_TOUCH` is a no-op here — the transient band (§5.1) is a later, priced refinement,
  not a prerequisite.
* **Never default the arena tier on.**  `MOE_EXPERT_GTT_ARENA` must stay opt-in until `tables > 0` *and* the
  allocation is bounded *and* an eviction arm passes.  With it unset, an APU must behave exactly as the placement
  mode: re-verify `627b2ebd7792` after every change (`arms.sh text --quick`).
* Register the work: a dated `WORKLOG.md` entry (newest first, `YYMMDD.VVV`, today's real date), the campaign
  README/IMPLEMENTATION status, `MEASUREMENTS.md` for numbers.  **Report findings to the maintainer before
  committing** (AGENTS.md) — especially if a measurement contradicts the design.

# 6. Models, and the definition of done

```
M35=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf      # 18.22 GiB of experts, the workhorse
MFN=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf  # ~40 GiB
MBIG=/llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf  # 123.45 GiB: cannot fit here
```

This session is done when **all** of these hold:

1. `tables > 0` and `arena=` equals the resident bytes in a `-lv 4` census of the arena tier.
2. No arena arm allocates beyond a stated cap (say 1.25 × the store extent) — enforced by an assertion you add, not
   by watching the box.
3. `-lme` smaller than the expert set: **evictions > 0, unregisters == 0**, and same-seed greedy text identical to
   the `MOE_EXPERT_GTT=0` arm on 35B-A3B.
4. `MBIG` runs to completion at a `-lme` the box can hold (that is the design's whole point: a >RAM model).
5. `scripts/check-gpu-faults.sh` clean, no `HSA_STATUS_ERROR_MEMORY_FAULT`, and the default path (`MOE_EXPERT_GTT_ARENA`
   unset) still gives `627b2ebd7792`.
6. The patch re-cut against `release.json`'s tree, its sha256 in the README status header, and a dated `WORKLOG.md`
   entry that says exactly what was measured and what was not.

# 7. First three actions

1. `cd ~/llama-cpp-rdna-boosts && git log --oneline -3` and read `WORKLOG.md` 261111.013; then
   `cd ~/llama-scratch && git log --oneline -1 && git status --short` (it should be clean at the campaign tree).
2. Read `IMPLEMENTATION.md` P3.0 + its OPEN BUG box, and `README.md` §5.2.
3. Reproduce B1 under the barriers, **`--fit off` first**, with the census arm — i.e. answer "does a table register,
   and does the explosion survive `--fit off`?" before writing any code.
