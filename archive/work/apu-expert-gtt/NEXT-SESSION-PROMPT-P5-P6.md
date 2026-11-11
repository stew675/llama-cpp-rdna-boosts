Paste everything below the line into a fresh session.  It is self-contained: it assumes no memory of the session that
wrote it, and it names every file, number and command you need.  It is the LAST session for this campaign: P5
(prefill streaming) and P6 (gate + promotion).

---

# 0. Where you are

Repo `~/llama-cpp-rdna-boosts` (the **delivery** repo), branch **`apu-expert-gtt`**, the handover commit is the **tip** of `apu-expert-gtt` (the last CODE commit is **`02d6e30e`**; the tip is the docs/handover commit on top of it),
pushed to `origin/apu-expert-gtt`.  The campaign is **P0 through P4f: COMPLETE and measured**; what is left is P5
and P6, and the promotion decision.

Box **halo**: Ryzen AI MAX+ 395 / **gfx1151 (RDNA3_5)**, UMA, 123 GiB RAM + 175 GiB swap, kernel 7.2.13,
**ROCm 10.1.0 at `/opt/rocm-10.1.0-gfx1151`**, `iommu=off`, `RLIMIT_MEMLOCK` unlimited, THP `madvise`, ONE GPU.

Development happens in the long-lived scratch tree, never in `~/llama.cpp` (that is the delivery checkout and must
stay clean):

```bash
cd ~/llama-scratch                                  # worktree, currently at 0cafbc344 / tree 3bc7adf4
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
cmake --build build-rocm --target llama-cli -j 16 -- -k      # ~10 s incremental (a .cu), ~1 min after a header
~/llama-cpp-rdna-boosts/scripts/scratch-tree.sh --rocm /opt/rocm-10.1.0-gfx1151 --gpu-target gfx1151 show
```

`scratch-tree.sh tip|state` **checks out** and refuses a dirty tree; for iterate-on-current-tree work use
`cmake --build build-rocm` directly.

Read, in this order, and trust them over any summary here:

1. `AGENTS.md` -- policy: apply/verify rules, the `-lv` table, "report findings before shipping", the WIP rules.
2. `archive/work/apu-expert-gtt/README.md` -- §0 the shape, §5 sizing, §5.2 the lock policy, **§6 the traps (read all of
   it; it is the accumulated "do not do that again" list)**, §7 open questions, §8 promotion criteria.
3. `archive/work/apu-expert-gtt/IMPLEMENTATION.md` -- the stage table (P0-P4f) and **the P5/P6 sections at the end**.
4. `archive/work/apu-expert-gtt/MEASUREMENTS.md` -- the numbers.  The sections that matter for P5/P6: §16 (the arena tier
   serves), §17 (the head pinned), §19 (never evicted), §20-§21 (the read transport + the direct fill), §22 (the
   overlap experiment: NO worker), **§23 (the heat-map trap -- read it before running anything)**, §24 (the
   RDNA3_5 UMA start arm).
5. `WORKLOG.md` entries **261111.012 ... 261111.024** (newest first; 024 is the handover of the current state).
6. `archive/work/apu-expert-gtt/arms.sh`, `guard.sh`, `cold-cache.py` -- the re-runnable tools (see §4 below).

# 1. What exists, and what has been measured (do not re-litigate)

The campaign's whole output is ONE cumulative patch:

* `archive/work/apu-expert-gtt/gtt-expert-store.patch` -- sha256 starts **`851e1871b2a59263…`**, tree `3bc7adf4…`, cut
  against the **r262** `release.json` tree `891d4acb…` (base `a691b2345`, tip `d65b0e889`), applies with `git am`
  onto the r262 tip and produces that tree.  Re-cut it after every change (`git -C ~/llama-scratch commit --amend`,
  `format-patch -1`, copy over), verify by `git am` in a throwaway worktree, and update the sha256 in the README
  status header + the new WORKLOG entry.

What that patch contains (all of it measured; the numbers are in `MEASUREMENTS.md`):

| piece | state |
|---|---|
| the **GTT store**: ONE anonymous region (`mmap` + `MADV_HUGEPAGE`), registered once `cudaHostRegister(Mapped\|ReadOnly)` **and `mlock`ed**, first-fit allocator, sliced per (tensor, expert) | works: 26 ms/GiB, 100 % huge pages, `mlock` free |
| `-lme` sizes it (N GiB exactly, or `auto` = `MOE_HOST_LIMIT_FRAC` of `MemAvailable`) | works |
| the lock policy: clamp to `RLIMIT_MEMLOCK`, a fitting model **declines**, a model that cannot fit **refuses** | works (`arms.sh limit`) |
| **P2 placement** (a set that fits: whole tables in the store, read in place) | works |
| **P3a/P3b arena** (`MOE_EXPERT_GTT_ARENA=1`, a set that does NOT fit: the store is the cache's slot store, the experts become holes in the read-only mapping, LFRU residency) | works: 120 tables / 4 082.3 MiB of 4 096 on 35B-A3B, 144 / 8 153.0 of 8 192 on Flash-Next, 144 / 61 047.0 of 61 440 on the 122B; **a 123.45 GiB model runs on this 123 GiB box** |
| **P3c** the MTP/nextn head pinned at full residency and OUTSIDE `-lme`, for the embedded head AND a separate drafter model; **never evicted** (three out-of-policy paths skip it, the census asserts it) | works: `768/768 slots, 0 evictions` against 7 670 trunk evictions |
| **P4a** the fills run on r262's adaptive `map`/`file`/`dio` transport | works |
| **P4b** the buffered arm reads STRAIGHT into the arena slot (no bounce, no H2D) | works: **+15.6 % decode warm** |
| **P4e** the overlap experiment | **answered: NO fill worker** (the drain in front of a direct fill is <= 263 us, never >= 1 ms; the fill is on the critical path) |
| **P4f** on RDNA3_5 + arena tier + direct fill, the ladder STARTS on the buffered arm | works: **+13.7 % warm** on a fitting model, **+56-67 %** on a >RAM model |
| kill switches | `MOE_EXPERT_GTT=0`, `_GTT_RO=0`, `_GTT_MLOCK=0`, `_GTT_ARENA` (opt-in), `_GTT_FILL_DIRECT=0` |

**`MOE_EXPERT_GTT_ARENA` is still default-OFF** and the tier is only reachable for a set that does not fit the
store.  Text identity is the oracle for correctness: `627b2ebd7792` (35B-A3B), `f586785d7183` (Flash-Next),
`9ccf62ed39c3` (122B), `ee95c45231c3` (122B, `-n 16`) -- same-seed greedy, hashed with **no `-lv`**.

# 2. What P5 is, and what it must show

`IMPLEMENTATION.md` **P5 -- prefill: stream only what is not resident**.

* Per layer: compute the chunk's used-expert set (the per-layer access plan already does this), **subtract the
  resident set**, and stream the difference *coalesced* -- this is `moe_cache_stage_table()`'s span shape with the
  GTT store as the destination.  The existing coalescing to reuse is `stage_span_fill()` /
  `MOE_HOST_POOL_STAGE_SPAN` (+38 % measured on the discrete pool in W4).
* Do not re-fetch a bank the previous chunk already holds; the census's per-layer read bytes must fall on a
  **repeated-prompt** arm.
* Ordering: a prefill fill is a bulk fill -- the warm arm is the copy engine, the cold arm a batched `pread`
  (parallel, not one `pread` per expert).  Measure both before choosing.
* **Interaction to watch**: `pool_transport_apply_locked()` REFUSES the `file` arm when
  `MOE_HOST_POOL_STAGE_SPAN=0` (the measured -41 % pair), so P5 touches the same switches as the transport ladder.
  With P4f the tier *starts* on `file` on RDNA3_5 -- if a P5 change turns the span off, the ladder will refuse
  `file` and step to `dio`, which is 11x slower here.  Check the start arm after any span change.

**Accept**: prefill logits identical to the all-resident arm via **`scripts/gate-prefill-logits.sh`** (this IS a
prefill-path change, so the KLD gate is mandatory, not just text: mean KLD <= 0.005 and same-top-p >= 98 %); text
identical; read bytes strictly below "fetch every used expert" on the repeated-prompt arm; pp rate reported against
the all-resident arm.

# 3. What P6 is

**P6: gate, docs, promotion** (`IMPLEMENTATION.md` P6, `README.md` §8, `IMPLEMENTATION.md` §7 for the gate spec).

1. **`scripts/gate-uma-experts.sh` does not exist yet** -- it and its **11 arms** are the deliverable
   (`IMPLEMENTATION.md` §7, the table: text identity / all-resident / >RAM / cold page cache / budget /
   no-movement / 6b admission / 6c `-lme` / kill switch / fill-fault / prefill / host-buffer A-B).  Model it on
   `scripts/gate-host-pool.sh` (same shape: byte-identical text per arm, a fault arm that must decline, censuses
   at `-lv 4`).  **Every arm must freeze the heat-map prior** (`MOE_EXPERT_HEATMAP=load`) -- `gate-host-pool.sh`
   already does, and MEASUREMENTS §23 is what happens when an arm does not.
   Arm 9 (prefill) asserts **the P4e answer**, not a pipeline: the recorded answer is "no fill worker, the fills
   are on the critical path" (§22).
2. **`ENVIRONMENT.md`** gets every campaign variable with its class: `MOE_EXPERT_GTT` (kill switch, default 1 on an
   integrated GPU), `MOE_EXPERT_GTT_RO` (kill switch), `MOE_EXPERT_GTT_MLOCK` (kill switch), `MOE_EXPERT_GTT_ARENA`
   (opt-in, or default-on if the maintainer decides -- see §5), `MOE_EXPERT_GTT_FILL_DIRECT` (default-on with a
   kill switch), plus the reused `MOE_HOST_LIMIT_FRAC` / `MOE_EXPERT_CACHE_TOUCH` / `MOE_EXPERT_HEATMAP` /
   `MOE_HOST_POOL_TRANSPORT` / `MOE_HOST_POOL_STAGE_SPAN`.  **The size is not a variable -- it is `-lme`.**
3. **`INVARIANTS.md`** gains the facts this campaign measured that it does not already have: gfx1151 has no
   `xnack` target at all; an APU's "VRAM" is the host pool; a registered RO range reads 230-233 GB/s against 60
   writable; `hipHostRegister` on `mmap` does NOT pin (probe 16 -- only `mlock` does, and it is the only thing the
   accounting reports); a pageable mapping may be a fill SOURCE and never a read target; `pread` into a registered
   range is legal and coherent; `O_DIRECT` needs the 4096 round-out; the GTT store's extent cannot be exceeded;
   a failed carve is FATAL (not a fallback); **the heat-map prior is an INPUT**.
4. **Promotion** (`README.md` §8): the set moves into `patches/` as a block (block 09/12 own that code per
   `BASELINE.md`), default-on per AGENTS.md's default-on policy with a kill switch, `patches/README.md` block note,
   `release.json` regenerated by `scripts/make-release.sh`, then the full gate ladder:
   `scripts/gate-per-block-build.sh`, `scripts/validate-set.sh`, `check-gpu-faults.sh`,
   `gate-prefill-logits.sh`, and the **MTP four-axis protocol** (`benchmarks/mtp-adaptive-methodology.md`) because
   this changes decode.  `scripts/fork-refresh.sh --build --gates --push` is the release-time flow.

# 4. Facts, tools and traps that will bite if you improvise

* **`-lv` is chosen by the arm**: text/hash arms take **no `-lv`** (hash with `scripts/extract-generated.py`);
  census/mechanism arms take **`-lv 4`**.  `-lv 3` is the trap.  **Two runs per arm, never one.**  A `-lv` run's
  text is POLLUTED -- I have twice measured "different hashes" that were just `-lv` (`-lv 4` on the 122B gives
  5 438 vs 5 565 chars between arms that are byte-identical).
* **Freeze the heat-map prior for every measurement arm**: `MOE_EXPERT_HEATMAP=load` (or `chmod a-w` the sidecar
  -- the file's mode IS the freeze signal and is logged).  Otherwise each run merges its tally into the prior and
  rewrites it (`<model>.moe-heatmap`, `.gguf` stripped, **0644**), arms contaminate each other, **run order becomes
  part of the result**, and a conclusion can invert: the `auto` ladder's latch rate looked like 4/5 with a drifting
  prior and is **0/5** with a frozen one (MEASUREMENTS §23).  `arms.sh` does this for every arm now.
* **Use `guard.sh` for every arm** (`AS_KB=100663296 LOCK_KB=8388608`, `AS_KB=0` for Flash-Next whose 27 GiB
  resident PLE table does not fit a 96 GiB address space, `LOCK_KB=unlimited` for a >RAM arm).  It also runs the
  1 s `SwapFree` monitor that is the only barrier that has ever caught a real bug.
* `cold-cache.py MODEL.gguf` drops a GGUF's page cache (no root needed) for a cold arm; for a >RAM model every
  read is cold anyway.
* **Naming**: never a bare "store" -- always a **GTT store**, and it is the expert **arena** but NOT the slab's
  "arena".  Never interchange the words.
* **Never apply `weight_buft_supported()` to a host master**; a host weight reaches the GPU through the scheduler's
  OFFLOAD path.
* **One store, many loads**: a second model in the process (a separate MTP drafter, a reload) must never size or
  grow the store and must never take the arena tier (`ggml_backend_dev_gtt_store_free_bytes` answers what is
  free; the hook reports 0 while the arena is live).  A failed carve FAILS the load -- size against the free bytes.
* **`-lme` below 2 048 MiB** makes the cache decline the arena tier entirely (`moe_cache_preflight_store`'s
  floor), so the tier silently becomes placement.  `-lme 1` also puts the *model* into placement, which is fine.
* **Flash-Next has NO embedded MTP head**: `--spec-type draft-mtp` without `-md` fails with `model doesn't
  contain MTP layers`.  Its drafter is a separate file (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, one `blk.48`
  block, 3 expert tables, 2 550 MiB) and it is **declined from the store by rule** (its weights are a whole-set
  device allocation, not cache tables, so they cannot be evicted -- the correct tier for it).
* Never run benchmarks in parallel; size `-t` (leave the GPU IRQ cores free); never push to upstream llama.cpp.

# 5. The two decisions the maintainer still owes (ask before shipping)

1. **`MOE_EXPERT_GTT_ARENA`'s default at promotion.**  AGENTS.md says a beneficial feature ships ON with a kill
   switch; the campaign has deliberately kept the arena tier opt-in because it is a *fittability* tier (it costs
   decode where the experts fit).  Either that stays a recorded exception, or it goes on with `=0` as the exit.
2. **The transient band** (README §5.1) is currently dropped by design: every miss fills, so
   `MOE_EXPERT_CACHE_TOUCH` is a no-op on this tier.  Reviving it is the only thing that would make admission a real
   policy here -- and only if a measured hit-rate says it is needed.

# 6. Measurement debt that promotion cannot skip

* **Multi-device is untested** (one GPU here): the per-device budget, the per-device head subtraction and the
  free-bytes query have never seen a `-sm tensor`/`-sm layer` split across APUs.
* **The MTP four-axis protocol** (`benchmarks/mtp-adaptive-methodology.md`) is owed for any decode-affecting
  change, and P4/P4b/P4f are exactly that.  The P3c/P4 arms so far show only `draft acceptance=1.0` on 2-token
  arms, which is not it.
* **No throughput claim** for the tier as a whole: the only decode numbers are the P4 A/Bs (one box, `-n 96`,
  3 reps, heat prior frozen) -- `MEASUREMENTS.md` §24.2 is the best pair (29.0 against 25.5 t/s warm; 7.5-7.8
  against 4.5-5.0 on the 122B).
* **Probe 15** (touch distance) is still owed; probes 01-14 and 16 exist.
* **The discrete side**: `scripts/gate-host-pool.sh` cannot run on this box (no discrete GPU).  The campaign's
  changes are inert there by construction (`g_gtt_arena` / `prop.integrated` gating) but that is a claim, not a
  measurement -- say so in the record.
* Parked by the maintainer, do not pick up: sizing the store for a **separate** drafter's head (a `common`/`--fit`
  term; `MEASUREMENTS.md` §18.4).

# 7. First three actions

1. `cd ~/llama-cpp-rdna-boosts && git log --oneline -3` and read `WORKLOG.md` `261111.024`; then
   `cd ~/llama-scratch && git log --oneline -1 && git status --short` (clean, tree `3bc7adf4`).
2. Read `IMPLEMENTATION.md` P5 + P6, `README.md` §6 (traps) and §8 (promotion), and `MEASUREMENTS.md` §23 (the
   heat-map trap) -- **before running anything**.
3. Reproduce the reference before changing anything, one text arm and one census arm, prior frozen:
   ```bash
   cd ~/llama-scratch && export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0
   M35=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf
   MOE_EXPERT_HEATMAP=load MOE_EXPERT_GTT_ARENA=1 AS_KB=100663296 LOCK_KB=8388608 \
     ~/llama-cpp-rdna-boosts/archive/work/apu-expert-gtt/guard.sh /tmp/gtt-arms/ref-text.log \
     ./build-rocm/bin/llama-cli -m $M35 -ngl 99 -p "The capital of France is" -n 20 --seed 42 --temp 0 \
       --no-display-prompt --single-turn -lme 4 --fit off      # expect sha=627b2ebd7792
   # the same arm at -lv 4 shows the census, including `GTT arena tier: ... read transport file (RDNA3_5 UMA start...)`
   ```
   `python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py <log>` prints the hash.

# 8. The definition of done for this session

1. P5 implemented and measured: **`scripts/gate-prefill-logits.sh` green** (mean KLD <= 0.005, same-top-p >= 98 %),
   text identical, and the read-byte census falling on a repeated-prompt arm; the pp rate against the all-resident
   arm reported (no invented throughput claims).
2. `scripts/gate-uma-experts.sh` written and green, every arm with the prior frozen, arm 9 asserting the P4e
   answer.
3. Dated `WORKLOG.md` entries for P5 and P6; the campaign README/IMPLEMENTATION status headers updated; the
   artifact re-cut against the r262 tree with its sha256/tree in the README header.
4. The two decisions in §5 asked and recorded; the debt in §6 either measured or explicitly carried into the
   promotion note.
5. **Report findings to the maintainer before committing anything delivery-affecting** (AGENTS.md), and do not
   push a `patches/` change without their go-ahead.
