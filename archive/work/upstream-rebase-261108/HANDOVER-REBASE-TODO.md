# HANDOVER — the re-base TODO list, then R17 (the release)

**Status: the ladder is CLIMBED (R01–R16 = all 16 blocks).  What is left is a short list of re-base-owned
defects/measurements, then the release ladder (R17) as the LAST task.**  Nothing here is delivery:
`patches/`, `release.json` and the fork's `rdna-boosts` branch stay untouched until R17.

**Read order:** this file → [`RUNGS.md`](RUNGS.md) (the ledger, every rung's numbers) →
[`DECISIONS.md`](DECISIONS.md) (`D1`–`D13`, `D15`) → [`HANDOVER.md`](HANDOVER.md) §0.1/§0.2 (the hard rules and
the environment, still verbatim) → [`HANDOVER-R16.md`](HANDOVER-R16.md) §6/§7 (the R17 ladder as first written and
the documentation owed) → `../../AGENTS.md` → `../../TODO.md` → `../kv-residency-cleanup/README.md` §2.1.

## §0 State (verified at handover, 2026-11-08)

| item | value |
|---|---|
| fork checkout | `~/llama.cpp`, branch **`rebase-r236`**, tip **`2aac85e78`**, tree **`a2e299275`**, clean |
| `master` there | the pristine upstream snapshot at **`a691b2345`** — never commit to it |
| commits over the base | **21** (`a691b2345..rebase-r236`): R01–R16 + 5 follow-ups (§0.1) |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-r236`**, head **`777f9987`**, pushed |
| `main` in the delivery repo | still the shipped **r235** — the switch-over is R17 |
| build (primary) | `~/bin/build-llama-rocm` (canonical, `-DGGML_HIP_NO_VMM=OFF`, VMM on).  Green at `2aac85e78` |
| comparison build (primary) | `~/llama.cpp/build-rocm-r235` = the pre-re-base `rdna-boosts` tree, same flags |
| upstream baseline (NEW 2026-11-08) | `~/llama.cpp/build-rocm-base` = **unpatched upstream `a691b2345`**, same flags, at the same source path — R17 step 5 records the prefill-logit base from it |
| reference numbers | [`../../benchmarks/2026-11-08-rebase-r235-vs-r236.md`](../../benchmarks/2026-11-08-rebase-r235-vs-r236.md) |

### §0.1 The five follow-up commits (they are part of r236, all pushed, all verified)

| sha | what |
|---|---|
| `328a155a9` | **UMA guard**: host-resident MoE experts (`-ncmoe`/`-cmoe`/`-lme`/a CPU expert `--override-tensor`) are REFUSED on any `GGML_BACKEND_DEVICE_TYPE_IGPU` device, one WARN, all weights stay on the device.  Inert on discrete GPUs |
| `e61e354d2` | the VMM pool's free-order assert now prints sizes/offsets (stderr, abort path), plus `GGML_CUDA_POOL_TRACE=1` (POSIX only) recording a backtrace per live block.  Off by default |
| `812deb221` | **the MTP-path pool-order fix**: `ggml_cuda_mul_mat_id_mmb_glu` took its long-lived `pool_stack` slot after five plain pool temporaries; the slot is now reserved first |
| `2aac85e78` | **the attendance pool declines on an integrated GPU** (message names RDNA3.5/gfx1151 and `TODO` #300) |
| `08d6c343b` | **the fused indexer ops' reachability census** (`GGML_CUDA_QSA_SCORE_LOG`, default off, block 11 — `TODO` #297's gate needs it; measured inert) |
| `9b7dfb47a` | **the BF16 src1 cache block is reserved before any plain pool allocation** (block 10 — `TODO` #298's live pool-order violation, reproduced and fixed) |
| — | (`2dc4511b8` is the R16 rung itself, i.e. the 16th block; `617ee7152`… are R13–R15) |

Cross-platform state: **gfx1201** (primary, 3 × R9700) and **gfx1100** (`fingon`) both build and run;
**gfx1151** (`halo`) builds and runs, the **window** and the **sparse** arm are validated there, the **pool is
rejected** (`TODO` #300).

### §0.2 The boxes (they behave differently — read this before choosing one)

| box | device | notes |
|---|---|---|
| primary (`this`) | 3 × gfx1201, 32 GiB each, 91 GiB RAM | the benchmark platform; 1/2/3-GPU arms available (`HIP_VISIBLE_DEVICES=0`, `0,2`, `0,1,2`) |
| `fingon` | 1 × gfx1100, 24 GiB, 61 GiB RAM | **unstable under load**: it rebooted 3× during full-parallel builds → build with `JOBS=4`.  Its 9B pp readings spread 2.5×, so **its timings are not usable**; correctness runs are fine.  `/tmp` is wiped on reboot.  No 4B model (use `Qwen3.5/9B/Q8_0`) |
| `halo` | 1 × gfx1151 (Strix Halo, UMA), 128 GiB | stable, slow.  A 2 GiB VRAM carve-out + GTT: `-ncmoe` is refused by design, the pool is rejected, the **window** works.  `-sm layer` multi-device arms need 2 GPUs (not available) |

**Transferring the tree to a box:** `git bundle create /tmp/b.bundle a691b2345..rebase-r236`, `scp` it, then
`git fetch /path/b.bundle rebase-r236:refs/heads/tmp && git checkout -B rebase-r236 tmp && git branch -D tmp`
(a direct fetch into a checked-out branch is refused).  Both boxes also have the delivery repo on
`rebase-r236`; update it the same way (`git fetch origin rebase-r236:refs/heads/tmp`).

## §1 The re-base TODO list (close these before R17)

**The campaign's handle list is `#279`, `#286`–`#298`, `#300`, `#301`** — the pre-existing backlog is `#116`,
`#131`–`#134`, `#270`, `#283`–`#285`.  **Twelve** of the campaign's handles are still open (`#279`, `#286`, `#287`,
`#289`, `#290`, `#291`, `#294`–`#298`, `#300`, `#301`); several were recorded as closed in the pre-re-base records and
must be closed **on r236**, which is a different thing.

| # | item | when |
|---|---|---|
| **#294** | the gates' kernel-journal window | **before R17** (makes every later gate verdict trustworthy) |
| **#297** | the qwen4exp sparse QSA arm has no gate | **before R17** (~30 lines of shell + one hash) |
| **#298** | sweep the `pool_stack_push` sites (the ordering class) | **before R17** |
| **#301** | r236 is −2 … −3 % on MoE **prefill** vs r235 | **before R17** — the only *measured* regression, decides "no slower" |
| **#279** | narrow-2 prefers `cudaMalloc` over evicting arena tables — measured CLOSED **pre-re-base**; the re-base moved the slab/arena so the basis is gone | **before R17** (one A/B) |
| **#286** | re-measure the Meta compute headroom (delivery 128 multiplier vs upstream's 256 floor, `D7`) | **before R17** (one 2-device arm) |
| **#290** | the dispatcher merge debt "lands with r236" — verify it in the **re-cut** patch, not the working tree | **at R17 step 1** |
| **#287** | re-cut `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` against `a691b2345`/`#31047` | **before R17** (delivery-neutral, upstream contribution) |
| **#289** | upstream PR owed: `find_slot()` must swap `mem_cell::rs_max` (fixed here at R04) | **before R17** (upstream contribution) |
| **#291** | the arena-vs-upstream-MoE-cache measurement (`D1`, MEASURING) | its own session — **the largest item**; a measurement, not a code change |
| **#303** | the dense `mul_mat_q` is 1.4 % slower than r235 at identical geometry (the last prefill gap) | its own session: [`HANDOVER-303.md`](HANDOVER-303.md) |
| **#304** | `MUL_MAT_ID` dup: `f16` at `ncols_dst = 9` is wrong (and nothing tests the path) | before R17 |
| **#287** | (halved) the mul-mat-id-dup re-cut is WIP: [`../upstream-mul-mat-id-dup-recut/`](../upstream-mul-mat-id-dup-recut/README.md) | its own session |
| **#295** | R17 must include a host-expert **two-slot DFlash** arm | **during R17** (a ladder arm, not a fix) |
| **#296** | `D8`'s CPU-win measurement | **during R17** (a ladder arm, not a fix) |
| #300 | gfx1151 attendance pool | **parked** (mitigated: the knob declines there), not a blocker |

So the pre-R17 work is **#279, #286, #287, #289, #294, #297, #298, #301** (+ #290 verified at R17 step 1),
**#291** is its own session, and the ladder absorbs **#295**, **#296**.  `TODO.md` now carries every one of these
handles with what "closing it" means — they were reconstructed from the campaign records because they had never been
carried into `TODO.md` (they were only in this directory's "open handles" list).


Four defects/gaps, one regression to localise, three decisions to close, plus housekeeping.  Everything below is
already written up in `TODO.md`/`DECISIONS.md`; this section says what to actually do.

### 1.1 `TODO` #301 — **r236 is −2 … −3 % on MoE PREFILL** (the only measured regression)

`benchmarks/2026-11-08-rebase-r235-vs-r236.md`.  Decode is flat on every arm; prefill is −3.2 % on **30B-A3B
Q4_K_M** (1 GPU), −2.0 % on **35B-A3B Q8_0** (3 GPU `-sm tensor`), −1.4 % on a dense 27B, flat on the dense 4B
and +0.7 % on a host-expert arm (`-ncmoe 30`).

* **Ruled out** (the gap survives each, and its absolute size stays ~130-160 t/s): MMB (`GGML_CUDA_MMB=0` −3.5 %),
  MMQ (`GGML_CUDA_DISABLE_MMQ=1` −3.2 %), the fusion dispatcher (`GGML_CUDA_DISABLE_FUSION=1` −1.6 %).
* **Shape of the evidence** says "per-launch or per-batch cost on the device-resident-expert path", not one kernel.
* **Next, cheapest first:** (a) rocprofv3 kernel list + counts on `30B-A3B Q4_K_M -p 512 -n 0` for both binaries and
  diff the kernel/host-function census; (b) the `-lv 4` launch census (graph nodes, `mul_mat_id` launches, the
  MoE dispatch counters) for both; (c) if neither names it, bisect the ladder: `scripts/gate-per-block-build.sh`
  already materialises all 16 block commits, so building the 16 tips and benching `-p 512` on 30B-A3B finds the
  rung that introduced it (~16 builds + 16 short benches, ccache makes the builds ~1-2 min each after the first).
* **Rung that most plausibly owns it:** R11 (block 10 slab/dispatcher/ring) and R06/R07 (block 05/06 FA/MMB) are
  the prefill-side rungs; R01 (block 00) is the allocator/band.  The gap is *absolute-constant*, which smells like
  a launch-count or a per-ubatch copy, not a arithmetic change.
* **Acceptance criterion:** either restore within ~1 % of r235 on the two MoE arms, or record a measured,
  understood cost and the reason it is acceptable, in the benchmark record + `WORKLOG.md`.

### 1.2 `TODO` #298 — sweep the `pool_stack_push` sites (the ordering class, one site fixed)

Rule (R13 `hc_combine_norm` + this session's `mmb_glu`): **no plain device-pool allocation may be created before a
`pool_stack_push` in the same scope** — the pool asserts LIFO (the VMM pool) and the non-VMM pool silently corrupts
its accounting.  Live sites to audit:

| site | check |
|---|---|
| `ggml/src/ggml-cuda/common.cuh:2799` `bf16_src1_cache_put` | called from the cuBLAS BF16 path **and** from `hc_combine_norm` — audit both callers for a preceding plain `ggml_cuda_pool_alloc` |
| `ggml/src/ggml-cuda/hyperconn.cu:905` `hc_base_copy_del` | the R13 fix site: verify it is still inside the pool stack and that nothing plain precedes it |
| `ggml/src/ggml-cuda/mmb.cu:1636` `mmb_cache_insert` | callers: `mmb_bf16_activation` and the shadow path |
| `ggml/src/ggml-cuda/mmb.cu:1978` `mmb_slot_reserve` | three call sites: `mmb_dense` (slot 1), `hc_gate_mix` (slot 3), `mul_mat_id_mmb_glu` (slot 2, **fixed**) |

Why it matters beyond the abort: the same violation is **layout/timing sensitive** — one gfx1201 build aborted 5/5
in a *dense 4B* warmup and the next build 0/4.  Use `GGML_CUDA_POOL_TRACE=1` (the switch `e61e354d2` added) to get
stacks; the assert itself always prints sizes/offsets now.  **A sweep is done when each site's callers have been
read, not when a build happens to run.**  Record the audit (site, caller, verdict) in the `TODO` item's record.

### 1.3 `TODO` #297 — the qwen4exp **sparse QSA decode arm has no gate**

`ggml/src/ggml-cuda/indexer-score.cu` called `ggml_rope_get_truncate(dst)` on a `GGML_OP_INDEXER_SCORE` node
(upstream `#30206` asserts the op type), so **every sparse-arm run aborted from R12 to R16** and nothing noticed:
the coherence arms and both qwen4exp gates run the *dense* arm.  The abort was found only because block 15's
residency gate's `pref` arm sets `LLAMA_QSA_DENSE_DECODE_UNTIL=0`.

* **Fix the gap, not the symptom:** add one sparse-arm text arm to `scripts/gate-qwen4exp-quant-coherence.sh` (or a
  small new gate): Flash-Next GSQ, 1 GPU, `LLAMA_QSA_DENSE_DECODE_UNTIL=0`, same-seed greedy `-n 64`, asserting the
  sparse text equals a recorded hash **and** that the fused indexer-score op actually launched (a `-lv 4` census or
  the `GGML_CUDA_MMB_LOG`-style counter — an arm that is unreachable must not be able to pass).
* `GGML_OP_INDEXER_SCORE` has **no `test-backend-ops` case** either (`-o FLASH_ATTN_QSA` covers the attention, not
  the score kernel).  Adding one is optional; the text arm is the requirement.

### 1.4 `TODO` #294 — the gates' kernel-journal window

9 gates end with `check-gpu-faults.sh`; **`gate-grid-stall-pad.sh`, `gate-spec-draft-window.sh` and
`gate-spec-reuse-window.sh` take no `--since`/`--no-faults`**, so a fault from an *earlier* arm in the same session
fails a clean gate (seen twice this campaign).  Fix: capture a timestamp before the first server starts and pass
`--since`, or add `--no-faults` like `gate-state-pipe.sh`.  Cheap; do it before R17 so the ladder's verdicts are
trustworthy.

### 1.5 The decisions still open (`DECISIONS.md`)

| id | what it owes | where |
|---|---|---|
| `D8` | which of block 12's CPU wins survive `#31854`-class rewrites: CPU `pp`/`tg` on a host-expert config, `test-backend-ops` for the touched quant types, the `-lv 4` CPU-path census.  = `TODO` #296 | R13 residue |
| `D9` | the graph cache / views / allocator: `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh` + a `test-backend-ops` pass for `#31584`'s view bounds | R11/R13/R16 |
| `D10` | all-reduce: `gate-prefill-logits.sh --sm tensor --device 0,1,2` with `--ab "GGML_CUDA_ALLREDUCE=hybrid"`, plus the `-lv 4` `RCCL ring` census | R02 |
| `D11` | the spec hidden-state contract (`#30141`/`#31660`): the MTP four-axis protocol | R03/R13/R16 |
| `D7` | Meta compute views (`#31584`-class): a `test-backend-ops` pass on a tensor split | R01 |
| housekeeping | **`D4` and `D6` still say OPEN** although `RUNGS.md` R04/R05 record them as decided/closed — reconcile the status lines (the ledger is authoritative) | R04/R05 |
| `D1` | the MoE cache measurement (`TODO` #291).  **Not a re-base blocker** — it is a feature measurement that predates the climb; leave it unless the maintainer says otherwise | R10 |

### 1.6 Parked, NOT a blocker (recorded so nobody re-opens it by accident)

* **`TODO` #300 — the gfx1151 attendance pool.**  Known-wrong on UMA; the knob now *declines* there, so no
  consumer can reach it.  The fix is owed but the maintainer parked it ("we'll fix it later"); reproducing it takes
  ~5 min (whole-cache-resident `pres` + `LLAMA_KV_POOL_VERIFY=1` on halo; **no** 20-minute prefill needed).
* `TODO` #116 (FP8), #131/#132 (host-expert pool on ≥3-device tensor splits), #133/#134 (arena re-copy; the *next*
  re-base), #270 (batched-bench `-c 16384` segfault at `-ncmoe 48`), #283–#285 (graph-opt streams, ring redirect,
  unified expert store): **pre-existing backlog, out of scope for this climb.**  #270 is the one to watch because
  R17 runs `llama-batched-bench -npl 1,4,8` — pick a context that is known to work, or note it.

## §2 Suggested order

1. **#294** (cheap, makes every later gate verdict trustworthy) → **#298** (audit + fix; the audit may also explain
   a stray abort you meet later) → **#297** (the sparse gate; ~30 lines of shell + one recorded hash).
2. **#301** (the real work: profile or bisect; it is the only *measured* regression and the one that decides whether
   r236 can be released as "no slower").
3. Close the decisions (§1.5) with their measurements — several are just running an existing gate on the final tip.
4. **R17** (§3).

A rung is "done" when it is committed, building and coherent; if an item cannot be closed, roll back to the last
green state, park the finding in this campaign directory and record it in `TODO.md`.

## §3 R17 — the release ladder (LAST task)

`HANDOVER-R16.md` §6 lists it; the deltas from this session are marked **NEW**:

1. `scripts/make-patches.sh` from a canonical fork rebuilt at `release.json.base` via `scripts/apply-all.sh`
   (never from the working chain), then `scripts/make-release.sh --base a691b2345`.
2. `scripts/validate-set.sh` (checksums + fresh-tarball strict apply).
3. `scripts/gate-per-block-build.sh` — all 16 blocks.  Expect **5 blocks to carry the follow-up commits' content**
   (the UMA guard lands in block 09's host-expert area, the pool fixes in block 15, the diagnostic in block 15, the
   indexer-score census in block 11): check which block owns each file before cutting, and put each follow-up's
   hunks in the block that owns the file (`llama-model-loader.*` = block 09; `ggml-cuda.cu` pool = block 10 or 15;
   `mmb.cu` = block 06; `llama-kv-cache.*` pool guard = block 15; `indexer-score.cu` = block 11;
   `ggml-cuda.cu`'s `mul_mat_cublas_impl` = block 10).
4. Coherence on the four arms + the host-expert and hybrid arms.  **NEW:** the `-n 16` arm now emits only a
   `[Start thinking]` preamble on the qwen3.5/3.6/3.8 models — a model-independent boilerplate, so use **`-n 64`**
   for cross-box text comparisons and keep `-n 16` only as the campaign's historical oracle.
5. `scripts/gate-prefill-logits.sh --record` **from a clean unpatched `a691b2345` build**, then compare on the patched
   build; also `--sm tensor --device 0,1,2` and `--kv q8_0` (block 01's all-reduce arm needs the split).
6. MTP four-axis + `llama-batched-bench -npl 1,4,8` (`TODO` #295: also a host-expert **two-slot DFlash** arm, whose
   acceptance axis is read at `-lv 4`; `TODO` #270 warns about `-c 16384` on Flash-Next).
7. Every block gate: `gate-norm-rows-alias`, `gate-hc-fusions`, `gate-host-pool`, `gate-park-idle-slots`,
   `gate-ckpt-reuse`, `gate-state-pipe`, `gate-spec-align`, `gate-spec-width-band`, `gate-kv-residency`,
   `gate-kv-window-split`, `gate-spec-draft-window`, `gate-spec-reuse-window`, `gate-qwen4exp-quant-coherence`,
   `gate-grid-stall-pad` (+ the **NEW** sparse arm from #297).
   **Runtimes on the primary box:** `gate-kv-residency` 25-40 min (window arms ~15 s, pool arms ~5 min each, the
   multi-device section included), `gate-kv-window-split` ~10 min, `gate-state-pipe` ~5 min,
   `gate-park-idle-slots` ~5 min, `gate-ckpt-reuse` ~4 min, `gate-spec-draft-window` ~5 min, the rest minutes.
8. `scripts/check-gpu-faults.sh` over the release session's window (**scope it with `--since`**, see #294).
9. `scripts/fork-refresh.sh --build --gates --push` — re-cut `rdna-boosts` from `.base`, verify tree **and** `.tip`,
   push to the **personal fork only**.
10. Records: `WORKLOG.md` (newest-first, `YYMMDD.VVV`), `release.json`, the current-state headers of
    `BASELINE.md`/`MANIFESTS.md`/`patches/README.md`/`ENVIRONMENT.md`/`GREEDY-PURITY.md`/`TODO.md`, then move this
    campaign to `archive/work/`.  `ENVIRONMENT.md` must **delete** the rows for what R13 removed (`GGML_LF_*`/
    `LLAMA_LF_*`, `LLAMA_STATE_GET_MERGE_GAP`) and **add** this session's: `GGML_CUDA_POOL_TRACE` (done),
    `SELFTEST_STRICT`/`SELFTEST_HOLD_VACUOUS_ARCHS` (gate knobs — document in the gate header, not
    `ENVIRONMENT.md`), and confirm the block-15 knob defaults.
11. Containers: the release images are built **off-host** by `scripts/container-watch.sh` on `radiance` after the tag;
    **do not** run the two-line container build on the primary box during a release.

## §4 Traps this climb paid for (add to `AGENTS.md` only if they generalise)

1. **A header change means a clean build.**  Adding a member to `llama_model_loader` shifted its layout; an
   incremental rebuild produced a binary that aborted in the VMM pool on a *dense 4B* and cost an hour of bisecting
   a phantom regression (the clean rebuild was fine).  `~/bin/build-llama-rocm clean` after touching a widely
   included header; ccache makes it ~6 min.
2. **A pool-order abort is layout/timing sensitive** — 5/5 vs 0/4 on builds that differ only by a diagnostic.  Never
   conclude "it passed once"; use `GGML_CUDA_POOL_TRACE=1` and read the two stacks.
3. **`--lazy-mode off` on a Flash-Next model loads 28 GB of PLE weights into RAM.**  On a 61 GiB box that OOMs and
   takes the session down with it.  Use `--lazy-mode auto` (the default), and on fingon `-ncmoe 44` for a drafter arm.
4. **Never `-ncmoe` on an integrated GPU** (it is now refused, but the message is a WARN: invisible without `-lv 2`).
5. **`pkill -f <pattern>` kills your own ssh session** when the pattern appears in the command line you sent — use
   the bracket trick (`pkill -f "gat[e]-kv-residency"`) or kill by pid.
6. **fingon reboots under load** (3× in one session): build with `JOBS=4`, and expect `/tmp` to be gone afterwards.
   A power cut can also leave a **zero-length branch ref** (`refs/heads/rebase-r236` after `git checkout -B`): delete
   the ref, `git symbolic-ref HEAD refs/heads/master`, `git reset --mixed master`, then re-fetch the bundle.
7. **The journal is shared** (#294): a gate's fault check can trip on a fault from an earlier arm.  Scope it.
8. Everything from `HANDOVER-R16.md` §8 still applies: unregistered graph inputs look like consumer bugs; no plain
   local pool allocation before a `pool_stack` push; `shared_ptr` checkpoints; HIP's `get_n_tensor` is CUDA-only so a
   batched copy through a Meta device is pageable; the `hc_combine_norm` pool-stack rule; coherence hashes are
   per-build.

## §5 Command reference

```bash
# state
cd ~/llama.cpp && git log --oneline -1 && git status --short --branch && git rev-parse HEAD^{tree}
cd ~/llama-cpp-rdna-boosts && git log --oneline -1

# build (primary / halo), and fingon
~/bin/build-llama-rocm            # incremental
~/bin/build-llama-rocm clean      # after a header change or a CMake option change
JOBS=4 EXTRA_CMAKE_FLAGS="-DGGML_HIP_NO_VMM=OFF" ~/bin/build-llama-rocm clean   # fingon

# the pre-re-base comparison binary (any box, same source path)
git checkout rdna-boosts && BUILD_DIR=build-rocm-r235 EXTRA_CMAKE_FLAGS="-DGGML_HIP_NO_VMM=OFF" \
    ~/bin/build-llama-rocm clean; git checkout rebase-r236

# coherence (no -lv; -n 16 is the historical oracle, -n 64 for cross-box text)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli -m <model> -ngl 99 -c 8192 \
  -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt -n 64 --seed 42 --temp 0 \
  --no-display-prompt --single-turn > /tmp/x.log 2>&1
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/x.log

# gates
MULTI_DEVICES=0 ./scripts/gate-kv-residency.sh     # halo (1 GPU); omit on the primary box
./scripts/check-gpu-faults.sh --since "30 min ago"
```

## §6 The delivery repo's own state

`rebase-r236` carries the campaign docs (`RUNGS.md`, `DECISIONS.md`, this file, `HANDOVER*.md`, `CONFLICT-MAP.md`),
the records for the follow-up fixes, and `benchmarks/2026-11-08-rebase-r235-vs-r236.md`.  **`main` is untouched** and
still the shipped r235; R17 switches it.  `patches/`, `release.json` and `rdna-boosts` on the fork are untouched.
