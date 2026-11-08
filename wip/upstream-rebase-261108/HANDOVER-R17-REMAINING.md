# HANDOVER — the five pre-R17 items and R17 itself (r236)

**Read order:** this file → `RUNGS.md` `P1`-`P8` (how the campaign's defects were found and fixed; `P8` is the most
recent) → `../../AGENTS.md` → `../../MANIFESTS.md` gate 22 + `../../scripts/gate-rebase-release.sh` →
`../../TODO.md` for the five rows → `DECISIONS.md` `D1`/`D3`/`D8` → `HANDOVER-REBASE-TODO.md` §3 for R17.

**The rule for this session: everything here is a *measurement that ends in a decision* or a *ladder arm that must
be wired into the gate*. Nothing here is an edit-and-hope. Nothing ships as a new TODO.**

## §0 State (verified at handover, 2026-11-08 evening)

| item | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-r236`**, head **`a4d0d5d0`**, pushed to `origin` (`git@github.com:stew675/llama-cpp-rdna-boosts.git`) |
| fork checkout | `~/llama.cpp`, branch **`rebase-r236`**, tip **`3b0978888`**, tree **`fe53d8a7a`**, clean |
| `master` in the fork | pinned at **`a691b2345`** (the fork point). **Build the attribution base from `git checkout --detach a691b2345`, never from `master`** — a local commit on `master` silently moves what every "unpatched upstream" arm measures and what R17 step 5 records as the KLD reference (that is how `#303` was mis-diagnosed) |
| builds | `build-rocm` = r236 (canonical), `build-rocm-r235` = the shipped set, `build-rocm-base` = unpatched `a691b2345` (**do not delete** — R17 step 5 re-records the KLD base from it) |
| `patches/`, `release.json`, the fork's `rdna-boosts` | **untouched** — cut only at R17 |
| the gate | `scripts/gate-rebase-release.sh` (new, `MANIFESTS.md` gate 22) |

### Closed on 2026-11-08 (do not reopen without new evidence)

| item | outcome | record |
|---|---|---|
| `#303` dense MMQ −1.4 % | **inherited, not ours.** r236's dense `mul_mat_q` is instruction-for-instruction identical to unpatched `a691b2345`'s (1436 instrs, 0 diff lines); the base commit itself costs +1.31 %, r235's old body −0.36 %, the delivery +0.09 %. Six levers tried (min removal, NVFP4 gating, launch_bounds occupancy, J width, the delivery's own dense MMB path, an ids-branch probe) — all null, all at instruction level | `benchmarks/2026-11-08-dense-mmq-303.md`, `WORKLOG` 261108.008, `RUNGS` `P7` |
| `#304` dup-id `f16` `n=9` | **FIXED** (wrong answer, not slow): the compact seam grid used a stride of 16 while the dispatch derived `cols_per_block` from `ncols_dst` clamped to 16, truncating each expert's range below 16. `MMF_IDS_COLS` names the contract; the duplicate cases are **enabled** as the standing oracle. `dup=1` 16/16, `MUL_MAT_ID` 947/947, `MUL_MAT` 2517/2517 | `WORKLOG` 261108.009, `RUNGS` `P8`, fork `3b0978888` |
| `#279` narrow-2 draft compute | closed on the new A/B (its arm, decode, is **+0.22 %**, so "do not implement" still holds) | `WORKLOG` 261108.009 |
| `#290` dispatcher merge debt | **verified** on the tree R17 cuts from: 0 `ggml_cuda_try_fuse_legacy` / `GGML_CUDA_FUSE_LEGACY` refs, one dispatcher TU with a single call site, `gate-norm-rows-alias.sh` PASS, `MUL_MAT_ID` 947/947 | `WORKLOG` 261108.010 |

## §1 The gate — lean on it, don't re-derive

```bash
./scripts/gate-rebase-release.sh                # everything, ~25-40 min, one verdict
./scripts/gate-rebase-release.sh --quick        # skip the per-block build + KLD gate while iterating
./scripts/gate-rebase-release.sh --perf-only    # the interleaved A/B against the shipped build
./scripts/gate-rebase-release.sh --only text,ops
```

Seven steps, each delegating its rule to the gate that owns it: fork/`master` state → clean build per block →
prefill-logit KLD → the `MUL_MAT_ID`/`MUL_MAT` oracles → coherence texts against pinned hashes → **interleaved wall
clock of every arm against `build-rocm-r235`** → the GPU-fault scan scoped to the run's own start time.  It **fails
rather than skips**, `WARN`/`INFO` do not fail it, and the exit code is the verdict.

**Current verdict (3 rounds):** `dense-27B-prefill` **−1.32 %** (−1.37/−1.21/−1.38, spread 0.17) **FAIL** — that is
`#303`, the inherited upstream cost; `moe-35B-prefill` +0.29 %, `moe-35B-decode` +0.22 %, ops 947/947 + 2517/2517,
text `1c5d32ac537d`, no GPU faults, `master` pinned.  **The single red arm is the release decision** (see §8).

**Extend the gate, don't work around it.** `#295` and `#296` below are ladder arms, so their resolution is to become
gate steps (a `server-arms` and a `cpu` step), which is also what stops them from being forgotten again.

## §2 `#295` — a host-expert multi-slot DFlash arm (DO THIS FIRST)

**Why first:** it guards a **silent** failure mode (draft acceptance → ~0 while the text stays correct), and R13
dropped the guard that used to prevent it (`n_seq > 1` + `-ncmoe` → host copies) because the device buffer and the
host copies measured identically.  Nothing else in the release ladder reaches a `-ncmoe` multi-slot DFlash run —
the MTP four-axis protocol is a single sequence.

**The arms** (R13's own numbers to reproduce, 2 slots, `-sm layer`, acceptance read at **`-lv 4`**):

| arm | R13 measurement to match (draft acceptance) |
|---|---|
| Flash-Next GSQ + DFlash2, `-ncmoe 24` | **0.53793 / 0.51678** "both ways" (78/145 and 77/149, mean len 2.59/2.54) |
| Flash-Next GSQ + DFlash2, `-ncmoe 10` | **0.55245 / 0.47134** both ways |
| 35B-A3B Q8_0 + its DFlash2, `-ncmoe 10` | **0.63918 / 0.62887** both ways |

`RUNGS.md` R13 (search `-ncmoe 24` / `0.53793`) has the exact invocation including the drafter attach; **use that
verbatim** rather than re-deriving the flags.  A `llama-server --parallel 2` run is required — a single-sequence
`llama-cli` arm cannot show it.

**Pass criterion:** every arm's acceptance is within noise of its R13 value, i.e. **not ~0**, and the generated text
is unchanged.  Then add the arm as a gate step (`--only server-arms`) with that criterion written into the step, and
record the numbers in the release `WORKLOG` entry.

**Traps:** `-lv 4` for the acceptance axis (a text arm takes **no** `-lv`); pin `HIP_VISIBLE_DEVICES` on every
multi-device arm; the journal is dirty for windows of this campaign (scope every fault check with `--since`).

## §3 `#296` — `D8`'s CPU-win measurement

**Why:** the re-base kept block 12's CPU hunks untouched (`arch-fallback.h`, `arch/x86/quants.c`, `ggml-cpu.c`,
`ggml-cpu.cpp`, `repack.cpp`); upstream rewrote the CPU backend around them (`#31854` only added a `type_traits_cpu`
entry), so *which* wins still pay is unknown.

**Arms:** CPU-side `pp`/`tg` on a host-expert config (a `-ncmoe` arm so the CPU path is actually exercised) with the
CPU backend forced (`--device`/`-ngl 0` as appropriate), `test-backend-ops` for the quant types those files touch
(the k-quant / i-quant rows in `arch/x86/quants.c`), and the delivery's `-lv 4` CPU-path census.  Compare against a
build with block 12's CPU hunks disabled if a kill switch exists; if none does, measure the arm against the r235
build (`build-rocm-r235`) which carries the same hunks, and against unpatched `a691b2345` (`build-rocm-base`) for
attribution — **the same two-fork-point method `#303` used.**

**Pass criterion:** each surviving win is named with a number and recorded in `patches/README.md` (block 12's
section); a hunk with no surviving win is dropped in the same change (and the drop is measured, not assumed).

## §4 `#292` — the dsv4 soft-arm numerics drift

**The fact:** `gate-hc-fusions.sh`'s two REPORTED dsv4 arms reproduce r235's numerics, not the re-based chain's — on
the BF16-hc class the chain gives 449 chars `d62340fbf566` while `dsv4_pre`/`dsv4_post` give 449 chars
`6a102dbc62d6`, and `6a102dbc62d6` is what the **chain** produced at r235.  So the HC chain moved with the re-base
while the fused pre/post ops still reproduce the old numbers.  The gate calls them soft ("reported"), so nothing
fails — but its header says byte-identity is expected on that class.

**Resolution:** quantify it with `scripts/gate-prefill-logits.sh` (**KLD + same-top-p on Flash-Next**), then either
(a) the drift is numerically nil → update the soft-arm expectation with the measured KLD and say why the byte-hash
differs, or (b) it is real → fix `hyperconn.cu` and make the arms hard.  **The decision is the deliverable**; record
it in the release `WORKLOG` entry and in the gate's header.

## §5 `#293` — keep or drop our own qwen4exp (Qwen3.8) model support

**The fact:** R12 (`D3`) kept the delivery's implementation on **+31 % decode with byte-identical text** on
Flash-Next 3-GPU `-sm tensor`, but it is a ~3200-line model file upstream keeps rewriting (the largest per-rung
conflict surface we own).  On **Qwen3.8-27B (1 GPU)** it was throughput-neutral **and its greedy text diverged**
(`b905c9763a58` ours vs `b21e94a5b6e1` upstream) — unexplained.

**The discriminating measurement:** Q8_0 on **2 and 3 GPUs `-sm tensor`** (the field config, where the fused
HC/indexer kernels are supposed to pay), ours vs upstream's, plus `gate-prefill-logits.sh` on the 27B to explain the
text divergence.  **Pass criterion:** if ours does not win on Q8_0 tensor-split, drop the model file and keep
upstream's — and then the fused kernels must be re-expressed as fusions over upstream's plain-op graph (that is a
block change, so it lands in this same pass, not as a new item).

**Purity note:** a text divergence between two *implementations* is not automatically a purity break — establish
whether it is a numeric difference (KLD) or a genuine graph difference before dropping either.

## §6 `#291` — the arena vs upstream's ClassLRU MoE cache (`D1`)

**The numbers we have** (1 × R9700, 35B-A3B Q8_0, `-ncmoe 20`, `-c 8192`, 16.3 GiB host experts, VMM build, text
identical in every arm): delivery arena 4080 MiB **547.6 / 53.5** t/s (pp/tg); upstream `--moe-cache-mib 4096`
**544.2 / 62.9**; streaming **470.9 / 49.0**.  Upstream's cache is **+17.6 % decode** at ~the same footprint, and the
delivery's tier used to be the faster one.

**The open questions:** is it the *policy* (ClassLRU vs the arena's heat/RFU admission), the *tier* (upstream caches
in a plain device buffer; ours lives in the slab, whose reserve/headroom may cost usable arena), the *config* (one
GPU `-sm layer` vs the field's 2-3 GPU `-sm tensor`), or the *measurement* (the arms are either/or — the coexistence
the merge preserves is untested)?  Also: `--moe-cache-cpu-share` (`#30966`) and ClassLRU (`#31233`) are capabilities
ours lacks.

**The measurement set:** one model at two `-ncmoe` levels, `MOE_EXPERT_CACHE_MIB` arms, the `-lv 4` census
(`expert access:`, `takeover=`, cold admission), decode **and** prefill, and an A/B of "arena only" vs "upstream
cache only" vs **both**.  Requires `GGML_HIP_NO_VMM=OFF` (upstream's default is ON, which makes the tier inert) —
`~/bin/build-llama-rocm` already does this.  **Decision:** keep both, keep one, drop one — recorded in
`DECISIONS.md` `D1` and the release `WORKLOG` entry.

## §7 R17 (after all five are resolved)

`HANDOVER-REBASE-TODO.md` §3 is the ladder.  The order that matches this campaign's evidence:

1. `scripts/make-patches.sh` (export the blocks from the canonical tree) → then **confirm `#290` in the re-cut
   block-10 patch** (`git grep -c 'try_fuse_legacy\|FUSE_LEGACY'` = 0) and run `scripts/validate-set.sh`.
2. `scripts/gate-rebase-release.sh` (full) — the verdict line goes into the release `WORKLOG` entry.
3. `scripts/gate-prefill-logits.sh --record` **from a detached `a691b2345` build** (the f16 base already exists; the
   `--kv q8_0` arm is still owed and needs its own base file).
4. `scripts/make-release.sh` (refreshes `release.json`; the `.tip` is what `scripts/fork-refresh.sh` prints after the
   re-cut) → `scripts/validate-set.sh` again.
5. `scripts/fork-refresh.sh --build --gates --push` — the fork's `rdna-boosts` only, never upstream.
6. The ROCm container images are built **off-host** by the watcher (`scripts/container-watch.sh` on `radiance`);
   after the tag nothing else is needed.  Never run the full image build on the primary box.

**Definition of done for r236:** every row in `TODO.md` that is re-base-derived is gone, the gate prints
`VERDICT: RELEASABLE` (the only permitted red is `dense-27B-prefill`, and only with the recorded attribution), and
`validate-set.sh` is green on the re-cut set.

## §8 The one release decision that is the maintainer's

`dense-27B-prefill` is **−1.32 %** against r235 and it is **not fixable from this repo** (`#303`: the kernel is
upstream's byte for byte, six levers null).  Options: (a) ship r236 and carry it as an inherited cost, recorded;
(b) hold the release, which trades 318 upstream commits for 1.3 % on one prefill arm; (c) take it upstream (the
repro is exact and in `benchmarks/2026-11-08-dense-mmq-303.md`).  **Do not quietly raise `--tolerance` to make the
gate green** — if that is the decision, record it in the release `WORKLOG` entry with the attribution.

## §9 Traps this campaign already paid for (they will bite again)

1. **`-lv` is picked by the arm**: no `-lv` for a text/hash arm, `-lv 4` for a census/mechanism arm.  `-lv 3`
   corrupts the text and still hides the ggml censuses.
2. **Pin `HIP_VISIBLE_DEVICES`** on every multi-device arm; the iGPU (gfx1036) is otherwise enumerated and the load
   aborts with `invalid kernel file`.
3. **A single-run `rocprofv3` census has a ±1.1 % floor** (two traces of the *same* binary differ by 1.1 %).  Use 2+
   interleaved rounds; interleave *builds within a round*, not rounds within a build.  Quote the arm wall clock for
   anything that has to be argued.
4. **The ISA is the noise-free arbiter** for "same template instantiation, different behaviour" —
   `wip/upstream-rebase-261108/isa-kernel-diff.py <buildA>/bin/libggml-hip.so* <buildB>/bin/libggml-hip.so* --kernel mul_mat_q`.
   Check it *before* benchmarking a codegen hypothesis.
5. **Never `git apply` the block series**; apply with `scripts/apply-all.sh` (strict `git am`).
6. **Scope every GPU-fault check with `--since`** to the gate's own start; the journal carries this campaign's own
   self-inflicted faults.
7. **`build-rocm-base` must keep existing**, and it must be built from a detached `a691b2345`.
8. **Report findings before committing** and wait for the maintainer's call on anything that changes the delivery.
