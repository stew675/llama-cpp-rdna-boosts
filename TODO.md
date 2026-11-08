# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**How this file is organised.**  The **index table** is the only list: every open, parked or in-flight
handle appears there exactly once, in handle order, and nothing else does.  The `## NNN.` sections that
follow carry the detail, in the same order; a row never states a status its section disagrees with, and a
closed item leaves both (the record is a `WORKLOG.md` entry plus the campaign it points at).  Unnumbered
**accepted limitations** — deliberately not fixed, do not re-report — come last, with the pointers to the
campaign records.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **16-block V2.1 set** (block 00 + blocks 01-15) at fork point **`6050b467c`**
(the 2026-10-25 re-base); read `.release` in `release.json` for the release id -- it is the source of
truth, not this line.  Block 15 is the **streamed KV** (the resident
window *and* the attendance pool, both opt-in).  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).
The r236 re-base (`wip/upstream-rebase-261108/`, base `a691b2345`) is the ladder in flight: its 16 blocks are
climbed, its remaining items are the `#279`/`#286`-`#301` rows below, and `patches/`/`release.json` are cut at R17.

## The open list

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM | performance / new type (`wip/fp8-support/`) | **PARKED** (2026-09-26) — builds on gfx1201, gate 2 fails (0.69x Q8_0) — §116 |
| **131** | `--host-experts pool` prefill collapse under a 3+-device `-sm tensor` split: the staging-path root cause, reachable only through the kill switch | performance (block 12; pre-existing backlog) | **OPEN (residue)** — the collapse itself is fixed (r207, measured), the root cause is not — §131 |
| **132** | the pool's pair decode deficit is the synchronous demand read (~29 % of a decode step) | performance (pre-existing backlog) | **OPEN** (2026-11-03) — re-open the overlap worker the ~6 % ceiling declined — §132 |
| **133** | a fully-resident (identity) expert table is re-copied out of the arena on every prefill ubatch | performance (delivery defect, default mode; backlog) | **OPEN** (2026-11-03) — 3-device prefill is slower than 1-device because of it — §133 |
| **134** | re-base the delivery onto a newer upstream tip (its own release, not a patch) | delivery (release work) | **IN PROGRESS** — this campaign: base `a691b2345` (past #134's `6b9c0cfcb`), ladder climbed, R17 owed — §134 |
| **135** | a `-sm tensor` split across more devices than the model divides evenly is degenerate and nothing refuses it at load | correctness / usability (a load-time refusal) | **OPEN** — the work is the guard, not a perf chase — §135 |
| **136** | `GGML_SCHED_DEVGATHER`'s prefill-band default costs 33 % prefill on 35B-A3B at `MOE_HOST_POOL_FRAC=50` | performance (delivery defect, default-on) | **OPEN** (2026-11-03) — needs a wider sweep before the default moves — §136 |
| **141** | a sub-complement host-expert budget declines the eager take-over at wide decode | performance (block 09, the pool tier) | **OPEN — measured limit** (2026-11-04) — the bound is static per shape on purpose — §141 |
| **142** | the pool tier's first prefill pays ~3 s (it is built and sized lazily) | performance (block 09/12; pre-existing r210) | **OPEN** — the lead is to move the population into the preflight — §142 |
| **270** | `llama-batched-bench -c 16384` segfaults on Flash-Next at `-ncmoe 48`, printing nothing | correctness (pre-existing, r215) | **OPEN** — a silent SIGSEGV is a bad failure mode; R17 must pick a context that works — §270 |
| **273** | three prefill transports where one should do | architecture (delivery) | **OPEN** — the host-set SIZE is the primary lever, residency secondary — §273 |
| **279** | narrow-2 (draft compute) must prefer its `cudaMalloc` fallback over evicting arena tables | performance (delivery, re-validation) | **OPEN on r236** — closed pre-re-base as "do not implement"; the slab/arena moved, so the basis is gone — §279 |
| **282** | a mid-run pinned-tier latch perturbs the logits by ~1e-5 and flips one near-tie token | purity (delivery; NOT corruption) | **OPEN** (2026-11-07) — measured KLD two orders inside the tolerance — §282 |
| **283** | `GGML_CUDA_GRAPH_OPT`'s branch streams collide with the staging copy stream's index | correctness (latent, default off) | **OPEN** — fix when the option is next touched — §283 |
| **284** | the ring's redirect restore should be deferred to `stage_free_ev` | VMEM (delivery) | **OPEN — cheap** — re-measure before believing the 6 % arithmetic — §284 |
| **285** | a unified expert store: one VA per device, expert-granular residency | design direction (not scheduled) | **OPEN** — not a prefill speed-up; six hard constraints recorded — §285 |
| **286** | the Meta compute-container headroom is carried two ways (the 128 multiplier *and* upstream's 256 floor) | correctness / allocation (`ggml-backend-meta.cpp`) | **OPEN** — found at R01 (`D7`); decide at R17 which constant sizes the buffer — §286 |
| **287** | re-cut `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` onto `#31047`'s `tile_spec` grid | upstream candidate, not delivery | **OPEN** — found at R01; a transcription plus a PR-body refresh — §287 |
| **289** | upstream `find_slot()` moves a recurrent cell's state but not its rollback bound | upstream candidate, not delivery | **OPEN** — fixed in the delivery at R04; upstream has the omission — §289 |
| **290** | the fusion dispatcher exists twice while upstream's arms are reconciled | delivery (`ggml-cuda-try-fuse.cu`, `ggml-cuda.cu`) | **OPEN until verified in the re-cut patch at R17 step 1** — resolved at R11, "closed" only once r236 ships it — §290 |
| **291** | upstream's ClassLRU MoE cache vs the delivery's host-expert arena at equal MiB | delivery vs upstream (`D1`) | **OPEN — deferred to its own session** (maintainer, 2026-11-08) — a measurement, not a code change — §291 |
| **292** | `gate-hc-fusions.sh`'s dsv4 soft arms reproduce r235's numerics, not the unfused chain's | gate expectation / `hyperconn.cu` | **OPEN** — created at R12; quantify with the KLD gate at R17 — §292 |
| **293** | decide whether to keep carrying the delivery's own qwen4exp (Qwen3.8) model support | model file (`src/models/qwen4exp.cpp`) + the fused HC/indexer kernels | **OPEN** — kept at R12 on a measured +31 % decode; the discriminating arm is Q8_0 on 2/3 devices — §293 |
| **295** | R17 must include a host-expert multi-slot DFlash arm (the detector for the guard R13 dropped) | ladder arm (R17) | **OPEN — R17** — the failure mode it catches is silent — §295 |
| **296** | `D8`'s CPU-win measurement is still owed | ladder arm (R17) | **OPEN — R17** — `pp`/`tg` on a host-expert config + `test-backend-ops` + the CPU-path census — §296 |
| **298** | the VMM pool's free-order invariant: one site fixed, the class is worth a sweep | correctness (pool ordering; layout/timing sensitive) | **OPEN** — audit each `pool_stack_push` site's callers; `GGML_CUDA_POOL_TRACE=1` gives the stacks — §298 |
| **300** | gfx1151 / RDNA3.5: the attendance pool (`LLAMA_KV_REGION_POOL`) is INCORRECT on a UMA device | correctness (RDNA3.5) | **PARKED** — the pool now declines on an integrated GPU, so nothing can reach it; the fix is owed — §300 |
| **301** | r236 is -2 .. -3 % on MoE **prefill** against the shipped r235 (decode flat) | performance (the only measured regression) | **OPEN** — it decides whether r236 may ship as "no slower" on the MoE shapes — §301 |

## Handle numbering

**Renumbered 2026-10-31 (docs-only).**  The r187-r189 sessions allocated `#190`-`#196` (and `#218`) as if the
sequence had advanced past a run of closed items, but `#119` was the last handle ever allocated and `#120`-`#189`
were never used.  They are **`#120`-`#126`** now (plus `#127` for the `cpy_2d_vec16` limitation), so the sequence
is contiguous.  The r187-r190 **commit messages** (immutable) still cite the old handles; they mean these rows.
The old numbers are **retired** — do not reuse them.

> **Handle numbering (corrected 2026-11-03, sequence re-checked 2026-11-04):** the real sequence runs to **#130**
> (closed, issue #263), so this file's `#268`/`#269` — allocated by the 2026-11-02 session — are themselves out of
> sequence, the same failure the 2026-10-31 renumbering note above describes.  **#131`-`#143 are allocated** (the
> 2026-11-03/04 items; `#143` closed in r214; `#270` is the r215 finding), so **new items continue from `#273`** (`#271`/`#272` closed in r234).  Renumbering `#268`/`#269` would also touch
> `WORKLOG.md` 261102.* and the campaign records that cite them, so it is left as a decision for the maintainer
> rather than done here.

---

## 116. Native FP8 E4M3 for RDNA4 — PARKED behind a GEMM regression (campaign `wip/fp8-support/`)

**Status: parked (2026-09-26).  Not scheduled, not in `patches/`, and nothing may be applied from it without an
explicit go-ahead (`AGENTS.md`, WIP rules).  The campaign is `wip/fp8-support/` (`README.md`, `PLAN.md`,
`HANDOVER.md`, `RESULTS-2026-09-26-rebase-gate.md`).**

**Why it exists.**  Single-request prefill is **70-81 % `mul_mat_q`** (sibling campaign
`archive/work/prefill-gap-attribution/`), so the vLLM gap is a GEMM problem, and the int8 MMQ is
**epilogue-bound**: every per-16 quantization pays a scaled dequant epilogue in the MMA path (~57 TFLOPS
against Q8_0's 83 and F16's 107 — `archive/work/per16-f16-mma/`).  FP8 tensor cores need no int8 dequant epilogue, gfx1201's fp8 WMMA peak is
~171-370 TFLOP/s-class against the int8 MMQ's ~41.5 T-MAC/s, llama.cpp has **no FP8 weight type at all** (so
today's "75 % of vLLM" comparison is engine *and* precision), and the port already measured **+13-17 % prefill
over Q8_0** on a 4B / gfx1201 on the maintainer's `cllm` branch (last touched 2026-08-06, base `6ea215d17`).

**What exists.**  `patches/` = the original 23-commit `cllm` series (provenance; `~/cllm` tip `7c17faffc` is
one unpushed commit ahead of `origin/cllm` and is the source of truth), `patches-rebased/` = the delivered
re-base: **13 commits, 37 files, +5374/-12**, on the r9 base, with the 10 superseded commits (chunked GDN,
ssm-conv, perf docs) dropped because block 02 ships a newer and different GDN.  The port is
`GGML_TYPE_F8_E4M3` (43, self-contained `[f32 scale][128 fp8]` blocks), `ggml-cuda/fp8.cu` (`mul_mat_fp8_wmma`:
CTA 128x64, 8 warps, `GROUP_M=4` pid swizzle, 16x16x128 tiles, register-staged, repacked weights; plus
`quantize_fp8` activation staging and a `dot4` GEMV for decode), direct HF safetensors loading, a
`convert_hf_to_gguf.py --outtype fp8_e4m3` path, a CPU quant + `GET_ROWS` (bit-exact vs CPU), and RDNA4 gating
(compile-time `RDNA4` + runtime `CC_IS_RDNA4`, scalar fallback otherwise).

**Where it stopped (gate 2 fails).**  4B, `pp512`, gfx1201, same box: Q8_0 **8146 ± 221** (the `cllm`-era
figure was 6177, i.e. the delivery's MMB/GEMM work is +32 %) vs `F8_E4M3` **5607 ± 15** (the `cllm` figure was
7184-7260) — **0.69x** where it used to be **1.16x**.  The machine is not the problem, the fp8 path is: the
kernel runs ~65 TFLOP/s here against the ~77 the old notes recorded **on the same source**, and
`fp8_repack_weights` takes **7.3 %** because it appears to run **per call** instead of once (it is meant to be
a lazy one-time layout change cached on the context — check `fp8_backend_cuda_context::fp8_repack_buf`
invalidation on the newer base).  GDN/ssm is 3 % and attention 1.3 %, so this is GEMM + repack + staging, not
the dropped commits.

**Next actions** (`PLAN.md`, in order): (1) interleaved A/B of `mul_mat_fp8_wmma` on the re-based tree against
`~/cllm` at the same `pp512` to isolate codegen vs launch configuration; (2) fix/confirm the repack caching
(target its 7.3 % to ~0); (3) re-run the 4B gate (fp8 must beat Q8_0 again); (4) only then convert
`/llm/models/Qwen3.8/27B/FP8/` and measure `pp8192` against the int8 baselines (Q8_0 1371 / Q6_K 975 /
Q4_K_XL 1254 t/s), which is the number that decides the campaign; (5) lift AITER's gfx1201
`A8W8_BLOCKSCALE` tiles / `GROUP_SIZE_M` / `kpack` (target `mul_mat_fp8_wmma` ≥ 110 TFLOP/s effective; AITER's
Triton reaches 121-137 against our 77-98).

**Risks / constraints.**  The **conversion is the correctness cliff**: HF fp8 (per-tensor/per-channel weight
scales, dynamic per-token activation) → the 128-block `block_f8_e4m3` grid, and a wrong scale mapping produces
plausible-but-wrong text — so the oracle is PPL against the vLLM fp8 record (fp8 6.2250 vs Q8_0 6.2464) plus the
same-seed greedy gate, never "it runs" (there is **no CPU oracle** for an fp8 `MUL_MAT`).  FP8 is a different
numerics path, so it can never be bit-identical to int8 and the purity contract is a PPL/oracle comparison.  A
**non-RDNA4 policy** (reject vs dequantize rather than a silent scalar fallback) has to be settled before any
promotion.  "Dequantize to bf16 + hipBLAS" is a **trap** — upstream already measured that on RDNA4 MMQ beats
dequant+hipBLAS (`ggml-cuda/mmq.cu:608`, PR #18537) — so the win must come from the native WMMA GEMM.  And the
4B result may not scale (the 27B has different shapes, a bigger lm_head and more attention).  Promotion shape,
if it ever gets there: a new weight type + kernels + loader + converter is a **large block** and a maintainer
call.

**The index row's detail (kept here so the table can stay a dashboard):**

native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM (campaign `wip/fp8-support/`)

**Kind:** performance / new type.

**PARKED** (2026-09-26).  Re-based onto the delivery as 13 commits (`~/llama-fp8`, branch `fp8-rebase`), builds clean on gfx1201, fully RDNA4-gated — but **gate 2 fails**: on a 4B the fp8 arm is **0.69x Q8_0** where the pre-rebase `cllm` branch had **1.16x**, because the delivery's MMB/GEMM work lifted Q8_0 **+32 %** (8146 vs 6177 t/s pp512) and the fp8 WMMA path did not follow (5607 vs 7184-7260).  Kernel trace: `mul_mat_fp8_wmma` 71.9 %, **`fp8_repack_weights` 7.3 % (running per call instead of once — the likely bug)**, staging 3.3 %, FA 1.3 %.  Not delivery work: nothing here is in `patches/`.  Resume: A/B the GEMM against `~/cllm`, fix the repack cache, re-run the 4B gate, only then convert and measure the 27B.  Detail in §116

The index row also recorded the r209-era kernel trace split: `mul_mat_fp8_wmma` **71.9 %**, `fp8_repack_weights` **7.3 %**, staging **3.3 %**, FA **1.3 %**.

## 131. `--host-experts pool` prefill collapses under a ≥3-device `-sm tensor` split (delivery defect)

**Status: FIXED AND SHIPPED in block 12 (amended, r207, 2026-11-03) — `--host-experts pool`
now gathers a pool-backed table in the prefill band by default, measured 1502.6-1569.2 t/s prefill against
`pinned`'s 471.8 at three devices with decode at -0.7 %, and 16.9 t/s with the gather switched off
(`GGML_SCHED_DEVGATHER=0`).  What remains OPEN is the *root cause inside the staging path*: it is still
pathologically slow for a pool-backed table at >= 3 devices and is now reachable only through the kill switch, so
either fix it or delete that branch for pool tables.  **Verification of the fix (2026-11-03):** the prefill-logit
KLD gate (`--ab "GGML_SCHED_DEVGATHER=0"`) reads **−0.000000 / 100.000 %** (bit-identical prefill), text is
byte-identical (5835 chars `dc178c29fe89`), and the rate is physically bounded (in-place pinned reads, 19 GB/s per
device vs a 28.6 GB/s calibration).  Two caveats worth carrying: `GGML_MOE_GATHER_ONCE=1` no longer produces the
Hole-B corruption at this width/quant (the control is vacuous — do not cite it), and an earlier `pinned` 75.2 t/s reading **did not reproduce** (three reruns: 394.1 / 425.7 / 425.2 t/s), so the
arm is stable; any "N x faster than pinned" prefill claim still needs a pinned control from the same group.  The
`llama-bench pp1024` disagreement (pool 130 ± 73 vs pinned 271 ± 2.5) is open and needs a `-r 3` repeat at
`-ub 1024`.  The `ENVIRONMENT.md` row for the three-valued `GGML_SCHED_DEVGATHER` shipped with it.**  Original finding text follows.

**Found 2026-11-03 on `soar` (`archive/work/host-pool-speed/` README §18.2, session 4), a defect in the DELIVERY's block 12
code, not in a campaign patch.**  Measured, 122B-A10B `-ncmoe 48`, `-sm tensor`, 5258-token prompt, `HIP_VISIBLE_DEVICES`
pinned to 3 R9700s: `--host-experts pool` prefills at **16.85 t/s** against `pinned`'s **564.65** (and 1126.8 with
`GGML_SCHED_DEVGATHER=1`); the same arms at 2 devices are 1024.1 (pool) vs 1603.8 (pinned), i.e. healthy.  The
defect is in the pool branch of `moe_cache_stage_table` (`!t.host_pinned`): the `pinned` arm stages the same volume
through the same kernel at full speed.  Probes that did NOT move it (all at 3 devices): `MOE_HOST_POOL_STAGE_BATCH=0`
(31.99), `MOE_HOST_POOL_DIO=0` buffered (**8.12**), `FRAC 95` (15.03, so it is not the uncovered-expert loop),
`FRAC 30` (206.30 — i.e. it scales with the PINNED POOL SIZE, 19.9 → 53.5 → 63.7 GiB gives 206 → 16.85 → 15.03).
Two live candidates: the box is over-committed (73 GiB mapping + 54-64 GiB pinned on 91 GiB RAM) so the staging's
pageable reads re-read from disk; or the peer-device reads of the pool alias (`pool->base_dev`) are slow and grow
with the pool's coverage.  **Diagnosis so far (2026-11-03, instrumented — the staging COPY is exonerated):** per-call timing of
`moe_cache_stage_table` gives **1.04 s of host time for 423 calls** (miss loop 0.76 s) and **0.66 s** for the
staged copies with a forced copy-stream sync; `rocprofv3` total kernel time is **5.04 s (pool) vs 4.08 s
(pinned)** at the same prompt with matching per-kernel dispatch counts (except `__amd_rocclr_fillBufferUnAligned`
3574 vs 1846 ≈ 4 extra allocation/fills per staging call); the host burns 2.3x the task-clock (83.2 vs 35.8 s);
and `MOE_EXPERT_CACHE_MIB=0` (no arena) recovers 4.3x (50.4 vs 11.8 t/s) while `GGML_MOE_CACHE_STAGE=0` does not
(16.1).  So the cost is the cache's **per-(table, device) prefill work around** the staging copy — the prime
suspect is the per-table `cudaMalloc(t.prs_dev)`/`cudaFree` pair (432 of them, 144 tensors × 3 devices) issued
*inside* the prefill; the fix, if so, is to allocate that 1 KiB map once at registration or use a per-device
scratch.  Next measurement: time that allocation (instrumented and queued) and the per-split scheduler time; do
not re-measure the prefill total, it is bracketed.  Workaround until
then: `GGML_SCHED_DEVGATHER=1` (pool-only use; on `pinned` the gather costs -47 % prefill at 2 devices).  Gates for
a fix: same-seed greedy text on 2 and 3 devices, `check-gpu-faults.sh`, and the prefill at ≥ `pinned`.

## 132. The pool's decode deficit on a tensor split is the synchronous demand read — re-open the overlap candidate

**Status: open, measured 2026-11-03 (`archive/work/host-pool-speed/` README §18.4).**  On the 2-GPU x8 pair at `FRAC 80` the
pool decodes 26.08 against `pinned`'s 39.72 (-34 %), and the fill accounting says **86 % of that gap is one
synchronous `O_DIRECT` read per uncovered expert per decode step**: 20.7 demand fills/token at ~0.55 ms = ~11.3 ms
of a 38.3 ms step (29 %), against 5.1 fills/token = ~5.2 ms of a 56 ms step (9 %) on one GPU — which is why §16.3
correctly declined an overlap worker at a ~6 % ceiling there.  On a multi-device split the miss set is per device
and the fill stays on one host thread, so the ceiling is ~25-30 % (and ~0 at full coverage / the 3-GPU shape where
the arena holds the whole set: 49.48-51.03 vs 51.97-52.04, i.e. **-1.8 %**, inside PR #267's own envelope).  The
tensor-split remap that this note used to rank against landed in r221 (`#269`, +16.1 % / +28.4 % at 2 / 3 devices),
so the demand read is what is left of the pair's decode deficit -- and it is a different cost (a fill, not a sync),
so the two do not overlap.  Protocol: re-screen with
`scripts/moe-trace-replay.py` first (`archive/work/host-pool-speed/README.md` §9.6 — an overlap does not change placement, so the
model cannot score it directly; use it to size the residual demand), then build the worker behind a kill switch and
gate it on the pair at `-ncmoe 48`, `FRAC` 80, with text + MTP acceptance + faults.

**The index row's detail (kept here so the table can stay a dashboard):**

The pool's pair decode deficit is the synchronous demand read (~11.3 ms/token = 29 % of the step); re-open the overlap worker the 1-GPU ~6 % ceiling declined (`archive/work/host-pool-speed/README.md` §16.3)

**Kind:** performance.

**OPEN** (2026-11-03, `archive/work/host-pool-speed/README.md` §18.4).  See §132

## 133. A fully-resident (identity) expert table is re-copied out of the arena on every prefill ubatch

**Status: open, measured 2026-11-03 on `soar` (`archive/work/host-pool-speed/` README §18.2).  Delivery performance defect in
the DEFAULT (`pinned`) mode at a large-arena configuration — not a pool issue.**  At `-ncmoe 48` on three RDNA4
GPUs the arena is `67110.0 MiB of 67110.0` (256/256 slots/table, i.e. the whole host expert set is *also* fully
VRAM-resident), and the prefill nevertheless stages **61 GiB for a 3049-token prompt — `arena=61031.5 MiB
host=510.5 MiB`, i.e. 99.2 % of it read back out of VRAM** into the op's per-split weight buffer, once per table
per ubatch, serialised against the compute (copy table → compute → copy next).  It runs at ~8 GB/s aggregate
(2.6 GB/s per device) over a 7.7 s prefill, so it is latency/occupancy-bound, not bandwidth-bound.  Symptoms:
`pinned` prefill at 3 GPUs (471.8-606.9 t/s) is **slower than at 1 GPU** (755.5) despite 3x the VRAM and 3x the
compute; the no-copy path (the pool gather, which resolves from the arena) scales normally (734 → 1300-1600).
Forcing the gather on `pinned` does not help (587.8 vs 564.7) because a pinned table's gather source is the pinned
*host* master over PCIe.

**Split-mode dependence (measured 2026-11-03):** the redundant re-copy happens under **both** `-sm tensor` and
`-sm layer` (both size the arena to 100 % of the set here), but its cost tracks the **number of staging calls**:
tensor split issues one per table *per device* (396-846 per prefill) and prefills at 394.1-425.7 t/s, layer split
issues one per table (244) and prefills at **944.3** — same bytes, 2.3x apart.  So the item is worst under
`-sm tensor`, and the pool's prefill advantage over `pinned` (2.7-3.7x there) shrinks to +6 % under `-sm layer`
(1000.7 vs 944.3).

**Candidate fix:** let an identity (fully resident) table serve the prefill op **in place** — the take-over /
aliasing path `moe_cache_get_table()` already implements for the decode band, which the scheduler does not use
above the band (`offload_op` is band-gated).  Needs: the take-over decision above the band for identity tables
only, a graph-capture audit (the band gate exists because capture bakes the redirect in), and the gates the
delivery requires for a pref/prefill change — `gate-prefill-logits.sh` must read KLD 0.000000 (the arms are the
same bytes, so anything else is a bug), same-seed greedy text, `check-gpu-faults.sh`.  **Value:** up to ~3x on the
default mode's prefill, which is also the baseline every pool claim is divided by.

**The index row's detail (kept here so the table can stay a dashboard):**

A fully-resident (identity) expert table is re-copied out of the arena on every prefill ubatch — 3-GPU `pinned` prefill is *slower* than 1-GPU (471-607 vs 755.5 t/s) at 100 % arena residency

**Kind:** performance (delivery defect, default mode).

**OPEN** (2026-11-03, `archive/work/host-pool-speed/README.md` §18.2).  See §133

## 134. Re-base the delivery onto upstream `6b9c0cfcb` — 207 commits, block 00 conflicts immediately

**Status: open, assessed 2026-11-03 (`WORKLOG.md` 261103.003).  Deferred out of the promotion session by design: it
is its own release.**  `6050b467c..upstream/master` = **207 commits**; `git am -3` of the delivery onto the new tip
fails at **block 00** (`ggml/src/ggml-cuda/mmq.cu`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`,
`src/llama-model.cpp`, `src/llama-vocab.cpp`).  Two upstream commits overlap this campaign's mechanisms and need a
semantic review, not just a textual merge: **`f5222f25c` "cuda: fuse MoE (MUL_MAT_ID) gate/up and GLU in MMQ"**
(the delivery's MoE/MUL_MAT_ID area) and **`740251a92` "cuda: gather batched tensor reads into mapped host memory"**
(the campaign's gather / mapped-host-memory area); also `8fc037672` (Q8_1 correction sum) and `1b6935b22`
(flash_attn_ext_rows, block 05's area).  Re-base plan: re-cut every block at the new base, per-block build ladder,
re-record the prefill-logit base on the new base, re-measure the headline arms (a re-base is a decode *and* prefill
change — `AGENTS.md`), then re-apply block 12's r207 amendment on top.

## 135. A `-sm tensor` split across more devices than the model divides evenly is degenerate, and nothing refuses it at load

A `-sm tensor` split across more devices than the model can divide evenly is **degenerate**, and nothing stops it at load: 35B-A3B has no even split above 2 GPUs, so a 3-device arm runs at ~20 t/s against ~55 (1 GPU) / ~52 (2 GPUs) — the third device *halves* throughput.  The item is now a **load-time guard**, not a perf investigation: detect an unsupported device count for the model and refuse it (vLLM-style) instead of silently running degenerate

**Kind:** correctness / usability (delivery — a load-time refusal).

**OPEN — limitation understood, the work is the GUARD** (maintainer, 2026-11-04).  The cause is **divisibility**: 35B-A3B has no even `-sm tensor` split above 2 devices, which is the same class as `gemma-4-E4B-it` + 3 devices in the accepted-limitations list (2 KV heads < 3 devices aborts in the split-state assert).  So the ~20 t/s is the *symptom* of a zero/uneven per-device share, **not** a host-expert performance bug — do not chase it as one.  What is needed: a **load-time check** that decides whether the requested device count can split this model's tensors evenly and, when it cannot, **refuses to offload that far** with a message naming the constraint and the supported counts, as vLLM's TP validation does (detect it from the model's own geometry — heads/expert counts — never a hard-coded model list).  Evidence: r209, 35B-A3B Q4_K_M `-ncmoe 40`, gfx1201 x3, `archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §7 — all-VRAM on the same 3 devices gives 138 t/s, so the cliff is the split geometry, not the device count

## 136. `GGML_SCHED_DEVGATHER`'s prefill-band default costs 33 % prefill at a half-covered pool

`GGML_SCHED_DEVGATHER`'s prefill-band default **costs 33 % prefill** on 35B-A3B at `MOE_HOST_POOL_FRAC=50` (422 -> 282 t/s at 3 devices, 5246-token prompt, two reps within 1 %), where r207 chose that default from a 122B/FRAC-95 measurement

**Kind:** performance (delivery defect, default-on).

**OPEN** (2026-11-03, `archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §4).  Needs a wider sweep (model x FRAC x ubatch) before touching the default — and note the same switch must be `0` for any prefill A/B of the expert fill path, because the gather serves the routed table itself

## 141. A sub-complement host-expert budget declines the eager take-over at wide decode

A **genuine sub-complement** host-expert budget (`-lme 8` = 43 % of the set, or any `MOE_HOST_POOL_FRAC` below 100) declines the eager take-over at wide decode: `llama-batched-bench -npl 8` gives **50.4 tg**, i.e. worse than the cache being off (129.2), because `slots + pool coverage < n_used * n_tok` cannot guarantee a complete remap and the run falls back to the pageable mapping

**Kind:** performance (block 09, the pool tier).

**OPEN -- measured limit** (2026-11-04, `WORKLOG.md` 261104.002 §4).  The bound is static per shape on purpose (a per-token decision cannot be baked into a captured graph), so the honest answers are a bigger arena, the pinned tier, or the retired `MOE_HOST_POOL_SIZING`/coverage work.  Candidates worth scoring offline with `scripts/moe-trace-replay.py` before any code: (a) require the arena to hold `n_used * band` slots for a pool-backed table (size it, do not decline it), (b) serve a neither-tier expert from the bounce ring *inside* the take-over instead of declining the table (needs the ring to be a legal cold source for one token), (c) accept the variance and let the graph re-capture.  Nothing is in flight; the r211 fix covers only the covering case

## 142. The pool tier's **first** prefill pays ~3 s before an arena exists

The pool tier's **first** prefill pays ~3 s: `llama-batched-bench -npp 512 -npl 1` measures **pp 128** for the pool against **588** for a pinned master (3.97 s vs 0.87 s), and the same 3 s is amortised invisibly inside a 5246-token prompt (prefill 1318 t/s, fine).  The pool is built and prepopulated lazily and the arena is sized only on the first DECODE-band pass, so the first prompt batch runs the pre-sizing staging path with no arena at all

**Kind:** performance (block 09/12, `moe-expert-cache`).

**OPEN -- pre-existing (r210), not a r211 regression** (2026-11-04, `WORKLOG.md` 261104.002 §4).  `MOE_HOST_POOL_PREWARM=0` does not move it (122 vs 130), so it is the build/first-touch, not the prewarm order.  The lead from `archive/work/host-expert-eager-push/P6-POOL-REGION.md` §7d is to move the population into the preflight (before the first timed batch) -- it was scoped as worth only ~0.09 s at `-n 300`, but this row is the arm where it is worth 3 s

## 270. `llama-batched-bench -c 16384` segfaults on Flash-Next at `-ncmoe 48` (both arms, no output)

**Status: open, found 2026-11-04 (r215).  Pre-existing, not that change — recorded so it is not re-discovered as one.**
`llama-batched-bench` on Qwen3.8-Flash-Next GSQ-RCO-IQ3_S (`-ngl 999 -ncmoe 48 -sm tensor -fa 1 -ctk/ctv q8_0
-ot 'per_layer_token_embd.*=CPU' -lm mmap`) exits with SIGSEGV (139) and **prints nothing at all** — not even the
model-load banner — for `-c 16384` at `-npp 512 -ntg 128 -npl 1` (and `-npl 2/4/8`, and `-npp 64 -ntg 64 -npl 8`).
`-c 8192` runs those same shapes normally (`-npp 16 -ntg 32 -npl 1,4,8` and `-npp 16 -ntg 32 -npl 8` both produce
the table).  Reproduced with `MOE_HOST_POOL_ALLOC=portable MOE_HOST_POOL_BULK=0`, i.e. the r214 code path, so it is
the workload shape / VRAM cliff on this model, not the r215 pool change.  Worth chasing only because a silent
SIGSEGV with no message is a bad failure mode (the archive's warning that this instrument has been unreliable
before applies: `archive/work/host-expert-eager-push/HANDOVER-R210.md` §3).  Start from `-c 16384 -npp 16` and
bisect the context, then check whether it is the compute buffer, the arena or the `-npp 512` prefill shape.

## 273. Three prefill transports where one should do

**Three prefill transports where one should do**: a host-resident expert table's prefill is served by the device gather, the staged fill or the scheduler's bulk upload, selected by `GGML_SCHED_DEVGATHER` x `GGML_MOE_CACHE_STAGE` and a tier test — each wins somewhere, so no single default is right.  Measured on r234: bulk wins at ≤ 67 % arena residency (+16 % at 44 %, +6 % at 67 %, 1 and 2 devices, both box phases), the staged fill wins at ≥ 83 % (up to +10 %; at 99 % it sends 0.0 % of the bytes over PCIe), on the pool tier the staged fill is mandatory (the alternative is the per-expert pinned bounce, 5-10x slower), and the gather is the pool default that wins at `-ub` ≤ 4096.  The trade is **bytes saved vs transactions issued**, and the resident set is scattered so the complement cannot be coalesced

**Kind:** architecture / delivery — a transport consolidation.

**OPEN — direction refined: the host-set SIZE is the primary lever** (bulk's prefill traffic is the host-set bytes per ubatch: measured 1872 -> 1989 t/s, +6.3 %, from `-ncmoe 48` to `36` — every transport scales with it), arena residency is secondary (only a transport that *reads* the duplicate can use it, which is why bulk measured flat across residency), and transaction count is third (the SDMA phase).  **Leading candidate for one transport**: a *contiguous* static VRAM tier (range chosen by a heat-maximising sliding window) + a **two-transaction fill** — one bulk DMA of the contiguous complement and one D2D overlay of the resident range — which has bulk's transaction count and staging's byte saving, and degenerates to plain bulk at 0 % residency.  It needs the freeze-vs-slab problem from `wip/moe-static-hot-tier/` solved and its decode hit-rate cost measured (a heat-*seeded* static set already costs ~1.8 points, r74).  Earlier framing, kept for the bulk default: consolidate on BULK, and the ≥ 83 %-residency cells are **not a target**: they are misconfigurations (high residency means the arena was sized from VRAM the user did not need to offload — `min(1, (free − reserve)/host set)` — so the right answer there is fewer `-ncmoe` layers or none, not a faster transport; and `-ncmoe` is the wrong lever to lower residency, capping the arena is).  What remains is an **arena-sizing policy** (decode hits vs prefill bytes), not a transport rule.  Rationale: simplest, most stable, already wins at ≤ 67 % on 1 and 2 deviceswith its high-residency hole closed by a **sizing** decision (a decode-sized arena — the arena exists for the decode band and its auto-size, which eats all free VRAM, is the only reason residency reaches ≥ 83 %) rather than by an in-place/remap read (the maintainer does not want to re-enter that; it is also the direction the archive refuted for the pool tier).  Parked experiment + all numbers: [`wip/expert-transfer-engine/`](wip/expert-transfer-engine/README.md) (`0001-no-pinned-staging.patch`, gated 41/41, text byte-identical, +6..16 % at low residency, **-4.6 %** on a 2-device 99 %-residency tensor split = the hole to close first).  Steps 1-2 there are the next work: does a capped arena cost decode?

## 279. narrow-2 (draft compute) must prefer its `cudaMalloc` fallback over evicting arena tables (r236 re-validation)

**Status: OPEN on r236** — it was closed on the pre-re-base tree, and the re-base moved the measurement basis
(see the last paragraph).  The pre-re-base result, kept for the reason it will be re-measured:

Measured **CLOSED 2026-11-07 as "do not implement"** on the pre-re-base tree
(`archive/work/host-expert-parity/PLAN.md` P1.5, `PR-267-REMAINDER.md` item 5): when the draft's narrow-2 compute
region fits *outside* the slab, taking the `cudaMalloc` fallback instead of evicting arena tables measured no better
than the eviction, so the branch was not taken.

The re-base moved the slab and the arena (R10/R11 re-derived the arena, R13 fixed the `hc_combine_norm` pool
ordering, R16 fixed the `mmb_glu` pool-ordering site and added `GGML_CUDA_POOL_TRACE`), so the *measurement basis*
is gone.  **Closing it on r236 means one A/B of the branch against `GGML_CUDA_SLAB_MIN_ARENA_MIB`** on the field
config (35B-A3B or Flash-Next, `-ncmoe`, `-sm layer`), with the arena table count / residency census in the record —
either it still measures neutral (close it again, with the r236 number) or the re-based slab changed the answer and
it ships.

## 282. A mid-run pinned-tier latch perturbs the logits by ~1e-5 (a path difference, not corruption)

**A mid-run pinned-tier latch (`MOE_HOST_POOL_PIN_FAULT=late`) perturbs the logits by ~1e-5, which flips one near-tie token in a 32-token greedy run**: the arm declines a MIXTURE of ops (some cache-served, some stood down) and that mixture reproduces neither `ref` nor the wholesale-stand-down arms, so a box that latches mid-run (the Windows aperture case in #278) does not answer identically to one that does not

**Kind:** purity (delivery — `gate-host-pool.sh` `pinlate` arm; NOT corruption).

**OPEN — found 2026-11-07 on `soar` while gating issue #281.**  `scripts/gate-host-pool.sh` is **36/37 on a clean r231 checkout** (`a19aab55e`) with this one check red: `pinlate` 171 chars `sha=b981e742aa13` against `ref` 182 / `a99ccbd97ecb`, deterministic across two builds and two runs, a one-token divergence at ~31 of 32 greedy tokens.  **Narrowed, four arms, all byte-identical controls:** the cold in-place read is NOT the cause (`MOE_EXPERT_CACHE_COLD_UVA=0` in both arms still diverges); the compute path is NOT the cause (`fail`/`late` stand every table down — `get_ok=0 decline_all=4929` — and still match `ref`, so the fallback computes the same logits); the decode device remap is NOT the cause (`GGML_MOE_DEVREMAP=0` still diverges).  What is left is the mixture itself (`pinlate`: `get_ok=552 decline_all=1711`, 119 rolled-back fills), i.e. a table serving some ops from the arena and others from the fallback inside one run.  `gate-host-pool.sh` now proves the fallback's byte-identity in the same run and reports this as a WARN instead of passing it blindly (a wholesale arm that diverges still FAILS).  **Next:** one instrumented run (the per-table `hits/misses/fills` line) to name the table whose op differs, and check the prefill-staging snapshot (`gprs_dev`/`prs_dev`) that is built before a rolled-back admission — the r219 class one layer deeper.  **MAGNITUDE MEASURED (2026-11-07, same session):** the delivery's own criterion on the gate's config (`llama-perplexity -c 512 --chunks 2 --kl-divergence --kl-divergence-base`) reads **median KLD -0.000000, 99.9 % -0.000040 worst, Same top p 100.000 %** -- ~1e-5, two orders of magnitude inside the 0.005 tolerance. So the mixture is a *path* difference (a ~1e-5 arithmetic difference, most likely a different kernel instantiation for the ops that fall back), **not wrong bytes**: `MOE_HOST_POOL_VERIFY=1` reports 0 pool-read mismatches in the latched arm, and a demoted-slot drain-before-serve diagnostic changed nothing.  The byte-identical *text* assertion was therefore the wrong instrument for this arm (it is the right one for the wholesale arms, which remain bit-exact); the gate now uses it there and reports the mixture as a WARN.  Re-entry: a latched box whose *logits* move outside the tolerance, or one that produces a wrong answer rather than a different near-tie

## 283. `GGML_CUDA_GRAPH_OPT`'s branch streams collide with the staging copy stream's index

**Status: open, latent (found 2026-11-07 while auditing transfer concurrency; `WORKLOG.md` 261107.014).**  The
fork/join scheduler maps a concurrent branch to `stream(device, branch_idx + 1)`, so **branch 0 lands on
`stream(device, 1)` -- exactly `copy_stream()`**, the stream every staging fill, `stage_gather` and the pool fill
use.  With `GGML_CUDA_GRAPH_OPT=1` a split's whole-table uploads would therefore queue *behind* branch compute
instead of overlapping it, and a second copy stream (the parked
`wip/expert-transfer-engine/0002-stage-copy-streams.patch`) would collide at index 2 the same way.  Default-off, so
nothing ships wrong today.  **Fix when the option is next touched:** give the branch streams their own index base
(or make `copy_stream()` reserve its index), and add a `-lv 4` assertion that no concurrent branch ever runs on the
copy stream's index.

## 284. The ring's redirect restore should be deferred to `stage_free_ev` (removes the third VRAM copy)

**Status: open, cheap, measured shape on a small config (`WORKLOG.md` 261107.014; `archive/docs/TENSOR-CORRUPTION.md`).**
`GGML_SCHED_STAGE_MODE=1` points the split input at the ring slot so the bytes move once instead of twice
(`H2D` + full-table `D2D` into `input_cpy`), but the generic ring's restore runs at the top of the next
`sched_stage_issue` -- the "the kernels have been launched" assumption the meta path disproved -- and the record
lists it as "mode 1 = abort, mode 0 = corrupt" on `-sm layer`.  The fix shape is written down there: **defer the
restore to the free-event boundary** (the slots already carry `stage_free_ev` for exactly this ordering).  What it
would buy is real VMEM, not speed on the old box: `input_cpy` is ~1.0 GiB of the 1.62 GiB work floor on Flash-Next
at `-ncmoe 46`, i.e. +4.5 % arena once a sizing policy can claim it; the one clean A/B on the record (meta path,
`-sm tensor`, IQ4_XS) put the D2D at **+0.3 %**, so re-measure before believing the 6 % the Flash-Next arithmetic
suggests.  Gates: same-seed text on Flash-Next at 48 % residency, `gate-prefill-logits.sh` KLD 0.000000.

## 285. A unified expert store: one VA per device, expert-granular residency (`wip/expert-transfer-engine/` §7.6)

**Status: open design direction, not scheduled (scoped 2026-11-07; `WORKLOG.md` 261107.014).**  Today a
host-resident table is represented by three allocations (arena slots, ring slot, `input_cpy`) with three sizing
rules and three reclamation semantics, and a slab chunk reclamation stands a whole table down
(`arena == nullptr && slots == 0`, `TODO` #271).  A single store -- one VA reservation per device, an
expert-granular page table (**~1.5 MiB for all 70 656 Flash-Next experts**), residency by policy, fills landing in
the table's own VA, the op reading `table + e*stride` with **no ids remap** -- would make a stand-down a host-backed
page instead of an unusable table, and collapse the three allocations and three rules into one.  It is **not** a
prefill speed-up: every prefill ubatch routes essentially the whole table (~15.5 GB of used experts), so there is no
cross-ubatch reuse, and a host-backed page read in place is 33 GB/s serialised against the copy engine's 55.8 GB/s
overlapped.  The baseline to beat is the current bulk fallback, which is **free** (r234: 1826 declined vs 1824
staged).  **Two constraints:** it must stay *yieldable* (a raw `cudaMalloc` outside the slab is the pre-slab crash
mode, issue #136), and block 15's VMM region maps in whole units and refuses an idle-slot park on a region-backed
cache -- read `ggml-cuda-vmm-region.cu` and `wip/kv-residency-cleanup/` before designing the page table.

## 286. Re-measure the Meta compute headroom (the delivery's 128 multiplier vs upstream's 256 floor)

`ggml/src/ggml-backend-meta.cpp` resolves the tensor-split compute headroom as
`compute_headroom = 128` (delivery) inside a `std::max(..., compute_headroom_min = 256)` (upstream `#31447`-era).
R01 kept **both** and recorded the question (`DECISIONS.md` `D7`, `RUNGS.md` R01).  Two constants with different
meanings now compete, and nobody has checked whether the delivery's 128 multiplier is still doing anything under
upstream's floor.

**Closing it:** a 2-device `-sm tensor` arm with the headroom forced by each mechanism (`--fit` on a config that is
marginal for VRAM, or a `test-backend-ops`/`llama-bench` pp arm at a context that makes the compute buffer the
limiter), showing which constant actually sizes the buffer and that the other can go — or that both are needed, with
the number.  Record the buffer sizes from the log, not the constants.

**The index row's detail (kept here so the table can stay a dashboard):**

**The Meta compute-container headroom is now carried two ways** — `compute_headroom = 128` per tensor (block 00) *and* upstream `#31447`'s `compute_headroom_min = 256` floor.  The floor may make most of the raised multiplier redundant, or the multiplier may still cover shapes the floor does not (upstream's case is a KV cache shared by many layers; ours is `~2*(n_rs_seq+1)` conv-state snapshot views per recurrent layer)

**Kind:** correctness/allocation (`ggml/src/ggml-backend-meta.cpp`; `BASELINE.md` provenance item 2, `INVARIANTS.md`).

**OPEN — found 2026-11-08 at re-base rung R01 (`wip/upstream-rebase-261108/DECISIONS.md` → `D7`).**  The merge keeps both (the floor is strictly additive), so nothing is at risk; what is unmeasured is whether `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` *plus* the 256 floor now allocates the hybrid recurrent graph that motivated the multiplier.  Measure at R17 on a hybrid recurrent model (qwen3.5/3.6/3.8 or deepseek41), then narrow the multiplier (correcting the invariant in the same change) or record why both stay

## 287. Re-cut `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` against `#31047`'s `tile_spec` grid

The upstream PR candidate for the duplicate `mul_mat_id` launch was cut against the old base; `#31047` moved the
grid/tile representation (`tile_spec`, `nchannels_y = 1` on the ids path — `GREEDY-PURITY.md` §84).  **Closing it:**
re-cut the patch against `a691b2345`, confirm it applies to unadulterated upstream and still removes the duplicate
launch (a launch-census before/after), then update `upstream/README.md`'s sha256.  Delivery-neutral; it is an
upstream contribution, so it must not change the delivery tree.

**The index row's detail (kept here so the table can stay a dashboard):**

**`upstream/UPSTREAM-PR-mul-mat-id-dup.patch` is stale: upstream rewrote the same MUL_MAT_ID tile scheduling in `#31047`** (`tile_spec`, one block per tile, no in-kernel column loop), so the duplicate-ids fix has to be re-expressed on that grid and the PR re-cut.  The delivery's own resolution is the reference: per-occurrence compaction in `mm_ids_helper` + upstream's seam writer, `test-backend-ops -o MUL_MAT_ID` **931/931** on gfx1201

**Kind:** upstream candidate (`upstream/`), not delivery.

**OPEN — found 2026-11-08 at re-base rung R01.**  The delivery side is solved (block 00 re-based on `rebase-r236` as commit `4aaffd1c0`), so the re-cut is a transcription plus a PR-body refresh; the `upstream/README.md` index row and the patch sha256 move with it.  Do not cut the PR against the delivery tree

## 289. Upstream PR owed: `find_slot()` must swap `mem_cell::rs_max`

Found and fixed **in the delivery** at R04: `find_slot()` recycles a cell by assigning `pos`/`src`/`seq_id` but leaves
`rs_max`, so the moved cell's rollback bound refuses legal rollbacks.  The delivery's fix is in block 03; the
**upstream PR is owed** (`HANDOVER.md` §5 item 8 has the full note).  **Closing it:** a patch under `upstream/` cut
against `a691b2345` that applies to unadulterated upstream, with the reproduction (a `test-recurrent-state-depth`
arm or the `-nrs` rollback scenario that fails without it).

**The index row's detail (kept here so the table can stay a dashboard):**

**Upstream `find_slot()` moves a recurrent cell's state but not its rollback bound** — the gather/re-order loop swaps `pos`, `src` and `seq_id` (and re-points the tails) while `mem_cell::rs_max` (`#31368`) stays with the physical cell index.  After a move, `seq_rm()` therefore reads the *destination* cell's bound while the snapshots come from the *source* cell (planes 1..`n_rs_seq` are moved by index, r195), so a legal rollback is refused: measured on the delivery's `test-recurrent-state-depth` Phase C, `n_rs_seq` 4 and 8 refused rollbacks 2..4 and 6..8 only in the batched-`[seq 0, seq 2]` arm, and **every** phase passes once the bound is swapped with the state

**Kind:** upstream candidate (`upstream/`), not delivery.

**OPEN — found and fixed in the delivery on 2026-11-08 at re-base rung R04** (`git show` the `find_slot` swap in `src/llama-memory-recurrent.cpp`: `std::swap(dst_cell.rs_max, src_cell.rs_max)`).  Unadulterated upstream has the same omission, so it needs a PR: a one-line fix plus the `test-recurrent-state-depth` Phase C scenario as the test (upstream's own `rollback_bound` cases do not move a cell).  `test-backend-ops`/`test-llama-context` cannot see it -- it needs the multi-sequence move

## 290. The dispatcher merge debt must actually LAND with r236 (resolved at R11)

R11 reconciled the two `ggml_cuda_try_fuse` copies into one and deleted the stale duplicates (`RUNGS.md` R11
defect 3, `GREEDY-PURITY.md` §84 rule 3) — but the resolution lives in the working chain, so "closed" is only true
once **r236 ships it**.  **Closing it:** after `scripts/make-patches.sh` at R17, confirm block 10's patch is the
single-dispatcher body (no `ggml_cuda_try_fuse_legacy()`, no delivery duplicates of upstream arms) and that
`gate-norm-rows-alias.sh` + `MUL_MAT_ID 931/931` are green on the *re-cut* tree, then close it in `WORKLOG.md`.

**The index row's detail (kept here so the table can stay a dashboard):**

**The fusion dispatcher exists twice while upstream's arms are reconciled**: block 10 moves it into `ggml-cuda-try-fuse.cu`, but upstream added 7 arms to the inline copy in the same re-base range (the MoE weighted-reduction fold, the norm+MUL→mmvq Q8_1 fold, the GDN/SSM state-scatter caches, the SSM conv+l2+gate/beta pre-scan, the DSV4 HC post op).  The interim merge keeps upstream's copy as `ggml_cuda_try_fuse_legacy()` behind the delivery's dispatcher (delivery first, fallback second)

**Kind:** delivery (`ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`, `ggml-cuda.cu`).

**RESOLVED in the re-base campaign (rung R11, fork `ed1d36436`) — lands with r236.**  Audit result: every upstream arm is already reachable from the delivery's dispatcher (`match_shared_expert_try_fuse` IS upstream's arm; `moe_weighted_reduction_try_fuse` is its superset; `gdn_cache_cpy`/`ssm_scan_cache_cpy` wrap upstream's), so only the DSV4 HC post arm needed folding in — at upstream's own position.  `ggml_cuda_try_fuse_legacy()` and the interim `GGML_CUDA_FUSE_LEGACY` switch are deleted, together with the superseded `ggml_cuda_match_moe_weighted_reduction` matcher/struct/constexpr (143 lines) that only they used.  The audit also proved the "fallback" copy was NOT upstream's text: it carried stale duplicates of delivery arms, one of which (the inline SSM conv-input fold) was not bit-identical in a state-carrying path — `GREEDY-PURITY.md` §84

## 291. The arena vs upstream's MoE cache: finish the measurement (`D1`, MEASURING)

`DECISIONS.md` `D1`.  The first numbers are in (`RUNGS.md` R10: upstream's cache +17.6 % decode on one arm) and the
maintainer deferred the analysis to its own session; the delivery's host-expert arena and upstream's cache may be
complementary (different tiers) or redundant.  **Closing it:** the measurement set R10 named — one model at two
`-ncmoe` levels, `MOE_EXPERT_CACHE_MIB` arms, the `-lv 4` census (`expert access:`, `takeover=`, cold admission),
decode **and** prefill, and an A/B of "arena only" vs "upstream cache only" vs both — then a decision recorded in
`DECISIONS.md` (keep both, keep one, drop one).  This is the largest item on the list; it is a measurement, not a
code change, unless the numbers say otherwise.

**The index row's detail (kept here so the table can stay a dashboard):**

**Investigate properly: upstream's MoE cache vs the delivery's host-expert arena at equal MiB.**  Measured once (1 × R9700, `Qwen3.6-35B-A3B Q8_0 -ncmoe 20 -c 8192`, 16.3 GiB host experts, text identical in every arm): delivery arena 4080 MiB **547.6 / 53.5** t/s (pp/tg), upstream `--moe-cache-mib 4096` **544.2 / 62.9**, streaming **470.9 / 49.0** — i.e. upstream's ClassLRU cache is **+17.6 % decode** over ours at ~the same footprint.  The delivery's tier used to be the faster one, so the number is surprising and unexplained

**Kind:** delivery vs upstream (`src/llama-moe-cache.cpp`, `ggml/src/ggml-cuda/moe-expert-cache.cu`).

**OPEN — needs a full measuring session (maintainer's call, 2026-11-08: defer).**  Open questions: is the difference the *policy* (ClassLRU vs the arena's heat/RFU admission), the *tier* (upstream caches in a plain device buffer; ours lives in the slab, and the slab's reserve/headroom may be costing us usable arena), the *config* (one GPU `-sm layer` vs the field's 2-3 GPU `-sm tensor`), or the measurement (the arms are either/or — the coexistence the merge preserves is untested; and the run needs `GGML_HIP_NO_VMM=OFF`, which is now in `~/bin/build-llama-rocm`).  Also note `--moe-cache-cpu-share` (`#30966`) and ClassLRU (`#31233`) are capabilities ours lacks

## 292. `gate-hc-fusions.sh`'s dsv4 soft arms reproduce r235's numerics, not the unfused chain's

**`gate-hc-fusions.sh`'s two REPORTED dsv4 arms drifted away from the chain on the BF16-hc class**: chain 449 chars `d62340fbf566`, `dsv4_pre`/`dsv4_post` 449 chars `6a102dbc62d6` — and `6a102dbc62d6` is what the *chain* produced at r235, so the HC chain moved with the re-base while the fused pre/post ops still reproduce the old numerics.  The gate classifies them soft ("reported") so nothing fails, but its own header says byte-identity is expected on this class

**Kind:** `scripts/gate-hc-fusions.sh`, `ggml/src/ggml-cuda/hyperconn.cu`.

**OPEN — created 2026-11-08 at re-base rung R12.**  Quantify with `gate-prefill-logits.sh` at R17 (KLD + same-top-p on Flash-Next), then either update the soft-arm expectation with the measured numbers or fix the drift, and record which

## 293. Decide whether to keep carrying the delivery's own qwen4exp (Qwen3.8) model support

**Decide whether to keep carrying our own qwen4exp (Qwen3.8) model support.**  R12 (`D3`) kept the delivery's implementation because it is **+31 % decode with byte-identical text** on Flash-Next 3-GPU `-sm tensor`, but it is a 3200-line model file that upstream keeps rewriting (`#30965`/`#31243`/`#31443`/`#30927`/`#31086`), i.e. the largest per-rung conflict surface we own.  On **Qwen3.8-27B** (1 GPU) it was throughput-neutral and its greedy text diverged from upstream's (`b905c9763a58` vs `b21e94a5b6e1`), which is unexplained

**Kind:** `src/models/qwen4exp.cpp`, `src/llama-kv-cache.*` (the keys-only indexer cache), `ggml/src/ggml-cuda/{hc-mix,hyperconn,fattn-qsa,indexer-score}.*`.

**OPEN — created 2026-11-08 (maintainer's call at R12).**  Assess in a follow-up session; the **discriminating measurement is Q8_0 on 2 and 3 GPU `-sm tensor`** (the field config, where the fused HC/indexer kernels are supposed to pay off), plus the KLD gate on the 27B to explain the text divergence.  If ours does not win on Q8_0 tensor-split, drop the model file and keep upstream's (the fused kernels then need re-expressing as fusions over upstream's plain-op graph)

## 295. R17 must include a host-expert multi-slot DFlash arm (the guard's detector)

R13 dropped the F1 host-experts guard (`n_seq > 1` + `-ncmoe` -> host copies) because the device buffer and
the host copies measured identically on the re-based slab (Flash-Next GSQ `-ncmoe` 10/24, 35B-A3B `-ncmoe`
10, 2 slots, `-sm layer`: 0.53793/0.51678, 0.55245/0.47134, 0.63918/0.62887, both ways).  The failure mode it
guarded against is **silent** (draft acceptance -> ~0 while the text stays correct), so the release ladder must
keep one arm that would catch a regression: a host-expert (`-ncmoe`) **two-slot** server with a DFlash drafter
and the acceptance axis read at `-lv 4`.  Add it to `RUNGS.md` R17's arm list if it is not there — the MTP
four-axis protocol alone does not cover a `-ncmoe` multi-slot DFlash run.

## 296. `D8`'s CPU-win measurement is still owed at R17

The re-base kept block 12's CPU hunks untouched (`arch-fallback.h`, `arch/x86/quants.c`, `ggml-cpu.c`,
`ggml-cpu.cpp`, `repack.cpp`; `#31854`'s F32-src1 `type_traits_cpu` entry did not collide with ours), but
*which* of those wins still pays on the rewritten CPU backend was never measured (`DECISIONS.md` `D8`).  The
R17 ladder should run the CPU-side `pp`/`tg` on a host-expert config, `test-backend-ops` for the touched quant
types, and the delivery's `-lv 4` CPU-path census, then either record the surviving wins in `patches/README.md`
or drop the dead ones.

## 298. The VMM pool's free-order invariant: one site fixed (MMB gate/up slot), the class is worth a sweep

**Fixed at r236 (`812deb221`)**: `ggml_cuda_mul_mat_id_mmb_glu` took its long-lived output slot (a `pool_stack`
block, released after the whole graph) *after* five plain pool temporaries, so at scope exit the temporaries
were freed below a live block.  That aborted **every** Flash-Next + MTP-drafter run on gfx1151 (r235 and r236
alike) and therefore blocked `gate-kv-residency.sh`'s pool section there; the pre-re-base binary aborts
identically, so it was pre-existing.  The rule is R13's `hc_combine_norm` rule: **no plain pool allocation may
be created before a `pool_stack` push in the same scope**.  Verified on halo: the sparse and pool arms now both
run and are byte-identical (`1e944569dbd6`).

It blocks `gate-kv-residency.sh`'s pool section on gfx1151 only while it is unfixed -- the arms all attach
the drafter -- and `812deb221` clears that.

**Open — the class, not the site.**  The same violation was seen once on gfx1201 in a *dense* 4B warmup (two
2 KiB blocks freed in allocation order) on a build that differs only by a diagnostic, i.e. it is latent and
layout/timing sensitive: 5/5 aborts on one build, 0/4 on the next.  So `GGML_CUDA_POOL_TRACE=1` exists now
(a backtrace per live block, printed on violation) and the assert always names the sizes/offsets.  A sweep of
every `pool_stack_push` site (`common.cuh`'s `bf16_src1_cache_put`, `hyperconn.cu`'s `hc_base_copy`,
`mmb.cu`'s `mmb_cache_insert`/`mmb_slot_reserve`) for plain allocations in the same scope above it is owed
before the next release — one site is fixed, the *rule* is what keeps breaking.

## 300. gfx1151: the attendance pool (`LLAMA_KV_REGION_POOL`) is INCORRECT on RDNA3.5 — do not claim it there

Found at R16 validating on `halo` (gfx1151 Strix Halo, UMA) with `scripts/gate-kv-residency.sh`.  The **window**
(Goal A) is fine (`wref == wres == wspill` `be904e86f248`, 16 flips / 64 MiB pinned host) and the **sparse**
decode arm is fine, but the **pool** (Goal B) produces wrong results:

| arm (Flash-Next GSQ-IQ3_XXS + MTP, `-c 32768` q8_0) | text | pool `bad` |
|---|---|---|
| `pref` (no pool, sparse ref) | `1e944569dbd6` | — |
| `pres`, whole cache resident (8192 pages), VERIFY off | `b08d3dbcb1e9` | 0 (not verified) |
| `pres`, whole cache resident, `LLAMA_KV_POOL_VERIFY=1` | `8c1b0d16ecd3` | **93 399 360** |
| `pfloor` (4104 pages, forced misses), VERIFY on | differs | **95 173 888** |

Three different texts from the same arm, and a *fully resident* pool (every block of the cache has a page) still
reports ~93 M byte mismatches against its own mirror -- i.e. this is not a residency/thrash effect, the resolve
is wrong.  The mismatch count moves run to run, so it is (at least partly) a race.  The gate therefore reports
`FAIL pres != pref`, a different draft acceptance, and `FAIL pfloor: bad != 0`; everything else in that gate
passes on the box (39 PASS).

Prime suspect: the pool is a **discrete-GPU design** -- a pinned host *mirror* plus a device pool filled by DMA.
On UMA the "mirror" and the weights are the same physical memory pool, so the MARK/CLAIM/MAP+FILL ordering (and
the `kv_pool_invalidate()` / dirty-refill protocol) is reasoning about two kinds of memory that are one here.
It may also be the per-device counter/page-table single-instance path (`bad` varies).

**Mitigated at r236 (`2aac85e78`): the pool now DECLINES on a shared host/device pool** (any
`GGML_BACKEND_DEVICE_TYPE_IGPU` device) with one message naming RDNA3.5/gfx1151 and this item, and the KV takes the
plain device buffer; `gate-kv-residency.sh` grades that decline (once per arm, no arm engaging the placement,
`pref == pres == pfloor`) instead of grading numbers that cannot exist.  So no consumer can trip the defect by
opting in on an APU.

**Still owed: the fix.**  Prime suspect is the tier's own model: a pinned host *mirror* plus DMA fills is a
discrete-GPU design, and on UMA the mirror and the weights are one physical pool, so the MARK/CLAIM/MAP+FILL
ordering, `kv_pool_invalidate()` and the dirty-refill protocol are reasoning about two kinds of memory that are
one.  The per-device counter/page-table single-instance path is the second candidate (`bad` varies run to run).
Start from the gate's `pres` + `LLAMA_KV_POOL_VERIFY=1` arm, which reproduces in ~5 min on halo and needs no
20-minute prefill (any `-c` works) — and from the fact that a *fully resident* pool is already wrong, so no
residency/thrash hypothesis is needed.

The window and the sparse arm are unaffected and validated there.  Pre-existing: the pool shipped in r235 and was
never run on RDNA3.5 before this validation.

## 301. r236 is −2 … −3 % on MoE PREFILL against the shipped r235 (decode flat); localise it

First head-to-head (`benchmarks/2026-11-08-rebase-r235-vs-r236.md`, 3 × R9700, interleaved `llama-bench`):
decode is flat (±0.1 %) on every arm, but prefill is **−3.2 %** on 30B-A3B Q4_K_M (1 GPU), **−2.0 %** on 35B-A3B
Q8_0 (3 GPU `-sm tensor`) and −1.4 % on a dense 27B, while the dense 4B is flat (−0.2 %) and a host-expert arm
(`-ncmoe 30`) is +0.7 %.  So it is device-resident-expert prefill, not decode and not the host-expert path.

It is **not** MMB (`GGML_CUDA_MMB=0`: −3.5 %), **not** MMQ (`GGML_CUDA_DISABLE_MMQ=1`: −3.2 %) and **not** the
fusion dispatcher (`GGML_CUDA_DISABLE_FUSION=1`: −1.6 %) — the gap survives each switch and its absolute size is
roughly constant (~130-160 t/s), which reads as a per-launch/per-batch cost rather than one kernel.  Next: profile
both binaries on `30B-A3B Q4_K_M -p 512` (rocprofv3 kernel list + the `-lv 4` launch census), or bisect the rung
ladder (`gate-per-block-build.sh` already produces all 16 trees' binaries); the bisect is ~16 builds + 16 benches.

Until then, do not claim "r236 is no slower" on MoE prefill — the delivery's headline wins are decode, and this is
a prefill cost on the MoE field shapes.

---

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **A prefill-dominated multi-slot mix (`-npl 4`, `-npl 8`) is slower with `MOE_HOST_POOL_RANK=decode`** (r224,
  block 12; **`#275`**, maintainer-accepted).  The decode ranking is a **single-user** win and its sign inverts
  under `llama-batched-bench`'s prefill-dominated mix: on the frozen real-world prior at `-lme 36`,
  `-npl 4` prefill is 1144.69 t/s against 1292.88 for `combined` (**-11.5 %**) and one `-npl 8` rep measured a
  -21 % total (memory was not the cause: `MemAvailable` stayed at 40 GiB).  Accepted because a single-user
  stream is the common case, and the workaround is documented rather than hidden: **`MOE_HOST_POOL_RANK=combined`
  restores the pre-r224 ranking while keeping the demote** — that is the setting for a deployment whose traffic
  is parallel slots.  The demote itself is safe in every cell measured (+5.3 % single-stream, +1.8/-0.9/+0.3 %
  at npl 1/4/8 in the earlier campaign sweep).  Record: `WORKLOG.md` 261106.012 + the r224 entry,
  `archive/work/host-expert-split/HANDOVER.md` §4.3.

- **The Windows pinned-host ceiling has no cap, only a measurement** (r223, issue #274 follow-up; **no `TODO`
  id** — waiting on the reporter's r227 log, `TODO` #95).  `hipHostAlloc` there is budgeted by WDDM against the
  adapter's shared (aperture) segment, and the published model for it, roughly `(total RAM / 2) / visible HIP
  devices` (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339), is **refuted as a predictor by the reporter's
  own numbers**: GPU shared usage stops at 9.32-9.35 GiB on a 47 GB box and again on the same box at 64 GB, with
  one visible adapter and ~60 GB of commit headroom (`WORKLOG.md` 261106.016).  A ceiling derived from the model
  was implemented and **deliberately withdrawn**, and r227 ships the measurement instead: the exit census prints
  the pinned tier's realized total, and on Windows the aperture segments' `CommitLimit`/`BytesResident` for all
  processes (`ggml_hip_win_shared_mem_info`, the `D3DKMTQueryStatistics` call `wddm-mem.cu` already makes), plus
  a loud, latching refusal for a pool allocation the driver will not give.  The shipped guard is
  mechanism-independent: the first refused `cudaHostAlloc` logs once and latches the pinned tier off (a Windows
  refusal surfaces as `hipErrorLaunchFailure`, which latches process-wide), and no new pool is attempted after
  that.  **The open item is one line of the reporter's next log**: if the aperture `CommitLimit` is near the
  ~9.3 GiB plateau, the cap is that segment and a ceiling is now defensible; if it is far larger, whatever else
  counts against the budget has to be found first.  Record: `WORKLOG.md` 261105.012 + 261106.016,
  `INVARIANTS.md` r223 fact 4 / r227 fact 4.
- **A sub-complement host pool is never served by the device remap** (r221; **no `TODO` id** — the pool
  side of closed `#269`).  A `--host-experts pool` table whose budget does not cover its tensor has, for some
  experts, neither an arena slot nor a pool slot, and a device-side `expert -> slot` map is built once per
  residency change and read blindly by the kernel, so it cannot express "this one must go to the eager path":
  measured `0 of 3840 offers served` under `-lme 16`.  The remap therefore serves a **covering** pool only
  (the default `-lme auto` mirror on a mapped load, +9.9 %) and a pinned master; a sub-complement pool keeps
  the eager path, which is also where `#141`'s arena bound already lives.  The device-count restriction the
  r203 bullet recorded is **gone** — `MOE_HOST_POOL_COLD_MULTIDEV` (default on since r210) is what the remap's
  pool cold source rides, and the remap itself is per-table/per-device, so a tensor split gets its own map
  (+16.1 % / +28.4 % at 2 / 3 devices).  Record: `WORKLOG.md` 261105.007, `INVARIANTS.md` r221,
  `GREEDY-PURITY.md` §82.

- **`cpy_2d_vec16` covers F32 only** (r189, `TODO` **#127**).  The vectorised fallback for the 2-D block copies
  r183's guard refuses is restricted to two **F32** tensors with 16-B-aligned width and pitches — the reachable
  case (qwen4exp's indexer key cache, which is F32 by construction).  Every other type and every unaligned
  geometry keeps `cpy_scalar`, which is bit-identical but element-per-thread.  Nothing in the delivery reaches a
  padded block copy of another type today (the census in `benchmarks/2026-10-31-cpy-2d-vec16.md` is F32-only), so
  this is a "extend it if a BF16/F16 copy shows up" note rather than work: the same 16-B unit is four whole floats
  but only **eight** halves, so a wider version needs its own alignment rule and its own oracle.  Re-check it when
  the KV campaign's sparse/pool arm is promoted (that arm is where the copy is made per *decode* step).

- **A verify step whose slot cannot draft at all still splits into two ubatches** (TODO #111, investigated and
  declined 2026-10-28).  A slot with `get_n_draft_max() <= 0` -- its last token of the response, or a full
  context waiting for a shift -- presents its sampled row alone while its neighbour presents `n_draft + 1`, so
  `split_equal()` puts the neighbour's draft rows in a second ubatch: `0+1` split decode steps per pair run on
  the 35B-A3B `-ncmoe 40` arm (`0+0` on the dense 27B arms, where the two responses do not overlap at a
  boundary).  Both alternatives were measured, one of them implemented and reverted: suppressing that step's
  drafts measures **-0.3 %** (below the floor, because the second ubatch carries the neighbour's *genuine*
  draft rows, so both arms are two passes), and cutting the neighbour's draft is the r153 cut, a **-6 %** loss.
  There is no correctness at stake (a verify width is a numerics axis, `GREEDY-PURITY.md` §63) and the candidate
  would add a cross-slot coupling, so nothing changed -- what did is the instrument:
  `scripts/gate-spec-align.sh` now counts these single-row splits separately and bounds them (`TAIL_MAX`,
  default 2) where it previously excluded them, so a change that *manufactures* them is still caught.
  Record: `benchmarks/2026-10-28-spec-align-111.md`.

- **Parking a memory whose second cache mirrors the attention cells is validated on qwen4exp only** (TODO
  #108, r146): `llama_memory_hybrid_idx` also serves glm5-next, whose indexer is the k-pool
  (`indexer_kpool > 0`) variant.  The park code, its mirror check and its invalidation are arch-agnostic
  and the k-pool layout is invalidated with everything else, but no glm5-next GGUF exists on this fleet, so
  only the qwen4exp arm is measured.  `llama_kv_cache_dsa` / `_iswa` (deepseek4, glm-dsa) report parking
  unsupported by design and keep #210 — they never half-park.

- **A dense `-sm tensor` split is not guaranteed to produce the same greedy text across device counts**
  (AGENTS "Greedy text is a per-device-count reference" was corrected 2026-10-24).  Qwen3.8-27B
  UD-Q4_K_XL, prose prompt, seed 42: 3 GPU 850 chars vs 2 GPU 876; the r126 build shows the same
  (897 vs 855), so it is the tensor split's rounding class, not a delivery defect.  Every `-sm layer`
  arm *is* text-identical across counts.  Gate a change within a device count.

- **A Windows VRAM overcommit is not catchable: it surfaces as a device loss, not an allocation failure
  (issue #216, accepted; nothing in the delivery can change it).**  **r209 note:** block 09 now ships the
  honest *figure* on Windows (`cudaMemGetInfo` returns the smaller of HIP's and the adapter's WDDM figure,
  `GGML_CUDA_WDDM_FREE=0` opt-out), so the slab, the cache sizing, `--fit` and `GGML_CUDA_MIN_FREE_MIB` no
  longer count another process's memory as free.  That does **not** make an overcommit fail, so this item
  stands as written; making the WDDM figure refuse through `GGML_CUDA_MIN_FREE_MIB` was never tried and is
  the open question (re-entry conditions: `archive/work/windows-host-memory/RESOLUTION.md`).  On WDDM an allocation set larger than
  the card does not fail, it pages into system memory; the first big prefill (a full `-ub` compute buffer
  plus the FA staging) then loses the device, and the process's *next* rocBLAS touch aborts with
  `rocBLAS error: Could not initialize Tensile host: No devices found` (rocBLAS's `TensileHost` prints
  that when `hipGetDeviceCount()` returns non-success; it allocates nothing on that path, so the message
  is a symptom, not an OOM inside rocBLAS).  Measured budget for the reported config (dense 27B IQ3_S,
  `-c 131072 -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024` + MTP n3, no mmproj): **14.8 GiB** of static device
  memory and a **16.0-16.2 GiB** first-prefill peak on a headless 32 GiB card, against a 16 GiB card that
  also drives the display.  Mitigations shipped/to use: the draft KV follows the main KV type (r125,
  block 02, -368 MiB at `-c 131072`), `-c`, `-ub`, a real `-ngl` offload (`-ngl` is a **layer count** and
  a NextN model has `n_layer + 1` layers, so `-ngl 85` on a 65-layer model is a no-op), and letting
  `--fit` size the context (it declines when `-ngl` or `-c` is set).  Do not re-investigate the rocBLAS
  message as a delivery defect without a Windows event log showing a TDR (`4101`/`4102`) *absent* at the
  abort.

- **`std::min<int64_t>` / `std::max<int64_t>` in other HIP TUs is a latent Windows-only build failure
  (TODO #95, accepted; no Windows toolchain here).**  With HIP on Windows (clang + the MSVC STL) clang's
  force-included `cuda_wrappers/algorithm` and the MSVC STL `<utility>` each define an inline
  `std::min<long long>` under **one** mangled name (the `enable_if` that makes clang's win overload
  resolution is not part of the MSVC mangling), so a translation unit that instantiates both fails to
  codegen with *"definition with same mangled name"* — r123's `moe-expert-cache.cu` was the first to
  trigger it (issue #217, fixed in r124 with the local ternary helpers; see `patches/README.md` block 09).
  The reporter's toolchain builds every other TU, but ~10 of them still carry the int64_t pattern
  (`argmax.cu`, `cpy-batch.cu`, `fattn.cu`, `mmvf.cu`, `fattn-common.cuh`, `gated_delta_net.cu`, `top-k.cu`,
  `ssm-scan.cu`, `ggml-cuda.cu`, `hc-mix.cu`): the collision only fires where the STL's own instantiation is
  emitted too, which is a header-instantiation accident, not a property of the call.  Not proactively
  rewritten because a non-AMD/Windows claim cannot be validated on this box (scope policy: non-AMD backends
  are out of validation scope); if a Windows build log names one of them, apply the same two-line helper
  fix there and bump the release.  Do not re-report as a new defect without a Windows build log.
  **r209 adds a case in point:** the Windows host-memory fold (blocks 09/11/12/15, PR #269) was proven inert
  on Linux by preprocessing the changed TUs, and it has **never been compiled by us** — no Windows toolchain
  here — so its own build is the reporter's.  A build log naming any of the ~10 `int64_t` TUs, or one of the
  four folded hunks, is the only thing that closes this.

- **The Meta split-state lookup-count headroom (TODO #80) — retired 2026-10-24, accepted.**  The measured
  defect (issue #184: the pre-warm walk allocating one heap node per visited tensor on every graph
  re-allocation) was fixed in r100 (block 09, `GGML_META_SS_SCRATCH`, default on).  What remained is the
  reporter's ~165k cache hits per re-allocation (30+ per node), and the only lever is the lookup **count**
  (the recursion/pruning structure of `calculate_split_state` and the pre-warm walk) — hashing the cached
  336-byte `ggml_tensor` copy to skip the `memcmp` costs about what the `memcmp` costs.  Not pursued because
  the frequency that makes it worth anything is his workload: it re-allocates **~230** graphs per run, while
  the delivery's own validated shapes re-allocate **3** in a whole 4-stream x 192-token Flash-Next IQ3_XXS
  + MTP window (`-sm tensor -ncmoe 47`, 63 graph reuses per stream; 4B Q8_0 2-GPU tensor: 12 in 4 x 128),
  so even at his own post-r100 16.0 ms per re-allocation the whole walk is at most ~1 % of a 4.2 s window
  and the lookup-count share of that is below this box's noise floor (r100's entire heap-churn removal
  measured +0.29 % on this shape).  Issue #184 is closed (2026-10-21) with the reporter confirming r100 in
  his production build (+1.6 % 4-stream decode, identical greedy output), and the walk's code is unchanged
  since r100.  A future change there still owes the r100 proof set (identical states under
  `GGML_META_SS_VERIFY=1`, identical greedy text and prefill KLD, same-binary A/B with
  `GGML_META_SS_SCRATCH=0`).  Records: `archive/work/issue-184-meta-r80r82/RESOLUTION.md` (addendum) +
  `probe-reallocs.py`, `WORKLOG.md` 261024.003.
- **`mmf32` covers the skinny F32 shapes only (r104, `TODO` #82 closed) — accepted.**  A full-F32 model's FFN
  GEMMs (`m` ≫ 256: dense F32/F16-weights-heavy prefill) still fall through to rocBLAS/hipBLASLt, so they keep
  that path's per-start solution lottery; the kernel's tile (32 output rows) is the wrong shape to be
  competitive there and no such model is validated.  Also untouched: BF16/f16 prefill (Flash-Next keeps both
  hipBLASLt's +51 % and its own lottery — a BF16 counterpart is its own job) and **RDNA3** (gfx1100), where the
  kernel stays opt-in (`GGML_CUDA_MMF32=1`) because it measured −1.3…−2.2 % at `pp2048` there; the lever would be
  a per-arch tile, not the gate.  RDNA3_5 was admitted and is default-on since r105 (`halo`: 12/12
  deterministic, +1.0 %/+1.6 % prefill, family-switch PASS).  Scope: `patches/README.md` (block 06), rule:
  `GREEDY-PURITY.md` §47.
- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share: `[1, 1, 0]` over 2 heads).  The abort is the
  split-state **ratio-consistency** assert in `ggml_backend_meta_get_split_state`
  (`split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]`
  — `ggml-backend-meta.cpp`; 1582 at r101, 1487 at r99: the line number moves, the assert does not, and
  older records that cite `:1177` are one more tree back).  Works on 1/2 GPUs and on 3 GPUs with
  `-sm layer`; maintainer's call: no fix.  r101's split-state work deliberately does not change this
  (`GREEDY-PURITY.md` §45 accepts a slice that is *empty or the whole* extent, and one head of two is
  neither); re-verified on r99 and r101, identical abort.  Every other model is unaffected (a future
  block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Gemma4's full-attention (head_dim 512) layers stay outside the rope + FWHT + KV-write fusion**
  (documented 2026-10-21, deliberate).  The fused kernel instantiates `fwht_n` in `{64, 128, 256}`, and
  Gemma4 is the only arch with a distinct SWA/full head dim (`key_length_swa` 256 vs `key_length` 512),
  so its 1/6 full-attention layers run `rms_norm_mul_rope_f32<256,true,float,0,false>` + `fwht_cuda<512>`
  for Q and K plus two `set_rows` — 6 launches instead of 2.  Extending it needs 4 new kernel
  instantiations **and** a fix to the fused kernel's V-row write (its `ncols <= block_size` precondition
  would quantize half a 512-wide V row), which edits the shipping 256 fold, for ~0.1 % on gfx1201.
  Accepted as a gap; the analysis (what to change, why the arithmetic is bit-identical by construction)
  is in [`archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md`](archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md) §5.6.
  (The iSWA builder order that made the 256-wide half of this fusion live is r101, `WORKLOG.md` 261021.003;
  `GREEDY-PURITY.md` §45 records the split-state rule that came with it.)
- **3-GPU plain (non-speculative) decode is a few percent slower than 2-GPU — accepted (this closes
  `TODO #51`; 2026-10-23, `archive/work/three-gpu-performance/RESOLUTION.md`).**  −2.0 % with a fully-resident
  arena (the 3-device internal all-reduce is 1.2 % of decode and no backend/size knob beats it), −10.5 % at
  91 % residency (the non-identity per-token routing readback, paid per device; the *marginal*-arena case is
  fixed in r117, an 8 %-short one is not), −10.2 % with
  `-sm layer` (no intra-token parallelism; a property of layer splitting, reproduced on dense 27B).
  Speculative decode does **not** lose: +14…+23 % at 3 devices when resident.  Production rule: 3 GPUs for
  speculative/MTP workloads, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).
- **`--spec-draft-mtp-vocab` two-band subset (closed 2026-10-22, measured, not shipped).**  A second band
  (keep `< N` **and** `>= n_vocab - M`) was meant to catch gemma4's space-piece band (229376..255999): it
  measured 0.610 acceptance at 37 % of the rows and was still 0.4 % *slower* on that payload.  With the
  width rule found in the 3-GPU arm (`N · n_devices < n_vocab`) a second band only makes the **mirrored**
  draft matmul wider, so it is a non-goal, not a parked idea.  Record:
  `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md` §5.

## Where the current lists live

- **The KV-residency campaign — ARCHIVED** (`archive/work/kv-residency/CLOSEOUT.md`): Goals A, B and C all ship
  as block 15, and `#121`-`#125` and `#129` are closed (the delivery records are `WORKLOG.md`
  261031.007-261031.009, 261031.014, 261101.001/.002).  **What is left of it is its own campaign**:
  [`wip/kv-residency-cleanup/`](wip/kv-residency-cleanup/README.md) — the missing residency gate, O1's go/no-go,
  a window for a *separate* draft model, a pool-size policy above ~200k, and the close-outs those imply.
  (`#124` — Goal A across `-sm tensor` — shipped in r204 and is retired; the design note
  `archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md` is history, and its five changes C1-C5 are what
  `patches/README.md` block 15 now describes.)

- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.

---