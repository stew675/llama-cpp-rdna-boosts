# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Renumbered 2026-10-31 (docs-only).**  The r187-r189 sessions allocated `#190`-`#196` (and `#218`) as if the
sequence had advanced past a run of closed items, but `#119` was the last handle ever allocated and `#120`-`#189`
were never used.  They are **`#120`-`#126`** now (plus `#127` for the `cpy_2d_vec16` limitation), so the sequence
is contiguous.  The r187-r190 **commit messages** (immutable) still cite the old handles; they mean these rows.
The old numbers are **retired** — do not reuse them.

> **Handle numbering (corrected 2026-11-03, sequence re-checked 2026-11-04):** the real sequence runs to **#130**
> (closed, issue #263), so this file's `#268`/`#269` — allocated by the 2026-11-02 session — are themselves out of
> sequence, the same failure the 2026-10-31 renumbering note below describes.  **#131`-`#143 are allocated** (the
> 2026-11-03/04 items; `#143` closed in r214; `#270` is the r215 finding), so **new items continue from `#271`**.  Renumbering `#268`/`#269` would also touch
> `WORKLOG.md` 261102.* and the campaign records that cite them, so it is left as a decision for the maintainer
> rather than done here.

**Current state:** the **16-block V2.1 set** (block 00 + blocks 01-15) at fork point **`6050b467c`**
(the 2026-10-25 re-base), release **`v16-6050b467c-r214`** (read `.release` in `release.json` -- it is the
source of truth, not this line).  Block 15 is the **streamed KV** (the resident
window *and* the attendance pool, both opt-in).  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).

| # | subject | kind | status |
|---|---|---|---|
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM (campaign `wip/fp8-support/`) | performance / new type | **PARKED** (2026-09-26).  Re-based onto the delivery as 13 commits (`~/llama-fp8`, branch `fp8-rebase`), builds clean on gfx1201, fully RDNA4-gated — but **gate 2 fails**: on a 4B the fp8 arm is **0.69x Q8_0** where the pre-rebase `cllm` branch had **1.16x**, because the delivery's MMB/GEMM work lifted Q8_0 **+32 %** (8146 vs 6177 t/s pp512) and the fp8 WMMA path did not follow (5607 vs 7184-7260).  Kernel trace: `mul_mat_fp8_wmma` 71.9 %, **`fp8_repack_weights` 7.3 % (running per call instead of once — the likely bug)**, staging 3.3 %, FA 1.3 %.  Not delivery work: nothing here is in `patches/`.  Resume: A/B the GEMM against `~/cllm`, fix the repack cache, re-run the 4B gate, only then convert and measure the 27B.  Detail in §116 |
| **131** | `--host-experts pool` prefill collapsed under a >=3-device `-sm tensor` split (block 12's staging path; root cause still unidentified) | performance (delivery defect) | **FIXED in r207** (pool tables gather in the prefill band, block 12: 1502-1569 vs `pinned` 471.8 prefill at 3 devices, decode -0.7 %; `GGML_SCHED_DEVGATHER=0` reproduces 16.9 t/s).  **Still OPEN:** the staging path's own cost, and the promotion.  See §131 |
| **132** | The pool's pair decode deficit is the synchronous demand read (~11.3 ms/token = 29 % of the step); re-open the overlap worker the 1-GPU ~6 % ceiling declined (`archive/work/host-pool-speed/README.md` §16.3) | performance | **OPEN** (2026-11-03, `archive/work/host-pool-speed/README.md` §18.4).  See §132 |
| **133** | A fully-resident (identity) expert table is re-copied out of the arena on every prefill ubatch — 3-GPU `pinned` prefill is *slower* than 1-GPU (471-607 vs 755.5 t/s) at 100 % arena residency | performance (delivery defect, default mode) | **OPEN** (2026-11-03, `archive/work/host-pool-speed/README.md` §18.2).  See §133 |
| **134** | Re-base the delivery onto upstream `6b9c0cfcb` (207 commits; block 00 conflicts; two overlapping upstream commits to review) | provenance / maintenance | **OPEN — its own release** (2026-11-03, `WORKLOG` 261103.003).  See §134 |
| **135** | A `-sm tensor` split across more devices than the model can divide evenly is **degenerate**, and nothing stops it at load: 35B-A3B has no even split above 2 GPUs, so a 3-device arm runs at ~20 t/s against ~55 (1 GPU) / ~52 (2 GPUs) — the third device *halves* throughput.  The item is now a **load-time guard**, not a perf investigation: detect an unsupported device count for the model and refuse it (vLLM-style) instead of silently running degenerate | correctness / usability (delivery — a load-time refusal) | **OPEN — limitation understood, the work is the GUARD** (maintainer, 2026-11-04).  The cause is **divisibility**: 35B-A3B has no even `-sm tensor` split above 2 devices, which is the same class as `gemma-4-E4B-it` + 3 devices in the accepted-limitations list (2 KV heads < 3 devices aborts in the split-state assert).  So the ~20 t/s is the *symptom* of a zero/uneven per-device share, **not** a host-expert performance bug — do not chase it as one.  What is needed: a **load-time check** that decides whether the requested device count can split this model's tensors evenly and, when it cannot, **refuses to offload that far** with a message naming the constraint and the supported counts, as vLLM's TP validation does (detect it from the model's own geometry — heads/expert counts — never a hard-coded model list).  Evidence: r209, 35B-A3B Q4_K_M `-ncmoe 40`, gfx1201 x3, `archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §7 — all-VRAM on the same 3 devices gives 138 t/s, so the cliff is the split geometry, not the device count |
| **136** | `GGML_SCHED_DEVGATHER`'s prefill-band default **costs 33 % prefill** on 35B-A3B at `MOE_HOST_POOL_FRAC=50` (422 -> 282 t/s at 3 devices, 5246-token prompt, two reps within 1 %), where r207 chose that default from a 122B/FRAC-95 measurement | performance (delivery defect, default-on) | **OPEN** (2026-11-03, `archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §4).  Needs a wider sweep (model x FRAC x ubatch) before touching the default — and note the same switch must be `0` for any prefill A/B of the expert fill path, because the gather serves the routed table itself |
| **141** | A **genuine sub-complement** host-expert budget (`-lme 8` = 43 % of the set, or any `MOE_HOST_POOL_FRAC` below 100) declines the eager take-over at wide decode: `llama-batched-bench -npl 8` gives **50.4 tg**, i.e. worse than the cache being off (129.2), because `slots + pool coverage < n_used * n_tok` cannot guarantee a complete remap and the run falls back to the pageable mapping | performance (block 09, the pool tier) | **OPEN -- measured limit** (2026-11-04, `WORKLOG.md` 261104.002 §4).  The bound is static per shape on purpose (a per-token decision cannot be baked into a captured graph), so the honest answers are a bigger arena, the pinned tier, or the retired `MOE_HOST_POOL_SIZING`/coverage work.  Candidates worth scoring offline with `scripts/moe-trace-replay.py` before any code: (a) require the arena to hold `n_used * band` slots for a pool-backed table (size it, do not decline it), (b) serve a neither-tier expert from the bounce ring *inside* the take-over instead of declining the table (needs the ring to be a legal cold source for one token), (c) accept the variance and let the graph re-capture.  Nothing is in flight; the r211 fix covers only the covering case |
| **142** | The pool tier's **first** prefill pays ~3 s: `llama-batched-bench -npp 512 -npl 1` measures **pp 128** for the pool against **588** for a pinned master (3.97 s vs 0.87 s), and the same 3 s is amortised invisibly inside a 5246-token prompt (prefill 1318 t/s, fine).  The pool is built and prepopulated lazily and the arena is sized only on the first DECODE-band pass, so the first prompt batch runs the pre-sizing staging path with no arena at all | performance (block 09/12, `moe-expert-cache`) | **OPEN -- pre-existing (r210), not a r211 regression** (2026-11-04, `WORKLOG.md` 261104.002 §4).  `MOE_HOST_POOL_PREWARM=0` does not move it (122 vs 130), so it is the build/first-touch, not the prewarm order.  The lead from `archive/work/host-expert-eager-push/P6-POOL-REGION.md` §7d is to move the population into the preflight (before the first timed batch) -- it was scoped as worth only ~0.09 s at `-n 300`, but this row is the arm where it is worth 3 s |
| **269** | PR #266's disposition: keep `GGML_MOE_DEVREMAP` (the sync-free decode remap, +3.3 % on this host's 122B arm), drop the rest — and the remap must become **tensor-split aware** before it is re-implemented over pool slots | capability / acceptance condition | **DECIDED 2026-11-02 (maintainer); OPEN, demoted** behind #131/#132 (2026-11-03).  Today `devremap_ok_locked()` needs a pinned master and `MOE_HOST_POOL_COLD` declines whenever `device_count != 1`, so a `-sm tensor` split gets no map at all; the map is per (table, device) while a pool slot is process-wide, so the cold-id encoding must be per-device-slice aware and the freshness invariant must hold per device.  Gates: same-seed greedy text and MTP acceptance on 1 GPU **and** the multi-device `-sm tensor` arm, `check-gpu-faults.sh` clean.  Detail: §269 |
Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

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

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

* **`MOE_HOST_POOL_COLD` needs a single VISIBLE device** (r203; **no `TODO` id** — the `#131` this bullet used to cite is the pool-prefill collapse below, a different finding).  The in-place pool read keys on
  `ggml_cuda_info().device_count == 1`, which is process-wide: on a multi-GPU box running a one-GPU arm (three
  R9700s plus the iGPU are four HIP devices here) it is **silently off**, costing that arm the +4.8 % TG the
  same config gets with `HIP_VISIBLE_DEVICES` pinned to one device.  The restriction is deliberate — the pinned
  pool is process-wide and its layout re-rank is not device-scoped — so it is a "widen it with a per-device
  layout epoch if a multi-device pool arm ever matters" note, not work.  The remap half (`GGML_MOE_DEVREMAP`)
  has no such limit (it is per-table, `t.device == device`) but declines on a pool-backed table anyway.  Record:
  `benchmarks/2026-11-01-host-expert-device-remap.md`, `GREEDY-PURITY.md` §82.

* **`cpy_2d_vec16` covers F32 only** (r189, `TODO` **#127**).  The vectorised fallback for the 2-D block copies
  r183's guard refuses is restricted to two **F32** tensors with 16-B-aligned width and pitches — the reachable
  case (qwen4exp's indexer key cache, which is F32 by construction).  Every other type and every unaligned
  geometry keeps `cpy_scalar`, which is bit-identical but element-per-thread.  Nothing in the delivery reaches a
  padded block copy of another type today (the census in `benchmarks/2026-10-31-cpy-2d-vec16.md` is F32-only), so
  this is a "extend it if a BF16/F16 copy shows up" note rather than work: the same 16-B unit is four whole floats
  but only **eight** halves, so a wider version needs its own alignment rule and its own oracle.  Re-check it when
  the KV campaign's sparse/pool arm is promoted (that arm is where the copy is made per *decode* step).


* **A verify step whose slot cannot draft at all still splits into two ubatches** (TODO #111, investigated and
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

* **Parking a memory whose second cache mirrors the attention cells is validated on qwen4exp only** (TODO
  #108, r146): `llama_memory_hybrid_idx` also serves glm5-next, whose indexer is the k-pool
  (`indexer_kpool > 0`) variant.  The park code, its mirror check and its invalidation are arch-agnostic
  and the k-pool layout is invalidated with everything else, but no glm5-next GGUF exists on this fleet, so
  only the qwen4exp arm is measured.  `llama_kv_cache_dsa` / `_iswa` (deepseek4, glm-dsa) report parking
  unsupported by design and keep #210 — they never half-park.

* **A dense `-sm tensor` split is not guaranteed to produce the same greedy text across device counts**
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
the arena holds the whole set: 49.48-51.03 vs 51.97-52.04, i.e. **-1.8 %**, inside PR #267's own envelope).  This
is now a bigger prize than the tensor-split remap (`#269`, +3.3 %).  Protocol: re-screen with
`scripts/moe-trace-replay.py` first (`archive/work/host-pool-speed/README.md` §9.6 — an overlap does not change placement, so the
model cannot score it directly; use it to size the residual demand), then build the worker behind a kill switch and
gate it on the pair at `-ncmoe 48`, `FRAC` 80, with text + MTP acceptance + faults.

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

## 269. PR #266's disposition: keep the device remap, drop the rest — and the remap must be tensor-split aware

**Status: decided 2026-11-02 (maintainer); 2026-11-03 update — still open, still gated, and now DEMOTED behind
`#131` (the pool prefill collapse at ≥3 devices) and `#132` (the ~25-30 % demand-fill overlap ceiling on a split).
Nothing in the 2026-11-03 multi-GPU session blocks it or changes its condition; the multi-device arm it needs now
exists and is healthy (F7 was an environment artefact: `MOE_EXPERT_CACHE_MIB=0`, not a `-sm tensor` registration
gap — `archive/work/host-pool-speed/README.md` §18.1), so the pair/3-GPU `-sm tensor` gates it names are runnable.**
Forward-looking: it constrains how `GGML_MOE_DEVREMAP` is re-implemented over pool slots, and it is the acceptance
condition for that work.  The maintainer's call:
PR #266's **`GGML_MOE_DEVREMAP` survives** the #266 reversion and stays a capability going forward, because it is
the one piece of #266 whose removal costs real throughput (the sync-free decode map; r203 measured +8.1 % on its
own cell, +3.3 % on this host's 122B arm — `archive/work/host-pool-speed/README.md` §15).  Everything else in #266 is
dropped, including `MOE_HOST_POOL_COLD`, which the pool-aware device gather now subsumes (measured: `COLD=0`
15.54 -> 19.57 t/s with the gather, `README.md` §14.4).

**The condition:** the remap must be **tensor-split aware**.  Today `devremap_ok_locked()` needs a pinned master
(`table_cold_ok()` -> `t.host_dev != nullptr`) and a `MOE_HOST_POOL_COLD` pool read declines whenever
`device_count != 1`, so a `-sm tensor` split gets no map at all.  An implementation that only works on one device
is not acceptable: the map is per (table, device) — a pool slot is process-wide and its device alias is valid on
every device — so the cold id encoding must be per-device-slice aware (each device's arena slots differ; the pool
slot does not) and the freshness invariant (a re-rank invalidates a lagging map: see `archive/work/host-pool-speed/`
§3.1's layout epoch / freeze) must hold per device.  Gates: same-seed greedy text and MTP acceptance on 1 GPU
**and** on the multi-device `-sm tensor` arm (`soar`), with `check-gpu-faults.sh` clean.

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
