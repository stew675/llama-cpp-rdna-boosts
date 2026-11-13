# The ladder — rungs, conflict sites, exit criteria and the ledger

One rung per delivery block, in block order (a later block's conflict is judged against the **rebased**
earlier blocks, so the order is not negotiable).  Block shas are the r277 chain
(`a691b2345..9d7c0502c`), the same commits `release.json` hashes.

How to work a rung: [`README.md`](README.md) §1.  Ledger columns: **status** `TODO` / `DOING` / `DONE` /
`DEFER`-`<decision>`; **sha** = the commit the rung produced on `rebase-r278`.

Legend for the conflict list: the paths the 2026-11-13 probe reported as conflicting for that block
(`CONFLICT-MAP.md`).  A path listed here is *expected* to conflict; a path that conflicts and is **not**
listed is a probe/cascade difference — note it in the rung log, do not assume the probe was wrong.

| rung | block | status | sha | decision points |
|---|---|---|---|---|
| R00 | setup | **DONE 2026-11-13** | — | — |
| R01 | 00 foundation | **DONE (merge+build+coherence); base fix DECIDED (opt 1)** | `43a418c10` | `D9`, **`D17`/B1** |
| R02 | 01 hybrid all-reduce | **BLOCKED on B1** | — | `D13` |
| R03 | 02 adaptive MTP | **TODO** | — | `D7`, `D8` |
| R04 | 03 chunked GDN | **TODO** | — | **`D2`** (stop-and-ask), `D14` |
| R05 | 04 BF16 KV | **TODO** | — | `D3` |
| R06 | 05 flash attention | **TODO** | — | **`D5`**, `D3` |
| R07 | 06 MMB prefill | **TODO** | — | `D5` |
| R08 | 07 k-quant boosts | **TODO** | — | `D4`, `D16` |
| R09 | 08 fused kernels | **TODO** | — | `D6`, `D12` |
| R10 | 09 host-resident MoE | **TODO** | — | **`D1`**, `D9`, `D15`, `D17`, `D18` |
| R11 | 10 infrastructure | **TODO** | — | `D6`, `D8`, `D10` |
| R12 | 11 qwen4exp | **TODO** | — | `D11` |
| R13 | 12 campaign remainder | **TODO** | — | `D10`, `D11`, `D15` |
| R14 | 13 MTP vocab subset | **TODO** | — | `D7` verify |
| R15 | 14 relaxed-tensor (**temporary**) | **TODO** | — | — |
| R16 | 15 streamed KV | **TODO** | — | `D3`, `D19` |
| R17 | 16 UMA GTT store | **TODO** | — | `D17` verify |
| R18 | release ladder | **TODO** | — | all `AWAITING` items resolved |

---

## R00 — setup · **DONE 2026-11-13**

* Campaign dir + `README.md` + `RUNGS.md` + `DECISIONS.md` + `CONFLICT-MAP.md` + `probe-conflicts.py`
  (+ `conflict-map.json`).
* New base facts: old `a691b2345` -> new **`4f9dfa0e3`**, 164 commits, 751 files, +40 612 / −24 226.
* `~/llama.cpp`: branch **`rebase-r278`** created at `4f9dfa0e3`; `master` untouched (pristine snapshot).
* `~/llama.cpp/build-rocm/bin-r277` — the r277 delivery binaries kept as the old-build reference
  (`build 12222, commit 9d7c0502c`).
* Probe: 17 blocks replayed with `git merge-tree` → 55 conflicted paths + 16 modify/delete entries; blocks
  01/04/13/14/16 clean.
* `wip/CAMPAIGNS.md` row + `WORKLOG.md` entry (campaign start; no delivery change).
* **R00 addendum (same day, found at R01):** the new base does not run with the delivery's required
  `GGML_HIP_NO_VMM=OFF` build.  Upstream `ac931b5bc` (`#30542`) makes the HIP VMM pool hang on ROCm
  10.1/gfx1201; a pure `master` build hangs too, and unpatched `master` with `GGML_HIP_NO_VMM=ON` works.
  Full reproduction, evidence and the four options are in [`BASE-BLOCKER.md`](BASE-BLOCKER.md); the register
  entry is `D17`.  A candidate fix (revert `ac931b5bc`, folded into block 00) is on `rebase-r278` and the
  rung below is green with it — it is `AWAITING` the maintainer's decision, not shipped.

---

## R01 — block 00: foundation (band uniformity, Meta wrapper, allocator) · **DONE 2026-11-13 (merge +
build + coherence); the folded base fix is DECIDED (`D17` option 1)**

* block sha `f15430d1e` · 34 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/llama-mmap.cpp` | `9d340ec34` (`#26014`) rewrote the file (+300, new `test-llama-mmap.cpp`) for a Windows-unbuffered load; `58e38bdf6` (`#31957`) narrowed the iGPU lazy fallback.  Our lazy/mlock hunks re-derive on the new load path — **`D9`**. |

**Note:** this block was *much* worse in the previous climb (12 paths, mostly `-Wshadow` rename ballast that
the r236 resolution dropped).  The current 34-file block is the resolved r236 shape, so expect a small,
mechanical merge.  Also verify the `GGML_COMPUTE_BUFFER_MARGIN_PCT` invariant still fires (upstream's Meta
compute-view floor is unchanged in this range).

**R01 result: `43a418c10`** (block 00 `f15430d1e` + `git revert ac931b5bc`; 36 files, +1049 / −357 against
the new base).  The only conflict was `src/llama-mmap.cpp`, and it resolved as an **upstream-wins rename**:
block 00 renames the `read_aligned_chunk` parameter `size` -> `size_l` (the `-Wshadow` pass) while upstream's
Windows-unbuffered rewrite independently renamed it `size` -> `len`; the bodies are otherwise identical, so
the resolution takes upstream's `len` in both platform variants and keeps the delivery's real content
(`llama_mlock::size()/release()/granularity()` + the `raw_lock` rename), which auto-merged.  No marker, no
`TODO(rebase)`.

**Exit:** build green with the canonical `GGML_HIP_NO_VMM=OFF` (36-file delta; only the pre-existing
`ld.lld` unroll notes); coherence on `Qwen3.5-4B Q8_0`, 1 GPU, `-sm layer`, seed 42 -> **100 chars
`sha=1c5d32ac537d`** (`Prompt 256.0 / Gen 92.1` t/s); `test-backend-ops -o MUL_MAT_ID` **1040/1040** on
gfx1201 (the duplicate-id fix reaches the new tile grid); `git grep '<<<<<<<'` empty.

**Base fix (DECIDED 2026-11-13, `D17` option 1):** the revert of `ac931b5bc` is folded into block 00 — it is
what makes the VMM=OFF build run at all on this box (without it the unpatched base hangs —
`BASE-BLOCKER.md`).  Dropped at the next re-base once upstream fixes the HIP pool.

## R02 — block 01: hybrid HIP all-reduce (RDNA4-gated) · **BLOCKED on `D17`/B1**

* block sha `d3470daa6` · 4 files · **expected conflicts: 0** (CLEAN)

**Expect:** `432433637` (`#31838`, fixed BLAS workspace) and `ac931b5bc` (`#30542`, VMM pool) are in the
same files but were not reported as conflicts.  Verify the RCCL/HIP all-reduce dispatch still selects the
delivery's hybrid arm on a 3-GPU `-sm tensor` arm, and that the gate is still reachable (`--sm tensor
--device 0,1,2`).  **`D13`**.

## R03 — block 02: adaptive MTP draft depth · **TODO**

* block sha `e16b66272` · 23 files · **expected conflicts: 3**

| conflicted path | what to expect |
|---|---|
| `common/speculative.cpp` | `a8fad21e9` (`#31927`) adds the MTP target-hidden-state device buffer + draft plumbing; `7874988b4` (`#31294`) adds a `--spec-draft-p-min` warning.  **`D7`**. |
| `src/llama-context.cpp` | the same `#31927` (+199 lines) — the device buffer's context surface meets the delivery's draft-depth control. |
| `src/llama-ext.h` | `8ca5a8008` (`#30565`) prunes the file (−64); the delivery adds to it.  Re-add *only* the delivery's declarations.  **`D8`**. |

**Exit:** adaptive-MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) is owed here if the
device-buffer path changes the measured decode; otherwise record why not.

## R04 — block 03: chunked gated-delta-net prefill · **TODO — STOPS FOR `D2`**

* block sha `dd81963db` · 25 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `src/models/delta-net-base.cpp` | **`ccd360f06` (`#32078`) removed the non-fused GDN and every ubatch-dependent graph node** (−405 lines here, plus `clef`/`clover-moe`/`kimi-linear`/`qwen35`/`qwen35moe`/`qwen4exp`/`models.h`/`llama-context.*`/`llama-cparams.h`/`llama-graph.h`).  **`D2`**. |

**Do not resolve blind.**  Read both sides first: (a) did upstream delete the *non-fused* builder our
chunked path replaced (then ours is the surviving implementation and the merge is a de-registration), or
(b) is our separate graph path now dead weight?  Measure prefill on a hybrid recurrent model with the
delivery path on and off, present to the maintainer, and wait.  **`D14`** (kernel side: `a8575a3ef`,
`998200491`) is a consistency check only.

## R05 — block 04: BF16 KV cache and native-BF16 flash attention · **TODO**

* block sha `7708d67ff` · 8 files · **expected conflicts: 0** (CLEAN)

**Expect:** applies clean, but `202c1a229` (`#28392`, FP8 KV) and `9e9619d03` (`#28413`, OCP FP8) are the
new neighbours — audit that the KV type set, the FA instances and the `supports_op` predicates stay
mutually consistent for every backend; a reachable-but-uninstantiated type is a crash
(`GREEDY-PURITY.md` §20).  **`D3`**.  No measurement beyond coherence: this block's own A/B is the BF16 KV
decode arm, re-run at R18.

## R06 — block 05: flash-attention band, native quantized K/V and derived kq mask · **TODO**

* block sha `a45c508e9` · 295 files · **expected conflicts: 8 + 16 modify/delete**

| conflicted path | what to expect |
|---|---|
| `ggml/CMakeLists.txt`, `ggml/cmake/common.cmake` | the FA instance generator/build grouping moved (`cd4ed902e`, `9d7b2d739`, `77e2db8ca`). |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | new head sizes 40/72 and the sm70 256 case. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | the matrix generator was rewritten; our instance list must be re-expressed in it, not appended by hand. |
| `src/llama-context.cpp`, `src/llama-cparams.h`, `src/llama-graph.h` | the kq-mask/FA-row plumbing meets `#31927`/`#32078`. |
| `tests/test-backend-ops.cpp` | `03d6c824e`/`1648d6411` moved the tests to the public API; our FA oracle entries move with it. |
| 16 × `template-instances/fattn-mma-f16-instance-*.cu` | **modify/delete**: upstream regenerated the ncols matrix; take upstream's set and re-add only the delivery-specific instances. |

**Expect:** **`D5`** — re-derive the delivery's FA band selector over the new instance matrix (including
the FP8 vec FA), keeping our band as the preferred path where it wins and upstream's kernel reachable.
`iq4_nl` instances are set by hand on this box (`GGML_CUDA_FA_QUANTS`) — re-set them when this rung lands.

## R07 — block 06: MMB quantized prefill GEMM · **TODO**

* block sha `7f17e7534` · 11 files · **expected conflicts: 1**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` (`#31533`) rewrote the MMQ/MMVQ dispatch and added `mmq-few.cu`; `0d69255c5` (`#26048`) added the nvfp4 w_s epilogue. |

**Exit:** prefill-logit gate (this is a prefill-kernel block), plus the MMB kill switch's own A/B.

## R08 — block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad) · **TODO**

* block sha `42c96a0ea` · 9 files · **expected conflicts: 2**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `0fb8f1577` few-row MMQ, `494d0f47b` src1-precision gating, `5d4c8eeb2` MoE tiles. |
| `tests/test-backend-ops.cpp` | `494d0f47b`/`03d6c824e` reworked the MUL_MAT oracle. |

**Expect:** **`D4`** (MMVF BF16 few-row — measure against the delivery dispatch) and **`D16`** (the
src1-precision gate must reach our path, not a fallback).

## R09 — block 08: fused kernels · **TODO**

* block sha `7f1ed0723` · 38 files · **expected conflicts: 7**

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d` (`#32001`) moved graph + concurrent-stream state out to `graph.cuh`; our fusion hooks follow it.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the fusion dispatch meets the graph move + cluster-dim key (`2e325dcd5`). |
| `ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu`, `quantize.cu(h)` | `0fb8f1577` + `5d4c8eeb2` + `0d69255c5`. |
| `ggml/src/ggml-cuda/norm.cu` | `31953bcf7` (`#32135`) rewrote BF16 RMS_NORM (−180/+180).  **`D12`**. |

**Exit:** `gate-norm-rows-alias.sh` on the merged tree; the fusion-alias gate if the out-node set moved.

## R10 — block 09: host-resident MoE (iface, scheduler, Meta, slab, cache engine) · **TODO**

* block sha `495d00d45` · 69 files · **expected conflicts: 8**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | `5f490d7bf` (`#31185`) external PLE + the `fit.*` reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9`, `7874988b4`.  **`D7`**. |
| `ggml/src/ggml-backend.cpp` | `806b3c2be` (`#31419`) upload reachability, `3bce8bd50` (`#32100`) prefetch.  **`D10`**. |
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d`, `432433637`.  **`D6`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the VMM pool (`ac931b5bc`) + graph move + MoE tile fix.  **`D17`**. |
| `src/llama-context.cpp`, `src/llama-context.h` | `951d465f5` (`#32157`) moved the MoE cache out of `llama-graph`; `a8fad21e9`.  **`D1`**. |
| `src/llama-mmap.cpp` | the new load path.  **`D9`**. |

**Expect:** port the delivery's cache hook to the new `llama-moe-cache` seam, keep the arena default-on and
upstream's cache dormant (the r236 either/or measurement stands).  **`D18`** is consistency only: do not let
the SYCL scheduler flags change the CUDA predicate.

## R11 — block 10: slab, fusion dispatcher and async input ring · **TODO**

* block sha `05f298fea` · 86 files · **expected conflicts: 10** (the largest cluster)

| conflicted path | what to expect |
|---|---|
| `ggml/src/ggml-cuda/common.cuh` | `17701e48d` graph/concurrent state move.  **`D6`**. |
| `ggml/src/ggml-cuda/cpy.cu`, `mmvf.cu`, `mmvq.cu`, `norm.cu` | `0fb8f1577`, `494d0f47b`, `31953bcf7`, `432433637`. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | graph move + cluster dims + few-row dispatch. |
| `src/llama-context.cpp` | `cb44fdb18` (`#32089`) graph-result reuse, `4fffab623` (`#32043`) metadata pages, `a8fad21e9`.  **`D10`**. |
| `src/llama-ext.h` | the prune (−64) vs the delivery's additions.  **`D8`**. |
| `src/llama-graph.cpp` | `951d465f5` MoE-cache move, `cb44fdb18`, `4fffab623`. |
| `tests/test-backend-ops.cpp` | the public-API test refactor. |

**Exit:** `gate-node-budget.sh` (the async ring adds graph nodes), `gate-grid-stall-pad.sh` if the pad site
moved.

## R12 — block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer) · **TODO**

* block sha `ea4963486` · 51 files · **expected conflicts: 5**

| conflicted path | what to expect |
|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | RPC surface + op table (`0eebb6edd` TRI backward, `57d04820d` Top-K) + the prune.  **`D11`**. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the op dispatch + the new graph/stream state. |
| `src/models/qwen4exp.cpp` | `ccd360f06` removed GDN/ubatch-dependent nodes from this model.  **`D2`**-adjacent. |
| `tests/test-backend-ops.cpp` | the public-API move. |

**Exit:** `gate-hc-fusions.sh` on the qwen4exp model, `gate-qwen4exp-quant-coherence.sh`.

## R13 — block 12: campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache) · **TODO**

* block sha `c7628e974` · 62 files · **expected conflicts: 4**

| conflicted path | what to expect |
|---|---|
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | `D11` + `D8`. |
| `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` | the FA generator (FA build grouping). |
| `src/llama-context.cpp` | `a8fad21e9`, `cb44fdb18`.  **`D10`**. |

## R14 — block 13: MTP draft vocabulary subset (`--spec-draft-mtp-vocab N`) · **TODO**

* block sha `f0c7f3359` · 19 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean, but upstream's MTP device-buffer path (`D7`) changes what a draft sees — verify the
vocab-subset filter still applies to the drafted tokens.

## R15 — block 14: `-sm relaxed-tensor` (**TEMPORARY**) · **TODO**

* block sha `6b4d9865e` · 11 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean.  Re-check upstream has not started allowing the arches the relaxation covers; if it has,
this block is dropped by decision, not by merge.

## R16 — block 15: the streamed KV — window (Goal A) + attendance pool (Goal B) + MTP draft window · **TODO**

* block sha `99d33117a` · 46 files · **expected conflicts: 5**

| conflicted path | what to expect |
|---|---|
| `common/fit.cpp` | the PLE/fit reshuffle.  **`D15`**. |
| `common/speculative.cpp` | `a8fad21e9` device-buffer MTP.  **`D7`**. |
| `ggml/include/ggml-rpc.h`, `ggml/src/ggml.c` | `D11`. |
| `src/llama-kv-cache.cpp` | `202c1a229` (`#28392`) added the FP8 KV type to every instance/predicate table.  **`D3`**. |

**Exit:** `gate-kv-residency.sh`, `gate-kv-window-split.sh`, `gate-spec-draft-window.sh`,
`gate-spec-reuse-window.sh`.  `D19` (`940a30ee3`) is a verify-only item here.

## R17 — block 16: the UMA GTT expert store · **TODO**

* block sha `9d7c0502c` · 16 files · **expected conflicts: 0** (CLEAN)

**Expect:** clean.  Verify **`D17`**: the halved VMM reservation + retry must not change the region the
GTT store asks for (`-lme` on `halo`/`soar`).  Gate: `gate-uma-experts.sh --quick`.

## R18 — release ladder · **TODO**

See [`README.md`](README.md) §4.  Not a rung: `make-patches.sh`, `make-release.sh`, `validate-set.sh`,
`gate-per-block-build.sh`, `gate-prefill-logits.sh --record` from a clean `4f9dfa0e3` build,
`gate-rebase-release.sh`, the block gates, `check-gpu-faults.sh`, fork refresh, containers, records, and
this campaign moved to `archive/work/`.

**Every `DECISIONS.md` item marked `AWAITING` must be `DECIDED` before R18 starts.**
