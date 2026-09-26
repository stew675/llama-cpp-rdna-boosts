> **SUPERSEDED 2026-09-15.**  Session 2 acted on this handover: **TODO item 2 is DONE**
> (`q4_1`/`q5_0`/`q5_1`/`iq4_nl` all have native arms; the three WIP bugs are fixed) and the
> r4 candidate was promoted — **`v16-790cf51aa-r4`**, tip `b19c70b341f9ed439bcda2a636fe6e5fa4fa634b`,
> tree `7fab975d9518b29aa7d890c1163f13a6c393c5df`, `validate-set.sh` strict 16/16, CI green.
> The sections below are kept as the session-2 starting state, not as current status: read
> `WORKLOG.md` 2026-09-15, `patches/README.md` (the 2026-09-15 block-15 amendment),
> `archive/work/issue-30-mtp-decode-regression/MEASUREMENTS.md` §I and `GREEDY-PURITY.md` §36 instead.
> `~/issue-30-followup-response.md` is **ready to post**.

# Issue #30 — session handover (2026-09-14)

**Read this file, then continue.** It is the complete state of the issue-#30 work: what is validated and
promotion-ready, what is not, the exact next steps in order, and the hard-won gotchas so you don't
re-derive them.

---

## 0. TL;DR

| change | state | where |
|---|---|---|
| **block 04** — arch-aware head-256 WMMA config + split-aware `ncols2` (deep-prefill fix) | **PROMOTED** (r3) | `patches/0004…`, release `v16-790cf51aa-r3` |
| **block 15 (r2)** — V4 native staging default for sub-F16 quants + the q4_0 arm | **PROMOTED** (r2) | `patches/0015…` |
| **block 15 (r4 candidate)** — mixed-K/V kernel contract fix + `get_alloc_size` q4_0 fix + the prefill band split + the arena + the RDNA3_5 arch gate | **VALIDATED on gfx1201+gfx1151, NOT promoted** | `archive/work/issue-30-mtp-decode-regression/patches/2026-09-14-todo21-prefill-arena-staging.diff` (7 files, 352 lines) |
| **Item 2** — native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl` | **IMPLEMENTED BUT BROKEN — not landable** | `…/patches/2026-09-14-item2-native-arms-WIP-BROKEN.diff` (727 lines) |
| **issue-#30 response draft** for @briansp2020 | written, **post only after r4** | `~/issue-30-followup-response.md` |
| TODO items 20 + 21 | **CLOSED** (2026-09-14) | `TODO.md` |
| purity tiering doctrine | **landed** | `GREEDY-PURITY.md` §36 + `AGENTS.md` critical-facts bullet |

**The fork working tree at `~/llama.cpp` currently contains the r4-candidate changes PLUS the broken
Item 2 work** (8 modified files, uncommitted). See §3 for how to separate them.

---

## 1. Environments

### soar (this box) — gfx1201, 3× R9700
* Fork: `/home/stew675/llama.cpp` (branch `rdna-boosts`, HEAD `a2c8d06a7` = r3 canonical tip).
* Build: `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`, then
  `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:build-rocm/bin`.
  (The `EXTRA_CMAKE_FLAGS` override is required: the script hardcodes a bare `-DCMAKE_HIP_FLAGS="-mllvm"`
  which CMake ≥4.3 breaks on.)
* Fast loop: `cmake --build build-rocm --target llama-cli llama-bench test-backend-ops -j 16`.
* Models: `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf`,
  `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`, `/home/stew675/Qwen3.5-4B-Q8_0.gguf` (purity).
* Stock reference build: `/home/stew675/stock-790/build-stock/bin` (base `790cf51aa`).
* Delivery repo: `/home/stew675/llama-cpp-rdna-boosts` (branch `main`, remote GitHub, tags r1/r2/r3).

### halo — gfx1151 (Strix Halo), ssh `halo`, 123 GB unified
* ROCm: `/opt/rocm-7.14-gfx1151`. Build script `~/bin/build-llama-rocm-714` (no `EXTRA_CMAKE_FLAGS`
  needed there). 4B/9B models:
  `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf`, `/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf`.
* **halo cannot reach GitHub** (no ssh key) — transfer files with `scp` from soar.
* `~/todo21/src` on halo = a fresh clone at `790cf51aa` + the r3 patch set applied via
  `scripts/apply-all.sh` (applied tree **`eb5b7583`** == `release.json`) + the r4-candidate diff, built
  into `build-todo21`. **This is a ready gfx1151 test bed** for the r4 candidate; re-apply + rebuild for
  anything newer.
* `~/todo21/delivery` on halo = a copy of the r3 `patches/` + `scripts/` + `release.json`.
  (halo's `~/llama-cpp-rdna-boosts` is **stale** — don't use it.)
* halo's `~/llama.cpp` is an old tip (no canonical `a2c8d06a7`); build from `~/todo21/src` instead.

---

## 2. What is validated and promotion-ready (the r4 candidate)

One diff: **`archive/work/issue-30-mtp-decode-regression/patches/2026-09-14-todo21-prefill-arena-staging.diff`**
(also copied byte-for-byte as `…-q4_0-mixed-kv-contract-and-scratch.diff`; same 7 files, 352 lines).
It contains **three** changes:

1. **The mixed-K/V NaN fix.** The tile kernel is instantiated with ONE `type_KV` for both operands and
   ignores the launcher's runtime native-type arguments; `launch_fattn` chose its native read **per
   tensor**. A mixed pair therefore fell back to the `F16` tile while the launcher skipped staging →
   the kernel read raw q4_0 as F16. `launch_fattn` now takes the kernel's native type
   (`kv_native_kernel`; tile passes its `type_KV`, vec passes `NONE`, MMA defaults to
   `FATTN_KV_NATIVE_PER_OPERAND`).
2. **The `get_alloc_size` q4_0 fix.** The TILE case was updated for the q8_0 arm but never q4_0, so a
   q4_0 cache reserved the F16 scratch the launcher no longer used: `-c 196608` q4_0 **849.04 → 123.04
   MiB**. The case now mirrors `ggml_cuda_flash_attn_ext_tile_case_type` exactly.
3. **The prefill band split + arena + arch gate.** `native_width = Q->ne[1] <= 8 || !prefill_stages`;
   `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)`. The prefill staging scratch comes from
   `ggml_backend_cuda_context::fattn_stage_get()` (a per-stream, 25 %-growth arena) instead of the
   compute-graph reserve (which sized it for `n_ctx` — the adaptive-MTP load failure). Bounded by
   `GGML_CUDA_FA_STAGE_MAX_MB` (default 512, 0 = unbounded).

### Numbers (all reproducible; logs in §5)

**gfx1201** — 27B UD-Q4_K_XL, q8_0 K/V, `pp150000`: **691.4** (1 GPU), **1076.9** (2-card `-sm tensor`),
**1199.0** (3-card) vs the old native 661.0 / 996.0 / 1111.4 and node-staging 690.4 / 1080.1 / 1203.6
(f16 703.4 / 1087.5 / 1218.6; stock q8_0 669.5 / 997.9 / 1085.8). q4_0 `pp150000` **694.5**;
q4_0 `tg64` 28.62 / 25.34 / 22.73 at d0/d32k/d65k. Reserve at `-c 196608`: q8_0 **123.04 MiB**,
q4_0 **123.04 MiB** (was 849.04), f16 122.77. Adaptive MTP `--spec-type draft-mtp-adaptive
--spec-draft-n-max 12 -c 196608 -ctk/ctv q8_0` **loads and generates (72.9 t/s)**. Same-seed greedy text
(staged vs native, ~4k prompt, `-n 128`, seed 42): q8_0 `472b282950b5`, q4_0 `118eb7f5fe85`, f16
`70960317a203` — identical both ways.

**gfx1201 correctness:** `test-backend-ops -o FLASH_ATTN_EXT` **5951/5951**.

**gfx1151** — 9B Q8_0: `test-backend-ops` **5951/5951**; width purity PURE for q4_0/q4_1/q8_0/f16;
`-c 196608` q8_0 reserve **89.04 MiB**; `pp16384` **1420.03**, `pp20480` **1373.26** (native-or-better —
this is the arch gate working). Native vs arena there: pp16k 1409.98/1404.53, pp20k 1368.92/1359.18,
pp32k 1266.30/1253.09, pp65k 1059.14/1042.86 → **no crossover**, native wins at every depth, so
RDNA3_5 keeps native prefill.

### Promotion recipe (the fork's block-15 amendment)

The r4 candidate belongs in **block 15** (the V4/V5 attention-memory block, whose tip `a2c8d06a7` is
HEAD, so it is a tip `--amend`, not a mid-chain replay):

```bash
cd ~/llama.cpp
git checkout .                                        # drop the Item 2 WIP (see §3)
git apply ~/llama-cpp-rdna-boosts/archive/work/issue-30-mtp-decode-regression/patches/2026-09-14-todo21-prefill-arena-staging.diff
git add -A && git commit --amend --no-edit            # rewrites block 15 -> new tip
TIP=$(git rev-parse HEAD); TREE=$(git rev-parse HEAD^{tree})
cd ~/llama-cpp-rdna-boosts
bash scripts/make-patches.sh /home/stew675/llama.cpp 790cf51aa "$TIP"
git diff "$TIP_BASE..$TIP" > rdna-boosts-all.patch    # TIP_BASE = 790cf51aa
bash scripts/make-release.sh --tip "$TIP" --tree "$TREE" --release v16-790cf51aa-r4
bash scripts/validate-set.sh                          # must PASS (strict 16/16 git am, tree asserted)
```

Then: update the docs (`WORKLOG.md` new dated entry at the **top**, `patches/README.md` block-15
amendment section, the `README.md`/`MANIFESTS.md`/`AGENTS.md` current-state headers, `BASELINE.md`),
commit + push `main`, tag + push `v16-790cf51aa-r4`. Then fill in the tag/sha in
`~/issue-30-followup-response.md` and post it.
**Reminder:** never push from `~/llama.cpp` (see AGENTS.md "Pushing policy").

---

## 3. The fork working tree right now (important)

`git status` in `~/llama.cpp` shows **8 modified files** = the r4 candidate (7) **plus** the broken
Item 2 additions. To get back to a clean promotion base:

```bash
cd ~/llama.cpp && git checkout .        # back to a2c8d06a7 (r3 tree)
```

The validated r4 candidate is fully preserved in the diff file named in §2 — do **not** promote from the
live working tree. (The extra 8th file is `ggml/src/ggml-cuda/fattn-tile.cu`; the r4 candidate's 7 are
`common.cuh`, `fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`, `fattn.cu`,
`ggml-cuda.cu`.)

---

## 4. The one open work item: Item 2 (native arms for q4_1/q5_0/q5_1/iq4_nl)

**Goal:** these four types have no native FA arm, so they stage through F16 and sit ~12-16 % behind f16 at
d32k (they are otherwise well supported and `W=1..8`-pure). A native arm should recover that plus remove
their staging scratch. **This is a throughput/memory item, not a correctness one** — and per the
maintainer decision (`GREEDY-PURITY.md` §36) it is explicitly *not* a purity goal.

**What is already written** (compiles; staging path unaffected — `GGML_CUDA_FA_KV_NATIVE=0` still gives
5951/5951):

* `ggml_cuda_fattn_dequantize_{q4_1,q5_0,q5_1,iq4_nl}_chunk()` in `fattn-common.cuh` — FP32 arithmetic +
  a single F16 cast, mirroring `convert.cu`'s `dequantize_block_q4_1` / `dequantize_block_cont_cuda<…,
  dequantize_q5_0/1>` / `dequantize_row_iq4_nl_cuda` and `dequantize.cuh`'s `dequantize_q5_0/1`.
* `FATTN_KV_NATIVE_{Q4_1,Q5_0,Q5_1,IQ4_NL}`; per-type `ggml_cuda_fattn_kv_<t>_supported()` + a shared
  `ggml_cuda_fattn_kv_layout_ok()`; a generic `ggml_cuda_fattn_tile_kv_native_type(K,V)` ("both operands
  the same native type"); `ggml_cuda_fattn_native_type_from_kernel<type_KV>()` +
  `ggml_cuda_fattn_native_ggml_type()`; the extended `ggml_cuda_fattn_kv_native_type()` chain.
* MMA + tile loader dispatch chains; the tile instantiation switch in `fattn-tile.cu`; the
  `get_alloc_size` TILE case now derives `tile_type` from the shared predicate.

**Why it fails — two symptoms, both reproducible:**

1. `test-backend-ops -o FLASH_ATTN_EXT` = **4703/5951**. *Every* failure is `hsk=72`
   (`hsk_padded=96`, `nr23=[4,1]`, `nb=32/75`, `mask=0`) and *every* one is a pure (same-type)
   `q4_1`/`q5_0`/`q5_1`/`iq4_nl` case. With `GGML_CUDA_FA_KV_NATIVE=0` → **5951/5951**.
2. Width probe, 4B, `P=64` (prefill): with the arms enabled, **all four types return the same hash
   `ad42a04252969383`** — consistent with the staged K/V coming back zeroed. With
   `GGML_CUDA_FA_KV_NATIVE=0` the hashes are type-specific and correct (q4_1 `9d5b230a253c0ecb`, q5_0
   `90a885ba0f7c9259`, q5_1 `77b5f1c579e0fb7f`, iq4_nl `3868fb0799fcd7ed`). Controls at the same shape:
   f16 `dc2046586db523ea`, q8_0 `2d490b3613fb9097`, q4_0 `d2a57d1628eec6a4` (the q4_0 arm is still fine).

**The key clue:** `P=64` is a **prefill** (`n_q = 64 > 8`), so this build *stages* these types (into the
new arena) rather than reading them natively — yet the staged result is wrong, while the *same types*
staged into the node scratch (`KV_NATIVE=0`) are right. So the first suspect is the **launcher's
staging branch / its agreement with `get_alloc_size` for the new types**, not the chunk
dequantizers.

**Next step (one debug build, one run — do this first):** add a temporary
`GGML_LOG_INFO("[FATTN] type=%d need_f16=%d kv_native=%d use_native=%d stage=%d contig=%d nelem=%lld
fn=%p fn_nc=%p\n", …)` in `launch_fattn` (gated by `getenv`) printing, for K and V: `K->type`,
`need_f16_K`, `kv_native_K`, `use_native_K`, `stage_K`, `ggml_is_contiguously_allocated(K)`,
`ggml_nelements(K)`, `ggml_get_to_fp16_cuda(K->type)` and `ggml_get_to_fp16_nc_cuda(K->type)`. Lead
candidate to check first: whether `ggml_get_to_fp16_nc_cuda()` even has entries for
`q4_1`/`q5_0`/`q5_1` (the AGENTS notes say only `iq4_nl` got non-contiguous converters added), and
whether the contiguous/non-contiguous branch is taken as expected — then whether `stage_K` is actually
true when the probe stages.

**Gate before promoting:** `test-backend-ops -o FLASH_ATTN_EXT` 5951/5951 on gfx1201 **and** gfx1151;
text `native == staging` per type; `W=1..8` one hash per type; `plain == draft-mtp`; and *then*
performance (see §6).

---

## 5. File inventory (all under `archive/work/issue-30-mtp-decode-regression/` unless noted)

* `README.md` — the action register (A–E) and the reconciliation story.
* `MEASUREMENTS.md` — the evidence: §A KV-type x depth, §B the q8_0/q4_0 V4 audit, §C adaptive-MTP at
  high context, §D deep-prefill, §E #28867, **§F** the prefill band split, **§G** the reporter's q4_0 NaN
  + the purity finding, **§H** the gfx1151 validation + the arch gate.
* `RECURRENT-SNAPSHOT-BUDGET.md` — the adaptive-MTP snapshot-budget deep dive (headroom work, the
  f32→bf16 opt-in to *measure* before offering).
* `patches/` — the diffs:
  * `2026-09-14-todo21-prefill-arena-staging.diff` == `2026-09-14-q4_0-mixed-kv-contract-and-scratch.diff`
    → **the r4 candidate** (352 lines, 7 files).
  * `2026-09-14-item2-native-arms-WIP-BROKEN.diff` → **the broken Item 2 WIP** (727 lines).
  * `2026-09-14-v4-default-plus-q4_0-native.diff` → the already-promoted r2 change (historical).
  * `2026-09-14-prefill-rdna-config-and-ncols2.diff` / `…-arch-split.diff` → the already-promoted r3
    change (historical).
* `tools/` — `bench_mtp.py`, `depth_sweep.sh`, `run_depth_audit.sh`, `runarm.sh`, `text_gate.sh`,
  `vwidth.sh`, `width_matrix.sh` + `width-matrix.cpp`, `purity.sh`, `sweep_nwarps.sh`,
  `sweep_pertype.sh`. Ad-hoc session scripts live in `/home/stew675/wip-issue30/` (`build.sh`,
  `pp-scan.sh`, `matrix-*.sh`, `remeasure.sh`, `adaptive-4axis*.sh`).
* Raw logs: `/home/stew675/wip-issue30/results/2026-09-14-todo21-*.txt` (perf/purity/text),
  `…-q4_0-nan.txt`, `…-item2-native-arms.txt` (includes the WIP status block). Copies of the
  `todo21-*` set are in `archive/work/issue-30-mtp-decode-regression/results/`.
* `~/issue-30-followup-response.md` — the reporter response to post after r4.

---

## 6. Gotchas and workflow rules (learned the hard way this session)

1. **Correctness first, performance last and separately.** Chasing a perf number before the axis is
   correct wastes long runs. (`test-backend-ops` and the width probe are seconds-to-minutes; `pp150000`
   is 4 minutes and only worth it once correctness is settled.)
2. **Never run parallel/background GPU benchmarks.** Builds in parallel are fine.
3. **`-sm tensor` masks single-card regressions.** Measure 1 GPU as well, always.
4. **`llama-cli` needs `--single-turn`** (`--no-display-prompt` for scripted output) or it drops into the
   chat loop and hangs. Wrap long commands in `timeout`. Log long runs to a file.
5. **Never benchmark `iq4_nl` on a stock/un-amended build** — no FA enablement there, so it runs
   host-only/CPU and never finishes.
6. **Mixed K/V cache types are hard-rejected** by llama.cpp — but `test-backend-ops` builds the op
   directly, which is how the mixed-K/V NaN was found. Use the op test as an oracle; use the width probe
   as a *detector*, not a certification.
7. **The width probe needs a constant `n_ctx`** — the shipped `width-matrix.cpp` varies it with `W`; use
   `/tmp/wm-fixed.cpp` on soar (copied to halo as `/tmp/wm-fixed.cpp`) which pins it.
8. **gfx1201 vs gfx1151 differ in kind, not just degree**: gfx1151 has no prefill crossover (native wins
   at every depth there) while gfx1201 staging wins; don't assume a gfx1201 result transfers.
9. **The reserve graph's `K->ne[1]` is `n_ctx`**, not the prefix (verified: 8192 at `-c 8192`, 196608 at
   `-c 196608`) — that is the root of the "scratch costs 726 MiB at load" problem, and why the arena
   exists.
10. **A multi-token graph is never CUDA-graph captured** in this tree (block 11's prefill skip), which is
    what makes a runtime allocation at prefill safe. Decode *is* captured — hence native (no scratch) at
    decode.
11. **The purity tiering is deliberate and permanent** (`GREEDY-PURITY.md` §36): f16/bf16/q8_0 guaranteed;
    q4_0/q4_1/q5_0/q5_1/iq4_nl best-effort. Don't re-open it as a defect hunt.
