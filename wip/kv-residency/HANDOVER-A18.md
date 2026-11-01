# HANDOVER for A18 — the KV-residency campaign's remaining items

Written 2026-10-31 at the end of session **A17** (the r194 corner-case session).  This file supersedes
[`HANDOVER.md`](HANDOVER.md) as the cold start; that one is the A-era record of how Goals A and B were built and
is still the place to look for their numbers.  [`COLD-START-PROMPT.md`](COLD-START-PROMPT.md) is the block to
paste into a fresh session and it points here.

**Read §1, §2, §5 (rules) and the per-item plan you pick before touching anything.**

---

## 1. Where things stand

* **Goals A, B and C are delivered**, all inside **block 15** of the delivery set: the resident window
  (`LLAMA_KV_RESIDENT_CELLS`, r178), the attendance pool (`LLAMA_KV_REGION_POOL`, r190), the streaming-arm
  policy that makes a streamed cache take the sparse decode arm (r190), and the MTP draft window
  (`LLAMA_SPEC_DRAFT_WINDOW`, r193, default ON) with the embedded-MTP slab reserve derivation.
* **The current release is `v16-6050b467c-r194`** (16 blocks, tree `977faeaa95f0284957805ac8e881573ff3410c67`).
  r194 is three corner fixes off a configuration battery: the pool is capped at the cache's cell count, the
  draft window no longer sizes a **separate** draft's reserve, and the VRAM-cliff headroom fact is documented.
* **What is left is four open items plus two adjacent ones**, §3.  None of them is a bug in what shipped:
  two are follow-ups a user would notice (memory pinned in full, a 2 ms top-k), one is hardware-blocked, and
  one is a pre-existing abort in an unrelated combination (`-nkvo` + host experts).

**A18 progress (this session).**  #123 and #125 were implemented and gated on their own branches
(`wip-kv-residency-123` `d51aedd93`, `wip-kv-residency-125` `83513d733`) and **promoted in r196** (blocks 15 and
11).  Records: [`C1-MIRROR-GROW.md`](C1-MIRROR-GROW.md) (the mirror pins 72 MiB of 3264 at a 5.3k run in
`-c 262144`; 1296 of 1632 at 103k, text identical to r194) and [`C2-RADIX-TOPK.md`](C2-RADIX-TOPK.md) (the op
was already a radix select; plain decode was on its cell-level form, now the block path: **2.062 -> 1.143
ms/token**, +4.0 % plain decode, MTP unchanged).  Superseded as the cold start by
[`HANDOVER-A19.md`](HANDOVER-A19.md).

## 2. What is delivered — do not redo any of it

| delivered | knob / feature | record |
|---|---|---|
| Goal A, the resident window (r178) | `LLAMA_KV_RESIDENT_CELLS=<cells>\|auto`, one device only (`-sm tensor` declines) | `patches/README.md` block 15 "The window", `A2-MIXED-BACKING.md` §3.7, `WORKLOG` 261030.002/.003/.005 |
| Goal B, the attendance pool (r190) | `LLAMA_KV_REGION_POOL=<cells per layer>`, qwen4exp trunk only | `patches/README.md` block 15 "The attendance pool", `BA-POOL-DESIGN.md` §7-§10 |
| the streaming decode arm (r190) | automatic; `LLAMA_QSA_DENSE_DECODE_STREAM=0` is the A/B arm | block 15 "r190 — a STREAMED cache takes the sparse decode arm" |
| Goal C, the MTP draft window (r193) | `LLAMA_SPEC_DRAFT_WINDOW` (default `auto` = ctx/4 clamped to [32768, 65536], `0` = off) | `O3-DRAFTER-RING.md` §2.3, `WORKLOG` 261031.014 |
| Goal A's `auto` policy (decided) | **P1, fittability only**; P2 rejected, P3 parked | `A2-MIXED-BACKING.md` §3.7.14, `TODO` #122 |
| r194 corner fixes | pool clamp; reserve scope; headroom 1536 on the cliff | `WORKLOG` 261031.015, block 15 "r194" |

Numbers already established, so a new session does not have to re-derive them: the pool sits **on the sparse
ceiling** at 5.3k / 103k / 180k (70.3 / 54.2 / 43.1 t/s with MTP-4, against dense-resident 70.9 / 58.3 / 48.7),
its own machinery is ~0.1 ms/token, and the MTP draft window is **+2.8 GiB of arena** on a `-ncmoe` MoE
(35B-A3B: arena 20 910 -> 23 715 MiB, decode 111.2 -> 113.7 t/s) and neutral on dense.

## 3. What is left, in the order I would take it

| item | what | size | blocked by |
|---|---|---|---|
| **#123** grow-on-demand pool mirror | the mirror is pinned in full at creation (3264 MiB at `-c 262144` q8_0) | M | nothing |
| **#125** the radix top-k | 0.9 ms per verify step under MTP, 2.2 ms/token plain at 103k on a sparse/pooled qwen4exp cache | S-M | nothing |
| **#129** `-nkvo` + host experts aborts | pre-existing (`r191` too), `GGML_ASSERT(id >= 0 && id < n_expert)` in the expert-cache fallback | S-M | nothing, but it is not block-15 work |
| **#124** Goal A across `-sm tensor` | designed only | L | **hardware** (one card here) |
| **#120** the r187 KV rotations are `q8_0`-only | `q4_0`/`q5_0`/`iq4_nl`/`bf16` keep their separate launches | M | nothing (block 10, adjacent) |
| **#127** `cpy_2d_vec16` is F32-only | the vectorised 2-D fallback covers F32 / 16-byte-aligned | S | nothing (block 10, adjacent) |

Two **new candidates** that came out of A17 and are not yet TODO rows (decide whether they are worth a row):

* **The qwen4exp separate MTP draft is not windowed.**  Goal C's window lives in `create_memory`'s plain
  `llama_kv_cache` branch, so it applies to an **embedded** MTP head only (qwen3.5/3.6/3.8).  Flash-Next's head
  is a `-md` GGUF with its own cache and its own indexer, and it keeps the full `-c`.  Extending the window to it
  is a different access pattern (KV injection on the main scheduler + a decoder graph on the alt one) and has to
  be measured, not assumed.
* **A pool-size policy.**  Measured in A17: `=16416` hits 97.9 % at a 5 246-token depth and 86.0 % at 102 800,
  and doubling it to `=32768` bought 0.2 points and no speed.  So the fixed cell count is fine at the shape we
  run, but nobody has asked what it should be for a 200k+ working set.

### 3.1 `TODO` #123 — grow-on-demand pool mirror

**The problem.**  `ggml_backend_cuda_kv_pool_mirror_buffer_type_alloc_buffer` is one `cudaMallocHost` of the
cache's whole cell-indexed size (`ggml-cuda.cu`, near `ggml_backend_cuda_kv_pool_mirror_buft`), taken when the
cache is created: 3264 MiB at `-c 262144` with q8_0 KV (the `TODO` row's 4.9 GiB is the f16 number), against a
204 MiB device pool.  A short run in a big `-c` therefore pins the whole context for nothing.
`LLAMA_KV_SPILL_RAM_PCT` (85 %) bounds it and the pool declines loudly rather than being OOM-killed, which is
why it shipped as a limitation in r190.

**The model to copy** is block 15's own host half: `llama_kv_cache`'s `grow_cells`/`grow_for` grow the windowed
KV's host backing as the watermark rises (`src/llama-kv-cache.cpp`), and the region provider
(`ggml_cuda_vmm_region`) already knows how to reserve VA and map **pinned host** chunks lazily
(`MIXED-BACKING.md`, `llama_kv_region_spill_reserve`).  The mirror is the same shape: reserve the VA, map a
chunk per grow step, never unmap (the mirror is authoritative and every cell can be read).

**Plan.**
1. Give the mirror reservation semantics: a region-backed buffer type whose `alloc_buffer` reserves and maps
   the first chunk (say 256 MiB), instead of one `cudaMallocHost`.
2. Grow it from `llama_kv_cache`'s existing watermark path (the same hook the window's host half uses), so a
   cache that never reaches depth 100k never pins 3 GiB.
3. Keep `memory_breakdown()`'s host charge correct (it reads the buffer size today) and keep
   `llama_kv_region_spill_reserve` accurate, or `--fit` stops seeing the mirror.
4. `kv_pool_invalidate()` and the fill/claim paths read the mirror by cell offset: a chunk that is not mapped
   yet must never be read, which is guaranteed if grows follow the watermark.

**Gates.** Pool text identity (`LLAMA_KV_POOL_VERIFY=1`, 0 mismatches) on a *grown* arm (a prompt that crosses
several grow steps, e.g. the 103k prompt), `LLAMA_KV_POOL_STATS` page/hit line unchanged, the three state gates
on a pooled cache, `--fit` audit (the mirror in the host bucket, the pool in VRAM), and the pinned-RAM ceiling
arm (`LLAMA_KV_SPILL_RAM_PCT=1` must still decline, not thrash).

### 3.2 `TODO` #125 — the radix top-k (the largest sparse-arm item left)

**Where it is.**  qwen4exp's sparse decode selects its attended blocks with a top-k over the indexer scores
(`src/models/qwen4exp.cpp`, the `build_qsa_topk` side; `GGML_OP_ALL_GATHER` and the pool resolve are next to
it).  A17's census work (A11, `BA-POOL-DESIGN.md` §8.4) attributes **0.9 ms per verify step under MTP** and
**2.2 ms/token** on plain decode at 103k, after A13 took the indexer score chain from 5.31 to 2.01 ms/token.

**Plan.**  Replace the sort/selection with a radix select (histogram over the top bits of the keys, then a
narrower pass), in-graph and captured-graph safe (no host readback, no transient scratch allocated per launch:
that is a hard invariant of this block, see `BA-POOL-DESIGN.md` §7.3).  Keep the selected **ids** identical to
today's for equal scores, because the pool's hit rate and the text depend on the selection.

**Gates.**  A width matrix at 1..8 query rows, the ids compared against the current kernel for the same scores
(the arm's oracle: same ids → same text), prefill-logit is *unaffected* (decode-only) but the 27B point is
cheap, `test-backend-ops` for whatever op is touched, and a `rocprofv3` census showing the 0.9 / 2.2 ms gone.
`benchmarks/prefill-logit-methodology.md` and `GREEDY-PURITY.md` §45/§63 apply if the selection changes at all.

### 3.3 `TODO` #124 — Goal A across a `-sm tensor` split (hardware-blocked)

Design only: [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) has the five changes (a Meta buft variant,
the grow fan-out, fit recognition, `region_capable`, fit sizing inside the split branch) and the four
`common/fit.cpp` findings a blind implementation gets wrong (`margins[0]` is simple device 0's target while
`free_dev` is the summed free; `kv_bytes_full[0]` is the whole logical KV; and the split `estimate()`'s
`ntd×` KV term).  Both block-15 placements decline a tensor split with one message; **do not remove those
refusals without the arms**.  Nothing here can be done on this box.

### 3.4 `TODO` #129 — `-nkvo` with host experts aborts (pre-existing, not ours)

Repro on the released r191 binary too: `-ncmoe 48 -nkvo`, qwen4exp, aborts in
`ggml_backend_sched_copy_input` (`ggml-backend.cpp:2920`, `GGML_ASSERT(id >= 0 && id < n_expert)`) while reading
the routing ids on the host in the expert-cache fallback path.  Reproduces with `LLAMA_KV_REGION_POOL=0` and
`--spec-type none`, so it is neither the pool nor Goal C.  A dense model with `-nkvo` is fine.

**Plan.**  Shrink it to a small MoE (a 4B-class MoE if one is on disk, else the 35B-A3B), then read that path:
which backend the ids tensor is on, whether the producer's stream is synced before the `get_async`, and what
`ids_backend` resolves to when the KV and the inputs are host-side.  Fix, then gate with a text arm against
`-nkv` (the KV on the device) on the same config, plus the standard coherence sweep.

### 3.5 Adjacent, block 10: `#120` and `#127`

Both are one-file changes with a measurement attached, and neither is campaign work, so take them only if the
campaign items stall.

* **#120**: the five r187 KV-rotation fusions key on the K/V cache's `SET_ROWS` being `GGML_TYPE_Q8_0`.  Each
  further type (`q4_0`, `q5_0`, `iq4_nl`, `bf16`) needs its own fused write kernel reproducing that type's
  group quantize map, plus the matching `is_row_write` predicate.  The `q8_0` arm is worth **+13.4 % tg128 /
  +10.1 % @4096** on Qwen3-30B-A3B, so a `q4_0` KV user (16 GiB cards) currently gets none of it.
* **#127**: `cpy_2d_vec16` covers F32 / 16-byte-aligned only.  Extending it to other element sizes is the
  cheap half of a regression fix nobody has needed since r189.

## 4. Environment

* Box: one R9700 (gfx1201, 32 624 MiB) plus an iGPU.  **Pin `HIP_VISIBLE_DEVICES=0` for every arm** (an
  unpinned `-sm tensor` enumerates the iGPU and aborts at model load with `invalid kernel file`).
* Build: `~/bin/build-llama-rocm` (ROCm 10.1.0 → `build-rocm/`), `~/bin/build-llama-rocm-714` for a 7.14 check
  (`build-rocm-714/`).  A `build-roc714-r191/` exists from the `#126` bisect and can be deleted.
  **Never pass `-DCMAKE_HIP_FLAGS`.**  `cmake --build build-rocm --target llama-cli llama-server -j 16` is the
  fast loop with ccache warm; keep ccache ≥ 30 GiB.
* Repos and branches:
  * delivery: `~/llama-cpp-rdna-boosts` (`main`), `release.json` = r194, tip `d7e708f08`.
  * fork: `~/llama.cpp`, branch **`rdna-boosts`** = the released tree (re-cut to `d7e708f08`), and the campaign
    branches `wip-kv-residency-c` (the Goal C + r194 work, `678e2a1d7`) and the older `wip-kv-residency`
    (Goal A measurements), `wip-kv-residency-b` (Goal B).  **Develop on a branch; never commit campaign code to
    `rdna-boosts`**, and never touch `patches/` or `release.json` outside a release.
* Models: `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS` (qwen4exp, 76 GB, has `runme` and the MTP sidecar
  `mtp-…-shared-Q8_0.gguf`), `Qwen3.8/27B/{Q8_0,Q4_K_XL}`, `Qwen3.6/35B-A3B/{Q8_0,Q4_K_M}`,
  `Qwen3.5/122B-A10B/Q4_K_XL`, `Qwen3.5/4B/Q8_0`.
* Prompts: `prompts/prose-rdna-boosts.txt` (5 256 tokens at the 27B), `prompts/kld-corpus-35b-in-domain.txt`
  (`cat` it 4x for a 102 800-token prompt, 7x for 180 035).  Both are hash-pinned in `prompts/README.md`.
* **Every Flash-Next arm carries `-ot per_layer_token_embd.*=CPU -lm none -lzm off`** (the 27 GiB PLE table
  otherwise moves identical runs by ±20 %).
* A working Flash-Next server shape (`/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/runme`): `-c 262144`,
  `-ncmoe 48`, q8_0 KV, `-np 1`, `LLAMA_KV_REGION_POOL=16416`, `GGML_CUDA_SLAB_HEADROOM_MIB=1536`, adaptive MTP
  n-max 7.  Measured on r194: 102 800-token prompt → prefill 575 t/s, decode 53.2 t/s, acceptance 0.603, pool
  hit 86 %.

## 5. Rules that cost sessions (keep them)

1. **`-n 2000`** for anything about the expert arena (`-n 128` is drowned by warm-up), and **`-ncmoe` = the
   model's full layer count** (35B-A3B and Flash-Next 48, 122B-A10B 48).
2. **Interleave arms** in one batch; this box's clocks drift.  Never two GPU jobs at once.
3. **A speculative arm's t/s is (tokens/step) x (step rate)**: check `draft_n`/`draft_n_accepted` before reading
   a speed delta, and use `-n 512` (or more), never a short window.
4. **Static MTP depth ≤ 4** for the campaign's reference arms; the delivery's tuned controller is
   `--spec-type draft-mtp-adaptive` (the runme uses n-max 7).
5. **Quote the device count** with every MTP number.  The `AGENTS.md` reference arm is **3 devices** (dense 27B
   layer 51.7 / tensor 90.7 t/s); this box measures 1 device.
6. **`ROCBLAS_USE_HIPBLASLT=0`** for every fresh-process numeric comparison (the library picks its solution per
   process start), and never judge run-to-run determinism on a greedy text hash.
7. `llama-cli` prints its accounting **only with `-v`**, and `-v` interleaves log text into the generation: hash
   text from a run **without** `-v` (`scripts/extract-generated.py`), take timings from a separate `-v` run.
8. Always `--single-turn --no-display-prompt`, always `timeout`, and check the journal with
   `scripts/check-gpu-faults.sh` rather than grepping a log for `page fault`.

## 6. Gates — the ones this campaign owes

* **Pool**: text identity against the pool-off arm, `LLAMA_KV_POOL_VERIFY=1` (0 mismatches),
  `LLAMA_KV_POOL_STATS=<path>` (a path, never stderr), `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`,
  `gate-state-pipe.sh` on a **pooled** cache, `test-backend-ops -o FLASH_ATTN_QSA`, and `check-gpu-faults.sh`.
* **Window / draft window**: text identity on a spilling arm, the standard prefill-logit point
  (`scripts/gate-prefill-logits.sh`, 0.000000 / 100.000 %), and for the draft window the MTP Protocol A arm
  (`-n 3000`, reasoning pinned, both split modes, device count quoted).
* **Delivery**: `scripts/validate-set.sh` (16/16 strict apply + tree equality), `gate-per-block-build.sh
  --changed-from <previous tip>`, and `scripts/gate-prefill-logits.sh` on the 27B Q8_0 point before any tag.
  A base is per ROCm line now, and a cross-toolchain compare is refused (that was `#126`).

## 7. Traps found the hard way (do not re-learn)

* **`gate-ckpt-reuse.sh` only passes `--spec-type` when `DRAFT` is set.**  For an embedded-head MTP arm pass
  `EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 3"` (state and park gates take the same trick).
* **`gate-park-idle-slots.sh` defaults `DRAFT` to the Flash-Next sidecar.**  For any other model pass `DRAFT=`
  explicitly or the MTP context fails on `token_embd.weight`'s shape.
* **A windowed draft needs `-c > 32768`**, else the adaptive policy correctly declines and the gate proves
  nothing about windows.
* **At the VRAM cliff** (`Flash-Next`, `-c 262144`, `-ncmoe 48`) the 768 MiB default slab headroom is one
  hipBLASLt workspace short: the first prefill fails with `hipBLASLt out of memory` while the server stays
  healthy, and it is order-dependent, so it looks flaky.  `GGML_CUDA_SLAB_HEADROOM_MIB=1536` fixed it 3/3.
* **A harness that kills a subshell leaves the server running**: kill the server pid, and wait for the port to
  be free before the next arm (two "hangs" in A17 were the harness, not the code).
* **`git push` over ssh fails on this box** (no working key): push with
  `git push --force-with-lease=rdna-boosts:<old-tip> https://github.com/stew675/llama.cpp.git rdna-boosts`.

## 8. Records

| file | what it holds |
|---|---|
| [`BA-POOL-DESIGN.md`](BA-POOL-DESIGN.md) | the pool: design, invariants (mark-then-claim, distinct-block claims, no transient scratch in a captured graph), §7-§10 measured |
| [`B1-STRATA-PLAN.md`](B1-STRATA-PLAN.md) | Goal B's plan of record, the placement matrix, why a recency window cannot stand in for attendance |
| [`A2-MIXED-BACKING.md`](A2-MIXED-BACKING.md) | the window: design, purity hunt (§3.7.8), spill cost (§3.7.9), the re-measurement (§3.7.11.4), the `auto` decision (§3.7.14) |
| [`A6-TENSOR-SPLIT-WINDOW.md`](A6-TENSOR-SPLIT-WINDOW.md) | the tensor-split design (`#124`) |
| [`O3-DRAFTER-RING.md`](O3-DRAFTER-RING.md) | the MTP draft window (§2.3 is the delivered session) |
| [`C1-MIRROR-GROW.md`](C1-MIRROR-GROW.md) | **A18**: the attendance pool's mirror grown on demand (#123) — promoted in r196 |
| [`C2-RADIX-TOPK.md`](C2-RADIX-TOPK.md) | **A18**: plain decode takes the indexer top-k's block radix path (#125) — promoted in r196 |
| [`HANDOVER.md`](HANDOVER.md) | the A-era cold start (Goals A and B), still the source for their commands |
| `patches/README.md` block 15 | the shipped description, including the r194 corner fixes |
| `WORKLOG.md` 261030.002/.003/.005, 261031.007-.009, .013-.015 | the dated delivery records |

## 9. Suggested order for A18

1. **`#123`** — **PROMOTED in r196** (see `C1-MIRROR-GROW.md`); the shipped arm is the park decline (saves
   the pinned RAM); growing the mirror to the park bound is the alternative if parking speed is preferred.
2. **`#125`** — **PROMOTED in r196** (see `C2-RADIX-TOPK.md`); the MTP half was already the block path, so
   the win is plain decode only.
3. **`#129`** — separate it from the campaign: it is a pre-existing abort in a combination nobody validated.
4. **`#120`/`#127`** if the session has time.
5. **`#124`** only on a multi-device host.
