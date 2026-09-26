# HANDOVER — general-purpose `mmb` (bf16/i8-WMMA dequant weight GEMM) + QSA/Q8_0 next steps

**Date:** 2026-09-20 (sessions 1-18).  **Status:** ACTIVE WIP, not part of the delivery, and the
code is **not** pushed to any llama.cpp fork.  This document is the self-contained entry point for
the next session; the "FOR THE NEXT SESSION" brief below is the whole handoff, and the UPDATE
sections after it are the dated history (newest first).  `README.md` is the running record and
`BF16-PRODUCER-PORT.md` the producer-port reference map.

> **2026-09-25 (r7) re-base:** the beta set is now re-cut onto delivery `v16-84e76d8a2-r7`
> (`7726e514…`, the block-14 Meta-tensor-split scheduler race fix), applied tree `24bb0f5acb…`.  The
> patch bodies are byte-identical to the r6-based set; only the `From` lines and `commits.txt`
> changed.  It was on r6's `1df5769c…` before, on r5's `469082e4…` before that, on r4's `70cc895a…`
> before that, on r3's `0daefe22…` before that, and on upstream master `84e76d8a2` before that.  See
> `README.md` and `BETA-TESTING.md`; the r12/r13 hashes in this handoff are the historical record.

> **2026-09-25 (r6) re-base:** the beta set is now re-cut onto delivery `v16-84e76d8a2-r6`
> (`504894e6…`, the block-15 bf16 native default flip), applied tree `1df5769c…`.  The patch bodies
> are byte-identical to the r5-based set; only the `From` lines and `commits.txt` changed.  It was on
> r5's `469082e4…` before, on r4's `70cc895a…` before that, on r3's `0daefe22…` before that, and on
> upstream master `84e76d8a2` before that.  See `README.md` and `BETA-TESTING.md`; the r12/r13 hashes
> in this handoff are the historical record.

> **2026-09-25 (r5) re-base:** the beta set is now re-cut onto delivery `v16-84e76d8a2-r5`
> (`de86c5e1…`, the block-15 follow-up that puts f16/bf16 on the RDNA4 GQA-6 decode/verify FA band),
> applied tree `469082e4…`.  The patch bodies are byte-identical to the r4-based set; only the `From`
> lines and `commits.txt` changed.  It was on r4's `70cc895a…` before, on r3's `0daefe22…` before
> that, and on upstream master `84e76d8a2` before that.  See `README.md` and `BETA-TESTING.md`; the
> r12/r13 hashes in this handoff are the historical record.

> **2026-09-25 (r4) re-base:** the beta set is now re-based onto delivery `v16-84e76d8a2-r4`
> (`5938da09…`, the block-15 RDNA4 GQA-6 decode/verify FA band), applied tree `70cc895a…` (no conflict
> resolution; the beta set does not touch the dense FA files r4 changed).  It was on r3's `0daefe22…`
> before, and on upstream master `84e76d8a2` before that.  See `README.md` and `BETA-TESTING.md`; the
> r12/r13 hashes in this handoff are the historical record.

> **2026-09-25 — the set is now consolidated and r13-based.**  The `closing-the-gap` campaign was
> folded into `patches/` (28 patches: the 12 core + 16 appended, ten closing patches folded into the
> core), and the whole set was re-based on the **r13** delivery tree (`bb7b6d07…`).  The apply recipe
> is r13 + strict `git am patches/*.patch` → tree **`468c6496…`**; see `README.md` and
> `wip/closing-the-gap/consolidation.md`.  The r12-era branch/base details below are historical.

---

## Branch and workflow — READ THIS FIRST (set 2026-09-20)

The WIP lives on **two dedicated branches**.  **New work goes to those branches, never to `main`.**

| what | where | branch | base |
|---|---|---|---|
| **code** | fork worktree `~/llama-wip-mmb` (a worktree of `~/llama.cpp`) | **`wip-mmb-general`** | `8a2567e1e` (the applied delivery tree) |
| **record + backup** | this repo (`llama-cpp-rdna-boosts`) | **`wip-mmb-general`** | cut from `main` at `1c2ec00` |

* The delivery-repo branch `wip-mmb-general` was cut from `main` at **`1c2ec00`**, **rebased onto
  `main` r12 (`4e37fa6`) on 2026-09-20**, and is pushed to `origin/wip-mmb-general` (tracking).  The
  code branch `wip-mmb-general` in `~/llama-wip-mmb` was rebased in the same pass: from the old
  applied delivery tree `8a2567e1e` onto the **r12 applied tree
  `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`** (`git rebase --onto <r12> 8a2567e1e wip-mmb-general`,
  38/38 clean, no conflicts).  The r12 delivery tree was rebuilt with
  `RDNA_BRANCH=r12-verify scripts/apply-all.sh` on a fresh `ebbb18522` worktree (tree verified ==
  `release.json`).  **All gates re-verified green on the rebased tree.**  **`main` is frozen for this work** — it keeps all
  the pre-existing WIP history exactly as it was, and is not committed to again until the maintainer
  calls the rebase.
* **Every new session:** `git -C ~/llama-cpp-rdna-boosts switch wip-mmb-general` (check
  `git branch --show-current` before committing) and commit the WIP record **there**.  Never commit
  the WIP record to `main`.
* The fork worktree stays on `wip-mmb-general`; regenerate the backup (`mmb-general.patch` +
  `patches/` + `commits.txt`) from it as usual and commit that to `wip-mmb-general`.
* **Rebase later, only on the maintainer's word.**  **DONE 2026-09-20 (r12).**  The record branch was
  `git rebase --onto 4e37fa6 1c2ec00 wip-mmb-general` (17/17; two AGENTS.md conflicts resolved by
  regenerating the file from r12's content plus this branch's WIP-branch pointers) and the code branch
  `git rebase --onto <r12 tree> 8a2567e1e wip-mmb-general` (38/38, no conflicts).  For the *next*
  rebase the pattern is: `git rebase --onto <new-main> 4e37fa6 wip-mmb-general` for the record and
  `git -C ~/llama-wip-mmb rebase --onto <new-delivery-base> <current-r12-base> wip-mmb-general` for
  the code; then regenerate the backup from the new base and re-run the gates.

---

## FOR THE NEXT SESSION — start here

**Session start checklist (do this first).**

1. `git -C ~/llama-cpp-rdna-boosts switch wip-mmb-general` — verify with `git branch --show-current`
   (must be **`wip-mmb-general`**, never `main`).
2. `cd ~/llama-wip-mmb`; confirm `git status` clean and `git log --oneline -1` = **`aa55dfef8`**.
3. Build with the §3 command; run the **§Gates** before/after any numerics-touching change (all
   currently green: PPL **10.6015**, greedy **`9c281c415082`**, width probe **PASS**).
4. Pick an item from **Remaining work** below.  The current mandate is the **IQ3_S dequant follow-up
   brief** (`### F. MMB restructure` → "Follow-up brief for a fresh session: the IQ3_S dequant body").
5. Commit the record to `wip-mmb-general` and push (`origin/wip-mmb-general`).

**Working from another machine / architecture?  Read `GROUPS.md` first** — it is the 5-group triage
sheet (what each group is, which parts are gated, what to check on gfx1100/gfx1201, and the gates).

**Mandate — the MMB line is DONE (session 24); what is left is promotion.**  The campaign's prefill
work is complete.  Landed in order: the general-purpose `mmb` weight GEMM; the QSA v3 path; the
bf16-producer port (incl. the `xn` stream); the `dsv4_hc`/concat/moe/unary non-temporal wins; the F32
split and the tiny-M kernel; the sessions 16–18 **indexer top-k** (2.94 % -> **1.47 %** of pp32768);
the sessions 21–22 **MMB structural** work (the A-panel double-buffered IQ3_S GLU tile, the
one-instruction `v_perm` bf16 RNE pack, the **A-panel dequant split**); session 23's
**tile-class threshold** fix (the small tile re-dequantized the A panel 4x); and session 24's
**`load_regs` field preload** (the GLU's dequant ran exposed) — together **MMB family 2980 -> 2453 ms**
and **total GPU kernel 4409 -> 3879 ms (−12.0 %)** on the target model, plus **pp8192 1037 -> 1116 t/s**
end to end, all **bit-identical**.  End-to-end prefill is now **~+40 % at pp2048 / +58 % at pp8192**
against the true delivery base (`benchmarks/2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md`).  All
§Gates are green.  **The only lever left inside MMB is the 642 ms of non-dequant GLU time** (WMMA +
LDS + barriers, pinned at 1 block/CU by the LDS budget) — the `BM=32` route in the follow-up brief;
everything else below is at its practical floor.  The other open item is **promotion (item E), which
is maintainer-gated**.

**Remaining work (prioritized — the whole list as of session 24).**  **Sessions 21-24 finished the MMB
line: the A-panel double-buffered GLU tile, the one-instruction `v_perm` bf16 RNE pack, the A-panel
dequant split, the tile-class threshold, and the `load_regs` IQ3_S field preload — Flash-Next MMB
family 2980 → 2453 ms, total GPU kernel 4409 → 3879 ms (−12.0 %), pp8192 1037 → 1116 t/s, all
bit-identical.**  The dequant is now preloaded and its remaining ALU is ~200 ms of the 841 ms GLU; the
larger remaining block is the **642 ms of WMMA/LDS/barrier time**, which is pinned at 1 block/CU by the
LDS budget — the `BM=32` route in the follow-up brief in `### F. MMB restructure` is the only idea left,
and it is speculative.  Everything else below is at its practical floor, and the remaining work is
**promotion (item E), which needs the maintainer**.

* **A. Indexer — CLOSED (session 25): at its practical floor.**  The family is now **1117 ms at
  pp32768 = 1.80 %** of the run (from 2.94 % at the start of this work).  Session 25 measured the
  remaining ideas and closed them:
  1. **Block-level gather/emit — investigated, DEAD.**  Reading how `blk_cells` is built settles it:
     `llama-memory-hybrid-idx.cpp:743` writes `cur_blk_cells[blk_of[j]*r + (idx%r)] = j` with
     `idx = ranked ? rank[j] : cells.pos_get(j)` — so a block's cells sit in **`idx%r` slot order**,
     which is ascending column order only in the unranked, contiguous-pool case (the ranked path needs
     a per-block sort of `r` cells).  Worse, the **dead/spare block is unfixable**: the `if
     (blk_of[j] >= 0)` guard means every unpooled cell (`cell_blk[j] == dead_bid`, incl. the tail,
     which `blk_tail` makes *always visible* and therefore likely selected) is **absent from
     `blk_cells`**, so a block walk still needs a full `n_kv` scan to find them.  That removes the
     entire point, and the ideal outcome was only ~0.4 % of the run.  **Do not build it.**
  2. **The "skip the whole block" intermediate — measured, NO GAIN.**  Skipping the per-cell key
     evaluation for blocks whose key cannot reach the prefix is provably output-preserving (a block
     below the prefix emits nothing and contributes nothing to the carry) and was implemented and
     A/B'd: `259.9 -> 262.2 ms` — nothing.  **The gather is not memory-bound on `cell_pos`.**
  3. **What DID pay: the scan.**  Acting on (2) pointed at the barrier-heavy Hillis-Steele scan
     (8 steps x 2 `__syncthreads` = ~19 barriers per 1024-cell tile).  A warp-shuffle scan + a
     2-barrier offset pass: **gather 259.9 -> 229.6 ms (-11.7 %)**, family -2.5 %, bit-identical.
  4. **Still open but not worth it** (each <0.2 % of the run for real complexity): `histogram_blocks`
     is the biggest remaining indexer kernel at **462 ms / 41 % of the family** — it recomputes the
     per-`(row,block)` key in each of the 3 passes and re-sums `wvis` per range.  Caching the key (or a
     compact per-block bin) from pass 1 would trade ~2 loads + 6 ALU per block-visit for 1 load; and
     a per-`(row,hb)` visible-count total from pass 1 would let passes 2-4 skip the `wvis` load.  Both
     are real but small, and neither is order-sensitive — the low-risk place to look if the indexer
     is ever revisited.
  2. Small polish: template the `indexer_topk_extra` branches (`blk_idx`/`cell_pos` are run-time checks
     in the hot loop); optionally fuse `hist_accum` into `select` (blocked on `select` being per-row
     while the suffix counts are per-(row, block)).  Also open: the block passes' key-bound cost
     (462 ms at 32K) and the `select`+`hist_accum` histogram-array traffic (152 ms).
  3. **Closed/refuted — do not redo:** "compact after pass 1" (cell tie groups are 50–60 % of the
     cache); a wider radix / fewer passes (the wider `select`/`hist_accum` reads offset the saved pass);
     pass 1's per-cell histogram atomics (session 19, now one per block).
* **B. Non-temporal sweep remainder (small) -- ASSESSED AND CLOSED (session 19).**  The remaining
  candidates were re-checked against the session-19 pp8192 profile: the qsa3 pack is **6.8 ms**
  (0.04 %), `ssm_conv_long_token_f32` was measured in session 14 and is noise-dominated (and wins on
  neither load nor store), `rms_norm_f32` must read `x` **twice** (a non-temporal load would evict the
  second read -- it is the largest non-MMB kernel at 3.5 %, but not via this hint), and the remaining
  unary producers (`unary_op_kernel<relu>` 31 ms, `scale_unary` 7 ms) are too small to matter.  The
  rule if it is ever revisited: **loads only, per-kernel, A/B load vs store** (a non-temporal *store*
  evicts the next op's input).
* **C. Recorded follow-ups (not scheduled).**  `ple_embd` is the last `mmb_cvt` (its producer is the
  GGUF loader, so it needs a load-time bf16 copy) -- now measured at **0.8 ms** per the session-11
  trace, i.e. not worth the loader change; `dsv4_hc`'s `mixed` as BF16-only would save ~21 MB/
  call but changes the GDN recurrence and MoE-routing numerics (deliberately out of scope).
* **D. Delivery items to fold in at promotion (block 14).**  `GGML_OP_INDEXER_FILL` is missing from
  `GGML_OP_NAME` (one line; already fixed in the WIP by `d1463bff3`); and the new **`blk_cells`
  `src[7]`** on `GGML_OP_INDEXER_TOPK` (the session-18 interface change).
* **E. Promotion (maintainer-gated).**  Rebase onto a canonical fork rebuilt at `ebbb18522` +
  `scripts/apply-all.sh`, regenerate `patches/` + `release.json`, and decide whether MMB rides as a
  **block-08 amendment**.  Every gate passes; it needs the maintainer's go-ahead + a beta window.
* **F. MMB structural (DONE; what is left is the GLU's non-dequant 642 ms).**
  **Session 21** landed the A-panel double-buffered (IQ3_S) GLU tile; **session 22** the one-instruction
  `v_perm` bf16 RNE pack (all but the IQ3 paths) **and the A-panel dequant split** (`mmb_dq_row_iq3s_p<PARTS>`,
  which supersedes `DBUF` for that tile); **session 23** the **tile-class threshold** (`THRESH` was 128,
  so a sub-128-row expert took `ceil(cnt/32)` BN=32 tiles, **each re-running the same A-panel dequant** —
  4x the weight dequant for the same WMMA; the optimum is `THRESH == BN_SMALL == 32`); and **session 24**
  the **`load_regs` IQ3_S field preload** (measured marginal dequant cost **348 ms = 35 %** via a
  barrier-guarded duplicate; the fix moved the exposed loads into the prefetch).  Together: `mrg`
  1318 → 841 ms, `mr` 383 → 361 ms, MMB family **2980 → 2453 ms**, total GPU kernel **4409 → 3879 ms
  (−12.0 %)**, 35B family 1008 → 962 ms, pp8192 1037 → **1116 t/s**, bit-identical.  **What is left is
  the GLU's 642 ms of WMMA/LDS/barrier time at 1 block/CU** — see the follow-up brief below.

  (Original framing, for context: the MMB weight GEMMs are **61.8 % of pp8192** on the target model.
  Session 6 closed the tuning — tiles/BN/VDR/shadow/int8 are all washes or worse — and the kernel is
  already software-pipelined, but the dequant ran on the same warps between the WMMA phases.  A
  warp-specialised producer + 2x LDS ring does **not** fit (64 KB LDS); A-panel double buffering does,
  and that is what shipped.)

**Progress (sessions 16-19):** the indexer is now **~2.2x faster** and still bit-identical.  Session 16
removed the full-width count pass (the gather's per-block greater/equal counts are derived from the
radix histograms).  Session 17 added **block-key sharing** on the `additive == nullptr` (derived
default) path: since the value is per-block, the histograms and the gather evaluate the block key once
per block's cells instead of once per cell.  Session 18 added a **block-level histogram path**: the op
now carries `blk_cells` (so it knows the ratio), pass 1 is cell-level and accumulates each block's
visible-cell count, and passes 2-4 bin that count at the block key (a block-aligned partition).  Session
19 cut pass 1's histogram atomics from one **per cell** to one **per block** (the block key and the
invisible `key_inf` bin are shared across a thread's run) and the block passes' visible-count atomic
from one **per thread** to one **per warp**.  Family
**177.1 -> 71.7 ms at pp8192 (-60 %)** and **2523.6 -> 1146.6 ms at pp32768 (-55 %, 2.94 -> 1.36 % of
the run)**, output **bit-identical** (PPL 10.6015, greedy `9c281c415082`, all §11 gates green).  The
failed "compact after pass 1" experiment is recorded in the session-16 UPDATE.  **The remaining indexer
levers are the 3 block passes (472 ms at 32K, key-bound), the gather (259 ms) and `select`+`hist_accum`
(152 ms); a block-level gather/emit would need each block's cells in column order (see session 18).**

**Environment**

| | |
|---|---|
| worktree | `~/llama-wip-mmb`, branch `wip-mmb-general`, tip **`49eff7f18`** (clean) |
| base | **r12 applied tree `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`** (was `8a2567e1e` before the 2026-09-20 rebase) |
| backup | this repo: `wip/mmb-general/mmb-general.patch` + `patches/` (**5 thematic patches**, see `GROUPS.md`) + `commits.txt`, on branch **`wip-mmb-general`** (rebased onto `main` r12 `4e37fa6`), pushed to `origin/wip-mmb-general` |
| target model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (94 GiB; the only qwen4exp with HC + QSA) |
| fast iteration model | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf` (21 GiB, `qwen35moe`; **no** HC/QSA — use it for `mmb_*` shapes) |
| reference | `~/pwilkin-llama-cpp` @ `f5daaa3cf` (branch `strix-halo`) |
| runtime | `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH` |

**Build / run** (build dir is configured, ccache on; a single-file change is ~2 min):

```sh
cd ~/llama-wip-mmb && cmake --build build-rocm -j16 --target llama-bench llama-perplexity llama-cli test-backend-ops
# warm the page cache before every bench; never run benches in parallel
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for f in /llm/models/Qwen3.8/Flash-Next/IQ4_XS/*-0000*.gguf; do dd if=$f of=/dev/null bs=4M 2>/dev/null; done
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 ./build-rocm/bin/llama-bench -m "$M" \
  -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -p 8192,2048 -n 0 -r 2
```

Knobs: `GGML_CUDA_MMB=1` = the base MMB gate, `GGML_CUDA_MMB_HC16=1` = the bf16-producer port (both
added by this WIP; the committed delivery runs with them off).  **The QSA v3 path is now a
COMPILE-TIME gate** (`LLAMA_QSA3_ENABLE`, default 1) -- see the profiling caveat below; `-DLLAMA_QSA3_ENABLE=0`
compiles it out.  `LLAMA_MMB_CVT_LOG=1` lists the activation conversions, `GGML_CUDA_MMB_LOG=1` the
MMB dense shapes.

> **PROFILING CAVEAT (session 15) -- read before trusting any `rocprofv3` profile.**  The
> `rocprofiler-register` shipped with ROCm 7.14 (`librocprofiler-register.so.0.6.0`, built Jul 9)
> calls `setenv()` with `GLOG_*` variables during early init.  That can reallocate the process
> environment under a concurrent `getenv()` in the target app and intermittently make an env gate
> read as **unset**.  Measured: with `GGML_CUDA_QSA3=1`, 15 pp8192 profiles took the qsa3 path and
> later ones silently took the slow `flash_attn_qsa` one, with no reproducer pattern.  Root cause is
> **ROCm issue #10196**, fixed upstream **2026-09-15** in **rocm-systems PR #11620**; our build
> predates it.  Consequences: (a) `GGML_CUDA_QSA3` was replaced by the compile-time `LLAMA_QSA3_ENABLE`
> (session 15) so qsa3 profiling is deterministic; (b) **any env-gated WIP path (`GGML_CUDA_MMB`,
> `GGML_CUDA_MMB_HC16`, the tuning knobs) can in principle flip under the profiler** -- verify the gate
> you care about from the kernel names in the trace, and corroborate perf claims with a *non-profiled*
> t/s A/B (no profiler => no `rocprofiler-register` => no race).  The indexer kernels are not
> env-gated, so they are safe to profile.

**Gates (all must stay green; a bit-identical change reproduces the hashes exactly):**

1. PPL c2048 on `prompts/prose-rdna-boosts.txt` — current build **10.6015** (`HC16=1`, the `xn`
   BF16-only work); `HC16=0` is 10.5771.  A numerics change re-baselines it (bar ~±0.68 at c2048;
   the tight ±0.027 bar needs c16384, which the prompt file cannot reach).
2. Same-seed greedy text via `scripts/extract-generated.py` -> **`9c281c415082`** (624 chars),
   `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 llama-cli -f prompts/prose-rdna-boosts.txt
   -n 128 --seed 42 --temp 0 -c 8192 -b 2048 -ub 2048 -ctk bf16 -ctv bf16 -fa 1 -ngl 99
   --no-display-prompt --single-turn` (`HC16=0` gives `c3f24ac9c114`).  The pre-session-12
   `9930c674a6ca` was a bit-identical-era hash; re-baselined there.
3. `test-backend-ops -o FLASH_ATTN_QSA` (22/22), `-o GATED_DELTA_NET` (46/46), `-o FLASH_ATTN_EXT`,
   **`-o INDEXER_TOPK`** (the indexer op's own test -- check it still runs cases, not 0/0).
4. `test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512` -> `width_purity=PASS`.
5. `plain == draft-mtp` greedy text (do **not** pass `-md` to the plain arm) — byte-identical.
6. Judge value-changing variants on `rocprofv3` kernel time, **not** end-to-end t/s (session 5e rule),
   and heed the profiling caveat above.
7. **Indexer determinism is part of the contract**: the block-14 gather was made deterministic
   (ascending column order) because the old atomic placement let the *list order* -- and at the rank
   boundary *which tied cells made it* -- vary run to run, which the f16-state recurrences amplified
   into llama-cli-visible nondeterminism.  Any rewrite must preserve that order (the `k_bin_bcast`
   tie-break precedent) and be checked with the same-seed text gate, not just t/s.

**Next work — detail (the reference for the prioritized list at the top of this brief)**

### A. The indexer — shapes, algorithm and measured cost (items A1–A3)

**Where it is.**  `ggml/src/ggml-cuda/indexer-topk.cu` (422 lines, the whole top-k) +
`indexer-score.cu` (the score kernels) + `ggml/src/ggml-cuda/ggml-cuda.cu` (the 3 op hooks) +
`src/models/qwen4exp.cpp` (`build_qsa_top_k`, ~line 1261, op call ~1563).  **None of it is touched by
this WIP** -- it is delivery block 14 (the `GGML_OP_INDEXER_*` ops are the delivery's own; pwilkin's
tree has **no** such op, it uses the generic `top_k_nary_search_cuda` in `top-k.cu`, so there is
nothing to port -- this is genuinely our code to optimise).

**Shapes** (qwen4exp: n_embd 2560, indexer n_head 4, head_size 128, `indexer_top_k` 2048, compress
ratio r=4):

| tensor | shape | note |
|---|---|---|
| `score` | `[n_blocks, n_tps, n_stream]` F32 | n_blocks = ceil(n_kv/r) |
| `cell_blk` | `[n_kv, n_stream]` I32 | maps each cell to its block |
| `additive` | `[n_kv, n_tps, n_stream]` F16 (or F32), may be **null** | the attention mask/bias.  **Null in the default prefill path** -- derived visibility (`cell_pos`/`q_pos`) + derived per-block bias means the mask is never created, so the top-k reads no additive.  The session-15 "134 MB mask read 4x" framing is **wrong for the default config**; the cost is the key-evaluation/gather work, not that stream. |
| `dst` | `[width, n_tps, 1, n_stream]` I32 | width = min(n_kv, `indexer_top_k` + r - 1) = **2051** |

`n_kv` = context length, `n_tps` = ubatch tokens (2048 at `-ub 2048`), 24 QSA layers drive it.
`value(c) = score[cell_blk[c], t, s] + additive[c, t, s]` is computed **on the fly** in every pass (the
block-14 win: the `[n_kv, n_tps]` F32 expanded tensor -- 512 MB at 64K -- never exists).  Each
`(tps, stream)` row is an independent top-k over `n_kv` cells.

**Algorithm** (`indexer_topk_radix_cuda_blocks`, the `additive == nullptr` default;
`indexer_topk_radix_cuda` is the `additive != nullptr` / A-B fallback):

1. `indexer_topk_radix_init` -- states[row].rank = k.
2. **4 radix passes**, 8 bits each, over a **block-aligned** partition: hist-block h covers blocks
   [h*bchunk, (h+1)*bchunk) and the cells [h*bchunk*r, min((h+1)*bchunk*r, n_kv)), with
   `r = blk_cells->ne[0]/n_blocks`.  `indexer_topk_histogram_pass1` (cell-level) accumulates
   `wvis[row*n_blocks + b]`, the block's visible-cell count, and bins each cell (visible at the
   block key, invisible at -inf); `indexer_topk_histogram_blocks` (passes 2-4) walks blocks, reads
   `wvis`, and bins the count at the block key -- three of the four passes do ~`n_blocks` work instead
   of ~`n_kv`.  `indexer_topk_radix_select` narrows the prefix.  (`indexer_topk_radix_histogram_grouped`
   is the session-17 cell-level variant, used by the fallback and by `LLAMA_INDEXER_NOBLOCK=1`.)
3. `indexer_topk_hist_accum` (session 16) folds each pass's **suffix** bin counts into the gather's
   per-block `g_cnt` (greater) and, on the last pass, `e_cnt` (equal): a cell with key > the final
   prefix is greater exactly at the first byte where it differs.  **This replaced the old full-width
   `indexer_topk_count` pass.**
4. `indexer_topk_base_scan` (exclusive prefix per row), then the gather over the same cell ranges
   (`indexer_topk_write_blocks_grouped`, session 17): it shares the block key across a thread's `VEC`
   cells, scans the per-thread totals, and a shared running carry keeps the ascending-column order
   exact.

The default path therefore does **one full cell pass (pass 1) + three block passes (`n_blocks`/`n_kv`
~ 1/4) + one cell gather pass**.

**Measured (session 19, gfx1151, bf16 KV, `-ub 2048`; session-18/15 numbers in brackets):**

| kernel | pp8192 | pp32768 | calls @32K |
|---|---:|---:|---:|
| histogram pass 1 (`_pass1`, cell-level) | 13.4 (24.0) | 266.0 ms (341.8) | 384 |
| histogram passes 2-4 (`_blocks`) | 19.8 (22.4) | 461.8 ms (470.4) | 1152 |
| gather (`_grouped`) | 20.1 (20.4) | 259.4 ms (262.5) | 384 |
| `indexer_topk_radix_select` | 11.3 | 87.8 ms | 1536 |
| `indexer_topk_hist_accum` | 5.2 | 64.3 ms | 1536 |
| `base_scan` + `init` | 2.0 | 8.0 ms | 384 |
| **family total** | **71.7 ms** (85.6 / 123.7 / 160.4 / 177.1) | **1146.6 ms** (1235.7 / 1362.3 / 2028.4 / 2523.6) | |

Sessions 16+17+18+19 together: **-60 % at pp8192** and **-55 % at pp32768 (2.94 -> 1.36 % of the run)**,
output bit-identical.  Session 18's block-level histogram is the win at short context (-31 % at 8K); at
32K it is only -9 % because the session-17 grouped pass already shared the block key, so the block
passes save only the per-cell visibility/binning work (the key is still evaluated once per block).
Session 19 then cut pass 1's histogram atomics from one per cell to one per block (-44 % pass1 at 8K)
and the block passes' visible-count atomic from one per thread to one per warp.

**Why the block passes are now the remaining cost.**  At 32K the family is the 3 block passes (41 %,
key-bound: one `score` gather + `blk_idx` per block), pass 1 (23 %), the gather (22 %), select (8 %) and
hist_accum (6 %).  The block passes are only ~2.2x cheaper per element than a cell pass because the
per-block key evaluation dominates both, and the `select`/`hist_accum` pair is memory-bound on the
histogram array.

**Hypotheses, ranked:**

1. **Compact after pass 1 -- REFUTED (session 16).**  Implemented and measured: **326 ms vs the 177 ms**
   family at pp8192.  The premise "only ~1/256 of the cells can still matter" is false here: the
   per-block indexer values are heavily tied (relu zeros, whole 4-cell blocks sharing a score, and a
   causal visible count that can approach k), so the cells whose top-8 key byte shares/beats the k-th
   value's are **~50-60 % of the cache** (measured `max_clen` 0.5-0.625 x `n_kv`).  Do not retry
   without a finer first bin.
2. **Block-granularity selection -- HISTOGRAM DONE (session 18), GATHER OPEN.**  The op now carries
   **`blk_cells`** (session 18) so it knows `r = blk_cells->ne[0]/n_blocks`, and the default path uses a
   block-aligned partition with a cell-level pass 1 (which counts each block's visible cells) and three
   block-level passes 2-4.  The **gather** is still cell-level: making it block-level (emit a selected
   block's cells directly) would need each block's cells in ascending **column** order.  `blk_cells`
   gives them in `idx % r` order, which equals column order only when the cache cell index is the
   position (single sequence, no ranked/mrope reorder); otherwise the block's cells would need a small
   sort, and the dead/spare block (`blk_idx == INT32_MAX`) is not in `blk_cells` at all (it is the
   unpooled incomplete tail + empty cells, still requiring a scan).  Estimated ~262 -> ~60 ms at 32K
   (~1.2x on the family) if it can be made order-exact.
3. **Fewer radix passes.**  `RADIX_BITS` is 8 (256 bins); 16 bits would be 2 passes but needs a 65536-bin
   histogram (too big for shared).  A 12+12+8 split is 3 passes but 4096 bins (16 KB shared, 16x the
   histogram memory) and, measured analytically, the wider `select`/`hist_accum` reads offset the saved
   pass -- a loss with the current structure.
4. **Specialise `indexer_topk_extra`.**  `blk_idx`/`cell_pos` presence is a run-time check inside
   `indexer_topk_value`/`indexer_topk_block_key`; templating the combinations would remove the
   per-call branches (small).
5. **Fuse `hist_accum` into `select`** (it already reads the whole histogram array) -- but `select` is
   per-row while the suffix counts are per-(row, block).

**How to measure.**  `rocprofv3 --kernel-trace` at `-p 8192` and `-p 32768` (exact commands in §4),
**plus a non-profiled `llama-bench -p 32768,65536 -r 2` t/s A/B** to corroborate (the profiler race
cannot touch the indexer, but the total is easier to trust unproxied).  A cheap iteration model is not
available for this op -- it only exists on qwen4exp (the `Qwen3.6-35B-A3B` has no indexer), so every
measurement is on the 94 GiB model.  Budget: model load ~1 min, pp8192 profile ~2 min, pp32768 profile
~4 min.

### 2. Rest of the non-temporal sweep (small)
Untested: the qsa3 pack kernels (6.5 ms), the `rms_norm`/other unary producers, `ssm_conv` under a
different shape.  Rule: **loads only, per-kernel, A/B load vs store** (sessions 13/14); `rms_norm_f32`
is not a candidate (reads `x` twice).

### D. Delivery items to fold into block 14

1. **`GGML_OP_INDEXER_FILL` missing from `GGML_OP_NAME`** -- found in session 9, already fixed in this
   WIP by `d1463bff3`; delivery `patches/0014` (block 14) still has it (every op from `INDEXER_FILL`
   onward is shifted by one in the name table).  One line to fold in at the next regeneration.
2. **`blk_cells` `src[7]` on `GGML_OP_INDEXER_TOPK`** (session 18) -- the block-level histogram path
   needs it; it must fold into block 14 at promotion (the `ggml_indexer_top_k` signature in
   `ggml/include/ggml.h`/`ggml/src/ggml.c`, and the call in `src/models/qwen4exp.cpp`).

### F. MMB restructure — the brief (prepared session 20, for a fresh session)

**Why.**  On the target model the MMB weight GEMMs are **61.8 % of pp8192** (9587 of 15508 ms,
measured session 20) — more than all the non-MMB prefill work combined.  (The older "~52 %" figure is
the same absolute MMB time against a larger denominator: qsa3 cut `flash_attn_qsa` from 16.6 % to
5.3 %, so MMB's *share* rose.)  Session 6 closed every cheap lever and the only remaining candidate is
a **structural** change to how the dequant and the WMMA share the warps.  This section is the
self-contained start-up for that work; read it with the session-6 UPDATE (the ceiling measurements)
and §5-§6 (the per-model tables + env knobs).

> **RESULT (sessions 21-24, `94251003e`): the restructure is DONE and landed FIVE wins.**
> A-panel double buffering (`DBUF`), the one-instruction `v_perm` bf16 RNE pack, the **A-panel
de-quant split**, the **tile-class threshold** (the small tile re-dequantized the A panel 4x), and the
> **`load_regs` field preload**.  Flash-Next MMB family **2980 → 2453 ms**, total GPU kernel
> **4409 → 3879 ms (−12.0 %)**, 35B family 1008 → 962 ms, pp8192 **1037 → 1116 t/s**, all
> **bit-identical**.  `DBUF` is now *off* for the IQ3_S
> GLU — the split supersedes it (with the split, `DBUF` costs +80 ms).  **What remains is the GLU's
642 ms of WMMA/LDS/barrier time** (the dequant's own remaining ALU is ~200 ms and is preloaded); see the
follow-up brief immediately below.  Full record: README
> "session 21"/"session 22"/"session 23"/"session 24".

#### Follow-up brief for a fresh session: the IQ3_S dequant body

**State.**  `mmb_dq_row_iq3s` (+ its `_p<PARTS>` split form) is the single hottest helper on the target
model: it dominates `routed_glu_kernel<64,32,16,16,5>` (1166 ms at pp8192-equivalent, the #1 MMB
kernel).  Measured: the dequant was **~51 % of the GLU call** before the split (6.71e9 values/call at
1.36 Tval/s isolated = 5.0 ms of 9.81 ms); the split cut the GLU by 11.6 %, and the body still
dominates the remainder.  Compare `mmb_dq_row36` (IQ4_NL) — the same *shape* of problem, solved with a
register LUT + `v_perm_b32`; IQ3_S cannot use that (512-entry grid).

**Where the instructions are** (isolated body, 765 SASS instrs before session 22's pack change):
bf16 RNE pack ~32 %, LUT index + 64-bit addressing ~26 %, sign select ~15 %, int→float + multiply
~17 %.  Measured ablations that move the needle: "no index math" **+36 %**, "grid const (keep the index
math)" **+10 %**.

**Ranked candidates.**

1. **DONE in session 24: the `load_regs` field preload** (`mmb_iq3s_preload` / `mmb_dq_iq3s_r`) —
   `routed_glu` 990 → 841 ms.  The same shape of fix is still UNPORTED to the GLU's other
   `store_lds`-loaded types (WTYPE 6/7/8/9/10), which read `Wbase` directly inside `store_lds`.  On the
   current models that is small (the IQ4_XS GLU is ~18 ms), so it is only worth it if a model puts a
   k-quant/IQ4_XS in the MoE gate/up.
2. **DONE/REFUTED in session 24: the `BM=32` occupancy route is dead.**  The GLU *is* pinned at
   1 block/CU (`hipOccupancyMaxActiveBlocksPerMultiprocessor`: `glu_big=1`, `glu_small=2`;
   `rocprofv3`'s `LDS_Block_Size` only reports *static* smem and cannot see the dynamic pad), and the
   dequant-per-output is `K/BN` — independent of `BM` — so `BM=32` looked like a free route to 2
   blocks/CU.  **Measured and refuted:** launching the small tile with 12 KB of dynamic smem forced it
   from 2 -> 1 blocks/CU and changed `mrg` by **nothing** (842.9 -> 838.7 ms, inside run noise).
   **This kernel is not occupancy-limited**, so halving `BM` would gain ~0 and cost a new split axis.
   Do not build it.  (Corollary: `1 block/CU` is *not* a defect here.  The GLU is limited by WMMA + LDS
   fragment traffic + 2 barriers per K-step contending on the same ports.)
2. **A fused gate+up dequant — reasoned DOWN, not measured.**  The gate and up weights share the *index
   expressions* but not the *values*: `qs`/`qh`/`sg`/`sc` are read from different tensors, so the LUT
   indices, the LUT loads and the scale math are all distinct — only the loop/setup structure could be
   shared.  Expected value is small, so it was not built.
3. **Cheaper index arithmetic.**  8 indices per `g` share one `qhg`; the high bit is
   `(qhg << (8-2*il)) & 256` (2 ops) plus the `qs` byte load and the `or`.  Precomputing the `qhg` bits
   or folding them into the byte load is worth trying before (2).
4. **Do NOT** re-attempt — **all measured**: an LDS-resident `iq3s_grid` (+0.6 % — the cost is the
   index/addressing math, not the load); sign-by-XOR (**not bit-identical**, `+0/-0`); the `v_perm` pack
   inside IQ3_S (**re-tested after the split: 1166 → 1297 ms**); and a **mid-N tile** (replacing the
   BN=32 class with BN=64 gives `mrg` 990 → 1048 ms — session 23).

**Method / traps (each of these cost time this session).**

* **The in-situ "no dequant" A/B is invalid** for routed/GLU kernels: corrupting the GEMM changes the MoE
  router's expert choice, which changes the workload itself (the small GLU tile "sped up" 1038 → 4 ms).
  Use the **fixed-token `llama-perplexity`** profiler (a fixed prompt ⇒ deterministic routing), *never*
  `llama-bench` (random prefill tokens make the routed timings non-comparable run to run).
* **Harness DCE is the default trap.**  A sink that consumes only some outputs, XORs an even number of
  identical values, or reuses the same input every iteration will be optimised away (this session saw a
  fake 22 Tval/s).  Vary the input per iteration and sum (not XOR) the outputs, then sanity-check the
  number against the roofline.  Prefer the real-kernel profile as the arbiter.
* **The gates, not the harness, are the bit-identity authority.**  The harness reported the (correct)
  `v_perm` pack as a mismatch and missed the split bug.
* **Any change to the A row mapping must move with its `if (row < BM)` guard** — the first split cut left
  the guard at `row == tid`, so 192 threads never ran and the shape was silently wrong.
* Thermal drift inflates later measurements (~+4 % on `mr` over a long session): A/B configs
  back-to-back, not across hours.  Benchmarks are min-of-N; never run them in parallel.

#### Where the code is

| path | what |
|---|---|
| `ggml/src/ggml-cuda/mmb.cu` (1586 lines) | everything: dequant helpers, tiles, kernels, host dispatch, shadow |
| `ggml/src/ggml-cuda/mmb.cuh` | API: `ggml_cuda_mul_mat_mmb` / `_id` / `_id_mmb_glu`, the `ggml_cuda_mmb_supported_*` predicates, the bf16 activation-cache + producer-slot API, `ggml_cuda_mmb_dense_will_take` / `_routed_will_take` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the op hooks (MUL_MAT / MUL_MAT_ID call the MMB path when a support predicate passes) and the graph-optimizer hooks (the fusion stand-down + the activation bf16 marking pass) |
| `tools/wmma-peak-gfx1151.cpp` | the peak microbenchmark (`hipcc --offload-arch=gfx1151 -O3`): **27.6 T-MAC/s bf16, 27.5 int8**, and the Q8_0-epilogue rows |

#### Geometry, tiles and the pipeline

Constants (`mmb.cu:15`): `MMB_BK = 64`, `MMB_NT = 256` (8 warps), `MMB_LDS_STRIDE = 72`.
Per-block LDS is `As[BM*72] + Bs[BN*72]` uint16 (RDNA3 CU LDS budget 64 KB): dense `128x128` = 36 KB,
dense `128x256` = 54 KB, routed-GLU `64x128` = 27 KB, tall `384x64` = **63 KB** (at the limit — the
tall tile has no room for another stage).

`mmb_tile_gemm` (`mmb.cu:405`; the GLU twin `mmb_tile_gemm_glu:640`) is the core:

* one WMMA tile per block: `TM = WTM/16` x `TN = WTN/16` bf16 accumulators per warp; 8 warps split
  `WAVES_M = BM/WTM` row-groups x `WTN/16` column-groups;
* K is walked in `MMB_BK = 64` steps; the software pipeline is **2-stage**: `load_regs(ks+1)`
  (weights -> registers) overlaps the WMMA of `ks`, and after the WMMA `store_lds(ks+1)` dequants
  the next weights into LDS.  **The dequant runs on the same warps, between the WMMA phases** — it
  is *not* overlapped with WMMA.  That is the gap;
* `__launch_bounds__(MMB_NT, 2)`;
* the A-side dequant: `WTYPE 0/1/3/4` load the quant fields into registers first, `WTYPE 5..10`
  (IQ3_S/Q5_K/Q6_K/IQ4_XS/Q3_K/IQ3_XXS) read `Wbase` directly in `store_lds`; helpers `mmb_dq_row*`
  (`mmb.cu:54-400`), one 64-value row per thread -> 64 bf16.

The tile selections (host side): dense `big ? <128,256,64,64> : <128,128,32,64>`; routed
`<128,128,32,64>` / `<128,32,32,16>`; fused GLU `<64,128,32,32>` / `<64,32,16,16>`; tall (IQ4_NL HC
down/inject only) `<384,64,96,32>` / `<384,32,96,16>`; F32 `mmb_f32split_kernel<128,128,32,64>` and
the tiny-M `mmb_tiny_m_f32_kernel<MMAX,TT,XBF16>`.

`WTYPE` is the last kernel template arg (the type switch inside `mmb_tile_gemm` / `load_regs` /
`store_lds`); the row byte width is `wrow_bytes`:

| WTYPE | ggml type | bytes / row unit | notes |
|---:|---|---|---|
| 0 | IQ4_NL | `(K/32)*18` | fused `mmb_dq_row36` |
| 1 | Q8_0 | `(K/32)*34` | fused `mmb_dq_row68` |
| 2 | BF16 | `K*2` | direct, no dequant (shadow / native bf16) |
| 3 | Q4_K | `(K/256)*144` | fused `mmb_dq_row_q4k` |
| 4 | Q5_1 | `(K/32)*24` | fused `mmb_dq_row_q5_1` |
| 5 | IQ3_S | `(K/256)*110` | `store_lds` reads `Wbase` |
| 6 | Q5_K | `(K/256)*176` | `store_lds` reads `Wbase` |
| 7 | Q6_K | `(K/256)*210` | `store_lds` reads `Wbase` |
| 8 | IQ4_XS | `(K/256)*136` | `store_lds` reads `Wbase` |
| 9 | Q3_K | `(K/256)*110` | `store_lds` reads `Wbase` |
| 10 | IQ3_XXS | `(K/256)*98` | routed only (its fused GLU is default-off) |

The target model (UD-IQ4_XS) is **IQ4_NL 52 % + IQ3_S 36 % + Q8_0 9.5 %** (the filename misleads —
the IQ4_XS type is ~1 %).  The fast 35B Q4_K_M is **Q4_K** expert + Q8_0 dense.

#### The dispatch chain

1. `ggml_cuda_mmb_supported_mm` (`:1198`) / `_supported_mmid` (`:1244`) / `_supported_glu` (`:1471`)
   gate by type/shape/contiguity, `T >= mmb_min_t()` (512, so prefill only) and `K % 64` (or `%256`
   for the k-quants); routed additionally needs `n_used <= 64` and `(T*n_used)>>16 < 1024`.
2. `ggml_cuda_mul_mat_mmb` (`:1252`): F32 (`mmb_f32split_kernel` / `mmb_tiny_m_f32_kernel`), the
   tall IQ4_NL tile, the shadow path, then `mmb_dense_kernel<...>` per weight type.
3. `ggml_cuda_mul_mat_id_mmb` (`:1440`) + `mmb_routed_kernel_dispatch` (`:1418`): per-expert grouping
   via `ggml_cuda_launch_mm_ids_helper` + `mmb_build_desc2` (`:946`; two classes — `>=THRESH=128`
   rows -> `128x128,32,64`, else `128x32,32,16`).
4. `ggml_cuda_mul_mat_id_mmb_glu` (`:1521`) + `mmb_routed_glu_kernel_dispatch` (`:1484`): the fused
   gate+up+SWIGLU.  **This is the #1 kernel.**
5. `mmb_bf16_activation` (`:999`): the activation is converted F32->bf16 **once** per (tensor, n)
   into a pool cache and reused; the old `mmb_cvt` pass is gone (sessions 8-12).  Do not re-add it.

#### The measured profile (gfx1151, target model, bf16 KV, ub 2048, pp8192; grand 15508 ms)

| kernel | ms | share | calls | tiles |
|---|---:|---:|---:|---|
| `mmb_routed_glu_kernel` | 3863.9 | **24.9 %** | 752 | `64x32,16,16` (2137) + `64x128,32,32` (1655) |
| `mmb_dense_kernel` | 3622.1 | **23.4 %** | 3728 | `128x128,32,64` (1958) + `128x256,64,64` (1640) |
| `mmb_routed_kernel` | 1270.7 | 8.2 % | 752 | `128x128,32,64` (603) + `128x32,32,16` (526) |
| `mmb_tiny_m_f32_kernel` | 445.4 | 2.9 % | 1136 | tiny-M F32 (hc inject M=4, M=8) |
| `mmb_f32split_kernel` | 381.2 | 2.5 % | 472 | F32 `128x128,32,64` (MoE router M=512) |
| **MMB family** | **9587** | **61.8 %** | | |

On the fast 35B Q4_K_M model the same family is ~49-52 % of pp8192 (session 6), so **use the 35B for
dense/GLU-tile iteration and the target model only when the change is IQ4_XS/IQ3_S/HC-specific.**
Dense grids seen: `blockIdx.x = M/128` (M = 128, 512, 640, 2560, 6144, 10240, 12288 ...); the GLU uses
`blockIdx.x = M/64`.  `rocprofv3` reports `Grid_Size_X = blocks.x * 256` (divide before matching).

#### The ceiling, and what is already closed

Dense **54 %** of the 27.6 T-MAC/s bf16 WMMA peak; fused GLU **~36 %** (it dequants gate and up
separately).  The loss is dequant-issue contention on the SIMD lanes, **not**:

* weight bandwidth — a bf16 shadow is **2.44x slower** (the 1-byte format's L2 residency is a
  feature; session 6);
* int8 — gfx1151 int8 == bf16 WMMA rate, and the Q8_0 epilogue makes int8 *slower* (session 6, and
  the `tools/wmma-peak-gfx1151.cpp` rows);
* tile/BN/VDR/`MMB_CACHE` knobs — all washes or worse (session 6 table);
* split-K — the dense kernel is already grid-rich (1024+ blocks).

#### The restructure candidates, in order

1. **Warp-specialised dequant + LDS ring (the session-6 "one remaining candidate").**  Split the 8
   warps into producers (dequant -> LDS) and consumers (WMMA), with a multi-stage ring of
   `MMB_BK`-wide `As`/`Bs` tiles so the producers stay ahead of the consumers.  Budget check: a
   2-deep `A` ring + 1-deep `B` for the dense `128x128` tile is `2*128*72*2 + 1*128*72*2 = 54 KB`
   (fits); a 3-deep ring needs 72 KB (does not), and dense `128x256` (2-deep A + 1-deep B = 72 KB)
   and tall `384x64` (63 KB already) do **not** fit.  **SESSION 21: implemented as `DBUF` (A 2-deep, B
   1-deep) — landed for IQ3_S GLU only; see the session-21 UPDATE.**
2. **Share the GLU dequant.**  The fused GLU walks the gate and up weights separately; if both were
   dequanted in one row step (a wider `mmb_dq_row` writing `Ag` and `Au`) or the two weights were
   interleaved at load time, the issue + index arithmetic would be shared.  Requires either a load-
   time repack (like the shadow) or accepting the two buffers as-is (then only the loop/index work is
   saved — a smaller win).
3. **Cheaper diagnostic first (recommended before any rewrite):** time the existing `128x128` dense
   kernel with the dequant **replaced by a constant** (throwaway, wrong output) — if it does not
   speed up ~proportionally to the 54 %-of-peak gap, the stall is elsewhere (LDS bank conflicts,
   `__syncthreads` count, or occupancy) and the ring will not pay.
4. The activation side is already optimal (`mmb_bf16_activation` caches the bf16 conversion once);
   the F32 path and tiny-M are separate kernels with their own ceilings (sessions 5/10/12 at the
   `F32SPLIT` comment).
5. **SESSION 22 measured the dequant body itself (do not re-derive):** it is **~51 % of the GLU**
   (6.71e9 values/call at 1.36 Tval/s isolated = 5.0 ms of 9.81 ms).  The **bf16 RNE pack is now one
   `v_perm_b32`** (`mmb_pack2`, except the IQ3 paths which keep `mmb_pack2_so` — the perm form is
   *slower* there, 1318 → 1355 ms) and that landed the win.  **Rejected: an LDS-resident `iq3s_grid`**
   (+0.6 % — the cost is the index/addressing math, ~26 %, not the LUT load) and **sign-by-XOR** (not
   bit-identical: a zero product is `+0.0` today, `-0.0` via XOR).  What is left: the **A-panel uses
   only 64 of 256 threads** for `BM = 64` (`row = tid + i*256`); splitting a row across the 8 `il`
   groups would use all 8 warps — but with `DBUF` on there is no barrier between `store_lds_a` and the
   WMMA, so the idle warps already run ahead.  **Measure before building.**

#### Iteration loop and gates

```sh
cd ~/llama-wip-mmb && cmake --build build-rocm -j16 --target llama-bench llama-perplexity llama-cli
M35=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf   # fast model
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-bench -m "$M35" -ngl 99 -fa 1 -b 2048 -ub 2048 -p 8192,2048 -n 0 -r 2
# kernel-level judge (t/s is too noisy for a few-percent kernel change):
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
rm -rf /tmp/prof && mkdir -p /tmp/prof && GGML_CUDA_MMB=1 \
  /opt/rocm-7.14-gfx1151/bin/rocprofv3 --kernel-trace -f csv -d /tmp/prof -o k -- \
  ./build-rocm/bin/llama-bench -m "$M35" -ngl 99 -fa 1 -b 2048 -ub 2048 -p 8192 -n 0 -r 1
# aggregate (Kernel_Name -> sum ms / calls; Grid_Size_X is blocks.x*256, divide it):
python3 -c 'import csv,collections;t=collections.Counter();c=collections.Counter();g=0
for r in csv.DictReader(open("/tmp/prof/k_kernel_trace.csv")):
 if r["Kind"]=="KERNEL_DISPATCH":
  n=r["Kernel_Name"];s=int(r["Start_Timestamp"]);e=int(r["End_Timestamp"]);d=(e-s)/1e6;t[n]+=d;c[n]+=1;g+=d
print(f"grand {g:.0f} ms");[print(f"{v:9.1f} {100*v/g:5.2f}% {c[k]:6d} {k[:80]}") for k,v in t.most_common(20)]'
```

* **Confirm the MMB gate from the kernel names in the trace** — under `rocprofv3` an env-gated path
  can read as unset (the `rocprofiler-register` race, session-15 caveat).
* A restructure that changes only the *order* of the dequant cannot change the arithmetic: the WMMA
  inputs are the same bf16 values, so the output is **bit-identical** and PPL/greedy must not move.
  A changed *tile/summation* order is a numerics change and re-baselines PPL (bar ~±0.68 at c2048).
* Gates: PPL c2048 **10.6015**, greedy **`9c281c415082`**, `test-logits-width-probe` **PASS**; the
  `W=1..8` decode/verify band is untouched (`mmb_min_t()=512`), but keep the `mmb_supported_*` shape
  guards consistent if the tile dispatch changes.

#### Reference

`~/pwilkin-llama-cpp` @ `f5daaa3cf` (branch `strix-halo`), `ggml/src/ggml-cuda/mmb.cu` (881 lines) is
the origin of the tile design — and it has the **same** tiles and **no** warp-specialised pipeline
(everything is already ported, incl. `mmb_tall`), so **there is nothing to port; this restructure is
ours to design.**

### 4. Promotion (maintainer-gated)
Every §11 gate passes.  Rebase onto a canonical fork rebuilt at `ebbb18522` + `scripts/apply-all.sh`,
regenerate `patches/`, decide whether MMB rides as a block-08 amendment.  **Do not do this
unilaterally** -- needs a beta window / go-ahead.

**Closed — do NOT redo** (each measured in the UPDATEs below):

* `mmb_dense` int8-IU8 WMMA and a bf16 weight shadow (session 6: int8 == bf16 on gfx1151; the shadow
  is 2.44x slower).  Every `mmb_*` tile/BN knob (wash or worse; 54 % / 36 % of WMMA peak).
* `ssm_alpha/beta` (M=48) via a WMMA f32 tile or an exact SIMT tile (session 10: rocBLAS stays — the
  WMMA split costs +0.04 PPL, the SIMT tile is slower).
* The qsa3 pack (session 7: 0.34 %, not 3 %; fused, bit-identical).
* The `mmb_cvt` bucket (sessions 8/9/11: eliminated bit-identically, incl. the dead-F32-store skip)
  and the `xn` BF16-only stream (session 12: +4.4 %/+3.8 % over session 11; the last `mmb_cvt` is
  the model-tensor `ple_embd`).
* The `dsv4_hc_pre` "kernel-local residual" as a *tiling/ILP* problem (session 13: it is L2/MALL
  pollution; non-temporal hints are the fix, vec4/vec8 are worse in situ).
* The `rms_norm_f32` second `x` read as a DRAM re-read (session 20: register-caching the per-thread
  `x` is a **wash** -- `<1024>` -1.2 %, `<256>` +3.0 %, net +0.2 %; the read is cache-resident).

---

> **Running summary (sessions 1-11, history).** A general prefill weight-GEMM on the tensor cores (dequant-to-bf16 → WMMA)
> is implemented for **every** weight type the delivery's models use, validated PPL-parity, and
> measured at up to **+68 % pp2048 / +56 % pp8192** on the Flash-Next Q4_K_M.  **QSA v3 (packed-block
> WMMA sparse attention)** landed in sessions 2/3 (QSA attention **2944 -> 728.6 ms, 4.04x**), and
> session 4 made it the default at every context length.  Sessions 5..5e then added the **shape-aware
> F32 split** (+2.1/+2.6/+3.1 %), the **tiny-M F32 kernel for the hc `*_inject` pair** (+2.3-3.3 %),
> ran **all the promotion gates to green** (including the `W=1..8` probe, which needed a new harness),
> and investigated **dsv4_hc** (no change landed — see 5e; the remaining lever there is bf16
> intermediates, a graph change).  **Session 6 closed the `mmb_*` optimization line**: the int8-IU8
> restructure is refuted on gfx1151 (int8 == bf16 WMMA; the Q8_0 epilogue makes it slower), a bf16
> weight shadow is 2.44x *slower*, and every tile knob is a wash or worse — the kernels are at their
> structural ceiling (dense Q8_0 at 54 % of the bf16 WMMA peak, GLU at 36 %).  The next prefill lever
> is **outside `mmb_*`** (FA 11.3 %, GDN 5.5 %, MoE concat+reduction 6.8 %, rms_norm ~5 %), or
> **promotion** (all gates green).  **Session 7** then measured the qsa3 pack (session-5b's "3 %"
> item), found it is **0.34 %** (the bucket was really the MoE `concat_transposed_src1_dim0` +
> base-graph F32 copies), and **fused it into one launcher pass, bit-identical: 50.5 → 6.5 ms,
> +0.26 % pp8192**.  The **indexer** (1 % at 8K, 3.2 % at 32K) stays deferred.  **Session 8** then began
> the bf16-producer port: the HC gate (a dense MUL_MAT the graph now marks BF16-only) and the HC
> normalized stream (the fused rms_norm+mul emits a BF16 copy into slot 0) are native-bf16 producers —
> **pp8192 950.8 -> 961.9 (+1.2 %), pp2048 973.1 -> 994.2 (+2.2 %)**, the gate is a measured numerics
> change and the xn copy is bit-identical; `hc_mixed`/`final_output` and the (640 ms / 3.8 %)
> `mmb_cvt` bucket are what remain.  **Session 9 finished it**: the graph now marks the activation of
> every MMB dense prefill GEMM "wants a BF16 copy", and the fused `rms_norm+mul`, `sigmoid+mul`,
> `scale+unary` and generic unary producers all emit it — so the whole `mmb_cvt` bucket is gone
> (only the `ple_embd` model tensor still converts), **bit-identical** (PPL unchanged, greedy text
> unchanged, all gates green), and **session 11 skipped the dead F32 stores** (+2.3 % / +3.3 %), for
> **pp8192 952.4 -> 974.0, pp2048 975.3 -> 1007.4**;
> the same session also found and fixed a **delivery** bug (see the session-9 UPDATE).  **Session 12**
> finished the producer port: the HC normalized stream `xn` is now **BF16-only** too (the
> `dsv4_hc_pre` src[0] and tiny-M inject consumers gained bf16 arms, and `xn` got a **dedicated
> producer slot** so its delayed consumers cannot be clobbered by the intervening generic copies) —
> a numerics change, PPL c2048 10.6015, **+4.4 % pp2048 / +3.8 % pp8192 over session 11** and the
> last `mmb_cvt` is the model-tensor `ple_embd`.  **Session 13** then closed session 5e's
> `dsv4_hc_pre` "kernel-local residual": the kernel was 18 % below its own ceiling because its
> streaming buffers thrashed the L2/MALL the GEMM weight streams use, and **non-temporal
> loads/stores** (value-preserving, bit-identical) take it 605 -> 493 us/call (**-18.6 %**) with
> `_post` 691 -> 660 ms; vec4/vec8 were tried and are worse in situ.  **Session 14** extended the
> non-temporal idea to the other streaming kernels and found it is **load-only and per-kernel**:
> `concat_transposed_src1_dim0` 357 -> 327 ms and `moe_weighted_reduction` 384 -> 344 ms, while
> `ssm_conv_long_token_f32` wins on neither (+79 ms both) — a non-temporal *store* evicts the output
> the next op wants, so load and store must be A/B'd separately.  **Session 15** turned up a
> `rocprofv3` trap -- `rocprofiler-register` `setenv()`s during early init and can make an env gate
> read unset, so qsa3 became a compile-time gate (`LLAMA_QSA3_ENABLE`) and the long-context indexer
> was finally measured cleanly (**2.94 % at pp32768**, ~linear in context) -- that is the next
> session's mandate.

**Current state (also in the brief above; kept here for history):**

| | |
|---|---|
| worktree | `~/llama-wip-mmb`, branch `wip-mmb-general`, tip **`aa55dfef8`** (clean) |
| base | **r12 applied tree `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`** |
| backup | `wip/mmb-general/mmb-general.patch` + `patches/` (**5 thematic patches**) + `commits.txt`, in this repo, pushed to `origin/wip-mmb-general` |
| verify | `git am` of `patches/` on the r12 tree — clean (**5/5**, applied tree `d365b43ddc87c472c33a121247931269f975aa43` == the 38-commit tree) |
| build | §3 | run | §4 |
| current numbers | the **session 24 UPDATE below** (the `load_regs` field preload -- routed_glu 990 -> **841 ms**, total GPU kernel 4015 -> **3879 ms**, pp8192 **1116 t/s**) then the **session 23 UPDATE** (the tile-class threshold) then the **session 20 UPDATE** (the `rms_norm` register-cache **refutation** -- the session-19 tip `2da50418d` is unchanged) and the **session 19 UPDATE** (indexer pass-1 per-block histogram atomics + the block-pass warp reduction) -- indexer family pp8192 **71.7 ms** / pp32768 **1146.6 ms** -- plus the **session 18 UPDATE** (the block-level histogram + `blk_cells` src), the **session 15 UPDATE** (qsa3 compile-time gate + the rocprofiler-register profiling caveat), the **session 14/13 UPDATEs** (non-temporal) and the **session 12 UPDATE** (`xn` BF16-only); plus the **delivery `GGML_OP_NAME` fix** |

**Historical ordering of the UPDATE sections:** 26 (newest, 2026-09-20, the r12 rebase + the 38->5 consolidation) -> 25 ( 2026-09-20, the indexer gather's warp-shuffle scan; the block-gather closed) -> 24 (the `load_regs` IQ3_S field preload) -> 23 (the tile-class threshold: the small tile's 4x A-panel dequant) -> 22 (the `v_perm` bf16 pack + the A-panel split) -> 21 (the A-panel double-buffered GLU tile) -> 20 (the `rms_norm` register-cache refutation) → 19 (2026-09-20, the indexer pass-1 per-block histogram atomics + the block-pass warp reduction) → 18 (2026-09-20, the indexer block-level histogram path + the `blk_cells` op src) → 17 (2026-09-20, indexer block-key sharing) → 16 (2026-09-20, the indexer count-pass elimination + the "compact after pass 1" refutation) → 15 (2026-09-20, the qsa3 compile-time gate + the rocprofiler-register profiling caveat) → 14 (2026-09-20, the non-temporal load sweep: concat/moe/unary) → 13 (2026-09-20, the `dsv4_hc` non-temporal fix) → 12 (2026-09-20, the `xn` BF16-only stream) → 11 (2026-09-20, the dead-F32-store skip in the producer port) → 10 (2026-09-20, `ssm_alpha/beta` profiled — rocBLAS stays) → 9 (2026-09-20, the full bf16-producer port) → 8 (2026-09-20, the HC gate + xn bf16 producers) → 7 (2026-09-20, the pack measurement) → 6 (2026-09-20, the
`mmb_*` ceiling) → 5e (dsv4_hc) → 5d (W=1..8 probe) → 5c (gates) → 5b (tiny-M) → 5 (profile + F32
split) → 4 → 3 → 2.**  §0-§14 after them are the original (session-1) body and are correct except where
an UPDATE says otherwise.

**Next work:** see the **"FOR THE NEXT SESSION"** brief at the very top of this file — its ordered
list is the authoritative one, and the historical "next-work order" lists inside the UPDATE sections
below are superseded.

---

## UPDATE — session 26 (2026-09-20): rebase onto **r12** + the WIP **consolidated 38 -> 5 thematic patches**

Full record: `GROUPS.md` (the triage sheet) + README "session 26".

**Rebase.**  The delivery moved to r12 (`origin/main` `4e37fa6`) while this branch was frozen.  Both
rebases are done and verified:

* **record branch** `git rebase --onto 4e37fa6 1c2ec00 wip-mmb-general` — 17/17; the two `AGENTS.md`
  conflicts were resolved by regenerating the file from r12's content plus this branch's WIP-branch
  pointers (r12's rewrite and our pointer both touch the layout table and the WIP-rule bullet);
  `README.md` and this file auto-merged.
* **code branch** — the r12 tree `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5` was rebuilt with
  `RDNA_BRANCH=r12-verify scripts/apply-all.sh` on a fresh `ebbb18522` worktree (tree verified ==
  `release.json`), then `git rebase --onto <r12> 8a2567e1e wip-mmb-general` — **38/38, no conflicts**.
  All gates re-verified green afterwards (PPL 10.6015, greedy `9c281c415082`, width PASS, the three
  op tests 2/2).

**Consolidation: 38 commits -> 5.**  The other machines need a manageable set, so the WIP is now five
thematic patches (see `GROUPS.md`):

| # | theme | originals |
|---|---|---:|
| 1 | `mmb` — the general-purpose bf16-WMMA dequant weight GEMM | 14 |
| 2 | `qsa3` — the packed-block WMMA sparse-attention path | 5 |
| 3 | the F32/tiny-M kernels + the default flips + the W=1..8 probe | 5 |
| 4 | HC16 native-BF16 producers + non-temporal accesses | 7 |
| 5 | the fused indexer top-k op | 7 |

The commits were **interleaved** (the `mmb` line is commits 1-9 *and* 33-37), so this is a
cherry-pick-and-squash, not a reorder.  **It is content-preserving and verified as such**: the 5-patch
result has tree **`d365b43ddc87c472c33a121247931269f975aa43`**, byte-identical to the 38-commit tip it
replaced, and `git am` of the 5 patches on a fresh r12 tree is clean.  The 38-commit history is still
in this repo's git history if the fine-grained form is ever needed.

**Not touched:** `~/llama.cpp`'s `rdna-boosts` branch (still at the pre-r12 `8a2567e1e`); the r12 tree
was built in a throwaway worktree on a distinct `r12-verify` branch.

---

## UPDATE — session 25 (2026-09-20): the **indexer gather's block scan** — warp shuffle instead of
## Hillis-Steele, gather **-11.7 %**; and the block-level gather/emit is CLOSED as unbuildable

Tip `aa55dfef8` (38th commit).  The session-24 follow-up ("look into the indexer block-gather").
Full record: README "session 25".

**Verdict first: the block-level gather/emit (the handover's item A.1) is DEAD, and the whole indexer
line is CLOSED.**  The handover had it as "the one real remaining optimisation"; reading
`llama-memory-hybrid-idx.cpp:743` shows why it cannot be built:

```c
cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;   // idx = ranked ? rank[j] : cells.pos_get(j)
```

* the cells of a block sit in **`idx % r` slot order**, not column order, so the ranked path needs a
  per-block sort of `r` cells;
* and decisively, the write is inside `if (blk_of[j] >= 0)`, so **every unpooled cell — including the
  tail, which `blk_tail` makes *always visible* and therefore likely selected — is absent from
  `blk_cells`**.  A block walk still needs a full `n_kv` scan to find them, which removes the entire
  point.  The ideal outcome was ~0.4 % of the run.  Do not build it.

**The "skip the whole block" intermediate was implemented, A/B'd and gained NOTHING.**  Skipping the
per-cell key evaluation for blocks whose key cannot reach the prefix is provably output-preserving
(a sub-prefix block emits nothing and contributes nothing to the carry): measured `259.9 -> 262.2 ms`.
**That is the useful signal** — the gather is *not* memory-bound on `cell_pos`.

**What that pointed at, and what landed.**  The gather ran a 256-thread **Hillis-Steele** inclusive
scan of (g_count, e_count): 8 steps x 2 `__syncthreads` = **~19 barriers per 1024-cell tile**.  It is
now a warp-shuffle scan plus one 2-barrier offset pass (integer sums are associative, so the prefix
values and the placement order are identical).

| pp32768, target model | before | after |
|---|---:|---:|
| `indexer_topk_write_blocks_grouped` (the gather) | 259.9 ms | **229.6 ms (−11.7 %)** |
| indexer family | 1146.5 ms (1.86 %) | **1117.4 ms (1.80 %)** |
| `indexer_topk_histogram_blocks` (×3) | 462.1 | 462.1 |
| `indexer_topk_histogram_pass1` | 265.6 | 265.6 |

**Bit-identical.**  Note which gate matters: the indexer selection width is 2051, so a prose prompt at
`-c 8192` never leaves the dense shortcut and the **normal PPL/greedy/width gates cannot reach this
path at all**.  The gate that does is a **long-context same-seed A/B** (16k-token prefill):
`9565ffa670c3` with and without.  Plus `test-backend-ops -o INDEXER_TOPK` and `-o FLASH_ATTN_QSA`
(2/2 each), PPL 10.6015, greedy `9c281c415082`, width PASS.

**Still open, not worth it** (each < 0.2 % of the run): `histogram_blocks` is the biggest remaining
indexer kernel (**462 ms = 41 % of the family**) and recomputes the per-`(row,block)` key in each of
its 3 passes and re-sums `wvis` per range.  Caching the key (or a compact per-block bin) from pass 1,
and a per-`(row,hb)` visible-count total to let passes 2-4 skip the `wvis` load, are both real.  Neither
is order-sensitive — the low-risk place to look if the indexer is ever revisited.

---

## UPDATE — session 24 (2026-09-20): the **`load_regs` field preload** — the GLU's dequant ran exposed.
## `routed_glu` **990 -> 841 ms (−15.1 %)**, total GPU kernel **4015 -> 3879 ms**, pp8192 **1116 t/s**

Tip `94251003e` (37th commit).  The session-23 follow-up (the ranked candidate in the §F brief).  Full
record: README "session 24".

**First, size the prize with a legitimate oracle.**  The in-situ "remove the dequant" A/B is invalid
(routing), so instead **duplicate** the IQ3_S dequant -- same values, idempotent, so the output and the
routing are unchanged -- behind an `asm volatile("" ::: "memory")` barrier so the compiler cannot CSE
it (without the barrier the duplicate is folded away: 3943 vs 4540 SASS instrs).  Result:
**`mrg` 990 -> 1338 ms**, so the dequant's marginal cost is **348 ms = 35 % of the GLU**, and the other
**642 ms is WMMA/LDS/barriers**.  Two consequences: the dequant IS on the critical path, and it is NOT
the majority of the kernel (the session-22 "~51 %" figure was an isolated-harness extrapolation).

**The fix: hoist the loads into the prefetch.**  `store_lds_a` runs *exposed* between the WMMA phases
(DBUF is off for this tile), so its `d`/`sc`/`qs`/`qh`/`sg` globals were issued and waited on inside
that exposed phase.  `mmb_iq3s_preload<PARTS>` now fetches only this thread's own `il` groups
(4/PARTS of them; the two `qs` bytes a grid lookup needs are adjacent, so they come back as one
`uint16`) into a small register struct **inside `load_regs`**, which already overlaps the previous
K-step's WMMA; `mmb_dq_iq3s_r` then runs purely out of registers.

| Flash-Next IQ4_XS | before | **after** |
|---|---:|---:|
| `routed_glu` (`mrg`) | 990.4 ms | **840.8 / 840.9** (two runs) |
| MMB family | 2592 | **2453** |
| **total GPU kernel** | 4015 | **3882 / 3879** (vs 4409 pre-session-21: **−12.0 %**) |
| pp8192 / pp2048 | 1089.1 / 1101.4 t/s | **1116.0 / 1128.3 (+2.5 %)** |

35B unchanged (the preload is on the IQ3_S path only).  Bit-identical: PPL **10.6015**, greedy
**`9c281c415082`**, width probe **PASS**.

**TRAP — the session-22 guard bug, walked into again.**  The first cut put the preload *inside*
`load_regs`' `A_ITEMS` loop under `if (row < BM && row < a_rows)` with `row = tid + i*MMB_NT`: only
threads 0..63 ran it and they all took `tid/BM = 0`, so parts 1..3 stayed stale and greedy collapsed to
**146 chars**.  The split row is `tid % BM` for *every* thread, so the preload must be **hoisted out of
that guard** — it now sits above the loop with a comment saying why.  **When you change the A row
mapping, the guard and every consumer of it must move together.**

**Where the pp8192 time actually goes (measured, `rocprofv3`, one `-p 8192 -n 0 -r 1` run = 2 passes;
divide by 2).  GPU kernel total 14396 ms = ~7198 ms per pass against a 7315 ms wall, i.e. the GPU is
~98 % busy — there is no launch-gap slack left to reclaim, kernel time IS the whole story:**

| per pass | ms | share | |
|---|---:|---:|---|
| `mmb_dense` | 1798 | **25.0 %** | Q8_0/Q6_K/BF16 dense GEMMs |
| `mmb_routed_glu` | 1399 | **19.4 %** | the IQ3_S/IQ4_XS MoE gate+up+swiglu |
| `mmb_routed` | 631 | 8.8 % | IQ4_NL MoE down |
| `qsa3_attn` | 408 | 5.7 % | |
| `dsv4_hc_post` | 334 | 4.6 % | |
| `gdn_bf16_scan` | 310 | 4.3 % | |
| `rms_norm_f32<1024>` | 270 | 3.8 % | session 20 refuted the register-cache |
| `mmb_tiny_m_f32` | 224 | 3.1 % | hc inject M=4/8 |
| `mmb_f32split` | 190 | 2.6 % | MoE router |
| `dsv4_hc_pre` | 189 | 2.6 % | |
| `moe_weighted_reduction` | 172 | 2.4 % | |
| `concat_transposed_src1_dim0` | 166 | 2.3 % | |
| `ssm_conv_long_token` | 157 | 2.2 % | |
| `k_bin_bcast` | 123 | 1.7 % | **the indexer (item A)** |

**MMB is 58.9 % of the pp8192 run**; `mmb_dense` is now the single largest kernel (25 %), and its Q8_0
dequant is cheap arithmetic, so it is the WMMA/LDS-bound one.

**Also measured this session (negative results, do not redo):**

* **`MMB_LDS_STRIDE` 72 -> 64 is catastrophic** (`mrg` 990 -> 2538 ms, the dense Q8_0 485 -> 2078).  The
  +8 padding exists to break LDS bank aliasing: a compact panel has a 128 B row stride, which aliases
  every row onto the same banks for the row-parallel WMMA fragment reads.
* **The 1-block/CU ceiling is real but IRRELEVANT — occupancy is not the limiter.**  `rocprofv3`'s
  `LDS_Block_Size` (static only) shows `mmb_routed_glu<64,128>` and `mmb_dense<128,128>` at 36864 B =>
  1 block/CU (registers allow 3; LDS binds at 64 KB/CU), and
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` confirms it (`glu_big=1`, `glu_small=2`).  But forcing
  the small tile 2 -> 1 blocks/CU with 12 KB of dynamic smem changed `mrg` by **nothing** (842.9 ->
  838.7).  **So the `BM=32` route to 2 blocks/CU is refuted by measurement** — do not spend the
  split-axis rewrite on it.  (Technique note: `rocprofv3`'s LDS column excludes dynamic shared memory,
  so a dynamic pad is invisible to it; a hardcoded pad plus the HIP occupancy API is the reliable check.)

---

## UPDATE — session 23 (2026-09-20): the **small tile re-dequantized the A panel 4x** — tile-class
## threshold 128 -> `BN_SMALL`, MMB family **2822 -> 2592 ms**, total GPU kernel **4254 -> 4015 ms**,
## pp8192 **+1.5 %**, bit-identical

Tip `49eff7f18` (36th commit).  The session-22 follow-up.  Full record: README "session 23".

**How it was found.**  Profiling the post-split tree per *shape* (not per name) showed the single
biggest kernel in the whole run is `mmb_routed_glu_kernel<64,32,16,16,5>` at **903 ms over 94 calls =
21 % of all GPU time**, versus 242 ms for the BN=128 sibling `<64,128,32,32,5>`.  Reading
`mmb_routed_glu_kernel` settles why: **`BM` tiles M (the expert's output rows = the A panel) and `BN`
tiles the expert's token rows**, and `mmb_build_desc2` sent every expert with `cnt < THRESH(128)` to the
BN=32 class.  A 128-row expert therefore took `ceil(128/32) = 4` blocks, **each of which re-dequantized
the same 64-row A panel** -- 4x the weight dequant for the same WMMA work and the same B-column padding.

**The threshold sweep** (Flash-Next IQ4_XS, fixed prompt, min-of-2, the GLU kernel):

| `THRESH` | 0 | 16 | **32** | 48 | 64 | 128 (old) |
|---|---:|---:|---:|---:|---:|---:|
| `mrg` ms | 1078.4 | 1028.6 | **992.0** | 1014.6 | 1031.4 | 1161.4 |

The optimum is exactly `THRESH == BN_SMALL == 32`, which is also the analytic rule: the small tile only
pays while **one** block covers the expert (then it does 1/4 the WMMA for the same dequant); past that
the big tile wins on dequant volume.  The routed (non-GLU) tile behaves the same: `mr` **395.9 ->
361.5 ms (-8.7 %)** at 32.

**Landed** (both defaults, each with an env A/B override):
`mmb_glu_thresh()` (`GGML_CUDA_MMB_GLU_THRESH`) and `mmb_routed_thresh()` (`GGML_CUDA_MMB_ROUTED_THRESH`).

| | old | **new** |
|---|---:|---:|
| Flash-Next `mrg` | 1161.4 | **990.4** |
| Flash-Next `mr` | 401.2 | **361.5** |
| Flash-Next MMB family | 2822 | **2592** |
| Flash-Next total GPU kernel | 4254 | **4015 (-8.9 % vs 4409)** |
| 35B Q4_K family | 987 | **962** |
| pp8192 / pp2048 (same binary, A/B) | 1073.1 / 1086.3 t/s | **1089.1 / 1101.4 t/s (+1.5 %)** |

**Bit-identical, and it cannot be otherwise**: `BN` only changes *which* token rows a block covers --
never the K reduction order of any output element, and the A-panel dequant is the same rows in the same
`ksh` order.  Verified: PPL **10.6015**, greedy **`9c281c415082`**, width probe **PASS**.

**Also measured this session (do not redo):** the `v_perm` bf16 pack for IQ3_S was re-tested *after* the
split (the register pressure and ILP changed) and still loses -- **1166 -> 1297 ms**; and replacing the
BN=32 class with a **mid BN=64 tile** loses too (**990 -> 1048 ms**), so the three-point tile ladder is
not worth it -- the two-class split at `THRESH == BN_SMALL` is the optimum.

**Read-through for the next person:** before reaching for a kernel rewrite, **profile per shape**.  This
21 %-of-the-run kernel was invisible in the per-name profile (it is just "mrg"), and the win was not in
the dequant *body* at all -- the brief had been pointing at the body for two sessions.

---

## UPDATE — session 22 (2026-09-20): the **bf16 RNE pack is one `v_perm_b32`** (all but the IQ3 LUT
## paths) — 35B MMB **−2.3 %**, Flash-Next total GPU **4309 ms (−2.3 % vs the pre-DBUF baseline)**;
## IQ3_S dequant audited, its LDS-grid and sign-XOR candidates measured and rejected

Tip `16c0fe07c` (34th commit).  The session-21 follow-up.  Full record: README "session 22".

**The audit ("did we already optimise the other types?"):** only **IQ4_NL** (`mmb_dq_row36`) and
**IQ4_XS** (`mmb_dq_row_iq4xs`) had been optimised — both hold their 16-entry codebook in 4 registers
and apply it with `v_perm_b32`.  **IQ3_S and IQ3_XXS were naive ports of upstream's
`dequantize_iq3_s`** — 16 divergent lookups into a 512/256-entry *global* grid per 64-value row.  The
k-quants (Q4_K/Q5_1/Q5_K/Q6_K/Q3_K) and Q8_0 are arithmetic (no LUT).  The register-LUT trick cannot
apply to a 512-entry grid.  Upstream uses the same global grid, so it is not an MMB regression.

**Cost:** the Flash-Next GLU dequantizes 6.71e9 values/call; a standalone harness (helper copied
verbatim, `iq3s_grid` extracted from `ggml-common.h`) measures 1.36 Tval/s → **~5.0 ms of the 9.81 ms
GLU call, i.e. the dequant is ~51 % of the GLU** (matches the session-21 ISA count).

**Landed:** `mmb_pack2` now rounds in integer and assembles the pair with **one `v_perm_b32`**
(`__builtin_amdgcn_perm(ua, ub, 0x03020706)`) instead of per-value `>>16` + shift/or.  Isolated IQ3_S
body **765 → 616 SASS instrs**, **1364 → 1701 Gval/s**.  35B MMB family **1008 → 985 ms (−2.3 %)**
(`mr` Q5_K/Q6_K **−7.9 %**, `mrg` Q4_K **−3.0 %**); Flash-Next Q6_K dense **−7.6 %**, total GPU
kernel **4409 → 4309 ms**.  **The IQ3_S dequant keeps the shift/or form** (`mmb_pack2_so`): it
*regresses* there (1318 → 1355 ms un-DBUF'd, 1225 → 1258 DBUF'd) — the LUT-latency-bound path waits on
all eight rounded words before the perm can start.

**Rejected (measured):** an **LDS-resident `iq3s_grid`** is +0.6 % (nothing) — the ablation shows the
cost is the **index/addressing math (~26 %)**, not the global LUT load; and **sign-by-XOR is NOT
bit-identical** (a zero product yields `+0.0` today vs `-0.0` via XOR, and `mmb_f2bf(±0.0) = 0x0000/0x8000`).

**Then the A-panel split landed (same session).**  `A_ITEMS = ceil(BM/256)` and the A row is
`tid + i*256`, so the GLU's `BM = 64` ran the A dequant on **warps 0-1 of 8 only** (dense `BM = 128`:
warps 0-3).  Isolated, 2 active warps reach **348 Gval/s vs 691** for 8 — 50 %, not 25 % — so the CU had
spare issue capacity.  `mmb_dq_row_iq3s_p<PARTS>` now has the `PARTS = MMB_NT/BM` threads sharing a row
take the `il` groups `{p, p+PARTS, ...}` for both `g` (same LDS dwords, disjoint offsets), with
`SPLIT_A = (WTYPE == 5 && MMB_NT > BM && MMB_NT % BM == 0)` switching the row mapping in
`store_lds_a` to `tid % BM`.  **`mrg` 1318.2 → 1166.4 ms (−11.6 %)**, total GPU kernel **4409 → 4254 ms
(−3.4 %)**, bit-identical; 35B unaffected (`SPLIT_A` false).  **It supersedes `DBUF` for this tile** —
with the split, `DBUF` is a regression (mrg 1166 → 1246) because the dequant no longer leaves warps idle
for the WMMA to run ahead into, so `DBUF` is switched off for the IQ3_S GLU (plumbing kept, disabled).

**Trap that cost a debug cycle:** the first cut put the split *inside* the `if (row < BM)` guard (where
`row = tid`), so only `tid < 64` ran and they all took `part = 0`; the other three `il` groups were
never written and greedy collapsed to 146 chars.  **A row-mapping change must move with its guard.**

**Top remaining follow-up (for a fresh session):** the IQ3_S dequant is *still* the dominant part of the
GLU.  Two candidates, in order: (1) **preload the block fields into registers in `load_regs`** (as
`WTYPE 0/1/3/4` already do) instead of the scattered byte loads in `store_lds`; (2) **a fused gate+up
dequant** — the GLU calls `mmb_dq_row_iq3s_p` twice with identical indices/scales and only different
data, so a dual version would roughly halve the index arithmetic (~26 % of the dequant).  See the
session-22-follow-up brief at the top of `### F. MMB restructure` for the method and the traps.

---

## UPDATE — session 21 (2026-09-20): the MMB restructure lands the **A-panel double-buffered GLU tile**
## (IQ3_S) — GLU **−6.7 %**, total GPU kernel **−1.8 %**, bit-identical; `DBUF2` refuted

Tip `7c7e7fd64` (33rd commit).  The session-20 mandate.  Full record: README "session 21".

**Recon (the facts a future session should not re-derive):**

* **LDS is 65536 B/CU** → the 36864 B dense/GLU tiles run **1 block/CU (8 of 64 waves)**.  The
  warp-specialised producer/consumer + 2x LDS ring does **not** fit; A-panel-only double buffering does.
* **The naive "no-dequant" A/B is INVALID for the routed/GLU kernels**: removing the dequant changes
  the GEMM result → the MoE router picks different experts → the workload itself changes (the small
  GLU tile "sped up" 1038 → 4 ms — a workload change, not a win).  Use the **fixed-token
  `llama-perplexity`** profiler (`/tmp/profT.sh` pattern: `rocprofv3` over `llama-perplexity -f
  prompts/prose-rdna-boosts.txt -c 2048`, min-of-N); `llama-bench`'s random prefill tokens make the
  routed kernels non-comparable run to run (only the dense kernels are stable there).
* **ISA (static, no confound):** the GLU `routed_glu_kernel<64,128,32,32,5>` body is **~3800 VALU vs
  32 WMMA** per K-step (dequant-issue bound); the dense `<128,128,32,64,1>` Q8_0 body is ~1550 VALU vs
  32 WMMA (not).  That is why `DBUF` helps the GLU and not the dense.
* **No hardware bf16 pack on gfx1151** (`v_cvt_pk_bf16_f32` is gfx12-only), and the compiler's `__bf16`
  conversion is **not** RNE (10/4096 differ from `mmb_f2bf`) — the software RNE pack must stay.

**Landed:** `DBUF` template flag on `mmb_tile_gemm`/`mmb_tile_gemm_glu` — the next K-step's weight
input (`As`/`Ag`/`Au`) is double-buffered and written *before* the WMMA of the current step, so the
de-quant overlaps it instead of serializing behind the single-buffer LDS hazard.  **Enabled for WTYPE 5
(IQ3_S) on the GLU `64x32` tile only:** `routed_glu_kernel<64,32,16,16,5>` 1018.8 → **942.0 ms**
(96 calls, −7.5 %), `mrg` 1318 → **1229** (−6.7 %), MMB family −2.6 %, **total GPU kernel 4409 → 4331 ms
(−1.8 %)**, perplexity 2.40 → **2.34–2.37 s/pass**.  Fits in LDS for dense `128x128`, routed
`128x128`/`128x32`, GLU `64x128`/`64x32`; **not** for dense `128x256` or the tall `384x*` tiles.

**Type-gated because it regresses everywhere else** (measured, so it stays off): dense Q8_0 `128x128`
**+37 %** (the Q8_0 dequant is only ~5 % of that kernel), routed IQ4_NL **+16 %**, GLU big tile **+14 %**,
and the GLU small tile on the **35B Q4_K** **+5.8 %** (287 → 304 ms).  The 35B is otherwise unaffected
(mrg 287.3 vs 287.4, family 1008 vs 1008).

**`DBUF2` (double-buffer B too → one sync per K-step) is REFUTED** (GLU 1225 → 1433 ms): the extra LDS
traffic costs more than the removed barrier.  Kept guarded/disabled in the tree for cheap re-testing.

**Gates:** PPL c2048 **10.6015** (×3), greedy **`9c281c415082`** (624 chars), width probe **PASS**
(maxdiff 0).  The `DBUF=false` refactor was verified bit-identical *before* enabling the flag.

**Next steps (ranked):** (1) a **cheaper IQ3_S dequant** — the generated code spends ~4 instr/value on
bf16 RNE packing and ~5 instr on 64-bit address math around each of the 16 `iq3s_grid` LUT lookups
(it is a plain `static const` device array, so each lookup is a divergent global load; moving it to LDS
or a 32-bit-offset form is the most promising); (2) a **fused gate+up dequant** sharing the grid-index/
scale arithmetic between the two panels; (3) nothing else fits in LDS.

**Harness note:** `/tmp/profT.sh <tag> <N>` / `/tmp/profP.sh <tag> <N>` (target/35B fixed-token
min-of-N) + `/tmp/anmin.py` were used for every number above; they are throwaway `/tmp` scripts, so
re-create them from the commands in the session-21 README entry if needed.

---

## UPDATE — session 20 (2026-09-20): the `rms_norm` register-cache experiment is REFUTED (a wash);
## the second `x` read is cache-resident

Scope: the largest non-MMB kernel, `rms_norm_f32<1024, true, false>` (the HC normalized stream `xn`,
**538 ms / 3.5 %** of pp8192), reads `x` twice -- once for the sum of squares, once for the output --
and the session-19 note had listed it as the one moderate non-MMB candidate (its dismissal in item B
was only about *non-temporal* loads, which would evict the second read).  The hypothesis was that the
second read is a DRAM re-read, so caching this thread's `x` values in registers across the block
reduction should cut it.

**Implementation.** A new `int NX` template parameter on `rms_norm_f32`; the first loop stashes each
thread's `x` into `float xr[NX]` and the second reads it from registers.  The index **must** be
compile-time: a dynamically indexed local array spills to local memory, so the loop is written as
`#pragma unroll for (k = 0; k < NX; ++k) { col = tid + k*block_size; if (col < ncols) ... }` and the
launcher selects `NX = 4` only when `ncols <= block_size*NX` (4 covers `<256>` always, and `<1024>` up
to ncols 4096 -- 2560 here).

**Measured (gfx1151, bf16 KV, ub 2048, rocprofv3 kernel trace, pp8192):**

| kernel | baseline (s19) | register-cache |
|---|---:|---:|
| `rms_norm_f32<1024,true,false>` | 538.4 ms | 531.7 ms (−1.2 %) |
| `rms_norm_f32<256,true,false>` | 278.9 ms | 287.1 ms (+3.0 %) |
| combined | 817.3 ms | 818.9 ms (+0.2 %) |
| grand total | 15408 ms | 15439 ms |

**Verdict: revert.**  The second `x` read is already cache-resident, so the kernel is not DRAM-bound
at all; the `<1024>` gain is inside run noise, and the `<256>` case *loses* because the unrolled
predicated loop always runs `NX` iterations while the dynamic loop ran only `ceil(ncols/block_size)`
(2-3).  This closes the `x`-re-read line for good: `rms_norm_f32` is latency/occupancy-bound on its
block reduction, not on the second read.  **Do not retry** (a shape-exact `NX` dispatch -- 2/3/4 --
could recover the `<256>` loss, but the ceiling is a noise-level ~1 % of one 3.5 % kernel).

The tree is back at the session-19 tip `2da50418d`; no record/backup change beyond this note.

---

## UPDATE — session 19 (2026-09-20): indexer histogram atomics go per cell -> per block (pass 1)
## and per thread -> per warp (the block passes) -- family **-16 % pp8192 / -7 % pp32768** vs s18

Continuation of the session-18 indexer mandate.  Two atomic-vs-algorithm re-associations, both
bit-identical (integer counts/sums):

1. **Pass 1 (`indexer_topk_histogram_pass1`)** was doing one `atomicAdd(&histogram[bin], 1)` **per
   cell**.  On the default `additive == nullptr` path every cell of a block shares the
   `indexer_topk_block_key` and every invisible cell shares `key_inf`, so a thread's `VEC` run bins into
   at most two bins.  It now accumulates the block's visible count (`cur_h`) and a per-thread pending
   counter for the invisible `key_inf` bin, flushing with **one atomic per block** plus **one atomic
   per thread** for the invisible stream.
2. **The block passes (`indexer_topk_histogram_blocks`)** had every one of their 256 threads do
   `atomicAdd(&s_sumw, mysum)` on a **single shared address** -- 256 serialized same-address atomics
   per CU per pass.  It now warp-reduces `mysum` with shuffles and does **one atomic per warp**
   (8 instead of 256).

### Measured (gfx1151, bf16 KV, ub 2048, rocprofv3 kernel trace; session-18 baseline in brackets)

| kernel | pp8192 | pp32768 |
|---|---:|---:|
| pass 1 (`_pass1`, cell-level) | **13.4** (23.7) | **266.0** (341.8) |
| passes 2-4 (`_blocks`) | **19.8** (22.4) | **461.8** (470.4) |
| gather (`_grouped`) | 20.1 (20.2) | 259.4 (262.5) |
| `indexer_topk_radix_select` | 11.3 | 87.8 |
| `indexer_topk_hist_accum` | 5.2 | 64.3 |
| `base_scan` + `init` | 2.0 | 8.0 |
| **family total** | **71.7** (85.6) | **1146.6** (1235.7) |

**Pass 1 -44 % at pp8192 / -22 % at pp32768; the block pass -11 % / -2 %; family -16.2 % / -7.2 %.**
The pass-1 win is larger at 8K because the invisible stream is the same fraction of the cache at any
depth but the partition has fewer hist-blocks per row at 8K, so the per-thread invisible flush
amortises over fewer blocks.  The block-pass win is larger at 8K for a similar reason (fewer hist-blocks
=> fewer `s_sumw` reduction targets).  Bit-identical: PPL c2048 **10.6015**, greedy
**`9c281c415082`** (624 chars), `test-logits-width-probe` **PASS** (maxdiff 0); both changes are
integer re-associations (the same bins/sums), so every gate that was green at session 18 stays green.

### What remains

The pass-1 cell scan is still cell-level (it must see every cell to count the per-block visibility),
and the block pass is still dominated by something other than its own reduction (only ~2-11 % moved),
so it is memory/latency-bound on the `wvis` + `score` streams.  The four remaining cost centres are
now the **three block passes (461.8 ms, key-bound)**, the **gather (259.4 ms, the shared-memory
scan)**, **`select`+`hist_accum` (152.1 ms, memory-bound on the histogram array)** and pass 1
(266.0 ms).  A block-level gather/emit is still the estimated ~1.2x family lever and still needs the
column-order handling from session 18.  A cheaper intermediate for the gather: it could take `wvis`
(already computed in pass 1) and skip the `cell_pos` load/compare for any block with zero visible
cells, without touching the output order.

---

## UPDATE — session 18 (2026-09-20): the indexer's block-level histogram path (+ the `blk_cells` op
## src) -- family **-52 % pp8192 / -51 % pp32768** vs session 15, bit-identically

Implements the first half of the session-17 scoping: the op now carries **`blk_cells`** (the
block -> cell map) as `src[7]`, so it can derive `ratio r = blk_cells->ne[0]/n_blocks` and use a
**block-aligned** histogram partition.  The default `additive == nullptr` path becomes
`indexer_topk_radix_cuda_blocks`:

* hist-block h covers blocks [h*bchunk, (h+1)*bchunk) and cells [h*bchunk*r, min((h+1)*bchunk*r, n_kv));
* `indexer_topk_histogram_pass1` is cell-level and accumulates `wvis[row*n_blocks+b]` (the block's
  visible-cell count) while binning each cell (visible at the block key, invisible at -inf);
* `indexer_topk_histogram_blocks` runs passes 2-4 over blocks only, binning `wvis[b]` at the block key
  and `range_cells - sum(wvis)` at the -inf key;
* the gather is unchanged (session-17 `_grouped`) but its partition now uses the block-aligned chunk.

Validated bit-identical: PPL c2048 **10.6015**, greedy **`9c281c415082`**, width probe **PASS**,
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` OK.  Also A/B-validated at large `n_kv`: a 4x-concatenated prose
prompt (64 KB), `-c 32768 -n 16 --seed 42 --temp 0`, gives the same text hash (**`7d2e5b3e46dd`**) with
the block path and with `LLAMA_INDEXER_NOBLOCK=1` (the session-17 path), so the block-aligned partition
and the dead-suffix handling agree with the cell-level reference at ~16k context.

### Measured (gfx1151, bf16 KV, ub 2048, rocprofv3 kernel trace; session-15 baseline in brackets)

| kernel | pp8192 | pp32768 |
|---|---:|---:|
| pass 1 (cell-level) | 24.0 | 341.8 |
| passes 2-4 (block-level) | 22.4 | 470.4 |
| gather | 20.4 | 262.5 |
| `select` | 11.6 | 87.9 |
| `hist_accum` | 5.3 | 65.0 |
| `base_scan` + `init` | 2.0 | 8.1 |
| **family** | **85.6 (177.1)** | **1235.7 (2523.6)** |

**-52 % at pp8192, -51 % at pp32768 (2.94 -> 1.47 % of the run).**  The block histogram is a big win at
short context (-31 % vs session 17 at 8K) but only -9 % at 32K, because the session-17 grouped pass
already evaluated the block key once per block -- the block passes save only the per-cell
visibility/binning work (the key is still evaluated once per block).

**End-to-end prefill record:** `benchmarks/2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md` — the
WIP is **+32 % at pp2048 and +43–48 % from pp4096 to pp32768** against the **true** delivery base
(`8a2567e1e`, built separately), of which ~+14–16 % at 8K–32K is the always-on WIP (qsa3, always-QSA,
non-temporal, indexer) and ~+27–29 % is MMB/HC16.  The earlier "MMB off" arm was the WIP tree with only
the MMB gates off, so it overstated the baseline by exactly that always-on work.

### Interface change (delivery-facing)

`GGML_OP_INDEXER_TOPK` gained `src[7] = blk_cells` (`ggml_indexer_top_k(..., blk_cells, k)`), asserted
`blk_cells->ne[0] % n_blocks == 0`.  The qwen4exp graph passes `inp->blk_cells` (already reachable via
the score chain, so no allocation change).  This must fold into delivery block 14 at promotion.  A/B
switches: `LLAMA_INDEXER_NOBLOCK=1` (session-17 path), `LLAMA_INDEXER_NOGROUP=1` (cell-level path).

### Memory note

`wvis` is `nrows x n_blocks` ints, i.e. `n_tps x n_kv/r` ints -- ~64 MB at 32K/ub2048, ~335 MB at the
model's 163840 max context (per op call, pool-reused).  Acceptable here; int16/uint8 with a shared
per-CU accumulation would cut it 2-4x if it ever matters.

---

## UPDATE — session 17 (2026-09-20): the indexer's block-key sharing -- histograms and gather
## evaluate the per-block key once per block, not once per cell (**-30 % pp8192, -46 % pp32768** vs
## session 15), bit-identically

Continuation of the session-16 indexer mandate.  On the default `additive == nullptr` path the value
is **per-block** (`score[cell_blk[c]] + derived bias`); only the visibility is per-cell.  Sessions 16's
kernels still evaluated the whole value (score gather + `blk_idx`) once per cell.  This session shares
it:

* `indexer_topk_radix_histogram_grouped<256,8,4>` -- each thread walks a contiguous run of 4 cells and
  recomputes `indexer_topk_block_key` only when `cell_blk` changes (the common `ratio == 4` case: once
  per thread step), then applies the per-cell visibility.  Same integer bin counts.
* `indexer_topk_write_blocks_grouped<4>` -- the same sharing in the gather.  Each thread owns 4
  contiguous cells, keeps per-cell `g`/`e` flags with a local prefix, and the block scan runs over the
  **per-thread totals**; the running carry still gives the exact ascending-column placement.
* The `additive != nullptr` path keeps the cell-level kernels (the value is then genuinely per-cell).
  `LLAMA_INDEXER_NOGROUP=1` forces the cell-level path for A/B.

### Measured (gfx1151, bf16 KV, ub 2048, rocprofv3 kernel trace; session-15 baseline in brackets)

| kernel | pp8192 | pp32768 |
|---|---:|---:|
| histogram (grouped) | 67.3 (83.5) | 859.6 (1235.7) |
| gather (grouped) | 22.4 (46.6) | 291.3 (668.3) |
| `hist_accum` | 16.9 (32.9) | 103.1 (513.2) |
| `select` | 15.0 (15.0) | 100.1 (97.6) |
| `base_scan` + `init` | 2.1 | 8.2 |
| **family total** | **123.7 (177.1)** | **1362.3 (2523.6)** |

**-30.2 % at pp8192, -46.0 % at pp32768 (2.94 -> 1.62 % of the run)**.  Bit-identical: PPL c2048
**10.6015**, greedy **`9c281c415082`** (624 ch), `test-logits-width-probe` **PASS** (maxdiff 0),
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK.  The grouped gather alone is -49 % of the
gather at 32K.

### Reconnaissance: the full block-level selection needs an op-interface change

The next step (resolve the top-k over `n_blocks` weighted by visible-cell count, then emit cells from
the selected blocks instead of scanning cells) was scoped.  It needs the op to carry **`blk_cells`**
(the block -> cell map; currently not a `src` of `GGML_OP_INDEXER_TOPK`) so the emit can enumerate a
block's cells; `r` is then `blk_cells->ne[0] / n_blocks`.  Two subtleties make it more than a kernel
swap: (1) the output order is ascending **column**, and within a block the `blk_cells` slot order is
`idx % r` (position/rank) order, which is *not* column order under mrope/ranked cells -- the per-block
cells would need a small sort; (2) the **dead/spare block** (unpooled incomplete-tail + empty cells,
`blk_idx == INT32_MAX`) has no `blk_cells` entries, so its visible cells still need a scan or a
memory-layer change to record them.  An op-interface change on delivery block 14 is a bigger,
delivery-affecting step; the block-key sharing landed here captures the per-block insight without it,
and a block-level `hist_accum`/emit remains the estimated further ~1.5-2x.

---

## UPDATE — session 16 (2026-09-20): the indexer's full-width count pass is gone (-19.6 % of the
## family at 32K); the "compact after pass 1" hypothesis is measured and refuted

Scope: the first indexer session (the session-15 mandate).  All of it is bit-identical -- PPL c2048
**10.6015**, greedy **`9c281c415082`** (624 ch), `test-logits-width-probe` **PASS** (maxdiff 0),
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK.

### What landed: derive the gather's counts from the radix histograms

The op used to do **6 full key-evaluation passes** over the `n_kv x n_tps` cells per layer: 4 radix
histograms + `indexer_topk_count` + `indexer_topk_deterministic_write`.  The count pass only produced
the gather's per-block `> prefix` / `== prefix` counts.  But **those counts are already in the radix
histograms**: a cell with `key > final_prefix` differs from the prefix at some byte `p`, is larger
there, and therefore sits in a bin above the selected one in exactly pass `p`; the equal cells are the
last pass's selected bin.

* `indexer_topk_hist_accum` folds each pass's suffix bin counts into `g_cnt` and, on the last pass,
  `e_cnt`.  Cost is O(`nrows x bpr x 256`) per pass, **independent of `n_kv`** -- it does not grow with
  context.
* The radix histogram blocks now cover **contiguous** column ranges (they used to interleave), so the
  per-block counts and the write's blocks agree.
* `indexer_topk_write_blocks` replaces `indexer_topk_deterministic_write`: one CU per (row, contiguous
  range), a **shared-memory** running carry over the range's 256-column tiles, same ascending-column
  placement.  (`indexer_topk_count` is deleted.)

### Measured (gfx1151, bf16 KV, ub 2048, rocprofv3 kernel trace; session-15 baseline in brackets)

| kernel | pp8192 | pp32768 |
|---|---:|---:|
| `indexer_topk_radix_histogram` | 84.5 (83.5) | 1250.3 (1235.7) |
| `indexer_topk_write_blocks` (was `deterministic_write`) | 41.9 (46.6) | 570.2 (668.3) |
| `indexer_topk_hist_accum` (was `count`) | 17.0 (32.9) | 102.7 (513.2) |
| `indexer_topk_radix_select` | 14.9 (15.0) | 97.0 (97.6) |
| `base_scan` + `init` | 2.1 (2.1) | 8.2 (8.7) |
| **family total** | **160.4 (177.1)** | **2028.4 (2523.6)** |

**-9.4 % at pp8192, -19.6 % at pp32768 (2.94 -> 2.39 % of the run)**, bit-identical.  The win is
long-context-weighted by construction: the eliminated pass scaled with `n_kv`, its replacement does not.

**Correction to the session-15 framing:** the additive mask is **not** the stream the histogram reads.
In the default prefill path the derived visibility + derived per-block bias mean the mask is never
created, so `additive == nullptr` and the kernel reads only `cell_blk`, the score gather and the derived
`blk_idx`/`cell_pos`.  The cost is key-evaluation/gather work, not mask bandwidth.

### The refuted hypothesis: "compact after pass 1"

Implemented fully (candidate list in ascending column order, count/scan/write, list-side radix passes
and gather) and measured: **326 ms vs the 177 ms** family at pp8192 -- a loss.  The premise assumes the
cells sharing the k-th value's top-8 key byte are ~1/256 of the cache.  They are **~50-60 %**:
`max_clen` measured 0.5-0.625 x `n_kv` at 4-8K.  The reason is the value structure -- the score is per
4-cell block, and the relu-heavy per-block scores are heavily tied (plus causal masking makes the
visible count approach k near the startup), so one 8-bit bin holds a huge tie group.  The compaction
gather + indirection then cost more than the three passes they replaced.  **Do not retry without a
finer first bin (see the brief's hypothesis 2).**

### Two implementation traps

1. **The running carry must be in shared memory.**  A per-thread register carry is invisible to the
   other threads, so tiles after the first placed cells with a stale base -- the same-seed text moved
   (`b4888d290fe2`); with a `__shared__` carry it is exactly `9c281c415082` again.
2. **`hist_accum` must be coalesced.**  The first form had one thread per (row, block) scanning 256
   bins: 53 ms per call (uncoalesced warp stride) and made the whole change a **loss** (197.5 ms
   family).  One 256-thread block per (row, block) with a warp-shuffle reduce is 17 ms at pp8192.

---

## UPDATE — session 15 (2026-09-20): qsa3 is now a compile-time gate; rocprofv3 was silently
dropping env-gated paths (`rocprofiler-register` setenv race)

While opening the indexer work, the pp32768 profile looked like it had found a long-context bug: the
qsa3 attention kernels were **absent** and the slow `flash_attn_qsa` VEC kernel ran instead (16.9 % of
the 96 s run).  It was not a long-context bug -- it was a **profiler artifact**.

**The trail, and what killed each hypothesis:**

* Non-profiled `llama-bench -p 8192` with `GGML_CUDA_QSA3=1` vs `=0`: **1015 vs 860 t/s**; at `-p 32768`
  **960 vs 822**.  So qsa3 was active at both -- but under `rocprofv3` the qsa3 kernels vanished and
  `flash_attn_qsa` ran at both 8K and 32K.
* 15 of the first pp8192 profiles had taken qsa3; later ones did not, with no code change in between.
* The graph-side gate printed `gate=1` (all terms satisfied) and the backend support predicate never
  returned false -- yet the profiled op ran without the packs.
* **Hard-gating qsa3 at COMPILE time (removing the env read) fixed it**: profiled pp8192 and pp32768
  then took qsa3 (480 / 1920 kernels).  That isolated the cause to the env read.

**Root cause (found by web search -- the pi web tool is gone, `curl` against the GitHub API worked):**
`rocprofiler-register` in the ROCm 7.14 build (`librocprofiler-register.so.0.6.0`, dated Jul 9) calls
`setenv()` with `GLOG_*` vars during early init.  That can reallocate the process environment under a
concurrent `getenv()` in the target app and intermittently make an env read return unset.  It is
**ROCm issue #10196** ("segfault with HIP / rocprofiler-register mutating env in early init"), fixed
upstream **2026-09-15** in **rocm-systems PR #11620**; our build predates the fix.

**Change:** the qsa3 gate is now `LLAMA_QSA3_ENABLE` (compile-time, default 1; `-DLLAMA_QSA3_ENABLE=0`
compiles the path out) instead of `GGML_CUDA_QSA3`.  Verified: profiled pp8192 takes qsa3 (480 kernels)
and pp32768 takes qsa3 (1920); non-profiled t/s unchanged (1024 at pp8192); PPL **10.6015** and greedy
**`9c281c415082`** unchanged; QSA / GDN / FA-EXT OK; width probe PASS; `plain == draft-mtp` identical.

**Methodology rule for every future session:** `rocprofv3` can silently drop an **env-gated** path, so
verify the gate you care about from the kernel names in the trace and corroborate with a **non-profiled**
t/s A/B (no profiler => no `rocprofiler-register` => no race).  The same risk applies to `GGML_CUDA_MMB`
and `GGML_CUDA_MMB_HC16`; they read stable in every profile so far, but a future "regression" that only
appears profiled should be checked here first.  The indexer kernels are not env-gated and are safe.

**Bonus:** with the artifact removed, the long-context indexer numbers are clean -- pp32768 indexer
**2523.6 ms = 2.94 %** (histogram 1235.7 / write 668.3 / count 513.2 ms), see the brief.

## UPDATE — session 14 (2026-09-20): the non-temporal hint generalises — but **loads only**, and
## per-kernel (concat −29.8 ms, moe −39.8 ms; ssm rejected), bit-identical

Session 13 warned the sweep of the other streaming kernels must not be done blindly.  This session did
it for the three session-5b/6 targets, testing **load-only / store-only / both** separately (three
profiles each, `rocprofv3` kernel time, gfx1151 pp8192):

| kernel | baseline | load-only | store-only | both | kept |
|---|---:|---:|---:|---:|---|
| `concat_transposed_src1_dim0` | 357.3 | **327.5** | 398.7 | 377.6 | **load-only (−29.8)** |
| `moe_weighted_reduction_f32_vec4` | 383.8 | **344.0** | ~416 (both−load) | 375.8 | **load-only (−39.8)** |
| `ssm_conv_long_token_f32` | 292.9 | (both−store) | 326.9* | 372.3 | **none** |
| `unary_gated_op_kernel` | 228.3 | **190.3** | — | — | **load-only (−38.0)** |
| `k_bin_bcast` (add/mul) | 244.0 | 296.8 | — | — | **none (load hurt +52.8)** |

\* `ssm` store-only is not reproducible (272.2 once, 326.9/327.0 twice) and the unchanged kernel's own
baseline swung 292.9 -> 313.6 between runs, so `ssm` is noise-dominated here and is left alone.  The two
kept kernels reproduce to <0.5 ms across runs (concat 327.9/327.9/327.5, moe 344.0/343.6/344.0).

**The rule:** a non-temporal **store** evicts the output the next op is about to read (concat's dst
feeds the weighted reduction immediately; moe's dst feeds the residual add), so it consistently loses;
a non-temporal **load** helps when the input is streamed once and the L2/MALL is polluted.  Apply it to
the **loads of pure-streaming kernels only**, and always A/B load vs store.

Implementation note: `__builtin_nontemporal_*` rejects HIP's `float4` struct (builtin scalars/vectors
only), so the moe kernel uses an `ext_vector_type(4)` view of the same 16 bytes.  The same limit breaks
the `__half` instantiations of the **templated** kernels, so `common.cuh` now has
`ggml_cuda_nt_load<T>()` — the hint for `float` via `if constexpr`, a normal load otherwise (a templated
kernel gets the hint on its f32 path only).  `k_bin_bcast` shows the sweep stays empirical: the same
hint cost **+52.8 ms** there.

Bit-identical (value-preserving hints): PPL c2048 **10.6015**, greedy **`9c281c415082`** unchanged,
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK, width probe PASS (maxdiff 0),
`plain == draft-mtp` byte-identical (`c0f8fb2b6fc7`).  Combined kernel saving ~70 ms of the 15.5 s
profile (−0.45 %); the ALL-total is noisier than the per-kernel times, so judge on the kernel trace.

**Still unmeasured:** the qsa3 pack, the `rms_norm`/unary producers, and `ssm_conv` under a different
block shape / `split_n_t`.

## UPDATE — session 13 (2026-09-20): the `dsv4_hc_pre` residual was L2/MALL pollution; non-temporal
## accesses close it (−18.6 % pre kernel time), bit-identically

Session 5e left this as "~18 % kernel-local headroom" with "runtime-stride address arithmetic and the
strided dst write" as the remaining suspects, and session 6's `mmb_*` sweep found only dead ends.
This session re-measured it in the bf16 era and found the actual cause: **the kernel is fine in
isolation; in situ its streaming buffers thrash the L2/MALL that the surrounding GEMM weight streams
need.**

**The measurement that cracked it.**

* A faithful standalone copy of the exact kernel/traffic/shape (n_embd=2560, hc=4, nt=2048; bf16
  `x` + bf16 `gate` read, F32 `dst` + bf16 `dst16` written) runs at **0.434-0.535 ms** (215 GB/s).
* The same kernel in the model takes **0.605 ms** (190 GB/s), and the trace shows `dsv4_hc_post` at
  206 GB/s on the same machine — so the environment is *not* the limit.
* `vec4`/`vec8` (2D grid, `uint2`/`uint4` loads, proved bit-identical) are **worse in situ**
  (625 / 642 us vs 605): the load-width/ILP line is closed.
* **Non-temporal loads/stores** (`__builtin_nontemporal_load`/`_store`) on the bulk accesses
  (`x`, `gate`, `dst`, `dst16` in `pre`; `x`, `residual`, `dst` in `post`) fix it:
  `dsv4_hc_pre_f32` **605 -> 493 us/call (-18.6 %, 190 -> 234 GB/s**, i.e. the pattern ceiling),
  `dsv4_hc_post_f32` **918 -> 880 us/call (691 -> 660 ms over 752 calls)**.  All-kernel total
  15596 -> 15501 ms.

They are **value-preserving cache hints, so the change is bit-identical**: PPL c2048 stays **10.6015**,
greedy **`9c281c415082`** (624 ch), `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK, width
probe PASS (maxdiff 0), `plain == draft-mtp` byte-identical (`c0f8fb2b6fc7`).  e2e is inside run noise
(pp2048 1041.6 -> 1047.2, pp8192 1015.6 -> 1017.9, same-session interleaved, `-r 2`) — judge it on the
kernel trace, per the standing rule.

**Two traps worth keeping:**

1. **A microbenchmark that isolates a kernel can miss a real in-situ win** — this one is invisible
   standalone (identical timing with and without the hints) because there is no competing traffic.
   The whole session-5e/6 "the pattern can do 232.6 GB/s" framing measured an *idle* part.
2. `mixed` (the `dsv4_hc_pre` output) is written F32 **and** bf16 because `all_bf16_consumers`
   rejects it: its consumers include the GDN `ssm_alpha/beta` `[2560x48]` F32 GEMM and the MoE
   router F32 GEMM (the expert GEMMs are bf16).  Making it BF16-only would save ~21 MB/call but
   changes the GDN recurrence and MoE routing input numerics — out of scope, recorded as a follow-up.

**Broader follow-up (unmeasured):** the same hint is untested on the other large pure-streaming
kernels (`concat_transposed_src1_dim0` 358 ms, `moe_weighted_reduction` 384 ms,
`ssm_conv_long_token_f32` 292 ms).  It must **not** be applied blindly — kernels with real reuse
(`qsa3_attn`'s K/V cache, the `mmb_*` weight panels) can only lose.

## UPDATE — session 12 (2026-09-20): the HC normalized stream `xn` is BF16-only — +4.4 % pp8192 /
## +3.8 % pp2048 over session 11, a numerics change (PPL re-baselined)

Session 11's dead-F32-store skip stopped at `xn`: its two delayed consumers -- `dsv4_hc_pre` src[0]
and the tiny-M `hc_*_inject` F32 GEMMs -- still read F32, so the graph only set `bf16_copy` and the
fused `rms_norm+mul` kept writing its F32 output (`rms_norm_f32` 1168 ms, ~7 %).  This session gives
both consumers a BF16 arm and lets `xn` be marked **BF16-only**:

* `dsv4-hc.cu`: `dsv4_hc_pre_f32` gains an `xbf16` template arm; when `xn` is BF16-only the launcher
  reads the bf16 slot instead of the (never written) F32 tensor.
* `mmb.cu`: `mmb_tiny_m_f32_kernel` gains an `XBF16` arm (4 bf16 per `uint2` load, RNE-expanded); the
  tiny-M launcher looks up the bf16 slot.  `ggml_cuda_mmb_reads_bf16_act()` is the new predicate
  "this dense GEMM reads the activation through the bf16 cache" (every weight type, plus the tiny-M
  F32 kernel) that `all_bf16_consumers` uses.
* `ggml-cuda.cu`: `all_bf16_consumers` now accepts a tiny-M F32 `MUL_MAT` and a `DSV4_HC_PRE`
  `src[0]` as bf16-aware consumers, so `xn` classifies BF16-only.

**A slot lifetime bug had to be fixed on the way (it cost a NaN PPL).**  Slot 0 is shared by every
"generic" activation copy, and `dsv4_hc_pre` *reads* `xn` from a slot while *writing* its own output
to slot 0; between the `xn` producer and its delayed consumers the intervening producers (e.g.
`lo = silu(scale(down))`) overwrite slot 0.  So `xn` now gets a **dedicated producer slot (4)**: the
marking pass assigns it when it sees the `DSV4_HC_PRE` consumer
(`ggml_cuda_mmb_mark_bf16_slot`), and producers reserve through `ggml_cuda_mmb_reserve_auto`
(dedicated slot if assigned, else 0).  The first attempt (no dedicated slot) produced a
coherent-*looking* but corrupt PPL of **492474** -- the huge apparent speedup was degenerate MoE
routing, exactly the session-5e trap: **run PPL before believing a prefill gain**.  Two smaller traps:
the mark is on the *fully-rooted* `mul` while `x->view_src` is only one level (a double reshape), so
the root walk must be recursive; and the tiny-M `X` row stride is in bf16 elements, not floats.

Gates: PPL c2048 **10.6015** (`HC16=0` 10.5771; session 11 was 10.6428 -- all within the ±0.68 bar);
greedy `-f prompts/prose-rdna-boosts.txt -n 128 --seed 42 --temp 0 -c 8192` **`9c281c415082`**
(624 chars, reproducible; `HC16=0` gives `c3f24ac9c114`); `FLASH_ATTN_QSA` / `GATED_DELTA_NET` /
`FLASH_ATTN_EXT` OK; width probe PASS (maxdiff 0); `plain == draft-mtp` byte-identical
(`c0f8fb2b6fc7`, 470 chars).

Same-session A/B (gfx1151 IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `-r 2`, two interleaved reps):

| tip | pp2048 | pp8192 |
|---|---:|---:|
| `af2f70580` (session 11) | 999 | 977 |
| this session (`xn` BF16-only) | **1043** | **1014** |

The kernel trace attributes it (`rocprofv3 --kernel-trace`, pp8192; all-kernel total 16744 -> 15594 ms
= **-6.9 %** for the whole `HC16` port, code loaded at `8a2567e1e` + this WIP): `mmb_cvt_f32_bf16`
648.8 -> 0.8 ms (the last conversion is `ple_embd`), `dsv4_hc_pre_f32` 743.8 -> 460.9 ms (bf16 `x` +
`gate`), `mmb_tiny_m_f32_kernel` 576 -> 431 ms (bf16 `X`), `rms_norm_f32<1024,true,false>` 649.5 ->
569.5 ms (the skipped F32 store), `unary_gated` 255.6 -> 227.6 ms.

Already-passing gates and the W=1..8 purity are unaffected by construction: in the decode/verify band
(`T <= 8`) the GEMMs are below `mmb_min_t`, so `xn` is never BF16-only there and the F32 path runs.

## UPDATE — session 11 (2026-09-20): the producer port's dead-F32-store skip — +2.3 % pp8192 /
## +3.3 % pp2048, still bit-identical

Session 9's producers wrote the F32 output **and** the BF16 slot copy.  A `rocprofv3` before/after shows
that of the 650 ms of `mmb_cvt` eliminated, only ~489 ms was net: the producers paid ~161 ms in extra
BF16 stores, and `rms_norm` in particular grew 1053 -> 1168 ms.  But for several activations the F32
output is **dead** — every consumer reads the BF16 copy through the MMB activation cache — so the F32
store is pure waste.

The graph optimizer now classifies each MMB dense GEMM activation's consumers: if every one (through
views/reshapes) is a quantized/bf16 `MUL_MAT` or `MUL_MAT_ID` (i.e. reads the BF16 cache), the
activation is marked **BF16-only** and the producer skips its F32 store; otherwise it keeps the old
"F32 + BF16 copy" behaviour.  The five producer kernels (`rms_norm+mul`, `sigmoid/silu+mul`,
`scale+unary`, generic unary, `dsv4_hc_pre`) each gained a `store_f32` flag.

It is **bit-identical** — the GEMM still reads the same BF16 value and nothing else read the F32
output: PPL c2048 stays 10.6428, greedy `9930c674a6ca` unchanged, the `W = 1..8` probe is pure,
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` pass.

Measured (gfx1151, IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`):

| config | pp2048 | pp8192 |
|---|---:|---:|
| `HC16=0` | 975.3 | 952.4 |
| `HC16=1` (session 9, always-emit) | ~994 | ~966 |
| `HC16=1` (this session, dead-store skip) | **1007.4** | **974.0** |

The `unary_gated` producer drops 302 -> 229 ms; `xn` stays on the copy path (`dsv4_hc_pre` src[0] and
the tiny-M inject still read F32), so `rms_norm` is unchanged.  **Next lever here:** give those two
consumers a BF16 arm and `xn` could go BF16-only too — `rms_norm` is 1168 ms and would drop ~30 %,
but it is a numerics change on the main hidden stream (the same class as the gate BF16), so it wants
the PPL gate.

---

## UPDATE — session 10 (2026-09-20): `ssm_alpha/beta` (M=48) — profiled, three kernels tried,
## **rocBLAS stays** (the faster WMMA tile costs +0.04 PPL)

Item 5 of the next-work list.  A `rocprofv3` kernel trace on the target model (gfx1151, IQ4_XS
Flash-Next, bf16 KV, `-b/-ub 2048`, pp8192, `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1 GGML_CUDA_MMB_HC16=1`)
confirms the target: the `ssm_alpha/beta` GEMMs are `[M=48, K=2560]` F32 and run on rocBLAS
`Cijk_Alik_Bljk_SB_MT32x32x8...` at **207.3 ms / 576 calls (0.360 ms)** = 1.26 % of the 16.43 s
kernel total.  Three replacements were built and measured against it:

| kernel | ms | per call | exact f32? | PPL c2048 |
|---|---:|---:|---|---:|
| rocBLAS MT32x32x8 (baseline) | 207.3 | 0.360 | yes | 10.6428 |
| `mmb_f32split_kernel<64,64,16,32>` (WMMA f16 hi/lo, BM=64, 32 T-blocks) | **179.7** | **0.312** | no | 10.6824 |
| `mmb_f32split_kernel<64,128,32,32>` (WMMA, BM=64, 16 T-blocks) | 247.0 | 0.429 | no | — |
| `mmb_small_m_f32_kernel<48,64,32>` (exact-f32 SIMT tile, W once / 64 tokens) | 251.9 | 0.437 | yes | 10.6748 |

**Verdict: keep rocBLAS (i.e. `MMB_F32SPLIT_MIN_M=128` and no small-M arm).**  Two independent
reasons, both measured:

1. The faster WMMA tile is **not worth its quality cost**.  The BM=64/BN=64 tile is 13 % faster on
the kernel (0.312 vs 0.360 ms), but its f16-hi/lo split is an approximation and `alpha`/`beta` gate
the GDN recurrence: **PPL c2048 10.6428 -> 10.6824 (+0.04)**.  End to end the win was ~0.17 %
(kernel-time, within the bench noise) — a bad trade for a systematic quality shift.
2. The **exact-f32** SIMT tile is *slower* than rocBLAS (0.437 vs 0.360 ms): a 48x64 output tile with
`RM=3, RN=4` has an FMA:LDS ratio of only ~1.7, so it is LDS/issue-bound, not memory-bound.  Even
exact, its PPL still shifts a little (+0.03) purely from the different f32 reduction order — the gate
is that sensitive.

Both experiments were reverted; the tree is back at `d6b803e70`.  A revisitable direction if someone
wants to push further: a properly register-blocked exact-f32 tile (`RM=4, RN=8, BK=16`) or rocBLAS
algorithm/split-K tuning — but it must beat 0.360 ms **and** stay exact-F32, because the WMMA split
already loses on quality alone.

---

## UPDATE — session 9 (2026-09-20): the bf16-producer port is done — the whole `mmb_cvt` bucket is
## eliminated, bit-identically (+1.3 % pp8192 / +2.0 % pp2048); plus a **delivery** op-name bug fix

Session 8 did the gate and the HC normalized stream.  This session generalised the mechanism and
closed the rest of the port.  Tip **`d6b803e70`**, three commits on top of the session-7 pack.

### What is done

**A general "activation wants a BF16 copy" mark.**  The graph optimizer now walks every MMB dense
prefill GEMM and marks its activation (`src1`'s root) `bf16_copy`.  Producers that can emit one write
it into the pinned slot 0 **in addition to** their F32 output, and the GEMM's `mmb_bf16_activation`
conversion finds it in the slot and is skipped.  The F32 output stays valid for every other consumer
and the copy is RNE-rounded exactly as `mmb_cvt_f32_bf16`, so the GEMM result is **bit-identical**.
The honoured producers are:

| producer | file | what it covers |
|---|---|---|
| fused `rms_norm+mul` | `norm.cu` | HC `xn`, every norm feeding a GEMM |
| fused `sigmoid/​silu+mul` (`ggml_cuda_op_unary_mul`) | `unary.cu` | `attn_gated`, `final_output` (the gated norm) |
| fused `scale+unary` | `unary.cu` | the HC `lo = silu(scale(down))`, the `hc_gate` activation |
| generic unary | `unary.cu` | any other unary activation |
| `dsv4_hc_pre` | `dsv4-hc.cu` | the HC mixed stream |

`LLAMA_MMB_CVT_LOG=1` now prints **only the `ple_embd` model-tensor conversion** (that tensor's
producer is the GGUF loader, so it cannot emit a copy).  The `hc_norm` / `hc_mixed` /
`final_output` / `attn_gated` / unary families are all gone.

### Measured (gfx1151, IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`)

| config | pp2048 | pp8192 | PPL c2048 | greedy sha |
|---|---:|---:|---:|---|
| `HC16=0` | 977.1 | 951.6 | 10.5771 | `9930c674a6ca` |
| `HC16=1` (gate + all copies) | **996.7** | **963.7** | **10.6428** | `9930c674a6ca` |

All the session-9 producers are bit-identical: the PPL and greedy hash are exactly the session-8
gate-only values, and `FLASH_ATTN_QSA` (22/22) / `GATED_DELTA_NET` (46/46) / `FLASH_ATTN_EXT` pass
while the `W = 1..8` width probe stays pure (maxdiff 0).  The **only** numerics change in the whole
port remains the session-8 gate BF16 (PPL +0.066, within the c2048 run's +-0.68 bar).

### A **delivery** bug found on the way: `GGML_OP_INDEXER_FILL` is missing from `GGML_OP_NAME`

`ggml/src/ggml.c`'s `GGML_OP_NAME` table has no `"INDEXER_FILL"` entry (the enum has
`GGML_OP_INDEXER_FILL`, added by delivery block 14; the base `8a2567e1e` confirms it is absent).
`ggml_op_name()` is `GGML_OP_NAME[op]`, so **every op from `INDEXER_FILL` onward is shifted by one**:
`GGML_OP_UNARY` prints as `MAP_CUSTOM1`, `HC_MIX` as `INDEXER_FILL`, and so on.  It is cosmetic in
the delivery (`ggml_op_name` is only used for logging -- checked), but it is exactly what made the
session-7 `MMB_CVT` log label the `lo = silu(scale(...))` producer `MAP_CUSTOM1` and sent this
session on a chase.  Fixed in the WIP by commit `d1463bff3`; **it belongs in the delivery** (a
one-line addition to block 14) when the promotion happens.

### What remains (small)

* `ple_embd` is the last `mmb_cvt`: its producer is the model loader, so a load-time bf16 copy would
  be needed (or it stays as-is; it is one conversion per graph).
* `dsv4_hc_pre`'s ~18 % kernel-local gap (session 5e) is the remaining kernel-local item;
  `ssm_alpha/beta` was investigated and closed in session 10 (rocBLAS stays).
* The delivery `GGML_OP_NAME` fix, and then **promotion** (all gates green).

---

## UPDATE — session 8 (2026-09-20): the bf16-producer port begins — the HC gate and the normalized
## stream are native-bf16 producers (+1.2 % pp8192 / +2.2 % pp2048)

Session 7's investigation (and `BF16-PRODUCER-PORT.md`) scoped this as a multi-session port.  Two of
its producers are now landed, behind `GGML_CUDA_MMB_HC16=1` (default off).  Tip **`ccf28bc65`**.

### What is done

**1. Mark lifetime + the gate marking pass.**  `ggml_cuda_mmb_marks_clear()` is now called on the
first optimize after a compute (the scheduler optimizes every split of one graph before computing
any of them, so marks must survive the remaining splits' optimize calls), and a new marking pass in
`ggml_backend_cuda_graph_optimize` marks a **gated `DSV4_HC_PRE`'s gate producer** BF16-only when
it is a dense `MUL_MAT` `[K=320, M=10240]`, MMB will take it, and every consumer (save the
view/reshape into the pre op) reads the BF16 copy.  Structure-only, as before pointers are not
assigned yet.  `MMB_HC16 gate marked BF16-only` is the one-shot log.

**2. The gate producer/consumer.**  MMB dense already had pinned slot 1 for exactly this gate
(`"1 = HC gate"`) but it was only allocated under `mmb_hc16()` and **nothing read it**.  It is now
allocated from the mark alone, and `dsv4_hc_pre` reads the BF16 copy through a new `wbf16` arm.  The
consumer falls back to F32 safely whenever the lookup is null (mark not set, producer not taken, or
`nt < MMB_MIN_T`), so the change is inert with the gate off.  This one **is** a numerics change
(the gate is written and read as BF16 instead of F32).

**3. The HC normalized stream (xn) producer.**  The fused `rms_norm + mul` of the HC normalized
stream (a broadcast `[n_embd, hc]` gamma over a 3D `[n_embd, hc, nt]` activation) now **also**
emits a BF16 copy of its output into pinned slot 0, RNE-rounded exactly as `mmb_cvt_f32_bf16`
would, so the following MMB dense down-projection finds it through `ggml_cuda_mmb_cache_lookup` and
skips its own conversion pass.  The F32 output is still written, so this is **bit-identical** (see
the PPL check below).  `LLAMA_MMB_CVT_LOG=1` shows all `hc_norm` conversions gone (57 -> 0).

### Measured (gfx1151, IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`)

| config | pp2048 | pp8192 | PPL c2048 | greedy sha |
|---|---:|---:|---:|---|
| `HC16=0` (baseline) | 973.1 | 950.8 | 10.5771 | `9930c674a6ca` |
| `HC16=1` gate only | 986.4 | 962.9 | 10.6428 | `9930c674a6ca` |
| `HC16=1` gate + xn | **994.2** | **961.9** | **10.6428** | `9930c674a6ca` |

Read it as: the gate is the only numerics change (**+0.066 PPL**, within the c2048 run's +-0.68 bar;
the same-seed greedy text is unchanged), and the xn copy is exactly bit-identical (the PPL of the
gate-only and gate+xn builds is identical to 4 decimals).  The xn conversion elimination accounts
for the pp2048 gain; pp8192 gains a little less because the run is more compute-bound.

**The win is real but the two producers are a small part of the port.**  The remaining `MMB_CVT`
bucket (642 ms / 3.8 % of pp8192) is now led by `hc_mixed` (`DSV4_HC_PRE`'s own output, 29 -> 40
conversions), `final_output` (`RESHAPE`) and the `node_*` (`MAP_CUSTOM1`) family; `hc_norm` is gone.
What is still needed: an `hc_mixed` producer arm on `dsv4_hc_pre` itself (we own that op, so it is
self-contained but every downstream consumer of `mixed` must then read BF16 — it feeds the
attention/ffn GEMMs and possibly non-GEMM ops), and the generic `final_output`/`MAP_CUSTOM1`
producers.  `dsv4_hc_post` stays at its bandwidth ceiling (session 5e).

**Trap for the next session:** the `dst16` arm added a runtime branch to the shared `rms_norm_f32`
kernel; it is inside the write loop, so bench with and without `GGML_CUDA_MMB_HC16` before assuming
the branch is free.  And the mark lifetime is the whole correctness story: a BF16 copy whose mark
outlives its graph is a garbage read, so any new producer must go through `ggml_cuda_mmb_marks_clear`.

---

## UPDATE — session 7 (2026-09-20): the qsa3 pack — 0.34 %, not 3 %, and now **fused** (0.26 % saved,
## bit-identical)

The session-5/5b next-work put "qsa3 attn + its PACK (4.9 % + **3.0 %**)" and proposed fusing the pack.
Measured on the target model (`/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`, 94 GiB, pp8192, `-b/-ub 2048`,
bf16 KV, `GGML_CUDA_MMB=1`, `rocprofv3`), **by diffing QSA3 on vs off** so the pack is attributed
exactly:

| kernel | ON ms | OFF ms | Δ = the pack |
|---|---:|---:|---:|
| `cpy_scalar<…cpy_1_scalar<__half,__half>>` (cont) | 41.0 | 0 | **41.0** |
| `cpy_scalar_transpose<__half>` | 9.5 | 0 | **9.5** |
| `cpy_scalar_contiguous<__hip_bfloat16,__half>` (cast) | 7.3 | 0 | **7.3** |
| `qsa3_attn_kernel` | 809.7 | 0 | 809.7 |
| `qsa3_merge_kernel` / `qsa3_rows_kernel` | 60.1 | 0 | 60.1 |
| `flash_attn_qsa` (VEC, replaced) | 0 | 3774.6 | −3774.6 |
| **qsa3 total** | | | **927.6 vs 3774.6 (4.07x)** |

**The entire qsa3 pack is 57.8 ms of the 16782 ms run = 0.34 %.**  With a **q8_0** KV cache it is
**86.3 ms = 0.48 %** (the extra `cpy_q_f32<…q8_0…>` 31.1 + `cpy_scalar_contiguous<float,__half>` 4.7
are the q8_0→F32→F16 chain).  The qsa3 win itself reproduces: pp8192 bf16 KV **827.2 → 966.8**
(+16.9 %), q8_0 **850.4 → 953.6** (+12.1 %).

**What the session-5 "PACK/copy (qsa3 pack) 3.0 %" bucket actually contained:**
`concat_transposed_src1_dim0` **357.5 ms (2.1 %)** — that is the **MoE output concat** (`GGML_OP_CONCAT`,
`unsigned int` elements; it is present with QSA3 **off** and has nothing to do with the pack) — plus
`cpy_scalar<float,float>` **110.7 ms (0.66 %)** (base-graph copies, also present with QSA3 off), plus
the real pack **57.8 ms (0.34 %)**.  Sum ≈ 3.1 %.  The label "(qsa3 pack)" was wrong.

**Ceiling for the proposed fusion:** the pack is cast(bf16→f16) + permute + `cont` + reshape + permute
+ `cont` = 3 passes over the cache.  The attn kernel cannot skip the pack: the packed-block layout is
what makes its per-block reads coalesced, and the same block is re-read by many groups, so packing
once amortises the re-layout.  **The maintainer asked for it anyway ("0.3 % is still 0.3 %"), so it
was implemented — see below.**

### Implemented (same session): the pack is now ONE launcher pass, bit-identical

The graph no longer builds `pk`/`pv`; it only materialises the natural contiguous F16 view
`[D=256, n_head_kv, n_kv]` (`qsa3_f16_cast`, no permute/cont).  Two new launcher kernels
(`qsa3_pack_keys_kernel` / `qsa3_pack_values_kernel`, `fattn-qsa3.cu`) do the whole re-layout in one
pass each; the launcher allocates the `pk`/`pv` scratch from `ctx.pool()`.  Both are pure permutations
of the same F16 values, so the result is **bit-identical**:

| check | old (graph pack) | new (fused) |
|---|---|---|
| PPL c2048 bf16 | 10.5771 | **10.5771** |
| greedy `sha=` (128 tok, seed 42) | `04ddb94b1529` (665 ch) | **`04ddb94b1529`** (665 ch) |
| `test-backend-ops` FLASH_ATTN_QSA / EXT | OK | **OK** |

pp8192 (IQ4_XS, `-b/-ub 2048`), pack-kernel time and t/s:

| KV | old pack | new pack | saved | t/s |
|---|---:|---:|---:|---|
| bf16 (default) | 50.5 ms (`cpy_scalar<half,half>` 41.0 + `cpy_scalar_transpose` 9.5) | **6.5 ms** (keys 4.0 + values 2.5) | **43.96 ms = 0.26 %** | 966.8 → **968.75** |
| q8_0 | 86.4 ms (incl. the q8_0→F32→F16 chain) | 49.1 ms | **37.34 ms = 0.22 %** | 953.6 → **959.0** |
| f16 | (same 2 conts) | same fused kernels (no cast needed) | — | 954.5 |

The F16/BF16 cast (`cpy_scalar_contiguous`, bf16 7.7 ms) and the quantized→F16 chain (q8_0 35 ms)
remain graph ops — fusing them is <0.05 % for the default bf16 config and needs per-type dequant in the
pack kernel, so it is left as a documented follow-up.

### bf16 HC intermediates + producer marking — investigated: **a multi-session port, not a cast**

The next-work #3 ("bf16 intermediates") and §10 ("bf16-producer marking") are the **same** underlying
change: make a producer emit bf16 natively, then have the consumer read it.  A plain `ggml_cast` is a
**net loss**, because the cast *is* the existing `mmb_cvt`:

| path | traffic per element |
|---|---|
| today: producer writes f32 → `mmb_cvt` (read f32, write bf16) → consumer reads bf16 | 4 + 4 + 2 + 2 = **12 B** |
| producer writes bf16 natively → consumer reads bf16 | 2 + 2 = **4 B** |
| extra `ggml_cast` (read f32, write bf16) → consumer reads bf16 | 4 + 4 + 2 + 2 = **12 B** (no win) |

So the win is entirely in the **producer**, and for `hc_norm` (`rms_norm * gamma`) / `hc_mixed`
(`DSV4_HC_PRE`) / `node_9` (`MAP_CUSTOM1`) that means a **fused producer that writes bf16**.

`LLAMA_MMB_CVT_LOG=1` (pp512, per layer) shows where the 642.5 ms of `mmb_cvt` goes:

| converted tensor | producer (`op` / `view_src`) | elements |
|---|---|---:|
| `hc_norm-N` | `RESHAPE` (view of the `rms_norm*gamma` `MUL`) | 5.24 M ×2 |
| `final_output-N` | `RESHAPE` | 3.15 M |
| `hc_mixed-N` | `DSV4_HC_PRE` + its `RESHAPE` | 1.31 M ×2 |
| `node-N` | `MAP_CUSTOM1` | 0.16 M |
| `ROCm0#ple_embd#0` | `NONE` (model tensor) | 1.31 M |

**What the reference (`~/pwilkin-llama-cpp`, `hc-mix.cu` + `ggml-cuda.cu`) does:** `hc_mix_reduce_bf16`
reads bf16 copies it gets from `ggml_cuda_mmb_cache_lookup(xn)` / `(gate)` — and those copies exist
because its graph-optimizer pass **marks native-bf16 producers** (`ggml_cuda_mmb_mark_bf16_only` on
MMB-chain outputs, the `rms_norm`+`mul` fusion with an `out_xn_bf16` output, the HC combine/norm
fusions).  It is gated `LLAMA_MMB_HC16` / `LLAMA_HC_BLK16` / `LLAMA_HC_RES16` and depends on
reference-only matching helpers (`ggml_cuda_match_hc_combine_norm`, `ggml_cuda_hc_mix_closed`) plus the
whole mark-lifetime machinery (`ggml_cuda_mmb_marks_clear` is **never called** in our tree).

**What our tree already does:** the GLU→routed-down bf16 reuse is in place via `mmb_slot[2]` (no `GLU`
producer appears in the `mmb_cvt` log), and `mmb_down16` / `mmb_blk16` / `mmb_res16` knobs exist but are
default-0 and unmarked.  `dsv4_hc_post` (685 ms) is *already at the bandwidth ceiling* (~216-252 MB per
launch / 0.9 ms ≈ 240 GB/s), so its `x` re-reads are L1-cached and restructuring it is a wash — the
lever really is bytes, i.e. native bf16.

**Scope:** port the reference's fused bf16 producers (at least the `rms_norm`+`mul` `out_xn_bf16`
fusion and the HC combine/norm matcher), wire `ggml_cuda_mmb_marks_clear` into the graph optimizer, add
the bf16 arm to `dsv4_hc_pre` + the `hc_mixed` consumer, and re-run the PPL gate (it is a numerics
change on the HC path).  That is a multi-session project with a `~2.3 %` ceiling on Flash-Next; it is
**not** a one-line cast.  Cheap inputs checked and rejected: `GGML_CUDA_MMB_CACHE` 32/128 is a wash on
Flash-Next (960/957/957 t/s), and a plain `ggml_cast` is a net loss by the table above.

**The turnkey brief for that port is [`BF16-PRODUCER-PORT.md`](BF16-PRODUCER-PORT.md)** — mandate,
environment, the reference file:line map, the ordered step plan, the gates and the pitfalls.  Nothing
has been started; the tree is at `2a73b02e4` with only the qsa3 pack landed this session.

**The correct targets (current profile, bf16 KV, total 16782 ms):** `dsv4_hc_pre` 743.9 + `_post` 685.3 =
**1429 ms (8.5 %)** via bf16 intermediates (~2.3 % win, next-work #3); `mmb_cvt_f32_bf16` **642.5 ms
(3.8 %, 3 insts)** via bf16-producer marking (§10); the MoE epilogue pair `moe_weighted_reduction`
384.1 + `concat_transposed_src1_dim0` 357.5 = **741.6 ms (4.4 %)** (delivery block 13/14, out of this
WIP's scope but the actual size the pack was credited with).

**Methodology (do not relearn): a bucket label is not evidence.**  Attribute a suspected cost by
turning the feature **off** and diffing the kernel trace; the session-5 bucket mixed three unrelated
copy families under one name.

---

## UPDATE — session 6 (2026-09-20): the `mmb_*` kernels are at their gfx1151 structural ceiling —
the Q8_0 IU8 restructure and the bf16 shadow are both refuted; the next lever is outside `mmb_*`

Session 5 left "`mmb_dense` (21 % of pp8192) + `mmb_routed_glu` (23 %) need a split-K / int8-IU8
restructure, not tuning" as the top item.  This session read the geometry, tried the tuning knobs and
the two restructure ideas, and **closed all of them**.  No change landed — the worktree is clean at
`7e431fc82` and every measurement below is from that build.

### What was measured

**New profile (the fast iteration model).**  The session-5 profile is on Qwen3.8-Flash-Next IQ4_XS
(94 GiB).  For iteration this session used **`Qwen3.6-35B-A3B-Q4_K_M`** (21 GiB, `qwen35moe`, GDN +
MoE, `rocprofv3 --kernel-trace`, `GGML_CUDA_MMB=1`, `-b/-ub 2048 -p 8192 -r 1`, two forward passes in
the trace, total kernel 6937 ms):

| kernel (by role) | ms | % | calls | note |
|---|---:|---:|---:|---|
| `mmb_dense_kernel<128,128,32,64,1>` **Q8_0** | **1514.8** | **21.8** | 1680 | attn_q/k/v/qkv + ssm_out + shexp |
| `flash_attn_ext_f16` | 784.4 | 11.3 | 80 | the full-attention layers |
| `mmb_routed_glu_kernel` Q4_K (small 32 + big 128) | 1093.3 | 15.8 | 320+320 | experts gate/up |
| `mmb_routed_kernel` Q5_K/Q6_K | 800.4 | 11.5 | 296+296+24+24 | experts down |
| `gdn_bf16_scan_cuda` | 380.2 | 5.5 | 240 | GDN |
| `concat_transposed_src1_dim0` | 234.6 | 3.4 | 240 | MoE output concat |
| `moe_weighted_reduction_f32_vec4` | 235.0 | 3.4 | 320 | MoE routing weights |
| `ssm_conv_long_token_f32` | 198.2 | 2.9 | 240 | GDN |
| `mmb_f32split` / `mmb_tiny_m_f32` / rocBLAS | 160.3+73.9+138.3 | 5.4 | | F32 (router + inject) |
| `rms_norm_f32` (all insts) | ~450 | ~6.5 | | |
| everything else | ~850 | 12 | | |

The MMB share is **~49 %** (dense 21.8 + GLU 15.8 + routed 11.5), matching the Flash-Next picture.

### 1. int8 WMMA is not faster than bf16 WMMA on gfx1151 (the §9 restructure is refuted)

`tools/wmma-peak-gfx1151.cpp` (new, kept; `hipcc --offload-arch=gfx1151 -O3`), 256-thread blocks,
NACC 8, both gfx11 builtins (`..._bf16_w32` with `v16s`, `..._iu8_w32` with `v4i`), plus a
simulated Q8_0 per-32-block scale epilogue on each:

| | T-MAC/s |
|---|---:|
| bf16 WMMA plain | **27.6** |
| int8 WMMA plain | **27.5** |
| bf16 WMMA + Q8_0 epilogue | 19.7 |
| int8 WMMA + Q8_0 epilogue | 14.1 |

**gfx1151's int8 and bf16 tensor cores run at the same rate**, and the Q8_0 epilogue then makes the
int8 path *worse* than the bf16 path.  The 174 T-MAC/s / "FP8 == INT8" figures in §9 and
`wip/q8-prefill-tuning` are **gfx1201** measurements; they do not transfer to Strix Halo.  The whole
"Q8_0 -> int8 IU8 WMMA, avoid the dequant staging" idea is **a net loss on the target arch** — the
current dequant-to-bf16 path is the correct design.  (27.6 T-MAC/s = 55.2 TFLOPS, matching the
handover's ~59 TFLOPS bf16 roof; the tool reports 20 CUs.)

### 2. A bf16 weight shadow is 2.44x *slower* (the on-the-fly dequant is right)

Tested in situ: the **same architecture** loaded as `Qwen3.6-35B-A3B-BF16` (66 GiB) makes MMB take the
dense weights as **WTYPE 2** (direct bf16, no dequant) instead of **WTYPE 1** (Q8_0 dequant).  Same
grid group `(16384,16)` = `attn_qkv` M=8192 K=2048, 320 calls each:

| weight format | ms | per call |
|---|---:|---:|
| Q8_0 (WTYPE 1) | 665.1 | 2.08 ms |
| BF16 (WTYPE 2) | **1621.7** | 5.07 ms |

2.44x slower.  The weight footprint doubles (17 -> 33 MB) and loses its L2 residency, so the kernel
becomes weight-stream bound; it is **not** dequant-ALU bound (if it were, removing the dequant would
help).  This also explains the earlier per-shape traffic arithmetic: the kernel is cache/bandwidth
bound on the A (weight) stream, and Q8_0's 1-byte format is a *feature*.  A persistent bf16 shadow of
the Q8_0 dense weights is therefore not a win even at the ~2.6 GB it would cost here, and the
"on-the-fly dequant, no shadow" decision is confirmed from the other direction.

**Trap:** the two models quantize different tensors, so the per-shape *call counts* differ
(e.g. `(4096,16)` is 320 calls in Q4_K_M, 640 in BF16).  Only the **same grid group** is comparable.
Do not compare totals.

### 3. Every cheap knob is a wash or worse (geometry *was* read and tuned)

All measured on the 35B Q4_K_M, pp2048/pp8192 t/s, `-r 2`:

| knob | pp2048 | pp8192 | vs default (2520 / 2296) |
|---|---:|---:|---|
| default | 2520.5 | 2296.0 | — |
| `GGML_CUDA_MMB_TILE=1` (force wide for all dense) | 2446.9 | 2231.4 | worse |
| dense `BM=64` (`<64,128,32,32,1>`) | 2354.1 | 2141.3 | **-6.6 %** (B panel reloads 2x) |
| GLU big `BM=128` (`<128,128,32,64,3>`) | 2495.5 | 2290.0 | ~-1 % |
| GLU `BN_SMALL=64` (`<64,64,32,16,3>`) | 2517.5 | 2297.7 | wash |
| `GGML_CUDA_MMB_CACHE=16` / `64` | 2514.5 | 2325 / 2323 | wash (noise; the f32->bf16 activation conversion is not redundant) |

This matches the session-1 finding that forcing narrow/wide changes pp8192 by +0.6 / -2.4 %: the
kernel is already at its tile optimum.  **`mmb_dense` runs at 14.8 T-MAC/s = 54 % of the 27.6 bf16
WMMA peak** (measured: 44.7 TFLOP of dense GEMM over 1514.8 ms); the GLU at ~10 T-MAC/s = **36 %**.
The missing fraction is dequant-issue contention (inherent, because Q8_0/Q4_K must be expanded to
bf16 for the tensor core), and the GLU pays it twice (gate + up).

### Methodology notes / traps

1. **The fast iteration model is the 21 GiB `Qwen3.6-35B-A3B-Q4_K_M`, not the 94 GiB Flash-Next.**
   It exercises the same `mmb_dense`/`mmb_routed_glu`/`mmb_routed` kernels (MMB is ~49 % of
   pp8192 kernel time vs ~52 % on Flash-Next), loads in seconds, and makes a 5-variant sweep
   affordable.  Confirm a finding on Flash-Next only if it is about the HC/`dsv4`/QSA paths (the 35B
   has none).
2. **`rocprofv3` grid columns are `grid_size_x` = blocks.x x 256 (the workgroup size), `grid_size_y`
   = blocks.y.**  Divide `grid_size_x` by 256 before matching to a shape; the MMB dense grid is
   `(M/128, T/128)` and the routed/GLU grid is `(M/BM, n_desc)`.
3. **A separate model is a clean in-situ WTYPE A/B.**  The BF16 model isolates WTYPE 1 vs 2 with the
   real kernel and real weights; no shadow plumbing is needed to test the idea.  Same for
   `Qwen3.6-35B-A3B-Q4_K_M` vs `UD-Q5_K_M` for WTYPE 3 vs 6/8.
4. **The microbenchmark builtin matters.**  gfx11's int8 WMMA takes `v4i` operands and has no
   `_gfx12` suffix; the gfx12 tool in `wip/q8-prefill-tuning/tools/` will not compile for gfx1151.

### Consequence for the next work

The session-5 "two big restructures" are done: the **int8 IU8** idea is refuted, the **bf16 shadow** is
refuted, and the **`mmb_routed_glu` geometry was read and every tile/BN knob tried**.  What was *not*
tried: a true split-K restructure (the kernel is already grid-rich — 1024+ blocks for the dense
shapes — so it would not raise occupancy) and a warp-specialised / double-buffered LDS pipeline
(the LDS budget at `BM=128` is 36 KB of the 64 KB, so double-buffering both A and B does not fit;
this is the one remaining candidate, and it is a non-trivial kernel rewrite).  The MMB path's
remaining gains must otherwise come from arithmetic that is *already* bf16 (nothing) or from outside
`mmb_*`.  The profile ranks the non-MMB prefill work as: **FA 11.3 %** > GDN scan 5.5 % > MoE
concat+weighted-reduction 6.8 % > rms_norm ~5 % > F32 5.4 %.  If the next session wants a prefill win
it should profile those on the delivery's own kernels; otherwise the responsible step is
**promotion** (all §11 gates are green — 5c/5d).

---

## UPDATE — session 5e (2026-09-19): dsv4_hc — investigated, NOT landed; the reference's win is bf16
## intermediates (a graph change), plus a residual ~18 % kernel-local gap

`dsv4_hc_pre_f32` + `dsv4_hc_post_f32` are 1432.5 ms (8.5 % of pp8192) and the reference (pwilkin
`hc-mix.cu`) is reportedly ~0.93 s against our 1.44 s.  **No change landed** — the tree is clean and
baseline pp8192 is 953.5 t/s post-revert.

### The machine's real ceiling, MEASURED (not inferred) — and the grid has to be swept

`tools/dram-bw-probe.cpp` (kept in this directory; `hipcc --offload-arch=gfx1151 -O3`), 192 MB buffers,
grid swept, 34 C, no other GPU work:

| pattern | GB/s | % of the 256 GB/s spec |
|---|---:|---:|
| pure sequential read (best grid) | **241.5** | 94.3 % |
| copy (read + write) | ~208-216 | 81-84 % |
| write-only | ~217 | 85 % |
| **the exact `dsv4_hc_pre` shape** (x + gate, 4 streams each, + dst), best grid | **232.6** | 90.8 % |
| the real `dsv4_hc_pre_f32` | **197** | 77.0 % |

So the part sustains **~240 GB/s** as I can measure it (a separate report puts it at ~255 GB/s
achievable, which would widen the gap a little), and — the part that matters — **the dsv4_hc_pre access
pattern is not inherently slow**: a clean kernel with the identical shape reaches 232.6 GB/s.  Our
kernel is therefore **~18 % below what its own pattern allows** (and ~23 % below the best pure read),
not pinned at a wall.  That is ~1.0-1.4 % of prefill, more than the 0.6 % a first pass concluded.

**Two corrections this took, both worth keeping:**

1. **The ceiling must be measured, not inferred.**  The first pass inferred it from our own kernels and
   concluded "already at the memory wall, nothing to gain".  Wrong.
2. **Sweep the grid before quoting a ceiling.**  The first pass ran grid=4096 everywhere and reported
   231.6 GB/s (read) / 227.2 (pattern); the *same kernels* at grid=16384 give **241.5 / 232.6** — ~5 %.
   Instruction sequence matters too: a 4-accumulator x4-unrolled read variant measured **worse**
   (221-230) than a plain single-accumulator grid-stride loop, so "more ILP" is not automatically more
   bandwidth on this part.  That matches the independent observation that this machine's number depends
   on the exact sequence used.

The split then:

* **~18 % kernel-local** (197 -> 232.6 GB/s) = ~133 ms = **~0.8 % of prefill**.  Not yet explained;
  what differs from the probe kernel is the sigmoid (measured free), the runtime-stride address
  arithmetic, and the strided `dst` write.
* **the dominant remaining lever is bytes**: bf16 intermediates take unique traffic 189 -> ~105 MB
  (1.8x), which at 232.6 GB/s is ~0.45 ms/launch vs 0.98 — **~2.3 % of prefill**, and that is what
  matches the reference's 1.44 -> 0.93 s.  It needs the `hc_norm`/`hc_gate` producers to write bf16, so
  it is a **graph-level change**.

The launch config was never the problem: `pre` runs `ceil(n_embd*n_tokens/256)` = 20480 blocks, fully
occupied, coalesced 128-byte reads.

**Two probe traps, both mine, each costing a GPU fault or a wrong answer:**

1. **Size buffers by bytes, index by `float4` count.**  Mixing them (`n4 = NB/4` against an `NB`-byte
   allocation) is a 4x out-of-bounds read that presents as `Memory access fault ... Page not present`
   — i.e. it looks like a HIP/driver problem, not an indexing bug.
2. **Count write bytes as the iterations actually performed**, not as one whole buffer.  Counting
   3x192 MB when only 50 MB was written reported **302 GB/s on a 256 GB/s part**; exceeding spec is
   the tell that the accounting, not the kernel, is wrong.

### Three levers tested at the KERNEL level; all exhausted

Measured with `rocprofv3` kernel time (see the methodology note below — end-to-end is unusable here):

| variant | `dsv4_hc_pre_f32` | verdict |
|---|---:|---|
| production (`expf`) | 745.0 ms | — |
| `__expf` (fast intrinsic) | 744.8 ms | **sigmoid costs nothing** |
| identity (no sigmoid at all) | 745.3 ms | **sigmoid costs nothing** |
| `float4` over the contiguous `i0` axis | neutral (end-to-end, value-preserving) | not load-width bound |
| hc loop unrolled + `__restrict__` on the 3 pointers | 729.9 ms (**-2.0 %**) | real but 0.09 % of prefill |

The -2 % was not kept: it buys 15 ms out of 16810 ms and costs a duplicated loop body.
**No register spilling in any variant** (VGPR 24 -> 32, `Scratch_Size` 0 throughout).

### What the reference actually does differently

The remaining lever is **bytes**, i.e. pwilkin's `hc_mix_reduce_bf16` — "Halogen-style 16-bit HC
intermediates", reading bf16 copies of `xn` and `gate`.  That takes the unique traffic from 189 MB to
~105 MB, a **1.8x** reduction, which matches the reported 1.44 -> 0.93 s almost exactly.  It needs the
`hc_norm` / `hc_gate` producers to write bf16, so it is a **graph-level change and a numerics change
on the HC path**, not a kernel-local fix.  (His other variant,
`hc_mix_reduce_f32_hc4_parallel`, spreads the hc loop over 4 warps with a shared-memory reduce; that
is a latency optimisation, and with the pattern itself reaching 227 GB/s in the probe it cannot be
where the gap is.)

**Conclusion:** two levers, neither a factor of two — **bf16 intermediates** (1.8x fewer bytes, a
graph change, ~2.3 % of prefill) and a residual **~18 % kernel-local gap** (~0.8 % of prefill; the first
pass put it at ~13 % because the probe had not swept its grid — see the ceiling section above).  Every
kernel-local lever that could have explained a *large* gap has now been measured and is exhausted.

### METHODOLOGY — two traps, both mine, both generalisable

1. **A value-changing kernel variant MUST be judged on `rocprofv3` kernel time, never end-to-end
t/s.**  The identity-sigmoid variant measured **-13 % end-to-end** (951 -> 824 t/s) while its own
kernel was **unchanged** (745.3 vs 745.0 ms).  The whole swing was downstream: `mmb_routed_glu`
3803.6 -> 5906.6 ms (+55 %), `mmb_routed` +392 ms, `qsa3_attn` +167 ms, on *identical launch counts*.
Blowing up the activations changes the MoE routing, and the descriptor-driven GLU launch is sized
worst-case with early-return slots, so different routing = different work.
**Prefill throughput on this model is data-dependent** — which also means any end-to-end A/B of a
numerics-changing patch needs its kernel profile checked, not just its t/s.
2. **The first diagnostic was invalid**: I passed the flag as a *runtime kernel argument*, so the
"no sigmoid" variant still compiled the `expf` **and** a select and did strictly *more* work.  The
`gated` flag is a template parameter for exactly this reason.  (Redone as a template, giving the
745.0 / 744.8 / 745.3 numbers above.)

---

## UPDATE — session 5d (2026-09-19): the W=1..8 probe — LAST GATE ITEM CLOSED

The `W = 1..8` logits matrix was the one §11 item never run, because no probe harness existed.  It
does now: **`tests/test-logits-width-probe.cpp`** (built with `cmake --build build-rocm --target
test-logits-width-probe`), adapted from `archive/work/strix-halo/issue25/logits-width.cpp`.

**Method.** Feed an identical prefix as ONE prefill batch, then decode a W-token batch
`[t_P .. t_{P+W-1}]`.  Row *j* of that batch sees exactly the same context for every W, so its logits
must not depend on W.  The probe runs W = 1..8 in separate contexts and checks (a) every row shared
with the W=8 batch matches it bit for bit, and (b) prints a per-row logits hash so two builds can be
diffed.  Env: `RS=0|from_w|<n>` (`n_rs_seq`), `FA`, `KV`.

**Result — gate PASSES.**

| prefill P | `MMB=0` row0 hash | `MMB=1` row0 hash | width purity (W=1..8) |
|---|---|---|---|
| **256** (below `MMB_MIN_T = 512`, so MMB is unreachable anywhere) | `6228d03bd2b501b4` | `6228d03bd2b501b4` — **identical** | PASS, maxdiff 0 |
| 1024 (MMB fires in prefill) | `ac4d5de3d40a2b1d` | `3703c13f03c4b25d` | PASS, maxdiff 0 |
| 2048 (MMB fires in prefill) | `1996b44e491de5c9` | `e3e4220fe83831da` | PASS, maxdiff 0 |

Read it in two halves, because the literal "MMB=1 == off" can only hold where MMB never runs:

* **Below the threshold** (`P = 256 < 512`) MMB is unreachable in *both* the prefill and the decode
  batch, and the two configs are **bit-identical across every row of every width**.  That is the
  "identical by construction" claim, demonstrated rather than asserted.
* **Above the threshold** the row-0 hashes differ — that is the **approved prefill re-baseline** (MMB
  replaces the MMQ reduction with a dequant-to-bf16 WMMA one), not a band defect.  What matters for
  purity is the second column: **`width_purity = PASS` with MMB on at every P** — no row of any width
  differs from the widest batch by even 1 ulp, so MMB introduces no width dependence.  The `T >= 512`
  gate is what keeps MMB out of the `W <= 8` band in the first place.

**Two probe bugs worth recording** (both cost a cycle and both are the same class as the `-md` trap):

1. `llama_batch_init(ubatch)` sizes the batch for the *micro*-batch; feeding a `P`-token prefill
   overruns it.  Worse, the first symptom was a **silent segmentation fault** (and a `GGML_ASSERT`
   abort for the `n_batch` half) rather than a clean error.
2. `llama_context_params.n_batch` is the max tokens per `llama_decode` call (the whole prefill batch)
   and `n_ubatch` the max per micro-batch — they are different knobs and both must scale with `P`.

Also fixed in the probe: `rows[W-1].data()` hashes the `std::vector` objects, not the floats — the
first run printed a plausible-looking `hash=` field that was hashing pointers.  The `width_purity`
column and the `row0_row1_hashes` line were always correct (they hash `.data()` of the inner vector).

---

## UPDATE — session 5c (2026-09-19): promotion gates actually run

Session 5's profile put the remaining big kernels (`mmb_routed_glu` 22.7 %, `mmb_dense` 21.1 %) out
of reach without a split-K / IU8 restructure, so this session ran the **promotion gates** instead —
the handover had listed all of them as never run.

The test models come from `test-llama-archs -o <dir>` (~200 tiny GGUF architectures), which is the
`test-generate-models` ctest fixture.  `build-rocm/bin` does not contain the test binaries by default.

| gate | result |
|---|---|
| `test-recurrent-state-rollback` qwen35-dense / nemotron_h-dense / deepseek4-moe | **PASS** (multi-seq split replay matched, max diff 0) |
| `test-recurrent-state-depth` (n_rs_seq 1..15 sweep) | **PASS** (`total failures = 0`) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** |
| `test-backend-ops -o GATED_DELTA_NET` | **46/46** |
| `test-backend-ops -o FLASH_ATTN_EXT` | **OK** (ROCm0) |

All with `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`.

**`plain == draft-mtp` greedy text on qwen4exp — PURE (byte-identical), gate PASSES.**

| config | plain | draft-mtp | verdict |
|---|---|---|---|
| `MMB=1 QSA3=1` | `bbd4bcb519e4` | `bbd4bcb519e4` | **identical** (1700 chars) |
| `MMB=0 QSA3=0` (features OFF) | `5120b28f2879` | `5120b28f2879` | **identical** (1720 chars) |

The hashes differ *between* the two configs and that is the approved **prefill re-baseline** (MMB and
QSA3 change prefill numerics by design) — but *within* each config the plain and `draft-mtp` arms are
byte-identical, which is the purity contract.  This also supplies two of the re-baseline hashes that
were previously missing from the record.

### METHODOLOGY TRAP — this result was reported WRONG first, and how

The first attempt concluded "`plain != draft-mtp`, pre-existing cause 3" from `320 chars vs 1942
chars`.  That was **entirely an artifact**:

* the **plain arm must not be given `-md`**.  Passing the draft model makes llama-cli initialise an
  MTP context even with `--spec-type none`, and it dies with `failed to initialize the context: this
  model is an MTP draft head without a trunk` -> `llama_server exited with code 1`, **exit status 1**;
* the failed run's stdout still contained the `Loading model... |\b-\b\\...` **spinner**, and the
  `grep -v '^\[' | tr -d '\b'` filter I used turned that spinner into exactly 320/318 "chars" — so I
  was hashing **a spinner against real text**;
* the 1942-char `draft-mtp` side was genuine; the 320-char side never generated a token.

Two rules out of it:

1. **Never pass `-md` to the plain arm** of a spec purity comparison.  Run the target model alone.
2. **Assert the arm actually generated something** before comparing — check `$?` and that the output
   is neither empty nor the spinner.  A comparison where one side is a load-failure artifact will
   always "diverge", and it looks exactly like a real near-tie flip.

### Other traps

1. The first purity matrix printed `LLAMA_QSA_OFF=1 ... PURE` where **both hashes were md5 of the
   empty string** — that run had failed to start.  A "pure" verdict where the two sides are empty is
   not a pass; always check the output is non-empty before comparing.  (`LLAMA_QSA_OFF=1` itself is
   genuinely unusable with `-md` here — `llama_server exited with code 1`, deterministic.)
2. `FLASH_ATTN_QSA` is now **22/22**, not the 18/18 in the older notes (cases were added);
   `GATED_DELTA_NET` is 46/46 as expected.

**Still not run:** the `W = 1..8` logits matrix with MMB on == off.  It needs a probe harness that does
not exist in the tree (there is no width-probe tool under `tests/` or `tools/`), and it is guaranteed
by construction anyway (`T >= 512` keeps MMB out of the whole `W <= 8` band).  Building that probe is
the remaining gate item.

---

## UPDATE — session 5b (2026-09-19): tiny-M F32 kernel — the hc `*_inject` GEMMs (+2.3-3.3 %)

Session-5 next-work #1 (the remaining F32 tiny-M) is **done**.

`hc_attn_inject` / `hc_ffn_inject` are M=4 (hc), K=10240, T=2048 — 1012 ms, 1.330 ms/launch, 5.7 % of
prefill.  Neither tile can serve them (M=4 gives no M parallelism: rocBLAS launches 64 blocks, the
128-row MMB tile pads 32x), and both read the 84 MB activation once yet sit at **63 GB/s** while
`rms_norm_f32` sustains **330 GB/s** — a *parallelism* wall, not bandwidth.  The two injects also
cannot be fused (different `xn`).

New `mmb_tiny_m_f32_kernel`: **one warp per token**, every lane accumulating a k-strided partial for
all M rows, so X is read once, coalesced, and reused across M in registers.  256 blocks, not 16-64.

| | per launch | GB/s |
|---|---:|---:|
| rocBLAS | 1.330 ms | 63 |
| MMB f32split | 1.825 ms | 46 |
| **tiny-M** | **0.500 ms** | **168** |

pp2048 933.0 → **964.1 (+3.3 %)**, pp4096 934.4 → **962.0 (+2.9 %)**, pp8192 934.6 → **955.9 (+2.3 %)**.
PPL c16384 bf16 3.3875 → 3.3851 (noise); decode identical (tg64 25.96; M=4 is a weight row count, so
the `T >= 512` gate still keeps the decode/verify band off this path by construction).

**Tuning (both negative, both recorded):** TT=1 beats TT=2/TT=4 (the W-traffic amortisation idea is
rejected — L2 serves the 164 KB W panels well enough); specialising `<8,TT>` → `<4,TT>` is +0.5 %.

**Trap:** the first version changed only the *launcher* and measured flat, because the new `M >= 128`
gate rejected the shape *before* the launcher ran — the kernel never executed.  The kernel trace
(symbol absent) caught it; a bench delta alone would have read as "the idea failed".

**Next-work order (revised: HISTORICAL — superseded by the AUTHORITATIVE list at the top of this file):**

1. ~~`mmb_dense` / `mmb_routed_glu` (52 % combined)~~ — still the top item in the authoritative list.
2. **`ssm_alpha/beta`** (207 ms, 0.359 vs a 0.082 floor, rocBLAS) — same ~60 GB/s parallel-wall shape
   as hc_inject had; the tiny-M kernel generalised to M=48 is the obvious next step.
3. **`dsv4_hc_pre` + `_post`** (8 %) — the HC prefill fusion (pwilkin's pair is ~0.93 s vs our 1.44 s).
   **Done in 5e** — investigated, nothing landed; the lever is bf16 intermediates.
4. **qsa3 attn + its PACK** (4.9 % + 3.0 %) — the pack is a pure copy; fusing it reclaims most of 3 %.
5. **The indexer** — 1 % at 8K, 3.2 % at 32K, growing.

---

## UPDATE — session 5 (2026-09-19): fresh profile reprioritises; F32 split by shape (+2-3 %)

**1. A fresh post-session-4 profile (MMB+QSA3 on, `-b/-ub 2048`) contradicts the session-4 ordering.**
% of total kernel time:

| family | pp8192 | pp32768 |
|---|---:|---:|
| `mmb_*` | 52.0 % | 48.2 % |
| **F32 rocBLAS** | **11.7 %** | **12.0 %** |
| `qsa3` attn/rows/merge | 4.9 % | 6.4 % |
| `rms_norm_f32` | 5.9 % | 5.5 % |
| `dsv4_hc_pre` + `_post` | 8.0 % | 7.4 % |
| PACK/copy (qsa3 pack) | 3.0 % | 3.5 % |
| indexer top-k family | 0.99 % | 3.2 % |

The **indexer is ~1 % at 8K**, not the #1 item - it was promoted on the pp2048 startup regime (0.5 %).
It does scale with `n_kv x n_tps` (3.2 % at 32K, output capped at 2051 cells) so it returns at depth,
but the **F32 path is 12 % at both** and was the bigger, better-scoped target.  Indexer deferred.

**2. F32 dense weights are now split by shape and default ON** (`GGML_CUDA_MMB_F32SPLIT=1`).
All-or-nothing was a wash because the paths disagree per shape:

| shape | rocBLAS | MMB f32split | winner |
|---|---:|---:|---|
| `ffn_gate_inp` M=512 K=2560 | 2.044 ms | **0.846 ms** | **MMB 2.4x** |
| `hc_attn/ffn_inject` M=4 K=10240 | **1.332 ms** | 1.825 ms | rocBLAS |
| `ssm_alpha/beta` M=48 K=2560 | **0.359 ms** | slower | rocBLAS |

Rule: MMB iff `M >= 128`.  pp2048 914.9 -> **933.9 (+2.1 %)**, pp4096 908.9 -> **933.0 (+2.6 %)**,
pp8192 902.0 -> **929.8 (+3.1 %)**; the old force-all mode 2 is worst at every length.  F32 GEMM
1947 -> ~1537 ms.  PPL c16384 bf16 3.3821 vs 3.3875 (bars ±0.027) = parity; decode untouched
(tg64 25.95 vs 26.00; the `T >= 512` gate is unchanged so `W=1..8` purity holds by construction).

**Trap, recorded so it is not relearned:** the first rule also took MMB when `K >= 4096` (expecting
the WMMA path to help the K=10240 hc inject pair).  It is wrong and costs 375 ms.  It looked right
only because bucketing launches by grid alone merged `hc_inject` + `ssm` into one 1.065 ms average.
**Split the bucket before believing a per-shape number.**

**Next-work order (revised, by measured cost: HISTORICAL — superseded by the list at the top):**

1. **F32 tiny-M**: the remaining ~1.5 s is `hc_inject` (M=4, 1.332 ms/launch, floor ~0.33) and
   `ssm` (M=48, 0.359 vs ~0.084 floor) - both rocBLAS-bound now.  A dedicated small-M kernel (or
   getting rocBLAS off its split-K) is the next ~8 % of prefill.
2. **`mmb_dense` / `mmb_routed_glu`** (52 % combined) - §9's Q8_0 IU8-WMMA and the routed-GLU
   geometry.
3. **`dsv4_hc_pre`+`_post`** (8 %) - the HC prefill fusion.
4. **qsa3 attn + its PACK** (4.9 % + 3.0 %) - the pack is a pure copy; fusing it into the kernel
   would reclaim most of its 3 %.
5. **The indexer** - 1 % at 8K, 3.2 % at 32K, growing; worth it once the above are done.

---

## UPDATE — session 4 (2026-09-19): two default flips + the F32 finding

**1. `LLAMA_QSA_DENSE_SHORTCUT` is now default OFF (always QSA)** — maintainer decision.  The
shortcut's rationale (below the selection width the top-k covers every cell, so sparse saves no
attention work) was **overtaken by qsa3**: in the fully-dense pp2048 regime `qsa3_attn_kernel` is
**137.8 ms vs 149.9 ms** for dense `flash_attn_ext_f16`, so qsa3 is already 8 % faster on the
attention even when nothing is skipped.  The path only still lost because indexer+top-k (20.9 ms)
cost more than the 12.1 ms saved.  Net end to end: +0.5 % pp512, -0.3 % pp1024, -1.1 % pp2048,
-1.1 % pp4096, -0.3 % pp8192 (≈ neutral, within run variance).  It removes the `n_kv == width`
numerics seam and makes qsa3 exercised at **every** context length - a `-c 2048` exercise used to be
silently dense, which is why the early "qsa3 is neutral" readings were vacuous.  Decode unaffected
(stays dense via `qsa_dense_decode_until`; verified tg64 25.80 -> 25.83).  PPL: c16384 bf16
3.3900 -> **3.3821**, q8_0 3.3879 -> 3.3861, c32768 4.3353 -> 4.3397 (noise).

**2. `GGML_CUDA_MMB_F32SPLIT` is now default 0.**  The F32 dense weights are all tiny-M (router
M=512, ssm alpha/beta M=48, hc `*_inject` M=4, shexp gate M=1) and both the MMB 128-row tile and
rocBLAS cost ~1.0 s at pp8192 (12 %).  rocBLAS measured faster: pp4096 896.0 -> **915.9**, pp8192
896.8 -> **902.9**.  A 16x256 small-M tile was tried and is **worse** (870/847 vs 895/885).

**Next-work order (revised):**

1. **The indexer** - the only thing keeping always-QSA from being a strict win.  pp2048:
   `indexer_topk_radix_histogram` 10.0 ms, `indexer_topk_deterministic_write` 4.7,
   `indexer_topk_count` 3.3, `indexer_topk_radix_select` 2.9 (n=96 each).
2. **A dedicated tiny-M F32 kernel** (or split-K): both current paths are ~10x off the memory-bound
   floor; A traffic = `(T/BN)*M*K*4`, B = `(M/BM)*T*K*4` (168 MB vs a 26 MB floor at M=512/T=2048).
3. The `mmb_*` kernels that still lead the profile (`mmb_routed_glu` 1873 ms real, `mmb_dense` 1758,
   `mmb_cvt_f32_bf16` 319) - §9/§10.
4. QSA v3 for gfx1200/gfx1100; then `qsa3_attn_kernel` itself (674.9 ms, ~93 % of qsa3).

---

## UPDATE — session 3 (2026-09-19): the qsa3 sort is done too

The `qsa3_rows_kernel` sort (session 2's remaining item, 441 ms of the 1152 ms qsa3 total) was
rewritten as a **bitmap counting sort**: `O(ns + nk/32)` instead of `O(ns^2)`, **order-exact** (PPL
bit-identical).  **Rows 441.0 -> 25.5 ms; qsa3 total 1151.9 -> 728.6 ms (4.04x vs the VEC kernel).**

Two facts from getting there, so session 4 does not repeat the detour:

1. **The index rows are NOT sets of 4-key blocks** (that was the natural guess from the source
   comments) - they are a handful of **long contiguous runs** (row0 = one run of 2051 keys, row2 =
   runs of 436/1611/4; 1-6 runs per row).  A block/group-based sort would have been *wrong*.  Always
   dump the real data before choosing the algorithm (`GGML_CUDA_QSA3_DUMP` was removed again; the
   procedure is in `README.md`).
2. A host-side `cudaMemcpy` of op data **must be ordered on `ctx.stream()`**, not the default stream,
   or it reads the tensor before the producing kernel has run.  The first dump was garbage for
   exactly this reason.

**Revised next-work order** (after session 3 - **SUPERSEDED by the session-4 UPDATE above**; kept
for history):

1. The `mmb_*` kernels lead the profile (`mmb_f32split` 2164 ms, `mmb_routed_glu` 2085+1626,
   `mmb_dense` 1904+1606, `mmb_cvt_f32_bf16` 648) - see §9/§10 for the Q8_0 IU8 and bf16-producer ideas.
2. QSA v3 for **gfx1200/gfx1100** (needs an RDNA4/RDNA3_0 WMMA variant; the wrapper is currently a
   deliberate no-op there, so only the VEC path runs).
3. `qsa3_attn_kernel` itself (674.9 ms, now 93 % of qsa3) - the remaining qsa3 cost.

---

## UPDATE — session 2 (2026-09-19): QSA v3 is DONE

**Read §8 as history; the work is landed.**  Everything below is still accurate for the MMB part.

Session 2 delivered the packed-block WMMA QSA prefill (§8) and extended it to every KV cache type:

* New file `ggml/src/ggml-cuda/fattn-qsa3.cu` (rows / merge / attn kernels, ported from the Strix
  Halo branch's `qsa-attn`).
* New optional op srcs `src[7]`/`src[8]` via `ggml_flash_attn_qsa_set_packed`; the pack itself is a
  pure graph composition (no new ggml op), built in `src/models/qwen4exp.cpp`.
* Gate: **`GGML_CUDA_QSA3=1`**, default OFF.  Prefill-only (`q->ne[1] >= 128`) and RDNA3_5, so the
  W = 1..8 decode/verify band still takes the VEC kernel and width purity holds by construction.
* gfx1201 + gfx1100 TU compiles verified (portable WMMA wrapper, no-op on RDNA4).

Measured (gfx1151, IQ4_XS, `-b/-ub 2048`): pp8192 **f16 819.4->861.7, bf16 791.2->854.5, q8_0
780.5->856.6**; PPL parity at c16384/c32768 on all three (e.g. bf16 3.3932 vs 3.3900); greedy text
coherent.  Kernel profile pp8192: **VEC 2944.4 ms -> qsa3 1151.9 ms** (attn 682.3 + rows 441.0 +
merge 28.6).

Three things session 3 must not re-derive - full detail in `README.md`:

1. **Do not remove the dense startup arm.**  `n_kv <= 2051` takes the dense path because the
   selection covers every cell there; forcing QSA there is measurably worse (pp2048 912.5 -> 903.2).
   Consequence: **qsa3 only engages above 2051 tokens**, so `-c 2048` tests silently prove nothing.
2. **The top-k rows are unsorted**, so the rank-sort in `qsa3_rows_kernel` is required.  Disabling it
   drops that kernel 441 -> 8 ms but blows the attn kernel up 682 -> 19593 ms and halves throughput.
3. **Cast on the natural contiguous cache view, never on a permuted one**, and route quantized types
   through F32 - the backend `dup` cannot permute a quantized tensor and only dequantizes to F32
   (otherwise: `ggml/src/ggml-cpu/ops.cpp:578` abort, once per QSA layer).

**Revised next-work order** (was: after session 2; superseded by the session-3 "UPDATE" above)

1. ~~`qsa3_rows_kernel` sort~~ **DONE in session 3** - bitmap counting sort, 441 -> 25.5 ms.  See the
   session-3 UPDATE at the top.  **Do not** reorder `idx` at the indexer: that tensor is shared with
   the VEC decode path, so it would be a decode numerics change.
2. The `mmb_*` kernels now lead the profile (`mmb_f32split` 2164 ms, `mmb_routed_glu` 2085+1626,
   `mmb_dense` 1904+1606, `mmb_cvt_f32_bf16` 648) - see §9/§10 for the Q8_0 IU8 and bf16-producer ideas.
3. QSA v3 for **gfx1200/gfx1100** (needs an RDNA4/RDNA3_0 WMMA variant; the wrapper is currently a
   deliberate no-op there, so only the VEC path runs).

---

## 0. TL;DR for the next session

1. Everything lives in the worktree `~/llama-wip-mmb`, branch **`wip-mmb-general`**, based on the
   maintainer's applied delivery tree at **`8a2567e1e`**.
2. The complete change is backed up here as
   **`wip/mmb-general/mmb-general.patch`** (combined, 3 files, +1451/−2) and
   **`wip/mmb-general/patches/0001..0009-*.patch`** (per-commit), plus `commits.txt`.
   Verified: `git apply --check` passes on a clean checkout of `8a2567e1e`.
3. Rebuild with the commands in §3; reproduce the numbers in §5.
4. **QSA v3 is DONE** (see the UPDATE section) — the remaining QSA work is the 441 ms sort in
   `qsa3_rows_kernel`.  The optional second lever is now **Q8_0 IU8-WMMA** (§9).

---

## 1. Why (context in one paragraph)

On Strix Halo (gfx1151) the delivery's prefill was ~2x behind the tuned RDNA3_5 stacks
(pwilkin `strix-halo` full env: 1349/1403 t/s on uniform IQ4_NL; halogen-flash 1246/1424 t/s on the
same llama.cpp GGUFs).  Root cause: our weight GEMMs run the **integer/vector MMQ** path while both
use **bf16 WMMA tensor cores** (roof ≈ 59 vs 29.7 TFLOPS on this part; prefill is compute-bound).
The parked port (`archive/work/wip-archive/iq4nl-prefill/`) proved the idea but was IQ4_NL-only and
was parked on the purity clash.  This session generalised it to the types the delivery's own models
actually use and made it purity-safe.  **GFX1201 note:** the kernels use the first-gen gfx11 WMMA
builtin, which gfx12 does not have — hence the RDNA3 gate and the portable wrapper (§6).

---

## 2. Where the code is / what changed

Two changesets in one WIP tree.

**MMB** (session 1):

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmb.cu` | new: the dequant row helpers, tile GEMM kernels, routed/GLU kernels, dispatch, predicates, shadow helpers |
| `ggml/src/ggml-cuda/mmb.cuh` | new: the exported API + `ggml_cuda_mmb_{dense,routed}_will_take` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MMB dispatch hooks in `ggml_cuda_mul_mat` / `_mul_mat_id` / `_glu`, and the **per-weight-type fusion stand-down** in the graph optimizer |

**QSA v3 / qsa3** (sessions 2-4):

| file | change |
|---|---|
| `ggml/src/ggml-cuda/fattn-qsa3.cu` | new (~600 lines): the **fused pack kernels** (session 7), rows/merge/attn kernels, the bitmap sort, the support check + launcher |
| `ggml/src/ggml-cuda/fattn-qsa.cu` / `.cuh` | qsa3 declarations + a short "prefer qsa3 when supported" hook in the VEC launcher |
| `ggml/include/ggml.h` + `ggml/src/ggml.c` | `ggml_flash_attn_qsa_set_packed` (op `src[7]`/`src[8]` = the natural F16 K/V views) |
| `src/models/qwen4exp.cpp` | `qsa3_f16_cast` + `qsa3_f16_natural` (the graph materialises only the F16 view; the launcher packs), the `GGML_CUDA_QSA3` gate, and the **`LLAMA_QSA_DENSE_SHORTCUT` default flip** |

Digest: **9 files, +2190/-11 across 13 commits**.

Both new `.cu` files are picked up by the existing `file(GLOB … "*.cu")`, but the glob is evaluated at
**configure** time - adding `fattn-qsa3.cu` needs a one-off `cmake -S . -B build-rocm` re-run (see §3).

### WTYPE map (used throughout the file)

| WTYPE | type | bytes per k-step (64 values) | row block |
|---|---|---|---|
| 0 | IQ4_NL | 36 | 18 B / 32 |
| 1 | Q8_0 | 68 | 34 B / 32 |
| 2 | BF16 (direct, no dequant) | 128 | 2 B / value |
| 3 | Q4_K | 144 (16 hdr + 128) | 144 B / 256 |
| 4 | Q5_1 | 48 | 24 B / 32 |
| 5 | IQ3_S | 110 | 110 B / 256 |
| 6 | Q5_K | 176 | 176 B / 256 |
| 7 | Q6_K | 210 | 210 B / 256 |
| 8 | IQ4_XS | 136 | 136 B / 256 |
| 9 | Q3_K | 110 | 110 B / 256 |
| 10 | IQ3_XXS | 98 | 98 B / 256 |

### Key design decisions (do not undo)

* **Prefill-only.**  `ggml_cuda_mmb_supported_*` require `T >= GGML_CUDA_MMB_MIN_T` (default 512).
  The decode/spec-verify band is `W = 1..8`, so MMB is unreachable there — `W=1..8` bit-identity and
  `plain == draft-mtp` hold **by construction**, not by measurement.
* **Per-weight-type fusion stand-down.**  The graph optimizer's MoE pair and SWIGLU→mmq fusions
  stand down only when MMB will actually take *that weight type*
  (`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`), never on the global gate.  This is the
  parked handover's resume-checklist item #5 and it is what stopped MMB-on from regressing models
  whose expert type it could not accelerate.
* **RDNA3_5 only by default.**  `mmb_enabled()` gates to `GGML_CUDA_CC_IS_RDNA3_5`; RDNA3_0
  (gfx1100/1101/1102) shares the gfx11 WMMA builtin but is untested, so it needs
  `GGML_CUDA_MMB_RDNA3=1`.  gfx12 is excluded (different builtin).
* **Portable WMMA wrappers.**  `mmb_wmma_bf16` / `mmb_wmma_f16` select the gfx11 builtin on
  non-`RDNA4` and are a **deliberate no-op** on `RDNA4`, so `mmb.cu` *compiles* for gfx1201 while
  never being launched there.  (Verified: `mmb.cu` and `fattn-qsa.cu` both compile for gfx1201 — §7.)
* **On-the-fly dequant, no shadow.**  A persistent bf16 shadow is impossible for a 120 GiB model's
  experts, so every non-trivial type is dequantized per k-step.  The legacy Q6_K shadow path still
  exists but is no longer needed for Q6_K (WTYPE 7 is on-the-fly).
* **IQ3_XXS fused GLU is default-off.**  Its routed path is a win, its fused GLU is a measured net
  loss (§5); `GGML_CUDA_MMB_IQ3XXS=1` re-enables the GLU arm for tuning.

---

## 3. Build / run

The build dir in the worktree is already configured; to rebuild:

```sh
cd ~/llama-wip-mmb
export ROCM_PATH=/opt/rocm-7.14-gfx1151
HIPCXX=$ROCM_PATH/lib/llvm/bin/clang HIP_PATH=$ROCM_PATH cmake -S . -B build-rocm \
  -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 -DHIP_PLATFORM=amd \
  -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_PATH/lib" \
  -DCMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS= \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-rocm -j 16 --target llama-bench llama-perplexity llama-cli llama-server
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
```

To recreate from scratch (the backup is authoritative):

```sh
git -C ~/llama.cpp worktree add --detach /tmp/mmb-rebuild 8a2567e1e
cd /tmp/mmb-rebuild
git apply /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/mmb-general.patch
# then the cmake configure+build above with -B build-rocm
```

**Caveat:** the WIP base is `8a2567e1e` (the maintainer's applied tree), **not** the current canonical
release tip (r9 = `76b10f1fb8391562e30364d6c307e6606100bc57`).  Before landing, the change must be
rebased onto a canonical fork rebuilt at the release base `ebbb18522` + `scripts/apply-all.sh`, then
the delivery `patches/` regenerated (see `AGENTS.md`).  None of the touched hunks are in code that
r9 changed (the r9 work was the block-15 tile kq-mask), so the rebase should be mechanical.

---

## 4. A/B and PPL commands used

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=<model>.gguf

# prefill A/B (MMB off vs on)
./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 \
    -p 2048,8192 -n 0 -d 0 -r 2
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 \
    -b 2048 -ub 2048 -p 2048,8192 -n 0 -d 0 -r 2

# PPL parity (the correctness gate for each new dequant)
F=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
```

Always **warm the model into page cache first** (`dd if=$M of=/dev/null bs=4M`); the box's page-cache
state swings prefill noticeably, and never run benches in parallel.

`GGML_CUDA_MMB=1` prints a few one-shot `MMB_GLU …` / `MMB_SHADOW …` lines — that is normal.

---

## 5. Measured results (gfx1151, ROCm 7.14, `-b/-ub 2048`, bf16 KV)

> **These were taken before the 2026-09-19 default flips** (`GGML_CUDA_MMB_F32SPLIT` 2->0 and
> `LLAMA_QSA_DENSE_SHORTCUT` ON->OFF).  The tables are still the right *method* and the MMB `off`
> column is unaffected, but the `on` column moves a little; re-measure rather than copy.

| model / test | MMB off | MMB on | PPL off → on |
|---|---:|---:|---|
| **Q4_K_M** pp2048 | 604.6 | **1015.2 (+68 %)** | — |
| **Q4_K_M** pp8192 | 568.9 | **888.1 (+56 %)** | 10.3328 → 10.2716 |
| IQ4_XS (Flash-Next) pp2048 | 723.3 | 838.0 (+15.9 %) | — |
| IQ4_XS pp8192 | 685.5 | 779.5 (+13.7 %) | 10.6938 → 10.6440 |
| Gemma4-26B-A4B Q8_0 pp2048 | 2058.5 | 2340.1 (+13.7 %) | — |
| Gemma4-26B-A4B Q8_0 pp8192 | 1723.5 | 1928.5 (+11.9 %) | parity (prose prompt is tokenizer-mismatched; both arms same ballpark) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp2048 | 2045.3 | 2303.0 (+12.6 %) | — |
| Qwen3.6-35B-A3B UD-Q5_K_M pp8192 | 1917.8 | 2141.3 (+11.7 %) | 14.4349 → 14.3907 |
| Qwen3.6-35B-A3B Q6_K pp2048 | 2128.8 | 2143.2 (+0.7 %) | — |
| Qwen3.6-35B-A3B Q6_K pp8192 | 1974.6 | 2012.3 (+1.9 %) | 14.3682 → 14.3296 |
| Qwen3.6-35B-A3B UD-Q3_K_M pp2048 | 2091.9 | 2295.2 (+9.7 %) | — |
| Qwen3.6-35B-A3B UD-Q3_K_M pp8192 | 1941.6 | 2120.2 (+9.2 %) | 14.6517 → 14.6248 |

**Interpretation**

* The Q4_K_M jump is its **Q4_K MoE expert GLU + routed down** (WTYPE 3); IQ4_XS is IQ3_S gate/up +
  IQ4_NL down; Q5_K_M/Gemma4 are their expert types.
* **Q6_K is correct but barely moves** (+2 %): its MMQ path is already efficient on this workload.
  Keep it for coverage, not as a headline win.
* **IQ3_XXS finding:** routed is a win, fused GLU is a loss.  `UD-Q3_K_M` full-on 1921 vs routed-only
  2120 vs off 1942 t/s pp8192 → the fused GLU is default-off (`GGML_CUDA_MMB_IQ3XXS=1` re-enables).
* **Model composition trap:** file names mislead.  *Flash-Next UD-IQ4_XS* is IQ4_NL 52 % + IQ3_S 36 %
  + Q8_0 9.5 % (IQ4_XS type only 1 %).  *MiniMax-M3 UD-IQ4_XS* is IQ3_S 56 % + IQ4_XS 35 % → now
  fully covered.  *UD-Q3_K_M* is IQ3_XXS 47 % + IQ4_XS 33 %.

### Coverage matrix (final)

| type | dense | routed + GLU | note |
|---|:--:|:--:|---|
| IQ4_NL, Q8_0, Q4_K, Q5_1, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K | ✓ | ✓ | |
| IQ3_XXS | ✓ | routed only | fused GLU default-off (net loss) |
| F16 / BF16 | — | — | already tensor-core via `mmf` WMMA |
| F32 | ✓ (f32split) | — | f16 hi/lo WMMA |
| Q2_K, IQ2_*, IQ1_* | — | — | deliberately out of scope |
| IQ3_M, IQ3_XS, IQ4_S, IQ4_M | — | — | **do not exist in current ggml** |

Not yet covered (optional): the legacy **Q4_0 / Q4_1 / Q5_0** (32-value blocks; Q5_1 is already done,
so these are small additions).

---

## 6. Environment knobs

MMB (`mmb.cu` / `mmb.cuh`):

| env | default | meaning |
|---|---|---|
| `GGML_CUDA_MMB` | 0 | master gate (default **off**) |
| `GGML_CUDA_MMB_MIN_T` | 512 | prefill-only threshold |
| `GGML_CUDA_MMB_RDNA3` | 0 | allow RDNA3_0 (untested) |
| `GGML_CUDA_MMB_GLU` | 1 | fused gate/up+swiglu arm |
| `GGML_CUDA_MMB_IQ3XXS` | 0 | enable the (net-loss) fused GLU arm for IQ3_XXS |
| `GGML_CUDA_MMB_BF16W` | 1 | BF16 dense weights via MMB |
| `GGML_CUDA_MMB_F32SPLIT` | **1** | F32 dense via f16-hi/lo WMMA.  **Mode 1 = shape-aware and DEFAULT since session 5**: MMB only when `M >= 128` (the MoE router, 2.4x faster); rocBLAS keeps `hc_*_inject` (M=4) and `ssm_alpha/beta` (M=48).  `0` = all rocBLAS, `2` = all MMB (pre-session-5, worst).  Measured pp8192 902.0 (0) / **929.8 (1)** / 903.0 (2) |
| `GGML_CUDA_MMB_TINY_M` / `_TINY_TT` | 1 / 1 | the hc `*_inject` warp-per-token kernel.  `_TINY_M=0` forces rocBLAS (A/B); `_TINY_TT` = tokens per warp (**1 is measured best**; 2 and 4 are worse — L2 serves the W panels) |
| `GGML_CUDA_MMB_F32SPLIT_MIN_M` / `_MIN_K` | 128 / 0 | the mode-1 shape rule.  **Do not add a K condition**: taking MMB for long K (the K=10240 inject pair) measured 1.825 vs 1.332 ms and cost 375 ms |
| `GGML_CUDA_MMB_TALL` | 2 | the tall-M tile class |
| `GGML_CUDA_MMB_SHADOW` / `_SHADOW_MB` | 0 / 6144 | legacy bf16 shadow (Q6_K/IQ4_NL); not needed now |
| `GGML_CUDA_MMB_TILE` | -1 | **diagnostic**: force narrow(0)/wide(1) tile |
| `GGML_CUDA_MMB_LOG` | 0 | **diagnostic**: one-shot `MMB_DENSE` shape log |

QSA / qsa3 (`fattn-qsa3.cu`, `src/models/qwen4exp.cpp`):

| env | default | meaning |
|---|---|---|
| `GGML_CUDA_QSA3` | (removed) | was the qsa3 env gate.  **Since session 15 the gate is compile-time `LLAMA_QSA3_ENABLE` (default 1)** -- the env var no longer exists, because `rocprofiler-register` could make it read as unset under `rocprofv3` (see the session-15 UPDATE) |
| `LLAMA_QSA_DENSE_SHORTCUT` | **0** | **FLIPPED ON->OFF on 2026-09-19 = always QSA** (maintainer decision).  `=1` restores the dense arm (the `LLAMA_QSA_SPARSE_FA=0` cross-check) |
| `LLAMA_QSA_DENSE_DECODE_UNTIL` | 65536 (gfx1151) | decode stays dense below this; independent of the above |
| `LLAMA_QSA_SPARSE_FA` | on | `=0` = the dense masked reference path |
| `GGML_CUDA_QSA3_DUMP` | (removed) | was a temporary idx-row dump; it is **gone from the tree** - re-add it if the sort/structure needs re-checking (the procedure is in `README.md`) |

---

## 7. Post-MMB profile + where the time now goes

> **This is the SESSION-1 profile** (MMB landed, no qsa3, `F32SPLIT`/shortcut at their old defaults).
> It is still the best *shape* inventory, but the ranking has moved - qsa3 took the QSA kernel from
> 2944 -> 728.6 ms and session 4 turned the F32 path off.  **Current numbers: the session-4 UPDATE**
> and the always-QSA section in `README.md`.

Q4_K_M pp8192, `GGML_CUDA_MMB=1`, rocprofv3 kernel trace (**total 17.75 s**, was ~28 s off):

| kernel | time | share |
|---|---:|---:|
| `flash_attn_qsa` | 2.94 s | **16.6 %** |
| `mmb_dense_kernel` (Q8_0 PLE + Q5_1) | 4.06 s | 23 % |
| `mmb_routed_glu` (Q4_K gate/up) | 2.46 s | 14 % |
| `mmb_routed` (Q4_K down) | 1.39 s | 7.8 % |
| `dsv4_hc_pre` + `_post` | 1.44 s | 8.1 % |
| `rms_norm_f32<1024>` | 0.65 s | 3.6 % |
| `gdn_bf16_scan` | 0.62 s | 3.5 % |
| `mmb_f32split` | 0.65 s | 3.7 % |
| `mmb_cvt_f32_bf16` | 0.65 s | 3.7 % |

**Both leading items were probed and are not tuning-reachable** (see `README.md` §investigation):

* **Q8_0 tiling is optimal** — forcing narrow/wide changes pp8192 by +0.6 % / −2.4 %
  (802 / 798 / 778 t/s) → the kernel is occupancy/LDS-bound.
* **QSA gather 8B→16B is neutral** (796 vs 798) → not load-issue-bound; it is
  compute/reduction/occupancy-bound.

---

## 8. QSA v3 — **DONE in session 2** (kept as the design record)

**Status: landed behind `GGML_CUDA_QSA3=1`; results and the remaining optimization are in the
"UPDATE" section above and in `README.md`.  The plan below is what was actually built - keep it for
the gfx1200/gfx1100 follow-up.**

**Goal (achieved):** replace, for prefill only, the VEC `flash_attn_qsa` score/PV with a packed-block
WMMA implementation (pwilkin's `qsa3`).  Result: 2944 ms -> 1152 ms on the kernel, +5-10 % prefill.

**Approved:** the maintainer accepts a prefill re-baseline (greedy prefill output changes) as long as
it is **consistent, deterministic and coherent**.  Keep the **VEC `flash_attn_qsa` for the W=1..8
band** (gate the new path `n_query >= 128`), so width purity and `plain == draft-mtp` are untouched
by construction.  Document the re-baseline with a new same-seed hash.

**Reference:** `~/pwilkin-llama-cpp` (branch `strix-halo`), `ggml/src/ggml-cuda/qsa.cu` (445 lines):
`qsa3_rows_kernel`, `qsa3_merge_kernel`, `qsa3_attn_kernel`, plus the graph's `qsa_pack_keys` /
`qsa_pack_values` tensors (his `src[6]`/`src[7]`).

**Concrete steps**

1. **Pack ops (graph side).**  Build contiguous f16 key/value block buffers once per graph from the
   cache + the selected indices.  In our tree that is `src/models/qwen4exp.cpp` +
   `GGML_OP_FLASH_ATTN_QSA` (see `build_qsa_top_k`, `build_qsa_store_k`) and
   `ggml/src/ggml-cuda/fattn-qsa.cu`.  Decide whether to add the buffers as extra op `src[]` or to
   build them inside the op's launcher (the latter avoids graph/allocator changes but re-packs per
   call).  The handover for the parked port (`archive/work/wip-archive/iq4nl-prefill/`) and
   `wip/prefill-arrangements/README.md` both scope this.
2. **Descriptor.**  Port `qsa3_rows_kernel` + `qsa3_merge_kernel`: merge `G = 4` consecutive queries'
   top-k lists into a sorted, deduplicated, block-aligned array of block ids + a 16-bit per-query
   membership mask + count (his `ublk`/`umask`/`ucount`).
3. **Attention.**  Port `qsa3_attn_kernel` (16×16×16 f16 WMMA over the packed blocks, membership
   mask folded into the score pass, online softmax as now).  Reuse the portable-WMMA pattern from
   `mmb.cu` so gfx1201 still compiles (or keep it gfx11-only with stubs, matching
   `gated_delta_net_chunked_bf16_gfx11.cu`).
4. **Gate + validate.**
   * prefill-only (`n_query >= 128`); VEC kernel unchanged in the band;
   * `test-backend-ops -o FLASH_ATTN_QSA` must stay green (add cases as needed);
   * the `W = 1..8` logits matrix on `LLAMA_QSA_SPARSE_FA` on/off/new must keep the band identical;
   * same-seed greedy text: new build vs old, once, to record the prefill re-baseline hash;
   * PPL vs the dense masked oracle (`LLAMA_QSA_SPARSE_FA=0`) within noise — **not** MTP acceptance,
     which is blind to a defect present in both draft and target (`GREEDY-PURITY.md` §21).

**Size estimate:** the scoping doc priced this at 6-10 days.  It is the single largest remaining item.

---

## 9. NEXT WORK #2 — Q8_0 IU8-WMMA dense path

**Goal:** replace the dequant-to-bf16 MMB Q8_0 dense path (4.06 s, 23 %) with an **int8 → int8 tensor
core** GEMM (`v_wmma_i32_16x16x16_iu8`), avoiding the LDS dequant staging that makes the current
kernel occupancy-bound.

**Facts already established**

* The big Q8_0 shapes: `attn_qkv M=10240 K=2560`, `attn_gate M=6144 K=2560`, `ssm_out M=2560 K=6144`,
  `hc_* M=320/10240`, `ffn_*_shexp M=640 K=2560`.
* The tile heuristic is optimal; the kernel is **occupancy/LDS-bound**, not tiling-bound.
* `v_wmma_i32_16x16x16_iu8` measured ~171-175 T-MAC/s on gfx1201 (see
  `wip/q8-prefill-tuning/README.md`); the same doc notes the mainline MMQ Q8_0 kernel reaches only
  31-34 % of the int8-WMMA ceiling and its k-loop has no double-buffering.

**Steps:** quantize the activations to int8 (Q8_1-style scales) — note this is a numerics change too,
so the same prefill-only gate applies — then a WMMA-iu8 tile kernel with int32 accumulators; A/B vs
the current bf16 Q8_0 MMB path on IQ4_XS/Q4_K_M.

---

## 10. Smaller remaining items

* **bf16-producer marking** — kill `mmb_cvt_f32_bf16` (0.65 s, 3.7 %): the parked port has the
  plumbing (`ggml_cuda_mmb_mark_bf16_only`, `ggml_cuda_mmb_cache_reserve`,
  `ggml_cuda_mmb_cache_lookup`) but nothing ever calls it.  Needs a graph-optimizer pass that marks a
  producer's output bf16-only when every consumer is MMB.  Prefill-only.
* **HC prefill fusion** — `dsv4_hc_pre`+`_post` are 8.1 %; pwilkin's `hc_combine_norm`/`hc_mix_reduce`
  are ~0.93 s vs our 1.44 s.  Block 14 already has the decode-band fused hc ops; the prefill arm is
  the gap.
* **Legacy Q4_0 / Q4_1 / Q5_0** — trivial next to what is done (32-value blocks).
* **Optional 7900 XTX test** (gfx1100, single card, small model) with `GGML_CUDA_MMB_RDNA3=1` — the
  tiling almost certainly needs an RDNA3_0 pass.  The maintainer wants gfx1151 exhausted first.

---

## 11. Purity / landing gates (unchanged from the parked handover)

Before this can be opt-in, let alone defaulted on:

1. `W = 1..8` logits matrix with `GGML_CUDA_MMB=1` == off (should be identical **by construction** —
   `T >= 512` — but must be run).
2. MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
3. `test-recurrent-state-rollback`.
4. `test-backend-ops` suites (FLASH_ATTN_QSA / GATED_DELTA_NET / FLASH_ATTN_EXT).
5. Same-seed **prefill re-baseline** documented with a fresh hash.
6. gfx1100 / gfx1201 compile + consistency (gfx1201 TU compile already verified — §12).
7. Default stays **off**; the env kill-switch (`GGML_CUDA_MMB=0`) is retained.

The repo rulebook is `AGENTS.md` + `GREEDY-PURITY.md`; landing must regenerate `patches/` from a
canonical fork rebuilt at the release base (`scripts/make-patches.sh` / `make-release.sh`), never from
the drifted working tree.

---

## 12. Verification already done (cumulative; update this when the tree changes)

**Session 1 (MMB):**

* gfx1151 build clean after every commit; `llama-bench` / `llama-perplexity` / `llama-cli` /
  `llama-server` all build.
* PPL parity recorded for: Q4_K_M, IQ4_XS, UD-Q5_K_M, Q6_K, UD-Q3_K_M (table in §5).

**Sessions 2-4 (qsa3 + the default flips):**

* **gfx1151 always**; `fattn-qsa3.cu` also **TU-verified for gfx1201 and gfx1100** (extract the command
  from `build-rocm/compile_commands.json`, swap `--offload-arch=gfx1151`; `mmb.cu` and `fattn-qsa.cu`
  verified the same way in session 1).
* `fattn-qsa3.cu` is **order-exact**: the bitmap sort reproduces the rank sort's output, proven by
  PPL being **bit-identical** across the rewrite (c16384 3.3900, c32768 4.3353, q8_0 3.3879 pre-flip).
* Post-flip PPL: c16384 bf16 **3.3821**, q8_0 **3.3861**, c32768 bf16 **4.3397**.
* Decode unaffected by the always-QSA flip: tg64 shallow 25.80 -> 25.83.
* Greedy text coherent on a >2051-token prompt; the prefill re-baseline is expected and approved.
* Combined patch `git apply --check` clean on a fresh `8a2567e1e` worktree (re-verified at session 4).

**Session 8 (the HC bf16 producers):**

* `GGML_CUDA_MMB_HC16=1` vs `0` on gfx1151, IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`:
  pp8192 950.8 -> 961.9 (+1.2 %), pp2048 973.1 -> 994.2 (+2.2 %) (repeated interleaved runs).
* PPL c2048: 10.5771 (off) -> 10.6428 (gate on), **10.6428 also with the xn copy** (the xn producer is
  bit-identical).  Same-seed greedy text `9930c674a6ca` in all three configs.
* `LLAMA_MMB_CVT_LOG=1`: all 57 `hc_norm` conversions disappear with `HC16=1`.
* Combined backup `mmb-general.patch` + `patches/0001..0019`: `git apply --check` clean on a fresh
  `8a2567e1e` (19 commits).

**Session 11 (the dead-F32-store skip):**

* `GGML_CUDA_MMB_HC16=1` vs `0`, gfx1151 IQ4_XS Flash-Next bf16 KV, `-b/-ub 2048`: pp8192
  952.4 -> 974.0 (+2.3 %), pp2048 975.3 -> 1007.4 (+3.3 %); the `unary_gated` producer 302 -> 229 ms.
* Bit-identical: PPL c2048 10.6428, greedy `9930c674a6ca`, width probe pure, FA/QSA/GDN ops pass.

**Session 9 (the full bf16-producer port):**

* `GGML_CUDA_MMB_HC16=1` vs `0` on gfx1151, IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`: pp8192
  951.6 -> 963.7 (+1.3 %), pp2048 977.1 -> 996.7 (+2.0 %) (interleaved repeated runs).
* PPL c2048 **10.6428** (identical to the session-8 gate-only build) and same-seed greedy text
  `9930c674a6ca` -- all session-9 producers are bit-identical.
* `test-backend-ops -o FLASH_ATTN_QSA` 22/22, `-o GATED_DELTA_NET` 46/46, `-o FLASH_ATTN_EXT` OK;
  `test-logits-width-probe` P=1024 width_purity=PASS (maxdiff 0); and the qwen4exp
  `plain == draft-mtp` greedy text is byte-identical (`9930c674a6ca` both, no `-md` on the plain arm).
* `LLAMA_MMB_CVT_LOG=1`: only `ple_embd` remains.
* The delivery `GGML_OP_NAME` `INDEXER_FILL` fix is confirmed against the base `8a2567e1e`.

**Still NOT run (blocking promotion, see §11):** nothing — the `W = 1..8` matrix is now done
(session 5d UPDATE at the top; probe `tests/test-logits-width-probe.cpp`, gate PASSES).  A same-seed
**prefill re-baseline hash** also exists now (`bbd4bcb519e4` with MMB+QSA3, `5120b28f2879` without).

---

## 13. Gotchas the next session should not re-learn

* `llama-cli` **must** be run with `--single-turn` (and usually `--no-display-progress`) or it blocks.
* Warm the page cache before every bench; never run benches in parallel.
* `llama-bench` `-b/-ub 16384` on the 94 GiB IQ4_XS can OOM at pp16384 (context creation fails); use
  `-ub 2048` (the server's config) for these models.
* The `-lm none -lzm on` flags are needed for the Q4_K_M/lazy models in `llama-bench`.
* `mmb_dq_row_*` for the K-/i-quants are loaded from the row base inside `store_lds` (the fields are
  non-contiguous), so their `load_regs` branch is intentionally a no-op — do not "optimise" it back
  into the register prefetch.
* A `case`/`if` added to the predicates must be added to **all** of:
  `supported_mm`, `supported_mmid`, `supported_glu`, `dense_will_take`, `routed_will_take`, **and**
  the three dispatch helpers — a missing one is either a silent fallback or an uninstantiated launch.
* `mmb.cu` uses `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (gfx11); gfx12 needs
  `…_gfx12` with different fragment types.  The wrapper is a no-op on RDNA4 on purpose.

---

## 14. Repo conventions

* This WIP lives under `wip/` in `llama-cpp-rdna-boosts` (the only push target:
  `github.com:stew675/llama-cpp-rdna-boosts`).  **Never** push anything out of `~/llama.cpp`.
* Commit records/doc updates to the delivery repo; keep the experimental code in the worktree and
  regenerate `mmb-general.patch` from it whenever the code changes.
* When the work is promoted, it becomes a block amendment (the parked handover suggests block 08 for
  the MMB module) and follows the promotion rule in `AGENTS.md`.
