# HANDOVER — r328 re-base ladder, 12 of 18 rungs climbed, resume at **R12 (paused mid-rung)**

> **This file is session-2/3's delta.**  A fresh session should start at [`COLD-START.md`](COLD-START.md)
> (self-contained, includes the DFlash measurement mandate and R12's resume plan); the prompt to paste is
> [`NEXT-SESSION-PROMPT.md`](NEXT-SESSION-PROMPT.md).

Written 2026-11-21 (session 2/3, which climbed R06, R07, R08, R09 and R10).  Read [`README.md`](README.md) §1 (method + traps),
[`RUNGS.md`](RUNGS.md) (the per-rung ledger and every resolution so far) and [`DECISIONS.md`](DECISIONS.md)
(the decision log) — this file is the delta: what is done, what is owed, and how to not repeat the mistakes of
the earlier sessions.

## §0 State, exactly

| item | value |
|---|---|
| working branch | **`rebase-r328`** in `~/llama.cpp`, clean, **HEAD = `dbe4fdd40`** (tree `2b6aec1f0`) |
| rungs done | **R00 … R11** + **two rung repairs** (R00's top-k gather, R05's `nb32` widening) |
| next rung | **R12 — PAUSED, resume it first** — block 12 (`dad1442b3`), 64 files, 10 conflicted paths.  Ten of them are resolved and captured in [`r12-wip/r12-resolved-but-unbuildable.diff`](r12-wip/r12-resolved-but-unbuildable.diff); the rung is blocked on **`common_speculative_impl_draft_dflash`**, which is two implementations of one class (upstream `#32016`'s `dflash_options`/`init()` seam vs block 12's inline ctor + DSpark/DFlash2/LiliCorr/ingest state).  **Read `r12-wip/README.md` first** — it lists what is settled and what is not.  Oracle: `benchmarks/mtp-adaptive-methodology.md` (four axes), not text |
| delivery repo | `~/llama-cpp-rdna-boosts` — **`patches/`, `release.json`, `rdna-boosts-all.patch` and the fork are untouched** (that is R18) |
| target release | next after `v17-4f9dfa0e3-r327`, i.e. **r328**; new fork point **`919794500`** |
| reference binaries | `~/llama.cpp/build-rocm/bin-r327` (the shipping r327 `bin/`) — **the attribution oracle for any abort** |
| oracle builds | `~/llama.cpp/build-base` (pristine base, hangs — see §4) and `~/llama.cpp/build-base-novmm` (pristine base, `GGML_HIP_NO_VMM=ON`, **the one that runs**) |
| ladder build | `~/llama.cpp/build-rocm` — `GGML_HIP_NO_VMM=OFF`, `GGML_CUDA_FA_QUANTS` includes `iq4_nl-iq4_nl`, `LLAMA_USE_PREBUILT_UI=OFF`; `cmake --build build-rocm -j 24` is the per-rung loop |

### The commit/tree table (oldest first) — compare **trees**, never SHAs, against a re-cut

```
919794500                    base
4cba0ad32  a3824ea0e  block 00: foundation
14a0414bb  6566ca3ac  block 01: hybrid HIP all-reduce
3ea8b0212  e12a34bad  R00 repair: top-k gather (block 10's kernel moved down)
3941bad4e  0799ebfad  block 02: adaptive MTP draft depth
e612388ba  971953b99  block 03: chunked GDN prefill
768d27561  0866d4229  block 04: BF16 KV + native BF16 FA
1c9bfa37c  13683655d  block 05: FA band + native quantized K/V
c3860af1e  b52803164  R05 repair: nb32 -> int64_t
ee377ec45  b5dc56996  block 06: MMB quantized prefill GEMM
1435e7873  d3b004581  block 07: k-quant boosts (mmvq VDR, Q6_K MMQ, Q1_0, grid pad)
510769147  2ed6cb541  block 08: fused kernels (fusions + fused MoE + norm-gated + gdn/ple conv)
24cecfb51  b05ee8543  block 09: host-resident MoE (iface, scheduler, Meta, slab, cache engine)
f5a090a31  394e0b708  block 10: slab, fusion dispatcher and async input ring
dbe4fdd40  2b6aec1f0  block 11: qwen4exp (model, hyper-connections, lazy reader, QSA/indexer)
```

## §1 What each rung cost, so R12..R17 can be planned

| rung | conflicts | what it actually took |
|---|---|---|
| R00 | 4 files / 7 regions | **D1** (take upstream `#32505`, drop the dup-ids port, retire the candidate) + re-port `mm_ids_helper_512_10` onto upstream's `ne_store` helper + keep the `#30542` revert.  Then **the R00 repair** (a VMM-pool top-k refusal, fixed by moving block 10's gather kernel down — **D15**) |
| R01 | 0 | one repair: HIP now static_asserts a 64-bit warp mask (`ggml_cuda_ce_xor` -> `ggml_cuda_shfl_xor`) |
| R02 | 5 files / 9 regions | all unions; **D5 resolved as MERGE** (upstream added sibling speculator classes, kept `impl_draft_mtp`), enum count 13 -> 14, docs **regenerated** with `llama-gen-docs` |
| R03 | 1 file / 2 regions | unions: upstream's `padded_q`/`permuted_k` layouts now also pass `/*n_rs_batch=*/1`; gate range `correlated ? -1e-4 : -0.5` |
| R04 | 2 files | **D3**: union of 7 rope arms + **dropped upstream's guard** that refused every non-BF16 rope into a BF16 cache (its own matrix lacked the arm ours supplies) |
| R05 | 7 files / **22 regions** | **D16** (maintainer): the delivery's FA band iteration model wins; upstream `#32407`'s rotation **not carried** — and `#32217`'s per-head mask + `#31617`'s typed masks still adopted.  Then the R05 repair: `nb32` -> `int64_t` |
| R06 | 2 files / 2 regions | both unions: upstream's few-row MMQ (`#32065`) keeps precedence over MMB's MUL_MAT_ID check (**it is compiled out under HIP**, so they are disjoint by backend), and block 06's `expert_bounds` output re-ported onto the R00 store-design helper (`n_dumped`, not `it_compact`).  Plus the **R06.2 finding** (block 06's MMB cache is pool-unsafe until block 10) |
| R07 | 1 file / 2 regions | both unions in `mmvq.cu`: upstream's Jetson `ORIN` table id + the delivery's RDNA2 `NAVI21` id, and the two `calc_nwarps` special cases side by side.  Upstream's RDNA3_5 `ne11` gate is a *dispatch threshold*, not a competitor to the delivery's VDR tables — adopted, unmeasurable on gfx1201.  D11 measured: grid pad +1.37 % pp2048 (text identical), `MMQ_J_MAX=64` −15.3 % → the delivery's cap stays at its default |
| R08 | 8 files / **13 regions** | the climb's worst.  Eight resolutions, all unions except two "theirs" (the chunked quantizer body, the enabled duplicate-id tests).  **D4 resolved without a choice** (additive mechanisms; the delivery's matcher is inert until R10).  **Four forced fixes**: five raw warp-mask sites in block 08's own code, the `SET_ROWS` q8_0 group kernel needs an F32-source gate (`#32387` added BF16 sources), a lost `#if` guard in `topk-moe.cu`, and an **auto-merge artifact** — upstream's `force_tiling=true` landed in the delivery's `has_gate` template slot in `mmq.cuh` and would have compiled and mis-dispatched.  Rung repairs are *not* squashed: they live in this commit (unlike R00/R05, which needed their own) |

## §2 Owed at specific later rungs (these are the ones that will be forgotten)

1. **R10 — drop block 10's `top_k_gather_rows` hunk.**  The R00 repair already put it in the tree; when block 10
   is cherry-picked, resolve that file **keep-local** and keep block 10's `ggml_cuda_cpy_as_memcpy_2d` guard
   (different site).  Duplicating the hunk is the failure mode.
2. **R10 — verify the R06.2 finding is fixed by block 10's `pool_stack`**, and use those arms to close the only
   coverage gap this rung leaves:
   ```bash
   # (a) the default-on RDNA4 dense type: must prefill, must not abort at ggml-cuda.cu:808
   HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf \
     -ngl 99 -sm layer --fit false -c 8192 -fa auto -ub 1024 -b 1024 -p "<6k-token prompt>" -n 8 \
     --seed 42 --temp 0 --no-display-prompt --single-turn      # r327 reads 1162.4 t/s, `50 chars sha=ebf2a5a9b00a`
   # (b) the forced routed arm — the ONLY coverage of R06's expert_bounds re-port
   HIP_VISIBLE_DEVICES=0,2 GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1 ./build-rocm/bin/llama-cli \
     -m /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf -ngl 99 -sm tensor -c 8192 -ub 1024 -b 1024 \
     -p "$(cat prompts/prose-rdna-boosts.txt)" -n 16 --seed 42 --temp 0 --no-display-prompt --single-turn
   #    text must equal the GGML_CUDA_MMB=0 hash: `73 chars sha=86b7f9b5aa80`
   ```
3. **R17 — `TODO` #351**: does upstream now support `-sm tensor` for `flint`?  That is block `0300`'s drop
   condition.  3-GPU arm on `/llm/models/RecursiveAI/Flint/Q8_0/flint-Q8_0.gguf`; also the refusal arm (must
   still print the pointer and exit 1) and `-sm layer` vs `-sm relaxed-tensor` text identity
   (`103 chars sha=cc4fbc3e8367` at r278).  Dropping a block is a maintainer decision, never a gate outcome.
4. **R18 — squash `3ea8b0212` into block 00** and **`c3860af1e` into block 05** (rung-repair pattern).
5. **R18 — re-measure `mm_ids_helper_512_10`** before quoting any number: the claim of record is
   **+4.5 % pp16384 / +5.2 % pp2048**; at this base it reads **+0.7 %** on a 5298-token prompt at 2 devices
   (`ON 1642.7/1638.7` vs `OFF 1635.4/1624.2`), text identical.  If it does not reproduce, the block-00 note and
   `patches/README.md`/`ENVIRONMENT.md` must say so.
6. **R18 — regenerate the generated docs and assert no drift.**  `llama-gen-docs` at R02 proved the committed
   tables at r327 are **stale**: block 00's `arg.cpp` adds the `--lazy-mode mlockr` help text and neither
   `tools/cli/README.md` nor `docs/speculative.md` lists `mlockr` (`arg.cpp` 2 hits, README 0).  Regenerate,
   then `git diff --exit-code` on the doc files.
7. **R18 — `TODO` #352**: move the container `web` stage from `tools/ui/` to `ui/apps/chat` (+ `ui/lib/ui`,
   `ui/lib/voice`) and the Priority-1 copy to `ui/apps/chat/dist`.  Only correct once the tree has the new layout.
8. **R18 — the FA gate for every latent risk this climb created**: `gate-prefill-logits.sh` (record a new base on
   the new build), `gate-spec-purity.sh`, `gate-spec-gdn-split.sh`, `gate-norm-rows-alias.sh` and the rest of
   the per-change table in `AGENTS.md`.  **D16** in particular means the FA band model is now the only iteration
   model in the tree, so the FA op matrix and the prefill-logit gate are its only guards.
9. **`TODO` #353** — Inkling-2 needs upstream [PR #32695](https://github.com/ggml-org/llama.cpp/pull/32695)
   (**unmerged** at 2026-11-20).  Nothing to do until it merges; then it is the local validation for the per-head
   FA mask and for the `nb32` widening.
10. **R18 — `GGML_CUDA_MMQ_J_MAX`'s usable floor** (`RUNGS.md` R07.3): `=8` aborts on Q6_K/gfx1201
   (`GGML_ASSERT(J_best > 0)`), *pre-existing at r327*, while `ENVIRONMENT.md` only says "a value < 8 is ignored".
   Either clamp the knob to ≥ 16 or document the floor; it is a one-line delivery change, the maintainer's call.
11. **R18 — the verify-band WMMA family's A/B** (`--wmma-verify`, block 07's r290 addition): opt-in and OFF by
   default, so no rung reaches it.  If it is ever flipped on, it needs the four-axis MTP protocol +
   `llama-batched-bench`, and its own byte-identity caveat (1 column always keeps the K-split kernel) belongs in
   `GREEDY-PURITY.md`.

## §3 The traps that actually bit (all are avoidable)

- **Build-completion detection.**  Do **not** poll `pgrep`/`grep` for a build.  Use a sentinel:
  `(cmake --build … > log 2>&1; echo "BUILD_RC=$?" >> log)` and poll the log for `BUILD_RC=`.
- **`GGML_CUDA_FA_QUANTS` is hand-set in `build-rocm/CMakeCache.txt`** and must contain `iq4_nl-iq4_nl` **only
  from R05 on**.  Before R05 the configure dies with `unknown type "iq4_nl"`.
- **HIP's 64-bit warp mask** (`#31617`) — bit twice: R01's all-reduce and R05's band code.  Any raw
  `__shfl*_sync(0xFFFFFFFF, …)` must become `ggml_cuda_shfl*`.
- **`-fa on` is required** for any KV-cache-type arm (bf16/q8_0).
- **27B on one 32 GiB card needs `--fit false -c 8192 -fa auto`**; the default `--fit` tries a 16 GiB KV
  allocation and dies in `llama_init_from_model`.
- **Docs are generated**: never hand-merge `tools/*/README.md` or `docs/speculative.md` — union what you can,
  then run `./build-rocm/bin/llama-gen-docs` and check the diff.
- **`-sm tensor` needs `HIP_VISIBLE_DEVICES` pinned** (the i5503D iGPU is enumerated otherwise).
  `--device 0,1,2` is **not** a valid flag at this base (`invalid device: 0`); the pin is enough.
- **Never run two GPU workloads at once** (benchmarks, op tests, or a build + a benchmark).
- **NEW (R06): an intermediate rung may abort where the finished set does not — and `bin-r327` is how you tell
  the difference in one run.**  R06's tree carries block 06 *without* block 10, and block 06's MMB activation
  cache frees pool buffers out of order, so a default-on RDNA4 dense type (IQ3_S) aborts at
  `ggml-cuda.cu:808` (`GGML_ASSERT(ptr == pool_addr + pool_used)`) while the same arm on `bin-r327` prefills
  normally.  Rule: **any abort gets the same arm re-run on `bin-r327` before it is called a rung regression**;
  when r327 is clean and this rung is not, name the *block that owns the fix* (here: block 10's
  `ctx.pool_stack_*`, `7bbf540d5`) and record it as an obligation, not as a defect.
- **NEW (R06): `llama-cli` is a client/server pair now** — an abort happens in the child, and the parent's log
  only shows the crash handler attaching.  Use `llama-server` + a `POST /completion` when you need the real
  backtrace in one process (that is how R06.2's `ggml_cuda_pool_vmm::free` <- `mul_mat_id_mmb_glu` frame was
  captured).
- **The MMB A/B levers are `GGML_CUDA_MMB` (master), `_MMF32`, `_TYPES=<csv>`, `_ROUTED`, `_GLU`, `_CACHE`.**
  On RDNA4 the default quantized *dense* set is `{IQ3_S, Q4_1, Q5_0}` and `routed=0`, so a Q8_0 model measures
  only the F32 arms unless `_TYPES=q8_0` forces the set (a documented A/B override, not a policy change).

## §4 Facts about this box's GPU stack that the ladder depends on

- **The pristine base hangs** on its first GPU compute with `GGML_HIP_NO_VMM=OFF` (the delivery's required
  config): `#30542`'s VMM pool retry loop spins inside libamdhip64.  `GGML_HIP_NO_VMM` is a **CMake option**, not
  an env var.  A base built with `GGML_HIP_NO_VMM=ON` runs fine — that is `build-base-novmm`, the oracle.
- The **`#30542` revert** (block 00) is what makes the pool work; it is carried a second time (**D2**) and its
  rationale is separate from block `0300` (see `DECISIONS.md` D2).
- **The VMM pool refuses a strided 2-D D2D copy into a VMM-backed destination** (`hipMemcpy2DAsync`,
  "invalid argument"; R00.6/D15) **and it is a strict LIFO stack** — a buffer freed under a newer one aborts
  (`ggml-cuda.cu:808`; R06.2).  Both classes are reachable with no delivery source change, and both are the
  delivery's own code's job to respect.
- Sanity oracle for every rung: **`100 chars sha=1c5d32ac537d`** on
  `Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt
  --single-turn`, `HIP_VISIBLE_DEVICES=0` (~242/91 t/s).  It has been identical at every rung so far.

## §5 Useful arms, with their measured baselines (do not re-derive)

| arm | command sketch | R06 reading |
|---|---|---|
| coherence (every rung) | 4B, as above | `1c5d32ac537d`, 242/91 t/s |
| 3-device tensor (block 01's arm) | `HIP_VISIBLE_DEVICES=0,1,2 … -sm tensor` | `1c5d32ac537d`; census `HIP AllReduce pipeline: 3 GPUs` |
| MTP adaptive (block 02) | 27B UD-Q4_K_XL, `--fit false -c 8192 -fa auto`, `--spec-type none` vs `draft-mtp-adaptive` | 29.3 -> **65.2 t/s**, text identical `55824e640aa0`, acceptance 0.95238 |
| chunked GDN (block 03) | 4B, 5298-token prompt, `GGML_CUDA_GDN_CHUNKED=1/0` | 6870.1 vs 6314.6 t/s (**+8.8 %**), text identical |
| bf16 KV + rope fusion (block 04) | 4B `-ctk bf16 -ctv bf16 -fa on`, `GGML_CUDA_DISABLE_FUSION=0/1` | 6786.5/88.8 vs 6417.0/76.2 (**+5.8 % pp, +16.5 % tg**) |
| native quantized K/V (block 05) | 4B `-ctk q8_0 -ctv q8_0 -fa on`, `GGML_CUDA_FA_KV_NATIVE=0/1` | **88.6 vs 83.9 t/s** decode (**+5.6 %**) |
| FA correctness (blocks 04/05 and anything touching `fattn-*`) | `test-backend-ops -o FLASH_ATTN_EXT` | **7428/7428, 0 FAIL**, ~4 min |
| MUL_MAT_ID (blocks 00/06) | `test-backend-ops -o MUL_MAT_ID` | **1078/1078, 0 FAIL**, 16 `dup=1` |
| GATED_DELTA_NET (block 03) | `test-backend-ops -o GATED_DELTA_NET` | 76/76 |
| MMB / MMF32 (block 06) | 4B Q8_0, `llama-bench -p 2048,16384 -n 0 -r 3 -ngl 99 -sm layer`, `GGML_CUDA_MMB=0` / `GGML_CUDA_MMF32=0` | default **7793.1 / 6768.4**; MMB off 7767.3 / 6727.0 (+0.33/+0.62 %); MMF32 off 7722.9 / 6699.9 (+0.9/+1.0 %) |
| MMB dense census (block 06) | 35B-A3B Q8_0, 2 dev `-sm tensor`, `GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_LOG=1` | 80 `MMB_DENSE` hits, prefill **4915.1 t/s**, clean |
| Flash-Next (host-expert field model, the known-red's repro) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, `HIP_VISIBLE_DEVICES=0,2 -sm tensor -c 8192 -ub 1024 -b 1024`, the 5298-token prompt | 1692.5 t/s prefill after the R00 repair |
| 35B-A3B Q8_0 prefill/decode (blocks 06/07/08 arms) | 2 dev `-sm tensor -c 8192 -ub 1024 -b 1024`, the 5298-token prompt | **4774.5 / 66.9 t/s**, `73 chars sha=86b7f9b5aa80` |
| 27B UD-IQ3_S dense (R06.2's abort arm) | 1 GPU `--fit false -c 8192 -fa auto`, ~6 k-token prompt | r327 **1162.4 / 37.8**, `50 chars sha=ebf2a5a9b00a`; this rung aborts unless `GGML_CUDA_MMB=0` (1175.9/32.9) |

The 5298-token prompt is `prompts/prose-rdna-boosts.txt` (sha256 in `prompts/README.md`); hash results with
`scripts/extract-generated.py`, never a grep/sed slice.

## §6 The standing bias the maintainer gave (and its limit)

> *"In every single rebase we've ever done, there was never a time where upstream did something better than the
> delivery repo"* — so where upstream and the delivery are **alternatives to the same mechanism**, take the
> delivery's and record the cost (**D16** is the first case: upstream's FA rotation dropped, verified absent).

**The limit, learned the hard way at R05** (`DECISIONS.md` D16 addendum): that bias is about a shared
*mechanism*, not about a **correctness** change upstream made for its *own* new code.  `nb32` is the example:
keeping the delivery's `int32_t` looked right and was wrong, because the per-head term upstream added multiplies
it by a head index.  Ask of any contested hunk: *is this a preference between two ways to do our thing, or is it
the arithmetic/contract of upstream's new feature?*  Take the delivery's side for the first, upstream's for the
second.  R06 added a third, cheaper case: **a dispatch where the two sides are disjoint by backend** (upstream's
few-row MMQ is HIP-compiled-out) needs no preference at all — keep both, upstream's first.

## §7 How to resume (short form)

```bash
cd ~/llama.cpp && git status && git branch --show-current      # must be rebase-r328, clean
git log --oneline -1                                            # expect dbe4fdd40
git cherry-pick dad1442b3                                       # R12 = block 12 (64 files, 9 conflicts)
(cmake --build build-rocm -j 24 > /tmp/r12.log 2>&1; echo "BUILD_RC=$?" >> /tmp/r12.log)
# poll for BUILD_RC, then: coherence -> the block's own switch -> the rung's log in RUNGS.md -> commit
```

**R10 closed four of the carried obligations and opened one** (see `RUNGS.md` R10): the dispatcher is now ONE TU
with upstream's `#32454` NORM+MUL arms in it (D4 shipped), R06.2's *dense* half is closed (27B IQ3_S prefills and
reads r327's `ebf2a5a9b00a`), the `top_k_gather_rows` hunk is de-duplicated by construction, and R09.3 is closed
(arena == arena-off).  **Still owed:** R06.2's *routed* half (the forced `GGML_CUDA_MMB_ROUTED=1` arm still
aborts on the pool LIFO assert — now R15's), and a **new decode finding** (R10.3: the arena arm reads 20.4 t/s
where R09 read 54.4, fusion- and slab-independent, owned by blocks 14/15).  **R10.3 is a RELEASE BLOCKER** — the
maintainer's instruction (2026-11-21): *"the decode regression must be fixed before we release; that can happen
later"*.  Do not cut r328 with it open, whatever the rest of the ladder says.

**Historical note — R10's four obligations were:** — run them as part of its verification, not later:
(1) `RUNGS.md` **R06.2** — after block 10's `pool_stack` lands, the 27B UD-IQ3_S dense arm (default MMB) must
prefill instead of aborting at `ggml-cuda.cu:808`, *and* the forced routed arm
(`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1` on 35B-A3B) must read `73 chars sha=86b7f9b5aa80`;
(2) **R08.2 / D4** — the narrow-row norm matcher must sit **ahead** of upstream's generic norm branches in
`ggml-cuda-try-fuse.cu`;
(3) **R09.3** — the triple arena / arena-off / arena-off+`GGML_MOE_HOST_OFFLOAD=0` must read **one** text, as r327
does (it reads two on the R09 tree).
Also still owed at R10: **drop block 10's `top_k_gather_rows` hunk** (the R00 repair already put it in the tree —
resolve that file keep-local and keep block 10's `ggml_cuda_cpy_as_memcpy_2d` guard) and `D8`/`D12`.

**R09 was the rung with the most upstream churn** (70 files, 136 upstream commits, 83 behind them) even though the
text merge is small (4 paths).  Its two `DECISIONS.md` items are both `MERGE`: **D8** (upstream moved the
nodes-per-tensor factor into `llama_model::graph_nodes_per_tensor()` and added a `32*n_layer_all` reserve when
`cparams.moe_cache_size > 0` — keep the virtual, keep the delivery's KV-rotation leaves and the hard "over the node
budget" refusal, and re-run `scripts/gate-node-budget.sh`) and **D9** (`4783b1145` replaced the architecture list
in `llama-kv-cache.cpp` with `llama_model::lid_k_hdmd()`; make qwen4exp's answer explicit in its model class
rather than in the cache's list).  Both are the delivery's *own* subsystems, so resolve them as merges and
**do not** drop delivery code: `D8`'s note about upstream's MoE-cache reserve says to check whether block 09's
arena needs its own term.
