# Issues #59 / #60 — analysis + verification (single gfx1100 box, no qwen4exp model)

> **UPDATE 2026-09-29:** the candidate gates below were re-integrated on r21 and the 4-head indexer
> policy was changed from an RDNA3_0 *unsupported* gate to a geometry-aware fused-vs-chain choice.
> See `SESSION-2026-09-29-wi1-wi2.md` and the top of `README.md`; the analysis below is kept as the
> original r20-context record.

**Session:** 2026-09-28, host `gfx1100` (one RX 7900 XTX, ROCm 7.14, `~/bin/build-llama-rocm-714`),
delivery tree = `v16-84e76d8a2-r20` (`release.json` tree `6f8369bf…`, built at `4be690b54`).

**Scope limit:** the reporters' model is **Qwen3.8-Flash-Next (qwen4exp) GSQ-RCO IQ3_XXS (~70 GiB)**.
There is **no qwen4exp model on this box** and one 24 GiB card cannot hold one, so neither issue can be
reproduced end-to-end here.  Everything below is either (a) a static proof from the r20 / r3 / r5 code,
(b) an **op-level** measurement (oracles), or (c) a memory measurement on the local `qwen35` 27B, which
shares the exact FA geometry (`head 256, GQA 6, q5_1 KV`) but is not qwen4exp.  Stated per claim.

---

## Verdicts in one line

| issue | reporter's ask | verdict |
|---|---|---|
| **#59(a)** qsa3 on gfx1100 | "keep qsa3 off on RDNA3_0 until re-tuned" | **Confirmed.** qsa3 was enabled on RDNA3_0 by the folded beta but the gfx1100 record explicitly left qwen4exp "trust-RDNA3_5"; the gate does not exist.  Candidate fix implemented + oracle-verified. |
| **#59(b)** fused indexer top-k | "make the indexer top-k revertible (gate/policy)" | **Confirmed, but the dominant cause is mis-attributed.** Patch `0005` changed two things: it rewrote `indexer-topk.cu`, **and** it added the **fused 4-head `LIGHTNING_INDEXER` prefill score** + the model switch to it.  `GGML_OP_INDEXER_TOPK` itself already exists in r3/r7 (block 14).  The reporter's own +4.5 % recovery via the (non-bit-exact) WMMA4 arm points at the score fusion as the dominant part; the top-k rewrite has its own, slower A/B knobs (`LLAMA_INDEXER_NOBLOCK`/`NOGROUP`).  Candidate gate implemented for the bit-safe score-fusion revert + oracle-verified. |
| **#60** r4/r5 long-prefill OOM | "fix or gate the r4/r5 head-256 GQA-6 FA band memory" | **Disproven for gfx1100.** The band is `GGML_CUDA_CC_IS_RDNA4`-only; on gfx1100 every changed expression is a provable no-op, so r3 and r5 have identical compute graphs.  The real long-prefill memory drivers are the **qsa3 F16 K/V packs** (r8+) and the **native FA prefill staging arena** (r3+, outside the graph reserve/`--fit`) — both already have knobs. |

---

## Issue #59 — gfx1100 deep-prefill regression from the r8 campaign

### 59(a) qsa3

**Where the policy is decided.** `ggml/src/ggml-cuda/fattn-qsa3.cu:585
ggml_cuda_flash_attn_qsa3_supported()` accepts `GGML_CUDA_CC_IS_RDNA3_0(cc) ||
GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc)`.  `RDNA3_0` was added by the gfx1100 overlay
(`archive/work/mmb-general/gfx1100/patches/0007`, folded into block 15 by r8).  The gfx1100 record
itself says (`gfx1100-porting.md` §6.4, `gfx1100-s2s4-results.md`): **"end-to-end qwen4exp performance
is trust-RDNA3_5"** and *"extend its support predicate from RDNA3_5 to RDNA3_0 **and re-tune**"*.  The
reporter's 2×7900XTX box is the first end-to-end measurement of that path; it measures ~13 % slower at
d64K, and the only lever that helps is the (non-bit-exact) `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100` —
i.e. a different kernel, not qsa3.  So qsa3-on-RDNA3_0 is exactly the "never re-tuned" case the
campaign flagged.

**There is also a hidden memory cost.**  `src/models/qwen4exp.cpp` `build_attn_qsa()` builds the qsa3
packs purely from shapes (`qsa3_pack && n_stream == 1 && q_p->ne[1] >= 128 && k_raw->ne[0] == 256 && …`)
and never asks the backend.  On any arch where the runtime qsa3 predicate is false the graph still
materialises **two full natural-F16 copies of the K/V cache** (`qsa3_f16_natural` → `ggml_cast` +
`ggml_cont`) that the VEC kernel then ignores.  At 126K tokens with 4 KV heads that is ~2×258 MiB of
dead compute-buffer memory.  This is why `LLAMA_QSA3_ENABLE=0` (which removes the packs *and* the
kernel) is the only thing that gives the reporter's r17 run its headroom.  A gate that only flips the
CUDA predicate would fix perf but **keep** the memory cost.

**Candidate fix (implemented, `candidate-gates.patch`).**
* `fattn-qsa3.cu`: `GGML_CUDA_QSA3` env override (`0` off everywhere, `1` opt in); **default off on
  RDNA3_0**, unchanged on RDNA3_5/RDNA4.
* `fattn-qsa.cu`: when `dst->src[7]/src[8]` are set (packed form), `ggml_cuda_flash_attn_qsa_supported`
  returns `ggml_cuda_flash_attn_qsa3_supported` — so the packed op's support *is* the qsa3 predicate,
  which is what lets the model probe it.
* `qwen4exp.cpp`: new `qsa3_op_supported()` builds a minimal packed probe and queries
  `ggml_backend_dev_supports_op`; the pack build now requires it.  So on gfx1100 (default) the packs
  are never materialised (no memory cost), and the VEC QSA kernel runs the prefill (the r7/r3
  behaviour the reporter wants).

**Verification (op level, gfx1100).** `test-backend-ops -o FLASH_ATTN_QSA`:
* default (qsa3 off on gfx1100): the four `qsa3=1` packed cases report `not supported [ROCm0]`, and
  `23/23 tests passed` (the VEC path still runs all unpacked cases).
* `GGML_CUDA_QSA3=1`: `26/26 tests passed`, including the packed WMMA cases.  So the gate is live, the
  opt-in is intact, and the fallback is correct.

### 59(b) the fused indexer

The reporter blamed patch `0005` ("fused indexer top-k op + QSA prefill score fusions").  Reading the
actual fold: `GGML_OP_INDEXER_TOPK` is **already in the r3/r7 delivery** (block 14 — verified: r3 has
`indexer-topk.cu`, `ggml_indexer_top_k` in `qwen4exp.cpp` and `GGML_OP_INDEXER_TOPK` in `ggml.h`), so
the op is not the new thing.  Patch `0005` rewrote `indexer-topk.cu` (671 lines, its own
`LLAMA_INDEXER_NOBLOCK`/`NOGROUP` A/B knobs, which the reporter measured as *slower*) and, for
qwen4exp, added
1. the **4-head case** to `ggml_cuda_lightning_indexer_generic` (`lightning-indexer.cu`: `n_embd==128
   && n_head==4`, the qwen4exp QSA prefill scorer), and
2. the model switch to `ggml_lightning_indexer` for the prefill score (`use_wmma` in
   `src/models/qwen4exp.cpp build_qsa_top_k`, env `LLAMA_QSA_SCORE_WMMA`, default ON).

Before `0005` the 4-head op was unsupported and the graph used the unfused `mul_mat+relu+head_sum`
chain.  The prefill score's WMMA4 arm (`indexer4_arch_enabled`, `lightning-indexer.cu:706`) is
**default-OFF on RDNA3_0** (`GGML_CUDA_LIGHTNING_INDEXER4_GFX1100`), so on gfx1100 the fused op runs
the **generic vec** kernel — which the reporter measures ~5.4 % slower than the r7 chain.  Enabling
the WMMA4 arm recovers +4.5 % but is not bit-exact (the reporter's own finding).  So the cheap,
bit-safe fix is to **not use the fused score op on gfx1100 by default** — i.e. the reporter's
"revertible" ask, applied to the bit-safe part.  (If a re-measure still shows a residual after this
gate, the next lever is `LLAMA_INDEXER_NOBLOCK=1`, which also has to be re-checked against the r7
top-k; note the reporter measured it as slower on the current tree.)

**Candidate fix (implemented, `candidate-gates.patch`).**
* `lightning-indexer.cu`: `ggml_cuda_lightning_indexer_supported` returns **false** for the 4-head
  shape on RDNA3_0 unless the WMMA4 arm is opted in.  The backend stays the single source of the
  arch policy via `indexer4_arch_enabled`.
* `qwen4exp.cpp`: new `lightning_indexer_op_supported()` probe gates `use_wmma`, so a rejected op
  makes the model build the unfused chain (r7 numerics/perf) instead of falling back to CPU.

**Verification (op level, gfx1100).** `test-backend-ops -o LIGHTNING_INDEXER`:
* default: all **81** `nh=4` cases report `not supported [ROCm0]` (→ the model builds the chain);
* `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1`: **81/81** `nh=4` cases `OK` (the WMMA4 arm).
The non-4-head (32/64) cases are unchanged.
`-o TOPK_QSA`, `-o FLASH_ATTN_EXT` and a qwen35 `pp512`/greedy smoke are green.

**Not verified:** the end-to-end prefill delta.  It needs a qwen4exp model; the op-level and static
evidence is all this box can give.  The candidate patch must not be promoted to `patches/` without a
gfx1100 qwen4exp re-measure (the campaign promotion rule).

---

## Issue #60 — the r4/r5 "334 MiB" long-prefill OOM

### The band cannot run on gfx1100 (static proof)

The r3→r5 delta is **four files** (`/tmp/lc-r3` tree `08fe2b77` vs `/tmp/lc-r5` tree `de86c5e1`,
both rebuilt from the delivery patches): `fattn.cu`, `fattn-common.cuh`, `fattn-mma-f16.cuh`,
`tests/test-backend-ops.cpp`.  The only functional change is the issue-#45 band:

* `ggml_cuda_fattn_band_wmma_applies` (`fattn-common.cuh:1636` / r20 `:1791`) begins
  `if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) return false;`.
* `GGML_CUDA_CC_IS_RDNA4(cc)` is `cc >= GGML_CUDA_CC_RDNA4` and `GGML_CUDA_CC_RDNA4 = OFFSET_AMD +
  0x1200`; gfx1100 is `OFFSET_AMD + 0x1100` (the campaign's own `MMB_CFG` dump prints
  `cc=0x1001100`), so the predicate is **false on gfx1100**.
* `launch_fattn` therefore takes `band_wmma == false`; `blocks_num.y` keeps its earlier `= 1`, so the
  new `nblocks_total = blocks_num.x * blocks_num.y` **equals** the old `blocks_num.x`, and the fixup
  expression is byte-for-byte the old one.  The reporter's `nblocks_total` hypothesis needs
  `blocks_num.y > 1`, which only the band sets.
* `fattn-mma-f16.cuh`: the new band entry is `if constexpr (!use_sparse && DKQ == 256 && DV == 256 &&
  ncols2 == 8) { if (gridDim.y > 1) { … return; } }` — skipped when `gridDim.y == 1`; the new
  `kb0_step` parameter defaults to `1`; and `blockIdx.x*gridDim.y + blockIdx.y` with `gridDim.y == 1,
  blockIdx.y == 0` reduces to `blockIdx.x`.  All no-ops.

**Conclusion:** on gfx1100, r3 and r5 are functionally identical; the `=0` band kill-switch "not
helping" is expected, because the band never applied.  The reporter's r3-PASS/r5-OOM bisect is not
explained by the r4/r5 code; it is most plausibly a **marginal-VRAM threshold** (the added compiled-in
band code can perturb the FA kernel's register/occupancy footprint, and the r3/r5 boxes were run at
different times) or an outside-config difference.  Recommendation: re-run r3 vs r5 **interleaved in
one session on a freshly booted box**, and if it reproduces, capture `rocm-smi` VRAM during the
prefill for both — the code says there is nothing to bisect.

### The real long-prefill memory drivers on gfx1100

Measured on this gfx1100 with the local `qwen35` 27B, `-c 65536 -ctk q5_1 -ctv q5_1`, at load:

| `GGML_CUDA_FA_KV_NATIVE` | `ROCm0` compute buffer | KV buffer |
|---|---:|---:|
| `1` (native, default) | **186.28 MiB** | 1536 MiB |
| `0` (F16 staging) | **400.28 MiB** | 1536 MiB |

i.e. the **native** path is *cheaper in the graph reserve* — it moves the whole-cache F16 K/V copy out
of the node allocation and into the **per-context staging arena** (`ctx.fattn_stage_try_get`, outside
the graph reserve and outside `--fit`; `GGML_CUDA_FA_STAGE_MAX_MB`, default 512 MiB/operand).  The
arena is additive at deep prefill and grows with `n_kv`, so a nearly-full device can OOM there while
the graph reserve is small.  That is consistent with the reporter's observations (`-ub 192` helps;
`GGML_CUDA_FA_KV_NATIVE=0` "avoids" it by removing the arena and folding the F16 scratch back into the
graph reserve, which was sized at init) — but note that workaround also disables the coupled
q8_0/q4_0/bf16/q5_1 native decode arms (the reporter's ~10 % decode cost).

Separately, on **r8+** the qwen4exp graph adds the two qsa3 F16 K/V packs (~2 × `n_kv × n_kv_heads ×
256 × 2` bytes); `LLAMA_QSA3_ENABLE=0` removes them, which is why the reporter's r17+qsa3-off run
passes.  The 59(a) candidate patch removes the packs *by default* on gfx1100 (backend probe), so it
also recovers that headroom.

**Actionable knobs that already exist** (no patch needed):
* `GGML_CUDA_FA_STAGE_MAX_MB=<MiB>` — bound/disable the native prefill staging arena per operand.
* `LLAMA_QSA3_ENABLE=0` — compile the qsa3 path (and its packs) out; the 59(a) patch makes this the
  runtime default on RDNA3_0.
* `-ub 128/192` — shrink the prefill graph.
* `LLAMA_KQ_MASK_DERIVED=0` — the reporter already lists it as neutral, so not a lever here.

**Follow-ups worth doing on real qwen4exp hardware:** (1) count the staging arena in `--fit` /
`llama_get_memory_breakdown` so the default fit does not under-reserve it; (2) re-measure qsa3 and the
lightning-indexer WMMA4 arm on gfx1100 for a bit-exact tuning; (3) after that, decide the RDNA3_0
defaults from data.

---

## Files / integration status

> **Current state (2026-09-29):** the branch is **rebased onto `main` = r21** and the r20-based
> integration was dropped in the rebase; `patches/`/`release.json` are r21's.  The complete fix is
> `fix-r20-full.patch` and must be re-integrated on r21 — see `FINDINGS-2026-09-29.md` (reporter
> results + work items).  The text below describes the earlier r20 run and is kept for reference.

* **Historically** integrated into the 16-patch delivery set on branch `wip/issues-59-60`
  (`patches/0015`), release string `v16-84e76d8a2-wip-issue59-60`, tip `2abadc5fd…`, applied tree
  `042a24cf…` (`scripts/validate-set.sh` strict 16/16).  The gates were folded into **block 15**.
* `fix-r20-full.patch` — the complete 8-file fix (gates + arena accounting).
* `candidate-gates.patch` — the original four-file standalone diff against r20 (no arena accounting).
* Verification on this box (gfx1100, no qwen4exp): build clean; `FLASH_ATTN_QSA` 23/23 default /
  26/26 with `GGML_CUDA_QSA3=1`; `LIGHTNING_INDEXER` nh=4 81/81 `not supported` default / 81/81 `OK`
  with `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1`; `TOPK_QSA` and `FLASH_ATTN_EXT` green; qwen35 q5_1
  coherence smoke clean.  Arena accounting: `-ctk/-ctv q5_1` `compute` row 513 -> 1537 MiB (+1024 =
  `2x512`), `GGML_CUDA_FA_STAGE_MAX_MB=64` -> 641 MiB (+128), f16 KV -> 513 MiB (none).
