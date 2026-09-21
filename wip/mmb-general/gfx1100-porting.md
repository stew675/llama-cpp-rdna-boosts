# gfx1100 (RDNA3_0) porting plan — the `mmb-general` WIP onto a single RX 7900 XTX

**Status:** PLAN / active handover (opened 2026-09-21).  Not part of the delivery.  This file is the
multi-session porting overlay for **gfx1100**.  It mirrors `gfx1201-porting.md` (the RDNA4 overlay)
and re-examines the prior "gfx1100 needs an RDNA3_0 pass / MMB is untested there" decisions in the
light of the substantial delivery + WIP changes since they were written — see **§2**.

> **HANDING THIS FILE TO A NEW SESSION?  Read §13 first** — it is the brief for the next unit of
> work (the session breakdown is §8).  This file is the *plan and record convention*; the raw numbers
> from each session go into a dated results file (`gfx1100-sNN-results.md`), not in here.

**Audience:** whoever picks this up next, with **no prior context**.  Read this with `GROUPS.md` (the
6-patch triage), `gfx1201-porting.md` (the RDNA4 port this is the sibling of) and `HANDOVER.md` (the
gfx1151 development record).  The group semantics stay as `GROUPS.md` describes.

**Session 1 log (2026-09-21):** the WIP was applied to the new `~/llama-wip-gfx1100` worktree
(branch `mmb-gfx1100`, `git am` **6/6**, tree `580db5174574f10cc92fb1cefa72281a65c77b12`) and built
green for gfx1100.  The full baseline matrix (B1-B9) was recorded on the delivery and the WIP: with
the gates off / arch-gated the WIP is **byte-identical to the delivery** on every gate, MTP is
healthy (27B 38→70.6 t/s, acceptance 0.78; MoE 113→151 t/s, 0.66), and the oracles are green.  Two
environment finds: the box exposes a **gfx1036 iGPU** that must be masked with
`HIP_VISIBLE_DEVICES=0`, and the G5 oracle is named **`TOPK_QSA`**, not `INDEXER_TOPK`.  Raw data:
**`gfx1100-s1-results.md`**.

**S2-S4 log (2026-09-21): the arch-neutral groups are decided and `qsa3` is ported to gfx1100.**
Raw data: **`gfx1100-s2s4-results.md`**.
* **G5 indexer:** always-on/generic, `TOPK_QSA` 4/4; performance is trust-RDNA3_5 (no model fits).
  The `GGML_OP_NAME` fill fix is already in patch 5 — no action.
* **G4 non-temporal:** a clean NT-off build (plain loads in `moe-weighted-reduction.cu`,
  `concat.cu`, `unary.cu`) interleaved at `r=5` shows it is **neutral on gfx1100** (all deltas
  ≤ ±0.23 %, sign model/depth-dependent) — unlike gfx1201's consistent +0.3-0.4 %.  **No code
  change** (the hints stay arch-neutral).
* **G3a always-QSA:** cannot be measured here; **the dense shortcut stays ON** on gfx1100 (the arch
  default).  Trust-RDNA3_5.  No code change.
* **G2 `qsa3` PORTED to gfx1100:** the predicate gained `RDNA3_0`; `FLASH_ATTN_QSA` **26/26**, and a
  `rocprofv3` trace confirms `qsa3_attn_kernel` + the pack/merge/rows kernels actually dispatch.
  Packaged as **patch `0007`** (`wip/mmb-general/gfx1100/patches/`).  End-to-end qwen4exp perf is
  trust-RDNA3_5.

Next: **S5-S7 (G1 `mmb`)** — the headline re-tune/re-scope.  Live work is §6.5.

**S5-S7 log (2026-09-21): `mmb` is a LARGE gfx1100 win.**  Raw data: **`gfx1100-s5s7-results.md`**.
* **S5 (open):** `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1` fires (`MMB_DENSE` + `MMB_GLU`); PPL parity
  (27B −0.9 %, 35B +0.4 %).
* **S6 (matrix):** with the gfx1151 default, the **quantized dense GEMM wins big** — 27B Q4_K_M
  **+14.4 %**, gemma-12B Q8_0 **+11.3 %** at pp8192 — the opposite of gfx1201's dense regression.
  The 35B MoE +2.4 %; gemma-26B-A4B **−3.1 %**.
* **The one loser is the F32 MoE-router split.**  gemma-26B's regression vanishes with
  `GGML_CUDA_MMB_F32SPLIT=0` (it only ever ran the f32 router; its quantized weights are Q4_0, not an
  MMB type), and the 35B gains further (+3 % more).  **Patch `0008`** defaults the F32 split off on
  RDNA3_0/RDNA4 (gfx1151 unchanged).  With that default: **27B +14.4 %, gemma-12B +11 %, 35B MoE
  +5.6 %/+4.7 %, gemma-26B neutral.**
* **Correctness:** decode unchanged (`mmb_min_t = 512`), width purity PASS, PPL parity; the 27B
  long-context text re-baselines with MMB on (expected — a different GEMM contraction, the qsa3
  class), 35B identical.  MTP 27B acceptance 0.80 (off 0.78), prefill 1154 t/s (off 1000).

Next: **S8 (G3b/c + HC16)** — then S9 (the delivery re-examination + full matrix) and S10
(freeze/merge).  See §6.6 and §8.

**S8-S9 log (2026-09-21): S8 is a no-op on gfx1100 by construction; S9's FA re-examination is
closed.**  Raw data: **`gfx1100-s8s9-results.md`**.
* **S8:** the F32 split is policy-off (patch `0008`); **HC16 is hard-gated to `RDNA3_5`** in
  `ggml-cuda.cu`, so `GGML_CUDA_MMB_HC16` is a no-op on gfx1100; the tiny-M kernel is qwen4exp-only.
* **S9 FA head cap:** the block-04 RDNA3_0 cap 256 **holds and the margin is large** — forcing
  head-512 WMMA (`GGML_CUDA_FA_WMMA_MAX_HEAD=576`) costs gemma-12B pp16384 **−9 %**, gemma-26B
  pp32768 **−15 %**.  Keep the cap.
* **S9 verify-width gate (rule 5):** green (B=4/B=8 equal delivery vs WIP on a dense K-quant gemma).
* **S9 open:** the `mmvq` RDNA3_0 `nwarps` and `VDR_Q8_0` re-sweeps, the block-13-fused vs
  MMB-routed kernel-time A/B, and the consolidated B1-B9 on the frozen tree (S10).

**S9 continued (2026-09-21): the delivery re-examination is CLOSED.**  Raw data:
**`gfx1100-s9-mmvq-results.md`**.
* **§2.6 MMB kernel-time (35B pp8192):** total **8514.6 → 8051.4 ms (−5.4 %)**; the monolithic
  `mul_mat_q` expert family is replaced by `mmb_routed_glu` + `mmb_routed`, dense by `mmb_dense`,
  attention unchanged.
* **§2.3 `mmvq` `nwarps`:** shape-dependent — `all-8` regresses (27B −3 %, 35B −3.3 %); the current
  table wins dense Q8_0 (+1.9 % on gemma-12B) and ties the 27B; **`all-1` wins the MoE** (35B +2 %
  decode, **+4.5 % MTP**, gemma-26B +4.7 %).  No clean `(type,K)` rule → **table kept**, `all-1`
  recorded as a MoE-only candidate (needs an M-based dispatch).
* **§2.4 VDR:** 4 vs 2 is a **wash** → keep 4.
* **§2.7 native KV:** all 8 types width-pure on gfx1100.

Next: **S10 (freeze, regenerate the overlay, merge back to `wip-mmb-general`)**.

**Deferred-tuning pass (2026-09-21, maintainer request): no code changes.**  Raw data:
**`gfx1100-s9-mmb-tuning-results.md`**.
* **MMB routed/GLU thresholds:** default **32 is optimal** (128 is −3.7..−3.9 %, 8 is −0.6..−1.5 %).
* **DBUF:** the big dense tile **cannot compile** (72 KB LDS > gfx1100's 64 KB) — why no dispatch
  passes it; the routed big tile fits but is neutral-to-slightly-negative → **refuted**.
* **IQ3_XXS GLU:** a wash → keep off.
* **`nwarps` MoE candidate:** the +4.5 % MTP win is real, but the shapes do **not** separate by
  `(type, K)` (or `M` alone), and the decode path uses the plain table (the `(type,K)` rule only
  reaches the verify body).  **Not landed** — recorded with the shape data and an `M >= 4096`
  hypothesis for a dedicated session.

---

## 0. TL;DR

* **gfx1100 is the arch where group 1 (`mmb`) has the best chance of transferring**, because it
  shares the **first-gen gfx11 WMMA builtin** with the validated gfx1151 reference.  It needs
  **none of the gfx12 fragment work** (`gfx1201-porting.md` §2).  It is currently gated behind
  `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`.
* **But feasibility ≠ value, and the gfx1201 record is the warning**: MMB with all types and the
  dense path on was a −4…−13 % regression there, and the wins were confined to the routed-MoE and
  qwen4exp-HC shapes.  gfx1100's delivery path is *also* re-tuned (block 04/10/13) and single-GPU,
  so **the gfx1151 win percentages do not transfer**.  Measure first, port second.
* **The arch-neutral groups (G5 indexer, G4 non-temporal) are the low-risk items** and are *partly*
  testable here.  G5 is qwen4exp-only (no model fits a 24 GB card) → unit-oracle + trust-RDNA3_5;
  G4's MoE/concat/unary hints **are** testable on the 26B-A4B / 35B-A3B models.
* **G2 (`qsa3`) is a one-line predicate change for gfx1100** (add `RDNA3_0`), then the
  `FLASH_ATTN_QSA` op oracle gives real unit coverage on this box.  End-to-end qwen4exp performance
  cannot be measured here → trust-RDNA3_5 (§9).
* **The real gfx1100 prize is the G1 re-tune / re-scope**: which weight types and which paths win on
  RDNA3_0.  gfx1151's full type set + dense-on default is what gfx1100 inherits today; gfx1201's
  measurement says that is not safe to assume.  This is the headline work (§6.5, S5-S7).
* **Re-examine the delivery's existing gfx1100 decisions** (§2) — they predate block 04/10/13/15 and
  the WIP, so several (the `mmvq` RDNA3_0 `nwarps` table, the `VDR_Q8_0` MoE=4 choice, the FA head
  cap 256, the native-KV auto policy) may be stale or interact with MMB.

---

## 1. Scope, target hardware, working state

| | |
|---|---|
| Host | **this box: 1× AMD Radeon RX 7900 XTX (gfx1100, RDNA3_0)**, Ryzen 9 7950X, 30 GiB RAM |
| GPU memory | **24 GB** — this is the hard constraint (no model sharding, no tensor parallel, no all-reduce) |
| ROCm | `/opt/rocm-7.14-gfx1100` (`hipconfig` 7.14.60850); the build script's `ROCM_714` |
| Delivery base | `~/llama.cpp` branch `rdna-boosts`, HEAD `c8dda33dd`, applied tree `8a80535e…` = the 16-block **r12** delivery |
| WIP base | the same r12 tree; the 6 WIP patches apply **6/6 clean** (verified this session) |
| **Code branch** | **`mmb-gfx1100`** in the worktree **`~/llama-wip-gfx1100`**, cut from `c8dda33dd`, the 6 WIP patches applied → **tree `580db5174574f10cc92fb1cefa72281a65c77b12`** (matches the documented WIP tree exactly) |
| **Record branch** | **`wip-mmb-general-gfx1100`** in this repo (`llama-cpp-rdna-boosts`), cut from `wip-mmb-general` at `1f2c92d` |
| Baseline build | the delivery binary already exists at `~/llama.cpp/build-rocm/bin/` (built 2026-09-20); for a clean A/B keep it or copy its `bin/` aside (rpath `$ORIGIN`) |
| Build | `cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714` (note: the script `rm -rf build-rocm`, so the WIP build and the delivery build must live in **different worktrees** — one per directory) |
| Dense mixed-type model | `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` (16.5 GB, `qwen35`) |
| Dense full-Q8_0 model | `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` (12.7 GB, gemma4) |
| MoE models | `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14.2 GB) and `/llm/models/Qwen3.6/35B-A3B/True-Q3_K_M/Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf` (17.1 GB, `qwen35moe`) |
| qwen4exp / Qwen3.8-Flash-Next | **not present / cannot fit** (94 GiB).  Its G2/G5/G3a work is unit-oracle + trust-RDNA3_5 only (§9) |
| Rule | **single GPU** — always `-ngl 99`, never `-sm tensor`; the `GGML_CUDA_ALLREDUCE`/hybrid-AR work is irrelevant here |

> **Branch-split rationale (maintainer decision, 2026-09-21).**  The gfx1201 port is still active on
> `wip-mmb-general`; this gfx1100 work lives on its **own** record branch and its **own** code
> worktree/branch so the two don't collide.  Merge the record branch back into `wip-mmb-general`
> (rebase or a merge commit) once the gfx1100 changes are frozen — see §10/§13.

**WARNING:** nothing under `wip/` may be folded into the delivery or applied to a *delivery* checkout
(`AGENTS.md` WIP rule).  This plan, the worktree and the local branches are WIP only.  Promotion is
maintainer-gated (`HANDOVER.md` §E).

### WIP patch layout (6 patches, tree `580db5174…`)

| # | patch | theme | gate | always-on? |
|---|---|---|---|---|
| 1 | `0001-WIP-mmb-…` | the general-purpose `mmb` bf16-WMMA dequant weight GEMM | `GGML_CUDA_MMB=1` + arch gate | no |
| 2 | `0002-WIP-qsa3-…` | `qsa3` packed-block WMMA sparse attention (+ the `FLASH_ATTN_QSA` oracle cases) | `LLAMA_QSA3_ENABLE` (default 1) + runtime arch gate | **yes** (qwen4exp) |
| 3 | `0003-WIP-the-F32-tiny-M-…` | F32/tiny-M kernels, the default flips (incl. the per-arch always-QSA gate), the W=1..8 width probe | `GGML_CUDA_MMB` / policy | mixed |
| 4 | `0004-WIP-HC16-…` | HC16 native-BF16 producers + non-temporal accesses | `GGML_CUDA_MMB_HC16=1` (producers) / none (NT) | mixed |
| 5 | `0005-WIP-indexer-…` | the fused indexer top-k op (`TOPK_QSA` oracle) | none (op-driven) | **yes** (qwen4exp) |
| 6 | `0006-WIP-mmb-RDNA4-…` | the `mmb` gfx12 fragment port **+ the arch-scoped weight-type/path policy** | `GGML_CUDA_MMB=1` + scope policy | no |

---

## 2. Re-examination of prior gfx1100 decisions (the requested review)

Every item below was decided **before** the recent substantial changes (the block 04/06/10/13/15
delivery work and the entire `mmb-general` WIP).  Each is therefore "re-open and re-measure", not
"trust".  The point is to find decisions that the newer code *invalidates*.

### 2.1 The `mmb` gate

* **Prior decision.**  `mmb_enabled()` (`mmb.cu`) defaults to `RDNA3_5 || RDNA4`; RDNA3_0
  (gfx1100/1101/1102) **shares the gfx11 builtin but is untested**, so it needs
  `GGML_CUDA_MMB_RDNA3=1`.  The WIP's own note says "the tiling likely needs an RDNA3_0 pass".
* **Why it is stale.**  The WIP has since grown from "IQ4_NL dense" to **11 weight types, dense +
  routed + GLU + F32-split + tiny-M + HC16** across three arches.  The gate was written when the
  only validated arch was gfx1151.  gfx1201 has since been measured (and *scoped*); gfx1100 is now
  the only gfx11 sibling left unmeasured.
* **Action (S5).**  Open it with `GGML_CUDA_MMB_RDNA3=1`, confirm MMB actually fires (`GGML_CUDA_MMB_LOG=1`,
  PPL parity), then re-scope/re-tune per §6.5.  Do **not** default it on until it is measured.

### 2.2 The `mmb` weight-type and path policy on RDNA3_0

* **Prior decision.**  `mmb_wtype_mask()` returns the **full set** (IQ family + `K_AND_Q8`) for
  RDNA3_0; `mmb_dense_flag()` is **ON** for RDNA3_0 (the dense stand-down is RDNA4-only).  I.e.
  gfx1100 inherits gfx1151's policy verbatim.
* **Why it is stale.**  The gfx1201 measurement (`gfx1201-s5s7-mmb-results.md` §3-§4) showed that
  "IQ weights win" is *not* the rule and that the **calling path** is at least as strong a factor:
  the generic dense tile lost for *every* type on RDNA4 while the routed/HC paths won.  RDNA3_0 is a
  *tuned-differently* sibling (`GROUPS.md`), and its delivery already carries the block-13 fused MoE
  MMQ paths that MMB would compete with.
* **Action (S6/S7) — DONE 2026-09-21.**  Measured per path: the quantized dense GEMM is a **large
  win on gfx1100** (27B Q4_K_M +14.4 %, gemma-12B Q8_0 +11.3 % at pp8192), and the routed MoE path
  wins too (~+4 %).  The **only** loser is the **F32 MoE-router split**; `mmb_dense_flag()` being ON
  meant gfx1100 inherited gfx1151's f32split-on, which cost gemma-26B-A4B −3.1 % and 35B-A3B −3.0 %.
  **Patch `0008`** now defaults the F32 split off on RDNA3_0/RDNA4 (gfx1151 unchanged), so a user
  who sets `GGML_CUDA_MMB=1` on gfx1100 cannot lose.  The full type set is kept (no type narrowing
  needed on gfx1100).  Data: `gfx1100-s5s7-results.md`.

### 2.3 The `mmvq` RDNA3_0 parameter table

* **Prior decision.**  `MMVQ_PARAMETERS_RDNA3_0` (`mmvq.cu`, gfx1100 sweep 2026-08-28): the whole
  band (`ncols_dst 1..8`) gets `nwarps=8` for `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q6_K/IQ4_NL`, `nwarps=1`
  otherwise.  `Q2_K/Q4_K/Q5_K/IQ4_XS` were measured to regress at 8.
* **Why it is stale.**  The later delivery work changed the mmvq landscape: the block-10 **VDR**
  work, the block-13 **per-`(type,K)` weight override** (`calc_nwarps_weight`, the Q8_0 short-K wide
  block) and the **band-uniformity** requirement (issue #30) all came after this table.  The RDNA4
  table itself was later revised to band-uniform `nwarps=1` + a per-`(type,K)` override, because the
  single-token-tuned values cost up to +35 % at the verify widths.  The gfx1100 table has **not**
  been re-validated against the verify widths or the new weight-kernel shape, and its per-type
  regressions (Q2_K/Q4_K/Q5_K/IQ4_XS at 1) may have moved.
* **Action (S9) — DONE 2026-09-21: shape-dependent, table left as-is.**  Two variant builds (`all-1`,
  `all-8`) vs the current table: `all-8` regresses (27B −3 %, 35B −3.3 %); `cur` wins dense Q8_0
  (gemma-12B +1.9 %) and ties the 27B; **`all-1` wins the MoE models** (35B +2 % decode,
  **+4.5 % MTP**; gemma-26B +4.7 %).  No clean `(type, K)` rule separates them (the 35B `shexp`
  K=2048 wants 1, gemma-12B dense K=3840 wants 8), and the RDNA4 `calc_nwarps_weight` short-K→8
  rule is the *opposite* of gfx1100's MoE preference.  **Keep the table; record `all-1` as a
  MoE-only candidate** needing a per-shape (M-based) dispatch.  Data: `gfx1100-s9-mmvq-results.md`.

### 2.4 The `VDR_Q8_0_Q8_1_MMVQ_MOE` choice

* **Prior decision.**  On RDNA4/RDNA3_0 the MoE-expert Q8_0 kernel uses **VDR = 4** (block-10's wide
  chunk): "gfx1100 verified 2026-08-28: tg128 123.74 → 127.7x (+3.x %), PPL near-lossless".  Dense
  mmvq keeps the upstream VDR.
* **Why it is stale.**  It was tuned on a single-token decode before the block-13 band work and the
  issue-#30 band-uniformity fix; the RDNA4 assignment (dense VDR reverted, MoE VDR=4, *per kernel*)
  is now the documented reference.  The interaction with MMB's routed path (which would replace the
  mmvq expert kernel on eligible weights) also did not exist then.
* **Action (S9) — DONE 2026-09-21: confirmed harmless (a wash).**  `VDR=2` vs `VDR=4` on the two MoE
  models is within noise (35B 126.01 vs 125.88 tg128; MTP 148.7 vs 149.1); the 2026-08-28 +3 % does
  not reproduce on these shapes, but VDR=4 does not hurt.  **Keep VDR=4.**  Data:
  `gfx1100-s9-mmvq-results.md`.

### 2.5 The FA WMMA head cap and `ncols2` rule (block 04)

* **Prior decision (2026-09-18, r5, issue #30).**  `GGML_CUDA_CC_IS_RDNA3_0(cc) ? 256` WMMA FA head
  cap (head 512 takes the **tile** kernel on gfx1100), and RDNA3_0 keeps the **stock AMD** `ncols2`
  rule under `-sm tensor` (irrelevant on 1 GPU, but the head cap is not).
* **Why it is stale.**  The block-06/13/15 FA instance reorganisation, block-15 **V3 derived kq
  mask** (now also on the tile kernel, r9) and **V4/V5 native KV** all changed the FA path since.
  The gfx1100 head-512 case (gemma4) is exactly what the 12B Q8_0 probe exercises, and the tile
  kernel now has the derived-mask path (a deep-prefill win on gfx1100 per r9).  The cap may still be
  right, but the surrounding kernel choice has moved.
* **Action (S9) — DONE 2026-09-21: the cap HOLDS.**  Forcing head-512 WMMA
  (`GGML_CUDA_FA_WMMA_MAX_HEAD=576`) on the current tree costs gemma-12B pp8192/pp16384 **−5.6 % /
  −9.1 %** and gemma-26B-A4B pp8192/pp32768 **−5.6 % / −14.6 %**.  The block-04 RDNA3_0 cap 256
  (head 512 → tile) is correct, and the margin is larger now than in the 2026-09-18 measurement
  (consistent with the r9 V3-on-tile improvement).  Data: `gfx1100-s8s9-results.md` §9.1.

### 2.6 The block-13 RDNA3_0 MoE fusion gate

* **Prior decision (2026-09-05).**  The fused gate+up+GLU **MMQ** arm and its `J_max_gate` caps were
  relaxed to RDNA3_0 after a gfx1100 validation (pp2048 4939 → 5405, +9.4 %); a Q3_K@96 probe lost to
  the cap 64.
* **Why it is stale.**  MMB's **routed** path (and its own fused GLU) would compete with the MMQ
  fusion on the same MUL_MAT_ID shapes.  The WIP's stand-down logic is per-tensor, so if MMB takes
  the routed weight the MMQ fusion steps aside — meaning the block-13 win and MMB's routed win are
  **alternatives, not additive**.  S7's "MMB routed wins more once the non-winning paths stop
  dragging" is exactly this.
* **Action (S7).**  A/B the block-13 fused path vs MMB routed on the MoE models, per type, with
  `rocprofv3` kernel time (both are routed shapes; end-to-end t/s alone is not admissible — §7).

### 2.7 The block-15 native KV / derived mask policies on gfx1100

* **Prior decision.**  V3 (`LLAMA_KQ_MASK_DERIVED`, on) is bit-identical across 4 KV types on
  gfx1100 (r9); V4 (`GGML_CUDA_FA_KV_NATIVE`) auto-enables native **q8_0/q4_0** and leaves **bf16
  off**; the 2026-09-15 amendment added native `q4_1/q5_0/q5_1/iq4_nl`.  The prefill staging arena
  is arch-gated `prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc)`, i.e. **RDNA3_0 stages**.
* **Why it is stale.**  These were validated on gfx1151/gfx1201 more than gfx1100; the auto policy is
  a *global* default, not per-arch.  The WIP's G2/G4 changes (qsa3, non-temporal) touch the same
  attention/memory paths.
* **Action (S9) — DONE 2026-09-21: all 8 native KV types are width-pure on gfx1100.**  `KV=`
  `f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl` all `PASS (worst maxdiff 0)` on the 27B with MMB on,
  and `FLASH_ATTN_EXT` is 5955/0-FAIL (it covers the new native arms).  The native auto policy is
  right for gfx1100; the per-type deep-prefill numbers remain trust-RDNA3_5/RDNA4.  Data:
  `gfx1100-s9-mmvq-results.md`.

### 2.8 Smaller per-arch delivery gates worth a sanity check

* `mmq.cu`: `ne11 <= (RDNA3_0 ? 128 : 256)` in a MUL_MAT_ID dispatch — re-check with the fused-MoE
  work.
* `mmf.cu`: `if (GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8)` — the multi-token redirect.
* `vecdotq.cuh`: `VDR_Q8_0_Q8_1_MMVQ_MOE=4` on RDNA3_0 (see §2.4).
* `allreduce-hip.cu`: the internal/hybrid AR is **RDNA4-only** — a non-issue on a single card.

---

## 3. Environment, build, run

### 3.1 Worktrees and builds

There is **one build directory per worktree** and the build script `rm -rf build-rocm`s it, so keep
the delivery and the WIP in separate worktrees:

```sh
# delivery (baseline) — already built
cd ~/llama.cpp && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714     # gfx1100, ccache
# or, for a cheap A/B, copy the binary aside once:  cp -a build-rocm/bin /tmp/base-bin

# WIP
cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714                  # first run: configure+build
# fast loop while iterating (no reconfigure):
cmake --build build-rocm --target llama-cli llama-bench llama-perplexity test-logits-width-probe test-backend-ops -j 16
```

Notes:
* The build script hardcodes `ROCM_714=/opt/rocm-7.14-gfx1100`, `-DGPU_TARGETS=gfx1100` and enables
  ccache (already populated for this tree from the gfx1100 delivery builds).
* New `.cu` files (`mmb.cu`, `fattn-qsa3.cu`, `indexer-topk.cu`) need a **configure** pass — the glob
  is evaluated at configure time.  The full script does that; the `cmake --build` loop does not.
* `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1100/lib` is harmless (the rpath is `$ORIGIN:…/lib`).

### 3.2 Apply / reset the WIP

The 6-patch set applies `git am` **6/6** onto a fresh r12 tree and yields applied tree
**`580db5174574f10cc92fb1cefa72281a65c77b12`** (verified this session).  To re-create the worktree:

```sh
cd ~/llama.cpp
git worktree add ~/llama-wip-gfx1100 -b mmb-gfx1100 rdna-boosts        # rdna-boosts is the r12 tree
cd ~/llama-wip-gfx1100
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/patches/*.patch   # 6/6
git rev-parse HEAD^{tree}    # -> 580db5174574f10cc92fb1cefa72281a65c77b12
```

### 3.3 Run

* **Mask the iGPU: always `HIP_VISIBLE_DEVICES=0`.**  This box enumerates a second HIP device —
  **gfx1036** (the Ryzen 7950X integrated GPU).  The build only targets gfx1100, so an unmasked
  `test-backend-ops` (or any tool that picks all devices) runs the gfx1036 device and aborts with
  `ROCm error: invalid kernel file` in the first kernel it launches.  Verified 2026-09-21.  Prefix
  every GPU command with `HIP_VISIBLE_DEVICES=0`.
* **Always** `llama-cli --single-turn` (and `--no-display-prompt` for scripted output) or it blocks.
  Wrap potentially-blocking commands in `timeout`.
* **24 GB budget.**  `-ngl 99` always; keep `-b/-ub` modest (`512/512` or `1024/1024`) and choose
  prompt lengths that fit the KV.  As a default: `-p 2048,8192` for the 14-17 GB models, and reserve
  `pp16384` for the 12.7 GB gemma-12B.  f16 KV unless a test varies the type.
* Warm the page cache first (`cat <model> >/dev/null`); never run benches in parallel.
* The 27B Q8_0 may need `-lm none -lzm on` on some subcommands; the Q4_K_M/Q3_K_M/Q4_K_XL files here
  do not (verify per model at S1).

---

## 4. Baseline gates to record before any port

Run every gate **twice**, on `~/llama.cpp/build-rocm` (delivery) and
`~/llama-wip-gfx1100/build-rocm` (WIP, all env gates off) — the second must be *identical* to the
first on every text/PPL/op gate.  (The only expected deltas are where a group is enabled.)

| # | gate | command sketch | pass |
|---|---|---|---|
| B1 | coherence, dense mixed | `llama-cli -m 27B-Q4_K_M -ngl 99 -p "The capital of France is" -n 20 --seed 42 --temp 0 --single-turn` | same-seed text identical build-to-build |
| B2 | coherence, MoE | same on `Qwen3.6-35B-A3B-Q3_K_M` (and the gemma-26B-A4B) | same-seed text |
| B3 | prefill, dense | `llama-bench -m <model> -ngl 99 -p 2048,8192 -n 0 -r 5` | record t/s |
| B4 | decode, dense | `llama-bench … -p 0 -n 128 -r 5`; a depth-16384 run (benchy protocol) where it fits | record t/s |
| B5 | prefill MoE | `llama-bench -m <MoE> -p 2048,8192 -n 0 -r 5` | record t/s |
| B6 | op oracles | `test-backend-ops -o FLASH_ATTN_EXT`, `-o FLASH_ATTN_QSA` (26/26 once G2 lands), `-o GATED_DELTA_NET`, **`-o TOPK_QSA`** | green |
| B7 | width purity | `HIP_VISIBLE_DEVICES=0 test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512` | `width_purity=PASS (worst maxdiff 0)` for the pure types, f16 KV |
| B8 | PPL | `llama-perplexity -m <model> -f prompts/prose-rdna-boosts.txt -c 2048 -b 2048 -ub 2048 -ngl 99 -fa 1` | record |
| B9 | MTP | `benchmarks/mtp-adaptive-methodology.md` (gemma-12B + its `-MTP.gguf`) | acceptance > ~0.45 at pos 1; MTP ≥ plain at depth 3 |

> **Oracle-name correction (this session).**  `GROUPS.md` and the gfx1201 records call the indexer
> oracle `-o INDEXER_TOPK`.  There is **no such test-case name**; the G5 oracle is registered as
> **`TOPK_QSA`** (`test-backend-ops.cpp`, `test_topk_qsa`).  Use `-o TOPK_QSA` — `-o INDEXER_TOPK`
> matches nothing and can look like a pass.  (The op enum is `GGML_OP_INDEXER_TOPK`; only the
> test-case name differs.)

**Important:** record every number in a new dated `gfx1100-sNN-results.md` (not in this plan).

### 4.1 S1 results (2026-09-21) — the baseline is recorded

The full S1 matrix is in **`gfx1100-s1-results.md`**.  Headline: the WIP (gates off / arch-gated) is
**byte-identical to the delivery on every gate**, so the S2+ A/Bs start from a trustworthy baseline.

The cheap highlights:

* WIP **builds green for gfx1100** (0 compiler errors).
* `HIP_VISIBLE_DEVICES=0 test-backend-ops -o TOPK_QSA` → **4/4** (G5 indexer, generic).
* `HIP_VISIBLE_DEVICES=0 test-backend-ops -o FLASH_ATTN_QSA` → **26/26**.  With the qsa3 predicate
  still excluding `RDNA3_0`, the four packed cases fall back to the VEC kernel, so this only proves
  the test is runnable here; **S4 re-runs it with the predicate changed and that is the real qsa3
  coverage on gfx1100.**
* `FLASH_ATTN_EXT` → **5955 cases, 0 FAIL**; `GATED_DELTA_NET` → 2/2.
* `test-logits-width-probe` → **`PASS (worst maxdiff 0)`** on all four models, f16 KV.
* MTP works and is a large win (27B 38→70.6 t/s, acceptance 0.78; MoE 113→151 t/s, acceptance 0.66).
* The gfx1100 `mmb`/`qsa3` device paths compile from the **same gfx11 arm** as gfx1151 — no gfx12
  work is needed, as `GROUPS.md` predicted.

---

## 5. Group inventory against gfx1100

| # | group | gfx1100 status today | port work | expected gfx1100 payoff | testable here? |
|---|---|---|---|---|---|
| G5 | indexer top-k op | generic, always-on, compiles | none (apply) | qwen4exp deep prefill; gfx1151 family 2.94 %→1.80 % of run | **unit only** (`TOPK_QSA`) — no model |
| G4 | non-temporal hints (`dsv4_hc`, `concat`, `moe-weighted-reduction`, fused gated-unary) | generic; `GGML_CUDA_MMB_HC16`-independent | none (per-kernel A/B, loads only) | `dsv4_hc_pre` −18.6 % on gfx1151; MoE/reduction hints likely small + | **yes** (MoE models) |
| G3a | always-QSA prefill flip | arch-gated: shortcut **ON** on gfx1100 | none | policy; only matters for qwen4exp | **no** — trust-RDNA3_5 (§9) |
| G2 | `qsa3` packed-block WMMA | gfx11 builtin available; predicate is `RDNA3_5 || RDNA4` | add `RDNA3_0` to the predicate | +7.6..+11.5 % qwen4exp prefill on gfx1151/gfx1201 | **unit** (`FLASH_ATTN_QSA` 26/26) |
| G1 | `mmb` bf16-WMMA dequant GEMM | gfx11 builtin; needs `GGML_CUDA_MMB_RDNA3=1` | open + re-tune/re-scope | the headline; gfx1151 was +32…+48 % prefill, gfx1201 scoped +3…+7 % | **yes** (dense + MoE models) |
| G3b/c | F32 shape split + tiny-M kernel | rides G1 | with G1 | MoE router / qwen4exp HC-inject shapes | **partly** (router on MoE) |
| G4 | HC16 bf16 producers | rides G1 (`GGML_CUDA_MMB_HC16`, default 0) | after G1 | kills `mmb_cvt`; +1-3 % on gfx1151 | **partly** (MoE paths) |

---

## 6. Port designs (gfx1100-specific)

### 6.1 G5 — indexer top-k (S2)

* **Files:** `ggml/src/ggml-cuda/indexer-topk.cu` (new, patch 5), `src/llama-memory-hybrid-idx.cpp`
  (`blk_cells`), `src/models/qwen4exp.cpp`.
* **Gating:** none; the op is generic CUDA/HIP and compiles for gfx1100 already.
* **What to prove here:** `test-backend-ops -o TOPK_QSA` green on gfx1100; the op is otherwise
  qwen4exp-only and has **no reachable model on a 24 GB card**.  Performance parity is
  trust-RDNA3_5 (§9).
* **Also fold in the delivery item** `HANDOVER.md` §D: `GGML_OP_INDEXER_FILL` is missing from
  `GGML_OP_NAME` (one line).
* **Status (S2, 2026-09-21): DONE.**  `TOPK_QSA` 4/4 on gfx1100; the `GGML_OP_NAME` fix is already
  present (patch 5); performance is trust-RDNA3_5.  No code change.

### 6.2 G4 — non-temporal hints (S2)

* **Files:** `common.cuh` (`ggml_cuda_nt_load`), `dsv4-hc.cu` (qwen4exp-only), `concat.cu`,
  `moe-weighted-reduction.cu`, `unary.cu` (fused gated-unary).
* **Gating:** none; a per-kernel A/B.  **Rule learned on gfx1151: load hints only, per kernel**; a
  non-temporal *store* evicts the next op's input.
* **What to test here:** the MoE-side hints (`concat`, `moe-weighted-reduction`, gated-unary) on the
  26B-A4B / 35B-A3B models; `dsv4_hc` is qwen4exp-only (trust-RDNA3_5).  Interleave rounds — the
  effect is small; a single noisy run decides nothing.
* **Status (S2, 2026-09-21): DONE — neutral on gfx1100.**  A clean NT-off build interleaved at `r=5`
  gives all deltas ≤ ±0.23 % with a model/depth-dependent sign (35B-A3B prefers off by ~0.2 % at
  pp32768; gemma-26B prefers on by ~0.2 % at pp16384; the 27B dense is a flat wash).  **No code
  change** — the hints stay arch-neutral.  Full table: `gfx1100-s2s4-results.md`.

### 6.3 G3a — always-QSA flip (S3)

* One hunk in `src/models/qwen4exp.cpp` (`LLAMA_QSA_DENSE_SHORTCUT` default).  gfx1100 currently
  takes the **shortcut ON** default (`qsa_arch_gfx() != 0x1151`), which is the conservative choice.
* **Cannot be measured here** (no qwen4exp model).  Keep the delivery/arch policy and document the
  trust-RDNA3_5 assumption.  If a future session gains access to qwen4exp on a multi-GPU box, the
  decision moves into the `qsa_arch_gfx()` policy alongside gfx1151.
* Do not conflate this with `qsa_dense_decode_until` / `qsa_dense_prefill_until`.
* **Status (S3, 2026-09-21): DONE — no code change.**  The shortcut stays **ON** on gfx1100 (the
  `qsa_arch_gfx() != 0x1151` default); the reverse decision is deferred to a box that can run
  qwen4exp.  Trust-RDNA3_5.

### 6.4 G2 — `qsa3` on RDNA3_0 (S4)

* **The change is one predicate line:** in `ggml/src/ggml-cuda/fattn-qsa3.cu`,
  `ggml_cuda_flash_attn_qsa3_supported()` currently accepts `RDNA3_5 || RDNA4`; add `RDNA3_0`:
  ```cpp
  if (!(GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc))) return false;
  ```
  The gfx11 arm of the fragment shim is already the compile-time `#else` (16-half full row,
  `m = 2*e + hi`), so gfx1100 uses exactly the validated gfx1151 code.  Keep the `q->ne[1] >= 128`
  prefill gate and the `n_stream == 1` graph gate — the W=1..8 decode/verify band and its width
  purity stay on the VEC kernel by construction.
* **Validation available here:** `test-backend-ops -o FLASH_ATTN_QSA` must be **26/26** on gfx1100
  (the packed cases from patch 2 now attach `src[7]/src[8]`, so they genuinely exercise the WMMA
  kernel).  This is real unit coverage even without a model.
* **Validation NOT available here:** the end-to-end qwen4exp prefill win.  Trust-RDNA3_5 (the gfx11
  code is identical); note it in the results file.
* **The delivery's `qwen4exp.cpp` policy already assumes gfx1100 = "QSA prefill always / dense
  decode"** (the 2026-09-07 crossover tables); qsa3 only makes the already-chosen QSA path faster.
* **Status (S4, 2026-09-21): DONE — PORTED.**  The predicate gained `RDNA3_0`; `FLASH_ATTN_QSA`
  **26/26**, and a `rocprofv3` kernel trace confirms `qsa3_attn_kernel` + the pack/merge/rows kernels
  actually dispatch on gfx1100.  Packaged as patch `0007`
  (`wip/mmb-general/gfx1100/patches/0007-WIP-qsa3-RDNA3_0-gfx1100.patch`).  End-to-end qwen4exp
  performance is trust-RDNA3_5 — see `gfx1100-s2s4-results.md`.

### 6.5 G1 — `mmb` on RDNA3_0 (S5–S7, the headline)

This is the largest and most valuable item.  Split it into three steps:

> **STATUS (S5-S7, 2026-09-21): DONE — and gfx1100 is the opposite of gfx1201.**  The quantized dense
> GEMM is a **large win** (27B Q4_K_M **+14.4 %**, gemma-12B Q8_0 **+11.3 %** at pp8192), the routed
> MoE path wins (~+4 % on the 35B), and the **only** loser is the **F32 MoE-router split**.  Patch
> `0008` defaults the F32 split off on RDNA3_0/RDNA4, after which: 27B **+14.4 %**, gemma-12B
> **+11 %**, 35B-A3B **+5.6 %/+4.7 %** (pp8192/32768), gemma-26B-A4B **neutral**.  Decode unchanged,
> width purity PASS, PPL parity.  Full data: `gfx1100-s5s7-results.md`.  The priming below
> ("expect a dense regression like gfx1201") **did not hold** — the dense tile is fine on RDNA3_0.

**6.5.1 Open and confirm (S5).**  `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`.  Confirm MMB fires
(`GGML_CUDA_MMB_LOG=1`) and PPL is at parity (a fragment-layout error is an exact-permutation error —
it would move PPL by orders of magnitude, not 0.1 %).  The gfx11 shim is the **already-validated**
gfx1151 code, so correctness should be a formality; confirm it rather than assume.

**6.5.2 The per-type / per-path matrix (S6).**  This is the decision that matters.  Use the overrides
already in patch 6:
* `GGML_CUDA_MMB_TYPES=<csv>` (e.g. `iq3_s,iq4_xs,q8_0,q4_k,…`) — the arch-scoped type mask.  On
  RDNA3_0 it currently returns the **full set**; verify that, or narrow it.
* `GGML_CUDA_MMB_DENSE=0|1` — separates the generic dense tile GEMM + F32 router (the gfx1201 losers)
  from the routed/MoE + HC paths (the gfx1201 winners).  On RDNA3_0 it currently defaults **ON**.
* Prime the sweep with the gfx1201 finding: with everything on it may be a large dense regression;
  the wins are likely **routed MoE** and possibly the qwen4exp HC shapes.  Models to sweep (single
  GPU, fitting 24 GB):
  * **Qwen3.6-35B-A3B Q3_K_M** — `qwen35moe`, routed experts (the gfx1201 +6.7 % case shape).
  * **Gemma-4-26B-A4B QAT UD-Q4_K_XL** — `gemma4` MoE, a broad type mix.
  * **27B UD-Q4_K_M** — dense, mixed K-quant (the dense-loss probe).
  * **Gemma-4-12B Q8_0** — dense, 100 % Q8_0 (the Q8_0-loss probe).
* Decide the gfx1100 default **per type and per path**, exactly as S7 did for RDNA4.  A user who
  sets `GGML_CUDA_MMB=1` on gfx1100 should not be able to lose on it.

**6.5.3 Re-tune the constants (S6/S7).**  The `mmb_*` tile/threshold constants are **gfx1151-tuned**;
do not transfer them.  Sweep at least `mmb_glu_thresh` / `mmb_routed_thresh` (default 32), `mmb_tall`,
`mmb_f32split_min_m`, and the `min_t` prefill threshold (default 512).  Follow the shape of
`gfx1201-porting.md` §6.5.3.  Start on the fast MoE model.
* **`MMB_BK = 64` is effectively fixed** — the `mmb_dq_row*` helpers hard-code 64 values/row and
  `MMB_LDS_STRIDE = MMB_BK + 8`.  Treat changing `BK` as a rewrite, not a knob.
* The `DBUF`/`DBUF2` flags are a **live, untried lever** (no current dispatch passes them true).
* All `mmb_*` tunables are cached in function-local statics: **changing an env var needs a new
  process**, not just a new model load.

**6.5.4 Purity / correctness gates for G1:** PPL parity, same-seed greedy, the width probe, and
`test-backend-ops`.  **S5-S7 result:** all pass; the 27B long-context greedy text re-baselines with
MMB on (a different GEMM contraction — the approved qsa3 class), the 35B is identical, and width
purity is PASS.  Decode/verify are untouched by construction (`mmb_min_t = 512`).  The MMB stand-down
is per-tensor, so an excluded (type,path) costs nothing (it keeps the delivery's MMQ path) — but any
new gate must be added to **all** of `supported_mm`, `supported_mmid`, `supported_glu`,
`dense_will_take`, `routed_will_take` or the graph and dispatch disagree (`gfx1201-porting.md` §13
trap 2).

### 6.6 G3b/c + G4 HC16 (S8)

* **F32 split (MoE router)** — currently ON for RDNA3_0 via `mmb_dense_flag()`.  Sweep
  `GGML_CUDA_MMB_F32SPLIT_MIN_M` / `_MIN_K`; the gfx1151 rule was `M >= 128` (the router) with
  `MIN_K = 0`.
* **tiny-M warp-per-token kernel** (qwen4exp HC `*_inject`, `M <= 8`) — `GGML_CUDA_MMB_TINY_TT`
  (default 1).  Only reachable for qwen4exp → trust-RDNA3_5 if no shape fires on the available models.
* **HC16 bf16 producers** — `GGML_CUDA_MMB_HC16=1` plus `GGML_CUDA_MMB_DOWN16`, `LLAMA_HC_BLK16`,
  `LLAMA_HC_RES16`.  Host-side graph marking; only meaningful with MMB active.  MoE producer paths
  are testable here; the HC ones are not.
* Purity for each: PPL + same-seed greedy.

---

## 7. Measurement protocol and gating strategy

### 7.1 Protocol (inherited from `gfx1201-porting.md` §13.0)

* `llama-bench … -n 0 -r 5`; **the first prefill test of an invocation is cold-start-limited** (up to
  −9 %).  Decide only on `r=5` and on **interleaved back-to-back rounds** (A, B, A, B in one warm
  session), never two benches at once.
* Warm the page cache for the large files.
* Prefer `pp8192` (or the deepest that fits) over `pp2048` for a decision.
* **Routed/GLU shapes (MoE experts): attribute with `rocprofv3` kernel time and/or a fixed-token
  `llama-perplexity`** — `llama-bench` uses random prefill tokens, so routed kernels are not
  comparable run-to-run.
* Never trust a gated path under `rocprofv3` without confirming the kernel name in the trace:
  `rocprofiler-register` can make an env gate read as *unset* (ROCm issue #10196).  It names the
  `GGML_CUDA_MMB_*` tunables (lazy `getenv`s), `LLAMA_QSA_DENSE_SHORTCUT`, etc.

### 7.2 Gating strategy (what is "gfx1100-specific")

1. **Device code** stays behind the existing `#if defined(RDNA3_0)` / gfx11 compile guards.  Do not
   touch the RDNA4 arms.
2. **Runtime arch gate:** `mmb_enabled()` keeps `RDNA3_0` behind `GGML_CUDA_MMB_RDNA3=1` until S6/S7
   decide a default.  `ggml_cuda_flash_attn_qsa3_supported()` gains `RDNA3_0` (the port).
3. **Per-arch tuning constants** must be selected by `ggml_cuda_info().devices[0].cc`, not compiled
   in — this is the same gap `gfx1201-porting.md` §7.3 / S11 flags.  The natural fix (where it does
   not already exist) is a small `mmb_arch_defaults(cc)` (or a per-cc conditional at each constant's
   accessor) with the env var as the A/B override.  **Do not regress gfx1151** — the cc switch must
   be a no-op there.  Precedents in-tree: `mmb_wtype_mask()`, `mmb_dense_flag()`, `qsa_arch_gfx()`.
4. **Never** let a gfx1100 change alter the gfx1151 or gfx1201 path without a guard.  The record
   branch is separate precisely so this stays reviewable.
5. New `.cu` files must be added to the build (glob re-configure) and registered consistently in the
   five MMB predicate sites (see §6.5.4).

---

## 8. Session work breakdown

Each session should: start on branch `mmb-gfx1100` in `~/llama-wip-gfx1100`, rebuild, run the
relevant gates, write a dated `gfx1100-sNN-results.md`, commit the record to the
`wip-mmb-general-gfx1100` branch, and update §11.

| session | goal | deliverables | exit gate |
|---|---|---|---|
| **S1** | build + baselines | WIP branch applied + built for gfx1100; delivery baseline built/preserved | build green; **B1-B3, B5-B8 recorded** (B4/B9 as far as they fit) |
| **S2** | arch-neutral wins ✅ | G5 indexer + G4 non-temporal applied/A/B'd; `GGML_OP_NAME` fill fix | `TOPK_QSA` 4/4; G4 **neutral** (no change); `FLASH_ATTN_EXT` 5955/0-FAIL |
| **S3** | G3a policy ✅ | gfx1100 always-QSA decision documented (trust-RDNA3_5) | shortcut stays ON (arch default); no code change |
| **S4** | G2 `qsa3` RDNA3_0 ✅ | predicate gains `RDNA3_0`; oracle re-run; kernel-trace confirmed | `FLASH_ATTN_QSA` **26/26** + `qsa3_attn_kernel` dispatched; patch `0007` |
| **S5** | G1 open ✅ | `MMB_RDNA3=1`; MMB fires; correctness | PPL parity (−0.9 %/+0.4 %); width probe PASS |
| **S6** | G1 per-type/path matrix + re-tune ✅ | **gfx1100 = dense wins big, F32 router loses**; patch `0008` | 27B +14.4 %, gemma-12B +11 %, 35B MoE +5.6 %, g26 neutral |
| **S7** | G1 routed/GLU ✅ | routed isolate (the bulk of the MoE win, ~+4 %) | kernel-time-free but `DENSE=0` isolate-backed; finer knobs deferred |
| **S8** | G3b/c + HC16 ✅ | F32 policy-off; HC16 hard-gated to RDNA3_5; tiny-M qwen4exp-only | **no measurable gfx1100 work by construction** |
| **S9** | delivery re-examination + matrix ✅ | FA cap **keep 256**; verify-width green; `nwarps` shape-dependent (table kept, MoE candidate); VDR=4 keep; 8/8 native KV pure; MMB kernel-time −5.4 % | all §2 items re-measured; one documented candidate |
| **S10** | freeze + regenerate + merge back | new WIP patch (likely patch 7 = gfx1100 scope/tuning); docs updated; `git am` N/N; merge record branch into `wip-mmb-general` | gfx1100 handed a clean state; branches reconciled |

Sessions S1-S7 are done.  S8 (G3b/c + HC16), S9 (delivery re-examination + full matrix) and S10
(freeze/merge) remain; see §13's successor log and the results files.

---

## 9. The "trust RDNA3.5" scoping (explicit assumptions)

A single 24 GB card cannot load qwen4exp / Qwen3.8-Flash-Next (94 GiB), so the qwen4exp-specific
groups cannot be validated end-to-end here.  This section records **exactly what is assumed** and
**what would falsify it**, so a future multi-GPU-RDNA3 session can close the gap.  The maintainer's
position (2026-09-21): trust the RDNA3_5 scoping for these until proven otherwise, because the
qwen4exp work is largely architecture-independent.

| item | what is assumed on gfx1100 | evidence it is safe | what would falsify it |
|---|---|---|---|
| **G2 `qsa3`** | the gfx11 fragment path is correct and the same win applies | it is literally the same compile-time gfx11 arm as gfx1151; `FLASH_ATTN_QSA` 26/26 on gfx1100 (S4) | an oracle failure → do not enable |
| **G2 `qsa3` perf** | +7.6..+11.5 % prefill at 4k-32k | gfx1151 and gfx1201 both measured it; the kernel is the same | a multi-GPU RDNA3 qwen4exp A/B |
| **G5 indexer** | the op is correct and the deep-prefill win scales | generic kernel; `TOPK_QSA` green on gfx1100 | a multi-GPU RDNA3 qwen4exp long-context text A/B |
| **G3a always-QSA** | shortcut stays **ON** (conservative) | gfx1201 showed always-QSA can be a big regression without a validated qsa3-era cost model; gfx1100 is not gfx1151 | a gfx1100 (or multi-GPU) qwen4exp `LLAMA_QSA_DENSE_SHORTCUT` A/B |
| **G4 `dsv4_hc` NT** | the −18.6 % `_pre` win transfers | same kernel, generic hint | a qwen4exp A/B |
| **G4 HC16 producers** | +1-3 % | host-side graph marking, arch-neutral | a qwen4exp A/B |
| **G3b/c tiny-M** | the HC-inject shape is unaffected | generic WMMA kernel | a qwen4exp A/B |

**Rule:** anything in this table that lands as **code** (G2's predicate, the indexer op, NT hints)
must still pass its **unit oracle on gfx1100** before shipping.  Only the *end-to-end performance
claim* is deferred.

---

## 10. Per-session record convention

* Code goes in `~/llama-wip-gfx1100` (branch `mmb-gfx1100`).  Do **not** commit it to `rdna-boosts`.
* Record the numbers + decisions in a dated **`gfx1100-sNN-results.md`** under `wip/mmb-general/` on
  branch **`wip-mmb-general-gfx1100`** (never `main`).
* Regenerate the WIP patch backup after each code change:
  ```sh
  cd ~/llama-wip-gfx1100 && git format-patch --start-number 1 c8dda33dd..HEAD -o /tmp/mmb-gfx1100
  # copy into wip/mmb-general-gfx1100/ (a new patch 0007 unless the files are owned by one patch),
  # commit the record + patches on the branch
  ```
  Land a code change as a **new patch** by default; fold it into a theme only when the files it
  touches are owned by exactly one existing patch (patches 1, 3 and 4 all touch `mmb.cu` — that is
  why the gfx1201 `mmb` work is patch 6).
* **Never push out of `~/llama.cpp`/`~/llama-wip-gfx1100`** (`AGENTS.md` Pushing policy).
* **Merge-back plan:** when gfx1100 is frozen (S10), rebase or merge `wip-mmb-general-gfx1100` onto
  the then-current `wip-mmb-general`, resolving the shared `wip/mmb-general/*.md` files by keeping
  both branches' sections.  The code changes merge as the gfx1100 patch(es).

---

## 11. Live checklist

- [x] gfx1100 record branch + code worktree created; WIP applies **6/6** (tree `580db5174574f10cc92fb1cefa72281a65c77b12`)
- [x] WIP builds for gfx1100 (S1, 2026-09-21; 0 errors)
- [x] **S1 baselines B1-B8 done** — WIP (gates off) byte-identical to the delivery on every gate; MTP healthy (27B 38->70.6 t/s acc 0.78; MoE 113->151 t/s acc 0.66); oracles green (`TOPK_QSA` 4/4, `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 2/2, `FLASH_ATTN_EXT` 5955/0-FAIL); width purity PASS on all 4 models (`gfx1100-s1-results.md`)
- [ ] G5 indexer — ✅ `TOPK_QSA` 4/4; perf trust-RDNA3_5; `GGML_OP_NAME` already fixed (S2)
- [ ] G4 non-temporal — ✅ A/B'd: **neutral on gfx1100**, no code change (S2)
- [ ] G3a always-QSA decision documented — ✅ shortcut stays ON (S3, trust-RDNA3_5)
- [ ] G2 `qsa3` predicate gains `RDNA3_0` — ✅ **26/26** + kernel trace; patch `0007` (S4)
- [x] G1 `mmb` opened (`MMB_RDNA3=1`), fires, PPL parity (S5)
- [x] G1 gfx1100 per-type/per-path: **dense is a big win, F32 router loses**; patch `0008` (S6)
- [x] G1 routed/GLU isolate: the routed path is the bulk of the MoE win (~+4 %) (S7)
- [x] G3b/c + HC16 — **no-op on gfx1100** (F32 policy-off; HC16 RDNA3_5-gated; tiny-M qwen4exp-only) (S8)
- [x] delivery re-examination DONE: FA cap 256 **keep**; verify-width gate green; **`nwarps` shape-dependent (table kept)**; **VDR=4 keep**; **all 8 native KV pure**; MMB kernel-time −5.4 % (S9)
- [x] G1 optional routed/GLU threshold + DBUF sweep — **done: no change** (thresh 32 optimal; DBUF refuted/not-viable; IQ3_XXS wash) (S9)
- [x] G1 `nwarps` MoE candidate — **investigated: real (+4.5 % MTP) but not expressible with the current dispatch axes**; deferred with data (S9)
- [ ] B1-B9 consolidated on the frozen tree, including MTP (S10)
- [ ] patch set regenerated + `git am` N/N + merged back to `wip-mmb-general` (S10)

---

## 12. Risks / open questions

1. **G1 value on gfx1100 is unknown**, and the gfx1201 lesson says "all-on" can be a large dense
   regression.  The scope split (types + paths) is the deliverable, not a blanket enable.  Budget
   S6/S7 generously.
2. **Page-cache / memory pressure.**  30 GiB RAM + 24 GiB VRAM with 12-17 GB models means swapping
   is a real risk; `llama-bench` benches must be serial and warmed, and `-ub` kept modest.
3. **No qwen4exp** → the largest group (G2/G5/G3a) is unit-only here.  The trust-RDNA3_5 table (§9)
   is the honest state; do not let a "green" unit oracle be reported as a performance validation.
4. **`rocprofiler-register` env-gate flakiness** (`GROUPS.md` §5): verify kernel names in the trace.
5. **Don't judge routed/GLU on end-to-end t/s** (`GROUPS.md` §6): use kernel time + fixed-token PPL.
6. **Re-tuning is a time sink.**  Timebox every sweep; only `r=5` interleaved rounds decide anything.

---

## 13. Brief for the next session (S1)

**Read this section, not the plan above, if you are starting fresh.**

1. **Set up.**  Confirm `git branch --show-current` is `wip-mmb-general-gfx1100` in this repo and
   `mmb-gfx1100` in `~/llama-wip-gfx1100`.  Confirm `git rev-parse HEAD^{tree}` in the worktree is
   `580db5174574f10cc92fb1cefa72281a65c77b12`; if not, re-apply the 6 patches (§3.2).  Confirm the
   delivery binary exists at `~/llama.cpp/build-rocm/bin/` and preserve it
   (`cp -a ~/llama.cpp/build-rocm/bin /tmp/base-bin`) before touching it — it is the baseline.
2. **Build the WIP for gfx1100:** `cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714`
   (configure + build; expect a few minutes with ccache).  Build the extra targets:
   `cmake --build build-rocm --target llama-cli llama-bench llama-perplexity test-backend-ops test-logits-width-probe -j 16`.
3. **Record the baselines (B1-B8, §4)** on **both** the delivery and the WIP (all gates off).  They
   must be identical everywhere the WIP has not enabled a group.  Write `gfx1100-s1-results.md`.
   * Start on the lightest pair (gemma-12B Q8_0 + Qwen3.6-35B-A3B Q3_K_M) to keep the session short.
   * Remember `--single-turn`, `--no-display-prompt`, `--seed 42 --temp 0`, and extract text with
     `scripts/extract-generated.py`.
4. **Sanity-check the oracles** you will rely on later: `test-backend-ops -o TOPK_QSA` (note:
   `-o INDEXER_TOPK` matches nothing — see §4), `-o FLASH_ATTN_EXT`, `-o GATED_DELTA_NET`, and
   `test-logits-width-probe` on gemma-12B f16 KV.
5. **Then S2** (G5 + G4 A/B), unless the maintainer redirects to S5 (the `mmb` prize) first.

**Do not** default `mmb` on, do not edit the delivery patches from this branch, and do not push
`~/llama-wip-gfx1100` anywhere.
