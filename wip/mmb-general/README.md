# WIP: generalising `mmb` (bf16-WMMA dequant weight GEMM) beyond IQ4_NL

**Status: ACTIVE (opened 2026-09-19).  Not part of the delivery.**  Code lives in the
`~/llama-wip-mmb` worktree (branch `wip-mmb-general`, based on the `~/llama.cpp`
delivery tree at `8a2567e1e`); nothing here is in `patches/`.

> **New session?  Read [`HANDOVER.md`](HANDOVER.md) first** — its "FOR THE NEXT SESSION" brief at the
top is the self-contained handoff (environment, build/run, gates, the prioritized remaining work).
> This file is the running (dated) record.

## Why

On Strix Halo (gfx1151) our prefill is ~2x behind the tuned RDNA3_5 stacks
(pwilkin `strix-halo` full env 1349/1403 t/s on uniform IQ4_NL; halogen-flash
1246/1424 on the same GGUF files) because our weight GEMMs run the integer/vector
MMQ path while both use bf16 WMMA tensor cores.  The parked port
(`archive/work/wip-archive/iq4nl-prefill/mmb-port.patch`, +18.4 %) was IQ4_NL-only,
so it did little for the delivery's own models.  This picks it back up in a
**general-purpose** form.

**GFX1201 note (why it was shelved):** the `mmb` kernels use the **first-gen gfx11**
WMMA builtin `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.  gfx12 needs
`..._bf16_w32_gfx12` (see `gated_delta_net_chunked_bf16_gfx11.cu` vs
`gated_delta_net_chunked_bf16.cu` in the delivery).  So this is RDNA3-gated by
construction; gfx1201/gfx1100 keep the existing MMQ/QSA path.

> **Re-examined 2026-09-21 (`gfx1201-porting.md` §2):** the builtin genuinely differs, but the RDNA4
> fragment layout is already probed/validated in-repo, so the RDNA4 port is a bounded
> *fragment-layout* port, not new algorithm.  **Status:** `qsa3` (group 2) was ported to RDNA4 in
> session 28 (+7.6..+11.5 % prefill — `gfx1201-s4-qsa3-results.md`); **`mmb` (group 1) is still
> gfx11-only** and is the remaining scoped follow-up (`gfx1201-porting.md` §6.5).

## Attribution — what is pwilkin's and what is ours (added 2026-09-20)

Compared against the reference `~/pwilkin-llama-cpp` @ `f5daaa3cf` (branch `strix-halo`) by direct
file inspection.  This matters because the two trees overlap heavily, and it is easy to re-derive his
work and think it is new (or vice versa).  What is verifiable is **presence/absence in his tree**;
who first had an idea is inherently fuzzier, and there may be history not visible in a file diff.

**His (we ported/integrated it):**

* the **MMB core** (dequant→bf16 WMMA weight GEMM) — the original parked port was explicitly a port of
  his work (`archive/work/wip-archive/iq4nl-prefill/mmb-port.patch`);
* **native-bf16 HC intermediates + producer marking** — `out_xn_bf16`, `store_xn_f32`,
  `ggml_cuda_mmb_mark_bf16_only(...)` for `xn`/`block_out`/`residual`/hc-mix-dst/gate/MoE-`ex`/`glu`,
  and `hc_mix_reduce_bf16`.  **Session 12's `xn` BF16-only is his idea**, applied to the delivery's own
  op (our `dsv4_hc_pre_f32` is a different kernel from his — ours does the sigmoid inline, his does
  not); the bf16-producer port (sessions 8-12) is a port, not an invention;
* `dsv4-hc.cu`, `hc-cn.cu`, `hc-mix.cu`, **`qsa3`** (`qsa.cu`), **`mmb_tall`** (384x64/384x32),
  `mmb_f32split_kernel`.

**Ours (absent from his tree):**

| | pwilkin @ f5daaa3cf | this WIP |
|---|---|---|
| **non-temporal cache hints** | **zero** — `grep nontemporal` = 0, no `slc`/`glc` anywhere | session 13: `dsv4_hc_pre` 605→493 us/call (−18.6 %) in situ |
| dense MMB weight types | IQ4_NL, Q8_0, Q6_K **only with a bf16 shadow** | + Q4_K, Q5_K, Q5_1, Q6_K on-the-fly, IQ4_XS, IQ3_S, Q3_K, IQ3_XXS |
| routed/GLU types | **IQ4_NL only** (`src0->type != GGML_TYPE_IQ4_NL → false`) | the same 8 types, fused GLU |
| tiny-M warp-per-token F32 (`hc_*_inject`) | no such kernel | `mmb_tiny_m_f32_kernel` |
| shape-aware F32 split | `mmb_f32split()` defaults **0** | default **1**, shape-gated |
| qsa3 itself | kernel exists | our optimizations: bitmap counting sort, fused pack, all KV types |

**Why the distinction is not academic:** his MMB is **IQ4_NL-only**, and the delivery's models are not
(Flash-Next is UD-**IQ4_XS**, the MoE iteration model is **Q4_K_M**), so his dense MMB does not fire on
them at all — the type generalization is what makes the headline gains land on the delivery's own
models.  Conversely, the non-temporal class (in-situ L2/MALL pollution) is something his tree does not
do anywhere, which is exactly why sessions 5e/6 — which were partly reasoning against his design —
missed it.

## Done (2026-09-19)

- `mmb_dq_row_q4k` / `mmb_dq_row_q5_1` — on-the-fly bf16 LDS dequant, no bf16 shadow
  (a 120 GiB model cannot afford one for its experts).
- `WTYPE 3` (Q4_K) and `WTYPE 4` (Q5_1) in `mmb_tile_gemm`, `mmb_tile_gemm_glu`,
  `mmb_dense_kernel`, `mmb_routed_kernel` and `mmb_routed_glu_kernel`.
- `ggml_cuda_mmb_supported_mm` / `_mmid` / `_glu` accept Q4_K/Q5_1 (K%256 guard for Q4_K).
- **Gate: RDNA3_5 only by default** (`GGML_CUDA_CC_IS_RDNA3_5`).  RDNA3_0 shares the gfx11 WMMA
  builtin but is untested, so it takes `GGML_CUDA_MMB_RDNA3=1` to open.  gfx12 is excluded (needs the
  `_gfx12` builtin).
- **Multi-arch build safe**: the WMMA calls go through `mmb_wmma_bf16`/`mmb_wmma_f16` wrappers;
  the RDNA4 branch is a deliberate no-op (runtime-gated off there), so `mmb.cu` compiles for gfx1201
  (verified on the recorded compile command) and gfx1151 is unchanged.
- Graph optimizer fusions (MoE pair, SWIGLU->mmq) now stand down only when MMB will
  actually take **that weight type** (`ggml_cuda_mmb_dense_will_take` /
  `_routed_will_take`) — resume-checklist item #5.

### Supported weight types (2026-09-19)

| type | dense | routed (expert) + GLU | note |
|---|:--:|:--:|---|
| IQ4_NL | ✓ | ✓ | the original port |
| Q8_0 | ✓ | ✓ | routed added; dense is the PLE table |
| Q4_K | ✓ | ✓ | Q4_K_M experts |
| Q5_1 | ✓ | ✓ | |
| Q5_K | ✓ | ✓ | UD-Q5_K_M experts |
| Q6_K | ✓ | ✓ | on-the-fly now (no 6 GiB shadow needed) |
| IQ4_XS | ✓ | ✓ | MiniMax-M3 UD-IQ4_XS 35 %, UD-Q3_K_M down experts |
| IQ3_S | ✓ | ✓ | IQ4_XS experts |
| Q3_K | ✓ | ✓ | Q3_K_S/M/L all map to this tensor type |
| IQ3_XXS | ✓ | routed only | fused GLU is a net loss, so it is default-off (`GGML_CUDA_MMB_IQ3XXS=1`) |
| F16 / BF16 | — | — | already tensor-core via `mmf` WMMA (no MMB needed) |
| F32 | ✓ (f32split) | — | f16 hi/lo WMMA |

Types with **no representation in current ggml** (named in older llama.cpp READMEs, checked
2026-09-19): `IQ3_M`, `IQ3_XS`, `IQ4_S`, `IQ4_M`.  `Q2_K`/`IQ2_*`/`IQ1_*` are deliberately out of
scope (quality).

### Measured (gfx1151, ROCm 7.14, `-ub 2048` bf16 KV; PPL on `prompts/prose-rdna-boosts.txt`, `-c 2048`)

| model / test | MMB off | MMB on |
|---|---:|---:|
| **Q4_K_M pp2048** | 604.6 | **1015.2 (+68 %)** |
| **Q4_K_M pp8192** | 568.9 | **888.1 (+56 %)** |
| Q4_K_M PPL | 10.3328 | 10.2716 |
| IQ4_XS pp2048 | 723.3 | 838.0 (+15.9 %) |
| IQ4_XS pp8192 | 685.5 | 779.5 (+13.7 %) |
| IQ4_XS PPL | 10.6938 | 10.6440 |
| Gemma4-26B-A4B Q8_0 pp2048 | 2058.5 | 2340.1 (+13.7 %) |
| Gemma4-26B-A4B Q8_0 pp8192 | 1723.5 | 1928.5 (+11.9 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp2048 | 2045.3 | 2303.0 (+12.6 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp8192 | 1917.8 | 2141.3 (+11.7 %) |
| Qwen3.6-35B-A3B UD-Q5_K_M PPL | 14.4349 | 14.3907 |
| Qwen3.6-35B-A3B Q6_K pp2048 | 2128.8 | 2143.2 (+0.7 %) |
| Qwen3.6-35B-A3B Q6_K pp8192 | 1974.6 | 2012.3 (+1.9 %) |
| Qwen3.6-35B-A3B Q6_K PPL | 14.3682 | 14.3296 |
| Qwen3.6-35B-A3B UD-Q3_K_M pp2048 | 2091.9 | 2295.2 (+9.7 %) |
| Qwen3.6-35B-A3B UD-Q3_K_M pp8192 | 1941.6 | 2120.2 (+9.2 %) |
| Qwen3.6-35B-A3B UD-Q3_K_M PPL | 14.6517 | 14.6248 |

The big Q4_K_M jump is the MoE expert GLU + routed down on WTYPE 3; IQ4_XS is IQ3_S gate/up +
IQ4_NL down; the Q5_K_M and Gemma4 Q8_0 gains are their expert types.  Q6_K moves little because its
MMQ path is already efficient on this workload.  UD-Q3_K_M moves little because only its **down**
experts are IQ4_XS (the gate/up are **IQ3_XXS**, still unsupported).  PPL parity everywhere says the
dequants are correct.

**Model composition note:** the file names mislead.  *Qwen3.8-Flash-Next UD-IQ4_XS* is by bytes
IQ4_NL 52 % + IQ3_S 36 % + Q8_0 9.5 %, with the IQ4_XS *type* only 1 %.  *MiniMax-M3 UD-IQ4_XS* is
IQ3_S 56 % + IQ4_XS 35 % + Q8_0 + Q6_K — now fully covered.

### Post-MMB profile (Q4_K_M pp8192, total 17.75 s, was ~28 s)

`flash_attn_qsa` **2.94 s (16.6 %)** is now the largest kernel; then `mmb_dense_kernel` 4.06 s
(Q8_0 PLE + Q5_1), `mmb_routed_glu` 2.46 s, `mmb_routed` 1.39 s, HC pre+post 1.44 s,
`mmb_cvt` 0.65 s, `mmb_f32split` 0.65 s.  Our VEC QSA already uses `v_dot2_f32_f16`, so its gap is
algorithmic (per-token gather + VEC vs packed-block WMMA), not instruction selection.

## UPDATE — session 26 (2026-09-20): rebased onto delivery **r12**, and the WIP consolidated **38 -> 5** thematic patches

The delivery moved to r12 (`origin/main` `4e37fa6`).  Both rebases done and verified: the record branch
17/17 (two `AGENTS.md` conflicts, resolved by regenerating from r12's content plus this branch's
pointers) and the code branch **38/38 with no conflicts**, onto the r12 applied tree
`8a80535e556bef57666d2eaa4d3eb4cf93fb83f5` (rebuilt with `RDNA_BRANCH=r12-verify scripts/apply-all.sh`
on a fresh `ebbb18522` worktree).  All gates re-verified green.

Then the 38 WIP commits were gathered into **5 "like items"** — `mmb`, `qsa3`, the F32/tiny-M +
default-flips + probe, HC16 + non-temporal, and the indexer — because the other machines need a
manageable set.  The commits were interleaved (the `mmb` line is 1-9 *and* 33-37), so this is a
cherry-pick-and-squash; it is **content-preserving and verified**: the 5-patch result has tree
`d365b43ddc87c472c33a121247931269f975aa43`, byte-identical to the 38-commit tip, and `git am` 5/5 on a
fresh r12 tree.

**`GROUPS.md` is the triage sheet** — per group: what it is, its env/compile gate, its arch note
(including that `mmb_wmma_*` is a deliberate no-op on RDNA4, so group 1 needs care on gfx1201), the
gates, and the recommended order for a new architecture.  Read it before applying anything on
gfx1100/gfx1201.

---

## UPDATE — session 27 (2026-09-21): first **gfx1201 (RDNA4, 3× R9700)** porting session — the
arch-neutral groups validated, and the always-QSA flip gated per arch

The 5-patch set was applied and built on gfx1201 and the arch-neutral groups were measured.  Full
detail: **`gfx1201-s1s2-results.md`**; the multi-session plan: **`gfx1201-porting.md`**.

**Headline (Flash-Next IQ4_XS, q8_0 KV, 3-GPU `-sm tensor`, `-b/-ub 2048`, vs the r12 delivery):**
prefill **+3.2 % / +5.0 % / +6.6 %** at pp32768/65536/98304, no regression at any depth, same-seed
greedy and all op oracles bit-identical.

**Per group:**

* **G5 (indexer top-k) — win, grows with depth:** +2.1 % pp8192 → +3.1 % pp32768 → **+8.0 % pp98304**
  (all-groups − no-G5).  This is the bulk of the deep win.  Same-seed text and `INDEXER_TOPK` 2/2.
* **G4 (non-temporal) — small consistent win:** +0.3-0.4 % at 32k/64k/98k in interleaved A/B rounds
  (the maintainer's rule: any win without a loss → keep it).  **Keep.**
* **G3a (always-QSA flip) — large regression on gfx1201:** pp2048 **−18.9 %**, pp8192 **−6.7 %**,
  because qsa3 is still gfx11-gated so the VEC QSA path serves the short-prefill band.  The flip is
  now **arch-gated: always-QSA only on gfx1151**, dense shortcut everywhere else (folded into patch 3).
* **G2 (qsa3) / G1 (mmb) — still gfx11-gated.**  The RDNA4 fragment-layout port is scoped (with the
  exact gfx11-vs-gfx12 lane mapping) in `gfx1201-porting.md` §§6.4-6.5, but not done.  The prior
  "RDNA4 is new work, not a port" note was re-examined: it is a bounded fragment port, the layout is
  already validated in-repo (`gated_delta_net_chunked_bf16.cu`).

**Patch set regenerated:** the G3a gate is folded into patch 3; the 5 patches apply `git am` 5/5 on a
fresh r12 tree and the applied tree is `c0f8ea75ba4037844affc396779e5307b2aafbfa` (byte-identical to
the tested tip).  `commits.txt` refreshed.  The set is now suitable for **both gfx1151 and gfx1201**
for the arch-neutral items; gfx1100 is next (see the gfx1100 job in `GROUPS.md`).

---

## UPDATE — session 28 (2026-09-21): **`qsa3` ported to RDNA4** — the second-biggest gfx1201 win

The packed-block F16 WMMA sparse-attention path (`fattn-qsa3.cu`, group 2) was ported from RDNA3_5 to
gfx12 and is the S4 milestone of `gfx1201-porting.md`.  Full data: **`gfx1201-s4-qsa3-results.md`**.

**The port** is one arch-selected shim at the top of the file plus 7 call sites: gfx12 stores 8
halfs/lane in the "two runs of four" layout (`k = 4*hi + {0..3}` and `4*hi + 8 + {0..3}`) and uses
the `_f16_w32_gfx12` builtin with the accumulator `m = 8*hi + e`; gfx11 keeps the 16-half full row and
`m = 2*e + hi`.  The gfx11 arm is a compile-time `#if`, so **gfx1151 is byte-identical**.  The
support predicate gained `RDNA4`; the `ne[1] >= 128` prefill gate is unchanged, so the decode/verify
band and its width purity are untouched by construction.  The two non-obvious parts: the value A
operand's two runs come from key-blocks `hi` and `hi+2` (not a contiguous row), and the probability
packing must still produce keys **0..15 in order** for the PV A operand.

**The kernel had no unit oracle before this** — on any arch.  `test-backend-ops -o FLASH_ATTN_QSA`
never attached `src[7]/src[8]`, and the qsa3 predicate requires both, so every run only ever
exercised the VEC kernel (`fattn-qsa.cu`).  Patch 2 now adds three packed cases (f16 ×2, bf16, at
`hsk=256`/`gqa=12`) plus the same shape on the VEC path as a baseline: **26/26 on gfx1201** (was 22).
With the tolerance forced to 0 to read the errors, qsa3 is ~4.2e-8 NMSE vs the same-shape VEC ~4e-9 —
f16-rounding level, not a layout error (a fragment-layout mistake is an exact permutation error, ~0.5×
contractions).

**Performance** (same build, only `LLAMA_QSA3_ENABLE` 0 vs 1, Flash-Next IQ4_XS, q8_0 KV, 3-GPU
tensor): **+7.6 % pp4096, +11.5 % pp16384, +10.4 % pp32768**.  That is the largest gfx1201 prefill win
after the G5 indexer.  The G3a arch gate was re-tested with qsa3 active — the dense shortcut is still
equal-or-better at every point, so **the gate stays**.

**Correctness note (important):** qsa3 changes the greedy text vs the VEC kernel at >2051-token
contexts.  That is *not* a port artefact — it is the documented, approved qsa3 prefill re-baseline
(`README.md` records `MMB=1 QSA3=1` → `bbd4bcb519e4` vs `MMB=0 QSA3=0` → `5120b28f2879`: "Hashes differ
between configs — that is the approved prefill re-baseline").  The S4 session also closed the missing
long-context text gate from S1/S2: with qsa3 off the WIP is **byte-identical to the delivery** on the
5246-token prose prompt, so G5/G4 are text-pure and the qsa3 delta is attributed exactly.

**Patch set regenerated:** the port and the new test coverage are folded into patch 2; the 5 patches
apply `git am` 5/5 on a fresh r12 tree and the applied tree is
`e9aa886ac06270f05a558be270069d7e7211a424`.  `commits.txt` refreshed; the local branch is
`mmb-port-qsa3`.  Next: **S5–S7 (G1 `mmb`)** — the remaining WMMA group.

---

## UPDATE — session 29 (2026-09-21): **`mmb` ported to RDNA4 — and the `GGML_CUDA_MMB` switch split**

S5–S7 of `gfx1201-porting.md` (group 1).  The gfx12 bf16 fragment shim is in and the gfx11 path is
verified **bit-identical** — but the finding that matters is that `mmb` is **not** a blanket win on
RDNA4, so the one switch became three axes.  Full matrix: **`gfx1201-s5s7-mmb-results.md`**.

**The port (S5).**  One arch-selected shim, 7 sites: gfx12 needs `..._bf16_w32_gfx12` and the 8-half
"two runs of four" fragment with the `m = 8*hi + e` accumulator; gfx11 keeps the 16-half row and
`m = 2*e + hi`.  Verified by compiling `mmb.cu` for **gfx1151** and diffing the device assembly:
**byte-identical except the `__hip_cuid_*` module-id symbol**, with every `codeLenInByte` and the
opcode histogram md5 equal.  **The trap:** routing the gfx11 load through a `__device__` helper left a
dead `lane >> 4` argument and perturbed the scheduler (semantics-preserving but not bit-identical);
the gfx11 arm has to be a **macro** so the preprocessed source is the original expression.

**Correctness (S6):** PPL parity — 14.3981 (off) vs 14.4087 (on) on the fast model, and 12.7378 vs
12.7221 over 7 chunks on the MoE IQ path (MMB demonstrably running via `LLAMA_MMB_CVT_LOG`).

**The finding (S7).**  With every weight type and the dense path enabled, MMB is a **−4…−13 %**
prefill regression on three models (dense Q8_0 −6 %, dense mixed-K −11 %, dense IQ-heavy −3 %), while
the wins are confined to the **routed MoE** path (+3.2 % on an 88.5 %-IQ MoE) and the **qwen4exp HC**
paths (+2.7…+11.8 %).  The generic quantized dense tile GEMM loses on RDNA4 for *every* type measured;
the F32 split router was the residual −1 % left on a K-quant model once the weights were already
type-excluded.  A 27B UD-IQ3_S (82 % IQ, **dense**) losing −3 % is what proves it is **not** simply "IQ
weights win" — the calling path matters at least as much as the type.

**The split (the deliverable).**  `GGML_CUDA_MMB` bundled **arch × weight-type × kernel-path ×
model-family**, and the type list was duplicated, arch-independent, in **five** places (drifted —
IQ3_XXS was treated differently in different copies).  It is now:

* **`mmb_wtype_ok()` / `mmb_wtype_mask()`** — one arch-scoped weight-type policy replacing all five
  copies.  RDNA4 = **IQ family** (`IQ4_NL`/`IQ3_S`/`IQ4_XS`/`IQ3_XXS`); RDNA3_5/RDNA3_0 keep the full
  set, so **gfx1151 is unchanged**.  Override: **`GGML_CUDA_MMB_TYPES=<csv>`**.
* **`mmb_dense_flag()`** — separates the generic quantized dense tile GEMM + the F32 router (the
  losers) from the routed MoE path and the qwen4exp HC paths (tall-M / tiny-M inject / gate-mix — the
  winners).  RDNA4 default: dense/router **off**; **`GGML_CUDA_MMB_DENSE=0|1`** overrides.
* `mmb_tall_shape()` keeps the qwen4exp HC tall-M tile out of the dense stand-down — without it the
  dense flag silently kills the HC win.

Because the graph's MMQ-fusion stand-down goes through these same predicates, an excluded type or
path costs **nothing** (it keeps the delivery's MMQ path).  Net with the scoped default: **neutral
wherever MMB would lose, +3…+7 % where it wins**, and a qwen4exp user can opt into the extra dense win
with `GGML_CUDA_MMB_DENSE=1`.  The open work is a **real gfx1201 dense tile geometry** — the current
one is gfx1151-tuned and is where every RDNA4 loss lives.

> **The remaining gfx1201 work is now scoped in `gfx1201-porting.md` §13 (S11-S15)** — read that
together with `HANDOVER.md` when picking the WIP up: S11 the arch-scoped `mmb_*` tuning constants,
S12 the routed/GLU tuning with kernel-time evidence (**and the routed-MoE default re-decision, see
session 30**), S13 the F32/HC16 paths, S14 the B1-B9 matrix (**MTP has never been run on gfx1201**),
S15 the freeze and hand-off.  S1-S10 are complete.

---

## UPDATE — session 30 (2026-09-21): **the RDNA4 dense tile geometry — the dense path is a per-TYPE win**

S10 of `gfx1201-porting.md` (the headline remaining win).  Full record:
**`gfx1201-s10-dense-geometry.md`**.

**The mechanism.**  `rocprofv3 --kernel-trace` carries `VGPR_Count`/`LDS_Block_Size` per launch, and
that settled it: the gfx1151-tuned dense tile needs **55296 B of LDS**, so exactly **one** workgroup
fits a CU (`sharedMemPerBlock` = 65536) — 8 warps, 2 per SIMD — while the delivery's MMQ uses **no
LDS at all** (it dequantizes into registers) and gets **3 blocks**.  On top of that, with `BM=128`
only **half** of the 256 threads dequantise the A (weight) panel (`A_ITEMS = ceil(BM/256)`), and that
dequant is serialised with the WMMA work.

**The fix.**  A **256x128** tile (WTM=64, WTN=64, TMxTN=4x4) keeps the same 55296 B but puts all 256
threads on the A dequant, and it makes the IQ3_S dense GEMM **beat the delivery's MMQ for the first
time**: 1.856 s vs 1.944 s = **−4.5 %** (27B UD-IQ3_S, pp4096, 1 GPU).  IQ4_XS still loses (+10.9 %)
and IQ3_XXS/IQ4_NL are break-even, and the whole-model interleaved A/B agrees exactly (IQ3_S-only
+1.0 %, the whole IQ family +0.04 % — IQ4_XS cancels the win).  **So the dense path is a per-weight
TYPE decision, not a per-arch one:** RDNA4 now enables it for **IQ3_S only**, via
`mmb_dense_tmask()`/`mmb_dense_type_ok()` (`GGML_CUDA_MMB_DENSE_TYPES=<csv>` is the A/B override;
`mmb_dense_flag()` still forces the whole path).  Landed as **patch 7** (`git am` 7/7, applied tree
`9ef573e5d0`), because patches 1/3/4/6 all touch `mmb.cu`.

**Result:** 27B UD-IQ3_S **+0.52 % pp8192 / +0.46 % pp32768** (interleaved r=5, two rounds agreeing
to 0.02 %); 27B Q8_0 exactly neutral; Flash-Next IQ4_XS +3.2 / +2.2 % (the S7 qwen4exp win
reproduces).  Same-seed greedy text is identical to the delivery **and to every valid geometry**, so
the geometry is numerics-neutral — but the sweep still needs a hash gate, because a geometry whose
declared `BN` is not `(8/(BM/WTM))*WTN` silently computes only part of the output and *looks* fast
(a `256x192/WTN48` arm measured a fake **−39 %**).  **gfx1151 is instruction-identical** (79/79
existing kernels, split on the `.size` delimiters; the 11 new 256x128 kernels are unreachable there).

**Correction to the S7 record (important).**  The `35B-A3B UD-Q3_K_M` **+6.7 % MoE win does not
reproduce**: the *unmodified S7 binary* now measures **−1.4 %** on this box.  The kernel breakdown
says why — the routed MMB (0.875 s) only **matches** the delivery's `mul_mat_q_routed_compact` +
`mul_mat_q<IQ3_XXS,64>` (0.880 s), and standing that fused kernel down costs an extra
`mm_ids_helper` launch (+0.065 s).  So `GGML_CUDA_MMB=1` on RDNA4 is currently a clear win only for
**qwen4exp (HC/QSA)** models and **IQ3_S-heavy dense** models; **the routed MoE default must be
re-decided in S12** (along with the `mm_ids_helper` overhead).  Also still open: the
`mmb_cvt_f32_bf16` activation conversion is a ~1 % tax whose 4-entry cache never hits
(`GGML_CUDA_MMB_CACHE` is the untested knob), and `DBUF` remains unwired (the shapes where it fits
are the small-BN ones where it does not pay — the code comment's −6.7 % is still unverified on RDNA4).

---

## UPDATE — session 25 (2026-09-20): the indexer gather's **warp-shuffle scan** (−11.7 % on the gather),
## and the block-level gather/emit closed as unbuildable

Tip `aa55dfef8` (38th commit).  Full record: HANDOVER "session 25".

### The block-level gather/emit is DEAD — and that closes the indexer line

This was the handover's "one real remaining optimisation".  `llama-memory-hybrid-idx.cpp:743`:

```c
cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;   // idx = ranked ? rank[j] : cells.pos_get(j)
```

* a block's cells are in **`idx % r` slot order**, not column order (the ranked path would need a
  per-block sort);
* and the write is inside `if (blk_of[j] >= 0)`, so **every unpooled cell — including the tail, which
  `blk_tail` makes always visible and therefore likely selected — is not in `blk_cells` at all.** A
  block walk therefore still needs a full `n_kv` scan to find them, which removes the entire point.

The ideal outcome was ~0.4 % of the run.  **Do not build it.**

### The useful negative: the block is not the cost, the barrier-heavy scan is

The handover's suggested intermediate — skip the per-cell key evaluation for blocks whose key cannot
reach the prefix — is provably output-preserving, so it was implemented and A/B'd:
**`259.9 -> 262.2 ms`, nothing.**  That says the gather is *not* memory-bound on `cell_pos`, which
pointed at what was: a 256-thread **Hillis-Steele** scan of (g_count, e_count) with 8 steps × 2
`__syncthreads` = **~19 barriers per 1024-cell tile**.

### What landed

Warp-shuffle inclusive scan + one 2-barrier offset pass.  Integer sums are associative, so the prefix
values — and therefore the ascending-column placement — are identical.

| pp32768, target model | before | after |
|---|---:|---:|
| `indexer_topk_write_blocks_grouped` (gather) | 259.9 ms | **229.6 ms (−11.7 %)** |
| indexer family | 1146.5 ms (1.86 %) | **1117.4 ms (1.80 %)** |

### The gate that matters here

The indexer selection width is 2051, so a prose prompt at `-c 8192` **never leaves the dense shortcut**
— the normal PPL/greedy/width gates cannot reach this path at all.  The gate that does is a
**long-context same-seed A/B** (16k-token prefill): `9565ffa670c3` with and without.  Plus
`test-backend-ops -o INDEXER_TOPK` / `-o FLASH_ATTN_QSA` (2/2 each), PPL 10.6015, greedy
`9c281c415082`, width PASS.

### Left, deliberately (each < 0.2 % of the run)

`histogram_blocks` is the biggest remaining indexer kernel (**462 ms = 41 % of the family**) and
recomputes the per-`(row,block)` key in each of its 3 passes, and re-sums `wvis` per range.  Caching the
key (or a compact per-block bin) from pass 1 and a per-`(row,hb)` visible-count total would remove
~2 loads + 6 ALU per block-visit.  Real, but neither is order-sensitive and neither is worth the
complexity today.

---

## UPDATE — session 24 (2026-09-20): the **`load_regs` field preload** — the GLU's dequant ran exposed.
## `routed_glu` **990 -> 841 ms (−15.1 %)**, total GPU kernel **4015 -> 3879 ms**, pp8192 **1116 t/s**

Tip `94251003e` (37th commit).  The session-23 follow-up.  Full record: HANDOVER "session 24".

### Sizing the prize first

The in-situ "remove the dequant" A/B is invalid (it changes the router -> the workload changes), so
instead **duplicate** the IQ3_S dequant — same values, idempotent, so the output *and* the routing are
unchanged — behind `asm volatile("" ::: "memory")` so the compiler cannot CSE it.  Without the barrier
the duplicate is silently folded away (3943 vs 4540 SASS instructions); with it:

| | `mrg` |
|---|---:|
| baseline | 990.4 ms |
| dequant duplicated (barrier-guarded) | **1338.5 ms** |

So the dequant's **marginal cost is 348 ms = 35 % of the GLU**, and the other **642 ms is
WMMA/LDS/barriers**.  The session-22 "~51 %" figure was an isolated-harness extrapolation; this is the
measured in-situ number, and it says the dequant is on the critical path but is *not* the majority.

### The fix

`store_lds_a` runs **exposed** between the WMMA phases (DBUF is off for this tile), so its
`d`/`sc`/`qs`/`qh`/`sg` global loads were issued *and waited on* inside that exposed phase.
`mmb_iq3s_preload<PARTS>` now fetches only this thread's own `il` groups (4/PARTS of them; the two `qs`
bytes a grid lookup needs are adjacent, so they come back as one `uint16`) into a small register struct
**inside `load_regs`** — which already overlaps the previous K-step's WMMA.  `mmb_dq_iq3s_r` then runs
purely out of registers.

| Flash-Next IQ4_XS | before | **after** |
|---|---:|---:|
| `routed_glu` (`mrg`) | 990.4 ms | **840.8 / 840.9** (two runs) |
| MMB family | 2592 | **2453** |
| **total GPU kernel** | 4015 | **3882 / 3879** (vs 4409 pre-session-21: **−12.0 %**) |
| pp8192 / pp2048 | 1089.1 / 1101.4 t/s | **1116.0 / 1128.3 (+2.5 %)** |

35B unchanged (the preload is on the IQ3_S path only).  Bit-identical: PPL **10.6015**, greedy
**`9c281c415082`** (624 chars), width probe **PASS**.

### Trap — the session-22 guard bug, walked into again

The first cut put the preload *inside* `load_regs`' `A_ITEMS` loop under
`if (row < BM && row < a_rows)` with `row = tid + i*MMB_NT`: only threads 0..63 ran it, and they all
took `tid/BM = 0`, so parts 1..3 stayed stale and greedy collapsed to **146 chars**.  The split row is
`tid % BM` for *every* thread, so the preload must be **hoisted out of that guard**.  It now sits above
the loop with a comment saying why.  **When you change the A row mapping, the guard and every consumer
of it must move together.**

### Negative results (measured — do not redo)

* **`MMB_LDS_STRIDE` 72 -> 64 is catastrophic** (`mrg` 990 -> 2538 ms; the dense Q8_0 485 -> 2078).  The
  +8 padding breaks LDS bank aliasing: a compact panel has a **128 B row stride**, which aliases every
  row onto the same banks for the row-parallel WMMA fragment reads.
* **The 1-block/CU occupancy ceiling is inherent.**  `rocprofv3`'s `LDS_Block_Size` — a column worth
  reading — shows `mmb_routed_glu<64,128>` and `mmb_dense<128,128>` at **36864 B**, i.e. **1 block/CU =
  8 warps** (registers would allow 3; LDS is binding at 64 KB/CU, and `<128,256>`/the tall tile are
  worse still).  Reaching 2 blocks/CU needs <= 32 KB, which needs stride 64 — the bullet above.

### Occupancy is NOT the limiter — the `BM=32` follow-up is refuted

What is left in the GLU is the **642 ms of WMMA/LDS/barrier time** (~36 % of the WMMA peak, session-6
figure).  The obvious suspect was occupancy: the tile is pinned at 1 block/CU by LDS, and halving `BM`
would halve the LDS while leaving dequant-per-output (`K/BN`) untouched — i.e. a free route to
2 blocks/CU.

**Measured and refuted.**  `hipOccupancyMaxActiveBlocksPerMultiprocessor` gives `glu_big=1,
glu_small=2`; so the small tile already *has* the 2 blocks/CU that halving `BM` would buy.  Forcing the
small tile 2 -> 1 blocks/CU with 12 KB of dynamic smem changed `mrg` by **nothing** (842.9 -> 838.7 ms,
inside run noise).  **Halving `BM` would gain ~0**, and `BM=32` would need `PARTS=8` with only 4 `il`
groups.  Do not build it.

*Technique note:* `rocprofv3`'s `LDS_Block_Size` reports **static** smem only — it read 23040 B both
padded and unpadded, which nearly produced a false conclusion.  Use the HIP occupancy API.

---

## UPDATE — session 23 (2026-09-20): the **small GLU tile re-dequantized the A panel 4x** — tile-class
## threshold 128 -> `BN_SMALL`, MMB family **2822 -> 2592 ms**, total GPU kernel **4254 -> 4015 ms
## (−8.9 % vs the pre-session-21 baseline)**, pp8192 **+1.5 %**, bit-identical

Tip `49eff7f18` (36th commit).  The session-22 follow-up.  Full record: HANDOVER "session 23".

### The find: profile per *shape*, not per name

The post-split profile named `mrg` (the routed GLU) as the #1 kernel, but the per-name view hides the
shape.  Splitting it out:

| kernel | ms | calls | share of the whole run |
|---|---:|---:|---:|
| `mmb_routed_glu_kernel<64, 32, 16, 16, 5>` | **903.1** | 94 | **21.3 %** |
| `mmb_routed_glu_kernel<64, 128, 32, 32, 5>` | 242.5 | 94 | 5.7 % |
| `mmb_routed_glu_kernel<64, 32, 16, 16, 8>` | 16.2 | 2 | 0.4 % |

Reading `mmb_routed_glu_kernel` settles what the two classes are: **`BM` tiles `M` (the expert's output
rows — the A panel / the weights), `BN` tiles the expert's token rows (the B panel)**.  `mmb_build_desc2`
sent every expert with `cnt < THRESH(128)` rows to the **BN=32** class.  So a 128-row expert took
`ceil(128/32) = 4` blocks — and **each of those 4 blocks re-dequantized the same 64-row A panel**.  Four
times the weight dequant for the same WMMA work and the same B-column padding.  The B-panel padding is
what the small class was *for*, and it is identical in both classes: `4 x 32 = 128` columns either way.

### The sweep

Flash-Next IQ4_XS, fixed prompt, min-of-2, the `mrg` (GLU) kernel:

| `THRESH` | 0 | 16 | **32** | 48 | 64 | 128 (old default) |
|---|---:|---:|---:|---:|---:|---:|
| `mrg` ms | 1078.4 | 1028.6 | **992.0** | 1014.6 | 1031.4 | 1161.4 |

The optimum is exactly `THRESH == BN_SMALL == 32` — which is also the analytic rule.  The small class
only pays while **one** block covers the expert (then it does 1/4 the WMMA for the *same* dequant); past
that the big class wins on dequant volume.  `THRESH=0` (all sizes on BN=128) is *worse* than 32, which
is the confirmation that the small class is still right for genuinely tiny experts.

The routed (non-GLU) tile has the same structure and the same fix: `mr` **395.9 -> 361.5 ms (−8.7 %)**.

### Landed

`mmb_glu_thresh()` / `mmb_routed_thresh()`, defaulting to `BN_SMALL` (32), each with an env A/B
override (`GGML_CUDA_MMB_GLU_THRESH` / `GGML_CUDA_MMB_ROUTED_THRESH`).

| | old | **new** |
|---|---:|---:|
| Flash-Next `mrg` (GLU) | 1161.4 | **990.4** |
| Flash-Next `mr` (routed) | 401.2 | **361.5** |
| Flash-Next MMB family | 2822 | **2592** |
| **Flash-Next total GPU kernel** | 4254 | **4015** (vs 4409 pre-session-21: **−8.9 %**) |
| 35B Q4_K: `mrg` / `mr` / family | 279.9 / 187.7 / 987 | **263.7 / 181.7 / 962** |
| pp8192 / pp2048 (same binary, env A/B) | 1073.1 / 1086.3 t/s | **1089.1 / 1101.4 (+1.5 %)** |

**Bit-identical, and it cannot be otherwise**: `BN` changes only *which* token rows a block covers —
never the K reduction order of any output element, and the A-panel dequant is the same rows in the same
`ksh` order.  Verified: PPL **10.6015**, greedy **`9c281c415082`** (624 chars), width probe **PASS**.

### Also measured this session — two more rejections (do not redo)

* **The `v_perm` pack for IQ3_S, re-tested after the split** (the register pressure and ILP had changed
  since session 22 rejected it): still loses, **`mrg` 1166 -> 1297 ms**.  The `mmb_pack2_so` shift/or form
  stays.  This is also the cleanest proof that **the kernel is latency-bound, not issue-bound** —
  `mmb_pack2_so` has *more* instructions than the `v_perm` form and is 11 % faster.
* **A mid BN=64 tile** (replace the BN=32 class with BN=64, `THRESH=64`): **990 -> 1048 ms**, worse.  The
  three-point tile ladder is not worth it; the two-class split at `THRESH == BN_SMALL` is the optimum.

### Read-through

Before reaching for a kernel rewrite, **profile per shape**.  This 21 %-of-the-run kernel was invisible
in the per-name profile (it is all just "mrg"), and the win was not in the dequant *body* at all — the
handover brief had been pointing at the body for two sessions, and the actual defect was a **tile-class
criterion**.  The body's remaining candidate (preload the fields in `load_regs`) is still open and ranked
first in the HANDOVER §F brief.

---

## UPDATE — session 22 (2026-09-20): the **bf16 RNE pack becomes one `v_perm_b32`** — the 35B MMB family
## **−2.3 %**, Flash-Next total GPU **−2.3 %** with session 21; IQ3_S dequant audited and re-derived

The session-21 follow-up (the "cheaper IQ3_S dequant").  The first thing to establish was **what had
actually been optimised**: see the audit below — IQ3_S was *not*.

### Dequant audit (the answer to "did we already do this?")

| WTYPE | type | shape of the helper | optimised? |
|---|---|---|---|
| 0 | IQ4_NL | `mmb_dq_row36` | **yes** — 16-entry LUT held in 4 registers, applied with `v_perm_b32` (the `(kv+128)`/`fma(...,-128*d)` trick keeps it bit-exact) |
| 8 | IQ4_XS | `mmb_dq_row_iq4xs` | **yes** — same register LUT (IQ4_XS shares the IQ4_NL codebook) |
| 1 | Q8_0 | `mmb_dq_row68` | arithmetic, no LUT |
| 3,4,6,7,9 | Q4_K, Q5_1, Q5_K, Q6_K, Q3_K | `mmb_dq_row_q4k`/`_q5_1`/`_q5k`/`_q6k`/`_q3k` | arithmetic (affine `fma(v,d,±m)`), no LUT |
| **5** | **IQ3_S** | `mmb_dq_row_iq3s` | **no** — 16 divergent `iq3s_grid[512]` **global** lookups, a direct port of upstream's `dequantize_iq3_s` |
| **10** | **IQ3_XXS** | `mmb_dq_row_iq3xxs` | **no** — same, `iq3xxs_grid[256]` + `ksigns_iq2xs` |

So the two i-quants with a 512/256-entry grid were added as naive ports (the register-LUT trick cannot
apply at that size — it needs 4 registers, i.e. 16 entries), while the two 16-entry codebooks got the
register treatment.  Upstream (`dequantize.cuh`, `vecdotq.cuh`, `mmq-load-tiles.cuh`) uses the same
global grid for IQ3_S, so this is upstream-wide, not an MMB regression.

### How much does it cost?  (a standalone harness, because the in-situ A/B is confounded)

`roofline`: the Flash-Next GLU dequantizes `20490 tiles x 64 rows x 40 ks x 2 panels x 64 values =
6.71e9` values per call.  A standalone `mmb_dq_row_iq3s` harness (`/tmp/dqbench`, the helper copied
verbatim + `iq3s_grid.h` extracted from `ggml-common.h`) measures **1.36 Tval/s**, i.e. **~5.0 ms of
the 9.81 ms** GLU call — **the dequant is ~51 % of the GLU kernel**, matching the session-21 ISA count
(~3800 VALU vs 32 WMMA per K-step).  The harness is also the bit-identity oracle (it catches `+0/-0`).

### What landed: one-instruction bf16 RNE pack

`v_cvt_pk_bf16_f32` does not exist on gfx11 and the compiler's `__bf16` **truncates** (10/4096 values
differ from RNE), so the pack was `mmb_f2bf(a) | (mmb_f2bf(b) << 16)` = round each word, `>>16`, then
shift/or.  It is now:

```
ua += 0x7fff + ((ua>>16)&1);  ub += 0x7fff + ((ub>>16)&1);
return __builtin_amdgcn_perm(ua, ub, 0x03020706u);
```

`v_perm_b32` bytes 0..3 come from its **second** operand and 4..7 from its first, so `0x03020706` picks
`{ua.b2, ua.b3, ub.b2, ub.b3}` = `{bf16(a), bf16(b)}` — one instruction replacing the per-value `>>16`
plus the shift/or.  Bit-identity was proven on 1M random float pairs (0 diffs) and on `d == 0` weights
(where `+0/-0` would otherwise differ), then by the gates.

Isolated IQ3_S dequant body: **765 → 616 SASS instructions**, **1364 → 1701 Gval/s**.

| workload | metric | before | after |
|---|---|---:|---:|
| 35B Q4_K_M (fixed prompt, min-of-3) | MMB family | 1008 ms | **985 (−2.3 %)** |
| | `mr` (Q5_K/Q6_K routed) | 203.6 | **187.4 (−7.9 %)** |
| | `mrg` (Q4_K GLU) | 287.4 | **278.5 (−3.0 %)** |
| Flash-Next IQ4_XS | Q6_K dense | 169.2 | **156.4 (−7.6 %)** |
| | Q8_0 dense / IQ4_NL routed | 491.3 / 387.5 | 488.7 / 384.6 (−0.5 / −0.8 %) |
| | **total GPU kernel (with session 21's DBUF)** | 4409 (no DBUF, no perm) | **4309 (−2.3 %)** |

**Plus the A-panel split (part 2 below): `mrg` 1318 → 1166 ms, total GPU kernel 4409 → 4254 ms
(−3.4 %), bit-identical.**

### Part 2: the A-panel dequant was running on 2 of 8 warps — now split across all of them

A structural find while looking for what was left: `A_ITEMS = ceil(BM/256)` and the A row is
`tid + i*256`, so with the GLU's `BM = 64` **only threads 0..63 (warps 0-1 of 8) ran the A dequant**
(and the dense `BM = 128` tiles used 128/256).  Isolated measurement of the consequence: **2 active
warps reach 348 Gval/s vs 691 for 8** — i.e. 50 %, not 25 %, so the CU had spare issue capacity and
the split was worth up to ~2x on the dequant.

`mmb_dq_row_iq3s_p<PARTS>` has the `PARTS = MMB_NT/BM` threads sharing a row take the `il` groups
`{p, p+PARTS, ...}` for both `g`, writing exactly the same LDS dwords (disjoint offsets, no race).
The row mapping in `store_lds_a` becomes `SPLIT_A ? tid % BM : tid + i*MMB_NT` so the threads that
would have skipped (`row >= BM`) now participate.

| Flash-Next IQ4_XS | baseline | session 21 `DBUF` | **`DBUF` + split** | **split, no `DBUF`** |
|---|---:|---:|---:|---:|
| `mrg` (IQ3_S GLU) | 1318.2 ms | 1224.5 | 1246.3 | **1166.4 / 1165.4** |
| total GPU kernel | 4409 | 4330 | — | **4269 / 4254** |

**The split SUPERSEDES `DBUF` for this tile.**  With the split the dequant no longer leaves warps idle
for the WMMA to run ahead into, so `DBUF` becomes a regression (+80 ms) — it is switched off for the
IQ3_S GLU (the plumbing stays, guarded and disabled).  `mrg` **−11.6 %**, total GPU kernel **−3.4 %**
vs the pre-session-21 baseline; the 35B is unaffected (`SPLIT_A` is false there, family 987 vs 985 ms).

**Trap (cost a full debug cycle):** the first cut put the split *inside* the existing `if (row < BM)`
guard, where `row = tid` — so only `tid < 64` ran and they all took `part = tid/64 = 0`; the other
three `il` groups were never written and greedy output collapsed to 146 chars.  **Any change to the
row mapping must move with its guard.**  The gates caught it; the harness did not (it had its own
flawed oracle).

**The IQ3_S dequant itself does not take the perm pack** — it *regresses* there (1318 → 1355 ms without
DBUF, 1225 → 1258 with; a clean 2x2 was run).  The LUT-latency-bound path apparently wants the shift/or
form, which can start as soon as its own two words are rounded rather than waiting on all eight.  So
`mmb_pack2_so` (the shift/or form) is kept and used by `mmb_dq_row_iq3s` / `_iq3xxs` only.

### Rejected this session (measured)

* **An LDS-resident `iq3s_grid`** (2 KB `__shared__`, loaded once per block, 32-bit `ds_load` instead of a
  64-bit-addressed global load): **+0.6 %** on the isolated dequant — i.e. nothing.  The ablation that
  *does* move the needle is "grid const (keep the index math)" (+10 %) and "no index math at all"
  (+36 %), so the cost is the **index/address computation (~26 %)**, not the LUT load.
* **Sign-by-XOR** (flip the float sign bit instead of `? -g1 : g1`): **NOT bit-identical.**  With a zero
  product the current ternary yields `+0.0` while XOR yields `-0.0`, and `mmb_f2bf(-0.0) = 0x8000` ≠
  `mmb_f2bf(+0.0) = 0x0000`.  (The harness caught this on explicitly zero-`d` rows; it would have been a
  silent weight corruption.)

### Next (ranked)

1. **The A-panel dequant only uses 64 of 256 threads.**  For every MMB tile `A_ITEMS = ceil(BM/256)` and
   the row is `tid + i*256`, so with `BM = 64` (both GLU tiles) only threads 0..63 dequantize while
   192 idle; the dense `BM = 128` tiles use 128/256.  Splitting each 64-value row across the 8 `il`
   groups (`PARTS = 256/BM` threads per row) would put all 8 warps on the dequant.  This is the largest
   identified structural item left — but with `DBUF` on, the idle warps already run ahead into the WMMA
   (there is no barrier between `store_lds_a` and the WMMA in the DBUF path), so the gain is bounded by
   how much of the dequant is still exposed.  Measure before building.
2. A **fused gate+up dequant** sharing the grid-index/scale arithmetic between the two panels.
3. Index/address arithmetic: precompute the 8 index-high-bits per `qhg` once, or preload the block's
   `qs`/`qh`/`sg` fields as wide vectors in `load_regs` (as `WTYPE 0/1/3/4` already do).
4. Nothing else fits in LDS; the warp-specialised 2x ring does not.

### Gates

PPL c2048 **10.6015**, greedy **`9c281c415082`** (624 chars), width probe **PASS** (maxdiff 0) — all
bit-identical.  34th commit `16c0fe07c`; backup regenerated and `git am` 34/34 verified.

---

## UPDATE — session 21 (2026-09-20): the MMB restructure lands an **A-panel double-buffered GLU tile**
## (IQ3_S): the weight dequant overlaps the WMMA (**−6.7 % GLU / −1.8 % total GPU time**), bit-identically

The session-20 mandate.  The target model's MMB weight GEMMs are **61.8 % of pp8192** (9587 of
15508 ms), the single largest lever left, and the brief's premise was "the dequant is serialized with
the WMMA behind a barrier".  This session measured the geometry, confirmed the premise for the
dequant-**heavy** GLU tile, and landed a targeted restructure.

### What was measured first (recon)

* **LDS is 65536 B/CU** (`hipDeviceProp_t`), so the 36864 B dense/GLU tiles give **1 block/CU = 8 of 64
  waves** — very low occupancy.  Any restructure must fit in LDS to be worth anything.
* **The dequant share is type-dependent, and the naive "no-dequant" A/B is invalid**: removing the
  dequant changes the GEMM result, which changes the MoE router's expert selection, which changes the
  routed/GLU workload itself (the small GLU tile collapsed 1038 → 4 ms — a workload change, not a
  speedup).  Only output-preserving experiments can be trusted here (the fixed-token `llama-perplexity`
  profiler was essential; `llama-bench`'s random prefill tokens also make the routed kernels
  non-comparable run to run).
* **ISA analysis (static, no measurement confound)** of the fully-unrolled body: the GLU
  `routed_glu_kernel<64,128,32,32,5>` loop has **~3800 VALU vs 32 WMMA** per K-step; the dense Q8_0
  `<128,128,32,64,1>` has ~1550 VALU vs 32 WMMA.  The GLU is dequant-issue bound; Q8_0 is not.
* **No hardware bf16 pack on gfx1151** (`v_cvt_pk_bf16_f32` is gfx12-only) and the compiler's `__bf16`
  conversion is *not* RNE (10/4096 values differ from `mmb_f2bf`), so the software RNE pack must stay.
* **The warp-specialised producer/consumer + 2x LDS ring does NOT fit**: the big tiles are already
  36864 B, and 2x is 72 KB > 64 KB.  A-panel-only double-buffering *does* fit.

### What landed: `DBUF` (A-panel double buffering)

`mmb_tile_gemm` / `mmb_tile_gemm_glu` gain a `DBUF` template flag: the next K-step's weight dequant
writes the **other** A buffer (`As`/`Ag`/`Au` are 2x; `Bs` stays single), so `store_lds_a(ks+1)` is
issued *before* the WMMA of `ks` and the dequant overlaps it, instead of being serialized behind the
single-buffer LDS hazard.  The loop becomes
`load_regs(k+1) → store_lds_a(k+1)→buf1 → WMMA(k) reads buf0 → sync → store_lds_b(k+1) → sync`.

LDS fits for: dense `128x128` (55296), routed `128x128`/`128x32` (55296/41472), GLU `64x128`/`64x32`
(55296/41472).  It does **not** fit for dense `128x256` (73728) or the tall `384x*` tiles.

**Enabled for WTYPE 5 (IQ3_S) on the GLU small tile only** — that is the one measured win:

| kernel (Flash-Next IQ4_XS, fixed prompt, min-of-N rocprofv3) | base | `DBUF` |
|---|---:|---:|
| `routed_glu_kernel<64,32,16,16,5>` (IQ3_S, 96 c) | 1018.8 ms | **942.0 ms (−7.5 %)** |
| `mrg` total | 1318.2 | **1229 (−6.7 %)** |
| MMB family | 2980 | **2902 (−2.6 %)** |
| **total GPU kernel time** | 4409–4415 | **4330–4336 (−1.8 %)** |
| perplexity `seconds per pass` | 2.40 | **2.34–2.37** |

**It is deliberately type-gated**: `DBUF` *regresses* the other tiles, so it stays off there —

* dense `128x128` (Q8_0, the dequant is only ~5 %): **+37 %** (491 → 671 ms);
* routed (IQ4_NL, both tiles): **+16 %**; routed small only: **+10 %**;
* GLU `64x128` big tile: **+14 %** (280 → 319 ms);
* GLU small tile on the **35B Q4_K** model: **+5.8 %** (287 → 304 ms) — so Q4_K keeps `DBUF` off;
  the 35B is completely unaffected by this build (mrg 287.3 vs 287.4, family 1008 vs 1008).

**`DBUF2` (also double-buffer B, one sync per K-step instead of two) is REFUTED**: the GLU went
1225 → **1433 ms** (+17 %) — the extra LDS read/write traffic costs more than the removed barrier
buys.  The code is kept, guarded and disabled, so it can be re-tested cheaply.

### Gates (all green, bit-identical)

PPL c2048 **10.6015** (×3), greedy **`9c281c415082`** (624 chars), `test-logits-width-probe` **PASS**
(worst maxdiff 0).  The change reorders the dequant but produces the same bf16 LDS values, so the WMMA
inputs are identical.  The `DBUF=false` refactor alone (split `store_lds` into `_a`/`_b`, add the
template flags) was verified bit-identical before any flag was enabled.

### Where this leaves the MMB restructure

The remaining ideas, ranked by the evidence:

1. **Cheaper IQ3_S dequant instructions.**  It is dequant-issue bound (~3800 VALU/32 WMMA).  The
   biggest single cost in the generated code is the per-value bf16 RNE pack (`v_bfe`/`v_add3`/`v_mov_b16`/
   `v_and_or` ≈ 4 instr/value) and the 64-bit address math around each of the 16 `iq3s_grid` LUT lookups
   (~5 instr each).  The LUT is a plain `static const` device array, so each lookup is a divergent global
   load; moving it to LDS or a 32-bit-offset form is the most promising next step.
2. **A fused gate+up dequant** that shares the IQ3_S grid-index/scale arithmetic between the gate and up
   panels (they use the same indices, only the data differs) — would roughly halve the index ALU.  Needs
   a per-type dual dequant.
3. The warp-specialised / double-buffered pipeline beyond `DBUF` does not fit in LDS for the big tiles.

---

## UPDATE — session 20 (2026-09-20): the `rms_norm` register-cache experiment is **refuted** (a wash)

The largest non-MMB kernel, `rms_norm_f32<1024, true, false>` (the HC `xn` stream, 538 ms / 3.5 % at
pp8192), reads `x` twice.  The hypothesis: the second read is a DRAM re-read, so caching each thread's
`x` in registers across the block reduction should cut it.  Implemented (`int NX` template param; the
index must be compile-time or the array spills, so the loop is unrolled `for k in 0..NX` with a
`col < ncols` guard, and the launcher picks `NX = 4` only when `ncols <= block_size*NX`).

Measured: `<1024,true,false>` 538.4 → **531.7 ms** (−1.2 %, noise), `<256,true,false>` 278.9 →
**287.1 ms** (+3.0 %), combined **+0.2 %**.  **Reverted.**  The second `x` read is cache-resident, so
the kernel is not DRAM-bound; the `<256>` loss is the unrolled predicated loop always running `NX`
iterations vs the dynamic loop's `ceil(ncols/block_size)`.  Do not retry (a shape-exact 2/3/4 dispatch
could recover the loss, but the ceiling is ~1 % of one 3.5 % kernel).  The tree stays at `2da50418d`.

**Next session's mandate: the MMB restructure** (the MMB weight GEMMs are **61.8 % of pp8192** on the
target model).  A self-contained brief — code map, geometry/LDS budget, the 2-stage pipeline and where
the dequant stalls, the dispatch chain, the measured per-kernel profile, the closed levers, the
restructure candidates, the iteration loop and gates — is in **`HANDOVER.md` §F "MMB restructure"**.

---

## UPDATE — session 19 (2026-09-20): indexer histogram atomics per cell → per block (pass 1) and
## per thread → per warp (block passes) (**family −16 % 8K / −7 % 32K** vs s18), bit-identically

Follow-up on the session-18 block path.  Two integer re-associations:

1. `indexer_topk_histogram_pass1` was doing one `atomicAdd(&histogram[bin], 1)` **per cell**.  But every
   cell of a block shares the block key and every invisible cell shares `key_inf`, so a thread's `VEC`
   run bins into at most two bins.  It now accumulates the per-block visible count (and one per-thread
   pending counter for the invisible `key_inf` bin) and flushes with **one atomic per block** (plus one
   per thread for the invisible stream).
2. `indexer_topk_histogram_blocks` had every one of its 256 threads do `atomicAdd(&s_sumw, mysum)` on a
   **single shared address** — 256 serialized same-address atomics per CU per pass.  It now warp-reduces
   `mysum` and does **one atomic per warp** (8 instead of 256).

The integer bin counts/sums are unchanged — only the atomic traffic shrinks.

| kernel | pp8192 (s18) | pp32768 (s18) |
|---|---:|---:|
| pass 1 (cell-level) | **13.4** (23.7) | **266.0** (341.8) |
| passes 2-4 (block-level) | **19.8** (22.4) | **461.8** (470.4) |
| gather | 20.1 | 259.4 |
| select | 11.3 | 87.8 |
| hist_accum | 5.2 | 64.3 |
| scan+init | 2.0 | 8.0 |
| **family** | **71.7** (85.6) | **1146.6** (1235.7) |

**Pass 1 −44 % / −22 %; block pass −11 % / −2 %; family −16.2 % / −7.2 %.**  Both wins are larger at 8K
because there are fewer hist-blocks per row there.  Bit-identical: PPL c2048 **10.6015**, greedy
**`9c281c415082`** (624 chars), width probe PASS (maxdiff 0).  The pass-1 kernel is still cell-level: it
must see every cell to count the per-block visibility.

**Next:** the block-level gather/emit is unchanged and still the open item (needs each block's cells in
column order; `blk_cells` slot order is `idx%r`, and the dead/spare block is not in `blk_cells`) →
estimated ~259 → ~60 ms at 32K.  The remaining cost centres are the three block passes (461.8 ms,
key-bound, now shown to be memory/latency-bound rather than reduction-bound) and `select`+`hist_accum`
(152 ms, memory-bound on the histogram array).

---

## UPDATE — session 18 (2026-09-20): indexer block-level histogram path (**−52 % pp8192 / −51 % pp32768**
## vs s15), bit-identically; `GGML_OP_INDEXER_TOPK` gains a `blk_cells` src

Implements the session-17 scoping.  The op now carries `blk_cells` as `src[7]`, so it derives
`r = blk_cells->ne[0]/n_blocks` and uses a **block-aligned** partition: pass 1 is cell-level and counts
each block's visible cells (`wvis`); passes 2-4 walk blocks only, binning `wvis[b]` at the block key
(and the invisible remainder at -inf).  The gather stays the session-17 cell-level `_grouped` kernel with
the block-aligned chunk, so the ascending-column order is unchanged.

| kernel | pp8192 (s17) | pp32768 (s17) |
|---|---:|---:|
| pass 1 (cell-level) | 24.0 | 341.8 |
| passes 2-4 (block-level) | 22.4 | 470.4 |
| gather | 20.4 (22.4) | 262.5 (291.3) |
| select | 11.6 (15.0) | 87.9 (100.1) |
| hist_accum | 5.3 (16.9) | 65.0 (103.1) |
| scan+init | 2.0 | 8.1 |
| **family** | **85.6 (123.7)** | **1235.7 (1362.3)** |

**−52 % / −51 % vs session 15 (2.94 → 1.47 % of the run at 32K).**  Bit-identical: PPL c2048 **10.6015**,
greedy **`9c281c415082`**, width probe PASS, `FLASH_ATTN_QSA` / `GATED_DELTA_NET` OK; and at ~16k context
(4x-concatenated prompt, `-c 32768 -n 16`) the block path and `LLAMA_INDEXER_NOBLOCK=1` agree
(`7d2e5b3e46dd`).  The block
histogram wins at short context (−31 % at 8K) but only −9 % at 32K, because the session-17 grouped pass
already shared the block key; the block passes save only the per-cell visibility/binning work.

**Delivery-facing:** the new `src[7] = blk_cells` must fold into block 14 at promotion.  A/B:
`LLAMA_INDEXER_NOBLOCK=1` (session-17 path), `LLAMA_INDEXER_NOGROUP=1` (cell-level path).  `wvis` is
`n_tps × n_kv/r` ints (~64 MB at 32K/ub2048, ~335 MB at 163840 context, pool-reused).

**Next:** the block-level gather/emit (needs each block's cells in column order; `blk_cells` slot order
is `idx%r`, and the dead/spare block is not in `blk_cells`) → estimated ~262 → ~60 ms at 32K.

## UPDATE — session 17 (2026-09-20): indexer block-key sharing (**−30 % pp8192 / −46 % pp32768** vs
## session 15), bit-identically; the full block-level selection needs an op change

Continuation of session 16.  On the default `additive == nullptr` path the value is **per-block**; the
session-16 kernels still evaluated it once per cell.  New `_grouped` histogram and gather kernels (VEC=4)
cache the block key across a thread's contiguous cells and scan the per-thread totals, so the gather's
ascending-column placement is unchanged.  `LLAMA_INDEXER_NOGROUP=1` forces the cell-level path.

| kernel | pp8192 (s15) | pp32768 (s15) |
|---|---:|---:|
| histogram (grouped) | 67.3 (83.5) | 859.6 (1235.7) |
| gather (grouped) | 22.4 (46.6) | 291.3 (668.3) |
| hist_accum | 16.9 (32.9) | 103.1 (513.2) |
| select | 15.0 (15.0) | 100.1 (97.6) |
| scan+init | 2.1 | 8.2 |
| **family** | **123.7 (177.1)** | **1362.3 (2523.6)** |

**−30 % / −46 % (2.94 → 1.62 % of the run at 32K)**.  Bit-identical: PPL c2048 **10.6015**, greedy
**`9c281c415082`**, width probe PASS, `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK.

**Next lever:** the full block-level selection (weighted radix over `n_blocks` + a block-level emit)
was scoped and needs the op to carry **`blk_cells`** (not currently a `src`) so the gather can enumerate
a block's cells; `r` is then `blk_cells->ne[0] / n_blocks`.  Ordering subtleties (ranked/mrope slot
order != column order; the dead/spare block has no `blk_cells` entries) make it more than a kernel swap
and it is an op-interface change on delivery block 14.  Estimated a further ~1.5–2x on the family.

## UPDATE — session 16 (2026-09-20): the indexer top-k loses its full-width count pass (**−9.4 % pp8192,
## −19.6 % pp32768** of the family), bit-identically; "compact after pass 1" is refuted

The indexer (`GGML_OP_INDEXER_TOPK`, delivery block 14) was the session-15 mandate.  Its gather used a
second full key-evaluation pass (`indexer_topk_count`) just to get the per-block `> prefix` /
`== prefix` counts.  Those counts are already latent in the radix histograms: a `key > final_prefix`
cell differs at some byte `p` and is larger there, so it lands in a bin above the selected one in
exactly pass `p`.  `indexer_topk_hist_accum` now folds each pass's suffix counts into `g_cnt`/`e_cnt`,
and `indexer_topk_write_blocks` (one CU per (row, contiguous range), shared running carry) replaces the
old count+scan+write.  `hist_accum` is O(`nrows x bpr x 256`) per pass -- **independent of `n_kv`** --
so the win grows with context.

| kernel | pp8192 (baseline) | pp32768 (baseline) |
|---|---:|---:|
| histogram (4 passes) | 84.5 (83.5) | 1250.3 (1235.7) |
| write | 41.9 (46.6) | 570.2 (668.3) |
| hist_accum (was count) | 17.0 (32.9) | 102.7 (513.2) |
| select | 14.9 (15.0) | 97.0 (97.6) |
| scan+init | 2.1 (2.1) | 8.2 (8.7) |
| **family** | **160.4 (177.1)** | **2028.4 (2523.6)** |

Bit-identical: PPL c2048 **10.6015**, greedy **`9c281c415082`**, width probe PASS, `FLASH_ATTN_QSA` /
`GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK.  Two traps: the write's running carry must be in **shared
memory** (a per-thread register gave `b4888d290fe2`), and `hist_accum` must be **coalesced** (one
256-thread block per (row, block) + warp reduce; the one-thread-per-block scan was 53 ms/call and made
the change a loss).

**Refuted: "compact after pass 1".**  Fully implemented and measured: **326 ms vs 177 ms** at pp8192.
The candidate set (cells whose top-8 key byte shares/beats the k-th value's) is **~50-60 %** of the
cache, not ~1/256, because the per-block relu scores are heavily tied.  See `HANDOVER.md` for the
numbers; the remaining lever is the radix histogram (60 % of the family) -- fewer passes or a
block-granularity selection.

## UPDATE — session 15 (2026-09-20): qsa3 becomes a compile-time gate; rocprofv3 silently drops
## env-gated paths (`rocprofiler-register` setenv race); indexer measured at long context

Opening the indexer work, the pp32768 profile appeared to show a long-context bug -- qsa3 attention
kernels absent, `flash_attn_qsa` at 16.9 % -- but it was a **profiler artifact**.  Full trail in
`HANDOVER.md`; the short version:

* non-profiled t/s proved qsa3 active at both 8K (1015 vs 860 with `GGML_CUDA_QSA3=0`) and 32K
  (960 vs 822), yet under `rocprofv3` the qsa3 kernels vanished;
* 15 early pp8192 profiles had taken qsa3, later ones did not, with no code change;
* hard-gating qsa3 at **compile time** (removing the env read) fixed it -- so the env read was the
  variable;
* root cause: `rocprofiler-register` in the ROCm 7.14 build (`librocprofiler-register.so.0.6.0`, Jul 9)
  calls `setenv()` with `GLOG_*` during early init, racing the app's `getenv()` and intermittently
  making an env gate read unset -- **ROCm issue #10196**, fixed upstream 2026-09-15 in **rocm-systems
  PR #11620**.

The qsa3 gate is now `LLAMA_QSA3_ENABLE` (compile-time, default 1; `-DLLAMA_QSA3_ENABLE=0` opts out).
Bit-identical (PPL 10.6015, greedy `9c281c415082`, all gates green).  **Rule: `rocprofv3` can silently
drop an env-gated path -- verify from kernel names and corroborate with a non-profiled t/s A/B.**

With the artifact gone, the **indexer** (the next session's mandate) measures clean: pp8192 **180.2 ms
(0.90 %)**, pp32768 **2523.6 ms (2.94 %)** -- histogram 1235.7, deterministic_write 668.3, count
513.2 ms -- at **1.88 ms/op @8K -> 6.57 ms/op @32K**, ~linear in context.  `GGML_OP_INDEXER_TOPK` is
delivery block 14 (not pwilkin's -- his tree uses the generic `top_k_nary_search_cuda`), so this is
our own op to optimise; the brief in `HANDOVER.md` has the shapes, the algorithm, the ranked
hypotheses and the gates.

## UPDATE — session 14 (2026-09-20): the non-temporal hint generalises — but **loads only**, and
## per-kernel (concat −29.8 ms, moe −39.8 ms; ssm rejected), bit-identical

Session 13 closed the `dsv4_hc` gap with non-temporal hints and warned the sweep of the other
streaming kernels must not be done blindly.  This session did it for the three session-5b/6 targets,
testing **load-only / store-only / both** separately (three profiles each, `rocprofv3` kernel time at
gfx1151 pp8192), and the answer is emphatically per-kernel:

| kernel | baseline | load-only | store-only | both | kept |
|---|---:|---:|---:|---:|---|
| `concat_transposed_src1_dim0` | 357.3 | **327.5** | 398.7 | 377.6 | **load-only (−29.8)** |
| `moe_weighted_reduction_f32_vec4` | 383.8 | **344.0** | ~416 (both−load) | 375.8 | **load-only (−39.8)** |
| `ssm_conv_long_token_f32` | 292.9 | (both−store) | 326.9* | 372.3 | **none** |
| `unary_gated_op_kernel` | 228.3 | **190.3** | — | — | **load-only (−38.0)** |
| `k_bin_bcast` (add/mul) | 244.0 | 296.8 | — | — | **none (load hurt)** |

\* not reproducible: `ssm` store-only measured 272.2 once and 326.9/327.0 twice — an unchanged kernel's
own baseline also swung 292.9 -> 313.6 between runs, so `ssm` is noise-dominated at this granularity
and is left alone.  The two kept kernels reproduce to <0.5 ms across runs (concat 327.9/327.9/327.5,
moe 344.0/343.6/344.0).

**The rule that falls out:** a non-temporal **store** evicts the output the next op is about to read
(concat's dst feeds `moe_weighted_reduction` immediately, moe's dst feeds the residual add) and
consistently loses; a non-temporal **load** helps when the input is streamed once and the L2/MALL is
polluted by the surrounding weight streams.  So apply the hint to the **loads of pure-streaming
kernels only**, and always A/B load vs store (a `both`-only test would have kept a regression here).

The moe kernel needed an `ext_vector_type(4)` view of `float4` for the hint — `__builtin_nontemporal_*`
rejects HIP's `float4` struct (it only accepts builtin scalars/vectors).  The same limitation bites the
**templated** kernels (`unary_gated_op_kernel`, `k_bin_bcast` are instantiated for `__half` too, and the
`__half` instantiations fail to compile).  The fix is `ggml_cuda_nt_load<T>()` in `common.cuh`: it takes
the hint for `float` via `if constexpr` and falls back to a normal load for every wrapper type, so a
templated kernel gets the hint on its f32 path only — which is the path these activations use.

`unary_gated_op_kernel` (the fused `sigmoid`/`silu+mul` that produces `attn_gated` and `final_output`)
wins with **load-only** too (−38.0 ms, two builds), but `k_bin_bcast` does **not** — a non-temporal
`src0` load cost **+52.8 ms** there (its broadcast `src1` operand / residual pattern evidently wants the
cache).  So the sweep stays empirical: same hint, opposite outcomes.

Bit-identical (value-preserving hints): PPL c2048 **10.6015**, greedy **`9c281c415082`** unchanged,
`FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK, width probe PASS, `plain == draft-mtp`
byte-identical.  Combined kernel saving ~70 ms of the 15.5 s profile (−0.45 %); the ALL-total is noisier
than the per-kernel times, so judge on the kernel trace.

**Still unmeasured (follow-up):** the qsa3 pack, the `rms_norm`/unary producers, and `ssm_conv` (no
hint found, but only three variants tested — a different block shape or a `split_n_t` change might
change the picture).  Note `rms_norm_f32` is **not** a candidate: it reads `x` twice (sum-of-squares,
then scale), so a non-temporal load would evict the row between the two passes.

## UPDATE — session 13 (2026-09-20): the `dsv4_hc_pre`/`_post` residual was L2/MALL pollution —
## non-temporal accesses close it (**−18.6 % pre, −31 ms post**), bit-identically

Session 5e left the `dsv4_hc_pre` residual as "~18 % kernel-local headroom" (a probe said the
pattern can do 232.6 GB/s, the kernel did 197) with "address arithmetic / strided dst write" as the
remaining suspects.  This session re-measured it in the bf16 era and found the real cause: **the
kernel is not slow, its buffers are polluting the cache the surrounding GEMM weight streams need**.

**The measurement that cracked it.**  A faithful standalone copy of the exact kernel
traffic/shape (n_embd=2560, hc=4, nt=2048; bf16 x + bf16 gate read, F32 dst + bf16 dst16 written)
runs at **0.434-0.535 ms** (215 GB/s), but the kernel in the model takes **0.605 ms** (190 GB/s).
The same trace shows `dsv4_hc_post` at 206 GB/s on the same machine, so the environment is not
bandwidth-limited.  Tiling is not it either: `vec4`/`vec8` (2D grid, `uint2`/`uint4` loads, proved
bit-identical) are both **worse in situ** (625 / 642 us vs 605).  What fixed it was making the bulk
accesses **non-temporal** (`__builtin_nontemporal_load`/`_store`, value-preserving hints):
`dsv4_hc_pre_f32` **605 -> 493 us/call (-18.6 %, 190 -> 234 GB/s)** and `dsv4_hc_post_f32`
**918 -> 880 us/call (691 -> 660 ms over 752 calls)**.  All-kernel total 15596 -> 15501 ms.

The hint is a no-op in the standalone microbench (there is no competing traffic there) — the same
reason session 5e/6 could not find it.  **Methodology rule: a microbenchmark that isolates a kernel
can miss a real in-situ win, because the win is about coexisting with the rest of the model.**

It is **bit-identical** (cache hints only): PPL c2048 stays **10.6015**, greedy
**`9c281c415082`**, `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT` OK, width probe PASS,
`plain == draft-mtp` byte-identical.  e2e is within run noise (pp2048 1041.6 -> 1047.2, pp8192
1015.6 -> 1017.9, same-session interleaved) — judge it on the kernel trace, per the standing rule.

**Two things ruled out on the way:** (1) `vec4`/`vec8` beat the scalar kernel standalone (~5 %) but
lose in situ, so the load-width/ILP line is closed for good; (2) `mixed` (the `dsv4_hc_pre` output)
is written as F32 **and** bf16 because `all_bf16_consumers` rejects it: its consumers include the GDN
`ssm_alpha/beta` `[2560x48]` F32 GEMM and the MoE router F32 GEMM (plus the expert GEMMs, which are
bf16).  Making it BF16-only would save ~21 MB/call but requires changing the GDN recurrence and MoE
routing input numerics — out of scope, documented as a follow-up.

**Broader follow-up (not done):** the same non-temporal treatment is untested on the other large
pure-streaming kernels (`concat_transposed_src1_dim0` 358 ms, `moe_weighted_reduction` 384 ms,
`ssm_conv_long_token_f32` 292 ms, the qsa3 pack).  It must **not** be applied blindly — kernels that
reuse data (the K/V cache in `qsa3_attn`, the weight panels in `mmb_*`) may lose.  Judge each on
`rocprofv3` kernel time.

## UPDATE — session 12 (2026-09-20): the HC normalized stream `xn` is now BF16-only — **+4.4 % pp2048 /
## +3.8 % pp8192** over session 11, a numerics change (PPL re-baselined)

Session 11's dead-F32-store skip stopped at `xn`: its two delayed consumers, `dsv4_hc_pre` src[0] and
the tiny-M `hc_*_inject` F32 GEMMs, still read F32, so the graph only set `bf16_copy` and the fused
`rms_norm+mul` kept writing the F32 output.  This session gives both consumers a BF16 arm and lets
`xn` be marked **BF16-only**:

* `dsv4-hc.cu`: `dsv4_hc_pre_f32` gains an `xbf16` template arm; when `xn` is BF16-only the launcher
  reads the bf16 slot instead of the (never written) F32 tensor.
* `mmb.cu`: `mmb_tiny_m_f32_kernel` gains an `XBF16` arm (the tiny-M launcher looks up the bf16 slot);
  `ggml_cuda_mmb_reads_bf16_act()` is the new predicate that says a dense GEMM reads the activation
  through the bf16 cache (all weight types, plus the tiny-M F32 kernel).
* `ggml-cuda.cu`: `all_bf16_consumers` now accepts a tiny-M F32 `MUL_MAT` and a `DSV4_HC_PRE`
  `src[0]` as bf16-aware consumers, so `xn` classifies BF16-only.

**A slot lifetime bug had to be fixed on the way (it cost a NaN PPL).**  Slot 0 is shared by every
generic activation copy, and `dsv4_hc_pre` *reads* `xn` from a slot while *writing* its own output to
slot 0; between the `xn` producer and its delayed consumers the intervening producers (e.g. `lo =
silu(scale(down))`) overwrite slot 0.  So `xn` now gets a **dedicated producer slot (4)**: the graph
optimizer assigns it when it sees the `DSV4_HC_PRE` consumer (`ggml_cuda_mmb_mark_bf16_slot`), and the
producers reserve through `ggml_cuda_mmb_reserve_auto` (dedicated slot if assigned, else 0).  The
first attempt (no dedicated slot) produced a coherent-looking but corrupt 492474 PPL — the huge
apparent speedup was degenerate MoE routing, exactly the session-5e trap: **always PPL before
believing a prefill gain**.

Gates: PPL c2048 **10.6015** (`HC16=0` 10.5771; session 11 was 10.6428 — all within the ±0.68 bar);
greedy `-f prompts/prose-rdna-boosts.txt -n 128 --seed 42 --temp 0 -c 8192` **`9c281c415082`** (624 ch,
reproducible; `HC16=0` gives `c3f24ac9c114`); `FLASH_ATTN_QSA` / `GATED_DELTA_NET` / `FLASH_ATTN_EXT`
OK; width probe PASS (maxdiff 0); `plain == draft-mtp` byte-identical (`c0f8fb2b6fc7`).

Same-session A/B (gfx1151 IQ4_XS Flash-Next, bf16 KV, `-b/-ub 2048`, `-r 2`, two interleaved reps):

| tip | pp2048 | pp8192 |
|---|---:|---:|
| `af2f70580` (session 11) | 999 | 977 |
| this session (`xn` BF16-only) | **1043** | **1014** |

The kernel trace attributes it (pp8192, `rocprofv3 --kernel-trace`, all-kernel total 16744 ->
15594 ms = **-6.9 %** for the whole `HC16` port): `mmb_cvt_f32_bf16` 648.8 -> 0.8 ms (this session's
last conversion is `ple_embd`), `dsv4_hc_pre_f32` 743.8 -> 460.9 ms (bf16 `x` + `gate`),
`mmb_tiny_m_f32_kernel` 576 -> 431 ms (bf16 `X`), `rms_norm_f32<1024,true,false>` 649.5 -> 569.5 ms
(the skipped F32 store), `unary_gated` 255.6 -> 227.6 ms.

## UPDATE — session 11 (2026-09-20): skip the dead F32 store in the producer port — **+2.3 % pp8192 /
## +3.3 % pp2048**, still bit-identical

Session 9's producers wrote F32 *and* the BF16 copy.  A profile showed that of the 650 ms of `mmb_cvt`
removed, only ~489 ms was net (the producers paid ~161 ms in extra BF16 stores).  But for activations
whose every consumer reads the BF16 cache, the F32 output is dead.  The graph optimizer now classifies
consumers: all-`bf16` (quantized `MUL_MAT`/`MUL_MAT_ID` through views) -> mark BF16-only and the
producer skips its F32 store; otherwise keep F32 + copy.  The five producer kernels gained a
`store_f32` flag.  Bit-identical (PPL 10.6428, greedy `9930c674a6ca`, width probe pure, FA/GDN pass).

| config | pp2048 | pp8192 |
|---|---:|---:|
| `HC16=0` | 975.3 | 952.4 |
| `HC16=1` (session 9, always-emit) | ~994 | ~966 |
| `HC16=1` (session 11, dead-store skip) | **1007.4** | **974.0** |

`unary_gated` 302 -> 229 ms.  `xn` still keeps its F32 (its `dsv4_hc_pre`-src0 and tiny-M consumers
read F32); adding BF16 arms there is the next ~1 % but is a numerics change on the hidden stream.

## UPDATE — session 10 (2026-09-20): `ssm_alpha/beta` (M=48) profiled — **rocBLAS stays**

The `ssm_alpha/beta` GEMMs are `[M=48, K=2560]` F32 and run on rocBLAS at **0.360 ms/call
(207.3 ms, 1.26 % of kernels)**.  Two faster-looking replacements lose: the BM=64 WMMA f32 tile is
0.312 ms but its f16-hi/lo split costs **+0.04 PPL** (alpha/beta gate the GDN recurrence), and an
exact-f32 SIMT tile is 0.437 ms (too low an FMA:LDS ratio).  `MMB_F32SPLIT_MIN_M=128` is unchanged;
both experiments reverted.  Full table in `HANDOVER.md` §session 10.

## UPDATE — session 9 (2026-09-20): the bf16-producer port is done — the whole `mmb_cvt` bucket is
## gone, bit-identically (+1.3 % pp8192 / +2.0 % pp2048); plus a **delivery** op-name bug

Session 8 did the HC gate and normalized stream.  This session generalised the mechanism: the graph
now marks the **activation of every MMB dense prefill GEMM** `bf16_copy`, and the fused
`rms_norm+mul`, fused `sigmoid/silu+mul`, fused `scale+unary`, generic unary and `dsv4_hc_pre` all
emit the copy into slot 0 alongside their F32 output.  The GEMM finds it in the slot and skips its
`mmb_cvt` pass.  All copies are RNE-rounded exactly as `mmb_cvt_f32_bf16` and the F32 outputs stay
valid, so everything is **bit-identical**: PPL c2048 stays 10.6428, greedy sha stays `9930c674a6ca`,
FLASH_ATTN_QSA/GATED_DELTA_NET/FLASH_ATTN_EXT pass, W=1..8 width probe pure.

| config | pp2048 | pp8192 |
|---|---:|---:|
| `HC16=0` | 977.1 | 951.6 |
| `HC16=1` | **996.7 (+2.0 %)** | **963.7 (+1.3 %)** |

`LLAMA_MMB_CVT_LOG=1` now shows only the `ple_embd` model-tensor conversion — the `hc_norm`,
`hc_mixed`, `final_output`, `attn_gated` and unary families are gone.

**Delivery bug found and fixed:** `GGML_OP_NAME` in `ggml/src/ggml.c` is missing `"INDEXER_FILL"`
(the enum has `GGML_OP_INDEXER_FILL` from delivery block 14; the base `8a2567e1e` confirms it).
`ggml_op_name()` is therefore shifted by one from there on (`UNARY` prints as `MAP_CUSTOM1`).  It is
cosmetic in the delivery but it mislabeled every `MMB_CVT` log; fixed by WIP commit `d1463bff3` and
flagged to move into the delivery.

## UPDATE — session 8 (2026-09-20): the bf16-producer port begins — HC gate + normalized stream (+1.2 %
## pp8192 / +2.2 % pp2048), behind `GGML_CUDA_MMB_HC16=1`

Session 7 scoped the bf16-producer port (`BF16-PRODUCER-PORT.md`); this session landed its first two
producers.  Tip `ccf28bc65`, two commits on top of the session-7 pack.

* **Mark lifetime**: `ggml_cuda_mmb_marks_clear()` is now called on the first optimize after a
  compute.
* **Gate**: the graph marks a gated `DSV4_HC_PRE`'s gate (dense `MUL_MAT [320 x 10240]`, MMB-taken,
  single consumer) BF16-only; MMB dense writes it into the pinned slot 1 (which already existed and
  was unused) and `dsv4_hc_pre` reads the BF16 copy through a new `wbf16` arm.  A real numerics
  change (PPL c2048 10.5771 -> 10.6428).
* **HC normalized stream (`xn`)**: the fused `rms_norm+mul` now also emits a BF16 copy into slot 0,
  RNE-rounded exactly as `mmb_cvt_f32_bf16`, so the MMB dense down projection skips its conversion
  pass.  Bit-identical (all 57 `hc_norm` `MMB_CVT`s disappear, PPL unchanged from the gate build).

| config | pp2048 | pp8192 | PPL c2048 |
|---|---:|---:|---:|
| `HC16=0` | 973.1 | 950.8 | 10.5771 |
| `HC16=1` gate+xn | **994.2** | **961.9** | 10.6428 |

**What remains:** the `hc_mixed` producer (`dsv4_hc_pre`'s own output — self-contained, but its
consumers must all read BF16) and the `final_output` / `MAP_CUSTOM1` families.  Same-seed greedy text
unchanged (`9930c674a6ca`).

## UPDATE — session 7 (2026-09-20): the qsa3 pack is 0.34 %, not 3 % — a misattribution corrected

Session-5b's next-work proposed fusing the qsa3 pack for ~3 % of prefill.  Measured by diffing QSA3
**on vs off** (`rocprofv3`, `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`, pp8192, bf16 KV,
`GGML_CUDA_MMB=1`), the **whole pack is 57.8 ms of 16782 ms = 0.34 %** (bf16 KV); with a q8_0 KV
cache, 86.3 ms = 0.48 %.  The session-5 "PACK/copy (qsa3 pack) 3.0 %" bucket was really
`concat_transposed_src1_dim0` (357.5 ms, the **MoE output concat**, present with QSA3 off too) +
`cpy_scalar<float,float>` (110.7 ms, base-graph copies) + the actual pack (57.8 ms).

**Implemented (same session):** the graph no longer builds `pk`/`pv`; it materialises only the natural
contiguous F16 view, and two new launcher kernels (`qsa3_pack_keys_kernel` / `qsa3_pack_values_kernel`)
do the whole re-layout in one pass each.  **Bit-identical** (PPL c2048 10.5771 both; greedy
`sha=04ddb94b1529` both), and the pack kernels go **50.5 -> 6.5 ms** (bf16, save **0.26 %** pp8192),
**86.4 -> 49.1 ms** (q8_0, save 0.22 %).  Full tables in `HANDOVER.md` §session 7.

**Then investigated the next targets (`dsv4_hc_pre`+`_post` bf16 intermediates 8.5 %, `mmb_cvt`
producer marking 3.8 %): they are the *same multi-session port*, not a cast.**  A plain `ggml_cast` is
a net loss — the cast *is* the existing `mmb_cvt` (12 B/elem vs 4 B/elem for a native-bf16 producer).
The reference gets bf16 for free from **fused producers** (`rms_norm`+`mul` with an `out_xn_bf16`
output; `ggml_cuda_mmb_mark_bf16_only` on MMB chains); our tree lacks them, `ggml_cuda_mmb_marks_clear`
is never called, and `LLAMA_MMB_CVT_LOG=1` shows the conversions are for `hc_norm` (5.24 M×2/layer),
`final_output` (3.15 M), `hc_mixed` (1.31 M×2) — all non-MMB producers.  `dsv4_hc_post` is already at
the bandwidth ceiling.  Cheap checks rejected: `GGML_CUDA_MMB_CACHE` 32/128 is a wash (960/957/957).
**Recommendation: scoped follow-up port, ~2.3 % ceiling.**

## UPDATE — session 6 (2026-09-20): the `mmb_*` kernels are at their gfx1151 ceiling

Session 5 left "`mmb_dense` (21 %) + `mmb_routed_glu` (16 %) need a split-K / int8-IU8 restructure".  This session closed both ideas, plus the bf16-shadow alternative.  **No code change** — the
worktree stays clean at `7e431fc82`.  Full detail, tables and traps are in `HANDOVER.md` §session 6;
the short version:

* **int8 WMMA is not faster than bf16 WMMA on gfx1151.**  `tools/wmma-peak-gfx1151.cpp` (new) measures
  **27.5 vs 27.6 T-MAC/s**; the Q8_0 per-block-scale epilogue then drops int8 to **14.1** vs bf16's
  **19.7**.  The 174 T-MAC/s figure that motivated §9 is **gfx1201**.  **The IU8 restructure is a
  net loss on the target arch.**
* **A bf16 weight shadow is 2.44x slower.**  The same `attn_qkv` shape is 2.08 ms as Q8_0 (WTYPE 1,
  dequant) and 5.07 ms as native BF16 (WTYPE 2), measured in situ on the BF16 twin of the model.  The
  kernel is weight-cache/bandwidth bound, not dequant-ALU bound; on-the-fly dequant is correct.
* **Every tile knob is a wash or worse:** dense `BM=64` -6.6 %, GLU big `BM=128` ~-1 %, GLU
  `BN_SMALL=64` wash, force-wide -2.8 %, activation cache 16/64 wash.  The geometry *was* tuned.
* **Efficiency:** `mmb_dense` Q8_0 = 14.8 T-MAC/s = **54 %** of the 27.6 bf16 peak; GLU ~**36 %**
  (it pays the dequant twice).  The residue is dequant-issue contention and is inherent.
* **Fast iteration model for the next session:** `Qwen3.6-35B-A3B-Q4_K_M` (21 GiB) exercises the same
  kernels with the same shares as the 94 GiB Flash-Next, but loads in seconds.  `rocprofv3`'s
  `grid_size_x` is `blocks.x x 256` — divide before matching a shape.

**Revised next step:** the `mmb_*` kernels' remaining gains need arithmetic that is already bf16
(none), so the prefill lever is **outside `mmb_*`** (FA 11.3 %, GDN 5.5 %, MoE concat+reduction 6.8 %,
`rms_norm` ~5 %), or **promotion** — all §11 gates are green (sessions 5c/5d), which is now the
highest-value step.

## Next

1. **QSA v3 packed-WMMA attention** — now the #1 kernel.  Plan: graph-side `qsa_pack_keys`/`_values`,
   the `qsa3_rows`/`qsa3_merge` block descriptor, then the `qsa3_attn_kernel` WMMA; prefill-only
   (`n_query >= 128`), VEC kept for the W=1..8 band.  Estimated ~1017 -> ~1160 t/s on Q4_K_M.
2. ~~**Q8_0 IU8-WMMA**~~ — **REFUTED on gfx1151 (session 6): int8 == bf16 WMMA, and the epilogue makes
   it slower.**  Do not pursue.  See the session-6 UPDATE above.
3. bf16-producer marking (kills `mmb_cvt`) and the HC prefill fusion.
4. Optional, only after gfx1151 is exhausted: a single 7900 XTX (gfx1100) small-model test with
   `GGML_CUDA_MMB_RDNA3=1` — the tiling likely needs an RDNA3_0 pass.

## QSA / Q8_0 investigation (2026-09-19)

Both post-MMB leaders were probed with cheap experiments before committing to a rewrite:

* **Q8_0 dense (23 %)** — shapes logged: `attn_qkv M=10240 K=2560`, `attn_gate M=6144 K=2560`,
  `ssm_out M=2560 K=6144`, `hc_* M=320/10240`, `ffn_*_shexp M=640 K=2560`.  Forcing the narrow or wide
  MMB tile changes IQ4_XS pp8192 by +0.6 % / -2.4 % (802 / 798 / 778 t/s), so the `M>=6144` heuristic
  is already optimal — the kernel is **occupancy/LDS-bound, not tiling-bound**.  The next move is an
  **int8 IU8-WMMA** variant (int8 weights+activations straight to the tensor cores; less LDS than the
  dequant-to-bf16 path), not tile tuning.
* **QSA (16.6 %)** — the f16/bf16 gather widened 8B -> 16B is **neutral** (796 vs 798 t/s), so it is
  not load-issue-bound.  It is compute/reduction/occupancy-bound: the VEC kernel does 1 query column
  x 16 heads per block and reduces with `v_dot2`, while a 16x16x16 WMMA tile does 16 heads x 16 cells
  per instruction — that is the `qsa3` gap.

Neither is reachable by tuning; both need a kernel restructure (QSA v3 below, and an IU8 Q8_0 path).
Diagnostics are left gated in the tree: `GGML_CUDA_MMB_LOG=1` (shape log),
`GGML_CUDA_MMB_TILE=0/1` (tile override).

## qsa3 — packed-block WMMA prefill for the QSA sparse attention (2026-09-19, session 2)

**DONE and validated** (was `NEXT WORK #1`).  Was `GGML_CUDA_QSA3=1` (opt-in); **since session 15
(2026-09-20) the gate is compile-time `LLAMA_QSA3_ENABLE`, default 1** (see the session-15 UPDATE).

### What it is

The VEC kernel (`fattn-qsa.cu`) walks the top-k list cell by cell with `v_dot2` and one query
column per block.  `fattn-qsa3.cu` (new, ported from the Strix Halo branch's `qsa-attn`) shares a
block of work across **G = 4 queries x 12 q-heads** (48 output rows) and runs the score and PV
passes on the **F16 WMMA tensor cores** over a package of 4 key blocks (16 keys) at a time.

Three kernels: `qsa3_rows_kernel` (per-row sortedness check + rank-sort), `qsa3_merge_kernel`
(merge 4 queries' rows into a sorted, deduplicated, block-aligned union + a 16-bit per-query
membership mask), `qsa3_attn_kernel` (16x16x16 F16 WMMA, mask folded into the score pass).

### Plumbing

* The pack is a **pure graph composition** (reshape + permute + cont) - **no new ggml op**.
  Helpers `qsa_pack_{keys,values}_graph` in `src/models/qwen4exp.cpp`.
* The op gained two optional srcs: `ggml_flash_attn_qsa_set_packed(a, packed_keys, packed_values)`
  (`src[7]`/`src[8]`).  NULL/NULL = the VEC path, unchanged.
* **Every KV type is supported**, because the kernels read only the F16 packs.  The cast must happen
  on the cache's *natural contiguous* view before any permute (`qsa3_f16_cast`), and the quantized
  types route through F32 - **the backend `dup` only dequantizes quantized->F32 and cannot permute a
  quantized tensor at all** (getting this wrong aborts in `ggml/src/ggml-cpu/ops.cpp:578`, once per
  QSA layer).
* **Prefill-only by construction**: the support check requires `q->ne[1] >= 128` and RDNA3_5, so the
  whole W = 1..8 decode/verify band keeps the VEC kernel and width purity is untouched.
* Portable WMMA wrapper (`qsa3_wmma_f16`, no-op on `RDNA4`) so a multi-arch build still compiles -
  verified for **gfx1201 and gfx1100** as well as gfx1151.

### Results (gfx1151, ROCm 7.14, IQ4_XS, `-b/-ub 2048`)

| KV type | pp4096 off -> on | pp8192 off -> on |
|---|---:|---:|
| f16 | 855.8 -> 893.8 (+4.4 %) | 816.0 -> 882.2 (+8.1 %) |
| bf16 | 834.2 -> **899.8 (+7.9 %)** | 791.9 -> **884.3 (+11.7 %)** |
| q8_0 | 827.1 -> **896.4 (+8.4 %)** | 784.6 -> **873.5 (+11.3 %)** |

All three converge at depth - the kernel reads the same F16 packs, so the KV type no longer matters
for the attention arithmetic.

**PPL parity** (wikitext):

| c | VEC | qsa3 |
|---|---:|---:|
| 16384, bf16 | 3.3932 | 3.3900 |
| 16384, q8_0 | 3.3861 | 3.3879 |
| 16384, f16 | 3.3883 | 3.3869 |
| 32768, bf16 | 4.3378 | 4.3353 |

All within +/-0.002 (the run's own error bar is +/-0.027).  Greedy text is coherent and agrees for
~40 tokens before the approved **prefill re-baseline** near-tie flip.

### Kernel profile (rocprofv3, pp8192)

| kernel | VEC | qsa3 |
|---|---:|---:|
| attention | 2944.4 ms (`flash_attn_qsa`) | **674.9 ms** (`qsa3_attn_kernel`) |
| rows / sortedness | - | 25.5 ms (`qsa3_rows_kernel`) |
| merge / union | - | 28.2 ms (`qsa3_merge_kernel`) |
| **total** | **2944.4 ms** | **728.6 ms (4.04x)** |

The rows figure is after the 2026-09-19 bitmap-sort rewrite below (it was 441.0 ms and the qsa3
total 1151.9 ms / 2.56x before it).

### Two findings worth not re-deriving

* ~~**The dense startup portion must stay.**~~ **SUPERSEDED 2026-09-19 - see the always-QSA section
  below; the shortcut is now default OFF.**  The original measurement (forcing QSA in the startup
  region was a pessimization, pp2048 912.5 -> 903.2) was taken *before* the bitmap sort, when the
  startup regime also paid the 441 ms rank-sort.  After the sort fix the same regime measures:
  **qsa3 137.8 ms vs dense `flash_attn_ext_f16` 149.9 ms (qsa3 8 % faster on the attention even when
  every cell is selected)**, with indexer+top-k 20.9 ms and qsa3 rows+merge 6.6 ms on the QSA side.
  So the path only still lost because the indexer cost more than the kernel saved.
* **The top-k rows are UNSORTED**, so `qsa3_rows_kernel` must sort them.  Proven by disabling the
  sort: the rows kernel drops 441 -> **8 ms** but the attn kernel explodes 682 -> **19593 ms** and
  throughput collapses 866 -> 474 t/s (unsorted rows break the merge kernel's binary searches).  So
  the sort is required and **the PPL parity above did exercise and validate it**.

### `qsa3_rows_kernel` sort rewrite - DONE (2026-09-19)

The rank sort was O(ns^2) and dominated qsa3 (441 ms of 1152 ms).  It is now a **bitmap counting
sort**: the row is a *set* of cell ids, so a `nk`-bit presence bitmap + a popcount scan enumerates
it in ascending order - **exactly the order the rank sort produced** - for O(ns + nk/32) per row.

Get the data first: an early assumption that the row is a set of whole 4-key blocks was **wrong**.
The real rows (dumped from a live run) are a handful of **long contiguous runs** - row0 is one run of
2051 keys, row2 is runs of 436/1611/4, i.e. 1-6 runs per row - so the bitmap is dense and the
popcount scan is cheap.  (This also explains why the row looks like "whole 4-key blocks" to an
aligned-group scan: a long run of consecutive keys has `ent[i] == ent[i-1]+1` everywhere.)

Implementation notes: the `nk`-bit bitmap plus 256 per-lane scan offsets share the rows kernel's
dynamic smem after the key array (48 KiB budget, i.e. `nk` up to ~1.5M); the kernel takes
`bitmap_words` and **falls back to the original rank sort when it is 0** (too large a cache).
Sentinels are appended after every valid key, matching the rank sort's placement.

**Result: rows 441.0 -> 25.5 ms (17x); qsa3 total 1151.9 -> 728.6 ms (4.04x vs VEC).**  PPL is
**bit-identical** to the pre-rewrite build (c16384 3.3900, c32768 4.3353, q8_0 3.3879) - the sort is
order-exact, not merely equivalent.

### Where the time goes now

With qsa3 on, the pp8192 profile is led by **`mmb_*` kernels** (`mmb_f32split_kernel` 2164 ms,
`mmb_routed_glu_kernel` 2085 + 1626 ms, `mmb_dense_kernel` 1904 + 1606 ms, `mmb_cvt_f32_bf16`
648 ms) - QSA is now **728.6 ms (4th-ish)** and no longer the #1 kernel.  The MMB follow-ups
(SS 7-9) are the bigger lever.

## Always-QSA prefill + F32 dense weights off (2026-09-19, session 4)

Two default flips, both measured; plus the revert of a failed experiment.

### 1. `LLAMA_QSA_DENSE_SHORTCUT` default ON -> OFF = **always QSA** (maintainer decision)

The shortcut sent `n_kv <= indexer_top_k + r - 1` (= 2051) to the dense masked FA arm on the
reasoning that there the top-k selects *every* cell, so sparse attention saves no work while still
paying the indexer.  **qsa3 changed that.**  Measured in the fully-dense startup regime (pp2048,
single ubatch, every cell selected, gfx1151, `rocprofv3`):

| | attention kernel | indexer + top-k | qsa3 rows/merge | total |
|---|---:|---:|---:|---:|
| dense (`shortcut=1`) | 149.9 ms (`flash_attn_ext_f16`) | - | - | **149.9 ms** |
| QSA (`shortcut=0`) | **137.8 ms** (`qsa3_attn_kernel`) | 20.9 ms | 6.6 ms | **165.3 ms** |

So qsa3's kernel is already **8 % faster than the dense FA kernel even when nothing is skipped** -
the path only still lost because the indexer + top-k cost 20.9 ms against the 12.1 ms the kernel
saved.  End to end the flip is ~neutral:

| pp | always-QSA | dense-shortcut | delta |
|---|---:|---:|---:|
| 512 | 703.8 | 700.3 | +0.5 % |
| 1024 | 839.0 | 841.5 | -0.3 % |
| 2048 | 909.3 | 919.4 | -1.1 % |
| 4096 | 904.0 | 913.9 | -1.1 % |
| 8192 | 900.1 | 902.9 | -0.3 % |

(within ~1 % run variance for most points).  What it buys: **no numerics seam at `n_kv == width`**,
and qsa3 is now exercised at *every* context length - a `-c 2048` PPL exercise used to be silently
dense, which is why the early "qsa3 is neutral" readings were vacuous.  **Decode is unaffected**: it
stays dense via the existing arch policy (`qsa_dense_decode_until` = 64K on gfx1151, always on
gfx1201) - verified `tg64` shallow 25.80 -> 25.83.

PPL moves the right way: c16384 bf16 **3.3900 -> 3.3821**, q8_0 3.3879 -> 3.3861; c32768 bf16
4.3353 -> 4.3397 (noise).  Greedy text coherent.  `LLAMA_QSA_DENSE_SHORTCUT=1` restores the dense
arm (still the `LLAMA_QSA_SPARSE_FA=0` cross-check).

**Remaining QSA gap = the indexer**, not attention: `indexer_topk_radix_histogram` 10.0 ms,
`indexer_topk_deterministic_write` 4.7, `indexer_topk_count` 3.3, `indexer_topk_radix_select` 2.9
(pp2048, n=96 launches each).  Halving that makes always-QSA a win even at pp2048.

### 2. `GGML_CUDA_MMB_F32SPLIT` default 2 -> 0 (MMB F32 dense weights off)

The F32 dense weights are all **tiny-M**: MoE router `ffn_gate_inp` M=512, `ssm_alpha`/`ssm_beta`
M=48, `hc_*_inject` M=4, `ffn_gate_inp_shexp` M=1 (per-layer inventory via `GGML_CUDA_MMB_LOG=1`).
Both paths cost ~1.0 s at pp8192 (12 % of prefill):

* `mmb_f32split_kernel<128,128,32,64>` computes a padded **128-row A tile**, so M=4 wastes 32x of its
  WMMA work (for `hc_attn_inject` M=4/K=10240 the launch computes ~53 GFLOP of WMMA for a 0.17 GFLOP
  problem) - 2164 ms in the profile, `n=2160`;
* `F32SPLIT=0` (rocBLAS, `Cijk_Alik_Bljk_SB_MT32x32x8_...`) lands on the same shape bound - 2076 ms,
  `n=1784`.

Both are ~10x off the memory-bound floor (A traffic = `(T/BN)*M*K*4`, B traffic = `(M/BM)*T*K*4`;
for M=512/T=2048 that is 168 MB against a 26 MB floor).  Since rocBLAS measured *faster*,
the MMB default is now **off**: IQ4_XS pp4096 896.0 -> **915.9**, pp8192 896.8 -> **902.9**.
`GGML_CUDA_MMB_F32SPLIT=1` opts it back in.

**Tried and rejected:** a 16x256 small-M tile for `M <= 64` (aimed at the 32x padding).  It is
**worse** - 870/847 t/s vs 895/885 for the 128 tile - because `BN=256` halves the block count in a
kernel that is already parallelism-starved, and `BM=16` does not help M=512.  Reverted; the real fix
is a dedicated tiny-M (or split-K) kernel.

## UPDATE — session 5 (2026-09-19): the fresh profile + the shape-aware F32 split

### Fresh post-session-4 profile — reprioritises the list

`rocprofv3` kernel trace, MMB + QSA3 on, IQ4_XS `-b/-ub 2048`, in % of total kernel time:

| family | pp8192 (17.77 s) | pp32768 (77.02 s) |
|---|---:|---:|
| `mmb_*` | 52.0 % | 48.2 % |
| **F32 rocBLAS** (`Cijk_...`) | **11.7 %** | **12.0 %** |
| `qsa3` attn/rows/merge | 4.9 % | 6.4 % |
| `rms_norm_f32` | 5.9 % | 5.5 % |
| `dsv4_hc_pre` + `_post` | 8.0 % | 7.4 % |
| PACK/copy (qsa3 pack) | 3.0 % | 3.5 % |
| **indexer top-k family** | **0.99 %** | **3.2 %** |
| indexer score | 0.98 % | 3.2 % |

**This contradicts the "indexer is next-work #1" ordering carried in from session 4.**  The indexer
is ~1 % at 8K (it was 0.5 % at the pp2048 startup regime that ordering was based on) and only reaches
3.2 % at 32K.  It does scale with `n_kv x n_tps` while its output is capped at 2051 cells, so it
matters at depth - but the **F32 path is 12 % at *both* depths** and was the larger target.  The
indexer is deferred, not dropped.

### The fix: F32 dense weights, split on shape (+2.1 / +2.6 / +3.1 %)

The F32 dense weights were all-or-nothing between rocBLAS and the MMB f32split tile, and the two
paths disagree about *which shape* each wins - so the previous default was a wash (mode 2 was even the
worst).  Per-shape, from the trace (the launcher grid is `ceil(M/128) x ceil(T/128)` workgroups, so
the launch identity is exact):

| shape | rocBLAS | MMB f32split | winner |
|---|---:|---:|---|
| `ffn_gate_inp` M=512 K=2560 | 2.044 ms | **0.846 ms** | **MMB 2.4x** |
| `hc_attn/ffn_inject` M=4 K=10240 | **1.332 ms** | 1.825 ms | rocBLAS 1.37x |
| `ssm_alpha/beta` M=48 K=2560 | **0.359 ms** | slower | rocBLAS |

**The discriminator is M alone.**  Mode 1 (new default) takes MMB only when `M >= 128`:

| pp | mode 0 (all rocBLAS) | **mode 1 (M>=128)** | mode 2 (all MMB) |
|---|---:|---:|---:|
| 2048 | 914.87 | **933.92 (+2.1 %)** | 901.73 |
| 4096 | 908.93 | **933.03 (+2.6 %)** | 901.01 |
| 8192 | 901.99 | **929.75 (+3.1 %)** | 902.97 |

F32 GEMM total 1947 -> ~1537 ms.  Why rocBLAS wins the small-M/short-K shapes: its `MT32x32x8`
kernel split-Ks hard (M=512/K=2560 launches 262144 blocks for ~1.0M outputs, ~64 threads per output,
so it pays partial-sum traffic) and loses the 2.4x there; on M=4/K=10240 the 128-row MMB tile wastes
WMMA rows and loses.

**Numerics parity**: PPL c16384 bf16 3.3821 (mode 0) vs 3.3875 (mode 1), error bars ±0.027.  Decode
untouched (tg64 25.95 vs 26.00) - the `T >= 512` gate is unchanged, so `W=1..8` purity still holds by
construction.  Greedy text coherent.

**Lesson (do not relearn): a first attempt added "or `K >= 4096`" to the rule, expecting the WMMA
path to help the K=10240 hc inject pair.  It is wrong and costs 375 ms.**  The trap that hid it:
bucketing launches by grid alone put `hc_inject` (760) + `ssm` (576) + others into one 1.065 ms
average that looked like an MMB win.  **Split the bucket before believing a per-shape number.**

## UPDATE — session 5e (2026-09-19): dsv4_hc investigated — ~18 % kernel-local headroom, and the real
lever is bytes (bf16 intermediates)

`dsv4_hc_pre_f32` + `dsv4_hc_post_f32` = 1432.5 ms (8.5 % of pp8192); pwilkin's reference is ~0.93 s
vs our 1.44 s.  **Nothing landed** — the tree is clean, baseline pp8192 restored to 953.5 t/s.

**Measure the ceiling, don't infer it — and sweep the grid.**  `tools/dram-bw-probe.cpp`
(`hipcc --offload-arch=gfx1151 -O3`), 192 MB buffers, grid swept, 34 C, no other GPU work:

| pattern | GB/s | % of 256 GB/s spec |
|---|---:|---:|
| pure sequential read (best grid) | **241.5** | 94.3 % |
| copy (read + write) | ~208-216 | 81-84 % |
| write-only | ~217 | 85 % |
| **the exact `dsv4_hc_pre` shape** (x + gate, 4 streams each, + dst), best grid | **232.6** | 90.8 % |
| the real `dsv4_hc_pre_f32` | **197** | 77.0 % |

The part sustains **~240 GB/s** as measured here (a separate report puts it at ~255 achievable, which
would widen the gap slightly) — and **the dsv4_hc_pre access pattern is not inherently slow**: a clean
kernel of the identical shape reaches 232.6 GB/s.  So our kernel sits **~18 % below what its own
pattern allows** (and ~23 % below the best pure read), not at a wall.

Two corrections this took, both my own:

1. **The ceiling was inferred from our own kernels** in the first pass, giving "already at the memory
   wall, nothing to gain".  It has to be measured.
2. **The grid must be swept before quoting a ceiling.**  grid=4096 everywhere gave 231.6 GB/s (read) /
   227.2 (pattern); the *same kernels* at grid=16384 give **241.5 / 232.6** — ~5 %.  Instruction
   sequence matters too: a 4-accumulator x4-unrolled read variant measured **worse** (221-230) than a
   plain single-accumulator grid-stride loop, so "more ILP" is not automatically more bandwidth on
   this part (consistent with this machine's number depending on the exact sequence used).

The split:

* **~18 % kernel-local** (197 -> 232.6) = ~133 ms = **~0.8 % of prefill**.  Unexplained so far; what
  differs from the probe kernel is the sigmoid (measured free), the runtime-stride address arithmetic,
  and the strided `dst` write.
* **the dominant lever is bytes** — bf16 intermediates cut unique traffic 189 -> ~105 MB (1.8x); at
  232.6 GB/s that is ~0.45 ms/launch vs 0.98 = **~2.3 % of prefill**, matching the reference's
  1.44 -> 0.93 s.  It needs the `hc_norm`/`hc_gate` producers to write bf16, so it is a **graph-level
  change**.

Kernel-level A/B (`rocprofv3`):

| variant | `dsv4_hc_pre_f32` |
|---|---:|
| production (`expf`) | 745.0 ms |
| `__expf` | 744.8 ms |
| identity (no sigmoid) | 745.3 ms |
| `float4` over the contiguous `i0` axis | neutral (end-to-end, value-preserving) |
| hc loop unrolled + `__restrict__` | 729.9 ms (**-2.0 %** = 0.09 % of prefill, not kept) |

No register spilling anywhere (VGPR 24->32, `Scratch_Size` 0).

**Two probe traps, both mine:** size buffers by *bytes* and index by *float4 count* (mixing them is a
4x out-of-bounds read that presents as `Memory access fault ... Page not present`, looking like a
HIP/driver problem); and count *write* bytes as the iterations actually performed, not one whole
buffer (counting 3x192 MB when 50 MB was written reported **302 GB/s on a 256 GB/s part** — exceeding
spec is the tell that the accounting, not the kernel, is wrong).

### METHODOLOGY — prefill time here is DATA-DEPENDENT; judge kernel variants on kernel time

The identity-sigmoid variant measured **-13 % end-to-end** (951 -> 824 t/s) while **its own kernel was
unchanged** (745.3 vs 745.0 ms).  The whole swing was downstream on *identical launch counts*:
`mmb_routed_glu` 3803.6 -> 5906.6 ms (+55 %), `mmb_routed` +392 ms, `qsa3_attn` +167 ms.  Changing the
activations changes the MoE routing, and the GLU launch is sized worst-case with early-return slots,
so different routing = different work.

**Rule: a value-changing kernel variant must be judged on `rocprofv3` kernel time, never end-to-end
t/s.**  Corollary for the rest of this WIP: every end-to-end A/B here also changed numerics (MMB,
QSA3), so those wins were confirmed in the kernel profile rather than taken from t/s alone.

Second trap from the same episode: the first version passed the flag as a **runtime kernel argument**,
so the "no sigmoid" variant still compiled the `expf` *and* a select — it did strictly *more* work.
A diagnostic whose whole point is to remove work must be a **template** parameter.

## UPDATE — session 5d (2026-09-19): W=1..8 width-purity probe — the last gate item closes

The `W = 1..8` logits matrix was the one §11 gate never run because no harness existed.  It does now:
**`tests/test-logits-width-probe.cpp`** (`cmake --build build-rocm --target test-logits-width-probe`),
adapted from `archive/work/strix-halo/issue25/logits-width.cpp`.  Gate **PASSES**:

| prefill P | `MMB=0` row0 | `MMB=1` row0 | width purity |
|---|---|---|---|
| 256 (below `MMB_MIN_T = 512`) | `6228d03bd2b501b4` | `6228d03bd2b501b4` — **identical** | PASS, maxdiff 0 |
| 1024 | `ac4d5de3d40a2b1d` | `3703c13f03c4b25d` | PASS, maxdiff 0 |
| 2048 | `1996b44e491de5c9` | `e3e4220fe83831da` | PASS, maxdiff 0 |

Below the threshold MMB is unreachable in both the prefill and the decode batch, so the configs are
bit-identical across every row of every width — the "identical by construction" claim demonstrated.
Above it the row-0 hashes differ by the **approved prefill re-baseline** (MMB replaces the MMQ
reduction with a dequant-to-bf16 WMMA one), while **`width_purity` stays PASS with MMB on** — MMB
introduces no width dependence, and `T >= 512` is what keeps it out of the `W <= 8` band.

Two probe bugs, same class as the `-md` trap: `llama_batch_init(ubatch)` sizes for the *micro*-batch
so a `P`-token prefill overruns it (silent SIGSEGV, plus a `GGML_ASSERT` for the `n_batch` half); and
`n_batch` (per `llama_decode`) vs `n_ubatch` (per micro-batch) are different knobs, both scaling
with `P`.  Also: `rows[W-1].data()` hashes the `std::vector` objects, not the floats.

## UPDATE — session 5c (2026-09-19): the promotion gates, run

With the remaining big kernels out of reach for a safe change (`mmb_routed_glu` 22.7 %,
`mmb_dense` 21.1 % — both need a split-K / IU8 restructure), this session ran the §11 gates that the
handover listed as never run.

Test models: `build-rocm/bin/test-llama-archs -o /tmp/test-models` (the `test-generate-models` fixture;
~200 tiny GGUFs).  The test binaries are not in `build-rocm/bin` by default — build them by target.

| gate | result |
|---|---|
| `test-recurrent-state-rollback` qwen35-dense / nemotron_h-dense / deepseek4-moe | **PASS** (max diff 0) |
| `test-recurrent-state-depth` (n_rs_seq 1..15) | **PASS** (`total failures = 0`) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** (older notes say 18/18 — cases were added) |
| `test-backend-ops -o GATED_DELTA_NET` | **46/46** |
| `test-backend-ops -o FLASH_ATTN_EXT` | **OK** |
| `plain == draft-mtp` greedy text (qwen4exp, MMB+QSA3 on **and** off) | **PURE, byte-identical** |

All with `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`.

### `plain == draft-mtp` on qwen4exp — PURE, gate PASSES

| config | plain | draft-mtp | verdict |
|---|---|---|---|
| `MMB=1 QSA3=1` | `bbd4bcb519e4` | `bbd4bcb519e4` | **identical** (1700 chars) |
| `MMB=0 QSA3=0` (features OFF) | `5120b28f2879` | `5120b28f2879` | **identical** (1720 chars) |

Hashes differ *between* configs — that is the approved prefill re-baseline — but *within* each config
the two arms are byte-identical.  These are also two of the previously-missing re-baseline hashes.

### METHODOLOGY TRAP — reported WRONG first

The first attempt concluded `plain != draft-mtp` ("320 chars vs 1942") and attributed it to pre-existing
cause 3.  **Fabricated by the harness:**

* the plain arm must **not** get `-md`; passing the draft model makes llama-cli initialise an MTP
  context even with `--spec-type none`, which fails (`this model is an MTP draft head without a
  trunk`, `llama_server exited with code 1`) and the run **exits 1 without generating**;
* its stdout still had the `Loading model... |\b-\b\\...` **spinner**, and `grep -v '^\[' | tr -d
  '\b'` turns that spinner into exactly 320/318 "chars" — so the comparison was **a spinner vs real
  text**;
* the 1942-char side was genuine; the 320-char side never generated a token.

Rules: **never pass `-md` to the plain arm**; **assert the arm generated output** (`$?`, non-empty, not
the spinner) before comparing.  A one-sided load failure always "diverges" and looks like a real
near-tie flip.

`LLAMA_QSA_OFF=1` is separately unusable with `-md` (`llama_server exited with code 1`).  And note
`FLASH_ATTN_QSA` is 22/22 now, not 18/18.

## UPDATE — session 5b (2026-09-19): tiny-M F32 kernel for the hc `*_inject` GEMMs (+2.3-3.3 %)

Next-work #1 from the session-5 list (the remaining F32 tiny-M) is done.

**The shape:** `hc_attn_inject` / `hc_ffn_inject` are M=4 (hc = the hyperconnection stream count),
K=10240, T=2048 — the largest remaining F32 cost at **1012 ms, 1.330 ms/launch, 5.7 % of prefill**.

**Why neither existing tile could serve it:** M=4 gives no M parallelism, so rocBLAS launches
`ceil(T/32)=64` blocks and the 128-row MMB f32split tile pads the A panel 32x.  Both read the 84 MB
activation exactly once (memory floor ~0.33 ms) yet sit at **~63 GB/s**, while `rms_norm_f32` on the
same part sustains **~330 GB/s** (1044 ms for 84 MB read + 84 MB write per launch).  So it is a
**parallelism wall**, not a bandwidth or arithmetic one — 16-64 blocks cannot keep enough loads in
flight.  (The two injects also cannot be fused: they consume different `xn`, pre-attn vs post-attn.)

**The fix:** `mmb_tiny_m_f32_kernel` — one warp per token, every lane accumulating a k-strided
partial *for all M rows*, so X is read once, coalesced (consecutive lanes read consecutive float4)
and reused across M in registers.  256 blocks at T=2048 instead of 16-64.

| | per launch | GB/s |
|---|---:|---:|
| rocBLAS | 1.330 ms | 63 |
| MMB f32split | 1.825 ms | 46 |
| **tiny-M warp-per-token** | **0.500 ms** | **168** |

End to end: **pp2048 933.0 → 964.1 (+3.3 %), pp4096 934.4 → 962.0 (+2.9 %), pp8192 934.6 → 955.9
(+2.3 %)**.  hc_inject total 1012 → 568 ms.  PPL c16384 bf16 3.3875 → **3.3851** (noise); decode
identical (tg64 25.96 both — M=4 here is a weight row count, not a token count, so the `T >= 512`
gate still keeps the whole decode/verify band off this path by construction).

**Tuning, both negative results recorded:**

* **Tokens-per-warp (TT): TT=1 is best.**  TT=2 (954.8) and TT=4 (952.3) measured *worse* than TT=1
  (957.4) at pp8192.  The motivation was W traffic — each lane walks a k-strided slice so a warp
  collectively reads all 164 KB of W (336 MB from L2 at 2048 warps) against X's 84 MB from DRAM.
  L2 serves the panels well enough that the extra live registers and reduced block count cost more.
* **MMAX specialisation: +0.5 %.**  The real M is 4, so `<4,TT>` (not `<8,TT>`) is the right
  instantiation; `<8,1>` is kept as the general path for M in 5..8.

**The trap worth carrying forward:** the first version changed only the *launcher* and measured
neutral (+/-0.3 %) — because the `M >= 128` rule I had just added rejected the shape in the *gate*
before the launcher was ever reached, so the kernel never ran.  A bench delta alone would have read
as "the idea failed".  **The kernel trace is what caught it: the symbol was simply absent.**  When a
new kernel measures flat, verify it actually executed.

## Gates before this could be opt-in, let alone defaulted on (from the parked handover)

- W = 1..8 logits matrix with `GGML_CUDA_MMB=1` == off (prefill-only, `T >= 512`).
- MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
- `test-recurrent-state-rollback`; `test-backend-ops` suites.
- Same-seed prefill re-baseline documented; gfx1100/gfx1201 compile + consistency.
