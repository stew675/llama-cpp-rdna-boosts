# GROUPS.md — the WIP patch set as 5 "like items"

**Why this file exists.**  The WIP started as 38 commits and is now **5 thematic patches** (`patches/`,
one per theme, applied in order on top of the r12 delivery tree).  This is the triage sheet for a
reviewer (or an agent) on **another architecture**: what each group is, which parts are gated, what is
expected to apply on gfx1100 / gfx1201, and how to prove it.

> **Porting to gfx1201 / gfx1100?  Read [`gfx1201-porting.md`](gfx1201-porting.md) first.**  It is the
> multi-session porting overlay: it re-examines (and partly supersedes) the "RDNA4 is a no-op / new
> work" framing below, gives the exact gfx11-vs-gfx12 WMMA fragment mapping, the per-group gfx1201
> assessment and gate plan, and the session breakdown.  The group *semantics* below are unchanged.

The consolidation is **content-preserving**: the 5-patch result has the **tree
`d365b43ddc87c472c33a121247931269f975aa43`**, byte-identical to the 38-commit tip it replaced
(`git am` 5/5 verified on a fresh r12 tree).  If a group is dropped, the remaining tree is simply the
corresponding subset; the original 38-commit history is still in this repo's git history.

## Apply order and base

```sh
# base = the r12 delivery tree (release.json: base ebbb18522, tree 8a80535e556bef57666d2eaa4d3eb4cf93fb83f5)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout ebbb18522
bash <this-repo>/scripts/apply-all.sh .        # -> branch rdna-boosts, tree 8a80535e... (r12)
git checkout -b wip-mmb-general
git am <this-repo>/wip/mmb-general/patches/*.patch   # 6/6, tree 580db5174...
```

## The five groups

| # | patch | theme | env gate | always-on? |
|---|---|---|---|---|
| 1 | `0001-WIP-mmb-…` | the general-purpose `mmb` bf16-WMMA dequant weight GEMM | `GGML_CUDA_MMB=1` **+ an arch gate** | no |
| 2 | `0002-WIP-qsa3-…` | `qsa3`: the packed-block WMMA sparse-attention path | compile-time `LLAMA_QSA3_ENABLE` (default 1) | **yes** |
| 3 | `0003-WIP-the-F32-tiny-M-…` | F32/tiny-M kernels, the default flips, the W=1..8 probe | `GGML_CUDA_MMB=1` (kernels) / none (flips) | mixed |
| 4 | `0004-WIP-HC16-…` | HC16 native-BF16 producers + non-temporal accesses | `GGML_CUDA_MMB_HC16=1` (producers) / none (NT hints) | mixed |
| 5 | `0005-WIP-indexer-…` | the fused indexer top-k op | none (op-driven) | **yes** |
| 6 | `0006-WIP-mmb-RDNA4-…` | the `mmb` RDNA4 (gfx12) fragment port **+ the arch-scoped weight-type/path split** | `GGML_CUDA_MMB=1` + the scope policy (`MMB_TYPES` / `MMB_DENSE`) | no |

> **Patch 6 and the theme split.**  S5-S7 (2026-09-21) landed as its own patch rather than folded:
> patches 1, **3 and 4** all touch `mmb.cu`, so there is no single theme to fold it into without a full
> re-cut.  Patch 6 is therefore the authoritative source for the RDNA4 scope policy
> (`mmb_wtype_ok` / `mmb_dense_flag`) — see `gfx1201-s5s7-mmb-results.md` §4.

### 1 — `mmb`: the general-purpose bf16-WMMA dequant weight GEMM

**Files:** `ggml/src/ggml-cuda/mmb.cu` (all of it), `mmb.cuh`, the `MUL_MAT`/`MUL_MAT_ID` op hooks and
the fusion stand-down + activation-marking passes in `ggml-cuda.cu`.

**What it is:** the whole `mmb_*` family — a dequant-to-bf16 weight GEMM on the WMMA units for every
weight type the delivery's models use (IQ4_NL, Q8_0, Q4_K, Q5_1, IQ3_S, Q5_K, Q6_K, IQ4_XS, Q3_K,
IQ3_XXS, BF16), dense / routed-MoE / fused gate+up+swiglu.  Includes the structural work: the A-panel
double-buffered GLU tile, the `v_perm` bf16 RNE pack, the IQ3_S A-panel dequant split, the tile-class
threshold (`THRESH == BN_SMALL`) and the `load_regs` IQ3_S field preload.

**Arch note — this is the group that needs your attention.**  The kernels dispatch through
`mmb_wmma_bf16` / `mmb_wmma_f16` (`mmb.cu` top), which select at compile time: gfx11 uses the
`__builtin_amdgcn_wmma_*_w32` intrinsics with the 16-half row fragment, **gfx12 (RDNA4) the
`..._w32_gfx12` intrinsics with the 8-half "two runs of four" fragment** (added 2026-09-21, S5).
Consequences to test, not assume:

* **gfx1201 (RDNA4): PORTED 2026-09-21 — but it is NOT a blanket win, so RDNA4 defaults to a
  *scoped* subset.**  With every weight type and the dense path enabled, MMB is a **−4…−13 %**
  prefill regression on three models; the wins are confined to the **routed MoE** path and the
  **qwen4exp HC** paths.  The gate was therefore split (see below) and RDNA4 defaults to
  IQ-family weights + routed/HC paths only — neutral wherever it would lose, **+3…+7 %** where it
  wins.  Full matrix: `gfx1201-s5s7-mmb-results.md`.  As a tune target this is the arch where the
  dense tile geometry has *never* been right (every RDNA4 loss lives in the gfx1151-tuned dense
  tile), so a genuine gfx1201 dense geometry is the open work, not a re-run.
* **gfx1100 (RDNA3_0):** the same gfx11 WMMA instructions exist, so this is the arch where group 1 has
  a real chance.  The delivery's own block 04 carries an RDNA3_0-specific FA head cap and `ncols2`
  rule, i.e. gfx1100 is a *tuned-differently* sibling, not a copy of gfx1151.  **Measure, do not
  transfer**: check the arch gate engages, then the gates below.
* **gfx1151 (RDNA3_5):** the reference.  End-to-end prefill +32 % at pp2048 / +43-48 % at 4k-32k.

**Measured on gfx1151 (target model `Qwen3.8-Flash-Next UD-IQ4_XS`, bf16 KV, `-ub 2048`):** MMB family
2980 → 2453 ms; pp8192 1037 → 1116 t/s.

### 2 — `qsa3`: the packed-block WMMA sparse-attention path

**Files:** `ggml/src/ggml-cuda/fattn-qsa*.cu*`, the QSA arm selection in `src/models/qwen4exp.cpp`.

**What it is:** the QSA sparse attention rewritten as a packed-block WMMA kernel (`2944 → 728.6 ms`,
4.04x), with every KV cache type supported (q4_0/q4_1/q5_0/q5_1 via `get_dequantize_V`), a bitmap
counting sort in `qsa3_rows_kernel` (441 → 25.5 ms), the pack fused into one launcher pass, and a
**compile-time** gate (`LLAMA_QSA3_ENABLE`, default 1) that replaced an env gate because
`rocprofiler-register` made env reads racy under the profiler.

**Arch:** the F16 WMMA path is **per-arch**: gfx11 (RDNA3_0/RDNA3_5) uses the first-gen builtin and
its 16-half full-row fragment; **gfx12 (RDNA4) uses `…_f16_w32_gfx12` with the 8-half "two runs of
four" fragment** (one `#if` shim, 7 sites — ported 2026-09-21, `gfx1201-s4-qsa3-results.md`).
**Always-on** (compiles in at `LLAMA_QSA3_ENABLE=1`); it is the qwen4exp attention path, so it only
matters for that model family.  Validate with the long-context gates (it is the attention kernel, not
a micro-optimisation): same-seed text at `-c 16384`+, `test-backend-ops -o FLASH_ATTN_QSA` (26 cases
since 2026-09-21 — **the packed path had no oracle before that**: the old cases never attached
`src[7]/src[8]`, so they only ever ran the VEC kernel).

### 3 — F32/tiny-M kernels, the default flips, and the W=1..8 probe

**Files:** `mmb.cu` (the F32 split + tiny-M kernels), the default flips in `ggml-cuda.cu` /
`qwen4exp.cpp`, `tools/` (the probe).

**What it is:** (a) F32 dense weights off by default + the **always-QSA prefill flip** (drops the
dense-shortcut regime — this is the single largest *shape* change in the prefill curve, and it is
policy, not a kernel); (b) the shape-aware F32 dense split (MMB for the MoE router only); (c) the
warp-per-token tiny-M F32 kernel for the `hc *_inject` GEMMs (M=4/M=8) and its MMAX=4 specialisation;
(d) the `W=1..8` logits width-purity probe (a `tools/` harness, not shipped code).

**Arch:** (a) and (d) are arch-neutral; (b)/(c) are `mmb_*` kernels and inherit group 1's arch
situation.  **The `always-QSA` flip is the one to A/B first** — it changes the prefill shape on every
arch, and on gfx1151 it was worth +10.6 % at pp4096 on its own.  **Updated 2026-09-21 (gfx1201 port):**
the flip is **not** portable — it is now gated per arch: the default is always-QSA **only on gfx1151**
(the arch it was measured on); every other arch keeps the delivery's dense shortcut.  On gfx1201 the
flip cost pp2048 **−18.9 %**, pp8192 **−6.7 %**, because the VEC QSA path (qsa3 was still gfx11-gated)
has no answer for the short-prefill band.  See `gfx1201-s1s2-results.md` §3a.  **Re-tested with qsa3
ported (S4):** the shortcut is still equal-or-better at pp4096..32768 (−0.15..−0.70 % for
always-QSA), so the gate stays — see `gfx1201-s4-qsa3-results.md` §3.1.

### 4 — HC16 native-BF16 producers + non-temporal accesses

**Files:** `ggml-cuda.cu` (the graph-optimizer marking passes), `norm.cu`, `dsv4-hc.cu`,
`concat.cu`, the fused gated-unary producer, `common.cuh` (`ggml_cuda_nt_load`).

**What it is:** the bf16-producer port — the graph marks "this activation wants a BF16 copy" and the
fused producers (`rms_norm+mul`, `sigmoid+mul`, `scale+unary`, generic unary) emit it directly, so the
`mmb_cvt` activation-conversion bucket disappears; the `xn` stream became BF16-only with its own
producer slot.  Plus the non-temporal sweep: `dsv4_hc` pre/post (−18.6 % on `_pre`), and non-temporal
**loads** in `concat_transposed_src1_dim0`, `moe_weighted_reduction` and the fused gated-unary producer.

**Arch:** the producer machinery is generic CUDA/HIP; the **non-temporal hints are the portable part**
and the easiest independent win to test on a new arch (the rule that was learned the hard way: **loads
only, per-kernel, A/B load vs store** — a non-temporal *store* evicts the next op's input).

### 5 — the fused indexer top-k op

**Files:** `ggml/src/ggml-cuda/indexer-topk.cu`, `src/llama-memory-hybrid-idx.cpp` (`blk_cells`),
`src/models/qwen4exp.cpp`.

**What it is:** `GGML_OP_INDEXER_TOPK` — the qwen4exp indexer's score-expand + mask + top-k fused into
one radix op (the naive graph materialised a 512 MB F32 tensor at 64K and re-read it 5 times), with a
**deterministic** ascending-column gather (the old atomic placement made the list order — and which
tied cells made the rank boundary — vary run to run, which the f16-state recurrences turned into
visible nondeterminism).  Now **1117 ms at pp32768 = 1.80 %** of the run, from 2.94 %.

**Arch:** generic.  **Always-on** for qwen4exp.  **Its correctness gate is special** — see below.

## gfx1201 results (2026-09-21) — what the port verified

The 5-patch set was applied to a 3× R9700 (gfx1201) box, built, and measured against the delivery.
Full data: `gfx1201-s1s2-results.md`; plan: `gfx1201-porting.md`.

| group | gfx1201 verdict | action |
|---|---|---|
| **G5 indexer** | **win**, grows with depth: +2.1 % pp8192 → **+8.0 % pp98304** | keep, always-on |
| **G4 non-temporal** | **small consistent win** at depth (+0.3–0.4 % at 32k/64k/98k) | keep, always-on |
| **G3a always-QSA flip** | **regression** (pp2048 −18.9 %, pp8192 −6.7 %) | **gated off on RDNA4** (default shortcut ON except gfx1151) |
| **G2 qsa3** | **win**: +7.6 / +11.5 / +10.4 % prefill at pp4096/16384/32768 (qsa3 vs VEC, same build) | **ported 2026-09-21** (`gfx1201-s4-qsa3-results.md`); folded into patch 2 |
| **G1 mmb** | **ported 2026-09-21, scoped**: +3…+7 % on the routed MoE and qwen4exp HC paths; neutral elsewhere (scoped default) | enabling it unscoped is a −4…−13 % regression on dense/K-quant models — see `gfx1201-s5s7-mmb-results.md` |

Net gfx1201 prefill vs the delivery (Flash-Next IQ4_XS, q8_0 KV, 3-GPU tensor, `-b/-ub 2048`):
**+3.2 % / +5.0 % / +6.6 %** at pp32768/65536/98304, no regression at any depth, same-seed greedy
and all op oracles bit-identical.  The patch set carries the G3a arch gate (patch 3, commit
`bdf97a390` on the pre-consolidation branch); it applies clean 5/5 and the applied tree is
`c0f8ea75ba`.

**G2 `qsa3` is ported to RDNA4** (2026-09-21) and is the second-biggest gfx1201 win after the indexer.
**G1 `mmb` is ported too, but scoped** — and the scoping is the interesting part: `GGML_CUDA_MMB`
bundled **arch × weight-type × kernel-path × model-family** into one switch, and the measurement shows
the four axes disagree on RDNA4 (a −4…−13 % regression with everything on, wins only on the routed
MoE and qwen4exp HC shapes).  It is now:

* **arch-scoped weight types** — `mmb_wtype_ok()` (one mask, replacing five duplicated hard-coded
  lists).  RDNA4 = the **IQ family** (`IQ4_NL`/`IQ3_S`/`IQ4_XS`/`IQ3_XXS`); RDNA3_5/RDNA3_0 keep the
  full set, so **gfx1151 is unchanged**.  A/B with **`GGML_CUDA_MMB_TYPES=<csv>`**.
* **path-scoped** — `mmb_dense_flag()` separates the generic quantized dense tile GEMM and the F32
  router (the losers) from the routed MoE path and the HC/tall-M/tiny-M paths (the winners).
  RDNA4 default: dense/router **off**; `GGML_CUDA_MMB_DENSE=0|1` overrides.

A user who sets `GGML_CUDA_MMB=1` on gfx1201 therefore **cannot lose** on it, and can opt into the
qwen4exp dense win with `GGML_CUDA_MMB_DENSE=1`.  The gfx11 (gfx1151/gfx1100) behaviour is unchanged
and keeps the full type set — the split only narrows what RDNA4 accepts.  Details:
`gfx1201-s5s7-mmb-results.md`.

## gfx1100 job (the next box)

gfx1100 (RDNA3_0, RX 7900 XTX) shares the **gfx11** WMMA builtin with gfx1151, so it needs **none of
the gfx12 fragment work**.  Its job is the mirror image of gfx1201's:

1. **Apply and build the set** (`scripts/apply-all.sh` + `patches/*.patch`; the tree is
   `c0f8ea75ba`).  It should compile as-is — all three arches are covered by the existing
   `#if defined(RDNA4)` no-op wrappers and the runtime gates.
2. **Arch-neutral groups first (G5 + G4):** apply, measure, gate.  Expect the G5 indexer to win and
   scale with depth like gfx1201 (it is the same generic kernel); the G4 hints are per-kernel A/B.
   **G3a:** the default is shortcut-ON on gfx1100 (non-gfx1151); if gfx1100 measures always-QSA as a
   win it can move into the gfx11 exception (the gate is one comparison in `qwen4exp.cpp`).
3. **G1 `mmb` — the real gfx1100 prize.**  gfx11 WMMA works there, so `GGML_CUDA_MMB=1` +
   `GGML_CUDA_MMB_RDNA3=1` opens it (the delivery's block-04 work already shows gfx1100 is a
   tuned-differently sibling), and gfx1100 keeps the **full** weight-type set (the RDNA4 scoping
   above does not apply).  **This is untested and needs a re-tune, not a transfer:** the tile
   geometry / `THRESH` / `VAULT` / `min_t` constants are gfx1151-tuned.  Follow `gfx1201-porting.md`
   §5 (G1 row) and §6.5.3 for the sweep shape; start on the fast `Qwen3.6-35B-A3B Q4_K_M` model.
4. **G2 `qsa3`** likewise: extend its support predicate from `RDNA3_5` to `RDNA3_0` and re-tune.
5. **Record** into a `gfx1100-porting.md` results file (same shape as `gfx1201-s1s2-results.md`) and,
   if the mmb constants need per-arch values, promote them to a `cc`-selected default (see
   `gfx1201-porting.md` §7 point 3) so gfx1151 is not disturbed.

**Do not** carry gfx1201's numbers to gfx1100 or vice-versa — they have different LDS budgets, WMMA
rates and (for MMB) different tuning points.  Measure on the box.

## Gates

**Common (any arch, any group):**

| gate | command | expected |
|---|---|---|
| PPL | `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 llama-perplexity -m <model> -f prompts/prose-rdna-boosts.txt -c 2048 -b 2048 -ub 2048 -ngl 99 -fa 1 -ctk bf16 -ctv bf16` | **10.6015** |
| greedy | same flags, `llama-cli -n 128 --seed 42 --temp 0 -c 8192 --no-display-prompt --single-turn`, then `scripts/extract-generated.py` | **624 chars sha `9c281c415082`** |
| width purity | `test-logits-width-probe <model> <prompt> 1024 512` | `width_purity=PASS (worst maxdiff 0)` |
| op oracles | `test-backend-ops -o FLASH_ATTN_QSA` / `-o GATED_DELTA_NET` / `-o INDEXER_TOPK` | 2/2 backends each |
| MTP | `benchmarks/mtp-adaptive-methodology.md` | acceptance > ~0.45 at pos 1; MTP ≥ plain at depth 3 |

**Group 5 needs a different gate than the rest.**  The indexer's selection width is 2051, so a prose
prompt at `-c 8192` **never leaves the dense shortcut** — PPL/greedy/width cannot reach the sparse
indexer at all.  Use a **long-context same-seed A/B** (≥ 16k-token prefill, `--seed 42 --temp 0`,
compare `extract-generated.py` hashes) plus `test-backend-ops -o INDEXER_TOPK`.  A result at 8k says
nothing about this group.

## Triage order for a new architecture

1. **Build the base (r12) alone** and record the baseline gates.  Everything below is relative to that.
2. **Group 5 is the arch-neutral, always-on item.**  **Group 2 (`qsa3`) now serves gfx1151 *and*
   gfx1201** (the RDNA4 fragment port landed 2026-09-21) and is gfx11-only for gfx1100 until its
   support predicate gains `RDNA3_0`.  Apply and gate these first; they are where a new arch gets a win
   without touching the WMMA assumptions.  **Group 3(a) (always-QSA) is NOT arch-neutral**: default the
   shortcut ON unless the arch is measured to prefer always-QSA (only gfx1151 is, today) — see the
   group-3 note above.
3. **Group 4's non-temporal hints** next: per-kernel, A/B load vs store.  On gfx1201 they are a small
   consistent win at depth (+0.3–0.4 %); on another arch measure, do not assume.
4. **Group 1 last** and only after checking `mmb_enabled()`'s arch gate on your box.  On RDNA4 it
   refuses today (the `mmb_wmma_*` builtins are no-ops there); on RDNA3_0 it may well engage, but treat
   the tile/threshold constants as gfx1151-tuned and re-measure them rather than transferring.
5. **Never trust a gated path under `rocprofv3`** without confirming it from the kernel names in the
   trace — `rocprofiler-register` `setenv()`s during early init and can make an env gate read as
   *unset* (ROCm issue #10196).  This bit this campaign twice; `LLAMA_QSA3_ENABLE` was made
   compile-time because of it.
6. **Do not judge a value-changing or perf variant on end-to-end t/s.**  Use `rocprofv3` kernel time,
   and for anything routed/GLU use a **fixed-token** run (`llama-perplexity` on a fixed prompt) —
   `llama-bench` uses random prefill tokens, so routed kernels are not comparable run to run.

## What this WIP is NOT

* **Not a delivery block.**  It is WIP on the `wip-mmb-general` branch; promotion needs the maintainer
  (see `HANDOVER.md` §E — rebase onto a canonical fork, regenerate `patches/`/`release.json`, and
  decide whether MMB rides as a **block-08 amendment**).
* **Validated on gfx1151 (RDNA3_5) and gfx1201 (RDNA4).**  gfx1201: G5/G4 win, `qsa3` wins
  (+7.6..+11.5 % prefill), the always-QSA flip is gated off, and `mmb` stays gfx11-gated (its RDNA4
  fragment port is future work — `gfx1201-porting.md` §6.5).  gfx1100 is next (see the gfx1100 job
  above).  Do not transfer tuning constants between arches.
* **Not a supported configuration.**  All WIP env gates (`GGML_CUDA_MMB`, `GGML_CUDA_MMB_HC16`, …)
  default off; the delivery runs with them off.
