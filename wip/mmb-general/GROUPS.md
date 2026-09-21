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
git am <this-repo>/wip/mmb-general/patches/*.patch   # 5/5, tree d365b43d...
```

## The five groups

| # | patch | theme | env gate | always-on? |
|---|---|---|---|---|
| 1 | `0001-WIP-mmb-…` | the general-purpose `mmb` bf16-WMMA dequant weight GEMM | `GGML_CUDA_MMB=1` **+ an arch gate** | no |
| 2 | `0002-WIP-qsa3-…` | `qsa3`: the packed-block WMMA sparse-attention path | compile-time `LLAMA_QSA3_ENABLE` (default 1) | **yes** |
| 3 | `0003-WIP-the-F32-tiny-M-…` | F32/tiny-M kernels, the default flips, the W=1..8 probe | `GGML_CUDA_MMB=1` (kernels) / none (flips) | mixed |
| 4 | `0004-WIP-HC16-…` | HC16 native-BF16 producers + non-temporal accesses | `GGML_CUDA_MMB_HC16=1` (producers) / none (NT hints) | mixed |
| 5 | `0005-WIP-indexer-…` | the fused indexer top-k op | none (op-driven) | **yes** |

### 1 — `mmb`: the general-purpose bf16-WMMA dequant weight GEMM

**Files:** `ggml/src/ggml-cuda/mmb.cu` (all of it), `mmb.cuh`, the `MUL_MAT`/`MUL_MAT_ID` op hooks and
the fusion stand-down + activation-marking passes in `ggml-cuda.cu`.

**What it is:** the whole `mmb_*` family — a dequant-to-bf16 weight GEMM on the WMMA units for every
weight type the delivery's models use (IQ4_NL, Q8_0, Q4_K, Q5_1, IQ3_S, Q5_K, Q6_K, IQ4_XS, Q3_K,
IQ3_XXS, BF16), dense / routed-MoE / fused gate+up+swiglu.  Includes the structural work: the A-panel
double-buffered GLU tile, the `v_perm` bf16 RNE pack, the IQ3_S A-panel dequant split, the tile-class
threshold (`THRESH == BN_SMALL`) and the `load_regs` IQ3_S field preload.

**Arch note — this is the group that needs your attention.**  The kernels dispatch through
`mmb_wmma_bf16` / `mmb_wmma_f16` (`mmb.cu` top), which use the **gfx11 `__builtin_amdgcn_wmma_*_w32`
intrinsics and are deliberately no-ops under `RDNA4`** (so the multi-arch build compiles).
Consequences to test, not assume:

* **gfx1201 (RDNA4):** `mmb_wmma_*` is a no-op there, so the GEMM cannot produce output and the host
  gate is expected to keep MMB off.  Verify the gate refuses (no wrong output, no crash) — and if the
  RDNA4 `v_wmma_*` path is wanted, that is new work, not a port.
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

**Arch:** generic HIP/CUDA, no arch-specific path.  **Always-on** (compiles in at `LLAMA_QSA3_ENABLE=1`).
It is the qwen4exp attention path, so it only matters for that model family.  Validate with the
long-context gates (it is the attention kernel, not a micro-optimisation): same-seed text at
`-c 16384`+, `test-backend-ops -o FLASH_ATTN_QSA` (18 cases incl. the CPU oracle).

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
arch, and on gfx1151 it was worth +10.6 % at pp4096 on its own.

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
2. **Group 2 + 3(a) + 5 are the arch-neutral, always-on items** — apply and gate them first; they are
   where a new arch gets a win without touching the WMMA assumptions.
3. **Group 4's non-temporal hints** next: per-kernel, A/B load vs store, the cheapest independent win.
4. **Group 1 last** and only after checking `mmb_enabled()`'s arch gate on your box.  On RDNA4 expect
   it to refuse (the `mmb_wmma_*` builtins are no-ops there); on RDNA3_0 it may well engage, but treat
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
* **Not validated outside gfx1151 (RDNA3_5).**  Every number in this repo is gfx1151.  Nothing here
  claims gfx1100/gfx1201 behaviour — that is exactly what the other boxes are for.
* **Not a supported configuration.**  All WIP env gates (`GGML_CUDA_MMB`, `GGML_CUDA_MMB_HC16`, …)
  default off; the delivery runs with them off.
