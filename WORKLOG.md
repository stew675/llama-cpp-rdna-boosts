# WORKLOG — dated delivery records

Reverse-chronological log of every delivery-affecting change to the
**rdna-boosts 15-patch set** (block amendments, community-fix
integrations, re-baselines, regeneration + clean-apply re-verifications).
Newest entry first.  The README's
[Current state](README.md) section is a lean summary and points here
for the full record; per-block technical notes live in
`patches/README.md`, the verification contract in `MANIFESTS.md`.

---

## 2026-09-11 (5) — F2 cause 2 FIXED: the MoE decode/verify band is band-uniform (block-13 amendment)

**qwen4exp is now width-pure `W = 1..8`**, so the designed `--spec-draft-n-max <= 7` verify batch is
bit-identical to the 1-token decode — the remaining *logit-level* condition for `plain == draft-mtp`.
Canonical tip **`bfaa83d8a`**, net tree **`4e5f2952f016f1ac160c53261f7b01d346322534`**; only
`ggml/src/ggml-cuda/mmvq.cu` changed (26 insertions / 11 deletions).

**Task 1 answered by measurement, and it moved the diagnosis.**  `[GD]` full-graph dumps show the graphs
are **identical** at every stage (2647/2404/2271/1863/1668/1565 nodes at both `W=4` and `W=5`), so the
previous entry's question ("fusion-applied vs graph-built-with-fewer-ops") is settled: the graph always
contains `MUL_MAT_ID(ffn_moe_gate)`, `MUL_MAT_ID(ffn_moe_up)`, `GLU(ffn_moe_swiglu)` at the same node
indices (`k=76/77/78`), and only the *fusion coverage* differs.  But the cause was **not** the
`mul_mat_id_glu_ops` fusion the previous entry blamed:

* `mul_mat_vec_q_moe`'s `__launch_bounds__` was `get_mmvq_mmid_max_batch_for_device<type>()*warp_size`
  — the upstream **per-type mmvq cap compiled into the kernel**, while the block is
  `(warp_size, ncols_dst)`.  Launching `IQ3_S` (cap 4) with `ncols_dst = 5` is 160 threads > the bound
  and dies with `ROCm error: unspecified launch failure`, so the cap is a *capability* limit, not just
  a heuristic.
* the same cap routes the upper band to MMQ: `ggml_cuda_mul_mat_id` takes `ne2 <= cap → mmvq` else
  `should_use_mmq → MMQ`, and `use_mmvq` (`ggml-cuda.cu:3730`) gates the `mul_mat_q_pair` fusion
  (which is what actually fired at `W = 5..7`).  mmvq (one warp per token, `mul_mat_vec_q_moe`) and
  MMQ reduce in different orders, so the band splits.
* the **UD-IQ4_XS quant mixes expert types per layer** — 47 layers `IQ3_S` gate/up (cap 4), layer 2
  `IQ4_XS` (cap 5), down `IQ4_NL`/`Q8_0` (cap 7) — which *predicts the census exactly*: fused layers
  48/48/48/48/1/0/0/0 for `W = 1..8` (measured `ffn_moe_up` MUL_MAT_ID counts 0/0/0/0/47/48/48).  That
  is the 4→5 and 5→6 boundary; the down's cap 7 is the 7→8 boundary.

**Fix.**  Block 13 already carries the invariant — `mul_mat_vec_q_switch_ncols_dst`'s `has_ids` branch
("this must cover `ncols_dst == 1` as well … the decode == verify invariant", added by block 13
2026-09-01) routes every `MUL_MAT_ID` to the column-generic MoE kernel.  The fix completes it for the
whole band:

1. `mmvq_mmid_max_batch_band(cap)` floors the per-type cap at `MMVQ_MAX_BATCH_SIZE` (the decode/verify
   band), applied to every AMD arch lookup, host *and* device;
2. `mul_mat_vec_q_moe`'s launch bound becomes `MMVQ_MAX_BATCH_SIZE*warp_size`, so the kernel can
   actually be launched across the band.

No other path changes: the caps' call sites are all `MUL_MAT_ID`-only, so dense models are untouched.

**Validation.**  `W = 1..8` all `3adeb313042a871b` (`-sm layer`) and `dcf1ae667f730879`
(`-sm tensor`) — i.e. every width equals that split's **pre-fix `W = 1` value**, so plain decode is
bit-unchanged and only `W = 5..8` moved onto it (the F1 "move the cheap side" pattern).  Also pure with
the state-sequence dimension exercised (`RS=0` and `RS=from_w`).  Controls: the **pre-fix vs post-fix
`plain` text is byte-identical** (`3ee9daee5c07`), and at the MTP gate config (`n_max 3` = `W=4`, a
no-op width) the runs are byte-identical: acceptance `0.76744` (66/86), generation 80.1 vs 80.0 t/s.
Text level: pre-fix `n_max 3` ≠ `n_max 7`; **with the fix they agree** (`8a50ea24e8d5`).

**Perf — the fix is a large win at the verify widths** (`llama-batched-bench`, fixed vs baseline
interleaved, swappable `libggml-hip.so`):

| model | batch 1 | 2 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| qwen4exp 3-GPU `-sm tensor` tg128 | 50.5 / 50.4 | 85.4 / 85.6 | 134.1 / 132.9 | **149.5 / 118.4 (+26 %)** | **162.5 / 130.6 (+24 %)** | **171.4 / 147.0 (+17 %)** | **178.0 / 155.4 (+14.5 %)** |
| 35B-A3B MoE 1 GPU tg128 | 98.3 / 98.1 | 156.4 / 156.2 | 254.2 / 254.1 | – | – | – | **341.3 / 289.9 (+17.8 %)** |
| 4B dense 1 GPU tg128 | 100.6 / 100.5 | 167.2 / 167.3 | 294.0 / 294.9 | – | – | – | 414.5 / 413.3 |

Widths inside the caps are unchanged (and bit-identical), dense is untouched, and MTP `n_max 7` goes
**41.8–42.5 vs 36.1 t/s (+16–18 %)** with acceptance `0.59375` vs `0.55556`.  The upstream per-type
mmvq caps were actively *costing* throughput on RDNA4 with the fork's mmvq + fused-GLU kernels.

**Cross-checks.**  `GATED_DELTA_NET` 4/4 backends OK; `FLASH_ATTN_EXT` 4/4 OK; MoE asterisk intact
(`ac8825358d9adfda` / `bd138ad2326fbbf2`); reserves unchanged (no allocation changes); clean-apply
simulation strict 15/15 with **0 whitespace warnings** and tree == canonical.

**Open — cause 3 (new, pre-existing, independent of cause 2).**  `plain` still differs from
`draft-mtp` text for qwen4exp even after the fix (`plain` `3ee9daee5c07` vs `n_max 3 == n_max 7`
`8a50ea24e8d5`), and the fix **cannot** be responsible: `n_max 3` uses `W = 4`, where the fix is a
verified no-op (bit-identical logits, byte-identical text, byte-identical acceptance).  Since the
single-step probe shows bit-identical logits for `W = 1..8` across both splits and the state-sequence
dimension, the divergence must be a **multi-step** effect — i.e. the speculative roll-back itself.
Prime suspect: the **masked (freed/stale) KV cells** written by rejected drafts, which block 14 keeps
at exactly `+0.0` in the HIP `fattn-tile`/`fattn-mma-f16` and Vulkan paths but **not** in qwen4exp's
**QSA sparse-attention path** (`fattn-qsa.cu`).  Next instrument: a multi-step probe (prefill P, then
feed a *fixed* token sequence, comparing the logits at each position between `W = 1` steps and `W = k`
chunks) — the single-step probe and `RS` dimension cannot see a cell that is only stale after a
roll-back.

## 2026-09-11 (4) — F2 cause 2 localised: it is the MoE gate+up+GLU fusion flipping at `n_q = 5`, not a kernel-dispatch band

**Instrument.**  The per-node `[ND]` dump (`GGML_CUDA_NODE_DUMP=1`, re-appliable from
`wip/kv-quant-purity-followups/tools/node-dump-instrumentation.patch`) on qwen4exp, `-sm layer`,
P=256, RS=0, at W=4/5/6/7.

**Result — the executed-op census is the signature, and it is unambiguous:** only one op's count changes
across the whole band, and it changes at exactly the boundary:

| width | `ffn_moe_down` | `ffn_moe_up` | total nodes |
|---|---|---|---|
| `W=1..4` | 48 | **0** | 1920 |
| `W=5`   | 48 | **47** | 1967 |
| `W=6,7` | 48 | **48** | 1968 |

So the MoE **gate+up+GLU fusion** (`mul_mat_id_glu_ops = {MUL_MAT_ID, MUL_MAT_ID, GLU}`,
`ggml-cuda.cu:3324`, admitted via `ggml_cuda_should_fuse_mul_mat`) is applied for `n_q <= 4` and
abandoned from `n_q = 5`, and the fused GLU epilogue and the separate `MUL_MAT_ID` + `GLU` chain do not
sum identically — which is the impurity.  The `W=6`/`W=7` pair is a **perfect calibration** (`+0` nodes,
`0` differing ops) — that is *why* they hash identically, and it validates the census (the previous
session's node-dump diff was unusable because it had no such calibration, and because shape equality is
not sufficient: cache/state tensors legitimately differ with W).

**Refuted by measurement (the pre-HC-fix exclusion list was unreliable — the `W=1` vs `W>=2` break
dominated those hashes):** the block-13 `get_mmvq_mmid_max_batch` cap and its MMQ pair arm (forcing MMVQ
across the band via a temporary `GGML_CUDA_MOE_MMVQ_BAND=1` is **byte-identical**, and `should_use_mmq`
is false for `n_q <= 8`, so that arm never fires in the band); the MoE expert kernel (`mul_mat_vec_q_moe`
is **provably width-invariant** — `rpb` derives from `blocks_per_row_x`, a K property, and
`block_dims = (warp_size, ncols_dst)` is one warp per token); `LLAMA_QSA_OFF`,
`GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE` (all leave `W=5` = `c999233926f0`; positive control
`LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` -> `bdaa8fc57381`, the recorded HC-off value, proving the
env plumbing); `ggml_cuda_should_use_mmvf(F32)` on gfx1201 = `ne11 <= 3` (a 3/4 boundary that does not
appear).

**Consequence for the brief:** cause 2 is a **fusion-coverage** band, not an `ncols_dst`/`ne11`
kernel-dispatch band — so it is *not* the same workstream as F3, and the fix is the F1/HC shape: keep the
gate+up+GLU fusion for the whole decode/verify band (`n_q <= 8`) rather than only `n_q <= 4`, measuring
the verify-throughput cost the way F1's was.  Next step: re-run the `GGML_CUDA_DISABLE_FUSION=1` width
matrix **post-HC-fix** (the earlier "survives all fusions disabled" observation predates it) to confirm
the unfused path is itself width-invariant.  Debug tooling to reuse: the node census above (nothing is
committed as code — it is the existing `[ND]` dump plus 30 lines of parsing), and the
`W=6` vs `W=7` calibration trick.

## 2026-09-11 (3) — Block 08 amended: the decode/verify band no longer spans two FlashAttention kernel families (F1 fixed)

**What changed.**  `ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu`) no longer returns
`BEST_FATTN_KERNEL_VEC` for small batches.  The fallback was upstream code (`11f0af550`, "for small
batch sizes the vector kernel may be preferable"): VEC for `n_q == 1` when `!gqa_opt_applies`, and for
`n_q <= 2` whenever K or V is quantized.  Both conditions are *always* inside the `n_q <= 8`
decode/verify band (prefill fell through to TILE anyway), so the branch only ever split the band; it is
deleted and the whole band uses TILE — the same shape of fix as the block-08 WMMA guard added
2026-08-29 (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff` in `launch_fattn`.

**Why.**  Measured with a new `GGML_CUDA_FA_TRACE` instrumentation (committed for reuse as
`wip/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch`): with `q8_0` or `q4_0` K/V the
chooser returned **VEC (100) at `n_q = 1,2` and TILE (200) at `n_q >= 3`**; the two families order the
online-softmax/PV reduction differently, so token-0 logits at `W = 1,2` disagreed with every verify
width.  The launcher's own plan was *already* width-independent (`ntiles_dst_eff`, `parallel_blocks`
== `ntiles_KV` at every width), which is why the earlier F1 suspects (KV-type staging, the KV-cache
write path, `stream_k` rounding) all measured clean.

**Measured (3x gfx1201, ROCm 7.14, unpinned).**
- 4B Q8_0 `q8_0/q8_0` **1 GPU `W=1..8` all `31a0c1bace68`**, 2-GPU `-sm tensor` `abebfb93`, 3-GPU
  `-sm tensor` `7fe106f5`; `q4_0/q4_0` `619c151e48c7` / `240bc37d` / `483a850e` — all four split
  configs pure, and every value is that config's *previous verify* value (only `W=1,2` moved).
- 27B Q8_0 `q8_0/q8_0` 3-GPU tensor `W = 1,2,3,4,5,8` all `d4156dbeb225`.
- f16/bf16 configs byte-identical (they never took VEC): 4B f16 `671d60969874`, bf16 `b5d7e7b4`.
- text level, 27B 3-GPU tensor, ctx 8192, 300 greedy tokens, `q8_0` KV: plain == `n_max 3` ==
  `n_max 7` = `3537bc2b36be` (before: plain `73b2565bce47`/2810 chars vs verify `3537bc2b36be`/2801);
  f16 control `f32aac948600` for both.  **Harness note:** `llama-cli`'s `/\|` spinner is ``-based and
  timing-dependent and the banner embeds the build SHA — apply backspaces and strip both before
  hashing; three "divergences" this session were spinner noise.
- MTP: 27B `n=96` q8_0 KV acceptance **0.90789 (69/76), identical** to the pre-fix build.  MoE
  asterisk unchanged (`ac8825358d9adfda` / `bd138ad2326fbbf2`, and both `bd138ad2326fbbf2` with
  `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`).
- perf (llama-bench, q8_0 KV, interleaved, same binary): 4B pp512 7609.8 -> 7597.9 (-0.15% = noise),
  tg128 98.03 -> 97.11 (**-0.9%**); 27B 3-GPU tensor pp512 2257.3 -> 2253.2 (-0.2%), tg128 38.46 ->
  38.28 (**-0.5%**).  Reserves byte-identical (27B ub2048 q8_0: dev 1920.3284 / host 880.3360).
- op suites **with the fix active**: `test-backend-ops -o FLASH_ATTN_EXT` **4591/4591, 4/4 backends**;
  `-o GATED_DELTA_NET` 46/46, 2/2.  Quantized-KV coherence (the only configs that move) —
  gemma-4-E4B / 27B / qwen4exp with q8_0 KV: deterministic across runs and coherent.
- clean-apply: fresh `9113cc188` + `scripts/apply-all.sh` -> strict **15/15 `git am`**, **0 whitespace
  warnings**, applied tree == **`4104e7d34dd8cf9cb5488d46dcbba1b17eaa32d3`**.
- block-15 beta re-cut against the new base: **`0c8099ca2`**, tree **`7335b923d`** — metadata/offset
  only, 0 changed body lines (block 15's `fattn.cu` hunks sit at lines 166-326, the fix at ~690).

**New canonical tip `1bcf4e82d`**, tree `4104e7d34` (block 08 = `38cffdece`; blocks 09-14 got new SHAs,
bodies metadata-only — 2 lines each).  `rdna-boosts-all.patch` refreshed (95 files).

**This is NOT F2 cause 2.**  qwen4exp's `W >= 5` residual (`{1..4} {5} {6,7} {8}`) is **completely
unchanged** by this fix (W=1,4 `3adeb313042a`, W=5 `c999233926f0`, W=8 `c56ebb61963a`), which
**refutes the "F1 and F2 cause 2 share a cause" hypothesis** — cause 2 is not the kernel-family
chooser (it is a matmul/MoE dispatch band, still open).

**F3 refinement.**  The "slow pure" KV types are not missing a native kernel: they are rejected by
`ggml_cuda_fattn_kv_type_supported()` unless the build sets **`GGML_CUDA_FA_ALL_QUANTS`** (this build:
OFF), so `ggml_cuda_get_best_fattn_kernel()` returns `NONE` *before* the VEC/TILE choice (0 `[FATPATH]`
lines for `q4_1` vs 1+ for `q8_0`) and the attention takes the generic fallback — width-invariant by
construction (hence pure) and ~3.4x slower.  F3's first experiment is therefore a build-flag A/B.

## 2026-09-11 (2) — Block 14 amended: the fused hyper-connection ops serve the decode/verify band (qwen4exp width purity, cause 1 of 2)

**What changed.**  `ggml/src/ggml-cuda/hc-mix.cu` (`ggml_cuda_op_hc_mix`, `ggml_cuda_op_hc_combine`) and
the two graph gates in `src/models/qwen4exp.cpp` no longer require `nt == 1`: the fused
hyper-connection (HC) chain now serves the whole **decode/verify band `1 <= nt <= 8`**
(`HC_FUSED_MAX_TOKENS`, asserted in both ops).  The four mix kernels and the combine kernel take the
token index from `blockIdx.y` and offset every per-token pointer with the tensor's own stride
(`inject` is read with its view stride); at `nt == 1` every added term is zero, so the decode result is
unchanged (verified byte-identical for f16/bf16/q8_0/q4_0).  A `<= 8`-token **prefill** chunk also takes
the fused path — it cannot be told apart from a verify batch, and both must use the decode arithmetic;
wider chunks keep the unfused chain.  The ops are otherwise the same arithmetic, so no kernel numerics
were touched (the env fallback `LLAMA_FUSED_HC_MIX=0 LLAMA_FUSED_HC_COMBINE=0` reproduces the pre-fix
adaptive-MTP numbers exactly).

**Why.**  qwen4exp failed the decode==verify invariant ("F2"): a 1-token decode used the fused HC ops
while an n-token verify batch used the unfused chain, so the two computed the same position differently
and plain decode and `draft-mtp` disagreed.  Root-caused 2026-09-11 into **two stacked causes** (the
second is a `W >= 5` kernel-dispatch band shared with F1); this lands **cause 1**, as a block-14
amendment (block 14 introduced `hc-mix.cu` and the qwen4exp HC paths, so it owns them — the same
owner-based rule used for the block-02/12/13 amendments, not block 00).

**Measured** (3x gfx1201, ROCm `/opt/rocm-7.14-gfx1201`, unpinned):
- width probe, qwen4exp IQ4_XS f16 KV P=256 RS=0: `-sm layer` W=1..4 all **`3adeb313042a871b`** (was
  W=1 `3adeb313042a` + W=2..4 `044715b66e72f077`), `-sm tensor` W=1..4 all **`dcf1ae667f730879`**;
  **W=1 byte-identical to the pre-fix build on both splits and for every KV type** (f16
  `3adeb313042a`, bf16 `42e1bcfa57c1`, q8_0 `cb018394fd37`, q4_0 `688835658f30`).
- W=5 `c999233926f0` / W=6,7 `a8c532e12f9c` / W=8 `c56ebb61963a` (`-sm layer`) still grouped = **cause 2**,
  the `ncols_dst`/`ne11` selection band at `W >= 5`, shared with the `q8_0`/`q4_0` KV impurity (F1).
- greedy text: plain == `--spec-type draft-mtp --spec-draft-n-max 3`, byte-identical (3275 chars);
  `n_max 7` still differs (cause 2).  qwen4exp is therefore width-pure for **`n_max <= 3`**.
- adaptive-MTP (f16 KV, n=96): acceptance **0.50000 -> 0.76744**, MTP generation **63.3 -> 79.9 t/s**.
  With a `q8_0` KV cache: 0.50000 -> 0.43089 — that configuration is already width-impure via F1 (its
  W=1 decode is also unchanged), so it must be re-measured once F1 is fixed; recorded, not gated.
- perf: `-sm tensor` f16 pp512 1288-1300 (**parity**), tg128 48.30/48.72 (**decode unchanged** vs the
  pre-fix build, and the fusion's +14% over the unfused fallback 42.28/42.32 is kept).
- no regressions: 27B 1 GPU W=1/W=8 `4089b4d4`, W=9 `72af52db`; MoE W1 `ac8825358d9adfda` / W3
  `bd138ad2326fbbf2`; reserves byte-identical (qwen4exp ub2048 q8_0 dev 6690.3987 / host 1262.6954 /
  kvbuf 956.26; f16 6642.1331 / 1262.4297 / 1800.00); `test-backend-ops -o FLASH_ATTN_EXT` and
  `-o GATED_DELTA_NET` both 4/4 OK; `llama-batched-bench` B=1..8 clean.
- clean-apply: fresh `9113cc188` + `scripts/apply-all.sh` -> strict **15/15 `git am`**, **0 whitespace
  warnings**, applied tree == canonical **`e36263da57b8985cb98018af59fe639be0290dc4`**.
- the block-15 beta patch was re-cut against the new base (**beta tip `54859fdda`**, tree
  **`543ccc015`**, parent `1d8f53594`): metadata/offset-only, **0 changed body lines**.

**New canonical tip `1d8f53594`**, tree `e36263da5`.  Blocks 00-13 are byte-identical to the previous
regeneration; only `patches/0014-…` changed (the `From`/`index`/hunk-offset metadata plus the band fix).

**Follow-ups.**  Cause 2 (`W >= 5`) — fix together with F1/F3 (`wip/kv-quant-purity-followups/`);
the HC ops have no `test-backend-ops` coverage (a CUDA-vs-CPU band test would close that gap).

- **Block 15 beta revalidation (2026-09-11): re-cut against the current 15-patch delivery + full re-validation.**
  The Block-15 beta patch was cut on `b425aa8f7` (block 14 of the old **14-block** chain, block 13
  `e61676292`) — before block 00 existed and before the 2026-09-11 block-02/12/13 amendments — so it
  was re-cut against the current delivery (base `389c5341f`, tree `928852cdc`) and re-validated end to
  end.  Block 15 is still **staged in `beta/block-15-campaign-wins/`, NOT a delivery patch**; this is a
  beta-record update, not a delivery change (no `patches/` file and no block SHA moved).

  - **Re-cut**: new beta tip **`fe4f55278`** (tree `ffe197e2f`, parent `389c5341f`); the patch in the
    beta directory was replaced.  Measured dependency delta: **exactly one file** —
    `ggml/src/ggml-cuda/fattn-common.cuh` `7442bc22a` → `22eec7d57`, i.e. block 00's
    `ntiles_dst_eff` fix inside `launch_fattn`; the other 22 touched files are byte-identical to the
    cut base, so the re-cut changes only the `From` line, that one `index` line and one `@@` hunk
    header (+8 offset).
  - **Numbering correction**: the first draft of the revalidation plan claimed the beta patch had to be
    renumbered `[PATCH 15/15]` → `[PATCH 16/16]`.  That was **wrong**: `make-patches.sh` uses
    `git format-patch --start-number 0`, so the denominator is the *last block index* — the delivered
    15-patch set is `[PATCH 00/14]`…`[PATCH 14/14]` and block 15 is correctly `[PATCH 15/15]`.
    Verified by regenerating the 16-commit range with the same convention: all 15 delivery patch
    *bodies* byte-identical, block 15 emitted as `[PATCH 15/15]`.
  - **Clean-apply**: fresh `9113cc188` + `scripts/apply-all.sh` → strict **15/15**, **0 whitespace
    warnings**, tree `928852cdc`; + the re-cut block-15 patch → 16 commits, tree `ffe197e2f`.
  - **Result: every 2026-09-10 Block-15 claim reproduced.**  Reserves to the last decimal (27B
    `1920.3284/880.3360` → `1121.1252/81.1329`; 4B `1800.3284/840.3360` → `1001.1252/41.1329` →
    `257.1252/41.1329`; gemma-4-E4B/E4B-31B and the qwen4exp W1/W2/W3 chain incl. indexer KV
    `956.26` → `318.76` and the bf16/V5 table with bf16+V5 costing exactly f16); the 27B width-purity
    probe hashes **identical to the delivered reference** (`4089b4d4` / `a4817ee6` / `91434ea9`,
    `W=9` `72af52db`/`b059daa6`/`bc3faabd`) so `n_max <= 7` holds and block 15 changes no FA numerics;
    V4/V5 on == off **bit-identically** (the flagged `launch_fattn` risk is cleared); same-seed output
    byte-identical across gates on 4B/27B (short + 40k)/both SWA gemmas/qwen4exp; MoE asterisk intact
    (`ac8825358d9adfda`/`bd138ad2326fbbf2`, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` → both
    `bd138ad2326fbbf2`); MTP 27B `0.90789` and MoE `0.58378` identical on both builds; op suites
    `FLASH_ATTN_EXT` **7859/7859 ROCm0 + 7859/7859 CPU** (6 derived), `GATED_DELTA_NET` OK,
    `test-alloc`/`test-batch-alloc` clean, W4 round trip 16.00 → 56.00 → 16.00 MiB; cost inside the
    documented envelope (4B prefill V3 −1.6 % / V4 −1.75 %, 27B V3 −0.3 %, decode flat; vs the
    delivery build 27B pp512 −1.7 % / tg128 flat, MoE pp512 −0.3 % / tg128 −1.25 %).  gfx1151 was
    **not** re-run (no such hardware on this host).
  - **Three PRE-EXISTING findings (not Block-15 regressions — identical hashes on the delivery build),
    now tracked in `wip/kv-quant-purity-followups/` + `TODO.md` + `GREEDY-PURITY.md` §12:** (F1) a
    **q8_0 or q4_0 K/V cache breaks the dense `n_max <= 7` purity guarantee** (`W=1 == W=2` then
    `W=3..8`; text-level plain `8ed58aa9` vs spec `da56855b` on the 27B) — the impure set is exactly
    the two types with a fast native both-quantized FA path; (F2) qwen4exp's fused sparse QSA path is
    not width-invariant; (F3) the sub-`q8_0` quants (q4_1/q5_0/q5_1/iq4_nl) are pure and 1800–2400 MiB
    but ~3.4x slower because they have no native FA path.  **Policy decided (maintainer
    2026-09-11): differing K/V cache types are rejected as an accepted limitation** (mixed pairs are
    1.7–3.6x slower than the same-type equivalent and never smaller; upstream #25871 already enforces
    same-K/V for DeepSeek V4).
  - Records updated: `beta/block-15-campaign-wins/{README,HANDOVER,BETA-TESTING}.md`,
    `GREEDY-PURITY.md` §12, `TODO.md`, `AGENTS.md`, `wip/kv-quant-purity-followups/`.

- **Block 13 amendment (2026-09-11, second): MoE `MUL_MAT_ID` decode/verify dispatch fix + the shared-expert fusion kill-switch.**
  Root-causes and closes the qwen35moe batch-width residual
  (`wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 2).  The residual was
  **not** in the MoE expert GEMM kernels.  A per-node, stride-aware dump of the
  decode graph that also covers fused-window destinations localised the first
  divergence to the **fused shared-expert window**
  (`ggml_cuda_op_shexp_down_gate`, gated `// decode only` on `ne[1] == 1`).  Two
  independent causes, both in that region:

  1. **`MUL_MAT_ID` never used the dedicated MoE kernel at `ncols_dst == 1`.**
     `mul_mat_vec_q_switch_ncols_dst` returned early only for `has_ids &&
     ncols_dst > 1`, so a single-token `MUL_MAT_ID` fell through to the **dense
     ksplit kernel with an ids gather** while a multi-token verify batch ran
     `mul_mat_vec_q_moe` -- two kernels, two accumulation orders, so a 1-token
     decode and an n-token verify batch of the same MoE matmul were not
     bit-identical.  The dense half of this was fixed earlier the same day (dense
     `MUL_MAT` rows always ksplit); the MMID half was still open
     ("`MUL_MAT_ID`/MoE keeps the item-split").  **Fixed**: route all `MUL_MAT_ID`
     through the MoE kernel -- it is column-generic (one warp per token column;
     `n_groups` and `warp_reduce_sum` depend only on `warp_size`), so decode and
     verify now share one path.  **+6.2% MoE decode** (tg128 95.62 -> 101.52),
     +1.4% pp512 (4790.6 -> 4858.6); dense 27B flat (tg128 31.95 -> 32.00,
     pp512 2021.9 -> 2033.5).
  2. **The fused shared-expert epilogue is not bit-exact with the unfused chain**:
     its gate dot uses `shexp_gate_sigmoid`'s own reduction order (not the
     standalone mmvq order), and its epilogue multiply was contracted into an FMA.
     The FMA is now removed (`__fmul_rn`, one rounding, matching the separate MUL
     kernel) -- necessary but not sufficient while the gate reduction differs.
     Making the whole window bit-exact needs the gate dot to reproduce
     `mul_mat_vec_q`'s order; scoped as future work.  The fusion is worth **+3.1%
     MoE decode** (101.5 vs 98.5 t/s), so it stays ON by default behind a
     first-class kill-switch: **`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`**.

  Verified: with the kill-switch set (plus fix 1) qwen35moe decode is
  **bit-identical** to the verify batch (`bd138ad2` both -- the W=3 value, so the
  decode path moves and the reference is preserved); by default both hashes are
  unchanged from the previous tip (no regression).  MoE MTP gate unchanged
  (acceptance 0.58378 = the canonical baseline exactly; per-pos 0.785/0.575/0.382;
  draft 130.9 vs plain 91.6 t/s).  Dense gates unchanged (`none == n_max 3 ==
  n_max 7` = `13acc229`; `n_max 8` still divergent = cause B).
  `test-backend-ops -o GATED_DELTA_NET` 2/2 OK.
  **Accepted residual**: by default the MoE decode/verify pair is still not
  byte-identical -- MoE is exempt from that gate by
  `benchmarks/mtp-adaptive-methodology.md` rule 3, and the gate it *is* held to
  passes.  Canonical chain re-cut: block 13 `c43beca1b` -> `855515420`, block 14
  `daf32f804` -> `389c5341f`, tip **`389c5341f`**, net tree **`928852cdc`**;
  clean-apply strict 15/15 `git am`, zero whitespace warnings, applied tree ==
  canonical.  All temporary instrumentation reverted.

- **Correction: `GGML_CUDA_ALLREDUCE=nccl` was never a bit-identical reference under `-sm tensor` (2026-09-11).**
  Several docs used "hybrid vs RCCL coherence IDENTICAL" as a validation gate
  (`AGENTS.md`'s verify recipe, `patches/README.md` block-13 note, `RUN.md`).
  It does not hold: the internal AR pipeline always BF16-round-trips
  (`GGML_CUDA_AR_BF16_THRESHOLD` defaults to 1) while the NCCL path reduces
  *small* tensors in FP32 ("Reduces as FP32 for small tensors and BF16 for
  large", `allreduce.cu`), so the two backends differ by design.  Measured
  2-GPU `-sm tensor`, 27B Q8_0, 300-token greedy `--spec-type none`:
  text `6e8ccd25` (hybrid) vs `6129e077` (nccl), and the token-0 logits differ
  in the decode/verify band (W=6: `a4817ee6` vs `73ff91bf`).  The gate only
  holds where no cross-device reduction happens at all -- 1 GPU and
  `-sm layer` both gave `8fd24746` under either backend, because the AR is
  never reached.  Past records that quote the gate (e.g. `BASELINE.md`'s dated
  validation lines) are left as written per the dated-record policy; they were
  probably true at the text level for the split mode used, or a near-tie
  collision.  Docs corrected: `AGENTS.md` (verify recipe -- now says
  "smoke comparison only", with the measurements), `patches/README.md`
  (block-13 note), `wip/sm-tensor-plain-vs-spec/RUN.md` (the gate list).
  Not a correctness bug: the default (hybrid) path is self-consistent, which is
  what the n_max sweep gates.  It is a reminder that **text equality is
  evidence for purity, never evidence against divergence** -- the same trap as
  the 3-GPU `n_max = 8` false negative recorded in the entry above.

- **Block 12 amendment: verification matrix completed, and the boundary is `W = 8` / `n_max = 7`, not `n_max = 8` (2026-09-11).**
  Completes the entry above with the full per-configuration probe matrix and a
  correction to how the boundary is established.  The verified guarantee is
  **`--spec-draft-n-max <= 7`** (an 8-token verify batch) on the dense 27B with
  MTP, for 1 GPU, 2-GPU `-sm layer`, 2-GPU `-sm tensor` and 3-GPU
  `-sm tensor` alike; the first violating depth is `n_max = 8` (a 9-token
  batch).  The target verifies the drafts *plus* the last committed token, so
  `K = n_max + 1` -- `n_max = 8` is a 9-token batch, one past the designed
  `Q->ne[1] > 8` limit (cause B).
  Raw-logit probe (27B Q8_0, `RS = 0`, P = 256), `W = 1..8` -> `W = 9`:
  `4089b4d4` -> `72af52db` (1 GPU); `4089b4d4` -> `72af52db` (2-GPU `-sm layer`);
  `a4817ee6` -> `b059daa6` (2-GPU `-sm tensor`); `91434ea9` -> `bc3faabd`
  (3-GPU `-sm tensor`).  Uniform: bit-identical through `W = 8` everywhere,
  divergent at `W = 9` everywhere.  1 GPU and `-sm layer` share a hash because
  layer splitting changes no kernel; tensor splitting is the only configuration
  with different numeric paths (and the only one cause A could affect).
  **Method correction:** the 3-GPU 300-token *text* gate at `n_max = 8` matched
  the plain run (`5037ef2e` both) even though the logits had already diverged --
  no greedy near-tie flipped inside that window.  Text equality is evidence for
  purity, never evidence against divergence; boundaries must be established with
  the probe.  (This is the near-tie rarity noted in
  `wip/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.)
  Cause B is left as-is by maintainer decision: correctness through `W = 8`
  is already far beyond what upstream delivers (upstream's CPU path diverges at
  the first width step, `W = 2`), and removing it would give up WMMA for
  9..N-token batches.  The gain from the block-12 fix (+12% MTP) is retained.

- **Block 12 amended: the hybrid all-reduce's size-based dispatch changed the reduction algorithm with the batch width (2026-09-11).**
  `ggml_backend_cuda_comm_is_small()` sent reductions below a per-device-count
  element count to the internal host-staged pipeline and everything above it to
  NCCL.  The two paths are **not bit-identical** (different summation order;
  the internal path always does the FP32->BF16 round-trip).  Under `-sm tensor`
  the reduced tensors scale with the batch width (`ne = ne0 * n_tokens`; ne0 =
  5120 on the 27B), so with the old 2-device value 32768 a **7-token**
  speculative verify batch (35840 elements) was reduced by NCCL while 1..6-token
  decode stayed on the internal pipeline: the same logical reduction, a
  different algorithm, purely because the batch got one token wider.  This -- not
  the GDN, and not MMVQ -- is what the earlier entries in this log called a
  pre-existing `n_max >= 6` divergence for 2-GPU `-sm tensor`.  (A second,
  *separate and deliberate* boundary remains at `W >= 9`: the FA launcher's
  tile-vs-WMMA switch at `Q->ne[1] > 8`, which caps the guaranteed range at
  `n_max <= 7` by design.  `GREEDY-PURITY.md` section 11 now states both causes
  and the per-configuration ranges.)
  **Fix:** raise the 2-device crossover 32768 -> 131072 (the 3-device value).
  The largest verify batch (`--spec-draft-n-max 16` -> 17 tokens = 87040
  elements) stays well under it, and still far below the internal pipeline's own
  1 MB (262144 element) cap, so nothing is pushed off the fast path.  Only
  7..25-token tensors change path; one-token decode and prefill (>25 tokens) are
  untouched.
  **Evidence** (27B Q8_0, 2-GPU tensor, `-ts 1/1`, probe/`llama-cli`):
  `GGML_CUDA_ALLREDUCE=internal` (one algorithm for every size) makes probe
  `W = 1/6/7/8` all `a4817ee6`; the fix does the same, with `W <= 6` keeping
  their previous hash, i.e. plain decode is bit-unchanged and only `W = 7,8`
  move onto the internal pipeline.  Text `none == n4 == n6 == n7` (`6e8ccd25`;
  previously pure only to `n_max 4`/5).  1-GPU `W=1 == W=8`; 3-GPU text
  `none == n6`; GDN K-independence `RS=6 W=6 == RS=0 W=1`; determinism `W=7`
  twice identical.
  **Perf is a win on both axes:** MTP `n_max 6` acceptance 0.509 -> 0.533 and
  63.6 -> 71.3 t/s (+12.1%); `n_max 12` 51.7 -> 58.0 t/s (+12.2%); llama-bench
  pp512 2009.08 -> 2004.12, pp4096 1905.00 -> 1907.41, tg128 31.92 -> 31.97
  (all unchanged within noise).
  **Localisation method** (temporary instrumentation, since reverted): a
  backend-side per-node digest dump in `ggml_backend_cuda_graph_compute`, gated
  on a phase file, reading each tensor through its own `nb[]` strides after a
  device sync.  `cb_eval` is unusable for this (it changes MoE numerics and
  aborts in the meta backend under tensor split), and `ggml_backend_tensor_get`
  flattens from the view's base pointer ignoring `nb[]`, which alone produced a
  field of false positives.  The dump showed the layer-0 GDN chain bit-exact and
  the first divergence exactly at the first cross-device reduction
  (`attn_residual-0`), whose buffer the meta backend rewrites in place between
  the producing MUL_MAT and its consumer.
  Canonical: block 12 `eec68b2ad` -> `cac14423e` (blocks 13/14 re-cut: `070096e16`
  -> `c43beca1b`, `30d119ea9` -> `daf32f804`), tip `daf32f804`, net tree
  `10f94d635`; clean-apply strict 15/15 `git am`, zero whitespace, applied tree
  == canonical; blocks 00-11 bodies byte-identical.

- **Block 02 amended (final form): the whole-batch K-independent chunked GDN prefill — free, no tail, no gate (2026-09-11).**
  Follows the KTAIL=16 entry below, which it supersedes.  Option B from
  `wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md`: instead of *sharing a
  sequential tail* between the plain (`K == 1`) and MTP (`K > 1`) prefills, both
  paths now make the **same call** — a batch with more than `max(K, 16)` tokens
  is chunked **whole**, exactly what `K == 1` already did, and anything smaller
  stays on the sequential kernel.  No tail, so the previous -0.3..-0.8 % tail
  cost goes to **zero**: 27B Q8_0 1 GPU pp512/2048/4096 = 1385.3/1356.4/1328.2
  vs 1384.7/1355.0/1327.8 for the old K-dependent boundary (parity), tg
  unchanged.  A batch larger than `max(K, 16)` cannot be a verify batch (those
  decode `<= K` tokens) and is never rolled back into, so its K snapshots are
  skipped; every verify batch keeps them.
  **Guard added** (the invariant is empirical): `llama_memory_recurrent::seq_rm`
  tracks the last batch's per-seq token count and logs a once-only warning if a
  rollback ever crosses that boundary.  Measured: 449 rollbacks over llama-cli
  `draft-mtp` n_max 1/4/8/16 + 20 in llama-server `--cache-reuse`, all preceded
  by a `<= K`-token batch, 0 warnings.
  **Removed**: `GGML_CUDA_GDN_ALIGN_BOUNDARY`, the `align_boundary` variable and
  both K-dependent branches (~118 lines) — they were unreachable with the gate
  ON, and the opt-out is superseded by `GGML_CUDA_GDN_CHUNKED=0`, which is
  *both* correct (all snapshots written) and bit-identical plain-vs-MTP.
  **Gate re-verification** (all corrected against the new build): 27B 2-GPU
  tensor `none == n1 == n4 == n5` (`6e8ccd25`), 3-GPU tensor `none == n4`,
  1-GPU 4B probe `671d6096`, 27B prefill probe `W = 1/3/5/6` all `a4817ee6`
  with `RS=6 W=6 == RS=0 W=1` (prefill now K-independent), `test-backend-ops -o
  GATED_DELTA_NET` OK, 3x determinism check identical.
  **Correction (important):** the `n_max <= 15` purity claim in the entries
  below — and in every doc — was **wrong**; it was never validated past
  `n_max = 4`.  The real `none == draft-mtp` range is **`n_max <= 5`**, and the
  cause is a **pre-existing** multi-token-verify-batch MUL_MAT dispatch
  difference (identical divergence pattern on the delivered KTAIL=16 build;
  pure at `RS=0` up to `W = 6`, breaks at `W = 7`, again at `W >= 9` where MMVQ
  hands over to MMQ).  Upstream master is affected too and *worse*: on upstream
  `9cf3bf256` (CPU, 4B) `W = 1` already differs from `W >= 2`.  Docs corrected
  (`GREEDY-PURITY.md` §11, `benchmarks/mtp-adaptive-methodology.md` rule 4,
  `patches/README.md`, `AGENTS.md`); root-causing it is
  `wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.
  Canonical re-cut: block 02 `63f8ab023` -> `6e81ed5ed`, tip **`30d119ea9`**,
  net tree **`29714ad1f`**; clean-apply strict 15/15 `git am`, zero whitespace,
  applied tree == canonical.

- **Block 02 amended: GDN alignment tail shortened to `KTAIL=16` (2026-09-11). — SUPERSEDED by the entry above.**  The aligned
  boundary's cost is entirely its sequential tail (which writes the K rollback snapshots), and the
  tail was 64 — ~16x longer than the default `--spec-draft-n-max 3` needs.  `KTAIL=16` covers
  `K <= 16` / `n_max <= 15`, including adaptive MTP's recommended `n_max = 12`; for deeper drafts the
  new `K > 16 ? K : 16` floor keeps the snapshots exact (those reproduce the pre-alignment `K > 1`
  boundary, i.e. correct-but-not-bit-identical, instead of reading stale snapshot slots).  Measured
  (27B Q8_0, 1 GPU, interleaved `-r 5`): `KTAIL=64` ≈ -1.5 %, **`KTAIL=16` ≈ -0.3..-0.8 %**,
  `KTAIL=8` ≈ 0; decode unchanged.  Bit-identity re-verified with `KTAIL=16`: 27B 2-GPU tensor
  probe W=1/3/5 and text `none == n1 == n2 == n4` (`e386b50d`), 3-GPU tensor `none == n4`, 4B 1-GPU.
  Canonical re-cut: block 02 `d60bb52ef` -> `63f8ab023`, tip **`30d119ea9`**, net tree
  **`29714ad1f`**; clean-apply strict 15/15 `git am`, zero whitespace, applied tree == canonical.
  Follow-ups (free GDN prefill alignment + the MoE batch-width residual) written up in
  `wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md`.

- **Block 02 amended: `GGML_CUDA_GDN_ALIGN_BOUNDARY` flipped to default ON (opt-out), 2026-09-11.**
  The K-independent chunked-GDN boundary is now enabled by default (`GGML_CUDA_GDN_ALIGN_BOUNDARY=0`
  opts out and restores the K-dependent boundary).  This is the second of the two independent fixes
  required for `--spec-type none == draft-mtp`: with the block-13 dense-MMVQ alignment in place, the
  default is `none == n1 == n2 == n4` on 27B 2-GPU tensor (`5037ef2e`), 3-GPU tensor (`f60b79d0`),
  2-GPU layer and 1-GPU (`7d566fee`), and on the 4B 1-GPU probe.  Cost of the default
  (27B Q8_0, 1 GPU, llama-bench, `-r 5`, two alternating runs): pp512 1393.7/1384.5 ->
  1363.6/1363.8 (**-1.8 / -1.5 %**), pp2048 1362.2/1358.3 -> 1337.3/1338.1 (-1.8 / -1.5 %),
  pp4096 1329.5/1328.5 -> 1309.2/1309.6 (-1.5 %); decode unchanged (tg128 20.43 -> 20.40).  The
  maintainer accepted the prefill cost to close the divergence.  Canonical chain re-cut: block 02
  `38641280b` -> `d60bb52ef`, tip **`33ccf7e28`**, net tree **`31e153fe3`** (later re-cut again for KTAIL=16:
tip `30d119ea9`, tree `29714ad1f`); clean-apply strict 15/15
  `git am`, zero whitespace warnings, applied tree == canonical.

- **Block 13 amended: dense decode/verify MMVQ kernel alignment (`mmvq.cu`, 2026-09-11).**
  Closes the remaining batch-width half of the `-sm tensor` plain-vs-spec divergence.  Root cause:
  the block-13 `ncols_dst == 1` dispatch kept **dense** rows with `K < 4096` on the item-split/rpb
  kernel while the ncols 2..8 dispatch (and `K >= 4096` ncols==1) unconditionally use the ksplit
  kernel; the two accumulate K in different orders, so a single-token dense `MUL_MAT` is not
  row-identical to the same row in a 2..8-token verify batch (~1e-6 at the first divergent
  projection, amplified by the recurrent GDN).  Visible as `--spec-type none` != `draft-mtp` on
  small dense models (`n_embd < 4096`, e.g. Qwen3.5-4B, the cheap 1-GPU repro) and under
  `-sm tensor` on any model whose per-GPU K shard drops below 4096 (Qwen3.8-27B 5120 -> 2560).
  Fix: `!has_ids || ncols_x >= 4096` — dense rows always ksplit; `MUL_MAT_ID`/MoE keeps the
  item-split (its multi-token kernel is `mul_mat_vec_q_moe`).  Verified per-process (token-0 logit
  hash, callback-free): 4B 1-GPU and 27B 2-GPU-tensor W=1/3/5 bit-identical (was 0.133 on the
  27B).  Perf neutral (4B/27B/MoE-A3B within noise, MoE tg128 95.66 -> 96.02); MTP gates
  0.487/36.5 (dense) and 0.675/153.1 (MoE); `GATED_DELTA_NET` 46/46; hybrid-vs-NCCL coherence
  identical.  **The default-config `-sm tensor` text equality additionally requires the block-02
  `GGML_CUDA_GDN_ALIGN_BOUNDARY=1` gate** (K-dependent chunked-GDN prefill boundary); that gate
  stays opt-in because it costs ~2-2.6% prefill.  Canonical chain re-cut: block 13
  `fc7f52f96` -> `029b07b30`, tip **`27bd754b6`**, net tree **`c0775c33c`**; clean-apply strict
  15/15 `git am`, zero whitespace warnings, applied tree == canonical.  Record:
  `wip/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`; block-13 notes in `patches/README.md`.

- **Block 02 amended: opt-in K-independent chunked-GDN boundary (`GGML_CUDA_GDN_ALIGN_BOUNDARY=1`, 2026-09-11).**
  Fixes the fork-only plain-vs-spec divergence found during the gfx1151 issue-#25 validation (the issue
  #25 *follow-up*): the chunked GDN prefill had a **K-dependent** chunk/sequential boundary (plain
  `K == 1` chunked the whole prompt; MTP `K == n_max + 1` chunked `n_tokens - K` + a K-token tail), so
  the post-prefill SSM state depended on `n_rs_seq` and `--spec-type none` disagreed with `draft-mtp`
  (greedy near-ties flipped).  The amendment adds a gated third branch that chunks `n_tokens - 64` and
  runs the sequential kernel over the last 64 for both `K == 1` and `K > 1`, giving one boundary and one
  state; the tail also emits the K snapshots (rollback <= 63 exact), and `n_seqs > 1` keeps the old
  whole-ubatch path.  **Default OFF** — the fork's existing boundary is deliberate and ~1.1-1.2 % faster
  prefill; the gate only guards the two existing branch conditions, so the default output is
  **byte-identical** (`d9bf6850`), while with the gate on `none == n2 == n4` (`1a9ef0a1`, which also
  equals the `GGML_CUDA_GDN_CHUNKED=0` reference on the short prompts).  gfx1201 probe (`RS=from_w`,
  P=256): `W1-W3/W3-W5 = 0.136693/0.182106` default (unchanged) -> `0.000000/0.000000` gated;
  `test-backend-ops -o GATED_DELTA_NET` 46/46 in default, gated and gated+fp32.  Record:
  `wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md`.  Canonical fork rebuilt at `9113cc188`,
  block 02 (`5cbfbafd9` -> `38641280b`) amended by rebase, new tip **`7b79930b2`**, net tree
  `fcf3e4bb7`; clean-apply **strict 15/15 `git am`**, zero whitespace warnings, applied tree ==
  canonical.  A separate `-sm tensor` (2/3-GPU) plain-vs-spec divergence — independent of GDN and of
  this gate — is documented there as an open follow-up (the server's 3-GPU tensor-split config is
  affected).

- **Block 00 (structural and architecture fixes) added; the set is now 15 patches and the masked-V
  freed-cell fixes are re-homed (2026-09-10).**  A new first block, `patches/0000`, holds baseline-level
  fixes every later block builds on:
  1. **FA small-batch KV-split width invariance (issue #25).**  `launch_fattn`'s non-stream-K
     `parallel_blocks` heuristic keys off `ntiles_dst`, which is a function of `Q->ne[1]`, so
     single-token decode (`n_q = 1`) and speculative verify batches (`n_q = 3`, `5`, …) chose different
     KV splits, fed different partial sums into the online-softmax/PV combine and produced different
     logits; greedy near-ties then flipped, so MTP `--spec-draft-n-max 2` and `4` streamed apart.  The
     heuristic now evaluates `ntiles_dst` as if `n_q == 1` for every `n_q <= 8` (prefill unchanged).
  2. **Vulkan masked-V / freed-cell fixes** (`flash_attn_cm1.comp`, `flash_attn.comp`): dead columns
     never read V.  These are baseline shaders, so they belong in the structural block.
  The **HIP** masked-V fixes do **not** belong in block 00: the `fattn-tile.cuh` half uses the native
  bf16 PV staging (`V_k0`/`KQ_k`/`nv_bfloat162`) that **block 03** introduces, and the
  `fattn-mma-f16.cuh` half fixes the same class of leak on that path — so, per the maintainer, both HIP
  halves were **moved into block 03** (the earliest block that exercises the leaking code).  Block 14 no
  longer carries any masked-V/freed-cell hunk.  The net tree is unchanged from the previous regeneration
  (`26690e4d9`).  The block-15 attention-memory campaign is unaffected and remains staged in
  `beta/block-15-campaign-wins/`.
  Layout: `0000` = block 00, `0001`–`0014` = the old blocks 01–14 (renumbered by
  `git format-patch --start-number 0`, so the file prefix still equals the block number; the subjects
  read `[PATCH 00/14]`…`[PATCH 14/14]`).  Canonical fork rebuilt at `9113cc188`, tip **`505637d6e`**;
  `scripts/apply-all.sh` and `scripts/make-patches.sh` updated (15 blocks, `0000` included);
  `rdna-boosts-all.patch` regenerated.
  Validation (3× gfx1201, ROCm 7.14): clean-apply sim → strict **15/15 `git am`, zero whitespace
  warnings**, applied tree `26690e4d9` == canonical; issue #25 → `--spec-draft-n-max 2 == 4` on 2-GPU
  p0/p2/p3 and 3-GPU p0, `draft-mtp-adaptive` == both; plain decode (`--spec-type none`) byte-identical
  to the pre-block-00 canonical on 2-GPU and 1-GPU; MTP acceptance gate holds (dense 0.479, MoE 0.669,
  MTP >> plain both).  A `structural-fixes` branch (block 00 + blocks 01–14, based directly on
  `9113cc188` = the fork's master) was pushed to the personal fork for the gfx1151 investigation; the
  upstream-PR candidate `upstream/UPSTREAM-PR-fa-kv-split-width.{patch,md}` was filed under `upstream/`.

- **Block 15 un-promoted from the delivery — it belongs only in `beta/block-15-campaign-wins/`
  (2026-09-10).**  Block 15 was promoted into `patches/0015` by mistake; the maintainer never
  approved cutting it as a delivery patch.  The delivery is a **14-patch set** again
  (`patches/0001`-`0014`, canonical tip `ff2b35f49`), `scripts/apply-all.sh` and
  `scripts/make-patches.sh` are back to 14 blocks, `rdna-boosts-all.patch` is the 14-block net,
  and the docs/headers no longer present Block 15 as delivered.  The block-15 work (including
  the 2026-09-10 V5 and RDNA3_5/gfx1151 amendments) continues to live only in
  `beta/block-15-campaign-wins/block-15-campaign-wins.patch` and is applied manually on top of
  the 14-block tree, pending the maintainer's promotion go-ahead.  The `0001`-`0014` bodies are
  unchanged from the promoted set; only the `From <sha>` line and the `[PATCH NN/15]` →
  `[PATCH NN/14]` series count differ.  Clean-apply sim: fresh worktree at `9113cc188` +
  `apply-all.sh` → strict 14/14 `git am`, zero whitespace warnings, applied tree `6ce36849` ==
  the canonical 14-block tree.  (The dated entries below that say "cut" / "15-patch" record the
  promotion as it happened; this entry reverses it.)

- **RDNA3_5 (gfx1151) validation of the 14-block delivery + the beta block-15 patch; V3 iGPU enablement + multi-stream
  guard folded into the beta block-15 patch (2026-09-10, single Strix Halo, ROCm 7.14).**  The first
  single-device iGPU run of the delivery (Radeon 8060S, `VMM: no`, 1 device).  Block-14
  masked-V fixes, V3 derived mask, V4 native q8_0 and V5 native bf16 were exercised with
  a BF16 KV cache in both arm states, per the sign-leak campaign matrix.  Two V3
  regressions found and fixed as a dated amendment to the **beta** block-15 patch (the delivery
  stays 14 patches; beta patch tip `377f8e790`):
  1. the derived-mask probe rejected `GGML_BACKEND_DEVICE_TYPE_IGPU`, so V3 was silently
     disabled on the HIP iGPU and its ~800 MiB compute + ~800 MiB host win was lost;
     `ggml_backend_dev_is_cuda()` / `ggml_backend_dev_implements_kq_derived()` now accept
     `IGPU` (ROCm/CUDA reg name still required);
  2. `n_seq_max > 1` aborted context creation in `ggml_flash_attn_ext_add_kq_derived`
     (`GGML_ASSERT(tok_lo->ne[0] == a->src[0]->ne[1])`): `build_attn_mha` derives the
     stream count from `k->ne[3]` (the cache's `n_stream` == `n_seq_max`), while
     `kq_mask_derivable()` only checked `ubatch.n_seqs_unq`; it now rejects
     `n_stream != 1`, so a multi-slot context keeps the packed mask (no abort) and a
     single-stream context keeps the win.  `llama-server --parallel 4`, which aborted on
     the pre-amendment tree, now serves and passes the 16-run gate.
  Validation on the amended tree: reserves reproduce the RDNA4 block-15 numbers exactly
  (4B ctx 204800/ub 2048 V3 −799.20 compute / −799.21 host, V5 bf16 968.86 → 256.86,
  V4 q8_0 1001.13 → 257.13; 27B f16/bf16/q8_0 488.86 / 1072.86→488.86 /
  1121.13→489.13; Flash-Next W on 3251.39/63.69 indexer 318.76, W off 6690.40/1262.70
  indexer 956.26).  Determinism: 14 ROCm + 7 Vulkan gate runs PASS 16/16, V3 on vs off
  byte-identical over 7 × 2064 cells, bf16 arm on/off (V5), q8_0 arm on/off (V4) and
  q4_0 arm on/off byte-identical; bf16 vs f16 differs only by cache precision.  Isolated
  probes clean in both arms on both backends (ROCm bf16 34/34, f16 36/36, Vulkan
  bf16/f16 36/36; only the documented deterministic live-cell bf16 diag ≤1.1e-13).
  `test-backend-ops` FLASH_ATTN_EXT 4596/4596 ROCm0 (FA_ALL_QUANTS=OFF; 5 derived cases
  OK) + 7859/7859 CPU, `test-alloc`/`test-batch-alloc` pass, W4 repro 16.00 MiB.  MTP
  acceptance identical V3 on/off and arm on/off (27B 0.79762, Flash-Next draft 0.52727).
  Arm cost on gfx1151 is *lower* than RDNA4 — V5 −0.4…−0.9 % prefill, V4 **+2.6 %** at
  pp20480, decode ±0.1 % (the large MALL absorbs the interleaved-view re-reads); V3
  ~−3.2 % pp20480.  Clean-apply sim: fresh worktree at `9113cc188` + `apply-all.sh` →
  strict 15/15 `git am`, zero whitespace warnings, applied tree `6f5d23b5` == amended
  canonical.  Block-13 fused MoE re-check on Q3_K_M: the isolated
  `GGML_CUDA_DISABLE_MOE_MMQ_FUSION` delta is ~0 on this build (fusion fires, coherence
  holds, decode untouched) — absolute prefill is ~10–13 % above the 2026-09-05 record,
  consistent with the 2026-09-06 model-neutral Strix folds capturing the same work.
  Raw matrix: `wip/strix-halo/GATE-2026-09-10-block15-rdna35.md`.

- **V5 native bf16 K/V folded into Block 15 (opt-in, same switch as V4) — D12 closed
  (2026-09-10).**  The bf16 lever is implemented, validated and packaged as a **dated
  amendment to block 15** (`patches/0015`, canonical tip `f5ab5350b` on `9113cc188`;
  the amendment touched `0015` only — `0001`-`0014` stayed byte-identical).  A bf16
  KV cache no longer needs the F16 staging scratch: bf16 and f16 tiles have the same
  byte layout, so the MMA loader converts each 16-byte staged chunk in registers
  (`__float22half2_rn(ggml_cuda_cast<float2>(bf16x2))`, bit-identical to the
  launcher's `ggml_get_to_fp16_cuda(GGML_TYPE_BF16)`) instead of copying from the
  scratch, and the scratch sizing + whole-cache conversion pass are skipped for that
  operand.  The per-operand staging source is now one shared type code
  (`fattn_kv_native_t{FATTN_KV_NATIVE_NONE,Q8_0,BF16}`, subsuming V4's flags), so the
  launcher, `get_alloc_size` and the kernels cannot disagree.  Scope per D10: the F16
  fragments/`cp_async` design is untouched (no bf16 WMMA fragments).
  **Measured (ctx 204800, bf16 KV, arm on vs off):** 4B ub 2048 968.86 -> **256.86**
  MiB/GPU (== the f16 cache; ub 1024 884.82 -> 128.82, ub 512 842.80 -> 64.80), 27B
  1072.86 -> **488.86** (ub 512 868.80 -> 122.80), gemma-4-E4B 1062.89 -> **404.89**,
  gemma-4-31B 2068.89 -> **716.89**; qwen4exp unchanged (f16 == bf16 == on/off there,
  its FA path never staged bf16) and its q8_0 control reproduced 3251.39/63.69
  exactly, confirming the refactor left V4 alone; ub 8 (TILE/verify) 8.09 either way.
  **Cost** (interleaved same-binary A/B, off -> on, bf16): 4B -0.22 % (pp2048),
  +0.27 % (8192), -1.06 % (20480), -2.36 % (40960); 27B -0.76 % (20480); decode
  within 0.1 %.  The conversion itself is free (native bf16 staging is within 0.17 %
  of an *f16* cache) — the loss is the removed scratch, which is a dense, normalised
  copy of the cache view (the GQA heads are interleaved: `nb[1]` is 2048 B for a
  512 B row on the 4B), while the native path re-reads the interleaved view on every
  staging pass.  **Decision: opt-in via `GGML_CUDA_FA_KV_NATIVE` (default 0), i.e.
  the maintainer's explicit instruction for this item ("treat it similarly to V4 ...
  gated by the same environment variable"), consistent with D9.**
  **Gates:** same-seed text byte-identical (arm on vs off vs f16) on 4B, 27B,
  gemma-4-E4B (ISWA) and gemma-4-31B (ISWA), short + 3k/40k prompts, with V3's
  derived mask active (wins additive: -799.2 derived mask, -712.0 bf16 scratch on
  the 4B); MTP 27B 0.82716 and qwen4exp 0.44262 identical on/off (q8_0 references
  0.76744/0.44262 unchanged); `test-backend-ops` FLASH_ATTN_EXT 7859/7859 ROCm0+CPU,
  with the 2704 bf16 K/V cases (all head sizes incl. the 576/512 MLA
  `v_is_view_of_k` layout) and 365 q8_0 cases green in both arm states, identical case
  lists.  Re-validated end to end **from the delivered patches**: fresh worktree at
  `9113cc188` -> `apply-all.sh` strict 15/15 `git am`, tree identical to `f5ab5350b`,
  build, reserves/coherence/MTP/op-suite all reproduced.  One pre-existing
  unrelated full-build warning recorded in `TODO.md`
  (`llama-kv-cache.h:274` `-Wunused-private-field` for W3's `v_enabled`).
  Records: the V5 amendment section in `patches/README.md`, the outcome section in
  `wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`, the block-15 beta record.

- **bf16-native MMA K/V planned as the next essential follow-up (D12); two pre-existing findings
  recorded (2026-09-10).**  With Block 15 cut, the maintainer picked the bf16 lever as the one
  follow-up.  The executable plan is **`wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`**:
  measured before-state in the *delivered* tree, the mechanism with exact call sites, the design
  (keep the F16 fragments, `cp_async` the raw bf16 bytes into the same shared offsets — a 16-byte
  chunk is 8 elements either way — then convert the tile in place), the validation protocol and a
  three-way ship rule (expectation: **on by default**, unlike V4, because the `cp_async` pipeline is
  kept).  Also `HANDOVER.md` D11/D12 and the §8 prompt.
  **Before-state (ctx 204800, V3 on, f16 = reference):** 4B (1 GPU) ub 2048 256.86 -> **968.86**
  (+712.00), ub 1024 +756.00, ub 512 +778.00; 27B (3-GPU Meta) ub 2048 488.86 -> **1072.86** (+584.00),
  ub 512 +746.00; ub 8 (TILE/verify) **identical** at 8.09 MiB; `GGML_CUDA_FA_KV_NATIVE=1` (V4) changes
  no bf16 row (it is q8_0-only).
  **Finding 1 (pre-existing, documented not fixed): mixed K/V types fall off the GPU attention path.**
  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`, either direction) reserves `graph splits = 18` (vs 2),
  moves ~1.5 GiB into the host compute buffer and loses the FA scratch; 4B pp2048/tg128: `q8_0/q8_0`
  7924.47/98.94, `bf16/q8_0` 640.25/61.57, `q8_0/bf16` 1048.66/68.54, `f16/q8_0` 852.57/54.39.  So
  "bf16 keys + q8_0 values" is not usable today; same-type K/V is the practical choice.  Fixing it
  needs the FA kernels to accept a mixed `(type_K, type_V)` pair — larger than V3/V4, out of scope.
  **Finding 2 (pre-existing): gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter
  (`n_head_kv = 2` < 3 devices); maintainer's call (D11): **document only, do not fix** — small model,
  unlikely configuration; it runs on 1/2 GPUs and with `-sm layer`.
- **Block 15 cut (2026-09-10) — the attention-memory campaign wins;
the set is now 15 patches (block-15 tip `09a137566` on the canonical fork
rebuilt at `9113cc188`), beta-staged in `beta/block-15-campaign-wins/`.**
  The campaign (`wip/arch-independent-memory/`, `wip/qwen4exp/qsa-memory/`)
  was merged into one block by replaying the validated work-branch tree
  onto block 14, then re-validated **as a combination** (the per-win
  records did not carry over on their own).  Six wins, each with an
  environment A/B gate; **V4 is opt-in** (an *enable* switch) per the
  maintainer's rule of 2026-09-10 (a sub-2 % loss with a large memory win
  and no cheap fix ships opt-in):

  | win | mechanism | gate (default) | measured (ctx 204800, q8_0 KV, ub 2048) |
  |---|---|---|---|
  | W1 | QSA score chain: relu before the 4-D reshape + `n_blocks`-chunked `ggml_concat` assembly | `GGML_QSA_SCORE_MEM` (1) | qwen4exp 6690.40 -> 4450.40 MiB/GPU (ub1024 3346.50 -> 2274.35) |
  | W2 | derived QSA per-block bias + derived visibility; bias/mask no longer materialised; input-fill null guards (incl. the `llm_graph_input_attn_k` one) | `GGML_QSA_DERIVED_BIAS` (1), `GGML_QSA_DERIVED_VIS` (1), `LLAMA_QSA_SPARSE_FA` (sparse) | qwen4exp 4450.40 -> **3251.39** MiB/GPU, host 1262.70 -> **63.69** MiB |
  | W3 | keys-only QSA indexer cache (`v_enabled` in `llama_kv_cache`; no V tensor, no V-side op) | `LLAMA_QSA_KEYS_ONLY` (1) | indexer KV 956.26 -> **318.76** MiB/GPU |
  | W4 | ggml-alloc releases view sources whose views are never consumed (the uncounted-view leak) | none — a bug fix; `beta/block-15-campaign-wins/ab/w4-revert.patch` | repro 56.00 -> 16.00 MiB; no reserve change on any model |
  | V3 | derived kq mask: `GGML_OP_FLASH_ATTN_EXT` src[5..7] carry compact per-cell state and the MMA FA kernel derives visibility in-kernel; the packed mask tensor is still built in every graph and simply loses its consumer (so no model allowlist and no mis-served consumer) | `LLAMA_KQ_MASK_DERIVED` (1; `0` = packed) | 4B 1800.33 -> **1001.13**, 27B 1920.33 -> **1121.13** MiB/GPU; host -799.21; gemma-4-E4B/-31B (ISWA) -809.18/-811.17; scales as `n_kv x n_tps x 2 B` |
  | V4 | native q8_0 K/V in the FA kernels: dequantise during the shared-tile staging (16-byte chunk = 8 elements = a quarter q8_0 block) instead of staging a whole-cache F16 copy | `GGML_CUDA_FA_KV_NATIVE` (**default 0 = opt-in**) | 4B -> **257.13**, 27B -> **489.13**, gemma-4-31B -1224 MiB/GPU; qwen4exp unchanged |

  **The wins compose additively** — qwen4exp ub 2048: pristine 6690.40 ->
  W1 only 4450.40 -> W2 only 5491.39 -> W1+W2 3251.39 (W1 -2240, W2
  -1199, W3 -637.5/GPU, V3 -799, V4 -744/-632); both W gates off
  reproduces the pristine 6690.40/1262.70 exactly.  **Cost**: V3 -1.28 %
  prefill (4B pp20480/ub 2048, interleaved same-binary A/B) / +0.28 %
  (27B), decode -0.32 %/-0.15 %; V4 a further -1.85 % (4B) / -1.72 %
  (27B) prefill — the loss is the lost `cp_async` pipeline (a quantized
  source cannot be copied asynchronously; a 2-byte-access pass changed
  nothing), decode within noise, hence opt-in.

  **Combination validation (all on the merged tree, and then re-run from
  the delivered patches — see below):** reserve matrix on 4B (1 GPU), 27B
  (3-GPU Meta), gemma-4-E4B (1 GPU), gemma-4-31B (3-GPU) and qwen4exp
  (3-GPU) at ub 2048/1024/512 x V4 off/on — every number matches the
  per-win records; same-seed generated text **byte-identical** on all
  five models across every gate combination (V3 x V4 on the dense
  models; W1/W2/W3/V3/V4 — 7 configurations — on qwen4exp) at a short and
  a 40k-token prompt; adaptive-MTP gate **unchanged** (27B inline draft
  0.76744 (66/86, mean 3.28) in all four gate combinations; qwen4exp
  draft 0.44262 (54/122) in all six, **equal to the block-14 baseline**,
  and MTP stays +26 % over plain decode at ctx 32768); `test-backend-ops`
  FLASH_ATTN_EXT on ROCm0 (both V4 gates) and CPU, the six derived FA
  cases, VIEW/CONT/CPY/DUP/CONCAT, `test-alloc`, `test-batch-alloc`; the
  W4 revert restores `ggml-alloc.c` byte-identically to block 14.

  **Two things worth recording.**  (1) Re-validation caught a real wiring
  bug before the cut: the W3 gate was passed to `v_enabled` with the
  wrong polarity, so the indexer cache stayed keys-only-disabled (956.26
  MiB) while `LLAMA_QSA_KEYS_ONLY=0` enabled it — fixed and re-verified
  (`956.26 -> 318.76` on the default, `956.26` with the gate off).  This
  is exactly what the combination pass is for.  (2) A **pre-existing**
  bug was found (it reproduces on block 14=HEAD, so it is not a block-15
  regression): `gemma-4-E4B-it` on **3 GPUs with `-sm tensor`** aborts in
  the meta splitter (`ggml-backend-meta.cpp:1177`) on a FLASH_ATTN_EXT
  node whose K source has zero extent on one buffer, because `n_head_kv =
  2` is fewer than the device count (2 heads / 3 devices leaves one
  device with nothing).  It runs on 1 GPU, on 2 GPUs and on 3 GPUs with
  `-sm layer`; the 27B (4 KV heads) and gemma-4-31B (4/16) are
  unaffected.  Diagnosed by instrumenting the failing assert to print the
  op/tensor/split geometry (temporary change, reverted).  Left unfixed —
  out of scope for this block — and documented in `patches/README.md`.

  **Regeneration and delivery mechanics.**  The reference fork checkout
  (`~/llama.cpp`, branch `rdna-boosts`) had been rebased onto a master
  that is **two commits newer than the recorded fork point** (`f3f1a8f27`
  iGPU lazy-load default + `304665fe7` SYCL IQ-type-for-MoE, both
  2026-09-08/09, i.e. after `9113cc188`), so `format-patch
  9113cc188..tip` there would have exported those two upstream commits as
  patches 0001/0002 — a latent trap for any future regeneration.  The
  patches were therefore regenerated from a **canonical fork rebuilt at
  `9113cc188`** via `scripts/apply-all.sh` (strict 15/15 `git am`, zero
  whitespace warnings), and the resulting tree was verified identical to
  the validated tree except for the 3 files of those two upstream commits
  (`ggml-sycl` x2, `src/llama-model.cpp` — outside the validated paths).
  The delivered `0001`-`0014` files were kept byte-for-byte (the
  regenerated ones differ only in the `From <sha>` line and the
  `[PATCH NN/15]` series count, verified content-identical hunk by hunk);
  `0015-rdna-boosts-block-15-campaign-memory-wins.patch` is new.
  `make-patches.sh`'s default tip is now `09a137566`, the canonical
  block-15 commit (the local branch `block15-canonical` in the fork
  checkout keeps that chain alive).  `rdna-boosts-all.patch` = `git diff
  9113cc188..09a137566` (98 files).

  **Clean-apply simulation (the delivered artifact, end to end):** fresh
  worktree at `9113cc188` -> `scripts/apply-all.sh` (15/15 strict
  `git am`) -> fresh `gfx1201` Release build -> reserves (4B 1001.13 /
  257.13, 27B 1121.13 / 489.13, qwen4exp 3251.39 with the indexer KV at
  318.76), byte-identical coherence on 4B/27B/gemma-4-E4B/qwen4exp with
  every gate flipped, MTP 0.76744 / 0.44262, and the op suites — all
  green.

  **Upstream-drop check (2026-09-10):** GitHub was unreachable from this
  host (SSH key denied), so the check ran against the recorded upstream
  base `9cf3bf256`: the `ggml-alloc` unused-view release (W4), the
  keys-only indexer cache (W3) and the `llm_graph_input_attn_k`
  null-mask guard are all **still absent upstream** (the first two apply
  cleanly, the guard's call site is still unguarded while its own
  `can_reuse_impl` accepts a null mask), so Block 15 keeps every hunk.
  Re-check after the next `git fetch` before filing the `upstream/`
  candidates.

  **Upstream candidates A1/A2 prepared on a pristine master worktree
  (2026-09-10):** the `upstream/` backlog is now empty (four candidates,
  each with its own `.md` evidence):
  - **A1 `UPSTREAM-PR-kv-cache-keys-only`** (win W3): verified on
    unadulterated master `9cf3bf256` (CPU build, the real 3-shard qwen4exp
    IQ4_XS GGUF) -- the upstream indexer KV buffer is **72.00 MiB at ctx
    8192 (K 24.00 + V 48.00)** and drops to **24.00 MiB (K only)** with the
    patch; same-seed text byte-identical; `test-alloc` all PASSED,
    `test-batch-alloc` 0 failures.  The shape is worth noting: the store
    overrides the *key* head to the indexer size (128) but inherits the
    model's *value* head (256), so the dead V is twice the K it never
    accompanies.  Method note: the first upstream A/B was measured with
    `git apply -3` (which stages), so `git checkout -- .` did not revert it
    and both runs measured the patched tree; the `git reset --hard` re-run
    is the real unpatched number above.
  - **A2 `UPSTREAM-PR-attn-k-null-mask-guard`** (part of win W2): verified
    on master -- applies clean, compiles, byte-identical same-seed text;
    recorded in its notes as **hardening, not a live fix** (every upstream
    construction site builds a mask, and `can_reuse_kq_mask` itself
    dereferences it, so the guarded branch is unreachable upstream today).
    It is what the sibling `attn_kv` class already does and the prerequisite
    for a future null-mask feature.
  Both patches were apply-checked on pristine master (individually and
  together: 4 files, +18/-7); the master worktrees were reset afterwards.

- **Block-14 amendment (2026-09-10) — freed-cell KV handling moved from the
  host-side zeroing to kernel-side masked-V elimination; the gfx1151-only
  `zero_freed` host zeroing (2026-09-09 amendment) is REMOVED (block-14 tip
  `ff2b35f49` on `9113cc188`, regenerated 2026-09-10; blocks 01-13 patch
  files byte-identical).**  `src/llama-kv-cache.{cpp,h}` are back to the
  upstream state — no `zero_freed`/`rows_hw`/`sharers` wiring, no env
  `LLAMA_KV_ZERO_FREED`, no per-free GPU memsets; evicting a resident KV
  sequence is pure host cell bookkeeping again on every device.  In its
  place block 14 now carries the three **kernel-side** fixes that make the
  content of fully-masked (freed/stale) flash-attention cells unreadable,
  so the host workaround is unnecessary:
  - HIP `fattn-tile.cuh` (packed-bf16 PV path): zero the per-warp V
    register copies of rows whose P is +0.0 across the warp's columns
    before the bf16 dot.
  - HIP `fattn-mma-f16.cuh`: after each V-tile slice is staged in shared
    memory, zero the rows the mask tile marks blocked (-inf) for every
    query column of the block; one extra uniform barrier, masked path
    (`ncols2 > 1 || mask_h`) only; compile-time excluded for the
    `V_is_K_view` and NVIDIA-swizzled (`swz_V`) paths.
  - Vulkan `flash_attn_cm1.comp` + `flash_attn.comp` scalar path: never
    read V of fully masked columns (dead columns keep V = +0.0).
  All three are unconditional in their kernel paths (no arch/env gating) —
  generic correctness fixes for masked/freed FA cells (batch serving, KV
  eviction) active by default on every device.  Root cause (Strix Halo,
  gfx1151): WMMA f16 `x + (-0.0)` is inexact, so a masked column leaked
  the sign of whatever V its cell last held; the fix guarantees masked
  cells contribute exactly +0.0 at the multiply.  Validation on the
  gfx1151 box (ROCm 7.14-gfx1151 + Vulkan RADV), host zeroing disabled:
  16/16 identical-request determinism gates PASS on every KV cache type
  each backend's FA supports — ROCm f16/bf16/q8_0/q4_0 (plus ON==OFF
  bit-identical over 2064 cells/run), Vulkan also q4_1/q5_0/q5_1/iq4_nl;
  `test-backend-ops` FLASH_ATTN_EXT vs CPU 4591/4591 (ROCm) and
  7822/7822 (Vulkan); CPU same-seed greedy 51/64 tokens identical,
  divergence only at a near-tie (CPU non-FA vs GPU FA numerics);
  depth-16384 llama-bench decode tg128 within 0.05% of pre-fix, pp within
  single-run drift.  Full record:
  `wip/strix-halo/kvzero/RECORD-2026-09-09.md` +
  `wip/kv-sign-leak/HANDOVER-2026-09-09-mma-f16.md`.  Delivery:
  regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical) + `rdna-boosts-all.patch`; clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip `ff2b35f49`; final-tree rebuild (delta vs the
  validated kernel-fix tree = the llama-kv-cache revert only) passes the
  16/16 gate and no longer logs the freed-cell zeroing.

- **Block-14 amendment (2026-09-09) — freed-cell KV-zeroing gated to gfx1151
  (fork block-01 commit `7c4d9c4e0`, block-14 tip `27485f1ca`, 14 commits on
  `9113cc188`; previous tip `0f2b7a4e1` superseded).**  Block 14's
  `seq_rm`/`seq_keep`/`clear` row zeroing (freed KV cells kept at +0.0 as a
  masked-column guard for the gfx1151/Strix-Halo WMMA f16 `x+(-0.0)`
  inexactness, ported from the strix lineage commit aad5adb08f) is now
  **enabled only when a KV-cache buffer device description carries `gfx1151`**
  (env `LLAMA_KV_ZERO_FREED=0/1` overrides the auto detection).  Everywhere
  else the pre-block-14 behavior is restored: evicting a resident KV sequence
  is pure host cell bookkeeping again.  Reason: without the gate, freeing an
  N-token sequence issued ~48×N per-cell 512-byte memsets (ggml's
  meta/multi-buffer memset decomposes one per-layer zeroing call into one
  synced `cudaMemsetAsync` per cell across the GPU head-split sub-buffers,
  each ~30-60 µs), so replacing a ~13k-token KV stalled ~18-24 s before the
  new prefill began on multi-GPU RDNA4 (3x R9700 gfx1201; reproduced on a
  plain dense 4B model too — model-agnostic).  Verified: on gfx1201 the
  A/B stall is gone (identical workload 24.5 s -> ~6 s) and the zeroing-off
  determinism gate passes (16 + 8 identical greedy requests, per-position
  top-8 logprobs float64-compared — the same gate that found the leak on
  gfx11); on the gfx1151 Halo box the gate enables
  ("freed-cell KV row zeroing enabled (gfx1151)") and the 16-run control is
  unchanged.  Regenerated `patches/0014` only (blocks 01-13 patch bodies
  byte-identical); clean-apply sim at `9113cc188` strict 14/14 `git am`,
  zero whitespace warnings, applied tree == fork tip `27485f1ca`.
  Follow-up (open): develop a performant gfx1151 flash-attn kernel-side fix
  so the host-side zeroing can be removed entirely.

- **Block-01 refresh (2026-09-09) — adaptive MTP draft depth updated to the
  llama.cpp PR #27210 review head (fork block-01 commit `7c4d9c4e0`,
  block-14 tip `0f2b7a4e1`, 14 commits on `9113cc188`).**  Block 01 was cut
  from PR #27210 (author: stew675) at its `0994374fd` state; the PR then
  advanced through a maintainer review round (`8408cdabf` comment fixes +
  `d236d41a2`, the review-response changeset).  The block is now refreshed
  to the PR head `d236d41a2`, still delivered as **one squashed patch
  block** (`git diff 9113cc188..d236d41a2` = 15 files, 519+/35-, applied
  as the single block-01 commit; blocks 02-14 re-based on top untouched).
  Review-round content now in block 01: `common_params_speculative::
  has_mtp()` helper (arg.cpp/common.cpp/server-context.cpp/init result
  refactored through it); a new `accept_partial()` virtual +
  `common_speculative_accept_partial()` so a partial acceptance the
  context could not apply (checkpoint-restore path in tools/server and
  examples/speculative-simple) is reported once and the following replay
  round cannot feed stale draft counts to the adaptive controller
  (non-adaptive accept path unchanged); the adaptive depth reset moves
  ahead of the empty-prompt early return in `begin()`; `
  --spec-draft-n-min-adaptive` rejects values < 1 and is documented
  (docs/speculative.md, tools CLI/server READMEs); the invalid-range
  check is `GGML_ABORT` -> `std::runtime_error`; draft-mtp +
  draft-mtp-adaptive together are rejected (shared ctx_dft); the delta-
  net conv-state snapshot-bound rationale comment; stale "defaults to 2"
  test comment fixed (default is 3) + value-0 rejection case.
  Regeneration mechanics: canonical fork rebuilt at `9113cc188` from the
  previous set (am-tip `050ec89ce`), block 01 replaced in place by the
  squashed PR-head changeset, blocks 02-14 `git rebase --onto` (clean,
  no conflicts — blocks 02-13 touch no block-01 file, block 14's
  common/arg/common.h hunks are disjoint).  Tree verification: old-tip..
  new-tip delta is exactly the review changeset (13 files, 129+/70-, ==
  `0994374fd..d236d41a2`), every other file byte-identical; regenerated
  0002-0013 patch bodies byte-identical to the previous delivery, 0014
  refreshed only in index lines/hunk offsets for the 3 common files;
  regenerated 0001 diff body byte-identical to the PR head changeset.
  Verification (local 3x R9700, gfx1201, ROCm 7.14): clean-apply sim at
  `9113cc188` strict 14/14 `git am`, zero whitespace warnings, applied
  tree == fork tip; rebuilt `test-arg-parser` + `test-speculative-
  adaptive` pass; plain-decode same-seed coherence (seed 42/temp 0,
  Qwen3.5-4B-Q8_0) token-IDENTICAL to the known-good `050ec89ce` build.
  The refresh touches no GPU kernels and no non-speculative host decode
  path — all changes live in the MTP-typed/adaptive code, the option
  parser and comments/docs.

- **Re-base (2026-09-08) — delivery moved to upstream master `9113cc188`
  (block-14 tip `78e67a3d8`).**  Upstream moved 14 commits past the
  `050dde50c` fork point (server checkpoint eviction, Kimi-K3 recurrent
  rollback, chat-parser split, ggml_prec spec, metal/vulkan/opencl fixes,
  spec single-device meta-wrapper handling #28390, and — decisive for this
  re-base — `d4389a4dd`/PR #28604 which **reverted #24233**, the very
  change block 06 diverged from).  An `apply-all.sh` run against the fresh
  master tip failed at block 06 in a way even `git am -3` cannot fix: the
  upstream revert deleted block 06's pre-image, so the block's change is a
  no-op on the new base (nothing left for the patch to do).  Resolution:
  block 06 was reduced to a host-buffer **rationale marker** commit (6
  comment lines above the now-unconditional `integrated = false` in
  `ggml-cuda.cu`), keeping the 14-block structure and all downstream block
  numbers intact; block 14's quantized-KV tensor-split gate merged
  **additively** with #28390's single-device `SPLIT_MODE_TENSOR` warn in
  `src/llama-context.cpp` (both kept, in sequence; #28390's code comment
  shows the same single-device-no-meta-wrapper intent as block 07, so no
  semantic collision).  Content verification against the previous delivery
  (re-applied at `050dde50c`): blocks 01-05 and 07-13 are byte-identical;
  block 06 differs as designed; block 14 differs only in the
  llama-context.cpp resolution region.  Regenerated at `9113cc188`
  (`f84549d23..78e67a3d8`) and clean-apply re-verified (strict 14/14
  `git am`, zero whitespace warnings, applied tree == fork tip
  `78e67a3d8`).  Coherence verified on the Strix box (gfx1151, ROCm 7.14):
  llama-cli same-seed output IDENTICAL to the canonical `72f0ee944` build
  (tensor + layer split x f16/q8_0/bf16 KV, and a long-prompt run at depth
  16384), clean runtime diagnostics, and the dense adaptive-MTP gate green
  on the new build (draft acceptance 0.833 at acc/pos 0.944/0.833/0.722;
  draft-mtp 20.3 t/s vs plain 7.9 t/s on the same prose prompt; MTP
  same-seed byte-identical old-vs-new).  The previous `050dde50c`-based
  regeneration (`d65a96084..ce641322e`) is superseded; the pre-re-base fork
  chain is preserved at `backup-rdna-boosts-bfcc4be99` and the known-good
  `72f0ee944` binary under `/tmp/rdna-ref-bin/` (session-local).

- **Block-14 amendment (3rd on 2026-09-08) — quantized-KV tensor-split
  gate:** the `q4_1`-family KV cache types (`q4_1`, `q5_0`, `q5_1`,
  `iq4_nl`) aborted during the first graph reserve under multi-GPU
  `SPLIT_MODE_TENSOR` on gfx1201 (3x R9700) —
  `ggml-backend-meta.cpp:538 GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)`
  — on both dense qwen35 (Qwen3.6-27B) and qwen4exp (Flash-Next), with
  `f32/f16/bf16/q8_0/q4_0` KV and layer split passing.  Root cause is
  **upstream**: reproduced on pristine vanilla llama.cpp at the fork
  point `050dde50c` (identical assert, non-qwen4exp Qwen3.5-4B; also at
  1 GPU, since upstream wraps even a single device in the Meta backend)
  and still unfixed on current upstream master.  Tensor split forces
  flash attention, whose CUDA/HIP kernels read the quantized K/V cache
  natively only for `q4_0`/`q8_0` (plus the float types); for the
  q4_1-family types the attention subgraph is externalized into
  op-NONE graph leaves (MIRRORED split state) which collide with the
  AXIS-0 elementwise gate branch of the qwen35/qwen4exp gated attention
  at the `attn_gated` `MUL` — the meta splitter cannot reconcile
  MIRRORED x AXIS-0.  Fix: a context-creation gate in
  `llama_init_from_model` (`llama-context.cpp`, block-14-owned in the
  set) that rejects K/V types outside FA's native set with a clear
  error when the Meta device is actually in use (tensor split over
  >= 2 GPUs; the fork's single-GPU "tensor" mode skips the Meta wrapper
  and is untouched — upstream, whose 1-GPU mode also wraps Meta, gets
  the clean error too).  Validated 2026-09-08 on gfx1201 (3x R9700,
  ROCm 7.14): KV-type matrix on dense 27B Q8_0 + Flash-Next IQ4_XS
  (3-GPU tensor) — `f32/f16/bf16/q8_0/q4_0` generate;
  `q4_1/q5_0/q5_1/iq4_nl` and `k=q4_1 v=bf16` / `k=bf16 v=q4_1` fail
  cleanly (zero asserts, actionable message); layer split + q4_1
  Flash-Next 25.9 t/s (unchanged); qwen4exp derived-cache pool-gate
  byte identity holds (tokens identical with the pool skipped vs
  `GGML_CUDA_QSA_INDEXER_CACHE=1`); dense-27B same-seed coherence A/B
  (gate stripped vs applied on the same tree) byte-identical;
  test-llama-archs qwen4exp all OK (NMSE 1.01e-13).  Canonical fork
  rebuilt at `050dde50c` (am-commits `d65a96084..ce641322e`, block-14
  tip `ce641322e`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `ce641322e`, full build
  clean, coherence byte-identical to the validation tree).
 the 2026-09-07 local
  delivery (`9850143`: block-14 **derived-cache pool gate**, regen at
  fork tip `bfcc4be99`) had never been pushed; the 2026-09-08 lineage on
  `origin/main` (issue #18 MUL_MAT_ID pair-fusion layout gate + issue
  #19 moe_weighted_reduction float4 remainder, both folded into blocks
  13/14; the block-14 compiler-warning cleanup; the qwen4exp
  tensor-split HIP gate — regen tip `2f1dc384b`) had been authored from
  a clone without it.  The two block-13/14 regens touched disjoint
  source hunks, so blocks 13/14 now carry all of it: QSA quantized-KV
  decode gate, derived-cache pool gate, the issue-18/19 fixes, the
  warning cleanup and the tensor-split backend gate.  Canonical fork
  rebuilt at `050dde50c` (am-commits `7df708e66..72f0ee944`, block-14
  tip `72f0ee944`); set regenerated with `scripts/make-patches.sh`;
  clean-apply sim re-verified 2026-09-08 (14/14 strict `git am`, zero
  whitespace warnings, applied tree == fork tip `72f0ee944`).
- **Block-14 amendment (2nd) — qwen4exp tensor-split backend gate
  (2026-09-08):** follow-up to the Vulkan validation sweep: block 14
  had removed upstream's `case LLM_ARCH_QWEN4EXP: // TODO: fix
  test-llama-archs` from `llm_arch_supports_sm_tensor`, enabling
  qwen4exp tensor split for every backend.  That is validated on
  ROCm/HIP only (3x R9700, NMSE 9.87e-14 vs CPU); on backends that
  cannot run the fused QSA/HC/WS4 ops on-device (Vulkan, Metal, SYCL;
  NVIDIA CUDA untested) the CPU-fallback subgraphs leave the meta
  splitter unable to reconcile mirrored-vs-split operand states and it
  aborts at graph reserve (`ggml-backend-meta.cpp` `handle_generic`,
  e.g. the qwen4exp gated-attention `MUL` on Vulkan — `test-llama-archs`
  died at the qwen4exp Meta row).  The enablement is now `#ifdef
  GGML_USE_HIP`, restoring upstream's clean "not implemented" error /
  arch-test SKIP on all other builds.  Verified: Vulkan — full
  test-llama-archs sweep completes RC=0 (457 rows, statuses identical
  to upstream 050dde50c, qwen4exp Meta SKIP like upstream), qwen4exp
  single-device still OK (9.01e-08, roundtrip OK), llama-cli
  qwen4exp `-sm tensor` fails with the upstream message; HIP —
  qwen4exp Meta still OK 9.87e-14 (validated path unchanged).  Canonical
  fork rebuilt at `050dde50c`; block-14 tip `13719e3ca` →
  `2f1dc384b`; set regenerated; clean-apply sim re-verified (14/14
  `git am`, zero whitespace warnings, applied tree == fork tip).
- **Block-14 amendment — compiler-warning cleanup (2026-09-08):** the
  block-14 sources warned under the `build-llama-vulkan` (system clang
  16.2.1, `-Wall -Wextra`) and `build-llama-rocm-714` (ROCm clang)
  host builds.  Five warnings, all from block-14 code, fixed and
  folded into the block-14 commit:
  - `ggml.c` — unused `n_blocks` local in the `ggml_indexer_fill`
    builder (removed).
  - `ggml-cpu.c` — `-Wswitch`: the exhaustive CPU compute-forward
    switch had no case labels for the new `GGML_OP_INDEXER_SCORE` /
    `GGML_OP_INDEXER_FILL` ops (GPU-only fused ops; the CPU plan
    phase already aborts on them as "op not implemented" before
    compute, so the case is an unreachable `GGML_ABORT`, mirroring
    `GGML_OP_COUNT`).
  - `ggml-cpu/ops.cpp` — two `-Wunreachable-code-break` warnings: the
    `break` after the noreturn `GGML_ABORT("fatal error")` in the
    `HC_MIX`/`HC_COMBINE` CPU type dispatchers' default cases
    (dropped, matching upstream convention).
  - `qwen4exp.cpp` — `idx_cache` was narrowed to `bool`, making the
    documented `GGML_CUDA_QSA_INDEXER_CACHE=2` debug probe
    (`idx_cache != 2`) tautologically true (`-Wtautological-constant-
    out-of-range-compare`); restored to an `int` with the 0/1/2
    tri-state so probe-2 (pool read without the fill) is reachable
    again.  `-Wsign-compare` in the gfx-id sniff loop (`size_t`
    counter vs `ggml_backend_dev_count()`).
  No generated-code or runtime-behavior change in default configs.
  Verified: the four TUs compile warning-free with the exact
  build-vulkan flags; full Vulkan + ROCm 7.14 (gfx1201) builds clean
  on the re-applied sim tree.  Canonical fork rebuilt at `050dde50c`;
  block-14 tip moved `3529b3497` → `13719e3ca`; set regenerated
  (14/14 `git am`, zero whitespace warnings, applied tree
  byte-identical to the fork tip); `rdna-boosts-all.patch` refreshed.
- **Block-14 amendment — MUL_MAT_ID pair-fusion layout gate (2026-09-08,
  issue #18):** community report + detailed root-cause analysis by
  `briansp2020` (production single-R9700 deployment of the 14-block
  set, ROCm 10): the block-13/14 MUL_MAT_ID gate+up pair fusion
  aborted the process with `GGML_ASSERT(ne11 == 1 && n_expert_used > 1)`
  in `ggml_cuda_mul_mat_q_pair` whenever two MUL_MAT_ID nodes shared
  src1/ids in a layout the fused kernel does not express (src1->ne[1] > 1
  or top-1 routing) — `test-backend-ops -b ROCm0` died in the
  MUL_MAT_VEC_FUSION group.  The dispatcher gate now requires the
  callee's layout preconditions; such pairs fall back to the per-node
  path, and the qwen4exp sparse-MoE pair (standard layout) still fuses.
- **Block-13 amendment — moe_weighted_reduction float4 remainder fix
  (2026-09-08, issue #19):** community report by `briansp2020`: the
  2026-09-06 mwr-float4 fold dropped the last `n_embd % 4` columns of
  every output row for `n_embd % 4 != 0` (silent wrong output;
  `MOE_WEIGHTED_REDUCTION(n_embd=63, ...)` failed).  The float4 quad
  kernel is now gated to `n_embd % 4 == 0` (where it is also
  alignment-safe) and the upstream scalar bounds-checked kernel covers
  the rest; the aligned path is byte-unchanged.
  Both fixes validated here (3x R9700 gfx1201, ROCm 7.14):
  `test-backend-ops -b ROCm0` **16590/16590** with the fusion active,
  MUL_MAT_VEC_FUSION 1265/1265, MOE_WEIGHTED_REDUCTION 6/6, same-seed
  llama-cli streams byte-identical (Flash-Next IQ4_XS 3-GPU and dense
  27B single-GPU; default vs `GGML_PAIR_OFF=1`/`GGML_PAIR_DENSE_OFF=1`),
  prefill A/B confirms the pair fusion still fires (pp2048/pp8192
  default > pair-off beyond noise), Flash-Next full model runs clean on
  CPU (`-ngl 0`).  Fork tip moved `3529b3497`; set regenerated;
  clean-apply sim re-verified (14/14 `git am`, zero whitespace
  warnings).  Full record:
  [`patches/README.md`](patches/README.md).
- **Block-14 amendment — QSA quantized-KV decode gate (2026-09-07):** a
  quantized KV cache type (e.g. `--cache-type-k q8_0`) aborted qwen4exp
  context init (`GGML_ASSERT` in `ggml_indexer_fill`: the fused decode
  indexer ops read raw cache rows in F32/BF16/F16 only, but the indexer
  sub-cache shares the main `--cache-type-k`).  `build_qsa_top_k` now
  falls back to the per-op chain for quantized indexer keys.  Validated
  on Strix Halo across the full KV-type matrix f32/f16/bf16/q8_0/
  q4_0/q4_1/iq4_nl/q5_0/q5_1 (start + generate, zero errors; BF16 fused
  path unregressed).  Fork tip moved `60aa4173d`; set regenerated.
- **Block-08 amendment — PR #15 integrated (2026-09-07):** community
  report + fix by DanoPTT (single R9700, production since 2026-09-07):
  block 08's mul_mat+bias fusion through a view node handed the
  mmvq/mmvf kernels a destination whose shape the guards never checked
  (a reshape moves tokens between dimensions on multi-sequence
  batches) → `GGML_ASSERT(ids || dst->ne[1] == 1)` abort.  Fix folded
  into the block-08 commit (delivery convention): require the
  through-view destination to satisfy the kernels' shape constraint
  before fusing.  Fork tip moved `3bebffd6b`; set regenerated;
  verified here (3x R9700): clean-apply sim tree-identical, build
  clean, test-backend-ops 6759/6759, dense same-seed byte-identical
  pre vs post fix, 3-GPU hybrid == RCCL, parallel 2-slot decode clean.
- **Re-baseline to upstream master `050dde50c` + block 14 (2026-09-07):**
  fork point moved from `465e49b9c` to the current master tip (22
  upstream commits; the ggml-cuda-touching ones — `b74f590ea` f16 FA
  divergent-barrier fix #27870, `73ab7599b` branchless Q4_K/Q5_K mmvq
  unpack #26705, `473599738` gfx90c HIP support #26454 — merged in
  disjoint hunks).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt
  from `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean at
  `050dde50c` after one manual block-04 conflict in
  `tests/test-backend-ops.cpp` — upstream LEAKY_RELU perf cases kept
  alongside block 04's) and **block 14 (qwen4exp support) was promoted
  from `beta/qwen4exp`** (fork delta `c261553a1..dd4301fb4`, re-based;
  one manual `common.cuh` conflict — upstream gfx90c APU macros kept
  alongside the block's `GGML_CUDA_CC_IS_GFX1151`).  Set regenerated
  with `scripts/make-patches.sh` (base `050dde50c`, canonical
  am-commits `90a816a68..3bebffd6b`, 14 blocks) and
  `rdna-boosts-all.patch` refreshed (87 files).  Clean-apply sim at
  `050dde50c` re-verified 2026-09-07 (applied tree byte-identical to
  the fork tip).  Full record:
  [`patches/README.md`](patches/README.md).
- **Re-baseline to upstream master `465e49b9c` (2026-09-06):** fork point
  moved from `9cffdcc80` to the current master tip (18 upstream commits
  past the fold-verified base `8b4b3558f`, 57 past the old fork point;
  the ggml-cuda-touching ones — `73a43d1f6` mmid/mmf race fixes #28475,
  `5fdfa6282` GDN l2-norm fix #28068 — merged in disjoint hunks, zero
  conflicts).  The `~/llama.cpp` `rdna-boosts` fork was rebuilt from
  `patches/` via `scripts/apply-all.sh` (13/13 `git am` clean, zero
  whitespace warnings; per-file content check on all 112
  upstream-touched files passed) and the set regenerated with
  `scripts/make-patches.sh` (base `465e49b9c`, canonical am-commits
  `45bf4d291..c261553a1`).  Two prerequisites: the 0044cfe fold had
  stripped the format-patch mail headers from 0002/0004/0008/0013 —
  restored from the pre-fold originals (delivery commit 0610b75) — and
  the block-13 message's fold-amendment trailer was re-dated to the
  fold's true date (tip amended `b4b760eb8` -> `c261553a1`).
  `rdna-boosts-all.patch` refreshed (45 files; was stale at 41,
  pre-fold).  Clean-apply sim at `465e49b9c` re-verified 2026-09-06
  (applied tree byte-identical to the fork tip).  The `qwen4exp` fork
  branch was rebuilt on the new base + the consolidated beta support
  patch (see `beta/qwen4exp/README.md`).
- **Campaign date re-stamp (2026-09-06):** the gfx1151/qwen4exp campaign
  docs had run a week ahead of the real calendar; every
  `wip/`/`beta/`/archive date (filenames + text) was collapsed onto the
  real git dates (2026-09-05/06) and the moved records' stale
  `benchmarks/2026-09-*` references were repointed at
  `wip/archive/qwen4exp/discovery/`.
- **Block-13 RDNA3.0 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps are now also on RDNA3_0 (gfx1100), validated on a single RX
  7900 XTX (ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M (ub 2048,
  1-GPU pinned): fusion fires, same-seed coherence IDENTICAL fused-on
  vs off, prefill gains pp2048 +9.4% (5405 vs 4939), pp16384 +7.8%
  (4487 vs 4162), decode unchanged (tg128 130.3 vs 130.4).  The
  RDNA4-tuned J caps transfer (uncapping regressed pp2048 5405 -> 4819
  / pp16384 4487 -> 4070, below the 3-op fallback; a Q3_K@96 probe
  also lost to the cap 64).  Set regenerated from a canonical fork
  rebuilt at `9cffdcc80` (13 am-commits, block-13 tip `8c2ace510`);
  clean-apply sim verified (zero whitespace warnings, applied tree
  byte-identical to the fork tip).  Full record:
  [`wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`](wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md).
- **Block-13 RDNA3.5 gate relaxation (2026-09-05, folded into block 13):**
  the fused MoE gate+up+GLU MMQ prefill arm + its `J_max_gate` tile
  caps were RDNA4-only; validated on Strix Halo (Ryzen AI MAX+ 395 /
  Radeon 8060S, gfx1151, ROCm 7.14) with Qwen3.6-35B-A3B True-Q3_K_M
  (ub 2048): same-seed coherence IDENTICAL fused-on vs off, prefill
  gains match RDNA4 (pp2048 +5.3% 1590 -> 1674, pp16384 +4.6% 1360 ->
  1423), decode unchanged (tg128 71.5). The RDNA4-tuned J caps
  transfer (uncapping regressed pp2048 1674 -> 1111 / pp16384 1423 ->
  1334).  Full record:
  [`wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`](wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md).
- **Block-13 MTP regression fixes (2026-09-02, folded into block 13):**
  (1) dense adaptive-MTP collapse (18.3 -> 27.5 t/s) — the mmvq
  item-split/rpb kernel is register-bound at multi-token decode batches
  (ncols 2..8 = the speculative verify step); fixed with a re-added
  pre-block-13 K-split kernel (`mul_mat_vec_q_ksplit`) for those batches
  and long-K single-token rows (plain decode 29.0 -> 30.1, output
  bit-identical to the 12-block build).  (2) MoE adaptive-MTP collapse
  (draft acceptance 0/1527, 53 t/s vs plain 90) — the block-08
  rms_norm->mmvq Q8_1 quantize-cache fold corrupts multi-token MUL_MAT_ID,
  so verify logits diverge from single-token decode; the fold is now gated
  to single-token MMID + plain MUL_MAT consumers (acceptance 0 -> 0.51,
  MTP 126 t/s vs upstream ~113).  MoE MTP had no baseline data, which is
  why it slipped.  Details + verification: `patches/README.md` block-13
  notes.  The adaptive-MTP baseline gate and expectations now live in
  [`benchmarks/mtp-adaptive-methodology.md`](benchmarks/mtp-adaptive-methodology.md)
  — run Protocol A there before shipping decode/fusion changes.
- **Fork tip:** the fork block-12 commit was amended 2026-09-04 with the
  runtime NCCL-failure fallback (issue #13); block 13 was amended
  2026-09-02 with the two MTP regression fixes, 2026-09-05 with the
  RDNA3.5 (Strix Halo) then RDNA3.0 (gfx1100) fused-MoE-MMQ gate
  relaxations and 2026-09-06 with the model-neutral Strix MoE mmq
  folds.  The set was regenerated 2026-09-06 from a canonical fork
  rebuilt at `465e49b9c` (13 am-commits, block-13 tip
  `c261553a1`); the clean-apply sim at `465e49b9c` applies with zero
  conflicts/whitespace warnings and its tree is byte-identical to the
  fork tip.
- **Fork point (baseline):** llama.cpp master at `465e49b9c` (re-based
  2026-09-06 from `9cffdcc80`, itself re-based 2026-09-02 from
  `0eadefebd`; 57 commits of drift from the old fork point — see
  `patches/README.md` for the dated re-base record, incl. the 2026-09-02
  manual merges vs upstream's #27970 (sparse-fa) and #25952 (fused MoE
  expert reduction)).
- **Set:** 14 patches in `patches/` (`0001`-`0014`).
- **Verified:** clean apply + full build + llama-cli same-seed coherence
  IDENTICAL (hybrid vs RCCL, 3-GPU) on the rebuilt fork; the clean-apply
  sim at `465e49b9c` applies with zero conflicts/whitespace warnings and
  its tree is byte-identical to the fork tip (`c261553a1`; 2026-09-06
  regeneration — earlier regenerations were re-verified on the RX 7900
  XTX box with sim build coherence identical + perf reproduced). tg64
  38.12 / tg512 41.08 and the block-13 numbers are unchanged — the
  re-base is content-identical plus upstream's additions.
- **Whitespace-clean apply:** the regenerated set applies with **zero git
  whitespace warnings** (`git am` 01-13; re-verified 2026-09-02 on a
  fresh checkout at `9cffdcc80`, re-verified 2026-09-04 after the
  block-12 amendment, re-verified 2026-09-05 after the block-13 RDNA3.5
  gate relaxation and again after the RDNA3.0/gfx1100 fold,
  re-verified 2026-09-06 on the `465e49b9c` re-base).
- **Deployment:** 3-GPU hybrid (`HIP_VISIBLE_DEVICES=0,1,2`, unpinned) gives
  depth-16384 decode 38.71 t/s (+21.8% vs 2-GPU). See
  [`patches/README.md`](patches/README.md) for block-12 env knobs and the
  server config.
- **RDNA4-only gate:** block 12 refuses to init off gfx1200/gfx1201 and
  falls back to RCCL (community RDNA3 verification pending).
- **Runtime NCCL-failure fallback (2026-09-04, issue #13):** block 12 no
  longer aborts when NCCL/RCCL fails at runtime — on the first failure it
  clears the sticky HIP errors on each AR device, warns once, stops using
  NCCL for the rest of the run and re-routes AllReduce to the internal
  pipeline (or the meta backend's butterfly).  This covers RCCL >= 2.30.4
  refusing kernel dispatch on a PCIe root port without AtomicOp completer
  support (e.g. PCH/Z390; `ncclCommInitAll` succeeds — see
  ROCm/ROCm#6520).  Folded into the block-12 commit; re-verified
  2026-09-04 (clean-apply sim, build, same-seed coherence IDENTICAL pre
  vs post fix on 27B Q8_0, depth-16384 tg unregressed: 2-GPU 32.48 ->
  32.40, 3-GPU 39.33 -> 39.31).
- **Block-12 AR_PROFILE fix (2026-09-01, PR #8):** AR-profile `devices[]`
  init order fixed — `GGML_CUDA_AR_PROFILE=1` no longer faults GPU 1
  under MTP (pre-fix reproduced on 3x R9700; post-fix clean, profiler
  dumps on every device).  Regenerated into the set; coherence unchanged.
- **Block-02 MTP chunked-GDN prefix (2026-09-01, PR #9):** block 02 now
  runs its chunked WMMA GDN on long single-sequence MTP prefills (prefix
  `n_tokens-K` + sequential K-tail) — +7.5% prefill at ~5.5k prompt,
  +7.7% at ~38k on 3x R9700, 64-token same-seed output token-identical
  to sequential.  Opt out: `GGML_CUDA_GDN_CHUNKED=0`.
