# GREEDY-PURITY findings — the dated narratives (moved out of `GREEDY-PURITY.md`, 2026-09-12 (11))

This is the **evidence and narrative half** of `../GREEDY-PURITY.md`: the full story of each closed-case
finding, moved here verbatim (section numbers and titles unchanged, so every `§N` citation that points
into `GREEDY-PURITY.md` still resolves — the summary entry there carries the claim, the numbers and the
rule; this file carries the investigation).  Read it when you need to know *how* something was found,
which instruments failed, or which false trails were ruled out; `GREEDY-PURITY.md` is what to read
before shipping.

Originally written for the 2026-08/09 block 10/12/13/14 work; see `WORKLOG.md` and
`patches/README.md` for the delivery-side record of each amendment.

## 11. The real purity range, and its two causes (2026-09-11 correction + fix)

Earlier notes (and the 2026-09-11 block-02 entries) claimed
`--spec-type none == draft-mtp` for `n_max <= 15`.  **That was wrong.**  The
claim had only ever been validated up to `n_max = 4`.  There are in fact **two
independent causes**, and they bound different configurations:

| config | `none == draft-mtp` guaranteed for | binding cause |
|---|---|---|
| 1 GPU (no split) | `n_max <= 7` (W <= 8) | B |
| 2-GPU `-sm layer` / no split | `n_max <= 7` | B |
| 2-GPU `-sm tensor` | `n_max <= 5` **before** this fix, `n_max <= 7` after | A, then B |
| 3-GPU `-sm tensor` | `n_max <= 7` | B |
| 2-GPU `-sm tensor` + `GGML_CUDA_ALLREDUCE=meta` | `n_max <= 7` | B |

The boundary is on the **verify batch width**, not on `--spec-draft-n-max`
directly: the target verifies the drafts *plus* the last committed token, so
`K = n_max + 1` and `n_max = 8` is already a 9-token batch.  Measured with the
raw-logit probe (27B Q8_0, `RS=0`, P=256; identical within a row = bit-identical
token-0 logits):

| config | `W = 1..8` | `W = 9` (= `n_max 8`) |
|---|---|---|
| 1 GPU | `4089b4d4` | `72af52db` |
| 2-GPU `-sm layer` | `4089b4d4` | `72af52db` |
| 2-GPU `-sm tensor` | `a4817ee6` | `b059daa6` |
| 3-GPU `-sm tensor` | `91434ea9` | `bc3faabd` |

So the guarantee is **`--spec-draft-n-max <= 7`** in every configuration, and
the first violating depth is `n_max = 8`.  (Before the fix, 2-GPU tensor broke
at `W = 7`.  1 GPU and `-sm layer` share a hash because layer splitting does
not change any kernel.)

**Use the probe, not the text gate, to establish a boundary.**  On 3 GPUs the
300-token greedy text at `n_max = 8` matched the plain run (`5037ef2e` in both)
while the logits had already diverged (`bc3faabd` vs `91434ea9`) -- no greedy
near-tie happened to flip inside that window.  Text equality is evidence *for*
purity, never evidence *against* divergence; this is the same near-tie rarity
noted in `../wip/sm-tensor-plain-vs-spec/HANDOVER-2026-09-11.md`.

**Cause A (fork-specific; FIXED 2026-09-11): block 12's size-based all-reduce
dispatch.**  `ggml_backend_cuda_comm_is_small()` routes a reduction to the
internal host-staged pipeline below a per-device-count element count (32768 for
2 devices, 131072 for 3, 262144 for 4+) and to NCCL above it.  The two paths
are **not bit-identical** (different summation order; the internal pipeline
always does the FP32->BF16 round-trip).  Under `-sm tensor` the reduced tensors
scale with the batch width (`ne = ne0 * n_tokens`, `ne0 = 5120` here), so a
**7-token** verify batch is 35840 elements and crossed the old 2-device limit
while 1..6-token decode stayed below it: the same logical reduction, a
different algorithm, purely because the batch got one token wider.  Raising the
2-device crossover to 131072 (the 3-device value) fixes it -- the largest
verify batch, `n_max 16` -> 17 tokens, is 87040 elements, well under it, and
still far below the internal pipeline's own 1 MB (262144 element) cap.
Measured: `GGML_CUDA_ALLREDUCE=internal` (one algorithm for every size) gives
`W = 1/6/7/8` **all** `a4817ee6`, and so does the fix, with `W <= 6` keeping
their original hash (plain decode bit-unchanged) and only `W = 7,8` moving onto
the internal pipeline.  MTP throughput at `n_max 6` went 63.6 -> 71.3 t/s
(+12%) and at `n_max 12` 51.7 -> 58.0 t/s (+12%), pp/tg unchanged -- the
internal pipeline is the faster path at these sizes, so the fix is a win on
both axes.  1 GPU has no cross-device reduction at all, 3 GPUs stay under their
boundary until `W = 26`, and `-sm layer` never splits the reduction dimension:
all three were unaffected.

**Cause B (deliberate; open): the flash-attention tile-vs-WMMA switch at
`Q->ne[1] > 8`.**  Beyond `W = 8` the FA launcher prefers the WMMA kernel (much
faster at prefill-scale batches) and its reduction order differs from the tile
kernel's.  This is not accidental -- the fork's own comment in
`ggml-cuda/fattn.cu` says speculative verify batches (`n_q <= 8`) must stay on
the tile kernel because decode never uses WMMA.  So `n_max <= 7` **is the
designed guarantee**, and any configuration whose reduction tensors stay under
Cause A's crossover is pure all the way through it.  Removing Cause B would
mean giving up WMMA for 9..N-token batches.

The table below is the original `n_max` sweep (pre-fix).  Both builds in it
carried Cause A (block 12 was identical in them) and Cause B, so it shows their
*joint* effect; the GDN prefill boundary was not a factor in either.

| `--spec-draft-n-max` | none / 1 / 4 / 5 | 6 / 7 | 8 / 9 / 10 | 12 | 16 |
|---|---|---|---|---|---|
| delivered build (KTAIL=16) | equal | `5037ef2e` | `e721b8b5` | `e721b8b5` | `5037ef2e` |
| whole-batch chunked prefill | equal | `b6d86d62` | `ed922c76` | `5037ef2e` | `4f3ee41c` |

The two builds diverge in exactly the same place, so this is **not the GDN
prefill boundary's doing**: block 02's change fixed the *prefill* (probe
`RS=6 W=6 == RS=0 W=1` — the prefill state is now K-independent), and this cap
is Cause A + Cause B, both of which were present in both builds.

**Localisation (as measured).**  Both causes are pure *decode-batch-width*
effects; neither is `K`, and neither is the GDN.  At `RS=0` (no snapshots at
all, `K = 1`, so the GDN cannot be involved) the probe separates on width alone:

```
2-GPU tensor, pre-fix:  W = 1..6 -> a4817ee6   W = 7,8 -> e286b75c   W = 9 -> 24f302f6
```

i.e. adding columns to the batch changes column 0's own result.  `W = 7` is
Cause A.  `W = 9` is Cause B, and it coincides with `MMVQ_MAX_BATCH_SIZE = 8` in
`mmvq.cuh` (beyond 8 columns `ggml_cuda_mul_mat` leaves the vector kernels for
MMQ); both the FA WMMA gate and the MMVQ/MMQ crossover sit at that boundary, so
Cause B is fixed at `W <= 8` regardless of which of the two fires first.
`GGML_CUDA_GDN_CHUNKED=0` does not help (it is a prefill switch); block 13's
dense `ncols==1` ksplit alignment does not either — it aligns `ncols = 1` with
the `2..8` *verify* dispatch, and these boundaries sit above that.

**Is upstream affected?**  Cause B is upstream-inherited; Cause A is
fork-specific (upstream has no internal AR pipeline).  Upstream master's own
`calc_nwarps`/`calc_rows_per_block`
(`ggml/src/ggml-cuda/mmvq.cu`) switch on `ncols_dst`, and
`ggml_cuda_mul_mat` selects MMVQ only for `ncols_dst <= MMVQ_MAX_BATCH_SIZE`,
so any batched evaluation uses different kernels than a one-token decode.
Measured on **upstream master `9cf3bf256`** (clean checkout, unmodified
`mmvq.cu`), CPU backend, 4B Q8_0, `P = 256`: `W = 1` gives `9024dd2e...`
while `W = 2..12` all give `3cd0eb0e...` — upstream diverges at the *first*
width step and is therefore **worse** than the fork, not better.  (That build
was CPU-only, so upstream's ROCm boundary was not measured; the fork's
`n_max <= 7` is a fork *result*, not an upstream guarantee.)

**Practical consequence.**  With Cause A fixed the guarantee is `n_max <= 7` for
2-GPU `-sm tensor` too (it already applied to 1 GPU, 3-GPU tensor and
`-sm layer`).  Do not use `none == draft-mtp` byte-equality as a gate above
that; use acceptance + MTP-vs-plain throughput (see
`benchmarks/mtp-adaptive-methodology.md`).  Adaptive MTP's recommended
`n_max = 12` remains outside the *guaranteed* range by Cause B, which is
deliberate.  Records: `patches/README.md` block-12 notes, the 2026-09-11
WORKLOG entry, `wip/sm-tensor-plain-vs-spec/FOLLOWUPS-2026-09-11.md` Part 3.

## 13. qwen4exp and the hyper-connection band (2026-09-11, Block 14 amendment)

qwen4exp (Qwen3.8-Flash-Next) has its own fused decode chain: the hyper-connection mixer
(`GGML_OP_HC_MIX`) and residual combine (`GGML_OP_HC_COMBINE`) replaced the unfused
`SCALE/SILU/MUL_MAT/SIGMOID/MUL/ADD` chain, but only for a **single-token** batch, so a 1-token decode
and an n-token verify batch took different arithmetic (measured in the per-node dump: 98
`HC_COMBINE` dispatches at `W=1`, **0** at `W>=2`).  Block 14, amended 2026-09-11, routes the whole
**decode/verify band `1 <= nt <= 8`** through the fused ops with the token index on `blockIdx.y`, so
every token in the band runs exactly the per-token kernel sequence a single-token decode runs.
Measured (3 GPUs, f16 KV, P=256, RS=0):

| split | W=1..4 | W=5 | W=6,7 | W=8 |
|---|---|---|---|---|
| `-sm layer`  | **all `3adeb313042a871b`** (= the W=1 decode) | `c999233926f0` | `a8c532e12f9c` | `c56ebb61963a` |
| `-sm tensor` | **all `dcf1ae667f730879`** (= the W=1 decode) | `2bfb89f59ec2` | `e8b1253ea93e` | `a7c5dfd26a56` |

So qwen4exp was width-pure for **`--spec-draft-n-max <= 3`** *at the time of this measurement*
(2026-09-11, block-14 amendment); **§15 fixes the `W >= 5` half, so the band is now `n_max <= 7`**.
The `W >= 5` grouping is **cause 2** (§11): the same
`ncols_dst`/`ne11` kernel-selection band **that was assumed to be the same site as the `q8_0`/`q4_0` KV
impurity in §12**.  That hypothesis was **refuted on 2026-09-11**: fixing F1 (§14) left cause 2's
`{5} {6,7} {8}` grouping completely unchanged, so cause 2 is a *separate* site (a matmul/MoE dispatch
band), not the FA kernel-family chooser.  Two caveats: a `<= 8`-token **prefill** chunk also takes the
fused path (indistinguishable from a verify batch — the point is that such a batch gets the decode
arithmetic); and with a `q8_0`/`q4_0` **KV cache** the cache's own impurity (§12) dominates, so the band
does not restore text equality there (the `W=1` decode is still unchanged).

## 14. F1 fixed (2026-09-11, block-08 amendment): the decode/verify band no longer spans two FA kernel families

§12's `q8_0`/`q4_0` impurity is fixed.  Root cause, found with a new kernel-chooser trace (committed for
reuse as `wip/kv-quant-purity-followups/tools/fa-kernel-chooser-trace.patch`, `GGML_CUDA_FA_TRACE=1`):
`ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu`) returned **VEC** for `n_q <= 2` with
a quantized K/V and **TILE** from `n_q = 3`.  The two families order the online-softmax/PV reduction
differently, so token-0 logits at `W = 1,2` disagreed with every verify width.  Both VEC conditions are
always inside the `n_q <= 8` band, so the branch is deleted and the band is TILE throughout — the same
shape as the block-08 WMMA guard (`Q->ne[1] > 8`) and block 00's `ntiles_dst_eff`.  The launcher plan
itself was already width-independent (`ntiles_dst_eff`; `parallel_blocks == ntiles_KV` at every width),
which is why every earlier F1 suspect measured clean.

| K/V cache (same type) | before §14 | after §14 |
|---|---|---|
| f16, bf16 | pure | pure, hashes **bit-identical** (they never took VEC) |
| q4_1, q5_0, q5_1, iq4_nl | pure | unchanged (see the F3 reframing below) |
| **q8_0** | `W=1,2` != `W=3..8` | **`W=1..8` bit-identical in all four split configs** |
| **q4_0** | same shape as q8_0 | **same** |

Measured (3x gfx1201): 4B `q8_0/q8_0` 1 GPU `W=1..8` all `31a0c1bace68`, 2-GPU `-sm tensor` `abebfb93`,
3-GPU `-sm tensor` `7fe106f5` (`q4_0`: `619c151e48c7` / `240bc37d` / `483a850e`); 27B Q8_0 3-GPU
`W = 1,2,3,4,5,8` all `d4156dbeb225`.  Text level (27B, 300 greedy tokens, q8_0 KV): plain ==
`n_max 3` == `n_max 7` = `3537bc2b36be` (was `73b2565bce47` vs `3537bc2b36be`); f16 control
`f32aac948600` for both.  Every new value equals that config's **previous verify** value: only
`W = 1,2` moved, so MTP is bit-unchanged (27B acceptance 0.90789 identical) and the cost is
decode-only — tg128 -0.9% (4B) / -0.5% (27B), pp512 ~-0.2%, reserves byte-identical.

**The pure range is still `n_max <= 7`** (the verify batch is `n_max + 1` and the designed FA
tile-vs-WMMA switch sits at `Q->ne[1] > 8`) — now for *every* supported KV type, not just the float
ones.

**F3 reframed.**  §12's "no native FA kernel" explanation is wrong in detail: `q4_1`/`q5_0`/`q5_1`/
`iq4_nl` are rejected by `ggml_cuda_fattn_kv_type_supported()` unless the build enables
`GGML_CUDA_FA_ALL_QUANTS` (OFF here), so `ggml_cuda_get_best_fattn_kernel()` returns `NONE` *before*
any VEC/TILE choice (0 `[FATPATH]` lines for `q4_1`, 1+ for `q8_0`) and attention takes the generic
fallback — width-invariant by construction (hence pure) and ~3.4x slower.  F3's first experiment is a
build-flag A/B of `GGML_CUDA_FA_ALL_QUANTS=ON` (plus §14's band rule, which keeps any newly-enabled
native path width-invariant).

**Harness lesson (three false positives in one session).**  `llama-cli`'s `/\|` spinner is `\b`-based
and timing-dependent, and the ASCII banner embeds the build SHA: apply backspaces, strip the banner and
the `[ Prompt: ... | Generation: ... ]` footer before hashing output — and always run a control that
must agree (the f16 pair) before believing any divergence.

## 15. F2 cause 2 fixed (2026-09-11, block-13 amendment): the MoE decode/verify band is band-uniform

§13's `W >= 5` grouping is fixed, and the *mechanism* is **not** the gate+up+GLU fusion coverage the
first pass blamed — it is upstream's **per-type mmvq cap** (`get_mmvq_mmid_max_batch_*`), which does two
things:

1. it sizes `mul_mat_vec_q_moe`'s `__launch_bounds__` (`cap × warp_size`) while the block is
   `(warp_size, ncols_dst)` — so it is a *capability* limit (launching `IQ3_S`, cap 4, with
   `ncols_dst = 5` is 160 threads > the bound and aborts with `unspecified launch failure`);
2. it chooses mmvq vs MMQ (`ggml_cuda_mul_mat_id`: `ne2 <= cap → mmvq`, else `should_use_mmq → MMQ`)
   and gates the `mul_mat_q_pair` fusion (`use_mmvq`, `ggml-cuda.cu:3730`) — which is what ran at
   `W = 5..7`.  **mmvq and MMQ reduce in different orders**, so every cap boundary inside the band is a
   numeric boundary.

This is a *graph-identical* bug: `[GD]` full-graph dumps give the same node counts (2647/2404/2271/1863/
1668/1565) at `W=4` and `W=5`, with `MUL_MAT_ID(ffn_moe_gate)` / `MUL_MAT_ID(ffn_moe_up)` /
`GLU(ffn_moe_swiglu)` at the same indices in both.  The quant is what makes it visible: the UD-IQ4_XS
file mixes expert types per layer (47 layers `IQ3_S` gate/up → cap 4; layer 2 `IQ4_XS` → cap 5; down
`IQ4_NL`/`Q8_0` → cap 7), which predicts the observed grouping **exactly** — fused layers
48/48/48/48/1/0/0/0 for `W = 1..8`, i.e. the 4→5 and 5→6 boundaries, and the down's cap 7 for 7→8.

**Fix** (block 13 amendment; it completes block 13's own `has_ids` "decode == verify invariant"):
floor the cap at `MMVQ_MAX_BATCH_SIZE` for every AMD lookup and size the MoE kernel's launch bound at
the band.  All cap call sites are `MUL_MAT_ID`-only ⇒ dense models cannot be affected (verified).

| split | W=1..8 (f16 KV, P=256, RS=0) |
|---|---|
| `-sm layer`  | **all `3adeb313042a871b`** (= the pre-fix W=1 decode) |
| `-sm tensor` | **all `dcf1ae667f730879`** (= the pre-fix W=1 decode) |

Every width moved onto that split's **pre-fix `W = 1`** value: plain decode is bit-unchanged, and only
`W = 5..8` moved (the F1 "move the cheap side" pattern).  Also pure with `RS=from_w`.  At the MTP gate
config (`n_max 3` = `W=4`, a no-op width) pre/post-fix runs are **byte-identical** (acceptance 0.76744,
80.0 vs 80.1 t/s) — the fix provably does not touch what already worked — and at `n_max 7` it is
**+16-18 % t/s** with acceptance 0.59375 vs 0.55556; `n_max 3` == `n_max 7` text (`8a50ea24e8d5`) where
they previously disagreed (`8a50ea24e8d5` vs `e6918a7af1f9`).

**The fix is also a large throughput win** (`llama-batched-bench`, interleaved, swappable
`libggml-hip.so`; fixed/baseline):

| model | b1 | b2 | b4 | b5 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|---|
| qwen4exp 3-GPU `-sm tensor` | 50.5/50.4 | 85.4/85.6 | 134.1/132.9 | **149.5/118.4** | **162.5/130.6** | **171.4/147.0** | **178.0/155.4** |
| 35B-A3B MoE 1 GPU | 98.3/98.1 | 156.4/156.2 | 254.2/254.1 | – | – | – | **341.3/289.9** |
| 4B dense 1 GPU | 100.6/100.5 | 167.2/167.3 | 294.0/294.9 | – | – | – | 414.5/413.3 |

So upstream's per-type mmvq caps were costing 14-26 % at exactly the speculative-verify widths on RDNA4
with the fork's mmvq + fused-GLU kernels.

**Cause 3 (open).**  `plain` still differs from `draft-mtp` text (`plain` `3ee9daee5c07` vs
`n_max 3 == n_max 7` `8a50ea24e8d5`) — and this fix cannot be responsible: at `n_max 3` (`W = 4`) it is
a verified no-op (bit-identical logits, byte-identical text, byte-identical acceptance).  A control
confirms the harness is deterministic and that plain decode is unchanged (pre-fix and post-fix plain
text are both `3ee9daee5c07`).  Because the single-step width probe is bit-identical across `W = 1..8`
on both splits *and* with the state-sequence dimension (`RS=0` and `RS=from_w`), the divergence must be
a **multi-step / roll-back** effect — **localised 2026-09-11 (further measurement): it is in the QSA *machinery*, and the site class is the same as cause 1's.**  `LLAMA_QSA_OFF=1` makes `plain` == `draft-mtp --spec-draft-n-max 3` **byte-identical** (`d4499ac8db72` both, 711 chars) — and the knob provably fires (the plain text moves `3ee9daee5c07` -> `d4499ac8db72`) — while `LLAMA_QSA_SPARSE_FA=0` (dense attention, indexer still on) leaves two different texts (`25f300a81b9e` vs `0d466b2dcf09`), so the defect is **not** the sparse-FA kernel but the **indexer/score machinery** (`indexer-topk.cu` + the `qwen4exp.cpp` gates).  Both QSA-side `n_tokens == 1` gates are the prime suspects — `src/models/qwen4exp.cpp:1094` (`idx_score_fused`, the fused indexer score) and `:1419` (`qsa_dense_decode_until`, the early-decode dense shortcut) — i.e. exactly the cause-1 pattern, and the single-step width probe cannot see them because it never reaches the sparse/indexer decode regime.  The divergence appears only after ~100 chars (~20 tokens) of a 3.3k-prompt greedy run (the first steps agree), so it is not a prefill-state difference; `GGML_CUDA_GDN_CHUNKED=0` moves both sides without making them agree (the known Issue #25 chunked-prefill item is a separate contributor, not this).  **Kill-switch for users meanwhile: `LLAMA_QSA_OFF=1`.**

## 16. Cause 3 fixed (2026-09-11, block-14 amendment): the QSA decode arm is band-uniform

After §13-§15 the dense models, the MoE and qwen4exp's hyper-connection band were pure, but qwen4exp
still had a **text** divergence: `--spec-type none` and `draft-mtp --spec-draft-n-max 3` / `7` shared
only ~100 of ~700 generated characters (f16 KV, `/tmp/prompt3k.txt`, 128 greedy tokens, 3-GPU
`-sm tensor`).  It was **not** a kernel and **not** the sparse-FA attention:

* `LLAMA_QSA_OFF=1` makes both runs **byte-identical** (711 chars), and the knob provably fires (the
  plain text moves `3ee9daee5c07` → `d4499ac8db72`), so the defect lives in the QSA **indexer**
  machinery (store / score / top-k selection);
* `LLAMA_QSA_SPARSE_FA=0` does **not** fix it (both texts move, both stay different) — the
  `fattn-qsa` kernel is exonerated;
* the single-step width probe is **pure** on both splits and with `RS=from_w`: its blind spot is
  exactly this bug (it prefills `P <= 2048` tokens and decodes one step, so `n_kv` stays below the
  QSA selection width and every width takes the same arm).

**Mechanism** (`build_layer_attn`, `src/models/qwen4exp.cpp`).  The indexer picks one of three arms:

```cpp
if (shortcut && n_kv <= width)                                        // 1: dense, store keys
else if (qsa_dense_decode_until > 0 && n_tokens == 1 && n_kv < ...)   // 2: dense policy arm
else  top_k = build_qsa_top_k(...);                                   // 3: sparse selection
```

`width = indexer_top_k + r - 1`; on qwen4exp `indexer_top_k = 2048`, `r = 4` (only every 4th layer has
an indexer), so `width = 2051`.  From the arm trace at the first decode graph: `n_kv = 2304 > 2051`, so
arm 1 no longer applies — and arm 2 is gated on `n_tokens == 1`:

| run | shape | arm |
|---|---|---|
| `--spec-type none` | `n_tokens=1 n_kv=2304` | **2 (dense)** |
| `draft-mtp n_max 3` | `n_tokens=4 n_kv=2304` | **3 (sparse top-k selection)** |

Same state, two attention regimes, purely because of the batch width.  (Both runs are identical for the
first 11 graph builds; the split starts at the first decode graph.)

**Fix**: `QSA_DECODE_BAND = 8` (the `n_max <= 7` purity band, the same constant class as
`HC_FUSED_MAX_TOKENS`), and arm 2 takes `n_tokens <= QSA_DECODE_BAND` instead of `n_tokens == 1`.
Prefill is untouched (`n_tokens` is far above the band, so it keeps arm 3 — the arch policy "prefill is
untouched, QSA always"), and on gfx1201 `qsa_dense_decode_until = 1 << 62`, so decode is now dense at
every width — which is the policy the arm's own comment describes.  Post-fix arm trace: `n_tokens=4
n_kv=2304` gives **arm 2** in *both* runs.

**Measured**: `plain == n_max 3 == n_max 7` = `804de0576868` (704 chars, f16 KV) and `plain == n_max 3`
= `75d8530c5bb1` (660 chars, q8_0 KV).  The plain stream moved with the fix (658 → 704 chars): the fix
also moves the shared 4-token non-decode shape at `n_kv = 2304` onto the dense arm, which is the same
"the band must take one path" trade as §14 — the *value* chosen is the policy-consistent dense one.

## 17. The MoE shared-expert epilogue is band-uniform too (2026-09-11, block-13 amendment)

§15 fixed the `W >= 5` split; what remained for Qwen3.6-35B-A3B was strictly `W=1 ac8825358d9adfda` vs
`W>=2 bd138ad2326fbbf2` — the fused shared-expert down epilogue (`dst = down(swiglu) *
sigmoid(gate(x)) + moe_out + ffn_residual`, a 6-node fusion in `ggml-cuda.cu`), gated

```cpp
down_mm->src[1]->ne[1] == 1 && gate_mm->src[1]->ne[1] == 1; // decode only
```

with an in-code note that the fused gate reduction (`shexp_gate_sigmoid`) does not reproduce the
standalone mmvq/MUL_MAT order — so `W=1` ran the fused epilogue and `W>=2` the unfused chain.

**Fix**: the two kernels are now token-generic, and the band takes the *fused* path (which keeps the
+3.1 % decode win instead of throwing it away):

* `shexp_gate_sigmoid` — one block (one warp) per token: the token index only selects the input column
  (`grid: (ncols)`), so each token's dot is computed by the same 32-thread warp reduction as before;
* `shexp_down_gated_q8_0` — one block per `(output row, token)` (`grid: (nrows, ncols)`), addressing
  `y_swiglu`, `moe_out`, `ffn_residual` and `dst` at the token's offset;
* **`nwarps` is pinned to the single-token value** (`calc_nwarps(type, 1, table_id)`): `calc_nwarps`
  returns 4 for `ncols_dst 1..4` but 2 for `5..8`, and `nwarps` sets `blocks_per_iter`, i.e. the
  reduction order — the same trap as §15's cap.  Pinning it is what makes every width bit-identical;
* the fusion arm accepts `1 <= ne[1] <= MMVQ_MAX_BATCH_SIZE` (with both matmuls the same width and the
  three epilogue operands contiguous); `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` still selects the unfused
  reference.

**Measured**: default probe `W = 1, 2, 3, 4, 8` all `ac8825358d9adfda` (the pre-fix `W=1`/fused value);
with the kill-switch all of them `bd138ad2326fbbf2` (a uniform unfused reference, = the pre-fix `W>=2`
value); qwen4exp was unaffected (`plain == n3 == 804de0576868`).  The asterisk is gone: the MoE decode
and verify batches now take one arithmetic, so the `n_max <= 7` guarantee covers MoE too.

## 23. A silently dead chain: the shadowed variable (2026-09-11 (11), Block 15 dense-arm blocker)

**The bug, in one line:** block 15's V2/V3 refactor added an *outer* `ggml_tensor * kq_mask_top_k =
nullptr;` in `build_attn_qsa` while the top-k mask chain *inside* the new `if (kq_mask != nullptr) { ... }`
wrapper kept its own `ggml_tensor * kq_mask_top_k = ggml_set_rows(...)` — a **new local** that shadowed the
outer one.  The chain was therefore built whenever the mask existed, and the attention
(`build_attn_mha(q, k, v, nullptr, kq_mask_top_k, ...)`) read the outer one: `nullptr`.  The dense masked
arm attended with no mask at all — a full causal leak, seen as PPL `1.0558` on qwen4exp for every KV type
where the delivery gives `6.49-6.55`, and as ≈`1.02` on *random* text where a working build gives `18.4`.

**Four generalisable lessons:**

1. **A graph tensor with no consumer is silently dropped.**  The chain's nodes were unreachable from the
graph output, so `ggml_build_forward_expand` never emitted them; with no consumer the allocator left the
packed mask unallocated, and block 15's own `if (self_kq_mask && self_kq_mask->buffer)` guard in
`llm_graph_input_attn_kv::set_input` then skipped `set_input_kq_mask` entirely.  So "the input is created
in the graph" proves nothing about it being *filled* — and an unfilled mask is indistinguishable from a
correct one until you measure quality.  A defensive check (assert the mask was filled when a consumer
exists) is worth considering for the beta.
2. **When an executed-graph dump is *missing* nodes, suspect the graph builder, not the allocator.**  The
   `[ND]` node dump showed the delivery's dense prefill consuming `attn_inp_kq_mask` 36 times (12 indexer
   layers × 3 devices) and emitting the `FILL`/`SET_ROWS`/zeros chain, while the beta consumed it **zero**
   times and emitted **no** `FILL` at all.  Dead code is absent from the graph *by construction* — that is
   the signature, and it points straight at the builder.
3. **`-Wshadow` would have caught this class outright.**  The fork does not enable it.  Adding it (at
   least for `src/` on the CI path) would make this whole failure mode a compile error; the fix itself is
   one token.  Filed as a follow-up.
4. **The leak instrument to reach for first is random text.**  A model that can see the target predicts
   *anything* — including noise — near-perfectly: `llama-perplexity -f /tmp/rand-text.txt --chunks 1
   -c 2560 -b 2560 -ub 2560` gave the broken build `1.0205` and the delivery `19.0589`.  Natural, or even
   repetitive, text is a *bad* leak detector (a repetitive prompt scores ≈1 in a perfectly healthy build,
   which sent this session down a false trail until random text settled it).  Keep both the oracle and the
   random-text probe in the gate list; they answer different questions ("is the fused path as good as the
   dense one?" vs "is the attention causal?").

