# HANDOVER — QSA fused sparse attention for quantized KV caches (qwen4exp)

**Point the next session at this file.**  It is self-contained for this task.  The shared
infrastructure (environment, models, probe, harnesses, landing procedure, trap list) is in
**`HANDOVER-2026-09-11-remaining-work.md`** (read §2–§5 and §12–§13) and the previous brief
**`HANDOVER-2026-09-11-f3-kv-diagonals.md`** (its §10 outcome section is the state you inherit).
`iq4_nl` is **out of scope here** — it is the *next* brief (F3 step 2); this one must not depend on it,
though the two meet at the same end state (a QSA kernel that can read `iq4_nl` too).

## 1. Mission

Make `GGML_OP_FLASH_ATTN_QSA` — the fused sparse/identity attention kernel qwen4exp's indexer layers
use — **read the quantized cache types natively**, so that on qwen4exp every KV cache type takes the
same attention path.  Today (after the 2026-09-11 F3 step-1 landing) `f16`/`bf16`/`q8_0` use the fused
op and `q4_0`/`q4_1`/`q5_0`/`q5_1` are **forced onto the dense masked path** by the block-14 gate
`qsa_kv_native` (`src/models/qwen4exp.cpp:1303`), because the QSA kernel's load dispatch cannot read
them (`ggml_cuda_flash_attn_qsa_supported()`, `ggml/src/ggml-cuda/fattn-qsa.cu:621`).

Step 1 of this task = `q4_0`, `q4_1`, `q5_0`, `q5_1` (the types the delivery already enables for plain
flash attention).  When F3 step 2 lands, `iq4_nl` joins them in the same place.

## 2. Why — the measured prize (do not skip; the shape of the win is not what you expect)

qwen4exp, 3 GPUs, `-sm tensor`, `llama-bench -p <N> -n 0 -r 2 -b 2048 -ub 2048 -fa on`, pp-only
(2026-09-11, canonical `6f07fe67a`):

| KV cache | attention path | pp8192 | pp32768 |
|---|---|---|---|
| f16 | **fused sparse QSA** (default) | 2368.3 | **2362.5** |
| f16 | dense masked (`LLAMA_QSA_SPARSE_FA=0`) | 2462.4 | 2075.9 |
| `q4_1` | **forced dense masked** (today's state) | 2416.7 | **2076.4** |

Read that carefully:

* Isolating the **path** at a constant cache type (rows 1–2) the fused sparse QSA is **−3.8 % at 8k but
  +13.8 % at 32k** — the crossover is between 8k and 32k, consistent with the block-14 arch-policy
  note ("QSA wins prefill from ~8K up").  **The prize is long-context prefill**: a quantized cache is
  currently denied it (2076.4 at 32k, i.e. exactly the dense number), and the fix should bring it to
  the f16-sparse level (~2360) for its own type.
* Comparing **cache types** (f16-sparse vs `q4_1`-dense) is *not* the path comparison — the smaller
  cache is worth a few percent on its own.  Hold the type constant when isolating the path, and hold
  the path constant when isolating the type.
* Below the crossover the fused op is a small *loss*, which is the trade the f16 reference already
  makes — "uniform with f16" is the goal, not "faster everywhere".

Secondary value: it removes the last **type-dependent kernel-family difference** in this model (the same
class of defect as F1/F2/HC/§19) and re-enables the fused op for the indexer layers' *shortcut* and
*dense-decode* arms, not just the sparse prefill arm.

## 3. State you are starting from

* **Canonical fork**: `/tmp/canon-llama`, branch `rdna-boosts`, tip **`6f07fe67a`**, net tree
  **`0c9dece6b0798e41360b8a8366187f38f37e1566`**, 15 blocks (00–14), clean, `build-base` built.
  (Block SHAs: 00 `1c7ab0e89`, 01 `aa4108b9d`, 02 `6e81ed5ed`, 03 `4dc962aa9`, 04 `03d004517`,
  05 `70f330aed`, 06 `d2fc2cb34`, 07 `110b5391d`, **08 `1a488fcf0`**, 09 `43cc5f850`, 10 `f5236f3ee`,
  11 `c71a657b0`, 12 `e008f9914`, 13 `1a88c92f5`, **14 `6f07fe67a`**.)
* **Delivery repo**: `~/llama-cpp-rdna-boosts`, `main` == `origin/main` == `cafe8f5`, 15 patches
  `0000`–`0014`, `make-patches.sh` default tip `6f07fe67a`.
* **Block 15 beta**: base `6f07fe67a` → beta commit **`8c377b958`**, tree `34527a292` (the sixth
  re-cut; branch `blk15-f3` in the canonical fork).  **Block 15 already modifies `fattn-qsa.cu`** (its
  derived-visibility / mask-elision work, W2/V3: `cell_vis`/`q_vis`), so **your block-14 amendment will
  invalidate it and the re-cut will conflict in that file for real** — budget for it (§8) and prefer
  edits that sit next to, not inside, block 15's hunks.
* **Owner**: `ggml/src/ggml-cuda/fattn-qsa.cu` and the qwen4exp arm gate are both **block 14's**
  (`git log 9113cc188..rdna-boosts -- ggml/src/ggml-cuda/fattn-qsa.cu` shows only block 14).  One
  block-14 amendment should carry the whole change.

## 4. What is VERIFIED about the mechanism (read before designing)

All checked in the delivery source on 2026-09-11:

1. **The kernel is templated on a single KV type** — `flash_attn_qsa<D, type_KV, use_logit_softcap>`
   (`fattn-qsa.cu:43`), i.e. K and V share it, which is exactly the K==V policy.
2. **It already dequantizes in-kernel.**  The smem staging has two arms: `if constexpr (type_KV ==
   GGML_TYPE_F16 || type_KV == GGML_TYPE_Q8_0)` (`:149`) with a **hand-written `block_q8_0` dequant into
   `half2` smem** ("Q8_0 inputs are dequantized to F16 here, once per tile", `:175-199`), and a
   BF16-native arm (`kv2_t = type_KV == GGML_TYPE_BF16 ? nv_bfloat162 : half2`, `:134`).  So the pattern
   you need already exists — it is per-type code, not a family decision.
3. **The reusable helpers exist**: `dequantize_V_{q4_0,q4_1,q5_0,q5_1,q8_0,f16,bf16}` in
   `ggml/src/ggml-cuda/fattn-common.cuh:378-600`, templated `<T, ne>` with `ne == 2 || ne == 4`, and
   for `T = half` they write **`half2`** — i.e. exactly the QSA smem element type.  Using them (one
   thread per `(cell, half2)` instead of the hand-written per-block code) is the small, uniform route;
   cross-check the smem row layout (`K_smem[cell][…]`, padded by one `half2`, `:135-136`) and the
   `M_smem` staging site (the comment at `:378` warns that only the F16/Q8_0 gather wrote it).
4. **No template-instance files**: QSA instantiations live in `fattn-qsa.cu` itself
   (`ggml_cuda_flash_attn_qsa_case<D, type_KV>`, `:453`) and the dispatch is a hand-written chain
   (`:576-618`) with `GGML_ABORT("unsupported K/V type")` as the fallback.  Adding 4 types × the 3 head
   sizes (`64/128/256`, `:597-612`) is a mechanical edit; qwen4exp uses **128**.
5. **The predicate** `ggml_cuda_flash_attn_qsa_supported()` (`:621`) requires `Q->type == F32`,
   `idx->type == I32`, `D ∈ {64,128,256}` and `kv_ok = (F16,F16) | (BF16,BF16) | (Q8_0,Q8_0)`.  It is
   consulted by the backend's `supports_op` (`ggml-cuda.cu:6858`), which is what decides whether the
   graph may build the op at all.
6. **The op has an identity mode** and this is the crucial consequence: `top_k == nullptr` (the
   `shortcut` and dense-decode arms) plus `env GGML_CUDA_QSA_IDENTITY` make the kernel read cells
   `0..n_top_k-1` with `identity = true`.  For **f16 today the decode band already runs the QSA
   *identity* kernel**, not `build_attn_mha`; for a quantized cache the block-14 gate makes it run
   `build_attn_mha` (`src/models/qwen4exp.cpp:1303-1305`).  So enabling QSA for these types will change
   the **decode** arithmetic too — new reference hashes (expected and fine) *and* a decode-perf
   question (§6).  Do not try to enable only the sparse arm: that would change f16's shortcut arm
   (which currently uses the QSA identity kernel) and make the two cache-type families diverge — the
   design is **all arms, all types**, which is exactly why per-arm enablement is wrong.
7. `LLAMA_QSA_SPARSE_FA=0` stays the A/B knob (dense masked), `LLAMA_QSA_OFF=1` is the whole-indexer-off
   reference, `LLAMA_QSA_DENSE_DECODE_UNTIL` overrides the decode crossover.  All three are useful
   controls and none of them should need changing here.
8. **`test-backend-ops` has NO `FLASH_ATTN_QSA` coverage** (grep the test file: nothing).  That is a
   real gap — the QSA kernel is the least-tested kernel in the fork and it is about to grow four load
   paths.  Adding at least a smoke test is part of the deliverable (§6).

## 5. The plan

1. **Enabling change** (one block-14 amendment, all in `fattn-qsa.cu` + 1 line in `qwen4exp.cpp`):
   * extend the smem K/V staging to `q4_0`/`q4_1`/`q5_0`/`q5_1` — either with `dequantize_V_*` (item 3)
     or with a per-type block like the existing Q8_0 arm; keep the smem type `half2` and the padded row
     layout, and make sure the `M_smem` staging is covered for the new arms;
   * add the four type branches (3 head sizes each) to `ggml_cuda_flash_attn_qsa`;
   * add them to `kv_ok` in `ggml_cuda_flash_attn_qsa_supported()`;
   * extend (do **not** delete) the `qsa_kv_native` list in `build_attn_qsa` so the graph gate and the
     backend predicate stay in lockstep — that gate is what prevents the "unsupported op is never split"
     meta-splitter abort the F3 session diagnosed;
   * leave `llama_kv_type_has_native_fa()` alone (it already allows the four types).
2. **Positive control before believing anything**: the trace/env knobs must show the new path being
   taken (`GGML_CUDA_QSA_IDENTITY`, `LLAMA_QSA_SPARSE_FA=0`), and the existing f16/bf16/q8_0 reference
   hashes must **not** move.
3. **Purity + correctness gates** (§6) — the decode-band arithmetic changes, so the full matrix is
   mandatory, not a regression check.
4. **Perf matrix** (§6) — prefill sparse-vs-dense per type, and the decode question.
5. **Land** (§8): block-14 amendment, patch regen, clean-apply sim, 7th beta re-cut, docs sweep, push.

## 6. Gates (all mandatory)

**Correctness — the decode band changes, so this is new-reference territory, not a no-op check:**

* `W = 1..8` width purity on qwen4exp, 3-GPU `-sm tensor` **and** `-sm layer`, `RS=0` and `RS=from_w`,
  for each of `q4_0`/`q4_1`/`q5_0`/`q5_1` + the three existing types as controls (the controls must
  reproduce today's values: f16 tensor `dcf1ae667f730879`, q8_0 tensor `7b4e8394ed779477`,
  q4_1 tensor `7d67c09d59b47267` — the q4_* ones are expected to *change*).
* `plain` == `--spec-draft-n-max 3` == `7` byte-identical text on qwen4exp (f16 QA'd harness, **default
  verbosity**); today's q4_1 value is `42dfe66f25ed` and will change — record the new one.
* MTP gate: pos-1 acceptance ≥ ~0.45 and MTP ≥ plain at `n_max 3` (today q4_1: 0.628, q8_0 0.700).
* Smoke on the other models/splits (27B `-sm tensor`/`-sm layer`, MoE-35B, gemma-4-E4B) — QSA is
  qwen4exp-only, so these must be **unchanged**.
* `test-backend-ops -o FLASH_ATTN_EXT` (currently 5599/5599, must stay) and `GATED_DELTA_NET` 4/4.
* **Add a `FLASH_ATTN_QSA` case to `tests/test-backend-ops.cpp`** (at minimum: run it for each supported
  KV type at `D = 128`, and compare against the CPU/non-fused reference the way the FA test does).
  If the harness cannot express the op's 5 inputs (Q/K/V/idx/mask) cheaply, say so explicitly in the
  block note and leave a coherence-level check instead — but do not leave the gap undocumented.

**Perf (interleaved, and *hold the cache type constant* when isolating the path):**

* `llama-bench -p 8192,32768,65536 -n 0 -r 2 -b 2048 -ub 2048 -fa on`, per type, default vs
  `LLAMA_QSA_SPARSE_FA=0` — the record of the §2 table, now for quantized caches.  Expect the fused
  path to lose a few percent at 8k and win double digits at 32k/64k.
* Decode: with a quantized cache, measure the QSA identity kernel against `build_attn_mha` (the current
  forced path) at depth — `tg128` at `pl=1..8` and, if the fused identity turns out to be *slower*, that
  is a **policy** question (the arch crossover `qsa_dense_decode_until` is per-arch, not per-type; a
  per-type rule would be a separate, deliberate decision — measure first, then decide, and if you do
  change the policy, the width gate applies to it too).
* KV reserves are unchanged by this work (the cache type is untouched) — but re-record them anyway if
  you change anything about the arm choice.

## 7. Traps

* **The four types' decode arithmetic will change** (QSA identity instead of `build_attn_mha`).  That is
  expected; treat the old hashes as the pre-change reference and add the new ones to
  `GREEDY-PURITY.md`/the block note.  What must *not* change is f16/bf16/q8_0.
* **QSA is prefill-heavy**: remember the arch policy puts prefill on the sparse arm above the indexer
  width (`indexer_top_k + r - 1` = 2051 for qwen4exp, i.e. the first ~2048-token chunk), so a pp8192
  benchmark is one "shortcut"/identity chunk and then sparse chunks, while a pp512 or pp2048 run
  measures almost nothing but the identity arm.
* **`dequantize_V_*`'s `i0` argument is an element index, not a byte index**, and `ne` must be 2 or 4;
  the QSA smem row is `half2`-indexed and padded by one `half2` for bank conflicts — get the row
  indexing right or the kernel will quietly read the wrong cells (the existing Q8_0 arm is the template).
* **Occupancy**: the kernel is tuned for 3 blocks/CU (`:131`); smem is
  `2 * WARP_SIZE * (D/2 + 1) * sizeof(kv2_t)` + a little.  Adding dequant *code* does not change smem,
  but check `max_blocks_per_sm` in the launcher if you add scratch.
* **Do not enable only the sparse arm** — see §4 item 6: it would change f16's shortcut arm and split
  the families again.
* `qwen4exp.cpp` will conflict in the 7th beta re-cut **inside the block-15 hunks** (both the helper
  `qwen4exp_qsa_sparse()` and the load path in `fattn-qsa.cu`); resolve by keeping block 15's structure
  and adding your type condition *inside* its predicates, exactly like the sixth re-cut did (that one
  threaded the type through `llama_cparams::type_k`, which is already in place for you).
* `--log-verbosity 4` interleaves log lines into the generated text — text-purity runs use the default
  verbosity; the acceptance line needs 4; run them separately.
* Never run benches in parallel; `-sm tensor` + gemma aborts in the meta splitter (3 GPUs, 2 KV heads)
  — use 1 GPU or `-sm layer` for the SWA smoke.

## 8. Landing

Exactly the standard flow (`HANDOVER-2026-09-11-remaining-work.md` §12): owner-based block amendment
(`git rebase -i 110b5391d` marking block 14 `edit` → apply → `git add` → `git commit --amend
--no-edit` → `git rebase --continue`; **`git apply` leaves changes unstaged, so the `git add` is
load-bearing**), regenerate with `./scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`,
bump the default tip in `scripts/make-patches.sh`, refresh `rdna-boosts-all.patch` by hand, clean-apply
sim (strict 15/15, 0 whitespace warnings, applied tree == canonical), **re-cut block 15** (now a real
`fattn-qsa.cu`/`qwen4exp.cpp` merge; verify the no-op claim against the delivery build at the gate
configs as the sixth re-cut did), docs sweep (WORKLOG entry, `patches/README.md` block-14 section,
`GREEDY-PURITY.md`, `AGENTS.md` chain/bullets, TODO, README/MANIFESTS/BASELINE tips, beta records), then
commit and push to the delivery repo's `origin` only.

**Definition of done for this task:** the four types take the fused QSA path on qwen4exp (all arms),
f16/bf16/q8_0 unchanged, the new hashes recorded, the width/text/MTP gates green on both splits, the
prefill A/B table above filled in for quantized caches, a `FLASH_ATTN_QSA` backend-op test (or a
documented reason there is none), and a dated WORKLOG entry.  Then hand back with a one-line pointer to
the `iq4_nl` brief (F3 step 2), which by then is only "one more `dequantize_V_iq4_nl` + the same four
edits".
