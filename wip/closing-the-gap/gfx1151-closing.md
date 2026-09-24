# gfx1151 — closing-the-gap: the MMVQ↔MMQ **band-boundary fixes** (validation & port brief)

**Audience:** the agent working on the **Strix Halo / Radeon 8060S (gfx1151, RDNA3_5)** box, `halo`.
**Goal:** the closing campaign was *developed here*, so most of it is already validated.  This file is
the **one new open item**: the two **band-boundary fixes** found on `soar` (gfx1201, RDNA4) on
2026-09-24.  One of them **already fires on gfx1151** (it is arch-independent AMD code), so it needs a
**revalidation here**; the other is **RDNA4-gated** and needs a **port/reject decision** on gfx1151.

**Companion briefs:** [`gfx1201-closing.md`](gfx1201-closing.md) (3× R9700 — where the fixes were
found and measured), [`gfx1100-closing.md`](gfx1100-closing.md) (1× 7900 XTX).
**Source of truth:** [`closing-the-gap.md`](closing-the-gap.md) + the `2026-09-*` records here.
**Delivery policy:** `AGENTS.md` (default-on policy, purity rules, **never push the `~/llama.cpp`
fork**).  Push the delivery repo only if the maintainer asks.

---

## 0. The one-paragraph summary

While chasing the qwen4exp `draft-mtp n_max` **7 → 8** drop (the "W = 9 verify cliff") on gfx1201,
the real cause turned out **not** to be the `mmq_rdna3_5_id_get_J` tile choice (the original
hypothesis) but the **`n_tokens = 8 → 9` MMVQ→MMQ family boundary**: `MMVQ_MAX_BATCH_SIZE` is 8, so
at 9 columns the dense `MUL_MAT` and the routed MoE `MUL_MAT_ID` both leave the vector kernels.  For
models whose weight row counts are multiples of 128 MMQ is *fast* (27B actually **gains** +28 % at
B=9); for weights whose row counts are **not** multiples of 128, MMQ takes its generic
non-128-row **`fallback`** config (`mul_mat_q_case`: `nrows_x % 128 != 0`), which is ~3× more
expensive per launch than the ksplit MMVQ kernel.  That fallback is exactly qwen4exp's case
(rows 4/15/18/256/320), so its B=9 collapsed −21 %.

Two fixes close it on gfx1201 (both kill-switchable, both default-on):

1. **Routed MoE band → 16** (`MMVQ_MOE_MAX_BATCH_SIZE`, `mmvq_mmid_max_batch_band` floor, the
   `mul_mat_vec_q_moe` launch bound + the extended `switch`).  The dedicated MoE kernel is one warp
   per token, so its per-token reduction does **not** depend on the column count — extending it to
   the whole supported verify range (`--spec-draft-n-max ≤ 15` ⇒ W ≤ 16) is safe and even *improves*
   purity above the W=8 contract.  **This part is arch-independent AMD code and therefore already
   live on gfx1151 — the #1 job here is to revalidate it.**
2. **RDNA4 dense odd-row band → 16** (in `ggml_cuda_mul_mat` and the pair-fusion stand-down): when
   `src0->ne[1] % 128 != 0`, keep the ksplit MMVQ kernel for `ne11 = 9..16` instead of falling into
   MMQ's slow fallback.  **This is `GGML_CUDA_CC_IS_RDNA4`-gated, so it does *not* fire on gfx1151;
   the #2 job is to measure the same batched-bench curve here and decide whether to widen the gate
   to RDNA3_5 (and RDNA3_0).**

gfx1201 result (qwen4exp IQ4_NL, 3-GPU `-sm tensor`): the B=8→9 step is **gone** — B=9 goes from
**202.6 → 270.0 t/s** (+33 %, now *above* B=8's 256), B=10..12 gain +5…+23 %.  This file hands that
to gfx1151.

> **Do not target the gfx1201/gfx1151 hash values across boxes.**  The gates are **intra-build**
> (`plain == draft-mtp`, width purity, acceptance) plus the A/B of the two kill-switches.

---

## 1. The machine & the campaign state

| | |
|---|---|
| GPU | **Strix Halo, Radeon 8060S (gfx1151, RDNA3_5)**, 123 GiB unified |
| ROCm | `/opt/rocm-7.14-gfx1151` (runtime `LD_LIBRARY_PATH`) |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (ccache); fast loop `cmake --build build-rocm --target llama-cli llama-bench llama-batched-bench test-backend-ops -j 16` |
| Typical fork branch | `gap-closing-hostbuf-integrated` = r13 + `beta/mmb-general` + `0001..0014`/`0016..0028` |
| **Every command** | `export HIP_VISIBLE_DEVICES=0` (single device) |
| Headline model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` + MTP sidecar `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |
| Other models | 35B-A3B UD-Q3_K_M / Q4_K_M, 27B UD-Q4_K_M / UD-Q4_K_XL / UD-IQ3_S / Q8_0, gemma-4-12B, gemma-4-26B-A4B, NanBeige BF16 |

The campaign's own records (`closed-the-gap.md`, `closing-the-gap.md`) remain the reference for the
rest of the gates; **do not re-run them all for this item** — the boundary fix is a
kernel-family-dispatch change and the gates that matter are §4.2–§4.5 below.

---

## 2. The fix, precisely (what to build)

The change is **folded into the WIP campaign as
[`patches/0028`](patches/0028-gap-closing-WIP-extend-the-MMVQ-routed-expert-band-and-RDNA4-dense-fallback.patch)
(commit `6b230ad59208`; the campaign is now 27 patches: `0001..0014` + `0016..0028`, still skipping
`0015`).  Apply it on top of the 26-patch tree with `git am`, or apply the whole set fresh — a fresh
r13+beta worktree + 27/27 reproduces tree `533eee3188ab7df9b6cf394adeaa31b46bd13ff2`.  Three files,
69 insertions:

### 2a. `ggml/src/ggml-cuda/mmvq.cuh`

```c
#define MMVQ_MAX_BATCH_SIZE 8      // unchanged: the dense decode/verify band
#define MMVQ_MOE_MAX_BATCH_SIZE 16 // new: the dedicated MUL_MAT_ID (routed expert) kernel band
```

### 2b. `ggml/src/ggml-cuda/mmvq.cu`

* `mmvq_mmid_max_batch_band(cap)` floors at **`MMVQ_MOE_MAX_BATCH_SIZE`** (was `MMVQ_MAX_BATCH_SIZE`) —
  this is the arch-independent part that reaches gfx1151.
* `mul_mat_vec_q_moe`'s `__launch_bounds__` widens to `MMVQ_MOE_MAX_BATCH_SIZE*warp_size` —
  **the part most likely to change gfx1151 occupancy for the *existing* W ≤ 8 band** (see §3).
* `get_mmvq_mmid_max_batch_impl` + a wrapper adding the **kill-switch**
  `GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1` (clamps back to 8).
* The `ncols_dst` dispatch gains `case 9..16` (dense `launch_ksplit`) and the entry assert becomes
  `(ids ? ne12 : ne1) <= MMVQ_MOE_MAX_BATCH_SIZE`.

### 2c. `ggml/src/ggml-cuda/ggml-cuda.cu`

* `ggml_cuda_mul_mat`: `bool use_mmvq = should_use_mmvq(...)`, then the **RDNA4-only** extension
  `… && GGML_CUDA_CC_IS_RDNA4(cc) && ne11 <= MMVQ_MOE_MAX_BATCH_SIZE && src0->ne[1] % 128 != 0`.
  Kill-switch `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1`.
* `ggml_cuda_mul_mat_id` / `ggml_cuda_mul_mat_id_needs_sync`: let the **quantized** id path use the
  extended cap (arch-independent), F32/F16 keep the 8-column MMVF band.
* `ggml_cuda_try_fuse` (the MUL_MAT(_ID) **pair fusion**): stand down when `has_ids` (now band 16) or
  when the RDNA4 dense odd-row rule applies.

**Kill-switches for interleaved A/B** (default = fixes on):
`GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1`, `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1`.

---

## 3. What already fires on gfx1151 vs what needs a decision

| piece | gfx1151 effect | job |
|---|---|---|
| `mmvq_mmid_max_batch_band` floor = 16 | **fires** — routed `MUL_MAT_ID` at `n_tokens 9..16` now takes `mul_mat_vec_q_moe` instead of the RDNA3_5 routed-compact MMQ (`mul_mat_q_routed_compact` / `mmq_rdna3_5_id_get_J`).  This is exactly the gfx1151-tuned code path the campaign ported. | **revalidate** (§4.2 / §4.4); if a type/path regresses, narrow the per-type `get_mmvq_mmid_max_batch_rdna3` table rather than reverting the band |
| `mul_mat_vec_q_moe` launch bound 8→16 warps | **fires** and can change register allocation / occupancy for the *existing* `n_tokens ≤ 8` decode band | **measure decode (`tg`) before/after** on the same model/build.  If W ≤ 8 regresses, take the templated mitigation in §5.1 |
| pair-fusion `use_mmvq` with `has_ids` | **fires** (MoE pair no longer merged into MMQ at 9..16) | covered by §4.2 |
| dense odd-row band | **does not fire** (`GGML_CUDA_CC_IS_RDNA4`) | **measure and decide** whether to widen to RDNA3_5 (§4.6) |
| `case 9..16` ksplit instantiations | **compiles** (build-time cost, see §5.2) | §5.2 build check |

**Why it matters here:** gfx1151 has its own `mmq_rdna3_5_id_get_J` tile table (rows-per-expert →
J 16/48/64/128, gfx1151-measured for the 2048×512 / 3072×1024 expert shapes) and the `use_compact`
dispatch.  The original gfx1201 hypothesis blamed that tile choice — profiling showed the tile was
**not** the cause on gfx1201 (the family boundary was).  gfx1151 must re-establish which of the two
it is *there*, with the batched-bench curve (§4.2) — acceptance-free, so it isolates the path from
the acceptance curve.

---

## 4. The test matrix

### 4.1 Baseline

Build the boundary-fix tree and the **pre-fix** tree (or use the kill-switches — cheaper, no second
build).  Record the S_TG curve with the fixes **on** and **off**, interleaved in one warm session
(`OMP_WAIT_POLICY` is *not* needed on this box — the APU host-buffer path is the gfx1151 default, but
use it anyway if any run pins the CPU).

### 4.2 The acceptance-free cliff detector (the primary gate)

`llama-batched-bench` at the verify shape.  This is the tool that made the cliff visible on gfx1201
and the one that ranks the arms:

```sh
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
cd ~/llama.cpp
BIN=build-rocm/bin/llama-batched-bench
Q=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf

# interleave the arms; the fix flags are the ONLY difference
for arm in on off_moe off_both on off_moe off_both; do
  case $arm in
    on)       unset GGML_CUDA_DISABLE_MMVQ_MOE_BAND GGML_CUDA_DISABLE_MMVQ_DENSE_BAND;;
    off_moe)  export GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1; unset GGML_CUDA_DISABLE_MMVQ_DENSE_BAND;;
    off_both) export GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1 GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1;;
  esac
  timeout 900 $BIN -m "$Q" -ngl 99 -c 8192 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 \
    -npp 16 -ntg 32 -npl 6,7,8,9,10,11,12 > /tmp/bb_$arm.out 2>/dev/null
  echo "=== $arm ==="; grep -E '^\| *16 \| *32 ' /tmp/bb_$arm.out
done
```

* Only **S_TG** is meaningful; the **PP column is noise at `npp 16`** (warmup + tiny prompt) — ignore
  it.  `npl` = the verify width B = `n_tokens`; B=8 is the last MMVQ band width, B=9 the first MMQ
  width.
* Run the same for **35B-A3B UD-Q3_K_M** (MoE, different expert type/geometry) and **27B UD-Q4_K_XL**
  (dense, rows divisible by 128 — the *control* that should show the pre-existing upward jump at B=9
  and be unchanged by the dense arm).
* Verdict: for qwen4exp the fixes-on curve must be **monotone across 8→9** (on gfx1201 it is
  256 → 270).  If gfx1151's off_both curve has a B=8→9 dip, the MoE arm (off_moe vs off_both) tells
  you whether it is the routed path or the dense path.

### 4.3 Correctness (must be byte-identical where promised)

The fix changes the *reduction family* at W = 9..16 only; W ≤ 8 is untouched by design, so:

```sh
# W=1 decode vs W=4 verify (n_max 3) must stay byte-identical and unchanged from the campaign:
LLAMA=build-rocm/bin/llama-cli
P=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
MD=/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
for arm in none n3; do
  case $arm in none) SPEC="--spec-type none"; MDARG="";;
               n3)   SPEC="--spec-type draft-mtp --spec-draft-n-max 3"; MDARG="-md $MD";; esac
  timeout 900 $LLAMA -m "$Q" $MDARG -ngl 99 -c 8192 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 \
    -fa auto -n 200 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
    -f "$P" $SPEC --ctx-checkpoints 0 > /tmp/p_$arm.log 2>&1
  python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/p_$arm.log | tail -1
done
```
* `none` must equal `n3` **and** equal the campaign's 8K hash (`3553e76d3a9e` on gfx1151 for the
  IQ4_NL + shared-Q8_0 sidecar config).  If the W ≤ 8 band moved, the **launch-bounds** change bit
  you — see §5.1.
* `n_max 8` (W=9) is **above the purity contract** (`plain == draft-mtp` is promised only for
  `n_max ≤ 7`); gate it with **acceptance + MTP-vs-plain throughput**, not a hash.  Do confirm it
  *runs* and its acceptance is sane (`> ~0.45` at pos 1).
* Width probe: `test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512` → PASS
  (worst 0) for W = 1..8.  The campaign runner also has the P=32768 extension.
* Oracles unchanged: `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46, `TOPK_QSA` 4/4,
  `LIGHTNING_INDEXER` 225/225, `FLASH_ATTN_EXT` (~5953/5953, count from **stdout-only**),
  `MUL_MAT`/`MUL_MAT_ID` unchanged.

### 4.4 The `n7/n8/adaptive` MTP matrix (the §13.6 follow-up belongs to this box)

`gfx1201-closed.md` §13.6/§13.7 explicitly left the `n_max 8` question to **gfx1151** (it is where
the original `n8 > n7` observation was made); the follow-up is tracked as `gfx1201-closing.md` §2.3
(OP-3).  With the boundary fix in place, run the full matrix:
R/C/K/P/X × `none`/`n7`/`n8`/`adaptive cap 8`, `-n 3000`, reasoning pinned per axis
(`benchmarks/mtp-adaptive-methodology.md` rule 0).  Compare against the **off_both** arm.
The gfx1201 unconfounded result is `n7` beats `n8` on every axis (session 6) — if gfx1151 shows the
opposite, the boundary path is the first thing to check.

### 4.5 Per-type question for the MoE band (only if §4.2 shows a regression)

`get_mmvq_mmid_max_batch_rdna3` caps routed types at 4–6 (`IQ4_NL`/`IQ4_XS` 6, `Q3_K`/`IQ3_*` 4,
`Q4_K`/`Q5_K`/`Q6_K` 4…).  The band floor overrides all of them to 16.  If a specific expert type is
slower on the `mul_mat_vec_q_moe` kernel at 9..16 on gfx1151, **the floor has to become per-type** —
narrowing `get_mmvq_mmid_max_batch_rdna3` alone does nothing, because
`mmvq_mmid_max_batch_band()` clamps its result *up* to `MMVQ_MOE_MAX_BATCH_SIZE` afterwards.  The
clean shape is a per-type band helper (or an explicit exception list) that returns, say, the
measured width for that type and 16 for the rest.  The kernel is one-warp-per-token, so the
reduction order is per-token and a per-type cap does not re-introduce a width impurity below the
cap.  Test with the model of that type (35B Q3_K_M / Q4_K_M, and a Q4_1/Q5_0 requant if needed).

### 4.6 Dense odd-row band — port/reject decision (gfx1151 does not get it as written)

The RDNA4 rule is `src0->ne[1] % 128 != 0`.  Widening it to RDNA3_5 is a **new arch gate**, so it
needs its own measurement:

1. Confirm the gfx1151 batched-bench curve of a model with odd rows.  qwen4exp's dense weights have
   rows 4/15/18/256/320; if gfx1151 shows a B=8→9 dip even with `off_moe`, the dense fallback is
   implicated.
2. If so, widen `GGML_CUDA_CC_IS_RDNA4(cc)` to `|| GGML_CUDA_CC_IS_RDNA3_5(cc)` (and evaluate
   RDNA3_0 separately per `gfx1100-closing.md`) and re-run §4.2.
3. Gate on the same `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND` kill-switch; keep the per-arch predicate so
   a box that does not need it is unaffected.

**Rationale for the `% 128` test:** it is not a guess about qwen4exp — it is the *actual* condition
`mul_mat_q_case` uses to select the fast vs `fallback` MMQ config
(`if (args.nrows_x % 128 == 0) fallback = false; else fallback = true;`).  A row count that is a
multiple of 128 is guaranteed to keep the fast config, so the rule provably never fires on clean
shapes (27B is unchanged on gfx1201).

---

## 5. Risks, mitigations, and the build

### 5.1 The launch-bounds widening (the one real regression risk)

`mul_mat_vec_q_moe` went from `__launch_bounds__(8*warp_size, 1)` to `(16*warp_size, 1)`.  The
compiler may allocate more registers for the wider assumption, reducing occupancy for the **W ≤ 8**
decode band.  gfx1201 measured B=8 unchanged (256.3 vs 255.7), but gfx1151 has a different
register/occupancy balance.

* Measure `tg` (depth-0 `llama-bench -n 128` and, if time, a depth sweep) and the W ≤ 8 batched rows
  **before/after**; the §4.3 `none`/`n3` hashes also catch a reduction-order shift.
* **Mitigation if W ≤ 8 regresses:** template the kernel on a `c_max_tokens` parameter and
  instantiate `mul_mat_vec_q_moe<type, RPB, CFUSE, 8>` for `ncols_dst ≤ 8` and `<…, 16>` for 9..16,
  so the narrow band keeps its original `__launch_bounds__`.  This is ~2× the MoE kernel
  instantiations (a build-time cost) but removes the occupancy coupling.  Land whichever the
  measurement prefers.

### 5.2 Build-time instantiation (`AGENTS.md` discipline)

The `switch (ncols_dst)` gains 8 dense `launch_ksplit` cases **per quantized type** (the `ksplit`
family is templated on `ncols_dst`).  On gfx1201 the incremental `mmvq.cu` rebuild was ~50 s with
ccache.  After the build, check nothing blew up and no FA/dispatch TU dominates:

```sh
nm -C build-rocm/ggml/src/ggml-cuda/CMakeFiles/ggml-hip.dir/fattn-tile.cu.o | grep -c tile_case
# dispatch TU must show 'U'; instance TUs 'T'/'W' (see AGENTS.md)
```

If the MMA/FA group is untouched (it should be — this is `mmvq.cu`/`ggml-cuda.cu` only), the clean
build stays in the usual range.

---

## 6. gfx1201 reference results (fixed columns: two interleaved rounds)

**qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`, `OMP_WAIT_POLICY=PASSIVE`, S_TG t/s:**

| B | delivery | + MoE band | + MoE + dense band |
|---:|---:|---:|---:|
| 6 | 209.8 | 210.9 / 209.4 | 209.8 / 209.7 |
| 7 | 235.0 | 235.7 / 235.3 | 235.8 / 235.0 |
| 8 | 255.7 | 257.3 / 256.3 | 256.4 / 255.4 |
| 9 | **202.6** | 225.7 / 225.3 | **270.0 / 268.7** |
| 10 | 226.1 | 245.4 / 243.9 | 289.0 / 288.2 |
| 11 | 245.0 | 263.7 / 262.8 | 302.0 / 301.7 |
| 12 | 264.7 | 281.1 / 279.1 | 324.5 / 324.0 |

**MTP (qwen4exp IQ4_NL, `draft-mtp`, prose, `-c 16384 -n 3000`, passive wait):** `n7` 123.4 t/s
both arms (W=8 unaffected); `n8` **94.3 (off) → 102.7 (on) t/s (+8.9 %)**, acceptance 0.6415 → 0.6285
(above the pos-1 gate).  `plain == n3` byte-identical in both arms.

**Controls (unchanged on gfx1201):** 27B UD-Q4_K_XL dense — the pre-existing B=8→9 *upward* jump
(194 → 249, all rows ÷128 → MMQ fast config) is identical in both arms; 35B-A3B Q3_K_M — neutral
(B=9 442 vs 431, rest within noise).

---

## 7. Traps

1. **`HIP_VISIBLE_DEVICES=0`** on every command.
2. **The PP column at `npp 16` is noise** — the verdict is **S_TG only**, plus the interleaved order.
3. **`n_max 8` (W=9) is above the purity contract** — use acceptance + MTP-vs-plain, not `plain ==
   draft-mtp`.  `n_max ≤ 7` (W ≤ 8) must stay byte-identical.
4. **`--ctx-checkpoints 0` at any depth.**
5. **Never benchmark in parallel**; interleave the arms in one warm session.
6. **Keep the two kill-switches separate** — `off_moe` vs `off_both` is how you attribute a change to
   the routed MoE path or the dense path.
7. **Do not push the `~/llama.cpp` fork**; fold any accepted port into a closing patch and record it
   in the delivery repo.

---

## 8. Session log

Newest first.  Append state, what changed, the tree/`From <sha>`, and the next action.

### 2026-09-24 — opened (from the gfx1201 W=9-cliff session)

Handover created.  **Nothing measured on gfx1151 yet.**  The gfx1201 fix is now
[`patches/0028`](patches/0028-gap-closing-WIP-extend-the-MMVQ-routed-expert-band-and-RDNA4-dense-fallback.patch)
(commit `6b230ad59208`); the first job here is to apply the 27-patch closing set on top of
`gap-closing-hostbuf-integrated`, build, and run §4.2/§4.3.  Open decisions: §3 (revalidate the MoE
band that already fires here), §4.6 (widen the dense odd-row gate?), §5.1 (launch-bounds occupancy).
Full gfx1201 evidence: [`2026-09-24-qwen4exp-w9-verify-cliff.md`](2026-09-24-qwen4exp-w9-verify-cliff.md)
and [`gfx1201-closed.md`](gfx1201-closed.md) §13.7 (session 8).
