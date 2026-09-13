# HANDOVER — issue #30 follow-up: per-kernel VDR (dense VDR=2, MoE VDR=4)

> Self-contained handoff.  Everything below was measured on this box (3x R9700
> gfx1201, ROCm 7.14 at `/opt/rocm-7.14-gfx1201`, 1 GPU pinned with
> `HIP_VISIBLE_DEVICES=0`).  Do **not** build in `~/llama.cpp` for delivery work —
> it is 2 upstream commits past the fork point; use the canonical worktrees.

## 0. The next task in one paragraph

The shipped issue-#30 fix (delivery commit `06c6356`, folded into blocks 08+10)
reverts the block-10 **VDR=4** mmvq boost **globally** and makes the RDNA4
`calc_nwarps` table **band-uniform `nwarps=1`**.  That fixes the dense regression,
but it also removes VDR=4 from **`mul_mat_vec_q_moe`**, where VDR=4 was a *win* —
so the delivery now regresses **MoE** (~+9.5 % single-token, ~−3.9 % MTP).  The
fix is a **per-kernel VDR split**: dense mmvq (`mul_mat_vec_q` item-split,
`mul_mat_vec_q_ksplit`, and their fused variants) keeps generic **VDR=2**;
`mul_mat_vec_q_moe` gets back **VDR=4** via its own selectors.  Both kernels stay
internally width-uniform, so the `plain == draft-mtp` invariant is preserved.
Then regenerate the patch set, re-run the validation, update docs, commit.

## 1. Environment / artifacts

| what | where |
|---|---|
| delivery repo (source of truth) | `~/llama-cpp-rdna-boosts` — HEAD `8646fff` (`06c6356` = the issue-#30 amendment; `8646fff` = user's `TrimREADME.md` on top) |
| canonical fork (amended, 16 blocks) | worktree `/home/stew675/canon-amend`, branch `canon-amend`, tip `1837856e3f8120449090c0f44594427573a541ed`, tree `56a1c5f23c54c038f78d7242dc05b181d872b69b` |
| └ block-08 commit | `06b72a9e2` (RDNA4 `calc_nwarps` = band-uniform `nwarps=1`) |
| └ block-10 commit | `80c79c95c` (VDR=4 reverted; `vecdotq.cuh` dropped from the block) |
| clean-apply + build | worktree `/home/stew675/deliver-verify` (branch `deliver-verify`, tree `56a1c5f23c…`), build dir `build/` (llama-cli/server/batched-bench/test-backend-ops) |
| stock reference | worktree `/home/stew675/stock-9113` at `9113cc188`, build `build-stock/` |
| pre-amendment build | `~/llama.cpp/build-rocm` (branch `rdna-boosts`, tip `c09f46ec5`) — **only** for A/B |
| width probe | `/tmp/lw-kv-verify` (built against `deliver-verify/build/bin`) |
| old patch set (pre-amendment) | `/tmp/patches-before-amend/` ← the VDR=4 code lives in `0010-*.patch` |
| scratch candidate patches | `/tmp/candidate-mtp-decode-fix-clean.patch`, `/tmp/candidate-pertype-knobs.patch` |
| helper scripts | `/tmp/vwidth.sh`, `/tmp/vtextgate.sh`, `/tmp/runarm.sh`, `/tmp/bench_mtp.py`, `/tmp/purity.sh` |
| untracked, do NOT commit | `.WORKLOG.md.swp` in the delivery repo root (user's editor swap) |

`~/llama.cpp` working tree is clean; leave it that way.

## 2. Why per-kernel VDR (the evidence)

Which knob touches which kernel (delivery source, `ggml/src/ggml-cuda/mmvq.cu`):

| knob | dense `mul_mat_vec_q` / `_ksplit` | MoE `mul_mat_vec_q_moe` |
|---|---|---|
| `calc_nwarps` (block-08 RDNA4 table) | yes | **no** — launches `(warp_size, ncols_dst)`, one warp per token, `__launch_bounds__(MMVQ_MAX_BATCH_SIZE*warp_size)`, never calls `calc_nwarps` |
| `get_vdr_mmvq` / `get_vec_dot_q_cuda` | yes | **yes** (`mmvq.cu` ~1216-1217) |

So the nwarps bug was **dense-only**, and VDR=4 was a *win* for the one-warp-per-token
MoE kernel (no multi-column register pressure).

Measured A/B (MoE = `Qwen3.6-35B-A3B-UD-Q4_K_M`, f16 KV, 1 GPU;
`llama-batched-bench -npp 16 -ntg 64 -npl 1,8`; interleaved reps):

| build | B=1 T_TG (s) | B=8 T_TG (s) | MoE MTP `n_max 3` | MTP acc |
|---|---|---|---|---|
| pre-amendment (VDR=4, nwarps=8) | 0.714 / 0.715 | 1.476 / 1.475 | **166.6 t/s** | 0.87179 |
| amended (VDR=2, nwarps=1) | 0.782 / 0.782 | 1.506 / 1.509 | **160.1 t/s** | 0.87179 |
| delta | **+9.5 % slower** | **+2.1 % slower** | **−3.9 %** | identical |

Dense headline (the regression the amendment fixes — `Qwen3.8-27B-UD-Q4_K_XL`,
q8_0 KV, 1 GPU; keep these as the target when re-validating):

| build | plain | B=1 | B=4 | B=8 | MTP n_max 7 | MTP n_max 3 | MTP-adaptive n_max 7 |
|---|---|---|---|---|---|---|---|
| stock `9113cc188` (width-impure) | 28.25 | 1.157 | 1.726 | 2.929 | 37.51 | — | — |
| delivery pre-amendment | 29.34 | 1.147 | 2.121 | 3.958 | 30.34 | — | 30.57 |
| **amended** | 28.62 | 1.175 | 1.657 | **2.798** | **36.32** | **40.15** | **38.47** |

Purity/op-suite on the amended build (must still hold after the per-kernel change):
- width probe `logits-dump-kv` W=1..8: 4B **all 8 native KV types** PURE
  (`f16 6ec6b7c8ec68bf20`, `bf16 2eec822768a3731b`, `q8_0 cf71bd4a204c93f7`,
  `q4_0 c2c0750532967fab`, `q4_1 ef5c76dba4de457e`, `q5_0 9624e5b4bf99f635`,
  `q5_1 19e2357c30e22df8`, `iq4_nl 2951a9b8c08d7ad3`); 27B `q8_0 45313682f9d41816`,
  `f16 bf3348c0a49e461c`, `bf16 e3ad7b8a5ab74ed1` PURE.
- text gate 27B all 8 KV types: `plain == mtp3 == mtp7`.
- `test-backend-ops`: ROCm0 **17999/17999**, 0 FAIL.
- MoE MTP acc `0.87179`.

## 3. Exact edit plan (in `/home/stew675/canon-amend`)

Amend the **block-10** commit again (the VDR machinery is block-10's; block 08 is
already correct and stays).  Use the same rebase method as the last amendment:
`git checkout <block10>`, edit, `git commit --amend --no-edit`,
`git rebase --onto <new> <old> canon-amend` (blocks 11-15 replayed cleanly before).

1. **`ggml/src/ggml-cuda/vecdotq.cuh`** — restore the block-10 VDR=4 additions
   (the `_vdr4`/`_vdr2` entry points and the defines).  The authoritative source is
   the old patch `/tmp/patches-before-amend/0010-*.patch` (its `vecdotq.cuh`
   section): `VDR_Q4_K_Q8_1_MMVQ 4`, `VDR_Q5_K_Q8_1_MMVQ 4`,
   `VDR_Q6_K_Q8_1_MMVQ 2`, the `#if defined(RDNA4)||defined(RDNA3_0)` Q8_0 VDR=4
   block, and the functions `vec_dot_q4_K_q8_1_impl_vmmq4`,
   `vec_dot_q4_K_q8_1_vdr4`, `vec_dot_q5_K_q8_1_impl_vmmq4`,
   `vec_dot_q5_K_q8_1_vdr4`, `vec_dot_q6_K_q8_1_impl_mmvq_vdr2`,
   `vec_dot_q6_K_q8_1_vdr2`.

2. **`ggml/src/ggml-cuda/mmvq.cu`** — split the selectors **per kernel**:
   - `get_vec_dot_q_cuda` / `get_vdr_mmvq`: keep the **dense** values — Q4_K/Q5_K/Q6_K
     generic (`vec_dot_q4_K_q8_1`/`_q5_`/`_q6_`, VDR 2/2/1).  Q8_0 stays at whatever
     the dense tests say (VDR=2 and 4 measured within noise on dense; VDR=2 is the
     safe choice).
     *Careful:* the Q8_0 generic `vec_dot_q8_0_q8_1` reads `VDR_Q8_0_Q8_1_MMVQ`
     directly, so a dense VDR=2 for Q8_0 needs that macro at 2 for the dense
     selector — or keep the macro at 4 and give dense its own 2-chunk wrapper.
   - Add `get_vec_dot_q_cuda_moe(type)` / `get_vdr_mmvq_moe(type)`:
     Q4_K → `vec_dot_q4_K_q8_1_vdr4` + 4, Q5_K → `..._vdr4` + 4,
     Q6_K → `..._vdr2` + 2, Q8_0 → VDR=4; default → `get_*` dense fallback.
   - `mul_mat_vec_q_moe` (~line 1216-1217): use the `_moe` selectors.
   - Leave the MoE launch geometry (`block_dims(warp_size, ncols_dst)`,
     `__launch_bounds__(MMVQ_MAX_BATCH_SIZE*warp_size)`, `rpb` from
     `mul_mat_vec_q_moe_launch`) untouched; leave `calc_nwarps` (block 08) as is.
   - The dense `MUL_MAT_ID` path never reaches the dense kernel (block-13 MMID
     dispatch fix routes all MMID to `mul_mat_vec_q_moe` for `ne2 <= 8`), so the
     split is clean.

3. Optional but consistent: apply the same per-kernel treatment to the block-15
   V4 native-q8_0 path only if it consumes the same macros (check
   `ggml_cuda_fattn_*`; it does not use `get_vdr_mmvq`).

## 4. Validation plan (do all of it)

1. Regenerate + clean-apply: `bash ~/llama-cpp-rdna-boosts/scripts/make-patches.sh
   /home/stew675/canon-amend 9113cc188 <new-tip>`; then in `deliver-verify`
   checkout `9113cc188`, delete the branch, `RDNA_BRANCH=deliver-verify
   scripts/apply-all.sh . …` → expect strict **16/16**, zero whitespace warnings,
   applied tree == `canon-amend` tip tree.  Re-add block-13's `--- … ---` RDNA3_5
   paragraph to `patches/0013-*.patch` (dropped by `git am` scissors — see
   `WORKLOG.md` 2026-09-12 (16) for the exact block).
2. Build `deliver-verify/build` (`BUILD_DIR=build /home/stew675/bin/build-llama-rocm-714`,
   or incremental `cmake --build build --target llama-server llama-batched-bench test-backend-ops -j16`).
3. Dense 27B `UD-Q4_K_XL` q8_0: batched-bench B=1,4,8 → expect ~amended (B=8 ≈ 2.80);
   MTP n_max 7 / mtp3 / adaptive → expect ≈ 36.3 / 40.2 / 38.5.
4. MoE 35B-A3B f16: batched-bench B=1,4,8 → expect ≈ pre-amendment (B=1 ≈ 0.71,
   B=8 ≈ 1.48); MTP n_max 3 → expect ≈ 166 t/s, acc 0.87179.
5. Purity: `/tmp/vwidth.sh` (4B 8 types, 27B q8_0/f16/bf16) and `/tmp/vtextgate.sh`
   (27B 8 types).  Hash values must match the lists in §2 (or re-pin + document why).
6. `test-backend-ops test` → ROCm0 17999/17999.
7. Docs: update the `patches/README.md` block-08/10 table rows + amendment section,
   `WORKLOG.md` (new dated entry; the existing entry is "2026-09-12 (16)"),
   `MANIFESTS.md` header, `AGENTS.md` (Critical-facts bullet + tip/tree refs),
   `benchmarks/mtp-adaptive-methodology.md`, `scripts/make-patches.sh` default tip,
   and regenerate `rdna-boosts-all.patch`.  Commit in the delivery repo.

Target line for the docs note: **the dense fix and the MoE win are independent
because the two knobs hit different kernels; the band-uniform constraint applies
per kernel, not across kernels.**

## 5. Traps / gotchas

- The current block-10 commit has **no** `_vdr4` code; `get_vdr_mmvq` must not
  return `VDR_Q4_K_Q8_1_MMVQ` unless `vecdotq.cuh` is restored and the macro is 2
  — Q4_K/Q5_K/Q6_K's generic dot does **not** read the macro, but Q8_0's does.
- `git am` drops any commit-message line starting `--- ` (block 13's RDNA3_5 note);
  re-add it to the regenerated patch by hand, as documented.
- The canonical fork tip SHA is rebuild-specific; only the **tree** hash is stable
  (`56a1c5f23c…` before this change).  Update `make-patches.sh`'s default tip after
  each rebuild.
- Do not commit `.WORKLOG.md.swp`.
- `mul_mat_vec_q_moe` at `ne2 > MMVQ_MAX_BATCH_SIZE` (or when `should_use_mmq`
  wins) goes to MMQ and is unaffected by VDR/nwarps — only the `ne2 <= 8` band is
  in scope.
- The block-15 `GGML_CUDA_FA_KV_NATIVE` (V4/V5) and the QSA paths do not use
  `get_vdr_mmvq`; qwen4exp `hc-mix.cu` **does** read `VDR_Q8_0_Q8_1_MMVQ` — if the
  Q8_0 macro value changes, qwen4exp arithmetic changes (purity-neutral, but its
  pinned hashes would need re-pinning; it was not re-validated here).
- `~/llama.cpp/build-rocm` and `/home/stew675/stock-9113/build-stock` are the two
  A/B baselines; `deliver-verify/build` is the build under test.

## 6. Reference: what shipped in `06c6356` (do not re-do)

- block 08: RDNA4 `calc_nwarps` → band-uniform `nwarps=1`.
- block 10: VDR=4 reverted in full; `vecdotq.cuh` removed from the block.
- clean-apply strict 16/16, zero whitespace, tree `56a1c5f23c54c038f78d7242dc05b181d872b69b`.
- docs updated in `patches/README.md` (new "2026-09-12 block-08 + block-10 amendment"
  section), `WORKLOG.md` (2026-09-12 (16)), `MANIFESTS.md`, `AGENTS.md`,
  `benchmarks/mtp-adaptive-methodology.md` (new rule 5: stock-relative verify-width
  `llama-batched-bench -npl 1,4,8` gate).

---

## OUTCOME (2026-09-12 (17)) — premise corrected

The per-kernel VDR was implemented as described (dense selectors upstream, `mul_mat_vec_q_moe`
VDR=4) and shipped in delivery commit **`6bb8ee3`** (canonical tip `a05225f73`, tree
`2833f1369bdea4cb45f68f85dbb2898fd98aab66`).  It keeps the dense fix and recovers the MoE **B=8**
expert win (`llama-batched-bench` 1.506 -> 1.452 s, better than the pre-(16) 1.494).  But **the
larger MoE single-token/MTP loss is NOT the VDR — it is the band-uniform `nwarps = 1` on the dense
layers.**  A diagnostic restoring the pre-(16) per-type `nwarps = 8` (RDNA4) recovers MoE B=1
0.783 -> 0.716 s and MoE MTP 161 -> 167 t/s, but costs dense MTP (35.9 -> 34.3 t/s at `n_max 7`)
and MoE B=8 (1.452 -> 1.499); the same Q8_0 weight type is in both models' decode paths, so no
per-type split satisfies both.  `nwarps = 1` is kept (the (16) dense verify fix requires it) and
the residual MoE single-token/MTP delta is a **documented trade**.

So the §2/§4 expectations in this handover (MoE B=1 ≈ 0.71, MTP ≈ 166) were based on the wrong
attribution; the per-kernel VDR cannot reach them.  Full numbers: the 2026-09-12 (17) entry in
`WORKLOG.md` and the (17) section in `patches/README.md`.

## OUTCOME (2026-09-12 (18)) — the targeted nwarps shipped

The residual MoE single-token/MTP delta was recovered in the **block-13 (18) amendment**: the dense
mmvq *weight* kernel (`mul_mat_vec_q_ksplit`) now picks `nwarps` per `(type, K)` — Q8_0 with
`K < 4096` -> 8, every other shape -> 1 (compile-time `long_k` bool; the pinned fusion ops keep
band-uniform `calc_nwarps`).  Measured: MoE B=1 +4 %, MTP n_max 3 +2 %, n_max 7 +10 % (acceptance
0.631 -> 0.731), at −2.8 % on the MoE batched B=8; the dense 27B is **bit-identical**.  The opposite
assignment (dense gets the MoE's wide VDR=4 on the same short-K shapes) was tested and **rejected**
(it cancels the MTP gain).  New canonical tip `907799de3`, tree `c2e284c2acc032238ef85cb35d427c1598ed0949`.
