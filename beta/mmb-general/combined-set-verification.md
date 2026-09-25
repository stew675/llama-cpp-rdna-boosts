# The combined set — gfx1151 + gfx1201 + gfx1100 (2026-09-21)

> **POSTSCRIPT — one patch was removed after this verification.**  The maintainer had the experimental
> per-M `nwarps` patch (`0013`) **moved out** of the set to [`../../wip/nwarps/`](../../wip/nwarps/),
> because it is default-OFF and breaks the `W=1..8` width-purity contract.  The 12-patch beta set was
> then `bca69f23dd…` on r12 (this file's §2-§4 describe the intermediate 13-patch state, tree
> `cd306e6b60…`, which is also verified).  Removing a default-OFF patch that no gfx1201 code path
> could reach cannot change the gfx1201 results in §4, and dropping it also removes the build cost
> measured in §5.
>
> **2026-09-25 — consolidated with `closing-the-gap` and re-based on r13.**  The shipped beta set is
> now **28 patches** (12 core + 16 appended, ten closing patches folded into the core), strict
> `git am` **28/28** from the **r13** tree (`bb7b6d07…`) to applied tree **`468c6496…`** —
> **tree-identical** to the combined `gap-closing-denseband` tree, so the gfx1201 cross-arch results
> below carry over unchanged.  The r12-era `bca69f23dd…` hash above is the pre-consolidation set.
>
> **2026-09-24 — re-based onto `84e76d8a2`.**  The delivery baseline moved to upstream master
> `84e76d8a2` (release `v16-84e76d8a2-r1`); the 28-patch set was re-based onto it and its applied
> tree is now **`2e4e8004…`** (previously `468c6496…` on r13).  Only patches `0014`/`0015` needed a
> conflict resolution (upstream's restructured `ggml_backend_cuda_graph_optimize` loop).  The
> gfx1201/gfx1100 cross-arch results below were measured on the r13-based tree; they carry over as
> behaviour claims, but re-measure on `2e4e8004` before quoting numbers.

One branch now carries all three architectures.  This file records the merge and the **cross-arch
verification** that the gfx1100 overlay does not disturb gfx1201 (or gfx1151).

## 1. What was merged

`origin/wip-mmb-general-gfx1100` (12 commits, branched from `c79d48f` = the S14-brief commit) merged
into `wip-mmb-general` as merge commit `10fc552`.  The branch was 3 commits behind us
(`1b76c4a` S14, `e9f47e5` S15, `3275f5e` the FLASH_ATTN_EXT correction), so the merge assembled
13-patch-era docs on top of those.

The merge was **clean apart from an auto-resolved `GROUPS.md`**: gfx1100 added a pointer to its new
`gfx1100-porting.md` at the top of the "gfx1100 job" section while the gfx1201 side rewrote the job
steps beneath it.  Git auto-merged both; nothing was lost.

gfx1100 brought:

| | |
|---|---|
| plan | `gfx1100-porting.md` (889 lines) + `gfx1100/README.md` |
| results | `gfx1100-s1-results.md`, `-s2s4-`, `-s5s7-`, `-s8s9-`, `-s9-mmb-tuning-`, `-s9-mmvq-`, `-s9-nwarps-`, `-s10-rebase-` |
| patches | `gfx1100/patches/{0011,0012,0013}` — correctly numbered after our `0010`, so **no collision** with the gfx1201 set |

## 2. The combined set applies clean

```
r12 c3ee45747 + patches/0001..0010 + gfx1100/patches/0011..0013
  -> git am 13/13
  -> tree cd306e6b6093b63468289edac24fbea3d270dbe2
```

Verified two ways: applied on top of the existing 10-patch fork state, and **fresh** from a detached
worktree at `c3ee45747` — both give the same tree, `cd306e6b60…`.  (Tree before the overlay:
`35fc853e63…`, the gfx1201-frozen tree.)

## 3. Why the overlay is safe for gfx1201 and gfx1151

Reviewed patch by patch before accepting the combination:

| patch | change | why it cannot affect RDNA4 / RDNA3_5 |
|---|---|---|
| **0011** `fattn-qsa3.cu` | `ggml_cuda_flash_attn_qsa3_supported()` gains `GGML_CUDA_CC_IS_RDNA3_0(cc)` | **purely additive**: RDNA4 and RDNA3_5 were already accepted, so the predicate's result is unchanged on both |
| **0012** `mmb.cu` | a new `if (GGML_CUDA_CC_IS_RDNA3_0(cc)) { c.f32split_mode = 0; }` arm in `mmb_arch_defaults(cc)` | **per-arch branch**: RDNA4 keeps S13's `f32split_mode = 1`, RDNA3_5 keeps the struct default `1` |
| **0013** `mmvq.cu` | a `small_m` template axis on the dense ksplit path | the host predicate is `table_id == MMVQ_PARAMETERS_RDNA3_0 && !has_fusion && nrows_x <= mmvq_rdna3_0_small_m()`, and that accessor is `getenv(...) ? atoi(...) : 0` — so `small_m` is **provably false** on every arch unless explicitly enabled on gfx1100 |

## 4. Verified on gfx1201, not assumed

Built the combined tree and re-ran the gates.  **Every gfx1201 value is identical to the frozen
10-patch result:**

| check | expected (frozen 10-patch) | combined 13-patch |
|---|---|---|
| `MMB_CFG` dump | `cc=0x1001201 … f32split=1(min_m=128,min_k=0) … routed=0` | **byte-identical** |
| 27B UD-IQ3_S `-n 24` | `119 chars sha=42cdf36d0633` | `42cdf36d0633` |
| 35B UD-Q3_K_M `-n 24` | `110 chars sha=461ca8cd0e88` | `461ca8cd0e88` |
| Flash-Next `-n 24` (3-GPU) | `135 chars sha=d73f9238f6d6` | `d73f9238f6d6` |
| `FLASH_ATTN_QSA` | 26 cases, green | 26 cases, green |
| `GATED_DELTA_NET` / `INDEXER_TOPK` | green | green |
| `test-logits-width-probe` (MMB=1) | `PASS (worst maxdiff 0)` | `PASS (worst maxdiff 0)` |
| 27B UD-IQ3_S pp8192 / pp32768 | 932.8 / 856.7 (S14 binary) | **934.86 / 856.71** |

The `MMB_CFG` line is the strongest single check: it proves 0012's RDNA3_0 arm did not leak into the
RDNA4 row.  The width probe is the one that would catch a `small_m` leakage into the reduction order,
because that is exactly the contract 0013 is documented to break when enabled.

## 5. The one cost: patch 0013 doubles the ksplit instantiation set

0013 threads a `small_m` **template** axis through `mul_mat_vec_q_ksplit`, and the host dispatch
branches on it at *runtime*, so both variants are compiled — on **every** arch, including the two
where the feature can never fire.

| `mmvq.cu.o` | 10-patch | combined 13-patch | delta |
|---|---:|---:|---:|
| object size | 8.5 MiB | 12 MiB | **+41 %** |
| `mul_mat_vec_q_ksplit` symbols | 828 | **1656** | **+100 %** |
| all `mul_mat_vec_q*` symbols | 1851 | 2679 | +45 % |

That is real build time on all three arches for a feature that is default-OFF **and** documented as
unshippable (it breaks W=1..8 width purity on MoE models; the only pure threshold gives no gain).
`mmvq.cu` is already one of the bigger TUs.  It is worth deciding whether this scaffold belongs in the
**default** apply set or should be applied only when running the nwarps experiment — the work is
preserved either way, since it is a separate patch.

## 6. State after the merge

* Branch: `wip-mmb-general` at the merge commit (plus the doc update that carries this file).
* Combined tree: `cd306e6b6093b63468289edac24fbea3d270dbe2`.
* gfx1201: **unchanged and re-verified** (§4) — the frozen `35fc853e63…` results still hold exactly.
* gfx1151: not re-run here, but 0011/0012/0013 cannot reach it (§3), and the gfx1201-side reasoning
  applies identically (`cc` is RDNA3_5, so 0012's arm is skipped and 0013's `table_id` test fails).
* gfx1100: its own records are in `gfx1100-porting.md` + the dated `gfx1100-s*-results.md`; the
  overlay was validated there 13/13 before the merge.
