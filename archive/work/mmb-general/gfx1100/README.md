# gfx1100 patch overlay — **FOLDED INTO THE BETA SET (2026-09-21)**

> **This overlay no longer exists as a separate step.**  At the `archive/work/mmb-general` promotion the two
> shippable overlay patches were folded into the beta set: they are now `patches/0011` (qsa3 on
> RDNA3_0) and `0012` (F32 split off on RDNA3_0) of the single 12-patch set.  The third — the
> experimental per-M `nwarps` rule — was moved out to [`../../../wip/nwarps/`](../../../wip/nwarps/)
> because it is default-OFF and breaks `W=1..8` width purity.
>
> Apply the beta set in one step: `git am <repo>/archive/work/mmb-general/patches/*.patch` (**12/12**, tree
> `bca69f23dd29acef2d8898c6fd492104e078eef1`).  The text below is the historical overlay record, kept
> because it explains the provenance, the rebase history and the file-ownership reasoning.

> **Rebased onto the updated `wip-mmb-general` (S10-S14, 10 canonical patches) on 2026-09-21.**
> The overlay was `0011`/`0012`/`0013`.  See
> [`../gfx1100-porting.md`](../gfx1100-porting.md) **§14** for the rebase brief, the verified
> conflict map and the §14.5 re-validation gate.

The gfx1100 work is a **WIP overlay on top of the canonical 10-patch `mmb-general` set**.  Keep it
separate so the record branch can be rebased/merged back into `wip-mmb-general` cleanly.

## Apply order

```sh
# 1. the canonical 10 patches (from wip/mmb-general/patches/, on the r12 delivery)
cd ~/llama.cpp
git checkout rdna-boosts                      # r12 tree 8a80535e
git worktree add ~/llama-wip-gfx1100 -b mmb-gfx1100 rdna-boosts
cd ~/llama-wip-gfx1100
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/patches/*.patch   # 10/10

# 2. the gfx1100 overlay (this dir)
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/gfx1100/patches/*.patch   # 3/3
git rev-parse HEAD^{tree}   # -> cd306e6b6093b63468289edac24fbea3d270dbe2 (verified 2026-09-21)
```

`0011`, `0012` and `0013` apply clean, in order, on top of the 10-patch canonical tree:

* `0011` (`fattn-qsa3.cu`): `ggml_cuda_flash_attn_qsa3_supported()` gains `RDNA3_0`, so the
  packed-block WMMA QSA path runs on gfx1100 (gfx11 fragment arm, identical to gfx1151).
  **On the 10-patch tree this is the only overlay patch that still touches a file the gfx1201
  S10-S13 work also touched** (`fattn-qsa3.cu` was not part of S10-S13; it applies clean).
* `0012` (`mmb.cu`): **rewritten for the S11 rework.**  The old edit patched
  `mmb_f32split_mode()`, which S11 replaced with the per-arch `mmb_arch_defaults(cc)` accessor.
  The new patch is a **`RDNA3_0` arm in `mmb_arch_defaults()`** setting `c.f32split_mode = 0`;
  RDNA3_5 keeps the struct default `1` and RDNA4 keeps S13's `1`.  The F32 MoE-router split is a
  loss on gfx1100.  Verify with `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1 GGML_CUDA_MMB_CFG=1` →
  `f32split=0`.
* `0013` (`mmvq.cu`): **experimental, DEFAULT-OFF** per-M `nwarps` rule for the dense weight ksplit
  path (`GGML_CUDA_MMVQ_RDNA3_SMALL_M`, default 0).  The rule gains 35B-A3B +2.1 % draft-mtp /
  gemma-26B +2.1 % decode at M≤2048/4096, but **breaks W=1..8 width purity on the MoE models**, and
  the only pure threshold (≤1024) gives no gain — so it must not be enabled until the width-invariant
  mapping is re-derived.  See `../gfx1100-s9-nwarps-results.md`.

## Rebase notes (2026-09-21, §14)

The gfx1201 branch moved from the 6-patch merge base `1f2c92d` to the 10-patch tip `c79d48f`.  The
four new patches (`0007`-`0010`) all rework `mmb.cu`; `fattn-qsa3.cu` and `mmvq.cu` were untouched.

* `0011` (was `0007`) and `0013` (was `0009`) **re-applied clean** — the gfx1201 S10-S13 work never
  touched `fattn-qsa3.cu` or `mmvq.cu`.
* `0012` (was `0008`) **conflicted** (its `mmb_f32split_mode()` body was replaced by the S11 accessor)
  and was **rewritten** as the `RDNA3_0` arm described above.
* The record-branch rebase (`wip-mmb-general-gfx1100` onto `wip-mmb-general`) was **clean** — no
  `GROUPS.md` conflict in the end (the gfx1100 pointer landed in a non-conflicting hunk).
* Fresh-apply verification: a worktree at `c8dda33dd` + the 10 canonical + the 3 overlay patches
  applies **13/13** and yields the same tree `cd306e6b6093b63468289edac24fbea3d270dbe2`.

## Why a patch instead of folding it into the canonical patches

The files are owned by canonical patches 2 (`fattn-qsa3.cu`), 1/3/4 (`mmb.cu`) and the delivery
(`mmvq.cu`).  Rewriting an owner patch would change its `From <sha>` line and `git am` content on the
shared `wip-mmb-general` branch for no behavioural reason; a separate patch keeps the merge-back a
clean replay.  If the maintainer prefers, they fold in at promotion time.

## Status

* `0011` (qsa3) is **validated** on gfx1100 at the unit level: `test-backend-ops -o FLASH_ATTN_QSA`
  26/26, and a `rocprofv3` kernel trace shows `qsa3_attn_kernel` + `qsa3_pack/merge/rows` dispatched.
  End-to-end qwen4exp performance is trust-RDNA3_5 (no qwen4exp model fits 24 GB).
* `0012` (mmb) **defaults the F32 MoE-router split off on RDNA3_0** (RDNA4/RDNA3_5 unchanged).
  With it, `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1` is a large gfx1100 win: 27B Q4_K_M +14.4 %,
  gemma-12B Q8_0 +11.3 %, 35B-A3B MoE +5.6 %/+4.7 %, gemma-26B-A4B neutral.  Decode unchanged,
  width purity PASS, PPL parity.  **Re-confirmed on the rebased 13-patch tree** (see
  `../gfx1100-s10-rebase-results.md`).
* `0013` (mmvq) is an **experimental, default-off** per-M `nwarps` scaffold; enabling it is a real
  MoE win but breaks width purity, so it is not enabled.  See `../gfx1100-s9-nwarps-results.md`.
* See `../gfx1100-s2s4-results.md` and `../gfx1100-s5s7-results.md` for the records.  The
  pre-rebase patch names (`0007`/`0008`/`0009`) in those dated files are the same three changes.
