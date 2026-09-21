# gfx1100 patch overlay

> **Rebasing onto the updated `wip-mmb-general` (S10-S14, 10 patches)?**  Read
> [`../gfx1100-porting.md`](../gfx1100-porting.md) **§14** — it has the verified conflict map and the
> exact `0008` rewrite (`0007`/`0009` apply clean; `0008` becomes an `RDNA3_0` arm in
> `mmb_arch_defaults`).

The gfx1100 work is a **WIP overlay on top of the canonical 6-patch `mmb-general` set**.  Keep it
separate so the record branch can be rebased/merged back into `wip-mmb-general` cleanly.

## Apply order

```sh
# 1. the canonical 6 patches (from wip/mmb-general/patches/, on the r12 delivery)
cd ~/llama.cpp
git checkout rdna-boosts                      # r12 tree 8a80535e
git worktree add ~/llama-wip-gfx1100 -b mmb-gfx1100 rdna-boosts
cd ~/llama-wip-gfx1100
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/patches/*.patch   # 6/6
git rev-parse HEAD^{tree}                     # -> 580db5174574f10cc92fb1cefa72281a65c77b12

# 2. the gfx1100 overlay (this dir)
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/gfx1100/patches/*.patch   # 0007
```

`0007`, `0008` and `0009` apply clean, in order, on top of the 6-patch tree:

```sh
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/gfx1100/patches/*.patch   # 3/3
git rev-parse HEAD^{tree}   # -> 2c89ce7219a993fa9c43c767f99b1e384db59656 (verified 2026-09-21)
```

* `0007` (`fattn-qsa3.cu`): `ggml_cuda_flash_attn_qsa3_supported()` gains `RDNA3_0`, so the
  packed-block WMMA QSA path runs on gfx1100 (gfx11 fragment arm, identical to gfx1151).
* `0008` (`mmb.cu`): `mmb_f32split_mode()` defaults the F32 split off on RDNA3_0/RDNA4 (gfx1151
  keeps it on), because the F32 MoE-router split is a loss on gfx1100.
* `0009` (`mmvq.cu`): **experimental, DEFAULT-OFF** per-M `nwarps` rule for the dense weight ksplit
  path (`GGML_CUDA_MMVQ_RDNA3_SMALL_M`, default 0).  The rule gains 35B-A3B +2.1 % draft-mtp /
  gemma-26B +2.1 % decode at M<=2048/4096, but **breaks W=1..8 width purity on the MoE models**, and
  the only pure threshold (<=1024) gives no gain — so it must not be enabled until the width-invariant
  mapping is re-derived.  See `../gfx1100-s9-nwarps-results.md`.

## Why a patch instead of folding it into the canonical patch 2

The file is owned by patch 2, but rewriting patch 2 would change its `From <sha>` line and `git am`
content on the shared `wip-mmb-general` branch for no behavioural reason; a separate patch keeps the
merge-back a clean replay.  If the maintainer prefers, it folds into patch 2 at promotion time.

## Status

* `0007` (qsa3) is **validated** on gfx1100 at the unit level: `test-backend-ops -o FLASH_ATTN_QSA`
  26/26, and a `rocprofv3` kernel trace shows `qsa3_attn_kernel` + `qsa3_pack/merge/rows` dispatched.
  End-to-end qwen4exp performance is trust-RDNA3_5 (no qwen4exp model fits 24 GB).
* `0008` (mmb) **defaults the F32 MoE-router split off on RDNA3_0/RDNA4** (gfx1151 unchanged).
  With it, `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1` is a large gfx1100 win: 27B Q4_K_M +14.4 %,
  gemma-12B Q8_0 +11.3 %, 35B-A3B MoE +5.6 %/+4.7 %, gemma-26B-A4B neutral.  Decode unchanged,
  width purity PASS, PPL parity.
* `0009` (mmvq) is an **experimental, default-off** per-M `nwarps` scaffold; enabling it is a real
  MoE win but breaks width purity, so it is not enabled.  See `../gfx1100-s9-nwarps-results.md`.
* See `../gfx1100-s2s4-results.md` and `../gfx1100-s5s7-results.md` for the records.
