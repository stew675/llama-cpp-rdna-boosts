# gfx1100 patch overlay

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

`0007` applies clean on top of the 6-patch tree.  It changes one file
(`ggml/src/ggml-cuda/fattn-qsa3.cu`): `ggml_cuda_flash_attn_qsa3_supported()` gains `RDNA3_0`, so the
packed-block WMMA QSA path runs on gfx1100 (gfx11 fragment arm, identical to gfx1151).

## Why a patch instead of folding it into the canonical patch 2

The file is owned by patch 2, but rewriting patch 2 would change its `From <sha>` line and `git am`
content on the shared `wip-mmb-general` branch for no behavioural reason; a separate patch keeps the
merge-back a clean replay.  If the maintainer prefers, it folds into patch 2 at promotion time.

## Status

* `0007` is **validated** on gfx1100 at the unit level: `test-backend-ops -o FLASH_ATTN_QSA` 26/26,
  and a `rocprofv3` kernel trace shows `qsa3_attn_kernel` + `qsa3_pack/merge/rows` dispatched.
  End-to-end qwen4exp performance is trust-RDNA3_5 (no qwen4exp model fits 24 GB).
* See `../gfx1100-s2s4-results.md` for the record.
