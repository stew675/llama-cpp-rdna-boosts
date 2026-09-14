# 04 — session handover (2026-09-13)

## What this session did

1. Scoped TODO item 1 (tiled Gated Delta Net) end to end.
2. Traced the source: `pwilkin/llama.cpp` `strix-halo` `964c6f2f0` (+ the mature
   `strix-halo-for-halobox` `8ab5a8373` form), and read pwilkin's own journey page.
3. **Recorded the numerics finding:** the tiled kernel is bit-exact (PPL identity with the
   sequential kernel on gfx1201); the 0.21 % PPL cost reported for the Strix Halo GDN work belongs
   to the *chunked* rewrite.
4. Built a **minimal prototype port** on gfx1201 (uncommitted, in `~/llama.cpp`) and measured it:
   op-level perf, end-to-end prefill, oracle correctness, and wikitext-2 PPL.
5. Wrote the documents in this tree.

## Current state

| item | state |
|---|---|
| this repo, branch `tiled-delta-net` | docs under `wip/tiled-gdn/` (committed) |
| `~/llama.cpp`, branch `rdna-boosts` | **prototype port applied uncommitted** in `ggml/src/ggml-cuda/gated_delta_net.cu`; patch saved at `wip/tiled-gdn/reference/port-spike-gated_delta_net.patch`; revert with `git checkout -- ggml/src/ggml-cuda/gated_delta_net.cu` |
| `~/llama.cpp/build-rocm` | built with the port; binaries reflect the spike |
| delivery `patches/` | untouched |
| pushes | none (per the pushing policy) |

There is a **pre-existing, unrelated uncommitted move** in this repo (`wip/iq4nl-prefill/` and
`wip/recurrent-rewind-depth/` → `archive/work/wip-archive/`).  It was not made by this session;
commit only `wip/tiled-gdn/` and leave it alone.

## The four facts to carry forward

1. **Tiled is exact.**  gfx1201 PPL: tiled == sequential == `6.5078` on all 64 chunks; chunked
   bf16 = `6.5092` (+0.0215 %, the chunked path's near-lossless deviation).
2. **Tiled is slower than what we ship.**  ~1.8× vs the sequential scan, but ~5× slower than the
   chunked bf16 default at the op level, and ~4 % slower end-to-end (pp2048/pp8192).  The entire
   GDN prefill lever is ~9 % on this box/model.
3. **The journey's ~2.2× is the prefill stack, not the GDN; the weight-set hinge is `mmb`.**  The
   repo already reproduced 1.77–1.79× on the Strix Halo box and ported pwilkin's `mmb` (parked,
   default off, +18.4 %); the residual is the QSA v3 sparse-attention kernel + HC fusions.  See
   `05-where-the-speed-comes-from.md` and `archive/work/wip-archive/iq4nl-prefill/`.
4. **The only likely wins are niche:** KDA prefill (chunked is non-KDA only), an exact opt-in /
   chunked-off fallback, and a gfx1100 fallback if the NW16 gfx11 chunked kernel does not fit.

## Recommended next experiments (in order)

1. **KDA prefill spike (highest expected value).**  Port
   `gated_delta_net_kda_tiled_128_cuda` from `strix-halo-for-halobox` `8ab5a8373` and A/B it
   against the sequential KDA kernel with the existing `kda=true` `test-backend-ops` cases.
   Requires a KDA model (Kimi-Linear / Kimi-K3) for end-to-end; if none is on the box, the op
   level is still decisive.
2. **Bitwise confirmation of the S_v=128 non-KDA port.**  Run the delivery's logits-hash /
   same-seed coherence gate and the `W=1..8` width matrix against the prototype.  PPL identity is
   strong but not bitwise.
3. **Generalize `S_v`.**  Add per-`S_v` `(NUM_WARPS, COLS, TOKEN_TILE)` configs for 64/32/16 and
   re-verify bit-exactness; this is what makes the fallback useful for non-128 models.
4. **Only then** decide on the delivery shape (new block vs block-02 amendment), and only as an
   opt-in / fallback, with `beta/` staging.

## Open questions for the maintainer

- Is a **bit-exact GDN prefill** worth a ~4 % prefill cost as an opt-in mode, given the repo's
  accuracy-first stance (BF16 KV)?  The default should stay chunked bf16 either way.
- Is the **KDA prefill** gap worth a block in its own right?  It is independent of the tiled-vs-
  chunked question and currently unimplemented on any path.
- Should the exact scan improvement be offered **upstream** as a standalone PR (it is generic and
  not RDNA-specific in concept), separate from this repo's delivery?

## How to resume

- Read `README.md` (verdict + numbers) → `01-kernel-analysis.md` → `02-port-assessment.md` →
  `03-validation-gfx1201.md`.
- The prototype patch is in `reference/`; apply it to a fresh `~/llama.cpp` at `rdna-boosts` and
  rebuild.  The exact commands are in `03-validation-gfx1201.md` §6.
- The environment: gfx1201 box, `/opt/rocm-7.14-gfx1201`, `HIP_VISIBLE_DEVICES=0`, one config per
  `test-backend-ops` invocation (never in parallel).
