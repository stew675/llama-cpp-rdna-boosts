# Packed-QSA — session handover (2026-09-13)

Read `PORT-PLAN.md` first; this file is the state + resume sheet.

## Repository / branch state

| repo | branch | state |
|---|---|---|
| `~/llama-cpp-rdna-boosts` (delivery) | **`packed-qsa`** (off `main` `2ebf725`) | `wip/tiled-gdn/`, `wip/prefill-arrangements/`, `wip/packed-qsa/`; pushed to `origin/packed-qsa` |
| `~/llama-cpp-rdna-boosts` (delivery) | `main` | untouched, at `2ebf725` (has `wip/tiled-gdn` + `wip/prefill-arrangements`) |
| `~/llama.cpp` (fork) | **`packed-qsa`** (off `rdna-boosts`) | `b214621da` = the tiled-GDN spike (`GGML_CUDA_GDN_TILED`), carried over |
| `~/llama.cpp` (fork) | `rdna-boosts` | clean, at the 16-block delivery tip |

**Rule for this work:** everything goes on `packed-qsa` in both repos.  Never `main`/`master`, never
push the fork.  (`~/llama.cpp/build-rocm` currently reflects the tiled-GDN spike; rebuild before
trusting any perf number.)

## The crystallised conclusion

The qwen4exp prefill gap on the Strix Halo box (pwilkin ~1.8x) decomposes into **three independent
arrangement classes**, not one:

| class | artifact | needs uniform IQ4_NL? | gap share |
|---|---|---|---|
| weights -> bf16 WMMA | `mmb` | **yes** (IQ4_NL-only predicates), removable per-type | −33 % family |
| **selected KV -> packed f16 blocks + WMMA** | **`qsa3`** | **no** (F16 KV only) | **~2.1 s / ~+117 t/s** |
| streams/layout -> bf16 marking, HC, conv1d | `mark_bf16_only`, `hc-cn`, `gdn-conv` | no | ~+74 t/s |

Facts that matter and are easy to lose:

- The **tiled GDN is exact** (bit-neutral); the "0.21 % PPL" was the **chunked** rewrite's
  near-lossless bf16.  The GDN is already ~1 % of a prefill pass and already beats pwilkin's tiled
  kernel on gfx1201 — it is **not** the lever.
- The **uniform-weight requirement belongs to `mmb`** (weight GEMM), not to QSA.  `qsa3` is gated on
  **F16 KV cache**, never on weight type.  It therefore applies to our mixed-expert checkpoints with
  no PPL-per-bit trade.
- All qwen4exp models on this box are **mixed-expert** (UD-IQ3_XXS / UD-IQ4_XS / UD-Q4_K_XL), so
  pwilkin's IQ4_NL-only `mmb` would not fire on any of them without a per-type dequant kernel.
- The QSA win is the **zero-format-constraint** half of the gap — which is why it is the first move.

## Resume: the next concrete steps

Implement `PORT-PLAN.md` P1 -> P5 on `packed-qsa`:

1. **P1** graph pack (`qsa_pack_keys`/`values`) + the two new `GGML_OP_FLASH_ATTN_QSA` sources +
   the shape/type gate.
2. **P2** port `qsa3_rows_kernel` + `qsa3_merge_kernel` (union + membership descriptor).
3. **P3** port `qsa3_attn_kernel`; **re-derive the f16 WMMA fragments for gfx12** (8-half) vs the
   reference's gfx11 (16-half) — `mma.cuh:1232` vs `:1239`.
4. **P4** support predicate + dispatch + VEC fallback (RDNA4 gfx1201 and RDNA3.5 gfx1151).
5. **P5** validate: op correctness vs VEC, PPL, `W=1..8` (packed is prefill-only), MTP acceptance,
   perf A/B (`flash_attn_qsa` >= 2x; qwen4exp ~+117 t/s).

### Decisions to make in P1/P3
- **Extend `GGML_OP_FLASH_ATTN_QSA`** with `src[7]/src[8]` (preferred) vs a new op.
- **Purity gate:** prefill-only (`n_query >= 128`) keeping the VEC band for `W=1..8` (safe) vs
  band-uniform (needs its own width matrix).
- **Visibility:** consume the derived `cell_vis`/`q_vis` (maskless) path or the base `mask`; the
  packed membership must reproduce the VEC path's visibility exactly.

## Environment / model notes

- Dev box: `soar`, 3x R9700 (gfx1201, RDNA4), ROCm 7.14 at `/opt/rocm-7.14-gfx1201`.
- pwilkin's numbers come from the **gfx1151 (Strix Halo)** box; the 2.1 s QSA share is a gfx1151
  measurement.  Confirming it on gfx1201 is a heavy run (3-GPU, ~93-105 GB model) and was deferred.
- The fork source is at `~/llama.cpp`; the reference at `~/pwilkin-llama-cpp` (`strix-halo`).
- QSA benchmark shape: D=256, 24 q-heads / 2 kv-heads (gqa 12), `n_top_k` up to 2051, prefill
  `n_query` >= 128.

## Reading order

1. `PORT-PLAN.md` (this tree) — the implementation plan.
2. `../prefill-arrangements/README.md` — the arrangement landscape + the GDN-is-spent argument.
3. `../tiled-gdn/05-where-the-speed-comes-from.md` — where the journey's ~2.2x really is.
4. `../../archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` —
   the archived `mmb` port + the kernel attribution (and its §12 purity rationale).
