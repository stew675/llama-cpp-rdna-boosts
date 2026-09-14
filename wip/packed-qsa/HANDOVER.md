# Packed-QSA — session handover (2026-09-13)

Read `PORT-PLAN.md` first; this file is the state + resume sheet.

## Repository / branch state

| repo | branch | state |
|---|---|---|
| `~/llama-cpp-rdna-boosts` (delivery) | **`packed-qsa`** (off `main` `2ebf725`) | `wip/tiled-gdn/`, `wip/prefill-arrangements/`, `wip/packed-qsa/`; pushed to `origin/packed-qsa` |
| `~/llama-cpp-rdna-boosts` (delivery) | `main` | untouched, at `2ebf725` (has `wip/tiled-gdn` + `wip/prefill-arrangements`) |
| `~/llama.cpp` (fork) | **`packed-qsa`** (off `rdna-boosts`) | `91f5e41a0` = **P3 part 1** (gfx12 f16 WMMA primitive + layout self-test) on `1697ad10e` = **P2** on `2b84c7c62` = **P1** on `b214621da` = the tiled-GDN spike |
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

1. ~~**P1** graph pack (`qsa_pack_keys`/`values`) + the two new `GGML_OP_FLASH_ATTN_QSA` sources +
   the shape/type gate.~~ **DONE 2026-09-13** — fork `2b84c7c62`, record in `P1-NOTES.md`
   (22/22 `FLASH_ATTN_QSA`, gate-off == gate-on same-seed text `b72fb4d76af5`, pack built,
   `n_blocks=256`).  See its "What P2/P3 need to know" for the call-outs.
2. ~~**P2** port `qsa3_rows_kernel` + `qsa3_merge_kernel` (union + membership descriptor);
   unit-check the union/mask against the `idx` rows.~~ **DONE 2026-09-13** — fork `1697ad10e`, record in
   `P2-NOTES.md`.  Host cross-check + 202-case self-test both green; the self-test found and fixed a
   duplicate-index block-split in the reference merge kernel.  See its "What P3 needs to know".
3. **P3** (part 1 **DONE 2026-09-13**, fork `91f5e41a0`) — the gfx12 f16 WMMA primitive
   (`qsa_mma_f16` / `qsa_frag_load`, the "two runs of four" 8-half fragment) is validated by
   `PACKED-QSA WMMA self-test: OK`, and the kernel design (grid/block, 8-way head-dim split,
   fragment address mappings, LDS P-transpose, masking) is in `P3-DESIGN.md`.  **Part 2: implement
   the kernel body** per that design, then A/B vs the VEC kernel.  Visibility source to resolve there.
4. **P4** support predicate + dispatch + VEC fallback (RDNA4 gfx1201 and RDNA3.5 gfx1151); decide
   the `-sm tensor` pack layout (currently asserts mirrored).
5. **P5** validate: op correctness vs VEC, PPL, `W=1..8` (packed is prefill-only), MTP acceptance,
   perf A/B (`flash_attn_qsa` >= 2x; qwen4exp ~+117 t/s).

### Decisions taken / still open
- **Extend `GGML_OP_FLASH_ATTN_QSA`** with `src[7]/src[8]` — **chosen (P1)**.
- **Purity gate:** prefill-only (`n_query >= 128`), VEC band for `W=1..8` — **chosen (P1)**.
- **Visibility (P3):** consume the derived `cell_vis`/`q_vis` (maskless) path or the base `mask`; the
  packed membership must reproduce the VEC path's visibility exactly.
- **`-sm tensor` pack layout (P4):** the splitter asserts the pack is mirrored; a kv-head-split pack
  is not representable.  Confirm the mirrored full-pack path or gate it off under tensor split.

## Environment / model notes

- Dev box: `soar`, 3x R9700 (gfx1201, RDNA4), ROCm 7.14 at `/opt/rocm-7.14-gfx1201`.
- pwilkin's numbers come from the **gfx1151 (Strix Halo)** box; the 2.1 s QSA share is a gfx1151
  measurement.  Confirming it on gfx1201 is a heavy run (3-GPU, ~93-105 GB model) and was deferred.
- The fork source is at `~/llama.cpp`; the reference at `~/pwilkin-llama-cpp` (`strix-halo`).
- QSA benchmark shape: D=256, 24 q-heads / 2 kv-heads (gqa 12), `n_top_k` up to 2051, prefill
  `n_query` >= 128.

## Reading order

1. `PORT-PLAN.md` (this tree) — the implementation plan.
2. `P1-NOTES.md` / `P2-NOTES.md` / `P3-DESIGN.md` (this tree) — the P1/P2 records + the P3 kernel design.
3. `../prefill-arrangements/README.md` — the arrangement landscape + the GDN-is-spent argument.
4. `../tiled-gdn/05-where-the-speed-comes-from.md` — where the journey's ~2.2x really is.
5. `../../archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md` —
   the archived `mmb` port + the kernel attribution (and its §12 purity rationale).
