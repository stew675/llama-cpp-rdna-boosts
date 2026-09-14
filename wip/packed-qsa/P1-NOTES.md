# Packed-QSA — P1 implementation notes

**Status:** P1 done (2026-09-13).  Fork branch `packed-qsa`, commit `2b84c7c62`.  Plan:
`PORT-PLAN.md`; state: `HANDOVER.md`.

## What P1 did

Graph-side plumbing only — no kernel change.  All of it default-off, so the delivered build is
byte-identical.

| file | change |
|---|---|
| `ggml/include/ggml.h` | declare `ggml_flash_attn_qsa_set_packed` |
| `ggml/src/ggml.c` | implement it: optional `src[7]`/`src[8]` on `GGML_OP_FLASH_ATTN_QSA`, with the packed shape contract asserted |
| `ggml/src/ggml-backend-meta.cpp` | `handle_flash_attn_qsa` now tolerates `src[7]`/`src[8]` (mirrored-or-unknown) |
| `src/models/qwen4exp.cpp` | `qsa_pack_keys`/`qsa_pack_values` (plain `reshape`/`permute`/`cont`) + `qwen4exp_qsa_packed()` gate + build/attach in `build_attn_qsa` |

### The op interface

```c
ggml_flash_attn_qsa_set_packed(a, packed_keys, packed_values);   // NULL/NULL clears
```

* `packed_keys`  `[16, 4, 16, n_blocks]` F16, contiguous, `n_blocks = k->ne[1]/4 * k->ne[2]`
* `packed_values` `[4, 256, n_blocks, 1]` F16, contiguous
* Asserted against `a->src[1]` (the permuted K): `k->ne[0] == 256`, `k->ne[1] % 4 == 0`.

### The gate (`LLAMA_QSA_PACKED`, default 0)

```
qwen4exp_qsa_packed()
  && n_stream == 1 && n_tokens >= 128
  && cparams.type_k == F16 && cparams.type_v == F16
  && k->ne[0] == 256 && k->ne[1] % 4 == 0
  && q_p->ne[2] == 12 * k->ne[2]            // gqa 12
```

Mirrors pwilkin's `layout_prefill`.  When the gate is off (default) `src[7]`/`src[8]` stay NULL and
`fattn-qsa.cu` (the VEC kernel) is exactly as before.

## Validation (gfx1201, 3x R9700, `-sm layer`)

* `test-backend-ops test -o FLASH_ATTN_QSA -b ROCm0` — **22/22**.
* `Qwen3.8-Flash-Next-UD-IQ3_XXS`, 512-token prefill, `-ctk f16 -ctv f16`,
  `LLAMA_QSA_DENSE_SHORTCUT=0` (to reach the sparse QSA op below the selection width):
  gate-off and gate-on same-seed text **identical** (`b72fb4d76af5`); prefill 125.4 vs 124.7 t/s
  (the pack's copy cost — the fingerprint that it fired).
* Pack fired on the real prefill: `n_kv=512, n_kv_heads=2, n_blocks=256, n_tokens=512` (the
  `set_packed` assertions passed).  The load-time `resolve_fused_ops` probe graph is also a
  `build_attn_qsa` caller, at `n_tokens=64` — correctly skipped by the `>= 128` gate.

## What P2/P3 need to know

1. **The graph is built more than once.**  The load-time `resolve_fused_ops` probe calls
   `build_attn_qsa` with `n_tokens=64`; the real prefill follows.  Any per-graph state (the
   one-shot log) or counter is shared across both.  Do not assume one build per run.
2. **`-sm tensor` + packed is unvalidated (P4).**  The pack folds `(n_kv, n_kv_heads)` into one
   block dimension, so a **kv-head-split** pack is not representable; the meta splitter now asserts
   `src[7]/src[8]` are mirrored-or-unknown, i.e. the pack must be **mirrored** (each device holds
   the full pack and reads its own kv-head blocks).  `-sm layer` is the validated config.
3. **Visibility is unresolved (P3).**  The pack is built independent of the mask/derived-vis arm.
   The reference `qsa3` is `maskless`; our VEC path has both a base `mask` and the derived
   `cell_vis`/`q_vis`.  P3 must pick one and prove the packed membership reproduces the VEC path's
   visibility exactly (a mismatch here is a silent quality bug — `GREEDY-PURITY.md` §21).
4. **`libllama` `LLAMA_LOG_INFO` is suppressed under `llama-cli`** (the server wrapper's callback
   filters to WARN), so the one-shot pack-built log is not visible there; use the timing
   fingerprint or a temporary `fprintf` when debugging.  The instrumentation used to confirm P1 was
   removed after the check.
5. **The `n_tokens >= 128` gate is what keeps `W = 1..8` on the VEC kernel** — do not lower it, and
   do not default the packed path on until the P5 gates pass.

## Next: P2

Port `qsa3_rows_kernel` + `qsa3_merge_kernel` (the union + membership descriptor) and unit-check the
union/mask against the `idx` rows.  Then P3 (`qsa3_attn_kernel` + **gfx12 f16 WMMA fragments** — 8-half
vs the reference's gfx11 16-half; `mma.cuh:1232` vs `:1239`).
