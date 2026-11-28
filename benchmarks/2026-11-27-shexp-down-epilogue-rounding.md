# The shared-expert down epilogue's rounding — `shexp_down_gated_q8_0` (r371, GitHub issue #377)

**Date:** 2026-11-27 · **Build:** the r370 tip and the r371 fix, both from `~/bin/build-llama-rocm`
(ROCm 10.2.0-gfx120X, clang 24, gfx1201 = 3× R9700) · **Model:** `Qwen3.6-35B-A3B-Q8_0.gguf` (the MoE
shared-expert shape; `ffn_down_shexp` is Q8_0 on every Qwen3.6 quant on this box, UD or not).

**Reporter:** @overdoingism, issue #377, from the source plus the emitted ISA of the r366 object (ROCm 10.0.0,
clang 23).  Everything below reproduces on the delivery's own toolchain, and adds what the report could not:
which arms *do* move, and what makes the defect observable at all.

## What the defect is

`shexp_down_gated_q8_0` stands in for three graph nodes, so its epilogue owes three roundings:

```
gated   = down * gate_sig        (MUL kernel, rounded)
ffn_out = gated + moe_out        (ADD kernel, rounded)
dst     = ffn_out + ffn_residual (ADD kernel, rounded)
```

The kernel wrote

```cpp
dst[o] = __fmul_rn(sum_down[t], gate_sig[t]) + moe_out[o] + ffn_residual[o];
```

under a comment asserting the multiply "must NOT be contracted into an FMA".  HIP clang contracts it anyway:
`__fmul_rn` lowers to a plain op and is contracted like any other (`GREEDY-PURITY.md` §102.2).  Decode ==
verify was never at risk — all 32 instantiations contract the same way — so this is a "the code says X, the ISA
says Y" defect, and the claim was the only thing broken.

## The mechanism, before and after

Extracted from the built object (`llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle
--targets=hip-amdgcn-amd-amdhsa--gfx1201`, `llvm-objdump -d`), 32 instantiations of
`shexp_down_gated_q8_0<4 nwarps × 8 ncols>`.

| build | epilogue chains ending in a store that carry a separate `v_mul_f32` | contracted (`v_fma_f32`/`v_fmac_f32`/`v_fma_mix_f32`) |
|---|---|---|
| r370 (pre-fix) | **2 of 146** | **144** |
| r371 (fixed) | **146 of 146** | **0** |

`<1,1>` before → after (the whole epilogue, `-lv`-free ISA):

```
-  v_fma_f32    v1, v1, s4, s5          # fma(sum_down, gate_sig, moe_out) -- ONE rounding
-  v_add_f32_e32 v1, s2, v1             # + ffn_residual
+  v_mul_f32_e32 v0, v0, v9             # gated   = round(sum_down * gate_sig)
+  v_add_f32_e32 v0, v0, v13            # ffn_out = round(gated + moe_out)
+  v_add_f32_e32 v0, v0, v18            # dst     = round(ffn_out + ffn_residual)
```

`<1,2>`..`<8,8>` before: `moe_out` loaded into the accumulator, then `v_fmac_f32_e32` (`+= sum_down*gate_sig`),
then `v_add_f32` (residual), per token.  After: a `v_mul_f32` and two `v_add_f32` per token, no fma form in
any epilogue.

**The scan has to be anchored on the STORE, not on the reduction.**  The obvious window ("everything after the
last `v_add_f32_dpp`") is right for `<1,1>`..`<4,8>` and wrong for `<8,8>`, where the compiler interleaves the
eight-token cross-warp reduction with the dot loop and software-pipelines the stores (measured on both objects).
Each output value is a chain that ends in a `global_store_b32`, so the gate looks back from every store for the
token's `v_mul_f32` and fails if a contracted form sits between the two.  That is `scripts/gate-shexp-down-gate.sh`
leg 1, and on the r370 object it reports **144 contracted chains across all 32 instantiations**.

## What the usual instruments can and cannot see

The fusion arms only at `down_mm->src[1]->ne[1] <= MMVQ_MAX_BATCH_SIZE` (8) tokens, so a batched prefill runs
the epilogue **only on its small tail ubatch** — which is the whole story below.

| arm (Qwen3.6-35B-A3B Q8_0) | r370 | r371 |
|---|---|---|
| greedy text, 3 GPU `-sm tensor`, short prompt (`-n 128`) | `2f79e37ca208` (518 chars) | `2f79e37ca208` |
| greedy text, 3 GPU `-sm tensor`, `prompts/prose-rdna-boosts.txt` | `a2cdef23588f` (504 chars) | `a2cdef23588f` |
| greedy text, 1 GPU `-ncmoe 40 --host-experts pool` | `2f79e37ca208` | `2f79e37ca208` |
| the same arms with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` (the chain) | `2f79e37ca208` / `a2cdef23588f` | same |
| prefill-logit KLD vs the chain, `-c 512 --chunks 40` 3 GPU `-sm tensor` | `-0.000000` / `100.000 %` | `-0.000000` / `100.000 %` |
| prefill-logit KLD r371 **vs the r370 build**, same point | — | `-0.000000` / `100.000 %` |
| two-chunk log-prob file (`--chunks 2`), r371 vs r370, byte compare | — | **identical** (0 of 63 322 400 finite floats differ) |
| **`-ub 1`** KLD, r371 vs the r370 log-prob base, 512 tokens, 3 GPU `-sm tensor` | — | **0.001354** / **98.431 %** (PPL 4.0962 → 4.0521) |

At `-ub 1` every token takes the fused epilogue and each step's hidden state feeds the next, so a 1e-7 shift
amplifies to a visible one (the first token already moves 53 of 248 324 log-prob codes, the third 21 691).  That
is the non-vacuity measurement: the fix changes the arithmetic, and the delivery's own prefill point is simply
blind to it.  Note also that the fused arm and the chain arm read the **same** text both before and after — the
fused gate reduction (`shexp_gate_sigmoid`) has its own, documented order difference from the standalone
`MUL_MAT` (`ggml-cuda-try-fuse.cu`), so the chain is not a text oracle for this fusion and never was.

## The one arm that does see it — and two stale anchors it exposed

`scripts/gate-server-arms.sh` (gate 24 step 5c) runs a **two-slot** `llama-server` with a DFlash drafter,
`--no-kv-unified`, 128 greedy tokens of the shipped prose prompt in both slots, so the two sequences verify in the
same decode step — i.e. the target runs verify batches of width > 1, where this epilogue runs too, and the
acceptance feedback loop carries a 1e-7 difference forward.  It is the only arm in the ladder that sees this change:

| 35b10 (35B-A3B Q8_0 + DFlash2, `-ncmoe 10`, two slots) | texts (sorted) | acceptance |
|---|---|---|
| the gate's recorded anchors | `50f29e574939 a56117e02dcf` | 0.60741 / 0.69106 |
| **r370 build** (pre-fix), twice | `50f29e574939 a56117e02dcf` — **anchors reproduce** | 0.60741 / 0.69106 |
| **r371 build**, three runs | **`10050d7d1b2f df3deb00c56b`** | **0.59559 / 0.60000** |

Deterministic on each build, so this is a one-time reference move of the same class as a K-partition change
(`GREEDY-PURITY.md` §111.2), not a purity loss: the arithmetic now matches the chain, and the reference text was
recorded on the contracted one.  Acceptance stays at the same level (~0.60, not a collapse).

**Two findings the run exposed, both fixed here.**

1. **`FN_D` named a file that no longer exists.**  The Flash-Next DFlash2 drafter moved to
   `/llm/models/Qwen3.8/Flash-Next/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` (`md5 65a35c6123…`, identical to the copy
   under `GSQ-RCO-IQ3_S/`), so `fn24`/`fn10` were failing as `FAIL (missing model or drafter)` before reaching a
   token.  The path is corrected in the gate.
2. **The Flash-Next anchors were already stale, and not from this change.**  With the path fixed, `fn24`/`fn10` read
   `9564942b467b e29d01e5edca` (0.69106 / 0.69672) on **both** builds, against the recorded
   `594368935245 b337648c3ef9` (0.59259 / 0.77193) — i.e. the pre-r371 tree already did not reproduce them, which is
   expected: r369 promoted the flash-next prefill transport/pipeline over blocks 09/10/12/15 in the same day
   (`archive/work/flash-next-prefill/`), and `ffn_down_shexp` on that model is IQ4_NL, so this epilogue cannot reach
   it at all.  All three arms are re-recorded from the r371 build (`--record`), and the gate is green: `fn24`/`fn10`
   `9564942b467b e29d01e5edca` 0.69106/0.69672, `35b10` `10050d7d1b2f df3deb00c56b` 0.59559/0.60000.

## The census

`GGML_CUDA_SHEXP_GATE_DEBUG=1` (`DIAGNOSTIC.md` §32), 4 generated tokens, 3 GPU `-sm tensor`:

| arm | `SHEXP DOWN GATE` lines |
|---|---|
| default | **600** (one per fused launch over the whole run: the prefill's tail ubatches plus each decode step) |
| `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` | **0** |

Before r371 there was no way to tell whether the kernel ran at all — and the two arms read the same text, so
"the census is not optional" (§102.5) was literal here.

## Rate

`llama-bench`, `-p 512 -n 128 -r 3`, 1 GPU, `-ncmoe 40 --host-experts pool`, `ROCBLAS_USE_HIPBLASLT=0`,
interleaved A/B/A rounds (2 rounds; this box's arm-to-arm spread is ±2 % on tg):

| build | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| r370 | 1656.3, 1646.0 | 63.6, 64.9 |
| r371 | 1663.8, 1654.4, 1664.5, 1672.0 | 62.5, 63.8, 63.8, 63.0 |

Flat inside the band, as it should be: the change adds two scalar instructions to a `threadIdx.x == 0` tail loop
of the epilogue, once per (row, token).

## Gates

* `scripts/gate-shexp-down-gate.sh` (new, `MANIFESTS.md` gate 48) — the ISA leg fails on the r370 object (144
  contracted chains) and passes on r371 (146/146 store-anchored chains, 32 instantiations); the census leg is
  600 / 0; each text arm is reproducible; the kernel journal is clean over the gate's window.
* `scripts/validate-set.sh` — green.
* `scripts/gate-prefill-logits.sh` (canonical point, the recorded r370 base): still `0.000000 / 100.000 %`, i.e.
  the dense 27B point is untouched (it has no shared expert) and the delivery's base does not need re-recording.
* `scripts/gate-server-arms.sh` — 3/3 PASS **after** the anchor re-record described above (its `35b10` anchors moved
  with this change; its `fn24`/`fn10` anchors were already stale and its `FN_D` path was stale, both fixed here).
* `scripts/gate-glu-inplace.sh` — PASS, and its host-expert MoE arm reads `2f79e37ca208` both ways, i.e. the same
  text as the greedy arms above.
* `test-backend-ops test -o MUL_MAT_ID` — PASS on the ROCm backend.
* `llama-batched-bench -npl 1,4,8` (35B Q8_0, 3 GPU `-sm tensor`, PP 512 / TG 128): 2866 / 4343 / 4615 t/s prefill
  and 109 / 307 / 469 t/s generation — informational; no baseline arm.
* `scripts/check-shlib-symbols.sh` — no new undefined symbol (the helpers are `static __device__`).
