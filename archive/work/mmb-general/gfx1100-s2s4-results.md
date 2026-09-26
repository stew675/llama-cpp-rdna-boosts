# gfx1100 port — S2/S3/S4 record (2026-09-21)

Sessions S2 (arch-neutral G5+G4), S3 (G3a policy) and S4 (G2 `qsa3` RDNA3_0) of
`gfx1100-porting.md`.  Raw evidence; the plan is the source of truth.

## S2a — G5 indexer

* The op is generic and always-on; `TOPK_QSA` is **4/4** on gfx1100 (S1).
* **No qwen4exp model fits 24 GB**, so its *performance* claim is trust-RDNA3_5 (§9 of the plan).
  There is no incidental effect on the available models: the S1 matrix showed WIP == delivery on
  every gate, and the indexer op is only instantiated by the qwen4exp graph.
* The delivery item (`GGML_OP_INDEXER_FILL` missing from `GGML_OP_NAME`) is **already fixed** by the
  WIP patch 5 (`ggml/src/ggml.c` lines 1088/1090) — no action.

## S2b — G4 non-temporal hints (the real S2 work)

Because the WIP is inert on non-qwen4exp models except for G4, the WIP-vs-delivery comparison on the
MoE models *isolated* the non-temporal hints.  To remove any doubt, a clean **NT-off** build was made
by replacing the hint loads with plain loads in exactly the three files that fire here
(`moe-weighted-reduction.cu` 4 sites, `concat.cu` 2 sites, `unary.cu` 1 site) and rebuilding
(`libggml-hip.so` md5 differs); the worktree was then restored and rebuilt.  Interleaved `r=5`:

### 35B-A3B Q3_K_M (MoE) — NT-off is consistently slightly *faster*

| point | NT-on | NT-off | Δ (off−on) |
|---|---:|---:|---:|
| pp16384 | 3423.23 / 3418.04 | 3425.99 / 3418.61 | +0.08 % / +0.02 % |
| pp32768 | 3006.14 / 3003.41 | 3011.20 / 3009.89 | +0.17 % / +0.22 % |

### gemma-26B-A4B (MoE) — a wash, sign flips with depth

| point | NT-on | NT-off | Δ (on−off) |
|---|---:|---:|---:|
| pp16384 | 2943.34 / 2941.32 | 2936.47 / 2936.26 | +0.23 % / +0.17 % |
| pp32768 | 2410.14 / 2414.87 | 2412.78 / 2409.42 | −0.11 % / +0.23 % |

### 27B UD-Q4_K_M (dense) — a flat wash

| point | NT-on | NT-off |
|---|---:|---:|
| pp8192 | 1022.33 / 1022.76 | 1022.44 / 1022.50 |
| pp16384 | 982.16 / 982.61 | 981.88 / 981.91 |

**Conclusion:** on gfx1100 the G4 non-temporal hints are **neutral** (all deltas ≤ ±0.23 %, and the
sign is model- and depth-dependent).  This is *unlike* gfx1201, where G4 was a consistent
+0.3-0.4 % at depth.  **Decision: no gfx1100 code change** — the hints stay arch-neutral/always-on,
because (a) no consistent loss is demonstrated, (b) the deltas are at the decision threshold, and
(c) gating them would add per-arch code for no measured gain.  The `dsv4_hc` hint is qwen4exp-only
(trust-RDNA3_5).  Recorded as a known-neutral item; revisit only if a future measurement shows a
larger effect at deeper context than 32k.

## S3 — G3a always-QSA policy

Cannot be measured on this box (no qwen4exp).  **Decision: keep the arch default — the dense
shortcut stays ON on gfx1100** (`qsa_arch_gfx() != 0x1151`), i.e. the delivery's behaviour.  This is
the conservative choice given that gfx1201 measured always-QSA as a large regression while qsa3 was
not yet validated there, and qsa3 on gfx1100 is only now enabled.  Trust-RDNA3_5 (§9).  **No code
change.**

## S4 — G2 `qsa3` on RDNA3_0 (the S4 deliverable)

**Change:** `ggml_cuda_flash_attn_qsa3_supported()` gained `RDNA3_0`
(`ggml/src/ggml-cuda/fattn-qsa3.cu`) plus the comment update.  gfx1100 takes the compile-time gfx11
fragment arm — the same code as the validated gfx1151 path.

**Validation (this box):**

1. `test-backend-ops -o FLASH_ATTN_QSA` → **26/26 passed** (2/2 backends).  Unlike the S1 run, the
   predicate now accepts gfx1100, so the four packed (`qsa3=1`) cases genuinely take the WMMA path.
2. **`rocprofv3` kernel trace** (`--kernel-trace`, `libggml-hip.so` md5 `91258d51…`) confirms the
   path actually dispatches on gfx1100:

   ```
   3  qsa3_attn_kernel
   3  qsa3_pack_keys_kernel
   3  qsa3_pack_values_kernel
   3  qsa3_merge_kernel
   3  qsa3_rows_kernel
   ```

   (3 packed cases × 1 launch each; the VEC `flash_attn_qsa` kernels run the remaining cases.)

3. **End-to-end qwen4exp performance is not measurable here** (the 94 GiB model does not fit 24 GB).
   The gfx11 kernel is the gfx1151-validated one, so the prefill win claim is **trust-RDNA3_5**
   (§9); the unit oracle + kernel dispatch are the gfx1100 evidence.

**Packaging:** the change is committed on the code branch `mmb-gfx1100` and exported as
**`wip/mmb-general/gfx1100/patches/0007-WIP-qsa3-RDNA3_0-gfx1100.patch`** (sha256 `cf81ff97…`),
applied after the canonical 6 patches — see `wip/mmb-general/gfx1100/README.md`.

## Carry-forward

* **S5-S7 (G1 `mmb`)** is the headline and the next unit of work.  Open it with
  `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`, confirm it fires + PPL parity, then sweep the per-type /
  per-path matrix (`GGML_CUDA_MMB_TYPES`, `GGML_CUDA_MMB_DENSE`) on the 35B-A3B / gemma-26B MoE and
  the 27B-Q4_K_M / gemma-12B-Q8_0 dense models.
* **S9** must re-check the delivery re-examination items (§2.3-§2.7 of the plan), which the S1-S4
  work has not touched: the `mmvq` RDNA3_0 `nwarps` table, the `VDR_Q8_0` MoE choice, the FA head
  cap 256 (the gemma-4 models are the probe), the block-13 MoE fusion vs MMB routed, and the
  native-KV auto policy.
