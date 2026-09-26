# Q2\_\*/IQ2\_\* model-tree scan + MMB IQ2 support (IQ2_S / IQ2_XS / IQ2_XXS), 2026-09-22

**Status:** IQ2 family ported, validated, **default ON** on non-RDNA4 (`patches/0018`).  Fork
`~/llama.cpp` branch `gap-closing-r13`, commit `bd984385e`.

## 1. The scan

A **header-only** GGUF scanner ([`tools/gguf-types.py`](tools/gguf-types.py)) — it reads the magic,
version, metadata-KV table and tensor-info table and never touches tensor data, so it runs over a
130+ GB model tree in seconds:

```sh
python3 archive/work/closing-the-gap/tools/gguf-types.py /llm/models
```

**Result: 132 `.gguf` files / 50 model dirs scanned, zero parse errors.**  Among `Q2_*`/`IQ2_*`,
only **`IQ2_S`** is present:

| model | IQ2_S tensors | shape | other types |
|---|---:|---|---|
| `MiniMax-M2.7/IQ3_S/MiniMax-M2.7-IQ3_S.gguf` | 124 | `ffn_gate_exps`/`ffn_up_exps` `[3072,1536,256]` (MoE) | Q6_K 250, IQ3_S 62, F32 373 |
| `DeepSeek/IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-*` (shards 2–4) | 84 | MoE experts | Q6_K, Q8_0, IQ3_XXS, IQ3_S, MXFP4, BF16, F32 |

No `Q2_K`, `Q2_0`, `IQ2_XS`, or `IQ2_XXS` anywhere.  (Unrelated finding: `Bonsai/Bonsai-8B.gguf` is
254× `Q1_0`; DeepSeek-V4-Flash-MXFP4/Q4_K_XL/Q8_K_XL are now MMB-accelerated by the `patches/0017`
MXFP4 support.)

## 2. What was added

The whole IQ2 family — `IQ2_S` (present) plus `IQ2_XS`/`IQ2_XXS` for completeness, because
`llama-quantize … IQ2_S` actually emits **IQ2_XS** on this box (a 105-tensor IQ2_XS model came out of
a plain `IQ2_S` request).

| type | WTYPE | block | bytes / 256 | value |
|---|---:|---|---:|---|
| IQ2_S | 16 | `half d; qs[64]; qh[8]; scales[8]` | 82 | `d*(0.5+s)*0.25 * iq2s_grid[qs[4ib+l] \| ((qh[ib]<<(8-2l))&0x300)][j] * sign` |
| IQ2_XS | 17 | `half d; qs[32] (uint16); scales[8]` | 74 | `d*(0.5+s)*0.25 * iq2xs_grid[q2[l]&511][j] * sign(ksigns_iq2xs[q2[l]>>9])` |
| IQ2_XXS | 18 | `half d; qs[32] (uint16)` | 66 | `d*(0.5+(a1>>28))*0.25 * iq2xxs_grid[aux8[l]][j] * sign(ksigns_iq2xs[(a1>>7l)&127])` |

Each `mmb_dq_row_iq2*` dequantizes one 64-value quarter of a 256-value super-block (`q4 = ksh & 3`
selects sub-blocks `ib = 2*q4, 2*q4+1`), the same shape as the existing `mmb_dq_row_iq3s`, and reads
the non-contiguous fields straight from the row base.  Wired through the dense, routed (`MUL_MAT_ID`)
and fused gate+up+GLU launches, the type masks/tables, and the `K % 256` super-block alignment check.

## 3. Gates (gfx1151)

| gate | IQ2_S | IQ2_XS | IQ2_XXS |
|---|---:|---:|---:|
| `MUL_MAT` oracle (MMB forced on, `GGML_CUDA_MMB_MIN_T=1`) | **14/14** | **14/14** | **46/46** |
| `MUL_MAT_ID` oracle (same) | **4/4** | **15/15** | **75/75** |

End-to-end:

* **Real model (MiniMax-M2.7-IQ3_S, iq2_s isolated with `GGML_CUDA_MMB_TYPES=iq2_s` vs `MMB=0`):**
  pp4096 **435.1 → 456.6 t/s (+4.9 %)**, PPL (wikitext, 4 chunks) **9.2623 → 9.3616 (+1.07 %)**.
* **Dense IQ2_XS model** (Nanbeige-3B requantized "IQ2_S", actually 105× IQ2_XS): PPL (16 chunks)
  **32.9266 → 32.9997 (+0.22 %)**, pp8192 **737.1 → 859.0 t/s (+16.5 %)**.
* IQ2_XXS has no model on this box — oracle-only.

The small PPL shifts are the MMB bf16-rounding numerics (the oracle proves the dequant); the ~+1 %
on the 2-bit MiniMax weights is the largest of the MMB set and is documented as the trade for +4.9 %
prefill.  Disable with `GGML_CUDA_MMB_TYPES` (drop `iq2_s`) if quality matters more.

## 4. Side finding (pre-existing, not this change)

`llama-imatrix` on `Nanbeige4.2-3B-BF16` fails with *"non-finite values detected in
blk.21.attn_output.weight"* with the default MMB build, and succeeds with `GGML_CUDA_MMB=0` **or**
`GGML_CUDA_MMB_BF16W=0`.  So the pre-existing **BF16-weight MMB dense path** (`bf16w`) produces
non-finite activations on that model.  This is independent of the IQ2 work (it reproduces with the
type mask irrelevant) and is recorded here as a follow-up, not fixed.

## Files

`ggml/src/ggml-cuda/mmb.cu`; scanner [`tools/gguf-types.py`](tools/gguf-types.py).  Patch:
[`patches/0018-…`](patches/).
