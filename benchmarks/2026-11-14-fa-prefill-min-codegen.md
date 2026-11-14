# One `min()` in the FA tile loader costs the head-256 prefill 13 % (2026-11-14, r281 — GitHub issue #311)

Why this file exists: issue **#311** reported that the r278 re-base (`4f9dfa0e3`) lost ~4 % of deep prefill
against the r250 tree (`a691b2345`) on a dense Qwen3.8-27B, with bit-identical text and identical draft
acceptance.  The regression is real, it is **not** in the delivery's own code, and it is **one line**: upstream
#30285 (`cuda : fattn mma head sizes 40 72`) wrote the K/V tile-loader bound as
`min(k0_stop, DKQ/2) - k0_start` so a **padded** head (40, 72) never reads past its real head size.  For an
unpadded head that `min()` against a constant is a no-op that is *not free*: it changes codegen for the whole
`flash_attn_ext_f16` instantiation and costs the RDNA4 head-256 prefill kernel **12.4 %** of its own time.

**Do not re-derive these numbers, and do not "simplify" the clamp back to a bare `min()`.**  The fix keeps
upstream's clamp for `DKQp > DKQ` and spells the unpadded arm as a constant-folded plain subtraction
(`DKQp == DKQ ? k0_stop - k0_start : min(k0_stop, DKQ/2) - k0_start`), which restores the pre-#30285
instruction selection *exactly* (same ISA instruction counts, below).

## The regression, on the reporter's shape

1 × R9700 (gfx1201, 32 GB), Linux, ROCm 10.1.0, the reporter's env
(`GGML_CUDA_DISABLE_GRAPHS=1 ROCBLAS_USE_HIPBLASLT=0 GGML_CUDA_ASYNC_INPUT=1
GGML_CUDA_FA_STAGE_PREALLOC=2 GGML_CUDA_FA_STAGE_MAX_MB=256`), dense **Qwen3.8-27B UD-Q5_K_XL**
(`/llm/models/Qwen3.8/27B/Q5_K_XL/Qwen3.8-27B-UD-Q5_K_XL.gguf`), `q8_0` K/V, FA on,
`llama-bench -ngl 99 -p 512,4096 -d 0,32768 -ub 512 -b 1024 -r 2`:

| test (t/s) | r250 `a691b2345`+16 | r280 `4f9dfa0e3`+17 | r281 (fix) | reporter P → R |
|---|---|---|---|---|
| pp512 @ d0 | 1251.7 | 1217.6 (−2.7 %) | **1254.4** | — |
| pp4096 @ d0 | 1226.9 | 1222.6 (−0.4 %) | **1224.9** | 1273.4 → 1265.3 (−0.6 %) |
| pp512 @ d32768 | 878.0 | 842.0 (−4.1 %) | **881.6** | 904.6 → 870.3 (−3.8 %) |
| pp4096 @ d32768 | 875.6 | 839.4 (−4.1 %) | **876.0** | 928.1 → 890.1 (−4.1 %) |
| tg128 @ d32768 | 25.19 | 25.22 | 25.21 | flat |

The same arm with `-ctk/-ctv f16` loses the same amount (r250 881.1 → r280 846.2 at `pp4096 @ d32768`), so
this is **not** the quantized-KV/staging path; it is the kernel.

## Where the time goes (rocprofv3 `--kernel-trace`, pp512 @ d30720, q8_0 KV)

| kernel | r250 | r280 | Δ |
|---|---:|---:|---:|
| `flash_attn_ext_f16<256,256,32,2,…>` (992 dispatches) | 4543.7 ms | 5108.2 ms | **+12.4 %** |
| `mul_mat_q<Q6_K,…>` (10 850) | 10188.2 ms | 10169.5 ms | −0.2 % |
| `mul_mat_q<Q5_K,…>` (10 788) | 9091.9 ms | 9083.3 ms | −0.1 % |

**Only the FA prefill instantiation moves.**  Launch geometry, resources and every runtime argument are
identical between the two builds (instrumented print): `nthreads 256`, `nbatch_fa 64`, `nbatch_K2/V2 128`,
`nbatch_combine 64`, `nstages 0`, `shmem 38400`, `swz 0`, `Q_in_reg 1`, same `grid 6144×8`, same VGPR 256 /
SGPR 128 / scratch 640, same 992 dispatches.  The `flash_attn_ext_f16<256,256,4,8,…>` band kernel (decode and
the MTP verify step) is unchanged (320.6 vs 321.9 ms), which is why decode does not move.

**It is codegen, and it is the TU.**  Extracting the device ISA from the two builds' instance objects
(`template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2-dkq256-dv256.cu.o`):

| | symbols in TU | `flash_attn_ext_f16<256,256,32,2,false…>` | `flash_attn_ext_f16_load_tile_rows<132,…>` | object |
|---|---:|---:|---:|---:|
| r250 | 230 539 | 107 342 | 4 594 | 1 501 968 B |
| r280 | 233 555 | 108 759 | 4 755 | 1 523 472 B |
| r281 (fix) | 230 539 | **107 342** | **4 594** | 1 501 096 B |

Compiling the **r250 header** with the r280 compiler, flags and `common.cuh` reproduces r250's numbers exactly,
so the trigger is inside `fattn-mma-f16.cuh`.  Bisecting that header's 34 hunks (r250 → r280) against the
instruction counts puts the jump on one hunk: `min(k0_stop, DKQ/2) - k0_start` (the `i0_diff` twin in the V
loader is the same shape and is spelled the same way in the fix).  The added heads-40/72 config rows, tile
specializations, `DECL_…` lines and the `DKQp/DVp` padding are all **not** the trigger (each was removed in
isolation and the counts stayed at 108 759 / 4 755).

## Gates on the shipped tree (`549f379e6`, tip `69296dde5`)

* `test-backend-ops -o FLASH_ATTN_EXT` **6997/6997** (the whole instance matrix, padded heads 40/72 included).
* `scripts/gate-prefill-logits.sh` **f16 0.000697 / 98.784 %** and **`--kv q8_0` 0.000580 / 98.882 %** —
  the same numbers r279/r280 read against the same bases, i.e. the fix is numerically a no-op.
* `test-logits-width-probe` (27B, P=3584): `width_purity=PASS`, worst maxdiff 0.
* Coherence: Qwen3.5-4B Q8_0 `196 chars sha=83eec5e9b4f0` on 1 GPU and 3 GPU `-sm tensor`, identical before and
  after the fix.
* MTP at depth (`llama-cli -f <62k tokens> -n 128 --spec-type draft-mtp --spec-draft-n-max 3`, `-c 65536`,
  `q8_0` KV, acceptance 13/26 live): generation **47.0 / 46.9 / 47.1 t/s** for r280 / r281 / r250, prompt
  **854.4 → 882.4 / 881.4 t/s**.  **Decode does not move**, so the decode half of issue #311 is not this
  cause (and it is not reproducible here at any of the three arms).

## The lesson

A semantically-neutral clamp in a device template is a **codegen** change for an instantiation it is dead in.
The delivery's own answer to "the build is slow" (`archive/work/build-time-regression/`) is the same lesson
from the other side: what a `__device__` function is *spelled* like matters even when the arithmetic is
provably identical, so a prefill change is judged by the kernel census and ISA, never by reading the diff.
This is also why a prefill regression can hide from text, PPL and acceptance gates: all three were identical
through the whole regression.
