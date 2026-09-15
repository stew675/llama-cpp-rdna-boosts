# Packed-QSA — Strix Halo / gfx1151 handover (2026-09-15)

**This file is self-contained and is the entry point for the gfx1151 agent.**  Read it top to
bottom.  Parent record: `HANDOVER.md` (gfx1201, now closed), results: `P3.5-NOTES.md`.

---

## 0. Why gfx1151 and where the gfx1201 work landed

We ported pwilkin's packed-block WMMA QSA attention into the delivery's QSA path.  On gfx1201 the
port now **wins +2.8 / +3.5 / +3.9 %** at ub2048/4096/6144 but stays **under the 5 % bar**, so
gfx1201/gfx1100 are closed as *not applicable* (fast RDNA4 VEC baseline + a ~5 % fixed merge share on
a 3.6 s pass).  On gfx1151 (this box) the **same approach wins +9…16 %** — finish it here.

The one durable kernel result from the gfx1201 push: the **gfx12 softmax transpose** (each lane
owns a row + 8 keys so the row softmax is 1 shuffle instead of 32).  It is *gfx12-specific* — the
gfx11 D layout already gives one row per lane, so the natural gfx11 form is already cheap.  See
`P3.5-NOTES.md` §9.3 for the gfx1201 numbers and the rejected variants (don't re-try them).

## 1. The box

* `ssh halo` (user `stew675`, Fedora, kernel 7.2.4).  UMA: 123 GiB RAM shared with the iGPU;
  `rocm-smi` shows a small "VRAM" but llama.cpp HIP uses GTT — a 87 GiB model + `-ub 16384` fits.
* ROCm: **`/opt/rocm-7.14-gfx1151`** (also 7.12).  Always
  `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib` (the built binaries also carry an rpath).
* Models: **`/llm/models/Qwen3.8/Flash-Next/`** — `IQ4_NL`, `IQ4_XS`, `Q4_K_M`, `Q4_K_XL`,
  `Q5_K_M-PLEQ4_0`.  The validation below used
  `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (87 GiB, 3 shards; pass the first shard and llama.cpp follows the split).
  Geometry: 48 layers, D=256, 24 q-heads / 2 kv-heads (gqa 12), `indexer.top_k=2048`, ratio 4 on
  every 4th layer → the QSA op fires on 12 layers.

## 2. Repos on `halo` (all already present)

| path | branch | state |
|---|---|---|
| `~/llama.cpp` | `rdna-boosts` @ `4328ce4cd` | the r5 delivery tree (`d735d6c1`, == `release.json.tree`); `build-rocm/` configured for gfx1151 |
| `~/llama-cpp-rdna-boosts` | `main` @ `a0d1de7` | delivery docs (r5) — **the packed-qsa docs are NOT here yet**, see §4 |
| `~/pwilkin-llama-cpp` | `strix-halo` @ `f5daaa3cf` | pwilkin's reference; **already built** at `~/pwilkin-llama-cpp/build-rocm` |

Build script: `~/bin/build-llama-rocm-714` (it rebuilds `build-rocm` from scratch, `gfx1151`).

## 3. What is proven on gfx1151 (reproduce this first)

pwilkin's **own** gfx11 kernel, `pp8192`, IQ4_XS, `-ngl 99 -fa on -ctk f16 -ctv f16`; both arms
carry the same QSA env stack and only `LLAMA_QSA_FA_V3` differs:

```bash
ssh halo
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib
cd ~/pwilkin-llama-cpp
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
Q="LLAMA_QSA_SPARSE=1 LLAMA_QSA_WHOLE_ATTN=1 LLAMA_QSA_BLOCK_SELECTION=1 \
   LLAMA_QSA_COMPACT_METADATA=1 LLAMA_QSA_DENSE_SHORTCUT=1 LLAMA_QSA_DIRECT_INDICES=1 \
   LLAMA_QSA_PACK_KEYS=1 LLAMA_QSA_PACK_VALUES=1 LLAMA_QSA_SCORE_BOUNDS=1 \
   LLAMA_QSA_NO_DENSE_MASK=1"
for UB in 2048 8192 16384; do
  B="-m $M -ngl 99 -fa on -ctk f16 -ctv f16 -b $UB -ub $UB -p 8192 -n 0 -r 2"
  echo -n "ub$UB base:   "; env $Q LLAMA_QSA_FA_V3=0 ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
  echo -n "ub$UB packed: "; env $Q LLAMA_QSA_FA_V3=1 ./build-rocm/bin/llama-bench $B 2>/dev/null | grep pp8192
done
```

Measured 2026-09-15:

| ubatch | base | packed | delta |
|---|---:|---:|---:|
| 2048 | 583.72 | 636.82 | **+9.1 %** |
| 8192 | 613.34 | 706.71 | **+15.2 %** |
| 16384 | 614.09 | 711.51 | **+15.9 %** |

(The full pwilkin env set, including the MMB/HC families, is archived in
`~/llama-cpp-rdna-boosts/archive/work/wip-archive/iq4nl-prefill/launcher-env.txt`; the MMB family is
a *separate* win for the uniform-IQ4_NL model, not needed for the QSA A/B.)

**If the numbers reproduce, the approach is confirmed on this box.**  If they do not, check that
all 11 `Q` env vars are exported and that the model loaded (silence + a plausible t/s).

## 4. Getting the delivery's packed-qsa code onto `halo`

The delivery's packed-qsa port lives on the **`soar`** fork branch `packed-qsa`
(`6f7a38b3f`), not on `halo`.  It is a net diff against the r5 tree, which `halo`'s `~/llama.cpp`
already matches, so it applies cleanly:

```bash
# on soar:
git -C ~/llama.cpp diff rdna-boosts..packed-qsa > /tmp/packed-qsa.diff
scp /tmp/packed-qsa.diff halo:/tmp/
# on halo:
cd ~/llama.cpp && git checkout -b packed-qsa rdna-boosts && git apply /tmp/packed-qsa.diff
# verify the tree matches soar's branch tip tree:
git rev-parse packed-qsa^{tree}      # compare with `soar` git rev-parse packed-qsa^{tree}
```

(Do **not** `git apply` the delivery's 16-block series; that rule is about `patches/`, not this
single diff.  Never push `~/llama.cpp` anywhere.)

## 5. The actual gfx1151 work: the 16-half WMMA fragment

`ggml/src/ggml-cuda/qsa-packed.cu` currently compiles only the **gfx12** path:

```c
typedef _Float16 qsa_v8h __attribute__((ext_vector_type(8)));
qsa_mma_f16(a,b,c)  ->  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12   // #if defined(RDNA4)
                    ->  NO_DEVICE_CODE                                     // #else
```

gfx1151 (gfx11 / RDNA3.5) needs the **16-half** fragment:

```c
typedef _Float16 qsa_v16h __attribute__((ext_vector_type(16)));
qsa_mma_f16(a,b,c)  ->  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32         // #elif defined(RDNA3)
```

**Reference implementation: pwilkin's `~/pwilkin-llama-cpp/ggml/src/ggml-cuda/qsa.cu`** — his
`qsa3_attn_kernel` *is* the gfx11 kernel, and it already wins.  The fragment address mappings and
the differences from our gfx12 kernel:

* **Q (A)**: gfx11 stages Q into LDS then loads two `uint4` (16 halves); our gfx12 loads 8 halves
  directly with `qsa_frag_load` ("two runs of four").  See `qsa.cu` lines ~192-214.
* **K (B)**: `krow = pkg + kb*1024 + (r&3)*16 + w*128`, two `uint4` per `t`.  Our pack layout is the
  same, so only the load width changes.
* **V (B)**: `d = 32*w + 16*t + r`, one `uint2` from each of the 4 union blocks concatenated into 16
  halves.
* **D / softmax**: gfx11's accumulator gives each lane **8 columns of one row** (row = `16*w + r`,
  keys `4*(e>>1) + 2*(e&1) + hi`), so the row softmax is a local reduce + **one** `shfl_xor(...,16)`.
  **Do not port the gfx12 transpose** — it is unnecessary here (the gfx11 layout is already the
  transposed form).  This is the single most important thing not to get wrong.
* The P tile / LDS staging and the output store differ (pwilkin uses an LDS `ostage` for coalesced
  16-byte row stores).

Everything else in our `qsa-packed.cu` — the P1 pack contract, the P2 merge/descriptor
(`qsa3_rows_kernel` + `qsa3_merge_kernel`), the dispatch in `fattn-qsa.cu`, the
`GGML_OP_FLASH_ATTN_QSA` `src[7]/src[8]` plumbing — is architecture-independent and already correct.

**Also needed** (P4 leftovers, from `P3.5-NOTES.md` / P4 notes):
* the real dispatch predicate / VEC fallback (`ggml_cuda_flash_attn_qsa_supported()` accepts on
  shape/type alone today);
* decide whether the packed path stays default-off (it should, until P5).

## 6. Validation gates before any promotion (P5)

* `GGML_CUDA_QSA_MERGE_CHECK=1` → merge self-test + host cross-check **OK**;
  `GGML_CUDA_QSA_ATTN_CHECK=1` → `rel ≈ 3e-5` vs the VEC kernel on identical inputs.
* `test-backend-ops -o FLASH_ATTN_QSA` **22/22**.
* `W = 1..8` logits matrix **unchanged** (the packed path is gated `n_tokens >= 128`, so decode and
  spec-verify stay on the VEC kernel — `GREEDY-PURITY.md`).
* PPL vs the VEC build (the packed path is a prefill re-baseline).
* MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
* End-to-end A/B at `pp8192/16384` (and the pwilkin reproduction above).

Reminder: nothing here is a delivery change until it goes through the promotion path (`beta/`
staging, env-gated A/B, maintainer go-ahead).  This is `wip/` work.

## 7. gfx1201 facts to carry over (do not re-derive)

* The **P1 pack is free** (−0.1 %); the **P2 merge A kernel** is the cost (~5 % on gfx1201, ~1.3 %
  expected here because the pass is ~4× slower).  Rejected merge "fixes": fusing the check into
  kernel B (loses parallelism), check-only kernel A, K prefetch, smaller `cap`.
* Large ubatch helps **here** (the win grows 9→16 %); on gfx1201 the op speedup *declines* past
  n_q≈2 K (ub8192 was 0.97x) — different VEC baselines.
* Benchmark hygiene: run one arm at a time, never parallel; `-sm tensor` is wrong here (single
  iGPU) — use the plain single-device flags above.  Halo single-ubatch runs are noisy (±50 t/s at
  ub2048); the ub8192/16384 numbers were tight (±0.2–3).
