# MEASUREMENTS — issue #30 follow-up

Raw runs and derived tables.  Every row states the arm, ROCm, GPU, command and log; conclusions live in
`README.md`.  One arm at a time, nothing else on the GPU.

Environment unless stated otherwise: 3x R9700 (gfx1201), ROCm 7.14 (`HIP 7.14.60850`), 1 GPU
(`HIP_VISIBLE_DEVICES=0`), model `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes; qwen35, 65
blocks, 24 q-heads / 4 kv-heads, key=value=256, `full_attention_interval=4` → 16 full-attention +
48 GDN layers), built with `~/bin/build-llama-rocm-714`.

Arms:

| arm | path | identity |
|---|---|---|
| **A** stock790 | `/home/stew675/stock-790/build-stock/bin` | `790cf51aa` detached |
| **B** delivery | `/home/stew675/llama.cpp/build-rocm/bin` | tree `a5683e1b008e` (16 patches, `rdna-boosts`) |
| **P** armP | not built yet | upstream master + the #28867 threshold |

---

## §C — adaptive MTP at high context fails to load (F-buf) — **REPRODUCED, root-caused**

**Command (the reporter's exact config, `llama-server`):**

```
HIP_VISIBLE_DEVICES=0 llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 196608 -ngl 99 -ctk q8_0 -ctv q8_0 -cram 24576 \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 12 --no-webui
```

**Result:** fails at load, exit before health:

```
I common_speculative_init_result: creating MTP draft context against the target model '...'
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 260.02 MiB on device 0: cudaMalloc failed: out of memory
E ggml_gallocr_reserve_n_impl: failed to allocate ROCm0 buffer of size 272646272
E graph_reserve: failed to allocate compute buffers
E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
E common_speculative_init_result: failed to create MTP context
E srv    load_model: failed to create MTP context
E srv  llama_server: exiting due to model loading error
```

Full log: `/tmp/mtp-server-fail.log`.  `llama-cli` with the same flags **loads** (its context params
differ enough to leave headroom), so the failure is server-specific only in the last 260 MiB.

**The memory it is competing for (from the 32k verbose log, `/tmp/mtp-server-v.log`):**

| buffer | @ -c 32768 | scaling | @ -c 196608 (projected) |
|---|---|---|---|
| model (ROCm0) | 16053 MiB | const | 16053 MiB (+ 682 host-mapped) |
| target KV (q8_0) | 1088 MiB | × ctx | **6528 MiB** |
| **target RS (recurrent snapshots)** | **7780.5 MiB** | **× n_seq × (1 + n_max)** | **7780.5 MiB** |
| target compute | 281 MiB | ~const | 281 MiB |
| draft KV (1 MTP layer) | 128 MiB | × ctx | 768 MiB |
| draft compute | 130 MiB | ~const | ~130 MiB |

Target RS breakdown (`/tmp/mtp-server-v.log`):

```
llama_context: n_rs_seq = 12
llama_context: n_rs_batch = 13
llama_memory_recurrent: ROCm0 RS buffer size = 7780.50 MiB
llama_memory_recurrent: size = 7780.50 MiB (4 cells, 64 layers, 4 seqs 12 rs_seq),
                        R (f32): 292.50 MiB, S (f32): 7488.00 MiB, P (f32): 0.00 MiB
```

**Derived.**  The RS buffer is `mem_size * (1 + n_rs_seq)` rows, `mem_size = n_seq_max = 4` (the server
default `n_parallel = 4`); `n_rs_seq = draft.n_max` via `common_params_speculative::need_n_rs_seq()`
(true for `has_mtp()`, i.e. the built-in MTP head).  Per snapshot plane: R = 22.5 MiB, S = 576 MiB,
**598.5 MiB in total**.

| configured ceiling `n_max` | `n_rs_seq` | planes | RS @ n_seq 4 | RS @ n_seq 1 |
|---|---|---|---|---|
| 3 | 3 | 4 | 2394 MiB | 599 MiB |
| 7 | 7 | 8 | 4788 MiB | 1197 MiB |
| **12 (adaptive default)** | 12 | 13 | **7781 MiB** | **1945 MiB** |

So the adaptive ceiling of 12 costs **+2992 MiB** over ceiling 7 and **+5387 MiB** over ceiling 3, and
the server's default `n_parallel = 4` multiplies all of it by 4.  The draft context's own `n_ctx` is set
to `llama_n_ctx(ctx_tgt)` in `common_speculative_init_result` (full target context), so its KV also
scales with context (768 MiB @ 196k).

**Candidate levers (tested/derived):**

1. `--parallel 1` — **CONFIRMED LOADS** at the exact failing config (`-c 196608 -ctk/ctv q8_0
   --spec-type draft-mtp-adaptive --spec-draft-n-max 12`): server reaches health in 4 s, `n_slots = 1`.
   Drops the RS set from `4 x 13 = 52` planes (7781 MiB) to `1 x 13 = 13` planes (**1945 MiB**), a
   5.8 GiB saving.  This is the immediate user workaround and proves the `n_seq` factor is decisive.
2. A memory-aware fit that lowers the *effective* adaptive ceiling / `n_rs_seq` when the target KV plus
   the RS set does not fit, with a warning (the controller must respect the effective ceiling).
   Lowest-risk, but it caps the depth on small cards — the thing the maintainer wants to avoid.
3. Reduce the RS snapshot set structurally: shared snapshot planes across sequences that are not
   currently verifying; snapshot precision; recompute-on-rollback (space-time trade).  All purity-
   sensitive, all need the §C gate.

**RESOLVED 2026-09-14 by the Action B V4 policy (same experiment build).**  The old build allocated a
**~744 MiB/GPU F16 staging scratch** for the q8_0 cache; the 260 MiB the draft context needed was the
last straw.  With the new default (native q8_0/q4_0 staging, no scratch) the exact failing config now
loads at the **default `n_slots = 4`** and generates (server health in 4 s; adaptive ceiling 12 at
`-c 196608` q8_0 -> 34.76 t/s, acceptance 0.3404).  Proof it is V4 and not something else: forcing the
old path with `GGML_CUDA_FA_KV_NATIVE=0` reproduces the failure byte-for-byte
(`allocating 260.02 MiB ... cudaMalloc failed: out of memory`, `failed to create MTP context`).  So the
MTP-at-depth load failure and the quantized-KV decode depth regression are **one root cause**, and the
V4 activation policy fixes both.

**Status:** load failure **FIXED by Action B's V4 policy**; the underlying RS-snapshot footprint (the
reason the margin was thin) remains a structural follow-up — lever 1 (`--parallel 1`) confirmed,
levers 2-3 not yet implemented.

---

## §A — KV-type × context-depth scaling — **DONE**

Script: `tools/run_depth_audit.sh` → `results/depth-A.tsv`.  `llama-bench -p 0 -n 64 -r 2 -d
0,16384,32768,65536`, 1 GPU, `-fa auto`, arms A (stock `790cf51aa`) and B (delivery).

`tg64` (t/s), 1 GPU:

| arm | KV | d0 | d16k | d32k | d65k | d65k/d0 |
|---|---|---|---|---|---|---|
| delivery | f16 | 29.17 | 27.76 | 26.45 | **24.26** | **83.2 %** |
| delivery | bf16 | 28.99 | 27.69 | 26.41 | **24.20** | **83.5 %** |
| delivery | q8_0 | 28.61 | 25.81 | 22.78 | 18.92 | 66.1 % |
| delivery | q4_0 | 28.61 | 26.16 | 23.33 | 19.72 | 68.9 % |
| stock790 | f16 | 28.19 | 27.06 | 25.84 | 23.76 | 84.3 % |
| stock790 | bf16 | 27.99 | 22.82 | 18.98 | 14.35 | 51.3 % |
| stock790 | q8_0 | 28.00 | 26.63 | 25.16 | 22.43 | 80.1 % |
| stock790 | q4_0 | 27.71 | 25.92 | 24.12 | 21.03 | 75.9 % |

**Conclusions.**

1. **The delivery's BF16 is exactly what block 03 exists for.**  Its flat-out BF16 slope (83.5 %) matches
   stock's *f16* slope (84.3 %) and it is ahead of stock f16 in absolute terms at every depth
   (28.99 vs 28.19 at d0; 24.20 vs 23.76 at d65k).  The equitable comparison is **delivery bf16 vs stock
   f16** (bf16 is the delivery's intended cache type); by that measure there is **no BF16 depth
   fall-off**, and stock's own bf16 (51.3 %) is the pathology the delivery removes.
2. **f16 also tracks stock** (83.2 % vs 84.3 %; delivery ahead at every depth).
3. **Quantized KV is the real regression.**  Delivery q8_0 retains 66.1 % vs stock q8_0's 80.1 %, and the
   delivery is *ahead* at d0 (+2.2 %) but *behind* at d65k (−15.6 %); q4_0 is the same shape
   (68.9 % vs 75.9 %, −6.2 % at d65k).  This is the depth-resolved form of the reporter's F-q8 and it is
   delivery-specific (the two arms share the base tree, so it is the blocks' quantized-KV path).

Prime suspect, from the tree: block 08's F1 fix (`GREEDY-PURITY.md` §14) deleted the VEC fallback that
upstream uses for `n_q <= 2` with a quantized K/V, so the whole delivery band takes the **tile kernel**,
which for a quantized cache is preceded by the **whole-cache f16 staging pass** (`need_f16_K/V = 1`);
that pass is proportional to `n_kv` and runs every decode step, which is exactly a cost that grows with
depth.  Upstream's VEC kernel reads the quantized rows natively, hence its flatter slope.  Next step is
to A/B the block-15 `V4` native-q8_0 lever (`GGML_CUDA_FA_KV_NATIVE=1`, which skips the staging pass)
and the `#27796` `nthreads_KQ_q` hypothesis.

---

## §B — quantized-KV native staging (F-q8 + the q4_0 gap) — **root-caused, fix prototyped and validated**

Experiment patch: `patches/2026-09-14-v4-default-plus-q4_0-native.diff` (fork working tree, 4 files).  It
(i) refines the block-15 `GGML_CUDA_FA_KV_NATIVE` activation policy — **auto** (unset) is now native
q8_0/q4_0 **ON**, bf16 **OFF**; `=1` forces both on, `=0` forces the old F16-staging path — and
(ii) adds a **native q4_0** arm to the tile + MMA kernels beside the existing q8_0/bf16 ones
(`ggml_cuda_fattn_dequantize_q4_0_chunk`, matching convert.cu's `dequantize_block_q4_0` arithmetic).

### Mechanism

Block 08's F1 fix (`GREEDY-PURITY.md` §14) deleted the VEC fallback that upstream uses for a quantized
K/V at small `n_q`, so the delivery's whole band takes the **tile** kernel, which for a quantized cache
is preceded by a **whole-cache F16 staging pass** (`need_f16_K/V = 1`).  That pass is proportional to
`n_kv` and runs on every decode step, so its cost grows with depth.  V4's native staging dequantizes
each staged tile directly from the cache and skips the pass.  The q8_0 arm existed but was opt-in; q4_0
had no arm at all.

### q8_0 (V4 default-on)

`tg64`, 1 GPU, `-fa auto`:

| config | d0 | d32k | d65k | d65k/d0 |
|---|---|---|---|---|
| delivery, V4 off (old default) | 28.61 | 22.78 | 18.92 | 66.1 % |
| **delivery, V4 on (new default)** | **28.86** | **25.77** | **23.29** | **80.7 %** |
| stock `790cf51aa` | 28.00 | 25.16 | 22.43 | 80.1 % |

`pp` (q8_0), same run: off 1270.6 / 1164.5 / 1046.3 vs on 1255.0 / 1149.9 / 1033.4 at pp4096 / 16384 /
32768 → **−1.2 % prefill** for **+23 % decode at d65k**.  (V4 also removes ~744 MiB/GPU of F16 scratch
at a 200k context, so it helps the same memory-constrained case the q8_0 cache is chosen for.)

### q4_0 (new native arm)

| config | d0 | d32k | d65k | d65k/d0 |
|---|---|---|---|---|
| delivery, staging (old) | 28.61 | 23.33 | 19.72 | 68.9 % |
| **delivery, native (new)** | **28.83** | **25.46** | **22.82** | **79.2 %** |
| stock `790cf51aa` | 27.71 | 24.12 | 21.03 | 75.9 % |

q4_0 native prefill: pp4096 1253.7, pp32768 1023.2 (same order as q8_0 native).  The native path is now
ahead of stock at every depth for both quantized types.

### Numerics / purity (the gate)

* **Text gate** (`-n 256`, greedy, prose prompt): q8_0 native == staging == `ab94eb7db4d4`; q4_0 native
  == staging == `edafcdc7f8df`.  The dequantized tile values are bit-identical to the F16 scratch.
* **Width matrix** (`width-matrix`, P=512, W=1..8, 1 GPU): every supported type is one hash; the new
  default moves none of them.

| KV | W=1..8 unique | first hash |
|---|---|---|
| f16 | 1 | `51a5cdf1c83a43ba` |
| q8_0 | 1 | `b9bc83dae1164f67` |
| q4_0 | 1 | `5778dc558408cbd9` |
| q4_1 | 1 | `f56f6d0908289e1c` |
| q5_0 | 1 | `f8e37d642c769809` |
| q5_1 | 1 | `475b29af9c4323ec` |
| iq4_nl | 1 | `7178d805c61fbec9` |

* `test-backend-ops -o FLASH_ATTN_EXT` not re-run in this session; the text gate exercises both the
  tile (decode) and MMA (prefill) native paths and matches the staging path, which is the stronger
  cross-path check.  **TODO before promotion:** run it for the q4_0 cache type.

**MTP gate** (the reporter's workload; `draft-mtp --spec-draft-n-max 3`, q8_0, prose, `-n 300`, greedy,
32k ctx, 1 GPU):

| mode | gen t/s | acceptance | text (no `-lv 4`) |
|---|---|---|---|
| auto (native) | 53.54 | 0.75182 | `78dbabf2160e` |
| off (staging) | 53.04 | 0.75182 | `78dbabf2160e` |

Acceptance identical and text bit-identical; the native path is flat-to-slightly-ahead at the default
depth.  (A first run *with* `-lv 4` showed different text hashes — the known statistics-interleaving
artifact, `GREEDY-PURITY.md` §14/§33; re-running without `-lv 4` is the correct gate.)

### Audit of the other enabled quant types (q4_1/q5_0/q5_1/iq4_nl)

They have **no native arm** (only q8_0/bf16/q4_0 do), so they still stage through F16.  They are
nevertheless **well supported**: they track stock's retention and are ahead in absolute terms.

`tg64` d0 / d32k, 1 GPU:  

| KV | delivery | stock `790cf51aa` | delivery d32k/d0 | stock d32k/d0 |
|---|---|---|---|---|
| q4_1 | 28.67 / 23.26 | 27.78 / 22.52 | 81.1 % | 81.1 % |
| q5_0 | 28.23 / 22.25 | 27.47 / 21.40 | 78.8 % | 77.9 % |
| q5_1 | 28.29 / 22.33 | 27.50 / 21.47 | 78.9 % | 78.1 % |
| iq4_nl | 28.22 / 23.01 | **not benchable**: stock has no `iq4_nl` FA enablement, so it runs host-only (CPU) and never finishes | 81.5 % | — |

They sit ~12-16 % behind f16 at d32k (f16 26.45), so a native arm would still be worth it — but the
severe q8_0/q4_0 fall-off was specific to those two types' staging kernels, not to staging in general.  
**Note:** do **not** benchmark `iq4_nl` on a stock/un-amended build (host-only, CPU-bound); that run was
killed after >10 min at 100 % CPU.  The delivery has the block-08 FA enablement, so its `iq4_nl` is fine.

### Harness note

`~/wip-issue30/wallp.cpp`'s KV-type parser only mapped `bf16`/`q8_0` and silently fell back to **f16** for
every other type (so a first q4_1/q5_0/q5_1/iq4_nl matrix read one f16 hash for all four).  The fixed
copy is `tools/width-matrix.cpp` (a full `kv_type_from_name`), built with `~/wip-issue30/build.sh`.

---

---

## §D — deep-prefill at depth (F-pp) — **root-caused: two FA-config issues, fixed in experiment**

**The regression is real and KV-type-independent** (so it is the common prefill path, not q8_0/staging/V4):
1 GPU, 27B UD-Q4_K_XL, `pp150000`, `llama-bench -p 150000 -n 0 -r 1`:

| KV | delivery (pre-fix) | stock `790cf51aa` |
|---|---|---|
| f16 | 609.5 | 686.9 |
| bf16 | 591.5 | — |
| q8_0 | 592.6 | 669.5 |

**Curve and shape.**  delivery f16 pp4k/16k/32k/64k = 1267.2/1162.9/1049.3/876.1 vs stock
1102.9/1047.0/982.6/877.2.  Fitting `t = a + b*n` gives delivery `a` smaller (the low-depth wins) but `b`
**~51 % larger** (5.73e-9 vs 3.79e-9) — a per-(query x KV)-cell attention cost, crossing over at ~64K.

### Cause 1 — the head-256 `ncols=64` WMMA config (the big one)

With `gqa_ratio = 6`, `use_gqa_opt && gqa_ratio > 4` -> `ncols2 = 8`; for `n_q > 8` the launcher picks
`ncols1 = 64/ncols2 = 8`, so **every** WMMA call (prefill and wide verify) uses the `ncols = 64` entry in
`ggml_cuda_fattn_mma_get_config_rdna`.  The delivery's entry was a **Strix Halo (gfx1151) "halo row"**
tuning (`nthreads 256, occupancy 1, nbatch_fa 32, nbatch_V2 64, Q_in_reg=false`) -- half the KV/V tile
and Q out of registers -- where stock (#28102) uses `(256, 2, 64, 128, 128, 64, 1, true)`.  That half-tile
shape is ~1.5x per attention cell on RDNA4/RDNA3_0, which is exactly the steeper slope.

**Fix (experiment):** make the RDNA config `cc`-aware -- `is_rdna3_5` keeps the halo row, RDNA4/RDNA3_0
take the upstream config.  (`RDNA3_5`/`RDNA4` are per-gfx in `vendors/hip.h`, so the host dispatch uses
`GGML_CUDA_CC_IS_RDNA3_5(cc)` and the device constexpr uses the `RDNA3_5` macro; the two agree.)

1 GPU f16: pp65536 876.1 -> **916.6** (stock 877.2), pp150000 609.5 -> **664.8** (stock 686.9).  bf16
pp150000 591.5 -> **644.1**.  The 4B q4_0 `W = 1..8` band stays **pure** (the config fix was kept separate
from the `switch_ncols2` block that the 2026-09-13 re-base blamed).

### Cause 2 — the AMD `switch_ncols2` preference (the last per-cell overhead)

Stock #28102 also added an AMD `switch_ncols2` block ("on RDNA it is preferable to minimize wasted
compute vs. duplicate I/O for the mask"): for `gqa_ratio = 6` it picks `ncols2 = 2` (6/2 exact; the
kernel width matches the head count) where the delivery's generic `gqa_ratio > 4` rule picks
`ncols2 = 8` (2 of 8 GQA lanes wasted).  The 2026-09-13 re-base omitted that block to hold the 4B q4_0
`W=1..8` band.

Adopting it (experiment) **on top of the config fix**:

| build | 1 GPU pp150000 | 3-GPU `-sm tensor` pp150000 | 4B q4_0 W=1..8 |
|---|---|---|---|
| generic ncols2=8 (config fix only) | 664.8 | **1219.9** | pure |
| **AMD block ncols2=2** | **703.7** | 1152.3 | pure |
| stock | 686.9 | 1111.8 | — |

Single-card f16 **703.7 (+2.4 % over stock)**, bf16 **675.8 (-1.6 %)**; tensor f16 **1152.3 (+3.6 %)**.
The 4B q4_0 band is pure with the AMD block (so the re-base's impurity must have needed the config +
block combination this tree no longer has).

**The tension is compute-vs-I/O, not a bug:** single card is compute-bound (ncols2=2, no wasted lanes),
`-sm tensor` is per-GPU bandwidth-bound (ncols2=8, less K/V re-read).  Both choices beat stock in both
modes; the ideal is split-aware ncols2 (a follow-up -- the chooser would need a split signal).

### Slope-testing without 150K runs

The per-token fit `t = a + b*n` from **pp8192/16384/32768/49152** reproduces the `b` ranking (the
150K point only amplifies it), so a candidate can be screened at 32-64K.  `-sm tensor` (3 GPUs) runs
150K in ~1/2 the time (1219.9 vs 664.8 t/s) but **masks single-card regressions** -- the halo config was
faster in tensor-only testing for exactly that reason.  Rule: any prefill change must be measured on
**1 GPU as well as `-sm tensor`**.

Experiment diff: `patches/2026-09-14-prefill-rdna-config-and-ncols2.diff`.

### Split-aware `ncols2` (the ideal, pending plumbing)

The single-card and `-sm tensor` optima are **different and both real**:

| rule | 1 GPU pp32K / pp64K / pp150K | `-sm tensor` pp150K |
|---|---|---|
| generic `gqa>4 -> ncols2=8` | 1096.3 / 916.6 (pre-AMD) / 664.8 | **1219.9** |
| AMD `gqa%2 -> ncols2=2` | **~ / 947.7 / 703.7** | 1152.3 |
| stock | 982.6 / 877.2 / 686.9 | 1111.8 |

**The signal exists.**  Instrumenting `switch_ncols2` on the 27B shows the FA op is **head-split under
tensor split**:

```
1 GPU:  n_q=512  Q.ne2=24  K.ne2=4  gqa=6
 tensor: n_q=512  Q.ne2=12  K.ne2=2  gqa=6   and   Q.ne2=6  K.ne2=1  gqa=6
```

So under `-sm tensor` the KV heads are distributed (`K.ne2` 2/1 vs 4) while the *ratio* stays 6.  The
chooser runs per split sub-op and sees the reduced head count, but **cannot know the model's total**
from the op alone, so a robust rule needs the split mode:

* **option A (proper):** a small `ggml_cuda` policy set from the llama layer, which already has
  `llama_model::split_mode()` — e.g. a `ggml_cuda_set_fa_tensor_parallel(bool)` called once at model load,
  read by `switch_ncols2`; tensor-parallel -> generic 8, otherwise -> stock's AMD 2.  This is a block-04
  amendment (new API plumbing), and it must be validated in both modes + the 4B q4_0 band.
* **option B (interim, safe):** adopt stock's AMD rule globally (it beats stock in *both* modes:
  single 703.7 +2.4 %, tensor 1152.3 +3.6 %), giving up the extra ~5.5 % tensor win until A lands.
* **option C (hacky):** infer from the op (`K.ne2` shrunk vs the full head count) — rejected: fragile for
  models with few KV heads and for `-sm layer`, where the op is **not** split (Q.ne2=24) and the AMD rule
  is correct despite `device_count > 1`.

**Recommendation:** option A (it is what "tensor split is already tuned, single card was never
re-checked" actually calls for); use option B if the promotion cannot wait for the API.

---

## §E — #28867 head-256 WMMA threshold — **investigated; the delivery does not have the regression**

The delivery already carries the effect of #28867 for the purity band via the **`Q->ne[1] > 8` guard** on
the WMMA branch (`ggml/src/ggml-cuda/fattn.cu`): a W<=8 verify is TILE regardless of the gqa-eff
threshold, which is exactly the range the reporter's master build (`16378d93f`/#28102, no guard) put on
WMMA.  The only uncovered range is `n_q = 9..N` (the `n_max 8..15` verify widths and batched serving), so
that is what was measured.

**Method.**  1 GPU, f16 K/V, recall prompt, greedy, `-n 400/500`, `draft-mtp`; the control forces the
whole head-256 range off WMMA with `GGML_CUDA_FA_WMMA_MAX_HEAD=128` (block-04 env).  Hashes confirm the
control is real: at `W=9` the default is WMMA and the control is TILE (f16 `609bc99910f8e616` vs
`8d7bb8d845f1154d`; q8_0 `925594dd6ecd7fd1` vs `ed19bc62010c2df8`).

| workload | WMMA (default) | TILE (forced) | acc / mean len |
|---|---|---|---|
| `n_max 8` (W=9), recall | 115.10 t/s | 115.72 t/s (+0.5 %) | 0.96712 / 8.67 — identical |
| `n_max 15` (W=16), recall | 147.19 t/s | 147.80 t/s (+0.4 %) | 0.93186 / 14.68 — identical |
| `llama-batched-bench` npl 1/8/9/16/32, f16 | — | — | neutral (<=0.6 %, either direction) |

The acceptance is bit-identical between the arms at both depths, so this is not a draft-quality
artifact: the two kernels are simply at parity for the head-256 verify on the delivery's tuned block-04
configs.  The reporter's ~20 % TILE advantage is a property of upstream master's head-256 WMMA configs
(the delivery deliberately kept its own in the 2026-09-13 re-base), not of the delivery.  Plain batched
decode at the 8->9 boundary is likewise neutral.

**Conclusion.**  No delivery change is needed to fix a regression — there is none.  Adopting the MFMA
style threshold (`n_q*gqa_eff > 64` for head>128) would be a **~0.4 % neutral** selection change (it
keeps `n_q=9..32` on TILE) and is *not* a purity change (the band is already bounded at `W=8` by the
matmul family switch at `MMQC/MMVF_MAX_BATCH_SIZE`, independent of FA).  Recommendation: leave the
delivery as-is; optionally match the MFMA threshold only if we want upstream alignment.

