# mixed-kv — the submission record (promoted as block 17 in r287, moved out as optional `0101` in r288)

> **Status: archived record; the live artifact is `optional/mixed-kv/`.**  PR #312 (@mjohanning99) was
> accepted and folded into the delivery as **block 17** in **r287**.  The **r288 block-number re-org** then
> moved it OUT of the default set, where it is **optional block `0101`**
> (`optional/mixed-kv/0101-mixed-kv.patch`; applied only with `scripts/apply-all.sh --include-optional`); its
> bytes are unchanged, so applied on the r288 default tree it reproduces the r287 release tree exactly
> (`89b25552…`).  This directory keeps the **original submission** as the record of the review (the QSA
> refusal and the one-V-type-function change are revision 2) and the measurements that led to the promotion;
> nothing here is part of the delivery, applied by `scripts/apply-all.sh`, or covered by the release gates.
> `mixed-kv.patch` is the r281 cut, sha256
> `dca100d4096699b8a27bdd2255ce502487e1f13345c3b65122a357bfe73115c0`; it applies at r281, **not** at the
> current tree.  The live artifact is verified by `scripts/gate-mixed-kv.sh` (run on the optional-applied
> build); the promotions and the move are recorded in `WORKLOG.md` r287 and r288.

# `mixed-kv` — a q8_0 K cache with a smaller V cache, read natively

**Optional artifact, not part of the delivery.  Cut for `r281`** (`release.json` `.tip` `69296dde5`, tree
`549f379e615684f900ca8059f15af69b64c5372b`).  Revision 2 (14 Nov 2026): the first cut was for `r280`; this one adds
the two changes the review asked for (see "Revision 2" below) and the correctness gates.

`--cache-type-k q8_0` with `--cache-type-v q5_1`, `q5_0` or `q4_0`, on models whose attention heads are all
256 wide (the Qwen3.5 / 3.6 / 3.8 hybrids, dense and MoE) and that do not use the sparse (QSA) attention.  The
delivery refuses every mixed pair today.

* `mixed-kv.patch` — the artifact.  sha256
  `dca100d4096699b8a27bdd2255ce502487e1f13345c3b65122a357bfe73115c0`
* applies at `r281` (`69296dde5`, tree `549f379e61…`) and produces tree `8a9f503ab21b557b43f8aad56320c866cddfceaf`
  (verified with `git apply --cached` into a fresh index of `69296dde5`)
* 11 files, 171 insertions / 42 deletions: `ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn-rdna.cuh`,
  `fattn-tile.cu`, `fattn.cu`, `template-instances/generate_cu_files.py` and three generated tile instances,
  `src/llama-context.cpp`, `tests/test-backend-ops.cpp`, `tests/test-logits-width-probe.cpp`
* the measurements further down were taken with revision 1 on `r279` (`f61ac4bd8`); `r279` -> `r281` changes
  `fattn-mma-f16.cuh` (#311, a constant-folded load bound, same results) and `tests/test-llama-archs.cpp` only

## Why the delivery refuses mixed pairs, and what this changes

`llama_init_from_model` rejects `type_k != type_v` with the note that every mixed pair measured 1.7-3.6x slower
than its same-type equivalent and never smaller.  The cause is in the kernel selection, not in the pair:

* the tile kernel has ONE K/V type parameter, so `ggml_cuda_fattn_tile_kv_native_type()` returns NONE for a
  mixed pair and the launcher stages BOTH operands through the F16 scratch, on every decode step;
* the RDNA4 decode/verify band asks for `kv_native(K) == kv_native(V)`, so a mixed pair leaves the band and
  takes that staged tile kernel too;
* the MMA kernel already takes a native type per operand (`kv_native.type_K` / `.type_V`), so prefill was
  never affected.

The patch gives the tile kernel a second type parameter for V (`type_V`, defaulting to `type_KV`).  K is loaded
with K's dequant arm and V with V's; the arms are the existing ones.  Three instances are generated, for head
256 only: K `q8_0` with V `q4_0`, `q5_0`, `q5_1`.  `ggml_cuda_fattn_kv_mixed_native()` names those pairs, the
band accepts them, and `ggml_cuda_fattn_tile_type()` / `_type_V()` keep the allocator and the dispatch on one
predicate.  The context gate accepts the three pairs when every attention layer has `n_embd_head_k == 256` and
`n_embd_head_v == 256`, and keeps refusing everything else, with the pair named in the error text.

`GGML_CUDA_FA_KV_MIXED=0` switches the native read of a mixed pair off (the staged path, as before).  It is a
kill switch: unset means on.  A same-type cache does not reach any of the new code.

## Revision 2: the QSA refusal, one V-type function, and the correctness gates (14 Nov 2026)

**Two changes**, both from the review on PR #312:

* `llama_init_from_model` keeps the same-type rule for `LLM_ARCH_QWEN4EXP`.  Its sparse attention op
  (`GGML_OP_FLASH_ATTN_QSA`) asserts `K->type == V->type` (`fattn-qsa.cu`), and its heads are 256 wide, so the
  head-256 gate alone would have admitted it.  Not demonstrated on a model: there is no Flash-Next file here.
* The tile dispatch (`ggml_cuda_flash_attn_ext_tile_case_type`) now takes the V type from
  `ggml_cuda_fattn_tile_type_V`, the function the allocator sizes the F16 scratch with, instead of switching on
  `V->type` behind its own predicate.  The kernel chosen is the same; the instantiation and the allocator now
  read one function, which is the contract the earlier drifts broke.

**Gates.** RX 9070 (gfx1201) + RX 7900 GRE (gfx1100), the delivery's default settings (no tuning variables).
"r281" is this patch on `69296dde5`; "r279" is the revision-1 engine (`f61ac4bd8` + revision 1), which runs the
same kernel code for these models (the QSA refusal does not apply to them, the dispatch change picks the same
instance).  Logs and scripts: `bench/mixedkv-gates-r281/` on the test box, available on request.

| gate | tree | result |
|---|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT`, default / `GGML_CUDA_FA_KV_NATIVE=0` / `GGML_CUDA_FA_KV_MIXED=0` | r279 | 7021/7021 in all six runs (two devices x three settings) |
| `test-backend-ops -o FLASH_ATTN_EXT`, default | r281 | 7021/7021 on gfx1100 and on gfx1201 |
| reserve, tensor split, `-c 32768` and `-c 131072` (KV, compute buffers, graph splits, the "converting K and V to f16" warning) | r279 | graph splits 2 / 2 and no warning for every pair; the same compute buffers for the mixed and the same-type pairs (`Meta()` 312.6 / 313.0 MiB, `CPU` 80.2 / 80.6 MiB); KV in the table below |
| allocator and kernel per width (`[FATPATH]` from `fa-kernel-chooser-trace.patch`, plus the same print of `need_f16_K/V` in `ggml_cuda_flash_attn_ext_get_alloc_size`), W = 1..8 | r281 | gfx1201: best = 400 at every width, gfx1100: best = 200 at every width; `need_f16_K = need_f16_V = 0` for every pair and width, as for q8_0/q8_0 |
| width purity, `logits-dump-kv.cpp` (looped over P and W in one process, one model load per cell; otherwise the tool's own steps), token-0 logits hash identical for W = 1..8 | r281 | PASS for q8_0/q8_0 and the three pairs on gfx1201 alone, `-sm layer` and `-sm tensor` at P = 256 / 700 / 1500; on gfx1100 alone at P = 256 / 700 / 1000 (that card cannot hold a 2048-token batch next to the 27B IQ4_XS, so its third length uses a 1024-token batch) |
| `plain == draft-mtp` (greedy, 128 tokens, `--spec-draft-n-max` 1 / 3 / 7), two prompt lengths (~370 and ~2100 tokens) | r279 | identical text in all 18 cells (three pairs x tensor / layer / gfx1201 alone x two prompts) |
| KLD against q8_0/q8_0 (WikiText-2 test, 8 x 4096, tensor split) | r279 | q5_1 0.0033 (same top p 98.30 %), q5_0 0.0039 (98.27 %), q4_0 0.0049 (97.83 %); q8_0/q8_0 against f16 is 0.0030 (table further down) |
| `scripts/check-gpu-faults.sh` over every log and the kernel journal | both | clean |

The q5_0 / q5_1 same-type pairs loaded and ran on the two-GPU `-sm tensor` split at both context sizes
(reserve rows below), so the q4_1-family abort the review mentions did not occur here.

KV at load, tensor split (`llama_kv_cache: size`):

| `-c` | q8_0/q8_0 | q8_0/q5_1 | q5_1/q5_1 | q8_0/q5_0 | q5_0/q5_0 | q8_0/q4_0 | q4_0/q4_0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 1088 MiB | 928 | 768 | 896 | 704 | 832 | 576 |
| 131072 | 4352 MiB | 3712 | 3072 | 3584 | 2816 | 3328 | 2304 |

Not covered by these gates: three GPUs; a qwen4exp run; the adaptive-MTP acceptance gate; the speed matrix
against both neighbouring same-type pairs with interleaved arms.

## Verification (r279, the delivery's default settings)

RX 9070 (gfx1201) + RX 7900 GRE (gfx1100).  "Stock" is `f61ac4bd8` built from the same tree with the same
options; "patched" is stock plus this patch.

| check | result |
|---|---|
| `test-backend-ops test -o FLASH_ATTN_EXT` | 7021/7021 on gfx1100 and on gfx1201, 3/3 backends.  The patch adds 24 cases: K `q8_0` with V `q4_0` / `q5_0` / `q5_1`, head 256, GQA 4 and 6, `nb` 1 / 3 / 8 / 32 |
| default path, patched against stock, K = V = `q8_0` (`llama-perplexity --kl-divergence`, ctx 512, 40 chunks) | mean KLD 0.000000, same top p 100.000 % on the tensor split, on gfx1201 alone and on gfx1100 alone; maximum KLD 0.000066 / 0.000059 / 0.000057 |
| the same tool, stock against stock, tensor split | mean KLD 0.000000, same top p 100.000 %, maximum KLD 0.000066: the maximum above is the tool's floor |
| stock with K `q8_0`, V `q5_1` | refused at load, as before |
| `test-logits-width-probe … 4096 1024`, `W = 1..8` | PASS, worst maxdiff 0, in all 16 cells below |
| kernel journal, GPU faults | none over any run |

Width probe (the patch adds `KVV=<type>` to the tool for the V cache; `KV` stays the K cache):

| K / V | tensor split, `RS=0` | tensor split, `RS=7` | gfx1201 alone | gfx1100 alone |
|---|---|---|---|---|
| `q8_0` / `q8_0` | `3a41a784e5b4ddfe` | `3a41a784e5b4ddfe` | `09a1bc34606b5a18` | `b253559ae2a370e6` |
| `q8_0` / `q5_1` | `a4378447daad2245` | `a4378447daad2245` | `af7adb1c0d46acba` | `d55f8656cf3f08c7` |
| `q8_0` / `q5_0` | `221e36202da68a4e` | `221e36202da68a4e` | `944372ffc6f1c15b` | `ae262c5d90e9e17b` |
| `q8_0` / `q4_0` | `49a85df9f1db0946` | `49a85df9f1db0946` | `9305f211d2100d44` | `8915b234a3eb92a6` |

Tensor split: Qwen3.8-27B Q4_K_M.  One card: Qwen3.8-27B UD-IQ4_XS.  Text: WikiText-2 test.

## What it measured

### Setup

* RX 9070 16 GB (gfx1201) in the CPU slot, PCIe 4.0 x16; RX 7900 GRE 16 GB (gfx1100) in the **chipset** slot of
  a B550 board, PCIe 3.0 x4.  Ryzen 7 5700X3D, 64 GB.  Linux 7.3.3 (CachyOS), `linux-firmware-amdgpu` 20261019.
* Built with ROCm 7.2.4 (AMD clang 22), `-DGGML_HIP=ON -DGPU_TARGETS="gfx1100;gfx1201"`, Release; run against
  the ROCm 10.1 runtime libraries (HIP 7.16).
* No `GGML_CUDA_*` / `LLAMA_*` variables set.  `-fa on -ngl 99 -ub 1024`.  Tensor split: `-sm tensor -dev
  ROCm1/ROCm0` (the 9070 first), even split.
* `llama-bench -p 2048 -n 64 -d 0,32768 -r 3` unless a table says otherwise; tok/s, the standard deviation of
  the three repeats in brackets where it matters.  Arms are compared inside one session only: the same arm
  moves by up to 0.7 % between sessions.

### VRAM at load, tensor split, `-c 204800`, Qwen3.8-27B Q4_K_M (plain `llama-server`, no draft, no mmproj)

| build, K / V | GRE + 9070, MiB | against `q8_0` / `q8_0` |
|---|---:|---:|
| stock, `q8_0` / `q8_0` | 12161 + 12168 | |
| patched, `q8_0` / `q8_0` | 12161 + 12168 | 0 |
| patched, `q8_0` / `q5_1` | 11661 + 11665 | −1003 |
| patched, `q8_0` / `q5_0` | 11561 + 11565 | −1203 |
| patched, `q8_0` / `q4_0` | 11361 + 11365 | −1603 |
| patched, `q8_0` / `q4_0`, `GGML_CUDA_FA_KV_MIXED=0` | 11929 + 11897 | −503 (at load; the staging arena is taken in use) |

### Quality against an f16 KV cache (tensor split, WikiText-2 test, 8 chunks of 4096)

| K / V | mean KLD | median | 99 % | same top p | PPL / PPL(f16) |
|---|---:|---:|---:|---:|---:|
| `q8_0` / `q8_0` | 0.0030 ± 0.0008 | 0.00041 | 0.015 | 98.44 ± 0.10 % | 1.0045 |
| `q8_0` / `q5_1` | 0.0036 ± 0.0008 | 0.00053 | 0.018 | 98.24 ± 0.10 % | 1.0061 |
| `q8_0` / `q5_0` | 0.0045 ± 0.0012 | 0.00057 | 0.018 | 98.21 ± 0.10 % | 1.0031 |
| `q8_0` / `q4_0` | 0.0057 ± 0.0013 | 0.00100 | 0.029 | 97.78 ± 0.12 % | 1.0087 |

### Speed, tensor split, Qwen3.8-27B Q4_K_M

| build, K / V | prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|
| stock, `q8_0` / `q8_0` | 470.4 | 38.19 (0.38) | 413.4 | 36.44 (0.59) |
| patched, `q8_0` / `q8_0` | 470.3 | 38.08 (0.35) | 413.3 | 36.35 (0.56) |
| patched, `q8_0` / `q5_1` | 470.3 | 37.90 (0.34) | 413.3 | 35.91 (0.56) |
| patched, `q8_0` / `q5_0` | 470.3 | 37.81 (0.33) | 413.1 | 35.62 (0.59) |
| patched, `q8_0` / `q4_0` | 470.4 | 37.99 (0.32) | 413.2 | 36.10 (0.58) |
| stock, `q8_0` / `q8_0`, repeated last | 470.2 | 38.19 (0.39) | 413.1 | 36.44 (0.57) |

Prefill does not move.  Decode on an empty cache does not move.  At 32768 cells the mixed pairs read 0.9-2.2 %
under `q8_0` / `q8_0`, inside the spread of three repeats.

### The staged path against the native one (tensor split, `GGML_CUDA_FA_KV_MIXED=0`)

| K / V, path | decode d0 | decode d32768 | decode d100000 |
|---|---:|---:|---:|
| `q8_0` / `q8_0`, stock | 38.19 | 36.44 | 32.12 (0.49) |
| `q8_0` / `q5_1`, native | 37.90 | 35.91 | 31.26 (0.43) |
| `q8_0` / `q5_1`, staged | 37.84 | 31.74 | **23.29** (0.28) |
| `q8_0` / `q4_0`, native | 37.99 | 36.10 | |
| `q8_0` / `q4_0`, staged | 37.95 | 32.05 | |

(`-d 100000` is `llama-bench -p 0 -n 64 -d 100000 -r 3`.)  The staged path is level on an empty cache, 11-12 %
behind at 32768 cells and 25 % behind at 100000: its cost is the whole-cache conversion, so it grows with
depth, which is the direction of the 1.7-3.6x the gate's comment records.  The native mixed pair is 2.7 %
under `q8_0` / `q8_0` at 100000 cells.

### Speed, one card, Qwen3.8-27B GSQ-RCO IQ3_S

All arms of one card in one session, stock repeated after the patched build:

| build, K / V | gfx1201: prefill d0 | decode d0 | prefill d32768 | decode d32768 | gfx1100: prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| patched, `q8_0` / `q8_0` | 857.7 | 35.59 | 649.1 | 32.19 | 387.1 | 33.24 | 315.3 | 30.12 |
| stock, `q8_0` / `q8_0` | 854.0 | 35.49 | 649.3 | 32.16 | 384.3 | 33.16 | 314.4 | 30.03 |
| stock, `q5_1` / `q5_1` | 854.0 | 35.33 | 648.1 | 30.69 | 384.0 | 32.96 | 314.6 | 28.66 |
| stock, `q4_0` / `q4_0` | 854.6 | 35.41 | 651.6 | 31.34 | 384.1 | 33.01 | 314.8 | 29.03 |
| patched, `q8_0` / `q5_1` | 855.2 | 35.38 | 649.5 | 31.02 | 383.8 | 32.93 | 314.3 | 29.51 |

Decode repeats are within 0.03-0.11 tok/s here.  The first arm of a session reads up to 0.8 % higher in
prefill whichever build it is (gfx1100: 387.2 for stock when stock ran first, 387.1 for the patched build
when it ran first, 384.3 for the other one both times), so the builds do not differ.

An earlier session with all three mixed pairs, against the patched `q8_0` / `q8_0` of that session
(31.95 on gfx1201, 30.01 on gfx1100) at 32768 cells: V `q4_0` 31.29 / 29.77 (−2.1 % / −0.8 %), V `q5_1`
30.96 / 29.55 (−3.1 % / −1.5 %), V `q5_0` 30.70 / 29.11 (−3.9 % / −3.0 %).

### Decode at depth is slower with the smaller V types — and that is not the mixing

At 32768 cells on one card a `q5_1` V decodes 1.7-3.6 % under `q8_0` / `q8_0`.  The delivery's own same-type
caches are slower still: stock `q5_1` / `q5_1` is 4.6 % under on both cards and stock `q4_0` / `q4_0` 2.6-3.3 %.
So the cost comes with the cache type, a mixed pair pays less of it than the matching small pair, and it keeps
the keys at 8 bits.  Nothing is lost on an empty cache or in prefill, and on the tensor split, where each card
reads two of the four KV heads, it sits inside the noise.

Part of it on gfx1201 is the fetch-ahead band, which only `q8_0` / `q8_0` takes.  One session, gfx1201 alone,
`llama-bench -p 0 -n 64 -d 0,32768 -r 3`:

| arm | decode d0 | decode d32768 |
|---|---:|---:|
| `q8_0` / `q8_0` | 35.32 | 31.90 |
| `q8_0` / `q8_0`, `GGML_HIP_FA_BAND_FETCH_AHEAD=0` | 35.34 | 31.42 (−1.5 %) |
| `q8_0` / `q4_0` | 35.35 | 31.27 (−2.0 %) |
| `q8_0` / `q5_1` | 35.34 | 31.00 (−2.8 %) |

So about 1.5 points of the gap are the missing fetch-ahead and the rest is the dequant arm (the nibble types
go byte by byte through FP32 where `q8_0` multiplies packed halves).  That reading is ours and is not tested
further.  Letting a mixed pair take the fetch-ahead variant for its `q8_0` K is an obvious follow-up; it is not
in this patch.

### A second model: Index-Translate 35B-A3B (qwen35moe) Q5_K_M, tensor split

| build, K / V | prefill d0 | decode d0 | prefill d32768 | decode d32768 |
|---|---:|---:|---:|---:|
| patched, `q8_0` / `q8_0` | 3319 | 112.9 (3.0) | 2406 | 110.9 (4.5) |
| stock, `q8_0` / `q8_0` | 3326 | 113.2 (2.4) | 2407 | 110.7 (4.4) |
| stock, `q8_0` / `q8_0`, again | 3323 | 113.3 (2.5) | 2405 | 110.9 (4.5) |
| patched, `q8_0` / `q5_1` (earlier session) | 3298 | 111.6 (2.5) | 2397 | 106.2 (4.4) |
| patched, `q8_0` / `q4_0` (earlier session) | 3298 | 111.8 (2.5) | 2403 | 106.8 (4.3) |

It loads and runs with a mixed pair; stock and patched agree; the decode spread is too wide to rank the pairs.
(The very first arm on this file, before it was in the page cache, read 2374 / 2094 in prefill and is left out.)

## What it does not cover

* Head sizes other than 256.  The mixed instances exist for `(256, 256)` only; the gate refuses a mixed pair on
  any other model, so nothing takes the staged path by accident.  Adding a size is the generator list, the
  extern declarations and the three `256` checks; the kernel body does not change.
* K types other than `q8_0`, and V types other than `q4_0` / `q5_0` / `q5_1`.
* gfx1151 and CDNA.  Three GPUs.  Speed on `-sm layer` (its correctness is in the revision-2 gates).
* A K-quant model on one card (the two 27B files that fit one 16 GB card here are IQ quants).
* Quality at depth (the KLD run is 4096-token chunks) and MTP acceptance per pair (the text equality with
  draft-mtp is in the revision-2 gates).
* The vec kernel.  `GGML_CUDA_FA_QUANTS` lists same-type pairs only, and the HIP chooser does not reach it for
  a quantized cache, so nothing here depends on it.

## Questions

* **Always compiled.**  The three tile instances are globbed like every other `fattn-tile*.cu`, whatever
  `GGML_CUDA_FA_QUANTS` says.  Compiling them on demand would need more than the CMake list: the gate lives in
  `llama_init_from_model`, which cannot see how the backend was built, so a default build would accept the
  pair and stage it.  It needs a backend query for "this pair is native".  I left it always-on; it is three
  translation units.
* **On by default.**  A mixed pair is only reached with two different `--cache-type-*` values, so no extra
  opt-in variable is asked for; `GGML_CUDA_FA_KV_MIXED=0` is the way out.
* **A block, not an artifact?**  Unlike `ce-q8-wire` this needs no special hardware and the default path is
  bit-identical, so it may belong in block 05.  It is cut as an artifact because the gates above are mine.
