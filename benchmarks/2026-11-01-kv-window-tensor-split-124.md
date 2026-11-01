# The windowed KV across a `-sm tensor` split (r204, `TODO` #124) — gate record

Dated record for the r204 release gate (`WORKLOG.md` 261101.016).  What shipped, what it measured, and the
four defects the arms found on the way.  Design note (pre-implementation, single-GPU reasoning):
[`../../archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`](../../archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md) §1-§8;
what block 15 now says: [`../../patches/README.md`](../../patches/README.md) → Block 15.

## Environment

* 3x R9700 (gfx1201), 32 GiB each, ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), `~/bin/build-llama-rocm`
  (`BUILD_DIR=build-rocm`), r204 tip **`aaf30b6b4`**, tree **`6260370…`**.
* Models: `Qwen3.5-4B-Q8_0` (small), `Qwen3.8-27B-Q8_0` (big), `Qwen3.8-27B-UD-Q4_K_XL` (four-axis),
  `Qwen3.6-35B-A3B-UD-Q4_K_M` (`-ncmoe 10`, host experts).
* Prompt: `prompts/prose-rdna-boosts.txt` (16074 B, 5246 tokens on the 27B tokenizer, `sha256 fabdec65…`).
* Every comparison ran with `ROCBLAS_USE_HIPBLASLT=0` and `-fit off` with `-c` pinned (a fresh-process
  comparison of two arms; `INVARIANTS.md` §44), generated text hashed by `scripts/extract-generated.py`.

## What the gate does

`scripts/gate-kv-window-split.sh` (new; release gate 16 in `MANIFESTS.md`), plus
`scripts/gate-state-pipe.sh` with `LLAMA_KV_RESIDENT_CELLS` set as its companion arm.

## Results

### Text purity (the arm that matters)

| arm | window | text |
|---|---|---|
| 3 devices, 4B, `-c 8192` f16 KV | off | 254 chars `sha=057ea27b1009` |
| " | `W=1024` | 254 chars `sha=057ea27b1009` |
| 2 devices, 27B Q8_0, `-c 524288` f16 KV | off | 309 chars `sha=7a7430617465` |
| " | `W=1024` | 309 chars `sha=7a7430617465` |
| " | `W=1024`, `LLAMA_KV_REGION_UNIT_MIB=1` | 309 chars `sha=7a7430617465` |
| " + `--spec-type draft-mtp --spec-draft-n-max 3` | off | 309 chars `sha=7a7430617465`, acceptance 0.46154 (36/78) |
| " | `W=1024` | 309 chars `sha=7a7430617465`, acceptance 0.46154 (36/78) |
| 1 device, 4B | off / `W=1024` | identical; the pre-#124 shape is untouched |
| 1 device, 4B, `-nkvo` | `W=1024` / off | identical, and the window is still declined (r197) |

The window is **real** in these arms, not a declined knob: `FLIP device <i>` appears 32 times per device at
the 4 MiB map unit and 128 times at a 1 MiB unit (the unit is floored to the driver's 2 MiB granularity, so
the finer arm is 2 MiB), and `grow_cells` reports `of which host 128.00 MiB` per device on the 2-device arm.

### The two halves of the prize

* **A context that only the window can serve:** 2 devices, 27B Q8_0, `-c 1048576` with an f16 KV (64 GiB of
  KV) loads and decodes with `W=131072`; the fully resident arm does not fit this box.
* **`--fit` on the split (27B Q8_0, 2 devices, `-c 524288`, f16 KV = 32 GiB):**

| arm | fit output |
|---|---|
| no window (r203 behaviour) | `n_gpu_layers` reduced to 42 (the estimate charges the whole KV once per device) |
| `LLAMA_KV_RESIDENT_CELLS=auto` | "no window is needed — the 524288-cell KV (32768.0 MiB, 16384.0 MiB of it on each of 2 devices) fits fully resident (35218.5 MiB of VRAM left over the 62520 MiB target)", `-c 524288 -ngl -1 -ts 31260,31260` |
| `LLAMA_KV_RESIDENT_CELLS=65536` | "the KV cache is WINDOWED on this 2-device tensor split: 65536 of 524288 cells resident (4096.0 MiB of 32768.0 MiB, one VMM region per device holding its slice = 2048.0 MiB each at an even split), 28672.0 MiB spilled to pinned system RAM", `-c 524288 -ngl -1` |
| `LLAMA_KV_SPILL_RAM_PCT=1`, `W=1024` | widened to 511289 cells (spill 812.4 MiB) — the ceiling working |
| `LLAMA_KV_SPILL_RAM_PCT=1`, `W=1024`, `-c 2097152` | "does not make this fit either" — window dropped, fitted as before (`-ngl 14`) |
| `LLAMA_KV_REGION=0`, `W=65536` | no window message at all |

The `auto` row is the biggest behaviour change in the release: with a window requested, the split's `--fit`
uses the **physical** KV term, so it stops shrinking a config the KV physically fits (the pre-existing
`n_devices x` over-count is untouched outside the window arm — its own item, `TODO` #122's neighbourhood).

### Throughput (not a correctness signal; recorded for the release)

`llama-batched-bench -npl 1,4,8 -npp 512 -ntg 32`, 2 devices, 27B Q8_0, `-c 32768`:

| B | plain pp t/s | plain tg t/s | `W=1024` pp t/s | `W=1024` tg t/s |
|---|---|---|---|---|
| 1 | 1316.20 | 31.86 | 1788.39 | 31.53 |
| 4 | 1927.77 | 109.69 | 1991.09 | 109.79 |
| 8 | 2040.22 | 191.69 | 2033.90 | 192.17 |

At 4352 cells the spill is ~3 MiB of a 32 GiB cache, so this is "the window costs nothing when it is not
needed", not a spill measurement.  Protocol A (adaptive-MTP methodology, 1 GPU, 27B UD-Q4_K_XL, `-n 3000`,
reasoning pinned per axis): plain -> `draft-mtp n3` **R 31.1 -> 59.3 (1.91x, acceptance 0.58686) / P 30.7 ->
74.5 (2.43x, 0.83908) / C 31.1 -> 80.2 (2.58x, 0.91117) / K 31.3 -> 85.9 (2.74x, 0.99200)** — the r90 record's
shape (1.96x/2.47x/2.67x/2.85x), so decode is untouched.

### The rest of the release gates

* **Coherence vs the r203 build** (`git stash` -> build -> hash -> unstash -> build), no window in any arm:
  3-GPU 4B `057ea27b1009`, 2-GPU 27B `7a7430617465`, 1-GPU 27B `7a7430617465`, 2-GPU host-expert 35B-A3B
  `-ncmoe 10` `e9db5cb8db0b` — **identical on both builds** in all four.
* **Prefill logits**: mean KLD **0.000000**, same-top-p **100.000 %** against the recorded gfx1201 base.
* **MTP draft window** (`gate-spec-draft-window.sh`): 6/6 requests `win == win0`, draft cache
  `(1024 + 512) x 2` cells, acceptance identical — PASS.
* **Per-block build**: 1/1 (`--changed-from baa382608`; block 15 only), 26 s.  `validate-set.sh` strict
  16/16, tree `6260370949c1975c1ec5cb085bf5e937a2677cf1`.
* **VMM selftest** (`GGML_CUDA_VMM_REGION_SELFTEST=1`): PASS, including a new arm for the multi-mapping flip
  and the unchanged "a straddling request is refused" arm.
* **`check-gpu-faults.sh`**: kernel journal + the windowed/split/MTP/state-pipe logs clean.

## The four defects the arms found

1. **A child region's init-time window commit ran past its own VA reservation.**  The commit rounds each
   tensor's window up to a whole map unit; on a split each child is its own reservation, so the last tensor's
   inflated window was refused (`the KV window (4.00 MiB) could not be committed in VRAM on device 1/2`, two
   per run on the 4B).  Fixed by clamping the commit to the region end **in whole units** — clamping to the
   tensor's own extent was tried first and is wrong: it leaves a short mapping off the unit grid, and the
   next slide then refuses to flip (`refusing to flip [2.00, 6.00) MiB: the mapping [2.00, 4.00) covers it
   only in part`).
2. **`ensure_span` refused a request that covered several whole mappings.**  One batch can advance the spill
   boundary by more than one unit, and at `LLAMA_KV_REGION_UNIT_MIB=1` (floored to the driver's 2 MiB
   granularity) the request covered two 2 MiB mappings and the exact-match check aborted the batch.  Whole
   mappings are now flipped one at a time; a *partial* coverage is still refused (the driver loses bytes).
3. **The batched state save issued a 2-D D2H from mixed backing.**  `hipMemcpy2DAsync` returns
   `invalid argument` when the source range begins in the pinned host mapping of a spilled region — the
   tensor-split checkpoint save aborted (`cache_k_l64 (view)`, 1024 B x 4782 rows).  The region-backed
   directions of `get/set_tensor_2d_async` now fall back to the region buffer's own copy, ordered behind the
   compute stream (a blocking default-stream copy does not wait for it — that alone produced a torn state).
4. **The fallback's "contiguous rows = one copy" shortcut ignored the caller's destination pitch.**  The
   split's splice reads a device's slice out of a *wider* row, so a tight copy misplaced every row after the
   first: the state-pipe gate failed on **all five cycles in all four async arms** ("restored text differs
   from the synchronous arm").  The shortcut now requires both pitches to equal the width.
   **`scripts/gate-state-pipe.sh` with `LLAMA_KV_RESIDENT_CELLS=1024` (2-GPU `-sm tensor`) is the arm that
   catches 3 and 4** — it went FAIL -> PASS 5/5 byte-identical to `LLAMA_STATE_ASYNC=0`, and its single-device
   equivalent was already covered.
