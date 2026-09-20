# Brief — bf16 producer port (HC intermediates + producer marking)

**Status:** READY-TO-START, not begun.  **Read `HANDOVER.md` first** (the WIP entry point), then this.
This is the turnkey brief for the next session: it is self-contained (mandate, environment, the
established facts, the reference map, the step plan, the gates, and a copy-paste prompt).

---

## 1. Mandate

Make the **producers write bf16 natively** so the HC path (and MMB→MMB chains) stop paying
`mmb_cvt` (f32→bf16 conversions) and read/generate bf16 directly.

- **Why it must be a producer change.** A plain `ggml_cast` is a **net loss** — the cast *is* the
  existing `mmb_cvt`.  Per element:

  | path | traffic |
  |---|---|
  | today: producer writes f32 → `mmb_cvt` (read 4, write 2) → consumer reads bf16 | 4+4+2+2 = **12 B** |
  | producer writes bf16 natively → consumer reads bf16 | 2+2 = **4 B** |
  | inserted `ggml_cast` → consumer reads bf16 | 4+4+2+2 = **12 B** (no win) |

- **Ceiling:** `mmb_cvt` is **642.5 ms = 3.8 %** of pp8192 on Flash-Next; `dsv4_hc_pre`+`_post` are
  1429 ms = **8.5 %**.  Realistic win ~2-3 %.
- **Numerics:** it is a **change on the HC path** → the PPL gate (record the new baseline; do not
  expect equality).

## 2. Environment

| | |
|---|---|
| worktree | `~/llama-wip-mmb`, branch `wip-mmb-general`, tip **`2a73b02e4`** |
| base | `8a2567e1e` |
| build | `cd ~/llama-wip-mmb && cmake --build build-rocm -j16 --target llama-bench llama-cli` (configured, ccache on) |
| runtime | `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH` |
| target model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (94 GiB; the only qwen4exp with HC + QSA -- do **not** use the `IQ4_NL/` dir) |
| reference | `~/pwilkin-llama-cpp`, branch `strix-halo`, commit **`f5daaa3cf`** |
| build target arch | gfx1151 (`halo`).  The mark pass is RDNA3_5-only in the reference. |

Reproduce the current baseline (warm the page cache first):

```sh
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
dd if=$M of=/dev/null bs=4M 2>/dev/null
GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1 ./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 \
  -ctk bf16 -ctv bf16 -b 2048 -ub 2048 -p 8192 -n 0 -r 2     # ~968.75 t/s
```
`LLAMA_MMB_CVT_LOG=1` prints (up to 200) one-shot `MMB_CVT <name> op=<op> ne=[...] view_src=<...> n=...`
lines — use it to see which producers are converting.

## 3. Established facts (measured in session 7 — do not re-derive)

- `LLAMA_MMB_CVT_LOG=1` at pp512, per layer: `hc_norm` (5.24 M, ×2, a `RESHAPE` view of the
  `rms_norm*gamma` `MUL`), `final_output` (3.15 M, `RESHAPE`), `hc_mixed` (1.31 M, ×2,
  `DSV4_HC_PRE` + its `RESHAPE`), `node_*` (0.16 M, `MAP_CUSTOM1`), `ROCm0#ple_embd#0` (1.31 M, `NONE`).
- The live prefill HC kernels are **`dsv4_hc_pre_f32` (743.9 ms) / `dsv4_hc_post_f32` (685.3 ms)** in
  `ggml/src/ggml-cuda/dsv4-hc.cu` (the qwen4exp graph builds `GGML_OP_DSV4_HC_PRE`/`_POST` at
  `src/models/qwen4exp.cpp:548` / `:620`).  Our tree *also* has `hc_mix_reduce_f32`
  (`hyperconn.cu:34`, the reference's counterpart) — confirm with a profile which is live before
  changing either.
- Our tree **already** reuses the GLU output as bf16 for the routed down via **`mmb_slot[2]`** (no GLU
  producer appears in the `MMB_CVT` log) — do not redo that.
- `dsv4_hc_post` is **already at the bandwidth ceiling** (~216-252 MB / 0.9 ms ≈ 240 GB/s), so its
  per-stream `x` re-reads are L1-cached; restructuring is a wash.
- **Prevalent false leads (already measured, do not repeat):** `GGML_CUDA_MMB_CACHE` 32/128 is a wash on
  Flash-Next (960/957/957 t/s); `GGML_CUDA_MMB_TILE`/tile shapes are exhausted; int8-IU8 and a bf16
  weight shadow are refuted on gfx1151 (see `HANDOVER.md` session 6).
- `ggml_cuda_mmb_marks_clear()` exists but is **never called** in our tree; `mmb_down16`/`mmb_blk16`/
  `mmb_res16` exist but are default-0; `ggml_cuda_mmb_mark_bf16_only` is never called either — the
  whole marking machinery is dead code today.

## 4. Reference map

| reference (`~/pwilkin-llama-cpp` @ `f5daaa3cf`) | what it does | our counterpart |
|---|---|---|
| `ggml/src/ggml-cuda/hc-cn.cuh:4-21` | `ggml_cuda_hc_combine_norm_args` **with the bf16 fields** (`out_xn_bf16`, `store_xn_f32`, `res_in_bf16`, `blk_in_bf16`, `res_out_bf16`) | `ggml/src/ggml-cuda/hyperconn.cuh:42` (no bf16 fields) |
| `ggml-cuda.cu:3880-3906` | fills the args at the fusion: `out_xn_bf16 = ggml_cuda_mmb_cache_reserve(...)`, `store_xn_f32 = !(out_xn_bf16 && is_bf16_only(out_xn))`, `res/blk_in_bf16` from marks | our fusion sites `ggml-cuda.cu:5465` and `:5628` |
| `hc-mix.cu:84-140` | `hc_mix_reduce_bf16` + host (`LLAMA_MMB_HC16`, `ggml_cuda_mmb_cache_lookup(xn)/(gate)`) | `hyperconn.cu:34`/`:90` (`hc_mix_reduce_f32`) and `dsv4-hc.cu:104` (`dsv4_hc_pre_f32`) |
| `ggml-cuda.cu:4971-5160` | the **marking pass** at the **top** of `ggml_backend_cuda_graph_optimize`, gated `LLAMA_MMB_HC16 >= 2` + RDNA3_5: marks `xn` (5000), `block_out` (5041), `residual` (5079), hc-mix dst (5097), the `320x10240` gate (5111), MoE `ex` (5133), `glu` (5151) | our `ggml_backend_cuda_graph_optimize` at `ggml-cuda.cu:6185` (**has early returns** — see pitfall 1) |
| `ggml-cuda.cu:4973-4977` | mark lifetime: `g_gt_after_compute` / `g_gt_first_split` key → `ggml_cuda_mmb_marks_clear()` on the first optimize after a compute | absent |
| `ggml-cuda.cu:5041-5050`, `5111` | `ggml_cuda_mmb_blk16()` / `res16()` gates | `mmb.cu:1313-1314` (default 0) |
| ref `hc-mix.cu` `hc_bf2f` | `__uint_as_float(((uint32_t) h) << 16)` | ours: `mmb.cu`/`dsv4-hc.cu` already have bf16 helpers |

## 5. Step plan

1. **Mark lifetime first.** At the very top of our `ggml_backend_cuda_graph_optimize`
   (`ggml-cuda.cu:6185`) — **before** the `enable_graph_optimization` / `use_cuda_graph` early returns —
   replicate the reference's clear-at-first-optimize-after-compute logic (ref `4973-4977`) and call
   `ggml_cuda_mmb_marks_clear()`.
2. **Port the marking pass** (ref `4980-5160`) gated `LLAMA_MMB_HC16 >= 2` + `GGML_CUDA_CC_IS_RDNA3_5`.
   It needs `ggml_cuda_match_hc_combine_norm(cgraph, i, ws, ca)` — our tree builds the args **inline** at
   `ggml-cuda.cu:5465`/`:5628`, so either port the reference matcher (`hc-cn`/`dsv4-hc` helpers) or
   factor our inline construction into a matcher.  Port the `xn`, hc-mix-dst and `glu` cases first
   (they are the ones our `MMB_CVT` log shows); `block_out`/`residual` need `LLAMA_HC_BLK16`/`_RES16`.
3. **Extend `ggml_cuda_hc_combine_norm_args`** (`hyperconn.cuh:42`) with the bf16 fields and teach
   `ggml_cuda_op_hc_combine_norm` (`hyperconn.cu:342`) to write `out_xn_bf16` / `res_out_bf16` (when
   `store_xn_f32 == false`) and read `res_in_bf16` / `blk_in_bf16`.
4. **Fill the new args** at our two fusion sites from the marks, as ref `3880-3906`.
5. **Add the bf16 arm to the HC reduce** (`dsv4_hc_pre_f32`, and `hc_mix_reduce_f32` if live): a bf16
   variant reading `xn`/`gate` (or `gate` only — `xn`'s producer is the `rms_norm*gamma` mul, so it
   needs step 3).  The arithmetic is the reference's `hc_mix_reduce_bf16`; keep the f32 accumulation.
6. Default the env knob, run the gates (below), and only then consider flipping the default on.

Start with the **`gate` (src[1] of `dsv4_hc_pre`)**: its producer is an MMB dense matmul (honours the
mark) and its only consumer is `dsv4_hc_pre` — a contained, safe first win.

## 6. Gates

- **PPL** (a re-baseline — record the new value, check within the run's ±0.027 bar): current
  `c2048 bf16 = 10.5771`; `c16384 bf16 = 3.3821` (session 4).  Long PPL needs a file with > 2·ctx
  tokens — the `prompts/prose-rdna-boosts.txt` file only tokenizes to 5246 tokens.
- **Greedy text**: current `scripts/extract-generated.py` → `665 chars sha=04ddb94b1529`
  (see the session-7 command).  It will move (re-baseline); record the new hash.
- `test-backend-ops -o FLASH_ATTN_QSA` and `-o FLASH_ATTN_EXT` must stay green.
- If the HC path can be reached at `W = 1..8`, re-run the width probe
  (`tests/test-logits-width-probe.cpp`); the decode/verify band must stay width-pure.
- `rocprofv3` kernel time (the value-changing variant rule — `HANDOVER.md` session 5e): judge on kernel
  time, not end-to-end t/s.

## 7. Pitfalls

1. **The marking pass must run before the scheduler assigns data pointers.**  In `graph_optimize` the
   data pointers are not set yet, so `ggml_cuda_hc_combine_norm_alias_ok`-style checks always fail —
   the reference treats the pass as *structure only*.  Our `ggml_backend_cuda_graph_optimize` returns
   early (`GGML_CUDA_GRAPH_OPT` + `use_cuda_graph`), so the marking must go **before** those returns or
   marks will never be set in the default configuration.
2. **Mark lifetime.**  A `mark_bf16_only` tensor's declared type stays `F32` while its buffer holds
   bf16, so **every** consumer must be bf16-aware.  Clear marks between graphs (step 1) or a later graph
   mis-reads f32 data as bf16.
3. **`mmb_blk16` requires `M % 8 == 0`** and skips when a `Dh` slot already owns the output.
4. **RDNA3_5 only** — the bf16 marks are gated to gfx1151 in the reference; keep that.
5. **The reference's helpers are not 1:1.**  Check `git grep` in our tree before assuming a matcher
   exists (`ggml_cuda_match_hc_combine_norm`, `ggml_cuda_hc_mix_closed` may not).
6. `cmake --build build-rocm -j16 --target llama-bench` is ~25 s after a single-file edit (ccache); a
   full model run is ~2-3 min — it is affordable to check in after every step.

## 8. Copy-paste prompt for a fresh session

> Continue the `wip/mmb-general/` work.  Read `wip/mmb-general/HANDOVER.md` (top to bottom) and
> `wip/mmb-general/BF16-PRODUCER-PORT.md`, then execute the plan in §5 there: port the reference's
> (`~/pwilkin-llama-cpp` @ `f5daaa3cf`) native-bf16 producers + graph-optimizer marking so the HC path
> stops paying `mmb_cvt`.  Start with the mark-lifetime clear at the top of
> `ggml_backend_cuda_graph_optimize` and the `hc_gate` marking (its only consumer is `dsv4_hc_pre`),
> then the `xn`/`hc_combine_norm` bf16 output.  Use the model and commands in §2, run the §6 gates, and
> judge on `rocprofv3` kernel time.  Keep the tree at a clean commit per step and regenerate the backup
> (`mmb-general.patch` + `patches/`) when the code changes.
