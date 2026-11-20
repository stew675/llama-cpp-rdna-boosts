# Measured conflict map — 18 blocks replayed onto `919794500`

Regenerate with `python3 probe-conflicts.py ~/llama.cpp` (text conflicts, `git merge-tree`, no worktree
touched) and `python3 probe-overlap.py ~/llama.cpp` (upstream churn on each block's files, plus the upstream
commits whose subject enters a delivery space).  Machine-readable output: `conflict-map.json`,
`overlap-map.json`.  Measured 2026-11-20, upstream drift `4f9dfa0e3..919794500` = 228 commits / 1944 files /
+147 254 −41 182.

**Caveat, by construction:** a conflicted block leaves conflict *markers* in the synthetic tree and the probe
chain continues from that tree, so a later block's count is slightly pessimistic.  Read "0 conflicts" as
solid and a high count as indicative.  This is not a resolution tool — the real climb is `git cherry-pick`
per rung.

## §1  Summary table

`conf` = conflicted paths in the probe.  `shared` = of the block's files, how many upstream also touched.
`churn` = upstream commits landing on those shared files (a block's total exposure).  `upcmt` = distinct
upstream commits behind that churn.

| rung | blk | sha | files | shared | churn | upcmt | conf | verdict |
|---|---|---|---|---|---|---|---|---|
| R00 | 00 | `de99fa2d0` | 37 | 27 | 93 | 55 | **4** | heavy: `ggml-cuda.cu` (13 up cmts), `mmid.cu`, `mmf.cuh` — **D1/D2/D3** |
| R01 | 01 | `a94e2214c` | 4 | 1 | 13 | 13 | **0** | clean apply; `ggml-cuda.cu` only for the all-reduce arm — verify by building |
| R02 | 02 | `cef19b032` | 23 | 20 | 95 | 53 | **5** | `common/speculative.cpp` + docs collide with the **DFlash spec refactor** (`32bdc3a89`) — **D8** |
| R03 | 03 | `36da15029` | 25 | 13 | 70 | 47 | **1** | only `tests/test-backend-ops.cpp` (26 up cmts); upstream CUDA has **no** chunked GDN, so the block still stands |
| R04 | 04 | `cfe40b5f9` | 8 | 7 | 51 | 33 | **2** | `ggml-cuda.cu` + `rope.cu` vs **`#32387` BF16 ROPE/SET_ROWS bf16-cache rope fusion** — **D5** |
| R05 | 05 | `7362b5958` | 341 | 23 | 96 | 51 | **7** | FA family: `fattn-*` vs `#31617` warp-sync refactor, `#32407` KV read order, `#32217` Inkling2; `llama-graph.cpp`; `test-backend-ops.cpp` |
| R06 | 06 | `f0564df7d` | 11 | 5 | 19 | 15 | **2** | `mmid.cu` (MMID rework) + `ggml-cuda.cu` — **D7** |
| R07 | 07 | `f5bdbefda` | 14 | 8 | 58 | 43 | **1** | `mmvq.cu` vs `#32308` MMVQ ne11 gate, `#31268` Orin, `#30672` accumulation gating |
| R08 | 08 | `fe283f27c` | 38 | 16 | 64 | 38 | **9** | worst ratio: `common.cuh`, `mmf.cuh`, `mmid.cu`, `mmq.cuh`, **`norm.cu` (`#32454` NORM+MUL fusion)**, `quantize.cu` (`#32065`), `set-rows.cu` (`#32387`), `topk-moe.cu`, tests — **D4/D5/D6/D7** |
| R09 | 09 | `123d53419` | 70 | 39 | **136** | **83** | **4** | highest exposure: `ggml-backend.cpp`, `common.cuh`, `ggml-cuda.cu`, **`llama-kv-cache.cpp` (`#32653` lid_k_hdmd, `#32380` kv-cells)** — **D9** |
| R10 | 10 | `7bbf540d5` | 87 | 30 | **135** | 71 | **9** | the FA family again + `fwht.cu`, `mmvq.cu`, `rope.cu`, `topk-moe.cu`; **node budget** (`#32648`/`#32636`) merges here |
| R11 | 11 | `61168c60b` | 52 | 30 | **140** | 66 | **5** | `ggml.c`, `ggml-rpc.h`, `llama-memory-recurrent.cpp`, **`src/models/qwen4exp.cpp`**, tests — qwen4exp has live upstream work |
| R12 | 12 | `dad1442b3` | 64 | 38 | **143** | 63 | **9** | broadest: `speculative.cpp`, `ggml.h`, x86 quants, `ggml.c`, **`dflash.cpp`**, **`server-context.cpp` (14 up cmts)**, `server-task.*` — **D8/D10** |
| R13 | 13 | `1014e6bf5` | 19 | 14 | 59 | 30 | **0** | clean apply (its files are shared but the hunks do not overlap) |
| R14 | 14 | `e88d45d9e` | 51 | 29 | 91 | 50 | **3** | `ggml-rpc.h`, `ggml.c`, **`src/llama-ext.h`** (`#31913` sticky NaN, `#32553` packing) |
| R15 | 15 | `35a9ae985` | 16 | 4 | 24 | 24 | **0** | clean apply |
| R16 | 16 | `286191b31` | 3 | 3 | 25 | 20 | **0** | clean apply — but its 3 files are `server-context.cpp`, `arg.cpp`, `common.h`, i.e. the server refactor's core (**D10**) |
| R17 | T | `e159adfc2` | 11 | 6 | 35 | 26 | **0** | clean apply; `#31435` (`--tensor-split` device count) is the thing to read into it (**D11**) |
|  | **total** | | | | | | **61** | |

Rungs that apply **clean** to the text merge: **R01, R13, R15, R16, R17**.  That is 5 of 18 — the rest need a
human.

## §2  The upstream commits that cause most of it

A short list, because the same handful of commits own most of the 61 paths:

| upstream | what | delivery exposure |
|---|---|---|
| `278a89cb7` | CUDA/HIP: refactor `_sync` warp primitives | `common.cuh`, `fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-vec.cuh`, `mmid.cu`, `quantize.cu`, `topk-moe.cu`, `fwht.cu` — **8 of the 61 paths**, blocks 05/08/09/10 |
| `5c008e000` | Inkling 2 (text+vision) | entire `fattn-*` family + `llama-graph.cpp` — blocks 05/10 |
| `b94959a64` | **duplicate expert ids** (`MUL_MAT_ID`) | `ggml.h`, `common.cuh`, `fattn.cu`, `ggml-cuda.cu`, `mmf.cu(h)`, `mmid.cu`, `top-k.cu`, tests — blocks 00/04/05/06/08/09/10/12 |
| `03ec5b432` | **CUDA NORM + MUL (+ ADD) fusion** | `ggml-cuda.cu`, `norm.cu(h)`, tests — blocks 08/10 |
| `7fce5439f` | **BF16 ROPE + SET_ROWS**, bf16-cache rope fusion | `ggml-cuda.cu`, `rope.cu`, `set-rows.cu`, `ggml.c`, `ops.cpp`, tests — blocks 04/08/10/11/12/14 |
| `8777c0c83` | few-row MMQ for `MUL_MAT_ID` | `ggml-cuda.cu`, `mmq-few.cu(h)`, `quantize.cu(h)`, tests — blocks 08/10 |
| `2e53a0bab`, `b79696f1f`, `c7518dbec` | HIP/RDNA MMQ + MMVQ tuning | `mmq.cuh`, `mmvq.cu` — directly on top of blocks 06/07/08's tuning tables |
| `32bdc3a89` | **spec + model: DFlash spec decoding refactor** | `common/common.h`, `common/speculative.cpp`, `docs/speculative.md`, both READMEs, `src/models/dflash.cpp` — blocks 02/12 |
| `429460dc0` | **server: `server_slot_sched`, prompt running hash** | `server-context.cpp`, `server-task.cpp`, `server-task.h` — blocks 12/16 |
| `62a6d3a30` + `a06044b93` | **`llama_batch_ext` migration**, old batch API deprecated | every tool/common file that builds a batch — blocks 02/12/16 (and the CLI arms of the whole gate set) |
| `652d559ed` + `4c4a0f0a4` | graph node budget into the model classes; reserve nodes for the MoE cache | `llama-context.cpp`, `llama-model.h`, `models.h`, `dflash.cpp` — blocks 09/10/12 (delivery has its own budget work here) |
| `4783b1145` | indexer key rotation -> `llama_model::lid_k_hdmd()` | `llama-kv-cache.cpp`, `llama-model.h`, `models.h` — blocks 09/11 |
| `ac830a9cb` | ggml backend scheduler tests | `ggml-rpc.h`, `ggml.c`, `ggml.h` — blocks 11/12/14 (mechanical) |
| `7716ec8cb` | `ggml_ftype` deprecation + version bump | `ggml.c`, `ggml.h` — blocks 11/12/14 (mechanical) |
| `0a7675746` | sticky-NaN fix in the recurrent state | `llama-graph.cpp`, `llama-ext.h` — blocks 05/14 |
| `16f80bd0c` | **vulkan** chunked GATED_DELTA_NET | tests only on our side; CUDA still has `//TODO: Add chunked kernel` |
| `4c4a0f0a4`/`c56be864f` | ggml-backend split rules | `ggml-backend.cpp`, `llama-moe-cache.cpp` — blocks 09/0300 |

## §3  Upstream moving into a delivery space (no text conflict required)

These are the collisions that a text-only probe cannot see, and the reason [`DECISIONS.md`](DECISIONS.md)
exists.  Full evidence per item there.

| space | upstream now has | delivery owns |
|---|---|---|
| MoE expert cache | `src/llama-moe-cache.cpp`, `docs/moe-cache.md`, `--moe-cache-size`, `--moe-cache-cpu-share`, node reserve `#32636`, split rule `c56be864f` | block 09's host-resident MoE (iface, scheduler, Meta, slab, cache engine) extends it; block 15's GTT store is a different tier |
| `MUL_MAT_ID` | duplicate ids `#32505`, few-row MMQ `#32065`, per-expert precision `#9fbe30c2a`, odd-columns abort fix `f477c1b90`, MoE-on-V100 MMQ, HIP MoE regression fix `05db09a4d` | blocks 00/06/07/08 (few-row band, MMB, mmid) |
| norm fusion | `#32454` NORM+MUL(+ADD) in `norm.cu`, `#32300` metal norm chain | block 08's `norm-gated.cu` (narrow-row wave-per-row, sigmoid gate) + the fusion dispatcher |
| BF16 | `#32387` BF16 rope/set_rows and a bf16-cache rope fusion; `#30672` SKU-aware fp16/bf16 accumulation gating | block 04 (BF16 KV cache + native BF16 FA) |
| spec decode | `#32016` DFlash refactor, `#31904` DFLASH2, `#32281` CI spec-dec triggers | blocks 02 (adaptive MTP depth), 12 (DFlash), 13 (MTP draft vocab) |
| GDN | `#32660` `supports_op` update, `#31327` mma kernel on CDNA, `#30207` chunked GDN for **vulkan** | block 03 (chunked GDN prefill, **CUDA**) — still unique |
| graph budget | `#32648` into model classes, `#32636` MoE-cache reserve | block 10's budget error + KV-rotation leaf accounting |
| indexer | `#32653` `lid_k_hdmd()` virtual, `#32483`/`#32476` qwen4exp fixes | block 11 (qwen4exp QSA/indexer, hyper-connections) |
| server slots | `#32154` `server_slot_sched` + prompt running hash, `#32434` lora/recurrent fix | block 16 (max-min fair admission, per-step chunk, duty cycle), block 12 (idle-slot parking, prompt cache) |
| tensor split | `#31435` up to `llama_max_devices()` configs | block 0300 (`-sm relaxed-tensor`) |

## §4  What this means for the order of the climb

The ladder is **block order** (a cherry-pick per block, in release order), because that is the only order in
which each rung's diff is the diff of one coherent feature.  But the *cost* is concentrated in R00 (MMID +
`ggml-cuda.cu`), R08 (the MMQ/norm/set-rows knot), R09-R12 (the MoE/KV/server/spec knot).  Budget:

- R00-R07: one session each is comfortable; R00 may take two (it carries D1/D2/D3).
- R08-R12: the heavy rungs.  Each is a resolution *plus* a decision or two.
- R13-R17: cheap; R16/R17 apply clean but need reading (server refactor, `--tensor-split`).
- R18: the release ladder (`scripts/gate-per-block-build.sh --all`, `validate-set.sh`, `make-patches.sh`,
  `make-release.sh`, `fork-refresh.sh`, then the worker container build).
