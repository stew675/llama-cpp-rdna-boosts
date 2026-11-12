# `-sm tensor` host experts: the Meta split copy leaves the tail guard of the last device uninitialised (r269)

**Status:** fixed in `v17-a691b2345-r269` (block 09 and block 10).  **Symptom:** a host-expert
`llama-server` on a `-sm tensor` split emits an unbroken run of `/` (or `3`) from the first decoded
token.  **Not a throughput finding:** the numbers here are correctness arms and a text hash.

## The report and the shape

The reporter's command loaded **Qwen3.8-Flash-Next UD-Q4_K_XL** (4 shards, 106 GiB) on 3 x R9700
(gfx1201), `--n-cpu-moe 44 --limit-host-experts auto --split-mode tensor`, `--ctx-size 204800`,
q8_0 K/V, adaptive MTP + ngram-mod.  The first request returned 512 `/` characters; a simple
`The capital of France is` request returned `333333...`.  Other quants of the same model
(Q4_K_M, IQ4_NL) were coherent, and `-sm layer` was coherent with the same quant.

Reproduction used the reporter's prompt (`sha256 e89f8e311d56...`, 662 chars, the Three.js page
request) at `CTX=32768` first and then at the reporter's `CTX=204800` / `-n_predict 512`.  The
pre-fix exact arm returned **512 `/`**; the post-fix arm returns **2208 chars, sha256
`53882027cf15`**, maximum `/` run **1**.

## Root cause

`copy_experts` (`ggml/src/ggml-backend.cpp`) uploads the used experts of a host table and appends a
`min(expert_size, 512)`-byte guard from the next expert, so the quantized `MUL_MAT_ID` MMQ's tail
over-read sees finite bytes instead of the next slot's uninitialised allocation.  Under
`-sm tensor`, `ggml_backend_meta_set_tensor_async()` splits that contiguous source range across the
devices.

For an axis-0 (`ffn_down_exps`) table the device slice is a **row slice**, and the source row of the
next expert is `row_bytes = n_embd / blck_size * type_size`.  Q8_0's 640-element row is **680 B**,
and the 3-device granularity-rounded shares are 128/256/256 elements, i.e. **136/272/272 B** per
device.  The 512-byte guard is distributed as 136 + 272 + 104 bytes: the **last device's slice
starts at byte 408 and only 104 of its 272 bytes are written**.  The remaining 168 bytes of that
slot are never initialised (normally NaN), and the MMQ tail over-read poisons the tile — the
repeated-`/` corruption.

This is why only this quant reproduces it.  Q4_K_XL's dynamic mix has Q8_0 `ffn_down_exps` in layers
4/30/46/47 (and a Q5_K gate/up in layer 2); Q4_K_M's down experts are Q5_1 (400 B rows, so the
512-byte guard covers a whole row) and every other quant tested has no Q8_0 expert table.  The
defect is in the **copy**, not the cache: `MOE_EXPERT_CACHE_MIB=0` still produces `/`, while
`GGML_META_SPLIT_COPY=0` (mirrored copies) is coherent.

## Fix

In the partial-tail path of the Meta split setter, after copying the bytes the source actually
provides, **zero the rest of each device's next-expert guard** (the first 512 bytes of the device
slot; the slot is contiguous across device rows, so an axis-0 slice that is narrower than 512 B is
filled across rows).  Head partial rows are untouched — their prefix belongs to a previous chunk.
The same fill is applied to the segmented (fused-table) async and synchronous setters.  No new
environment variable, no default changed; the guard is still finite, it is now deterministic.

## Verification (gfx1201, ROCm 10.1.0, 3 x R9700 unless noted)

| arm | result |
|---|---|
| reporter's config and prompt, `-c 204800`, 512 tokens | pre-fix **512 `/`**; post-fix **2208 chars `53882027cf15`, max `/` run 1** (three builds) |
| same model, `-c 32768`, cache off | pre-fix **96 `/`**; post-fix **388 chars `3ea48d7566b3`** |
| same model, `-sm layer` (control) | coherent, 390 chars |
| same model, `GGML_META_SPLIT_COPY=0`, cache off (control) | coherent, 384 chars `1233cbd8e0a8` |
| Q4_K_M, same `-sm tensor -ncmoe 44` config, cache off (`SPEC=0`) | pre-fix and post-fix **both 436 chars `21f493f3ab52`** (no drift on the working quant) |
| IQ4_NL, same config | coherent, 389 chars `1d00ec05be02` |
| `gate-qwen4exp-quant-coherence.sh QUANTS="Q4_K_XL Q4_K_M" NGPU=3 NCMOE=44 CTX=32768 NGEN=96 QSA_ARM=0` | **PASS** both quants; gather on == gather off (`b1af8904a110` / `7160cb056885`), 0 slash runs, kernel journal and every log clean |
| `gate-prefill-logits.sh` (27B Q8_0, 40 chunks) | mean KLD **0.000000**, same-top-p **100.000 %** |
| `gate-per-block-build.sh --fast --changed-from e533fb2af --tip 42f90c9b8` | **3/3** (block 09 321 s, block 10 327 s, tip 6 s) |
| `validate-set.sh` | 17 blocks, checksums + strict apply, tree `db3494fa`, Windows scan 291 files clean |

**Not established / not chased:** the `GGML_META_SPLIT_COPY=0` kill-switch arm with the expert cache
**on** aborts in `ggml_backend_cuda_get_tensor_2d_async` on this model (a 2-D read of a mirrored
copy); the shipped default does not take that arm and the cache-off control above is the one that
was used.  This is recorded as an observation, not as a fixed path.

## Records

`WORKLOG.md` 261112.005; `INVARIANTS.md` r269; `GREEDY-PURITY.md` §87; `patches/README.md` blocks 09
and 10.
