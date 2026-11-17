# Issue #323: a host-expert routed MoE is offloaded to the GPU at EVERY ubatch width

**Claim.** A routed `MUL_MAT_ID` whose expert bank is in the delivery's *host-expert tier* (`-ncmoe`/`-cmoe`
with the `ROCm_Host` pinned master or the `ROCm_HostMap` model mapping behind the pool) used to be judged by
the generic `GGML_OP_OFFLOAD_MIN_BATCH` (default **32**) whenever it fell **above the expert cache's decode
band (16 on RDNA4)**, so a 17..31-token ubatch ran its MoE on the **CPU**.  It now always runs on the GPU.
That window is where a chat server lives: every short prompt chunk, every follow-up turn, and every prefill
step of a two-chat session lands in it.

Reporter: @briansp2020, [issue #323](https://github.com/stew675/llama-cpp-rdna-boosts/issues/323).  Fix:
r305, block 09, `ggml_backend_cuda_device_offload_op()`; `GGML_MOE_HOST_OFFLOAD=0` is the kill switch and
restores the pre-r305 decision exactly (every number below was taken with it as the "before" arm).

## Why the generic threshold is wrong for this op

`GGML_OP_OFFLOAD_MIN_BATCH` exists because offloading a **host weight** means uploading it whole before the
op runs, so for a small batch the copy costs more than the CPU compute.  Neither half of that holds for a
routed MoE in this tier:

* the GPU arm does **not** upload the bank -- the expert cache serves the routed set from the arena/pool, or
  streams only the experts that are not resident, through the delivery's own coalesced parallel read engine
  (block 15's prefill stream, block 12's staging ring);
* the CPU arm reads the **same bytes** out of the pageable model mapping -- one page fault and
  `do_sync_mmap_readahead` per 4 KiB, every OpenMP thread queued on one page-cache lock, which is exactly the
  profile the reporter posted -- and then computes them at CPU rates.

So the CPU arm is dominated on both axes, and the *only* widths where it was ever chosen were 17..31 tokens
(at or below the band the cache take-over already forced the GPU; at 32 and above the generic threshold did).
It is also a purity fix: the backend a routed op ran on depended on how many tokens happened to share its
ubatch.

## Measured (gfx1201, 2 x R9700, ROCm 10.1.0, cold page cache)

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (83.6 GB, 512 experts, 48 blocks), `-ngl 99 -ncmoe 47 -lme 47 -c 16384
-ctk q8_0 -ctv q8_0 --load-mode dio --lazy-mode auto`, `HIP_VISIBLE_DEVICES=0,2`, a **19-token** prompt,
`-n 1 --no-warmup --temp 0 --seed 42`; `sync; echo 3 > /proc/sys/vm/drop_caches` before every run.  Numbers
are llama-cli's own prompt-eval rate for that single ubatch (higher is better); `mmid@CPU` is the count of
`MUL_MAT_ID` nodes the scheduler placed on the CPU in the `GGML_SCHED_DEBUG=2` dump.

| arm | before (r304) | mmid@CPU | after (r305) | mmid@CPU | ratio |
|---|---:|---:|---:|---:|---:|
| `-sm tensor`, pool tier (pageable master) | **2.4 / 2.5** | 138 | **76.2 / 78.5** | 0 | **31x** |
| `-sm layer`, pool tier | 2.5 | 138 | 80.5 | 0 | 32x |
| `-sm tensor`, `--host-experts pinned` (whole-tensor `ROCm_Host`) | 58.7 | 138 | 70.7 | 0 | +20 % |
| `-sm tensor`, `MOE_EXPERT_CACHE_MIB=0` (cache off) | 1.1 | 2271 | 7.9 | 0 | 7.2x |
| **control**: 436-token prompt (already >= 32), `-sm tensor` | 442.2 | 0 | 435.2 | 0 | inert |

The control is the point of the fix's scope: at 32 tokens and above nothing moves (the generic threshold was
already offloading), and below the band nothing moves (the cache take-over already was).  Only the
17..31-token window changes.

The pinned-master arm is the "cheapest possible CPU arm" -- the experts are already in RAM, nothing is read
from disk -- and the GPU arm still wins there by 20 % **on the fastest consumer CPU available** (9950X3D), so
the rule is not tuned to a slow CPU.  The cache-off arm is the same width on a *plain* whole-table upload,
which is the most expensive GPU arm the delivery has, and it still wins by 7.2x.

## Purity

Same-seed greedy text at the 19-token width on `Qwen3.6-35B-A3B UD-Q4_K_M`, `-sm layer -ncmoe 40`,
`-n 24`, `--temp 0 --seed 42`:

| arm | generated text |
|---|---|
| r305 default (GPU) | `105 chars sha=c16075337035` |
| r305 with `GGML_MOE_HOST_OFFLOAD=0` (the pre-r305 CPU arm) | `105 chars sha=c16075337035` |
| r305 with `GGML_OP_OFFLOAD_MIN_BATCH=9` (the reporter's workaround) | `105 chars sha=c16075337035` |
| all-VRAM (`-ncmoe 0`), the numerical oracle | `105 chars sha=c16075337035` |

So at this width the two arms agree with each other and with the all-VRAM build on this prompt; the fix's
justification is the 31x, not a purity repair.

## Not covered by this rule

* the **APU GTT expert store** (`-lme` on an IGPU): its store is a DEVICE buffer type, so the op is assigned
  by buffer ownership and never reaches this decision (and `-ncmoe` is refused there anyway);
* a **plain CPU-buffer** expert bank (`--override-tensor 'exps=CPU'`, i.e. the delivery's tier bypassed):
  the generic threshold still applies, because there is no pool/stream to serve the GPU arm from and a
  whole-table upload per ubatch really is the CPU's competition there;
* `-sm tensor` where the loader merges each device's host-master buffer into a per-layer **Meta** buffer:
  that weight is already owned by the meta backend, so the decision is the same before and after (measured:
  `-sm tensor` + `Qwen3.6-35B-A3B-Q4_K_M -ncmoe 40` puts 0 `MUL_MAT_ID` nodes on the CPU either way).  The
  Flash-Next rows above are the shape where the per-device host buffer survives, which is the reporter's.

## Reproduction

```bash
# before / after, on the same binary (the kill switch is the "before"):
HIP_VISIBLE_DEVICES=0,2 GGML_SCHED_DEBUG=2 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  -ngl 99 -sm tensor -ncmoe 47 -lme 47 -c 16384 -ub 6144 -b 6144 \
  -f <a 19-token prompt> -n 1 --no-warmup --temp 0 --seed 42 --no-display-prompt --single-turn -lv 5
# add GGML_MOE_HOST_OFFLOAD=0 for the pre-r305 arm; add `-sm layer` for the second row
```
