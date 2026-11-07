# `two-lane-prefill` — `LLAMA_PREFILL_LANES`: consecutive-ubatch prefill overlap

**Optional artifact, not part of the delivery.  On `r229`** (cut against `release.json` tree
`622d017f7a1fac093b24b7bb8dc65a8e15de4cb7`).

From **PR #268 by [@slavap](https://github.com/slavap)**, rebased onto the delivery, reviewed and reduced
(see *What changed in the rebase* below).  The submitter's own description and measurements are in the PR;
this file records what *this repo* verified and what it cannot reach.

* `two-lane-prefill.patch` — the artifact.  sha256 `1515bd63023602bec6b0546f82580a4552200a9e1353dc8ef4a93c0f2ea808fa`
* applies at `release.json` tree `622d017f7a…`, produces tree `e40eabe6f10b27f405d3006ab7c6c5b69703119c`
* ```bash
  bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>
  ```

## What it does

`LLAMA_PREFILL_LANES=2` (default `1`) keeps a **second backend scheduler** with its own compute buffers
and, in the decode loop, prepares two *consecutive* wide prefill ubatches on the two lanes and
submits their splits alternately (`ggml_backend_sched_graph_compute_async_pair`, `a0 b0 a1 b1 …`).  One
device's routed-expert work then overlaps the other device's attention.  Lanes never move work: `-ts` /
`-ot` placement decides which device owns what, the interleave only changes submission order.

Since the r229 cut, a batch of [LLAMA_LANES_SPLIT_MIN (default 1024; 0 disables), 2 x n_ubatch) tokens
is split into two wide sub-ubatches (the first taking LLAMA_LANES_SPLIT_FRAC of it, default 0.55,
rounded up to 64, the chunked-GDN chunk), and the pair gate no longer requires both ubatches
output-free: the batch output row lands in the second sub-ubatch and is extracted per lane, on the
scheduler of each lane.  Before that cut, a prompt batch of fewer than two ubatches never paired, and
only the output-free head of a longer prompt paired.

It degrades to the existing single-lane path when `< 2` non-CPU **backends**, pipeline parallelism,
`no_alloc`, or a draft context (`ctx_other`) is present; a ubatch is only paired when it is wide,
output-free, and no nextn (MTP) or F1 layer-input readback is active, so MTP and DFlash stay single-lane.
A pair is admitted only when both lanes have the same split structure and the cache-hazard analysis finds
no cross-backend hazard (`GGML_SCHED_PAIR_HAZARD=0` keeps the name guard alone; `GGML_SCHED_PAIR_DEBUG`
logs the decision and, at `>= 3`, per-lane/per-backend GPU event spans).

## Which layouts it reaches — and which it cannot

This matters more than the knob does, so it is measured, not assumed (3× R9700, gfx1201):

| layout | result |
|---|---|
| `-sm tensor` (any device count) | **never engages.**  This tree's tensor split is ONE backend (a Meta device wrapping N GPUs), so the feature's `n_gpu >= 2` gate sees 1: `sched_reserve: LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0)` |
| plain `-sm layer`, all layers on GPU | **never engages.**  The scheduler's own pipeline parallelism is enabled in exactly that configuration and the feature declines: `ignored (n_gpu = 3, pipeline_parallel = 1)` |
| `-sm layer` + `--override-tensor` (`-ot`) | **engages** — a tensor override disables pipeline parallelism, which is how the submitter's MoE layout works: `pair decided: splits a=4 b=4, interleaved=1` |
| `-sm row`, `-sm none` with 2+ GPU backends | engages (pipeline parallelism is layer-split-only) |

The submitter's **dense** reference layout (`-sm layer -ts 1,1`, no `-ot`) is refused on this tree for the
same reason, so the +13.3 % it reports was not measured with this patch on this base.

## What it measures here

Same binary, env-gated A/B, pp8192 / `-ub 2048`, `-r 3`, two rounds in ABBA order, pair engaged in both:

| shape, `-sm layer -ot` | 1 lane | 2 lanes |
|---|---|---|
| MoE 35B-A3B Q8_0 | 6132.3 / 6182.2 t/s | 6130.5 / 6188.7 t/s → **+0.03 %** |
| dense 27B Q6_K_XL | 1210.1 / 1201.0 t/s | 1204.0 / 1202.5 t/s → **nil** |

**It is an imbalance-hiding technique, not a general prefill win.**  The submitter's +34 % / +13 % are an
RX 7900 XTX (gfx1100) + Radeon Pro V620 (gfx1030) pair, whose devices differ ~3× in throughput: the fast
card waits for the slow one on every split, and a second ubatch in flight fills that wait.  On devices of
equal throughput every split is already the critical path, so a second ubatch adds queueing and no
throughput.  That is why this is an artifact and not a block: none of the delivery's validated boxes has an
imbalanced multi-device pair.

Correctness here: greedy text byte-identical between lanes (`prompts/prose-rdna-boosts.txt`, `-n 32`,
`--temp 0 --seed 42`, `-ub 512`, sha `61d387d6e21c` on the `-ot` layout, `d34f2e4e737c` on `-sm tensor`),
identical to the delivery build's text in every case; no abort with MTP (`-md`); the default
(`LLAMA_PREFILL_LANES` unset) path is unchanged.

*The r229 cut changed the submission structure (sub-ubatch split + output-aware pairing); the numbers
above are for the r220 cut and need re-running on this cut before they can be cited again.*

### Submitter measurement (r229 cut, imbalanced pair)

RX 7900 XTX (gfx1100) + Radeon Pro V620 (gfx1030), MoE Q6_K_XL, `-sm layer -ts 1,0 -ot
blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1 -dev ROCm0,ROCm1 --load-mode dio`, pair engaged:

| test | 1 lane | 2 lanes | delta |
|---|---|---|---|
| pp2048 | 859.68 t/s | 1382.54 t/s | **+60.8 %** |
| pp2048 @ d32768 | 786.93 t/s | 1200.86 t/s | **+52.6 %** |
| tg256 | modestly up | | |

Correctness: greedy text byte-identical L1 vs L2 at short and ~2100-token prompts; L1 (lanes unset)
matches the delivery baseline.  For llama-cli on MTP models: auto-MTP must be disabled
(--spec-type none); the feature declines MTP workloads by design (nextn gate).

## What changed in the rebase onto r220

1. **Dropped what the delivery now carries.**  PR #268 also diagnosed and fixed the shared per-device
   copy event (`cpy_tensor_async_nowait` + a per-copy event ring).  That part was **promoted separately**
   into block 12 (`WORKLOG.md` 261105.006) because it is a real defect on its own — a queued HIP stream
   wait binds to the *latest* record, so a copy in flight re-binds consumers of copies still queued.  The
   artifact now uses `ggml_backend_sched_next_copy_event` / `cpy_tensor_async_nowait` from the delivery
   instead of defining them (and the duplicated ring and free loop the naive rebase produced are gone).
2. **CUDA portability.**  The PR spelled `hipEventDefault` / `hipEventElapsedTime` unconditionally;
   `ggml-cuda.cu` still builds under nvcc for CUDA, and those have no hipify alias.  They now follow this
   file's own `#ifdef GGML_USE_HIP … #else cudaEventDefault … #endif` form, so the CUDA build is intact.
3. **No cost on the default path.**  The PR switched *every* backend event to a **timing** event so the
   elapsed-time diagnostic could work.  Events are timing-enabled only when `GGML_SCHED_PAIR_DEBUG>=3`
   (the sole consumer), so applying this artifact costs the default path nothing.
4. **Docs corrected** to state the reachability table above rather than implying `-sm layer -ts 1,1`
   works.

## What changed in the re-cut onto r229

The r220-r229 delivery set touches none of the files this patch edits, so the base side of the cut is
byte-identical to the r220 cut; only the two `src/llama-context.*` files differ from it.

The r220 cut paired two consecutive whole ubatches and only when both were output-free, so a prompt
batch of fewer than two ubatches never paired (its single ubatch carries the batch output row), and a
2..3-ubatch batch paired only its output-free head.  On an imbalanced pair the feature therefore
measured neutral in the submitter R223 test, and the r220-cut numbers above (balanced devices) never
showed a win.  The r229 cut adds:

1. **The sub-ubatch split.**  A batch of [LLAMA_LANES_SPLIT_MIN (default 1024; 0 disables), 2 x
   n_ubatch) tokens is split into two wide ubatches, the first taking LLAMA_LANES_SPLIT_FRAC of it
   (default 0.55, rounded up to 64, the chunked-GDN chunk); the batch output row lands in the second
   ubatch.  Batches of 2 x n_ubatch or more keep the plain n_ubatch split.
2. **Output-aware pairing.**  The pair gate no longer requires both ubatches output-free; either
   partner may carry the outputs, which are extracted per lane after compute, on the scheduler of each
   lane (extract_outputs, with sched-aware readback sync; the single-lane path runs the same function
   on sched).  The readback sync is skipped on the final ubatch (the caller consumes its outputs and
   syncs).  MTP (nextn) and F1 layer-input readback workloads stay single-lane as before.

## Known open points (unchanged from the PR)

* On the pair-failure path (`graph_compute_pair != SUCCESS`) lane 1's ubatch has already been applied to
  the memory module by `prepare_ubatch`, and the loop then continues into the single-lane path, which
  applies it a second time.  Error path only; a double `mctx->apply()` for the same positions is not
  obviously idempotent for every memory type.
* `include/llama.h` gains `llama_n_prefill_lanes()`, a public accessor for an internal feature.
* The second scheduler costs a second set of compute buffers (reported at startup as
  `compute buffer size = … MiB (prefill lane 2)`).

## Env knobs (see `ENVIRONMENT.md` for the class of each)

| variable | default | what |
|---|---|---|
| `LLAMA_PREFILL_LANES` | `1` | `2` enables the pair.  Anything else (or a layout the table above refuses) logs one WARN and stays single-lane |
| `LLAMA_LANES_SPLIT_MIN` | `1024` | `0` disables the sub-ubatch split (whole-ubatch pairing only, the r220 behaviour) |
| `LLAMA_LANES_SPLIT_FRAC` | `0.55` | fraction of the batch the first sub-ubatch takes (clamped to [0.1, 0.9], rounded up to 64) |
| `GGML_SCHED_PAIR_DEBUG` | `0` | `1`: log each pair decision; `>= 2`: also run the hazard self-test; `>= 3`: per-lane/per-backend GPU event spans (creates timing events) |
| `GGML_SCHED_PAIR_HAZARD` | on | `0` keeps only the split-name guard, skipping the cache-hazard analysis (diagnostic) |

## Why this is not a block

Promotion needs a validated win on one of the delivery's own configurations (`optional/README.md`).  If an
imbalanced multi-device pair ever becomes a validated target, this artifact is the candidate — and the
hazard analysis, the split-structure guard and the reachability table are what it would be promoted on.
