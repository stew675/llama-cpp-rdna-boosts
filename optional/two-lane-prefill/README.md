# `two-lane-prefill` — `LLAMA_PREFILL_LANES`: consecutive-ubatch prefill overlap

**Optional artifact, not part of the delivery.  On `r229`** (cut against `release.json` tree
`622d017f7a1fac093b24b7bb8dc65a8e15de4cb7`).

From **PR #268 then PR #279 by [@slavap](https://github.com/slavap)**, rebased onto the delivery, reviewed and
amended (see *What changed in the rebase* below).  The submitter's own description and measurements are in the
PRs; this file records what *this repo* verified, what it measures here, and what it cannot reach.

* `two-lane-prefill.patch` — the artifact.  sha256 `c451d99696aa13218147e2117db3d411804cb74a6b2a41e98c6009203e87457e`
* applies at `release.json` tree `622d017f7a…`, produces tree `0d8d509d78bb5ba2f3797af0a39e9466fd1e6668`
  (verified by strict `git am` on a fresh checkout of `.tip`)
* ```bash
  bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>
  ```

## What it does

`LLAMA_PREFILL_LANES=2` (default `1`) keeps a **second backend scheduler** with its own compute buffers and, in
the decode loop, prepares two consecutive wide prefill ubatches on the two lanes and submits their splits
alternately (`ggml_backend_sched_graph_compute_async_pair`, `a0 b0 a1 b1 …`), so one device's work overlaps
another device's.  Lanes never move work: `-ts` / `-ot` placement decides which device owns what, the interleave
only changes submission order.

Since the r229 cut there are two halves, and **both are needed** (the numbers below are the proof):

1. **The sub-ubatch split.**  A batch of [`LLAMA_LANES_SPLIT_MIN` (default 1024; `0` disables), 2 x n_ubatch)
   tokens is split into two wide ubatches, the first taking `LLAMA_LANES_SPLIT_FRAC` of it (default 0.55,
   rounded up to 64 = the chunked-GDN chunk); the batch's output row lands in the second ubatch.  Before this
   cut only *whole* ubatches paired, so a prompt batch of fewer than two ubatches never paired at all.
2. **Output-aware pairing.**  The pair gate no longer requires both ubatches to be output-free: the outputs are
   extracted per lane after compute, on the scheduler of each lane (`extract_outputs`, with a sched-aware
   readback sync).  MTP (`nextn`) and F1 layer-input readback stay single-lane.

It degrades to the existing single-lane path when `< 2` non-CPU **backends**, pipeline parallelism, `no_alloc`,
or a draft context is present, and a workload that reads `nextn` / F1 layer-input rows declines the pair **and
now also declines the split** (see amendment 1).

A pair is admitted only when both lanes have the same split structure and the cache-hazard analysis finds no
cross-backend hazard (`GGML_SCHED_PAIR_HAZARD=0` keeps the name guard alone; `GGML_SCHED_PAIR_DEBUG` logs the
decision and, at `>= 3`, per-lane/per-backend GPU event spans).

## Which layouts it reaches — and which it cannot

This matters more than the knob does, so it is measured, not assumed (3× R9700, gfx1201):

| layout | result |
|---|---|
| `-sm tensor` (any device count) | **never engages.**  This tree's tensor split is ONE backend (a Meta device wrapping N GPUs) so the `n_gpu >= 2` gate sees 1: `sched_reserve: LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0)` |
| plain `-sm layer`, all layers on GPU | **never engages.**  The scheduler's own pipeline parallelism is enabled in exactly that configuration: `ignored (n_gpu = 3, pipeline_parallel = 1)` |
| `-sm layer` + any `--override-tensor` (`-ot`) | **engages** — a tensor override disables pipeline parallelism (`pair decided: splits a=4 b=4, interleaved=1`).  **This is the only reachable layout on ROCm.** |
| `-sm row` | **not usable at all on ROCm**: `llama_model_load: device ROCm0 does not support split buffers` (the r220 cut listed this as a reachable layout — it is an inference, not a measurement, and it is wrong for this backend) |
| `-sm none` | one GPU backend, i.e. the `-sm tensor` case |

The submitter's **dense** reference layout (`-sm layer -ts 1,0`, no `-ot`) is refused here for the same reason,
so the +13.3 % PR #268 reported for it was not measured with this patch on this base.

## What it measures here

`-sm layer -ts 1/1/1 -ot 'blk\.0\.attn_q\.weight=ROCm0'` (a placement-neutral override: it only turns pipeline
parallelism off), `-ub 2048`, `-t 16`, `-r 5`, two rounds in ABBA order unless noted.  Same binary, env-gated.

### The gain needs both halves (MoE 35B-A3B Q8_0, pp2048)

| arm | t/s | vs lanes unset |
|---|---|---|
| lanes unset (one 2048-token ubatch) | 6621 | — |
| lanes=2, `LLAMA_LANES_SPLIT_MIN=0` (**pair alone**: one ubatch, nothing to pair) | 6703 | +1.2 % (nil) |
| lanes unset, `-ub 1152` (**split alone**: the same two ubatch boundaries, run serially) | 5608 | **−15.3 %** |
| lanes=2, split + pair | **8436** | **+27.4 %** |

So the r220 cut's measurement was right and irrelevant: pairing two *already wide* ubatches is worth nothing
here (this is why the r220 artifact measured +0.03 %), while pairing the two halves of **one** wide ubatch is
worth a quarter of the prefill — a single unsplit ubatch serializes the layer-split pipeline and the split
gives the second lane something to run.  The dense 27B Q6_K_XL shows the same 2×2: split alone −3.6 %
(1246 vs 1293), pair alone −1.2 % (1277, nil), split + pair **1823 (+41.0 %)**.

### Shape sweep (35B-A3B Q8_0)

| shape | 1 lane | 2 lanes | |
|---|---|---|---|
| pp512 (below the split minimum: nothing to split or pair) | 4179–4191 | 4149–4166 | neutral |
| pp2048 | 6621 | 8436 | **+27.4 %** |
| pp4096 | 6486 | 8245 | **+27.1 %** |
| pp8192, `-b 8192` (one decode call → 4 ubatches, paired 2×2) | 6240 | 9251 | **+48.3 %** |
| pp8192, `-b 2048` (4 decode calls, each one split batch) | 7530 | 7941 | +5.5 % |
| dense 27B pp2048 | 1293 | 1823 | **+41.0 %** |
| dense 27B pp8192 | 1488 | 1745 | +17.3 % |
| 27B `…-mtp.gguf` pp2048 (llama-cli prompt eval) | 844.7 | 1078.5 | +27.7 % |
| tg128 (MoE, dense, `…-mtp`) | 96.46 / 19.22 / 32.5 | 96.7 / 19.23 / 32.6 | nil |

The gain is largest when ubatches are actually paired inside one decode call; feeding a long prompt in `-b`
chunks (one split batch per call) leaves +5.5 %.

### Correctness

* **The default path is bit-identical.**  `scripts/gate-prefill-logits.sh` against a base recorded from the
  delivery build: mean KLD **0.000000**, same-top-p **100.000 %**; pp2048 6604–6628 vs 6618–6636 t/s (ABBA).
* **The pair is bit-exact.**  lanes=2 vs lanes unset at the same ubatching measures **exactly** the divergence
  of a plain `-ub 1152` vs `-ub 2048` change on the same build — mean KLD `0.004088`, same-top-p `97.898 %`,
  identical to six digits (this is the §46 "benign reordering" scale — a smaller ubatch, nothing more).  So the
  interleave adds no approximation; the split's divergence *is* the ubatch change, which is why greedy text
  staying identical is luck of the greedy path and not identity.
* Same-seed greedy text (`-n 32 --temp 0 --seed 42`, `/tmp` 2145-token slice of
  `prompts/prose-rdna-boosts.txt`): lanes=2 == lanes unset, sha `f61214fad249` on the MoE and `8c1c642fc7cd` on
  the dense 27B, before and after amendment 1, and with a DFlash2 draft after amendment 1.
* Perplexity (`--ctx 2048 --chunks 8 --ub 2048`, every token an output row, i.e. **both** lanes carry outputs)
  restores the split's exact single-lane value — `llama-perplexity` output identical to the `-ub 1152` arm.

### What it costs (all measured, all documented rather than tuned away)

* **A second compute-buffer set per armed context**: ~1.4–1.5 GiB at `-ub 2048` (`ROCm0 441 + ROCm1 430 +
  ROCm2 430 + host 65` MiB, logged once as `compute buffer size = … (prefill lane 2)`).
* **A second context in the same process runs ~8 % slower with the lanes armed** (pp8192: 7976 → 7347 on the
  2nd context, while lanes unset is flat at 7513–7519 across three contexts).  Unexplained; the extra buffer
  set is the obvious suspect.
* **Multi-sequence prefill regresses**: `llama-batched-bench -npp 2048 -ntg 128 -npl 1,4,8` → PP/S_TG
  6289→8101 (+28.8 %), 8363→8437 (+0.9 %), **8776→8467 (−3.5 %)**; TG 92.98→92.96, 243.2→246.4, 322.7→322.9.
  Server workloads with many concurrent long prompts therefore do not want this.
* **Arming the lanes on a draft context costs ~4 % prefill / ~2 % decode** even though the pair declines
  (35B-A3B + DFlash2 pp2048: 4042/4026 vs 3894/3867).  Leave `LLAMA_PREFILL_LANES` unset for draft workloads.

## What changed in the rebase

**r220 (the first artifact cut):**

1. **Dropped what the delivery now carries.**  PR #268 also diagnosed and fixed the shared per-device copy
   event (`cpy_tensor_async_nowait` + a per-copy event ring).  That part was **promoted separately** into block
   12 (`WORKLOG.md` 261105.006) because it is a real defect on its own — a queued HIP stream wait binds to the
   *latest* record, so a copy in flight re-binds consumers of copies still queued.  The artifact now uses
   `ggml_backend_sched_next_copy_event` / `cpy_tensor_async_nowait` from the delivery instead of defining them.
2. **CUDA portability.**  The PR spelled `hipEventDefault` / `hipEventElapsedTime` unconditionally;
   `ggml-cuda.cu` still builds under nvcc for CUDA and those have no hipify alias.  They now follow this file's
   own `#ifdef GGML_USE_HIP … #else cudaEventDefault … #endif` form.
3. **No cost on the default path.**  Events are timing-enabled only when `GGML_SCHED_PAIR_DEBUG>=3`.
4. **Docs corrected** to state the reachability table rather than implying `-sm layer -ts 1,1` works.

**r229 (this re-cut, PR #279):**

5. **The split is gated exactly like the pair** — `!cparams.embeddings_nextn && !cparams.lf_layer_inp_dev`, the
   pair gate's own conditions.  Without it a draft workload declined the pair but was still split, i.e. it paid
   a narrower ubatch and gained nothing: 35B-A3B Q8_0 + DFlash2 pp2048 measured **3981.8 → 3427.4 t/s
   (−13.9 %)** and its greedy text diverged.  With the gates agreeing, text is identical and the residual is the
   ~4 % arming cost above.
6. **Restored the delivery comment line** that the r220 cut dropped in `src/llama-context.h`
   (`// one scheduler each switch re-splits and re-allocates the graph. …`), which had left the MTP sched2 note
   reading "… every cycle; with / scheduler, so both stay allocated".  The artifact no longer deletes delivery
   text it does not mean to.

## Known open points

* On the pair-failure path (`graph_compute_pair != SUCCESS`) lane 1's ubatch has already been applied to the
  memory module by `prepare_ubatch`, and the loop then processes it again in the single-lane path.  Error path
  only; a double `mctx->apply()` for the same positions is not obviously idempotent for every memory type.
* `include/llama.h` gains `llama_n_prefill_lanes()`, a public accessor for an internal feature.
* The ~8 % second-context penalty and the −3.5 % 8-sequence regression above are observed, not explained.

## Env knobs (see `ENVIRONMENT.md` for the class of each)

| variable | default | what |
|---|---|---|
| `LLAMA_PREFILL_LANES` | `1` | `2` enables the pair.  Anything else (or a layout the table above refuses) logs one WARN and stays single-lane |
| `LLAMA_LANES_SPLIT_MIN` | `1024` | minimum batch size (tokens) for the sub-ubatch split; `0` = whole-ubatch pairing only (the r220 behaviour) |
| `LLAMA_LANES_SPLIT_FRAC` | `0.55` | fraction the first sub-ubatch takes (clamped to [0.1, 0.9], rounded up to 64) |
| `GGML_SCHED_PAIR_DEBUG` | `0` | `1`: log each pair decision; `>= 2`: also run the hazard self-test; `>= 3`: per-lane/per-backend GPU event spans (creates timing events) |
| `GGML_SCHED_PAIR_HAZARD` | on | `0` keeps only the split-name guard, skipping the cache-hazard analysis (diagnostic) |

## Why this is not a block

Promotion needs a **validated win on one of the delivery's own configurations**, and this cut now has one
(gfx1201, +27 %/+41 % prefill in the `-ot` layout) — but it also has three open regressions (a draft context
armed and declined, a second context, 8-sequence batched prefill) and it is **inert in both splits the delivery
actually validates** (`-sm tensor` is one backend; plain `-sm layer` has pipeline parallelism on).  Reaching it
costs a second compute-buffer set and a hand-written `-ot` override.  So it stays an artifact: a consumer with
an `-ot` layout gets a large prefill win for a knob and a documented set of costs.

It becomes a promotion candidate when it reaches `-sm tensor` (pairing inside the Meta device, or a tensor-split
hazard analysis), when plain `-sm layer` can use it (the pair needs to coexist with, or replace, the scheduler's
pipeline parallelism), or when the three regressions above are gone.
