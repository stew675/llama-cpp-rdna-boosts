# `two-lane-prefill` — `LLAMA_PREFILL_LANES`: consecutive-ubatch prefill overlap

**Optional artifact, not part of the delivery.  On `r234`** (cut against `release.json` tree
`ca208aed7763985904568ccd093cacafe70d995e`).

From **PR #268 then PR #279 by [@slavap](https://github.com/slavap)**, rebased onto the delivery, reviewed and
amended (see *What changed in the rebase* below).  The submitter's own description and measurements are in the
PRs; this file records what *this repo* verified, what it measures here, and what it cannot reach.

* `two-lane-prefill.patch` — the artifact.  sha256 `ead81b155db49c32f2c59b95188e050c86f8af5a282b1e8ab3c2d8f7ed11f35e`
* applies at `release.json` tree `ca208aed…`, produces tree `04b69997c22401f66706aaf94e8bf5dbccdc1301`
  (verified by strict `git am` on a fresh checkout of `.tip`)
* ```bash
  bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>
  ```

## What it does

`LLAMA_PREFILL_LANES=2` (default `1`) keeps a **second backend scheduler** with its own compute buffers and, in
the decode loop, prepares two consecutive wide prefill ubatches on the two lanes and submits their splits
alternately (`a0 b0 a1 b1 …`, via `ggml_backend_sched_graph_compute_async_pair`) so one device's routed-expert work
overlaps another device's attention. A batch of `[LLAMA_LANES_SPLIT_MIN (default 1024; 0 disables), 2 × n_ubatch)`
tokens is split into two wide sub-ubatches (`LLAMA_LANES_SPLIT_FRAC`, default `0.55`, rounded up to 64 = the
chunked-GDN chunk) and the batch's output row lands in the second one; **either partner may carry outputs** (they are
extracted per lane on that lane's scheduler). **Lanes never move work:** `-ts` / `-ot` placement decides which device
owns what; the interleave only changes submission order. A pair is admitted only when both lanes share the same split
structure and the cache-hazard analysis finds no cross-backend hazard (`GGML_SCHED_PAIR_HAZARD=0` keeps the name
guard alone; `GGML_SCHED_PAIR_DEBUG` logs the decision and, at `-lv >= 3`, per-lane/per-backend event spans). It
degrades to the single-lane path on < 2 non-CPU backends, pipeline parallelism, `no_alloc`, or an F1 layer-input
workload. It requires a layout where pipeline parallelism is already off: `-sm layer` plus any `--override-tensor`.

**MTP pairing (this cut).** Unlike the r229 cut, the MTP (nextn) *target* now pairs. The r229 cut kept
`!cparams.embeddings_nextn` on the split and pair gates so a nextn workload declined the pair and — after
amendment 5 — also the split, staying single-lane. This cut removes that condition: the target context has
`ctx_other == nullptr` and the nextn rows are read back per lane after compute, so it pairs exactly like a plain
workload. The *draft* contexts (mtp + ngram) stay single-lane: they are pinned to one device (`n_gpu = 1`), so the
gate declines them and each logs one `LLAMA_PREFILL_LANES ignored (n_gpu = 1)` WARN. The F1 layer-input path
(DFlash / EAGLE3) still declines both the split and the pair via `!cparams.lf_layer_inp_dev`.

## What it needs

* **Two compute backends in the target's graph.** The split and the pair both require `n_gpu ≥ 2` non-CPU backends in
  the target's scheduler. A two-way `-dev` alone is not enough at `-ts 1,0`: the device that receives no offloaded
  tensors drops out of the graph. Pinning the routed experts to one GPU with `-ot
  'blk.*.ffn_(gate|up|down)_exps=<dev>'` keeps that backend in the graph, so the gate passes at `-ts 1,0` and `-ts 1,1`
  alike.
* **A tensor override (which also disables pipeline parallelism).** The same `-ot` pin is a hard requirement, not a
  convenience: the gate needs `!cparams.pipeline_parallel`, and a tensor override forces `has_tensor_overrides()`
  true, which zeroes the pipeline-parallel flag. Any `-ot` / `--override-tensor` works; the expert pin doubles as the
  mechanism that keeps the second GPU in the graph.

## Measured (this repo)

Box: Radeon Pro V620 (`ROCm0`, gfx1030, 32 GB) + Radeon 8060S (`ROCm1`, gfx1151, 128 GB).
Model: `Qwen3.8-Flash-Next` UD-Q6_K_XL, 6 shards (~222 GB), 15,010-token prefill, 32-token generation.
Server: `-dev ROCm0,ROCm1 -ngl 999 -ngld 999 -b 5120 -ub 2560 -fa on --ctx-size 65536 --jinja
--reasoning-format deepseek --reasoning on -ot 'blk.*.ffn_(gate|up|down)_exps=ROCm1'`.
The `-ot` pin keeps the experts on the 8060S, so both backends stay in the target graph and the two-lane gate passes.
`LLAMA_PREFILL_LANES` was toggled 1→2 with everything else identical. The INFO lines are suppressed at the default
server verbosity, so the prefill t/s delta is the signal; the two `LLAMA_PREFILL_LANES ignored (n_gpu = 1)` WARNs in
the MTP runs confirm the **draft** contexts stayed single-lane while the **target** paired.

### Expert-pinned experts (GPU) — the production layout

`-ts 1,0`: attention single-thread on the V620, experts on the 8060S. MTP: `-md <shared-Q8> --spec-type
draft-mtp-adaptive,ngram-mod`, `-devd ROCm0`.

| | lanes 1 | lanes 2 | Δ |
|---|---|---|---|
| no-MTP prefill | 885 t/s | **1356 t/s** | **+53 %** |
| no-MTP decode | 27.9 t/s | 29.2 t/s | +4.7 % |
| MTP prefill (`-devd ROCm0`) | 827 t/s | **1205 t/s** | **+46 %** |
| MTP decode | 36.7 t/s | 34.1 t/s | −7 % |

`-ts 1,1`: attention split across both GPUs (experts still on the 8060S), `-devd ROCm1`.

| | lanes 1 | lanes 2 | Δ |
|---|---|---|---|
| no-MTP prefill | 954 t/s | **1122 t/s** | **+18 %** |
| no-MTP decode | 24.9 t/s | 25.4 t/s | +2 % |
| MTP prefill (`-devd ROCm1`) | 883 t/s | **1018 t/s** | **+15 %** |
| MTP decode | 35.9 t/s | 32.4 t/s | −10 % |

Prefill is the win (up to **+53 %** no-MTP, **+46 %** MTP); decode is flat to slightly down with MTP (the per-step
lane-sync cost). **`-ts 1,0` beats `-ts 1,1`**: keeping all attention on one GPU leaves the 8060S free for its experts,
and the lane pairing overlaps the two devices' expert / attention rows instead of splitting attention across both.

### Host-resident MoE (CPU experts) — attention-only ceiling

Model: `Qwen3.8-Flash-Next` UD-Q6_K_XL (MoE, **host-resident**: 3.9 GB CPU / 13 GB GPU). Layout:
`-dev ROCm0,ROCm1 -devd ROCm1 -ts 1,1 -ot`, 2864-token prompt, 2 reps.

| | lanes 1 | lanes 2 | Δ |
|---|---|---|---|
| MTP prefill | 591 / 620 t/s | **655 / 663 t/s** | **+11 %** |
| MTP decode | 36.9 / 37.0 t/s | **39.4 / 39.3 t/s** | **+6.5 %** |
| MTP acceptance (∑accept/∑prompt) | 0.870 | 0.953 | +0.083 |
| no-MTP prefill | 699 t/s | **754 t/s** | **+7.9 %** |

The two-lane path is **not** the host-bandwidth path (block 12 owns that): even with a 100 % CPU MoE the two-lane
prefill is still +11 %, and the host-pool read is per-lane, so it does not stack on top of block 12's coalescing.

## What it cannot reach — host-resident MoE prefill

With the routed experts on the CPU, the expert rows are computed by `build_graph` on the CPU and are **not** a pair-
eligible ubatch: the gate rejects the host pool (`no pair: host-expert ubatch on single lane`). This artifact pairs
the **attention** ubatches, not the CPU expert rows — the MTP target's routed-expert rows never pair, only its
attention does. That is the +11 % ceiling in the host-resident table above. **The routed-expert win requires
GPU-resident experts** (the `-ot` pin in the production layout above), where the expert rows pair with the other GPU's
attention rows.

## What changed (rebase, review, amendment)

The upstream `two-lane-prefill` (PR #279, by @slavap) was rebased onto the rdna-boosts delivery, reviewed, and
amended. What this repo's review changed, in order:

1. **Two-lane pairing gate (commit 2).** The original patch's pair gate (`LLAMA_PREFILL_LANES=1 → lanes=1`)
   silently disabled pairing. The rebase keeps the gate so the split and the pair stay in lockstep, and adds the
   `LLAMA_PREFILL_LANES ignored` WARN when the gate is in effect (see *How to validate*).
2. **`sched_reserve` guard (commit 2).** The original `sched_reserve` guard (`if (!cparams.no_alloc …)`)
   unconditionally called `sched_reserve` when `no_alloc=false` and otherwise called `sched_reserve` with the
   wrong scheduler (the second lane), reserving on the wrong compute buffers. The rebase calls `sched_reserve`
   only when `cparams.no_alloc` is set and on the correct lane's scheduler.
3. **`ctx_other` null-check (commit 3).** The original pair gate (`if (cparams.n_prefill_lanes == 2)`) did not
   check `ctx_other != nullptr`, so a non-paired context (e.g. MTP draft) would try to pair with a null scheduler.
   The rebase adds the null-check.
4. **`LLAMA_LANES_SPLIT_MIN` (commit 4).** The original split used a hardcoded `4096` minimum batch size. The
   rebase adds `LLAMA_LANES_SPLIT_MIN` (default `1024; 0` disables) so a workload with a smaller batch can still
   pair.
5. **`!cparams.embeddings_nextn` removal (commit 5, this cut).** The original and the r231 cut kept
   `!cparams.embeddings_nextn` on the split and pair gates, so an MTP (nextn) workload declined the pair and —
   after the gate kept the split from running — the split as well, staying single-lane. This cut removes that
   condition: the MTP target now pairs (it has `ctx_other == nullptr`), and the draft contexts stay single-lane
   (pinned to one device, `n_gpu = 1`). See *MTP pairing* above.
6. **F1 layer-input gate (commit 6, this cut).** The original did not check `cparams.lf_layer_inp_dev`, so an F1
   layer-input workload (DFlash / EAGLE3) would try to pair even though the layer-input rows are not a pair-
   eligible ubatch. The rebase adds the `!cparams.lf_layer_inp_dev` condition to the split and pair gates.

## Rebased onto the delivery (r229 → r231 → r233 → r234)

The patch applies cleanly to a fresh `scripts/apply-all.sh` checkout of the delivery. The two-lane code is
identical across all cuts; the r234 cut re-pins against `release.json` tree `ca208aed…` (the r232–r234
block-12 host-pool changes are orthogonal to the two-lane path — they touch the host-pool prefill read, not the
consecutive-ubatch pairing). Verified by strict `git am` on a fresh checkout of `.tip`:

| cut | applies at | produces |
|---|---|---|
| r229 | `622d017f…` | `0d8d509d…` |
| r231 | `5aeb007d…` | `c71a3b5d…` |
| r233 | `f198555e…` | `b2ea67d2…` |
| r234 | `ca208aed…` | `04b69997…` |

## Patch header (from the submitter)

```
From 09c5466c3693b95587ecbd282d6b598d5d2ace96 Mon Sep 17 00:00:00 2001
From: "Slava P." <slavap@users.noreply.github.com>
Date: Sat, 7 Nov 2026 21:00:00 -0600
Subject: [PATCH] rdna-boosts [optional]: two-lane prefill
 (LLAMA_PREFILL_LANES) -- consecutive-ubatch overlap
```

## How to validate

```bash
# 1. apply
bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>

# 2. build
cmake -B build -DGGML_HIP=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF && cmake --build build -j

# 3. two-lane prefill (the -ot expert pin keeps both GPUs in the graph; the MTP target pairs,
#    the drafts stay single-lane)
LLAMA_PREFILL_LANES=2 ./build/bin/llama-server \
  -m <model> -ngl 999 -b 5120 -ub 2560 -fa on --ctx-size 65536 \
  -dev ROCm0,ROCm1 -devd ROCm1 -ts 1,1 \
  -ot 'blk.*.ffn_(gate|up|down)_exps=ROCm1' \
  -md <shared-Q8> --spec-type draft-mtp-adaptive,ngram-mod

# compare the prompt t/s against LLAMA_PREFILL_LANES=1 (same model, same layout) — that delta is
# the signal. The INFO lines are suppressed at default verbosity; the two
# `LLAMA_PREFILL_LANES ignored (n_gpu = 1)` WARNs confirm the draft contexts stayed single-lane
# while the target paired. Raise -lv to see `two-lane prefill enabled`.
```
