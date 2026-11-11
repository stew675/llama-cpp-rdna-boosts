# `two-lane-prefill` — `LLAMA_PREFILL_LANES`: consecutive-ubatch prefill overlap

**Optional artifact, not part of the delivery.  Re-cut on `r259`** (against `release.json` tree
`7995db440ff79d805b4e1ad946ebfae605a56616`; the r254, r247, r246, r243, r241 and r229 cuts and their measurements
are kept below as the dated record).  The r259 rebase is **no longer purely mechanical**: the artifact's own code
is unchanged (its +/− line set is identical across the r247, r254 and r259 cuts), but a **host-expert layout
(`-ncmoe` + `-ot`) aborted in the pair path** — pre-existing since the r246 cut, never seen because every arm
below keeps the experts device-resident — and that is fixed in this cut (`What changed in the rebase`, item 13),
and that tier then measures as a *loss* at a wide ubatch (item 14).

From **PR #268, #279, #284, #296 then #301 by [@slavap](https://github.com/slavap)**, rebased onto the delivery,
reviewed and amended (see *What changed in the rebase* below).  The submitter's own description and measurements
are in the PRs; this file records what *this repo* verified, what it measures here, and what it cannot reach.

* `two-lane-prefill.patch` — the artifact.  sha256 `64f92b608728d48005d38c1562e089d6f67c434393219efb2e83beb6bf7572a9`
* applies at `release.json` tree `7995db44…` (r259), produces tree `d1c3f0f9c370efc7707caab10ebc6c67c03bfe9a`
  (verified by strict `git am` on a fresh checkout of the r259 `rdna-boosts` branch; the r254 cut produced
  `896446c422…` on `3f2cad2a…`, the r247 cut `17f51fb30c…` on `8f99d2dc6d…`, and with the
  `host-pool-fill-src` artifact applied first — `apply-all.sh`'s order — this cut produces `e6e213c6ce…`)
* ```bash
  bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>
  # --include-optional steps through the artifacts and asks [y/N] for each one
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
   readback sync).  F1 layer-input readback stays single-lane.

Since the **r246 cut** an **MTP (nextn) target pairs too**: its prefill runs on 0-output nextn ubatches, and an
unmasked nextn row is per token, so the host offsets the lane split changes are the ubatch's own.  A **draft
context never arms** — `ctx_other` for Gemma4-assistant / EAGLE3 / DFlash / qwen4exp, and `ctx_type == MTP` for
an MTP draft, which the fork does *not* mark with `ctx_other` (it is the target's own model).

It degrades to the existing single-lane path when `< 2` non-CPU **backends**, pipeline parallelism, `no_alloc`,
or a draft context is present, and a workload that reads **F1** layer-input rows declines the pair **and also
declines the split** (see amendment 11).

A pair is admitted only when both lanes have the same split structure and the cache-hazard analysis finds no
cross-backend hazard (`GGML_SCHED_PAIR_HAZARD=0` keeps the name guard alone; `GGML_SCHED_PAIR_DEBUG` logs the
decision and, at `>= 3`, per-lane/per-backend GPU event spans).

## Which layouts it reaches — and which it cannot

This matters more than the knob does, so it is measured, not assumed (3× R9700, gfx1201):

| layout | result |
|---|---|
| `-sm tensor` (any device count) | **never engages.**  This tree's tensor split is ONE backend (a Meta device wrapping N GPUs) so the `n_gpu >= 2` gate sees 1: `sched_reserve: LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0, draft = 0)` |
| plain `-sm layer`, all layers on GPU | **never engages.**  The scheduler's own pipeline parallelism is enabled in exactly that configuration: `ignored (n_gpu = 3, pipeline_parallel = 1, draft = 0)` |
| `-sm layer` + any `--override-tensor` (`-ot`) | **engages** — a tensor override disables pipeline parallelism (`pair decided: splits a=4 b=4, interleaved=1`).  **This is the only reachable layout on ROCm.** |
| any layout, with a draft (`-md draft-dflash` / `-eagle3`, or `--spec-type draft-mtp`) | **never engages** — the draft context declines at the arm gate: `ignored (n_gpu = 3, pipeline_parallel = 0, draft = 1)`, and allocates no second buffer set |
| `-sm row` | **not usable at all on ROCm**: `llama_model_load: device ROCm0 does not support split buffers` (the r220 cut listed this as a reachable layout — it is an inference, not a measurement, and it is wrong for this backend) |
| `-sm none` | one GPU backend, i.e. the `-sm tensor` case |

The **expert tier is orthogonal to reachability**: `-ncmoe` / `--host-experts` changes nothing about which
layouts arm the pair (`-sm layer -ot` still engages, and pairs), it changes whether the pair is *worth it* — on
host-resident experts it wins at the default ubatch shape and loses at a wide one (*What it measures here*).

The submitter's **dense** reference layout (`-sm layer -ts 1,0`, no `-ot`) is refused here for the same reason,
so the +13.3 % PR #268 reported for it was not measured with this patch on this base.

## What it measures here

`-sm layer -ts 1/1/1 -ot 'blk\.0\.attn_q\.weight=ROCm0'` (a placement-neutral override: it only turns pipeline
parallelism off) and `-ub 2048 -b 2048` throughout; the `llama-bench` arms add `-t 16 -r 3`, the `llama-cli` arms
run the tool's default thread count (one arm per line, A B B A order).  Same binary, env-gated.

### The r259 cut (PR #301) — re-measured here, and the host-expert tier

3× R9700 gfx1201, the same layout, same binary, env-gated, A B B A, `prompts/prose-rdna-boosts.txt` for the
`llama-cli` arms.  Two ubatch shapes: the delivery's `-b 2048 -ub 2048`, and the **tool defaults**
(`-ub 512 -b 2048`) — what a consumer who tunes nothing actually gets.

| arm | 1 lane | 2 lanes | |
|---|---|---|---|
| MoE 35B-A3B Q8_0, `llama-bench pp2048 -b 2048 -ub 2048` | 6544.9 / 6524.6 | **8070.2 / 8056.0** | **+23.4 %** |
| the same, default `-ub 512` (whole 512-token ubatches pair) | 4077.5 / 4051.0 | **6001.6 / 6033.7** | **+48.1 %** |
| the same, default `-ub 512`, `-ncmoe 40` (host-resident experts) | 3917.6 / 3930.8 | 4126.2 / 4157.1 | +5.6 % |
| the same, `-b 2048 -ub 2048`, `-ncmoe 40` | 6276.7 / 6276.7 | 5544.4 / 5530.4 | **−11.8 %** |
| 27B `GSQ-RCO-IQ3_S-mtp`, `--spec-type draft-mtp --spec-draft-n-max 3` — prefill | 954.4 / 963.5 | **1239.4 / 1238.1** | **+29.2 %** |
| the same — decode (`-n 32`) | 56.7 / 55.9 | 56.8 / 56.5 | nil |
| the same model, `--spec-type none` — prefill | 1061.6 / 1053.9 | **1295.2 / 1293.1** | **+22.3 %** |

**Text and the default path.**  Greedy `-n 32 --temp 0 --seed 42` text is `e2dde701fc04` (35B-A3B Q8_0) for
lanes=1 == lanes=2 == lanes unset, on `-sm tensor` and on the `-ot` layout, and with and without `-ncmoe 40` —
the delivery's own anchor for that model; the 27B arms read `a4a403d64c40` both ways and under both speculator
settings (the r246 anchor).  `scripts/gate-prefill-logits.sh` on the **default path** (lanes unset, 27B Q8_0)
reads mean KLD **0.000000 / same-top-p 100.000 %** against the base recorded from the r259 build, and
`check-gpu-faults.sh` is clean over every arm.

**The host-expert tier is where this knob loses.**  With `-ncmoe 40` the pair still engages (3 pairs logged,
`llama-cli -ub 2048 -b 2048`: prefill 5135.4 -> 4218.4 t/s, −17.9 %), but that tier is link-bound: the delivery's
H2D staging already overlaps the host→device traffic, a second compute-buffer set takes VRAM the expert-cache
arena wants, and the two lanes contend for the same link.  At the default `-ub 512` shape it still gains, because
there the second lane overlaps *layer* work instead of waiting on the link.  **On the host-expert tier leave
`LLAMA_PREFILL_LANES` unset** at a wide ubatch.

### The r246 cut (PR #296) — the MTP target pairs

3× R9700 gfx1201, `prompts/prose-rdna-boosts.txt` (5 298 tokens), `-ub 2048 -b 2048`, A B B A:

| arm | 1 lane | 2 lanes | |
|---|---|---|---|
| dense 27B `GSQ-RCO-IQ3_S-mtp`, `--spec-type draft-mtp --spec-draft-n-max 3` — prefill | 1047.7 / 1044.3 | **1361.1 / 1351.1** | **+29.9 %** |
| the same, decode (`-n 32`) | 63.4 | 63.2 | nil |
| the same model, `--spec-type none` — prefill | 1086.5 | **1436.3** | **+32.2 %** |
| MoE 35B-A3B Q8_0, `llama-bench pp2048` (`-t 16 -r 3`) | 6695.8 / 6751.0 | **8288.2 / 8322.9** | **+23.8 %** |
| MoE 35B-A3B Q8_0 + DFlash2 draft (`--spec-type draft-dflash`) — prefill | 5308 / 5346 / 5340 / 5300 | 5273 / 5287 / 5260 / 5280 | −0.9 % |
| `llama-batched-bench -npp 2048 -ntg 128 -npl 1,4,8` (PP/S) | 6214.6 / 8381.7 / 8822.9 | 7475.8 / 8258.2 / 8284.8 | +20.3 % / −1.5 % / **−6.1 %** |

`GGML_SCHED_PAIR_DEBUG=1`: `pair decided: splits a=4 b=4, interleaved=1`, once per batch — 5 298 tokens at
`-b 2048` is 3 batches, each split into 1152 + 896 (or 1152 + 50) and paired, so 3 pairs per prompt (a 15 k-token
prompt pairs several batches' worth).  The draft contexts log
`LLAMA_PREFILL_LANES ignored (n_gpu = 3, pipeline_parallel = 0, draft = 1)` (one WARN each — two draft
contexts) and allocate no lane-2 buffer set; the target logs `two-lane prefill enabled`.  An MTP draft has
`n_gpu = 2..3`, not 1: its backends are the model's whole device list, which is why the `ctx_type` predicate
is needed at all.

**The pair adds no approximation** (r246, 27B Q8_0, `-c 2048 --chunks 20 -ub 2048 -b 2048` + `-ot`,
`llama-perplexity --kl-divergence-base` recorded from the lanes-unset arm): lanes=2 reads mean KLD
**0.001394 / same-top-p 99.101 %**, and a plain `-ub 1152` arm (the same 1152 + 896 boundaries, run serially)
reads *exactly the same two numbers*.  So the divergence is the ubatch change, to every printed digit.

**The default path is bit-identical** to the r246 delivery build: `scripts/gate-prefill-logits.sh` mean KLD
**0.000697** / same-top-p **98.784 %** — the release's own reading — and `check-gpu-faults.sh` clean.

**Greedy text and MTP acceptance are unchanged**: `-n 32 --temp 0 --seed 42`, lanes=1 == lanes=2 ==
`--spec-type none`, sha `a4a403d64c40`; DFlash2 arm sha `e2dde701fc04` both ways; draft acceptance **0.62500
(20 / 32)** in both MTP arms (the submitter's arm shows 0.742 → 0.677 across the lanes — that arm leaves the
depth adaptive, so acceptance follows the speedup; a fixed `--spec-draft-n-max` is what makes the comparison
exact).

### The r243/r229 cuts (dated record): the gain needs both halves (MoE 35B-A3B Q8_0, pp2048)

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

* **The default path is bit-identical** (r246): `scripts/gate-prefill-logits.sh` mean KLD **0.000697** /
  same-top-p **98.784 %** against the base recorded from the r246 delivery build — the release's own reading.
* **The pair is bit-exact** (r246): lanes=2 vs lanes unset at the same ubatching measures **exactly** the
  divergence of a plain `-ub 1152` vs `-ub 2048` change on the same build — mean KLD `0.001394`, same-top-p
  `99.101 %`, identical to every printed digit (the r229 point read `0.004088 / 97.898 %`; this is the §46
  "benign reordering" scale — a smaller ubatch, nothing more).  So the interleave adds no approximation; the
  split's divergence *is* the ubatch change, which is why greedy text staying identical is luck of the greedy
  path and not identity.
* Same-seed greedy text (`-n 32 --temp 0 --seed 42`, the 5 298-token `prompts/prose-rdna-boosts.txt`): lanes=2 ==
  lanes unset == `--spec-type none`, sha `a4a403d64c40`; a DFlash2 draft arm sha `e2dde701fc04` both ways.  MTP
  draft acceptance identical, `0.62500 (20 / 32)` in both arms (r229 record: `f61214fad249` on the MoE and
  `8c1c642fc7cd` on the dense 27B).
* Perplexity (`--ctx 2048 --chunks 8 --ub 2048`, every token an output row, i.e. **both** lanes carry outputs)
  restores the split's exact single-lane value — `llama-perplexity` output identical to the `-ub 1152` arm (r229).

### What it costs (all measured, all documented rather than tuned away)

* **The host-expert tier loses at a wide ubatch** (r259): `-ncmoe 40`, `llama-bench pp2048 -b 2048 -ub 2048`,
  6276.7 -> 5544.4 t/s (**−11.8 %**); `llama-cli` on the 5 256-token prompt, 5135.4 -> 4218.4 (**−17.9 %**).  It
  gains **+5.6 %** at the default `-ub 512`, so the shape decides.  Before the r259 cut it did not regress — it
  aborted (`What changed in the rebase`, item 13).

* **A second compute-buffer set per armed context**: ~1.4–1.5 GiB at `-ub 2048` (the r229 record: `ROCm0 441 +
  ROCm1 430 + ROCm2 430 + host 65` MiB, logged once as `compute buffer size = … (prefill lane 2)`).  A **draft**
  context no longer pays it (r246): the two MTP draft contexts on the 27B arm show only their own 134 MiB device /
  20 MiB host set, no `(prefill lane 2)` lines.
* **A second context in the same process runs ~8 % slower with the lanes armed** (r229 observation, *not*
  re-measured on this cut; pp8192: 7976 → 7347 on the 2nd context, while lanes unset is flat at 7513–7519 across
  three contexts).  Unexplained; the extra buffer set is the obvious suspect.
* **Multi-sequence prefill regresses**, and more so than at r243: `llama-batched-bench -npp 2048 -ntg 128 -npl
  1,4,8` → PP/S_TG 6214.6→7475.8 (+20.3 %), 8381.7→8258.2 (−1.5 %), **8822.9→8284.8 (−6.1 %)**; TG
  92.28→92.56, 242.2→239.1, 320.7→320.9 (r243 record: +28.8 / +0.9 / −3.5 %).  Server workloads with many
  concurrent long prompts therefore do not want this.
* **A draft workload's target still pays the arming cost** even though nothing pairs — `embeddings_layer_inp_dev`
  is a *decode-time* gate, not an arm-gate one.  Measured on this cut (35B-A3B Q8_0 + DFlash2, 5 298-token
  prompt): **−0.9 %** prefill, text identical (the r229 arm measured ~4 % prefill / ~2 % decode on a
  `llama-bench pp2048` arm; the difference is the shape, not a change in the mechanism).  An MTP workload no
  longer pays it at all (the draft is refused at the arm gate).

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

**r229 (the PR #279 re-cut):**

5. **The split is gated exactly like the pair** — `!cparams.embeddings_nextn && !cparams.lf_layer_inp_dev`, the
   pair gate's own conditions.  Without it a draft workload declined the pair but was still split, i.e. it paid
   a narrower ubatch and gained nothing: 35B-A3B Q8_0 + DFlash2 pp2048 measured **3981.8 → 3427.4 t/s
   (−13.9 %)** and its greedy text diverged.  With the gates agreeing, text is identical and the residual is the
   ~4 % arming cost above.
6. **Restored the delivery comment line** that the r220 cut dropped in `src/llama-context.h`
   (`// one scheduler each switch re-splits and re-allocates the graph. …`), which had left the MTP sched2 note
   reading "… every cycle; with / scheduler, so both stay allocated".  The artifact no longer deletes delivery
   text it does not mean to.

**r241 (the re-cut after the delivery moved under it):**

7. **`ggml_backend_sched_compute_splits` became a per-split function.**  The delivery now has
   `ggml_backend_sched_compute_split(sched, split_id, st)` plus the loop wrapper (which is exactly the structure
   the pair path needs), and since r229 the loop body also gained `sched_moe_register_tables()` before it and
   `ggml_backend_sched_prefetch_inputs()` between the two input-copy passes.  The artifact's cut carries both
   delivery calls, so a pool's host tables are still registered before a graph's first split and the prefetch
   still runs between the non-weight and the weight copies.
8. **`extract_layer_inputs()` lost its `host_also` argument** (the delivery's in-graph layer-input device store
   removed the post-graph host fill).  The artifact now adds only `sched_x` — the scheduler that owns the graph,
   which lane 1 must pass or its readback would query lane 0's scheduler — and keeps the r241 body and semantics
   verbatim.
9. **`graph_reserve()` takes `split_only` where it used to take `no_alloc`**, and `cparams.lf_layer_inp_dev`
   became `cparams.embeddings_layer_inp_dev`.  Both are renames at the two places the artifact arms the lanes.
10. **Removed: the `lf_wait_features_consumed()` ordering call** the r229 cut put at the end of
    `prepare_ubatch()`.  The function does not exist on r241 (the in-graph store made it unnecessary), and
    `prepare_ubatch` keeps r241's tail unchanged.

**Spot check at r241** (3 × R9700 gfx1201, 35B-A3B Q8_0, `-sm layer -ts 1,1,1` plus
`-ot 'blk\.0\.attn_q\.weight=ROCm0'`, `-b 2048 -ub 2048`, the 5 256-token
`prompts/prose-rdna-boosts.txt` with `-n 0`, two interleaved reps): prefill **5844.7 / 5827.9 → 6638.2 /
6614.8 t/s (+13.5 %)**.  Same-seed greedy text unchanged (`4ac3a523160f`, the delivery's own anchor for that
model), and under `-sm tensor` the lanes are refused and the text is the dense anchor (`1c5d32ac537d`).  The
number is smaller than the r229 table's +27 % because that table is `llama-bench pp2048`, one wide ubatch split
into two halves; this arm is `llama-cli` feeding a 5 256-token prompt in `-b` chunks, where only the chunks that
fit the split window pair.

**r243 (the re-cut after the r241 delivery moved again):** no behavioural delta and no conflict resolution —
the artifact was re-cut only so its index lines name the r243 blobs instead of r241's (a stale blob line is what
lets a 3-way apply land a hunk in the wrong place; `WORKLOG.md` 261109.008).  Its numbers are the `+13.5 %` spot
check above, carried over.  `release.json` was not touched.

**r244-r245 (PR #284 reviewed, NOT taken):** PR #284 re-cut the artifact onto r234 and, on top of the re-base,
removed `!cparams.embeddings_nextn` from the split and pair gates so an **MTP target** pairs.  It was turned
down with `CHANGES_REQUESTED` (`WORKLOG.md` 261109.011): the delta measures (+14.7 %, 946.7/942.0 →
1085.1/1080.6 t/s on 3 × R9700) but it would also *pair the MTP draft*, which the artifact's arm gate let arm
— an MTP draft is the target's own model with `ctx_type = MTP`, so `ctx_other` is nullptr, `n_gpu` is the
model's whole device list, and the draft was allocating a second scheduler it could never pair on.  The
requested change was the explicit predicate `ctx_other != nullptr || ctx_type == MTP` plus a re-cut on the
current tree with the gate delta and the layout/hardware stated.

**r246 (this re-cut, PR #296):**

11. **The MTP target pairs; the MTP draft is refused at the arm gate** (`ctx_type == MTP` added).  The split and
    pair gates keep only `!embeddings_layer_inp_dev`, so they agree with each other and with what actually
    pairs.  Measured on this cut: target prefill **+29.9 %** (27B MTP, 5 298-token prompt), MoE **+23.8 %**,
    dense **+32.2 %**; MTP greedy text and draft acceptance unchanged; the pair measures the plain-`-ub 1152`
    KLD exactly (all in *What it measures here*).
12. **The draft predicate is one boolean and it is logged.**  The submitted cut left the two conditions inline
    and the WARN at `ignored (n_gpu = %d, pipeline_parallel = %d)`, so a refused MTP draft printed a line whose
    two conditions both pass — the reader cannot see that the new predicate fired.  It reports
    `… draft = %d` again, and the `ctx_other`-vs-`ctx_type` explanation sits above the gate where the next
    reader will look (`ggml_backend_sched_graph_compute_async_pair`'s WARN is the other half of that story).

**r259 (this re-cut, PR #301 plus this repo's amendment):**

13. **FIXED: the pair aborted on a host-expert layout.**  `LLAMA_PREFILL_LANES=2 -sm layer -ot … -ncmoe 40`
    died at the first prefill on `GGML_ASSERT(sched_a->is_alloc && sched_b->is_alloc)` in
    `ggml_backend_sched_graph_compute_async_pair`.  `prepare_ubatch()` grew the `lane` parameter in this
    artifact, but the delivery's two lane-0 placement blocks inside it — the ring narrow→wide drop
    (`ring_carved && ring_arm && n_tokens_prev < 64`) and the `drop_now` wide→narrow drop/re-reserve — still
    dropped and re-reserved `sched` (lane 0) unconditionally, so the lane-1 prepare ran them *again* after lane
    0's ubatch had been prepared and handed the pair an unallocated lane 0.  Both blocks are now `lane == 0`
    (they manipulate lane 0's scheduler and the context's lane-0 size expectations; a lane-1 ubatch must not
    re-place lane 0).  Pre-existing since the r246 cut — every arm that cut measured keeps the experts
    device-resident, so the ring was never carved and the blocks never fired.  After the guard the arm runs and
    its text is the single-lane text (`e2dde701fc04`).
14. **Base drift r254 → r259**, no code delta: r258 (block 09, the identity expert table served IN PLACE:
    `ggml-backend-impl.h` gains `moe_cache_in_place`, `ggml-backend.cpp`'s staging decision and take-over call
    change, `ggml_backend_buffer_get_usage()` answers for a buffer-less table) and r259 (block 01, the `q8ce`
    wire) moved the files the artifact touches.  The artifact's +/− line set is identical across the r247, r254
    and r259 cuts (checked per file on sorted line multisets), so the arm gate, the MTP-target pairing, the draft
    refusal and the scheduler refactor are unchanged.  Note the r254 cut's own delta note attributed the
    `GGML_SCHED_STAGE_MODE` retirement to r254: it was **r250** (block 09, `TODO` #284) — the resolution was right,
    the attribution was not.
15. **Dropped a stray blank line** the artifact had been adding to `ggml/include/ggml-cuda.h` since the r220 cut
    (its only change to that file).  That file is untouched now.

## Known open points

* On the pair-failure path (`graph_compute_pair != SUCCESS`) lane 1's ubatch has already been applied to the
  memory module by `prepare_ubatch`, and the loop then processes it again in the single-lane path.  Error path
  only; a double `mctx->apply()` for the same positions is not obviously idempotent for every memory type.
* `include/llama.h` gains `llama_n_prefill_lanes()`, a public accessor for an internal feature.
* The ~8 % second-context penalty and the −6.1 % 8-sequence regression above are observed, not explained.
* The host-expert wide-ubatch loss (−11.8 % / −17.9 %, r259) is measured and explained as link-bound contention,
  not tuned away; it is documented rather than gated, because the layout that gains is the one the artifact is
  for.
* The delivery's own draft detection (`graph_reserve()`'s narrow-2 route, `process_ubatch()`'s `is_draft`, the
  narrow-2 floor re-pin in `sched_reserve()`) still tests `ctx_other` alone, so none of it fires for a
  qwen3-family MTP draft.  That is pre-existing delivery behaviour (`WORKLOG.md` 261109.011 item 3), not this
  artifact's; the artifact only had to stop letting the draft arm.

## Env knobs (see `ENVIRONMENT.md` for the class of each)

| variable | default | what |
|---|---|---|
| `LLAMA_PREFILL_LANES` | `1` | `2` enables the pair.  Anything else (or a layout the table above refuses, or a draft context) logs one WARN and stays single-lane |
| `LLAMA_LANES_SPLIT_MIN` | `1024` | minimum batch size (tokens) for the sub-ubatch split; `0` = whole-ubatch pairing only (the r220 behaviour) |
| `LLAMA_LANES_SPLIT_FRAC` | `0.55` | fraction the first sub-ubatch takes (clamped to [0.1, 0.9], rounded up to 64) |
| `GGML_SCHED_PAIR_DEBUG` | `0` | `1`: log each pair decision; `>= 2`: also run the hazard self-test; `>= 3`: per-lane/per-backend GPU event spans (creates timing events) |
| `GGML_SCHED_PAIR_HAZARD` | on | `0` keeps only the split-name guard, skipping the cache-hazard analysis (diagnostic) |

## Why this is not a block

Promotion needs a **validated win on one of the delivery's own configurations**, and this cut now has several
(gfx1201, +23.4 % MoE `pp2048`, +48.1 % at the default ubatch shape, +22.3 % dense prefill, **+29.2 % on an MTP
target**, all in the `-ot` layout) — but it still carries open regressions (a second context, −6.1 % at 8
sequences, −0.9 % on a layer-input draft's target, **−11.8 % on the host-expert tier at a wide ubatch**) and it
is **inert in both splits the delivery actually validates** (`-sm tensor` is one backend; plain `-sm layer` has
pipeline parallelism on).  Reaching it costs a second compute-buffer set and a hand-written `-ot` override, and
it reaches a *draft* workload only through its target.  So it stays an artifact: a consumer with an `-ot` layout
and device-resident experts gets a large prefill win for a knob and a documented set of costs; a consumer on the
host-expert tier should not set it.

It becomes a promotion candidate when it reaches `-sm tensor` (pairing inside the Meta device, or a tensor-split
hazard analysis), when plain `-sm layer` can use it (the pair needs to coexist with, or replace, the scheduler's
pipeline parallelism), or when the regressions above are gone.
