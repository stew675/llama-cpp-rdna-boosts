# Flash-Next prefill — why do other engines reach 2900-3500 t/s where we reach 2100-2600?

**Status: PROMOTED at r369 (2026-11-27) and ARCHIVED here.**  The staging transport ships as a cross-cutting
layer over blocks **09/10/12/15** (each block's `patches/README.md` note says exactly which hunks landed there and
why), `WORKLOG.md` **261127.018** is the promotion entry, and `INVARIANTS.md` **r369** carries the facts —
including item 8, the attribution rule the fold was run by (a hunk goes to the block whose tree still holds its
preimage, and a positional-initializer row list belongs to the LAST block that writes it).  The record is kept
whole: the campaign's own numbers, reviews, handovers and harnesses are below, unchanged except that the
harnesses' `../../` paths are now `../../../` (this directory is one level deeper than `wip/` was).  Follow-ups:
`TODO` #385-#388; the tensor split's fill-ahead/in-place work is `wip/tensor-split-stage-overlap/`.

**The status it was promoted from:** a measured rewrite in the prototype; nothing in `patches/`, no `release.json`
change, no fork push.  The campaign is **squashed to one commit** (`SQUASHED-campaign-r368.patch`) and **code-reviewed**
([`REVIEW-squashed.md`](REVIEW-squashed.md), all findings fixed), which is the form it goes to promotion in.

**A session starts at [`HANDOVER-session7.md`](HANDOVER-session7.md) — and session 7 has DONE it: the final
code-review pass (seven defects, all fixed in the same single commit, tip `79d48465e`, patch 5119 lines), the §4
tests re-run ([`RESULTS-session7.md`](RESULTS-session7.md)), and then a STOP before the promotion ladder.**  What is
left is the maintainer's: their own user-side check, the **block decision** (selector + the re-keyed zero-carve rule
into block 09; pipeline / transport region / cushion / depth authority and the VMM pool's fail-soft growth into block
10), and the `REVIEW-squashed.md` §9 design question (delete the Meta backend's staged transport, or finish it).
Nothing is cut, renumbered or folded without that answer, and the gate ladder has not been started.

**Two measurement rules this campaign paid for (2026-11-27)**: a **text/coherence arm takes NO `-lv`** -- and a
transport/census arm is a separate `-lv 4` run -- because `extract-generated.py` does not strip log lines, so a `-lv 4`
"text" hash is polluted by the interleaved timestamped lines (it faked a staged-vs-gather divergence); and **judge the
tail** of the generated text, not just the head, because a coherence break often appears half-way through
(`./tailcheck.sh LOG 200`).  With both applied, the tensor-split staged transport and the gather are **byte-identical**
(321 chars `sha=184dc307181b` at 40 / 80 / 100 % coverage).

**The previous hand-over's order was** (and sessions 6-7 followed it): a fresh code review with the no-tombstone
rule (no dead-code remnants in the source, `ENVIRONMENT.md` or `DIAGNOSTIC.md`), the quick tests, then **STOP** for
the maintainer's user-level test — promotion (which begins with the block decision) only on their explicit
all-clear.

The history behind it: [`RESULTS-session5.md`](RESULTS-session5.md) (the `transport_base` cushion, the buffer
count from the arena's residency, and the re-base onto **r368**), [`RESULTS-session4.md`](RESULTS-session4.md)
(the reporting audit, the `-sm tensor` repair, and the transport selector that makes the staged pipeline the
default at prefill width: 1 GPU 2108 → 2445 t/s, +16 %), and [`RESULTS-session3.md`](RESULTS-session3.md) /
[`RESULTS-session2.md`](RESULTS-session2.md) before them.  §1-§8 below are the *scoping* record of session 1
(the competitor teardown and the hypothesis register); later sessions superseded its conclusions.

The question the campaign started from is not "can we win a few percent"; it is *what about our prefill
architecture caps us ~35 % below engines that stream the same 40 GiB expert set over a slower link than ours*.
Read §1-§3 before touching code; §4 is the hypothesis register and §5 the tools to attack it. §8 lists what is
already excluded so you do not re-measure it.

## 0. The question, the box, and the rules

**The question.** On `Qwen3.8-Flash-Next` (qwen4exp, 48 layers, 512 experts, 10 used, a 26.8 GiB PLE, ~76 GB
IQ3_XXS) with the experts in the host tier, our prefill reads **2108 t/s** on a Gen5 x8 card, **2262** on a Gen5
x16 card and **1712** on a Gen5-x4 riser, for an 89 182-token prompt at `-b 8192 -ub 8192`. Other engines and
other boxes read **2950-3500+** for the same prompt. Two independent facts say that is not link physics:

* `@DanoPTT`'s R9700 sits on **PCIe 3.0 x16 (12.5 GB/s, 4.4x below our x16)** and his *good* state prefills the
  same prompt at **2921-2953 t/s**;
* **Strata** (Niko1221/Strata, on this same box, same GGUF) is at **~3500 t/s** prefill in its current release.

**The definition of done.** Either (a) a measured, gated change that moves our number materially (the first
target is to beat 2900 on the x16 card, i.e. +28 %), or (b) a written, measured reason why our architecture
cannot reach it and what it would cost to change. A negative that is *understood* is a valid outcome; a
"tried things, no win" is not.

**The box.** `radix` = 1 x R9700 on Gen5 x16 (55.0 GB/s H2D), 93 GiB. This box = 3 x R9700, GPU0 on Gen5 x8
(28.7 GB/s), GPU1 on an M.2->Gen5 x4 riser (14.4 GB/s), GPU2 on an M.2 riser at x4; 91 GiB, 9950X3D. Model:
`/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (both boxes
have it). Prompt: `l1-90k.txt`, **generated by the harness** (not committed) as `prompts/prose-rdna-boosts.txt` x17 —
89 182 tokens by the server, 89 199 by the cli.

**The flags** are `@DanoPTT`'s from GitHub issue #368 and they are the shape to measure:
`--fit off -ngl all --n-cpu-moe 47 --host-experts pool --load-mode auto --lazy-mode on -c 131072 -fa on
-ctk q8_0 -ctv q8_0 -b 8192 -ub 8192 -t 8 --cache-ram 4096` with
`MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=85 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_GATHER_NSPLIT=4
ROCBLAS_USE_HIPBLASLT=1`.  `-np 2 --kv-unified` are server-only and are in `visit.sh`.

**Rules.** This is `wip/`: nothing here is applied to the fork or folded into `patches/` without the
maintainer's go-ahead, and a new block is always asked first (`AGENTS.md`, WIP and promotion rules). Every
candidate change needs the delivery's own purity gates — same-seed greedy text **and** the prefill-logit KLD
point for any prefill-kernel or approximate-path change, the MTP four-axis protocol for anything touching
decode, and `validate-set.sh` for the set itself.

## 1. What the competitors are actually doing (read these before hypothesising)

**Strata**, on this box, is the closest thing to an answer we can read. Its own configuration
(`~/Strata/strata-iq3_xxs.json`) and docs (`~/Strata/docs/DETAILS.md`, `~/Strata/docs/AMD_HIP_PERFORMANCE.md`,
`~/Strata/docs/BATCHING.md`, `~/Strata/docs/KV_PREFETCH.md`) show an engine built **around the prefill transfer**,
not around llama.cpp's graph:

| Strata mechanism | what it does | why it matters to us |
|---|---|---|
| `--prefill auto` with a **byte-sized streamed expert ring** ("picks the largest chunk that keeps that ring") | chunk up to 16384 on gfx12 today (docs §1218: 32768 ceilings), 8192 as the floor | our `-ub 16384` measured **+9.5 %**, and we stop there because the wide compute view is VRAM-hungry |
| **prompt staging ring of N slots** (`STRATA_PREFILL_RING`, 256 slots below ~6K, 192 to 12K, 96 above) | experts are streamed through a *slot ring* with async copies | our per-ubatch expert traffic is a **device-side gather kernel on the compute stream** (no copy-engine overlap) — §2/§4 H1 |
| `STRATA_PREFILL_CPU_SHARE*` | the **CPU computes part of the prompt** while the GPU streams experts | our prefill leaves 24 of 32 cores idle; on a link-bound shape this is free FLOPs — H5 |
| `--prefill-pipe` (multi-GPU), `STRATA_PREFILL_COPY_EVENT_EVERY`, `COPY_WAIT_DEDUPE` | chunk pipelining and copy-event bookkeeping across cards/chunks | we stage per split, per ubatch, and wait |
| `STRATA_HIPBLASLT_TUNING` (per-arch, per-library-version tables), `STRATA_PREFILL_MMQ` | **calibrated** dense-GEMM selection, llama.cpp's MMQ kernels reused | our MUL_MAT class is 26 % of the op budget and we take the default library choice for every shape — H3 |
| `--expert-profile` (precomputed ranked routing) + `--expert-cache auto` + `--resident-cpu-experts` | residency chosen from a *profile*, resident copy in RAM, adaptive exchange | our arena is heat-seeded and uniform-slot; not obviously the same thing |
| `--kv int8`, `--kv-resident 32768` | 8-bit KV, a resident window | ours is q8_0 with no window in this shape |

Strata's October log on this box (`~/Strata/strata-iq3_xxs.log`) shows **1636 t/s** for a 70 580-token prompt on
ROCm 10.1 with an older build, i.e. the ~3500 figure is a recent gain from exactly this kind of tuning. It is a
different engine with a different weight layout (a "pack"), so **do not try to port it**; read it for the
*ideas*, then measure whether each idea is available in our design.

**`@DanoPTT`'s two states** (issue #368) are the other half of the puzzle and they are a *transport* story, not a
kernel story: see §2.3.

## 2. Our measured baseline (2026-11-26, r366, all reproducible with §5)

### 2.1 The link does matter, but it is not the cap

| our GPU | H2D calibration | L1 prefill | L1 decode | note |
|---|---:|---:|---:|---|
| GPU1, M.2->Gen5 x4 riser | 14.4 GB/s | **1712 t/s** | 26.1 | Dano's Gen3-x16 class |
| GPU0, Gen5 x8 | 28.7 GB/s | **2108** | 32.6 | this box's usual arm |
| `radix`, Gen5 x16 | 55.0 GB/s | **2262** | 44.8 | |
| Dano, Gen3 x16 | 12.5 GB/s | **2951** (good) / 1620 (slow) | 52 / 31 | Windows |

Fitting a per-ubatch fixed cost across the three of ours gives **~0.9 s per 8192-token ubatch at x8 and ~2.4 s at
Gen5-x4** — i.e. ~34 GB of expert traffic per ubatch, the whole routed set. So our prefill *is* transfer-heavy,
and a 4x slower link costs 1.5 s/ubatch on our box. Dano's good state pays **none** of it, which is the single
most important clue in this campaign.

### 2.2 The op budget says we are compute-capped below ~2770 t/s

`GGML_CUDA_OP_TIMING=1` (per-op HIP events; it perturbs absolute rates by ~2 %, its *shares* are the point), one
89 182-token prefill on GPU0:

| class | share | absolute |
|---|---:|---:|
| FUSED (dispatcher fusions) | 26.5 % | 8.68 s |
| `MUL_MAT` (dense: attention/GDN projections, shared experts) | 24.7 % | 8.11 s |
| `MUL_MAT_ID` (routed MoE) | 25.0 % | 8.19 s |
| `FLASH_ATTN_QSA` | 7.9 % | 2.58 s |
| `DSV4_HC_PRE`, `INDEXER_TOPK`, `GLU`, `CPY`, `LIGHTNING_INDEXER` | 11 % | 3.5 s |
| **all ops** | | **32.8 s = 0.360 ms/token = 2770 t/s** |

Wall time is 41.5 s (0.466 ms/token, 2152 t/s), so **~0.9 s/ubatch (20-23 %) is not in any op** — that is the
expert transport, launched from the scheduler's input loop, not a graph node.

### 2.3 The arithmetic that bounds the gap

| who | ms/token | implies |
|---|---:|---|
| our ops alone | **0.360** | 2770 t/s ceiling from the current kernels |
| our wall, x8 | 0.466 | +0.11 ms/token of transport |
| Dano, good state | **0.339** | below our **ops-only** cost → he pays no transport *and* is ~6 % faster per op |
| Dano, slow state | 0.618 | +0.28 ms/token = one full expert-set pass at 12.5 GB/s |
| Strata, ~3500 | **0.234** | 35 % below our ops-only cost → **not reachable by removing transport alone** |

So there are two independent deficits, and the second one is the hard one:

1. **transport, ~0.9 s/ubatch** (20-23 % at x8, ~35 % at Gen3-x16). Removable in principle; Dano's good state
   shows a transport-free prefill exists.
2. **per-op compute, 0.360 vs Strata's ≤0.234 ms/token.** Nothing in the transport register explains this; it is
   the kernel/mix/architecture question, and it is where the 3500 lives.

Dano's own census sharpens (1): his two states differ by **138 gather calls** over the prefill — exactly one
table role across his 46 host layers — so one role's tables switch **transport** between states (2070 calls with
text A vs 2208 with text B). His residency map is identical in both, so this is not "what is resident": it is
*which path a role's routed op takes*.

## 3. What is NOT the answer (measured this session — do not redo)

| arm (GPU0 unless noted) | L1 prefill | verdict |
|---|---:|---|
| default (r366) | 2108 | baseline |
| `-ub 16384` | **2310** | +9.5 %, text identical **on this prompt**; needs a width-purity gate before it is a recommendation |
| `-ub 4096` | 1769 | per-ubatch fixed cost confirmed |
| `MOE_EXPERT_CACHE_MIB=0` (no arena/pool, plain `-ncmoe` streaming) | **2329** | +10.5 % prefill but decode 32.6 -> **13.1**: the cache's *prefill* overhead is real, its decode value is huge. Not a config, a target |
| `MOE_EXPERT_CACHE_MIB=8192 / 26000` | 1920 / 2128 | more residency barely moves prefill — suspicious, see H1 |
| `MOE_GATHER_NSPLIT=32` | 2172 | +2 % |
| `GGML_CUDA_GDN_CHUNKED_BF16=0` | 1891 | chunked GDN is worth its keep |
| `--ctx-checkpoints 0` | 2153 | +2 % |
| `--lazy-mode mlock` (pin the whole 27.4 GiB PLE) | 2125 | wash: the PLE is already page-cache-warm here. Re-test on a 64 GiB box |
| `GGML_CUDA_SLAB_RING_MIB=6144` (r361 layout) | 1992 | the r366 fix is strictly better |
| GPU0 vs GPU1 vs radix | 2108 / 1712 / 2262 | §2.1 |
| dense 4B Qwen3.5, same prompt, ub 4096/8192/16384 | 4256 / 4154 / 3971 | no per-ubatch fixed cost in the generic scheduler: the 0.9 s is specific to the host-expert path |

Also ruled out by `@DanoPTT` on his box: the H2D calibration line does **not** track his flip (slow visits 12.2-12.5
GB/s inside the good range 6.7-12.6), `GPU_MAX_HW_QUEUES=2` does not, `ROCBLAS_USE_HIPBLASLT=0` does not.

## 4. Hypothesis register (ranked; each has an experiment, an instrument and a signal)

**H1 — our prefill expert transport is a synchronous device-side gather where Strata uses an async slot ring.**
The scheduler's gather (`moe_cache_gather_host` -> `moe_cache_gather_kernel`) runs on the *compute* stream at
input-loop time, so it neither overlaps the previous split's compute nor uses the copy engine, and every ubatch
re-reads ~34 GB. Strata streams the same bytes through a slot ring with async copies and a configurable slot
count, and sizes the chunk to the ring.
*Experiment:* (a) re-run the `GGML_SCHED_DEVGATHER=0` arm on r366 (the ring/staging path) and compare its
transport share with the gather's; (b) instrument the gather: bytes moved, per-ubatch wall, and whether a second
copy stream or a persistent slot ring ahead of the split overlaps it; (c) check why **more residency does not
lower prefill** (§3: 8 GB -> 26 GB arena moved 1920 -> 2128) — if the gather really resolved arena-first for the
routed experts, +18 GB of resident experts should have saved far more. *Expected signal:* the gather's bytes/time
move off the critical path; +5-20 % at x8, much more on a slow link. *Instrument:* `GGML_CUDA_OP_TIMING=1`,
`MOE_EXPERT_CACHE_TABLES=1`, the `device gather:` census, `GGML_CUDA_SLAB_STATS=1`. *Risk:* low (block 09/10
internals, no arithmetic change if the bytes are identical).

**H2 — the ubatch is too small for the fixed cost, and our chunk is not budgeted like theirs.**
Our fixed cost per ubatch is ~0.9 s at 8192 tokens; `-ub 16384` recovers 9.5 % and the growth is blocked by the
wide compute view's VRAM, which is exactly the thing Strata's "largest chunk that keeps the ring" sizes around.
*Experiment:* measure `-ub 16384` and `-ub 24576/32768` with the r366 slab (the wide band now amortises the aux
region), record VRAM headroom and any `SLAB_WIDE_AUX` refusal, and run the width-purity gate (text + KLD at the
same ubatch on a few models). *Expected signal:* +9..20 %; the ceiling is `-c` and the arena. *Instrument:*
`SLAB_STATS`, `slot print_timing`, `benchmarks/prefill-logit-methodology.md`. *Risk:* a width change can flip a
near-tie, so it needs the gates, and the bigger wide view may cost decode VRAM.

**H3 — the dense-GEMM (prefill MUL_MAT) selection is leaving 30 %+ on the table.**
`MUL_MAT` + `FUSED` are 51 % of our op budget and both go through our MMB/MMQ decision + the library (hipBLASLt)
for some shapes. Strata ships *calibrated per-arch, per-library-version* hipBLASLt tables and an MMQ prefill
path, and reports per-shape decisions at start.
*Experiment:* dump the per-shape decision (`GGML_CUDA_MMB_CFG`/the `MMB_CFG` line, the MMB/MMQ census) for the
prefill graph, then A/B the alternatives per shape: `ROCBLAS_USE_HIPBLASLT=0/1`, `GGML_CUDA_MMB=0`,
`GGML_CUDA_MMB_CFG` overrides, and a hipBLASLt per-shape calibration sweep of our own (the lottery check
`scripts/check-hipblaslt-lottery.sh` shows 10.2's solution list is one entry per call, so the *library's* choice
is not the lever — our own kernel choice may be). *Expected signal:* one or two shapes with a big delta; the sum
is the headline. *Instrument:* `GGML_CUDA_OP_TIMING=1` per-shape names (`MUL_MAT [KxMxT]` is printed), plus a
targeted micro-bench harness. *Risk:* changing a GEMM path moves prefill logits => the KLD gate is mandatory.

**H4 — qwen4exp's own ops are 18 % of the budget and may have a cheaper prefill form.**
`FLASH_ATTN_QSA` 7.9 %, `DSV4_HC_PRE` 4.8 %, `INDEXER_TOPK` 3.2 %, `LIGHTNING_INDEXER` 1.6 %, `GLU` 2.2 %.
*Experiment:* for each, ask whether the prefill width admits a fused/blocked form we do not use (we already have
hyper-connection fusions — `scripts/gate-hc-fusions.sh` — so check what is *not* fused), and measure the op's
scaling with width. *Expected signal:* 2-6 % total. *Risk:* model-specific correctness (the HC gate exists).

**H5 — the CPU is idle during prefill; Strata's `STRATA_PREFILL_CPU_SHARE` is not.**
*Experiment:* measure the CPU's idle fraction during an 89k prefill (we know the *server* uses ~1 core, and the
GPU is the bottleneck here), then evaluate the cheapest useful split: PLE rows, the dense attention/GDN
projections, or the shared-expert FFN for a band of the ubatch on the CPU while the GPU streams experts.
*Expected signal:* on our box probably small (we are compute-bound on the GPU, and the CPU is 4x slower per
FLOP); on Dano's Gen3 box potentially large, which is why it is in the register. *Risk:* high complexity, purity.

**H6 — Dano's flip is a per-start *transport-path* choice, and reproducing it may be the cheapest route to a win.**
His two states differ by 138 gather calls and 2.28 s/ubatch, with identical residency. Either a role's tables are
served without a copy in the good state, or the path that moves whole tables is chosen in the bad one. Our arms
always land in the "pays transport" state.
*Experiment:* ask him for the per-chunk `slot print_timing` block from one good and one slow visit plus the
`device gather:` census from each (already requested in issue #368); meanwhile grep our own decision points for
anything that can differ per start with identical configuration — `moe_cache_take_over` (identity tables),
`sched_input_gatherable` (pool_backed + band), `moe_cache_stage_ready`, the `GGML_SCHED_DEVGATHER` mode, the
`prefill_stream` band — and make each one PRINT its decision at `-lv 4` so a pair of visits can be diffed.
*Expected signal:* if his good state is transport-free, that path is worth up to 23 % on our x8 box and far more
on a slow link. *Risk:* low (diagnostics first).

**H6b — the maintainer's own hypothesis: the Windows HIP/driver library does something on his box that our Linux
one does not.** The delivery's transport stack is the same C++ on both, but three things are *not*: WDDM's
allocation accounting (a freed allocation stays committed — `INVARIANTS.md`, `common.cuh`'s FA-ladder notes), the
`O_DIRECT`-vs-`FILE_FLAG_NO_BUFFERING` pool read paths (r227/r294), and the fact that his **12.5 GB/s** is a Gen3
link where a copy-engine transfer and a compute-stream kernel read are far more distinguishable than on Gen5.
*Experiment:* do not chase a driver-only difference first — make H6's decisions PRINT, and let his pair of visits
tell us whether the *path choice* differs. If it does not, this becomes the leading hypothesis and the question is
whether a Linux arm can be forced into the same path (e.g. `GGML_SCHED_DEVGATHER`, the ring, or `--host-experts
pinned`) and reads his good state's rate.

## 5. Tools in this directory (copied from the session that produced §2)

```sh
./arm.sh <tag> [ENV=V ...]          # one llama-cli arm at Dano's flags; env: DEV (gpu index), UB, EXTRA,
                                    #   PROMPT, VERBOSE; writes runs/<tag>/run.log and prints the rates
./arm_gpu.sh <tag> <gpu> [ENV=V...] # the same, plus a rocm-smi sampler -> runs/<tag>/smi.txt (sclk/mclk/power)
./visit.sh <tag> [ENV=V ...]        # one SERVER visit (S1 -> L1 89k -> S2 -> L2 cache hit), fresh server,
                                    #   includes -np 2 --kv-unified, cold heat prior, prints per-request timings
```

Useful instruments (see `DIAGNOSTIC.md` for the catalogue):

* `GGML_CUDA_OP_TIMING=1` — per-op HIP-event totals and shares, the single most useful prefill instrument.
* `GGML_CUDA_SLAB_STATS=1` — layout at the wide arm/drop: `work (narrow/ring/wide_aux)`, `arena`, `outside_*`,
  `aux_refused`; `GGML_CUDA_WORK_ALLOC_DEBUG=1` adds each compute view's placement.
* `MOE_EXPERT_CACHE_TABLES=1` — per-layer residency, stand-down and re-arm tallies; with `-lv 4` also the
  `device gather:` census and the MoE fusions.
* `slot print_timing:` lines in any server log (**default verbosity**) — per-chunk prompt tokens and seconds.
  This is the cheapest way to compare against Dano's box.
* `rocm-smi --showclocks --showpower --showuse` during a run — confirm sclk/mclk/power; our GPU0 runs 3359 MHz /
  1258 MHz / up to 430 W, so the card is not the limiter.
* Purity: same-seed greedy text (`scripts/extract-generated.py`) and `scripts/gate-prefill-logits.sh` for anything
  touching a prefill kernel or a default-on approximation; `benchmarks/mtp-adaptive-methodology.md` for decode.

Hygiene that has bitten before: `HIP_VISIBLE_DEVICES` pinned on every arm; `llama-cli` needs `--single-turn`
(and `--no-display-prompt`) and everything inside a `timeout`; no `-lv` on a text/hash arm and `-lv 4` on a
census arm (`-lv 3` is the trap); never run two benchmarks at once; keep large artifacts off `/tmp` (tmpfs).

## 6. A suggested first session

1. Re-derive §2.2 on the x16 box (`radix`) so the op budget is not x8-biased, and record the per-shape
   `MUL_MAT [KxMxT]` times (H3's raw material).
2. H1(a): the `GGML_SCHED_DEVGATHER=0` vs default transport comparison on r366 at `-ub 8192` and `-ub 16384`,
   with `SLAB_STATS` and the gather census. Decide whether the gather's bytes can be taken off the critical path
   — this is the cheapest 20 % on the table.
3. H2: the width sweep 8192/16384/24576/32768 with text + KLD, and record what blocks each step (VRAM, arena,
   slab refusal). If 32768 works, the fixed cost is amortised 4x.
4. H3 begins only after 2-3: a per-shape GEMM table is the biggest single unknown, but it needs a harness.
5. Send `@DanoPTT`'s per-chunk `print_timing` into H6 whenever it arrives; that pairs with 2.

## 7. What a win looks like, and how it ships

A validated change with a kill switch, the delivery's gates green (`validate-set.sh`, the prefill-logit KLD,
same-seed text, and the specific gate for whatever moved), recorded in `WORKLOG.md` + `benchmarks/`, and folded
into the **owning block** (09/10 for the transport/slab, 06 for the GEMM path) — *after* the maintainer approves
the block decision. Nothing in this directory is applied to the fork, ever, without that.

## 8. Excluded (do not re-measure)

`--lazy-mode mlock` vs `on` (wash here; PLE page-cache-warm), GDN chunked off (-11 %: keep it), MMB off (-1 %),
`--ctx-checkpoints 0` (+2 %), `MOE_GATHER_NSPLIT` 4->32 (+2 %), cache size 8/26 GB (1920/2128 vs 2108),
`-ub 4096` (-16 %), the r361 ring-hole layout (1992 vs 2108), a dense model's per-ubatch fixed cost (none: 4256
at ub 4096, so §2.2's 0.9 s is the host-expert path only), the H2D calibration as Dano's state carrier (his own
16-visit sweep), `GPU_MAX_HW_QUEUES=2` and `ROCBLAS_USE_HIPBLASLT=0` (his box).

## 9. Open questions for others

* `@DanoPTT`: per-chunk `slot print_timing` from one good and one slow visit, the two `device gather:` census
  lines, and whether a good visit's `-lv 4` census shows any table **without** a copy (asked 2026-11-26).
* Strata's exact version + flags behind the ~3500 figure (our copy's log shows 1636 t/s on ROCm 10.1 for a 70K
  prompt; the newer number is what the target should be measured against, and its `--prefill`/ring settings are
  in `~/Strata/docs/DETAILS.md` §1218-§1220).

## 10. Session 2 (2026-11-26, second session) — the ring is not a ring, and the fill engine is the transport

> **SESSION 3 REPLACED ITS SINGLE-RUN NUMBERS WITH AN INTERLEAVED MATRIX AND ADDED THE CORRECTNESS BLOCK — read
> [`RESULTS-session3.md`](RESULTS-session3.md) for the current numbers.**
>
> **THE TRANSPORT IS NOW THE PING-PONG AND THE SLOT RING IS DELETED — for what is implemented, the invariants,
> the slab/stage-region facts and the next session's task (a fit/slab/arena REPORTING audit), read
> [`HANDOVER-session4.md`](HANDOVER-session4.md) first.** The engine finding below holds (the
> kernel fill is the slow engine, +24.6 % for a copy-engine fill, ranges disjoint); the redirect's "+6 %" does
> **not** — at 5 samples per arm it is +2-4 %, inside the staged path's own 5-9 % spread — and the gather is
> **not** retirable (it is also the decode-width path). Session 3 also adds a 122B-A10B arm with no PLE
> (**+39.7 % prefill**, text identical) and the cross-geometry results (a tensor split loses 12 %).

**Nothing here is in `patches/`.** Three experiment patches are in [`experiments/`](experiments/) and were
measured on a build carrying them (`~/llama.cpp` uncommitted; `release.json` untouched, so `validate-set.sh`
is unaffected). All numbers are GPU0 (Gen5 x8) unless the row says `radix` (Gen5 x16), `-ub 8192`, Dano's
flags from §0, and `MOE_EXPERT_HEATMAP=load` (residency pinned — §89 makes the heat prior an *input* to greedy
text, so an unpinned A/B compares two different configs).

### 10.1 Measured (prefill t/s, 89 234-token prompt)

| arm | x8 | x16 | decode | note |
|---|---:|---:|---:|---|
| default (pool + device gather) | 2099-2117 | 2355-2359 | 32.7 | baseline; very reproducible on x16 (±0.1 %) |
| forced ring, cache's two-tier kernel fill, arena stand-down refused (pre-r366 behaviour) | 1790-1791 | — | 26 | the A/B control |
| same + `MOE_HOST_POOL_STAGE_STANDDOWN=1` (a stood-down table may still stage) | 1831-1843 | — | 26-27 | the split-level refusal was real, but this is only +3 % (the knob is **deleted** in the delivery candidate: session 7 removed the gate and its hook caller, `REVIEW-squashed.md` §10.2) |
| same + **bulk fill** (one copy-engine DMA per table from the pool mirror) | **2265-2367** | — | 26-36 | **+10-12 % over the gather, same-seed text identical** |
| bulk fill + one-split scheduler lookahead | 2272-2359 | — | — | **no change** — the transfer was already overlapped |
| `MOE_EXPERT_CACHE_MIB=0` (no cache at all: ring + one async pageable DMA per table) | 2329-2482 | 2531 | — | the existence proof, but decode collapses |

Same-seed greedy text (128 tokens, heat prior pinned, `scripts/extract-generated.py`): gather
`sha=43d0b1f4629e`, bulk fill `sha=43d0b1f4629e` — **byte-identical**, so the fill is a pure transport change.

### 10.2 Why the two intended transports both lose (the answer to "the arena is 640 GB/s, the link is 28")

The intended design — fill the ring from the arena for resident experts, from the host only for the gaps — is
`moe_cache_stage_table`'s two-tier kernel, and it is **the slowest of the three fills**, because of its *engine*,
not its byte count:

| fill | host bytes / ubatch | engine | stream | effective |
|---|---:|---|---|---:|
| device gather | ~19 GB (arena D2D for the rest) | kernel | **compute** | ~20 GB/s, fully exposed (0.94 s/ubatch) |
| two-tier cache stage | ~19 GB | kernel | copy | **~10 GB/s** → 1.9 s/ubatch |
| bulk DMA (this session) | **40 GB** | copy engine | copy | ~28 GB/s → 1.4 s/ubatch |

So the fill that moves **2.1x the host bytes wins by 23 %**: a kernel reading the `cudaHostAlloc`-mapped pool
reaches roughly a third of the link, and the copy engine reaches ~all of it. The gather is byte-optimal and
still loses because it runs on the *compute* stream, where nothing hides it.

The ring cannot rescue either one because **it is not a pipeline**: `ggml_backend_sched_compute_splits` issued
`sched_stage_issue(split N)` and then, in the same iteration, drained split N — so a transfer could only hide
behind the *previous* split's compute. Its 8 slots exist to hold the several host weights *of one split*, not to
run ahead. §10.1 shows the honest result of fixing that: with a bulk fill there is nothing left to hide, so a
one-split lookahead buys nothing. It would matter with the kernel fill.

### 10.3 Other findings that change the arithmetic

* **Compute budget, corrected.** The routed MoE is 415 TFLOP and the dense `MUL_MAT` 527 TFLOP (the earlier
  "8.8 TFLOP MoE" forgot the 47 host layers), i.e. **~29 TFLOPS aggregate over 32.75 s of ops**. Strata's
  ~3500 t/s implies ~37 TFLOPS and Dano's 2951 ~31, so the compute gap is **1.28x, not 1.5x** — the ops side is
  a tuning problem, not a different architecture. Shapes: `n_embd` 2560, expert FFN 640, 512 experts, 10 used,
  hyper-connections 4 @ rank 320.
* **`FUSED REPEAT` + `FUSED MUL` are 3.9 s = 12 % of the op budget.** `build_moe_ffn`'s `weight_before_ffn` path
  does `ggml_repeat_4d(cur[n_embd,1,n_tokens] → [n_embd,10,n_tokens], F32)` and then multiplies it: the repeat
  **materialises 839 MB of F32 per call** (measured via the new shape print, `experiments/0003`) and the mul
  writes another 839 MB. No REPEAT+MUL fusion exists for this pair. Fusing them writes 839 MB once, which should
  return ~2.6 s (≈8 %).
* **`moe_cache_stage_ready()` refused staging for a POOL-backed table whenever its arena was stood down**
  (`t.arena != nullptr && t.slots > 0`), although a pooled table's source is the pool. The refusal is per
  *split*, so a stood-down layer sent its whole split to the scheduler's pageable per-table bounce: **174-189 GB
  per run through one host `memcpy` + one sub-MiB H2D per expert** under a bounded pinned ring. That is the
  erratic/starved GPU the session saw. Fixed in `experiments/0001` (worth only +3 % on its own, because the
  two-tier kernel it then reaches is the slow engine above).
* **The pool's covering mirror is already `slot == expert`** (`pool_fill_mirror_locked`: "the file layout IS the
  slot layout"), so one table is one contiguous run of `cudaHostAlloc(mapped)` memory and one DMA is a
  byte-identical fill. Nothing used that.
* **`-ub > 16384` is blocked by VRAM arbitration, not compute**: the ~21-22.6 GiB arena grows into the ~11 GiB
  wide prefill view (`SLAB_WIDE_AUX refused: slab=1 wide_armed=0`, `HipVMM Failure: out of memory`), and an
  explicit `MOE_EXPERT_CACHE_MIB` only shrinks it at slab init — the settle grows it back before the first wide
  ubatch. Explicitly out of scope for this campaign (maintainer, session 2).

### 10.5 The hybrid fill (implemented 2026-11-26, session 2) — the shape asked for from the start

`experiments/0001` now carries the design the slab/arena work should have started from: **each engine does the
half it is good at.** `MOE_HOST_POOL_STAGE_BULK=2` (the default of the patch):

* the **host** side is a handful of **coalesced run DMAs** on the copy engine, straight out of the pool's
  expert-ordered mirror — adjacent non-resident experts become ONE transfer (measured 68 transfers per table
  instead of 512, i.e. ~5 experts per run), and `MOE_HOST_POOL_STAGE_MERGE=<n>` deliberately over-fetches
  across resident runs shorter than `n` experts to lengthen them (a merge needs no patch pass: the pool mirrors
  every expert, so the boundary residents simply come from the host too);
* the **arena** side is **ONE kernel launch** for the whole table (`skip_miss`, the host runs already wrote
  everything else) — a kernel is the right engine here because it never crosses PCIe.

Measured (ub 8192, x8, heat prior pinned, `MOE_EXPERT_HEATMAP=load`):

| `-ncmoe` | gather (baseline) | whole-table DMA (`BULK=1`) | **hybrid (`BULK=2`)** |
|---|---:|---:|---:|
| 48 (max arena, 21.8 GiB) | 2100 | 2232 | **2301 / 2309 / 2361** |
| 47 | 2117 | 2231-2367 | 2231 |
| 36 (12 layers resident) | 2119 | 2468-2510 | 2427 |

So the hybrid is +11 % over the gather at `-ncmoe 48`, but only +4 % over one DMA of the whole table — and
`MERGE=4` is a small *loss* (2265 vs 2301), exactly as the arithmetic said it would be at 53 % coverage
(over-fetched bytes cost more than the extra run length saves). Keep it at `1` (coalesce only).

**What this proves:** making the fill three times more transfer-efficient bought ~4 %. The residual at
`-ncmoe 48` is ~5.5 s of `wall − ops` and it is **not** transfer volume; it needs a split-level timeline to
attribute (the ring→input D2D on the compute stream is ~0.7 s of it, and the arena's fill ramp is the next
candidate — but the fill timing census does not fire for a pool-promoted fill, so that is unmeasured, not
assumed).

Note `-ncmoe 36` remains the fastest prefill arm *because a whole-layer-resident table is read in place and
never copied at all* — a bigger effect than any transfer-efficiency win. The maintainer's decision is to stay at
`-ncmoe 48`, where the arena (and therefore decode) keeps its size, and to fix the transport instead. That
ordering is what this section does.

### 10.6 The slot ping-pong (branch `wip/prefill-slot-pingpong`, `experiments/0004`)

The design the maintainer asked for from the start: **the ring slot IS the compute buffer.** A staged host
weight is consumed *in the slot* instead of being copied into the graph-allocated split input first, the slot is
released after that split's compute is enqueued, and the fill for split N+1 is issued while split N runs
(`0002`). That is the retired r250 "redirect" form -- safe here for the reason r250 lacked: such a split is a
multi-token prefill split, and **`GGML_CUDA_GRAPH_MAX_TOKENS` is 16**, so no captured graph can bake the input
pointer in. `GGML_SCHED_STAGE_REDIRECT=0` restores the copy.

Measured at `-ncmoe 48`, ub 8192, x8, heat prior pinned, hybrid fill:

| arm | prefill |
|---|---:|
| copy into the split input (pre-change) | 2229 / 2274 |
| **consume the slot in place** | **2305 / 2313 / 2466 / 2471** |

Mean **~2390 vs ~2250 (+6 %)**, and same-seed text is identical (`sha=43d0b1f4629e`, 128 tokens) — the content
of the buffer is unchanged, only where the consumer reads it from. Against the 2100 gather baseline this is
**+14 %**, with the ops-only ceiling at ~2725.

### 10.7 The GPU is NOT I/O-starved — a third pipeline stage would target the wrong thing

The maintainer proposed a 3-stage pipeline (host fill -> arena-expert fill -> process). Measured with
`arm_gpu.sh` on the redirect arm, during the prefill: **`GPU use` median 100 %, p25 92 %** (the 79 % mean
includes model load). The GPU is saturated, so there is no I/O stall left to hide: the ~5.8 s of
`wall - ops` at this arm (38.6 s vs 32.75 s) is **unbracketed GPU work**, not waiting. At `-ncmoe 47` that
5.8 s was the device gather (a compute-stream kernel outside the op table); with the hybrid fill its remaining
components are the arena-side fill kernel and DRAM contention from the copy engine's writes, and the split
between them is **unmeasured**.

Consequence for the campaign's ordering: the transport is now within ~14 % of the ops ceiling and the GPU is
busy, so further pipeline depth buys little. The remaining 12-14 % to 2900 lives in the op set — the
`ffn_moe_weighted` REPEAT+MUL (3.9 s, 12 % of ops) first, then the GEMM rate (~29 TFLOPS aggregate, against
radiance's implied ~68 on the same per-card link and Strata's ~37).

### 10.4 What to do next, in order

1. **Land the bulk fill** (`experiments/0001`): one copy-engine DMA per table out of the pool mirror when the
   mirror invariant holds, falling back to the two-tier kernel otherwise. +10-12 % at ub 8192, text identical.
   It needs the *default* transport on the pool tier to become "stage at prefill width" rather than "gather"
   (today `GGML_SCHED_DEVGATHER` default mode 1 gathers them), and the slab's ring hole must be carved for it
   (r362 declines the hole only while the tier *gathers*).
2. **Fuse the `ffn_moe_weighted` REPEAT+MUL** (~2.6 s, +8 %): a block-10 dispatcher fusion writing the mul's
   output directly from `cur` and the per-expert weights.
3. Then the remaining exposure (~5 s, 13 % at the bulk fill) is *not* transport and not the ring: it is what is
   left of `wall − ops` and needs a split-level timeline to attribute (the PLE page-ins and the ring→input D2D
   are the two candidates that have not been separated yet).
4. Re-measure on `radix` (x16) after 1-2; the x16 box is where the 2900 target lives and where the gather is
   already only 13 % of the wall.
