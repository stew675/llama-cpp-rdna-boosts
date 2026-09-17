# Adaptive MTP baseline methodology & validation records

Why this file exists: on 2026-09-02, a regression in the delivery's decode
kernels collapsed **adaptive MTP / draft-mtp on MoE models** (and cost ~33%
on dense MTP). It went unnoticed for weeks because no validation gate
exercised MTP end-to-end: the llama-benchy decode suites (see
`benchmarks/`) run plain (non-speculative) decode only, and MTP numbers
never made it into a dated baseline. This file is the MTP-specific gate:
protocols, canonical commands, expected numbers (dense + MoE), and the
acceptance/verification rules that would have caught the regression.

Status: 2026-09-02 — gate defined; dense + MoE baselines recorded (pre- and
post-fix). All runs on 3x R9700 (gfx1201), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`), fork build flags as in
`archive/work/wip-archive/qwen35moe-prefill/bench-config.md`; 1-GPU runs pin `HIP_VISIBLE_DEVICES=0`
(GOLDEN RULE 1: without the pin llama.cpp layer-splits and decode drops
~97 -> ~81 t/s on the MoE model).

**2026-09-17 update:** the re-base onto `ebbb18522` picked up upstream #28549 ("Enable CUDA graph for
MTP draft"), which the isolated A/B now shows is worth **+0.3-1.4 %** on the four-axis adaptive gate
(scaling with draft depth) for free — see `2026-09-17-mtp-pr28549-ab.md`.  The MTP reference cell for
those measurements is the 2-card Q8_0 Qwen3.8-27B row below; it reproduces the recorded 2026-09-16
numbers within ~1-2 %.

## Why decode benches cannot see MTP regressions

MTP adds two decode shapes that plain decode never produces:
- **verify batches**: the target is decoded at ncols = n_draft+1 (2..13),
  i.e. multi-token MUL_MAT and multi-token MUL_MAT_ID (MoE) rows;
- **draft-context decode**: single-token steps through the masked/nextn
  path, re-captured graphs, and (for single-head MTP) sequential drafting.

llama-bench tg decodes 1 token/step regardless of `-b`, so it only covers
the ncols==1 path. Kernel or fusion bugs confined to ncols 2..13 (or to the
draft context) are invisible to llama-bench/benchy and to single-token
same-seed coherence. The 2026-09-02 regression is the worked example: the
block-08 rms_norm->mmvq Q8_1 quantize-cache fold corrupted multi-token
MUL_MAT_ID, so MoE verify logits diverged from single-token decode and MTP
draft acceptance collapsed to 0/1527 (draft-mtp ~53 t/s vs plain ~90, where
MTP should accelerate). Dense models and single-token MoE decode were
unaffected, so every existing gate passed.

The 2026-09-12 issue-#30 regression is a second worked example of a different class:
the patch set made the mmvq knobs **band-uniform** (a purity requirement -- `nwarps` and
VDR both participate in the K-split accumulation order, so decode and the verify batch
must agree) but left them at their **single-token-tuned** values, which cost up to +35% on
the verify widths (dense Qwen3.8-27B UD-Q4_K_XL, `q8_0` KV, `llama-batched-bench` B=8
2.929 s stock / 3.958 s delivery).  Acceptance stayed flat (0.484 vs 0.466), so the
acceptance rule passed; `llama-bench tg128` passed (single token is the one width that
did *not* regress); block 13's `pl=8` check was old-vs-new *within* the delivery.  Only a
**stock-relative** batched decode at the verify widths exposed it.

Gate addition (2026-09-12, rule 5): before shipping any decode/verify or mmvq change, run
an interleaved stock-vs-new `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` with a
quantized KV cache on a dense K-quant model and require the new build to be within noise
of stock at B=1 and **no worse** at B=4/B=8.  The batched TG numbers are the instrument
(they are acceptance-free); `draft-mtp` / `draft-mtp-adaptive` end-to-end throughput is
the confirmation.  The amended verify path measured B=8 2.798 s vs stock 2.929 s; see the
2026-09-12 (16) WORKLOG entry and the block-08 + block-10 amendment section in
`../patches/README.md`.

Follow-up (2026-09-12 (17)): the VDR half of the fix is now **per kernel** (dense upstream, MoE
expert `mul_mat_vec_q_moe` keeps block-10's VDR=4).  The MoE expert kernel is not reached by
`calc_nwarps` (one warp per token), so the band-uniform `nwarps=1` still applies to the MoE model's
*dense* layers — that is the source of the residual MoE single-token/MTP delta vs the pre-(16)
build (a diagnostic restoring per-type `nwarps=8` recovers MoE B=1 0.783 -> 0.716 s and MTP
161 -> 167 t/s but costs the dense 27B MTP 35.9 -> 34.3 t/s).  The gate is unchanged, but when
judging a change also measure the **affected model class**: a knob that is dense-purity-mandated
can still cost a MoE model's dense layers (and vice versa).

Follow-up (2026-09-12 (18)): that residual is recovered by making the dense mmvq **weight** kernel
pick `nwarps` per `(type, K)` — a Q8_0 weight with `K < 4096` (the MoE attention qkv/gate and the
lm_head) takes the wide block (8), every other shape stays at 1; the choice is per tensor shape, so
`W = 1..8` still agree.  Measured: MoE B=1 **+4 %**, MTP `n_max 3` **+2 %**, `n_max 7` **+10 %**
(acceptance 0.631 -> 0.731), at −2.8 % on the MoE batched B=8; the dense 27B is **bit-identical**
(its Q8_0 weights are `K >= 5120` -> `long_k` -> 1).  **The pinned fusion ops (GDN/SSM,
shared-expert, the gate fusions) must keep the band-uniform `calc_nwarps`** — their
`calc_nwarps(GGML_TYPE_Q8_0, 1, ...)` is a single-token reduction-order anchor, and leaking the rule
into them breaks the 27B `f16` width purity (verified).  The **opposite** assignment (giving the dense
kernel the MoE's wide VDR=4 on the same short-K shapes) was measured and **rejected**: +1.6 % batched
B=8 but it cancels the MTP gain (acceptance back to 0.63115) — the two knobs have independent
per-kernel optima.

## Protocols

### Protocol A — fast per-build gate (llama-cli, fixed seed)

Canonical (dense): `bench.sh` style, seed 42, temp 0, predict >= 3000:

```sh
HIP_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=0 <build>/bin/llama-cli \
  --model <model> --fit false --top-k 20 --threads 8 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --predict 3000 --load-mode mlock \
  --cache-ram 16384 --ctx-size 102400 --flash-attn auto --temperature 0.0 \
  --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all \
  --cache-type-k bf16 --cache-type-v bf16 --ctx-checkpoints 64 \
  --reasoning-budget 65536 --reasoning-preserve --repeat-penalty 1.0 \
  --presence-penalty 1.5 --seed 42 --single-turn \
  --spec-type draft-mtp --prompt "$(cat prompts/prose-rdna-boosts.txt)"
```

Run the same with `--spec-type none` and compare.  The prompt is the versioned
`prompts/prose-rdna-boosts.txt` (16074 B, 5298 tokens, `sha256 fabdec65…`) — record the hash with any
result, and never edit a shipped prompt in place (see `prompts/README.md`).

The drafting model is the **MTP head built into the target GGUF** (`blk.<block_count-1>.nextn.*`,
`*.nextn_predict_layers`), used automatically when no `-md` is passed.  Do not pass the old standalone
`mtp-*.gguf`; it is a different drafter and changes acceptance.

**Reasoning mode is part of the protocol (2026-09-13).**  Qwen3.8 emits a thinking trace for
instruction-like prompts, so a run left on the template default measures *thinking*, not the workload:
at `-n 256` the "code" prompt never reached any Python and the "prose" prompt answered with a thinking
trace.  The four-axis gate therefore sets reasoning explicitly per axis:

```sh
case "$axis" in reasoning) REA=on;; *) REA=off;; esac   # R reasons; P/C/K generate content
... --reasoning $REA ...
```

`--reasoning off` is the llama.cpp flag (`--chat-template-kwargs '{"enable_thinking":false}'` is the
equivalent template-level knob); both arms must use the same setting.  The earlier adaptive-MTP numbers
(including the first cut of `2026-09-13-adaptive-mtp-4-axis.md`) were measured without it and are not
comparable -- see `2026-09-13-adaptive-mtp-4-axis-n12.md` for the corrected table.

**Generation length is part of the protocol (2026-09-13).**  **Never gate MTP on a short run.**  The
drafter needs context to predict what comes next, and the adaptive controller needs hundreds of verify
rounds to settle; a few hundred tokens measures the warm-up transient, not the mode.  Use **`-n 3000`**
for the four-axis gate (`-n 2000` is the hard floor).  The workloads are hundreds of lines of code, a
multi-thousand-word prose piece, a multi-thousand-character derivation, and a full recall passage --
none of which fit in 256 tokens.  The effect is not subtle: the code axis at adaptive ceiling 12
measured **-5%** vs fixed `n3` at `-n 256` (mean accepted length 4.32) and **+28%** at `-n 3000`
(mean accepted length 7.02).  A 256-token spot check reports the transient and can invert the ranking.
This is the same class of blind spot as the reasoning flag: a gate that does not reproduce the real
usage shape optimises for the wrong thing.

Gate rules:
0. **Length**: the four-axis gate runs at `-n 3000` (floor `-n 2000`).  A shorter run may be used as a
   smoke check for correctness (acceptance > 0, text purity), never as a performance verdict.  Record
   `-n` with every number.
1. **Acceptance**: with `--log-verbosity 4`, the `draft acceptance` /
   `acc per pos` lines must show a healthy rate on prose (>= ~0.45 at pos 1
   for these models; the 2026-09-02 regression showed 0.000). A collapse to
   0.0 = verify/logit numerics divergence (draft-vs-verify mismatch), not a
   tuning issue.
2. **MTP must not lose to plain decode on the same build**: `draft-mtp`
   Generation t/s should be >= `none` Generation t/s on predictable content
   and within ~10% on generic prose at draft depth 3. (Do NOT test at
   `--spec-draft-n-max 12` fixed depth: fixed-depth over-drafting is
   expected to lose; the adaptive configs C3/C6 below are the meaningful
   high-depth tests.)  **As of 2026-09-13 (issue #30) the CLI clamps
   `--spec-draft-n-max` at 15 (a visible notice; `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0`
   keeps a larger value) and keeps depths 8..15 with a visible purity notice**, so
   a baseline measured above 15 must be re-measured at 15 (or with the env set)
   before comparing.  See
   `../archive/work/block-15-campaign-wins/BETA-TESTING.md` for the notice semantics.
3. **Same-seed determinism vs the previous release** (dense): outputs must
   be byte-identical between the build under test and the known-good build.
   On MoE this is not required (fusion-ordering numerics drift is an
   accepted trade-off); sane output is the bar there.
4. **Purity range is `n_max <= 7`** (2026-09-11, after the block-12 fix):
   `--spec-type none` == `draft-mtp` is byte-identical up to an 8-token verify
   batch, which is the designed limit -- beyond it the flash-attention launcher
   switches to WMMA (`Q->ne[1] > 8`) **and** the matmuls switch from the
   MMVQ/MMVF decode families to MMQ (`ncols == MMVQ_MAX_BATCH_SIZE` = 8), both of
   which change the reduction order.  On 2-GPU `-sm tensor` the range used to
   stop at `n_max = 5` because of a second, fork-specific cause (block 12's
   size-based all-reduce dispatch changed the reduction algorithm when the batch
   crossed 32768 elements = 7 tokens); that was fixed on 2026-09-11.  Do not use
   `none == draft-mtp` equality above `n_max = 7` as a gate; use acceptance +
   MTP-vs-plain throughput instead.  See `../GREEDY-PURITY.md` §11.  **Since
   2026-09-13 (issue #30) the CLI no longer enforces the purity range by
   clamping:** `--spec-draft-n-max 8..15` is allowed with a visible notice that
   `none` vs `draft-mtp` may differ, and only `> 15` is clamped (the recurrent
   rollback snapshot bound -- a correctness bound, not the purity one).
   **Purity above 7 is also length-dependent**, which is another reason to run the
   gate long: at `-n 3000` the adaptive ceiling 12 is byte-identical to plain on
   the reasoning and recall axes but diverges on prose and code (a near-tie
   flips once the run is long enough), while at `-n 256` all four happened to
   match.  Report purity at the gate length, not a short check.

### Protocol B — server harness (dense canonical, long-context workloads)

The full adaptive-MTP comparison harness lives at
`/home/stew675/stew675/adaptive-results-perf/` (`run_sweep.py`; configs C0
no-spec / C3 draft-mtp-adaptive n 3..12 / C6 adaptive+ngram; workloads
R=reasoning, K=verbatim recall, P=prose, C=code; server = 2-GPU tensor
split `HIP_VISIBLE_DEVICES=0,2`, Q8_0 27B dense, f16 KV, ctx 262144).
Run passes 1 and 2, repeat >= 2, and compare against the baselines below.
The dense expectations table was recorded 2026-08-23 (rdna-boosts era
`e0aa19e25`) and re-verified 2026-09-02 on the fixed 13-block build —
see the results table.  **The adaptive ceiling is 12** (`--spec-draft-n-max 12`,
the mode's recommended depth); since the 2026-09-13 clamp relaxation it is no
longer capped at 7.  The current four-axis measurement is
[2026-09-13-adaptive-mtp-4-axis-n12.md](2026-09-13-adaptive-mtp-4-axis-n12.md).

## Baselines

### Dense — Qwen3.8-27B Q8_0, 2-GPU tensor split (0,2), f16 KV

Predicted tokens/s (predicted_per_second). C0 = plain decode.

| config | workload | 2026-08-23 perf-era | 2026-09-02 fixed 13-block | base (upstream) C0 |
|---|---|---|---|---|
| C0 | R | - | 31.8 | 30.0 |
| C0 | K | - | 31.6 | 29.9 |
| C0 | P | - | 31.9 | 30.1 |
| C0 | C | - | 31.8 | 30.0 |
| C3 | R | 53.2 | 53.0 | - |
| C3 | K | 149.2 | 148.2 | - |
| C3 | P | 57.6 | 59.5 | - |
| C3 | C | 78.0 | 79.1 | - |
| C6 | R | 53.0 | 53.1 | - |
| C6 | K | 329.8 | 324.6 | - |
| C6 | P | 57.6 | 59.4 | - |
| C6 | C | 75.6 | 79.1 | - |

Dense verdict: MTP baseline fully restored (C3/C6 parity or better;
C0 plain decode also ahead of upstream base). C3 speedups vs C0 on the same
build: R 1.67x, K 4.7x, P 1.87x, C 2.5x — MTP is a strong accelerant.

### MoE — Qwen3.6-35B-A3B (qwen35moe, nextn MTP head)

1-GPU UD Q4_K_M (has the nextn head), seed 42, temp 0, draft depth 3
(default n_max). There was NO pre-2026-09-02 MoE MTP baseline — that is why
the regression was not caught (the MoE decode suites are MTP-free).

| build | plain (none) | draft-mtp | draft acceptance | verdict |
|---|---|---|---|---|
| upstream 9cffdcc80 | 75.0 | 113.4 | 0.49 | MTP accelerates +51% |
| shipped 13-block (pre-fix) | 88.6 | 53.2 | 0.000 (0/1527) | MTP collapses (bug) |
| fixed 13-block (2026-09-02) | 89.5 | 125.8 | 0.51 | MTP accelerates +41% |

MoE verdict: post-fix MTP acceptance (0.51) equals the fully-unfused
internally-consistent numerics (0.49) and upstream — the draft-vs-verify
numerics relationship is not depressing acceptance. Fix summary and the
mechanism in `patches/README.md` (block 13 notes, 2026-09-02).

### MoE multi-GPU — Qwen3.6-35B-A3B-UD Q8_0, tensor split (new baselines, 2026-09-02)

First-ever MoE 2- and 3-GPU rows (no prior baseline existed). Model has the
nextn/MTP head. Env conventions as `run_sweep.py`: `NCCL_PROXY_CPUSET=8..15`,
`NCCL_P2P_DISABLE=1`, `GGML_CUDA_DISABLE_GRAPHS=0`, tensor split. Fixed
13-block build (delivery tip `8f2838d1`). llama-bench: `-t 16 -r 2 -ub 2048
-p 512 -n 128`; Protocol A MTP: temp 0, seed 42, ctx 32768, bf16 KV,
predict 512.

| config | tg128 f16 | tg128 bf16 | pp512 f16 | pp512 bf16 | plain (proto A) | draft-mtp | draft acceptance |
|---|---|---|---|---|---|---|---|
| 2-GPU (0,2) | 97.63 | 97.50 | 4603 | 5009 | 91.2 | 125.1 | 0.542 |
| 3-GPU (0,1,2) | 103.17 | 102.91 | 4770 | 4764 | 96.6 | 124.6 | 0.529 |

MoE decode scales weakly across GPUs at batch 1 (expert compute per token;
tensor split cannot share experts) — 2-GPU ~= 1-GPU Q6_K (~98), 3-GPU ~103.
MTP accelerates at both splits (+37% 2-GPU, +29% 3-GPU) with acceptance
~0.53 — the hybrid-AR + MTP combination is healthy on multi-GPU.

### MoE single-token decode anchors (MTP-free, for reference)

Canonical commands in `archive/work/wip-archive/qwen35moe-prefill/bench-config.md`. Recorded
baseline (2026-09-02, fork tip) vs fixed 13-block (2026-09-02):

| test | baseline | fixed 13-block |
|---|---|---|
| Q6_K 1-GPU tg128 | 97.6 | 98.8 |
| Q8_0 2-GPU tensor tg128 | 95.6 | - |

## What to run before shipping a decode/fusion change

1. Protocol A on the dense Q4_K_XL-UD and the MoE Q4_K_M-UD (none +
   draft-mtp, seed 42) — acceptance > ~0.45 and MTP >= plain.
2. Protocol B (C0/C3/C6, passes 1+2) on the dense Q8_0 row.
3. If any fusion/try_fuse/multitoken-kernel code changed, also run MoE MTP
   at draft depth 3 with fusion ON vs `GGML_CUDA_DISABLE_FUSION=1` and
   compare acceptance (must match within noise; a 0-acceptance split = the
   fused multi-token path diverged again).
4. MoE multi-GPU MTP (2-GPU 0,2 and 3-GPU 0,1,2, Q8_0-UD): draft-mtp vs
   plain Protocol A rows — acceptance ~0.53 and MTP >= plain at both
   splits (baselines in the MoE multi-GPU table above).
