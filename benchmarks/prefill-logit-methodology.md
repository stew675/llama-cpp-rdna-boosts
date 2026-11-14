# Prefill-logit (KLD) validation gate

Why this file exists: on 2026-10-08, issue #113 showed that the existing same-seed coherence gate and
the MTP gate are **blind to a prefill-logit divergence**.  The BF16/WMMA chunked GDN path was shipping
as the default and shifted the prefill logits substantially (mean KLD 0.032 on gfx1201, 0.62 on
gfx1100) while decode was clean, the greedy text looked normal, and every op-level oracle
(`test-backend-ops -o GATED_DELTA_NET`, 46/46) passed.  Only a KL/perplexity comparison against a
known-good base saw it.  This file is the prefill-logit gate: the protocol, the canonical command, the
thresholds, and the scope of changes it must run for.

Status: **2026-10-08 (r29) — gate defined and mandatory.**  Run it before tagging a release whose block
set touches a prefill kernel (and after any change to the delivery's prefill dispatch).  The first
recorded run is the r29 issue-#113 example (`WORKLOG.md` 2026-10-08 r29).

**The captured-graph band needs its own base.**  Run the gate at **`--ub 8`** as well whenever the change
touches a copy/ordering path or anything the decode band reaches: the default ubatch does not capture a
graph, and `--ub 8` stores a separate base file (`...-ub8.kld`) so the two never mix (added at r90).

**Automation.**  `scripts/gate-prefill-logits.sh` implements this protocol and is the canonical entry
point:

```bash
./scripts/gate-prefill-logits.sh --record                 # establish the known-good base (writes $BASE.meta)
./scripts/gate-prefill-logits.sh                          # compare the current build against the base
./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED_BF16=0"   # exact-fallback A/B
./scripts/gate-prefill-logits.sh --ub 8 --record   # the captured-graph band gets its OWN base
./scripts/gate-prefill-logits.sh --ub 8            # ... and its own comparison (-ub 8 -b 8)
./scripts/gate-prefill-logits.sh --kv q8_0 --record        # the quantized-KV prefill paths (r156)
./scripts/gate-prefill-logits.sh --kv q8_0                 # ... and their own comparison
./scripts/gate-prefill-logits.sh --device 0,1,2 --sm tensor --record   # the ALL-REDUCE arm (block 01):
./scripts/gate-prefill-logits.sh --device 0,1,2 --sm tensor            #   it has its own base file (r285)
```

**The device set and the split mode are a second point too (r285, `TODO` #323).**  An all-reduce change is
only reachable on >= 2 devices with a `tensor` split, so a release that touches block 01 needs **two** bases:
the canonical single-device `layer` reference and the multi-device `tensor` arm.  The `.meta` pinned `device=`
and `sm=` since r272 and a compare refuses a mismatch, but the FILE NAME did not, so recording one overwrote
the other.  Since r285 the name tags a non-canonical point (`...-dev0-1-2-smtensor.kld`) and leaves the
canonical one untagged (so every base recorded before r285 is still found), and both can be held on disk at
once.  The rule is unchanged: a base is one POINT, so record/label the point a release used.

**A quantized KV cache is a second point (r156, issue #237).**  The gate's default is an f16 cache, which
**skips every quantized-cache prefill path**: the F16 FA prefill **staging arena**, `GGML_CUDA_FA_KV_NATIVE`'s
tile read, and block 05's quantized arms.  `--kv TYPE` passes `-ctk/-ctv TYPE`, stores the base under its own
name (`...-kvq8_0.kld`, with the type pinned in the meta) and is what a change to any of those must be judged
against; the f16 base stays the release reference (the two points differ in base PPL: 5.9325 vs 5.9317 on the
27B Q8_0 wikitext cell).

`--record` writes a `$BASE.meta` sidecar (build ref, model, corpus sha256, ctx/chunks, base sha256, PPL)
and refuses a compare whose model/corpus/settings do not match the base unless `--allow-base-mismatch`
is given.  The manual commands below are exactly what the script runs; use them directly only if the
script's pins do not fit.

## What it catches

Any change that moves the prefill forward pass enough to shift the prompt logits, but not enough to
change greedy decode text or trip a `test-backend-ops` oracle: a new or changed prefill kernel family
(GDN/SSM chunked, flash attention, MMQ/MMVF/MMB prefill GEMMs, MoE prefill), a changed reduction
order, an accumulator or recurrent-state precision change, or a new default-on approximate path.

The coherence gate samples the decode distribution *starting from the post-prefill state*; a uniform
prefill shift is invisible to it.  The MTP gate exercises decode/verify shapes, not the prefill
kernels.  This gate is the missing third axis.

## The protocol

Record reference logits from a known-good build (the previous release, or the exact/sequential
fallback arm), then compare the candidate cell by cell.  `llama-perplexity` reports the mean KL
divergence, the same-top-p fraction and the per-chunk table.

### Pinned model + corpus + flags

| item | value |
|---|---|
| model (gfx1201, 32 GiB) | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| model (24 GiB cards, e.g. gfx1100) | `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| corpus | `/llm/models/wikitext-2-raw/wiki.test.raw` (a code corpus is a useful second arm; issue #113 reproduced on both) |
| flags | `-c 512 --chunks 40 -ngl 99 -fa on -t 8` |
| device pin | `HIP_VISIBLE_DEVICES=0` (single GPU, so the result is card-local) |

**Batch / `n_seq` matters.**  Leaving the batch at its default keeps `n_seq = batch_size / n_ctx > 1`
(4 with `-c 512 -b 2048`), which exercises the multi-sequence path; the r30 `A_sc` stride aliasing bug
only showed up for `n_seqs > 1`, so do **not** force `-b == -c` (`n_seq = 1`) when validating a GDN
change.  Run both settings if in doubt.  (The r29 release shipped with the bug precisely because the
earlier investigation compared at `n_seq = 1`.)

40 x 512 chunks is the r29 calibration: it separates mean KLD 0.03 from 0.0005 in well under a minute
of compute per arm.  Do not shrink it without re-calibrating; a shorter window is noisier.

### 1. Record the base (once per known-good build)

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity -m <model> -f <corpus> \
  -c 512 --chunks 40 -ngl 99 -fa on -t 8 --kl-divergence-base /tmp/prefill-base.kld
```

The base file is large (multi-GB); it is a scratch artifact and is **not** committed.  Record its
sha256, the build id and the base PPL in the release's `WORKLOG.md` entry.

### 2. Run the candidate

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity -m <model> -f <corpus> \
  -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
  --kl-divergence-base /tmp/prefill-base.kld --kl-divergence
```

### 3. Acceptance

| metric | threshold |
|---|---|
| mean KLD | `<= 0.005` |
| same top p | `>= 98.0 %` |
| max KLD | inspect only; no hard limit (a single tokenizer/context edge outlier is not a regression) |

r29 calibration: the clean fp32 chunked GDN arm measured mean KLD **0.00054** / same-top-p **98.8 %**
(gfx1201, 27B Q8_0) and **0.000037** / **99.7 %** (gfx1100, Q4_K_M); the rejected bf16 arm measured
**0.0324** / **93.6 %** and **0.6228** / **79.4 %**.  The thresholds sit ~10x above the clean value and
well below the regression.

### 4. Exact-fallback A/B (when the change adds or edits an approximate kernel)

When the change introduces or modifies a kernel that has an exact or sequential fallback, additionally
compare the candidate's **default** against that **fallback** arm directly (for GDN: `GGML_CUDA_GDN_CHUNKED=0`)
and require the same thresholds.  This is the test that would have caught issue #113: the fallback arm
was clean, the default was not.  A change is only allowed to be a default-on approximate path if this
A/B passes; otherwise it stays opt-in.

> **The A/B must reach the candidate arm (r91).**  `--ab "K=V"` applies `K=V` to the **base** arm only,
> and `run_ppl` inherits the caller's environment — so `GGML_CUDA_GDN_CHUNKED=0 ./gate-prefill-logits.sh
> --ab "GGML_CUDA_GDN_CHUNKED=0" ...` forces **both** arms to the fallback and reports ~0 by
> construction (`-0.000000 / 99.99 %`).  Set the env on the script only for arms that must apply to
> both.  Check that the candidate really took the new path (e.g. tail/branch log line, or an A/B whose
> result is not ~0 for a change that is not bit-exact).

### 5. Self-KLD control first (r91) — the instrument is model-specific

Before interpreting any number, run `--record` (base = candidate) and require it to be ~0.  On gfx1201,
35B-A3B Q3_K_M at `--ctx 2048 --chunks 16` is **not a usable instrument**: three self-KLD runs of the
**unpatched** r90 build (same binary, `GGML_CUDA_ASYNC_INPUT=0`) measured 0.004582 / 97.355 %,
0.004957 / 97.318 % and 0.004639 / 97.336 % — i.e. the model's own run-to-run prefill variation
sits *at* the 0.005 / 98.0 thresholds, so a real 0.002-0.003 regression is invisible and a clean build
can "fail".  `Qwen3.8-27B-Q8_0` (0.000000 / 99.994 %) and `Qwen3.8-Flash-Next-IQ3_XXS` (0.000000 /
100.000 %, 2 GPU) are clean on the same box and are the models to gate on.  A noisy self-KLD is a
*measurement* result, not a gate result — do not tune a threshold to it.

## Family switches: judge at a calibrated point, against a same-kernel control

A **kernel-family switch** (a tile -> WMMA flash-attention row set, a new head cap) does not share a
reduction order with the kernel it replaces, so its divergence *is* the tuning and cannot be driven to
zero by any row parameter. The gate's absolute limits therefore need two extra guards, implemented as
`scripts/gate-prefill-logits.sh --family-switch`:

1. **The point must be one the model is confident about** (base PPL inside the calibrated domain; the
   gate was pinned at base PPL 5.93).  At a point it finds hard, *every* reordering measures far
   outside the limits, so the number says nothing about the candidate.  Measured on gfx1100 with
   gemma-4-26B-A4B-QAT (2026-10-21, `archive/work/fa-rdna3-head512-purity/`):

   | point (gemma-4-26B-A4B QAT, f16 K/V) | base PPL | same-kernel control `-ub 512` vs `-ub 256` |
   |---|---:|---|
   | wikitext-2, `-c 512 --chunks 40` (the default point) | 607 | **0.104989 / 85.000 %** |
   | the model's own in-domain text, `-c 512 --chunks 30` | 13.7 | 0.036654 / 93.660 % |
   | the same at `-c 2048 --chunks 10` (`prompts/fa-family-switch-gemma4-26B-A4B.txt`) | **2.65** | 0.013962 / 97.221 % |

   A benign ubatch change is 21x the KLD limit at the wikitext point.  The model is a *reasoning* model,
   so raw wikitext is 45x outside its confident range; that is why block 05's head-512 rows "failed"
   there (0.0462 / 90.42 %) and pass at the in-domain point (0.00178 / 98.97 %).

2. **The control must be measured on the spot.**  The same kernel at a different ubatch is the floor of
   what a reordering costs at that point; a candidate no worse than it carries no evidence of a defect.
   The mode records the control base next to the fallback base (same arm, `--control-ub`, default 256)
   and always prints it.  **The absolute limits stay the criterion**: the control explains *whose* excess a
   miss is (the switch's or the point's), and `--allow-control-pass` is an explicit opt-in for the narrow
   case of a switch that provably cannot meet them — by default a candidate cannot pass just because the
   point is insensitive.

`--record --family-switch --ab "<feature-off env>"` records both bases and refuses a point whose base PPL
is above `--family-max-ppl` (default 10).  The **op-level accuracy reference** is the second leg, and it
is deterministic: with `FA_DBG_NMSE=1` `test-backend-ops -o FLASH_ATTN_EXT` prints the NMSE already
computed against the CPU backend for every case, so the two families can be compared on the model's own
prefill shape (`hsk=hsv=512, nr23=[8,1], kv=32768, nb=512`).  On gfx1100 the head-512 WMMA rows and the
tile kernel are within a factor 1.3 of each other there (f16 5.99e-5 vs 5.71e-5, q8_0 2.30e-5 vs
1.64e-5, threshold 5e-4), i.e. neither is an accuracy regression; at smaller query widths WMMA is 1.1-2.8x
closer to the CPU reference.

### A base is only valid for the arm it was recorded on (2026-10-21)

§5 above says a noisy self-KLD means the model is not a gate model.  It does not say why, and two
obligations follow once the cause is known.  Both are now decided policy (maintainer, 2026-10-21):

1. **Find the cause before you drop the model.**  35B-A3B's run-to-run prefill variation is not inherent
   rounding: on gfx1201 / ROCm 10.1.0 it is the **hipBLASLt per-process solution choice** on the F32
   `ssm_alpha`/`ssm_beta` and MoE-router GEMMs (`GREEDY-PURITY.md` §44, `archive/work/hipblaslt-bypass/`).  Measured
   at `-c 512 --chunks 8`, 1 GPU: a candidate against a base recorded on the **same unpinned build** gave
   mean KLD **0.007279** / same-top-p **96.618 %**, and the *reference* arm `ROCBLAS_USE_HIPBLASLT=0` gave
   **0.007271** / 96.618 % from that same base — i.e. the gate was reading the lottery, not the candidate,
   and its spread (0.0068-0.0073 here) straddles the 0.005 limit.  Pinning the solution in our own code makes
   this model's self-KLD exactly **0.000000 / 100.000 %** (`archive/work/hipblaslt-bypass/evidence/p3-arm-g-pin-implemented.txt`).
2. **Record the base on the arm that will ship, and treat a deliberate family switch as a new baseline**
   (with `scripts/gate-prefill-logits.sh --family-switch` for the judgement itself: a point the model is
   confident about, and the on-the-spot same-kernel control above as the floor).
   When a default change moves the prefill arithmetic into a different kernel family — a new GEMM family, a
   pinned solution, a different accumulator precision — the old base is no longer the reference.  Re-record it
   **from the new default**, and log the old-vs-new KLD in the release entry as the **family-switch magnitude**
   (for a hipBLASLt pin on 35B-A3B: ~0.0073 / 96.6 %).  Do not compare a new family against an old-family
   base and report the difference as a regression: that is §46's "an absolute gate cannot decide a family
   switch" in its prefill form.  The pinned-vs-pinned control must still read exactly 0.000000, which is what
   makes the number interpretable.

## Scope: when the gate is required

Run it before a release (and after any change to `patches/`) whenever the block set touches:

- the GDN/SSM chunked prefill (`gated_delta_net*`);
- the flash-attention prefill kernels or their dispatch;
- MMQ / MMVF / MMB prefill GEMMs or a new quant path;
- MoE prefill, the shared-expert band, or the expert cache;
- any default-on approximate path (a precision change or a reduction-order change).

Not required for decode-only changes (the MTP gate covers those) or doc-only changes.

## Recording

Add the base build id/hash and the candidate mean KLD / same-top-p for each arm to the release's
`WORKLOG.md` entry.  If the gate fails, the change does not ship as a default; either fix it, make it
opt-in, or flip the existing default off (the r29 resolution).
