# The host-expert prefill shape has a text oracle again — and its prefill logits are a recorded KLD point (r310, `TODO` #344)

**Claim.** On the Flash-Next host-expert shape (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`, 1 GPU, `-ncmoe 47`) the
greedy text **is** reproducible once the heat-map prior and the BLAS library are pinned, so the shape no longer
needs to be excluded from text comparisons; and its prefill logits are a **usable prefill-logit (KLD) point**
(`0.000000` self-KLD), so it can be judged on the gate instead.  With both instruments in place the r306
admission rule's altered case on this shape — "a 512-token step becomes a 2048-token step" — reads **byte-identical
text** between the r305 and the r309 builds.

Why this was open: `TODO` #344 (r306).  The admission rule (block 16, `-ib`) was verified where a text oracle
exists (the dense 4B: identical text in all four arms), but on the host-expert shape six identical-config runs
read **four different texts** and the GGUF then shipped **no `.moe-heatmap`** for `MOE_EXPERT_HEATMAP=load` to pin
the residency with (`GREEDY-PURITY.md` §89), so the shape's default-path numerics under r306 were unmeasured.

**Both premises have changed, and neither is taken on faith here:**

* This GGUF **now ships a heat map**: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.moe-heatmap` (601 376 B) sits next to
  the two shards, i.e. the r75 **unified** name (`<base>.moe-heatmap`, split marker stripped), which is exactly
  what `MOE_EXPERT_HEATMAP=load` reads.
* The spread **does not reproduce** on the r309 build: 10 llama-cli runs and 4 server starts across three
  (heat map, BLAS library) states were each stable *within* their state.

## 0. The box and the shape

| item | value |
|---|---|
| GPU | 1 x R9700 (gfx1201, 34 GiB), ROCm 10.1.0 (`rocm-10.1.0-gfx120X`) |
| CPU / RAM | 9950X3D, 32 threads, 91 GiB RAM (74.9 GiB available; the run's peak RSS is **71.1 GiB**) |
| model | `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (2 shards, 75.8 GB) |
| flags that reach the tier | `-ngl 99 -fa on --n-cpu-moe 47 --load-mode dio --lazy-mode mlockr -ot 'per_layer_token_embd.*=CPU'` |
| drain | `HIP_VISIBLE_DEVICES=0`, `ROCBLAS_USE_HIPBLASLT=0`, `MOE_EXPERT_HEATMAP=load` |

The census says which tier served it: arena **18 988.1 MiB**, host pool (L2) **141 tensors / 40 012.5 MiB pinned,
h = 1.0000 (34 866/34 866), fills = 72 192, evictions = 0** — a *covering* pool, so no expert is ever missing
and the residency question is settled before the first token.

## 1. The text oracle: pinned heat map + pinned BLAS library

`llama-cli`, `prompts/prose-rdna-boosts.txt` (16 074 B, 5246 tokens), `-c 8192 -n 64 --seed 42 --temp 0
--no-warmup --single-turn --no-display-prompt`, text hashed with `scripts/extract-generated.py`
(`<chars> sha=<12 hex>`):

| heat map | `ROCBLAS_USE_HIPBLASLT` | runs | text |
|---|---|---:|---|
| `MOE_EXPERT_HEATMAP=load` (frozen prior) | `0` | 3 | `296/91ef8940d66e` **3/3** |
| `MOE_EXPERT_HEATMAP=0` (no prior at all) | `0` | 3 | `296/91ef8940d66e` **3/3** |
| `MOE_EXPERT_HEATMAP=load` | unset (`hipBLASLt` on, the default) | 4 | `302/fb8da5c09068` **4/4** |

Two conclusions, and the second one is the one that matters for every other gate:

1. **The residency prior is not the carrier of the old spread here.**  A frozen prior and no prior at all read
   the *same* text — because the pool covers the expert set (`h = 1.0000`), residency selects nothing on this
   shape at this `-c`.  The r306 spread was therefore not reproduced by re-running the same configuration, and
   the two state changes since then are r307's pass scoping (it removed 138 whole-table refills — 206-450 MiB
   each, ~27.8 GiB, ~10.4 s — from every 30-34-token prompt, `INVARIANTS.md` r307) and this box's pins.
   Bisected in §1b (42 runs, twelve states): the prior, the arena size, the residency breadth and r307's repair
   are all eliminated, and the coverage is now a gate rather than a claim (§1c).
2. **The BLAS library IS part of the text's identity on this shape**: `hipBLASLt` on reads `302/fb8da5c09068`,
   off reads `296/91ef8940d66e`, each reproducible.  That is §44's rule reaching a *host-expert* shape as well,
   so a text arm here pins `ROCBLAS_USE_HIPBLASLT` exactly as it pins the heat map (`r284`), and a comparison
   that pins only one of the two is comparing states, not builds.

**The recipe for a text arm on this shape** (and the one `MANIFESTS.md` gate 1 now names):

```bash
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0 \
  ./build-rocm/bin/llama-cli -m .../GSQ-IQ3_XXS-00001-of-00002.gguf -f prompts/prose-rdna-boosts.txt \
  -ngl 99 -fa on -c 8192 -n 64 --seed 42 --temp 0 --no-warmup --single-turn --no-display-prompt \
  --n-cpu-moe 47 --load-mode dio --lazy-mode mlockr -ot 'per_layer_token_embd.*=CPU'
# repeat, and report the spread (here: none)
```

## 1b. The bisect: what the r306 spread was NOT (42 runs, twelve states)

The r306 note's own explanation (§89 residency) did not survive measurement, so the states were eliminated one
at a time instead of assumed.  All arms: llama-cli, the same 5246-token prompt, `-n 64 --seed 42 --temp 0`, the
tier flags, 1 GPU; `-c 16384 -b 2048 -ub 2048` is the r306 arm's 5-step shape (and it really is a **partial**
arena there: `h = 0.4593`, 6 629 fills, `evictions = 0`):

| state varied | runs | result |
|---|---:|---|
| heat map **mutating** (`MOE_EXPERT_HEATMAP=1`, a writable `MOE_EXPERT_HEATMAP_DIR`) | 5 | one text, 5/5 (the map was rewritten each run: 588 848 B) |
| heat map **frozen** (`=load`) | 3 + 3 + 2 | one text (2-3/3 per config) |
| heat map **absent** (`=0`) | 3 | one text |
| arena **8 GiB forced** vs the 23.8 GiB auto size (`MOE_EXPERT_CACHE_MIB`), library off | 2 + 2 | one text both ways |
| arena 20 GiB / 8 GiB forced, library **on** | 2 + 2 | one text both ways |
| build **r305** `73295b281` (pre-r306 admission, pre-r307 repair) vs **r309** | 6 + 6 | one text both (and the same text as each other) |
| `ROCBLAS_USE_HIPBLASLT` **on** vs **off** | 4 vs 3-4 | **different text**, each reproducible within its state |

So: not the prior, not the arena size, not the residency breadth, not r307's repair, not the library's free-VRAM
sensitivity (the arena-size arm *is* a 16 GiB VRAM delta and moved nothing).  What could not be reconstructed is
that session's **prompt** and **box state** — and the residual explanation is a greedy near-tie flip (`§46`/`§61`:
a hash is not an instrument for a near-tie, a spread over repeats is), which is exactly why the coverage below is
mechanical rather than a quoted hash.

## 1c. The gate: `scripts/gate-host-expert-text.sh`

The durable form of "this shape has an oracle" is a **repeat check**, not a sentence in a record.  The new gate
(defaults = this shape, `RUNS=3`, `PIN="MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0"`) runs the arm N times
and fails on **any** spread, refuses to run at all if a pin is missing from `PIN` (the r303/r309 "an arm that
leaves a defaulted switch alone is two arms" mistake), reports the flipped-library arm as *evidence* (it must
show a different text, or the pin is inert for that prompt — information, not a defect), and `--kld` adds the
prefill-logit point.  Measured on r310:

```
./scripts/gate-host-expert-text.sh                # 3/3 identical 319 chars sha=a455bfe7267a; library arm 336/e98c06d6427a
./scripts/gate-host-expert-text.sh --quick --kld  # 2/2 identical + KLD 0.000000 / 100.000 %
```

## 2. The r306 case: a 512-token step becomes a 2048-token step — same text

The rule's altered case on this shape is the **step width**, not the ubatch width: at `-b 2048 -ub 512` a lone
prompt's ubatch boundaries are unchanged (consecutive 512 blocks either way, `archive/work/prefill-chunk-adaptive/`
§3), so only the *step* count differs.  r305 capped every step at one `n_ubatch`; r309's default takes the whole
`-b` window when nothing decodes.  The cap is no longer reachable on the r309 binary (that is the point of r306),
so the A/B is **two binaries**: r305 (the canonical commit `73295b281` in a scratch worktree, built with the
same `~/bin/build-llama-rocm`) and r309 (`build-rocm`, `87d3629df`).

`llama-server -np 1 -c 8192 -b 2048 -ub 512 -lv 5 --no-warmup`, one `POST /completion` of the same 5246-token
prompt (`n_predict 64`, `temperature 0`, `seed 42`, `cache_prompt false`); the text comes over HTTP, so `-lv 5`
cannot pollute it.  Steps are the `fair share: added=` values from the log:

| build | admission steps (prompt tokens per step) | text |
|---|---|---|
| r305 `73295b281` (cap on every step) | `512 x 9, 122, 512, 4` (12 steps) | `98/65ba627f8eca` **2/2** |
| r309 `87d3629df` (default, wide when nothing decodes) | `2048, 2048, 634, 512, 4` (5 steps) | `98/65ba627f8eca` **2/2** |

The mechanism is provably live on both arms (12 x 512-wide steps against 2 x 2048-wide), the walls are 22 s and
21 s (load included), and the text is **byte-identical, 4/4**.  Both arms' tail is the same `512, 4` — that is
the prompt-checkpoint placement (`4 + n_ubatch`), not the cap.

## 3. The prefill-logit point: `0.000000` self-KLD, and r305 -> r309 moves nothing

`scripts/gate-prefill-logits.sh` on the shape, canonical flags (`-c 512 --chunks 40 -fa on -t 8`, f16 KV,
wikitext), `--device 0`, the tier flags in `--extra` and the two environment pins in the new `--pin`:

```bash
MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0 ./scripts/gate-prefill-logits.sh \
  --model .../Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --device 0 --record \
  --pin "MOE_EXPERT_HEATMAP=load,ROCBLAS_USE_HIPBLASLT=0" \
  --extra "-ncmoe 47 --load-mode dio --lazy-mode mlockr -ot per_layer_token_embd.*=CPU"
```

| arm | mean KLD | same top p | base PPL |
|---|---:|---:|---:|
| r309 build, self-test (`--record` then compare) | **-0.000000** | **100.000 %** | 2.8553 |
| r305 build `73295b281` against the r309-recorded base | **-0.000000** | **100.000 %** | — |

Base: `prefill-logit-gfx1201-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002-xncmoe-47-load-mode-dio-lazy-mode-mlockr-ot-per-layer-token-embd.-CPU-pMOE-EXPERT-HEATMAP-load-ROCBLAS-USE-HIPBLASLT-0.kld`,
sha256 `effe46712734…`, 40 x 512 tokens, ~35 s per pass, peak RSS 71.1 GiB, kernel journal clean.

Two things this buys, beyond the r306 verdict:

* **The instrument is valid on the tier.**  `--record` compares the base to itself across two *processes*, so
  `0.000000 / 100.000 %` says the logits are bit-reproducible: the residency mechanism that makes the shape's
  *kernels* differ elsewhere does not reach the logits here.  A shape that cannot be re-read cannot be a base.
* **The r305..r309 delta is measured on the tier, not argued.**  That window contains r307 (the take-over
  promise scoped to its staging pass — a `moe_cache_repair_copy` on the prefill path), which is exactly the class
  of change the gate exists for; it reads nothing on 40 chunks of host-expert prefill.

The pins do **not** move the logits: the unpinned record of the same point produced the same base sha256
(`effe46712734…`), which is why they belong in the point's *identity* (the file name + meta `pins=`) and not in
the measurement.

## 4. What this closes, and what it does not

* `TODO` #344 is closed on **both** alternatives it named: the text oracle exists with a stated spread (none,
  14/14 runs), and `gate-prefill-logits.sh` runs on the model (`0.000000`).
* `scripts/gate-prefill-logits.sh` gained `--pin "K=V,..."` (both arms, folded into the base file name, recorded
  in the meta, refused on mismatch) — without it a base recorded under one BLAS/heat-map state could be found
  and compared against another, which on this shape is a text *and* a kernel-set difference.
* **Out of scope by existing doctrine, not left open:** every arm here is single-slot.  A two-slot pair and the
  wide-step multi-prompt interleaving (`split_equal`) are non-oracles by `GREEDY-PURITY.md` §63/§38 (per-config
  determinism is a *single-slot* statement; `--kv-unified` serving is not deterministic), which predates this item
  — such a shape is judged by KLD plus throughput, and this record claims nothing else about it.
* **Bisected, and it is not any of the obvious candidates:** §1b eliminates the prior, the arena size, the
  residency breadth and r307's repair across 42 runs; only the BLAS library state moves the text, reproducibly.
  If a future release sees a spread again, `scripts/gate-host-expert-text.sh` fails instead of a reading being
  argued about, and the first things to compare are `ROCBLAS_USE_HIPBLASLT` and the heat map's mode.
