# Graph replay up to 16 tokens — `GGML_CUDA_GRAPH_MAX_TOKENS` (r326, block 10; contributor PR #336)

Dated: 2026-11-20 (local).  gfx1201 (1× R9700 32 GB for the arms below, 3× for the `-sm tensor` arms),
ROCm 10.1.0 (`/opt/rocm-10.1.0-gfx120X`), ccache on.  Release: `v17-4f9dfa0e3-r326` (block 10 amended).
Contribution: [`stew675/llama-cpp-rdna-boosts#336`](https://github.com/stew675/llama-cpp-rdna-boosts/pull/336)
(`@overdoingism`, cut on r324 against `4f9dfa0e3`), assessed on r325 and folded with the corrections below.

## The question

`ggml_cuda_graph_is_multi_token()` has compared a graph's token count with `MMVQ_MAX_BATCH_SIZE` (**8**)
since r20: at or below 8 tokens a graph is a fixed shape every step, so HIP graph replay pays; above it the
r20 heuristic assumed "true pre-fill, a varying ubatch, where capture never amortises".  A multi-slot server
under `LLAMA_SPEC_ALIGN` breaks that assumption — 4 slots × `draft-mtp n3` is a **fixed 16-row** verify every
step — so the band is now a knob, `GGML_CUDA_GRAPH_MAX_TOKENS`, **default 16** (`8` restores r325 exactly).

Two things had to be measured before it could ship as a default: what the band is worth where the width
repeats, and what it costs where the width *moves* (block 02's per-sequence depth, a deep fixed draft, the
n-gram family).

## 1. It is inert at the delivery's default drafting (both device counts)

`--spec-draft-n-max 7` (the delivery default) can produce at most an 8-row verify, so no 9..16-row graph
exists and the census is identical at both settings — 27B UD-Q4_K_XL, 600 greedy tokens, same-seed text:

| arm | `GGML_CUDA_GRAPH_TRACE` census | text |
|---|---|---|
| 1 GPU, `=8` | `direct=53 replay=1713 capture=18 properties_changed=14` | `2640 chars sha=9ecdbc5ff34b` |
| 1 GPU, default (16) | **identical** | **identical** |
| 3 GPU `-sm tensor`, `=8` | `direct=3657 replay=21819 capture=813` | — |
| 3 GPU `-sm tensor`, default | **identical** | — |

`ngram-mod` (a variable-width drafter) is inert too: 3 GPU `-sm tensor`, `--spec-draft-n-max 15`, four runs
across both settings all read `direct=4245 replay=111069 capture=1548 properties_changed=771`.

## 2. What it is worth where the width repeats (4-slot server, 16 rows every step)

`archive/work/spec-batch-budget/multislot-mtp-probe.py`, one R9700, `-np 4 --kv-unified --fit off`,
`-c 14048`, 4 × 3000 greedy tokens (`ignore_eos`, `cache_prompt false`), q8_0 K/V, `-fa on`, `-b/-ub 2048`,
`ROCBLAS_USE_HIPBLASLT=0`, order **A B B A A B**, then four more interleaved pairs (8 runs/arm total);
A = `GGML_CUDA_GRAPH_MAX_TOKENS=8`, B = default.

| model | A (=8) | B (16) | B vs A |
|---|---|---|---|
| 27B UD-Q4_K_XL | 179.2 / 177.7 / 177.7 / 177.8 (mean **178.1**) | 185.8 / 184.1 / 184.8 / 182.6 (mean **184.3**) | **+3.5 %** (every B run above every A run) |
| 4B Q8_0 (state-matched pairs) | 640.8 / 641.1 | 652.1 | +1.8 % |

Census, same arm (this is the mechanism, and it reproduces the submitter's counts within a couple of calls):

| arm | `direct` | `replay` | `capture` | `properties_changed` |
|---|---|---|---|---|
| A (=8) | 1973 | 3210 | 62 | 50 |
| B (16) | 180 | 4746 | 156 | 97 |

> **Measurement note.** These numbers are only reproducible with **no other CPU user on the box**.  The first
> pass of this arm ran with a browser resident and read A 176.8-184.3 / B 179.1-183.3 — a ±4 % spread that
> hides a 3.5 % effect.  With the browser gone the two arms separate cleanly and the spread inside an arm is
> 0.8 % (A) / 1.7 % (B).  On this box the 4B 4-slot arm additionally has two acceptance states at 0.98 / 0.88
> (mean draft len 3.95 vs 3.69), so only state-matched 4B pairs are quoted — the same class of instability as
> `TODO` #349.

## 3. `llama-batched-bench -npl 1,4,8,16`

`-c 8192 -b 2048 -ub 512 -npp 128 -ntg 128`, q8_0 K/V, `-fa on`, S_TG t/s, order A B B A:

| model | npl | A (=8) | B (16) | B vs A |
|---|---|---|---|---|
| 27B | 1 / 4 / 8 | 31.84 / 99.75, 99.90 / 157.04, 156.89 | 31.84 / 99.73, 100.01 / 156.98, 156.89 | inert (±0.1 %) |
| 27B | **16** | 242.69, 242.49 | 245.57, 245.44 | **+1.2 %** |
| 4B | 1 / 4 / 8 | 102.17 / 343.00, 345.18 / 532.85, 533.00 | 101.96, 102.06 / 342.98, 343.26 / 532.68, 532.79 | inert (±0.4 %) |
| 4B | **16** | 799.06, 797.71 | 807.09, 808.29 | **+1.2 %** |

Graph calls per run: `140 / 378 / 3` (A) -> `13 / 504 / 4` (B).  The submitter's ROCm 10.0.0 box reported
**+2.5 % / +4.1 %** for the `npl 16` row; on ROCm 10.1.0 the delta is a third of that and the absolute levels
are ~2-4 % lower in *both* arms, i.e. this ROCm line launches cheaper.  The direction and the "1/4/8 inert"
claim reproduce; the magnitude is library-dependent, which is why it is recorded per ROCm line.

## 4. What it costs where the width moves — and why that arm cannot be judged here

3 GPU `-sm tensor`, `--spec-draft-n-max 15` (a deep draft whose verify width moves), 27B, 400 greedy tokens:
captures rise **150-240 -> 1350-1430** (9×) and `properties_changed` 279 -> 1053-1926, i.e. a moving width
pays a capture per new width, exactly as the code comment says.

**The rate in that shape is bimodal in both arms** (9 runs span 46.5-57.6 t/s, and the `-lv 4` acceptance
differs between the states: 0.1395 vs 0.166-0.168).  Above an 8-row verify this set's text is not
run-to-run stable (`GREEDY-PURITY.md` §11), and the draft depth adaptation follows the acceptance that the
drifting text produces — so neither arm's rate is attributable to the knob.  A "only capture when the width
repeated" prototype does **not** remove the churn: the depth saturates at 15, so the width stays 16 while the
FA mask extent moves each step.

This is the accepted limitation of the band (recorded in `ENVIRONMENT.md` §7.4): it needs
`--spec-draft-n-max >= 8`, i.e. the user has already left the purity band, and the kill switch
(`GGML_CUDA_GRAPH_MAX_TOKENS=8`) restores r325 behaviour exactly.

## 5. The pre-fill class (not a verify, and not in the original submission)

The band is a token count, not "a verify": a **9..16-row pre-fill ubatch** (a short prompt, a prefill tail, a
small `-ub`) is captured too.  With `-ub 12` over a 5298-token prompt: **21** captures of `n_tokens=12` by
default, **0** at `=8`, and the prefill reads **+1.2 %** (`269.6 / 269.4` -> `272.7 / 272.8` t/s).  That is
why the r326 change also had to make the FA staging arena capture-safe: above the 8-row band a
native-capable K/V cache *stages*, so a 9..16-row staged FA op can now be inside a captured graph.  The arena
is no longer grown from inside a capture (`ggml_backend_cuda_context::capturing`); the call reads the K/V
cache natively instead, which is the same arithmetic and the same tokens.  Gate:
`scripts/gate-graph-replay-width.sh` arm D (the capture-time staged call is logged once and counted).

## 6. Gates run for r326

* `scripts/gate-graph-replay-width.sh` — **PASS** (reach: 12-14 captures at `n_tokens=16` with `replay` 986 vs
  `direct` 52; kill switch: 0 captures at 9..16 and fewer graphs captured overall; delivery default:
  **identical census and byte-identical text**; pre-fill: 21 captures at `n_tokens=12` vs 0 with the guard's
  line logged once; GPU faults clean).  **Negative-tested**: the same gate on the shipped r325 binary fails
  3 of its checks (no 9..16 capture in the default arm, the "kill switch" changes nothing, no pre-fill
  capture).
* `scripts/gate-prefill-logits.sh` — **PASS** (mean KLD `0.000000`, same-top-p `100.000 %`): the band changes
  *which* graphs are replayed, not a kernel or an arithmetic.
* MTP Protocol A (`benchmarks/mtp-adaptive-methodology.md`) — inert at the default depth, same text.
* `llama-batched-bench` — table 3 above.
* `scripts/gate-per-block-build.sh --fast` (all 18 rungs) and `scripts/validate-set.sh` — see `WORKLOG.md`
  261120.003.r326.

## Where the numbers came from

Reproduce with: the probe (`--env GGML_CUDA_GRAPH_MAX_TOKENS=8` vs unset, `--env ROCBLAS_USE_HIPBLASLT=0`),
`GGML_CUDA_GRAPH_TRACE=1` for every census, `ggml_backend_cuda_context`'s `capturing` line for arm D of the
gate, one fresh process per run, and nothing else running on the CPU (see the measurement note).  Prompts:
`prompts/prose-rdna-boosts.txt` (16074 B, 5298 tokens, sha256 `fabdec65…`) and the probe's per-slot prefix.
