# MTP CPU-spin: automatic single-threading of tiny CPU split graphs (OP-1)

**Status:** the runtime half of OP-1 is **DONE** and folded into the WIP set as
[`patches/0029`](patches/0029-gap-closing-WIP-run-tiny-CPU-split-graphs-on-the-calling-thread.patch).
No `OMP_*`/`KMP_*` env var is needed any more: a plain run with the fixed build keeps the MTP decode
CPU at ~1.2 cores and beats the passive-wait reference.  Campaign branch tip tree (r13-based,
`closing-gfx1201`): **`fa9cf6d1e654333d458ade3655c4a0d540225827`** (28 patches; `campaign-all.patch`
regenerated).

> **Pre-existing base drift (not caused by `0029`).**  The campaign's base is **delivery r13**
> (16-block tree `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`, r13+beta `79136a15…`) but this repo's
> `patches/` + `release.json` are still **r12** (`8a80535e…`), so a *fresh* apply through
> `scripts/apply-all.sh` + `beta/` lands on r13+beta `bca69f23…` and closing tip `cb937fe4…`.  The
> **only** delta is one hunk in `common/speculative.cpp` (the `gemma4-assistant` `is_mem_shared`
> guard, r13), verified by `git diff`; `0029` is orthogonal (it touches only
> `ggml/src/ggml-cpu/ggml-cpu.cpp`) and applies cleanly on both.  Both trees were checked:
> *fresh r12-based apply* 28/28 `git am`, tree `cb937fe4ca60d5e2df561d133f550de886edf52a`;
> *campaign r13 branch* tree `fa9cf6d1…`.

The structural half (get the input/PLE embedding off the CPU entirely) and the draft-sampler offload
(OP-1.4) are **still open** — see the end.

---

## 1. The problem (session 6 root cause)

Under `-sm tensor` on the discrete 3-GPU box, the scheduler puts a **tiny CPU split** at the front of
every graph.  It holds the host-mapped input/PLE embeddings' `GET_ROWS`:

* the target graph: `GET_ROWS model.input_embed` (`token_embd.weight`, 341 MiB, `CPU_Mapped`);
* the MTP draft graph: 3 × `CPY` + 3 × `SCALE` (0 bytes, the HC init state) + `GET_ROWS mtp_tok_embd-48`.

The 27 GiB `per_layer_token_embd` cannot fit in the 96 GiB of VRAM beside the 89 GiB of weights, and
the default `--load-mode mmap` turns the selected `ROCm_Host` back into a plain `CPU_Mapped` buffer
(`llama-model-loader.cpp`: "avoid using a host buffer when using mmap"), so the gather is dispatched
to the CPU backend.  `0024`/`0025` only cover the single-device / APU cases.

Each such graph is an OpenMP parallel region.  With the default active wait the runtime keeps the
idle workers spinning for `KMP_BLOCKTIME` (200 ms, set in `ggml_cpu_init`).  Plain decode runs ~55
graphs/s, so the pool settles between them (~2 cores); MTP calls `llama_decode` `n_max+1` times per
token (~250–350 graphs/s), the pool never sleeps, and all 15 worker threads spin.  Measured cost on
the current tree (qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, prose, `-c 16384 -n 1500`, seed 42):

| arm | Generation | CPU | acceptance |
|---|---:|---|---:|
| default | **81.0 t/s** | ~15.0 cores | 0.84791 |
| `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` | 108.5 t/s | 0.4 cores | 0.84791 |
| `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` (new kill-switch) | 85.1 t/s | 14.9 cores | 0.84791 |

i.e. ~25 % of qwen4exp MTP throughput and 14 cores go to the idle-wait spin.  Acceptance is
byte-identical in every arm, so it is pure scheduling waste.

## 2. The fix (`0029`)

`ggml_backend_cpu_graph_compute` chooses its own thread count from the graph:

```c
static int ggml_backend_cpu_graph_n_threads(const struct ggml_cgraph * cgraph, int n_threads);
```

A graph with **≤ 32 nodes and ≤ 16 MiB of node outputs** runs inline on the calling thread
(`n_threads = 1` → `ggml_graph_compute` takes its non-OpenMP branch, no parallel region, nothing to
spin).  Everything larger keeps the configured count.  `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1`
restores the old behaviour for A/B and bisection (default-on, disable-only, per `AGENTS.md`).

Why the bounds: the targeted split graphs are 1–7 nodes of a few hundred KiB each, and the whole
spec-decode verify band stays inside them (`≤ 16 tokens × 640 KiB`, even the PLE-shaped gather).  An
ordinary model graph — CPU-only inference, or any prefill-sized gather (≥ 64 tokens → ≥ 40 MiB) — is
far above both bounds and is untouched.  The affected ops (`GET_ROWS`/`CPY`/`SCALE`) are
thread-count invariant, so the output is bit-identical.

**Measured, fixed build, default env (`gfx1201`):**

| arm | Generation | CPU | acceptance |
|---|---:|---|---:|
| MTP `n3`, prose, `-n 1500` | **111.7 t/s** | 1.2 cores | 0.84791 |
| MTP `n3`, prose, `-n 1500`, kill-switch | 85.1 t/s | 14.9 cores | 0.84791 |
| plain decode `-n 800` | 52.9 t/s (vs 53.0 kill-switch) | 1.4 / 1.4 cores | — |

**`llama-server` (the acceptance config, no env, `-t 15`, `/completion` `n_predict 800`, prose):**

| arm | predicted t/s | CPU |
|---|---:|---|
| default (fix) | **108.8** | 1.4 cores |
| `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` | 98.2 | 12.0 cores |

So the fix **beats the passive-wait reference** (111.7 vs 108.5 on the CLI) with the CPU quiet, the
server path behaves the same, and plain decode is unchanged.

## 3. Purity and regressions

* **Intra-build purity:** `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3`,
  prose `-n 300`, no `-lv`: both **`a79d0d14855b`** (943 chars).
* **Heuristic invariance:** the same `draft-mtp n3` run with
  `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` is also **`a79d0d14855b`** — the heuristic is a pure
  scheduling change.
* **CPU-only unaffected:** 4B Q8_0 `-ngl 0` `tg64` 9.69 vs 9.60 t/s (r = 5) — the full model graph is
  never inside the bounds.
* **GPU prefill unaffected:** qwen4exp `pp2048` 2448.5 (fix) vs 2466.6 (kill-switch) — 0.7 %, inside
  the fix arm's ±27.5; the prefill CPU split is 20 MiB at ub 2048, above the bound by construction.

## 4. OP-2 / OP-3 — four-axis + `n7`/`n8`/adaptive re-baseline (fixed build, **no** env)

qwen4exp IQ4_NL + `mtp-…-shared-Q8_0.gguf`, 3-GPU `-sm tensor`, q8_0 KV, `-c 16384`, `-b/-ub 2048`,
`-n 3000`, seed 42, reasoning pinned per axis (`benchmarks/mtp-adaptive-methodology.md` rule 0).
Fixed `draft-mtp --spec-draft-n-max 7|8`; adaptive = `draft-mtp-adaptive --spec-draft-n-max 8`;
`none` = plain decode (`-md` omitted — `-md` on a plain arm aborts).  Generation t/s (parenthesised =
draft acceptance):

| axis | prompt | none | n7 | n8 | adaptive |
|---|---|---:|---:|---:|---:|
| R reasoning | `reasoning.txt` | 54.6 | 79.6 (0.365) | 78.1 (0.352) | **86.8 (0.592)** |
| C code | `code-python.txt` | 54.8 | **139.2 (0.764)** | 138.8 (0.732) | 136.6 (0.743) |
| P prose | `prose-rdna-boosts.txt` | 53.8 | **128.1 (0.706)** | 124.1 (0.655) | 123.0 (0.685) |
| K recall | `recall.txt` | 52.4 | 163.6 (0.963) | **167.4 (0.957)** | 154.3 (0.967) |
| X phase-switch | `code-reasoning-mixed.txt` | 54.7 | **99.2 (0.490)** | 84.6 (0.390) | 98.8 (0.661) |

CPU was quiet in **every** arm (steady mean 0.7–1.7 cores; the one-off per-run maxima are load-time
artefacts).

**What changed vs the session-6 table (which was measured before `0028` and with the passive env):**

| axis | n7 old → new | n8 old → new | adaptive old → new |
|---|---:|---:|---:|
| R | 75.4 → 79.6 | 57.1 → **78.1** | 83.8 → 86.8 |
| C | 131.1 → 139.2 | 107.3 → **138.8** | 108.8 → 136.6 |
| P | 119.3 → 128.1 | 95.1 → **124.1** | 107.1 → 123.0 |
| K | 156.4 → 163.6 | 129.2 → **167.4** | 129.1 → 154.3 |
| X | 92.2 → 99.2 | 62.9 → **84.6** | 95.4 → 98.8 |

The old `n7 >> n8` gap was almost entirely the **`0028` W=9 MMVQ→MMQ cliff**, not the acceptance
curve: with that fixed the two depths are near-tied.  **`n_max 8` is no longer dominated** — it now
*wins* on recall (167.4 vs 163.6) and is within noise on C, while `n7` keeps prose/X and the adaptive
controller keeps reasoning/phase-switch.  The `gfx1201` `n_max 8` question is therefore **addressable
on this box**, not an artefact of the gfx1151 box; the depth choice is workload-dependent, which is
exactly what the adaptive ceiling-8 controller exploits on R/X.

## 5. OP-5.1 — `0013` (`idx_relu_sum`) is redundant on RDNA4

Temporarily widened the `0013` call-site gate to RDNA4 (`ggml-cuda.cu`, `GGML_CUDA_CC_IS_RDNA3_5(cc)
|| GGML_CUDA_CC_IS_RDNA4(cc)`), rebuilt, and ran qwen4exp `pp8192` with
`GGML_CUDA_IDX_RELU_SUM_LOG=1`:

* **`0016` default ON:** the matcher fires **0** times — the fused `LIGHTNING_INDEXER` op has already
  replaced the `mul_mat + relu + head-sum` chain (`use_wmma` at `n_tps >= 128`, `idx_dim == 128`,
  `n_idx_h == 4`).
* **`LLAMA_QSA_SCORE_WMMA=0` (0016 off):** the matcher fires **4** times
  (`nb=640 heads=4 rows=512`) — the chain and the matcher are both intact; `0016` is what removes it.

**Verdict: nothing to port.**  The RDNA4 arm would be dead code while `0016` is default ON.  The gate
change was reverted (build restored).  `0023`/`0011` stay parked as before.

## 6. Still open (unchanged by this session)

* **OP-1 structural half** — putting the input/PLE `GET_ROWS` on a device (shard `per_layer_token_embd`
  across the tensor-split devices, or run the gather on the device over the host-mapped pointer) would
  remove the CPU split rather than serialise it.  `0029` makes it a *latency/scheduling* non-problem,
  so this is now optional.
* **OP-1.4** — `llama_context::set_sampler` still rejects the backend sampler under
  `SPLIT_MODE_TENSOR`, so the draft `top_k(10)` chain round-trips through the CPU each draft step.
* **OP-3** per-type MoE band floor; another odd-row dense model.
* **OP-4** `llama-imatrix`, the `0001`/`0008` isolated A/Bs, the M-RoPE image case.
* **OP-5.2** `0011` HC BF16 streams (needs the `0027` meta `graph_optimize` path; now present).

## 7. Harness (new, kept)

* `tools/mon.py` — per-thread CPU sampler (`/proc/<pid>/task/*/stat` deltas); prints
  `SUMMARY cpu_cores mean=… p50=… max=… steady_mean=… busy_threads(last)=…`.
* `tools/mtp-run.sh <tag> <outdir> [env…] -- <cmd…>` — runs a command under the sampler and writes
  `<tag>.log` + `<tag>.cpu`.
* `tools/matrix-axis.sh <axis> <prompt> <reasoning>` — the four-mode axis comparison above.
* `tools/server-mtp.sh <tag> <outdir> [env…]` — the `llama-server` health-wait + `/completion` MTP
  check under the same sampler (used for the acceptance-config A/B above).
* Raw logs under `tools/runs/` (git-ignored).
