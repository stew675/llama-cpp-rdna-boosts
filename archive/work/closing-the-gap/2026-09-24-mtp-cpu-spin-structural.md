# MTP CPU-spin: the structural fix (input embedding off the CPU) — **investigated, opt-in**

**Status:** the structural fix **works** (it removes the CPU split entirely) but it is **not** the
throughput optimum, so it is folded as an **opt-in** (`patches/0030`, `LLAMA_DEVICE_INPUT=1`) and the
shipped default stays the runtime fix from [`0029`](2026-09-24-mtp-cpu-spin-automatic.md) plus the
upstream host input placement.

**Verdict in one line:** on gfx1201 the Meta-split GPU gather is **~2.6 % slower** MTP than the
single-threaded CPU gather, so `0029` is the right default; `0030` is the structural alternative and a
bisection knob.  Campaign tip tree with both: **`99b429a60d441f814c84737cfa57803bc15a2f6d`** (29
patches, verified by a fresh r13 + beta + closing `git am`).

---

## 1. What the CPU split actually is (established in §2.1 of the brief)

For qwen4exp under `-sm tensor` on the discrete 3-GPU box the scheduler peels a tiny CPU split off each
graph.  In the *target* graph it is one `GET_ROWS` on `token_embd.weight` (`model.input_embed`); in the
MTP *draft* graph it is the same table (`mtp_tok_embd-48`, the shared head borrows the target's
`token_embd`) plus three 0-byte `CPY`/`SCALE` state copies.

The `per_layer_token_embd` (`ple_embd`, ~27 GiB) is **not** a graph op: `build_inp_ple()`
(`src/models/qwen4exp.cpp`) host-gathers it in `set_input` whenever its buffer is host memory
(`host_gather`) or the managed lazy reader is active.  So the only thing that needs moving is the small
`token_embd` table — a fact §2.1.2's candidate 3 ("shard the 27 GiB table") masks.

`token_embd` lands in host memory for two compounding reasons: the input layer is always assigned the
CPU buffer list (`llama_model.cpp`, `dev_input`), and the `use_mmap` override in
`llama_model_loader.cpp` turns the selected `ROCm_Host` buffer back into the plain mapped CPU buffer.
The scheduler then cannot hand the `GET_ROWS` to a GPU (`ggml_backend_cuda_device_supports_buft()`
only accepts `ROCm_Host` when `integrated`, and a discrete GPU is not), so it goes to the CPU backend.

## 2. The two candidate structural fixes

### A. Place the input layer on the output device (`patches/0030`) — **works**

`LLAMA_DEVICE_INPUT=1` sets `pimpl->dev_input = get_layer_buft_list(n_layer_all)`, so `token_embd` is
allocated in the output device's buffer (under `-sm tensor` that is the **Meta** buffer, replicated as
`GGML_BACKEND_SPLIT_AXIS_MIRRORED` — the generic split state for `token_embd`), and the `GET_ROWS`
runs inside the GPU graph.  The per-layer token embedding is special-cased back to the CPU list so it
stays host-resident and host-gathered.

Result: **0 CPU splits**, and the CPU is quiet even with `0029` disabled
(`GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1`): qwen4exp IQ4_NL `draft-mtp n3`, prose, `-n 800`,
kill-switch, `LLAMA_DEVICE_INPUT=1` → **107.8 t/s / 1.0 cores** (vs 81.6 t/s / 14.9 cores with the old
host placement and no 0029).

But it is slower than the default.  Interleaved A/B (`-n 1500`, same build, same seed/prompt, alternating
arms):

| arm | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| default (host input, `0029` heuristic) | **111.4** | **111.4** | **111.4** |
| `LLAMA_DEVICE_INPUT=1` (device input) | 108.7 | 107.5 | 108.8 |

→ the Meta-split GPU gather costs **~2.6 %** versus the single-threaded CPU gather (which runs while the
GPU waits for the first split anyway).  It also replicates `token_embd` in VRAM (3 × 341 MiB here).
Acceptance (`0.85417`) and same-seed text (`a79d0d14855b`) are **identical** in both arms.

### B. Keep the table in host memory, read it from the device — **not reachable under `-sm tensor`**

The natural "no VRAM" structural fix (brief §2.1.2 candidate 4) is to leave `token_embd` in
`ROCm_Host` and let the GPU read the pinned host memory.  It fails for a structural reason: under
`-sm tensor` the model's device list is a **single Meta device**, so the scheduler's only GPU backend is
the Meta backend — and `ggml_backend_meta_device_supports_buft()` only accepts Meta buffer types.  The
per-device ROCm backends are not scheduler backends, so widening the ROCm device's `supports_buft()`
does not help: the `GET_ROWS` still has nowhere to go but the CPU.  (On an APU the single device *is*
the scheduler backend and `integrated` already accepts the host buffer — that is exactly why `0025`
works there and not on a discrete multi-GPU box.)  Making B work would require the Meta backend (and
every simple device behind it) to accept host buffers, for a per-launch read of a few rows — not worth
the blast radius.

## 3. Validation of `0030` (opt-in)

* **Purity:** qwen4exp `--spec-type none` == `draft-mtp n3` == **`a79d0d14855b`** (943 chars), and the
  same hash as the default and the pre-structural build; the opt-in arm alone is also `a79d0d14855b`.
* **Coherence:** 27B UD-IQ3_S `6073add19dac` (the recorded reference), both 1 GPU and 3-GPU `-sm
  tensor`; 35B-A3B plain == `draft-mtp`; qwen4exp `-sm layer` plain == `draft-mtp` (`54070cb0be03`).
* **Prefill:** qwen4exp `pp2048` 2475.1 ± 8.2 t/s (default-build 2448-2467) — neutral/slightly better.
* **CPU-only:** the `-ngl 0` path returns the CPU from `get_layer_buft_list()` and is unchanged by
  construction.
* **Fresh apply:** r13 + `archive/work/mmb-general` + closing `0001..0014`/`0016..0030` (skip `0015`)
  reproduces tree `99b429a60d441f814c84737cfa57803bc15a2f6d` (29/29 `git am`).

## 4. Verdict / what to keep

* **Keep `0029` default-on** — it is the throughput optimum and the actual fix for the spin.
* **`0030` is opt-in** (`LLAMA_DEVICE_INPUT=1`): it is the only way to remove the CPU split under
  `-sm tensor`, it is byte-identical, and it trades ~2.6 % MTP for ~0.3 cores and no CPU-split
  dependence on the `0029` heuristic.  Use it when CPU headroom matters more than the last 2.6 % (e.g.
  a busy server), or to bisect a suspected CPU-split issue.
* Do **not** default it on: `AGENTS.md`'s default-on policy is for features that *improve* the metric,
  and this one measures slower on the maintained boxes.
* The `per_layer_token_embd` special-case is correct for every host-gather PLE model (qwen4exp,
  gemma3n/gemma4) and only affects the opt-in path today.
