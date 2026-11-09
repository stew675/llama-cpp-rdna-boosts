# archive/work/ple-gather (CLOSED, promoted in r243) — contributor PR #289

**Status: CLOSED 2026-11-09 — promoted into the delivery as a block-11 amendment in `v16-a691b2345-r243`.**
Nothing here is delivery work and nothing is open.  The code is in `patches/0011-*`; the record of *shipping* it is the
`WORKLOG.md` entry `261109.008` and `INVARIANTS.md` r243; the variable catalogue rows are in `ENVIRONMENT.md`; the
per-block verification contract row is `MANIFESTS.md` block 11.  This file is the campaign record: what the
contributor offered, what we measured, what we changed, and what we deliberately left out.

Contributor: **Brian (`briansp2020`)** — <https://github.com/stew675/llama-cpp-rdna-boosts/pull/289>, one patch
`0001-qwen4exp-gather-the-host-PLE-rows-on-several-threads.patch` (`src/models/qwen4exp.cpp`, +53/−9) cut against
**r235**, plus a README, filed under `wip/ple-gather-r235/`.

Artifacts kept here:

| file | what | sha256 |
|---|---|---|
| `0001-ple-gather-as-submitted.patch` | the patch exactly as the PR carries it (`git format-patch` against r235) | `2fcd7e1ba010da8e6d2684e04d0faefbc3385538de6df91c0dc122949cfd2216` |
| `0001-ple-gather-final-cut.diff` | the **delivery's** form of the change (`git diff` against the r242 tree): the re-cut plus the three adjustments below | `cfdc9f38969c146841444438ebfc38492d3cafdd85f117e06c287de5f7dab587` |

## What it does

`llm_graph_input_ple::set_input` (the host-table path that replaced a CPU `get_rows` node, so the prefill graph stays
in one GPU split) gathers `ple_n_heads` hashed rows per token out of `per_layer_token_embd`, dequantizes them into
`emb_scratch` and `ggml_backend_tensor_set`s that into the PLE graph input — while both GPUs wait at the ubatch
boundary.  Two changes:

1. **Write the graph input in place.**  When the input tensor lives in a host buffer, the rows are dequantized
   straight into `emb->data`; the scratch vector and its whole-input copy are skipped.  Device, split and RPC
   buffers, and a meta buffer with a non-host child, keep the old path.  `LLAMA_PLE_DIRECT=0`.
2. **Fan the row range out.**  One thread per 4096 rows over up to 8 `std::thread`s, so a decode batch (< 8192 rows)
   stays on the calling thread; a worker that cannot be started leaves its rows to the caller.  `LLAMA_PLE_THREADS`.

## The contributor's numbers (2x R9700, gfx1201, ROCm 10.1, stock r235 `77b4f43a3`)

Flash-Next GSQ-RCO IQ3_S, `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto
--cache-ram 1024`, `-c 262144`, host pool off, slab headroom 1536: prefill **2090/2088 -> 2110/2107 t/s (+0.9 %)** on
a 101k prompt and **2142/2139 -> 2153/2144 (+0.3 %)** on a 31.5k one; on their production build (`-ub 4096`)
**+0.8-1.2 %**.  Identical greedy/prompt-set/101k hashes vs stock, 7 other models coherent, KLD 0 on all 6 checks,
`test-backend-ops` 21 679/21 680 (the one failure also fails on stock), 0 GPU faults.  Their motivation: "about 300k
rows and a 200 MB copy per ubatch" at `-ub 4096`.

## What we measured (re-cut onto the r242 tree, gfx1201, 2x R9700 `0,2`)

| arm | base (r242) | PR | Δ |
|---|---|---|---|
| delivery shape (`-ub 4096` + attendance pool), 31.5k, 6v6 | 2338.5 | 2369.2 | **+1.31 %** |
| `-ub 4096`, no pool, 31.5k, 2v2 | 2286.8 | 2313.3 | **+1.16 %** |
| their field shape (`-ub 6144`, no pool), 31.5k, 6v6 | 2384.3 | 2385.7 | **+0.06 %** (neutral) |
| their field shape, 101k, 3v3 | 2444.1 | 2451.9 | **+0.32 %** |
| same binary, both switches OFF (control), 31.5k, 4 runs | — | 2344.4 | — |
| … threads only (`LLAMA_PLE_DIRECT=0`), 4 runs | — | 2360.6 | **+0.69 %** |
| … direct write only (`LLAMA_PLE_THREADS=1`), 4 runs | — | 2348.5 | **+0.17 %** |

Every PR run in the delivery shape (2360.9-2380.5) sits above every base run (2327.1-2354.9); the field-shape arms
are inside this box's spread.  **The win is ubatch-boundary shaped**: at `-ub 4096` a 31.5k prompt crosses 8
boundaries instead of 6, and at `-ub 6144` the same absolute per-boundary cost hides behind the host-expert fill.

Rule-5 width check (`llama-batched-bench -npl 1,4,8 -npp 512 -ntg 32`, 2 GPU `-sm tensor`, q8_0 KV): PP B1 592.5 ->
597.2, B4 2226.0 -> 2242.9, B8 2465.0 -> **2539.8** t/s; decode 56.0/160.3/237.1 -> 56.2/157.5/240.7 (inside the
spread).

Correctness: greedy text identical in every arm — `9de3ae5502bc` (31.5k, `-n 64`), `e6ecbe2cd1c8` (`draft-mtp n3`,
`-n 256`), `0f5005a4398f` (1 device `-sm layer`), `83eec5e9b4f0` (Qwen3.5-4B control); prefill-logit
**-0.000000 / 100.000 %** (`scripts/gate-prefill-logits.sh --sm tensor --device 0,2`, base recorded from the r242
build); no GPU faults across every run (`scripts/check-gpu-faults.sh`).

## Where the PR text is wrong

- **"About 300k rows and a 200 MB copy per ubatch" at `-ub 4096`** — measured: **65 536 rows / 41.9 MB**
  (`ple_n_heads = 16`, `ple_head_dim = 160`, IQ4_NL 90 B/row; 98 304 / 62.9 MB at `-ub 6144`).  ~4.6x overstated.
  The census that produced the figure is `LLAMA_PLE_DEBUG=1` (`ple gather: 65536 rows, 7 threads, direct=1`).
- **`+0.9 %` / `+0.3 %` on the field shape do not reproduce** here (+0.32 % / +0.06 %).  Their `-ub 4096` production
  numbers do (+1.2-1.3 %, if anything conservative), and that is the shape the delivery validates qwen4exp on.

## What was adjusted before it could ship

1. **Re-cut onto the current tree.**  The touched region is byte-identical between r235 and r242 (only line numbers
   drifted), so the patch applied with `git am -3` and no manual resolution.
2. **The thread ceiling follows the operator.**  The width is now the smallest of 8,
   `std::thread::hardware_concurrency()` and `cparams.n_threads_batch` (`--threads` / `-tb`), instead of a hardcoded
   8.  ggml's threadpool cannot be reused: its only entry point is `ggml_graph_compute(cgraph, threadpool,
   n_threads)`, it has no getter (`ggml_backend_cpu_set_threadpool` is write-only) and no `parallel_for`, and it is
   idle while the GPUs run the prefill anyway — while this codebase's own host fan-outs
   (`llama_lazy_reader::gather`, `llama_state_copy_jobs`) already spawn their own threads.  Reusing the pool would
   mean a second work-item protocol in `ggml-cpu.c`'s threading core to save ~0.02 % of a prefill.
3. **The direct write resolves its buffer like `ggml_backend_tensor_set()`** (`view_src ? view_src->buffer : buffer`),
   so a view input cannot be mis-detected; a false negative simply takes the old path.
4. **A census was added** (`LLAMA_PLE_DEBUG=1`, one INFO line per `set_input`): a silent optimisation makes a later
   arm vacuous (r225/r238).  The 65 536/41.9 MB figure above came from it.
5. **Records**: `ENVIRONMENT.md` rows, the `MANIFESTS.md` block-11 verification row, `INVARIANTS.md` r243, the
   `patches/README.md` block-11 note, the block-11 commit message, and `WORKLOG.md` 261109.008.

## Held back deliberately

- **A persistent gather pool** instead of spawn-per-ubatch: it would save ~0.3 ms of thread creation per boundary
  (~0.02 %) and cost lifetime/teardown handling in the graph input.
- **The `MADV_WILLNEED` range merge** the contributor mentions as a follow-up: that is `wip/ple-residency`'s
  territory (parked), and the delivered per-page loop is load-bearing per that campaign's own record (removing it
  costs 40 % of prefill).
- **An arch sweep** (gfx1100/gfx1151): the change is host code with no kernel, no backend predicate and no arch
  branch, the 1-device arm is neutral with identical text, and the 4B control shows it unreachable outside
  qwen4exp.  On gfx1151 the whole expert set fits in VRAM, so `per_layer_token_embd` is a *device* table and the
  host-gather arm is not even reached without `--override-tensor 'per_layer_token_embd.*=CPU'`.
