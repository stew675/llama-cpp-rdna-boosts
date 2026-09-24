# gfx1201 — closing-the-gap: **remaining work** (live brief)

**Audience:** the agent continuing the campaign on the **3× Radeon AI PRO R9700 (gfx1201, RDNA4)** box,
`soar`.
**Goal:** pick up the **open items** below.  The campaign's validation/porting pass on this box is
largely done — what is left is the MTP CPU-spin automatic path selection (OP-1, the big one), a
throughput re-baseline, the `0028` follow-ups, a few unrun gates, and two low-priority/parked items.

**Companion files:**
| file | what |
|---|---|
| [`gfx1201-closed.md`](gfx1201-closed.md) | **the full history and details of everything CLOSED** on this box — session log §11, gate results, per-patch verdict table §7, DONE work items §12.1-§12.3, `0016` port, lossy-transfer result, the §13.7 cliff fix.  Cite it for anything marked DONE here. |
| [`gfx1151-closing.md`](gfx1151-closing.md) | the gfx1151 revalidation/port brief (the MMVQ band halves and the `n7`/`n8` matrix are handed there). |
| [`gfx1100-closing.md`](gfx1100-closing.md) | the single-7900-XTX brief. |
| [`closing-the-gap.md`](closing-the-gap.md) · [`README.md`](README.md) | the campaign handover + patch inventory. |
**Delivery policy:** `AGENTS.md` — default-on policy (a beneficial feature ships on, an env var only
**disables**), purity rules, and **never push the `~/llama.cpp` fork**.  Push the delivery repo only if
the maintainer asks.

---

## 0. Status at a glance

**Closed** (one line each in §1 → full detail in `gfx1201-closed.md`): the whole initial validation
pass (build, oracles, width purity, same-seed coherence, intra-build `plain == draft-mtp`, qwen4exp
prefill A/B, PPL parity, the rule-5 batched gate), the four RDNA4 ports/enablements (`0003`,
`0016`, `0017`, `0027`), the `0004` conv-fusion `-sm tensor` purity fix, the `0010`/`0011` lossy-transfer
negative, and the `0028` W=9 verify-cliff fix.

**Open** (this file):

| id | item | priority | where |
|---|---|---|---|
| **OP-1** | MTP CPU-spin: **automatic path selection, no `OMP_*`/`KMP_*` env vars** (~35 % of qwen4exp MTP lost; confirmed live) | **highest** | §2.1 |
| **OP-2** | re-baseline the gfx1201 qwen4exp MTP throughput (all pre-fix numbers are ~35 % low) + make the harness assert the CPU is quiet | high (composes with OP-1) | §2.2 |
| **OP-3** | `0028` follow-ups: full `n7`/`n8`/adaptive matrix *with* the fix, per-type MoE band tuning, another odd-row dense model | medium | §2.3 |
| **OP-4** | validation gates not run: four-axis MTP with the fix, `llama-imatrix`, `0001`/`0008` isolated A/Bs, the M-RoPE image case | medium | §2.4 |
| **OP-5** | remaining RDNA3_5-gated kernels (`0013` redundancy check; `0011`/`0023` parked) | low | §2.5 |
| **OP-6** | build-time critical path (the `fattn-mma-f16` per-type instance blow-up) | low / parked in `TODO.md` | §2.6 |

---

## 1. Closed items — one line each (full details in `gfx1201-closed.md`)

| closed item | result (one line) | detail |
|---|---|---|
| §5 build-time check | PASS — `fattn-tile.cu.o` = 96 `tile_case` `U`, 0 defined; no implicit type axis | `gfx1201-closed.md` §5 / §11 |
| §6.5 op oracles | `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46, `TOPK_QSA` 4/4, `LIGHTNING_INDEXER` 225/225, `FLASH_ATTN_EXT` 5954 OK / 0 FAIL | §11 gate results |
| §6.3 width purity | PASS (worst 0) 27B UD-IQ3_S + 35B UD-Q3_K_M | §11 gate results |
| §6.1 same-seed coherence | 4B/27B IQ3_S byte-identical to r13+beta; 27B Q8_0 / 35B deltas proven to be the MMB prefill **re-baseline**, not impurities | §11 integration |
| §6.2 intra-build purity | `plain == draft-mtp` PURE at 8K/40K/128K on dense + qwen4exp | §11 gate results |
| §6.6 qwen4exp MTP acceptance | 0.81388 (spin-free number now 0.84791 — see OP-2) | §11 gate results |
| §6.6 qwen4exp prefill A/B | closing adds **+4.4 %** pp32768, +3.1 % pp65536, +1.0 % pp98304 over r13+beta | §11 gate results |
| §6.8 PPL parity | 35B UD-Q3_K_M +0.51 %, 27B UD-IQ3_S −0.04 % | §11 gate results |
| arch A/Bs | `0021` byte-identical; `0022` inert (gfx1201 dense-always); `0016` ported (below) | §11 arch A/Bs |
| `0004` conv1d fusion | **fixed** — GDN/PLE conv fusion not bit-identical under `-sm tensor`, now gated to single-device graphs | `2026-09-23-gfx1201-conv-fusion-tensor-split.md` |
| `0003` `hc_gate_mix` RDNA4 port | **PORTED, default ON** — bit-identical, +5.8/+5.4/+5.3 % qwen4exp IQ4_NL prefill | §12.2 |
| `0016` `QSA_SCORE_WMMA` RDNA4 port | **PORTED, default ON** — +1.7/+3.2/+6.5/+12.3 % prefill, oracle 225/225 | §11 session 5 / `2026-09-23-qsa-score-wmma-rdna4.md` |
| `0017` MMB quant coverage | Q4_1 + Q5_0 **enabled** on RDNA4 (+6..14 %), Q4_0 excluded (−2 %) | §12.1 |
| `0027` meta `graph_optimize` | **forwarded under `-sm tensor`** (`MMB_OPT` 0 → 390/5418) | §12.3 |
| `0010`/`0011` lossy prefill | **NEGATIVE** — no gfx1201 win (APU/unified-memory effect) | §11 lossy-transfer |
| `0019`/`0023` HC16 | inert on RDNA4 (RDNA3_5-gated at the call site) | §7 verdict table |
| `0025` host-buffer input | no-op on a discrete GPU (`prop.integrated=0`) | §7 verdict table |
| rule-5 batched verify gate | PASS (within noise at B=1/4/8) | §11 |
| `0028` W=9 verify cliff | **FIXED** — MMVQ band boundary; qwen4exp B=9 202.6 → 270.0 t/s, `n_max 8` MTP +8.9 % | §11 session 8 / `2026-09-24-qwen4exp-w9-verify-cliff.md` |
| per-patch verdict table | filled for every closing patch | §7 |
| porting layers | "no port needed" = only the beta prerequisite carried the RDNA4 kernel ports; the closing set's own RDNA3_5-only kernels are OP-5 | §11 porting layers |

---

## 2. Open items (the work)

### 2.1 OP-1 — MTP CPU-spin: **automatic** path selection, no env vars  ← the headline

**Goal (maintainer, 2026-09-24):** a user must be able to run a typical `llama-server` config with
**no** `OMP_*`/`KMP_*` environment variables and get the fast path.  The delivery has to **detect** the
degenerate scheduling and choose the high-performance path itself, **default-on** (an env var may only
*disable* it, per `AGENTS.md`).

#### 2.1.1 The problem, and the fresh reproduction (2026-09-24, current tree = 0028)

`gfx1201-closed.md` §11 session 6 root-caused it: under `-sm tensor` the scheduler puts a **CPU split**
(the input / PLE `GET_ROWS`: `model.input_embed`, `ple_embd`, `mtp_tok_embd`) at the front of every
graph.  MTP calls `llama_decode` ~`n_max+1`× per token, so the OpenMP **active-wait** pool never sleeps
and spins all 16 cores.  Plain decode does ~55 such graphs/s and the pool settles (~2 cores); MTP does
~250–350/s and pins every core.

Re-confirmed on the current tree (with the `0028` fix), qwen4exp IQ4_NL + `mtp-…-shared-Q8_0.gguf`,
3-GPU `-sm tensor`, q8_0 KV, `draft-mtp n3`, prose, `-c 16384 -n 1500`:

| env | Generation | CPU | acceptance |
|---|---:|---|---:|
| default | **79.3 t/s** | ~16 cores pinned | 0.84791 |
| `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` | **107.2 t/s** | 1–2 cores | 0.84791 |

**+35 %** and 14 cores recovered, acceptance **byte-identical**.  The env mitigation must not remain a
user requirement.

#### 2.1.2 The structural fix — get the input embeddings off the CPU

If there is no CPU split there is nothing to spin.  `0024` (single-device input-on-GPU) / `0025`
(host-buffer input + scheduler guard) already cover the **APU / 1-device** cases.  On a **discrete
multi-GPU** box `n_devices() != 1` and `prop.integrated == 0`, so neither applies and the host-mapped
`token_embd` / `per_layer_token_embd` / `mtp_tok_embd` `GET_ROWS` stays on the CPU.

Candidates, cheapest first:
1. **draft `mtp_tok_embd`** — one small table, a per-draft-step cost; put its `GET_ROWS` on a device.
2. **target `token_embd`** — one table.
3. **`per_layer_token_embd`** (~27 GiB) — the hard one: a `GET_ROWS` is **row-parallel**, so the table
   shards cleanly across the tensor-split devices (each device holds a row range; indices are routed to
   the owning shard, or every shard gathers with a masked add).  Investigate whether the meta backend's
   split machinery can carry the embedding table + `GET_ROWS`, or whether the existing `-sm tensor`
   `ncols2`/`GET_ROWS` split-state handlers already do (see `ggml_backend_meta_* handle_get_rows`).
4. Alternative: keep the table host-resident but run the `GET_ROWS` **on the device** over a
   host-mapped pointer — this is the `0025` idea generalised past `integrated`.  Cheaper VRAM, but the
   device reads host memory (discrete GPUs can via HMM/`hipHostMalloc`-mapped, at a bandwidth cost;
   only the gathered rows are read).

The **structural** option is the right one if it validates; it removes the CPU graph rather than hiding
it.

#### 2.1.3 The runtime fix — automatic spin mitigation (the detection half)

Even with a CPU split, the active-wait spin is pure waste.  Explore, in order:
* **Per-split thread count**, not per-process: the CPU split here is tiny.  Find where the
  scheduler / CPU backend picks the split's thread count and make **small CPU splits run single-thread**
  (`n_threads = 1` → no OpenMP fork → no barrier to spin on).  Most surgical and automatic by
  construction.  Check `ggml_backend_cpu_graph_compute` / the threadpool wiring
  (`ggml_backend_cpu_set_n_threads`, the `set_n_threads` list in `llama_context`).
* **Passive/low-spin pool**: detect the shape (GPU backend present + CPU split is input-only) and set
  the pool's wait policy / `KMP_BLOCKTIME` equivalent programmatically (check whether
  `kmp_set_blocktime(0)` or a `ggml_threadpool` pause/priority mode is reachable from the CPU backend;
  if not, whether the pool can be created with the right mode).
* A startup heuristic that logs **one clear line** when it engages, plus an env **kill-switch**
  (default-on, disable-only).

Whatever is chosen: **automatic, default-on, arch-neutral, env kill-switch for bisection only** — not an
env opt-in.  It must not regress plain decode, prefill, or CPU-only runs.

#### 2.1.4 Also fix — draft sampler backend offload under `-sm tensor`

`llama_context::set_sampler` rejects the backend sampler outright when
`model.split_mode() == LLAMA_SPLIT_MODE_TENSOR` (`"backend sampling not supported with SPLIT_MODE_TENSOR;
using CPU"`), so the draft `top_k(10)` chain runs on the CPU every draft step.  Make the backend sampler
tensor-split-aware (or keep it on a device) so no per-step CPU round-trip is needed.  Small, but it
composes with 2.1.2/2.1.3.

#### 2.1.5 Acceptance criteria

* A plain `llama-server` with **no** `OMP_*`/`KMP_*` env keeps MTP decode at ~1–2 CPU cores and
  reproduces the passive-wait throughput (**~107 t/s**, qwen4exp IQ4_NL `draft-mtp n3`, prose) with
  **byte-identical output and identical acceptance** (0.84791 / the session-6 0.83204).
* Plain decode and prefill unregressed; CPU-only builds unaffected; the `qwen35`/`qwen35moe` models
  (no PLE) stay on their current numbers (already clean, §11 session 6).
* **Re-baseline the gfx1201 qwen4exp MTP t/s** (see OP-2) and make the harness robust: the MTP gate
  should not depend on an env var being set.  Consider teaching the reporting harness to assert the CPU
  is quiet (a core-count sanity check) so a future degenerate-scheduling regression cannot silently
  poison the numbers.

#### 2.1.6 Tools / harness carried over (from the session-6 record)

* `/tmp/mon.py` — per-thread CPU sampler (`/proc/<pid>/task/*/stat` deltas); the instrument that made
  the spin visible (2 cores → 16 cores).
* `gdb -p <pid> -batch -ex "thread apply all bt"` — the stack that located the draft `llama_decode`.
* `GGML_SCHED_DEBUG=1` (with `-lv 5`) — per-graph split/backend assignment; showed the `CPU` split ahead
  of the `Meta` split and its `model.input_embed`/`ple_embd`/`mtp_tok_embd` inputs.
* `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` — the A/B that proved the spin was idle-wait, not work.
* Session-6 command shape: IQ4_NL 9-shard + `mtp-…-shared-Q8_0.gguf`, `-sm tensor`, q8_0 KV,
  `-b/-ub 2048`, `-c 16384`, `-n 1500..3000`, seed 42, temp 0.

> Full original plan text (the same candidates, with the session-6 narrative): `gfx1201-closed.md` §13
> and §11 session 6.  The `qwen35`/`qwen35moe` models have **no PLE** and their `token_embd` fits in
> VRAM, so their CPU split is empty — they are the clean control (they must stay unchanged).

### 2.2 OP-2 — re-baseline the gfx1201 qwen4exp MTP throughput

Every qwen4exp MTP t/s figure measured on this box **without** the passive-wait env is ~35 % low
(session 2's 57.5 t/s, session 6's pre-fix table, …).  The session-8 numbers use
`OMP_WAIT_POLICY=PASSIVE` and are the unconfounded reference.  Do:
* re-measure the four-axis (`R`/`C`/`K`/`P`) + phase-switch (`X`) set with the passive env (or the
  OP-1 fix), `-n 3000`, reasoning pinned per axis (`benchmarks/mtp-adaptive-methodology.md` rule 0);
* correct/annotate the stale records so a future session does not cite the confounded numbers;
* add the CPU-quiet assertion to the harness (§2.1.5).

### 2.3 OP-3 — `0028` follow-ups (the MMVQ band boundary)

`0028` is verified for the cliff itself but not exhaustively:
* **full `n7`/`n8`/adaptive matrix with the fix** — session 6 covered the matrix *before* the fix; only
  the prose axis + one `n8` pair were re-measured after.  Use the same protocol (`-n 3000`, reasoning
  pinned) and the `OMP_WAIT_POLICY=PASSIVE` (or OP-1) build.
* **per-type MoE band** — the routed-expert band floor is currently unconditional 16 on AMD; if a
  specific expert type is slower on `mul_mat_vec_q_moe` at 9..16, the floor has to become per-type (see
  the gfx1151 brief §4.5 — narrowing the per-arch table alone does nothing because the floor clamps up
  afterwards).
* **another odd-row dense model** — the RDNA4 dense rule (`nrows_x % 128 != 0`) was only exercised on
  qwen4exp; 27B/35B are the clean controls.  If one is available, A/B a dense model with odd rows.
* **gfx1151/gfx1100 revalidation/port** is handed off — see [`gfx1151-closing.md`](gfx1151-closing.md).

### 2.4 OP-4 — validation gates not yet run

* qwen4exp **four-axis MTP set with the fix** (only prose re-measured in session 8).
* `llama-imatrix` (`0019`/`0023`) — the NanBeige model is absent here; substitute another BF16 model.
  Note HC16 is RDNA3_5-gated, so on RDNA4 this is really a scheduler/split regression check.
* `0001` (`hc_combine_norm`) / `0008` (M=4 HC inject) **isolated** A/Bs — they are exercised by the
  qwen4exp gates but never singled out.
* §6.6 **M-RoPE image case** (`0005`) — needs the vision projector; the gfx1151 repro was an image after
  ~12k tokens of text.

### 2.5 OP-5 — remaining RDNA3_5-gated kernels (low priority)

* **`0013` prefill indexer relu+head-sum** (`idx_relu_sum`, call-site `GGML_CUDA_CC_IS_RDNA3_5`):
  `0016`'s port already banks this reduction (the fused lightning-indexer computes
  `bias + sum_h relu(dot_h)`), so a separate enablement is likely redundant on RDNA4.  Verify by
  diffing `GGML_CUDA_IDX_RELU_SUM` on/off **after** the `0016` port — if the graph no longer contains
  that chain, there is nothing to port (this is a 10-minute check).
* **`0011` HC BF16 streams / `0023` HC16**: `0011` needs the meta `graph_optimize` path (`0027`, done)
  to run under `-sm tensor`; `0023`'s HC16 is RDNA3_5-gated and not bandwidth-bound on discrete RDNA4 →
  **park** unless a 48 GB single-GPU RDNA4 box appears.

### 2.6 OP-6 — build-time critical path (parked in `TODO.md`)

The `fattn-mma-f16` per-type instance set is the remaining clean-build critical path (0.90 → 7.26 MB
per instance TU, 6.7 → 229 s; each instance file carries one WMMA kernel copy per KV type).  The tile
half was fixed in r5; the MMA half needs a code-path change (finer generated-file granularity, or a
runtime KV-type dispatch in the loader) with its own A/B.  Not gfx1201-specific, but measured here —
see `wip/build-time-regression/` and `TODO.md`.

---

## 3. Operational reference

### 3.1 The box + models

| | |
|---|---|
| GPU | **3× AMD Radeon AI PRO R9700 (gfx1201, RDNA4)** |
| Host | Ryzen 9 9950X3D2, 184 GiB RAM |
| ROCm | build: `/opt/rocm-7.14.1-gfx102X`; runtime: `/opt/rocm-7.14-gfx1201` |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (ccache) |
| Runtime | `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH` |
| Multi-GPU rule | **`-sm tensor` + `GGML_CUDA_ALLREDUCE=hybrid` (default)** for all qwen4exp / 3-GPU runs |

| model | role |
|---|---|
| `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | dense; needs `-lm none -lzm on`; built-in nextn → no `-md` |
| `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` | dense; MMB; built-in nextn |
| `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` | dense; rule-5 batched verify gate |
| `/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | MoE prefill; built-in nextn |
| `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | MoE; built-in nextn |
| `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` | dense Q8_0; FA head-512/tile policy |
| `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | MoE; F32-router isolate |
| `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/…-00001-of-00003.gguf` + `mtp-…-Q4_K_M.gguf` | qwen4exp headline (3-GPU `-sm tensor`) |
| `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/…-00001-of-00009.gguf` + `mtp-…-shared-Q8_0.gguf` | qwen4exp (used by the MTP/spin work) |
| `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf`, `/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf` | small dense smoke |

(Paths are the ones the records used — `ls` to confirm the layout.)

### 3.2 Apply the full stack (27-patch closing set)

The delivery repo is `~/llama-cpp-rdna-boosts`.  The current `~/llama.cpp` checkout is branch
`closing-gfx1201` = delivery r13 + `beta/mmb-general` + closing `0001..0014`/`0016..0028` (tip tree
**`533eee3188ab7df9b6cf394adeaa31b46bd13ff2`**).  A fresh apply:

```sh
WORK=$HOME/llama-cpp-rdna-boosts
cd ~/llama.cpp
git fetch --all
git checkout ebbb18522
git checkout -b closing-gfx1201
bash "$WORK"/scripts/apply-all.sh .                      # delivery r13 (16 blocks)  -> tree bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12
git am "$WORK"/beta/mmb-general/patches/*.patch          # 12 beta patches          -> tree 79136a15cac1920c0dd334b4c119a9cb42f9143b
for p in "$WORK"/wip/closing-the-gap/patches/0*.patch; do
  case "$p" in *0015-*) echo "skip 0015 (superseded by r13 block 00)"; continue;; esac
  git am "$p"
done
git rev-parse HEAD^{tree}                                # expect 533eee3188ab7df9b6cf394adeaa31b46bd13ff2
```

**Traps:** `0015` must be **skipped**; `0024` must be applied **before** `0025` (the loop order handles
this).  The single-patch alternative is `git apply` of `wip/closing-the-gap/campaign-all.patch` on the
r13+beta tree.

### 3.3 Build

```sh
cd ~/llama.cpp
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop:
# cmake --build build-rocm --target llama-cli llama-bench llama-perplexity \
#   test-backend-ops test-logits-width-probe llama-batched-bench llama-imatrix -j 16
```

### 3.4 Gate commands

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
# M = <model gguf>, MD = <mtp sidecar for qwen4exp>
P=$WORK/prompts/prose-rdna-boosts.txt

# coherence (per model) — 27B Q8_0 adds -lm none -lzm on
build-rocm/bin/llama-cli -m "$M" -ngl 99 -fa auto -ctk f16 -ctv f16 -c 8192 -n 24 \
  --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off -f "$P" > /tmp/coh.log 2>&1
python3 "$WORK"/scripts/extract-generated.py /tmp/coh.log

# intra-build purity (the real contract; ALWAYS --ctx-checkpoints 0 at depth)
#   --spec-type none  vs  --spec-type draft-mtp --spec-draft-n-max 3   must be byte-identical

# width probe
build-rocm/bin/test-logits-width-probe "$M" "$P" 1024 512       # width_purity=PASS (worst 0)

# MMB config + A/B
GGML_CUDA_MMB_CFG=1 build-rocm/bin/llama-bench -m "$M" -ngl 99 -p 2048 -n 0

# op oracles (separate stdout/stderr; FLASH_ATTN_EXT from stdout-only)
for op in FLASH_ATTN_QSA GATED_DELTA_NET TOPK_QSA LIGHTNING_INDEXER FLASH_ATTN_EXT; do
  build-rocm/bin/test-backend-ops -o $op > /tmp/orc-$op.out 2>/tmp/orc-$op.err
done

# MTP (qwen4exp must pass -md mtp-…-shared-Q8_0.gguf; 27B/35B have built-in heads — no -md)
OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 build-rocm/bin/llama-cli -m "$M" -md "$MD" \
  -ngl 99 -sm tensor -c 16384 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto \
  -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off -f "$P" \
  --spec-type draft-mtp --spec-draft-n-max 3 --ctx-checkpoints 0 -lv 4
```

Oracles expected: `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46, `TOPK_QSA` 4/4,
`LIGHTNING_INDEXER` 225/225, `FLASH_ATTN_EXT` 5954 OK / 0 FAIL.

### 3.5 Harness + traps

* **Warm the page cache** for the multi-shard 87 GiB model before any A/B — the first cold pp8192 read
  cost ±121 t/s (a spurious −21 %); the warm re-run was clean.
* **Kill leftover benches**: `pkill -9 -x llama-bench` — an orphan holds ~22 GiB/GPU and the next load
  dies with `ggml-backend-meta.cpp:1848 GGML_ASSERT(meta_buf_ctx->bufs[i])`.
* **Capture stdout/stderr to files and parse after** — never `| grep | head` a bench (it can hang).
* Use `pgrep -x llama-bench` (not `-f`; `-f` matches your own shell).
* **`--ctx-checkpoints 0`** for anything at depth.
* **Never benchmark in parallel**; interleave A/B arms in one warm session.
* **`-md` on a `plain` arm aborts** (a draft head without a trunk); qwen4exp must pass the sidecar, the
  dense/MoE models must not.
* **`FLASH_ATTN_EXT` cannot be counted from a merged `2>&1` log** (ANSI + stream interleaving).
* **Fold a fix** with `git commit --fixup=<commit>` then
  `GIT_SEQUENCE_EDITOR=true GIT_EDITOR=true git rebase --autosquash <commit>~1`; regenerate with
  `git format-patch -1 <sha> --stdout --no-numbered`; replace the patch file and re-verify a fresh
  r13+beta worktree + the 27 patches reproduces the branch tip tree.
* **Do not push the `~/llama.cpp` fork**; push the delivery repo only on explicit request.

### 3.6 Report template

For each gate: the **exact command**, the **build/tree**, the **`MMB_CFG`** line, the **numbers** (and
the interleaving order for A/Bs), and the **extracted hash** where a text gate applies.  State the gate
name, not "it was slower".  For MTP use `benchmarks/mtp-adaptive-methodology.md` rule 0
(`-n 3000`, reasoning pinned).  Append results to the session log below, and move completed items into
§1 with a pointer to the detail (which belongs in `gfx1201-closed.md` once written up).

---

## 4. Session log (open-work sessions, newest first)

### 2026-09-24 — brief split

`gfx1201-closing.md` was split: all closed work moved to [`gfx1201-closed.md`](gfx1201-closed.md), this
file now tracks only the remaining items (OP-1…OP-6).  State at the split: closing set 27 patches,
tip tree `533eee3188ab7df9b6cf394adeaa31b46bd13ff2`; `0028` (the W=9 cliff) folded and verified; the
CPU-spin drop re-confirmed live (79.3 → 107.2 t/s with the passive env).  Previous sessions 1–8:
`gfx1201-closed.md` §11.
