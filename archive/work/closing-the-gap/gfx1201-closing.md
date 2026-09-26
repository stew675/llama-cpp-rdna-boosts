# gfx1201 — closing-the-gap: **closure record** (all campaign items closed)

**Audience:** the agent continuing the campaign on the **3× Radeon AI PRO R9700 (gfx1201, RDNA4)** box,
`soar`.
**Goal:** pick up the **open items** below.  **Every campaign item is now closed** — OP-1 (`0029`
default-on, `0030` opt-in), OP-2, OP-3 (no per-type MoE band), OP-4 (`llama-imatrix` + the
`0001`/`0008` A/Bs), OP-5.1/OP-5.2, and OP-6 (build time: no change, ccache).  This file is now a
closure record.  **§0.5 lists the outcomes; §0 the two follow-up leads** (not campaign gaps).

**Companion files:**
| file | what |
|---|---|
| [`gfx1201-closed.md`](gfx1201-closed.md) | **the full history and details of everything CLOSED** on this box — session log §11, gate results, per-patch verdict table §7, DONE work items §12.1-§12.3, `0016` port, lossy-transfer result, the §13.7 cliff fix.  Cite it for anything marked DONE here. |
| [`gfx1151-closing.md`](gfx1151-closing.md) | the gfx1151 revalidation/port brief (the MMVQ band halves and the `n7`/`n8` matrix are handed there). |
| [`gfx1100-closing.md`](gfx1100-closing.md) | the single-7900-XTX brief. |
| [`2026-09-24-op3-per-type-moe-band.md`](2026-09-24-op3-per-type-moe-band.md) | OP-3: the per-type MoE band measurement + the launch-bound trap (closed, no per-type band). |
| [`2026-09-24-op4-imatrix-and-hc-gates.md`](2026-09-24-op4-imatrix-and-hc-gates.md) | OP-4 (`llama-imatrix` smoke + the pre-existing `-sm tensor` imatrix corruption, the `0001`/`0008` A/Bs) and OP-5.2 (`0011` under `-sm tensor`). |
| [`2026-09-24-op6-build-time-closure.md`](2026-09-24-op6-build-time-closure.md) | OP-6 build time: closure (the largest MMA TU is 82 s vs a 236 s makespan; no change). |
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
negative, the `0028` W=9 verify-cliff fix, and (2026-09-24) all of **OP-1** — the automatic MTP CPU-spin
fix (`0029`), the opt-in structural input placement (`0030`), the OP-2/OP-3 four-axis re-baseline, the
OP-5.1 `0013` redundancy verdict, and OP-1.4 (draft sampler, won't-fix) — plus the 2026-09-24 closures:
**OP-3** (no per-type MoE band), **OP-4** (`llama-imatrix` smoke + the `0001`/`0008` A/Bs), **OP-5.2**
(`0011` parked) and **OP-6** (build time, no change).

**All campaign items are closed** — the per-item outcomes are in §0.5 and §1; the per-patch verdicts
and full history are in `gfx1201-closed.md`.

**Follow-up leads** (discovered here, not campaign gaps):

| lead | status | where |
|---|---|---|
| `-sm tensor` `llama-imatrix` | **confirmed upstream, open** — the imatrix's activation read under the Meta backend returns wrong values (the gather layout is correct; `llama-perplexity` is fine).  A **pure `ebbb18522`** worktree reproduces the exact corruption (`ffn_down` corr 0.0370; PPL 54211 vs 9.13), so it is an **upstream llama.cpp** bug, not the fork/delivery.  Workaround: `-sm layer`.  Upstream-PR candidate once the stale/aliased buffer is pinned | `2026-09-24-op4-imatrix-and-hc-gates.md` |
| M-RoPE image case (`0005`) | **run, clean** — closing and baseline (pre-fix) both pass the image+MTP repro (up to 19949-token prompt + 4096-token image + 1000 generated); the gfx1151 trigger does not reproduce on gfx1201 with the shared head.  `0005` retained as a port.  Harness: `tools/mrope-image-mtp.sh` | `2026-09-24-op4-imatrix-and-hc-gates.md` |

---

## 0.5 Next session — the run plan (start here)

**State.**  `~/llama.cpp` branch `closing-gfx1201` = delivery r13 + `beta/mmb-general` + closing
`0001..0014`/`0016..0030`, tip tree **`99b429a60d441f814c84737cfa57803bc15a2f6d`** (29 patches);
campaign repo `gap-closing` @ `4be503f`.

```sh
cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
# fast loop: cmake --build build-rocm --target llama-cli llama-bench llama-batched-bench llama-imatrix test-backend-ops -j 16
```

### OP-3 — per-type MoE band: **CLOSED 2026-09-24, no per-type band**

The unconditional floor at 16 is a net win for every expert type in the relevant range (`n_max <= 12`,
`B <= 13`); only IQ3_XXS/IQ4_XS regress, and only from `B=13` (`-2.2%`), which is the price of the
decode/verify purity the band exists to give.  The `__launch_bounds__` widening trap is also cleared
(no `B <= 8` regression).  Record + full data:
[`2026-09-24-op3-per-type-moe-band.md`](2026-09-24-op3-per-type-moe-band.md).  The harness that
produced it is kept below for reference.

```sh
cd ~/llama.cpp
M=/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf    # then Q4_K_M
for arm in on off on off; do
  if [ $arm = off ]; then export GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1; else unset GGML_CUDA_DISABLE_MMVQ_MOE_BAND; fi
  timeout 900 build-rocm/bin/llama-batched-bench -m "$M" -ngl 99 -sm tensor -c 8192 -b 2048 -ub 2048 \
    -npp 16 -ntg 32 -npl 1,4,8,9,10,12,16 > /tmp/bb_${arm}_$RANDOM.out 2>/dev/null
done
grep -E '^\| *16 \| *32 ' /tmp/bb_*.out      # B=9..16 rows are the band; B<=8 is the control

# and the qwen4exp cliff itself (IQ4_NL, the model 0028 was cut on; the w9-record harness):
#   ... -ctk q8_0 -ctv q8_0 -npp 16 -ntg 32 -npl 6,7,8,9,10,11,12
```

**Result (band ON vs OFF, `S_TG t/s`):** Q4_K/Q5_K +26% at B=9 tapering to +10% at B=16; Q5_K/Q6_K
+16% at B=12; IQ4_NL +13% at B=9; Q3_K +2.5% at B=12 then negative from B=14; IQ3_XXS/IQ4_XS +2.8%
at B=9, -0.2% at B=12, -2.2% at B=13.  If the relevant ceiling is ever raised past 12, the fix is a
kernel split (two 8-warp blocks along the token axis), **not** a per-type cap.  The gfx1151 per-type
mitigation shape is [`gfx1151-closing.md`](gfx1151-closing.md) §4.5.

### OP-4 — the unrun gates: **CLOSED 2026-09-24** (d deferred)

(a) the imatrix smoke check, (b) the `0001` A/B, (c) the `0008` A/B and (d) the M-RoPE image + MTP
repro are all **run** (2026-09-24).  Headline: the imatrix smoke check passes, but `-sm tensor`
`llama-imatrix` is corrupt (**pre-existing**; `llama-perplexity` is fine) — new follow-up lead, out of
scope.  `0001` / `0008` both validated.  (d) runs clean on the closing build **and** the pre-fix
baseline (the gfx1151 trigger does not reproduce on gfx1201 with the shared head).  Record:
[`2026-09-24-op4-imatrix-and-hc-gates.md`](2026-09-24-op4-imatrix-and-hc-gates.md).  Harnesses below.

**(a) `llama-imatrix` (`0019`/`0023`) - a split/scheduler smoke check.**  HC16 is RDNA3_5-gated, so on
RDNA4 this only proves the 3-GPU `-sm tensor` forward survives a long calibration run.
`/llm/models/Qwen3.8/27B/BF16/…-00001-of-00002.gguf` is the only full BF16 model (54 GiB, 2 shards;
fits 3x32 GiB).

```sh
build-rocm/bin/llama-imatrix -m /llm/models/Qwen3.8/27B/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf \
  -f prompts/prose-rdna-boosts.txt -ngl 99 -sm tensor -o /tmp/imatrix-bf16.dat
```
Require: no abort / no meta-split assert, a written imatrix, no NaN.  A build prefix is needed
(`cmake --build build-rocm --target llama-imatrix -j 16`).

**(b) `0001` `hc_combine_norm` isolated A/B.**  By default (`fused_dsv4_hc_post=false`) the plain HC
chain is built and the graph optimizer's `hc_combine_norm` matcher — the one `0001` revived — fuses
it; `LLAMA_FUSED_DSV4_HC_POST=1` forces the alternative `DSV4_HC_POST` op (the slower arm per the §7
verdict).  A/B the two on qwen4exp IQ4_NL `pp8192,32768` (`llama-bench -r 3`, interleaved), confirm
with `LLAMA_HC_CN_DEBUG=1` that the matcher fires, and record the delta + the greedy text at the gate.
Prior art: [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md).

**(c) `0008` M=4 HC inject isolated A/B.**  `GGML_CUDA_MMB_TALL_MIN_M=0` restores the M=4 inject into
the 384-row tall tile; A/B qwen4exp IQ4_NL `pp8192,32768`.  Expect the default (`16`) to win.  Prior
art: [`2026-09-22-mmb-tall-min-m.md`](2026-09-22-mmb-tall-min-m.md) (the `0008` record).

**(d) M-RoPE image case (`0005`) - lowest priority.**  Needs the vision projector
(`/llm/models/Qwen3.8/Flash-Next/Q4_K_M/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`): 3-GPU `-sm tensor`,
text to ~12k tokens then an image, and check for the M-RoPE `X < Y` / cell-window abort.
`llama-mtmd-cli` or `llama-server` + an image request.  If the tooling isn't ready, note it as not-run
rather than blocking on it.

### OP-5.2 — `0011` under `-sm tensor`: **CLOSED 2026-09-24, parked**

`0027` closes the gap: the HC marking pass now runs under `-sm tensor` (`HC_BLK16 comb=` 582×).  The
A/B is **+1.3 %**, all from `LLAMA_HC_RES16` (`blk16` is inert — `prod=0` on every candidate).  Lossy
→ stays default-OFF.  Record:
[`2026-09-24-op4-imatrix-and-hc-gates.md`](2026-09-24-op4-imatrix-and-hc-gates.md).

### OP-6 — `fattn-mma-f16` build-time: **CLOSED 2026-09-24, no further change**

The OP-6 premise is the **pre-r6** state: r6's per-head split already took the largest MMA instance TU
to **2.9 MB / 81.8 s**, and the clean `ggml-hip -j16` build is ~236 s — **throughput-bound, not
tail-bound** — so the remaining per-KV-type split (same total instantiation work) cannot help.  The
runtime-dispatch / `__noinline__` options were tried and rejected in r6; ccache is the answer.  Record:
[`2026-09-24-op6-build-time-closure.md`](2026-09-24-op6-build-time-closure.md).

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
| **OP-1** MTP CPU-spin | **DONE (runtime half, `0029`)** — tiny CPU split graphs run single-threaded; default 81.0 → 111.7 t/s, ~15 → 1.2 cores, acceptance/text byte-identical; env **kill-switch only** | `2026-09-24-mtp-cpu-spin-automatic.md` |
| **OP-2** MTP re-baseline | **DONE** — four-axis + `n7`/`n8`/adaptive, no env, CPU quiet every arm; `n8` near-tied with `n7` and wins recall (the old gap was `0028`) | `2026-09-24-mtp-cpu-spin-automatic.md` §4 |
| **OP-3** matrix half | **DONE** — the full `n7`/`n8`/adaptive matrix with the fix | `2026-09-24-mtp-cpu-spin-automatic.md` §4 |
| **OP-3** per-type MoE band | **CLOSED, no per-type band** — the unconditional floor at 16 wins for every routed-expert type in the relevant `B <= 13` range (k-quants +26%, IQ4_NL +13% at B=9); only IQ3_XXS/IQ4_XS regress and only from `B=13`.  The `__launch_bounds__` widening has no `B <= 8` cost (138-way register-identical, ±0.6% runtime) | `2026-09-24-op3-per-type-moe-band.md` |
| **OP-4** gates | **DONE** — (a) imatrix smoke passes but `-sm tensor` imatrix is corrupt (**pre-existing**, perplexity fine; root cause = the Meta eval-callback activation read); (b) `0001` default fusion +5 % over the op; (c) `0008` default `TALL_MIN_M=16` ahead; (d) M-RoPE image+MTP runs clean on closing **and** pre-fix baseline | `2026-09-24-op4-imatrix-and-hc-gates.md` |
| **OP-5.2** `0011` `-sm tensor` | **PARKED** — `0027` closes the marking gap (`comb=` 582×); A/B +1.3 %, all from `LLAMA_HC_RES16` (`blk16` inert); lossy → default-OFF | `2026-09-24-op4-imatrix-and-hc-gates.md` |
| **OP-6** build time | **CLOSED, no change** — the "7.26 MB / 229 s" premise is pre-r6; the largest MMA TU is now 2.9 MB / 82 s vs a 236 s `-j16` makespan (throughput-bound), so the per-KV-type split cannot help; ccache is the answer | `2026-09-24-op6-build-time-closure.md` |
| **OP-5.1** `0013` on RDNA4 | **REDUNDANT** — matcher fires 0× with `0016` ON, 4× with `LLAMA_QSA_SCORE_WMMA=0`; no port | `2026-09-24-mtp-cpu-spin-automatic.md` §5 |
| **OP-1** structural input placement | **INVESTIGATED, opt-in (`0030`)** — `LLAMA_DEVICE_INPUT=1` moves the input layer to the output/Meta device (0 CPU splits, byte-identical) but the Meta-split GPU gather is ~2.6 % slower MTP than host input + `0029`, so it is not defaulted | `2026-09-24-mtp-cpu-spin-structural.md` |
| **OP-1.4** draft sampler under `-sm tensor` | **CLOSED, won't fix** — blocked by the Meta backend (`handle_per_row` asserts on the vocab-split logits; needs a distributed top-k) and not a measurable win even under `-sm layer` (75.6 vs 75.1 t/s, noise) | `2026-09-24-mtp-draft-sampler-tensor-split.md` |
| per-patch verdict table | filled for every closing patch | §7 |
| porting layers | "no port needed" = only the beta prerequisite carried the RDNA4 kernel ports; the closing set's own RDNA3_5-only kernels are OP-5 | §11 porting layers |

---

## 2. Open items (the work)

### 2.1 OP-1 — MTP CPU-spin: **automatic** path selection, no env vars  ← the headline

> **RUNTIME HALF DONE 2026-09-24 (`patches/0029`).**  `ggml_backend_cpu_graph_compute` now runs a
> tiny CPU split graph (≤ 32 nodes / ≤ 16 MiB of node outputs — i.e. the host-mapped input/PLE
> `GET_ROWS` and the MTP draft's state copies) inline on the calling thread, so no OpenMP region is
> entered and nothing spins.  Default run (no env): qwen4exp IQ4_NL `draft-mtp n3` prose `-n 1500`
> **81.0 → 111.7 t/s**, **~15 → 1.2 cores**, acceptance **0.84791** and same-seed text
> **`a79d0d14855b`** identical (also identical with the heuristic forced off).  The delivery is
> **default-on with a disable-only kill-switch** (`GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1`).
> Detail: [`2026-09-24-mtp-cpu-spin-automatic.md`](2026-09-24-mtp-cpu-spin-automatic.md).
>
> **Still open:** 2.1.2 (the structural fix — remove the CPU split rather than serialise it; now
> optional) and 2.1.4 (draft sampler backend offload under `-sm tensor`).

**Goal (maintainer, 2026-09-24):** a user must be able to run a typical `llama-server` config with
**no** `OMP_*`/`KMP_*` environment variables and get the fast path.  The delivery has to **detect** the
degenerate scheduling and choose the high-performance path itself, **default-on** (an env var may only
*disable* it, per `AGENTS.md`).  **This is now satisfied by `0029`** (see the note above).

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

#### 2.1.2 The structural fix — get the input embeddings off the CPU  ← **INVESTIGATED, opt-in (`0030`)**

> **Result (2026-09-24): the structural fix works but is ~2.6 % slower for MTP, so it is opt-in
> (`LLAMA_DEVICE_INPUT=1`) and the default stays host input + `0029`.**  Detail:
> [`2026-09-24-mtp-cpu-spin-structural.md`](2026-09-24-mtp-cpu-spin-structural.md).  Interleaved A/B
> (`-n 1500`): host input + `0029` **111.4/111.4/111.4** t/s vs `LLAMA_DEVICE_INPUT=1`
> **108.7/107.5/108.8** t/s; acceptance and same-seed text identical.**  The opt-in gives **0 CPU
> splits / 1.0 cores**.  Candidate 4 (device reads the host-mapped table) is **not reachable under
> `-sm tensor`** — the Meta backend is the only scheduler GPU backend and rejects host buffers.

If there is no CPU split there is nothing to spin.  `0024` (single-device input-on-GPU) / `0025`
(host-buffer input + scheduler guard) already cover the **APU / 1-device** cases.  On a **discrete
multi-GPU** box `prop.integrated == 0`, so they do not apply and the host-mapped `token_embd` /
`mtp_tok_embd` `GET_ROWS` stays on the CPU.  (Note `n_devices()` under `-sm tensor` is **1** — the Meta
device — so the old `n_devices() != 1` framing is wrong; and `per_layer_token_embd` is not a graph op
at all: the model host-gathers it in `set_input`.)  Only the small `token_embd` table needs moving.

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

#### 2.1.3 The runtime fix — automatic spin mitigation (the detection half)  ← **DONE (`0029`)**

The chosen option was the first candidate — **per-graph thread count in the CPU backend**, no
scheduler surgery: [`patches/0029`](patches/0029-gap-closing-WIP-run-tiny-CPU-split-graphs-on-the-calling-thread.patch)
adds `ggml_backend_cpu_graph_n_threads()` (≤ 32 nodes **and** ≤ 16 MiB of node outputs →
`n_threads = 1`).  No OpenMP region is entered for the split, so the active-wait pool is never re-armed
and the spin is gone.  Default-on, disable-only kill-switch
`GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1`.  The passive-`KMP_BLOCKTIME` option was not needed.
See the note at the top of §2.1 and [`2026-09-24-mtp-cpu-spin-automatic.md`](2026-09-24-mtp-cpu-spin-automatic.md).

#### 2.1.4 Also fix — draft sampler backend offload under `-sm tensor`  ← **CLOSED, won't fix**

> **Closed 2026-09-24 (not small, and not measurable).**  (1) **Blocked structurally:** under
> `-sm tensor` `output.weight` is vocab-split so the logits are `GGML_BACKEND_SPLIT_AXIS_0`, and the
> sampler's `TOP_K`/`ARGSORT` dispatch to `handle_per_row()`, which asserts on AXIS_0 — lifting the
> guard aborts at `ggml-backend-meta.cpp:544`.  Fixing it needs a **Meta-backend distributed top-k**
> (per-shard top-k + merge, or a MIRRORED logits all-gather).  (2) **Not worth it even where it
> works:** under `-sm layer` the backend-sampling A/B is within noise (75.6 vs 75.1 t/s, ON vs
> `--no-spec-draft-backend-sampling` — the same-arm spread, 70.8→75.6, is larger).  The upstream guard
> (PR #23287) is deliberate.  Detail:
> [`2026-09-24-mtp-draft-sampler-tensor-split.md`](2026-09-24-mtp-draft-sampler-tensor-split.md).

`llama_context::set_sampler` rejects the backend sampler outright when
`model.split_mode() == LLAMA_SPLIT_MODE_TENSOR` (`"backend sampling not supported with SPLIT_MODE_TENSOR;
using CPU"`), so the draft `top_k(10)` chain runs on the CPU every draft step.  Make the backend sampler
tensor-split-aware (or keep it on a device) so no per-step CPU round-trip is needed.  Small, but it
composes with 2.1.2/2.1.3.

#### 2.1.5 Acceptance criteria  ← **MET 2026-09-24**

* A plain `llama-server` with **no** `OMP_*`/`KMP_*` env keeps MTP decode at ~1–2 CPU cores and
  reproduces the passive-wait throughput (**~107 t/s**, qwen4exp IQ4_NL `draft-mtp n3`, prose) with
  **byte-identical output and identical acceptance** (0.84791 / the session-6 0.83204).
  → **111.7 t/s at 1.2 cores, acceptance 0.84791, text `a79d0d14855b`** (fix build, no env; the
  passive reference on the same build was 108.5 t/s).  `llama-server` itself (the acceptance config,
  `-t 15`, `/completion n_predict 800`): **108.8 t/s / 1.4 cores** vs 98.2 / 12.0 with the kill-switch.
* Plain decode and prefill unregressed; CPU-only builds unaffected; the `qwen35`/`qwen35moe` models
  (no PLE) stay on their current numbers (already clean, §11 session 6).
  → plain 52.9 vs 53.0 t/s; CPU-only 4B `tg64` 9.69 vs 9.60; `pp2048` 2448 vs 2467 (noise).
* **Re-baseline the gfx1201 qwen4exp MTP t/s** (see OP-2) and make the harness robust: the MTP gate
  should not depend on an env var being set.  Consider teaching the reporting harness to assert the CPU
  is quiet (a core-count sanity check) so a future degenerate-scheduling regression cannot silently
  poison the numbers.
  → **done**: `tools/mtp-run.sh` + `tools/mon.py` always sample per-thread CPU and report
  `SUMMARY cpu_cores … busy_threads(last)=…` next to every run; `tools/matrix-axis.sh` is the
  four-mode axis driver.

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

### 2.2 OP-2 — re-baseline the gfx1201 qwen4exp MTP throughput  ← **DONE 2026-09-24**

Every qwen4exp MTP t/s figure measured on this box **without** the passive-wait env is ~35 % low
(session 2's 57.5 t/s, session 6's pre-fix table, …).  The session-8 numbers use
`OMP_WAIT_POLICY=PASSIVE` and are the unconfounded reference.  Do:
* re-measure the four-axis (`R`/`C`/`K`/`P`) + phase-switch (`X`) set with the passive env (or the
  OP-1 fix), `-n 3000`, reasoning pinned per axis (`benchmarks/mtp-adaptive-methodology.md` rule 0);
* correct/annotate the stale records so a future session does not cite the confounded numbers;
* add the CPU-quiet assertion to the harness (§2.1.5).

**Done** (fixed build, **no** env; table and the old→new delta in
[`2026-09-24-mtp-cpu-spin-automatic.md`](2026-09-24-mtp-cpu-spin-automatic.md) §4):

| axis | none | n7 | n8 | adaptive |
|---|---:|---:|---:|---:|
| R reasoning | 54.6 | 79.6 (0.365) | 78.1 (0.352) | **86.8 (0.592)** |
| C code | 54.8 | **139.2 (0.764)** | 138.8 (0.732) | 136.6 (0.743) |
| P prose | 53.8 | **128.1 (0.706)** | 124.1 (0.655) | 123.0 (0.685) |
| K recall | 52.4 | 163.6 (0.963) | **167.4 (0.957)** | 154.3 (0.967) |
| X phase-switch | 54.7 | **99.2 (0.490)** | 84.6 (0.390) | 98.8 (0.661) |

CPU quiet in every arm.  The old `n7 >> n8` collapse was `0028`'s W=9 cliff: with it fixed `n8` is
near-tied everywhere and **wins recall** — the `n_max 8` question is workload-dependent on gfx1201 too.

### 2.3 OP-3 — `0028` follow-ups (the MMVQ band boundary)

`0028` is verified for the cliff itself but not exhaustively:
* **full `n7`/`n8`/adaptive matrix with the fix** — session 6 covered the matrix *before* the fix; only
  the prose axis + one `n8` pair were re-measured after.  Use the same protocol (`-n 3000`, reasoning
  pinned) and the `OMP_WAIT_POLICY=PASSIVE` (or OP-1) build.
  → **DONE 2026-09-24** (all five axes, fixed build, no env — see §2.2 / the OP-1 record §4).  The old
  `n7 >> n8` gap was `0028`'s W=9 cliff; `n8` is now near-tied and wins recall.
* **per-type MoE band** — the routed-expert band floor is currently unconditional 16 on AMD; if a
  specific expert type is slower on `mul_mat_vec_q_moe` at 9..16, the floor has to become per-type (see
  the gfx1151 brief §4.5 — narrowing the per-arch table alone does nothing because the floor clamps up
  afterwards).
  → **DONE 2026-09-24: no per-type band needed** — the unconditional floor at 16 wins for every type
  in the relevant `B <= 13` range; only IQ3_XXS/IQ4_XS regress (from `B=13`), which is the purity
  trade.  See [`2026-09-24-op3-per-type-moe-band.md`](2026-09-24-op3-per-type-moe-band.md).
* **another odd-row dense model** — the RDNA4 dense rule (`nrows_x % 128 != 0`) was only exercised on
  qwen4exp; 27B/35B are the clean controls.  If one is available, A/B a dense model with odd rows.
  → **DONE 2026-09-24: none available, and the controls are confirmed clean.**  A GGUF header scan
  (`gguf` reader, all 2-D tensors) shows 27B IQ3_S / 35B-A3B Q3_K_M / gemma-4-12B have **zero**
  weights with `ne[1] >= 128 && ne[1] % 128 != 0` (their only odd rows are `ssm_alpha`/`ssm_beta`,
  ne1 32/48).  qwen4exp's odd-row dense weight is `output_hc_down.weight` (**ne1 = 320**, one per
  layer, 97 total); every other qwen4exp weight with ne1 ≥ 128 is `% 128 == 0`.  So the RDNA4 dense
  band is qwen4exp/`output_hc_down`-specific by construction and the 27B/35B controls carry no such
  weight — there is no other box model to A/B.
* **gfx1151/gfx1100 revalidation/port** is handed off — see [`gfx1151-closing.md`](gfx1151-closing.md).

### 2.4 OP-4 — validation gates not yet run

* qwen4exp **four-axis MTP set with the fix** (only prose re-measured in session 8).
  → **DONE 2026-09-24** (OP-2, all five axes incl. the phase-switch prompt).
* `llama-imatrix` (`0019`/`0023`) — the NanBeige model is absent here; substitute another BF16 model.
  Note HC16 is RDNA3_5-gated, so on RDNA4 this is really a scheduler/split regression check.
  → **DONE 2026-09-24: clean run, but `-sm tensor` imatrix is corrupt (pre-existing; perplexity fine).**
  See [`2026-09-24-op4-imatrix-and-hc-gates.md`](2026-09-24-op4-imatrix-and-hc-gates.md).
* `0001` (`hc_combine_norm`) / `0008` (M=4 HC inject) **isolated** A/Bs — they are exercised by the
  qwen4exp gates but never singled out.
  → **DONE 2026-09-24: `0001` default fusion +5 %; `0008` default `TALL_MIN_M=16` ahead (interleaved).**
* §6.6 **M-RoPE image case** (`0005`) — needs the vision projector; the gfx1151 repro was an image after
  ~12k tokens of text.  **→ DONE 2026-09-24: ran clean on the closing build AND the pre-fix baseline
  (the trigger does not reproduce on gfx1201 with the shared MTP head); `0005` retained as a port.**
  Harness: `tools/mrope-image-mtp.sh`.

### 2.5 OP-5 — remaining RDNA3_5-gated kernels (low priority)

* **`0013` prefill indexer relu+head-sum** (`idx_relu_sum`, call-site `GGML_CUDA_CC_IS_RDNA3_5`):
  `0016`'s port already banks this reduction (the fused lightning-indexer computes
  `bias + sum_h relu(dot_h)`), so a separate enablement is likely redundant on RDNA4.  Verify by
  diffing `GGML_CUDA_IDX_RELU_SUM` on/off **after** the `0016` port — if the graph no longer contains
  that chain, there is nothing to port (this is a 10-minute check).
  → **DONE 2026-09-24: REDUNDANT.**  Widened the gate to RDNA4, rebuilt, ran qwen4exp `pp8192` with
  `GGML_CUDA_IDX_RELU_SUM_LOG=1`: **0 matches** with `0016` ON (default) and **4 matches** with
  `LLAMA_QSA_SCORE_WMMA=0`.  `0016` removes the chain; the port would be dead code.  Gate change
  reverted.  Detail: the OP-1 record §5.
* **`0011` HC BF16 streams / `0023` HC16**: `0011` needs the meta `graph_optimize` path (`0027`, done)
  to run under `-sm tensor`; **← DONE 2026-09-24**: `0027` closes the gap (`comb=` 582×), the A/B is
  +1.3 % all from `LLAMA_HC_RES16` (`blk16` inert), lossy so it stays default-OFF → **parked**.  `0023`'s
  HC16 is RDNA3_5-gated and not bandwidth-bound on discrete RDNA4 → **park** unless a 48 GB single-GPU
  RDNA4 box appears.

### 2.6 OP-6 — build-time critical path (parked in `TODO.md`)

The `fattn-mma-f16` per-type instance set is the remaining clean-build critical path.  → **CLOSED
2026-09-24: no change.**  The "7.26 MB / 229 s per TU" premise is the pre-r6 state; r6's per-head split
leaves the largest TU at **2.9 MB / 82 s** against a ~236 s `-j16` makespan, so the build is
**throughput-bound** and a per-KV-type split cannot help.  The runtime-dispatch / `__noinline__`
options were rejected in r6 (build worse / -1.5-2.5 % prefill); ccache is the answer.  See
[`2026-09-24-op6-build-time-closure.md`](2026-09-24-op6-build-time-closure.md), `archive/work/build-time-regression/`
and `TODO.md`.

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

### 3.2 Apply the full stack (29-patch closing set)

The delivery repo is `~/llama-cpp-rdna-boosts`.  The current `~/llama.cpp` checkout is branch
`closing-gfx1201` = delivery r13 + `beta/mmb-general` + closing `0001..0014`/`0016..0030` (tip tree
**`99b429a60d441f814c84737cfa57803bc15a2f6d`**).  A fresh apply:

```sh
WORK=$HOME/llama-cpp-rdna-boosts
cd ~/llama.cpp
git fetch --all
git checkout ebbb18522
git checkout -b closing-gfx1201
bash "$WORK"/scripts/apply-all.sh .                      # delivery r13 (16 blocks)  -> tree bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12
git am "$WORK"/beta/mmb-general/patches/*.patch          # 12 beta patches          -> tree 79136a15cac1920c0dd334b4c119a9cb42f9143b
for p in "$WORK"/archive/work/closing-the-gap/patches/0*.patch; do
  git am "$p"
done
git rev-parse HEAD^{tree}                                # expect 99b429a60d441f814c84737cfa57803bc15a2f6d
```

> **Base (2026-09-24): the repo is now on delivery r13.**  `gap-closing` was rebased onto `main`, so
> `patches/` + `release.json` are the **r13** set (`bb7b6d07…`) and a fresh apply reproduces the branch
> tip tree `99b429a6…` exactly (29/29 `git am`, verified: block `bb7b6d07`, beta `79136a15`, closing
> `99b429a6`).

**Traps:** the superseded `0015` was removed (it is r13 block 00), so the `0*.patch` glob no longer includes it; `0024` must be applied **before** `0025` (the loop order handles
this).  The single-patch alternative is `git apply` of `archive/work/closing-the-gap/campaign-all.patch` on the
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
# With 0029 the passive-wait env is NO LONGER NEEDED for the fast path — keep it only as the A/B
# reference (add `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` for the other arm).
build-rocm/bin/llama-cli -m "$M" -md "$MD" \
  -ngl 99 -sm tensor -c 16384 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto \
  -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off -f "$P" \
  --spec-type draft-mtp --spec-draft-n-max 3 --ctx-checkpoints 0 -lv 4
# CPU-quiet gate: `bash tools/mtp-run.sh <tag> <outdir> -- <the same cmd>` reports
# `SUMMARY cpu_cores … busy_threads(last)=…` next to the run (expect steady ~1.2 cores).
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
* **Fold a fix** into an existing patch with `git commit --fixup=<commit>` then
  `GIT_SEQUENCE_EDITOR=true GIT_EDITOR=true git rebase --autosquash <commit>~1`; regenerate with
  `git format-patch -1 <sha> --stdout --no-numbered`; replace the patch file and re-verify a fresh
  r13+beta worktree + the **29** patches reproduces the branch tip tree.  A **new** change is a new
  numbered patch (`git format-patch -1 <sha> --stdout --no-numbered > patches/00NN-…patch`) and the
  `campaign-all.patch` `git diff <r13+beta tree>..<tip>` is regenerated too.
* **Do not push the `~/llama.cpp` fork**; push the delivery repo only on explicit request.

### 3.6 Report template

For each gate: the **exact command**, the **build/tree**, the **`MMB_CFG`** line, the **numbers** (and
the interleaving order for A/Bs), and the **extracted hash** where a text gate applies.  State the gate
name, not "it was slower".  For MTP use `benchmarks/mtp-adaptive-methodology.md` rule 0
(`-n 3000`, reasoning pinned).  Append results to the session log below, and move completed items into
§1 with a pointer to the detail (which belongs in `gfx1201-closed.md` once written up).

---

## 4. Session log (open-work sessions, newest first)

### 2026-09-24 — session 9d: handover for the remaining OP-3/OP-4/OP-5.2/OP-6

No measurements.  The brief was rewritten for a fresh session: **§0.5 is now the run plan** with the
build/state header for every remaining item and the exact commands/gates for
OP-3 (per-type MoE band), OP-4 (`llama-imatrix`, isolated `0001`/`0008` A/Bs, M-RoPE image),
OP-5.2 (`0011` under `-sm tensor`, expected flat) and OP-6 (`fattn-mma-f16` build time), and §2.3-§2.6
now point at it.  State at handover: `closing-gfx1201` tree `99b429a6…` (29 patches), campaign
`gap-closing` @ `4be503f`.

### 2026-09-24 — session 9c: OP-1.4 closed (won't fix) + the qwen4exp prefill re-check

**OP-1.4 — draft sampler backend offload under `-sm tensor`: CLOSED, won't fix.**  Lifting the upstream
guard aborts in the Meta backend (`ggml-backend-meta.cpp:544 handle_per_row: src_ss[0].axis !=
AXIS_0`), because `output.weight` is vocab-split so the logits are AXIS_0 and a global top-k cannot be
expressed; it needs a Meta-backend distributed top-k.  And the win is not there anyway: under `-sm
layer` (where it works) the backend-sampling A/B is within noise (**75.6 vs 75.1 t/s**, ON vs
`--no-spec-draft-backend-sampling`; same-arm spread 70.8→75.6).  Detail:
[`2026-09-24-mtp-draft-sampler-tensor-split.md`](2026-09-24-mtp-draft-sampler-tensor-split.md).  No code
change (the temporary `LLAMA_BACKEND_SAMPLING_TENSOR` probe was reverted).

**Prefill re-check (user request).**  The ~3350 t/s recollection is **qwen4exp IQ4_NL pp8192** (the
`0003` hc_gate_mix record: 3346.9/3345.8/3340.2).  On the current build: **pp8192 3381.0 ± 13.6**,
pp32768 3280.8, pp65536 3130.3 — **no regression** (slightly better).  IQ4_XS pp8192 was ~3083 in the
2026-09-23 `0016` record; the current build measures higher than the r13+beta baseline there too
(pp8192 2920 vs 2571, pp32768 3172 vs 2858; both climbing with page-cache warmth), so no regression
there either.  The shallow-pp numbers are dominated by page-cache state for these near-VRAM-limit
models (switching between the two qwen4exp models evicts the other's ~27 GiB host PLE), not by the
closing patches.

### 2026-09-24 — session 9b: OP-1 structural (opt-in `0030`) + the r13 rebase

**Rebase.**  `gap-closing` was rebased onto `main` (r13); the repo's `patches/`/`release.json` are now
r13 and the documented fresh apply reproduces the branch tip tree exactly.  Only `WORKLOG.md`
conflicted (main's r13 entry vs the campaign's 2026-09-22/21 entries); both sides kept, r13 first.

**OP-1 structural — investigated, folded as opt-in `0030` (`LLAMA_DEVICE_INPUT=1`).**  The structural
fix works: it moves the small `token_embd` table to the output/Meta device and the `GET_ROWS` runs in
the GPU graph, giving **0 CPU splits** and ~1.0 cores even with `0029` disabled.  But the Meta-split
GPU gather is **~2.6 % slower** MTP than the host input + `0029` single-thread gather (interleaved A/B
`-n 1500`: host **111.4/111.4/111.4**, device **108.7/107.5/108.8**), so it is **not** defaulted.
Acceptance and same-seed text are identical (`0.85417` / `a79d0d14855b`).  Candidate 4 (device reads the
host-mapped table) is not reachable under `-sm tensor` (the Meta backend is the only GPU backend and
rejects host buffers).  Full detail:
[`2026-09-24-mtp-cpu-spin-structural.md`](2026-09-24-mtp-cpu-spin-structural.md).  Validated: 27B IQ3_S
`6073add19dac` (1 GPU + 3-GPU tensor), 35B-A3B plain == `draft-mtp`, `-sm layer` plain == `draft-mtp`,
`pp2048` 2475 t/s, fresh apply 29/29 → tree
`99b429a60d441f814c84737cfa57803bc15a2f6d`.

**Still open (next-session plan in §0.5):** OP-3 per-type MoE band, OP-4 (`llama-imatrix`, isolated
`0001`/`0008` A/Bs, M-RoPE image), OP-5.2 `0011`, OP-6 build time.  (OP-1 is fully closed: `0029`
shipped, `0030` opt-in, 1.4 won't fix.)

### 2026-09-24 — session 9: OP-1 automatic CPU-spin fix (`0029`) + OP-2/OP-3 re-baseline + OP-5.1

Full detail: [`2026-09-24-mtp-cpu-spin-automatic.md`](2026-09-24-mtp-cpu-spin-automatic.md).

**OP-1 runtime half DONE.**  `ggml_backend_cpu_graph_compute` now runs a tiny CPU split graph
(≤ 32 nodes / ≤ 16 MiB of node outputs — the host-mapped input/PLE `GET_ROWS` and the MTP draft's
state copies) inline on the calling thread, so no OpenMP region is entered and the active-wait pool
never spins.  Automatic, default-on, arch-neutral, disable-only kill-switch
`GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1`; folded as
[`patches/0029`](patches/0029-gap-closing-WIP-run-tiny-CPU-split-graphs-on-the-calling-thread.patch)
(tip tree `fa9cf6d1e654333d458ade3655c4a0d540225827`, `campaign-all.patch` regenerated).

* qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, prose, `-c 16384 -n 1500`, seed 42:
  default **81.0 t/s / ~15 cores → 111.7 t/s / 1.2 cores**, passive-wait reference 108.5 / 0.4 cores,
  kill-switch 85.1 / 14.9.  Acceptance **0.84791** in every arm.
* Purity: `plain == draft-mtp n3` text **`a79d0d14855b`**, and the heuristic on/off gives the same
  hash.  Plain decode 52.9 vs 53.0 t/s; CPU-only 4B `tg64` 9.69 vs 9.60; `pp2048` 2448 vs 2467.
* **OP-2 / OP-3 matrix DONE** (fixed build, **no** env): five axes × `none`/`n7`/`n8`/adaptive cap 8,
  `-n 3000`, reasoning pinned, CPU quiet everywhere.  The old `n7 >> n8` gap was `0028`'s W=9 cliff —
  with it fixed `n8` is near-tied on R/C/P and **wins recall** (167.4 vs 163.6); adaptive keeps R
  (86.8) and X (98.8).
* **OP-5.1 DONE: `0013` is redundant on RDNA4** — with the gate widened to RDNA4 the matcher fires
  0× with `0016` ON and 4× with `LLAMA_QSA_SCORE_WMMA=0`; gate change reverted.
* Harness added: `tools/mon.py` (per-thread CPU sampler), `tools/mtp-run.sh`, `tools/matrix-axis.sh`;
  every run now reports `SUMMARY cpu_cores … busy_threads`.
* **Still open:** OP-1.2 structural (input/PLE off the CPU), OP-1.4 draft-sampler offload, OP-3
  per-type MoE band + odd-row dense model, OP-4 `llama-imatrix` / `0001`+`0008` A/Bs / M-RoPE image,
  OP-5.2 `0011`, OP-6 build time.

### 2026-09-24 — brief split

`gfx1201-closing.md` was split: all closed work moved to [`gfx1201-closed.md`](gfx1201-closed.md), this
file now tracks only the remaining items (OP-1…OP-6).  State at the split: closing set 27 patches,
tip tree `533eee3188ab7df9b6cf394adeaa31b46bd13ff2`; `0028` (the W=9 cliff) folded and verified; the
CPU-spin drop re-confirmed live (79.3 → 107.2 t/s with the passive env).  Previous sessions 1–8:
`gfx1201-closed.md` §11.
