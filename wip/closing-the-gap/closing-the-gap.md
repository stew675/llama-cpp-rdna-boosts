# Closing the gap — `beta/mmb-general` vs the other solution's `strix-halo` prefill

**Date:** 2026-09-20 (snapshot) · **updated:** 2026-09-22 (end of session 3)
**Box:** `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`), 123 GiB RAM / 124 GB unified VRAM
**Scope:** a 1:1 prefill comparison on **the other solution's uniform-IQ4_NL model** (not just our mixed UD-IQ4_XS), a kernel-level profile diff on the uniform model, and a gate-ablation of the other solution's stack on this box to price the still-missing families. This is an investigation record, not a delivery change.

> **Fresh session: read this whole file; the "START HERE" block is the handoff.**  The body below
> (§0–13 + the dated records) is the dated investigation record, kept for its measurements.

---

## START HERE — fresh-session handover (end of session 3, 2026-09-22)

### Where we are

* **Target is `-b 8192 -ub 8192`.**  For **long-context** (pp65536+) use **`-b/-ub 4096`**: at
  `-ub 8192` that point is right at the memory limit and the GPU oscillates (memory shortfall) at
  ~844 t/s, while `-ub 4096` stays pegged at 100 % and runs **1093 t/s** (maintainer, 2026-09-22).
  `-ub 16384` is root-caused and parked (below); do not spend time on it unless the PLE-lazy fix is
  picked up.
* At a **matched** ubatch we were **~10 % behind at pp8192 / ~15 % behind at pp16384** on the uniform
  IQ4_NL model — the number the §13 phase plan closes; items 1+2 have since narrowed it.
* **Phase-1 item 1 (HC fusions) DONE** (session 2, **`patches/0003`**): `hc_combine_norm` matcher
  revived (+1.5 %) and `hc_gate_mix` wired default-on (+1.2–1.5 %) — width-pure, same-seed-text
  identical.
* **Phase-1 item 2 (depthwise conv1d) DONE** (session 3, **`patches/0004`**): `gdn-conv.{cu,cuh}` +
  `ple-conv.{cu,cuh}` ported, default-on (`GGML_CUDA_DISABLE_CONV_FUSION=1` disables), **bit-identical**
  (fused == unfused row-0 hash + width probe PASS), **+3.0/+3.2 %** qwen4exp IQ4_NL and **+6.5/+7.1 %**
  35B-A3B (pp8192/32768, `-ub 8192`) —
  [`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md).
* **Phase-1 item 3.5 first fix DONE** (session 3, **`patches/0005`**): the QSA block window is sized by
  the highest stored position (`b0f31f587`), fixing the M-RoPE-image + MTP assert —
  [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md).
* **CLOSED NEGATIVE — do not redo:** item 1's `hc_combine_norm_f32_b256` swap (not bit-identical — it
  changes the greedy text `1b59d651f2c3` → `fc7c8a10ea45` — and 0.7–0.8 % slower; the reference's
  554 ms is its **BF16** HC traffic (`hc16`/`blk16`/`res16` compiled in), not the thread count) and
  item 5's `concat_transposed` drop (already gone at `-ub 8192`; the remaining BF16 MoE epilogue is a
  lossy/memory candidate, not a 3–4 % win) —
  [`2026-09-21-hc-cn-b256-rejected.md`](2026-09-21-hc-cn-b256-rejected.md).
* **Next:** the **two remaining item-3.5 QSA correctness fixes** (`40c0b9c38` maskless-only-where-qsa3-
  consumes, `14fff4f97` −1 sentinels) — an **audit** against our derived-visibility QSA (scoping below)
  — or **Phase-1 item 4** (`norm-gated.cu`/`rms_rows`, ~1.2 % on our tree; `idx-relu-sum` is already
  banked by our fused indexer score), then **item 6** (`qsa3_attn` body, 817 vs the reference's 618 ms),
  item 7 (tall tile), item 8 (QSA graph flags).

### Do these in order

1. **Run the full `beta/mmb-general` BETA-TESTING gate suite on the current default build**
   ([`../../beta/mmb-general/BETA-TESTING.md`](../../beta/mmb-general/BETA-TESTING.md)).  Gate semantics:
   **Gate 1 = `GGML_CUDA_MMB=0`** (byte-identical to r12), **Gate 2 = the default** (no env).  Plus the
   width probe and the MTP gate.  Green before any promotion, and do not trust performance numbers as
   "the product" until then.  **Still owed: the MTP gate (Gate 4) and the MMB-off byte check (Gate 1).**
   Already green (sessions 2-3): `GATED_DELTA_NET`, `INDEXER_TOPK`, `FLASH_ATTN_QSA` 26/26,
   `FLASH_ATTN_EXT` 5955/0, width probe PASS on 27B + qwen4exp + 35B-A3B, conv fused==unfused row-0
   hashes.
2. **Next code item** — either the remaining item-3.5 QSA audit (scoping below) or item 4
   (`norm-gated`).  Then item 6 (`qsa3_attn`).
3. Keep the **default-on policy**: every beneficial feature is ON; its env var only *disables* it.  Never
   run a benchmark with a feature left off.

### Rebuild / run (copy-paste)

```sh
cd ~/llama.cpp && ~/bin/build-llama-rocm-714                      # full build (ccache; ~4 min warm)
# fast loop:  cmake --build build-rocm --target llama-bench llama-cli -j 16
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MM=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
for f in ${MU%/*}/*-0000*.gguf; do cat "$f" >/dev/null; done   # warm page cache first
# ubatch-8192 baseline (ours, full default set, no env):
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 8192 -ub 8192 -p 8192,32768 -n 0 -r 2
# long context: use -ub 4096 (at -ub 8192 the pp65536 point memory-thrashes at ~844 t/s;
# -ub 4096 is clean and pegged at 100 %, ~1093 t/s):
~/llama.cpp/build-rocm/bin/llama-bench -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 4096 -ub 4096 -p 65536 -n 0 -r 1
# gates used repeatedly (qwen4exp IQ4_NL):
#   width probe:   ~/llama.cpp/build-rocm/bin/test-logits-width-probe "$MU" \
#                    prompts/prose-rdna-boosts.txt 1024 512     (expect width_purity=PASS, worst maxdiff 0)
#   op oracles:    test-backend-ops -o GATED_DELTA_NET / INDEXER_TOPK / FLASH_ATTN_QSA / FLASH_ATTN_EXT
# the other solution's same-config reference (its launcher env):
( set -a; . archive/work/wip-archive/iq4nl-prefill/launcher-env.txt; set +a; \
  ~/pwilkin-llama-cpp/build-rocm/bin/llama-bench -m "$MU" -dev ROCm0 -ngl 999 -fa on \
  -lm none -lzm on-direct -ctk f16 -ctv f16 -b 8192 -ub 8192 -p 2048,8192,16384 -n 0 -r 2 )
```

### Next-task scoping — the two remaining item-3.5 QSA correctness fixes (audit)

Our tree was at the **pre-fix** state for all three of the reference's correctness commits, so item 3.5
is a port, not a mere audit.  The **first is done** (`b0f31f587`, `patches/0005`).  The other two target
the reference's `tail_idxs` / `compact` / `maskless` design, which our QSA does **not** share (ours has
`cell_vis`/`q_vis` derived visibility + `blk_idx`/`blk_tail`), so each has to be mapped to our
equivalent first:

* **`40c0b9c38` — maskless only where the qsa3 kernel consumes it.**  The reference's `LLAMA_QSA_NO_DENSE_MASK`
  made decode non-deterministic (10/10 → 1/10 identical greedy outputs) because maskless was decided
  from the graph input alone, while qsa3 also needs both packed layouts and `>= 128` queries (true in
  prefill, false in decode), so every decode step attended unmasked over stale cells.  The fix decides
  maskless at the use site and asserts the invariant in the dispatcher.  **Audit question for us:** our
  maskless/derived path is `LLAMA_KQ_MASK_DERIVED` + `GGML_QSA_DERIVED_VIS`; check where our derived
  visibility is decided vs where the kernel that consumes it is chosen, and whether a decode step can
  take a maskless path over stale cells.  The instrument is **greedy determinism over N identical
  requests** (the reference's 10/10 → 1/10), not throughput.
* **`14fff4f97` — keep the −1 selection sentinels out of the masked attention path.**  In the reference,
  complete-block selection lists selected cells with −1 for invisible blocks / empty tail slots, which
  only the maskless selected-key kernel understands; the tails were allocated whenever block selection
  applied, so a non-scalar visibility (2-D image positions, several sequences) sent the selection to
  the **masked** path, whose `set_rows` wrote row −1 (illegal memory access on gfx1151, reproduced with
  an image after 12k tokens of text).  The fix gates the tail allocation on `scalar` and uses the
  block-expanded top-k otherwise.  **Audit question for us:** our `blk_idx` uses −1 (incomplete block)
  and `INT32_MAX` (spare tail block) sentinels — find every consumer that could reach a `set_rows` or a
  masked path and confirm the sentinels are only ever consumed by the derived/maskless path.

Read the reference diffs with `git -C ~/pwilkin-llama-cpp show 40c0b9c38` / `14fff4f97`; the files to
map are `src/models/qwen4exp.cpp` (`qwen4exp_use_block_selection`, `qwen4exp_select_complete_blocks`)
and `src/llama-memory-hybrid-idx.{h,cpp}`.  If a port is not directly applicable, record the audit
result (present / N/A + why) in the item-3.5 record rather than forcing a change.

### Alternative next item — Phase-1 item 4 (`norm-gated` / `rms_rows`)

The reference's `norm-gated.cu::rms_rows_f32` is a wave-per-row RMS norm for narrow rows
(`ncols <= 256`) with an optional sigmoid gate, worth ~1.2 % on our tree (our narrow-row norms already
run as `rms_norm_f32<256,{true,false}>`; the reference's pair is ~135 ms less).  It claims to be
**bitwise identical** to `norm.cu`'s `rms_norm_f32<256,...>` (per-warp xor trees + a xor tree over the
8 partials).  **Gate it the same way item 2 was gated:** verify the reduction order against our
`rms_norm_f32<256>` first (the `_b256` lesson: a reduction-order mismatch changes the greedy text and is
not shippable), then width probe + same-seed text + an A/B.  `idx-relu-sum` is **already banked** (our
fused `GGML_CUDA_QSA_INDEXER_SCORE` computes `bias + sum_h relu(dot_h)` in one kernel).

### Current state (exact)

| what | where / value |
|---|---|
| fork `~/llama.cpp` | branch **`gap-closing`** @ **`9449f3446`** = r12 + the 12 `beta/mmb-general` patches + the 5 gap-closing commits |
| fork build | `~/llama.cpp/build-rocm` (gfx1151, ROCm 7.14), full feature set **default** |
| this repo | branch `gap-closing` (published to `origin`), `wip/closing-the-gap/patches/0001..0005` |
| the other solution | `~/pwilkin-llama-cpp` @ `b0f31f587`, `build-rocm` |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93 GiB, qwen4exp) |
| MoE test model | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| MTP sidecar | `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (the IQ4_NL `shared-Q8_0` head does **not** load — see §12) |

Rebuild: `cd ~/llama.cpp && ~/bin/build-llama-rocm-714`.  Runtime:
`export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0`.
The five `gap-closing` fork commits are exported to [`patches/`](patches/) so the code survives a fork
reset.

### What NOT to redo (session-3 conclusions)

* **`hc_combine_norm_f32_b256`** — not bit-identical, slower; the reference's speed is its BF16 HC
  traffic.  If HC-combine speed is revisited, the change is the **BF16** `blk16`/`res16` path (lossy,
  needs the maintainer's call), not the thread count.
* **`concat_transposed` drop (item 5)** — already gone at `-ub 8192`.
* **`-ub 16384`** — parked; the `-ub 8192` long-context point memory-thrashes, so use `-ub 4096` there.
* **MMB-off byte-identity and the MTP gate** — still owed, not yet done.

---

## Session-3 record (2026-09-21): Phase-1 item 2 — the depthwise conv1d is DONE

### Result

The other solution's `gdn-conv.{cu,cuh}` + `ple-conv.{cu,cuh}` are ported into the delivery, default
**ON** (`GGML_CUDA_DISABLE_CONV_FUSION=1` disables).  Fork tip **`1004c65db`**, exported as
[`patches/0004`](patches/0004-gap-closing-WIP-port-the-depthwise-conv1d-fusions-GD.patch).  Full record:
[`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md).

| model | pp | off | default (fused) | delta |
|---|---:|---:|---:|---:|
| qwen4exp IQ4_NL | 8192 | 1223.1 | **1259.9** | +3.0 % |
| qwen4exp IQ4_NL | 32768 | 1158.4 | **1195.6** | +3.2 % |
| qwen35moe 35B-A3B Q4_K_M | 8192 | 2128.1 | **2265.9** | +6.5 % |
| qwen35moe 35B-A3B Q4_K_M | 32768 | 1561.9 | **1672.5** | +7.1 % |

**Bit-identical**: the width-probe row-0 logits hash is identical fused vs disabled
(qwen4exp `268e0673300b7a33`, 35B-A3B `e97c9e304ce1ca8f`), `width_purity=PASS (worst maxdiff 0)` on
both, and same-seed greedy text is identical (`bb820cccf620`, 637 chars).

### Two adaptations our tree needed

1. `ple_conv_check`: our post-re-base `grouped_norm` emits a **3-D** `[n_embd, hc, T]` MUL, so the
transpose's view root has `ne[2] = hc`; the reference's `x->ne[2] == 1` test rejected every PLE layer.
The check now derives `C`/`T` from the transpose and only requires a contiguous F32 root of `C*T`
elements (the flat layout is the same `[hc_dim, T]` buffer either way).
2. `gdn_conv_check`: the shared `build_conv_state` (qwen35moe/qwen35/qwen3next) writes the snapshot
`cpy(view(concat), dst)` with a raw view; the sweep rejected that CPY.  It now accepts a CPY whose
source is a 3-column view of the concat (covered by `tail_from`) and rejects any other concat reader.

### Gates still owed (unchanged from session 2, plus the conv fusion)

* The **MMB-off byte-identity** check and the **MTP** gate (Gate 1/Gate 4).
* The **27B dense** width-probe run (session 2/3 ran qwen4exp + 35B-A3B).
* Re-check the delivery's **MoE/general GDN prefill records** now that the GDN fusion also fires on
qwen35moe/qwen35/qwen3next (the snapshot-cpy adaptation); the 35B-A3B numbers above are the first
signal.

---

## Session-2 record (2026-09-21): ubatch 8192 target, the 16k diagnosis, and the gatemix win

### Decision — ubatch 8192 is the target

Target **`-b 8192 -ub 8192`** for the head-to-head and all further prefill work.  The `-b 16384 -ub 16384`
regime is parked for a later session.

Uniform IQ4_NL (the other solution's checkpoint), gfx1151, `-ctk f16 -ctv f16`, `-n 0 -r 2`, ours with
**no env** (full default set), the other solution with its full launcher env:

| pp | ours, `-ub 8192` (pre-gatemix) | other, `-ub 8192` | ours, `-ub 2048` (old) | other, `-ub 16384` |
|---:|---:|---:|---:|---:|
| 2048  | 1159.1 | 1233.6 | 1181.5 | 1233.0 |
| 8192  | 1212.6 | 1346.5 | 1149.3 | 1338.8 |
| 16384 | 1179.0 | 1387.5 | 1129.2 | 1399.3 |

So ubatch 8192 is itself a real gain over our ubatch 2048 (+5.5 % at pp8192) and gives a stable,
reproducible baseline: at a **matched** ubatch we are **~10 % behind at pp8192 / ~15 % behind at
pp16384**, which is the number the §13 phase plan exists to close.  It also confirms the old "~4 %
behind" was partly the ubatch mismatch (ours ub2048 vs its ub16384).  Session 2 then added gatemix
(+1.2–1.5 % at pp8192/32768); re-measure the baseline with the current default before comparing.

### Gatemix — Phase-1 item 1's second half, DONE

The `hc_gate_mix_kernel` + `ggml_cuda_hc_gate_mix` existed in `mmb.cu` with no call site, so the HC gate
GEMM (`w_up @ lo`, `[320 -> 10240]`) dispatched standalone and the mix ran in the `dsv4_hc_pre` op.
Session 2 wired the matcher/call site in `ggml_cuda_try_fuse` (`94694a38e`, `patches/0003`):
`ggml_cuda_hc_mix_closed()` plus a branch at the gate `MUL_MAT` that handles **both** the unfused chain
and our delivery's explicit `ggml_dsv4_hc_pre` op (recovering the `[hc*n_embd, T]` activation the bf16
cache is keyed on from the gate GEMM's own activation input).  Default **on** on gfx1151
(`LLAMA_HC_GATEMIX=0` disables); RDNA4/RDNA3_0 stay off.

| pp | `LLAMA_HC_GATEMIX=0` | default | delta |
|---:|---:|---:|---:|
| 8192  | 1198.8 | **1212.6** | +1.2 % |
| 32768 | 1138.6 | **1155.6** | +1.5 % |

Width probe **PASS** (worst maxdiff 0) and same-seed greedy text **identical** (`471d102e7b7d`).  The
fusion shifts prefill logits by a bf16-epilogue ULP (three width-pure variants; the text is stable) and
is prefill-only, so the decode/verify band is untouched.  Full detail:
[`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md).  Caveat: the kernel is **IQ4_NL-only**,
so the mixed UD-IQ4_XS model is unchanged (Q8_0 gate) — a follow-up.

### Why `-ub 16384` fails (root cause, deferred not fixed)

`llama-bench -b 16384 -ub 16384 -p 16384` fails at `llama_init_from_model` because the single pp-graph
reserve needs a **33794 MiB** compute buffer and `cudaMalloc` returns OOM.  Measured breakdown at
`n_tokens = 16384` (allocator trace):

| term | size | notes |
|---|---:|---|
| `result_output` | 15520 MiB | `[n_vocab=248320, n_tokens]` f32 — the reserve uses `n_outputs = n_tokens` |
| qwen4exp HC pin | ~18114 MiB | every layer's `block_out` pinned as a graph output (~1.1 MiB/token) |
| graph working set | ~160 MiB | reused heavily |

* **The other solution allocates the same 15520 MiB `result_output`** (verified by enabling
  `GGML_ALLOCATOR_DEBUG` in its tree and running the same config) — its pp reserve is
  `n_outputs = 16384` too.  So this term is **not** something they solved and we did not; it is
  upstream's worst-case logits reserve.  Its total is 16980 MiB because it has **no HC pin**; our
  unpinned total is 15680 MiB — i.e. our graph is already ~1.3 GiB *smaller* than theirs (block-15
  W4/V3 work), and the entire 18 GiB excess is the pins.
* The pin lives in `src/models/qwen4exp.cpp::build_hc_combine` (`ggml_set_output(block_out)` /
  `ggml_set_output(inject)`) and exists so the fused `hc_combine_norm` matcher can read the **narrow**
  `block_out` base without the allocator reusing its buffer for the norm output.  It is what lets the
  matcher run at all; without the pin the matcher declines and the unfused (bit-identical but ~8 %
  slower) chain runs.  At `n_tokens <= 8192` the pin is affordable (~9 GiB); at 16384 it is not.
* A `nt <= 8192` guard on the pin **does** make 16384 create a context and run (verified: mixed IQ4_XS
  `-b/-ub 16384 -p 16384` -> 1069 t/s, compute reserve 15.68 GiB), and is bit-identical to the pinned
  path (`test-logits-width-probe` W=1..8 worst maxdiff 0).  It was **reverted** with the rest of the
  experiments because it is a workaround that costs the fusion above 8192; not needed for the 8192 target.
* **Do not ship the "matcher without the pin" path unverified.**  With the flag requirement removed and
  the pin off, the matcher fires but the logits differ from both the pinned fusion and the unfused chain
  (width probe W1 hash `5b7861bc` vs `02ece229`) — the alias check missed a real overlap.  The pin/flag
  gate is load-bearing.

### The PLE residency (the other half of the memory picture)

The 27.45 GiB `ROCm_Host` model buffer is **entirely `per_layer_token_embd.weight`** (27466 MiB; the
remaining `token_embd.weight` 644 MiB).  Findings:

* With **`-lzm on`** the PLE is mmap-lazy: it moves to a 26.8 GiB **CPU** mapping and `ROCm_Host` drops to
  0.63 GiB.  This is exactly what the other solution's launcher gets from `-lzm on-direct` (its summary:
  `ROCm0 67591 / ROCm_Host 341 / CPU_Mapped 27465` MiB).
* With the **default `-lzm auto`** our `llama-bench` does **not** lazy-load it: the loader sees
  `lazy_read::mode == 0 (OFF)`.  `params.lazy_mode` is parsed as AUTO (1) in `llama-bench`
  (`LAZYPARSE v0=1`) but `lazy_read::add` sees 0, so the value is lost between `to_llama_mparams()` and
  `llama_model_load()`.  `llama-bench` also has no `--lazy-buffer-size`, so the managed ~5 GiB capped
  reader is unreachable from it.
* **Candidate fixes for the later 16k session** (in order of preference): (a) fix the lazy-mode
  propagation so the PLE is mmap-lazy by default (frees ~27 GiB of pinned host memory and likely makes
  the pinned 16384 reserve fit without touching the HC matcher); (b) expose `--lazy-buffer-size` in
  `llama-bench` and use the managed ~5 GiB reader; (c) keep the `nt <= 8192` pin guard as a fallback.
  (a)/(b) preserve the HC design, which is the point.

### Current state (exact)

| what | where / value |
|---|---|
| this repo, `main` | `== origin/main == 830770a` (clean) |
| this repo, `gap-closing` | **the WIP branch; all session-2 work is committed here** (`b8aeaa1`, `c1c0b33`) |
| fork `~/llama.cpp` | branch **`gap-closing`** @ **`94694a38e`** (local; based on `mmb-beta` = r12 + the 12 `beta/mmb-general` patches + the three gap-closing WIP commits) |
| fork build | `~/llama.cpp/build-rocm` (gfx1151, ROCm 7.14), built 2026-09-21; full feature set **default** (incl. `hc_gate_mix`) |
| pre-port WIP (reference) | `~/llama-wip-mmb` @ `90bf12997` (`wip-mmb-general`), build at `build-rocm` |
| the other solution | `~/pwilkin-llama-cpp` @ `b0f31f587`, **rebuilt** (`build-rocm`) |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93 GiB, qwen4exp) |
| MTP sidecar | `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (**not** the IQ4_NL dir's `shared-Q8_0` — see §12) |

Rebuild: `cd ~/llama.cpp && ~/bin/build-llama-rocm-714`.  Runtime:
`export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH; export HIP_VISIBLE_DEVICES=0`.
The **three** `gap-closing` fork commits are also exported to [`patches/`](patches/) (`0001..0003`, tip
`94694a38e`) so the code survives a fork reset.

### The policy (also in `AGENTS.md`)

**A beneficial feature that has passed the gates is ON by default; its env var only DISABLES it.**  Applied
in `0c860fe77`: `GGML_CUDA_MMB` default **on** (`=0` disables; RDNA3_0 included, `GGML_CUDA_MMB_RDNA3=0`
disables that arm); MMB `hc16` default **on** (`GGML_CUDA_MMB_HC16=0` disables); the HC `hc_combine_norm`
matcher default **on** (`LLAMA_FUSED_DSV4_HC_POST=1` forces the slower `DSV4_HC_POST` op); the HC
**`hc_gate_mix`** default **on** on gfx1151 (`LLAMA_HC_GATEMIX=0` disables; session 2).  **Never** run a
benchmark with a feature left off — if you type `FOO=1 <bench>`, ask why the default is not already `1`.

### What changed this session (vs the 2026-09-20 snapshot)

* **MMB/HC16 were opt-in and I benchmarked with them off** (835 t/s instead of 1136).  Fixed by the policy
  above; the real default is now the full set (body §G).
* **Found and fixed a latent bug:** the two graph-optimizer HC16 marking sites in `ggml-cuda.cu` read a
  hardcoded `getenv(...) : 0` and **ignored the arch config**, so HC16 never engaged even when the config
  said on.  Now default from the config.
* **Revived the `hc_combine_norm` prefill matcher** (fork `121ad7935`): three bugs — see
  [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md).  The matcher default now beats
  `DSV4_HC_POST` by ~+1.5 % prefill.
* **Session 2: wired the `hc_gate_mix` fusion and made it default-on** (fork `94694a38e`, `patches/0003`)
  — +1.2–1.5 % at pp8192/32768, width-pure, text-identical.  See `2026-09-21-hc-combine-norm.md`
  and the session-2 record above.  Phase-1 item 1 is therefore done.  Caveat: the kernel is IQ4_NL-only,
  so the mixed UD-IQ4_XS model is unchanged.
* **Session 2: ubatch 8192 adopted as the target** and the `-ub 16384` failure root-caused and deferred
  (see the session-2 record above).
* **MTP qualified** (parked): see [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md).
  Our plain decode is ahead, fixed-depth MTP speedup is at parity, and the only real MTP gap is
  `nextn_shared_target_tensors` support.

### Numbers to reproduce (gfx1151, qwen4exp IQ4_NL, prefill, `-b/-ub 2048` — session-1 record, superseded
by the ubatch-8192 table above)

| pp | **default (full set)** | `GGML_CUDA_MMB=0` | pre-port WIP (full set) |
|---:|---:|---:|---:|
| 2048 | 1158.6 | — | 1167.1 |
| 8192 | 1136.2 | 835.1 | 1136.4 |
| 32768 | 1070.9 | 811.1 | — |

### Strategic framing (maintainer, 2026-09-21)

We **should** be significantly faster: our chunked GDN graphs are **~5× faster than the other solution's tiled GDN**,
and our MMB optimizations are more refined.  The recurring pattern is that we optimize very well once we
find what is missing — the other solution simply **has more things**.  So the work is to keep finding the missing
pieces (the HC `gate_mix`/combine+norm fusions, the depthwise conv1d, the gated RMS-norm, the indexer
relu-sum, the MoE bf16 epilogue, the sparse QSA decode + incremental indexer) and land them default-on.

### Known caveats before touching the fork

* The fork branch `gap-closing` carries **env-gated debug traces** (`LLAMA_HC_CN_DEBUG` in `ggml-cuda.cu`
  and `ggml.c`); they are inert by default but should be removed before any patch is cut.
* The revived `hc_combine_norm` matcher and the `hc_gate_mix` fusion have passed the **width probe and
  same-seed text** gates (uniform IQ4_NL) but not the full BETA-TESTING suite; the MMB-off byte check and
  the MTP gate are still owed.
* `LLAMA_FUSED_DSV4_HC_PRE`/`_POST` env toggles are WIP A/B knobs.
* **Follow-up:** the delivery's `hc_combine_norm_f32` is the 1024-thread/3-column variant; the reference's
  `hc_combine_norm_f32_b256` (256 threads, two packed elements/thread) is the obvious kernel swap behind
  the same matcher.  Also `hc_gate_mix_kernel` is IQ4_NL-only (Q8_0 for the mixed models is a follow-up).
* The `-ub 16384` bug is **out of scope for MMB**; it is a delivery graph/allocator issue (see the
  session-2 record).

---

## Update 2026-09-21 (session 1) — current state of play

> Session-1 record; the session-2 handoff is the START HERE block at the top.  This section still has
> the reference inventory (what each side has, commit deltas, priorities) that the phase plan builds on.

This section supersedes the stale references in the body. The body's measurements remain valid as
**dated, gated-tree** evidence, but two references have moved and the plan needs five additions.

### A. Our side: `wip/mmb-general` was promoted to `beta/mmb-general`, 5 → 12 patches

The body compared a **5-patch, gfx1151-only WIP** at tip `90bf12997` (`~/llama-wip-mmb`). The current
reference is **`beta/mmb-general`** — **12 patches**, applied tree
**`bca69f23dd29acef2d8898c6fd492104e078eef1`**, verified `git am` **12/12** on top of the r12 delivery
(`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).

* The new patches are the **gfx1201 (RDNA4) port** (0006–0010) and the **gfx1100 (RDNA3_0) deltas**
  (0011–0012). They are arch-scoped; on **gfx1151** the code is the core the body measured, so the body's
  gfx1151 numbers carry over except where §C says otherwise.
* **What the 12-patch beta still does NOT add** (grep of the applied tree, not inference):
  `gdn-conv.cu`, `ple-conv.cu`, `norm-gated.cu`, `idx-relu-sum.cu` and `hc-cn.cu` are still **absent**
  (items 2–6 below).  **Item 1 is done in session 2**: `hc_gate_mix_kernel` was wired (the matcher/call
  site, not the kernel) and is default-on on gfx1151 — see the session-2 record at the top.  So body
  §8.2 **items 2–6 remain open**, item 1 is closed.

### B. The other solution's side: `f5daaa3cf` → `b0f31f587`, 10 new commits

The body pinned `f5daaa3cf` (2026-09-12). The branch tip is **`b0f31f587`** (2026-09-16). The delta:

**Prefill-relevant**

| commit | what | why it matters here |
|---|---|---|
| `40a9f4d01` | *hip: extend MMB quants and fuse Flash-Next F32 PLE* | MMB quant coverage goes from a handful of types to **23** (adds Q4_0/Q4_1/Q5_0, Q2_K, the whole IQ1/IQ2 family, MXFP4, NVFP4) through a new `mmb-quant.cuh` generic dispatcher; the direct **PLE conv** now also takes **F32** weights (Flash-Next's PLE weights). Real gap: our beta covers **10** types and has **no** `ple-conv.cu` at all. |
| `40c0b9c38` | *qsa: drop the dense mask only where the qsa3 kernel will consume the op* | Correctness **and** a prefill win on its tree: the mask-forced workaround cost `pp16384` 945.68 → 1067.80 t/s; the fix decides maskless at the use site and `GGML_ASSERT`s the invariant in the dispatcher. The failure mode it fixed was decode non-determinism (10/10 → 1/10) that collapsed long sessions into repetitions. Our derived-visibility/maskless path should be audited against this. |

**Decode / MTP-relevant**

| commit | what | why it matters here |
|---|---|---|
| `d67d58836` | *hip: enable sparse QSA decode and incremental indexer state* | New **sparse selected-cell decode** kernels (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh` WMMA) that read selected F16 K/V cells directly, plus an **incremental indexer-key cache** (`src/qsa-prefix-state.h`, `llama-memory-hybrid-idx.*`). Measured on its tree: serial depth-40000 **25.85 → 28.82 t/s**, MTP 40680-token **31.17 → 35.57** (first) / **32.69 → 39.10** (repeat), for 104 MiB @65k / ~416 MiB @256k of cache. This is a **new axis** the body only gestured at (its item 11). Our delivery has a *different* QSA-sparse-FA decode path plus an incremental **derived-block-vector** cache (`GGML_CUDA_QSA_INDEXER_CACHE`, default on) — overlapping, not equivalent; needs a 1:1 audit. |
| `0f2950198` | *qwen4exp: skip unused HIP decode indexer work* | A temporary dense-decode bypass, **superseded** by `d67d58836`. Listed only so it is not mistaken for the current state. |

**Correctness / housekeeping**

| commit | what |
|---|---|
| `b0f31f587` | QSA block window sized by the highest stored position, not the occupied-cell count (fixed an M-RoPE image + MTP crash ~300 tokens after an image). |
| `14fff4f97` | Keep the `-1` selection sentinels out of the masked attention path. |
| `ac1ebb4e0` | **Compile in the tuned defaults and drop the env gating** — `mmb_enabled()`, `gdn_conv_enabled()`, `norm_gated_enabled()`, `norm_rows_enabled()`, `ple_conv_enabled()` now return `true`, and the `LLAMA_*` experiment switches are gone. |
| `0cfb81512` | Drop stale comment references to the removed gates. |
| `be905cf7d` | Recurrent cache: no warning for positions in a stateless cache. |
| `31b38632c` | server: a zero draft length means speculation off. |

### C. Impact on the plan

1. **The §6 ablation price-list can no longer be reproduced against the other solution's current HEAD.**
   `ac1ebb4e0` deleted the env switches Appendix D zeroed. Re-measure against `b0f31f587` as a *default*
   build, or bisect by reverting the compiled-in defaults; do not re-run the old env ablations.
2. **The prefill gap inventory (§8.2 items 1–6) is unchanged** — the beta set did not close any of them.
   Item 3 is now *larger*, because the other solution also fuses the **F32 PLE** conv.
3. **Item 11 is promoted from a footnote to a first-class decode item** (sparse QSA decode + incremental
   indexer). It is the one newer feature the other solution has that is a measurable, self-contained optimisation
   rather than a refinement, and it is orthogonal to the prefill campaign — workable in parallel.
4. **MMB quant coverage:** our beta's 10 types vs its 23. Unlikely to move the uniform-IQ4_NL gap
   (both fire there), but a completeness/robustness gap for arbitrary GGUFs (Q4_0/Q4_1/Q5_0 and
   MXFP4/NVFP4 are common). Low-to-medium priority.
5. **Three cheap correctness items to port/audit** independent of perf: `40c0b9c38` (maskless only
   where qsa3 consumes it), `b0f31f587` (position-vs-cell block window), `14fff4f97` (sentinel
   handling). They prevent long-session corruption and are far cheaper than the perf items.
6. **MTP is not a missing optimisation in the other solution's favour — it is a different axis.** It has upstream
   `draft-mtp` with a **fixed** `n_max` and only upstream's per-step `p_min`/`n_min` early stop; there
   is **no** cross-round adaptive controller in its tree. Our `draft-mtp-adaptive` controller is a
   depth-policy advantage that composes with its per-step decode gains. **Measured 2026-09-21** (see
   [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and §12): our plain decode is
   ahead (+2–6 %), the fixed-depth MTP **speedup is at parity** (ours `n3` 1.90/1.78/2.05 vs its
   1.91/1.79/2.02 on code/prose/recall), and our adaptive wins recall (2.40x) but over-drafts code and
   prose on qwen4exp — a tuning item, not a structural one.  The one real MTP gap is
   **`nextn_shared_target_tensors` support**: our build cannot load the shared MTP sidecar its IQ4_NL
   model ships (every draft position past the first fails an M-RoPE `X < Y` check), so we fell back to
   the `Q4_K_M` sidecar for the comparison.

### D. Body §6/§9 caveat

The kernel-level comparisons (`mmb_dense` +809 ms, HC, `rms`, `qsa3_attn` +195 ms) were made against
`f5daaa3cf`. Before trusting them again, re-profile `b0f31f587`: its tree gained the `mmb_quant`
dispatcher and dropped env gating, and the QSA decode/indexer changes add kernels to the trace.

### E. Where the current beta was built and validated

Applied and built on **gfx1151** on 2026-09-21 for the beta re-validation window
(`beta/mmb-general/BETA-TESTING.md`). Build: `~/bin/build-llama-rocm-714` from the `mmb-beta` branch of
`~/llama.cpp` (r12 + 12 patches, tree `bca69f23dd…`). The gfx1151 numbers in the body were measured on
the pre-beta WIP; the beta re-run is what confirms they still hold.

### F. Priority sequence (maintainer, 2026-09-21)

**Recall speed + correctness → decode speed + correctness → MTP tuning + correctness.**  The MTP
qualification is therefore **done to "is our MTP behind its?" depth only** and parked; its result and
the one real MTP gap are in [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and
§12 below.  The headline: our plain decode is ahead, the fixed-depth MTP **speedup** is at parity, and
the only MTP gap is **`nextn_shared_target_tensors` support** (we cannot load the shared MTP head
the other solution's IQ4_NL model ships).  The body's prefill items 1–8 are the "recall" phase.

### G. Default-on policy + the recovered full-set numbers (2026-09-21)

**Policy (now in `AGENTS.md`):** a beneficial feature that has passed the gates is **ON by default**; the
env var only **disables** it.  Applied on fork branch `gap-closing`:

* `GGML_CUDA_MMB` default **ON** (`=0` disables), RDNA3_0 included (`GGML_CUDA_MMB_RDNA3=0` disables that arm);
* MMB `hc16` default **ON** (`GGML_CUDA_MMB_HC16=0` disables) — the two graph-optimizer HC16 marking
  sites in `ggml-cuda.cu` had a hardcoded env default of 0 and ignored the arch config, so HC16 never
  engaged without an explicit env (this is why the earlier full-set A/B looked flat);
* the HC `hc_combine_norm` matcher default **ON** (`LLAMA_FUSED_DSV4_HC_POST=1` forces the slower op).

gfx1151, qwen4exp IQ4_NL, `-b/-ub 2048`, prefill, **no env at all**:

| pp | default (full set) | `GGML_CUDA_MMB=0` | delta |
|---:|---:|---:|---:|
| 2048 | 1158.6 | — | |
| 8192 | 1136.2 | 835.1 | **+36 %** |
| 32768 | 1070.9 | 811.1 | +32 % |

Parity with the pre-port WIP at pp8192 (1136.4), i.e. the gfx1201/gfx1100 port did **not** strand the
win.  The earlier search for “1180–1220” was the `-ub 16384` regime (blocked by the context bug, item 3),
run/thermal variance, and ~1 % port cost — not a lost kernel.  Same-seed default run reproducible
(`extract-generated 01509ffcc688` twice); full `BETA-TESTING` gates still pending.

---

## 0. TL;DR (2026-09-20 snapshot)

1. **On the other solution's model the WIP is no longer 2x behind.** It is **within ~4% at matched `-ub 2048`** and **~9% behind at the other solution's best config (`-ub 16384`)**. The WIP took the uniform model from the delivery base's **734 t/s → 1149 t/s at pp8192/ub2048 (+57%)**; the other solution gets 1194. Our earlier "2x behind" figure was the mixed model measured against *its fast path not firing there*.

2. **The remaining gap is NOT MMB and NOT the QSA attention kernel.** The WIP already **beats** the other solution on `mmb_routed_glu` (−111 ms), the GDN recurrence (−250 ms), the F32 path (−239 ms vs its rocBLAS), the qsa3 sort (−86 ms) and the `mmb_cvt` bucket (−121 ms). The gap is concentrated in **three families the other solution fuses and we do not**:
   * the **hyper-connection (HC) prefill fusions** — `hc_combine_norm` + `hc_gate_mix` — worth **−19.5%** on its stack when disabled;
   * the **depthwise conv1d** (`gdn_conv_direct`/`ple_conv`) — worth **−10.5%**;
   * the **gated RMS-norm** (`norm-gated`/`rms_rows`) and the **indexer relu-sum** — worth −2.9% / −1.3%.

3. **Our delivery's `hc_combine_norm` (in `hyperconn.cu`) is present but never fires** — verified 0 calls with and without the WIP's HC16 gate. This is the single highest-value fix on the table: the other solution's equivalent fires 190× (554 ms) and it has an additional 408 ms `hc_gate_mix_kernel` we do not have at all. The missing gate-mix fusion also moves ~190 gate GEMMs *into* our `mmb_dense` (1266 launches vs its 956), which is most of the `mmb_dense` +809 ms delta.

4. **A separate, pre-existing delivery bug:** our tree (base r12 *and* the WIP) **cannot create a context at `n_batch == n_ubatch == n_ctx == 16384`** (`-b 16384 -ub 16384 -p 16384`), independently of MMB/HC16/QSA and of offload. the other solution's tree runs the same config at **1399 t/s**. This caps the useful ubatch and is why we have no pp16384/ub16384 point.

5. **Priority:** (a) make/fix the HC `combine_norm` + gate-mix fusion; (b) port the depthwise conv1d; (c) the `-ub 16384` context bug; (d) `norm-gated` + `idx-relu-sum`; (e) tune `qsa3_attn` and the `mmb_dense` tall tile. (a)+(b) are ~30% of end-to-end prefill on the other solution's numbers, which is exactly the "1300+" delta.

---

## 1. Why this investigation

`wip/mmb-general/` generalized the other solution's `mmb` weight GEMM, ported its `qsa3` attention and the bf16-producer machinery, and measured **+43–48%** over the delivery base on **our mixed UD-IQ4_XS** model. But the other solution's headline `1300+` numbers were on **its uniform-IQ4_NL** checkpoint. The open question was: *what still stands between us and those numbers?*

The previous gap analysis (README "Attribution", `archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-…`) said the gap was the QSA kernel and the weight GEMM; both were since ported. This session re-measured everything 1:1 on the actual the other solution GGUF, profiled both stacks, and priced the residual with the other solution's kill-switches.

---

## 2. Environment, builds, model

| | |
|---|---|
| WIP build | `~/llama-wip-mmb/build-rocm/bin/llama-bench`, tip **`90bf12997`** (38 commits / 5 thematic patches), base r12 applied tree `8a80535e…`, `LLAMA_QSA3_ENABLE=1` (compile-time) |
| delivery base | fresh worktree `/tmp/llama-r12-base` @ **`8568aaddb`** (block 15, r12 tree), built for this session (6 min with ccache) |
| the other solution's build | `~/pwilkin-llama-cpp/build-rocm/bin/llama-bench`, branch `strix-halo` @ **`f5daaa3cf`** |
| uniform model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93.16 GiB, 176.94 B params) |
| mixed model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (87.24 GiB) |
| the other solution env | `archive/work/wip-archive/iq4nl-prefill/launcher-env.txt` (its `install.sh` "optimized" set, verbatim) |

Rules observed: **page cache warmed** (`cat` all shards to `/dev/null`) before every run; **no parallel benches**; `-p … -n 0 -r 2`; `rocprofv3 --output-format csv` (the ROCm 7.14 rocpd/SQLite writer aborts without it); `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib`; `HIP_VISIBLE_DEVICES=0`.

The other solution runs are `-dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct`; WIP runs are `-ngl 99 -fa 1`. Both `-ctk f16 -ctv f16`. WIP all-on = `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1`.

> **Run-to-run variance is real: ~±2–3%** on this box (the other solution pp8192/ub16384 measured 1320.9 and 1338.8 in the same session; WIP all-on measured 1191.7 and 1220.5). Treat single-digit differences as noise; the profile and the ablations are the reliable signals.

---

## 3. Throughput — the 1:1 comparison

### 3.1 Uniform IQ4_NL (the other solution's checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | 733.6 | 737.1 |
| **base r12** | 16384 | — | 755.6 | **ctx failed** |
| **WIP all-on** | 2048 | 1181.5 | 1149.3 | 1129.2 |
| **WIP all-on** | 16384 | 1176.3 | 1220.5 | **ctx failed** |
| **the other solution full env** | 2048 | 1233.2 | 1194.2 | 1187.0 |
| **the other solution full env** | 16384 | 1233.0 | **1338.8** | **1399.3** |

* Base → WIP: **+56.7%** (pp8192/ub2048), **+61.5%** (pp8192/ub16384), **+53.2%** (pp16384/ub2048).
* WIP as % of the other solution: **96.2%** (ub2048 pp8192), **91.2%** (ub16384 pp8192), **95.1%** (ub2048 pp16384). The other solution's best config (ub16384) is the one we cannot fully run.

### 3.2 Mixed UD-IQ4_XS (our checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | ~707* | 710.5 |
| **base r12** | 16384 | — | 747.5 | — |
| **WIP all-on** | 2048 | 1137.9 | 1110.3 | 1082.2 |
| **WIP all-on** | 16384 | 1139.0 | 1192.3 | **ctx failed** |
| **the other solution full env** | 2048 | 901.4 | 899.3 | 1046.9 |
| **the other solution full env** | 16384 | 904.0 | 1070.1 | 1130.8 |

\* from `benchmarks/2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md` (same box).

On the mixed model the WIP is **+23–26% over the other solution** at pp2048–8192/ub2048, because the other solution's `mmb_supported_mmid`/`_glu` predicates reject anything but **IQ4_NL**, so the model's **IQ3_S gate/up experts (36% of bytes, ~2/3 of the MoE FLOPs)** and its Q8_0 dense tensors fall back to its MMQ path. Ours accelerates them. At pp16384/ub16384 the other solution catches up (its 1130.8 vs our ub2048 1082.2).

**Conclusion:** the mixed-model comparison is *not* apples-to-apples in the direction the original "1.79x behind" implied. Each stack wins on the model its fast path was built for. The honest 1:1 is the uniform model, where the residual gap is ~4–9%.

---

## 4. The `-ub 16384` context-creation failure (delivery bug)

### 4.1 Symptom

```
llama_bench: error: failed to create context with model '…/Qwen3.8-Flash-Next-UD-IQ4_XS-…gguf'
```
`llama-bench` calls `llama_init_from_model` and gets `nullptr`. No underlying error is printed. It reproduces on **both** checkpoints, on the **base r12 build** as well as the WIP, and the other solution's tree runs the same config fine (its 1399.3 on uniform).

### 4.2 Bisect matrix (uniform model, `-p 16384`)

| `n_batch` | `n_ubatch` | `n_ctx` | result |
|---:|---:|---:|---|
| 16384 | 16384 | 16384 | **ctx failed** |
| 16384 | 8192 | 16384 | 1192 t/s ✓ |
| 8192 | 16384 | 16384 | 1166 t/s ✓ (ubatch clamped to batch) |
| 16384 | 4096 | 16384 | 1162 t/s ✓ |
| 16384 | 2048 | 16384 | 1119 t/s ✓ |
| 16384 | 16384 | 8192 | 1221 t/s ✓ (`-p 8192`) |

**Trigger:** the triple **`n_batch == n_ubatch == n_ctx` = 16384** — i.e. a *full-batch, full-context prefill graph*.

### 4.3 Ruled out

* **Not OOM / not model VRAM:** still fails with `-ngl 50` (half the model unoffloaded). `rocm-smi` shows ~0.6 GiB VRAM used during the failing init (the model is mmap'd/GTT).
* **Not the WIP:** the **base r12** build fails identically → pre-existing delivery bug.
* **Not MMB/HC16:** fails with `GGML_CUDA_MMB=0 GGML_CUDA_MMB_HC16=0`.
* **Not QSA:** fails with `LLAMA_QSA_OFF=1`, `LLAMA_QSA_DENSE_SHORTCUT=1`, and `LLAMA_QSA_SPARSE_FA=0`.
* **`llama-cli` with the identical cparams works** (`n_ctx=16384 n_batch=16384 n_ubatch=16384`), because its init `graph_reserve` builds for `n_tokens = 64`; `llama-bench`'s init reserves the full-ubatch graph and dies there.

### 4.4 Hypothesis

The compute-graph reserve for a 16384-token × 16384-KV graph allocates one or more very large tensors (an `[n_kv, n_tokens]` mask/bias class tensor is 1 GiB as F32, 512 MiB as F16; the packed kq mask is `n_kv*n_tokens*2`), or hits an allocator/shape limit. The failure is silent, so the next step is a debug build that prints the reserve failure (or `gdb` on `llama_init_from_model`) — **not yet done**. Whatever it is, it is independent of the WIP and it costs us the `-ub 16384` regime where the other solution is 9% ahead.

---

## 5. Kernel profile diff — uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=1

Both runs profiled with `rocprofv3 --kernel-trace`. WIP grand kernel sum **13591 ms**; the other solution **11393 ms** (ratio 1.19). (Kernel-sum ratio > t/s ratio because the profiler captures the whole process; use the *family deltas*, not the absolute ratio.) Both traces confirmed the relevant fast paths were live: WIP `mmb_dense`/`mmb_routed_glu`/`mmb_routed` present, `qsa3_attn` present, `flash_attn_qsa` absent; the other solution `mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `hc_gate_mix_kernel`, `gdn_conv_direct_kernel` all present.

### 5.1 Family table (Δ = WIP − the other solution, ms)

| family | WIP ms | WIP % | the other solution ms | other % | **Δ(WIP−other)** |
|---|---:|---:|---:|---:|---:|
| `mmb_dense` | 4008.6 | 29.5 | 3199.4 | 28.1 | **+809.2** |
| `rms_norm` (all) | 1059.9 | 7.8 | 505.3 | 4.4 | **+554.5** |
| MoE concat+reduction | 744.8 | 5.5 | 278.6 | 2.5 | **+466.1** |
| `ssm_conv_long_token` (conv) | 303.0 | 2.2 | 7.0 | 0.1 | **+296.0** |
| `qsa3_attn` | 813.8 | 6.0 | 618.4 | 5.4 | **+195.4** |
| elementwise | 706.9 | 5.2 | 555.2 | 4.9 | +151.8 |
| copy | 243.4 | 1.8 | 129.0 | 1.1 | +114.4 |
| indexer | 156.0 | 1.2 | 63.6 | 0.6 | +92.4 |
| mmq/mmvq | 120.5 | 0.9 | 55.1 | 0.5 | +65.4 |
| `mmb_tiny_m` (F32) | 54.7 | 0.4 | 0.0 | 0.0 | +54.7 |
| `mmb_routed` | 979.0 | 7.2 | 944.3 | 8.3 | +34.7 |
| `mmb_f32split` | 295.9 | 2.2 | 273.0 | 2.4 | +23.0 |
| HC (dsv4/hc_*) | 1103.8 | 8.1 | 962.0 | 8.4 | +141.8 |
| `mmb_routed_glu` | 1863.6 | 13.7 | 1974.7 | 17.3 | **−111.0** |
| `mmb_other`/`mmb_cvt` | 1.1 | 0.0 | 122.0 | 1.1 | **−120.8** |
| rocBLAS | 0.0 | 0.0 | 239.4 | 2.1 | **−239.4** |
| GDN | 938.3 | 6.9 | 1188.2 | 10.4 | **−250.0** |

(Watch the bucketing: the other solution's `gdn_conv_direct_kernel` 250 ms landed in the GDN row, so the true conv comparison is our `ssm_conv_long_token_f32` 303 vs its `gdn_conv` 250 + `ple_conv` 7. And its GDN "1188" = `gated_delta_net_tiled` 936 + `gdn_conv_direct` 250; our pure recurrence is 938 — **parity**.)

### 5.2 The `mmb_dense` detail (raw instantiations)

| tile `WTYPE` | WIP ms / calls | the other solution ms / calls | Δ |
|---|---:|---:|---:|
| `<128,256,64,64,0>` | 1627.3 / 168 | 1735.7 / 168 | −108 (we win) |
| `<128,128,32,64,0>` | 1060.0 / 594 | 785.6 / 498 | +274 / **+96 calls** |
| `<384,64,96,32,0>` (tall) | 1004.6 / **380** | 589.6 / **190** | +415 / **+190 calls** |
| misc | 316.7 | 228.5 | +88 |
| **total** | **4008.6 / 1266** | **3199.4 / 956** | **+809 / +310 calls** |

Our dense MMB launches **1266** GEMMs vs its **956** (+310), and the tall `384x64` tile runs **twice** as often (380 vs 190). A large part of this is structural, not tile tuning: the other solution's **`hc_gate_mix_kernel`** (408 ms, 190 calls) fuses the HC gate GEMM + sigmoid + mix and *removes* ~190 dense GEMMs from its `mmb_dense`; we run those in `mmb_dense` and then do the mix separately in `dsv4_hc_pre/post`.

### 5.3 The RMS/HC detail

| | WIP | the other solution |
|---|---|---|
| `rms_norm_f32<1024,true>` | **622.2 ms / 196 calls** | — |
| `rms_norm_f32<256,true>` | 288.8 / 168 | 0.3 / 24 |
| `rms_norm_f32<256,false>` | 148.8 / 144 | 166.4 / 144 |
| `rms_norm_f32<1024,false>` | — | 35.9 / 10 |
| **`rms_rows_f32<true>`** | — | **220.0 / 72** |
| **`rms_rows_f32<false>`** | — | **82.7 / 72** |
| `dsv4_hc_post_f32<false>` | **739.0 / 188** | — |
| `dsv4_hc_pre_f32<true,true,true>` | 364.7 / 190 | — |
| **`hc_combine_norm_f32_b256`** | **0** | **554.3 / 190** |
| **`hc_gate_mix_kernel<4>`** | **0** | **407.7 / 190** |

`rms_rows_f32` is the other solution's fused **gated** RMS-norm (`LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS`); the 622 ms `rms_norm_f32<1024,true>` is our HC normalized stream. It folds the HC combine + norm into `hc_combine_norm_f32_b256`, and it has a whole `hc_gate_mix` kernel we have no analogue of.

### 5.4 The MoE detail

| | WIP | the other solution |
|---|---|---|
| `concat_transposed_src1_dim0` | **375.3 / 74** | 0 |
| `moe_weighted_reduction_f32_vec4` | **369.5 / 96** | — |
| `moe_weighted_reduction_bf16_v4` | — | **213.3 / 94** |

The other solution's MoE epilogue reads **bf16** expert outputs (its `LLAMA_MMB_DOWN16` / `store_f32=0` routed-down) and avoids the `concat_transposed` materialisation entirely. We still materialise the concat and reduce in F32. (The WIP added non-temporal hints to these two kernels in session 14, but did not remove the concat or move to bf16 inputs.)

### 5.5 QSA

Same kernel name, same 24 calls, **813.8 vs 618.4 ms** — our `qsa3_attn_kernel` is 32% slower at identical work. Our `qsa3_rows`/`merge` are **faster** (64.6 vs 151.0). So the port's *sort/merge* is a win and the *attention body* is a regression, or its `qsa.cu` has an arch/tile difference the port did not carry.

---

## 6. What the missing pieces are worth — the other solution's ablations on this box

Run on the uniform model, `-b/-ub 16384`, `-p 8192`, r=2, source its `launcher-env.txt` and zero one family at a time. This is the cleanest "what is missing" price list, because it is the *same tree, same model, same box*.

| arm | pp8192 t/s | Δ vs full | % |
|---|---:|---:|---:|
| **FULL (baseline)** | 1320.9 | — | — |
| **NO `HC_*` (all 6)** | **1063.6** | **−257.3** | **−19.5%** |
| **NO `GDN_CONV`+`PLE_CONV`** | **1182.6** | **−138.3** | **−10.5%** |
| NO `NORM_GATED`+`NORM_ROWS` | 1282.1 | −38.8 | −2.9% |
| NO `IDX_RELU_SUM` | 1303.2 | −17.7 | −1.3% |
| NO `MMB_DOWN16` | 1321.8 | +0.9 | +0.1% (nil) |

The `HC_*` set zeroed is `HC_CN_SHAPE`, `HC_GATEMIX`, `HC_MIX_FUSE`, `HC_BLK16`, `HC_RES16`, `HC_PACK_DI`. The `GDN_CONV`/`PLE_CONV` ablation removes the *whole* direct-conv path (kernel + the concat/tail/reorder chain it replaces), so its 10.5% is more than the 257 ms of the two kernels themselves.

For cross-reference, the archived `iq4nl-prefill` Phase-1 ranking (pp16384, older build) measured: NO HC −289 (−21%), NO MMB −470 (−33%), NO QSA −696, NO CONV −62, NO NORM −30, NO IDX −10. The HC/NORM/IDX magnitudes reproduce; the fresh CONV number is larger because the `-ub 16384` single-shot prefill exposes the concat chain more.

---

## 7. WIP's own gate contributions on this model (for contrast)

Uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=2:

| arm | pp8192 | Δ |
|---|---:|---:|
| WIP all-on | 1191.7 | — |
| WIP `HC16=0` | 1097.5 | HC16 bf16 producers **+8.6%** |
| WIP `MMB=0 HC16=0` | 856.2 | MMB **+28.2%** |
| WIP all-on `LLAMA_QSA_DENSE_SHORTCUT=1` | 1179.2 | always-QSA **+1.1%** |
| WIP all-on `LLAMA_QSA_OFF=1` | 1087.7 | QSA **+9.6%** |

So on the uniform model the WIP's MMB and bf16-producer work are doing exactly what they should. The gap is elsewhere.

---

## 8. Gap inventory (file + gate level, vs the other solution @ `f5daaa3cf`)

### 8.1 Ported / integrated (not the gap)

| the other solution work | status |
|---|---|
| `mmb.cu` dequant→bf16 WMMA weight GEMM | ported **and generalized** to 9 weight types (`wip/mmb-general/patches/0001`) |
| `qsa.cu` qsa3 rows/merge/attn | ported as `fattn-qsa3.cu` (`patches/0002`) |
| bf16-producer marking (`mark_bf16_only`, `out_xn_bf16`) | ported (`patches/0004`) |
| F32 split / tiny-M | ported/ours (`patches/0003`) |
| non-temporal hints | **ours** (its tree has zero) |
| fused indexer top-k | **ours** (`patches/0005`; its tree uses `top_k_nary_search_cuda`) |
| `dsv4_hc_pre`/`hc_mix_reduce` | in delivery block 14 / WIP |

### 8.2 Missing or inactive

| # | the other solution feature | its file / gate | our status | measured worth here |
|---|---|---|---|---|
| 1 | **HC gate-mix fusion** | `mmb.cu::hc_gate_mix_kernel`, `LLAMA_HC_GATEMIX` | **absent** | inside the −19.5% HC ablation |
| 2 | **HC combine+norm fusion (b256)** | `hc-cn.cu::hc_combine_norm_f32_b256` | delivery has `hyperconn.cu::hc_combine_norm_f32` (1024-thread) but it **never fires** (0 calls) | inside the −19.5% HC ablation |
| 3 | **depthwise conv1d, GDN + PLE** | `gdn-conv.cu`, `ple-conv.cu`; `LLAMA_GDN_CONV`/`LLAMA_PLE_CONV` | **absent** (we still build `concat`+transpose + `ssm_conv_long_token_f32`) | **−10.5%** |
| 4 | **gated RMS-norm** | `norm-gated.cu::rms_rows_f32`; `LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS` | **absent** | −2.9% |
| 5 | **indexer relu-sum** | `idx-relu-sum.cu`; `LLAMA_IDX_RELU_SUM` | **absent** | −1.3% |
| 6 | **MoE bf16 epilogue / concat elimination** | `moe_weighted_reduction_bf16_v4` + `LLAMA_MMB_DOWN16` | F32 epilogue + concat still materialised | ~+466 ms kernel time |
| 7 | **`hc_combine_norm` b256 variant** | `hc-cn.cu` | only the 1024-block form exists | — |
| 8 | QSA graph-side options | `qwen4exp.cpp`: `QSA_WHOLE_ATTN`, `_BLOCK_SELECTION`, `_COMPACT_METADATA`, `_DIRECT_INDICES`, `_NO_DENSE_MASK`, `_QUERY_STRIP`, `_SCORE_BOUNDS`, `_SCORE_WMMA`, `_TOKEN_EMBD` | not ported; **partly redundant** with delivery block-14/15 derived-visibility / keys-only / fused indexer score (never audited 1:1) | small |
| 9 | HC knobs `HC_CN_SHAPE`/`HC_MIX_FUSE`/`HC_BLK16`/`HC_RES16`/`HC_PACK_DI` | HC variants | partial (we have the bf16 `xn` stream but not the variants) | inside −19.5% |
| 10 | depthwise conv2d | `conv2d-dw.cu` | absent | 0 on these models |
| 11 | MTP-side QSA | `LLAMA_MTP_QSA`, `_MTP_QSA_MIN_T`, `LLAMA_MTP_EH_FLATTEN` | absent | **decode/MTP, not prefill** |
| 12 | host/loader | `LLAMA_PLE_PREFETCH`, `LLAMA_LOAD_LOCALS`, `--lazy-mode on-direct` | our analogue | load-time only |

---

## 9. Root-cause notes and hypotheses

### 9.1 `hc_combine_norm` does not fire — highest-value item

* The delivery's `ggml_cuda_op_hc_combine_norm` lives in `ggml/src/ggml-cuda/hyperconn.cu`; the graph-optimizer match is at `ggml-cuda.cu:5514` and `:5677` (two sites) and is a long `ok_a…ok_f` shape/type/alias predicate.
* `ggml_cuda_hc_combine_norm_supported` would accept this model (`n_embd=2560 ≤ HC_CN_MAX_EMB=3072`, `warp_size=32`, `hc≤16`), so the **supported** gate is not the blocker.
* Empirically it is **0 calls** on the uniform model with **both** `GGML_CUDA_MMB_HC16=1` and `=0`, and the fallback is `dsv4_hc_pre_f32<true,true,true>` + `rms_norm_f32<1024,true>` + `dsv4_hc_post_f32<false>`.
* Therefore the failure is in the *pattern match* (`ok_*`, `ggml_can_fuse_subgraph_ext`, alias `overlap`), i.e. our `qwen4exp` graph no longer presents the shape the delivery's matcher expects, **or** the matcher was only ever validated in the beta and has been dormant since. the other solution's equivalent fires 190× on the same model.
* **Action:** instrument the matcher (log which `ok_*` fails per layer), fix the pattern, or port the other solution's `hc-cn.cu` + `hc_gate_mix_kernel` directly. Then add `LLAMA_HC_GATEMIX`-equivalent: fused gate GEMM + sigmoid + mix, which also removes ~190 `mmb_dense` launches and the `rms_norm_f32<1024>` pass.

### 9.2 Depthwise conv1d

Our path: `build_conv_state`/concat + `ggml_ssm_conv` → `ssm_conv_long_token_f32` (303 ms), plus the surrounding `concat_cont`/`cpy_scalar`/transpose traffic. The other solution's `gdn_conv_direct_kernel` reads `state`+`x` directly and writes the conv output (+ optional silu), 250 ms, and `ple_conv_kernel` 7 ms, with **no concat tensor**. Porting `gdn-conv.cu`/`ple-conv.cu` (and their graph-optimizer match hooks, `*_match_at_concat`/`*_match_at_conv`/`*_match_at_tap` + `*_write_tail`/`*_direct`) is self-contained and worth ~10.5% end-to-end on the other solution's measure.

### 9.3 `mmb_dense` +809 ms / +310 launches — mostly structural, not tile tuning

The WIP already closed the tile-tuning question (session 6: every tile/BN/VDR knob is a wash or worse; the kernel is at 54% of bf16 peak). The delta is that the other solution **does fewer GEMMs**: `hc_gate_mix` absorbs ~190 gate GEMMs, and its tall tile runs 190× not 380×. Fixing item 1 should collapse most of this; the tall-tile 2x is worth a separate look (is the same A-panel dequantized/routed twice, or is our `mmb_tall` predicate applied to a tensor it handles with `<128,128>`?).

### 9.4 `qsa3_attn` +195 ms at identical launch counts

Our port is 32% slower on the body while our sort is faster. This is a kernel-shape/arch issue, not a graph issue. A/B the WIP `fattn-qsa3.cu` against the other solution's `qsa.cu` on this exact model (the WIP's own qsa3 measurements were on the mixed model / gfx1201 for some arms). Possible causes: the pack layout (`qsa_pack_keys/values` graph vs its `src[6]/src[7]`), the `G=4`/`umask` handling, or the `ncols2`/`Q->ne[1]` selection.

### 9.5 The `-ub 16384` context bug

Pre-existing delivery (base r12 fails, WIP fails, the other solution works), all WIP gates ruled out, not model VRAM. Blocks the pp16384/ub16384 point where the other solution is strongest. Needs a debug print/gdb on the reserve, then a fix in the base graph/allocator. It also means our ub16384 numbers above are only valid up to pp8192.

---

## 10. Recommended next steps (prioritized)

| # | action | expected | effort |
|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) **or** port `hc-cn.cu`; add the `hc_gate_mix` fusion | large — the `HC_*` ablation is **−19.5%** | 2–4 days; pattern debug may be hours |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + their graph-optimizer matches | **−10.5%** (+ fewer concat/copy kernels) | 2–3 days |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation (unlocks `-ub 16384` and pp16384/ub16384) | access to the other solution's best regime | 0.5–2 days |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9% / −1.3% | 1–2 days |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` (its `moe_weighted_reduction_bf16_v4`, `MMB_DOWN16`) | ~+466 ms kernel (~3–4%) | 1–2 days |
| 6 | Tune/port-align `qsa3_attn` body against `qsa.cu` | ~+195 ms (~1.5%) | 1–2 days |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 day |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 equivalents | small / likely redundant | 0.5 day |

Items 1+2 alone are ~30% of end-to-end prefill on the other solution's ablations — comfortably the difference between our 1221 and 1300+.

---

## 11. Caveats and data provenance

* **Variance.** Single runs on this box swing ±2–3%; the the other solution baseline measured 1320.9 and 1338.8 in the same session. All family ablations share one session so their *relative* deltas are meaningful, but one or two points are within noise.
* **Kernel-sum ratio ≠ t/s ratio.** The profiles capture the whole process (including warm-up), so use the family deltas, not `13591/11393 = 1.19`.
* **Profiler caveat (`rocprofiler-register`, ROCm issue #10196).** Under `rocprofv3`, an env-gated path can read as *unset* (measured to flip `GGML_CUDA_QSA3` before it was made compile-time). I verified the fast paths were live from the kernel names in each trace (`mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `gdn_conv_direct_kernel` all present). The WIP's MMB/HC16 are still env-gated and could in principle flip; the family table is consistent with the un-profiled throughput, so it did not.
* **The `mmb_dense`/`rms`/HC kernels are *not* the same code in the two trees**, so their per-kernel times are not a pure A/B; the ablation (§6) is the authoritative price of the missing behaviour.
* **`-ub 16384` is required to reproduce the other solution's 1339/1399**; our ub16384 numbers only exist up to pp8192 because of the context bug.
* **Not done:** gdb/debug of the context-creation failure; a 1:1 audit of the QSA graph-side flags; a from-scratch attempt to make `hc_combine_norm` fire; and any actual port work.

---

## Appendix A — raw throughput (t/s, `llama-bench -n 0 -r 2`)

```
Uniform IQ4_NL, base r12:
  ub2048  pp8192 733.62 ± ?      pp16384 737.06 ± 5.73
  ub16384 pp8192 755.59 ± 3.98   pp16384 FAIL
Uniform IQ4_NL, WIP all-on:
  ub2048  pp2048 1181.46 ± 3.31  pp8192 1149.32 ± 4.25  pp16384 1129.16 ± 1.19
  ub16384 pp2048 1176.28 ± 4.01  pp8192 1220.52 ± 1.53  pp16384 FAIL
Uniform IQ4_NL, the other solution full env:
  ub2048  pp2048 1233.16 ± 39.79 pp8192 1194.24 ± 6.02  pp16384 1187.04 ± 0.49
  ub16384 pp2048 1233.03 ± 32.96 pp8192 1338.77 ± 37.03 pp16384 1399.34 ± 0.00

Mixed UD-IQ4_XS, base r12:
  ub16384 pp8192 747.48 ± 9.26   ; ub2048 pp16384 710.48 ± 7.75
Mixed UD-IQ4_XS, WIP all-on:
  ub2048  pp2048 1137.89 ± 1.59  pp8192 1110.32 ± 1.99  pp16384 1082.21 ± 0.35
  ub16384 pp2048 1139.03 ± 5.17  pp8192 1192.31 ± 1.11  pp16384 FAIL
Mixed UD-IQ4_XS, the other solution full env:
  ub2048  pp2048 901.36 ± 96.74  pp8192 899.30 ± 61.23  pp16384 1046.86 ± 1.18
  ub16384 pp2048 904.04 ± 91.36  pp8192 1070.10 ± 29.01  pp16384 1130.80 ± 5.31
```

## Appendix B — the other solution family ablations (uniform, `-b/-ub 16384`, pp8192, r=2)

```
FULL                     1320.88 ± 37.97
NO NORM_GATED+ROWS       1282.10 ± 31.20   -38.78  -2.94%
NO GDN_CONV+PLE_CONV     1182.58 ± 31.49   -138.30 -10.47%
NO IDX_RELU_SUM          1303.17 ± 41.83   -17.71  -1.34%
NO MMB_DOWN16            1321.76 ± 33.37   +0.88   +0.07%
NO HC_* (all 6)          1063.62 ± 30.46   -257.26 -19.48%
```

## Appendix C — WIP gate contributions (uniform, `-b/-ub 16384`, pp8192, r=2)

```
WIP all-on                1191.66 ± 4.29
WIP HC16=0                1097.46 ± 3.47
WIP MMB=0 HC16=0           856.21 ± 11.06
WIP all-on DENSE_SHORTCUT=1 1179.19 ± 6.23
WIP all-on QSA_OFF=1       1087.71 ± 25.75
```

## Appendix D — exact commands

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MM=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
W=/home/stew675/llama-wip-mmb/build-rocm/bin/llama-bench
BASE=/tmp/llama-r12-base/build-rocm/bin/llama-bench
OTHER=/home/stew675/pwilkin-llama-cpp/build-rocm/bin/llama-bench
ENV=/home/stew675/llama-cpp-rdna-boosts/archive/work/wip-archive/iq4nl-prefill/launcher-env.txt

# warm the page cache
for f in /llm/models/Qwen3.8/Flash-Next/IQ4_NL/*-0000*.gguf; do dd if=$f of=/dev/null bs=4M; done

# WIP all-on
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2

# the other solution full env
( set -a; . $ENV; set +a; \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2 )

# the other solution ablation (e.g. HC)
( set -a; . $ENV; set +a; \
  LLAMA_HC_CN_SHAPE=0 LLAMA_HC_GATEMIX=0 LLAMA_HC_MIX_FUSE=0 LLAMA_HC_BLK16=0 LLAMA_HC_RES16=0 LLAMA_HC_PACK_DI=0 \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 8192 -n 0 -r 2 )

# profile (csv; the rocpd writer aborts on this ROCm)
rm -rf /tmp/prof && mkdir -p /tmp/prof
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 /opt/rocm-7.14-gfx1151/bin/rocprofv3 \
  --kernel-trace -f csv --output-format csv -d /tmp/prof -o k -- \
  $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 -b 16384 -ub 16384 -p 8192 -n 0 -r 1

# delivery base (built this session)
cd ~/llama.cpp && git worktree add --detach /tmp/llama-r12-base 8568aaddb
# then configure/build as in wip/mmb-general/HANDOVER.md §3
```

---

## 12. MTP qualification (2026-09-21): adaptive (ours) vs fixed (the other solution's)

This is the MTP half of the gap analysis, added because the other solution's newer commits are decode/MTP-heavy
and it is easy to read its MTP t/s as a gap. It is **not** the same axis as our advantage, and the
qualification below is what the 2026-09-21 plan asks for before either side is claimed.

### 12.0 Result (measured 2026-09-21 — see [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md))

Two findings, and one correction to the premise:

* **Our plain decode is ahead** of its on qwen4exp IQ4_NL (code 32.4 vs 31.1, prose 31.8 vs 29.9,
  recall 32.4 vs 31.9 t/s).  Absolute MTP t/s therefore flatters its stack; the fair metric is the
  **speedup over each tree's own plain decode**.
* **At fixed depth the MTP speedup is at parity** — ours `n3` **1.90x / 1.78x / 2.05x** vs its fixed
  **1.91x / 1.79x / 2.02x** (code / prose / recall).  It did **not** adopt our controller
  (`common/speculative-adaptive.h` is absent from its tree) and it is not ahead.
* **Our adaptive controller is mixed on qwen4exp** — the opposite of the 27B dense record.  It wins
  **recall** (2.34–2.40x) but over-drafts code and prose at `n_max 9..12` (code `adaptive 12`
  per-position acceptance falls 0.94 → 0.45 → 0.22 → 0.07); `adaptive 7` already beats `n3` on code
  (63.1 vs 61.4 t/s).  So the qwen4exp adaptive **ceiling is a tuning item**, not a structural gap.
* **The one real MTP gap is correctness/compat, not speed: `nextn_shared_target_tensors`.**  The sidecar
  the other solution's IQ4_NL model ships is a *shared* MTP head; our build fails every draft position past the
  first on an M-RoPE `X < Y` check, so the head cannot be used at all.  The comparison above used the
  non-shared `Q4_K_M` sidecar, which both trees run clean.

### 12.1 Structural standing

| | the other solution (`b0f31f587`) | ours (r12 + `beta/mmb-general`) |
|---|---|---|
| spec type | upstream **`draft-mtp` only** | `draft-mtp` **and** `draft-mtp-adaptive` |
| depth | **fixed** `--spec-draft-n-max` (default 3), capped at `n_mtp_layers` when chaining heads | adaptive controller picks the depth each round; `--spec-draft-n-start`, `n_min_adaptive`, clamp at 15 |
| cross-round feedback | none — only upstream's **within-round** `p_min`/`n_min` early stop | credit-bucket `common_speculative_adaptive` (delta = `n_accepted - depth`; full accept credits `max(1, n_accepted-1)`; surplus/deficit carried; `drop_pressure = max(60, 10*depth)`, `climb_budget = 20 + 6*(depth-1)`, cold start `cap-3`) |
| per-step cost | **new** sparse selected-cell decode (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh`) + incremental indexer key state (`d67d58836`): serial d40000 **25.85 → 28.82 t/s**, MTP 40680 **31.17 → 35.57** / **32.69 → 39.10** | our own QSA-sparse-FA decode + derived-block-vector cache; no dedicated selected-cell decode kernel for this model |

**The two optimise different things and compose.** Its `d67d58836` lowers the cost of each verify/draft
step; our block-01 controller decides *how deep* to draft. Median accepted length is the quantity the
controller moves and its kernels do not.

Delivery evidence for the controller (all at `-n 3000`, `benchmarks/mtp-adaptive-methodology.md` rule 0):
+13 % prose, +28 % code, +61 % recall vs fixed `n3` (`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`),
and +72 % recall for `draft-mtp-adaptive` + `ngram-mod` (`benchmarks/2026-09-17-mtp-ngram-combo.md`).
At `-n 256` the same controller *lost* to fixed `n3` (-5 % code) — the length rule matters here as much
as anywhere.

### 12.2 Hypothesis and falsification

**Hypothesis:** on the same model and workload our adaptive depth beats our fixed `n3` (and its fixed
`n3`) by a margin larger than its per-step decode gains, because the depth policy is the term the
per-step kernels do not touch.

**Falsifiers:**
- if `draft-mtp-adaptive` ≤ `draft-mtp --spec-draft-n-max 3` on the four axes at `-n 3000` on
  qwen4exp, the controller does **not** transfer to this model (a real finding — it would need a
  model-specific investigation);
- if its *absolute* MTP t/s exceeds ours by more than its per-step kernel advantage explains
  (measured as our fixed-`n3` vs its reported fixed-`draft-mtp`), our depth policy is not the whole
  story and the decode kernels are the gap after all.

### 12.3 Protocol — single-build A/B (the portable claim)

On our beta build (`~/llama.cpp/build-rocm`, r12 + 12 patches), **the other solution's uniform IQ4_NL model**,
gfx1151, `-ctk f16 -ctv f16`, seed 42 / temp 0, `-n 3000` (reasoning pinned: `on` for R, `off` for
P/C/K), per `benchmarks/mtp-adaptive-methodology.md`. Four arms per axis:

```sh
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
BIN=~/llama.cpp/build-rocm/bin/llama-cli
for axis in P C R K; do for arm in none fixed3 adaptive adaptive12; do
  case $axis in R) REA=on;; *) REA=off;; esac
  case $arm in none) SPEC="--spec-type none";;
                fixed3) SPEC="--spec-type draft-mtp --spec-draft-n-max 3";;
                adaptive) SPEC="--spec-type draft-mtp-adaptive";;
                adaptive12) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 12";; esac
  # <prompt for $axis>, --log-verbosity 4 -> capture Generation t/s + acceptance + acc per pos
  $BIN -m "$MU" -ngl 99 -fa 1 --reasoning $REA $SPEC \
    --seed 42 --temp 0 --predict 3000 --single-turn --no-display-prompt \
    -p "$(cat prompts/<axis-prompt>.txt)" 2>&1 | tee /tmp/mtpq-${axis}-${arm}.log
done; done
```

Also run the **40680-token prompt** (its long case) for A1/A2/A3 only; record Generation t/s and mean
accepted length. `-n` is recorded with every number (rule 0).

### 12.4 Cross-build comparison — separate the axes

The other solution's 35.83 / 39.01 t/s include its per-step kernels, so our absolute numbers are expected to be
lower. Decompose, do not compare totals:

* **depth-policy delta** = `adaptive` − `fixed3` on *our* build (its kernels absent from both arms);
* **per-step-cost delta** = our `fixed3` vs its reported fixed-`draft-mtp` (same model, same depth) —
  this prices the sparse-decode + incremental-indexer gap in milliseconds per step;
* only the residual neither term explains is a genuine MTP gap.

### 12.5 Conclusion and the plan it implies

Measured, not predicted: our fixed-depth MTP is at parity with its, our plain decode is ahead, and our
adaptive controller is a clear win on recall and a tuning problem on code/prose for this model.  The
plan is therefore:

* **do not** treat its MTP as a speed gap;
* fold the other solution's per-step decode path in as **item 9** (sparse selected-cell decode + incremental
  indexer) — that is the term its absolute numbers get for free;
* add **`nextn_shared_target_tensors` support** as a correctness/compat item (it gates its own model's
  MTP head);
* park the qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) until the MTP phase, per the maintainer's
  priority sequence.

### 12.6 What NOT to conclude

* **Do not** read its 39.10 t/s as "our adaptive MTP is 39 t/s behind" — it is measuring a fixed-depth
  stack plus its decode kernels on a different tree.
* **Do not** compare absolute MTP t/s without each build's own plain decode next to it.
* **Do not** compare at `-n 256`: our controller's warm-up transient inverts the ranking there.
* **Do not** use `none == draft-mtp` byte purity above `n_max 7` as the MTP gate; use acceptance and
  MTP-vs-plain throughput (rule 4).

---

## 13. Revised action plan — phased (2026-09-21)

**Maintainer's priority sequence (2026-09-21): recall speed + correctness → decode speed + correctness
→ MTP tuning + correctness.**  The 2026-09-20 §10 order was: (1) HC combine_norm/gate-mix, (2) depthwise
conv1d, (3) `-ub 16384` context bug, (4) norm-gated + idx-relu-sum, (5) MoE bf16 epilogue, (6) qsa3_attn
body, (7) tall tile, (8) QSA graph flags; items 1–9 survive, regrouped below.

### Phase 1 — recall (long-context prefill/attention) speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) and **wire the existing `hc_gate_mix_kernel`** | large — `HC_*` ablation **−19.5 %** | 2–4 d | **DONE 2026-09-21**: matcher revived (+1.5 % prefill) and `hc_gate_mix` wired + default-on on gfx1151 (+1.2–1.5 % at pp8192/32768, width-pure, text-identical) — [`2026-09-21-hc-combine-norm.md`](2026-09-21-hc-combine-norm.md), `patches/0003`. Follow-up: IQ4_NL-only kernel (mixed UD model unchanged) |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + matches (now incl. **F32 PLE**) | **−10.5 %** | 2–3 d | **DONE 2026-09-21 (session 3)**: ported default-on, bit-identical, +3.0/+3.2 % qwen4exp IQ4_NL and +6.5/+7.1 % 35B-A3B at `-ub 8192` — [`2026-09-21-gdn-ple-conv-fusions.md`](2026-09-21-gdn-ple-conv-fusions.md), `patches/0004`.  Two adaptations (3-D `grouped_norm` root + the shared-builder snapshot cpy) |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation | unlocks `-ub 16384` | 0.5–2 d | pre-existing delivery bug |
| 3.5 | **Port the three correctness fixes** (`40c0b9c38`, `b0f31f587`, `14fff4f97`) | prevents long-session corruption | 0.5–1 d | **first one DONE 2026-09-22**: `b0f31f587` (QSA block window by highest stored position), `patches/0005` — [`2026-09-22-qsa-block-window-fix.md`](2026-09-22-qsa-block-window-fix.md).  The other two target the reference's `tail_idxs`/`compact`/`maskless` design and remain an **audit** against our derived-visibility QSA |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9 % / −1.3 % | 1–2 d | |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` | ~+466 ms kernel (~3–4 %) | 1–2 d | beta has `MMB_DOWN16` gated off; wire it + the bf16 reduction |
| 6 | Tune/port-align `qsa3_attn` body vs `qsa.cu` | ~+195 ms (~1.5 %) | 1–2 d | re-profile `b0f31f587` first |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 d | |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 | small / likely redundant | 0.5 d | |

Items 1+2 remain ~30 % of end-to-end prefill on the other solution's ablations.

### Phase 2 — decode speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 9 | **Port sparse QSA decode + incremental indexer state (`d67d58836`)** | **+11–20 % MTP/decode** | 2–4 d | this is the per-step term its absolute numbers get for free; audit vs our `GGML_CUDA_QSA_INDEXER_CACHE` (default on) first |
| 10 | MMB quant coverage: Q4_0/Q4_1/Q5_0/Q2_K/IQ1/IQ2/MXFP4/NVFP4 | completeness | 1–2 d | low priority for the delivery's models |

Our **plain decode is already ahead** of its (+2–6 % on qwen4exp, §12), so item 9 is a *hold/repay*
item, not a catch-up.

### Phase 3 — MTP tuning + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 12 | **`nextn_shared_target_tensors` support** | gates the other solution's IQ4_NL MTP head | 1–2 d | our build fails every draft position past the first (M-RoPE `X < Y`); see §12.0 |
| 11 | qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) + a long-prompt run | recovers the recall win without over-drafting code/prose | 0.5–1 d | `adaptive 7` already beats `n3` on code; the 27B result does not transfer at `n_max 12` |

Item 11/12 are parked until Phase 1–2 land, per the maintainer's sequence.

