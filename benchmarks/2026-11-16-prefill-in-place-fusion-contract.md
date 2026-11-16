# The prefill MoE fusions never met an in-place table — and what actually moves the text (TODO #327, #335)

**Question.** `TODO` #327 was the maintainer's chosen answer to `GREEDY-PURITY.md` §89: the three prefill MoE
fusions read the expert table **directly** instead of through the `ggml_cuda_mul_mat_id` redirect, so a table
served in place (r258/r266, `MOE_EXPERT_PREFILL_IN_PLACE`) made them stand down, and residency therefore
selected the kernel set. The chosen fix was (2) of three: *teach the fusions the in-place contract*. This
record is the implementation's measurement, and it answers three questions:

1. Do the prefill MoE fusions fire for a cache-managed (host-resident) table at all?
2. Does teaching them the contract change any text?
3. With the fusions out of the picture, what *does* make greedy text a function of residency?

**Answers in one line each.**

1. **No.** A routed `MUL_MAT_ID` whose weight lives in a host buffer is its own **single-node split**, so
   `gate`, `up` and `GLU` never share a graph and every prefill MoE fusion declines for *shape*, not for the
   cache. The above-band stand-down this item was written to remove was **dead code**.
2. **No** — measured byte-identical on the one configuration where the prefill MoE fusions do fire (a fully
   device-resident table, where the cache has no tables at all).
3. The in-place **prefill read itself**, in two shapes: a coherent near-tie difference (r284's differential,
   still unexplained at the kernel level) and, under VRAM pressure, an outright **degenerate output** whose
   carrier is a **captured graph replaying an arena pointer whose table the slab evicted mid-run**. That is
   the new item (#335); it is not the fusions.

## 1. The fusions cannot fire for a cache-managed table (the census)

`ggml_cuda_try_fuse` runs per **split** graph. With `-ncmoe`, a routed MoE op's weight is a host buffer, so the
scheduler puts that op in a split of its own. Census of the split graph sizes on 35B-A3B Q3_K_M, `-ncmoe 48`,
`-ub 2048`, 2 devices `-sm layer` (`GGML_CUDA_FUSE_TRACE=1`):

| split graph | count |
|---|---|
| 1 node (`ffn_moe_gate-N` alone) | 117 |
| 2 nodes | 117 |
| 26 / 72 / 74 / 99 nodes (attention + the device-resident layers) | 3 / 27 / 3 / 84 |

and the new `-lv 4` prefill-fusion census (this change) reads **all four kinds at zero** on every
cache-managed arm:

| arm | `get_ok` (in-place reads) | `mmb_glu` | `mmq_triple` | `pair` | `swiglu_down` |
|---|---|---|---|---|---|
| 35B-A3B Q8_0, 2 dev `-sm layer`, `-ncmoe 48` | 951 | 0/0 | 0/0 | 0/0 | 0/0 |
| 35B-A3B Q8_0, 2 dev `-sm tensor`, `-ncmoe 48` | 1902 | 0/0 | 0/0 | 0/0 | 0/0 |
| 35B-A3B Q4_K_M, 2 dev, `-ncmoe 48` | 1887 | 0/0 | 0/0 | 0/0 | 0/0 |
| 35B-A3B Q3_K_M, `-ub 2048`, `-ncmoe 48` | 951 | 0/0 | 0/0 | 0/0 | 0/0 |
| Flash-Next GSQ-IQ3_XXS + MTP sidecar | 7510 | 0/0 | 0/0 | 0/0 | 0/0 |
| **35B-A3B Q3_K_M, fully device-resident (no `-ncmoe`)** | 0 | 0/0 | 0/0 | **0/117** | 0/0 |

(`in_place/staged`; the census line is `report_locked: prefill MoE fusions: …`.)

So the *only* configuration where a prefill MoE fusion fires is the one where no expert table is
cache-managed, and the guard's above-band `moe_cache_table_serves()` stand-down could never have been
reached. `GGML_CUDA_MMB=0` moving r284's prior-absent arm is therefore **MMB's dense GEMM** (the router and
the shared expert, which do fire and are resident on the device), not the routed GLU — the r284 record's
attribution of the differential to the fusion stand-down was wrong.

## 2. The in-place contract, implemented and inert

`moe_cache_fusion_serve()` (new, `moe-expert-cache.{h,cu}`) answers the same per-lane question
`moe_cache_get_table` answers and returns one of three placements:

* `MOE_CACHE_SERVE_NONE` — no lane served: keep the graph's staged tensors (today: every routed arm above).
* `MOE_CACHE_SERVE_IN_PLACE` — every lane served: launch on the arena copies (`src0_c`, `gate_c`, and the
  remapped ids when the tier publishes a prefill remap).
* `MOE_CACHE_SERVE_MIXED` — the lanes disagree: **stand the fusion down**. One lane's staged copy was never
  filled (the scheduler's in-place decision is per split), so the per-op chain, which resolves each table on
  its own, is the only correct consumer of that state.

Taught sites: the MMB routed gate+up+GLU triple, the MMQ gate+up+GLU triple (both the RDNA arm and the
NVIDIA dense-arm one), the MMQ shared-quantize pair (`ggml_cuda_mul_mat_q_pair` gained optional `src0`/`ids`
overrides), and the MMQ `swiglu→down` fold. The guard's above-band stand-down is gone; its below-band
per-table rule is unchanged. `MOE_EXPERT_PREFILL_IN_PLACE` stays a pure kill switch and no environment
variable is added.

**Purity evidence:** the staged (routed-pair) arm is byte-identical between the r296 build and this one —
Qwen3.6-35B-A3B Q3_K_M fully device-resident, 1 GPU, `-ub 2048 -c 8192`, `prompts/prose-rdna-boosts.txt`,
`-n 16 --seed 42 --temp 0`: `73 chars sha=86b7f9b5aa80` **both** builds, with `pair=0/117` in the census. The
coherence gate (Qwen3.5-4B Q8_0, 20 tokens, `sha=1c5d32ac537d`) is identical on 1 device `-sm layer` and 3
devices `-sm tensor`.

## 3. What does move the text: the in-place **prefill read**, two ways

Both are reachable only with `MOE_EXPERT_PREFILL_IN_PLACE=1` (the default). `MOE_EXPERT_PREFILL_IN_PLACE=0`
stages the prefill and collapses each pair to one text, exactly as r284 recorded — that part of r284 stands;
only its *cause* was misattributed.

### 3a. The coherent case (r284's differential, still unexplained)

Flash-Next GSQ-RCO-IQ3_XXS + `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `HIP_VISIBLE_DEVICES=0,1`, `-sm layer`,
`-ngl 99 -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off -c 32768 -ctk q8_0 -ctv q8_0`,
`-f prompts/kld-corpus-35b-in-domain.txt -n 64 --seed 42 --temp 0`, `ROCBLAS_USE_HIPBLASLT=0`,
`MOE_EXPERT_HEATMAP=load` (frozen; the prior file moved aside for the "absent" arm):

| prior | `MOE_EXPERT_PREFILL_IN_PLACE` | text |
|---|---|---|
| absent | 1 | `f40ae789f219` (837 chars) |
| absent | 0 | `7a3474cbe5f1` (835 chars) |
| present | 1 | `7a3474cbe5f1` |
| present | 0 | `7a3474cbe5f1` |

Census of the differing arm: 131 of 144 tables **fully resident** (`512/512 slots`), `takeover=359`,
`get_ok=7510`, `heat coverage 95.68 %`; with the prior present none of them is identity, so the whole prefill
stages. No MoE fusion fires in either arm.

### 3b. The degenerate case (new, `TODO` #335)

Same model and drafter, `-c 16384`, `prompts/prose-rdna-boosts.txt`, 64 tokens — i.e. the same shape with
VRAM clamped to **2 MiB free**:

| arm | text | prompt t/s |
|---|---|---|
| `MOE_EXPERT_PREFILL_IN_PLACE=1` | 82 chars `sha=a753000294c9` — `…\n3333333…` (degenerate) | 853 |
| `MOE_EXPERT_PREFILL_IN_PLACE=0` | 328 chars `sha=7c5a512a8447` | 587 |
| in place + `GGML_CUDA_DISABLE_GRAPHS=1` | 297 chars `sha=04ca2590aebf` (**coherent**) | — |
| in place + `GGML_CUDA_SLAB=0` | 328 chars `sha=7c5a512a8447` (coherent) | 948 |
| in place + `MOE_EXPERT_CACHE_MIB=20000` | 297 chars `sha=04ca2590aebf` | 604 |
| in place + `MOE_EXPERT_CACHE_MIB=0` (no cache) | 296 chars `sha=91ef8940d66e` | — |
| in place, no drafter (residency not forced partial) | both arms `7c5a512a8447` | 1001 / 840 |
| `-c 8192` (no eviction: `MOE_EXPERT_HEATMAP=load`, 6.7 GiB arena free) | both arms `7c5a512a8447` | 672 / 603 |

The carrier is the **slab's mid-run arena eviction**, and the mechanism is a **captured graph replaying an
arena pointer** rather than a wrong expert read:

* `moe_cache_evict_slab_range: evicted 3946.0 MiB … 12 table(s) stood down until the compute-buffer drop
  re-arms them`, followed 34 log lines later by `moe_cache_rearm: re-armed 12 …` — both **before** the
  `[ Prompt: … ]` footer, i.e. inside the prompt of the corrupted arm.
* The arena **itself is correct**. `MOE_EXPERT_CACHE_VALIDATE=3` sampled 157 in-place dispatches (36 of them
  with `tables=144 resident=132 down=12`): **0** `MISMATCH`, **0** `NONFINITE-HEAD`, **0** `NONUNIFORM-LAYER`.
  So the bytes the redirect hands to the kernel are the model's, at the moment of dispatch.
* Filling the copy the take-over skipped does **not** fix it (an experimental arm that kept the scheduler's
  copy and still read the arena is equally degenerate), so it is not the missing `input_cpy`.
* Removing the captures (`GGML_CUDA_DISABLE_GRAPHS=1`) or the slab (`GGML_CUDA_SLAB=0`) or the pressure
  (`MOE_EXPERT_CACHE_MIB=20000`, `-c 8192`) restores a coherent text. That is the signature of
  `ggml_cuda_graph_mem_freed()`'s generation check not covering an eviction that lands **inside** a replayed
  graph: the replay's launches were already issued with the old arena address while `moe_cache_evict_slab_range`
  handed the bytes back to the slab's free list (the eviction itself only bumps the generation, which
  invalidates the *next* capture/replay).

`TODO` #335 carries the reproducer, the three workarounds and the open question.

## What this means for the delivery

* Blocks **06** (MMB), **08** (the old-08 fusions) and **09/12** (the cache/scheduler contracts) change as
  described; blocks 14/15 are untouched. No new environment variable; `MOE_EXPERT_PREFILL_IN_PLACE` stays a
  kill switch and the per-fusion bisect arms (`GGML_CUDA_MMB_GLU`, `GGML_PAIR_ROUTED_OFF`,
  `GGML_CUDA_DISABLE_FUSION`) are unchanged.
* The new `-lv 4` census (`report_locked: prefill MoE fusions: <kind>=<in_place>/<staged> … mixed_lanes=N`) is
  the instrument that makes the first answer checkable in any future run: it is what a gate reads to prove
  residency did not select the kernel set.
* `scripts/gate-prefill-logits.sh`'s `--ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` is still the numeric instrument for
  §3a, and it cannot see either case without a drafter or a pressure arm — `llama-perplexity` cannot load one.
  That is the measurement #327's "owed item 2" asked for and it is still owed, now with a named mechanism.
