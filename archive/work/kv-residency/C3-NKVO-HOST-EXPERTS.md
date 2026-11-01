# C3 — `-nkvo` + host experts (TODO #129): a routed op whose ids are computed by its own split

Session A19, 2026-11-01.  Developed on branch `wip-kv-residency-129` in `~/llama.cpp` (commit
`e6ec2f9c2`) cut from the r196 tree, then **delivered as `r197`**: the two hunks are an amendment to **block
09** and the sibling window guard below is an amendment to **block 15** (`WORKLOG.md` 261101.002,
`release.json` tip `ba7c2b7b8`).  `TODO` #129 is closed.  What follows is the diagnosis as written before the
release, plus §7 for the delivery.

---

## 1. What it is

`-ncmoe <n> -nkvo` (host-resident experts + the KV cache in host RAM) aborts during the first
`llama-cli` graph:

```
ggml-backend.cpp:2920: GGML_ASSERT(id >= 0 && id < n_expert) failed
```

in `ggml_backend_sched_copy_input`, the host-expert path's routing readback.  Reproduced on the r196
tip with a small MoE (a 24-layer `bailingmoe3`, Ling-3.0-tiny) in seconds — no need for qwen4exp, which
is where it was first found.

## 2. Root cause — the split pass, not the expert cache

The abort is a **routing-readiness** bug, not a cache bug:

* `ggml_backend_sched_copy_input` prunes a host expert table's upload to the experts the routing
  selects: it reads the ids tensor (`node->src[2]`) from the split backend, and that read happens when
  the split's *inputs* are copied — i.e. **before the split computes**.
* The pass-5 split rule (`ggml_backend_sched_split_graph`) starts a new split for a host weight
  "so the memory of the previously offloaded weights can be reused" — but only
  `if (node_backend_id == cur_backend_id && split->n_inputs > 0)`.  That guard is upstream's and
  unchanged.
* **This is a delivery-introduced reachability, not an upstream bug** (checked against
  `origin/master` = `6050b467c`): upstream's scheduler does no expert pruning at all, and its prune
  callback (`llama_context::sched_copy_experts`) fires only when the **first node of the split** is the
  `MUL_MAT_ID` that reads the input (`const ggml_tensor * node = ggml_graph_node(graph, 0);
  if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) return false;`).  In this shape the split's
  first node is `attn_output-1 (cont)`, so upstream declines to prune and copies the whole table — no
  ids readback, no abort.  Block 09 moved that pruning into the scheduler and generalised it to *any*
  `MUL_MAT_ID` consumer of the input inside the split (needed so a merged gate/up/down split hooks all
  three tables), and that is what reads ids this split has not computed yet.  So: **our regression,
  exposed by `-nkvo`** — part A's stand-down is upstream's behaviour for that shape, and part B goes
  further and gives the split a boundary so the pruning still works.
* With `-nkvo` the attention runs on the host, so the GPU split starts at the attention output.  That
  split has **no input to reuse yet** (`n_inputs == 0`), so the host-weight rule is skipped and the split
  swallows the router chain *and* every routed op of the layer.  The ids are therefore produced by the
  same split whose inputs are being copied: the readback returns the previous split's leftover memory.

Instrumented proof (`NKVODBG`, removed before the commit): the failing arm's split graph is
`[0] attn_output-1 (cont) … [31] ffn_moe_argsort-1(ARGSORT)*IDSRT … [3x] ffn_moe_gate-1(MUL_MAT_ID)*NODE`
with `n_inputs = 1` at the copy, while the default arm's split starts **at** `ffn_moe_gate-1` and the
router is in the previous split.  The int32 readback values were float bit patterns
(`-1153925685, 1020136602, …`).

**The abort is the lucky case.**  A stale word that happens to land in `[0, n_expert)` passes the
assert and silently selects the wrong experts — so this needs a guard even after the split is fixed.

Not the report's suspects: independent of `LLAMA_KV_REGION_POOL`, of `--spec-type`, and of the expert
cache (`MOE_EXPERT_CACHE_MIB` unset in the repro).  A dense model with `-nkvo` has no host weight and is
unaffected.

## 3. The fix (two parts, one block)

| part | where | what |
|---|---|---|
| **A — correctness guard** | `ggml_backend_sched_copy_input` | if the ids root (the `view_src` chain of `node->src[2]`) is a node of **this** split, stand every routing-dependent shortcut down (pruned upload, `moe_cache_gather`, `moe_cache_update`) and take the general input path's whole-table copy.  Same fallback the existing `node_n_consumers > 1` case already uses.  The identity `moe_cache_take_over` needs no ids and is untouched. |
| **B — keep it fast** | `ggml_backend_sched_split_graph` | start the split **at** a `MUL_MAT_ID` that reads a host weight *and* whose ids root is computed above it in the same split (`next_moe_layer` keeps the layer's gate/up/down in the new split).  The router stays in the previous split — the shape every other config already has — so the pruning and the expert cache work again. |

Both belong to **block 09** (`patches/0009-…`, the file's `cache_op` / ids-readback hunks).

Guard A alone is correct but pays the whole-table upload per op; B is what makes the feature usable:

| arm (1 x R9700, `-n 128`, `-r 2`, interleaved) | tg128 | notes |
|---|---|---|
| Ling-3.0-tiny `-ncmoe 24`, KV on GPU (reference) | **176.0 / 176.4** | unchanged pre/post fix |
| Ling-3.0-tiny `-ncmoe 24 -nkvo`, pre-fix | **abort** | `GGML_ASSERT(id >= 0 && id < n_expert)` |
| … guard **A** alone | **24.6** | whole-table upload per routed op |
| … guard A + split **B** | **82.7** | pruning + cache restored (prefill 4480 vs 4435, i.e. A/B-neutral: the `-nkvo` prefill loss is the CPU attention) |

Cache check on the same arm with `MOE_EXPERT_CACHE_MIB=3000 -nkvo`: arena 2998.3 MiB, **17547/22584
reaches = 0.777 hit**, identical text — i.e. B really restored the update hook (under A alone the cache
would never be fed).  Graph census on the `-nkvo` arm: `direct=708 replay=7198 capture=118
properties_changed=96`, so the new boundary still captures and replays.

## 4. Gates run

Text is byte-identical between the `-nkvo` arm and the **default (KV on GPU) arm of the same config**,
and identical to the **pre-fix default arm** (so the default path is untouched):

| model (arch) | arm | text |
|---|---|---|
| Ling-3.0-tiny IQ4_NL (`bailingmoe3`, 24 L, 128 experts) | `-ncmoe 24` ± `-nkvo`, `-n 48`, seed 42, temp 0 | 251 chars `caa1f08f0a5e` (both; also with `-cmoe`, with `--host-experts pool`, and with the cache at 3000 MiB) |
| 35B-A3B UD-Q4_K_M (`qwen35moe`, 48 L) | `-ncmoe 48` ± `-nkvo` | 206 chars `cf7f8b23f404` |
| 35B-A3B + `--spec-type draft-mtp --spec-draft-n-max 3` (`-n 96`) | `-ncmoe 48` ± `-nkvo` | 330 chars `a2bf5eebdd09` |
| gpt-oss-20b MXFP4 (`gpt-oss`, 24 L, 32 experts) | `-ncmoe 24` ± `-nkvo` | 190 chars `7800c9430515` |
| Flash-Next GSQ-IQ3_XXS (`qwen4exp`, 48 L, the original report) | `-ncmoe 48 -c 8192` ± `-nkvo`, `-ot per_layer_token_embd.*=CPU -lm none -lzm off` | 91 chars `ffed60de47ae` |
| Qwen3.5-4B Q8_0 (dense) | ± `-nkvo` | 220 chars `6ff72e08d38b` (path not reached: no host weight) |

* Default-arm speed is flat pre/post (Ling tg 176.05/155.77 → 176.01/176.28; 35B tg 100.27/101.53 →
94.89/101.11, pp 1373/1465 → 1468/1464 — this box's pp arm is the noisy one); guard A's scan is skipped
entirely unless the input is a **host** expert weight, so a full-GPU arm takes no new code path.
* `scripts/check-gpu-faults.sh` (kernel journal + the four `-nkvo` logs): clean.
* Not run / not runnable here: any **multi-device** arm (`-sm tensor`/`-sm layer` on 2+ GPUs — this box
  has one R9700 plus an iGPU the build cannot target); the prefill-logit gate (no kernel, op or dispatch
  change — the default path's splits are unchanged, and `gate-prefill-logits.sh` compares the default
  27B Q8_0 point, so it is owed at the release that ships this, not by this change).

## 5. Delivery (done in r197)

1. ~~Amend **block 09** with the two hunks.~~  **Done in r197** (`patches/0009-…` regenerated,
   `release.json` tip `ba7c2b7b80252e313197e758b0070246d2ef0952` / tree
   `eaf673210a4b03c2121eccae2f10302310b2ae62`, `validate-set.sh` 16/16, per-block build 2/2, prefill-logit
   0.000000 / 100.000 %, coherence + MTP arms in `WORKLOG.md` 261101.002).
2. ~~An `upstream/` candidate.~~  **No** — upstream cannot reach it: its prune callback only fires when
   the routed op is the split's *first* node, and this shape's split starts at the attention-output
   `CONT`, so upstream copies the whole table.  The reachability came with block 09's generalised
   consumer search, so this is a delivery regression and there is nothing to file upstream.
3. `-nkvo` + host experts stays a *fittability* combination, not a recommended one: 82.7 vs 176 t/s on
   the Ling arm (the `-nkvo` penalty itself, `archive/work/kv-residency/INVENTORY.md` §"38.71 vs 100.67").
   Worth a line in `ENVIRONMENT.md`/block 09's note only if it ships.

## 6. Reproduction

```bash
# pre-fix: aborts at ggml-backend.cpp:2920
HIP_VISIBLE_DEVICES=0 timeout 300 build-rocm/bin/llama-cli \
  -m /llm/models/Ling/3.0-tiny/IQ4_NL/Ling-3.0-tiny-IQ4_NL.gguf \
  -ngl 99 -ncmoe 24 -nkvo -n 48 --seed 42 --temp 0 -p "The capital of France is" \
  --single-turn --no-display-prompt
# compare against the same command without -nkvo (must be the same text)
```

## 7. The delivery (r197)

Both hunks are an amendment to block 09, and the sibling fault found while validating them is an amendment
to block 15:

* **block 15 — `-nkvo` declines the window.**  A window is a DEVICE region plus a pinned host tail, but the
  plan block's guard never tested `offload`, so `-nkvo LLAMA_KV_RESIDENT_CELLS=64` announced "WINDOWED KV
  ENABLED: the newest 64 of 4096 cells stay in VRAM" and then built a plain **pageable CPU buffer** of the
  whole context (`CPU KV buffer size = 27.00 MiB`), reserved pinned spill budget for a cache that cannot
  spill, and left `kv_resident_cells` set — which `is_streaming()` turned into **qwen4exp's sparse decode
  arm** for a KV with no region at all (that arm's premise is a pinned host mirror resolved through a device
  pool).  The window now declines once with the reason (the pool already declined on `!offload`), and
  `is_streaming()` tests `kv_region` — the placement actually taken — which also covers a device cache whose
  region buft the backend refused.  Text unchanged everywhere: `-nkvo` + window already read as `-nkvo`
  alone (251 chars `caa1f08f0a5e`), the GPU-window arm still enables (`ROCmKVRegion0+win64c`).

Release gates recorded in `WORKLOG.md` 261101.002: coherence on nine arms (dense 27B/4B, Ling, 35B-A3B ±
`draft-mtp`, gpt-oss-20b, Flash-Next default/`-nkvo`/attendance-pool, the Ling window and `--host-experts
pool` arms), MTP prose `-n 3000` (`draft-mtp` == `none`, acceptance 1449/1791 identical, t/s flat), the
stock batched-bench `-npl 1,4,8` arm, prefill-logit **0.000000 / 100.000 %**, per-block build **2/2**, park
PASS, ckpt-reuse PASS, state-pipe PASS, `check-gpu-faults.sh` clean.
