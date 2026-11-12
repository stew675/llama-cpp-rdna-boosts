# Upstream-PR proposal: budget `graph_max_nodes()` for the KV Hadamard rotation's per-layer graph cost

> **Upstream candidate** · status: prepared 2026-11-12 (r267), **built and measured on `a691b2345`** · not filed.
> Delivery: **block 12** (the `graph_max_nodes()` / node-budget part; the delivery also makes
> `resolve_fused_ops()` treat an unreservable probe as "cannot verify, disable", which does not exist upstream) ·
> patch: [`UPSTREAM-PR-kv-rotation-node-budget.patch`](UPSTREAM-PR-kv-rotation-node-budget.patch) (12 files, +79/-0,
> applies cleanly to **`a691b2345`** with `git apply --check`, sha256 `2b4baafbdeb9e292b16d4ce852d0db1fd1e4ab309df6d2b5d7fb3c267d291173`).

## TL;DR

`llama_context::graph_max_nodes()` sizes the scheduler's hash table and its node/leaf id arrays, and both are
indexed over **nodes + leafs**, but its default branch is the per-tensor heuristic `max(1024, 8 * n_tensors)`.  A
**quantized** KV cache also carries the Hadamard rotation (#21038), and since **#31073** ("kv-cache: keep rotation
constants on the GPU") every layer's rotation constants are their own device tensors, i.e. one graph **leaf** per
rotated tensor per layer on top of the four `llama_mul_mat_hadamard` chains (3 nodes each).  That is **12 nodes +
2 leafs per layer**, which the per-tensor budget does not model.  A model with few tensors per layer and many
layers falls off it: **phi-4** (`phi3`, 40 layers, 243 tensors, 4 matmuls per layer because its QKV is fused)
needs **2062 entries (1649 nodes + 413 leafs)** against a **2053-entry** table (`max(1024, 8*243) = 1944`, next
prime 2053) and aborts before the server listens:

```
ggml/src/ggml-impl.h:337: fatal error          # "visited all hash table entries" in ggml_hash_find_or_insert
#3 ggml_backend_sched_split_graph
#4 llama_context::graph_reserve
#5 llama_context::resolve_fused_ops(...)::{lambda(llm_fused_op_probe const&, bool&)#1}::operator()
```

The `split_only` probe path in `graph_reserve()` calls `ggml_backend_sched_split_graph` directly, so it has none of
the `hash_set.size >= n_nodes + n_leafs` asserts the reserve/alloc paths carry: the model dies with no llama-level
message at all.  The same model with `f16`/`f16` KV builds a 1169-node graph and loads (no rotation), which is what
made this look like a quantized-KV defect rather than a budget one.

The fix makes the budget ask the memory module: `llama_memory_i::graph_extra_entries()` returns the entries the
KV cache's own graph builders add (**7 per rotated tensor per layer**: 6 nodes + 1 leaf), counted from the rotation
tensors `init_rotation()` actually created (so the ctor's predicate is not duplicated), and
`llama_kv_cache_iswa` / `llama_memory_hybrid` / `llama_memory_hybrid_iswa` sum their attention caches.  phi-4 goes
1944 -> **2504**; an iSWA model's two sub-caches both count (gemma-4-12B: 6016).  `graph_reserve()` also refuses a
graph that does not fit and names both numbers instead of dying inside ggml.  A graph that is *under* the budget is
untouched, so this is allocation sizing only: no kernel, no dispatch row, no arithmetic.

## Reproduce (unmodified `a691b2345`)

```sh
# aborts (~3 s in, before the model is loaded)
hip assign, then:
llama-cli -m phi-4-IQ4_XS.gguf -c 16384 -ub 2048 -b 2048 -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 99 -p hi -n 2 --single-turn --no-display-prompt --fit off
```

Two census lines make the mechanism explicit (add a temporary `GGML_LOG_INFO` to
`ggml_backend_sched_split_graph` and to `llama_context::graph_reserve`):

```
DEBUGPROBE graph_reserve: n_tokens=1 n_seqs=1 split_only=1      # the kv_rows / FA probe, 1 token
DEBUGPROBE split_graph: n_nodes=1649 n_leafs=413 hash_set=2053  # 2062 > 2053
```

With the patch applied the same command loads and generates; the same-model `f16` arm and the
`LLAMA_ATTN_ROT_DISABLE=1` arm generate the same text (`a1b7eb2ee7a6`, 31 chars, seed 42, temp 0).

## Validation (this box: 1x Radeon AI PRO R9700, gfx1201, ROCm 10.1.0)

| arm | unmodified `a691b2345` | + this patch |
|---|---|---|
| phi-4 IQ4_XS, `-fa on -ctk q8_0 -ctv q8_0`, `llama-cli` | **abort** (`ggml-impl.h:337`, 2062 vs 2053) | loads, 31 chars `a1b7eb2ee7a6` |
| phi-4, `f16`/`f16` | loads | loads, `a1b7eb2ee7a6` (unchanged) |
| phi-4, `q8_0` + `LLAMA_ATTN_ROT_DISABLE=1` | loads (no rotation tensors) | loads (budget term drops with the tensors) |
| Qwen3.5-4B Q8_0 / Qwen3.8-27B UD-Q4_K_XL / gemma-4-12B, `q8_0` KV | loads | loads, greedy text byte-identical to the unpatched build (`1c5d32ac537d` / `d45e2221805a` / `cd85935d5fee`) |
| build | `all` target, ROCm 10.1.0, gfx1201 | `all` target, clean tree, same flags |

The `all` build of the patched `a691b2345` is a fresh tree at a new source path (`/tmp/up302`), so this patch
compiles from scratch, not incrementally.  The delivery's own build of the same change is what r267 shipped, and
its release gates (`validate-set.sh`, per-block build, coherence on four models incl. an iSWA and a hybrid one,
`gate-prefill-logits.sh` f16 and `--kv q8_0` at 0.000000 / 100.000 %, MTP smoke, `check-gpu-faults.sh`) are
recorded in `WORKLOG.md` 261112.003.

## Not validated

* **Non-AMD backends.**  Nothing here is backend-specific (it sizes host-side scheduler tables), but it was only
  run on gfx1201.
* **Other budget-starved models.**  The term is exact for the KV rotation; it does not give the heuristic a general
  margin, so a model whose graph is inflated by something *else* can still overflow.  The refusal message
  (`the graph needs N entries (X nodes + Y leafs) but the scheduler was reserved for M`) is what turns that from a
  bare abort into a diagnosable failure, and it is the reason to take this patch even where the term is not needed.
* **Copies into the hash table.**  `ggml_backend_sched_new()` sizes the table from the budget and the split pass
  then inserts input/output copies (upstream's own `FIXME: needs to be size*2 to account for leafs`).  This patch
  does not address that slack; it only makes the budget cover the graph itself.
* **In a delivery build** this patch must not be applied on top of the full block set (block 12 carries the same
  hunks already).

## Alternative considered, not taken

Growing the scheduler's hash table (and the id arrays, `context_buffer`, and rehashing or refusing) inside
`ggml_backend_sched_split_graph` would fix the whole class, including the copy slack.  It was declined here
because the table is indexed by hash id and the ids persist across graphs unless the caller resets, so a
mid-life rehash has to preserve them; that is a scheduler-internals change with a regression surface far larger
than a budget term, and the budget term is what the caller already controls.
