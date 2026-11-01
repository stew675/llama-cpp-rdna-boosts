# A6 — a windowed KV across a `-sm tensor` split: design note

**Status: DESIGN ONLY — not implemented, not measurable on this box (one R9700).**  r181 ships the window
**single-device only**: the region buffer type is reached through `ggml_backend_dev_backend_reg(dev)`, and a
Meta (tensor-split) device has no reg, so the window is declined with one message (HANDOVER invariant 13,
`WORKLOG.md` 261030.005).  This note is the plan for closing that gap, plus the cross-check against
`common/fit.cpp`'s split accounting that any implementation owes.  Nothing here has been built.

Read with: HANDOVER §3 (the mechanism and invariants), `A2-MIXED-BACKING.md` §3.7.11.4 (what the window is
worth) and §3.7.12 (r181).  Everything below is 1-GPU reasoning from the code; the arms need a 2/3-device host.

---

## 1. What a tensor split actually is here (facts, with the sites)

| fact | where |
|---|---|
| `-sm tensor` puts **one Meta device** into `model.devices`; the loader's split shards every tensor | `src/llama.cpp:166-190` |
| `cache_k_l*/cache_v_l*` are split **AXIS_0** (the KV-head/embd axis), granularity = whole KV heads, extents taken from `attn_output.weight` | `src/llama-model.cpp:461, 609-611, 859-878` |
| so the **cell axis (dim 1) is never split**: every device holds *all* `kv_size` cells and a slice of dim 0. A cell index therefore means the same cell on every device — the window `W` is naturally shared | consequence |
| **exceptions that are MIRRORED** (each device holds the whole cache): `cache_idx_*` (qwen4exp QSA indexer), and the whole KV + dsv4 state for `LLM_ARCH_DEEPSEEK4/41` / dsv4-in-DFLASH | `src/llama-model.cpp:461-462, 556-566, 582-584` |
| the Meta buft is built from each child's **plain** device buft and cached per device (`std::map<dev, buft>`), so there is exactly one meta buft instance today | `ggml/src/ggml-backend-meta.cpp:565-591` |
| the Meta device reports **summed** free/total over its children | `ggml/src/ggml-backend-meta.cpp:123-134` |
| `supports_buft` compares the *devices* behind the bufts, not buft identity — a variant meta buft on the same meta device is accepted | `ggml/src/ggml-backend-meta.cpp:267-284` |

## 2. Why the data path is already split-aware (the important half)

`llama_kv_cache` builds *one* tensor set per buft and hands it to `ggml_backend_alloc_ctx_tensors_from_buft`,
i.e. the Meta buft's `alloc_buffer_n`.  That already fans out per child:

1. **allocation**: `alloc_buffer_n` builds each child's split simple tensors and calls
   `ggml_backend_alloc_ctx_tensors_from_buft(ctx_simple, simple_buft)` **per child**, with that child's slice
   sizes (`ggml-backend-meta.cpp:2659-2745`).  With a *region* child buft this is a per-device VMM reservation
   of the child's slices.
2. **window commit**: `init_tensor_impl` calls `ggml_backend_buffer_init_tensor(init_buf, t_ij)` on each child
   tensor (`ggml-backend-meta.cpp:2054`), and the region's `init_tensor` commits `W` cells at the *tensor's own*
   `nb[1]`/`ne[1]` (`ggml-cuda.cu:2890-2925`).  The meta split scales `nb[1]` for an axis-0 slice
   (`ggml-backend-meta.cpp:1991-1992`), so each device commits `W × its own cell stride` — the per-device share,
   no new code.  The ordering rule (resident VRAM committed before `SLAB_INIT`, invariant 1) is preserved
   because this happens at buffer allocation, per device, before the slab arms.
3. **lazy host growth and the spill budget**: `grow_for` runs from each child's `set_tensor`/`get_tensor`
   (the meta buffer's `set/get/memset/clear` all fan out), and `grow_cells` uses each tensor's own `nb1`; the
   pinned-RAM accounting (`ggml_cuda_kv_spill_allow`) is charged **per mapping**, i.e. per device, so the
   runtime ceiling is already aggregate-correct (`ggml-cuda.cu:2807, 2989`).

So the window's *commit*, *slide* and *spill accounting* need no new logic per device.  What is missing is the
plumbing that lets a child buft be a region buft at all, plus the two decision sites.

## 3. What is missing (five changes)

**C1 — a Meta buft variant whose simple bufts are per-device region bufts at the same `W`.**
Generalise `ggml_backend_meta_device_get_buffer_type` (`ggml-backend-meta.cpp:566-592`) to take the simple-buft
vector, and cache the variant by `(meta dev, simple bufts, W)`.  The buft context already stores
`simple_bufts` (`:435-458`), so nothing else changes.  Keep the iface, and in particular keep
`get_alloc_size = ggml_backend_cuda_buffer_type_get_alloc_size`-equivalent (max over children) so the fit's
`no_alloc` KV figure stays the **whole logical** KV (see §4).

**C2 — a grow fan-out, and *no silent no-op*.**
The registered proc `ggml_backend_cuda_kv_region_grow` dispatches on the buffer
(`ggml-cuda.cu:2747-2753`) and **returns 0 (success) for a buffer that is not a CUDA region buffer** — so if a
meta buffer ever reached it, growth would be a silent success on unmapped VA and the first batch above the
mapped extent would *fault the GPU* (`grow_regions` aborts on a refusal, not on this).  The meta arm must
iterate the children and require every child's `grow_cells` to succeed.  That needs two exported accessors
(`ggml_backend_meta_buffer_n_bufs`, and the child buffer of index `i`); both are static today
(`ggml-backend-meta.cpp:825-838`).  A missing accessor must be a hard error, never a 0 return.

**C3 — `--fit` must recognise the Meta region buft.**
`fit_buft_is_kv_region` resolves the reg to a *simple child* (r181) and then asks the CUDA predicate, which
compares `buft->iface.alloc_buffer` — false for a Meta buft.  Make it walk the meta buft's simple bufts and
require all of them to answer (needs either the same exported accessors, or a meta-side predicate registered as
a proc).  Without this the fit sees `kv_bytes = 0` and warns "needs one region-backed KV cache".

**C4 — the cache's `region_capable` test.**
`src/llama-kv-cache.cpp:408` asks `llama_kv_region_dev_reg(model.dev_layer(0)) != nullptr`; under a split that
is the Meta device (null).  It must accept a Meta device whose children answer, and the buft lookup
(`:576`) must call a meta-aware `llama_kv_region_buft_split` (a new proc, e.g.
`ggml_backend_meta_kv_region_buft_split(meta_dev, cells)`, that builds C1 from the children's
`..._kv_region_buft_split(child, cells)`).

**C5 — the fit's window decision must run for a split.**
Today the split branch (`common/fit.cpp:604-838`) **returns** before the window block (`:857-970`), and the
window block's `have_kv = nd == 1` is satisfied by a split (`nd` is the number of *model* devices = 1 for a
Meta wrapper) only by accident.  The decision must be taken **before** (or inside) the split branch, because it
changes the per-device `ttarget` the split balances against: the window frees VRAM that the arena reserve
(`cache_reserve.margin(tdevs[id], …)`, `:640-668`) would otherwise take.

## 4. Cross-check against `common/fit.cpp`'s split accounting

Measured/inspected on the r181 tree; this is the part a blind implementation would get wrong.

| quantity | what it is under a split | verdict for the window |
|---|---|---|
| `nd` (`:535`) | `llama_model_n_devices` = **1** (the Meta device) | the window block is *reachable*; it is not the blocker |
| `dmds_full[0].free` | **sum** of the children's free | correct aggregate budget |
| `margins[0]` (`:540-543`) | `margins_s[0]` = **simple device 0's** `--fit-target`, **not** the sum | **must be `sum(margins_i)`** or the budget silently omits `(ntd-1) × target` |
| `kv_bytes_full[0]` (probe) | the **whole logical KV** (the probe is `no_alloc`, so `llama_kv_cache::memory_breakdown` takes the `alloc_ctx_tensors_from_buft_size` branch over the *unsplit* tensors, `llama-kv-cache.cpp:1400-1410`) | `cell_bytes = kv_bytes/kv_cells` is the **aggregate** per-cell cost, so `W × cell_bytes` is the aggregate resident cost — consistent with the aggregate free. Do **not** divide it per device |
| per-device feasibility | each child commits `W × child nb1`; a device short of that fails the commit (logged, window not effective) | the fit must check **min over devices**: `free_i − margin_i − used_other_i ≥ W × cell_bytes_i`. Aggregate-only sizing can pass while one device cannot hold its share |
| split `estimate()` (`:809-838`) | `model + ntd*(ctx+comp)`, i.e. `mb.context` treated as per-device — but the probe's KV component is the whole logical KV, so the KV is counted `ntd×` | pre-existing conservatism, out of scope here. A windowed arm makes it worse (whole KV instead of `W` cells); a split window should feed the estimate `W × cell_bytes / ntd` per device |
| pinned-spill **plan** (`llama-kv-cache.cpp:483`, logical `cell_all`) | the plan is per logical cache, while `grow_cells` charges per device | correct for an axis-0 split; **under-counts `n_devices×` for a MIRRORED cache** (`cache_idx_*`, dsv4 KV). Invariant 14 says bound the *plan*: it must be built from the per-device geometry, not the logical cell bytes |
| `-sm tensor` + `-ncmoe` | `cache_reserve.margin(tdevs[id], host_bytes)` reserves the arena **per device** in `ttarget` | the window's VRAM has to be subtracted before that reserve — i.e. C5 is not cosmetic |

Two more consequences worth stating:

* **A mirrored cache's window costs `n_devices×`.**  `cache_idx_*` is MIRRORED on every device, so `W` cells
  there are `n_devices × W × cell_bytes` of VRAM for a cache that cannot spill unevenly.  On a 3-device
  Flash-Next split the indexer window (102 MiB at `-c 65536`) becomes 306 MiB — the fit's aggregate formula
  handles it (both `kv_bytes` and `cell_bytes` are whole-cache figures for a mirrored tensor), but the *prize*
  is only a third of the attention window's for the same cells.
* **The arena prize is per device**, so on a split the window's value is its own VRAM share per device
  (`A2-MIXED-BACKING.md` §3.7.11.4 A: +2.2-2.9 GiB and +3.9-6.7 points of residency on the 35B/122B at `-c
  262144`, single device).  A split already pays its own decode shape (+38 % `tensor` vs `layer` at 2 devices,
  `AGENTS.md`), so the split window is worth measuring on its own terms, not extrapolated.

## 5. Invariants a split window must keep

* **1** — the resident window is committed at buffer creation, before `SLAB_INIT`, **on every device** (the
  child `init_tensor` does it; a device that refuses logs and the window is not effective).
* **2/3/7/11** — the slide boundary is quantised in whole map units **per device**, the slide is monotone, and
  every tensor must start on that device's granularity.  The child buft's `get_alloc_size` already rounds per
  tensor (`ggml-cuda.cu:3140-3149`); do not move that rounding up to the Meta buft.
* **8** — a staging copy to/from a host-location mapping must be a kernel copy.  The meta buffer's
  `set_tensor`/`get_tensor` fan out to the child's region methods (safe); the meta buffer has **no**
  `cpy_tensor` (falls back to a host bounce), but `ggml_backend_tensor_copy` between two *child* buffers of the
  same buft reaches the plain CUDA `cpy_tensor` — verify that path on a spilling split before shipping
  (defrag, `seq_cp`, `state` device copies).
* **12** — a region-backed cache declines the park (r181/`0014`); keep it, including for a split.
* **13** — no un-checked `ggml_backend_dev_backend_reg`; a reg-less device reads as "the backend does not
  answer", and the new meta procs are reached through the **meta** device's own resolution (there is no reg
  there either — the fit must keep resolving through a child for the *CUDA* answers).
* **14** — bound the planned spill (§4, mirrored caches).

## 6. What must not be done

* **No per-device `W`.**  A cell index must mean the same cell on every device; the attention op reads all
  slices for one cell.  A device short of its share is a *decline*, not a smaller window.
* **Do not route the window through the Meta *weight* buft.**  The KV buft is a separate instance keyed by
  `(device, cells)`; weights must keep the plain children.
* **Do not let the grow proc return success on an unhandled buffer** (C2) — that is a GPU fault, not an error.
* **Do not grow the region for a park** (invariant 12).
* **Do not size a split window against aggregate free alone** (§4) and do not re-propose a
  percentage-of-slab form of `W` (deferred by decision).

## 7. Validation plan (a 2/3-device host; NOT possible here)

Same shape as the single-device gate set, per split mode, on a config that provably spills:

```bash
# 1. it must not assert, and must say exactly one thing per requested window
HIP_VISIBLE_DEVICES=0,1,2 llama-cli -m <27B Q4_K_XL> -ngl 99 -sm tensor -c 8192 -ctk q8_0 -ctv q8_0 \
    -f prompts/prose-rdna-boosts.txt -n 64 -v ...            # no knob: NO KV message
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_KV_RESIDENT_CELLS=1024 ...   # one WINDOWED line per cache, per run
HIP_VISIBLE_DEVICES=0,1,2 LLAMA_KV_RESIDENT_CELLS=auto ...   # one fit report, no "did not run"

# 2. prove the spill BEFORE believing any arm (per device: `KVREGION_FREE ... host=` > 0)
LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_DUMP=/dev/null ... -c 8192 -n 8
LLAMA_KV_RESIDENT_CELLS=1024 LLAMA_KV_REGION_DEBUG=1  ... | grep -c 'FLIP device'

# 3. purity: text identical across the split's *own* device-count reference (a 2-vs-3 text difference is
#    the tensor split's rounding class, GREEDY-PURITY.md §45 - gate WITHIN a device count)
ROCBLAS_USE_HIPBLASLT=0 ... hash via scripts/extract-generated.py     # plain == windowed
scripts/gate-prefill-logits.sh --sm tensor --device 0,1,2 --kv q8_0 \
    --ab "LLAMA_KV_RESIDENT_CELLS=1024"                                # -0.000000 / 100.000 %

# 4. the fit: a pinned -c that does not fit must REPORT the split window, and a big -c with a tiny
#    LLAMA_KV_SPILL_RAM_PCT must report the ceiling and DROP the window (invariant 14; the 235 GiB arm)
llama-fit-params -m <27B Q8_0> -c 262144 -ctk q8_0 -ctv q8_0
llama-fit-params -m <4B> -c 8388608 -LLAMA_KV_SPILL_RAM_PCT=1 ...

# 5. the corners on a spilling split: the coherence trio, the parking gate (it must take the save-and-clear
#    path and stay byte-identical to lazyoff), gate-state-pipe.sh, gate-ckpt-reuse.sh, check-gpu-faults.sh
```

Expected: the same numbers the single-device arms produced (`W × cell_bytes` per device, the arena reserve
growing by the KV's uncommitted share per device), **not** a rescaling of the single-device table.

## 8. Open questions for the maintainer

1. **Is a split window wanted at all?**  It is a bigger piece of work (C1-C5) than the single-device slice, and
   its prize is per device: on a `-ncmoe` split the arena gains its own share, on a dense split there is nothing
   to give the VRAM to.  The single-device form already covers "run a 27B Q8_0 at 262144 on one 32 GB card".
2. **Mirrored caches:** allow the window on `cache_idx_*`/dsv4 KV at `n_devices×` cost, or decline it there?
3. **Where does the fit's split `estimate()` conservatism (KV counted `ntd×`) get fixed** — here, or as its own
   item?  It is independent of the window and it makes `--fit` shrink a multi-device context earlier than needed.
