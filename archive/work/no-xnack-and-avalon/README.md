# The no-XNACK constraint, and what Avalon does instead

> **RETIRED AND ARCHIVED (2026-11-04).**  An analysis note (no code, no campaign).  Its question is answered,
> its recommended techniques are shipped or superseded by the r210/r211 host-expert tier, and the mode it was
> written against never shipped.  Read [`RESOLUTION.md`](RESOLUTION.md) first.

**Status (2026-11-02): analysis note, no code, no campaign.**  Opened after the `--host-experts window` prefill
campaign hit a fatal GPU page fault on a model larger than RAM (`WORKLOG.md` 261102.012/.013), to answer one
question against a working implementation: *can host-expert reads be done at speed on a GPU with no XNACK?*
Source read: `~/Avalon` (a llama.cpp fork for Qwen3.8-Flash-Next, `README.md` claims 98 t/s on an RX 7900 XT,
RDNA3, PCIe 3.0, **16 GB of RAM** for an 89 GiB model), commits to `c24c03d84`.

## The constraint, stated once

A GPU without XNACK cannot take a page fault and retry.  Device accesses must therefore target memory that is
**present**, and `hipHostRegister` does not provide presence — it installs page tables over reclaimable
page-cache pages (`INVARIANTS.md`, device-visible host memory must be resident).  Two mechanisms exist:
hold the pages (`mlock`, i.e. pinning — the RAM saving disappears), or **never let an SM read them**.  Avalon
takes the second.  (On MI200/MI300 `HSA_XNACK=1` + retry faults give the first for free; RDNA has neither.)

## What Avalon does (code-cited)

| mechanism | where | why it is safe on no-XNACK |
|---|---|---|
| every expert compute read is **VRAM** | `moe_cache_prefill_residents()` hands the MMQ node `moe_cache_entry_addr(pool, s)`, and `pool.gpu_base = slab` came from `cudaMalloc` | VRAM is always present |
| host -> VRAM moves are **copy-engine** copies | `moe_split_copy_by_kernel()` returns **false on HIP** by default (`GGML_CUDA_MOE_SPLIT_COPY=kernel` opts in) | the CE refills a reclaimed registered page instead of faulting (the asymmetry `probe-cold-read`'s header records) |
| the mmap is **registered**, for the copies | `moe_cache_pin_tensor()`: `cudaHostRegister(read-only\|portable\|mapped)`, page-rounded, per-tensor ranges merged where they overlap | makes the DMA direct instead of ROCm's staged bounce: measured **fills 115 -> 61 us, generation +1.6 %**; a copy ending outside a registered range costs prefill **-8 %**, one range per expert **-25 %** |
| the only host memory an SM reads is **pinned** | `moe_hosted_host_alloc()` = `hipHostMalloc(hipHostMallocMapped\|Coherent)` + `cudaHostGetDevicePointer` | pinned pages are always resident; this is the CPU/GPU split ("the CPU computes the missing experts") |
| speed from overlap, not from a memory trick | PREGATE prefetch of the next layer's experts on the copy engine; VRAM lent between the KV cache / compute buffers and the expert cache; CPU working the same layer; q8_0/q4_0 KV | none of it touches the residency question |

**So the answer to the question is: they did not make in-place host reads safe — they removed them from the
compute path.**  Our `--host-experts window` is the opposite bet (SM in-place reads of the mapping), which is why
it needs `mlock` and therefore stops being a RAM saving (`WORKLOG.md` 261102.013).

## What is worth taking

1. **Copy-engine prefetch of the next layer's experts during prefill** (PREGATE).  Pure overlap, no residency
   question, mode-agnostic (so it survives `-sm tensor`).  This is the piece behind their 1 400-1 850 t/s prefill
   on a **PCIe 3.0** link.
2. **Lend slab space to a prefetch ring and take it back.**  *Almost all of this we already have* — `r168`'s
   wide-aux growth decided once per wide pass plus `r87`'s forced prefill -> decode drop (`patches/README.md`
   block 10, `INVARIANTS.md`).  Avalon applies the same pattern to the **expert-cache slab** instead of the FA
   staging arena, so the only new thing is a *second consumer* of the lend; worth doing only on top of (1).
3. **Keep the registered-mapping alias fill** — we have it (the beta's `MOE_EXPERT_WINDOW_ALIAS_FILL`); they price
   the same trick independently at 115 -> 61 us per fill, which corroborates the direction.
4. **CPU-compute the missing experts concurrently** — upstream's `-ncmoe` CPU path; Avalon interleaves it with GPU
   work per layer.  A larger change, and a different campaign.

## What is NOT worth taking

* **Tail-expert skipping** (`GGML_CUDA_MOE_TAIL_SKIP`, on by default there, and in Strata).  It deletes a term
  from the layer's arithmetic for +5.3 % short-prompt / +11.9 % deep generation.  Refused outright and recorded as
  `GREEDY-PURITY.md` §83, including the rule for benchmarking against an engine that ships it.

## Caveats

Avalon is one model, one GPU, `-sm tensor` unmentioned; its results transfer as *techniques*, not as numbers.
Its VM/lending code is also far more invasive than anything the delivery would take from it — the two items above
are the only ones this note recommends.
