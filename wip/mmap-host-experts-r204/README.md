Thank you for taking #266 into r203, and for the careful notes on where each piece landed. Reproducing it block by block was a lot of work.

This one is a bigger ask, so please treat it as a proposal. You removed `--host-experts mmap` in r33 (#116) because a pageable host master faults on a no-XNACK GPU. We have kept a variant running on our side since then, because on our one-GPU box (60683 MiB RAM) it leaves about 10 GB more RAM available. Pinned mode pins all 46900 MiB of host experts; here the pinned bytes are capped by a budget, min(60 % RAM, RAM - 8 GiB), which is 36409 MiB on our box. The 10 patches below are that variant, cleaned up on r204 and built on #266's device remap. If you would rather not carry mmap again, that is completely fine. The numbers may still be useful.

The default stays `pinned`. Everything here is behind `--host-experts mmap`.

**How it reads the experts without faulting**
- 0001 restores the loader path: host experts stay in the pageable model mapping (CPU_Mapped).
- 0002: the GPU never reads pageable memory directly. Experts are registered with the GPU sparsely, through a file alias, and only when needed, so every read is from registered (pinned) pages.
- 0003 keeps the expert-cache arena across prefill transitions, so the decode cache is not rebuilt after every prompt.

**RAM tier and decode** (0004-0008)
- 0004 adds a RAM tier: a pinned buffer that holds the cold experts of each mmap table (`GGML_MOE_RAM_TIER_MIB`), with a per-expert source address (`cold_ptrs`: its tier slot, or the file alias). An expert is in either a VRAM slot or a RAM-tier slot (exclusive), so the pinned footprint is the tier, not the whole expert set.
- 0005 makes #266's device remap work for these tables: a cold expert is read in place over PCIe through its `cold_ptrs` entry. The pool cold read (#266 0002) never applies to a tier table.
- An admission is a swap: the victim goes VRAM -> RAM, and the incoming expert goes RAM -> VRAM. 0006 copies the likely victims back ahead of time on a side stream, into K spare RAM slots (`GGML_MOE_TIER_PRECOPY`, `GGML_MOE_TIER_SPARES`), so a swap rarely waits for its copy-back. 0007 hardens its error paths (a failed event record drains the side stream; a failed wait fails the swap).
- 0008 sets the defaults: 8 spares, and a tier table admits an expert from its 3rd touch (`GGML_MOE_TIER_TOUCH`).

**Bounding the pinned memory** (0009-0010)
- 0009: one pin budget covers the tier and the sparse registration together (`GGML_MOE_PIN_BUDGET_MIB`, default min(60 % RAM, RAM - 8 GiB)). If the whole tier does not fit, the hottest tables get one first (a partial tier) instead of all-or-nothing, and the warm-up registrations are bounded too. Before this, VmLck reached ~43.8 GB on our box (tier plus unbounded warm-up registrations). Now it stays at the budget.
- 0010: a read the cache declines (for example, a registration the budget refused) goes through a small per-table pinned bounce buffer (an 8-entry ring). It has no kill switch on purpose. The previous fallback copied straight from the pageable mapping, and the default budget declines some reads too: on r204 at the default budget that gave wrong greedy output and 47,771 KFD refusals, then aborted. The root cause: ROCclr pinned each pageable range it copied, until KFD's system-memory limit (RAM - RAM/64 - 1.5 GiB per the amdgpu source) refused new mappings ("exceeds resident system memory limit"; 8,378,295 times across three runs at a 20 GB budget on r200). The next lazy code-object load (the first 2-sequence `ssm_conv`) then failed with "no kernel image is available". With the bounce: 0 refusals, no crash.

**Gates** (Radeon AI PRO R9700, gfx1201, ROCm 10.1, r204 + the patches, your defaults, Flash-Next GSQ-RCO IQ3_S on 1 GPU, MTP with a Q4_K draft head, q8_0 KV, 256K context, slab headroom 4096)
- llama-server greedy: byte-identical in every arm (sha 3ae2e7418782): stock r204 pinned, patched pinned, patched mmap at the default budget, and patched mmap at a 20 GB budget.
- 259.6k-token needle: correct on patched mmap.
- 0 GPU faults, 0 KFD refusals.

**Numbers** (llama-server, t/s, arms alternated; greedy = two passes per run; conc2 = sum of two concurrent streams; RAM available = MemAvailable minimum over warm-up, short runs and the needle, sampled every 2 s. Stock pinned allocates its host experts with hipHostMalloc, which does not show in VmLck, so its pinned bytes are the host-expert total the cache logs at load)

| | greedy (mean) | conc2 (mean) | pinned host-expert bytes | RAM available (min) |
|---|---|---|---|---|
| stock r204 `pinned` | 74.7 | 95.3 | 46900 MiB (all host experts) | 6.3-6.9 GB |
| patched `pinned` | 74.1 | 97.3 | | |
| patched `mmap`, default budget | 72.4 | 89.7 | 36409 MiB (VmLck) | 16.4-17.2 GB |
| patched `mmap`, 20 GB budget | 28.4 | 32.4 | 19999 MiB (VmLck) | |

- 259.6k needle prefill: patched mmap 218 s (1189 t/s) vs stock pinned 203 s (1278 t/s).
- 0006+0008 together: greedy 72.9 vs 66.0 with `GGML_MOE_TIER_PRECOPY=0` (r200, your defaults), and 77.45 vs 73.24 on our r196 production build. Neither piece alone was significant.
- Known limit: below the full tier, the colder tables run without one and decode drops steeply (20 GB budget: 28.4 t/s, 59 tables without a tier). A tier shared across tables would help there.

Notes:
- `pinned` output is unchanged and its speed is within our spread. One part of 0003 (the narrow-2 fallback: the draft compute region keeps its cudaMalloc fallback when it fits outside the slab, instead of evicting arena tables) applies to every mode; in our pinned runs it did not change the arena evictions (17 per run, stock and patched alike).
- Tested on the R9700 with this model, single device only.
- The CUDA side is untested, as we have no NVIDIA card.

We have further improvements on the way, which we would send separately.

Files: `0001-*.patch` ... `0010-*.patch` (`git format-patch` against r204; they apply with `git am`) and this README. We are happy to run other shapes, budgets or models on our cards, and to rework any of it if you prefer a different shape. No rush at all.

(Claude-assisted, as before.)
