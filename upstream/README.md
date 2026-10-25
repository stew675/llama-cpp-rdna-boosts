# upstream/ — standalone upstream-PR candidates

## Base (2026-10-24 re-base)

Every candidate here was cut against `097cf33d9`.  After the `v14-3f03b1772-r127` re-base, each
sub-directory's README carries a **Re-base status** line: 9 patches still apply cleanly
(`git apply --check`) to `3f03b1772` but have not been re-validated there, and 8 need a re-cut.  None of
this blocks a release — nothing in `upstream/` is part of the delivery.

Each candidate is a **self-contained change that could be filed as a PR against unadulterated
`ggml-org/llama.cpp`**, in its own sub-directory:

```
<name>/
├── README.md                  # what it is, why, the validation, what was NOT validated
└── UPSTREAM-PR-<name>.patch   # the patch, cut against the master commit named in the README
```

**None of these is a deliverable.**  Every one is either *already inside a delivery block* (here as the
reference and the starting point for a PR branch) or *not in the delivery at all*.  Applying one on top of
the full delivery would double-apply the hunks.

**Filing one:** re-create the branch from current upstream master, re-run the README's own *Validation*
section there (the recorded numbers are from the fork-point tree), and check the README's "what was NOT
validated".  If a hunk is ever accepted upstream, drop it from the delivery at the next regeneration — the
delivery then carries only the fork-local remainder.

| directory | what | in the delivery | status |
|---|---|---|---|
| [`fa-kv-split-width/`](fa-kv-split-width/README.md) | ggml-cuda: make the FA KV-split (`parallel_blocks`) heuristic query-width independent for small batches, so decode and every spec-verify width reduce identically | **block 00** | validated (gfx1201, issue #25), applies clean to its base |
| [`fa-decode-verify-kernel-family/`](fa-decode-verify-kernel-family/README.md) | ggml-cuda: keep the decode/verify band on one FA kernel family (a quantized K/V cache took VEC for `n_q <= 2`, TILE from 3) | **block 05** | validated (3x gfx1201) |
| [`fa-512-rdna-head-cap/`](fa-512-rdna-head-cap/README.md) | ggml-cuda: let RDNA4 use the WMMA FA kernel for head 512 (prefill-only), instead of always falling to the tile kernel | **block 05** (r92, issue #188) | builds and runs on `097cf33d9`; `FLASH_ATTN_EXT` 3993/3993 |
| [`attn-k-null-mask-guard/`](attn-k-null-mask-guard/README.md) | llama: `llm_graph_input_attn_k` tolerates an absent kq mask (guard the fill, let `can_reuse_impl` accept null) | **block 05** | hardening only — no reachable null-mask path upstream today — reached in the delivery at `-np 1` (hybrid-K, r129) |
| [`get-rows-iq4-nl-sub-qk-k/`](get-rows-iq4-nl-sub-qk-k/README.md) | ggml-cuda: `GET_ROWS` on an `iq4_nl` row that is not a whole `QK_K` super-block (keeps the node on the GPU) | **block 08** | validated on gfx1201 (split count 142 → 22, +25-36 % prefill) |
| [`moe-router-tie-break/`](moe-router-tie-break/README.md) | ggml-cuda: make the fused MoE router bit-identical to the generic chain (softmax order, division, stable argsort tie-break) | **block 08** | validated on gfx1201 |
| [`q1-0-amd-perm/`](q1-0-amd-perm/README.md) | ggml-cuda: port the Q1_0 table lookups off HIP's private-memory `__byte_perm` to `__builtin_amdgcn_perm` | **block 07** | bit-identical (16.7M cases); no speed claim |
| [`ggml-alloc-unused-view/`](ggml-alloc-unused-view/README.md) | ggml-alloc: release view sources whose views are never consumed (a real leak; repro included) | **block 00** | repro + `test-alloc`/`test-batch-alloc` verified on master |
| [`mul-mat-id-dup/`](mul-mat-id-dup/README.md) | ggml-cuda: handle repeated expert ids in `MUL_MAT_ID` (per-occurrence compact rows, expanded column grids, no first-match `break`) | **block 00** (since r66) | dup cases 16/16 (base 0/16), full suites green |
| [`meta-offload-op/`](meta-offload-op/README.md) | ggml-backend-meta: let the tensor split op-offload host weights, and let a `MIRRORED` tensor serve an arbitrary byte range | **block 09** | measured standalone on pristine master (505 → 1690 t/s) |
| [`meta-split-unalloc-alias-and-alloc-fail/`](meta-split-unalloc-alias-and-alloc-fail/README.md) | ggml-backend-meta: resolve an alias through its view chain when the allocator never initialised it, and fail an allocation with NULL instead of asserting | **block 00** (+ the `offload_op` gate in **block 09**) | builds on `097cf33d9`; validated at 24 GiB-card parity |
| [`per-device-host-buffers/`](per-device-host-buffers/README.md) | ggml-cuda + llama + ggml-backend: per-device host buffer types and host-expert offload placement | **block 09** | validated on 2x / 3x R9700 |
| [`moe-host-expert-pinning/`](moe-host-expert-pinning/README.md) | llama: keep host-resident MoE expert weights pinned instead of downgrading them to the pageable mapping | **block 09** | +83 % `-sm tensor -ncmoe 99` pp8192, bit-identical text |
| [`sched-cross-device-input-order/`](sched-cross-device-input-order/README.md) | ggml-backend: order a cross-device split-input copy after the destination backend's queued work | **block 09** (issue #103 / PR #104) | fork form validated; upstream form applies clean, not built upstream |
| [`ggml-sched-probe/`](ggml-sched-probe/README.md) | ggml sched: probe the stored layout before the alloc-fallback's full device sync (single-device fast path) | **block 11** (qwen4exp) | single-GPU validated |
| [`kv-cache-keys-only/`](kv-cache-keys-only/README.md) | llama: keys-only KV caches — `llama_kv_cache` gains `v_enabled` (no V tensor, no V-side op) | **block 11** (+ the plumbing in **block 08**) | verified on master: indexer KV 72.00 → 24.00 MiB at ctx 8192 |
| [`lazy-mode-off-resident/`](lazy-mode-off-resident/README.md) | llama: make `--lazy-mode off` keep `TENSOR_READ_LAZY` tensors resident under `--load-mode mmap` (the help text promised it; the mapping won) | **block 00** (r122) | applies clean to `097cf33d9`; validated in the delivery tree |

## Backlog (not written up yet)

* **The masked-V / freed-cell kernel fixes** (HIP `fattn-tile.cuh`, HIP `fattn-mma-f16.cuh`, Vulkan
  `flash_attn.comp` / `flash_attn_cm1.comp`) — validated on gfx1151 + Vulkan, folded into the delivery,
  but **not** re-cut against unadulterated master: the HIP halves sit on the fork's native-bf16/WMMA FA
  path, so the upstream form needs a fresh port.

## Note for old paths

Until 2026-10-20 the candidates were flat files here (`UPSTREAM-PR-<name>.md` + `.patch`).  A record
written before then that cites `upstream/UPSTREAM-PR-<name>.md` means
`upstream/<name>/README.md`; the patch is unchanged and lives beside it.
