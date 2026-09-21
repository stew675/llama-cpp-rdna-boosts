# gfx1100 port — S1 groundwork record (2026-09-21)

Session: the first gfx1100 (single RX 7900 XTX, RDNA3_0) session on the `mmb-general` WIP.  This is
the raw evidence behind the plan in `gfx1100-porting.md`; the plan is the source of truth, this file
is the dated record.  **Status: S1 is only partially done** — the build and the cheap unit gates are
verified; the full baseline matrix (B1-B9) is the next session's work.

## Setup

* Host: 1× AMD Radeon RX 7900 XTX (gfx1100, 24 GB) + a Ryzen 7950X (which also exposes a **gfx1036**
  integrated GPU over HIP — see the trap below).
* ROCm `/opt/rocm-7.14-gfx1100` (`hipconfig` 7.14.60850).
* **Base** = delivery r12, `~/llama.cpp` branch `rdna-boosts`, HEAD `c8dda33dd`, applied tree
  `8a80535e556bef57666d2eaa4d3eb4cf93fb83f5`; already-built binary at `~/llama.cpp/build-rocm/bin/`.
* **WIP** = new worktree `~/llama-wip-gfx1100`, branch `mmb-gfx1100`, cut from `c8dda33dd`, the **6**
  WIP patches applied `git am` **6/6** → applied tree
  **`580db5174574f10cc92fb1cefa72281a65c77b12`** (exactly the documented WIP tree).
* Record branch: `wip-mmb-general-gfx1100` (this repo), cut from `wip-mmb-general` at `1f2c92d`.

## 1. The gfx1036 trap (environment)

`test-backend-ops` (and any tool that enumerates all HIP devices) sees **two** ROCm devices:

```
Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), Wave Size: 32, VRAM: 24560 MiB
Device 1: AMD Radeon Graphics,     gfx1036 (0x1036), Wave Size: 32, VRAM: 15607 MiB
```

The build targets **gfx1100 only**, so the gfx1036 device launches a kernel with no image and aborts:

```
ROCM error: invalid kernel file
current device: 1, in function ggml_cuda_kernel_launch ...
hipGetLastError()
...
ggml_cuda_get_rows_switch_src0_type<float>
```

**Fix: prefix every GPU command with `HIP_VISIBLE_DEVICES=0`.**  (This is not a code bug — it is the
headless iGPU being enumerated.  It cost the first `TOPK_QSA` run a core dump.)

## 2. Build

`cd ~/llama-wip-gfx1100 && ~/bin/build-llama-rocm-714` → **exit 0, 0 compiler errors**, all default
targets linked (`llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops`, `llama-app`,
`test-chat`).  ccache was warm for the delivery TUs; the WIP's changed TUs (`mmb.cu`, `fattn-qsa3.cu`,
`indexer-topk.cu`, `ggml-cuda.cu`, the FA group) compiled fresh.  `test-logits-width-probe` is built
by the same script (registered in `tests/CMakeLists.txt` by patch 3).

## 3. Unit gates (all with `HIP_VISIBLE_DEVICES=0`)

| gate | result |
|---|---|
| `test-backend-ops -o TOPK_QSA` (G5 indexer op) | **4/4 passed**, `2/2 backends passed` |
| `test-backend-ops -o FLASH_ATTN_QSA` (G2; qsa3 still gfx1100-gated off) | **26/26 passed**, `2/2 backends passed` |
| `test-logits-width-probe` gemma-12B Q8_0, `prompts/prose-rdna-boosts.txt`, `P=1024`, f16 KV | **`width_purity=PASS (worst maxdiff 0)`** |

Notes:

* **`TOPK_QSA` is the G5 oracle** — the name in `GROUPS.md` / the gfx1201 records (`-o INDEXER_TOPK`)
  is wrong; only the op enum is `GGML_OP_INDEXER_TOPK`.  `-o INDEXER_TOPK` matches no test case and
  can look like a pass.  (Corrected in `GROUPS.md`/`gfx1100-porting.md`.)
* **The `FLASH_ATTN_QSA` 26/26 is not yet qsa3 coverage on gfx1100.**  `ggml_cuda_flash_attn_qsa3_supported()`
  still requires `RDNA3_5 || RDNA4`, so the four packed (`qsa3=1`) cases fall back to the VEC kernel
  and the test only proves the harness runs.  **S4 adds `RDNA3_0` to the predicate and re-runs this;
  that is the real unit validation of the gfx11 qsa3 path on gfx1100.**
* Width probe input: `prompts/prose-rdna-boosts.txt` = 5491 tokens, sha256
  `fabdec65f5859e5508cc863a6e5f976706d5a770bb53eb1b406dc5aee3667727`.

## 4. What is NOT done (S1 remains open)

* The full baseline matrix **B1-B9** (coherence, prefill/decode, PPL, MTP) on the delivery **and**
  the WIP with all gates off.  Only the three unit gates above are recorded.
* No model has been loaded yet on this box, so nothing is known about memory fit / bench flags.
* The delivery binary at `~/llama.cpp/build-rocm/bin/` predates this session and was **not** rebuilt
  or copied aside — do that first in the next session (or keep it unaided, since the WIP lives in a
  separate worktree).

## 5. Carry-forward

See `gfx1100-porting.md` §13 for the S1 brief; the next concrete step is to build the delivery-side
baseline binaries (or preserve the existing ones), then run B1-B3/B5-B8 on both trees.
