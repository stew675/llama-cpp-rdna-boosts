# llama-cpp-rdna-boosts

**AMD RDNA performance and correctness work for [llama.cpp](https://github.com/ggml-org/llama.cpp),
delivered as a verified, reproducible patch set.**

> **Fast decode. Deep speculative drafting. Less VRAM. Greedy output that still means what the
> model meant.**
>
> 16 self-contained patches for a clean llama.cpp checkout — apply all of them, or only the ones
> you want.

---

## What this is

`llama-cpp-rdna-boosts` packages the RDNA/ROCm work of the
[`stew675/llama.cpp`](https://github.com/stew675/llama.cpp) fork (`rdna-boosts` branch) as a
**16-patch series** (block `00` + blocks `01`–`15`) that applies to a clean upstream llama.cpp
checkout at a pinned **fork point**.

It is not a fork people have to live in. It is a **delivery**: a frozen base, an ordered set of
`git am` commits, a `release.json` that names every artifact by sha256, and a validation workflow
that proves the patches still apply and still produce the recorded tree.

The headline work:

- **Multi-token prediction (MTP)** that is worth using — an adaptive draft depth, a drafter and
  verifier that agree numerically, and a verify path fast enough that deep drafts pay. It grew out
  of upstream PR [#27210](https://github.com/ggml-org/llama.cpp/pull/27210) and has since been carried
  well past it: the controller, its tuning, and the deeper depths it can afford are all downstream of
  this repo's numerics and batch-verification work.
  → **[MTP & Adaptive MTP](MTP-and-Adaptive-MTP)**
- **Prefill speed** — a fused chunked gated-delta-net (GDN) kernel, RDNA4 WMMA flash-attention,
  a fused MoE gate+up+GLU MMQ path, and pair-fusion fixes.
- **Decode speed** — k-quant mmvq boosts, a fused MoE expert tail, and a hybrid HIP all-reduce
  for the small-tensor decode path.
- **Memory** — an attention-memory campaign that frees several GiB of VRAM without changing a
  single output bit.
- **New model support** — qwen4exp / Qwen3.8-Flash-Next (QSA sparse attention, hyper-connections,
  managed lazy reader, MTP draft head).

## What it aims to achieve

| Goal | How we get there |
|---|---|
| **Make RDNA fast** | Tune the kernels and dispatches for `gfx1100` / `gfx1151` / `gfx1201`, with runtime arch selection (one multi-arch binary). |
| **Never trade away correctness for speed** | The *greedy-purity* rule: a single-token decode and a speculative verify must compute the same thing. If a fast path can't do that, it gets fixed or gated — not shipped. |
| **Make other backends uninjured** | The set is upstream-shaped. CUDA/MUSA/SYCL/Vulkan/CPU stay consistent (no aborts, no uninstantiated kernel pairs) even where the work is unreachable for them. |
| **Leave headroom for context** | Free the scratch and staging buffers that made deep drafts at long context fail to load. |
| **Be verifiable** | Every release ships the patches, `release.json`, and `SHA256SUMS`. CI re-applies the set on a fresh tarball of the fork point and checks the applied tree. |
| **Stay honest about what is promised** | Distinguish a *text-level guarantee* from a *logits-level measurement*; publish the exact scope of both. |

## Supported hardware

The set targets the **AMD RDNA 3 / 3.5 / 4** families:

| Family | Arch | Example parts |
|---|---|---|
| RDNA 3 | `gfx1100` | RX 7900 XTX / XT, RX 7800 XT |
| RDNA 3.5 | `gfx1150` / `gfx1151` | Strix Point / Strix Halo APUs |
| RDNA 4 | `gfx1200` / `gfx1201` | RX 9060 XT; RX 9070 / 9070 XT |

RDNA4 sees the most benefit (the WMMA flash-attn path, chunked GDN, k-quant boosts and the
internal all-reduce were built and validated there first), but as much as possible is back-ported:
the chunked GDN has a dedicated first-gen WMMA port for gfx11, WMMA flash-attn runs on RDNA3.0/3.5
with tuned head limits, and block 10 adds a dedicated RDNA3.5 mmvq table. Only block 12's internal
all-reduce is genuinely RDNA4-only; elsewhere it falls back to RCCL.

## Quick start

```bash
# 1. Fresh llama.cpp at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$BASE"

# 2. Apply the set — one commit per block, on a fresh `rdna-boosts` branch
bash <path-to-this-repo>/scripts/apply-all.sh .
#    strict 16/16 `git am` on the recorded base

# 3. Build (trim GPU_TARGETS to your arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 \
      -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 4. Coherence gate — same-seed output must match a known-good build
./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 \
  --no-display-prompt --single-turn
```

> **Always pass `-ctk X -ctv X` (matching types).** Mixed K/V cache types are hard-rejected by
> design — every mixed pair measured 1.7–3.6× slower and never smaller.

## The 16 blocks

| Patch | What it does |
|---|---|
| `0000` | **Structural & architecture fixes** — FA small-batch KV-split width invariance (decode and every verify width reduce identically) + Vulkan masked-V fixes. The base everything else sits on. |
| `0001` | **Adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`) — the credit-bucket depth controller. |
| `0002` | **Fused chunked GDN prefill** (bf16/WMMA, RDNA4 + gfx11 ports) + the MTP chunked-prefix path. |
| `0003` | **BF16 KV cache + native-BF16 flash-attn.** |
| `0004` | **RDNA4 WMMA flash-attn** + Q6_K mmq prefill perf. |
| `0005` | **CPU bit-identical decode/verify batches.** |
| `0006` | Host-buffer revert for discrete GPUs. |
| `0007` | Meta device-wrapper skip. |
| `0008` | **Fused-core prefill kernels** + GPU bit-identical results (needs 03+04). |
| `0009` | Meta-buffer compute-container headroom. |
| `0010` | **k-quant boosts** — Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR + q8_1 quantize-cache fusions, plus a dedicated RDNA3.5 table. |
| `0011` | Skip CUDA graphs for multi-token prefill (decode keeps graph replay). |
| `0012` | **Hybrid HIP all-reduce** — internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-gated. Includes the opt-in copy-engine (SDMA) mode. |
| `0013` | **Fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — the MoE prefill fusion and the decode/verify band fixes. |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA, fused indexer top-k, HC_MIX/HC_COMBINE, managed lazy reader, MTP draft head, per-arch decode policy. |
| `0015` | **Attention-memory wins** — derived kq mask, native q8_0/q4_0/bf16 K/V in FA, QSA score/bias/visibility reductions, keys-only indexer cache (~3.4 GiB/GPU + ~1.2 GiB host on qwen4exp). |

## The rules we hold ourselves to

> **Greedy purity first.**
>
> Speculative decoding compares the verifier's `argmax` against the draft token. If a 1-token
> decode and an *n*-token verify of the same state compute even a ULP differently, a near-tie can
> flip and a perfectly good draft gets rejected. So we make them compute the same thing, and we
> gate on it.
>
> A fix that restores that agreement may cost a few percent of raw throughput. **We land it,
> record the cost, and repay it structurally later.** Acceptance is worth more than a fraction of
> a GEMM.

Everything else follows from that: validate at the workload's real length, measure the model
class you touched, and never present a short spot-check as a verdict.

## Releases & verification

Frozen deliveries are published as GitHub Releases, tagged `v16-<fork-point>-r<N>`. A tag push is
the only thing that cuts a release; each one carries `rdna-boosts-all.patch`, `patches.tar.gz`,
`release.json` and `SHA256SUMS`. `scripts/validate-set.sh` (and CI) re-checks the artifact hashes,
the strict apply on a fresh tarball of the fork point, and the applied tree.

`release.json` is the single source of truth — never hand-edit the hashes.

## Explore the wiki

- **[MTP & Adaptive MTP](MTP-and-Adaptive-MTP)** — the full story: what MTP is, how the drafter and
  verifier were made to agree, how the verify batch was made cheap, and how the adaptive depth
  controller turns all of it into a real speedup over static draft lengths.
- **[MTP Quick Reference](MTP-Quick-Reference)** — the flags, the depth policy, and the commands.

In the repository:

| | |
|---|---|
| [`README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/README.md) | consumer overview + workflow |
| [`patches/README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md) | apply details, env knobs, per-block notes |
| [`GREEDY-PURITY.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/GREEDY-PURITY.md) | the purity rulebook (read before shipping) |
| [`benchmarks/mtp-adaptive-methodology.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/mtp-adaptive-methodology.md) | the MTP gate protocol |
| [`WORKLOG.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/WORKLOG.md) | dated record of every change |

## Community

This has become a community effort. Special thanks to the people who found issues and offered
fixes: [@1337hero](https://github.com/1337hero), [@bakon11](https://github.com/bakon11),
[@briansp2020](https://github.com/briansp2020), [@eoprede](https://github.com/eoprede),
[@tungel](https://github.com/tungel), and [@DanoPTT](https://github.com/DanoPTT).

And thank you to the
[Strix Halo llama.cpp project](https://github.com/halo-box/strix-llama.cpp) for the prefill-tuning
inspiration, and of course to the [llama.cpp](https://github.com/ggml-org/llama.cpp) team whose
work everything here rests on.

## License

Same as llama.cpp (MIT).
