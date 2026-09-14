# Next-session prompt — recover the quantized-KV prefill cost (TODO item 21)

> Paste the block below into the next session.

---

Session prompt — recover the quantized-KV prefill cost (TODO item 21)

Context: working in `~/llama-cpp-rdna-boosts` (delivery repo) and `~/llama.cpp` (fork). The Issue #30
wider-configuration campaign is complete and promoted: block 15 carries the `V4` activation policy
(native q8_0/q4_0 staging is the **default** for sub-F16 quants; `GGML_CUDA_FA_KV_NATIVE` unset=auto,
1=force on, 0=force the old F16-staging path) plus the new q4_0 native arm; block 04 carries the arch- and
split-aware RDNA prefill config. Delivery is at release **`v16-790cf51aa-r3`** (tip `a2c8d06a7`, tree
`eb5b7583`); the fork `rdna-boosts` is clean at that tip. Read
`wip/issue-30-mtp-decode-regression/README.md`, `MEASUREMENTS.md` §B (the V4 prefill table) and §D (the
prefill fix), `RECURRENT-SNAPSHOT-BUDGET.md`, `patches/README.md` (the 2026-09-14 block-04/block-15
sections), `GREEDY-PURITY.md` §34/§35, and **TODO item 21**.

Task: close the gap in **TODO item 21** — the V4 native staging is a decode/memory win but a *prefill*
cost that grows with depth and split. Measured on the 27B UD-Q4_K_XL (`pp150000`, f16 control): q8_0 is
**−4.2 % (1 GPU), −7.8 % (2-card `-sm tensor`), ~−8 % (3-card)** off the staging path (only ~−1.2 % at
pp32K), and it is **parity with stock** (delivery q8_0 −1.2 / −0.2 / +2.4 % across 1/2/3 cards) while f16
is +2.4 / +6.9 / +9.6 %. Goal: make q8_0 (and q4_0) prefill match the f16 margin, without losing the
**+23 % q8_0 decode at d65K** or the **adaptive-MTP `-c 196608` ceiling-12 load** (both depend on V4
removing the ~744 MiB scratch).

Mechanism to attack: the native loaders dequantize each 16-byte K/V chunk synchronously into the smem tile
(`ggml_cuda_fattn_dequantize_q8_0_chunk` / `_q4_0_chunk` in `ggml/src/ggml-cuda/fattn-common.cuh`), so a
quantized source cannot use the `cp_async` pipeline the F16 staging copy gets. Candidate approaches:
(a) native at decode/verify only (`Q->ne[1] <= 8`), staging at prefill — must be paired with scoping the
F16 conversion so the decode-only graph keeps the memory (else the MTP load regresses);
(b) pipeline the native staging (double-buffered dequant into smem) to recover cp_async-equivalent
throughput; (c) a hoisted/shared prefill conversion. Relevant files: `fattn-common.cuh` (predicates,
dequant helpers, launcher/staging), `fattn-tile.cuh`/`fattn-tile.cu` and `fattn-mma-f16.cuh` (native
loaders + dispatch), `fattn.cu` (chooser/config). Work as an **uncommitted experiment** on top of
`a2c8d06a7`; save the diff under `wip/issue-30-mtp-decode-regression/patches/` before proposing promotion
and do not touch `patches/` until validated.

Environment: 3× R9700 (gfx1201), ROCm 7.14 (`/opt/rocm-7.14-gfx1201`), build
`BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`, then
`LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:build-rocm/bin`. Models:
`/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf`, `.../Q8_0/Qwen3.8-27B-Q8_0.gguf`,
`~/Qwen3.5-4B-Q8_0.gguf`; prompts in `prompts/`; harnesses in
`wip/issue-30-mtp-decode-regression/tools/` (`bench_mtp.py`, `depth_sweep.sh`, `run_depth_audit.sh`,
`runarm.sh`, `text_gate.sh`, `vwidth.sh`, `width_matrix.sh` + `width-matrix.cpp`, `purity.sh`,
`sweep_nwarps.sh`, `sweep_pertype.sh`), with the ad-hoc session scripts in `/home/stew675/wip-issue30/`
(`build.sh`, `adaptive-4axis*.sh`, `matrix-*.sh`, `remeasure.sh`).

Gates (do not weaken): q8_0/q4_0 prefill >= stock by the f16 margin; same-seed text native == staging
(`tools/text_gate.sh`); `W = 1..8` one logits hash for every supported KV type (4B q4_0 especially);
MTP `n_max 3` acceptance unchanged; adaptive `-c 196608 -ctk/ctv q8_0 --spec-draft-n-max 12` still loads
at the default `n_slots=4`; and **always measure 1 GPU as well as `-sm tensor`** (the tensor split masks
single-card effects). Protocol: screen with the `t = a + b*n` slope fit at pp8192/16384/32768/49152, one
arm at a time (3 GPUs may be used in parallel only if the benchmark outputs stay self-consistent).
