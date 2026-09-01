# llama-bench configuration reference (qwen35moe + friends)

WHY THIS FILE EXISTS: on 2026-09-02 the Q6_K decode numbers "stopped
reproducing" (81 t/s instead of 97-98). It was NOT a regression: the run
had omitted `HIP_VISIBLE_DEVICES=0`, so llama.cpp silently layer-split the
model across ALL 3 R9700s, and layer-split decode pays a per-token RCCL
all-reduce sync. Every number in the docs below depends on the GPU pinning
and split mode. Use this file as the single source of truth for what was
actually run.

## Hardware

- 3x AMD Radeon AI PRO R9700 (gfx1201), ROCm `/opt/rocm-7.14-gfx1201`
  (build toolchain AND runtime; do NOT mix in system ROCm 7.1.1).
- 184 GB RAM, 16 cores. Runtime PM fix applied (GPU 0/1/2 dpm/power on):
  `echo on | sudo tee /sys/bus/pci/devices/0000:{03,06,09}:00.0/power/control`
- llama-bench binary: `build-rocm/bin/llama-bench` (GGML_HIP=ON,
  GPU_TARGETS=gfx1201). Docs-era reference build:
  `~/prs/llama.cpp/build-rocm/bin` @ `554691a72`.

## Models

| file | blocks | params | size | notes |
|------|--------|--------|------|-------|
| `Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf` | 40 | 34.66B | 27.29 GiB | May-1 file; the validated 1-GPU config |
| `Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` | 41 | 35.51B | 35.19 GiB | 2-GPU only (too big for 32 GiB) |
| `Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 41 | 35.51B | 21.10 GiB | has MTP/nextn head; NOT comparable to Q6_K |
| `True-Q3_K_M/...` | 41 | 35.51B | | 2-GPU tensor-split load-fix test |
| `Q5_K_M/...` | 41 | 35.51B | | load-fix test |

NOTE: the UD/41-block files carry a `nextn` (MTP) layer that Q6_K lacks;
decode numbers between the 40-block and 41-block files are NOT directly
comparable.

## THE GOLDEN RULES (learned the hard way)

1. **Single-GPU decode/prefill (Q6_K, Q4_K_M, ...): ALWAYS pin the GPU.**
   `HIP_VISIBLE_DEVICES=0` in front of llama-bench. WITHOUT it, llama.cpp
   layer-splits across all visible GPUs by default, and decode drops from
   ~97 t/s to ~81 t/s (per-token RCCL sync). The docs-era "1-GPU verified
   config" numbers ALL used `HIP_VISIBLE_DEVICES=0`.
2. **2-GPU runs: state the split mode explicitly.**
   `HIP_VISIBLE_DEVICES=0,1 --split-mode tensor` for tensor split (the
   Q8_0 config). Layer split (`--split-mode layer`, the default when no
   mode given but >1 GPU used) is a DIFFERENT number with different
   sync costs.
3. **Model file matters.** 40-block (Q6_K) vs 41-block (UD, has MTP head)
   decode is not comparable. State the exact .gguf path in every result.
4. **`-ub` (ubatch) changes prefill a lot** (512 vs 2048 = +65% on
   qwen35moe). Decode is unaffected by ub (n=1 per step), but prefill
   tables MUST record the ub used.
5. `-t` threads: 16 for all the numbers below (decode insensitive 8-32,
   prefill mildly; the tables below are all -t 16).

## Canonical command lines (copy-paste)

### Q6_K 1-GPU (the verified decode config)

```sh
# decode
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf \
    -ngl 99 -t 16 -r 1 -ub 2048 -p 512 -n 128
# prefill (ub=512 vs ub=2048 comparison)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf \
    -ngl 99 -t 16 -r 2 -p 512 -n 0 -ub 512
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf \
    -ngl 99 -t 16 -r 2 -p 2048 -n 0 -ub 2048
# long decode for clock/thermal sampling (n large so you can catch it live)
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf \
    -ngl 99 -t 16 -r 1 -ub 2048 -p 512 -n 2048
```

### Q8_0 2-GPU tensor-split

```sh
HIP_VISIBLE_DEVICES=0,1 ./build-rocm/bin/llama-bench \
    -m /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf \
    -ngl 99 -t 16 -r 1 -ub 2048 -p 512 -n 128 --split-mode tensor
```

### 2-GPU load-fix / slow-load tests (any quant, tensor split)

```sh
HIP_VISIBLE_DEVICES=0,1 timeout 480 ./build-rocm/bin/llama-bench \
    -m <model.gguf> -ngl 99 -t 16 -r 1 -ub 2048 -p 512 -n 0 --split-mode tensor
# slow-load probe: time from "load_tensors:" to the first pp row (>= 480s
# timeout; the Q6_K/Q3_K pre-fix load took ~3 min, looked like a hang)
```

### Coherence (same-seed greedy output, not llama-bench)

```sh
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-cli \
    -m /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf \
    -ngl 99 -t 16 -p "The capital of France is" -n 20 --temp 0 --seed 42 --single-turn
```

## Baseline numbers table (2026-09-02 re-verified, current fork tip)

All tg = decode (tokens/s), pp = prefill (tokens/s), `-t 16 -r 1 -ub 2048`
unless noted. "docs-era" = the protected `~/prs/llama.cpp` build @
`554691a72`; "current" = fork tip `482837e5a` (block 13 + multi-token
fusion). The two agree within noise.

| test | current | docs-era | docs ref |
|------|---------|----------|----------|
| Q6_K 1-GPU tg128 | 97.59 | 97.72 | 97.5-98.5 (fused-moe-mmq, plan-qwen4exp) |
| Q6_K 1-GPU tg512 | 99.09 | | |
| Q6_K 1-GPU tg2048 | 98.62 | | decode flat with n |
| Q6_K 3-GPU layer-split tg128 (no env) | 81.10 | | DO NOT compare to 1-GPU |
| Q8_0 2-GPU tensor-split tg128 | 95.59 | | 97.5 (q8-tensor-split-result.md) |
| Q6_K 1-GPU pp512 ub2048 | 3145 | | |
| Q4_K_M UD 1-GPU tg128 | 82.76 | | 41-block MTP model, different arch |
| Q6_K 2-GPU tensor-split pp512 (load-fix check) | 4455-4526 | 4526-4575 | 4538 (true-q3 doc) |

## WARNING: never compare layer-split vs single-GPU decode

The only reason tg512/tg1024/tg2048 ever looked slower (82/82/80) was
that those runs omitted `HIP_VISIBLE_DEVICES=0` (3-GPU layer split).
Pinned to one GPU, decode is flat with n (tg128 97.6, tg512 99.1, tg2048
98.6).  Decode is also flat with context depth (92.4 @ KV=512 and KV=16K,
pre-block-13 docs).  Any apparent decode "regression" that vanishes when
`HIP_VISIBLE_DEVICES=0` is added was never a real regression.

## What "regression check" means (established 2026-09-02)

A build is NOT regressed if it matches the docs numbers under the SAME
command line:
- Q6_K decode: `HIP_VISIBLE_DEVICES=0 ... -p 512 -n 128 -ub 2048` -> ~97-98
- Q8_0 2-GPU: `HIP_VISIBLE_DEVICES=0,1 ... --split-mode tensor -n 128` -> ~95-97
- Prefill: always record `-ub`; ub=2048 is the deployment config.

If a "regression" appears, FIRST re-run with the exact command above
before touching the build. The 09-02 scare was a missing
`HIP_VISIBLE_DEVICES=0`, not a code change.
