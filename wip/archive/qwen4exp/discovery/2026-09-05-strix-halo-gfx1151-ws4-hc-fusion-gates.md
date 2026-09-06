# Strix Halo (RDNA3.5 / gfx1151) — WS4 prefill hyperconn-fusion gates (final build)

Date: 2026-09-05 (run from the 2026-09-05 session handoff)
Machine: Ryzen AI MAX+ 395 w/ Radeon 8060S (Strix Halo APU), 16 C / 123 GB
unified, single gfx1151 (rocm-smi carve-out 2 GiB; model runs from unified
memory).  ROCm `/opt/rocm-7.14-gfx1151` (build toolchain AND runtime).
Build: `~/llama.cpp`, branch `qwen4exp`, tip `248e47704` (= `dca0526a8` =
master `8b4b3558f` + blocks 01-13 + the three beta/qwen4exp patches + THIS
commit), built with the canonical `~/bin/build-llama-rocm-714`
(gfx1151 / ROCm 7.14, GGML_HIP_GRAPHS/MMQ_MFMA/NO_VMM/RCCL on, NO `-mllvm`
flag).  Compile-time provenance note: the binaries baked the pre-commit HEAD
`dca0526a8` (they were built from the working tree while the WS4 changes were
still uncommitted); content verified current — no source file is newer than
`build-rocm/bin/libggml-hip.so.0.23.0` (2026-09-05 17:13) and the tree is
clean at `248e47704` (17:33).
Model: `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
(3 shards, 87.24 GiB, 176.94 B params, IQ4_XS 4.25 bpw; fully resident).
Policy: NON-MTP.

## What was validated

The WS4 **prefill hyperconn fusions** (hc_mix_reduce_f32 + hc_combine_norm_f32;
ported from halo-box, A-adapted, the both-on bug fixed) **DEFAULT ON** — the
WS4 gate set on the final committed build:

1. clean warm-clock r3 depth-0 pp ladder (pp512..16384 descending, long-pp
   first), fusion default vs `GGML_CUDA_DISABLE_HC_FUSION=1` (same build);
2. depth 12k/32k pp rows + tg@depth regression (decode must not regress);
3. memory-stability ladder (−r3 through 32k, no crash/leak);
4. coherence spot-check on the final build (llama-cli same-seed text,
   fusion on == off).

## Protocol

- llama-bench, `-ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16
  -ctv f16 --load-mode none`, prompts `-p 512 -p 1024 -p 2048 -p 4096
  -p 8192 -p 16384 -n 0` (depth-0 ladder DESCENDING in one process so the
  first test is the long pp16384 warm-clock run) and `-d 12288` / `-d 32768`
  for the depth rows (`-p 512 -p 1024 -p 2048 -p 4096 -n 128` per depth; the
  depth fill is the first (untimed) work in each process, so the clock is
  warm before the first measured row).
- A/B toggle in the SAME build: `GGML_CUDA_DISABLE_HC_FUSION=1` = unfused
  prefill path; unset = fused (deliverable default).
- Warm page cache first (dd the 3 shards). One benchmark at a time,
  standalone invocations only. Raw files: `/tmp/gateA/gateA-*.md/.log`.
- Coherence: llama-cli same-seed greedy (`-p "The capital of France is"
  -n 20 --temp 0 --seed 42 --single-turn --no-display-prompt`), generated
  text diffed with the timing line stripped.
- The logit-level bit-exactness evidence (llama_decode harness, pp + 40
  greedy decode steps, per-step logits hashed, at 6/140/500/1400-token
  prompts; deterministic across fresh processes; llama-cli text on == off)
  was established on THIS build in the 2026-09-06 session — see
  `wip/strix-halo/notes-ws1-survey.md` (2026-09-06 LATE entry).

## Results

### Gate 1 — depth-0 warm-clock r3 ladder (t/s, mean ± stdev)

| test | fusion ON (default) | fusion OFF | Δ (ON/OFF) |
|------|--------------------:|-----------:|-----------:|
| pp16384 | 527.79 ± 5.28 | 490.51 ± 3.75 | **+7.6%** |
| pp8192  | 497.37 ± 0.82 | 463.10 ± 1.53 | **+7.4%** |
| pp4096  | 437.65 ± 2.33 | 406.12 ± 1.42 | **+7.8%** |
| pp2048  | 349.52 ± 2.33 | 321.23 ± 3.14 | **+8.8%** |
| pp1024  | 354.42 ± 6.76 | 331.50 ± 5.96 | **+6.9%** |
| pp512   | 315.79 ± 4.01 | 300.32 ± 3.66 | **+5.2%** |

Fusion default ON is +5-9% across the whole depth-0 ladder (meets/beats the
expected +5-6% at pp2048-16384), same build, clean warm-clock r3 protocol.

### Gate 2 — depth rows (each depth = one process; pp rows then tg128; r3)

d12288 (KV pre-filled to 12,288):

| test | fusion ON | fusion OFF | Δ |
|------|----------:|-----------:|--:|
| pp512 @ d12288  | 233.77 ± 12.42 | 244.20 ± 23.67 | within noise |
| pp1024 @ d12288 | 326.38 ± 1.09  | 291.80 ± 17.71 | +11.9% |
| pp2048 @ d12288 | 321.78 ± 0.85  | 308.61 ± 0.60  | **+4.3%** |
| pp4096 @ d12288 | 380.38 ± 50.80 | 360.48 ± 46.46 | +5.5% |
| tg128 @ d12288  | 22.04 ± 0.15   | 22.12 ± 0.04   | flat (decode) |

d32768 (KV pre-filled to 32,768):

| test | fusion ON | fusion OFF | Δ |
|------|----------:|-----------:|--:|
| pp512 @ d32768  | 253.13 ± 10.02 | 254.89 ± 10.33 | flat |
| pp1024 @ d32768 | 289.20 ± 17.79 | 272.01 ± 0.45  | +6.3% |
| pp2048 @ d32768 | 311.39 ± 1.07  | 297.53 ± 0.66  | **+4.7%** |
| pp4096 @ d32768 | 365.95 ± 47.60 | 347.54 ± 42.84 | +5.3% |
| tg128 @ d32768  | 20.10 ± 0.05   | 20.01 ± 0.12   | flat (decode) |

Notes: the small-pp rows at depth are noisy (±10-50; KV-state restore between
tests + clock); the pp2048 rows (±0.6-1.1) are the reliable signal and keep a
+4-5% fusion gain at depth. tg128 is FLAT at both depths — decode is
untouched (the fusion fires only at prefill; the `ne[1]==1` guard holds).

### Gate 3 — memory stability

Six sequential llama-bench processes (each −r 3; ctx up to 36,864 =
32768 depth + 4096 prompt on the 87 GiB resident model): all exited 0, no
OOM/error lines in any log, ~116 GB available restored after each process
(leak-free), page cache stays warm between runs. No crash through 32k depth.

### Gate 4 — coherence (final-build spot-check)

llama-cli `-n 20 --temp 0 --seed 42`, fusion ON vs `GGML_CUDA_DISABLE_HC_FUSION=1`:
generated text IDENTICAL (only the timing line differs). Consistent with the
session's logit-level bit-exact + determinism proof on this build.

## Status / routing

- WS4 gates PASSED on the final build; fusion stays DEFAULT ON (opt-outs
  `GGML_CUDA_DISABLE_HC_FUSION` / `_HC_MIX` / `_HC_COMB` remain).
- Delivery routing (2026-09-06): the qwen4exp-tree change (commit
  `248e47704`) is packaged as a 4th `beta/qwen4exp` patch
  (`ws4-hc-prefill-fusions.patch`) applied after the three existing patches
  (apply order verified on the beta base; see `beta/qwen4exp/README.md`).
  NEVER push from `~/llama.cpp`.
- Follow-ons (same 2026-09-06 session / 2026-09-05): re-derive the
  remaining A-vs-B depth-0 pp gap post-fusion (fusion ON vs the community
  build) and WS3 #2 (dense shortcut below the selection width in
  `build_layer_attn`) / WS3 #3 (routed-compact MoE mmq for i-quants,
  RDNA4-gated) — see `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-post-fusion-gap.md`.
  WS6 re-base still NOT indicated.
