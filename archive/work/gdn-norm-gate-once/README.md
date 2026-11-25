# gdn-norm-gate-once — the GDN norm-gate fusion matches once per GDN layer, not twice

Promoted into **block 10** at `v17-919794500-r355` (contributor PR #361, `overdoingism`; the campaign was
submitted as `wip/gdn-norm-gate-once/` and never lived under `wip/` here — its record is this directory).
The code is `patches/0010-...`; the facts are `INVARIANTS.md` r355, the narrative `WORKLOG.md` 261124.011,
the block reference `patches/README.md` block 10.

## What the contributor found

A follow-up to #357 (block 10's SILU widening, r353).  #357 made `ggml_cuda_gdn_norm_gate_match()` run on
Qwen3.x, and that matcher ends with **three whole-graph reader counts**: `ggml_cuda_count_uses` for the
`RMS_NORM` and for its `MUL`, and the gate `MUL`'s inside `ggml_cuda_find_mul_q8_1_matmul`.  It ran **twice
per GDN layer** — at the `RMS_NORM` (the skip) and again at the gate node — and the gate node then called
`ggml_cuda_find_mul_q8_1_matmul()` once more.  Seven whole-graph walks per GDN layer where the unfused gate
takes one, on every `graph_compute` that runs the node loop (graphs off, a direct run or recapture, prefill,
multi-token verify).  A replayed graph pays nothing.

Two changes:

1. **The gate node takes the skip's decision.**  The skip records the gate node (`gdn_norm_gate_at`), and the
   pending norm has exactly one reader — the gate's `MUL` — so no other node can match it.  The norm is the
   pending `MUL`'s `RMS_NORM` source; the matmul is the node after the gate `MUL`, or the one after its no-op
   reshape (the nodes `ggml_cuda_find_mul_q8_1_matmul` accepted).
2. **The three counts are taken in one walk** (`ggml_cuda_count_uses3`), and
   `ggml_cuda_find_mul_q8_1_matmul()` takes an already-counted value through a defaulted parameter.  Every
   other caller is unchanged and the conditions are the same, so the decisions are.

The contributor's measurements (one R9700 each, ROCm 10.0): Windows/Ryzen 7 5700G `try_fuse` per
`graph_compute` 14.4 -> 7.5 ms, graphs-off decode 26.9 -> 28.2 t/s (fold off: 28.1); Linux/Ryzen 9 3900X
13.5 -> 7.1 ms with graphs-on decode unchanged (GPU-bound); text, fold census and graph-call counts identical
on 11 Linux cases and the reporter's #359 deep case.

## The reporter's finding, which the submitted text got wrong

The PR body (and its commit message) said the change **does not explain #359**.  The reporter then found
their harness sets `GGML_CUDA_DISABLE_GRAPHS=1` by default, so the fold's host cost *was* their −1.8 %:
r353 + this change reads **+0.4 %** against r345 where r353 alone read −1.8 %.  The attribution is corrected
in `WORKLOG.md` 261124.011 and `INVARIANTS.md` r355; the shipped commit message no longer carries the
superseded sentence (the amendment rewrote block 10's commit).

## The reviewer's two amendments to the submitted form (both made before the cut)

* **The derivation is bounds-checked and happens before the state is cleared.**  The submitted
  `cgraph->nodes[s + 2]->op == MUL_MAT ? nodes[s + 2] : nodes[s + 3]` read `nodes[s + 3]` with no bounds
  check and asserted *after* the read — and the reshape branch is the one taken on **every** shape measured
  (7 354 of 7 354 decisions), so it is the live path, not a theoretical one.  `s + 2 < n` / `s + 3 < n`
  now guard it, and the assert (extended to the relation it relies on: `direct || nodes[s + 2]` is the
  `RESHAPE` over `mul`) runs **before** `gdn_norm_pending` / `gdn_norm_gate_at` are cleared, so a broken
  invariant trips a tripwire instead of fusing a wrong matmul.
* No default argument on the matcher, no new switch, no message text in the block commit (block 10's
  amendment convention since r353).

## Verification (this box: gfx1201, ROCm 10.2, one GPU job at a time)

* **Decision equivalence, measured not argued.**  A temporary instrumented arm re-ran the PRE-change
  `ggml_cuda_gdn_norm_gate_match()` at every gate node and compared decision and tensors:
  **7 354 decisions, 0 mismatches** — qwen35 dense 27B (1 GPU and 3-GPU `-sm tensor`, graphs on and off),
  qwen35moe 35B-A3B, and **qwen4exp / Flash-Next** (the SIGMOID arm the contributor could not test).
  The instrument is not shipped.
* **Host time** (`GGML_CUDA_GCDBG=1`, graphs off, 27B tg128): `fuse` **1323.3 -> 752.4 ms** over the same
  234 158 `try_fuse` calls (258 `graph_compute`) = **5.13 -> 2.91 ms per `graph_compute`**, the same ~2x
  ratio the contributor measured on slower CPUs.
* **Decode**: graphs off **31.13 -> 31.77 t/s (+2.1 %)**; graphs on flat (32.15 / 32.18) and `pp512` flat —
  graph replay does not pay host time, exactly as claimed.
* **Text**: byte-identical across builds on 1 GPU (graphs on and off), 3 GPU `-sm tensor` (on and off),
  35B-A3B, and a two-slot `--kv-unified` server, with the fold census identical too
  (27B `silu=96`@`-ub 1` / `528`@`-ub 8` + `declined=528`; 35B-A3B `60`/`300`/`300`).
* Release ladder at r355: `gate-gdn-norm-gate.sh --kld` PASS (KLD −0.000000 / 100.000 % at `-ub 1` and
  `-ub 4`), `gate-prefill-logits.sh` canonical 0.000000 / 100.000 %, `gate-lora-fusion.sh` PASS,
  the MTP four-axis protocol (all four texts identical to the r352/r353 record), `llama-batched-bench
  -npl 1,4,8` flat, `test-fusion-gate-beta-multiseq` 77/77, `check-shlib-symbols.sh` 16/16,
  `check-gen-docs.sh` ok, `validate-set.sh` GREEN — details in `WORKLOG.md` 261124.011.

## Not covered

RDNA3, NVIDIA (HIP build only), Windows, several GPUs (the contributor's own gaps), and the contributor's
HIP-10.2-runtime / 10.0-compiler hybrid.  `gate-spec-purity.sh` was not run: the change touches no FA tile
launcher, no KV extent and no speculator, and the four-axis protocol is the arm that dimensions the verify
batch for it.

## Left open (`TODO` #379)

Even with the fold off, `ggml_cuda_try_fuse` costs ~7 ms per `graph_compute` on these boxes (27B, ~890
calls) because *other* matchers also count readers over the whole graph.  Reader counts taken once per graph
would cut that further wherever the node loop runs — a matcher-design change, so it is not part of this one.
