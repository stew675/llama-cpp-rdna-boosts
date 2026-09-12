# 2026-09-12 — QSA *forced-sparse* q8_0 residual (TODO item 4): deep dive

Box: Strix Halo APU (gfx1151), 1 device, ROCm 7.14 (`/opt/rocm-7.14-gfx1151`).
Build: the delivered 15-patch set, worktree `/home/stew675/ll25/verify` (tip `9f46e926f`, tree
`f4791066f4a582316b1ca95f51c96cd10b905ef7`).
Model: Qwen3.8-Flash-Next UD-IQ4_XS + `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.  Prompt `/tmp/p5000.txt`
(4293 tokens).  Baseline gate: `plain (--spec-type none)` vs `draft-mtp --spec-draft-n-max 3`, greedy,
`--seed 42 --temp 0`, text hashed with `tools/textgen.py`.

This continues `RECORD-2026-09-12-qsa-sparse-width.md`.  The residual under study:
forced sparse (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`) + `-ctk q8_0` + the `p5000` prompt
→ `plain a57bc13bbf2a` vs `n3 3124adfd2b94` (reproduced this session).

## 0. Summary

The residual is **not** a decode/verify kernel-width dependence.  A multi-step, teacher-forced replay of
the plain greedy sequence at every verify width is **bit-pure** (200 positions, W = 1..8, with rollback
schedules and deliberately-wrong rolled-back tokens).  The recurrent snapshot rollback restore is exact.
`n_rs_seq`, `n_outputs_max`, CUDA-graph capture and the chunked-GDN prefill boundary are all **not** the
difference between the two runs (the GDN chunk-size sequence is literally identical).  The one measurable
*structural* plain-vs-MTP difference on this model is that the MTP driver turns on the target's
`embeddings_nextn` export (`common/speculative.cpp:1431`), which changes qwen4exp's last-layer output
gather deferral and shifts the **prefill's last-position logits by a ULP** — a real logits-level
plain≠MTP defect, but on its own it does not flip the replayed greedy tokens, so it is the leading
*partial* cause, not the whole story.  The honest state: the residual is a driver-level (plain-vs-MTP)
divergence that no probe built so far reproduces; the next step is a faithful mini-MTP driver.

## 1. The residual is not a width dependence

New instrument `qsa-item4/mstep.cpp` (multi-step teacher-forced probe):

* pass 1 decodes `N` greedy tokens at width 1, recording, for each step `s`, the hash `H1[s]` of the
  logits that predicted token `T[s]` (context = prompt + `T[0..s-1]`);
* pass 2 replays the *same* token sequence `T` in batches of `W`, comparing every row's logits against
  `H1` (row `j` of the batch at positions `P+k..P+k+W-1` predicts `T[k+j+1]`).

Extras: `RB` = roll back `W-RB` positions after every batch (a spec verify + partial rollback schedule),
`JUNK` = put a different token in the rolled-back rows, `TAIL` = move the prefill's chunked→sequential
GDN boundary, `NEXTN` = `llama_set_embeddings_nextn(ctx, true, false)` on pass-2's context, `RS1`/`RS2` =
separate `n_rs_seq` per pass, `NOM` = `n_outputs_max`.

| config (forced sparse, `-ctk q8_0`, `P=4293`, `N=200`) | result |
|---|---|
| `W=1` vs `W=4` (both `RS=0`) | **0 mismatches** |
| `W=2`, `W=8` (both `RS=0`) | **0 mismatches** |
| `W=4 RB=3 RS=3` (verify + keep 1 + rollback schedule) | **0 mismatches** |
| `W=4 RB=3 RS=3 JUNK=1` (rolled-back rows hold unrelated tokens) | **0 mismatches** |
| `W=4 RB=0/3 JUNK=1 RS=3 NEXTN=1` | **1 mismatch — the prefill row only** (§3) |
| `RS1=0 RS2=2`, `RS1=0 RS2=3 NOM=3`, `W=1` | **0 mismatches** |

So: the target model's forward is **width-pure** across the whole decode/verify band, the recurrent
snapshot rollback restores the sequential state exactly, and **the content of the rolled-back tokens does
not leak** into the post-rollback state.  The single-step probe used on 2026-09-12 (identical at `P=4000`)
was simply not sampling the near-tie; this probe covers all 200 positions of the run.

## 2. What the sharp signature says

Re-measured `n_max` sweep on the `p5000`/q8_0 forced-sparse case (`N=128`):

| `--spec-draft-n-max` | plain | n<sub>max</sub> |
|---|---|---|
| 1 | `a57bc13bbf2a` | **`a57bc13bbf2a` (pure)** |
| 2, 3, 5, 7 | `a57bc13bbf2a` | `3124adfd2b94` (all four **identical**, first diff at char 458) |

and MTP is genuinely active at `n_max 1` (Generation 31.1 t/s vs plain 23.7).  So the divergence is a
**binary toggle at `n_max >= 2`**: once a wider verify exists the whole trajectory moves to the other
side of a near-tie, and every `n_max >= 2` lands on the same side (so the verify width itself does not
change the arithmetic — consistent with §1).

## 3. The one measurable plain-vs-MTP difference: `embeddings_nextn`

`common/speculative.cpp:1431` calls `llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false)` when the
MTP driver starts.  Reproducing that on the probe's pass-2 context:

```
NEXTN=0: prefill logits ad3acaa75d19ddf2
NEXTN=1: prefill logits b624a79f19b1b1f0     <-- differ
```

Every *decode* position stays pure; only the prefill's last-position logits move.  Cause is visible in
`src/models/qwen4exp.cpp`: the MTP export needs a hidden row for **every** token, so

```c
const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;
if (il == n_layer - 1 && inp_out_ids && gather_now) { ...gather cur/inject/res_hc early... }
```

is false in the MTP target, and the last layer's `hc_combine`/`hc_mix`/attention then run on the **full
ubatch** instead of the 1 output row, with the gather deferred to after `t_h_nextn` is taken.  Same math,
different batch width → ULP.  (The deferral is an upstream pattern shared by many models; qwen4exp's
hyperconnection chain is what makes it visible here.)  It is a genuine logits-level violation of the
`plain == draft-mtp` guarantee, but in teacher-forced replay the flipped-ULP prefill logits do **not**
change the generated tokens (`NEXTN=0/1` and `POUT=1..4` all give the same `Thash`), so it does not by
itself explain the char-458 divergence.

## 4. Ruled out

* **`n_rs_seq`** — `RS1=0` vs `RS2=2/3` over 200 positions: pure.  (Also the recorded hashes were taken
  with `RS=from_w`; the value is neutral for the forward.)
* **`n_outputs_max`** (MTP target uses `1 + n_max`) — `NOM=3` vs 1: pure.
* **CUDA graph capture** — the real gate diverges identically with `GGML_CUDA_GRAPH_OPT=0`.
* **Chunked-GDN prefill boundary** — this *is* a real hazard (moving it by 1 token changes the greedy
  text: `TAIL=0/1/2/3` → four different token hashes) and it explains why plain is perturbed by
  `GGML_CUDA_GDN_CHUNKED=0` at **char 49**.  But the *actual* chunked-GDN call sequence is **identical**
  between the plain and MTP runs (an instrumented `gated_delta_net.cu` logged every chunked launch:
  144 calls each, same sizes `{2048, 2044, 207}`, zero diff).  So the boundary is not the plain-vs-MTP
  difference.
* **`GGML_CUDA_GDN_CHUNKED=0` / `GGML_CUDA_DISABLE_FUSION=1` "reconcile"** — confirmed on the current tip
  (`GDN_CHUNKED=0`: `aba1c745cd2d` both sides) but they are **perturbations, not localisers**: with
  `GDN_CHUNKED=0` the *plain* stream itself moves at char 49.  The earlier record's suspicion was right.
* **The fused indexer score / derived cache** — `GGML_CUDA_QSA_INDEXER_SCORE` is a no-op here: with a
  quantized indexer key cache (`--cache-type-k q8_0` is applied to the indexer store too) `idx_key_float`
  is false, so the fused op is bypassed by design and the per-op dequantising chain runs (unchanged in
  the source).

## 5. Disposition

* TODO item 4 stays **open**, re-scoped again: it is a driver-level (plain-vs-MTP) divergence, not a
  kernel-width one; the leading *measurable* cause is the `embeddings_nextn` last-layer gather deferral
  (§3), which is a real defect worth fixing on its own.
* Next step (needs a new instrument): a **faithful mini-MTP driver** — target + draft contexts on the same
  model, `embeddings_nextn` on, the real draft proposals and the driver's rollback — dumping
  `llama_get_logits_ith` per step for both `--spec-type none` and `draft-mtp`, to find the first step
  where the target's logits differ.  Everything cheaper has been exhausted.
* Kill switch for users is unchanged: `LLAMA_QSA_OFF=1` (and the delivery default — dense decode below
  64K — is pure on this prompt).

## 6. Repro / instruments

```sh
cd /home/stew675/ll25/verify
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
clang++ -O2 -std=c++17 -I include -I ggml/include -I src \
  /home/stew675/llama-cpp-rdna-boosts/wip/strix-halo/qsa-item4/mstep.cpp -o /tmp/mstep \
  -Lbuild-rocm/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm/bin
# width purity (expect mismatches=0):
LLAMA_QSA_DENSE_DECODE_UNTIL=0 W=4 RB=3 RS=3 JUNK=1 N=200 CTX=8192 CTK=q8_0 CTV=q8_0 SPLIT=layer NGL=99 \
  /tmp/mstep "$M" /tmp/p5000.txt 4293 2048
# the embeddings_nextn prefill ULP (expect mismatches=1 at pos=4293, only with NEXTN=1):
LLAMA_QSA_DENSE_DECODE_UNTIL=0 W=4 NEXTN=1 N=200 CTX=8192 CTK=q8_0 CTV=q8_0 SPLIT=layer NGL=99 \
  /tmp/mstep "$M" /tmp/p5000.txt 4293 2048
```

The text gate is `qsa-item4`-independent: `wip/kv-quant-purity-followups/tools/` (`textgen.py`), or the
session's `/tmp/g1151-item4/gate.sh` harness (env `TAG`, `KV`, `CTX`, `TXT`, `NG`; sets
`LLAMA_QSA_DENSE_DECODE_UNTIL=0` on request).
