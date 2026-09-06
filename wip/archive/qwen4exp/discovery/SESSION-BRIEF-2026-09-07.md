# Strix Halo session brief — 2026-09-06 end state (handoff for the next session)

CONTINUE from this file + the plan (~/make-strix-halo-faster.md) + the evidence
trail (wip/strix-halo/notes-ws1-survey.md). Read all three first.

## Commits / tree state

- ~/llama.cpp (branch qwen4exp, tip 248e47704, base dca0526a8 + THIS commit):
  WS4 prefill hyperconn fusions DEFAULT ON (bit-exact), deterministic
  indexer-topk gather, kv-cache stale-cell zeroing. Build: ./build-rocm via
  ~/bin/build-llama-rocm-714 (canonical; no -mllvm flag). All temp debug removed.
- ~/llama-cpp-rdna-boosts: tip 4d54c74 (wip/strix-halo notes committed).
  wip/strix-halo/ = experimental notes only, NOT delivery.

## What is DONE (2026-09-05/06)

1. WS1 gap reproduced/pinned: depth-0 pp A vs B (IQ4_XS, ub 2048): 2.0x@512,
   2.25x@2048, 1.8x@4k, 1.5x@8k, 1.2x@16k; our decode wins at depth (A +8% @32k).
2. WS2 attribution: ~50-60% = per-ubatch elementwise/norm tail (B fuses the
   hyperconn chains at prefill; A didn't), ~20-25% = expert i-quant mmq vs B's
   routed-compact, ~10% shallow QSA vs dense; rocBLAS parity; NOT upstream drift.
3. WS4 port IMPLEMENTED and now DEFAULT ON and BIT-EXACT:
   - hyperconn.cu/.cuh (kernels + host) and two try_fuse matcher blocks in
     ggml-cuda.cu. Env opt-outs: GGML_CUDA_DISABLE_HC_FUSION / _HC_MIX / _HC_COMB.
   - Both-on divergence FIXED (two stacked bugs): (a) the comb block was nested
     inside if(hc_mix_on) so 'comb-only' never fired; (b) the comb matcher read
     block_out from the REPEAT INPUT (b->src[0]) whose buffer the allocator
     reused for hc_inject (the standalone REPEAT ran before the fused window) -
     now reads the live REPEAT output (mul operand), block_out_hc flag indexes
     its [n_embd,hc,T] rows. Fused == unfused BIT-IDENTICAL (llama_decode harness,
     pp + 40 greedy steps, per-step logits hashed) at 6/140/500/1400-token
     prompts, deterministic across fresh processes; llama-cli text on == off.
   - Perf (single-run warm-clock, this build): pp2048 342.8 vs 326.7 (+4.9%),
     pp4096 430.7 vs 406.7 (+5.9%), pp16384 518.0 vs 487.5 (+6.3%).
4. Real determinism fix (the original llama-cli flakiness): indexer-topk.cu's
   atomicAdd-based gather scrambled the QSA list ORDER run-to-run (>1 block/row);
   ulp-level QSA softmax drift got amplified by the f16-state recurrences into
   llama-cli-visible nondeterminism (5/5 fresh runs differed, fusions off too).
   Replaced with a deterministic ascending-column count/scan/write. Fresh-process
   harness + llama-cli are now bit-identical. This bug was in OUR qwen4exp
   feature code (the custom radix top-k), not upstream and not B (B uses a
   CUB-argsort path).
5. kv-cache stale-cell zeroing ported from B (aad5adb08) - fixes cross-request
   cell reuse (masked-out rows leaking stale V sign on RDNA WMMA f16).

## Provenance answers (for the record)

- WS4 both-on bug = OUR port's adaptation/splicing errors (B's reference matcher
  passes the mul operand directly; the A-adaptation's REPEAT unwrap + the brace
  nesting were introduced in the port). Not upstream, not B.
- Original llama-cli nondeterminism = OUR pre-existing qwen4exp indexer-topk
  kernel. Not upstream (no QSA/indexer upstream), not B.
- KV stale cells = a FIX B had and A lacked (now ported).

## NEXT SESSION (in order)

1. WS4 gates on the final build (fusion default ON):
   - clean warm-clock r3 pp ladder depth 0 (pp512..16384, descending, per the
     plan protocol) default vs GGML_CUDA_DISABLE_HC_FUSION=1;
   - depth 12k/32k pp rows + tg@depth regression (must not regress; fusion is
     prefill-only via the ne[1]==1 guard but verify);
   - memory-stability ladder (-r3 through 32k, no crash/leak).
2. Dated bench record benchmarks/2026-09-06-strix-halo-*.md (mirror the
   block-13 record format) + delivery routing per AGENTS (qwen4exp tree change ->
   beta/qwen4exp patch amendment or new patch; NEVER push from ~/llama.cpp);
   update TODO.md.
3. Re-derive the remaining A-vs-B pp gap (now ~348->~360s range expected at
   pp2048 with the fusion) and re-run the depth-0 ladder vs B; then WS3 #2
   (dense shortcut below the selection width in build_layer_attn) and #3
   (routed-compact MoE mmq for i-quants, RDNA4-gated) as follow-ons; gfx1201
   validation through the normal delivery flow. WS6 re-base still NOT indicated.

## Hygiene (all from AGENTS.md + plan §9)

No parallel/background benches (also: back-to-back llama_decode/llama-cli runs
of this 87GB model intermittently produce EMPTY output in loops - run standalone
or spaced, verify non-empty output before diffing). Warm page cache (dd shards)
before benching; first test after process start = long pp (cold GPU clock);
depths 0/12k/32k only for now (64k/128k deferred); ~116 GB effective VRAM,
~420 GB disk; one server (port 8033); push policy: delivery repo only.
