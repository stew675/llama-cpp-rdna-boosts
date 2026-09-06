# Strix Halo (RDNA3.5 / gfx1151) — prefill root-cause investigation (2026-09-05)

Question (user framing): A and B run the SAME model on the SAME GPU; the tokens must be ingested
and processed either way. At low depths (single-ubatch pp512-4096) A is far behind B in
llama-bench (pp2048 400 vs 771 = 1.93x; pp512 409 vs 647, pp1024 416 vs 733). Is that real A
slowness or a measurement artifact — and either way, WHERE does A's extra time go, and why is
B's identical-work path faster? This record is the investigation state; fix work follows.

## Environment

A (qwen4exp `1682d32a9` + WS3#4 weighted-down `a18e24f97`, default = dense-shortcut ON) vs B
(`c7af5c6c2`), same box/model, IQ4_XS UD Qwen3.8-Flash-Next 87.24 GiB, ub/b 2048, f16 KV, FA on.
Bench protocol = llama-bench `-ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16
--load-mode none`. Real path = llama-cli (sync per decode). Same-session numbers below.

## Measured facts (all this session)

| pp2048 measurement | A | B | A/B |
|---|---|---|---|
| llama-bench (warm-clock r3, same session) | ~400 t/s (5.1-5.6 s/rep) | ~771 t/s (2.66 s/rep) | 1.93x |
| llama-cli, identical flags (`--load-mode none`) | 494.3 t/s | — | — |
| llama-cli, normal load | 494.8 t/s | 653.0 t/s | 1.32x |
| rocprof GPU-busy per pp2048 decode (pair) | 2.94 s | 2.41 s | 1.22x |
| A decode split structure (env-instrumented, reverted) | split0 CPU 4 nodes (`model.input_embed`) = **2700 ms**; split1 GPU 7201 nodes submit 10-220 ms | host submit ~0.25 s/ub | — |

Per-ubatch decomposition of A's pp2048 (one 2048-token ubatch per decode):

- GPU kernel busy ~2.9 s vs B ~2.4 s (+0.5 s: kernel deltas below).
- A CPU input-embedding (PLE get_rows, split0) ~2.7 s under llama-bench decode — ABSENT on the
  llama-cli path (A cli == 494 both load modes). B pays ~0 (B host submit ~0.25 s/ub total).
- llama-bench rep wall ~5.6 s ≈ CPU-embed 2.7 (GPU idle, host serialized) + GPU 2.9.

## Corrected framing (per user)

The llama-bench delta is NOT "B games the benchmark": B's embedding/ingest path is fast in
llama-bench AND llama-cli. A's input-embedding path is genuinely slow in the configuration
llama-bench produces (2.7 s for a 2048-row PLE lookup — should be tens of ms), and A is still
~1.3x behind B at pp2048 on the real path even when that 2.7 s does not fire. Treat the 2.7 s as
an A-side defect to find and fix (real workloads can hit the same path), not a protocol excuse.
Whether B optimized the same path (PLE placement / get_rows / host pipeline) is an open diff.

## Open root-cause: A's CPU input-embedding 2.7 s (llama-bench decode only)

- Why CPU at all under llama-bench but GPU/CPU-fast under llama-cli with the same flags?
  `--load-mode none` measured identical in llama-cli (494.3 vs 494.8) -> either llama-cli
  ignores the flag (verify) or the placement differs by ctx/model-construction, not load mode.
- Hypotheses for 2.7 s / 2048 rows (~1.3 ms/row — absurd for a 160-elem IQ4_NL row):
  (a) host-mmap PLE row path with per-row overhead (managed n-gram reader / locking),
  (b) CPU IQ4_NL get_rows implementation far from optimal for this shape/threadpool,
  (c) a per-decode full-table or O(table-metadata) touch (28.8 GB table),
  (d) llama-bench random-token decode (ids over the full 320M vocab) vs cli text ids hitting a
  pathological row-index path.
- Method: profile the CPU get_rows op (CPU-side timing; check threadpool = 15); diff A vs B
  qwen4exp input-embedding/PLE handling (A = per-layer PLE + managed lazy reader machinery; B =
  plain load?). B's per-ubatch host submit is ~0.25 s — diff the host pipeline too.

## Real kernel deltas at pp2048 (rocprof A vs B, ~0.5 s/pass GPU) — fix targets in order

1. concat: A `concat_non_cont` 0.41 s/111 calls vs B `concat_transposed_src1_dim0` 0.13 s/111
   (3x; B has a specialized transposed-src1 concat). PORT B's kernel.
2. swiglu-input quantize: B fuses `quantize_mmq_q8_1_swiglu` (141 calls, 0.105 s); A runs
   separate swiglu + quantize (A quantize total 0.50 s/1608 calls vs B 0.29 s/1326 + 0.105).
   PORT B's fused quantize.
3. Q8_0 gate/up mmq J128: A 1.56 s vs B 1.38 s on the SAME 1182 calls (+13%) — same kernel;
   the feeding differs (quantize/input prep) — investigate.
4. A-only `mm_ids_helper<10>` 423 calls / 0.31 s launch overhead per ubatch — see if B's
   equivalent is fused/cheaper.
5. Host per-ubatch submit (A more than B's ~0.25 s) — re-measure after the embed fix.
   Near-parity already: routed-compact expert mmq (IQ4_NL/IQ3_S), rocBLAS GEMMs, hc_combine/mix
   (A +17% on hc_combine), dense FA (+18% A).

## Why the low-depth rows are the worst (and the deep rows are not)

pp512/1024/2048 = ONE ubatch per decode: the CPU-embed cost (if it fires) + the per-ubatch host
deltas are paid per decode and NOT amortized, and B's dense-attention cost at these depths is
still small. At pp8192/16384 (multi-ubatch) the embed does not repeat per ubatch and A's sparse
attention beats B's growing dense cost -> gaps 1.25x/1.04x. At depth 12k/32k A wins TG and
approaches/wins PP. So "massive distance at low depths" == the single-ubatch rows == where the
ingest defect + per-ubatch deltas concentrate.

## Plan (next session, in order)

1. ROOT-CAUSE + FIX the CPU input_embed path (2.7 s -> <50 ms): confirm whether llama-cli honors
   --load-mode; find why llama-bench's decode routes the PLE embedding to a CPU split; profile
   the CPU get_rows; diff A/B PLE placement + input-embedding build. Expect this to recover
   most of the pp512/1024/2048 llama-bench delta (A's warm GPU+host pp2048 would be ~2.9-3.3 s
   -> ~620-700 t/s vs B 771).
2. PORT B's transposed concat + swiglu-input quantize kernels (bit-exact/coherence check per the
   WS4 recipe, then same-session A/B).
3. Investigate the Q8_0 gate/up mmq feed delta (+13%).
4. Re-establish the WARM real-path (llama-cli after a warmup decode) low-depth ladder A vs B for
   an honest baseline at every data point (the earlier cli numbers were cold-clock).
5. Re-derive the full matrix on the new default after (1)-(3); update records.

Raw evidence: /tmp/gateA/bnd-*.log (split instrumentation), ub-*.log (decode walls),
cli-pp2048-{A,B}*.txt, /tmp/prof/pp{A,B}_results.db (+ rocprof-sum.py). Instrumentation was
reverted; tree clean at a18e24f97 (includes the decode weighted-down port).
