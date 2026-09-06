# Strix Halo (RDNA3.5 / gfx1151) — A-vs-B gap re-derived on the WS3 #2 default-ON build

Date: 2026-09-10 session (continuation). Purpose: equal/surpass the community repo (B) for
PREFILL and TG at EVERY data point (depths 0/12k/32k x pp512..16384 + tg128). This record is the
new baseline after the WS3 #2 ggml fix + `LLAMA_QSA_DENSE_SHORTCUT` default ON (A `1682d32a9`):
A's default now runs dense-below-width attention like B, so depth-0 rows compare regime-matched.
Protocol: llama-bench `-ngl 99 -t 15 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none`,
r3 for A, prompts descending in one process (pp16384 first, warm clock); B same-box
`c7af5c6c2` — see the B@depth protocol note below. Model: IQ4_XS UD Qwen3.8-Flash-Next (87.24 GiB).
Raw: /tmp/gateA/gap-{A,B}-*.log.

## Depth-0 (same session)

| test | A (r3) | B (r3) | gap B/A |
|------|-------:|-------:|--------:|
| pp512 | 408.55 ± 0.49 | 647.11 ± 22.06 | 1.58x |
| pp1024 | 416.15 ± 10.48 | 733.41 ± 15.25 | 1.76x |
| pp2048 | 400.14 ± 0.74 | 770.79 ± 14.97 | 1.93x |
| pp4096 | 487.30 ± 0.87 | 678.53 ± 99.57 (noisy rep) | 1.39x |
| pp8192 | 543.55 ± 0.55 | 678.66 ± 0.96 | 1.25x |
| pp16384 | 575.12 ± 1.45 | 599.86 ± 1.05 | 1.04x |
| tg128 | 25.07 ± 0.01 | 25.96 ± 0.06 | A behind 3.5% |

## Depth-12k (A r3, B r1 — see protocol note)

| test | A | B | gap B/A |
|------|-----:|-----:|--------:|
| pp512 | 350.31 | 494.20 | 1.41x |
| pp1024 | 358.75 | 514.55 | 1.43x |
| pp2048 | 345.70 | 534.26 | 1.55x |
| pp4096 | 436.06 | 509.10 (r3, completed before B crashed) | 1.17x |
| tg128 | 22.46 | 22.04 | A 1.02x (wins) |

## Depth-32k (A r3, B r1)

| test | A | B | gap |
|------|-----:|-----:|-----:|
| pp512 | 317.70 ± 8.76 | 300.86 | A 1.06x (wins) |
| pp1024 | 329.33 ± 18.75 | 313.41 | A 1.05x (wins) |
| pp2048 | 334.01 ± 0.75 | 296.78 | A 1.13x (wins) |
| tg128 | 20.11 ± 0.03 | 18.58 | A 1.08x (wins) |

## Reading

- Deficits are concentrated in SHALLOW pp (depth-0 pp512..8192, gap 1.25-1.93x, worst pp2048
  1.93x; pp16384 ~1.04x) and depth-12k pp (1.17-1.55x). tg@0 is A behind ~3.5% (25.07 vs
  25.96); tg at depth is already A >= B (22.46 vs 22.04 @12k, 20.11 vs 18.58 @32k).
- Per-ubatch delta analysis (single-ubatch rows, ms/ubatch): pp512 0.46, pp1024 1.06, pp2048
  2.46 — a cost that GROWS with the ubatch token count (a per-token component ~1.2 us/token at
  pp2048), NOT a flat per-ubatch fixed cost. At pp16384 the same first ubatch exists but the
  mean is pulled to parity by A's sparse attention beating B's dense at n_kv 4096-16384 (B's own
  rate collapses with n_kv: pp2048@0 771 -> @12k 534 -> @32k 297, while A stays ~flat 400 ->
  346 -> 334). So the shallow-row deficit is a per-token cost A pays that B does not — matching
  the WS1 attribution's unfused MoE routing/reduction + elementwise/norm/cpy tail (B fuses into
  hc_* + weighted-expert-sum ops; A's tail unfused outside the WS4 hyperconn coverage).
- B same-box runs 10-30% slower than the community PR#18 table (e.g. pp2048@0 771 vs their 844;
  pp2048@32k 297 vs 386): the community table is NOT a valid same-box reference; use same-box B.

## B@depth protocol note (this session)

B aborts reproducibly (exit 134) in `argsort_f32_i32_cuda_cub` (top-k, ggml-cuda.cu:108) when
llama-bench's depth-12k/32k tests run at `-r 3` (warmup + state save/restore via
llama_state_seq_get/set_data between reps). Depth-0 (no depth state machinery) and `-r 1` at
depth both run fine. B is untouched (reference only). Recommendation for future same-box B@depth
rows: `-r 1` (or investigate the state-restore x cub interaction on B's side later if a B-level
comparison needs r3). A is unaffected (deterministic count/scan indexer, no cub argsort).

## Next attack (shallow-row per-token tail)

Per the ledger, the largest remaining component at pp512-4096@0 is the MoE routing/reduction
tail: port B's `ggml_cuda_op_weighted_expert_sum` / `ggml_cuda_mul_mat_id_weighted_rdna3_5`
(IQ4_NL down-proj fused with the n_used=10 weighted sum) graph fusions into A's ggml-cuda.cu,
then re-measure pp512-2048@0. Verify A's graph pattern matches B's before porting. A secondary
target for tg@0: A 25.07 vs B 25.96 (+3.5% — some of this closed already by the shortcut tg win;
decode levers tracked in beta/qwen4exp README).
