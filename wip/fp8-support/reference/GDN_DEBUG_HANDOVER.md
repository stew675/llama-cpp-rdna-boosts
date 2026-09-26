# GDN CHUNKED DEBUGGING HANDOVER (session 2026-08-06, mid-debug)

## STATUS 2026-08-06 (LATE): BUG FOUND AND FIXED - PPL 359971 -> 6.2352

ROOT CAUSE: the tiled phase A closure (attn = I+A+A^2+A^3 via 16x16 block
products) is mathematically WRONG for a 64x64 strictly-lower A. The claim
"A^4 = 0" only holds if the diagonal 16x16 blocks are zero, but they are NOT:
a strictly-lower matrix has nonzero sub-diagonal entries inside every diagonal
block, so the nilpotency index is 64 (not 4). Missing terms (e.g. A(2,2)*A(2,0)
in A^2(2,0), and all A^3 terms that stay inside diagonal blocks) are zero only
when A is tiny.

Why the model exposed it and tests/debug did not:
- model: L2-normalized k (gram~1), beta up to 0.94, gates near 0 (gate[0] ~
  -0.0007) -> A is a DENSE strictly-lower matrix with |A| up to 0.81; the
  closure error propagates into k_cumsum/k_cumdecay -> garbage output (PPL
  359971).
- tests use gates -20..-1e-4 / -2..-0.5 with small beta -> A tiny -> the
  missing block terms ~0 -> tests pass.
- the T=2 warmup dump validated because a 2-token chunk has A^2=0 (never
  exercises the block products).
- the handover's "python closure validation" used strong-decay data (A^4 ~
  4e-10) -> validated the wrong claim.

FIX (in gated_delta_net.cu, uncommitted): replaced the closure with EXACT 16x16
block forward substitution (still 256 threads, keeps the tiling):
  D_r = (I - A(r,r))^{-1} via row substitution D_r(i,:) = e_i + sum_{k<i} B(i,k) D_r(k,:)
  X(r,c) = D_r * sum_{m=c}^{r-1} A(r,m) * X(m,c)   for c < r
(plus __shared__ float s_acc[16][16]). Verified in numpy against np.linalg.inv
on the model's dumped data (maxdiff ~1e-7) before building.

VERIFICATION (all on stewfp8-ow.gguf, /tmp/corpus_pride.txt, -c 512 -b 2048 -n 4):
- chunked (default, graphs ON): PPL 6.2352  (sequential baseline 6.2572)
- 47/47 GATED_DELTA_NET backend tests pass (incl. multi-seq cases)
- layer-consistent dump (see below) now fully matches numpy: phase A scratch
  k_cumsum 2.3e-6 / k_cumdecay 7e-8 / attn_causal 2.8e-6, phase B dst 1.1e-7,
  state_out 2.8e-7, sequential cross-check 5.6e-8.
- generation works: llama-cli --single-turn --no-conversation gives coherent
  text (needs those flags in this fork, see HANDOVER.md).

ALSO LEARNED THIS SESSION:
- GGML_HIP_GRAPHS=0 is a CMAKE OPTION, NOT a runtime env var. The runtime
  disable is GGML_CUDA_DISABLE_GRAPHS=1. All previous "graphs off" runs were
  actually graphs ON. CUDA graphs are NOT the cause of the garbage anyway
  (PPL 359971 reproduces with graphs properly disabled).
- the old GDN_DUMP was layer-misaligned because the context streams are
  cudaStreamNonBlocking and the blocking cudaMemcpy (legacy default stream)
  does not wait for them. Fixed dump: all copies via cudaMemcpyAsync/
  cudaMemcpy2DAsync on `stream` + one cudaStreamSynchronize AFTER phase B
  launch, then write files. First gdn call in perplexity is T=2 (warmup); the
  real calls are T=128 n_seqs=4 n_chunks=2 (batch 2048 split into 4 seqs).

REMAINING (perf): chunked pp512 = 5335 t/s vs sequential 5878 t/s (~9%
slower). gdn kernels per pp512: phase A 14.1ms (293us x 48) + phase B 33.9ms
(705us x 48) = 48ms vs sequential 31.8ms (120us x 264). Phase B (128-thread
original) is the bottleneck; the 4-slice 256-thread phase B rewrite (was
15.5ms) is LOST (never committed, no stash) and must be re-derived + re-
validated against the sequential kernel before it can land. The dump
infrastructure (GDN_DUMP + /tmp/validate_gdn_dump.py) is exactly what to use
for that re-validation.

---

## ORIGINAL STATUS: THE CHUNKED DELTA-NET IS CORRECT IN ALL TESTS/DEBUG BUT PRODUCES GARBAGE ON THE REAL MODEL (PPL 359971 vs sequential 6.2572)

DO NOT continue from memory. Read this whole file, then re-establish state from the sections below.

---

## 1. What the code is supposed to do

Chunked gated delta rule in `ggml/src/ggml-cuda/gated_delta_net.cu`:
- phase A `gdn_chunk_prepare` (256 threads/CTA, grid (H*n_seqs, n_chunks)): per (head, seq, chunk of 64):
  gates cumsum (Hillis-Steele) -> s_decay; stage k into smem s_k[64][129]; gram = k@k^T (4x4 tiles);
  w[i][j] = -beta_i*exp(decay_i-decay_j)*gram (strictly lower, diag/upper zeroed); closure attn = I+A+A^2+A^3
  via 16x16 block products (FIXED to accumulate a20+a30+a31, see bug list); k_cumsum = attn@(beta*v);
  k_cumdecay = attn@(beta*k*exp(decay)); attn_causal = L.*(q@k^T) strictly causal (j<=i); delta, P_last.
- phase B `gdn_chunk_state` (128 threads, grid H*n_seqs*16): per (head, seq, V-slice of 8): sequential over chunks:
  v_new = k_cumsum - k_cumdecay@S; o = exp(decay)*(q@S) + attn_causal@v_new; S = P_last*S + (k*delta)@v_new.
- Dispatch (default ON since a few builds ago): `!KDA && !keep_rs_t && S_v==128 && n_tokens>1 &&
  (state_slot_stride==0 || state_slot_stride==S_v*S_v*H*n_seqs)`.
  Disable for sequential comparison: `GGML_CUDA_GDN_CHUNKED=0` (env, default-on: env absent OR nonzero -> chunked).
- Scratch (pool): k_cumsum [H*n_seqs*n_chunks*64*128] + k_cumdecay [same] + attn_causal [H*n_seqs*n_chunks*64*64]
  + decay [H*n_seqs*n_chunks*64] + delta [same] + P_last [H*n_seqs*n_chunks]. FLAT OFFSETS (per-HS contiguous
  within each array, NOT interleaved per-HS):
  k_cumsum=0, k_cumdecay=2097152 (for 32 heads x 8 chunks), attn_causal=4194304, decay=5242880,
  delta=5259264, P_last=5275648, total=5275904 (for H=32, n_chunks=8).
  The phase A indexes each array with (hsc+chunk) where hsc=hs*n_chunks (this is CORRECT).

## 2. Verified-correct building blocks (do not re-debug these)

- All 47/47 GATED_DELTA_NET backend-ops tests PASS (I added multi-seq cases (32,128,64,2) and (32,128,512,4)
  to tests/test-backend-ops.cpp).
- Standalone debug `/tmp/gdn_debug.cpp` (H=32, S=128, T=64 or 512, CPU sequential reference) passes with
  maxerr ~1.27e-4 for: strong decay (gates -20..-1e-4), weak decay (-2..-0.5), zero state, random state,
  T=64 and T=512, graphs ON and OFF.
- In-kernel DBG prints (env GDN_DBG=1, passed as kernel arg `dbg`) show for hs=0 chunk=0 of the model:
  - s_decay (smem) = correct cumsum of gates read from the tensor (e.g. -2.17, -4.19, -6.27...; d31~-54, d63~-108).
  - decay_scratch write = s_decay, at flat offset 5242880+tid (readback in kernel matches).
  - M2 (k_cumsum) computes accv00 = b0*v00 exactly.
  - closure values: w[32][0]=0, attn[32][0]=0 (blocks ~0 because decay per chunk is strong: exp(d32-d0)=exp(-52)=0;
    the closure is effectively I+A for this model).
- The closure block-LU math was validated in python against forward substitution (identical to ~1e-10, A^4~4e-10).

## 3. THE CRITICAL TRAP (read twice)

The GDN_DUMP instrumentation (`env GDN_DUMP=1`, dumps scratch + inputs to /tmp/gdn_scratch_model.bin +
/tmp/gdn_inputs_model.bin from inside the launch function) is LAYER-MISALIGNED: the CPU comparison
(computed from the dumped inputs) does NOT match the dumped scratch, e.g.:

- dumped inputs gate[0] = -1.72853 (some layer)
- dumped scratch decay[0] = -0.20981 (a DIFFERENT layer)
- the same-run DBG shows BOTH gate[0]=-0.20981 (one layer) and gate[0]=-1.72853 (next layer) prints.

So the dump-based comparisons ("k_cumsum maxdiff 3.97", "decay maxdiff 54.8") are MEANINGLESS until the
dump is made layer-consistent. The dump runs per layer (file overwritten each time); something about the
graph-capture / async-launch / printf-flush ordering causes the scratch file and the input file to come from
different layers even with GGML_HIP_GRAPHS=0. The phase A's in-kernel readbacks (DBG) are the only trustworthy
evidence, and they show the phase A computing CORRECT intermediate values.

NEXT STEP: fix the dump to be layer-consistent. Simplest: print the layer's gate[0] to stderr alongside each
file write (same fprintf), or dump the scratch AND inputs to a per-layer filename (e.g. append the gate[0] value),
or disable the dump except for a specific layer (compare the DBG's gate[0] with the file's gate[0]).

## 4. Bugs found and fixed THIS session (in gated_delta_net.cu, uncommitted)

1. (EARLIER session, committed) decay_scratch write raced across heads: tid 0..127 wrote into 64-float per-head
   regions -> fixed with `if (tid < GDN_CHUNK)`.
2. (THIS session) closure block products p0/p12/p3 OVERWROTE the original A[2][0]/A[3][0]/A[3][1] blocks instead
   of accumulating -> fixed by saving a20/a30/a31 in registers before the product writes, computing a3 from the
   isolated A^2[2][0], then writing a20+p0, a30+p12+a3, a31+p3. Effect on this model is ~0 (blocks ~0) but it is
   a real math bug for weaker decay.
3. (THIS session) `getenv()` cannot be called inside a __global__ kernel on HIP -> pass debug flag as kernel arg.
4. Debugging lesson: cudaMemcpy/cudaStreamSynchronize/cudaEvent* inside the launch during CUDA graph capture are
   unreliable ("operation not permitted when stream is capturing" for sync/event; memcpy reads stale memory).

## 5. Current PPL/generation state (model stewfp8-ow.gguf, corpus /tmp/corpus_pride.txt, -c 512 -b 2048 -n 4)

- sequential (GGML_CUDA_GDN_CHUNKED=0): PPL 6.2572 (baseline, good).
- chunked (default): PPL 359971.4813 (garbage).
- Earlier during session: 18057 (2-slice phase B era), 16502 (HEAD 128-thread phase B + tiled phase A, broken
  closure), 359971 (same + fixed closure). All garbage; numbers changed with each phase-B/closure change, so the
  bug is REAL and in the chunked path, but NOT reproduced by tests/debug.

## 6. What changed since the last GOOD PPL (6.2996, which was: tiled phase A + ORIGINAL 128-thread phase B)

Sequence: phase A tiling (build52 era, PPL 6.2996 verified) -> 2-slice 256-thread phase B -> 4-slice 256-thread
phase B (bench 6059) -> chunked default-on -> PPL garbage from here on -> dumped/debugged -> closure fix.
NOTE: the 6.2996 was measured BEFORE the phase B rewrites; the phase B rewrites may contain the bug, OR the
tiled phase A has a bug that the debug doesn't catch (tests/debug use random q/k/v and gates -20..-1e-4 or -2..-0.5;
the model's exact data may expose something else).

## 7. Outstanding hypotheses for the model-only failure

- The phase A writes are correct in-kernel but something OVERWRITES the scratch between phase A and phase B
  (e.g. another op's scratch on the same pool buffer, or the phase A of the next layer racing the phase B of the
  current layer in the graph). CHECK: the launch frees `scratch` (ggml_cuda_pool_alloc destructor) at the end of
  the launch; the phase B launch is a SEPARATE pool alloc that may REUSE the same buffer - verify the phase B's
  alloc does not alias the phase A's scratch while phase B still needs it. THIS IS THE STRONGEST HYPOTHESIS:
  the phase A's scratch is freed at the end of the phase-A block before the phase B launch block allocates its own
  (possibly aliasing) scratch.
- The model's actual data (fp8-dequantized q/k/v, softplus gates, L2-normalized q/k) exposes an edge case:
  e.g. s_bp = beta*exp(decay) where decay can be near 0 (gate ~ -0.0009) making L ~ 1 and w entries non-negligible
  for FAR pairs in chunks where decay span is small; the block products then matter and the closure (even fixed)
  may be wrong. VALIDATE: python-compute the chunked path on the model's dumped inputs (once the dump is
  layer-consistent) and compare against the sequential reference.
- Phase B multi-chunk chain state accumulation differs from sequential in a way the tests' tolerance hides.
  The debug's maxerr 1.27e-4 at T=512 is suspiciously constant across gate ranges - investigate whether the debug
  reference itself is subtly wrong (compare debug CPU reference against the sequential kernel's output at
  GGML_CUDA_GDN_CHUNKED=0 with identical inputs).

## 8. How to reproduce and debug

Build: `/tmp/llama-hip-full` (cmake, GGML_HIP=ON, gfx1201). Rebuild targets: `cmake --build /tmp/llama-hip-full -j16 --target llama-bench llama-cli llama-perplexity test-backend-ops`.
Model: `/llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf`.
PPL: `./llama-perplexity -m <model> -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0` (chunked by default; `GGML_CUDA_GDN_CHUNKED=0` for sequential).
Tests: `test-backend-ops test -b ROCm0 -o GATED_DELTA_NET` (47 cases now).
Debug prints: `env GDN_DBG=1 ./llama-bench -m <model> -p 512 -n 1 -r 1 -dev ROCm0 | grep "DBG"`.
Standalone: `/tmp/gdn_debug.cpp` (edit H/S/T/gate ranges/state, rebuild with the g++ command in its header comment).
Perf profile: `rocprofv3 -r -d /tmp/rocp_x -f csv -- ./llama-bench -m <model> -p 512 -n 8 -r 1 -dev ROCm0`, then parse the kernel_trace.csv.
Bench: chunked pp512 ~6008-6059 (was 5925 sequential; the chunked currently wins on speed but is WRONG on output).

## 9. Debug instrumentation currently in the code (remove before any commit)

- GDN_DUMP block in the launch (scratch + inputs dump, layer-misaligned, see section 3).
- GDN_DBG prints in phase A (s_decay, M2 accv, closure values) gated by kernel arg `dbg` (passed from
  getenv("GDN_DBG") in the launch). Plus a stale "DBG gram" print placed BEFORE the M1 (reads uninitialized smem -
  ignore its values).
- `#include <cstdlib>` added at the top.
- tests/test-backend-ops.cpp: added (32,128,64,2) and (32,128,512,4) cases (KEEP these - they are useful).

## 10. Uncommitted changes in the working tree (git status shows gated_delta_net.cu + tests)

Last commit: bb31b45d5 (chunked kernel opt-in, correct but slow). The current working tree has: phase A tiling
(256 threads, block-LU closure), phase B 2-slice/4-slice experiments (currently HEAD's 128-thread phase B is in
the file after the bisect), chunked default-ON, closure fix, dump/dbg instrumentation. Decide what to keep
(phase A tiling + closure fix + default-on are the intended direction; phase B rewrites need re-validation;
dump/dbg must be removed).

## 11. Performance context (RESOLVED 2026-08-06: chunked now beats sequential)

- Phase A tiling took gdn_chunk_prepare from 76ms -> 12-14ms per 48 launches.
- Phase B was rewritten (128 threads/16 slices -> 256 threads/4 slices, thread-per-column
  with warp-broadcast loads, see PERF_HANDOVER.md section 12): 33.8ms -> 12.8ms.
- Chunked pp512 = 6068-6072 vs sequential 5878 (chunked WINS). PPL 6.2426, 47/47 tests.
- The lost 4-slice phase B (15.5ms) is superseded: the new kernel (12.8ms) is simpler
  and validated. The dump infrastructure for re-validation lives in /tmp/validate_gdn_dump.py.
- Decode (tg64 ~89-90) is untouched (sequential path for n_tokens=1).
- Next perf target: phase A (14.2ms, closure-barrier-bound at 1 CTA/CU).

## 12. Files

- ggml/src/ggml-cuda/gated_delta_net.cu (main work area)
- tests/test-backend-ops.cpp (added multi-seq cases)
- /tmp/gdn_debug.cpp (standalone debug, uses the built libs)
- /tmp/gdn_scratch_model.bin, /tmp/gdn_inputs_model.bin (dumps - layer-misaligned, see section 3)
- /tmp/validate_chunked.py (original python chunked-math validation, from earlier session)
- PERF_HANDOVER.md (session logs; section 11 covers the chunked kernel design)
