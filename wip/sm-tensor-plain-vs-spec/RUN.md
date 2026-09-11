# Running the plain-vs-spec / batch-width gates

Builds (GFX1201, ROCm 7.14 at `/opt/rocm-7.14-gfx1201`):

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
CANON=/tmp/canon-llama                 # canonical fork, branch rdna-boosts
cmake --build $CANON/build-base --target llama-cli llama-bench test-backend-ops -j 16
# if /tmp is gone: clone ggml-org/llama.cpp, checkout 9113cc188, run
#   scripts/apply-all.sh .        -> branch rdna-boosts, tree 29714ad1f
#   then cmake configure/build (see patches/README.md for the flags)
```

Models:

```
27B dense : /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
4B        : /home/stew675/Qwen3.5-4B-Q8_0.gguf
MoE       : /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

## 1. Bit-identity probe (the primary instrument)

`logits-dump-singlewidth.cpp` — ONE decode width per process, prints the
token-0 logit hash, **no** `cb_eval` (the callback changes MoE numerics, so it
is not a valid instrument; `CB=0` disables it explicitly).  Prefills P=256
tokens then decodes a batch of W, printing `[L] W=<w> logits0_hash=...`.

```sh
clang++ -O2 -std=c++17 -I $CANON/include -I $CANON/ggml/include \
  logits-dump-singlewidth.cpp -o /tmp/lw-dump2 \
  -L $CANON/build-base/bin -lllama -lggml -lggml-base -Wl,-rpath,$CANON/build-base/bin

M27=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
# 2-GPU tensor split, MTP-faithful K (= W)
for w in 1 3 5; do
  W=$w CB=0 SPLIT=tensor RS=from_w NGL=99 HIP_VISIBLE_DEVICES=0,1 \
    timeout 900 /tmp/lw-dump2 $M27 wip/sm-tensor-plain-vs-spec/p0long.txt 256 512 2>/dev/null | grep '^\[L\]'
done   # all three hashes must be EQUAL
```

Env: `W` width, `CB=0` no callback, `SPLIT=none|tensor|layer|row`,
`RS=0|<n>|from_w` (`n_rs_seq`; `from_w` = W-1, the MTP case), `FA=0|1`,
`NGL`.  `RS=0` measures the pure batch width; `RS=from_w` measures the
K-dependence.

`logits-width-tensor.cpp` is the older 3-contexts-in-one-process version —
**its W=1 numbers can be a multi-context artefact; prefer the single-width
tool.**

## 2. Text gate (the acceptance the maintainer cares about)

```sh
BIN=$CANON/build-base/bin/llama-cli
P0="Explain in detail how a hash map works internally, including collision handling, load factor, and resizing. Then write a complete implementation in Python."
for spec in "none:none" "draft-mtp --spec-draft-n-max 1:n1" \
            "draft-mtp --spec-draft-n-max 2:n2" "draft-mtp --spec-draft-n-max 4:n4"; do
  a="${spec%%:*}"; t="${spec##*:}"
  HIP_VISIBLE_DEVICES=0,1 $BIN -m $M27 -ngl 99 -sm tensor -ts 1/1 -c 8192 \
    -ctk f16 -ctv f16 -fa auto -p "$P0" -n 300 --seed 42 --temp 0 --top-k 1 \
    --no-display-prompt --single-turn --no-warmup --spec-type $a > /tmp/g-$t.log 2>&1
done
python3 extract.py /tmp/g-none.log /tmp/g-n1.log /tmp/g-n2.log /tmp/g-n4.log
# all four must print the SAME sha1
```

`--single-turn` is mandatory or `llama-cli` hangs.  Also worth running
`-sm layer` and 1 GPU (`HIP_VISIBLE_DEVICES=0`, no `-sm tensor`).

## 3. Other gates

- `test-backend-ops -o GATED_DELTA_NET` (expect OK).
- Coherence: 4B, 3-GPU tensor, default vs the known-good build — same-seed
  output must be identical.  Do **not** use `GGML_CUDA_ALLREDUCE=nccl` as a
  bit-identical reference under `-sm tensor`: the internal AR always
  BF16-round-trips (`GGML_CUDA_AR_BF16_THRESHOLD` default 1) while NCCL reduces
  small tensors in FP32, so the two differ by design (measured 27B 2-GPU tensor
  greedy text `6e8ccd25` hybrid vs `6129e077` nccl, logits W=6 `a4817ee6` vs
  `73ff91bf`).  It only matches for splits with no cross-device reduction
  (1 GPU, `-sm layer`).
- MTP gate: `benchmarks/mtp-adaptive-methodology.md` Protocol A (acceptance
  >= ~0.45, MTP t/s >= plain).
- perf: `llama-bench -m <27B> -ngl 99 -p 512,2048,4096 -n 128 -r 5`
  (never run benches in parallel; interleave A/B runs to cancel thermal drift).
