# BETA-TESTING — 0016 `cpy_2d_vec16`

Nothing here is applied by default.  Apply the release set, then this candidate on top of it.

## 1. Apply + build

```bash
git clone https://github.com/stew675/llama-cpp-rdna-boosts && cd llama-cpp-rdna-boosts
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <release.json.base>                     # 6050b467c for r188
bash <this-repo>/scripts/apply-all.sh --include-beta .
#   expect: 16 blocks + this candidate = 17 commits, and it must report the tree
#   "6514698a25b39cac4cd815dfc929e96c9b3deda2" for the candidate (it is NOT release.json's tree)
~/bin/build-llama-rocm
```

## 2. What a PASS looks like

The change is **inert** except on the copy the guard refuses, so the primary evidence is that nothing observable
changes — and then that the cost appears and disappears on the arm that hits it.

**(a) Text identity (Flash-Next, 1 GPU).**  Same-seed greedy must be byte-identical to the build without the patch:

```bash
export HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib
M=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MD=/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
BASE="-m $M -md $MD -ngl 99 -ncmoe 48 -ot 'per_layer_token_embd.*=CPU' -lm none -lzm off -ctk q8_0 -ctv q8_0 \
      --seed 42 --temp 0 --no-display-prompt --single-turn --spec-type draft-mtp --spec-draft-n-max 4"
P=$HOME/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt

eval ./build-rocm/bin/llama-cli $BASE -c 16384 -f $P -n 64 > /tmp/t-dense.log 2>&1
LLAMA_QSA_DENSE_DECODE_UNTIL=0 eval ./build-rocm/bin/llama-cli $BASE -c 16384 -f $P -n 64 > /tmp/t-sparse.log 2>&1
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/t-dense.log    # 296 chars sha=91ef8940d66e
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/t-sparse.log   # 332 chars sha=dd1981213373
```

**(b) The prefill-logit gate** (must be 0.000000 / 100.000 %):

```bash
LLAMA_ARG_N_CPU_MOE=48 ROCBLAS_USE_HIPBLASLT=0 ~/llama-cpp-rdna-boosts/scripts/gate-prefill-logits.sh \
  --model $M --ctx 16384 --chunks 2 --kv q8_0 --base /tmp/fnx.kld --record     # then re-run without --record
```

**(c) The cost is real, on the arm that hits it** — the *sparse* decode arm at depth, and a kernel census that shows
which copy kernel ran:

```bash
# census: two lengths of the SAME arm, then subtract (prefill and the lazy loads cancel)
rocprofv3 --kernel-trace --stats -f csv -o short -- ./build-rocm/bin/llama-cli $BASE -c 131072 -f /tmp/prompt-103k.txt -n 200  -v
rocprofv3 --kernel-trace --stats -f csv -o long  -- ./build-rocm/bin/llama-cli $BASE -c 131072 -f /tmp/prompt-103k.txt -n 2000 -v
~/llama-cpp-rdna-boosts/wip/kv-residency/kv-kernel-census.py short_kernel_stats.csv long_kernel_stats.csv 200 2000
#   FAIL (bug present) : cpy_scalar<cpy_1_scalar<float,float>> ~5.2 ms/token, total ~27.0
#   PASS (fix active)  : cpy_2d_vec16 ~1.8 ms/token, cpy_scalar <0.05, total ~23.7
```

`/tmp/prompt-103k.txt` = `cat prompts/kld-corpus-35b-in-domain.txt` four times (102 899 tokens).  Better still, run
`LLAMA_QSA_DENSE_DECODE_UNTIL=0` (the sparse arm) as the census arm: it is the one this patch exists for, and it
needs no other campaign code.

## 3. Report

Post the text hashes, the gate line, the two census totals and (if you have one) a **Windows/HIP build log** — the
whole file is HIP on Windows too, and this box has no Windows runner.  A pass folds this into **block 10**; a
refutation comes back with the census CSV pair, and the patch is dropped (the fallback stays as it is).
