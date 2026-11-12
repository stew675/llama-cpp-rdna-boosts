# Routing traces

Not committed (a 538-token prompt + 200 decode tokens on the 122B is ~4 MB).  Reproduce the one the admission
replay used:

```sh
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_HEATMAP=load \
MOE_EXPERT_TRACE=$PWD/q8-lme90-code533-n200.trace \
  ~/llama-scratch/build-rocm/bin/llama-cli -m /llm/models/Qwen3.5/122B-A10B/Q8_0/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf \
  -lme 90 -ub 4096 -b 4096 -ngl all --fit off --seed 42 --temp 0 --no-display-prompt --single-turn \
  -c 8192 -n 200 -f ~/llama-cpp-rdna-boosts/prompts/code-python.txt
```

then `../admission-sim.sh` (it defaults to that path and the run's frozen `.moe-heatmap`).
