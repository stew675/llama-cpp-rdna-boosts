#!/usr/bin/env bash
# A/B harness for wip/slab-draft-aux-reserve -- the slab's draft (aux) reserve accounting.
#
# Why a SERVER harness and not llama-bench: llama-bench feeds random tokens, so the expert routing is
# synthetic, the heat prior (the residency the slab sizes itself against) is not exercised, and a cell
# is one shot.  This runs the real `runme` shape on llama-server, warms each arm with one real turn,
# then measures ONE turn of ~7k prefill + 2000 greedy tokens.  AGENTS.md rule 5: the heat prior is
# pinned (MOE_EXPERT_HEATMAP=load, which never writes the sidecar) so every arm starts from the same
# residency; INVARIANTS r221 fact 7: the arms are INTERLEAVED, never run as a block.
#
# Usage:  ./ab-server.sh [ROUNDS] [ARMS...]
#         ./ab-server.sh 2 A B E          # default: 2 rounds of A B C E
#
# Arms (see README.md §1):
#   A  as shipped          -- MTP draft, draft experts DEVICE-resident (the reference)
#   B  --spec-draft-cpu-moe -- draft experts HOST-resident  -> exposes the aux-reserve overwrite
#   E  B + MOE_EXPERT_CACHE_AUX_RESERVE_MIB=<derived>       -> the workaround for the overwrite
#   C  no draft at all     -- the arena ceiling reference (what the arena would be without MTP)
#   D  no draft, MOE_EXPERT_CACHE_MIB=<A's arena>           -- arena-only control (C - D isolates the
#                                                             arena's decode slope from MTP's effect)
set -uo pipefail

HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
MODEL_DIR=${MODEL_DIR:-/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S}
SERVER=${SERVER:-$HOME/llama.cpp/build-rocm/bin/llama-server}
OUT=${OUT:-/tmp/slab-draft-aux}
PORT=${PORT:-8033}
MODEL=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
MTP=mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
AUTO_ARENA_MIB=${AUTO_ARENA_MIB:-12365}     # arm A's auto arena on the reference box; arm D pins it
AUX_DERIVED_MIB=${AUX_DERIVED_MIB:-1223}    # the derived draft-CONTEXT reserve (a runme log line says it)

mkdir -p "$OUT"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm-10.1.0-gfx1100/lib}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export GGML_CUDA_SLAB_HEADROOM_MIB=${GGML_CUDA_SLAB_HEADROOM_MIB:-1280}
export MOE_EXPERT_HEATMAP=load              # pin the residency prior; `load` never writes the sidecar

# One prompt, identical in every arm: the shipped prompts concatenated (README §8 records its hash and
# its 6 991-token length).  Rebuilt every run -- a interrupted or stale file silently changes the arm.
PROMPT=$OUT/prompt.txt
: > "$PROMPT"
first=1
for p in prose-rdna-boosts recall code-python reasoning code-reasoning-mixed; do
  [ $first = 0 ] && printf '\n\n' >> "$PROMPT"
  first=0
  cat "$REPO/prompts/$p.txt" >> "$PROMPT"
done
# the committed RESULTS were measured with exactly this file:
#   23 023 B, 6 991 tokens, sha256 d1eaaa5eefced21e3b96c84edc5690140a5d850573a15be84658c27cf73c1ce3


common=( --model "$MODEL" --alias qfn --host 127.0.0.1 --port "$PORT"
         --fit off --n-gpu-layers all --n-cpu-moe 48 --ctx-size 80000 --parallel 1
         --flash-attn on --limit-host-experts auto
         --cache-type-k q8_0 --cache-type-v q8_0
         --override-tensor 'per_layer_token_embd.*=CPU'
         --load-mode auto --lazy-mode auto --threads 7 --cache-ram 4096
         --batch-size 2048 --ubatch-size 2048
         --temp 0.0 --top-k 20 --min-p 0.01 --repeat-penalty 1.0 --presence-penalty 0.0
         -lv 4 )

draft_args=( --spec-draft-model "$MTP" --spec-type draft-mtp --spec-draft-n-max 3 )

vram_used() { echo $(($(cat /sys/class/drm/card1/device/mem_info_vram_used)/1024/1024)); }

run_arm() {
  local arm=$1
  local round=$2
  local tag="${arm}${round}"
  local log="$OUT/arm-$tag.log"
  local extra=()
  local env_extra=()

  case "$arm" in
    A) extra=( "${draft_args[@]}" ) ;;
    B) extra=( "${draft_args[@]}" --spec-draft-cpu-moe ); env_extra=( LLAMA_MTP_DRAFT_OP_OFFLOAD=0 ) ;;
    E) extra=( "${draft_args[@]}" --spec-draft-cpu-moe )
       env_extra=( LLAMA_MTP_DRAFT_OP_OFFLOAD=0 MOE_EXPERT_CACHE_AUX_RESERVE_MIB=$AUX_DERIVED_MIB ) ;;
    C) ;;
    D) env_extra=( MOE_EXPERT_CACHE_MIB=$AUTO_ARENA_MIB ) ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac

  cd "$MODEL_DIR" || return 1
  env "${env_extra[@]}" "$SERVER" "${common[@]}" "${extra[@]}" > "$log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 240); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then up=1; break; fi
    kill -0 $pid 2>/dev/null || { echo "$tag: server died (see $log)" >&2; return 1; }
    sleep 1
  done
  [ $up = 1 ] || { echo "$tag: no /health after 240 s" >&2; kill $pid; return 1; }

  # warm-up turn: fires the graph re-reserve and the first arena sizing, not measured
  python3 - "$OUT" "$REPO/prompts/reasoning.txt" <<'PY'
import json, sys
json.dump({"prompt": open(sys.argv[2]).read(), "n_predict": 32, "temperature": 0, "cache_prompt": False},
          open(sys.argv[1] + "/body-warm.json", "w"))
PY
  python3 "$HERE/req.py" "http://127.0.0.1:$PORT/completion" "$OUT/body-warm.json" >/dev/null 2>&1

  # measured turn: the ~7k prompt, 2000 greedy tokens, EOS ignored so the length is fixed
  python3 - "$OUT" "$PROMPT" <<'PY'
import json, sys
json.dump({"prompt": open(sys.argv[2]).read(), "n_predict": 2000, "temperature": 0,
           "ignore_eos": True, "cache_prompt": False, "top_k": 20, "min_p": 0.01,
           "repeat_penalty": 1.0, "presence_penalty": 0.0},
          open(sys.argv[1] + "/body-meas.json", "w"))
PY
  local vram_load; vram_load=$(vram_used)
  python3 "$HERE/req.py" "http://127.0.0.1:$PORT/completion" "$OUT/body-meas.json" > "$OUT/timing-$tag.json" 2>"$OUT/err-$tag.txt"
  local vram_run; vram_run=$(vram_used)

  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  for _ in $(seq 1 60); do [ "$(vram_used)" -lt 300 ] && break; sleep 1; done

  python3 - "$tag" "$log" "$vram_load" "$vram_run" "$OUT" <<'PY'
import json, re, sys
tag, log, vl, vr, out = sys.argv[1:6]
t = json.load(open(f"{out}/timing-{tag}.json"))
if abs(t["prompt_n"] - 6991) > 700:
    print(f"WARNING: prompt_n={t['prompt_n']}, expected 6991 -- was the prompt assembled correctly?",
          file=sys.stderr)
txt = open(log, errors="replace").read()
def last(pat):
    m = re.findall(pat, txt)
    return m[-1] if m else None
auto   = last(r"MoE expert cache \(auto\): arena ([\d.]+) MiB of ([\d.]+) MiB host experts \(([\d.]+)% residency\)")
turn   = last(r"MoE arena = ([\d.]+) \(([\d.]+) hit / ([\d.]+) reaches this turn\), arena ([\d.]+) MiB")
accept = last(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")
spec   = last(r"statistics\s+draft-mtp: #calls\(b,g,a\) =\s*([\d ]+), .*dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms")
slab   = last(r"slab ([\d.]+) GiB \(chunk [\d.]+ MiB\): work ([\d.]+) GiB \+ arena ([\d.]+) GiB")
slabinit = last(r"SLAB_INIT dev=0 free=([\d.]+) GiB reserve=([\d.]+) GiB headroom=([\d.]+) GiB aux=([\d.]+) GiB")
row = {
    "tag": tag,
    "pp_tps": t["pp_tps"], "prompt_n": t["prompt_n"],
    "tg_tps": t["tg_tps"], "predicted_n": t["predicted_n"],
    "arena_MiB": auto[0] if auto else (turn[3] if turn else None),
    "residency_pct": auto[2] if auto else None,
    "hit_rate": turn[0] if turn else None,
    "draft_n": t["draft_n"], "draft_accepted": t["draft_n_accepted"],
    "acceptance": accept[0] if accept else None,
    "mean_len": accept[3] if accept else None,
    "draft_gen_ms": spec[2] if spec else None,
    "slab_work_arena_GiB": list(slab) if slab else None,
    "slab_init_free_reserve_headroom_aux_GiB": list(slabinit) if slabinit else None,
    "vram_load_MiB": vl, "vram_run_MiB": vr,
    "text_sha12": t["text_sha256_12"],
}
print(json.dumps(row))
PY
}

ROUNDS=${1:-2}; shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(A B C E)

: > "$OUT/results.jsonl"
for r in $(seq 1 "$ROUNDS"); do
  for a in "${ARMS[@]}"; do
    run_arm "$a" "$r" | tee -a "$OUT/results.jsonl"
  done
done
echo "done -> $OUT/results.jsonl"
