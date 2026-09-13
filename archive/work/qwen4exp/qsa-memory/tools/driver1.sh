#!/bin/bash
# Sequential validation matrix driver (keys-only build). No time constraints; run one GPU job at a time.
set -u
export HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0
PROG=/tmp/kt/progress.log
mkdir -p /tmp/kt
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

# ---- default greedy completion body ----
mkbody(){ # $1 outfile, $2 prompt, $3 n_predict
  python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'n_predict':int(sys.argv[2]),'temperature':0,'seed':42,'top_k':1,'top_p':1,'min_p':0,'cache_prompt':True,'stream':False}))" "$2" "$3" > "$1"
}
mkbody /tmp/kt/b1.json "The capital of France is" 40
mkbody /tmp/kt/b2.json "The largest ocean on Earth is the" 40
mkbody /tmp/kt/blong.json "The history of the French Revolution began in 1789 when" 4600

run_smoke(){ # $1 tag $2 port $3 ctx $4 batch $5 ub $6 par ${7:-} ktype ${8:-} unified ${9:-} extra...
  local TAG=$1 PORT=$2 CTX=$3 BATCH=$4 UB=$5 PAR=$6 KTYPE=${7:-q8_0} UNIFIED=${8:-0}; shift 8
  say "== $TAG: ctx=$CTX batch=$BATCH ub=$UB par=$PAR kv=$KTYPE unified=$UNIFIED extra='$*'"
  TAG="$TAG" PORT="$PORT" CTX="$CTX" BATCH="$BATCH" UB="$UB" PAR="$PAR" KTYPE="$KTYPE" \
    UNIFIED="$UNIFIED" EXTRA="$*" BODY1="$(cat /tmp/kt/b1.json)" BODY2="$(cat /tmp/kt/b2.json)" \
    bash /tmp/kt/smoke.sh 2>&1 | tee -a "$PROG"
}

# T1: non-power-of-2 / non-divisor ubatch in inference (batch 2048, ub 1536 -> 2048 % 1536 = 512)
run_smoke t1-ub1536 8091 204800 2048 1536 1 q8_0 0
# T2: bf16 KV types
run_smoke t2-bf16 8092 204800 2048 2048 1 bf16 0
# T3: f32 KV types
run_smoke t3-f32 8093 204800 2048 2048 1 f32 0
# T4: multi-slot (parallel 2, n_stream=2)
run_smoke t4-par2 8094 204800 2048 2048 2 q8_0 0
# T5: kv-unified + parallel 2
run_smoke t5-unified 8095 204800 2048 2048 2 q8_0 1
say "driver: functional smoke tests complete"
