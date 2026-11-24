PORT=18116; unset LD_LIBRARY_PATH
HIP_VISIBLE_DEVICES=0 setsid ~/llama.cpp/build-rocm/bin/llama-server -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -np 2 -c 16384 --port $PORT --prefill-decode-arms-mode on -lv 2 >/tmp/fd4b.log 2>&1 &
S=$!; for _ in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
echo "1 GPU, pair ON: fds=$(ls /proc/$S/fd | wc -l)"
kill -TERM -$S 2>/dev/null; wait $S 2>/dev/null
