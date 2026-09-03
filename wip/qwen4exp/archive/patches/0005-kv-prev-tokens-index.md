# 0005 - KV prev-tokens index (PR #27992-style, our implementation)

Applies: src/llama-kv-cache.{cpp,h}, src/llama-kv-cells.h (query side).

What it does: adds `seq_pos_token_le` (O(log n) predecessor lookup) +
`llama_kv_cache::get_prev_tokens_indexed` + the LLAMA_KV_PREV_TOKENS
mode switch (FAST default / VERIFY / OFF) + diagnostic counters.

History: unit-tested 9480 lookups 0 failures; model verify mode 0
mismatches; index 0.18us vs scan 13.8us/call at 2K ctx. At 16K the
win is FLAT (683 vs 692 t/s); the win is at 240K ctx (2.7x decode).

NOTE (2026-09-02): the base `seq_pos` cell-position map is now ALREADY
upstream (landed via #27991 "kv-cache: optimize restoring
non-contiguous cells" in the 0eadefebd re-base). This patch adds the
query side on top and applies cleanly to the current fork (rdna-boosts
482837e5a / qwen4exp 0ff151bda). Unit test: test-kv-prev-tokens.cpp
(compile standalone, not in CMake).

Superseded by / merged upstream? No - upstream has the map but not the
query. Re-check before re-applying if upstream merges PR #27992.
