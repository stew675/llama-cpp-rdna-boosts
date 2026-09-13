diff --git a/src/llama-kv-cache.cpp b/src/llama-kv-cache.cpp
index 65afbd8c3..5ede0ea05 100644
--- a/src/llama-kv-cache.cpp
+++ b/src/llama-kv-cache.cpp
@@ -8,7 +8,9 @@
 #include <algorithm>
 #include <array>
 #include <cassert>
+#include <chrono>
 #include <cmath>
+#include <cstdlib>
 #include <cstring>
 #include <limits>
 #include <map>
@@ -1826,6 +1828,88 @@ bool llama_kv_cache::has_cell_ext() const {
     return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
 }
 
+//
+// (seq, pos) -> token index for the n-gram predecessor lookups in get_prev_tokens()
+//
+// the llama_kv_cells::seq_pos index stores the rows of the cells per (seq, pos), which allows resolving the
+// predecessor tokens in O(needs * log n) instead of scanning all used cells once per ubatch. controlled by the
+// LLAMA_KV_PREV_TOKENS environment variable:
+//
+//   fast (default):   use the index for eligible ubatches, fall back to the general scan when not applicable
+//   verify:           run the general scan (source of truth) and the index for every call, compare them and log
+//                     mismatches (with a heartbeat of the scan / index cost). intended for testing and debugging
+//   off:              general scan only (the index is still maintained, but never queried)
+//
+
+enum class llama_prev_tokens_idx_mode {
+    VERIFY,
+    FAST,
+    OFF,
+};
+
+static llama_prev_tokens_idx_mode llama_prev_tokens_idx_mode_get() {
+    static const llama_prev_tokens_idx_mode mode = []() {
+        const char * env = getenv("LLAMA_KV_PREV_TOKENS");
+
+        if (env && strcmp(env, "verify") == 0) { return llama_prev_tokens_idx_mode::VERIFY; }
+        if (env && strcmp(env, "off")    == 0) { return llama_prev_tokens_idx_mode::OFF;    }
+
+        return llama_prev_tokens_idx_mode::FAST;
+    }();
+
+    return mode;
+}
+
+// diagnostic counters - decode is serialized per context, benign otherwise (reporting only):
+static struct {
+    uint64_t n_calls        = 0;
+    uint64_t n_needs        = 0;
+    uint64_t n_mismatch     = 0;
+    uint64_t n_not_applic   = 0; // ubatches the index cannot serve (multimodal)
+    uint64_t n_fast_hits    = 0;
+    double   t_scan_us      = 0.0;
+    double   t_index_us     = 0.0;
+} g_prev_tokens_diag;
+
+bool llama_kv_cache::get_prev_tokens_indexed(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
+    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions do not encode the
+    // token order - the general scan resolves those predecessors by ubatch order instead
+    if (!ubatch.token) {
+        return false;
+    }
+
+    res.clear();
+    res.resize(ubatch.n_tokens*n, LLAMA_TOKEN_NULL);
+
+    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
+        // same choice as the general scan's consumer (the n-gram architectures reject shared tokens)
+        const llama_seq_id seq_id = ubatch.seq_id[i][0];
+
+        for (uint32_t j = 0; j < n; ++j) {
+            const llama_pos p = ubatch.pos[i] - (llama_pos) (n - j);
+
+            if (p < 0) {
+                continue;
+            }
+
+            llama_token tok = LLAMA_TOKEN_NULL;
+
+            for (uint32_t s = 0; s < n_stream; ++s) {
+                llama_token t = LLAMA_TOKEN_NULL;
+
+                // later stream wins, matching the general scan which merges all streams into one hist
+                if (v_cells[s].seq_pos_token_le(seq_id, p, &t)) {
+                    tok = t;
+                }
+            }
+
+            res[i*n + j] = tok;
+        }
+    }
+
+    return true;
+}
+
 void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
     const uint32_t n_tokens = ubatch.n_tokens;
 
@@ -1836,6 +1920,45 @@ void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, st
         return;
     }
 
+    const llama_prev_tokens_idx_mode idx_mode = llama_prev_tokens_idx_mode_get();
+    const auto t_start = std::chrono::steady_clock::now();
+
+    if (idx_mode == llama_prev_tokens_idx_mode::FAST) {
+        const auto t_i0 = std::chrono::steady_clock::now();
+
+        if (get_prev_tokens_indexed(ubatch, n, res)) {
+            const auto t_i1 = std::chrono::steady_clock::now();
+
+            g_prev_tokens_diag.n_calls++;
+            g_prev_tokens_diag.n_needs += (uint64_t) n_tokens*n;
+            g_prev_tokens_diag.n_fast_hits++;
+            g_prev_tokens_diag.t_index_us += std::chrono::duration<double, std::micro>(t_i1 - t_i0).count();
+
+            if (g_prev_tokens_diag.n_calls % 2000 == 0) {
+                LLAMA_LOG_WARN("%s: PLE-IDX mode=fast calls=%llu needs=%llu hits=%llu not_applicable=%llu mismatches=%llu idx_avg=%.3f us used=%u\n",
+                        __func__,
+                        (unsigned long long) g_prev_tokens_diag.n_calls,
+                        (unsigned long long) g_prev_tokens_diag.n_needs,
+                        (unsigned long long) g_prev_tokens_diag.n_fast_hits,
+                        (unsigned long long) g_prev_tokens_diag.n_not_applic,
+                        (unsigned long long) g_prev_tokens_diag.n_mismatch,
+                        g_prev_tokens_diag.t_index_us / g_prev_tokens_diag.n_calls,
+                        v_cells[0].get_used());
+            }
+
+            return;
+        }
+
+        // the index is not applicable to this ubatch - run the general scan below
+        g_prev_tokens_diag.n_not_applic++;
+
+        res.clear();
+        res.resize(n_tokens*n, LLAMA_TOKEN_NULL);
+
+        LLAMA_LOG_WARN("%s: PLE-IDX mode=fast fell back to general scan (n_tokens=%u multimodal=%d)\n",
+                __func__, n_tokens, !ubatch.token);
+    }
+
     // note: apply_ubatch() has already stored the current ubatch
     //       the window below thus covers tokens of this very ubatch as well, which is what we want
     llama_pos p_min = std::numeric_limits<llama_pos>::max();
@@ -1928,6 +2051,64 @@ void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, st
             res[i*n + j] = lookup(seq_id, p);
         }
     }
+
+    if (idx_mode == llama_prev_tokens_idx_mode::OFF) {
+        return;
+    }
+
+    // verify the index against the general scan, which remains the source of truth (no speedup intended here):
+    // any MISMATCH line means the index and the scan disagree and must not be trusted yet
+    {
+        const auto t_g1 = std::chrono::steady_clock::now();
+
+        std::vector<llama_token> res_idx;
+
+        const auto t_i0 = std::chrono::steady_clock::now();
+        const bool applicable = get_prev_tokens_indexed(ubatch, n, res_idx);
+        const auto t_i1 = std::chrono::steady_clock::now();
+
+        g_prev_tokens_diag.n_calls++;
+        g_prev_tokens_diag.t_scan_us  += std::chrono::duration<double, std::micro>(t_g1 - t_start).count();
+        g_prev_tokens_diag.t_index_us += std::chrono::duration<double, std::micro>(t_i1 - t_i0).count();
+
+        if (!applicable) {
+            g_prev_tokens_diag.n_not_applic++;
+        } else {
+            for (uint32_t i = 0; i < n_tokens; ++i) {
+                for (uint32_t j = 0; j < n; ++j) {
+                    const llama_token a = res[i*n + j];
+                    const llama_token b = res_idx[i*n + j];
+
+                    if (a == b) {
+                        continue;
+                    }
+
+                    if (g_prev_tokens_diag.n_mismatch < 32) {
+                        LLAMA_LOG_ERROR("%s: PLE-IDX MISMATCH calls=%llu i=%u j=%u seq=%d pos[i]=%d scan=%d index=%d\n",
+                                __func__,
+                                (unsigned long long) g_prev_tokens_diag.n_calls,
+                                i, j, ubatch.seq_id[i][0], ubatch.pos[i], a, b);
+                    }
+
+                    g_prev_tokens_diag.n_mismatch++;
+                }
+            }
+
+            g_prev_tokens_diag.n_needs += (uint64_t) n_tokens*n;
+        }
+
+        if (g_prev_tokens_diag.n_calls % 2000 == 0) {
+            LLAMA_LOG_WARN("%s: PLE-IDX mode=verify calls=%llu needs=%llu mismatches=%llu not_applicable=%llu scan_avg=%.1f us index_avg=%.3f us used=%u\n",
+                    __func__,
+                    (unsigned long long) g_prev_tokens_diag.n_calls,
+                    (unsigned long long) g_prev_tokens_diag.n_needs,
+                    (unsigned long long) g_prev_tokens_diag.n_mismatch,
+                    (unsigned long long) g_prev_tokens_diag.n_not_applic,
+                    g_prev_tokens_diag.t_scan_us  / g_prev_tokens_diag.n_calls,
+                    g_prev_tokens_diag.t_index_us / g_prev_tokens_diag.n_calls,
+                    v_cells[0].get_used());
+        }
+    }
 }
 
 size_t llama_kv_cache::total_size() const {
diff --git a/src/llama-kv-cache.h b/src/llama-kv-cache.h
index c4d8699de..eab552de3 100644
--- a/src/llama-kv-cache.h
+++ b/src/llama-kv-cache.h
@@ -243,6 +243,11 @@ public:
     // note: used by n-gram input embeddings
     void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;
 
+    // (seq, pos) -> token index variant of get_prev_tokens - O(needs * log n) instead of scanning all cells
+    // returns false if the ubatch is not eligible (multimodal predecessors are resolved by ubatch order)
+    // see llama_kv_cells::seq_pos_token_le() and the LLAMA_KV_PREV_TOKENS modes in llama-kv-cache.cpp
+    bool get_prev_tokens_indexed(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;
+
 private:
     const llama_model & model;
     const llama_hparams & hparams;
diff --git a/src/llama-kv-cells.h b/src/llama-kv-cells.h
index a4292c79e..78da75953 100644
--- a/src/llama-kv-cells.h
+++ b/src/llama-kv-cells.h
@@ -246,7 +246,7 @@ public:
         assert(seq_id >= 0);
 
         seq[i].reset(seq_id);
-        seq_pos_dec(seq_id, pos[i]);
+        seq_pos_dec(seq_id, pos[i], i);
 
         if (seq[i].none()) {
             pos[i] = -1;
@@ -270,7 +270,7 @@ public:
             seq[i].reset();
 
             seq[i].set(seq_id);
-            seq_pos_inc(seq_id, pos[i]);
+            seq_pos_inc(seq_id, pos[i], i);
 
             return false;
         }
@@ -340,7 +340,7 @@ public:
         assert(!seq[i].test(seq_id));
 
         seq[i].set(seq_id);
-        seq_pos_inc(seq_id, pos[i]);
+        seq_pos_inc(seq_id, pos[i], i);
     }
 
     // return the sequence id of this cell
@@ -367,7 +367,7 @@ public:
             return -1;
         }
 
-        assert(seq_pos[seq_id].begin()->second > 0);
+        assert(!seq_pos[seq_id].begin()->second.empty());
 
         return seq_pos[seq_id].begin()->first;
     }
@@ -382,11 +382,36 @@ public:
             return -1;
         }
 
-        assert(seq_pos[seq_id].rbegin()->second > 0);
+        assert(!seq_pos[seq_id].rbegin()->second.empty());
 
         return seq_pos[seq_id].rbegin()->first;
     }
 
+    // the token of the newest cell (largest row) at the largest position <= p for the given sequence
+    // this reproduces the result of an ascending scan over all cells building a (seq, pos) -> token map, where
+    // the last cell written for a (seq, pos) pair wins - see llama_kv_cache::get_prev_tokens()
+    // return values:
+    //  +1: found, *tok is set
+    //   0: the sequence has no cell at or before p
+    int seq_pos_token_le(llama_seq_id seq_id, llama_pos p, llama_token * tok) const {
+        assert(seq_id >= 0);
+        assert(seq_id < LLAMA_MAX_SEQ);
+
+        const auto & m = seq_pos[seq_id];
+
+        auto it = m.upper_bound(p);
+        if (it == m.begin()) {
+            return 0;
+        }
+
+        --it;
+        assert(!it->second.empty());
+
+        *tok = ext[*it->second.rbegin()].tok;
+
+        return 1;
+    }
+
     // note: call only if the cell is not empty
     llama_pos pos_get(uint32_t i) const {
         assert(i < pos.size());
@@ -516,7 +541,7 @@ private:
     // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
     std::vector<seq_set_t> seq;
 
-    // the set seq_pos[s][p] tells us how many times the position p is currently present for sequence s
+    // the set seq_pos[s][p] contains the rows (cell indices) of all cells currently occupying position p for sequence s
     // if the position p is not present, seq_pos[s][p] is not set
     // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
     //
@@ -524,28 +549,34 @@ private:
     //  - during performing a cache reuse via (rm + add)
     //  - some vision models have input embeddings with repeating positions
     //
-    std::map<llama_pos, int> seq_pos[LLAMA_MAX_SEQ];
+    // keeping the rows (and not just a reference count) allows resolving the token of a (seq, pos) pair without
+    // scanning all cells. the rows are sorted, so *rbegin() reproduces the "last cell wins" result of an ascending
+    // scan over all cells - see llama_kv_cache::get_prev_tokens() and llama_kv_cells::seq_pos_token_le()
+    //
+    std::map<llama_pos, std::set<uint32_t>> seq_pos[LLAMA_MAX_SEQ];
 
     // helper functions for updating `seq_pos`, once cell at a time:
 
-    void seq_pos_dec(llama_seq_id s, llama_pos p) {
+    void seq_pos_dec(llama_seq_id s, llama_pos p, uint32_t row) {
         auto it = seq_pos[s].find(p);
         assert(it != seq_pos[s].end());
 
-        if (--it->second == 0) {
+        it->second.erase(row);
+
+        if (it->second.empty()) {
             seq_pos[s].erase(it);
         }
     }
 
-    void seq_pos_inc(llama_seq_id s, llama_pos p) {
-        seq_pos[s][p]++;
+    void seq_pos_inc(llama_seq_id s, llama_pos p, uint32_t row) {
+        seq_pos[s][p].insert(row);
     }
 
     // remove cell i
     void seq_pos_rm(uint32_t i) {
         for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
             if (seq[i].test(s)) {
-                seq_pos_dec(s, pos[i]);
+                seq_pos_dec(s, pos[i], i);
             }
         }
     }
@@ -554,7 +585,7 @@ private:
     void seq_pos_add(uint32_t i) {
         for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
             if (seq[i].test(s)) {
-                seq_pos_inc(s, pos[i]);
+                seq_pos_inc(s, pos[i], i);
             }
         }
     }
