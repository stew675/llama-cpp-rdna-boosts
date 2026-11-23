// STEP 3 probe: ONE `llama_memory` used by TWO contexts on TWO THREADS (invariants I3/I4 of DESIGN.md §4).
//
// WHAT IT ASSERTS
//   1. A wide PREFILL arm and a narrow DECODE arm run at the same time against one shared memory, on different
//      sequences, and neither corrupts the other: the prefill arm's sampled token and the decode arm's whole
//      token stream are IDENTICAL to the same program run solo (`--solo`).  Each arm's attention rows are its
//      own sequence's cells (the unified cache uses per-sequence KV rows), so this identity is a real
//      requirement, not a coincidence of the layout.
//   2. Nothing deadlocks.  The lock covers bookkeeping only; each arm's compute runs outside it.  A deadlock
//      shows up as the watchdog below ("did not finish within N s").
//   3. I4 is EXERCISED, not merely present: the prefill arm's prompt is deliberately WIDER than its `n_ubatch`,
//      so `init_batch()` prepares several ubatches at once (each one applies its cells as a trial and then the
//      cell map is ROLLED BACK).  During the whole prefill the decode arm is searching that same, apparently
//      EMPTY, cache -- which is exactly the window the reservation bitmap exists for.  Against a build without
//      the bitmap the decode arm takes cells the prefill arm has already prepared, and this probe diverges.
//
// Usage:
//   probe-n1-threads MODEL.gguf            # the two-thread arm
//   probe-n1-threads MODEL.gguf --solo     # the same program, one arm at a time (the control)
//   [--rounds N] [--steps N] [--prompt-tokens N] [--watchdog S]
//
// Both invocations print one line per arm: `PREFILL round r -> tok`, then `DECODE <tokens...>`, and a final
// `DIGEST prefill=<n> decode=<hash>`.  The gate is a `diff` of the two digests.

#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_fail++;
    }
}

struct arm {
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
};

static llama_context * mk_ctx(llama_model * model, llama_context * borrow, int n_batch, int n_ubatch,
                              int n_seq_max) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = 8192;
    p.n_batch         = n_batch;
    p.n_ubatch        = n_ubatch;
    p.n_seq_max       = n_seq_max;
    p.n_threads       = 8;
    p.n_threads_batch = 8;
    p.no_perf         = true;
    p.ctx_memory      = borrow;
    return llama_init_from_model(model, p);
}

static std::vector<llama_token> tok_of(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                           true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(),
                           true, false);
    }
    tokens.resize(n > 0 ? n : 0);
    return tokens;
}

// decode `tokens` at `pos0` on `seq`, last row's logits only; returns the argmax of the last row
static int32_t decode_argmax(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq,
                             const std::vector<llama_token> & tokens, llama_pos pos0) {
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (batch == nullptr) {
        return -1;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, seq, tokens[i]);
        if (idx < 0) {
            // -1 = the batch is full (it is sized by the context's n_batch): say so, do not return a silent -1
            printf("FAIL llama_batch_ext_add_token -> %d at token %zu of %zu (n_batch too small for this batch)\n",
                   idx, i, tokens.size());
            g_fail++;
            llama_batch_ext_free(batch);
            return -1;
        }
        const llama_pos pos = pos0 + (llama_pos) i;
        (void) llama_batch_ext_set_pos(batch, idx, &pos);
        (void) llama_batch_ext_set_output_logits(batch, idx, i + 1 == tokens.size());
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (rc != 0) {
        printf("FAIL llama_process -> %d\n", rc);
        g_fail++;
        return -1;
    }
    // -1 = the last output row.  A batch WIDER than n_ubatch is split, and only the rows of the LAST ubatch
    // survive in the logits buffer, so the batch-index form (`tokens.size() - 1`) is wrong here.
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return -1;
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int32_t best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static uint64_t hash_tokens(const std::vector<llama_token> & v) {
    uint64_t h = 1469598103934665603ull;   // FNV-1a
    for (llama_token t : v) {
        h ^= (uint64_t) (uint32_t) t;
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf [--solo] [--rounds N] [--steps N] [--prompt-tokens N] [--watchdog S]\n",
               argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    bool   solo    = false;
    int    rounds  = 3;
    int    steps   = 32;
    int    nprompt = 1900;   // < n_batch (2048), > n_ubatch (512): one batch, several ubatches
    int    watchdog = 300;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--solo"))   { solo = true; }
        else if (!strcmp(argv[i], "--rounds")) { rounds  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--steps"))  { steps   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--prompt-tokens")) { nprompt = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--watchdog"))      { watchdog = atoi(argv[++i]); }
        else { printf("unknown arg %s\n", argv[i]); return 2; }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // a prompt deliberately WIDER than the prefill arm's n_ubatch (512), so init_batch() prepares several
    // ubatches at once -- the I4 window
    std::vector<llama_token> prefill = tok_of(vocab, "The quick brown fox jumps over the lazy dog. ");
    std::vector<llama_token> big;
    while ((int) big.size() < nprompt) {
        big.insert(big.end(), prefill.begin(), prefill.end());
    }
    big.resize(nprompt);

    const std::vector<llama_token> prime = tok_of(vocab, "List three uses of a bucket:");

    // owner: wide (prefill-shaped); borrower: narrow (decode-shaped)
    llama_context * ctx_a = mk_ctx(model, nullptr, 2048, 512, 4);
    if (ctx_a == nullptr) { printf("FAIL: owner context\n"); return 1; }
    llama_context * ctx_b = mk_ctx(model, ctx_a, 8, 8, 4);
    if (ctx_b == nullptr) { printf("FAIL: borrower context\n"); return 1; }
    if (llama_get_memory(ctx_a) != llama_get_memory(ctx_b)) {
        printf("FAIL: the borrower does not hold the owner's memory\n");
        return 1;
    }
    printf("owner %p / borrower %p / one memory %p | prefill %zu tokens (n_ubatch 512), %d rounds | %d decode steps\n",
           (void *) ctx_a, (void *) ctx_b, (void *) llama_get_memory(ctx_a), big.size(), rounds, steps);

    std::vector<int32_t> prefill_tok;
    std::vector<llama_token> decode_tok;
    std::atomic<int> done_a{0}, done_b{0};

    auto run_prefill = [&]() {
        for (int r = 0; r < rounds; r++) {
            const int32_t t = decode_argmax(ctx_a, vocab, 0, big, 0);
            printf("PREFILL round %d -> tok %d\n", r, t);
            prefill_tok.push_back(t);
            // free seq 0's cells so the next round prefills again (this is a request boundary, not a decode)
            const bool ok = llama_memory_seq_rm(llama_get_memory(ctx_a), 0, -1, -1);
            if (!ok) { printf("FAIL: seq_rm(0)\n"); g_fail++; }
        }
        done_a = 1;
    };

    auto run_decode = [&]() {
        // prime the decode arm's own sequence
        int32_t t = decode_argmax(ctx_b, vocab, 1, prime, 0);
        if (t < 0) { printf("FAIL: decode prime\n"); g_fail++; done_b = 1; return; }
        const int n_prime = (int) prime.size();
        for (int s = 0; s < steps; s++) {
            const std::vector<llama_token> one = { (llama_token) t };
            t = decode_argmax(ctx_b, vocab, 1, one, (llama_pos) (n_prime + s));
            if (t < 0) { printf("FAIL: decode step %d\n", s); g_fail++; break; }
            decode_tok.push_back((llama_token) t);
        }
        done_b = 1;
    };

    const auto t0 = std::chrono::steady_clock::now();
    if (solo) {
        run_prefill();
        run_decode();
    } else {
        std::thread ta(run_prefill);
        std::thread tb(run_decode);
        // watchdog: a deadlock or a hang must fail the probe, not wait forever (the project's supervision rule)
        while (!(done_a && done_b)) {
            const auto dt = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - t0).count();
            if (dt > watchdog) {
                printf("FAIL: the two arms did not finish within %d s (deadlock? prefill done=%d decode done=%d)\n",
                       watchdog, (int) done_a, (int) done_b);
                fflush(stdout);
                std::_Exit(3);   // leave the hung threads; the watchdog is the verdict
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ta.join();
        tb.join();
    }
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    printf("DECODE");
    for (llama_token t : decode_tok) { printf(" %d", (int) t); }
    printf("\n");
    printf("DIGEST prefill=%016llx decode=%016llx round_ms=%lld\n",
           (unsigned long long) hash_tokens(prefill_tok), (unsigned long long) hash_tokens(decode_tok),
           (long long) dt);

    check((int) prefill_tok.size() == rounds, "the prefill arm completed every round");
    check((int) decode_tok.size() == steps,  "the decode arm completed every step");

    llama_free(ctx_b);   // the borrower first (I2)
    llama_free(ctx_a);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", g_fail == 0 ? (solo ? "PASS (solo control)" : "PASS (two arms, one memory)")
                               : "FAIL: see the lines above");
    return g_fail == 0 ? 0 : 1;
}
