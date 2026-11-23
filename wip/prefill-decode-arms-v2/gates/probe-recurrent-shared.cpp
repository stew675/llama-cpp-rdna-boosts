// STEP 5 probe: ONE recurrent memory, TWO contexts, and the per-batch PLACEMENT extracted into the context
// (invariant I5 of DESIGN.md §4 -- the upstream `TODO: extract all the state like head and n here`).
//
// THE CLAIM
//   `llama_memory_recurrent` used to keep this batch's graph-visible state ON THE MODULE (`head`, `n`, `rs_z`),
//   so two contexts building graphs read each other's placement, and `s_copy()` -- a graph-time GETTER --
//   RESET `rs_idx[seq]` as a side effect.  The placement now belongs to `llama_memory_recurrent_context`.
//
// SECTION 1 (deterministic, no threads, no timing) -- THE NON-VACUITY CHECK
//   Two batch contexts are prepared and applied against the SAME shared recurrent memory, and BOTH are kept
//   alive; then each one's `(head, n, rs_z)` is read back.  Extracted: the two read DIFFERENT placements (each
//   batch's own).  Not extracted: both read the module's single value, i.e. they are EQUAL -- which is exactly
//   what the pre-STEP-5 build prints.
//
// SECTION 2 (two threads) -- THE TEXT
//   The wide arm (seq 0, recurrent prefill) and the narrow arm (seq 1, one token per step) run on two threads
//   against the one memory; both digests must equal the same program run one arm at a time (`--solo`).
//
// Usage:  probe-recurrent-shared MODEL.gguf [--solo] [--rounds N] [--steps N]
//         (a GDN/SSM hybrid: the probe REFUSES a model with no recurrent module rather than passing vacuously)
// Build:  <clang++> -std=c++17 -O1 -pthread probe-recurrent-shared.cpp -I<repo>/include -I<repo>/src \
//             -I<repo>/ggml/include -L<build>/bin -lllama -lggml -Wl,-rpath,<build>/bin

#include "llama.h"

#include "llama-batch.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
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

static llama_context * mk(llama_model * model, llama_context * borrow, int n_batch, int n_ubatch, int n_ctx,
                          int n_seq_max) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = n_ctx;
    p.n_batch         = n_batch;
    p.n_ubatch        = n_ubatch;
    p.n_seq_max       = n_seq_max;
    p.n_threads       = 8;
    p.n_threads_batch = 8;
    p.no_perf         = true;
    p.ctx_memory      = borrow;
    return llama_init_from_model(model, p);
}

static std::vector<llama_token> tok_of(const llama_vocab * v, const std::string & t) {
    std::vector<llama_token> tk(t.size() + 16);
    int n = llama_tokenize(v, t.c_str(), (int32_t) t.size(), tk.data(), (int32_t) tk.size(), true, false);
    if (n < 0) {
        tk.resize(-n);
        n = llama_tokenize(v, t.c_str(), (int32_t) t.size(), tk.data(), (int32_t) tk.size(), true, false);
    }
    tk.resize(n > 0 ? n : 0);
    return tk;
}

// the recurrent child of whatever memory this is (null if the model has none)
static llama_memory_recurrent * recr_of(llama_memory_t mem) {
    if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hyb->get_mem_recr();
    }
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

// the recurrent context inside a (possibly hybrid) memory context
static const llama_memory_recurrent_context * recr_ctx_of(const llama_memory_context_i * mctx) {
    if (auto * hyb = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        return hyb->get_recr();
    }
    return dynamic_cast<const llama_memory_recurrent_context *>(mctx);
}

// one full llama_process, returning the argmax of the last row
static int32_t decode(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq,
                      const std::vector<llama_token> & tokens, llama_pos pos0) {
    llama_batch_ext * b = llama_batch_ext_init(ctx);
    if (b == nullptr) {
        return -1;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        const int32_t idx = llama_batch_ext_add_token(b, seq, tokens[i]);
        if (idx < 0) { llama_batch_ext_free(b); return -1; }
        const llama_pos pos = pos0 + (llama_pos) i;
        (void) llama_batch_ext_set_pos(b, idx, &pos);
        (void) llama_batch_ext_set_output_logits(b, idx, i + 1 == tokens.size());
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    llama_batch_ext_free(b);
    if (rc != 0) {
        printf("FAIL llama_process -> %d\n", rc);
        g_fail++;
        return -1;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return -1;
    }
    const int nv = llama_vocab_n_tokens(vocab);
    int32_t best = 0;
    for (int i = 1; i < nv; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static uint64_t hash_tokens(const std::vector<llama_token> & v) {
    uint64_t h = 1469598103934665603ull;
    for (llama_token t : v) {
        h ^= (uint64_t) (uint32_t) t;
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf [--solo] [--rounds N] [--steps N]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    bool solo = false;
    int rounds = 4, steps = 16;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--solo"))   { solo = true; }
        else if (!strcmp(argv[i], "--rounds")) { rounds = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--steps"))  { steps  = atoi(argv[++i]); }
        else { printf("unknown arg %s\n", argv[i]); return 2; }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context * a = mk(model, nullptr, 1024, 512, 4096, 4);
    llama_context * b = a ? mk(model, a, 8, 8, 4096, 4) : nullptr;
    if (a == nullptr || b == nullptr) {
        printf("FAIL: could not create the context pair\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(a);
    llama_memory_recurrent * recr = recr_of(mem);
    printf("memory %p, recurrent module %p (%u cells), borrower holds the same memory: %d\n",
           (void *) mem, (void *) recr, recr != nullptr ? recr->size : 0u,
           (int) (llama_get_memory(b) == mem));
    check(recr != nullptr, "this model HAS a recurrent module (a GDN/SSM hybrid -- otherwise the probe is vacuous)");
    if (recr == nullptr) {
        return 1;
    }

    printf("\n1. two live batch contexts on ONE recurrent memory hold DIFFERENT placements (I5)\n");
    {
        // two ubatches of DIFFERENT widths, one per sequence, prepared through the real path
        // (llama_batch_ext -> llama_batch_allocr -> memory->init_batch) and applied.  Both contexts are KEPT
        // alive, so the two placements are readable at the same moment.
        const uint32_t n_a = 32;   // seq 0: a wide batch (ONE recurrent cell -- a cell holds a sequence's state)
        const uint32_t n_b = 8;    // seq 1: a narrow batch

        auto make_ctx = [&](llama_seq_id seq, uint32_t n) -> llama_memory_context_ptr {
            llama_batch_ext * be = llama_batch_ext_init(a);
            if (be == nullptr) { return nullptr; }
            for (uint32_t i = 0; i < n; i++) {
                const int32_t idx = llama_batch_ext_add_token(be, seq, 0);
                const llama_pos pos = (llama_pos) i;
                (void) llama_batch_ext_set_pos(be, idx, &pos);
                (void) llama_batch_ext_set_output_logits(be, idx, false);
            }
            llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
            if (!balloc.init(*be, *vocab, false)) {
                llama_batch_ext_free(be);
                return nullptr;
            }
            auto mctx = mem->init_batch(balloc, n, false);
            llama_batch_ext_free(be);
            return mctx;
        };

        llama_memory_clear(mem, true);

        auto mctx_a = make_ctx(0, n_a);
        auto mctx_b = make_ctx(1, n_b);
        check(mctx_a != nullptr && mctx_b != nullptr, "both batch contexts were prepared");
        if (mctx_a && mctx_b) {
            check(mctx_a->apply(), "context A applied");
            check(mctx_b->apply(), "context B applied");
            // ... and both are STILL ALIVE here: this is the state in which the old module had one value

            const auto * ra = recr_ctx_of(mctx_a.get());
            const auto * rb = recr_ctx_of(mctx_b.get());
            check(ra != nullptr && rb != nullptr, "both memory contexts expose their recurrent context");
            if (ra != nullptr && rb != nullptr) {
                printf("     A: head=%u n=%u rs_z=%d | B: head=%u n=%u rs_z=%d\n",
                       ra->get_head(), ra->get_n_rs(), ra->get_rs_z(),
                       rb->get_head(), rb->get_n_rs(), rb->get_rs_z());
                check(ra != rb, "the two contexts are distinct objects");
                check(ra->get_n_rs() != rb->get_n_rs() || ra->get_head() != rb->get_head(),
                      "the two placements DIFFER at the same instant (a module-held placement would be equal)");
                // a recurrent cell holds a whole SEQUENCE's state, so both batches occupy ONE cell each; what
                // distinguishes the placements is WHICH cell, and whose sequence lives in it
                check(ra->get_head() != rb->get_head(), "each context points at its OWN cell");
                const auto & cell_a = recr->cells[ra->get_head()];
                const auto & cell_b = recr->cells[rb->get_head()];
                check(cell_a.has_seq_id(0) && !cell_a.has_seq_id(1),
                      "A's cell holds sequence 0 (and not 1)");
                check(cell_b.has_seq_id(1) && !cell_b.has_seq_id(0),
                      "B's cell holds sequence 1 (and not 0)");
            }
        }
        // the placements are released with the contexts; reset the memory for section 2
        mctx_a.reset();
        mctx_b.reset();
        llama_memory_clear(mem, true);
    }

    printf("\n2. two threads, one memory: the text is the same as one arm at a time\n");
    const std::vector<llama_token> seed_ = tok_of(vocab, "The quick brown fox jumps over the lazy dog. ");
    std::vector<llama_token> big;
    while (big.size() < 512) {
        big.insert(big.end(), seed_.begin(), seed_.end());
    }
    big.resize(512);
    const std::vector<llama_token> prime = tok_of(vocab, "List three uses of a bucket:");

    std::vector<int32_t>  tok_a;
    std::vector<llama_token> tok_b;
    std::atomic<int> done_a{0}, done_b{0};

    auto run_a = [&]() {
        int32_t t = decode(a, vocab, 0, big, 0);
        tok_a.push_back(t);
        int n_pos = (int) big.size();
        for (int r = 1; r < rounds; r++) {
            std::vector<llama_token> one = { (llama_token) t };
            t = decode(a, vocab, 0, one, (llama_pos) n_pos++);
            tok_a.push_back(t);
        }
        done_a = 1;
    };
    auto run_b = [&]() {
        int32_t t = decode(b, vocab, 1, prime, 0);
        int n_pos = (int) prime.size();
        for (int s = 0; s < steps; s++) {
            std::vector<llama_token> one = { (llama_token) t };
            t = decode(b, vocab, 1, one, (llama_pos) n_pos++);
            if (t < 0) { g_fail++; break; }
            tok_b.push_back((llama_token) t);
        }
        done_b = 1;
    };

    const auto t0 = std::chrono::steady_clock::now();
    if (solo) {
        run_a();
        run_b();
    } else {
        std::thread ta(run_a);
        std::thread tb(run_b);
        while (!(done_a && done_b)) {
            const auto dt = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - t0).count();
            if (dt > 300) {
                printf("FAIL: the two arms did not finish within 300 s (deadlock?)\n");
                fflush(stdout);
                std::_Exit(3);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ta.join();
        tb.join();
    }
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

    printf("\nDECODE");
    for (llama_token t : tok_b) { printf(" %d", (int) t); }
    printf("\nDIGEST arm_a=%016llx arm_b=%016llx round_ms=%lld\n",
           (unsigned long long) hash_tokens(tok_a), (unsigned long long) hash_tokens(tok_b), (long long) dt);
    check((int) tok_a.size() == rounds, "the wide arm completed every round");
    check((int) tok_b.size() == steps,  "the narrow arm completed every step");

    llama_free(b);
    llama_free(a);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", g_fail == 0 ? (solo ? "PASS (solo control)" : "PASS (one recurrent memory, two contexts)")
                               : "FAIL: see the lines above");
    return g_fail == 0 ? 0 : 1;
}
