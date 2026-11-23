// STEP 3 probe: the cell RESERVATION (invariant I4) -- deterministic, single-threaded, two prepares.
//
// THE CLAIM (DESIGN.md §4, I4)
//   `llama_kv_cache::prepare()` places a batch's ubatches and then ROLLS THE CELL MAP BACK, so between a
//   prepare() and the decode that commits it the cells look EMPTY.  A second prepare() -- i.e. the other
//   context, or the same one preparing the next batch -- must not be handed them.  `prepare()` therefore
//   RESERVES what it hands out, and only a real `apply_ubatch()` (or a destroyed batch context, or a failed
//   prepare) releases it.
//
// WHY IT IS NOT A TEXT TEST
//   A collision does not change one process's text: two batches write the same cells and each then reads
//   cells the other overwrote, which shows up as fluent-but-wrong output under load, or not at all.  The
//   mechanism is a SET ALGEBRA question, so it is asked directly:
//
//       s1 = prepare(ubatch A)        // reserves A's cells
//       s2 = prepare(ubatch B)        // MUST be disjoint from s1
//       apply_ubatch(s1)              // commits A, releases its reservation
//       s3 = prepare(ubatch C)        // may reuse nothing of B; may reuse A
//
//   On a build without the reservation bitmap, `s2 == s1` (the same cells, because the map was rolled back)
//   -- this probe is that control's instrument, and it needs no threads and no timing.
//
// ONE THREAD ONLY: this is the I4 question.  The lock (I3) is a separate mechanism and its oracle is the
// two-thread soak (`probe-n1-threads.cpp`).
//
// Usage:  probe-kv-reservation MODEL.gguf      (a DENSE model: its memory must BE a llama_kv_cache)
// Build:  <clang++> -std=c++17 -O1 probe-kv-reservation.cpp -I<repo>/include -I<repo>/src \
//             -I<repo>/ggml/include -L<build>/bin -lllama -lggml -Wl,-rpath,<build>/bin

#include "llama.h"

#include "llama-batch.h"
#include "llama-model.h"
#include "llama-vocab.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_fail++;
    }
}

// a hand-built single-sequence ubatch: n tokens of `seq` at positions pos0..pos0+n-1
struct ubatch_hold {
    llama_ubatch ub = {};
    llama_ubatch::data_t d;

    ubatch_hold(uint32_t n, llama_seq_id seq, llama_pos pos0) {
        d.token.assign(n, 0);
        d.pos.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            d.pos[i] = pos0 + (llama_pos) i;
        }
        d.n_seq_id.assign(n, 1);
        d.seq_id_data.assign(n, seq);
        d.seq_id.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            d.seq_id[i] = &d.seq_id_data[i];
        }
        d.seq_id_unq.assign(1, seq);
        d.seq_idx.assign(LLAMA_MAX_SEQ, 0);
        d.output.assign(n, 0);

        ub.b_equal_seqs = 1;
        ub.n_tokens     = n;
        ub.n_seq_tokens = n;
        ub.n_seqs       = 1;
        ub.n_seqs_unq   = 1;
        ub.n_pos        = 1;
        ub.token        = d.token.data();
        ub.embd         = nullptr;
        ub.embd_state   = nullptr;
        ub.embd_row     = nullptr;
        ub.pos          = d.pos.data();
        ub.n_seq_id     = d.n_seq_id.data();
        ub.seq_id       = d.seq_id.data();
        ub.seq_id_unq   = d.seq_id_unq.data();
        ub.seq_idx      = d.seq_idx.data();
        ub.output       = d.output.data();
        ub.type         = nullptr;
    }
};

// the cell indices a prepare() result holds, in order
static std::vector<uint32_t> cells_of(const llama_kv_cache::slot_info & si) {
    std::vector<uint32_t> v;
    for (size_t s = 0; s < si.idxs.size(); s++) {
        for (uint32_t i : si.idxs[s]) {
            v.push_back(i);
        }
    }
    return v;
}

static bool disjoint(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    for (uint32_t x : a) {
        for (uint32_t y : b) {
            if (x == y) {
                return false;
            }
        }
    }
    return true;
}

static void print_cells(const char * what, const std::vector<uint32_t> & v) {
    printf("     %s: %zu cell(s) [%u..%u]\n", what, v.size(), v.empty() ? 0 : v.front(),
           v.empty() ? 0 : v.back());
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf [--ntok N]\n", argv[0]);
        return 2;
    }
    uint32_t ntok = 512;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ntok")) {
            ntok = (uint32_t) atoi(argv[++i]);
        }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // the cells are metadata; keep this probe off the GPU entirely
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", argv[1]);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 8192;
    cp.n_batch   = 2048;
    cp.n_ubatch  = 2048;
    cp.n_seq_max = 4;
    cp.kv_unified = true;   // ONE stream: two sequences then share one cell array, which is where a collision lives
    cp.no_perf   = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        printf("FAIL: could not create the context\n");
        return 1;
    }
    auto * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
    if (kv == nullptr) {
        printf("FAIL: this model's memory is not a llama_kv_cache (use a DENSE model)\n");
        return 1;
    }
    printf("cache: %u cells, %u stream(s)\n", kv->get_size(), kv->get_n_stream());
    if (ntok * 3 > kv->get_size()) {
        printf("FAIL: --ntok %u does not fit three ubatches in %u cells\n", ntok, kv->get_size());
        return 1;
    }

    printf("\n1. an UNBORROWED cache does not reserve (the single-context path is untouched)\n");
    {
        kv->clear(true);
        ubatch_hold a(ntok, 0, 0);
        ubatch_hold b(ntok, 1, 0);
        const auto s1 = kv->prepare({ a.ub });
        const auto s2 = kv->prepare({ b.ub });
        check(!s1.empty() && !s2.empty(), "two prepares succeeded");
        if (!s1.empty() && !s2.empty()) {
            print_cells("prepare A", cells_of(s1[0]));
            print_cells("prepare B", cells_of(s2[0]));
            // with ONE context the caller applies s1 before preparing s2, so an overlap here is the documented
            // single-context behaviour and the reason the bitmap is opt-in (a borrower must exist first)
            check(!disjoint(cells_of(s1[0]), cells_of(s2[0])),
                  "no borrower: prepare B MAY reuse A's cells (the bitmap is off, as designed)");
        }
        kv->clear(true);
    }

    printf("\n2. a BORROWED cache reserves: prepare() x2 must be disjoint (I4)\n");
    llama_context_params bp = llama_context_default_params();
    bp.n_ctx     = 8192;
    bp.n_batch   = 2048;
    bp.n_ubatch  = 2048;
    bp.n_seq_max = 4;
    bp.kv_unified = true;
    bp.no_perf   = true;
    bp.ctx_memory = ctx;   // the real trigger: notify_borrowed() comes from here
    llama_context * borrower = llama_init_from_model(model, bp);
    check(borrower != nullptr && llama_get_memory(borrower) == llama_get_memory(ctx),
          "the borrower holds the same memory");

    {
        kv->clear(true);
        ubatch_hold a(ntok, 0, 0);
        ubatch_hold b(ntok, 1, 0);
        ubatch_hold c(ntok, 2, 0);

        const auto s1 = kv->prepare({ a.ub });
        const auto s2 = kv->prepare({ b.ub });
        check(!s1.empty() && !s2.empty(), "both prepares succeeded");
        if (s1.empty() || s2.empty()) {
            return 1;
        }
        const auto c1 = cells_of(s1[0]);
        const auto c2 = cells_of(s2[0]);
        print_cells("prepare A", c1);
        print_cells("prepare B", c2);
        check(c1.size() == ntok && c2.size() == ntok, "each prepare placed the whole ubatch");
        check(disjoint(c1, c2), "prepare B's cells are DISJOINT from prepare A's (they were reserved)");

        // commit A: its reservation must be released, and B's must still hold
        kv->apply_ubatch(s1[0], a.ub);
        const auto s3 = kv->prepare({ c.ub });
        check(!s3.empty(), "a third prepare succeeded");
        if (!s3.empty()) {
            const auto c3 = cells_of(s3[0]);
            print_cells("prepare C", c3);
            check(disjoint(c3, c2), "prepare C's cells are DISJOINT from the still-reserved prepare B");
            // A was committed, so its cells are legitimately reusable only after the sequence is removed
            printf("     (C overlapping A's committed cells: %s -- A is still live here, so this must be no)\n",
                   disjoint(c3, c1) ? "no" : "yes");
            check(disjoint(c3, c1), "prepare C did not overwrite A's COMMITTED cells");
        }
    }

    printf("\n3. a batch context dropped without applying releases its reservation (cancelled batch)\n");
    {
        kv->clear(true);

        // the real path: a llama_batch_ext -> llama_batch_allocr -> memory->init_batch(), which PREPARES (and
        // therefore reserves) the ubatches; the context is then destroyed without apply() -- an abort/cancel
        llama_batch_ext * be = llama_batch_ext_init(ctx);
        check(be != nullptr, "the batch was created");
        std::vector<llama_token> tokens(ntok, 0);
        for (uint32_t i = 0; i < ntok; i++) {
            const int32_t idx = llama_batch_ext_add_token(be, 0, 0);
            const llama_pos pos = (llama_pos) i;
            (void) llama_batch_ext_set_pos(be, idx, &pos);
            (void) llama_batch_ext_set_output_logits(be, idx, false);
        }

        llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
        check(balloc.init(*be, *llama_model_get_vocab(model), false), "the batch was sanitised");
        {
            const auto mctx = kv->init_batch(balloc, ntok, false);
            check(mctx != nullptr && mctx->get_status() != LLAMA_MEMORY_STATUS_FAILED_PREPARE,
                  "init_batch prepared (and reserved) the batch");
            // dropped HERE, without a single apply()
        }
        llama_batch_ext_free(be);

        ubatch_hold f(ntok, 1, 0);
        const auto sf = kv->prepare({ f.ub });
        check(!sf.empty(), "a prepare after the dropped context succeeded");
        if (!sf.empty()) {
            print_cells("prepare F", cells_of(sf[0]));
            check(sf[0].idxs[0][0] == 0,
                  "the dropped batch context RELEASED its cells (F starts at cell 0, not after the leak)");
        }
        kv->clear(true);
    }

    llama_free(borrower);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("\n%s\n", g_fail == 0 ? "PASS: prepare() reserves its cells while the cache is shared"
                                : "FAIL: see the lines above");
    return g_fail == 0 ? 0 : 1;
}
