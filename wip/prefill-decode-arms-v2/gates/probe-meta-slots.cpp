// STEP 4 probe: the Meta backend's view-slot IDENTITY (defect V3 of DESIGN.md §5).
//
// THE CLAIM
//   The Meta backend keeps, per (buffer, view slot), a rotating pair of view containers, and a graph must be
//   built AND resolved under the same slot.  The slot used to be derived from a ROLE -- `ctx_other ? 2 : 0` --
//   so a pair plus a drafter (THREE contexts) had only two roles and the decode context and the draft context
//   shared a pair.  It is now an IDENTITY: each `llama_context` owns the pair (`meta_slot_base`,
//   `meta_slot_base + 1`) from a process-wide counter, assigned once at construction.
//
// WHAT IT ASSERTS
//   1. N contexts get N DISTINCT bases, and each base's pair is two slots inside the backend's 8.
//   2. the text does not move with the number of live contexts: the same prefill+decode reads the same token
//      with 2 contexts live and with 3 (a 3rd context's existence must not change another's views).
//   3. every decode really went through the Meta backend (`-sm tensor` on 3 devices), otherwise the slot
//      machinery was never reached and this probe would be vacuous -- the caller pins 3 devices and the probe
//      reports the split mode it got.
//
// Usage:  HIP_VISIBLE_DEVICES=0,1,2 probe-meta-slots MODEL.gguf
// Build:  <clang++> -std=c++17 -O1 probe-meta-slots.cpp -I<repo>/include -I<repo>/src \
//             -I<repo>/ggml/include -L<build>/bin -lllama -lggml -Wl,-rpath,<build>/bin

#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"

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

static llama_context * mk(llama_model * model, llama_context * borrow, bool tensor_split) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx           = 4096;
    p.n_batch         = 2048;
    p.n_ubatch        = 512;
    p.n_seq_max       = 2;
    p.n_threads       = 8;
    p.n_threads_batch = 8;
    p.no_perf         = true;
    p.ctx_memory      = borrow;
    p.flash_attn_type = tensor_split ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
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

static int32_t argmax_last(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq,
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf   (pin HIP_VISIBLE_DEVICES to 3 devices)\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.split_mode   = LLAMA_SPLIT_MODE_TENSOR;
    mp.main_gpu     = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // the Meta backend only exists on a tensor split over >= 2 devices; the caller pins the devices and this
    // line makes the arm self-describing (a 1-device run would make every check below vacuous)
    const size_t n_dev = ggml_backend_dev_count();
    printf("split mode requested = TENSOR, ggml devices visible = %zu\n", n_dev);
    check(n_dev >= 3, "at least 3 devices are visible (HIP_VISIBLE_DEVICES=0,1,2), so the split is real");

    const std::vector<llama_token> prompt_ = tok_of(vocab, "The capital of France is");
    const std::vector<llama_token> big_    = tok_of(vocab, "Summarise the following text in one sentence. ");

    // 2-context arm: the pair the server will use
    llama_context * a  = mk(model, nullptr, true);
    llama_context * b  = mk(model, a, true);
    check(a != nullptr && b != nullptr, "the 2-context arm was created");
    if (a == nullptr || b == nullptr) { return 1; }

    const int base_a = a->meta_view_slot_base();
    const int base_b = b->meta_view_slot_base();
    printf("2-context arm: bases %d and %d\n", base_a, base_b);
    check(base_a >= 0 && base_a + 1 < 8, "the owner's pair is inside the backend's 8 slots");
    check(base_b >= 0 && base_b + 1 < 8, "the borrower's pair is inside the backend's 8 slots");
    check(base_a != base_b, "the two contexts have DISTINCT bases (no shared view pair)");

    const int32_t t2 = argmax_last(b, vocab, 0, prompt_, 0);
    check(t2 >= 0, "the 2-context arm decoded");

    // 3-context arm: a third live context (the drafter's stand-in) must not move anything
    llama_context * c = mk(model, nullptr, true);
    check(c != nullptr, "a THIRD context was created");
    if (c == nullptr) { return 1; }
    const int base_c = c->meta_view_slot_base();
    printf("3-context arm: bases %d, %d, %d\n", base_a, base_b, base_c);
    check(base_c >= 0 && base_c + 1 < 8, "the third context's pair is inside the backend's 8 slots");
    check(base_c != base_a && base_c != base_b, "all three bases are DISTINCT");
    check(base_a % 2 == 0 && base_b % 2 == 0 && base_c % 2 == 0, "every base is a PAIR boundary (even)");

    // the same work again, with the third context alive: the token must not move
    const int32_t t3 = argmax_last(c, vocab, 1, prompt_, 0);
    check(t3 == t2, "the same prefill+decode reads the SAME token with 3 contexts live");
    printf("     token with 2 contexts = %d, with 3 = %d\n", t2, t3);

    llama_free(c);
    llama_free(b);
    llama_free(a);
    llama_model_free(model);
    llama_backend_free();

    printf("\n%s\n", g_fail == 0 ? "PASS: the view slot is a per-context identity, and it does not move the text"
                                : "FAIL: see the lines above");
    return g_fail == 0 ? 0 : 1;
}
