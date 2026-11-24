// STEP 2 probe: `llama_context_params::ctx_memory` -- the borrow (invariant I1/I2 of DESIGN.md §4).
//
// WHAT IT ASSERTS
//   1. IDENTITY (I1): a context created with `ctx_memory = a` reports `llama_get_memory(b) ==
//      llama_get_memory(a)` -- one memory object, not a view and not a copy.
//   2. THE NEGATIVE CONTROL: two contexts created WITHOUT the borrow hold two DIFFERENT memories.  Without
//      this arm, "the pointers are equal" proves nothing about the borrow (the trivial implementation of the
//      whole feature would be to make every context share one memory).
//   3. THE WORK (the point): the next-token argmax after a prompt is IDENTICAL whether the prompt is
//      prefilled on the owner and the next token decoded on the BORROWER, or both on the owner -- i.e. the
//      borrower really reads and extends the owner's cells.  Run single-threaded, so this is a *borrow* test
//      and not a locking test (STEP 3 interleaves two threads and owns that question).
//   4. THE CONTRACT REFUSALS: `ctx_memory` of a different model is refused; `ctx_memory` together with
//      `ctx_other` is refused (I8).  Both must return nullptr, not a half-built context.
//   5. TEARDOWN (I2): the borrower is freed BEFORE the owner, and the owner's memory still works afterwards
//      (a decode on the owner after the borrower is gone).  A borrower that freed the memory would fail
//      here; a borrower that freed it twice would abort.
//
// Build/run (see the STEP 2 record for the exact line):
//   <clang++> -std=c++17 -O1 probe-ctx-memory.cpp -I<repo>/include -I<repo>/ggml/include \
//       -L<build>/bin -lllama -lggml -Wl,-rpath,<build>/bin -o /tmp/probe-ctx-memory
//   HIP_VISIBLE_DEVICES=0 /tmp/probe-ctx-memory <model.gguf> [<model2.gguf>]

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        g_fail++;
    }
}

static llama_context * mk_ctx(llama_model * model, llama_context * borrow, bool ctx_type_mtp = false) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx        = 512;
    p.n_batch      = 512;
    p.n_ubatch     = 512;
    p.n_seq_max    = 1;
    p.n_threads       = 8;
    p.n_threads_batch = 8;
    p.no_perf      = true;
    p.ctx_memory   = borrow;
    if (ctx_type_mtp) {
        p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    return llama_init_from_model(model, p);
}

// one decode of `tokens` on `ctx`, with the last row's logits requested; returns the argmax of the last row
static int32_t decode_argmax(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0,
                             const llama_vocab * vocab) {
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (batch == nullptr) {
        printf("  FAIL llama_batch_ext_init\n");
        g_fail++;
        return -1;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        if (idx < 0) {
            printf("  FAIL llama_batch_ext_add_token -> %d\n", idx);
            g_fail++;
            llama_batch_ext_free(batch);
            return -1;
        }
        const llama_pos pos = pos0 + (llama_pos) i;
        if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
            printf("  FAIL llama_batch_ext_set_pos\n");
            g_fail++;
            llama_batch_ext_free(batch);
            return -1;
        }
        // only the last row's logits are needed
        (void) llama_batch_ext_set_output_logits(batch, idx, i + 1 == tokens.size());
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (rc != 0) {
        printf("  FAIL llama_process -> %d\n", rc);
        g_fail++;
        return -1;
    }
    const float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (logits == nullptr) {
        printf("  FAIL no logits for the last row\n");
        g_fail++;
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

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s MODEL.gguf [MODEL2.gguf]\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // a short prompt; the tokens are whatever this tokenizer makes of it
    const std::string prompt = "The capital of France is";
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(),
                           (int32_t) tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(),
                           (int32_t) tokens.size(), true, false);
    }
    if (n <= 0) {
        printf("FAIL: tokenize -> %d\n", n);
        return 1;
    }
    tokens.resize(n);
    printf("model loaded, prompt = %d tokens\n\n", n);

    llama_context * a = mk_ctx(model, nullptr);
    if (a == nullptr) { printf("FAIL: owner context\n"); return 1; }

    llama_memory_t mem_a = llama_get_memory(a);

    printf("1. identity (I1)\n");
    llama_context * b = mk_ctx(model, a);
    check(b != nullptr, "the borrower context was created");
    if (b == nullptr) { return 1; }
    llama_memory_t mem_b = llama_get_memory(b);
    printf("     owner %p memory %p | borrower %p memory %p\n", (void *) a, (void *) mem_a,
           (void *) b, (void *) mem_b);
    check(mem_a != nullptr && mem_a == mem_b, "llama_get_memory(borrower) == llama_get_memory(owner)");

    printf("\n2. negative control: a context WITHOUT the borrow gets its own memory\n");
    llama_context * c = mk_ctx(model, nullptr);
    check(c != nullptr, "the second owner context was created");
    if (c == nullptr) { return 1; }
    llama_memory_t mem_c = llama_get_memory(c);
    check(mem_c != nullptr && mem_c != mem_a, "an independent context's memory differs from the owner's");

    printf("\n3. the borrow does the work\n");
    // The two arms must be INDEPENDENT (a second arm that reuses the first arm's cells would pass for the
    // wrong reason), so each arm gets a fresh (owner, borrower) pair and runs the same three steps:
    //     prefill the prompt -> sample t1 -> decode t1 -> sample t2.
    // Arm 1 does all three on the owner; arm 2 prefills on its owner, then decodes t1 AND t2 on the BORROWER.
    // The claim: t1 and t2 are identical in both arms, i.e. the borrower reads the cells the owner wrote and
    // extends them at the right positions.
    // arm 1: everything on the owner
    const int32_t t1 = decode_argmax(a, tokens, 0, vocab);
    std::vector<llama_token> one = { (llama_token) t1 };
    const int32_t t2 = decode_argmax(a, one, (llama_pos) tokens.size(), vocab);
    printf("     owner-only:  prompt -> t1=%d -> t2=%d\n", t1, t2);

    // arm 2: a fresh pair; the prompt on its owner, the two decoded rows on its BORROWER
    llama_context * a2 = mk_ctx(model, nullptr);
    llama_context * b2 = a2 != nullptr ? mk_ctx(model, a2) : nullptr;
    check(a2 != nullptr && b2 != nullptr, "the second (owner, borrower) pair was created");
    int32_t t1b = -1, t2b = -1;
    if (a2 != nullptr && b2 != nullptr) {
        t1b = decode_argmax(a2, tokens, 0, vocab);
        std::vector<llama_token> oneb = { (llama_token) t1b };
        t2b = decode_argmax(b2, oneb, (llama_pos) tokens.size(), vocab);
        printf("     borrower arm: prompt(owner) -> t1=%d -> t2=%d (decoded on the BORROWER)\n", t1b, t2b);
    }
    check(t1 >= 0 && t2 >= 0 && t1b >= 0 && t2b >= 0, "every decode produced logits");
    check(t1 == t1b, "t1 (sampled from the owner's prefill) is the same from either arm");
    check(t2 == t2b, "t2 decoded ON THE BORROWER equals t2 decoded on the owner (the cells were shared)");

    // the owner continues from the borrower's row: position tokens.size()+1 on the FIRST pair
    std::vector<llama_token> one2 = { (llama_token) t2b };
    const int32_t after = decode_argmax(a, one2, (llama_pos) tokens.size() + 1, vocab);
    check(after >= 0, "the owner can continue after the borrower's row");

    if (b2 != nullptr) { llama_free(b2); }
    if (a2 != nullptr) { llama_free(a2); }

    printf("\n4. contract refusals\n");
    {
        // ctx_memory + ctx_other on one context is a contradiction (I8)
        llama_context_params p = llama_context_default_params();
        p.n_ctx = 256; p.n_batch = 256; p.n_ubatch = 256;
        p.ctx_memory = a;
        p.ctx_other  = c;
        p.no_perf    = true;
        llama_context * bad = llama_init_from_model(model, p);
        check(bad == nullptr, "ctx_memory together with ctx_other is refused");
        if (bad != nullptr) { llama_free(bad); }
    }
    if (argc >= 3) {
        llama_model * other = llama_model_load_from_file(argv[2], mp);
        if (other == nullptr) {
            printf("  SKIP the different-model refusal: could not load %s\n", argv[2]);
        } else {
            llama_context_params p = llama_context_default_params();
            p.n_ctx = 256; p.n_batch = 256; p.n_ubatch = 256;
            p.ctx_memory = a;          // belongs to `model`, not to `other`
            p.no_perf = true;
            llama_context * bad = llama_init_from_model(other, p);
            check(bad == nullptr, "ctx_memory of a DIFFERENT model is refused");
            if (bad != nullptr) { llama_free(bad); }
            llama_model_free(other);
        }
    } else {
        printf("  SKIP the different-model refusal (pass a second model path to run it)\n");
    }

    printf("\n5. teardown order (I2): free the borrower first, the owner must still work\n");
    llama_free(b);
    // decode on the owner after the borrower is gone: a borrower that freed or damaged the memory fails here
    std::vector<llama_token> one3 = { (llama_token) after };
    const int32_t after2 = decode_argmax(a, one3, (llama_pos) tokens.size() + 2, vocab);
    check(after2 >= 0, "the owner still decodes after the borrower was freed");

    llama_free(c);
    llama_free(a);
    llama_model_free(model);
    llama_backend_free();

    printf("\n%s\n", g_fail == 0 ? "PASS: ctx_memory borrows one memory object and changes nothing else"
                                : "FAIL: see the lines above");
    return g_fail == 0 ? 0 : 1;
}
