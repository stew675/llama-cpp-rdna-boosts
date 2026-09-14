// Width-purity probe for the decode/verify band.
//
// Prefills P tokens (P > the QSA selection width by default), then decodes a
// batch of W tokens and hashes the token-0 logits.  For a width-invariant
// decode/verify path the token-0 logits must be identical for every W: a real
// speculative verify of W rows must give the same first-row result as a W=1
// decode of the same state.
//
// env: W (batch width), P (prefill), RS (n_rs_seq), CTK/CTV, SPLIT, NGL, FA,
//      EXTRA (tokens appended to P to cross the 2051 selection width)
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_type kv_type_from_name(const char * n) {
    if (!strcmp(n,"f16"))    return GGML_TYPE_F16;
    if (!strcmp(n,"bf16"))   return GGML_TYPE_BF16;
    if (!strcmp(n,"q8_0"))   return GGML_TYPE_Q8_0;
    if (!strcmp(n,"q4_0"))   return GGML_TYPE_Q4_0;
    if (!strcmp(n,"q4_1"))   return GGML_TYPE_Q4_1;
    if (!strcmp(n,"q5_0"))   return GGML_TYPE_Q5_0;
    if (!strcmp(n,"q5_1"))   return GGML_TYPE_Q5_1;
    if (!strcmp(n,"iq4_nl")) return GGML_TYPE_IQ4_NL;
    fprintf(stderr, "unknown KV type '%s'\n", n); exit(2);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf text.txt [P=3000] [ubatch=4096]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const int P      = argc > 3 ? atoi(argv[3]) : 3000;
    const int ubatch = argc > 4 ? atoi(argv[4]) : 4096;
    const int W      = getenv("W") ? atoi(getenv("W")) : 1;
    const int n_ctx  = P + 256;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    { const char * ng = getenv("NGL"); if (ng) mp.n_gpu_layers = atoi(ng); }
    { const char * split = getenv("SPLIT");
      if (split && !strcmp(split, "tensor")) mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
      else if (split && !strcmp(split, "layer")) mp.split_mode = LLAMA_SPLIT_MODE_LAYER; }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    { FILE * f = fopen(text_path, "rb"); if (!f) { perror("text"); return 1; }
      char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) text.append(b, r); fclose(f); }

    std::vector<llama_token> toks(65536);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n = 0;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        n = P + 32;
        for (int i = 0; i < n; ++i) toks[i] = (llama_token) ((7*i + 3) % n_vocab);
    } else {
        n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    }
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    if (n < P + 32) { fprintf(stderr, "text too short: %d < %d\n", n, P + 32); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = ubatch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
    { const char * rs = getenv("RS");
      cp.n_rs_seq = (rs == nullptr) ? 0u : (uint32_t) atoi(rs);
      cp.n_rs_batch = cp.n_rs_seq + 1; }
    { const char * fa = getenv("FA");
      cp.flash_attn_type = (fa && !strcmp(fa, "0")) ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                          : (fa && !strcmp(fa, "1")) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                          : LLAMA_FLASH_ATTN_TYPE_AUTO; }
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    { const char * ck = getenv("CTK"); if (ck) cp.type_k = kv_type_from_name(ck); }
    { const char * cv = getenv("CTV"); if (cv) cp.type_v = kv_type_from_name(cv); }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    // prefill in chunks of ubatch so a large P is fine
    llama_batch b = llama_batch_init(ubatch, 0, 1);
    for (int p0 = 0; p0 < P; p0 += ubatch) {
        const int n_b = (P - p0) < ubatch ? (P - p0) : ubatch;
        b.n_tokens = 0;
        for (int i = 0; i < n_b; ++i) {
            b.token[b.n_tokens] = toks[p0 + i]; b.pos[b.n_tokens] = p0 + i;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 0;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
    }

    // decode the width-W batch at pos P..P+W-1
    b.n_tokens = 0;
    { const char * extra = getenv("EXTRA");
      if (extra) { for (int i = 0; i < atoi(extra); ++i) {
          b.token[b.n_tokens] = toks[P + i]; b.pos[b.n_tokens] = P + i;
          b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 0;
          b.n_tokens++;
      } } }
    for (int j = 0; j < W; ++j) {
        b.token[b.n_tokens] = toks[P + j]; b.pos[b.n_tokens] = P + j;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 1;
        b.n_tokens++;
    }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }

    const int n_kv = (int) llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    {
        const float * l = llama_get_logits_ith(ctx, 0);
        const int nv = llama_vocab_n_tokens(vocab);
        uint64_t hh = 1469598103934665603ULL;
        const unsigned char * pp = (const unsigned char *) l;
        for (size_t bb = 0; bb < (size_t) nv*sizeof(float); ++bb) { hh ^= pp[bb]; hh *= 1099511628211ULL; }
        printf("[L] W=%d P=%d n_kv=%d logits0_hash=%016llx nv=%d\n", W, P, n_kv, (unsigned long long) hh, nv);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
