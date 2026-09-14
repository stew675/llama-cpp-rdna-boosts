// Real-model decode/verify width-purity matrix: load once, then for each W in a
// list, build a fresh context, prefill P tokens, decode a W-token batch, and
// hash the token-0 logits.  All W must agree for a width-invariant verify.
//
// usage: wallp model.gguf text.txt P [ubatch] W1,W2,...
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf text.txt P ubatch W1,W2,...\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const int P      = atoi(argv[3]);
    const int ubatch = atoi(argv[4]);
    std::vector<int> ws;
    { std::string s = argv[5]; size_t p = 0;
      while (p < s.size()) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size();
          ws.push_back(atoi(s.substr(p, c - p).c_str())); p = c + 1; } }

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
    if (n < P + 32) { fprintf(stderr, "text too short: %d < %d\n", n, P + 32); return 1; }

    for (int W : ws) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = P + 256 + W; cp.n_batch = ubatch + W; cp.n_ubatch = ubatch + W; cp.n_seq_max = 1;
        { const char * rs = getenv("RS");
          cp.n_rs_seq = (rs == nullptr) ? 0u : (uint32_t) atoi(rs);
          cp.n_rs_batch = cp.n_rs_seq + 1; }
        { const char * fa = getenv("FA");
          cp.flash_attn_type = (fa && !strcmp(fa, "0")) ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                              : (fa && !strcmp(fa, "1")) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                              : LLAMA_FLASH_ATTN_TYPE_AUTO; }
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        { const char * ck = getenv("CTK");
          if (ck) cp.type_k = !strcmp(ck,"bf16")?GGML_TYPE_BF16:!strcmp(ck,"q8_0")?GGML_TYPE_Q8_0:GGML_TYPE_F16; }
        { const char * cv = getenv("CTV");
          if (cv) cp.type_v = !strcmp(cv,"bf16")?GGML_TYPE_BF16:!strcmp(cv,"q8_0")?GGML_TYPE_Q8_0:GGML_TYPE_F16; }

        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "ctx init failed (W=%d)\n", W); return 1; }

        llama_batch b = llama_batch_init(cp.n_batch, 0, 1);
        for (int p0 = 0; p0 < P; p0 += ubatch) {
            const int n_b = (P - p0) < ubatch ? (P - p0) : ubatch;
            b.n_tokens = 0;
            for (int i = 0; i < n_b; ++i) {
                b.token[b.n_tokens] = toks[p0 + i]; b.pos[b.n_tokens] = p0 + i;
                b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 0;
                b.n_tokens++;
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed (W=%d)\n", W); return 1; }
        }
        b.n_tokens = 0;
        for (int j = 0; j < W; ++j) {
            b.token[b.n_tokens] = toks[P + j]; b.pos[b.n_tokens] = P + j;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 1;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed (W=%d)\n", W); return 1; }

        const float * l = llama_get_logits_ith(ctx, 0);
        uint64_t hh = 1469598103934665603ULL;
        const unsigned char * pp = (const unsigned char *) l;
        for (size_t bb = 0; bb < (size_t) n_vocab*sizeof(float); ++bb) { hh ^= pp[bb]; hh *= 1099511628211ULL; }
        printf("[L] W=%d P=%d logits0_hash=%016llx nv=%d\n", W, P, (unsigned long long) hh, n_vocab);

        llama_batch_free(b);
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
