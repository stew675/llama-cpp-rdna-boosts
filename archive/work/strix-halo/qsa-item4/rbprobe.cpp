// Rollback-restore probe: does a spec-verify batch + partial rollback leave the same state
// as a pure sequential decode?
//
//   path A (reference):  decode a (pos p), b (pos p+1), ... one at a time, then X at p+W
//   path B (verify):     decode batch [t_p .. t_{p+W-1}] in ONE call, seq_rm the last R
//                        positions, then decode X at p+(W-R)
//
// If the snapshot/rollback restores the state exactly, the X logits are bit-identical.
// env: W, R, RS, CTX, CTK, CTV, NGL, SPLIT
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fh(const float * a, int n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * p = (const unsigned char *) a;
    for (size_t i = 0; i < (size_t) n * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static ggml_type kvt(const char * n) {
    if (!strcmp(n,"f16")) return GGML_TYPE_F16;
    if (!strcmp(n,"q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(n,"bf16")) return GGML_TYPE_BF16;
    if (!strcmp(n,"q4_0")) return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}
static llama_context * mkctx(llama_model * model, int CTX, int ub, uint32_t rs, ggml_type tk, ggml_type tv) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = CTX; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.n_rs_seq = rs;
    cp.type_k = tk; cp.type_v = tv;
    return llama_init_from_model(model, cp);
}
static bool prefill(llama_context * ctx, const std::vector<llama_token> & toks, int P, int ub) {
    llama_batch b = llama_batch_init(ub, 0, 1);
    for (int i = 0; i < P; ++i) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = 0;
        b.n_tokens++;
        if (b.n_tokens == ub) {
            if (llama_decode(ctx, b) != 0) return false;
            b.n_tokens = 0;
        }
    }
    if (b.n_tokens > 0 && llama_decode(ctx, b) != 0) return false;
    llama_batch_free(b);
    return true;
}
// decode one token at `pos`
static bool dec1(llama_context * ctx, llama_token t, int pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1; b.token[0] = t; b.pos[0] = pos;
    b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model text P ubatch\n", argv[0]); return 1; }
    const char * mpath = argv[1]; const char * tpath = argv[2];
    const int P = atoi(argv[3]); const int ub = atoi(argv[4]);
    const int W = getenv("W") ? atoi(getenv("W")) : 4;
    const int R = getenv("R") ? atoi(getenv("R")) : 2;
    const uint32_t rs = getenv("RS") ? (uint32_t) atoi(getenv("RS")) : 0;
    const int CTX = getenv("CTX") ? atoi(getenv("CTX")) : 8192;
    const ggml_type tk = getenv("CTK") ? kvt(getenv("CTK")) : GGML_TYPE_F16;
    const ggml_type tv = getenv("CTV") ? kvt(getenv("CTV")) : tk;

    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    if (getenv("NGL")) mp.n_gpu_layers = atoi(getenv("NGL"));
    if (getenv("SPLIT") && !strcmp(getenv("SPLIT"), "layer")) mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    std::string text;
    { FILE * f = fopen(tpath, "rb"); if (!f) { perror("text"); return 1; }
      char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) text.append(b, r); fclose(f); }
    std::vector<llama_token> toks(262144);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    if (n < P + W + 4) { fprintf(stderr, "text too short: %d\n", n); return 1; }

    // tokens: verify batch = toks[P..P+W-1], kept prefix = toks[P..P+accept-1], probe = toks[P+accept]
    const int accept = W - R;                    // tokens kept after the rollback
    const llama_token probe = toks[P + accept];

    uint64_t hA = 0, hB = 0;
    {   // path A: sequential decode of the first `accept` tokens, then the probe token
        llama_context * c = mkctx(model, CTX, ub, rs, tk, tv);
        if (!c || !prefill(c, toks, P, ub)) { fprintf(stderr, "A failed\n"); return 1; }
        for (int i = 0; i < accept; ++i) if (!dec1(c, toks[P + i], P + i)) { fprintf(stderr, "A dec failed\n"); return 1; }
        if (!dec1(c, probe, P + accept)) { fprintf(stderr, "A probe failed\n"); return 1; }
        hA = fh(llama_get_logits_ith(c, -1), nv);
        llama_free(c);
    }
    {   // path B: one W-wide batch, then roll back R positions, then the probe token
        llama_context * c = mkctx(model, CTX, ub, rs, tk, tv);
        if (!c || !prefill(c, toks, P, ub)) { fprintf(stderr, "B failed\n"); return 1; }
        llama_batch b = llama_batch_init(W, 0, 1);
        b.n_tokens = W;
        for (int j = 0; j < W; ++j) {
            b.token[j] = toks[P + j]; b.pos[j] = P + j;
            b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
        }
        if (llama_decode(c, b) != 0) { fprintf(stderr, "B batch failed\n"); return 1; }
        llama_batch_free(b);
        if (R > 0) {
            llama_memory_t mem = llama_get_memory(c);
            if (!llama_memory_seq_rm(mem, 0, P + accept, -1)) { fprintf(stderr, "B seq_rm(%d..) returned false\n", P + accept); }
        }
        if (!dec1(c, probe, P + accept)) { fprintf(stderr, "B probe failed\n"); return 1; }
        hB = fh(llama_get_logits_ith(c, -1), nv);
        llama_free(c);
    }
    printf("[rb] W=%d R=%d RS=%u accept=%d  hA=%016llx hB=%016llx  %s\n",
           W, R, rs, accept, (unsigned long long) hA, (unsigned long long) hB,
           hA == hB ? "SAME" : "DIFFER");
    llama_model_free(model); llama_backend_free();
    return hA == hB ? 0 : 2;
}
