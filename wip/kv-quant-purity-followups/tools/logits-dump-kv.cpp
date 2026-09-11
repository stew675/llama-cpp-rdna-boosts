// Per-node tensor dump probe: runs ONE decode-batch width (env W) after a fixed
// prefill and prints layout-aware digests of every computed node in execution
// order.  For each node we print h0..h3 where hp = hash of the elements whose
// index along dim p is 0 (i.e. the "slice" at 0 on dim p).  For a tensor whose
// batch/token dim is p, hp is the token-0 slice, which must be identical
// between W=1 and W=3.  Non-contiguous views are skipped (their linear bytes
// are not the logical elements).
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
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
    if (!strcmp(n,"mxfp4"))  return GGML_TYPE_MXFP4;
    fprintf(stderr, "unknown KV type '%s'\n", n); exit(2);
}

static int  g_w      = 0;
static bool g_enable = false;
static int  g_idx    = 0;

static bool dump_cb(ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        return g_enable;
    }
    if (!g_enable) {
        return true;
    }

    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const bool cont = ggml_is_contiguous(t);
    const size_t esz = ggml_element_size(t);
    const size_t nb  = (size_t) ne0 * ne1 * ne2 * ne3 * esz;

    char srcdim[80] = "";
    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1]) {
        snprintf(srcdim, sizeof srcdim, " K=%lld", (long long) t->src[1]->ne[0]);
    }
    if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[1]) {
        snprintf(srcdim, sizeof srcdim, " K=%lld ntok=%lld", (long long) t->src[1]->ne[0], (long long) t->ne[2]);
    }

    if (!cont) {
        printf("[D] W=%d idx=%d op=%-18s SKIP(noncontig) ne=[%lld,%lld,%lld,%lld]%s\n",
               g_w, g_idx++, ggml_op_name(t->op), (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3, srcdim);
        return true;
    }

    std::vector<unsigned char> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    uint64_t h[4] = { 1469598103934665603ULL, 1469598103934665603ULL, 1469598103934665603ULL, 1469598103934665603ULL };
    const int64_t dims[4] = { ne0, ne1, ne2, ne3 };
    for (size_t e = 0; e < (size_t) ne0*ne1*ne2*ne3; ++e) {
        size_t r = e;
        int64_t ix[4];
        for (int d = 0; d < 4; ++d) { ix[d] = (int64_t)(r % (size_t) dims[d]); r /= (size_t) dims[d]; }
        const unsigned char * p = buf.data() + e*esz;
        for (int d = 0; d < 4; ++d) {
            if (ix[d] == 0) { for (size_t b = 0; b < esz; ++b) { h[d] ^= p[b]; h[d] *= 1099511628211ULL; } }
        }
    }
    printf("[D] W=%d idx=%d op=%-18s CON ne=[%lld,%lld,%lld,%lld] h0=%016llx h1=%016llx h2=%016llx h3=%016llx%s\n",
           g_w, g_idx++, ggml_op_name(t->op), (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
           (unsigned long long) h[0], (unsigned long long) h[1], (unsigned long long) h[2], (unsigned long long) h[3], srcdim);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf text.txt [P=256] [ubatch=512]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const int P      = argc > 3 ? atoi(argv[3]) : 256;
    const int ubatch = argc > 4 ? atoi(argv[4]) : 512;

    g_w = getenv("W") ? atoi(getenv("W")) : 1;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    { const char * ng = getenv("NGL"); if (ng) mp.n_gpu_layers = atoi(ng); }
    { const char * split = getenv("SPLIT");
      if (split && !strcmp(split, "tensor")) mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
      else if (split && !strcmp(split, "layer")) mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
      else if (split && !strcmp(split, "row"))   mp.split_mode = LLAMA_SPLIT_MODE_ROW; }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    { FILE * f = fopen(text_path, "rb"); if (!f) { perror("text"); return 1; }
      char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) text.append(b, r); fclose(f); }

    std::vector<llama_token> toks(8192);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    if (n < P + 8) { fprintf(stderr, "text too short: %d\n", n); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = ubatch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
    { const char * rs = getenv("RS");
      cp.n_rs_seq = (rs == nullptr) ? 0u
                  : (strcmp(rs, "from_w") == 0) ? (uint32_t) (g_w - 1)
                  : (uint32_t) atoi(rs); }
    { const char * fa = getenv("FA");
      cp.flash_attn_type = (fa && (!strcmp(fa, "0") || !strcmp(fa, "off"))) ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                          : (fa && (!strcmp(fa, "1") || !strcmp(fa, "on"))) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                          : LLAMA_FLASH_ATTN_TYPE_AUTO; }
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    // revalidation extension: allow the KV type to be varied so the V4 (q8_0) and
    // V5 (bf16) native-staging arms can be exercised in the width matrix too.
    { const char * ck = getenv("CTK"); if (ck) cp.type_k = kv_type_from_name(ck); }
    { const char * cv = getenv("CTV"); if (cv) cp.type_v = kv_type_from_name(cv); }
    { const char * cb = getenv("CB"); if (cb == nullptr || strcmp(cb, "0") != 0) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; } }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    const int W = g_w;
    llama_batch b = llama_batch_init(ubatch, 0, 1);
    b.n_tokens = 0;
    for (int i = 0; i < P; ++i) {
        b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 0;
        b.n_tokens++;
    }
    g_enable = false;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
    b.n_tokens = 0;
    // REPEAT=1: batch = W copies of token P (so a batch-content dependence shows up as a
    // difference from the W=1 run; a pure width/kernel dependence does not care).
    const bool repeat = getenv("REPEAT") != nullptr && atoi(getenv("REPEAT")) != 0;
    for (int j = 0; j < W; ++j) {
        b.token[b.n_tokens] = repeat ? toks[P] : toks[P + j]; b.pos[b.n_tokens] = P + j;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 1;
        b.n_tokens++;
    }
    g_enable = true;
    { FILE * ff = fopen("/tmp/nodedump_on", "w"); if (ff) fclose(ff); fprintf(stderr, "[MARK] nodedump_on created\n"); }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }
    remove("/tmp/nodedump_on");

    {
        const float * l = llama_get_logits_ith(ctx, 0);
        const int nv = llama_vocab_n_tokens(vocab);
        uint64_t hh = 1469598103934665603ULL;
        const unsigned char * pp = (const unsigned char *) l;
        for (size_t bb2 = 0; bb2 < (size_t) nv*sizeof(float); ++bb2) { hh ^= pp[bb2]; hh *= 1099511628211ULL; }
        printf("[L] W=%d logits0_hash=%016llx nv=%d\n", W, (unsigned long long) hh, nv);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
