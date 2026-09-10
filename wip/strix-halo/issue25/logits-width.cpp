// Direct target-logits batch-width probe (issue #25).
//
// Feeds an identical prefix (one prefill batch) and then decodes a batch of W
// tokens [t_P .. t_{P+W-1}]. Row j of that batch sees the exact same context
// regardless of W, so the logits at row 0/1/2 must be identical for every W if
// the batched forward is width-consistent. Any difference is the verify-batch
// width dependence, isolated from MTP/spec logic entirely.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int argmax(const float * a, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (a[i] > a[best]) best = i;
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf text.txt [P=256] [ubatch=512]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const char * text_path  = argv[2];
    const int P       = argc > 3 ? atoi(argv[3]) : 256;
    const int ubatch  = argc > 4 ? atoi(argv[4]) : 512;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    { FILE * f = fopen(text_path, "rb"); if (!f) { perror("text"); return 1; }
      char buf[65536]; size_t r; while ((r = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, r); fclose(f); }

    std::vector<llama_token> toks(8192);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
    if (n < 0) { fprintf(stderr, "tokenize failed (need %d)\n", -n); return 1; }
    if (n < P + 8) { fprintf(stderr, "text too short: %d tokens, need %d\n", n, P + 8); return 1; }
    printf("tokens=%d P=%d ubatch=%d\n", n, P, ubatch);

    const int widths[] = {1, 3, 5};
    std::vector<std::vector<float>> rows[3]; // per width: captured rows
    std::vector<int> rownum[3];

    for (int wi = 0; wi < 3; ++wi) {
        const int W = widths[wi];
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = ubatch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
        // Replicate the MTP verify-batch recurrent-state snapshot count:
        // common sets n_rs_seq = draft.n_max, so a W-token batch with n_max=W-1
        // uses n_rs_seq = W-1 (K = n_rs_seq+1).  RS=zero forces plain decode;
        // RS=<n> pins a fixed n_rs_seq.
        {
            const char * rs = getenv("RS");
            cp.n_rs_seq = (rs == nullptr) ? 0u
                        : (strcmp(rs, "from_w") == 0) ? (uint32_t) (W - 1)
                        : (uint32_t) atoi(rs);
        }
        const char * fa = getenv("FA");
        cp.flash_attn_type = (fa && (!strcmp(fa, "0") || !strcmp(fa, "off"))) ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                            : (fa && (!strcmp(fa, "1") || !strcmp(fa, "on"))) ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                            : LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

        llama_batch b = llama_batch_init(ubatch, 0, 1);
        // prefill t[0..P-1]
        b.n_tokens = 0;
        for (int i = 0; i < P; ++i) {
            b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 0;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
        // decode batch [t_P .. t_{P+W-1}]
        b.n_tokens = 0;
        for (int j = 0; j < W; ++j) {
            b.token[b.n_tokens] = toks[P + j]; b.pos[b.n_tokens] = P + j;
            b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = 1;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }
        const int nv = llama_vocab_n_tokens(vocab);
        for (int j = 0; j < W; ++j) {
            float * l = llama_get_logits_ith(ctx, j);
            rows[wi].push_back(std::vector<float>(l, l + nv));
            rownum[wi].push_back(j);
        }
        printf("W=%d n_rs_seq=%u captured %d rows\n", W, cp.n_rs_seq, W);
        llama_batch_free(b);
        llama_free(ctx);
    }

    // rows are aligned by j (context index) across widths
    printf("\n%-4s %-14s %-14s %-10s %-10s %-10s\n", "row", "max|W1-W3|", "max|W3-W5|", "W1-argmax", "W3-argmax", "W5-argmax");
    int diffs = 0;
    for (int j = 0; j < 3; ++j) {
        const std::vector<float> & a  = rows[0][j < (int) rows[0].size() ? j : 0];
        const std::vector<float> & b3 = rows[1][j];
        const std::vector<float> & b5 = rows[2][j];
        float d35 = 0; for (size_t k = 0; k < b3.size(); ++k) d35 = std::max(d35, std::fabs(b3[k] - b5[k]));
        float d13 = -1.0f;
        if (j < (int) rows[0].size()) { d13 = 0; for (size_t k = 0; k < a.size(); ++k) d13 = std::max(d13, std::fabs(a[k] - b3[k])); }
        int am1 = j < (int) rows[0].size() ? argmax(a.data(), a.size()) : -1;
        int am3 = argmax(b3.data(), b3.size()), am5 = argmax(b5.data(), b5.size());
        printf("%-4d %-14.6f %-14.6f %-10d %-10d %-10d\n", j, d13, d35, am1, am3, am5);
        if (d35 > 0) diffs++;
    }
    printf("\nW3-vs-W5 rows 0..2: %s\n", diffs ? "NONZERO (verify batches differ by width)" : "all 0 (verify batches bit-consistent)");

    llama_model_free(model);
    llama_backend_free();
    return diffs ? 2 : 0;
}
