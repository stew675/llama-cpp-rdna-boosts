// Deterministic logit dumper for the managed-lazy validation.
// Usage: logitdump -m MODEL -o OUT [-p PROMPT | -n NTOKENS] [--seed S] [--sample K]
//                 [-ngl N] [-c N] [--split-mode M] [--lazy-buffer-size N(K|M|G)]
// Dumps logits for sampled token positions (every K-th plus the last) to OUT.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static size_t parse_size(const char * s) {
    size_t mult = 1;
    std::string num(s);
    if (!num.empty()) {
        const char c = num.back();
        switch (c) {
            case 'K': case 'k': mult = 1024;              num.pop_back(); break;
            case 'M': case 'm': mult = 1024*1024;         num.pop_back(); break;
            case 'G': case 'g': mult = 1024ul*1024*1024;  num.pop_back(); break;
            default: break;
        }
    }
    return (size_t) strtoull(num.c_str(), nullptr, 10) * mult;
}

int main(int argc, char ** argv) {
    std::string model_path, prompt, out_path;
    int ngl = 0;
    int n_ctx = 4096;
    int n_tokens = 0;            // 0 = use the prompt
    uint32_t seed = 1234;
    int sample_k = 0;            // 0 = dump every position
    int repeat = 0;              // repeat a 512-token block (locality for cache hits)
    size_t lazy_buf = 0;
    llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m" && i+1 < argc) { model_path = argv[++i]; }
        else if (a == "-p" && i+1 < argc) { prompt = argv[++i]; }
        else if (a == "-o" && i+1 < argc) { out_path = argv[++i]; }
        else if (a == "-n" && i+1 < argc) { n_tokens = atoi(argv[++i]); }
        else if (a == "--seed" && i+1 < argc) { seed = (uint32_t) atoi(argv[++i]); }
        else if (a == "--sample" && i+1 < argc) { sample_k = atoi(argv[++i]); }
        else if (a == "--repeat" && i+1 < argc) { repeat = atoi(argv[++i]); }
        else if (a == "-ngl" && i+1 < argc) { ngl = atoi(argv[++i]); }
        else if (a == "-c" && i+1 < argc) { n_ctx = atoi(argv[++i]); }
        else if (a == "--lazy-buffer-size" && i+1 < argc) { lazy_buf = parse_size(argv[++i]); }
        else if (a == "--split-mode" && i+1 < argc) { split_mode = (llama_split_mode) atoi(argv[++i]); }
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            fprintf(stderr, "usage: logitdump -m MODEL -o OUT [-p PROMPT | -n NTOKENS] [--seed S] [--sample K] [-ngl N] [-c N] [--split-mode M] [--lazy-buffer-size N]\n");
            return 1;
        }
    }
    if (model_path.empty() || out_path.empty() || (prompt.empty() && n_tokens == 0)) {
        fprintf(stderr, "usage: logitdump -m MODEL -o OUT [-p PROMPT | -n NTOKENS] [--seed S] [--sample K] [-ngl N] [-c N] [--split-mode M] [--lazy-buffer-size N]\n");
        return 1;
    }
    if (n_tokens > 0) {
        n_ctx = std::max(n_ctx, n_tokens + 32);
    }

    fprintf(stderr, "logitdump: loading %s (ngl=%d, lazy_buf=%zu, split=%d)\n", model_path.c_str(), ngl, lazy_buf, (int) split_mode);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers    = ngl;
    mp.n_lazy_buf_size = lazy_buf;
    mp.split_mode      = split_mode;
    llama_model * model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = n_ctx;
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 16;
    cp.n_threads_batch = 16;
    llama_context * ctx = llama_new_context_with_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // token stream: either the tokenized prompt, or a seeded random stream
    std::vector<llama_token> toks;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (n_tokens > 0) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> dis(0, 200000);
        std::vector<llama_token> block;
        if (repeat > 0) {
            block.resize(512);
            for (int i = 0; i < 512; i++) {
                block[i] = (llama_token) dis(gen);
            }
        }
        toks.reserve(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            toks.push_back(repeat > 0 ? block[i % 512] : (llama_token) dis(gen));
        }
    } else {
        const int32_t probe = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, false);
        const int32_t n = probe < 0 ? -probe : probe;
        toks.resize(n > 0 ? n : 0);
        if (n > 0) {
            llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, true, false);
        }
    }
    if (toks.empty()) {
        fprintf(stderr, "empty token stream\n");
        return 1;
    }
    fprintf(stderr, "logitdump: %zu tokens\n", toks.size());

    // sample positions: every K-th plus the last
    std::vector<bool> want_logits(toks.size(), false);
    std::vector<size_t> sampled;
    for (size_t i = 0; i < toks.size(); i++) {
        if (sample_k > 0 ? (i % sample_k == 0) : true) {
            want_logits[i] = true;
            sampled.push_back(i);
        }
    }
    want_logits[toks.size() - 1] = true;
    if (sampled.empty() || sampled.back() != toks.size() - 1) {
        sampled.push_back(toks.size() - 1);
    }

    // decode in n_batch chunks (llama_decode requires n_batch >= batch size);
    // positions accumulate in the KV cache across chunks
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> logits_storage;
    logits_storage.reserve(sampled.size() * (size_t) n_vocab);

    const auto t0 = llama_time_us();
    llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);
    size_t chunk_start = 0;
    while (chunk_start < toks.size()) {
        const size_t chunk_end = std::min(chunk_start + cp.n_batch, toks.size());
        const size_t n_chunk = chunk_end - chunk_start;
        for (size_t i = 0; i < n_chunk; i++) {
            batch.token[i]     = toks[chunk_start + i];
            batch.pos[i]       = (llama_pos) (chunk_start + i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = want_logits[chunk_start + i];
        }
        batch.n_tokens = n_chunk;
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode failed at %zu\n", chunk_start);
            return 1;
        }
        // harvest this chunk's sampled positions
        for (size_t i : sampled) {
            if (i >= chunk_start && i < chunk_end) {
                const float * lg = llama_get_logits_ith(ctx, i - chunk_start);
                logits_storage.insert(logits_storage.end(), lg, lg + n_vocab);
            }
        }
        chunk_start = chunk_end;
    }
    const double dt = (llama_time_us() - t0) / 1e6;

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path.c_str());
        return 1;
    }
    uint64_t pos_u64 = sampled.size();
    fwrite(&pos_u64, 8, 1, f);
    fwrite(logits_storage.data(), sizeof(float), logits_storage.size(), f);
    fclose(f);
    fprintf(stderr, "logitdump: wrote %zu sampled positions x %lld floats to %s (%.1f tok/s)\n",
            sampled.size(), (long long) n_vocab, out_path.c_str(), toks.size() / dt);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    return 0;
}
