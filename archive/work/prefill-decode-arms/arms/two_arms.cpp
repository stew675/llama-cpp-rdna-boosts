// two_arms -- WIP instrument for the prefill-decode-arms campaign (NOT part of the delivery).
//
// It is the campaign's design WITHOUT the server: one process, one model, ONE prefill arm (its own context +
// thread) and ONE decode arm (its own context + thread, sharing the prefill arm's KV), with a request queue
// between them and the sequence state handed over at the request boundary.
//
//   prefill arm  ctx A  n_batch = n_ubatch = (the model's prefill width)   owns the prompts
//   decode arm   ctx B  n_batch = n_ubatch = ARMS_B_BATCH (decode only)    owns the sampled tokens
//
// The handover is the campaign's T1: A's llama_state_seq_get_data_ext() writes the sequence's ATTENTION KV and
// its RECURRENT state; B (a co-owner of the KV, `ctx_other_share_memory`) SKIPS the KV half on read -- its
// llama_kv_cache returns early with `other != nullptr` -- so what actually crosses the boundary is exactly the
// recurrent state.  That is the whole of T1, and `ARMS_MODE=handover` verifies it against the single-context
// text oracle before anything else runs.
//
// Modes (ARMS_MODE):
//   single   : ONE context (ctx A only), every request run sequentially -- the TEXT REFERENCE
//   arms     : ctx A prefills a request's prompt, hands the sequence to ctx B, ctx B decodes it; the next
//              request's prompt is prefilled on A WHILE B decodes == the campaign's product shape
//   handover : one request; prefill on A, hand over, decode on B (the T1/T3 oracle, with the handover timed)
//   pf-alone : ctx A alone runs the prompts (the prefill reference wall)
//   dec-alone: ctx B alone runs the decoder's generation (the decode reference wall)
//
// Scenario: ARMS_SCENARIO="<prompt_tokens>:<n_predict>@<admit_delay_ms>,..." -- each item is one request.  The
// prompt is two_ctx's 4932-token one (TWO_CTX_PF_FILE / arms/p4k prompt), repeated if a request asks for more.
//
// Text oracle: every request's generated token ids are hashed; `arms`/`handover` must read the same hashes as
// `single` for the same scenario.  Purity pins for a host-expert arm: MOE_EXPERT_HEATMAP=load and
// ROCBLAS_USE_HIPBLASLT=0 (r310).
//
// Build: ./build-two-arms.sh   (links ~/llama.cpp/build-2ctx of the campaign's WIP branch)

#include "common.h"
#include "arg.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

// --- tiny sha256 (FIPS 180-4), so the tool needs no external crypto lib -------------------------------
struct sha256_ctx {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t len = 0;
    uint8_t buf[64] = {0};
    size_t buflen = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) | (uint32_t(p[4*i+2]) << 8) | uint32_t(p[4*i+3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19)  ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        len += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof(buf) - buflen);
            memcpy(buf + buflen, p, take);
            buflen += take; p += take; n -= take;
            if (buflen == 64) { block(buf); buflen = 0; }
        }
    }

    std::string final() {
        const uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (buflen != 56) { update(&z, 1); }
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) { lenb[i] = uint8_t((bits >> (56 - 8*i)) & 0xff); }
        update(lenb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) { snprintf(out + 8*i, 9, "%08x", h[i]); }
        return std::string(out);
    }
};

static std::string sha_of(const std::vector<llama_token> & toks) {
    sha256_ctx c;
    for (auto t : toks) {
        const int32_t v = (int32_t) t;
        c.update(&v, sizeof(v));
    }
    return c.final();
}

static double now_ms() { return ggml_time_us() / 1000.0; }

static double ipct(std::vector<double> xs, double p) {
    if (xs.empty()) { return 0.0; }
    std::sort(xs.begin(), xs.end());
    const size_t k = std::min(xs.size() - 1, (size_t) (p / 100.0 * (xs.size() - 1) + 0.5));
    return xs[k];
}

static double vmax(const std::vector<double> & xs) {
    return xs.empty() ? 0.0 : *std::max_element(xs.begin(), xs.end());
}

// --- one request ---------------------------------------------------------------------------------------
struct request_t {
    int  id = 0;
    size_t n_prompt_tok = 0;   // how many prompt tokens to use
    int  n_predict = 32;
    double admit_ms = 0.0;     // absolute, when the request becomes available to the prefill arm

    llama_seq_id seq = -1;
    double t_admit_real = 0.0; // absolute, when the prefill arm actually picked it up
    double t_prompt_start = 0.0;
    double t_prompt_done = 0.0;
    double t_handover = 0.0;   // handover wall
    size_t handover_bytes = 0;
    bool   handover_device = false;
    // the handover is EXECUTED by the decode arm (see arms_t): the prefill arm only posts it, and waits for
    // handover_done before it starts the next prompt (so a take never reads a memory the prefill arm is
    // allocating cells in)
    bool   handover_posted = false;
    bool   handover_done   = false;
    double t_first_tok = 0.0;

    // decode-arm bookkeeping
    llama_token pending = -1;  // token to ingest next
    llama_pos   pos = 0;       // position of `pending`
    bool        on_decode_arm = false;
    bool        done = false;

    std::vector<llama_token> generated;
    std::vector<double>      arrivals_ms;
};

// --- the shared state between the two arms -------------------------------------------------------------
struct arms_t {
    std::mutex              m;
    std::condition_variable cv_ready;    // the decode arm waits for a handed-over sequence
    std::condition_variable cv_done;     // the prefill arm waits for the decode arm to drain at the end
    std::condition_variable cv_taken;    // the prefill arm waits for a posted handover to be executed
    size_t n_taken = 0;                  // handovers executed
    bool   wait_all = false;             // ARMS_WAIT_ALL: do not step until all handovers are done
    std::vector<request_t> * reqs = nullptr;
    size_t   n_prefilled = 0;
    bool     prefill_done = false;
    bool     decode_done  = false;

    // the handover must not run while the decode arm is mutating ITS memory module's cell metadata (the two
    // memories are separate objects: only the attention KV cells are shared).  A step is short (one token per
    // active slot), so the prefill arm waits at most one step at a request boundary.
    std::mutex mem_m;

    // prefill windows for the "ITL during a prefill" metric
    std::mutex pf_m;
    std::vector<std::pair<double, double>> pf_windows;
};

static bool in_any_window(const std::vector<std::pair<double, double>> & w, double t) {
    for (const auto & p : w) {
        if (t >= p.first && t <= p.second) { return true; }
    }
    return false;
}

// --- greedy sampling (the campaign's text oracle is plain greedy) --------------------------------------
//
// ARMS_LOGITS_DIR=<dir>: dump every row the arms SAMPLE (the prompt's last row and every decode step) as raw
// float32 into <dir>/<mode>-r<rep>-q<req>-s<step>.f32 and append one line per row to <dir>/manifest.txt with
// the row's sha256, its argmax and its top-5.  This is the diff instrument for RESULTS-2026-11-21.md §3.4 item
// 3: the SAME token decoded on ctx A and on ctx B produces two rows, and the rows (not the sampled ids, which
// are identical by construction when the text matches) carry the numeric difference.  Instrument only.
static std::string g_logits_dir;
static std::string g_mode_tag = "arms";
static int         g_rep_tag  = 0;
static std::mutex  g_logits_m;

static llama_token argmax_of(const float * logits, int n_vocab) {
    llama_token best = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[best]) { best = v; }
    }
    return best;
}

static llama_token sample_row(llama_context * ctx, int32_t i, int req, int step) {
    const float * logits = llama_get_logits_ith(ctx, i);
    if (logits == nullptr) { return -1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const llama_token best = argmax_of(logits, n_vocab);

    if (g_logits_dir.empty()) { return best; }

    char name[128];
    snprintf(name, sizeof(name), "%s-r%d-q%d-s%d.f32", g_mode_tag.c_str(), g_rep_tag, req, step);
    const std::string path = g_logits_dir + "/" + name;
    if (FILE * fp = fopen(path.c_str(), "wb")) {
        fwrite(logits, sizeof(float), (size_t) n_vocab, fp);
        fclose(fp);
    }

    sha256_ctx h;
    h.update(logits, sizeof(float) * (size_t) n_vocab);

    // the top-5 of the row: the numeric difference is often invisible in the argmax
    std::vector<int> top(5, -1);
    for (int v = 0; v < n_vocab; ++v) {
        for (int k = 0; k < 5; ++k) {
            if (top[k] < 0 || logits[v] > logits[top[k]]) {
                for (int j = 4; j > k; --j) { top[j] = top[j - 1]; }
                top[k] = v;
                break;
            }
        }
    }
    char tops[256];
    tops[0] = 0;
    for (int k = 0; k < 5 && top[k] >= 0; ++k) {
        char one[64];
        snprintf(one, sizeof(one), "%s%d:%.6f", k ? " " : "", top[k], (double) logits[top[k]]);
        strncat(tops, one, sizeof(tops) - strlen(tops) - 1);
    }

    std::lock_guard<std::mutex> lk(g_logits_m);
    if (FILE * mf = fopen((g_logits_dir + "/manifest.txt").c_str(), "a")) {
        fprintf(mf, "ROW mode=%s rep=%d req=%d step=%d row=%d n_vocab=%d argmax=%d sha=%s top5=%s\n",
                g_mode_tag.c_str(), g_rep_tag, req, step, (int) i, n_vocab, (int) best,
                h.final().substr(0, 16).c_str(), tops);
        fclose(mf);
    }
    return best;
}

// --- prompt ingestion on an arm ------------------------------------------------------------------------
// returns false on a failed chunk
static bool ingest_prompt(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & toks,
                          int n_batch, llama_pos & pos_out) {
    n_batch = std::min(n_batch, (int) llama_n_batch(ctx));
    llama_pos pos = 0;
    for (size_t off = 0; off < toks.size(); off += (size_t) n_batch) {
        const size_t end = std::min(toks.size(), off + (size_t) n_batch);
        common_batch batch(ctx);
        for (size_t i = off; i < end; ++i) {
            batch.add(toks[i], pos++, seq, end == toks.size() && i + 1 == end);
        }
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            fprintf(stderr, "seq %d: prompt chunk failed (rc=%d)\n", (int) seq, rc);
            return false;
        }
    }
    // llama_process() is ASYNC: the prompt's wall is only over after a synchronise.  Without this every
    // prefill number is an enqueue time (a warm 512-token prompt "takes" 3 ms instead of ~115 ms).
    llama_synchronize(ctx);
    pos_out = pos;
    return true;
}


// --- the decode arm's step loop -------------------------------------------------------------------
// One token per active slot in ONE batch per step (continuous batching of the generating slots).  Used by
// the two-arm modes and by `batch-pair` (the same loop on a single context), so the batch WIDTHS match.
// Returns false on a failed step.
static bool decode_loop(llama_context * ctx, std::vector<request_t> & reqs, const llama_vocab * vocab,
                        std::mutex * mem_m) {
    for (;;) {
        std::vector<request_t *> active;
        for (auto & r : reqs) {
            if (r.on_decode_arm && !r.done) { active.push_back(&r); }
        }
        if (active.empty()) {
            return true;
        }

        std::unique_lock<std::mutex> lk;
        if (mem_m) { lk = std::unique_lock<std::mutex>(*mem_m); }

        common_batch batch(ctx);
        for (auto * r : active) {
            batch.add(r->pending, r->pos, r->seq, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "decode arm: step failed (batch of %zu)\n", active.size());
            return false;
        }

        double t_step_done = 0.0;
        for (int i = 0; i < (int) active.size(); ++i) {
            auto * r = active[i];
            llama_token tok = sample_row(ctx, i, r->id, (int) r->generated.size());   // the first row's sample synchronises the step
            if (t_step_done == 0.0) { t_step_done = now_ms(); }
            if (tok < 0) { r->done = true; continue; }
            r->generated.push_back(tok);
            r->arrivals_ms.push_back(t_step_done);
            r->pos += 1;
            r->pending = tok;
            const bool stop = llama_vocab_is_eog(vocab, tok) || (int) r->generated.size() >= r->n_predict;
            if (stop) {
                r->done = true;
                r->on_decode_arm = false;
                llama_memory_seq_rm(llama_get_memory(ctx), r->seq, -1, -1);
            }
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_parallel = 4;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    const std::string mode = getenv("ARMS_MODE") ? getenv("ARMS_MODE") : "arms";
    g_mode_tag = mode;
    if (const char * d = getenv("ARMS_LOGITS_DIR")) {
        g_logits_dir = d;
        mkdir(g_logits_dir.c_str(), 0777);   // EEXIST is fine
        if (FILE * mf = fopen((g_logits_dir + "/manifest.txt").c_str(), "a")) {
            fprintf(mf, "# mode=%s\n", mode.c_str());
            fclose(mf);
        }
    }
    const std::string prompt_a = [&] {
        std::string s = "The history of computing is a long and tangled story.";
        if (const char * f = getenv("TWO_CTX_PF_FILE")) {
            FILE * fp = fopen(f, "rb");
            if (fp == nullptr) { fprintf(stderr, "cannot open TWO_CTX_PF_FILE %s\n", f); return s; }
            char buf[65536];
            size_t n;
            s.clear();
            while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) { s.append(buf, n); }
            fclose(fp);
        }
        return s;
    }();

    // scenario: "<prompt_tokens>:<n_predict>@<admit_delay_ms>,..."
    std::vector<request_t> reqs;
    {
        const std::string sc = getenv("ARMS_SCENARIO") ? getenv("ARMS_SCENARIO") : "4932:128@0";
        size_t i = 0;
        while (i < sc.size()) {
            size_t j = sc.find(',', i);
            if (j == std::string::npos) { j = sc.size(); }
            const std::string item = sc.substr(i, j - i);
            i = j + 1;
            if (item.empty()) { continue; }
            request_t r;
            r.id = (int) reqs.size();
            const size_t c1 = item.find(':');
            const size_t c2 = item.find('@');
            r.n_prompt_tok = (size_t) atoll(item.substr(0, c1).c_str());
            r.n_predict    = atoi(item.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1).c_str());
            r.admit_ms     = c2 == std::string::npos ? 0.0 : atof(item.substr(c2 + 1).c_str());
            if (r.n_prompt_tok == 0) { r.n_prompt_tok = 64; }
            if (r.n_predict   <= 0) { r.n_predict = 32; }
            reqs.push_back(r);
        }
    }
    // one sequence per request (a slot); n_ctx_seq = n_ctx / n_seq_max, so do not over-allocate
    const int n_parallel = std::max<int>(2, (int) reqs.size());

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = common_model_load(params.model.get_paths(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the prompt every request draws its tokens from
    std::vector<llama_token> pf_tokens = common_tokenize(vocab, prompt_a, true, false);

    const int n_reps = getenv("ARMS_REPS") ? std::max(1, atoi(getenv("ARMS_REPS"))) : 1;

    // A rep reuses the SAME sequence ids: a request's KV and recurrent state are freed when it finishes, and
    // n_ctx_seq = n_ctx / n_seq_max, so raising n_seq_max for the reps would shrink each sequence's context
    // (the campaign's 4932-token prompt needs n_ctx_seq > 4932).  This is also what a server does with its
    // slots -- the id is a slot, not a request.
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max     = n_parallel;
    cparams.n_outputs_max = n_parallel;

    const bool want_a = mode != "dec-alone";
    const bool want_b = mode != "single" && mode != "pf-alone" && mode != "batch-pair";

    llama_context * ctx_a = nullptr;
    llama_context * ctx_b = nullptr;

    if (want_a) {
        ctx_a = llama_init_from_model(model, cparams);
        if (!ctx_a) { fprintf(stderr, "ctx A failed\n"); return 1; }
    }
    if (want_b) {
        llama_context_params cparams_b = cparams;
        // the DECODE context is sized for decode: a 1-token-per-slot batch needs neither the prefill width
        // nor its compute buffers (the campaign's §2: this is what makes the pair fit at arena 5000)
        const int b_batch = getenv("ARMS_B_BATCH") ? atoi(getenv("ARMS_B_BATCH")) : 8;
        cparams_b.n_batch  = b_batch;
        cparams_b.n_ubatch = b_batch;
        if (ctx_a != nullptr) {
            cparams_b.ctx_other              = ctx_a;
            cparams_b.ctx_other_share_memory = true;
        }
        ctx_b = llama_init_from_model(model, cparams_b);
        if (!ctx_b) { fprintf(stderr, "ctx B failed\n"); return 1; }
    }

    printf("MODE %s  ctx A %s  ctx B %s  n_parallel %d\n", mode.c_str(),
           ctx_a ? "on" : "off",
           ctx_b ? (ctx_a ? "on (KV shared with A)" : "on") : "off", n_parallel);

    const int n_batch_a = (int) params.n_batch;

    for (int rep = 0; rep < n_reps; ++rep) {
        // a rep is the whole scenario again on fresh sequence ids: the FIRST rep pays the cold expert arena
        // (~3x slower, ~13000 fills for 120 tables), so only rep >= 1 is a warm number
        for (auto & r : reqs) {
            r.generated.clear();
            r.arrivals_ms.clear();
            r.t_admit_real = r.t_prompt_start = r.t_prompt_done = r.t_handover = r.t_first_tok = 0.0;
            r.handover_bytes = 0;
            r.handover_device = false;
            // the handover flags are per REQUEST, and a stale `handover_done` from the previous rep makes the
            // prefill arm skip its wait -- so the next prompt would be prefilled while the take is still reading
            // this arm's recurrent metadata (the ordering rule of HANDOVER §4.2).  Reset them with the rest.
            r.handover_posted = false;
            r.handover_done   = false;
            r.t_handover = 0.0;
            r.t_first_tok = 0.0;
            r.pending = -1;
            r.pos = 0;
            r.on_decode_arm = false;
            r.done = false;
            r.seq = (llama_seq_id) r.id;
        }
        if (n_reps > 1) {
            printf("REP %d\n", rep);
        }
        g_rep_tag = rep;

        // ARMS_RESET_CACHE: wipe the cell metadata and every allocation cursor at the rep boundary, so each rep
        // starts exactly like rep 0.  Each context owns its OWN cursor (`v_heads`) over the SAME shared cells,
        // so without this the two arms start a warm rep from two different places -- the single-context modes
        // have one cursor and therefore one layout per rep.
        if (getenv("ARMS_RESET_CACHE") && atoi(getenv("ARMS_RESET_CACHE")) != 0) {
            if (ctx_a) { llama_memory_clear(llama_get_memory(ctx_a), false); }
            if (ctx_b) { llama_memory_clear(llama_get_memory(ctx_b), false); }
        }

        // ---------------------------------------------------------------------------------------------
        // reference modes: everything on ONE context, one request at a time
        // ---------------------------------------------------------------------------------------------
        if (mode == "batch-pair") {
            // the single-context reference for a CONCURRENT decode set: every prompt is prefilled first, then
            // the same decode loop as the decode arm runs all slots in one batch.  The batch widths match the
            // two-arm arm only when that arm's slots are all handed over before it steps (ARMS_WAIT_ALL=1).
            for (auto & r : reqs) {
                std::vector<llama_token> toks(pf_tokens.begin(),
                        pf_tokens.begin() + std::min(pf_tokens.size(), r.n_prompt_tok));
                while (toks.size() < r.n_prompt_tok) {
                    const size_t take = std::min(pf_tokens.size(), (size_t) r.n_prompt_tok - toks.size());
                    toks.insert(toks.end(), pf_tokens.begin(), pf_tokens.begin() + take);
                }
                r.n_prompt_tok = toks.size();
                r.t_admit_real = r.t_prompt_start = now_ms();
                llama_pos pos = 0;
                if (!ingest_prompt(ctx_a, r.seq, toks, n_batch_a, pos)) { return 1; }
                r.t_prompt_done = now_ms();
                llama_token tok = sample_row(ctx_a, -1, r.id, 0);
                if (tok < 0) { fprintf(stderr, "no logits\n"); return 1; }
                r.generated.push_back(tok);
                r.arrivals_ms.push_back(now_ms());
                r.pending = tok;
                r.pos     = pos;
                r.on_decode_arm = true;
            }
            if (!decode_loop(ctx_a, reqs, vocab, nullptr)) { return 1; }
            for (const auto & r : reqs) {
                printf("REQ %d sha=%s gen=%zu prompt_tokens=%zu pf_ms=%.1f pf_tps=%.1f\n",
                       r.id, sha_of(r.generated).substr(0, 12).c_str(), r.generated.size(), r.n_prompt_tok,
                       r.t_prompt_done - r.t_prompt_start,
                       1000.0 * (double) r.n_prompt_tok / (r.t_prompt_done - r.t_prompt_start));
            }
            continue;
        }

        if (mode == "single" || mode == "pf-alone" || mode == "dec-alone") {
            llama_context * ctx = (mode == "dec-alone") ? ctx_b : (ctx_a ? ctx_a : ctx_b);

            for (auto & r : reqs) {
                std::vector<llama_token> toks(pf_tokens.begin(),
                        pf_tokens.begin() + std::min(pf_tokens.size(), r.n_prompt_tok));
                while (toks.size() < r.n_prompt_tok) {
                    const size_t take = std::min(pf_tokens.size(), (size_t) r.n_prompt_tok - toks.size());
                    toks.insert(toks.end(), pf_tokens.begin(), pf_tokens.begin() + take);
                }
                const double t0 = now_ms();
                r.t_admit_real   = t0;
                r.t_prompt_start = t0;
                llama_pos pos = 0;
                if (!ingest_prompt(ctx, r.seq, toks, n_batch_a, pos)) { return 1; }
                r.t_prompt_done = now_ms();
                r.n_prompt_tok  = toks.size();

                const int n_dec = (mode == "pf-alone") ? 0 : r.n_predict;
                for (int i = 0; i < n_dec; ++i) {
                    llama_token tok = sample_row(ctx, -1, r.id, i);
                    if (tok < 0) { fprintf(stderr, "no logits\n"); return 1; }
                    r.generated.push_back(tok);
                    r.arrivals_ms.push_back(now_ms());
                    if (llama_vocab_is_eog(vocab, tok) || i + 1 == n_dec) { break; }
                    common_batch batch(ctx);
                    batch.add(tok, pos++, r.seq, true);
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        fprintf(stderr, "decode failed\n"); return 1;
                    }
                }
                llama_memory_seq_rm(llama_get_memory(ctx), r.seq, -1, -1);
                r.done = true;
            }

            for (const auto & r : reqs) {
                const std::string sha = sha_of(r.generated);
                const double pf_ms = r.t_prompt_done - r.t_prompt_start;
                printf("REQ %d sha=%s gen=%zu prompt_tokens=%zu pf_ms=%.1f pf_tps=%.1f\n",
                       r.id, sha.substr(0, 12).c_str(), r.generated.size(), r.n_prompt_tok, pf_ms,
                       pf_ms > 0 ? 1000.0 * (double) r.n_prompt_tok / pf_ms : 0.0);
                if (mode == "single") {
                    printf("TEXT %d: %s\n", r.id, common_detokenize(vocab, r.generated, false).c_str());
                }
            }
            continue;
        }

        // ---------------------------------------------------------------------------------------------
        // the two arms
        // ---------------------------------------------------------------------------------------------
        arms_t st;
        st.reqs     = &reqs;
        st.wait_all = getenv("ARMS_WAIT_ALL") && atoi(getenv("ARMS_WAIT_ALL")) != 0;

        std::atomic<bool> prefill_done_flag{false};   // (kept for symmetry with the campaign's other tools)

        std::thread t_pf([&] {
            const double t_origin = now_ms();
            for (auto & r : reqs) {
                const double due = t_origin + r.admit_ms;
                while (now_ms() < due) { std::this_thread::sleep_for(std::chrono::microseconds(500)); }

                std::vector<llama_token> toks(pf_tokens.begin(),
                        pf_tokens.begin() + std::min(pf_tokens.size(), r.n_prompt_tok));
                while (toks.size() < r.n_prompt_tok) {
                    const size_t take = std::min(pf_tokens.size(), (size_t) r.n_prompt_tok - toks.size());
                    toks.insert(toks.end(), pf_tokens.begin(), pf_tokens.begin() + take);
                }
                r.n_prompt_tok = toks.size();

                r.t_admit_real   = now_ms();
                r.t_prompt_start = now_ms();
                {
                    std::lock_guard<std::mutex> lk(st.pf_m);
                    st.pf_windows.emplace_back(r.t_prompt_start, 0.0);
                }

                llama_pos pos = 0;
                if (!ingest_prompt(ctx_a, r.seq, toks, n_batch_a, pos)) { break; }
                r.t_prompt_done = now_ms();
                {
                    std::lock_guard<std::mutex> lk(st.pf_m);
                    st.pf_windows.back().second = r.t_prompt_done;
                }

                // the first token is sampled on the arm that produced the logits
                llama_token tok = sample_row(ctx_a, -1, r.id, 0);
                if (tok < 0) { fprintf(stderr, "req %d: no logits after the prompt\n", r.id); break; }
                r.generated.push_back(tok);
                r.arrivals_ms.push_back(now_ms());
                r.t_first_tok = r.arrivals_ms.back();

                // the decode arm ingests this token at the prompt's end (position = prompt length)
                r.pending = tok;
                r.pos     = pos;

                // ---- HANDOVER: post the slot to the decode arm --------------------------------------
                // The decode arm executes the state take at the top of its next step (under mem_m), so the
                // handover is serialised with the decode steps and the prefill arm never waits on a lock the
                // decoder holds in a loop.  The prefill arm DOES wait for the take to complete before it starts
                // the next prompt: the take reads this arm's recurrent cell metadata, which the next prompt's
                // find_slot() would be mutating.
                r.t_handover = 0.0;
                {
                    std::lock_guard<std::mutex> lk(st.m);
                    r.handover_posted = true;
                    st.cv_ready.notify_all();
                }
                {
                    std::unique_lock<std::mutex> lk(st.m);
                    const double t_w = now_ms();
                    // ARMS_NO_WAIT_TAKE=1 is the CONTROL for the ordering rule (HANDOVER §4.2): it lets the next
                    // prompt be prefilled while the take is still reading this arm's recurrent metadata -- the
                    // race that the stale `handover_done` flag used to create in warm reps.  Measured: with the
                    // race, R1's prefill lands in recurrent cell 1 instead of 0 and its warm text moves; it is
                    // NOT a property of the design, and it is what pass 1/2 read as a "host-tier" effect.
                    if (!(getenv("ARMS_NO_WAIT_TAKE") && atoi(getenv("ARMS_NO_WAIT_TAKE")) != 0)) {
                        st.cv_taken.wait(lk, [&] { return r.handover_done || st.prefill_done; });
                    }
                    r.t_handover = now_ms() - t_w;
                }

                r.pending       = tok;
                r.pos           = pos;

                {
                    std::lock_guard<std::mutex> lk(st.m);
                    st.n_prefilled++;
                    st.cv_ready.notify_all();
                }
            }

            {
                std::lock_guard<std::mutex> lk(st.m);
                st.prefill_done = true;
                st.cv_ready.notify_all();
            }
            prefill_done_flag.store(true);
        });

        std::thread t_dec([&] {
            for (;;) {
                std::vector<request_t *> active;
                std::vector<request_t *> taking;
                {
                    std::unique_lock<std::mutex> lk(st.m);
                    st.cv_ready.wait(lk, [&] {
                        for (auto & r : reqs) {
                            if ((r.on_decode_arm || r.handover_posted) && !r.done) { return true; }
                        }
                        return st.prefill_done;
                    });
                    for (auto & r : reqs) {
                        if (r.handover_posted) { taking.push_back(&r); }
                        else if (r.on_decode_arm && !r.done) { active.push_back(&r); }
                    }
                    if (st.wait_all && st.n_taken < reqs.size()) {
                        active.clear();   // hold every step until the whole set has been handed over
                    }
                    if (active.empty() && taking.empty()) {
                        if (!st.prefill_done) {
                            // the prefill arm still has requests to come (a prompt in flight, or a handover
                            // it has not posted yet): this arm must NOT exit
                            continue;
                        }
                        st.decode_done = true;
                        st.cv_done.notify_all();
                        return;
                    }
                }

                if (!taking.empty()) {
                    std::lock_guard<std::mutex> lk(st.mem_m);
                    for (auto * r : taking) {
                        const double t_h0 = now_ms();
                        bool taken = false;
                        if (std::strcmp(getenv("ARMS_HANDOVER") ? getenv("ARMS_HANDOVER") : "device", "copy") != 0) {
                            taken = llama_memory_seq_state_take(llama_get_memory(ctx_b), llama_get_memory(ctx_a), r->seq);
                        }
                        if (!taken) {
                            const size_t sz = llama_state_seq_get_size_ext(ctx_a, r->seq, LLAMA_STATE_SEQ_FLAGS_RECR_ONLY);
                            std::vector<uint8_t> buf(sz);
                            const size_t nw = llama_state_seq_get_data_ext(ctx_a, buf.data(), sz, r->seq, LLAMA_STATE_SEQ_FLAGS_RECR_ONLY);
                            if (nw == 0) { fprintf(stderr, "req %d: state get failed\n", r->id); exit(1); }
                            const size_t nr = llama_state_seq_set_data_ext(ctx_b, buf.data(), nw, r->seq, LLAMA_STATE_SEQ_FLAGS_RECR_ONLY);
                            if (nr != nw) { fprintf(stderr, "req %d: state set failed (%zu != %zu)\n", r->id, nr, nw); exit(1); }
                            r->handover_bytes = nw;
                            llama_memory_seq_rm_recr(llama_get_memory(ctx_a), r->seq, -1, -1);
                        }
                        r->t_handover      = now_ms() - t_h0;
                        r->handover_device = taken;
                    }
                    {
                        std::lock_guard<std::mutex> lk2(st.m);
                        for (auto * r : taking) {
                            r->handover_posted = false;
                            r->handover_done   = true;
                            r->on_decode_arm   = true;
                            st.n_taken++;
                        }
                        st.cv_taken.notify_all();
                        st.cv_ready.notify_all();
                    }
                    continue;   // collect the (now larger) active set on the next pass
                }

                // one step: one token per active slot.  mem_m keeps a concurrent handover out of this
                // memory module's cell metadata (the memories are separate objects; only the KV cells are
                // shared -- see the campaign's README, landmine 7).
                {
                    std::lock_guard<std::mutex> lk(st.mem_m);

                    common_batch batch(ctx_b);
                    for (auto * r : active) {
                        batch.add(r->pending, r->pos, r->seq, true);
                    }
                    if (llama_process(ctx_b, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        fprintf(stderr, "decode arm: step failed -- releasing every active sequence\n");
                        for (auto * r : active) {
                            llama_memory_seq_rm(llama_get_memory(ctx_b), r->seq, -1, -1);
                            r->done = true;
                            r->on_decode_arm = false;
                        }
                        std::lock_guard<std::mutex> lk2(st.m);
                        st.decode_done = true;
                        st.cv_done.notify_all();
                        return;
                    }
                    double t_step_done = 0.0;
                    for (int i = 0; i < (int) active.size(); ++i) {
                        auto * r = active[i];
                        llama_token tok = sample_row(ctx_b, i, r->id, (int) r->generated.size());   // the first sample synchronises the step
                        if (t_step_done == 0.0) { t_step_done = now_ms(); }
                        if (tok < 0) { r->done = true; continue; }
                        r->generated.push_back(tok);
                        r->arrivals_ms.push_back(t_step_done);
                        r->pos += 1;
                        r->pending = tok;
                        const bool stop = llama_vocab_is_eog(vocab, tok) ||
                                          (int) r->generated.size() >= r->n_predict;
                        if (stop) {
                            r->done = true;
                            r->on_decode_arm = false;
                            llama_memory_seq_rm(llama_get_memory(ctx_b), r->seq, -1, -1);
                        }
                    }
                }
            }
        });

        t_pf.join();
        t_dec.join();

        // wip/prefill-decode-arms (T0): the co-owner pair's allocation HANDSHAKE, checked at the request boundary.
        // A leaked reservation (a prepare nothing applied) would shrink the shared cache silently, so it is printed
        // rather than hidden; the counters are the reserve/commit/release traffic of the rep.
        {
            const bool ok_a = ctx_a ? llama_memory_alloc_sync(llama_get_memory(ctx_a), "request-boundary") : true;
            const bool ok_b = ctx_b ? llama_memory_alloc_sync(llama_get_memory(ctx_b), "request-boundary") : true;
            printf("HANDSHAKE ok=%d\n", (ok_a && ok_b) ? 1 : 0);
        }

        std::vector<std::pair<double, double>> windows;
        {
            std::lock_guard<std::mutex> lk(st.pf_m);
            windows = st.pf_windows;
        }

        double pf_total_ms = 0.0;
        size_t pf_total_tok = 0;
        for (const auto & r : reqs) {
            const std::string sha = sha_of(r.generated);
            const double pf_ms = r.t_prompt_done - r.t_prompt_start;
            pf_total_ms  += pf_ms;
            pf_total_tok += r.n_prompt_tok;
            const std::string hb = r.handover_device
                    ? std::string("device")
                    : std::string("copy,") + std::to_string(r.handover_bytes) + "B";
            printf("REQ %d sha=%s gen=%zu prompt_tokens=%zu pf_ms=%.1f pf_tps=%.1f "
                   "handover_ms=%.2f(%s) ttft_ms=%.1f wall_ms=%.1f\n",
                   r.id, sha.substr(0, 12).c_str(), r.generated.size(), r.n_prompt_tok, pf_ms,
                   pf_ms > 0 ? 1000.0 * (double) r.n_prompt_tok / pf_ms : 0.0,
                   r.t_handover, hb.c_str(),
                   r.t_first_tok > 0 ? r.t_first_tok - r.t_admit_real : 0.0,
                   r.arrivals_ms.empty() ? 0.0 : r.arrivals_ms.back() - r.t_admit_real);
        }
        printf("TOTAL prefill_tokens=%zu prefill_ms=%.1f pf_tps=%.1f\n",
               pf_total_tok, pf_total_ms, pf_total_ms > 0 ? 1000.0 * (double) pf_total_tok / pf_total_ms : 0.0);

        std::vector<double> all, during;
        for (const auto & r : reqs) {
            for (size_t i = 1; i < r.arrivals_ms.size(); ++i) {
                const double d = r.arrivals_ms[i] - r.arrivals_ms[i - 1];
                all.push_back(d);
                if (in_any_window(windows, r.arrivals_ms[i])) { during.push_back(d); }
            }
        }
        size_t n_in_window = 0;
        for (const auto & r : reqs) {
            for (double a : r.arrivals_ms) {
                if (in_any_window(windows, a)) { n_in_window++; }
            }
        }
        printf("ITL all=%zu p50=%.1f p95=%.1f max=%.1f | during_prefill=%zu tokens=%zu p50=%.1f p95=%.1f max=%.1f\n",
               all.size(), ipct(all, 50), ipct(all, 95), vmax(all),
               during.size(), n_in_window, ipct(during, 50), ipct(during, 95), vmax(during));
    }

    if (ctx_b) { llama_free(ctx_b); }
    if (ctx_a) { llama_free(ctx_a); }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
