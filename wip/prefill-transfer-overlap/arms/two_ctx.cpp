// two_ctx -- WIP instrument for the prefill-transfer-overlap campaign (NOT part of the delivery).
//
// Task 1 of HANDOVER §4.2: one process, ONE model, TWO llama_contexts that share the KV cache, each
// processing its own sequence.  The oracle is text: the two sequences' greedy output must be byte-identical
// to the same two sequences run in ONE context.
//
// It compiles against a build of ~/llama.cpp (the campaign's candidate tree), not against the delivery:
//
//   see arms/build-two-ctx.sh
//
// Modes (--mode):
//   single : one context (n_seq_max = 2), run seq 0 then seq 1 sequentially -- the REFERENCE
//   dual   : ctx A runs seq 0, ctx B (params.ctx_other = A, ctx_other_share_memory = true) runs seq 1,
//            still sequentially in time.  Same text as `single` == the shared KV is coherent.
//   dual-parallel : A prefills seq 0 while B decodes seq 1 CONCURRENTLY (two host threads, one per
//            context).  The text oracle still applies: the sequences are independent, so a concurrent
//            dual run must read the SAME text as `single`/`dual`.  This is the submission experiment's
//            control arm (no gate, no yield point yet -- both run flat out).
//
// Every mode prints `SEQ <id> <n_tokens> sha=<first 12 hex of sha256 of the token ids>` plus the tokens, so
// a diff of two runs is mechanical.

#include "common.h"
#include "arg.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <string>
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

// --- the campaign's gate (HANDOVER §4.4) ----------------------------------------------------------------
// The decoder may only spend `F x the prefill's elapsed wall` on its own steps -- "hand the decoder at most
// the fraction of the prefill that the gather window is worth".  F = 1 is one-for-one, F = 0 is DUTY=0-shaped
// (no decode at all while a prefill runs), and no gate at all is the flat-out arm.
struct gate_t {
    std::atomic<double> pf_start_ms{0.0};     // A's first prompt chunk, absolute
    std::atomic<double> pf_end_ms{0.0};       // A's prompt finished, absolute
    std::atomic<double> dec_spent_ms{0.0};    // the decoder's own step time, accumulated
    double F = 1.0;
    bool   enabled = false;
};

static double now_ms() { return ggml_time_us() / 1000.0; }

// --- the split-boundary switch (Task 2) ----------------------------------------------------------------
// `llama_set_split_yield_callback` fires once per graph split that issued a MoE gather, after the gather is
// enqueued and before the split's compute -- the campaign's per-layer window.  The prefill thread OFFERS a
// window; the decoder thread is parked waiting for one, wakes, enqueues exactly ONE step on ITS OWN thread
// (its own per-thread stream) and reports the enqueue; the prefill thread waits only for that report (bounded)
// so the decoder's kernels are queued ahead of the split's compute.  A step is offered only while the decoder
// is waiting and only `F x elapsed / step` of them have been placed -- HANDOVER §4.4's rule.
struct yield_t {
    std::mutex              m;
    std::condition_variable cv;
    uint64_t a_seq = 0;      // windows offered
    uint64_t b_done = 0;     // windows the decoder has enqueued for
    bool     b_ready = false;
    bool     b_stop  = false;
    bool     enabled = false;
    bool     pf_running = false;
    double   F = 1.0;
    double   pf_start_ms = 0.0;
    double   est_step_ms = 11.0;
    int      placed = 0;
    int      offered = 0;
    int      skipped_not_ready = 0;
    int      skipped_budget = 0;
    double   wait_ms = 3.0;   // how long the prefill may stall for the decoder's enqueue
    double   a_stall_ms = 0.0;
    double   a_stall_max_ms = 0.0;
};

static void yield_offer(void * ud) {
    auto * y = (yield_t *) ud;
    std::unique_lock<std::mutex> lk(y->m);
    if (!y->enabled || y->b_stop || !y->pf_running) { return; }
    if (y->F <= 0.0)                                                 { y->skipped_budget++;  return; }
    if (!y->b_ready)                                                 { y->skipped_not_ready++; return; }
    const double budget = y->F * (now_ms() - y->pf_start_ms) / y->est_step_ms;
    if ((double) y->placed >= budget)                                { y->skipped_budget++;  return; }

    y->a_seq++;
    y->offered++;
    y->cv.notify_all();
    const double t_w = now_ms();
    y->cv.wait_for(lk, std::chrono::microseconds((long) (y->wait_ms * 1000.0)),
                   [&] { return y->b_done >= y->a_seq || y->b_stop; });
    const double d = now_ms() - t_w;
    y->a_stall_ms += d;
    if (d > y->a_stall_max_ms) { y->a_stall_max_ms = d; }
}

// the decoder: park for a window before a step.  Returns false when the prefill is over and the decoder
// should go back to decoding freely.
static bool yield_park(yield_t * y, uint64_t & seen) {
    std::unique_lock<std::mutex> lk(y->m);
    y->b_ready = true;
    y->cv.notify_all();
    y->cv.wait(lk, [&] { return y->a_seq > seen || y->b_stop || !y->pf_running; });
    if (y->a_seq <= seen) {
        y->b_ready = false;
        return false;      // the prefill finished (or stopped): decode freely from here
    }
    seen = y->a_seq;
    y->b_ready = false;
    y->placed++;
    return true;           // a window was granted: enqueue one step now
}

static void yield_enqueued(yield_t * y, uint64_t seen) {
    std::lock_guard<std::mutex> lk(y->m);
    y->b_done = seen;
    y->cv.notify_all();
}

// --- the per-sequence runner ---------------------------------------------------------------------------
struct seq_result {
    std::vector<llama_token> generated;
    std::vector<double>      arrivals_ms;   // ABSOLUTE epoch ms (ggml_time_us/1000) per generated token
    double t_start_ms   = 0.0;              // absolute, right before the first prompt chunk
    double t_prompt_ms  = 0.0;              // absolute, the prompt's last chunk returned
    double t_wall_ms    = 0.0;              // prompt + generation
    size_t n_prompt     = 0;
};

static seq_result run_seq(
        llama_context * ctx,
        const std::string & prompt,
        int n_predict,
        llama_seq_id seq_id,
        int n_batch,
        std::atomic<bool> * go,
        std::atomic<int>  * ready,
        gate_t * gate = nullptr,
        bool is_prefill = false,
        bool is_decoder = false,
        yield_t * yl = nullptr) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    // a context sized for DECODE (a second context, n_batch = 8) ingests its prompt in small chunks
    n_batch = std::min(n_batch, (int) llama_n_batch(ctx));

    std::vector<llama_token> toks = common_tokenize(vocab, prompt, true, false);

    seq_result res;

    if (go != nullptr) {
        // rendezvous: both arms start their first decode at the same instant (the campaign's control arm)
        ready->fetch_add(1);
        while (!go->load()) { std::this_thread::sleep_for(std::chrono::microseconds(200)); }
    }

    const int64_t t_start = ggml_time_us();
    res.t_start_ms = t_start / 1000.0;
    res.n_prompt   = toks.size();

    auto decode_chunk = [&](const std::vector<llama_token> & chunk, llama_pos pos0, bool logits_last) -> bool {
        common_batch batch(ctx);
        for (size_t i = 0; i < chunk.size(); ++i) {
            const bool want = logits_last && i + 1 == chunk.size();
            batch.add(chunk[i], pos0 + (llama_pos) i, seq_id, want);
        }
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        return rc == 0;
    };

    if (gate != nullptr && is_prefill) {
        gate->pf_start_ms.store(now_ms());
    }
    if (yl != nullptr && is_prefill) {
        std::lock_guard<std::mutex> lk(yl->m);
        yl->pf_start_ms = now_ms();
        yl->pf_running  = true;
    }

    // prompt, in n_batch chunks (the last chunk asks for logits)
    llama_pos pos = 0;
    for (size_t off = 0; off < toks.size(); off += (size_t) n_batch) {
        const size_t end = std::min(toks.size(), off + (size_t) n_batch);
        std::vector<llama_token> chunk(toks.begin() + off, toks.begin() + end);
        if (!decode_chunk(chunk, pos, end == toks.size())) {
            fprintf(stderr, "seq %d: prompt decode failed\n", (int) seq_id);
            return res;
        }
        pos += (llama_pos) chunk.size();
    }

    res.t_prompt_ms = ggml_time_us() / 1000.0;
    if (gate != nullptr && is_prefill) {
        gate->pf_end_ms.store(res.t_prompt_ms);
    }
    if (yl != nullptr && is_prefill) {
        std::lock_guard<std::mutex> lk(yl->m);
        yl->pf_running = false;
        yl->b_stop     = true;
        yl->cv.notify_all();
    }

    // greedy loop: argmax over the last row, no sampler (the campaign's text oracle is plain greedy)
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            fprintf(stderr, "seq %d: no logits at step %d\n", (int) seq_id, i);
            return res;
        }
        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) { best = v; }
        }
        res.generated.push_back(best);
        res.arrivals_ms.push_back(ggml_time_us() / 1000.0);
        if (llama_vocab_is_eog(vocab, best) || i + 1 == n_predict) {
            break;
        }
        if (gate != nullptr && gate->enabled && is_decoder) {
            // wait until the prefill's elapsed wall x F covers what this decoder has already spent
            for (;;) {
                const double ps = gate->pf_start_ms.load();
                const double pe = gate->pf_end_ms.load();
                if (ps == 0.0 || pe > 0.0) { break; }                    // no prefill in flight
                if (gate->dec_spent_ms.load() <= gate->F * (now_ms() - ps)) { break; }
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        // Task 2: wait for the prefill to offer a window (only while it is running); the step is enqueued
        // from THIS thread, so it lands on this thread's own per-thread stream
        static uint64_t yl_seen = 0;
        bool yl_window = false;
        if (yl != nullptr && yl->enabled) {
            yl_window = yield_park(yl, yl_seen);
        }

        const double t_step = now_ms();
        if (!decode_chunk({ best }, pos, true)) {
            fprintf(stderr, "seq %d: decode failed at step %d\n", (int) seq_id, i);
            return res;
        }
        if (gate != nullptr && is_decoder) {
            gate->dec_spent_ms.store(gate->dec_spent_ms.load() + now_ms() - t_step);
        }
        if (yl != nullptr && yl_window) {
            yield_enqueued(yl, yl_seen);
        }
        pos += 1;
    }

    res.t_wall_ms = (ggml_time_us() - t_start) / 1000.0;
    return res;
}

static double ipct(std::vector<double> xs, double p) {
    if (xs.empty()) { return 0.0; }
    std::sort(xs.begin(), xs.end());
    const size_t k = std::min(xs.size() - 1, (size_t) (p / 100.0 * (xs.size() - 1) + 0.5));
    return xs[k];
}

static void report(int seq_id, const seq_result & r, const llama_vocab * vocab, bool with_text) {
    const std::string sha = sha_of(r.generated);
    const double prompt_ms = r.t_prompt_ms - r.t_start_ms;
    printf("SEQ %d n=%zu sha=%s prompt_tokens=%zu prompt_ms=%.1f pf_tps=%.1f wall_ms=%.1f\n",
           seq_id, r.generated.size(), sha.substr(0, 12).c_str(), r.n_prompt, prompt_ms,
           prompt_ms > 0 ? 1000.0 * r.n_prompt / prompt_ms : 0.0, r.t_wall_ms);
    if (with_text) {
        printf("TEXT %d: %s\n", seq_id, common_detokenize(vocab, r.generated, false).c_str());
    }
}

// the decoder's inter-token times, split by whether they landed inside the prefill's window
static void report_itl(const seq_result & dec, const seq_result & pf, const char * label) {
    std::vector<double> all, during;
    for (size_t i = 1; i < dec.arrivals_ms.size(); ++i) {
        const double d = dec.arrivals_ms[i] - dec.arrivals_ms[i - 1];
        all.push_back(d);
        if (dec.arrivals_ms[i] >= pf.t_start_ms && dec.arrivals_ms[i] <= pf.t_prompt_ms) {
            during.push_back(d);
        }
    }
    printf("ITL %s total=%zu during_prefill=%zu p50=%.1f p95=%.1f max=%.1f | during p50=%.1f p95=%.1f max=%.1f\n",
           label, all.size(), during.size(),
           ipct(all, 50), ipct(all, 95), all.empty() ? 0.0 : *std::max_element(all.begin(), all.end()),
           ipct(during, 50), ipct(during, 95), during.empty() ? 0.0 : *std::max_element(during.begin(), during.end()));
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_parallel = 2;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();

    std::string mode = getenv("TWO_CTX_MODE") ? getenv("TWO_CTX_MODE") : "dual";
    int n_predict = params.n_predict > 0 ? params.n_predict : 32;

    // sequence A's prompt: a file (the campaign's arm) or the built-in one-liner
    std::string prompt_a = "The history of computing is a long and tangled story.";
    size_t      n_prompt_a_cap = 0;   // 0 = the whole file
    if (const char * f = getenv("TWO_CTX_PF_FILE")) {
        FILE * fp = fopen(f, "rb");
        if (fp == nullptr) { fprintf(stderr, "cannot open TWO_CTX_PF_FILE %s\n", f); return 1; }
        char buf[65536];
        size_t n;
        prompt_a.clear();
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) { prompt_a.append(buf, n); }
        fclose(fp);
    }
    if (const char * t = getenv("TWO_CTX_PF_TOKENS")) {
        n_prompt_a_cap = (size_t) atoll(t);
    }
    const std::string prompt_b = "Count slowly from one to five hundred, one number per line.";

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = common_model_load(params.model.get_paths(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }

    int n_reps = getenv("TWO_CTX_REPS") ? atoi(getenv("TWO_CTX_REPS")) : 1;
    if (n_reps < 1) { n_reps = 1; }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max     = 2 * n_reps;   // one (A, B) pair of sequence ids per rep
    cparams.n_outputs_max = 2 * n_reps;

    const bool want_a = mode != "dec-alone";
    const bool want_b = mode != "pf-alone";

    llama_context * ctx_a = nullptr;
    llama_context * ctx_b = nullptr;

    if (want_a) {
        ctx_a = llama_init_from_model(model, cparams);
        if (!ctx_a) { fprintf(stderr, "ctx A failed\n"); return 1; }
    }
    if (want_b) {
        llama_context_params cparams_b = cparams;
        // the DECODE context is sized for decode: its compute buffers are the VRAM a second context costs,
        // and a 1-token batch needs neither the prefill width nor its buffers
        if (const char * bb = getenv("TWO_CTX_B_BATCH")) {
            cparams_b.n_batch  = atoi(bb);
            cparams_b.n_ubatch = atoi(bb);
        }
        if (ctx_a != nullptr) {
            // the generic second context: same model, the KV cells shared, its own sequences
            cparams_b.ctx_other              = ctx_a;
            cparams_b.ctx_other_share_memory = true;
        }
        ctx_b = llama_init_from_model(model, cparams_b);
        if (!ctx_b) { fprintf(stderr, "ctx B failed\n"); return 1; }
        if (ctx_a == nullptr) {
            // dec-alone: give B a cache of its own (no sharing needed for a single-sequence control)
            cparams_b.ctx_other = nullptr;
        }
    }

    // wip/prefill-transfer-overlap: stream priorities, the hardware form of "spend the window, not the
    // compute" -- a LOW decode context is scheduled only into the SMs the prefill's gathers leave idle.
    if (const char * p = getenv("TWO_CTX_A_PRIO")) {
        const int prio = strcmp(p, "low") == 0 ? -1 : (strcmp(p, "high") == 0 ? 1 : atoi(p));
        printf("PRIO A = %d (%s)\n", prio, llama_set_stream_priority(ctx_a, prio) ? "set" : "NOT SUPPORTED");
    }
    if (ctx_b != nullptr) {
        if (const char * p = getenv("TWO_CTX_B_PRIO")) {
            const int prio = strcmp(p, "low") == 0 ? -1 : (strcmp(p, "high") == 0 ? 1 : atoi(p));
            printf("PRIO B = %d (%s)\n", prio, llama_set_stream_priority(ctx_b, prio) ? "set" : "NOT SUPPORTED");
        }
    }

    printf("MODE %s  ctx A %s  ctx B %s\n", mode.c_str(),
           ctx_a ? "on" : "off", ctx_b ? (cparams.ctx_other_share_memory ? "on (KV shared with A)" : "on") : "off");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_batch = (int) params.n_batch;

    for (int rep = 0; rep < n_reps; ++rep) {
    const llama_seq_id seq_a = (llama_seq_id) (2*rep + 0);
    const llama_seq_id seq_b = (llama_seq_id) (2*rep + 1);
    if (n_reps > 1) {
        printf("REP %d (seq A=%d, seq B=%d)\n", rep, (int) seq_a, (int) seq_b);
    }

    if (mode == "single") {
        report(0, run_seq(ctx_a, prompt_a, n_predict, seq_a, n_batch, nullptr, nullptr), vocab, true);
        report(1, run_seq(ctx_a, prompt_b, n_predict, seq_b, n_batch, nullptr, nullptr), vocab, true);
    } else if (mode == "dual") {
        report(0, run_seq(ctx_a, prompt_a, n_predict, seq_a, n_batch, nullptr, nullptr), vocab, true);
        report(1, run_seq(ctx_b, prompt_b, n_predict, seq_b, n_batch, nullptr, nullptr), vocab, true);
    } else if (mode == "dual-parallel") {
        std::atomic<bool> go{false};
        std::atomic<int>  ready{0};
        seq_result ra, rb;
        std::thread ta([&] { ra = run_seq(ctx_a, prompt_a, n_predict, seq_a, n_batch, &go, &ready); });
        std::thread tb([&] { rb = run_seq(ctx_b, prompt_b, n_predict, seq_b, n_batch, &go, &ready); });
        while (ready.load() < 2) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        go.store(true);
        ta.join();
        tb.join();
        report(0, ra, vocab, true);
        report(1, rb, vocab, true);
    } else if (mode == "pf-alone" || mode == "dec-alone") {
        // the two reference walls for the pf-dec arm
        if (mode == "pf-alone") {
            const seq_result ra = run_seq(ctx_a, prompt_a, 2, seq_a, n_batch, nullptr, nullptr);
            report(0, ra, vocab, false);
        } else {
            // the decoder's own prompt plus the generation, in ONE call (a second call would restart at
            // position 0 and the allocator refuses it -- X < Y)
            const std::string dec_prompt = prompt_b + " Give me a long list of numbers, one per line.";
            const seq_result rb = run_seq(ctx_b, dec_prompt, n_predict, seq_b, n_batch, nullptr, nullptr);
            report(1, rb, vocab, false);
            report_itl(rb, rb, "dec-alone");
        }
    } else if (mode == "pf-dec") {
        // THE CAMPAIGN'S PHYSICS ARM: A prefills the long prompt while B decodes.
        gate_t  gate;
        yield_t y;
        if (const char * yy = getenv("TWO_CTX_YIELD")) {
            y.enabled = atoi(yy) != 0;
            y.F = getenv("TWO_CTX_GATE_F") ? atof(getenv("TWO_CTX_GATE_F")) : 1.0;
            if (getenv("TWO_CTX_WAIT_MS")) { y.wait_ms = atof(getenv("TWO_CTX_WAIT_MS")); }
            llama_set_split_yield_callback(ctx_a, yield_offer, &y);
            printf("YIELD MODE: one decode step per gathered split, F=%.2f\n", y.F);
        }
        if (const char * g = getenv("TWO_CTX_GATE_F")) {
            gate.enabled = true;
            gate.F = atof(g);
        }
        std::atomic<bool> go{false};
        std::atomic<int>  ready{0};
        seq_result ra, rb;
        std::thread ta([&] { ra = run_seq(ctx_a, prompt_a, 4, seq_a, n_batch, &go, &ready, &gate, true, false, y.enabled ? &y : nullptr); });
        std::thread tb([&] {
            // B's own short prompt, then a long decode, in one call: the decode is what must overlap
            const std::string dec_prompt = prompt_b + " Give me a long list of numbers, one per line.";
            rb = run_seq(ctx_b, dec_prompt, n_predict, seq_b, n_batch, &go, &ready, &gate, false, true, y.enabled ? &y : nullptr);
        });
        while (ready.load() < 2) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        go.store(true);
        ta.join();
        tb.join();
        report(0, ra, vocab, false);
        report(1, rb, vocab, false);
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "pf-dec%s", gate.enabled ? " (gated)" : " (flat out)");
        report_itl(rb, ra, lbl);
        if (y.enabled) {
            std::lock_guard<std::mutex> lk(y.m);
            printf("YIELD F=%.2f placed=%d offered=%d skipped(not-ready)=%d skipped(budget)=%d "
                   "prefill_stall_ms=%.1f max=%.1f\n",
                   y.F, y.placed, y.offered, y.skipped_not_ready, y.skipped_budget,
                   y.a_stall_ms, y.a_stall_max_ms);
        }
        if (gate.enabled) {
            printf("GATE F=%.2f decoder_spent_ms=%.1f prefill_wall_ms=%.1f\n",
                   gate.F, gate.dec_spent_ms.load(), ra.t_prompt_ms - ra.t_start_ms);
        }
    } else {
        fprintf(stderr, "unknown TWO_CTX_MODE '%s'\n", mode.c_str());
        return 1;
    }
    } // rep

    if (ctx_b) { llama_free(ctx_b); }
    if (ctx_a) { llama_free(ctx_a); }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
