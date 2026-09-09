// fattn-diag.cpp — is the leak per-cell (partial columns) or only fully-masked columns?
// Staggered causal mask over nq rows at positions pos0..pos0+nq-1; garbage ONE column
// c* = pos0+1 which is masked ONLY for row 0 (all other rows unmasked at c*).
// Row 0's output must be invariant to V[c*]; if it changes -> per-element leak.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <dlfcn.h>

typedef ggml_backend_t (*init_fn)(int);
typedef int (*count_fn)(void);
static init_fn g_init; static count_fn g_count;

static float rnd(unsigned * s){ *s^=*s<<13;*s^=*s>>17;*s^=*s<<5; return ((float)(*s&0xFFFFFF)/8388608.0f)-1.0f; }

int main(int argc, char ** argv){
    if (argc < 7) { fprintf(stderr,"usage: %s <lib> <f16|bf16> <hsk> <nq> <kv> <pos0>\n", argv[0]); return 2; }
    const char * lib = argv[1];
    const bool bf16 = !strcmp(argv[2],"bf16");
    const int64_t hs = atoll(argv[3]), nq = atoll(argv[4]), kv = atoll(argv[5]), pos0 = atoll(argv[6]);
    const bool is_hip = strstr(lib,"hip") != nullptr;
    void * dl = dlopen(lib, RTLD_NOW|RTLD_GLOBAL);
    if(!dl){fprintf(stderr,"dlopen: %s\n", dlerror()); return 2;}
    g_init = (init_fn)dlsym(dl, is_hip?"ggml_backend_cuda_init":"ggml_backend_vk_init");
    g_count= (count_fn)dlsym(dl, is_hip?"ggml_backend_cuda_get_device_count":"ggml_backend_vk_get_device_count");
    ggml_backend_t backend = g_init(0);
    ggml_type kv_type = bf16?GGML_TYPE_BF16:GGML_TYPE_F16;
    ggml_init_params ip = {64ull*1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs, nq);
    ggml_tensor * k = ggml_new_tensor_2d(ctx, kv_type, hs, kv);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, kv_type, hs, kv);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf((float)hs), 0.0f, 0.0f);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(mask);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    unsigned s = 42;
    std::vector<float> qd(hs*nq); for(auto&x:qd) x=rnd(&s);
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size()*4);
    // K/V: real values everywhere (valid cells)
    std::vector<ggml_fp16_t> kd(hs*kv), vd(hs*kv);
    for(int64_t c=0;c<kv;++c) for(int64_t d=0;d<hs;++d){ kd[c*hs+d]=ggml_fp32_to_fp16(rnd(&s)*2.f); vd[c*hs+d]=ggml_fp32_to_fp16(rnd(&s)*0.5f); }
    // mask: row r at position pos0+r: 0 for c<=pos0+r, else -inf
    std::vector<ggml_fp16_t> md(kv*nq);
    for(int64_t r=0;r<nq;++r) for(int64_t c=0;c<kv;++c) md[r*kv+c]= (c<=pos0+r)? ggml_fp32_to_fp16(0.f):ggml_fp32_to_fp16(-INFINITY);
    auto upload=[&](const std::vector<ggml_fp16_t>&vv){ ggml_backend_tensor_set(v, vv.data(), 0, vv.size()*2); };
    auto upk=[&](const std::vector<ggml_fp16_t>&kk){ ggml_backend_tensor_set(k, kk.data(), 0, kk.size()*2); };
    upk(kd);
    if(bf16){ std::vector<ggml_bf16_t> kb(kd.size()), vb(vd.size());
        for(size_t i=0;i<kd.size();++i){ kb[i]=ggml_fp32_to_bf16(ggml_fp16_to_fp32(kd[i])); vb[i]=ggml_fp32_to_bf16(ggml_fp16_to_fp32(vd[i])); }
        ggml_backend_tensor_set(k, kb.data(),0,kb.size()*2); ggml_backend_tensor_set(v, vb.data(),0,vb.size()*2);
    }
    ggml_backend_tensor_set(mask, md.data(), 0, md.size()*2);
    auto compute=[&](std::vector<float>&o){ o.assign(hs*nq,0.f); ggml_backend_graph_compute(backend, gf); ggml_backend_tensor_get(out,o.data(),0,o.size()*4); };
    // baseline
    std::vector<float> o0,o1,o2;
    // (V for c* not set specially yet: vd[c*] is the legit rnd value.)
    const int64_t cstar = pos0+1; // masked for row 0 only (requires nq>=2, kv>cstar)
    std::vector<ggml_fp16_t> vA = vd, vB = vd;
    for(int64_t d=0;d<hs;++d){ vA[cstar*hs+d]=ggml_fp32_to_fp16(0.f); vB[cstar*hs+d]=ggml_fp32_to_fp16((d&1)?-37.0f:37.0f); }
    upload(vA); compute(o0);
    upload(vA); compute(o1);
    upload(vB); compute(o2);
    // compare row 0 only (must be invariant), and rows 1.. (change legitimately at row1 only, but c* unmasked for rows>=1)
    int det=0, leak=0; float lmax=0; int lp=-1;
    for(int64_t d=0; d<hs; ++d){ // row 0
        if(o0[d]!=o1[d]) det++;
        if(o0[d]!=o2[d]) { leak++; float dd=fabsf(o0[d]-o2[d]); if(dd>lmax){lmax=dd;lp=d;} }
    }
    printf("%s %s hsk=%lld nq=%lld kv=%lld pos0=%lld row0-invariant -> %s (det=%d leak=%d lmax=%.3e @%d)\n",
        is_hip?"hip":"vk", bf16?"bf16":"f16", (long long)hs,(long long)nq,(long long)kv,(long long)pos0,
        det?"DET":(leak?"LEAK-ROW0":"OK"), det, leak, lmax, lp);
    return 0;
}
