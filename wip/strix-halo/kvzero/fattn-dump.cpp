// dump FA output for a given config to a file (value-preservation checks)
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <dlfcn.h>
typedef ggml_backend_t (*init_fn)(int);
static float rnd(unsigned * s){ *s^=*s<<13;*s^=*s>>17;*s^=*s<<5; return ((float)(*s&0xFFFFFF)/8388608.0f)-1.0f; }
int main(int argc,char**argv){
    // argv: lib mode hs nq kv nvalid nh nh_kv outfile
    const char*lib=argv[1]; bool bf16=!strcmp(argv[2],"bf16");
    int64_t hs=atoll(argv[3]),nq=atoll(argv[4]),kv=atoll(argv[5]),nv=atoll(argv[6]),nh=atoll(argv[7]),nh_kv=atoll(argv[8]);
    const char*out=argv[9];
    void*dl=dlopen(lib,RTLD_NOW|RTLD_GLOBAL); if(!dl){fprintf(stderr,"dlopen %s\n",dlerror());return 2;}
    init_fn init=(init_fn)dlsym(dl,strstr(lib,"hip")?"ggml_backend_cuda_init":"ggml_backend_vk_init");
    ggml_backend_t backend=init(0);
    ggml_type kt=bf16?GGML_TYPE_BF16:GGML_TYPE_F16;
    ggml_init_params ip={64ull*1024*1024,nullptr,true};
    ggml_context*ctx=ggml_init(ip);
    ggml_tensor*q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hs,nq,nh);
    ggml_tensor*k=ggml_new_tensor_3d(ctx,kt,hs,kv,nh_kv);
    ggml_tensor*v=ggml_new_tensor_3d(ctx,kt,hs,kv,nh_kv);
    ggml_tensor*mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,kv,nq);
    ggml_tensor*o=ggml_flash_attn_ext(ctx,q,k,v,mask,1.0f/sqrtf((float)hs),0.f,0.f);
    ggml_set_input(q);ggml_set_input(k);ggml_set_input(v);ggml_set_input(mask);
    ggml_cgraph*gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,o);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    unsigned s=7;
    std::vector<float> qd(hs*nq*nh); for(auto&x:qd)x=rnd(&s);
    ggml_backend_tensor_set(q,qd.data(),0,qd.size()*4);
    std::vector<ggml_fp16_t> kd(hs*kv),vd(hs*kv),md(kv*nq);
    // mode: nv==kv -> fully live; nv<kv -> masked tail with garbage V
    for(int64_t c=0;c<kv;++c)for(int64_t d=0;d<hs;++d){kd[c*hs+d]=ggml_fp32_to_fp16(rnd(&s)*2.f);vd[c*hs+d]=ggml_fp32_to_fp16(rnd(&s)*0.5f);}
    for(int64_t c=nv;c<kv;++c)for(int64_t d=0;d<hs;++d)vd[c*hs+d]=getenv("FDUMP_ZERO_TAIL")?ggml_fp32_to_fp16(0.f):ggml_fp32_to_fp16((((c*131+d*17+7)%101-50)/100.f));
    if (getenv("FDUMP_CAUSAL")) {
        for(int64_t r=0;r<nq;++r)for(int64_t c=0;c<kv;++c)md[r*kv+c]=ggml_fp32_to_fp16(c<=r?0.f:-INFINITY);
    } else {
        for(int64_t r=0;r<nq;++r)for(int64_t c=0;c<kv;++c)md[r*kv+c]=ggml_fp32_to_fp16(c<nv?0.f:-INFINITY);
    }
    auto up=[](ggml_tensor*t,const void*d,size_t n){ggml_backend_tensor_set(t,d,0,n);};
    up(k,kd.data(),kd.size()*2); up(mask,md.data(),md.size()*2);
    if(bf16){std::vector<ggml_bf16_t>vb(vd.size());for(size_t i=0;i<vd.size();++i)vb[i]=ggml_fp32_to_bf16(ggml_fp16_to_fp32(vd[i]));up(v,vb.data(),vb.size()*2);}
    else up(v,vd.data(),vd.size()*2);
    ggml_backend_graph_compute(backend,gf);
    std::vector<float> od(hs*nq*nh);
    ggml_backend_tensor_get(o,od.data(),0,od.size()*4);
    FILE*f=fopen(out,"wb"); fwrite(od.data(),4,od.size(),f); fclose(f);
    fprintf(stderr,"dumped %zu floats to %s\n",od.size(),out);
    return 0;
}
