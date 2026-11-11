#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
__global__ void sumw(const unsigned int* p, unsigned long long* out, size_t n4){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n4; i+=stride) acc += ((volatile const unsigned int*)p)[i];
  atomicAdd(out,acc);
}
static unsigned long long gpu_sum(const void* p,size_t bytes){
  unsigned long long* d; CK(hipMalloc(&d,8)); CK(hipMemset(d,0,8));
  sumw<<<512,256>>>((const unsigned int*)p,d,bytes/4);
  CK(hipGetLastError()); CK(hipDeviceSynchronize());
  unsigned long long h=0; CK(hipMemcpy(&h,d,8,hipMemcpyDeviceToHost)); hipFree(d); return h;
}
static unsigned long long expect(unsigned v,size_t bytes){ // sum of u32 words, mod 2^64
  unsigned long long n=bytes/4; unsigned long long w=(unsigned)v*0x01010101ULL; return w*n;
}
int main(int argc,char**argv){
  size_t SZ=1ull<<30; int ro = argc>1 && !strcmp(argv[1],"writable");
  unsigned char* buf=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(buf,SZ,MADV_HUGEPAGE);
  memset(buf,0x11,SZ);
  void* dev=buf;
  CK(hipHostRegister(buf,SZ,hipHostRegisterMapped|(ro?0:hipHostRegisterReadOnly)));
  CK(hipHostGetDevicePointer(&dev,buf,0));
  printf("registered %s\n", ro?"WRITABLE (coherent)":"ReadOnly (cached, 233 GB/s)");
  for(int it=0; it<4; it++){
    unsigned char pat = (it%2)?0x22:0x11;
    unsigned long long s = gpu_sum(dev,SZ);
    printf("  iter %d: gpu_sum=%016llx  expected(%02x)=%016llx  %s\n", it, s, pat, expect(pat,SZ), (s==expect(pat,SZ))?"FRESH":"STALE/other");
    /* host overwrites the whole region in place, no unregister, no flush */
    memset(buf, pat==0x11?0x22:0x11, SZ);
  }
  return 0;
}
