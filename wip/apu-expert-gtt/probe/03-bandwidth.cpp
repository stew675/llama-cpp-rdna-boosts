#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <vector>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
__global__ void rd(const unsigned int* p, unsigned long long* out, size_t n4){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n4; i+=stride){ const volatile unsigned int* q=p; unsigned int a0=q[i*4],a1=q[i*4+1],a2=q[i*4+2],a3=q[i*4+3]; acc+=(unsigned long long)a0+a1+a2+a3; }
  if(acc==0xdeadbeefULL) atomicAdd(out,acc);
}
static void bench(void* p,size_t bytes,const char* tag){
  size_t n4=bytes/16; unsigned long long* d; CK(hipMalloc(&d,8)); double best=1e9;
  for(int r=0;r<6;r++){ hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
    rd<<<512,256>>>((const unsigned int*)p,d,n4); hipEventRecord(b); CK(hipEventSynchronize(b));
    float ms=0; hipEventElapsedTime(&ms,a,b); if(ms<best)best=ms; hipEventDestroy(a); hipEventDestroy(b);}
  hipFree(d); printf("  %-52s %7.1f GB/s\n",tag,bytes/(best*1e6));
}
int main(){
  size_t SZ=1ull<<30; unsigned dev_flags=hipHostRegisterMapped;
  { unsigned char* d; CK(hipMalloc(&d,SZ)); CK(hipMemset(d,1,SZ)); bench(d,SZ,"hipMalloc (device/GTT)"); }
  unsigned char* anon=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  memset(anon,2,SZ);
  void* d1=anon; CK(hipHostRegister(anon,SZ,dev_flags)); CK(hipHostGetDevicePointer(&d1,anon,0));
  bench(d1,SZ,"anon, Mapped (writable)");
  void* d2=anon; CK(hipHostUnregister(anon));
  anon=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); memset(anon,2,SZ);
  d2=anon; CK(hipHostRegister(anon,SZ,dev_flags|hipHostRegisterReadOnly)); CK(hipHostGetDevicePointer(&d2,anon,0));
  bench(d2,SZ,"anon, Mapped|ReadOnly");
  int fd=open("/tmp/xnack-probe/bw.bin",O_RDONLY);
  unsigned char* fc=(unsigned char*)mmap(nullptr,SZ,PROT_READ,MAP_SHARED,fd,0);
  for(size_t i=0;i<SZ;i+=4096)(void)fc[i];
  void* d3=fc; CK(hipHostUnregister(anon)); CK(hipHostRegister(fc,SZ,dev_flags|hipHostRegisterReadOnly)); CK(hipHostGetDevicePointer(&d3,fc,0));
  bench(d3,SZ,"file/page-cache, Mapped|ReadOnly");
  return 0;
}
