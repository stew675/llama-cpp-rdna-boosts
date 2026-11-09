#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
__global__ void rd(const unsigned int* p, unsigned long long* out, size_t n4){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n4; i+=stride){ const volatile unsigned int* q=p; unsigned int a0=q[i*4],a1=q[i*4+1],a2=q[i*4+2],a3=q[i*4+3]; acc+=(unsigned long long)a0+a1+a2+a3; }
  if(acc==0xdeadbeefULL) atomicAdd(out,acc);
}
static double gpu_read(const void* p,size_t bytes,const char* tag){
  size_t n4=bytes/16; unsigned long long* d; CK(hipMalloc(&d,8)); double best=1e9;
  for(int r=0;r<6;r++){ hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
    rd<<<512,256>>>((const unsigned int*)p,d,n4); hipEventRecord(b); CK(hipEventSynchronize(b));
    float ms=0; hipEventElapsedTime(&ms,a,b); if(ms<best)best=ms; hipEventDestroy(a); hipEventDestroy(b);}
  hipFree(d); double gbs=bytes/(best*1e6);
  printf("    %-46s %7.1f GB/s\n",tag,gbs); return gbs;
}
int main(int argc,char**argv){
  const char* src=argv[1]; size_t SZ=1ull<<30;
  unsigned char* buf=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(buf,SZ,MADV_HUGEPAGE);
  int fd=open(src,O_RDONLY|O_DIRECT);
  double t=now(); ssize_t got=pread(fd,buf,SZ,0); double od=now()-t;
  printf("  O_DIRECT pread 1 GiB into the buffer: %.2f s = %.1f GB/s (%zd bytes)\n", od, SZ/od/1e9, got);
  close(fd);
  /* can a WRITABLE buffer be registered READ-ONLY for the GPU? */
  hipError_t e1=hipHostRegister(buf,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly);
  void* d1=buf; if(e1==hipSuccess) CK(hipHostGetDevicePointer(&d1,buf,0));
  printf("  hipHostRegister(Mapped|ReadOnly) on a PROT_WRITE buffer: %s\n", e1==hipSuccess?"OK":hipGetErrorString(e1));
  if(e1==hipSuccess) gpu_read(d1,SZ,"GPU read: registered ReadOnly");
  CK(hipHostUnregister(buf));
  CK(hipHostRegister(buf,SZ,hipHostRegisterMapped));
  void* d2=buf; CK(hipHostGetDevicePointer(&d2,buf,0));
  gpu_read(d2,SZ,"GPU read: registered writable");
  CK(hipHostUnregister(buf));

  /* refill cycle: unregister -> overwrite in place -> register RO again */
  t=now(); CK(hipHostRegister(buf,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly)); double r1=now()-t;
  t=now(); CK(hipHostUnregister(buf)); double u1=now()-t;
  t=now(); CK(hipHostRegister(buf,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly)); double r2=now()-t;
  CK(hipHostUnregister(buf));
  printf("  refill cycle: register %.2f ms, unregister %.2f ms, re-register %.2f ms\n", r1*1e3,u1*1e3,r2*1e3);

  /* what a warm->hot promotion costs when the source is an ordinary page-cache page */
  int fd2=open(src,O_RDONLY); unsigned char* map=(unsigned char*)mmap(nullptr,SZ,PROT_READ,MAP_PRIVATE,fd2,0);
  for(size_t i=0;i<SZ;i+=4096)(void)map[i];
  t=now(); memcpy(buf,map,SZ); double mc=now()-t;
  printf("  memcpy page cache -> buffer 1 GiB: %.3f s = %.1f GB/s\n", mc, SZ/mc/1e9);
  return 0;
}
