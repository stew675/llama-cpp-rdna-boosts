// 13-dma-from-mapped-file.cpp -- the WARM fill arm of the GTT expert store: the copy engine pulls from a
// REGISTERED, read-only MAP_SHARED mapping of the GGUF into the registered anon slot.  (P1.0 assumption #3.)
//
// WHY: on this part PageableMemoryAccess=0, so the only source a copy engine may read is a *registered* range
// (probe 02).  A registered RO file/page-cache mapping reads at 230.5 GB/s (probe 03), and the store's
// destination reads at 230.8 -- but the *fill rate* between them, at expert granularity, is unmeasured.
//
// The point of this arm is that the load is issued on the compute stream, i.e. it is in the same submit as the
// MUL_MAT_ID that needs it: no host round trip, no host memcpy, no staging ring.
//
// Build: hipcc -O2 -o 13-dma-from-mapped-file 13-dma-from-mapped-file.cpp
// Run:   ./13-dma-from-mapped-file <file> [offset=0]
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

__global__ void sum(const unsigned char* p, unsigned long long* out, size_t n){
  size_t stride=(size_t)gridDim.x*blockDim.x, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long acc=0;
  for(; i<n; i+=stride){ const volatile unsigned char* q=p; acc+=q[i]; }
  atomicAdd(out,acc);   // the total is order-independent, so it must equal the CPU sum
}
static unsigned long long gpu_sum(const void* p,size_t n){
  unsigned long long* d; CK(hipMalloc(&d,8)); CK(hipMemset(d,0,8));
  sum<<<512,256>>>((const unsigned char*)p,d,n); CK(hipDeviceSynchronize());
  unsigned long long v; CK(hipMemcpy(&v,d,8,hipMemcpyDeviceToHost)); hipFree(d); return v;
}
static unsigned long long cpu_sum(const unsigned char* p,size_t n){ unsigned long long v=0; for(size_t i=0;i<n;i++) v+=p[i]; return v; }

int main(int argc,char**argv){
  if(argc<2){ printf("usage: %s <gguf> [offset]\n",argv[0]); return 2; }
  const char* path=argv[1];
  const off_t off=argc>2?(off_t)atoll(argv[2]):0;
  const size_t WIN=256ull<<20;                     // the mapped window (a slice of the file)
  const size_t SIZES[2]={696*1024, 16*1024*1024};  // one expert slab, one layer span (4 KiB aligned)
  printf("13-dma-from-mapped-file: %s @ %lld, window %zu MiB\n", path, (long long)off, WIN>>20);

  int fd=open(path,O_RDONLY);
  if(fd<0){ perror("open"); return 2; }
  const off_t base=(off/4096)*4096;
  unsigned char* fmap=(unsigned char*)mmap(nullptr,WIN,PROT_READ,MAP_SHARED,fd,base);
  if(fmap==MAP_FAILED){ perror("mmap file"); return 2; }
  // the source is a pageable mapping: legal ONLY as a fill source, so register it first
  hipError_t e=hipHostRegister(fmap,WIN,hipHostRegisterMapped|hipHostRegisterReadOnly);
  printf("  hipHostRegister(Mapped|ReadOnly) on a MAP_SHARED file mapping: %s\n",
         e==hipSuccess?"OK":hipGetErrorString(e));
  if(e!=hipSuccess){ munmap(fmap,WIN); close(fd); return 2; }
  void* sdev=fmap; CK(hipHostGetDevicePointer(&sdev,fmap,0));

  // the destination: the store's shape (anon + THP + registered ReadOnly)
  const size_t DSZ=32ull<<20;
  unsigned char* dst=(unsigned char*)mmap(nullptr,DSZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  madvise(dst,DSZ,MADV_HUGEPAGE);
  memset(dst,0xAA,DSZ);
  CK(hipHostRegister(dst,DSZ,hipHostRegisterMapped|hipHostRegisterReadOnly));
  void* ddev=dst; CK(hipHostGetDevicePointer(&ddev,dst,0));

  hipStream_t st; CK(hipStreamCreate(&st));
  for(int i=0;i<2;i++){
    const size_t n=SIZES[i];
    const off_t at=(off-base);
    for(int warm=0; warm<2; ++warm){                 // warm=0: source may be cold, warm=1: page cache hit
      double t=now(); CK(hipMemcpyAsync(ddev, (const char*)sdev+at, n, hipMemcpyDeviceToDevice, st));
      CK(hipStreamSynchronize(st)); double dt=now()-t;
      printf("  %-14s %7.3f MiB pull %s in %8.3f ms = %6.2f GB/s\n",
             i?"layer span":"one expert", n/1048576.0, warm?"(warm)":"(first)", dt*1e3, n/dt/1e9);
    }
    const unsigned long long c=cpu_sum(fmap+at,n), g=gpu_sum((char*)ddev,n);
    printf("    verify %zu bytes: file %llu slot %llu %s\n", n, c, g, c==g?"(match)":"(MISMATCH)");
    // and the same bytes written by the host, for the "no copy engine" reference
    double t=now(); memcpy(dst,fmap+at,n); double dt=now()-t;
    printf("    reference: host memcpy into the slot  %8.3f ms = %6.2f GB/s\n", dt*1e3, n/dt/1e9);
  }
  CK(hipStreamDestroy(st)); CK(hipHostUnregister(dst)); CK(hipHostUnregister(fmap));
  munmap(dst,DSZ); munmap(fmap,WIN); close(fd);
  return 0;
}
