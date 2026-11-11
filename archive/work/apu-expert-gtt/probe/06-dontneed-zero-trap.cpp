#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){printf("HIPFAIL %s: %s\n",#x,hipGetErrorString(e));exit(2);} }while(0)
__global__ void sumw(const unsigned int* p, unsigned long long* out, size_t n){ size_t s=(size_t)gridDim.x*blockDim.x,i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; unsigned long long a=0; for(;i<n;i+=s)a+=((volatile const unsigned int*)p)[i]; atomicAdd(out,a);}
static unsigned long long gs(const void*p,size_t b){ unsigned long long*d; CK(hipMalloc(&d,8)); CK(hipMemset(d,0,8)); sumw<<<512,256>>>((const unsigned int*)p,d,b/4); CK(hipDeviceSynchronize()); unsigned long long h; CK(hipMemcpy(&h,d,8,hipMemcpyDeviceToHost)); hipFree(d); return h;}
int main(){
  size_t SZ=1ull<<30;
  unsigned char* a=(unsigned char*)mmap(nullptr,SZ,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  memset(a,0x33,SZ); void* d=a;
  CK(hipHostRegister(a,SZ,hipHostRegisterMapped|hipHostRegisterReadOnly)); CK(hipHostGetDevicePointer(&d,a,0));
  printf("  GPU sum before DONTNEED: %016llx   (0x33333333 pattern expected)\n", gs(d,SZ));
  madvise(a,SZ,MADV_DONTNEED);
  printf("  GPU sum after  DONTNEED: %016llx   %s\n", gs(d,SZ),
         gs(d,SZ)==0 ? "<<< SILENTLY ZEROED (no fault!)" : "(content survived)");
  return 0;
}
