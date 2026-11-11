#include <hip/hip_runtime.h>
#include <cstdio>
static void a(int d, hipDeviceAttribute_t at, const char* n){
  int v=-1; hipError_t e = hipDeviceGetAttribute(&v, at, d);
  printf("  %-34s %s\n", n, (e==hipSuccess)? std::to_string(v).c_str() : "n/a");
}
int main(){
  int n=0; hipGetDeviceCount(&n);
  for(int d=0; d<n; ++d){
    hipDeviceProp_t p{}; hipGetDeviceProperties(&p, d);
    printf("dev %d: %s | arch=%s | integrated=%d | canMapHostMemory=%d | unifiedAddressing=%d\n",
           d, p.name, p.gcnArchName, p.integrated, p.canMapHostMemory, p.unifiedAddressing);
    printf("  totalGlobalMem=%.1f GiB isLargeBar=%d memBusWidth=%d\n",
           p.totalGlobalMem/1073741824.0, p.isLargeBar, p.memoryBusWidth);
    a(d, hipDeviceAttributeManagedMemory, "ManagedMemory");
    a(d, hipDeviceAttributeConcurrentManagedAccess, "ConcurrentManagedAccess");
    a(d, hipDeviceAttributePageableMemoryAccess, "PageableMemoryAccess");
    a(d, hipDeviceAttributeUnifiedAddressing, "UnifiedAddressing");
    a(d, hipDeviceAttributeCooperativeLaunch, "CooperativeLaunch");
  }
  return 0;
}
