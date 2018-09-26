#ifndef __CUGL_APPS_DYNFU_CUDA_SORT_CUH__
#define __CUGL_APPS_DYNFU_CUDA_SORT_CUH__

namespace dynfu
{
namespace gpu
{
__device__ __forceinline__
unsigned int warpSortByKey(float key, unsigned int val)
{
  int lane_id = threadIdx.x % warpSize;

  return;
}
}
}

#endif /* __CUGL_APPS_DYNFU_CUDA_SORT_CUH__ */
