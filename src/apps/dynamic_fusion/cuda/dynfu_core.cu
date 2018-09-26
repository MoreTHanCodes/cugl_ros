#include <cugl_ros/apps/dynamic_fusion/cuda_internal.h>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/reduce.cuh>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/tsdf.cuh>

namespace dynfu
{
namespace gpu
{
__global__
void resetSurfelFlagsKernel(DeviceArrayHandle<uint> surfel_flags)
{
  const int stride = blockDim.x * gridDim.x;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < MAX_TRIANGLES_SIZE;
       i += stride)
  {
    surfel_flags.at(i) = 1;
  }
}

void resetSurfelFlags(DeviceArray<uint> &surfel_flags)
{
  int block = 256;
  int grid = min(divUp(MAX_TRIANGLES_SIZE, block), 512);

  resetSurfelFlagsKernel<<<grid, block>>>(surfel_flags.getHandle());
  checkCudaErrors(cudaGetLastError());
}
} // namespace gpu
} // namespace dynfu
