#ifndef __CUGL_APPS_DYNFU_CUDA_KNN_FIELD_CUH__
#define __CUGL_APPS_DYNFU_CUDA_KNN_FIELD_CUH__

#if defined(__CUDACC__)
  #define __CUGL_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
  #define __CUGL_HOST_DEVICE__
#endif

namespace dynfu
{
namespace gpu
{
struct CoarseKnnFieldHandle
{
  __CUGL_HOST_DEVICE__ CoarseKnnFieldHandle(unsigned int *grid_start_arg, unsigned int *grid_end_arg, unsigned short *index_list_arg)
      : grid_start(grid_start_arg),
        grid_end(grid_end_arg),
        index_list(index_list_arg)
  {
  }

  __CUGL_HOST_DEVICE__ unsigned int getStartAt(unsigned int grid_hash) const
  {
    return grid_start[grid_hash];
  }

  __CUGL_HOST_DEVICE__ unsigned int getEndAt(unsigned int grid_hash) const
  {
    return grid_end[grid_hash];
  }

  __CUGL_HOST_DEVICE__ unsigned short getIndexAt(unsigned int x) const
  {
    return index_list[x];
  }

  __CUGL_HOST_DEVICE__ void setStartAt(unsigned int grid_hash, unsigned int value)
  {
    grid_start[grid_hash] = value;
  }

  __CUGL_HOST_DEVICE__ void setEndAt(unsigned int grid_hash, unsigned int value)
  {
    grid_end[grid_hash] = value;
  }

  __CUGL_HOST_DEVICE__ void setIndexAt(unsigned int x, unsigned short value)
  {
    index_list[x] = value;
  }

  unsigned int *grid_start;
  unsigned int *grid_end;
  unsigned short *index_list;
};

struct FineKnnFieldHandle
{
  __CUGL_HOST_DEVICE__ FineKnnFieldHandle(unsigned int *grid_start_arg, unsigned short *index_list_arg)
      : grid_start(grid_start_arg),
        index_list(index_list_arg)
  {
  }

  __CUGL_HOST_DEVICE__ unsigned int getStartAt(unsigned int grid_hash) const
  {
    return grid_start[grid_hash];
  }

  __CUGL_HOST_DEVICE__ unsigned short getIndexAt(unsigned int x) const
  {
    return index_list[x];
  }

  __CUGL_HOST_DEVICE__ void setStartAt(unsigned int grid_hash, unsigned int value)
  {
    grid_start[grid_hash] = value;
  }

  __CUGL_HOST_DEVICE__ void setIndexAt(unsigned int x, unsigned short value)
  {
    index_list[x] = value;
  }

  unsigned int *grid_start;
  unsigned short *index_list;
};
} // namespace gpu
} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_CUDA_KNN_FIELD_CUH__ */
