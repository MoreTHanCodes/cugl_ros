#ifndef __CUGL_APPS_DYNFU_CUDA_REDUCE_CUH__
#define __CUGL_APPS_DYNFU_CUDA_REDUCE_CUH__

// Note: we only support 32bit type (float or int) for the current

namespace dynfu
{
namespace gpu
{
// // Utility class used to avoid linker errors with extern
// // unsized shared memory arrays with templated type
// template<class T>
// struct getSharedMemory
// {
//   __device__ __forceinline__ operator       T *()
//   {
//     extern __shared__ int __smem[];
//     return (T *)__smem;
//   }
// 
//   __device__ __forceinline__ operator const T *() const
//   {
//     extern __shared__ int __smem[];
//     return (T *)__smem;
//   }
// };
// 
// template <class T>
// __device__ __forceinline__
// T warpReduceSum(T val)
// {
//   for (int offset = warpSize/2; offset > 0; offset /= 2)
//   {
//     val += __shfl_down_sync(val, offset);
//   }
// 
//   return val;
// }
// 
// template <class T>
// __device__ __forceinline__
// T blockReduceSum_New(T val)
// {
//   // T *sdata = getSharedMemory<T>();
//   static __shared__ T sdata[32];
//   
//   int lane_id = threadIdx.x % warpSize;
//   int warp_id = threadIdx.x / warpSize;
// 
//   val = warpReduceSum<T>(val);
// 
//   if (lane_id == 0)
//     sdata[warp_id] = val;
// 
//   __syncthreads();
// 
//   val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane_id] : T(0);
// 
//   if (warp_id == 0)
//     val = warpReduceSum<T>(val);
// 
//   return val;
// }
// 
// template <class T>
// __device__ __forceinline__
// T blockReduceSum_Old(T val)
// {
//   T *sdata = getSharedMemory<T>();
// 
//   int blockSize = blockDim.x;
//   int tid = threadIdx.x;
// 
//   sdata[tid] = val;
// 
//   __syncthreads();
// 
//   // do reduction in shared mem
//   if ((blockSize >= 512) && (tid < 256))
//     sdata[tid] = val = val + sdata[tid + 256];
// 
//   __syncthreads();
// 
//   if ((blockSize >= 256) &&(tid < 128))
//     sdata[tid] = val = val + sdata[tid + 128];
// 
//    __syncthreads();
// 
//   if ((blockSize >= 128) && (tid <  64))
//     sdata[tid] = val = val + sdata[tid +  64];
// 
//   __syncthreads();
// 
//   if ( tid < 32 )
//   {
//     // Fetch final intermediate sum from 2nd warp
//     if (blockSize >=  64) val += sdata[tid + 32];
// 
//     // Reduce final warp using shuffle
//     val = warpReduceSum<T>(val);
//   }
// 
//   return val;
// }

__device__ __forceinline__
float warpReduceSum(float val)
{
  for(int offset = warpSize/2; offset > 0; offset /= 2)
  {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // TODO: only for cuda 8 or older
    // val += __shfl_down(val, offset);
  }

  return val;
}

__device__ __forceinline__
float blockReduceSum(float val)
{
  __shared__ float shared[32];
  
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  val = warpReduceSum(val);

  if(lane_id == 0)
    shared[warp_id] = val;

  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane_id] : 0.f;

  if(warp_id == 0)
    val = warpReduceSum(val);

  return val;
}
} // namespace gpu
} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_CUDA_REDUCE_CUH__ */
