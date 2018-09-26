#include <cugl_ros/apps/dynamic_fusion/cuda_internal.h>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/utils.cuh>

namespace dynfu
{
namespace gpu
{
/*
 * Image Processing Functions built on CUDA
 */
__global__
void depthToColorKernel(float2 thresholds,
                        float depth_scale,
                        int width, int height,
                        const DeviceArrayHandle2D<ushort> src_depth,
                        DeviceArrayHandle2D<uchar4> dst_color)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height)
  {
    ushort depth_raw = src_depth.at(y, x);
    float depth_in_meters = (float)(depth_raw) * depth_scale;

    uchar4 rgba = make_uchar4(20, 5, 0, 1);

    if(depth_in_meters > thresholds.x && depth_in_meters < thresholds.y)
    {
      uchar f = (uchar)(255.0f * (depth_in_meters - thresholds.x) / (thresholds.y - thresholds.x));
      rgba.x = 255 - f;
      rgba.y = 0;
      rgba.z = f;
    }

    dst_color.at(y, x) = rgba;
  }
}

void depthToColor(float2 thresholds,
                  float depth_scale,
                  int width, int height,
                  const DeviceArray2D<ushort> &src_depth,
                  DeviceArray2D<uchar4> &dst_color)
{
  dim3 block(32, 16);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));

  depthToColorKernel<<<grid, block>>>(thresholds, depth_scale,
                                      width, height,
                                      src_depth.getHandle(),
                                      dst_color.getHandle());

  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void convertColorKernel(int width, int height,
                        bool draw_mask,
                        int2 mask_start,
                        int2 mask_end,
                        const DeviceArrayHandle2D<uchar> src_color,
                        DeviceArrayHandle2D<uchar4> dst_color)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height)
  {
    uchar color_r = src_color.at(y, 3*x + 0);
    uchar color_g = src_color.at(y, 3*x + 1);
    uchar color_b = src_color.at(y, 3*x + 2);

    bool in_mask = (draw_mask && x > mask_start.x && y > mask_start.y && x < mask_end.x && y < mask_end.y);

    uchar4 rgba = (in_mask) ? make_uchar4(255, 0, 0, 1) : make_uchar4(color_r, color_g, color_b, 1);

    dst_color.at(y, x) = rgba;
  }
}

void convertColor(int width, int height,
                  bool draw_mask,
                  int2 mask_start,
                  int2 mask_end,
                  const DeviceArray2D<uchar> &src_color,
                  DeviceArray2D<uchar4> &dst_color)
{
  dim3 block(32, 16);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));

  convertColorKernel<<<grid, block>>>(width, height,
                                      draw_mask,
                                      mask_start,
                                      mask_end,
                                      src_color.getHandle(),
                                      dst_color.getHandle());

  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void truncateDepthKernel(float2 depth_thresholds,
                         float4 color_thresholds,
                         bool draw_mask,
                         int2 mask_start,
                         int2 mask_end,
                         const Intrinsics intrin,
                         const DeviceArrayHandle2D<ushort> src_depth,
                         const DeviceArrayHandle2D<uchar> src_color,
                         DeviceArrayHandle2D<ushort> dst_depth_16u,
                         DeviceArrayHandle2D<float> dst_depth_32f)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < intrin.width && y < intrin.height)
  {
    // truncation via depth
    ushort depth_raw = src_depth.at(y, x);
    float depth_in_meters = (float)(depth_raw) * intrin.depth_scale;

    // truncation via color (HSV space)
    float color_r = __uint2float_rd(src_color.at(y, 3*x + 0));
    float color_g = __uint2float_rd(src_color.at(y, 3*x + 1));
    float color_b = __uint2float_rd(src_color.at(y, 3*x + 2));

    const float max = fmaxf(color_r, fmaxf(color_g, color_b));
    const float min = fminf(color_r, fminf(color_g, color_b));

    float color_v = max / 255.f;
    float color_h, color_s;

    if (max == 0.f)
    {
      color_h = 0.f;
      color_s = 0.f;
    }

    const float diff = max - min;
    color_s = diff / max;

    if (min == max)
    {
      color_h = 0.f;
    }

    if (max == color_r) color_h = 60.f * ((color_g - color_b) / diff);
    else if (max == color_g) color_h = 60.f * (2.f + (color_b - color_r) / diff);
    else color_h = 60.f * (4.f + (color_r - color_g) / diff);

    if (color_h < 0.f) color_h += 360.f;

    bool in_mask = (draw_mask && x > mask_start.x && y > mask_start.y && x < mask_end.x && y < mask_end.y);

    bool valid = (depth_in_meters > depth_thresholds.x &&
                  depth_in_meters < depth_thresholds.y &&
                  color_h > color_thresholds.x &&
                  color_h < color_thresholds.y &&
                  color_s > color_thresholds.z &&
                  color_v > color_thresholds.w &&
                  !in_mask);


    if(!valid)
    {
      depth_raw = 0;
      depth_in_meters = 0;
    }

    float xl = (x - intrin.cx) / intrin.fx;
    float yl = (y - intrin.cy) / intrin.fy;
    float lambda = sqrtf(xl * xl + yl * yl + 1);

    dst_depth_16u.at(y, x) = depth_raw;
    dst_depth_32f.at(y, x) = depth_in_meters * lambda;
  }
}

void truncateDepth(float2 depth_thresholds,
                   float4 color_thresholds,
                   bool draw_mask,
                   int2 mask_start,
                   int2 mask_end,
                   const Intrinsics &intrin,
                   const DeviceArray2D<ushort> &src_depth,
                   const DeviceArray2D<uchar> &src_color,
                   DeviceArray2D<ushort> &dst_depth_16u,
                   DeviceArray2D<float> &dst_depth_32f)
{
  dim3 block(32, 16);
  dim3 grid(divUp(intrin.width, block.x), divUp(intrin.height, block.y));

  truncateDepthKernel<<<grid, block>>>(depth_thresholds,
                                       color_thresholds,
                                       draw_mask,
                                       mask_start,
                                       mask_end,
                                       intrin,
                                       src_depth.getHandle(),
                                       src_color.getHandle(),
                                       dst_depth_16u.getHandle(),
                                       dst_depth_32f.getHandle());

  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void bilateralFilterKernel(int cols, int rows,
                           float sigma_space2_inv_half,
                           float sigma_color2_inv_half,
                           const DeviceArrayHandle2D<ushort> src_depth,
                           DeviceArrayHandle2D<ushort> dst_depth)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= cols || y >= rows)
    return;

  const int R = 6;
  const int D = R * 2 + 1;

  int value = src_depth.at(y, x);

  int tx = min(x - D / 2 + D, cols - 1);
  int ty = min(y - D / 2 + D, rows - 1);

  float sum1 = 0;
  float sum2 = 0;

  for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
  {
    for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
    {
      int tmp = src_depth.at(cy, cx);

      float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      float color2 = (value - tmp) * (value - tmp);

      float weight = __expf (-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

      sum1 += tmp * weight;
      sum2 += weight;
    }
  }

  int res = __float2int_rn (sum1 / sum2);

  dst_depth.at(y, x) = max(0, min(res, SHRT_MAX));
}
                     
void bilateralFilter(int width, int height,
                     const DeviceArray2D<ushort> &src_depth,
                     DeviceArray2D<ushort> &dst_depth)
{
  float sigma_color = 30;
  float sigma_space = 4.5;

  dim3 block(32, 16);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));

  // cudaFuncSetCacheConfig(bilateralFilter, cudaFuncCachePreferL1);
  bilateralFilterKernel<<<grid, block>>>(width, height,
                                         0.5f / (sigma_space * sigma_space),
                                         0.5f / (sigma_color * sigma_color),
                                         src_depth.getHandle(),
                                         dst_depth.getHandle());

  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void createVertexImageKernel(const Intrinsics intrin,
                             const DeviceArrayHandle2D<ushort> src_depth,
                             DeviceArrayHandle2D<float4> dst_vertex)
{
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if(u < intrin.width && v < intrin.height)
  {
    float z = intrin.depth_scale * (float)(src_depth.at(v, u));

    float vx = z * (u - intrin.cx) / intrin.fx;
    float vy = z * (v - intrin.cy) / intrin.fy;
    float vz = z;

    float4 vert = (z != 0) ? make_float4(vx, vy, vz, 1.f) : make_float4(quiet_nanf(), quiet_nanf(), quiet_nanf(), 1.f);

    dst_vertex.at(v, u) = vert;
  }
}

void createVertexImage(const Intrinsics &intrin,
                       const DeviceArray2D<ushort> &src_depth,
                       DeviceArray2D<float4> &dst_vertex)
{
  dim3 block(32, 16);
  dim3 grid(divUp(intrin.width, block.x), divUp(intrin.height, block.y));

  createVertexImageKernel<<<grid, block>>>(intrin,
                                           src_depth.getHandle(),
                                           dst_vertex.getHandle());

  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
float3 computeDominantEigenVector(float cov[6])
{
  // use power method to find dominant eigenvector
  float3 v = make_float3(1.f, 1.f, 1.f);

  // 8 iterations seems to be more than enough
  for (int i = 0; i < 8; i++)
  {
    float x = v.x * cov[0] + v.y * cov[1] + v.z * cov[2];
    float y = v.x * cov[1] + v.y * cov[3] + v.z * cov[4];
    float z = v.x * cov[2] + v.y * cov[4] + v.z * cov[5];
    float m = max(max(x, y), z);
    float iv = 1.f / m;
    v = make_float3(x * iv, y * iv, z * iv);
  }

  return v;
}

__global__
void createNormalImageKernel(int width, int height,
                             const DeviceArrayHandle2D<float4> src_vertex,
                             DeviceArrayHandle2D<float4> dst_normal)
{
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u >= width || v >= height)
    return;

  dst_normal.at(v, u) = make_float4(quiet_nanf(), quiet_nanf(), quiet_nanf(), 0.f);

  if (isnan(src_vertex.at(v, u).x))
    return;

  const int kx = 7;
  const int ky = 7;
  const int kstep = 1;

  int ty = min(v - ky / 2 + ky, height - 1);
  int tx = min(u - kx / 2 + kx, width - 1);

  float3 centroid = make_float3(0.f);
  int counter = 0;
  for (int cy = max(v - ky / 2, 0); cy < ty; cy += kstep)
  {
    for (int cx = max(u - kx / 2, 0); cx < tx; cx += kstep)
    {
      float3 vertex = make_float3(src_vertex.at(cy, cx));
      if (!isnan(vertex.x))
      {
        centroid += vertex;
        ++counter;
      }
    }
  }

  if (counter < kx * ky / 2)
    return;

  float counter_inv = 1.f / counter;
  centroid *= counter_inv;

  // store cov as an upper triangular mat in row-major order
  float cov[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int cy = max(v - ky / 2, 0); cy < ty; cy += kstep)
  {
    for (int cx = max(u - kx / 2, 0); cx < tx; cx += kstep)
    {
      float3 vertex = make_float3(src_vertex.at(cy, cx));
      if (!isnan(vertex.x))
      {
        float3 cent_to_vert = vertex - centroid;

        cov[0] += cent_to_vert.x * cent_to_vert.x;
        cov[1] += cent_to_vert.x * cent_to_vert.y;
        cov[2] += cent_to_vert.x * cent_to_vert.z;
        cov[3] += cent_to_vert.y * cent_to_vert.y;
        cov[4] += cent_to_vert.y * cent_to_vert.z;
        cov[5] += cent_to_vert.z * cent_to_vert.z;
      }
    }
  }

  // approximate the dominant eigenvector of the covariance matrix
  // float3 n = computeDominantEigenVector(cov);

  typedef Eigen33::Mat33 Mat33;
  Eigen33 eigen33 (cov);

  Mat33 tmp;
  Mat33 vec_tmp;
  Mat33 evecs;
  float3 evals;
  eigen33.compute(tmp, vec_tmp, evecs, evals);

  dst_normal.at(v, u) = make_float4(normalize(evecs[0]), 0.f);
}

void createNormalImage(int width, int height,
                       const DeviceArray2D<float4> &src_vertex,
                       DeviceArray2D<float4> &dst_normal)
{
  dim3 block(32, 16);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));


  createNormalImageKernel<<<grid, block>>>(width, height,
                                           src_vertex.getHandle(),
                                           dst_normal.getHandle());

  checkCudaErrors(cudaGetLastError());
}

} // namespace gpu
} // namespace dynfu
