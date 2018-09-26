#ifndef __CUGL_APPS_DYNFU_CUDA_TSDF_CUH__
#define __CUGL_APPS_DYNFU_CUDA_TSDF_CUH__

namespace dynfu
{
namespace gpu
{
const int DIVISOR = 32767;

__device__ __forceinline__
void updateVoxel(float tsdf, uchar4 rgba, int weight, TsdfVoxelType &tsdf_voxel, RgbaVoxelType &rgba_voxel)
{
  float tsdf_ref = float(tsdf_voxel.x) / DIVISOR;
  int weight_ref = int(tsdf_voxel.y);
  // int r_ref = int(rgba_voxel.x);
  // int g_ref = int(rgba_voxel.y);
  // int b_ref = int(rgba_voxel.z);

  float tsdf_new = (tsdf * float(weight) + tsdf_ref * float(weight_ref)) / float(weight + weight_ref);
  int weight_new = max(min(weight + weight_ref, 256), 0);
  // int r_new = min((int(rgba.x) * weight + r_ref * weight_ref) / (weight + weight_ref), 255);
  // int g_new = min((int(rgba.y) * weight + g_ref * weight_ref) / (weight + weight_ref), 255);
  // int b_new = min((int(rgba.z) * weight + b_ref * weight_ref) / (weight + weight_ref), 255);
  int r_new = int(rgba.x);
  int g_new = int(rgba.y);
  int b_new = int(rgba.z);

  if (weight_ref != 0 && fabs(tsdf_new - tsdf_ref) > 0.3f)
  {
    tsdf_new = tsdf_ref;
    weight_new = weight_ref;
  }

  int fixedp = max(-DIVISOR, min(DIVISOR, __float2int_rz(tsdf_new * DIVISOR)));

  tsdf_voxel = make_short2(fixedp, weight_new);
  rgba_voxel = make_uchar4(r_new, g_new, b_new, 0);
}

__device__ __forceinline__
void updateVoxel_neg(TsdfVoxelType &tsdf_voxel)
{
  float tsdf_ref = float(tsdf_voxel.x) / DIVISOR;
  int weight_ref = int(tsdf_voxel.y);

  int weight_new = max(min(weight_ref - 1, 256), 0);

  int fixedp = max(-DIVISOR, min(DIVISOR, __float2int_rz(tsdf_ref * DIVISOR)));

  tsdf_voxel = make_short2(fixedp, weight_new);
}
} // namespace gpu
} // namespace dynfu
#endif /* __CUGL_APPS_DYNFU_CUDA_TSDF_CUH__ */
