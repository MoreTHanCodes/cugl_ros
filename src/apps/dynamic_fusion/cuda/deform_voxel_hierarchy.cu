#include <cugl_ros/apps/dynamic_fusion/cuda_internal.h>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/reduce.cuh>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/tsdf.cuh>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/utils.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

// #define CUDA8

namespace dynfu
{
namespace gpu
{
// Image Textures
texture<float4, cudaTextureType2D, cudaReadModeElementType> vertex_texture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> normal_texture;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_texture;
texture<uchar, cudaTextureType2D, cudaReadModeElementType> color_texture;

texture<float4, cudaTextureType2D, cudaReadModeElementType> vertex_true_texture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> normal_true_texture;

// Voxel Blending Textures
texture<TransformType, 1, cudaReadModeElementType> transform_texture;
texture<HashEntryType, 1, cudaReadModeElementType> block_hash_table_texture;
texture<float4, 1, cudaReadModeElementType> blend_weight_table_texture;
texture<int4, 1, cudaReadModeElementType> blend_code_table_texture;
// Marching Cubes Textures
texture<int, 1, cudaReadModeElementType> tri_texture;
texture<int, 1, cudaReadModeElementType> num_verts_texture;
// Voxel Block Textures
texture<TsdfVoxelType, 1, cudaReadModeElementType> tsdf_voxel_texture;
texture<RgbaVoxelType, 1, cudaReadModeElementType> rgba_voxel_texture;
// Mesh Textures
texture<float4, 1, cudaReadModeElementType> mesh_vert_texture;
texture<float4, 1, cudaReadModeElementType> mesh_norm_texture;
// Deform Energy Term Textures
texture<float4, 1, cudaReadModeElementType> elem_data_texture;
texture<uint, 1, cudaReadModeElementType> elem_id_texture;
texture<float, 1, cudaReadModeElementType> elem_weight_texture;

__device__ uint blocks_done = 0;
__device__ int global_count = 0;
__device__ int output_count;
__device__ float global_value = 0.f;
__device__ float output_value;

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void bindImageTextures(int width, int height,
                       const VertexImage &vertex_image,
                       const NormalImage &normal_image,
                       const DepthImage32f &depth_image_32f,
                       const ColorImage8u &color_image_8u)
{
  cudaChannelFormatDesc vertex_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture2D(0, vertex_texture, reinterpret_cast<const void *>(vertex_image.getDevicePtr()), vertex_desc, width, height, vertex_image.getPitch()));

  cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture2D(0, normal_texture, reinterpret_cast<const void *>(normal_image.getDevicePtr()), normal_desc, width, height, normal_image.getPitch()));

  cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaBindTexture2D(0, depth_texture, reinterpret_cast<const void *>(depth_image_32f.getDevicePtr()), depth_desc, width, height, depth_image_32f.getPitch()));

  cudaChannelFormatDesc color_desc = cudaCreateChannelDesc<uchar>();
  checkCudaErrors(cudaBindTexture2D(0, color_texture, reinterpret_cast<const void *>(color_image_8u.getDevicePtr()), color_desc, 3*width, height, color_image_8u.getPitch()));
}

void unbindImageTextures()
{
  checkCudaErrors(cudaUnbindTexture(vertex_texture));
  checkCudaErrors(cudaUnbindTexture(normal_texture));
  checkCudaErrors(cudaUnbindTexture(depth_texture));
  checkCudaErrors(cudaUnbindTexture(color_texture));
}

void bindImageTrueTextures(int width, int height,
                           const VertexImage &vertex_image,
                           const NormalImage &normal_image)
{
  cudaChannelFormatDesc vertex_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture2D(0, vertex_true_texture, reinterpret_cast<const void *>(vertex_image.getDevicePtr()), vertex_desc, width, height, vertex_image.getPitch()));

  cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture2D(0, normal_true_texture, reinterpret_cast<const void *>(normal_image.getDevicePtr()), normal_desc, width, height, normal_image.getPitch()));
}

void unbindImageTrueTextures()
{
  checkCudaErrors(cudaUnbindTexture(vertex_true_texture));
  checkCudaErrors(cudaUnbindTexture(normal_true_texture));
}

void bindBlendingTextures(const DeviceArray<TransformType> &transform_array,
                          const DeviceArray<HashEntryType> &block_hash_table,
                          const DeviceArray<float4> &blend_weight_table,
                          const DeviceArray<int4> &blend_code_table)
{
  cudaChannelFormatDesc transform_desc = cudaCreateChannelDesc<TransformType>();
  checkCudaErrors(cudaBindTexture(0, transform_texture, reinterpret_cast<const TransformType *>(transform_array.getDevicePtr()), transform_desc));

  cudaChannelFormatDesc block_hash_table_desc = cudaCreateChannelDesc<HashEntryType>();
  checkCudaErrors(cudaBindTexture(0, block_hash_table_texture, reinterpret_cast<const HashEntryType *>(block_hash_table.getDevicePtr()), block_hash_table_desc));

  cudaChannelFormatDesc blend_weight_table_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture(0, blend_weight_table_texture, reinterpret_cast<const float4 *>(blend_weight_table.getDevicePtr()), blend_weight_table_desc));

  cudaChannelFormatDesc blend_code_table_desc = cudaCreateChannelDesc<int4>();
  checkCudaErrors(cudaBindTexture(0, blend_code_table_texture, reinterpret_cast<const int4 *>(blend_code_table.getDevicePtr()), blend_code_table_desc));
}

void unbindBlendingTextures()
{
  checkCudaErrors(cudaUnbindTexture(transform_texture));
  checkCudaErrors(cudaUnbindTexture(block_hash_table_texture));
  checkCudaErrors(cudaUnbindTexture(blend_weight_table_texture));
  checkCudaErrors(cudaUnbindTexture(blend_code_table_texture));
}

void bindBlendingTableTextures(const DeviceArray<float4> &blend_weight_table,
                               const DeviceArray<int4> &blend_code_table)
{
  cudaChannelFormatDesc blend_weight_table_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture(0, blend_weight_table_texture, reinterpret_cast<const float4 *>(blend_weight_table.getDevicePtr()), blend_weight_table_desc));

  cudaChannelFormatDesc blend_code_table_desc = cudaCreateChannelDesc<int4>();
  checkCudaErrors(cudaBindTexture(0, blend_code_table_texture, reinterpret_cast<const int4 *>(blend_code_table.getDevicePtr()), blend_code_table_desc));
}

void unbindBlendingTableTextures()
{
  checkCudaErrors(cudaUnbindTexture(blend_weight_table_texture));
  checkCudaErrors(cudaUnbindTexture(blend_code_table_texture));
}

void bindTransformTextures(const DeviceArray<TransformType> &transform_array)
{
  cudaChannelFormatDesc transform_desc = cudaCreateChannelDesc<TransformType>();
  checkCudaErrors(cudaBindTexture(0, transform_texture, reinterpret_cast<const TransformType *>(transform_array.getDevicePtr()), transform_desc));
}

void unbindTransformTextures()
{
  checkCudaErrors(cudaUnbindTexture(transform_texture));
}

void bindBlockHashTableTextures(const DeviceArray<HashEntryType> &block_hash_table)
{
  cudaChannelFormatDesc block_hash_table_desc = cudaCreateChannelDesc<HashEntryType>();
  checkCudaErrors(cudaBindTexture(0, block_hash_table_texture, reinterpret_cast<const HashEntryType *>(block_hash_table.getDevicePtr()), block_hash_table_desc));
}

void unbindBlockHashTableTextures()
{
  checkCudaErrors(cudaUnbindTexture(block_hash_table_texture));
}

void bindMarchingCubesTextures(const DeviceArray<int> &tri_table,
                               const DeviceArray<int> &num_verts_table)
{
  cudaChannelFormatDesc tri_desc = cudaCreateChannelDesc<int>();
  checkCudaErrors(cudaBindTexture(0, tri_texture, reinterpret_cast<const int *>(tri_table.getDevicePtr()), tri_desc));

  cudaChannelFormatDesc num_verts_desc = cudaCreateChannelDesc<int>();
  checkCudaErrors(cudaBindTexture(0, num_verts_texture, reinterpret_cast<const int *>(num_verts_table.getDevicePtr()), num_verts_desc));
}

void unbindMarchingCubesTextures()
{
  checkCudaErrors(cudaUnbindTexture(tri_texture));
  checkCudaErrors(cudaUnbindTexture(num_verts_texture));
}

void bindTsdfVoxelTexture(const DeviceArray<TsdfVoxelType> &tsdf_voxel_array)
{
  cudaChannelFormatDesc tsdf_voxel_desc = cudaCreateChannelDesc<TsdfVoxelType>();
  checkCudaErrors(cudaBindTexture(0, tsdf_voxel_texture, reinterpret_cast<const TsdfVoxelType *>(tsdf_voxel_array.getDevicePtr()), tsdf_voxel_desc));
}

void unbindTsdfVoxelTexture()
{
  checkCudaErrors(cudaUnbindTexture(tsdf_voxel_texture));
}

void bindRgbaVoxelTexture(const DeviceArray<RgbaVoxelType> &rgba_voxel_array)
{
  cudaChannelFormatDesc rgba_voxel_desc = cudaCreateChannelDesc<RgbaVoxelType>();
  checkCudaErrors(cudaBindTexture(0, rgba_voxel_texture, reinterpret_cast<const RgbaVoxelType *>(rgba_voxel_array.getDevicePtr()), rgba_voxel_desc));
}

void unbindRgbaVoxelTexture()
{
  checkCudaErrors(cudaUnbindTexture(rgba_voxel_texture));
}

void bindMeshTextures(const DeviceArray<float4> &vertex_array,
                      const DeviceArray<float4> &normal_array)
{
  cudaChannelFormatDesc mesh_desc = cudaCreateChannelDesc<float4>();

  checkCudaErrors(cudaBindTexture(0, mesh_vert_texture, reinterpret_cast<const float4 *>(vertex_array.getDevicePtr()), mesh_desc));
  checkCudaErrors(cudaBindTexture(0, mesh_norm_texture, reinterpret_cast<const float4 *>(normal_array.getDevicePtr()), mesh_desc));
}

void unbindMeshTextures()
{
  checkCudaErrors(cudaUnbindTexture(mesh_vert_texture));
  checkCudaErrors(cudaUnbindTexture(mesh_norm_texture));
}

void bindTermElemTextures(const DeviceArray<float4> &elem_data,
                          const DeviceArray<uint> &elem_ids,
                          const DeviceArray<float> &elem_weights)
{
  cudaChannelFormatDesc data_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture(0, elem_data_texture, reinterpret_cast<const float4 *>(elem_data.getDevicePtr()), data_desc));

  cudaChannelFormatDesc id_desc = cudaCreateChannelDesc<uint>();
  checkCudaErrors(cudaBindTexture(0, elem_id_texture, reinterpret_cast<const uint *>(elem_ids.getDevicePtr()), id_desc));

  cudaChannelFormatDesc weight_desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaBindTexture(0, elem_weight_texture, reinterpret_cast<const float *>(elem_weights.getDevicePtr()), weight_desc));
}

void unbindTermElemTextures()
{
  checkCudaErrors(cudaUnbindTexture(elem_data_texture));
  checkCudaErrors(cudaUnbindTexture(elem_id_texture));
  checkCudaErrors(cudaUnbindTexture(elem_weight_texture));
}

void bindRegTermElemTextures(const DeviceArray<float4> &elem_data,
                             const DeviceArray<uint> &elem_ids)
{
  cudaChannelFormatDesc data_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaBindTexture(0, elem_data_texture, reinterpret_cast<const float4 *>(elem_data.getDevicePtr()), data_desc));

  cudaChannelFormatDesc id_desc = cudaCreateChannelDesc<uint>();
  checkCudaErrors(cudaBindTexture(0, elem_id_texture, reinterpret_cast<const uint *>(elem_ids.getDevicePtr()), id_desc));
}

void unbindRegTermElemTextures()
{
  checkCudaErrors(cudaUnbindTexture(elem_data_texture));
  checkCudaErrors(cudaUnbindTexture(elem_id_texture));
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void resetVoxelBlockArray(DeviceArray<TsdfVoxelType> &tsdf_voxel_block_array)
{
  size_t voxel_block_array_size = tsdf_voxel_block_array.getSize();

  checkCudaErrors(cudaMemset(reinterpret_cast<void *>(tsdf_voxel_block_array.getDevicePtr()), 0, voxel_block_array_size * sizeof(TsdfVoxelType)));
}

void resetVoxelBlockArray(DeviceArray<RgbaVoxelType> &rgba_voxel_block_array)
{
  size_t voxel_block_array_size = rgba_voxel_block_array.getSize();

  checkCudaErrors(cudaMemset(reinterpret_cast<void *>(rgba_voxel_block_array.getDevicePtr()), 0, voxel_block_array_size * sizeof(RgbaVoxelType)));
}

__global__
void resetTransformArrayKernel(DeviceArrayHandle<float4> transform_array)
{
  const int stride = blockDim.x * gridDim.x;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < MAX_GRAPH_NODES_SIZE;
       i += stride)
  {
    int start = 4 * i; // 4x4 matrix (4 * float4)
    Affine3d transform; // set identity matrix as default
    transform.getValue(&(transform_array.at(start))); // write to global memory
  }
}

void resetTransformArray(DeviceArray<TransformType> &transform_array)
{
  int block = 256;
  int grid = min(divUp(MAX_GRAPH_NODES_SIZE, block), 256);

  resetTransformArrayKernel<<<grid, block>>>(transform_array.getHandle());
  checkCudaErrors(cudaGetLastError());
}

void resetBlockHashTable(DeviceArray<HashEntryType> &block_hash_table)
{
  size_t block_hash_table_size = block_hash_table.getSize();

  checkCudaErrors(cudaMemset(reinterpret_cast<void *>(block_hash_table.getDevicePtr()), 0xFFFFFFFF, block_hash_table_size * sizeof(HashEntryType)));
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
float3 getGridCoo(int x, int y, int z, float length, float offset)
{
  float3 coo = make_float3(x, y, z);
  coo += offset;
  coo *= length;

  return coo;
}

__device__ __forceinline__
float3 getGridCoo(int3 grid, float length, float offset)
{
  float3 coo = make_float3(grid);
  coo += offset;
  coo *= length;

  return coo;
}

__device__ __forceinline__
int getGridHash(int x, int y, int z, int dim)
{
  int hash = x * dim * dim + y * dim + z;

  return hash;
}

__device__ __forceinline__
int getGridHash(int3 grid, int dim)
{
  int hash = grid.x * dim * dim + grid.y * dim + grid.z;

  return hash;
}

__device__ __forceinline__
int3 getGrid(int hash, int dim)
{
  int3 grid;
  grid.z = (hash % dim);
  grid.y = (hash / dim) % dim;
  grid.x = (hash / dim) / dim;

  return grid;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getVoxelBlendingTableKernel(int voxel_blend_table_entry_size,
                                 int voxel_block_inner_hash_size,
                                 int voxel_block_inner_dim,
                                 int dim_ratio,
                                 float voxel_length,
                                 float voxel_block_length,
                                 float node_block_length,
                                 float two_mul_sigma_sqr_inv,
                                 DeviceArrayHandle<float4> voxel_blend_weight_table,
                                 DeviceArrayHandle<int4> voxel_blend_code_table)
{
  // DO NOT use shared memory here!
  float voxel_blend_weight[27];
  int voxel_blend_code[27];

  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < voxel_blend_table_entry_size;
       i += stride)
  {
    int pass = i / voxel_block_inner_hash_size;
    int3 voxel_block_grid = getGrid(pass, dim_ratio);
    float3 voxel_block_origin = getGridCoo(voxel_block_grid, voxel_block_length, 0.f);

    int voxel_block_inner_hash = i % voxel_block_inner_hash_size;
    int3 voxel_block_inner_grid = getGrid(voxel_block_inner_hash, voxel_block_inner_dim);
    float3 voxel = voxel_block_origin + getGridCoo(voxel_block_inner_grid, voxel_length, 0.5f);

    // load data
    int shift = 0;
    for (int x = -1; x < 2; x += 1)
    {
      for (int y = -1; y < 2; y += 1)
      {
        for (int z = -1; z < 2; z += 1)
        {
          float3 node = getGridCoo(x, y, z, node_block_length, 0.5f);

          float3 diff = voxel - node;
          float diff_dot = dot(diff, diff);
          float weight = expf(-1 * diff_dot * two_mul_sigma_sqr_inv);
          int code = ((x + 1) << 16) + ((y + 1) << 8) + (z + 1);

          voxel_blend_weight[shift] = weight;
          voxel_blend_code[shift++] = code;
        } // z
      } // y
    } // x

    // sort by weight
    for (int j = 0; j < 4 * VOXEL_BLENDING_MAX_QUAD_STRIDE; j += 1)
    {
      for (int k = 26; k > j; k -= 1)
      {
        float weight_left  = voxel_blend_weight[k-1];
        float weight_right = voxel_blend_weight[k];

        int code_left  = voxel_blend_code[k-1];
        int code_right = voxel_blend_code[k];

        if (weight_right > weight_left)
        {
          voxel_blend_weight[k-1] = weight_right;
          voxel_blend_weight[k]   = weight_left;

          voxel_blend_code[k-1] = code_right;
          voxel_blend_code[k]   = code_left;
        }
      } // k
    } // j

    // write result
    for (int j = 0; j < VOXEL_BLENDING_MAX_QUAD_STRIDE; j += 1)
    {
      float4 weight_entry_quad = make_float4(voxel_blend_weight[4*j + 0],
                                             voxel_blend_weight[4*j + 1],
                                             voxel_blend_weight[4*j + 2],
                                             voxel_blend_weight[4*j + 3]);

      int4 code_entry_quad = make_int4(voxel_blend_code[4*j + 0],
                                       voxel_blend_code[4*j + 1],
                                       voxel_blend_code[4*j + 2],
                                       voxel_blend_code[4*j + 3]);

      voxel_blend_weight_table.at(i * VOXEL_BLENDING_MAX_QUAD_STRIDE + j) = weight_entry_quad;
      voxel_blend_code_table.at(i * VOXEL_BLENDING_MAX_QUAD_STRIDE + j) = code_entry_quad;
    }
  } // grid-stride loop
}

void generateVoxelBlendingTable(float voxel_length,
                                int voxel_block_inner_dim,
                                float voxel_block_length,
                                int voxel_block_dim,
                                float node_block_length,
                                int node_block_dim,
                                DeviceArray<float4> &voxel_blend_weight_table,
                                DeviceArray<int4> &voxel_blend_code_table)
{
  int dim_ratio = voxel_block_dim / node_block_dim;

  int voxel_blend_pass_num =  dim_ratio * dim_ratio * dim_ratio;

  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;

  int voxel_blend_table_entry_size = voxel_block_inner_hash_size * voxel_blend_pass_num;

  // TODO: find the optimal value of the following parameter
  float sigma = 1.3f * node_block_length / 3.0f;
  float two_mul_sigma_sqr_inv = 1.f / (2.f * sigma * sigma);

  int block = 256;
  int grid = min(divUp(voxel_blend_table_entry_size, block), 512);

  getVoxelBlendingTableKernel<<<grid, block>>>(voxel_blend_table_entry_size,
                                               voxel_block_inner_hash_size,
                                               voxel_block_inner_dim,
                                               dim_ratio,
                                               voxel_length,
                                               voxel_block_length,
                                               node_block_length,
                                               two_mul_sigma_sqr_inv,
                                               voxel_blend_weight_table.getHandle(),
                                               voxel_blend_code_table.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getNodeBlendingTableKernel(int dim_ratio,
                                float node_block_length_prev,
                                float node_block_length_curr,
                                float two_mul_sigma_sqr_inv,
                                DeviceArrayHandle<float4> node_blend_weight_table,
                                DeviceArrayHandle<int4> node_blend_code_table)
{
  // DO NOT use shared memory here!
  float node_blend_weight[27];
  int node_blend_code[27];

  const int i = threadIdx.x;

  // int3 node_block_grid_prev = getGrid(i, dim_ratio);
  // TODO: only for test
  int3 node_block_grid_prev = make_int3(0, 0, 0);

  float3 node_prev = getGridCoo(node_block_grid_prev, node_block_length_prev, 0.5f);

  // load data
  int shift = 0;
  for (int x = -1; x < 2; x += 1)
  {
    for (int y = -1; y < 2; y += 1)
    {
      for (int z = -1; z < 2; z += 1)
      {
        float3 node_curr = getGridCoo(x, y, z, node_block_length_curr, 0.5f);

        float3 diff = node_prev - node_curr;
        float diff_dot = dot(diff, diff);
        float weight = expf(-1 * diff_dot * two_mul_sigma_sqr_inv);
        int code = ((x + 1) << 16) + ((y + 1) << 8) + (z + 1);

        node_blend_weight[shift] = weight;
        node_blend_code[shift++] = code;
      } // z
    } // y
  } // x

  // sort by weight
  for (int j = 0; j < 4 * NODE_BLENDING_MAX_QUAD_STRIDE; j += 1)
  {
    for (int k = 26; k > j; k -= 1)
    {
      float weight_left  = node_blend_weight[k-1];
      float weight_right = node_blend_weight[k];

      int code_left  = node_blend_code[k-1];
      int code_right = node_blend_code[k];

      if (weight_right > weight_left)
      {
        node_blend_weight[k-1] = weight_right;
        node_blend_weight[k]   = weight_left;

        node_blend_code[k-1] = code_right;
        node_blend_code[k]   = code_left;
      }
    } // k
  } // j

  // write result
  for (int j = 0; j < NODE_BLENDING_MAX_QUAD_STRIDE; j += 1)
  {
    float4 weight_entry_quad = make_float4(node_blend_weight[4*j + 0],
                                           node_blend_weight[4*j + 1],
                                           node_blend_weight[4*j + 2],
                                           node_blend_weight[4*j + 3]);

    int4 code_entry_quad = make_int4(node_blend_code[4*j + 0],
                                     node_blend_code[4*j + 1],
                                     node_blend_code[4*j + 2],
                                     node_blend_code[4*j + 3]);

    node_blend_weight_table.at(i * NODE_BLENDING_MAX_QUAD_STRIDE + j) = weight_entry_quad;
    node_blend_code_table.at(i * NODE_BLENDING_MAX_QUAD_STRIDE + j) = code_entry_quad;
  }
}

void generateNodeBlendingTable(float node_block_length_prev,
                               int node_block_dim_prev,
                               float node_block_length_curr,
                               int node_block_dim_curr,
                               DeviceArray<float4> &node_blend_weight_table,
                               DeviceArray<int4> &node_blend_code_table)
{
  int dim_ratio = node_block_dim_prev / node_block_dim_curr;

  int node_blend_pass_num = dim_ratio * dim_ratio * dim_ratio;

  int node_blend_table_entry_size = node_blend_pass_num;

  // TODO: find the optimal value of the following parameter
  float sigma = 1.3f * node_block_length_curr / 3.0f;
  float two_mul_sigma_sqr_inv = 1.0f / (2.0f * sigma * sigma);

  int block = node_blend_table_entry_size;
  int grid = 1;

  getNodeBlendingTableKernel<<<grid, block>>>(dim_ratio,
                                              node_block_length_prev,
                                              node_block_length_curr,
                                              two_mul_sigma_sqr_inv,
                                              node_blend_weight_table.getHandle(),
                                              node_blend_code_table.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getImageHomeBlockKernel(int width,
                             int height,
                             Affine3d transform,
                             float voxel_block_length,
                             int voxel_block_dim,
                             DeviceArrayHandle<uint> pending_node_hash_array)
{
  const int u = threadIdx.x + blockIdx.x * blockDim.x;
  const int v = threadIdx.y + blockIdx.y * blockDim.y;

  if (u >= width || v >= height)
    return;

  float4 vertex = tex2D(vertex_texture, u, v);
  float3 proj_vertex = make_float3(transform * vertex);
  float3 offset = make_float3(0.5f * voxel_block_length);

  proj_vertex -= offset;

  int3 voxel_block_grid;
  voxel_block_grid.x = __float2int_rz(proj_vertex.x / voxel_block_length);
  voxel_block_grid.y = __float2int_rz(proj_vertex.y / voxel_block_length);
  voxel_block_grid.z = __float2int_rz(proj_vertex.z / voxel_block_length);

  uint hash = (uint)getGridHash(voxel_block_grid, voxel_block_dim);

  if (isnan(vertex.x) ||
      voxel_block_grid.x < 0 ||
      voxel_block_grid.y < 0 ||
      voxel_block_grid.z < 0 ||
      voxel_block_grid.x >= voxel_block_dim ||
      voxel_block_grid.y >= voxel_block_dim ||
      voxel_block_grid.z >= voxel_block_dim)
  {
    hash = 0xFFFFFFFF;
  }

  int start = u * height + v;

  pending_node_hash_array.at(start) = hash;
}

__global__
void extendHomeBlockKernel(int block_dim,
                           size_t block_size,
                           const DeviceArrayHandle<uint> home_block_hash_array,
                           DeviceArrayHandle<uint> active_block_hash_array)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < block_size;
       i += stride)
  {
    uint home_hash = home_block_hash_array.at(i);

    int3 block_grid = getGrid((int)home_hash, block_dim);

    const int block_neighbor_size = 8;
    int shift = i * block_neighbor_size;
    for (int x = block_grid.x;
         x < block_grid.x + 2;
         x += 1)
    {
      for (int y = block_grid.y;
           y < block_grid.y + 2;
           y += 1)
      {
        for (int z = block_grid.z;
             z < block_grid.z + 2;
             z += 1)
        {
          uint active_hash;

          // Note all boundary blocks will be as inactive blocks
          if (x < 0 || x >= block_dim ||
              y < 0 || y >= block_dim ||
              z < 0 || z >= block_dim)
          {
            active_hash = 0xFFFFFFFF;
          }
          else
          {
            active_hash = (uint)getGridHash(x, y, z, block_dim);
          }

          active_block_hash_array.at(shift++) = active_hash;
        } // loop on z
      } // loop on y
    } // loop on x
  }
}

__global__
void initVoxelBlockKernel(size_t voxel_block_size,
                          const DeviceArrayHandle<uint> pending_node_hash_array,
                          DeviceArrayHandle<uint> voxel_block_hash_array,
                          DeviceArrayHandle<uint2> voxel_block_hash_table)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < voxel_block_size;
       i += stride)
  {
    uint block_hash = pending_node_hash_array.at(i);
    voxel_block_hash_array.at(i) = block_hash;

    uint2 hash_entry = make_uint2(1, uint(i));
    voxel_block_hash_table.at(block_hash) = hash_entry;
  }
}
                                
bool initVoxelBlock(const Intrinsics &intrin,
                    const Affine3d &transform,
                    float voxel_block_length,
                    int voxel_block_dim,
                    size_t &voxel_block_size,
                    DeviceArray<uint> &pending_node_hash_array,
                    DeviceArray<uint> &voxel_block_hash_array,
                    DeviceArray<HashEntryType> &voxel_block_hash_table)
{
  // Step 1: Home Block Hashing
  int width = intrin.width;
  int height = intrin.height;

  dim3 block(32, 16);
  dim3 grid(divUp(width, block.x), divUp(height, block.y));

  getImageHomeBlockKernel<<<grid, block>>>(width,
                                           height,
                                           transform,
                                           voxel_block_length,
                                           voxel_block_dim,
                                           pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 2: Home Block Sorting
  thrust::device_ptr<uint> pending_node_hash_array_ptr(pending_node_hash_array.getDevicePtr());

  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + width * height);

  // Step 3: Home Block Unique
  thrust::device_ptr<uint> unique_end;
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + width * height);

  voxel_block_size = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  // remove inactive block hash
  uint hash_array_last;
  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&hash_array_last),
                             reinterpret_cast<void *>(raw_pointer_cast(unique_end) - 1),
                             sizeof(uint), cudaMemcpyDeviceToHost));

  if (hash_array_last == 0xFFFFFFFF)
    voxel_block_size -= 1;

  if (voxel_block_size >= MAX_VOXEL_BLOCK_SIZE)
  {
    std::cout << std::endl << "Voxel block size exceeds limitation!" << std::endl;
    return false;
  }

  if (voxel_block_size <= 0)
  {
    std::cout << std::endl << "No voxel block detected!" << std::endl;
    return false;
  }

  // Step 4: Home Block Copy
  thrust::device_ptr<uint> voxel_block_hash_array_ptr(voxel_block_hash_array.getDevicePtr());
  thrust::copy(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + voxel_block_size,
               voxel_block_hash_array_ptr);

  // Step 5: Home Block Hash Extension
  block.x = 256;
  block.y = 1;
  grid.x = min(divUp(voxel_block_size, block.x), 512);
  grid.y = 1;

  extendHomeBlockKernel<<<grid, block>>>(voxel_block_dim,
                                         voxel_block_size,
                                         voxel_block_hash_array.getHandle(),
                                         pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 6: Active Block Sorting
  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + 8 * voxel_block_size);

  // Step 7: Active Block Unique
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + 8 * voxel_block_size);

  voxel_block_size = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  // remove inactive block hash
  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&hash_array_last),
                             reinterpret_cast<void *>(raw_pointer_cast(unique_end) - 1),
                             sizeof(uint), cudaMemcpyDeviceToHost));

  if (hash_array_last == 0xFFFFFFFF)
    voxel_block_size -= 1;

  if (voxel_block_size >= MAX_VOXEL_BLOCK_SIZE)
  {
    std::cout << std::endl << "Voxel block size exceeds limitation!" << std::endl;
    return false;
  }

  // Step 8: Update Voxel Block Array & Voxel Block Table
  grid.x = min(divUp(voxel_block_size, block.x), 512);
  grid.y = 1;

  initVoxelBlockKernel<<<grid, block>>>(voxel_block_size,
                                        pending_node_hash_array.getHandle(),
                                        voxel_block_hash_array.getHandle(),
                                        voxel_block_hash_table.getHandle());
  checkCudaErrors(cudaGetLastError());

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getVertexHomeBlockKernel(float voxel_block_length,
                              int voxel_block_dim,
                              size_t vertex_size,
                              const DeviceArrayHandle<float4> vertex_array,
                              const DeviceArrayHandle<uint> active_flags,
                              DeviceArrayHandle<uint> home_block_hash_array)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < vertex_size;
       i += stride)
  {
    uint flag = active_flags.at(i);
    float3 vertex = make_float3(vertex_array.at(i));
    float3 offset = make_float3(0.5f * voxel_block_length);

    vertex -= offset;

    int3 voxel_block_grid;
    voxel_block_grid.x = __float2int_rz(vertex.x / voxel_block_length);
    voxel_block_grid.y = __float2int_rz(vertex.y / voxel_block_length);
    voxel_block_grid.z = __float2int_rz(vertex.z / voxel_block_length);

    uint hash = (uint)getGridHash(voxel_block_grid, voxel_block_dim);

    home_block_hash_array.at(i) = (flag == 1) ? hash : 0xFFFFFFFF;
  }
}

__global__
void updateVoxelBlockKernel(int voxel_block_dim,
                            size_t voxel_block_size_old,
                            size_t voxel_block_size_new,
                            const DeviceArrayHandle<uint> pending_node_hash_array,
                            DeviceArrayHandle<uint> voxel_block_hash_array,
                            DeviceArrayHandle<uint2> voxel_block_hash_table)
{
  const int stride = blockDim.x * gridDim.x;
  const int lane_id = threadIdx.x % warpSize;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < voxel_block_size_new;
       i += stride)
  {
    uint block_hash = pending_node_hash_array.at(i);

    uint2 hash_entry = voxel_block_hash_table.at(block_hash);
    hash_entry.x = 1; // set as active voxel block for TSDF voluem fusion

    voxel_block_hash_table.at(block_hash) = hash_entry;

    /*
     * Warp-Aggregated Atomics
     */

    // Step 1: leader election
#ifdef CUDA8
    unsigned int valid = __ballot(hash_entry.y == 0xFFFFFFFF);
#else
    unsigned int active = __activemask();
    unsigned int valid = __ballot_sync(active, hash_entry.y == 0xFFFFFFFF);
#endif

    int leader = __ffs(valid) - 1;

    // Step 2: computing the total increment (in the warp)
    int total = __popc(valid);

    if (total == 0)
      continue;

    // Step 3: performing the atomic add
    unsigned int lane_mask_lt = (1 << lane_id) - 1;
    unsigned int rank = __popc(valid & lane_mask_lt);

    int warp_old = 0, thread_offset = 0;

    if (lane_id == leader)
      warp_old = atomicAdd(&global_count, total);

    // Step 4: broadcasting the result
#ifdef CUDA8
    warp_old = __shfl(warp_old, leader);
#else
    warp_old = __shfl_sync(valid, warp_old, leader);
#endif

    thread_offset = voxel_block_size_old + warp_old + (int)rank;

    if (thread_offset < MAX_VOXEL_BLOCK_SIZE && hash_entry.y == 0xFFFFFFFF)
    {
      hash_entry.y = uint(thread_offset);

      voxel_block_hash_array.at(thread_offset) = block_hash;

      voxel_block_hash_table.at(block_hash) = hash_entry;
    }

    bool full = (voxel_block_size_old + warp_old + total >= MAX_VOXEL_BLOCK_SIZE);

    if (full)
      break;

  } // grid-stride loop

  __syncthreads();

  // prepare for future scans
  if (threadIdx.x == 0)
  {
    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_count = global_count;
      global_count = 0;
      blocks_done = 0;
    }
  }
}

bool updateVoxelBlock(float voxel_block_length,
                      int voxel_block_dim,
                      size_t vertex_size,
                      const DeviceArray<float4> &vertex_array,
                      const DeviceArray<uint> &active_flags,
                      size_t &voxel_block_size,
                      DeviceArray<uint> &pending_node_hash_array,
                      DeviceArray<uint> &voxel_block_hash_array,
                      DeviceArray<HashEntryType> &voxel_block_hash_table)
{
  // Step 1: Home Block Hashing
  int block = 256;
  int grid = min(divUp(vertex_size, block), 512);

  getVertexHomeBlockKernel<<<grid, block>>>(voxel_block_length,
                                            voxel_block_dim,
                                            vertex_size,
                                            vertex_array.getHandle(),
                                            active_flags.getHandle(),
                                            pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 2: Home Block Sorting
  thrust::device_ptr<uint> pending_node_hash_array_ptr(pending_node_hash_array.getDevicePtr());

  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + vertex_size);

  // Step 3: Home Block Unique
  thrust::device_ptr<uint> unique_end;
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + vertex_size);

  size_t home_block_size = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  // remove inactive block hash
  uint hash_array_last;
  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&hash_array_last),
                             reinterpret_cast<void *>(raw_pointer_cast(unique_end) - 1),
                             sizeof(uint), cudaMemcpyDeviceToHost));

  if (hash_array_last == 0xFFFFFFFF)
    home_block_size -= 1;

  if (home_block_size >= MAX_VOXEL_BLOCK_SIZE)
  {
    std::cout << std::endl << "Home block size exceeds limitation!" << std::endl;
    return false;
  }

  if (home_block_size <= 0)
  {
    std::cout << std::endl << "No home block detected!" << std::endl;
    return false;
  }

  // Step 4: Home Block Copy
  thrust::device_ptr<uint> home_block_hash_array_ptr(pending_node_hash_array.getDeviceWritePtr());
  thrust::copy(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + home_block_size,
               home_block_hash_array_ptr);

  // Step 5: Home Block Hash Extension
  block = 256;
  grid = min(divUp(home_block_size, block), 512);

  extendHomeBlockKernel<<<grid, block>>>(voxel_block_dim,
                                         home_block_size,
                                         pending_node_hash_array.getWriteHandle(),
                                         pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 6: Active Block Sorting
  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + 8 * home_block_size);

  // Step 7: Active Block Unique
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + 8 * home_block_size);

  size_t voxel_block_size_new = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  // remove inactive block hash
  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&hash_array_last),
                             reinterpret_cast<void *>(raw_pointer_cast(unique_end) - 1),
                             sizeof(uint), cudaMemcpyDeviceToHost));

  if (hash_array_last == 0xFFFFFFFF)
    voxel_block_size_new -= 1;

  if (voxel_block_size_new >= MAX_VOXEL_BLOCK_SIZE)
  {
    std::cout << std::endl << "New voxel block size exceeds limitation!" << std::endl;
    return false;
  }

  // Step 8: Update Voxel Block Array & Voxel Block Table
  block = 256;
  grid = min(divUp(voxel_block_size_new, block), 512);

  updateVoxelBlockKernel<<<grid, block>>>(voxel_block_dim,
                                          voxel_block_size,
                                          voxel_block_size_new,
                                          pending_node_hash_array.getHandle(),
                                          voxel_block_hash_array.getHandle(),
                                          voxel_block_hash_table.getHandle());
  checkCudaErrors(cudaGetLastError());

  int size;
  checkCudaErrors(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));

  if (voxel_block_size + size >= MAX_VOXEL_BLOCK_SIZE)
  {
    std::cout << std::endl << "Updated voxel block size exceeds limitation!" << std::endl;
    return false;
  }

  voxel_block_size += size;

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getNodeBlockHashKernel(int node_block_dim_prev,
                            size_t node_block_size_prev,
                            const DeviceArrayHandle<uint> node_block_hash_array_prev,
                            int dim_ratio,
                            int node_block_dim,
                            DeviceArrayHandle<uint> pending_node_hash_array)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < node_block_size_prev;
       i += stride)
  {
    uint node_block_hash_prev = node_block_hash_array_prev.at(i);

    int3 node_block_grid = getGrid(node_block_hash_prev, node_block_dim_prev);
    node_block_grid.x /= dim_ratio;
    node_block_grid.y /= dim_ratio;
    node_block_grid.z /= dim_ratio;

    uint node_block_hash = (uint)getGridHash(node_block_grid, node_block_dim);

    pending_node_hash_array.at(i) = node_block_hash;
  }
}

__device__ __forceinline__
uint expandBits(uint v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;

  return v;
}

__global__
void initNodeBlockKernel(float volume_length,
                         int level,
                         float node_block_length,
                         int node_block_dim,
                         size_t node_block_size,
                         size_t node_size,
                         const DeviceArrayHandle<uint> pending_node_hash_array,
                         DeviceArrayHandle<uint> node_block_hash_array,
                         DeviceArrayHandle<uint2> node_block_hash_table,
                         DeviceArrayHandle<uint> node_hash_array,
                         DeviceArrayHandle<float4> node_array,
                         DeviceArrayHandle<uint> node_morton_code_array)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < node_block_size;
       i += stride)
  {
    uint node_block_hash = pending_node_hash_array.at(i);
    uint2 hash_entry = make_uint2(0xFFFFFFFF, uint(i));

    int3 node_block_grid = getGrid(node_block_hash, node_block_dim);

    float4 node = make_float4(getGridCoo(node_block_grid, node_block_length, 0.5f), 1.f);

    float x = min(max(node.x * 1024.0f / volume_length, 0.0f), 1023.0f);
    float y = min(max(node.y * 1024.0f / volume_length, 0.0f), 1023.0f);
    float z = min(max(node.z * 1024.0f / volume_length, 0.0f), 1023.0f);

    uint xx = expandBits((uint)(x));
    uint yy = expandBits((uint)(y));
    uint zz = expandBits((uint)(z));
    uint morton_code = xx * 4 + yy * 2 + zz;

    node_block_hash_array.at(i) = node_block_hash;

    node_block_hash_table.at(node_block_hash) = hash_entry;

    node_hash_array.at(i + node_size) = (node_block_hash & 0x0FFFFFFF) | (level << 28);
    node_array.at(i + node_size) = node;
    node_morton_code_array.at(i + node_size) = morton_code;
  }
}

bool initNodeBlockHierarchy(float volume_length,
                            int node_block_dim_prev,
                            size_t node_block_size_prev,
                            const DeviceArray<uint> &node_block_hash_array_prev,
                            int level,
                            float node_block_length,
                            int node_block_dim,
                            size_t &node_block_size,
                            DeviceArray<uint> &pending_node_hash_array,
                            DeviceArray<uint> &node_block_hash_array,
                            DeviceArray<HashEntryType> &node_block_hash_table,
                            size_t &node_size,
                            DeviceArray<uint> &node_hash_array,
                            DeviceArray<float4> &node_array,
                            DeviceArray<uint> &node_morton_code_array)
{
  int dim_ratio = node_block_dim_prev / node_block_dim;

  // Step 1: Home Block Hashing
  int block = 256;
  int grid = min(divUp(node_block_size_prev, block), 512);

  getNodeBlockHashKernel<<<grid, block>>>(node_block_dim_prev,
                                          node_block_size_prev,
                                          node_block_hash_array_prev.getHandle(),
                                          dim_ratio,
                                          node_block_dim,
                                          pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 2: Home Block Sorting
  thrust::device_ptr<uint> pending_node_hash_array_ptr(pending_node_hash_array.getDevicePtr());

  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + node_block_size_prev);

  // Step 3: Home Block Unique
  thrust::device_ptr<uint> unique_end;
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + node_block_size_prev);

  node_block_size = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  if (node_block_size >= MAX_GRAPH_NODES_SIZE)
  {
    std::cout << std::endl << "Node block size exceeds limitation!" << std::endl;
    return false;
  }

  if (node_size + node_block_size >= MAX_GRAPH_NODES_SIZE)
  {
    std::cout << std::endl << "Node size exceeds limitation!" << std::endl;
    return false;
  }

  // we do not consider phantom block here
  // Step 4: Update Deformation Graph & Block Hierarchy
  grid = min(divUp(node_block_size, block), 512);

  initNodeBlockKernel<<<grid, block>>>(volume_length,
                                       level,
                                       node_block_length,
                                       node_block_dim,
                                       node_block_size,
                                       node_size,
                                       pending_node_hash_array.getHandle(),
                                       node_block_hash_array.getHandle(),
                                       node_block_hash_table.getHandle(),
                                       node_hash_array.getHandle(),
                                       node_array.getHandle(),
                                       node_morton_code_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  node_size += node_block_size;

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void updateNodeBlockKernel(float volume_length,
                           int level,
                           float node_block_length,
                           int node_block_dim,
                           size_t node_block_size_old,
                           size_t node_block_size_new,
                           size_t node_size,
                           const DeviceArrayHandle<uint> pending_node_hash_array,
                           DeviceArrayHandle<uint> node_block_hash_array,
                           DeviceArrayHandle<uint2> node_block_hash_table,
                           DeviceArrayHandle<uint> node_hash_array,
                           DeviceArrayHandle<float4> node_array,
                           DeviceArrayHandle<uint> node_morton_code_array)
{
  const int stride = blockDim.x * gridDim.x;
  const int lane_id = threadIdx.x % warpSize;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < node_block_size_new;
       i += stride)
  {
    uint node_block_hash = pending_node_hash_array.at(i);

    int3 node_block_grid = getGrid(node_block_hash, node_block_dim);

    float4 node = make_float4(getGridCoo(node_block_grid, node_block_length, 0.5f), 1.f);

    float x = min(max(node.x * 1024.0f / volume_length, 0.0f), 1023.0f);
    float y = min(max(node.y * 1024.0f / volume_length, 0.0f), 1023.0f);
    float z = min(max(node.z * 1024.0f / volume_length, 0.0f), 1023.0f);

    uint xx = expandBits((uint)(x));
    uint yy = expandBits((uint)(y));
    uint zz = expandBits((uint)(z));
    uint morton_code = xx * 4 + yy * 2 + zz;

    uint2 hash_entry = node_block_hash_table.at(node_block_hash);

    /*
     * Warp-Aggregated Atomics
     */

    // Step 1: leader election
#ifdef CUDA8
    unsigned int valid = __ballot(hash_entry.y == 0xFFFFFFFF);
#else
    unsigned int active = __activemask();
    unsigned int valid = __ballot_sync(active, hash_entry.y == 0xFFFFFFFF);
#endif

    int leader = __ffs(valid) - 1;

    // Step 2: computing the total increment (in the warp)
    int total = __popc(valid);

    if (total == 0)
      continue;

    // Step 3: performing the atomic add
    unsigned int lane_mask_lt = (1 << lane_id) - 1;
    unsigned int rank = __popc(valid & lane_mask_lt);

    int warp_old = 0, node_block_offset = 0, node_offset = 0;

    if (lane_id == leader)
      warp_old = atomicAdd(&global_count, total);

    // Step 4: broadcasting the result
#ifdef CUDA8
    warp_old = __shfl(warp_old, leader);
#else
    warp_old = __shfl_sync(valid, warp_old, leader);
#endif

    node_block_offset = node_block_size_old + warp_old + (int)rank;
    node_offset = node_size + warp_old + (int)rank;

    if (node_block_offset < MAX_GRAPH_NODES_SIZE &&
        node_offset < MAX_GRAPH_NODES_SIZE &&
        hash_entry.y == 0xFFFFFFFF)
    {
      hash_entry.y = uint(node_block_offset);

      node_block_hash_array.at(node_block_offset) = node_block_hash;
      node_block_hash_table.at(node_block_hash) = hash_entry;

      node_hash_array.at(node_offset) = (node_block_hash & 0x0FFFFFFF) | (level << 28);
      node_array.at(node_offset) = node;
      node_morton_code_array.at(node_offset) = morton_code;
    }

    bool full = (node_block_size_old + warp_old + total >= MAX_GRAPH_NODES_SIZE) ||
                (node_size + warp_old + total >= MAX_GRAPH_NODES_SIZE);

    if (full)
      break;
  } // grid-stride loop

  __syncthreads();

  // prepare for future scans
  if (threadIdx.x == 0)
  {
    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_count = global_count;
      global_count = 0;
      blocks_done = 0;
    }
  }
}

bool updateNodeBlockHierarchy(float volume_length,
                              int node_block_dim_prev,
                              size_t node_block_size_prev,
                              const DeviceArray<uint> &node_block_hash_array_prev,
                              int level,
                              float node_block_length,
                              int node_block_dim,
                              size_t &node_block_size,
                              DeviceArray<uint> &pending_node_hash_array,
                              DeviceArray<uint> &node_block_hash_array,
                              DeviceArray<HashEntryType> &node_block_hash_table,
                              size_t &node_size,
                              DeviceArray<uint> &node_hash_array,
                              DeviceArray<float4> &node_array,
                              DeviceArray<uint> &node_morton_code_array)
{
  int dim_ratio = node_block_dim_prev / node_block_dim;

  // Step 1: Home Block Hashing
  int block = 256;
  int grid = min(divUp(node_block_size_prev, block), 512);

  getNodeBlockHashKernel<<<grid, block>>>(node_block_dim_prev,
                                          node_block_size_prev,
                                          node_block_hash_array_prev.getHandle(),
                                          dim_ratio,
                                          node_block_dim,
                                          pending_node_hash_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // Step 2: Home Block Sorting
  thrust::device_ptr<uint> pending_node_hash_array_ptr(pending_node_hash_array.getDevicePtr());

  thrust::sort(pending_node_hash_array_ptr,
               pending_node_hash_array_ptr + node_block_size_prev);

  // Step 3: Home Block Unique
  thrust::device_ptr<uint> unique_end;
  unique_end = thrust::unique(pending_node_hash_array_ptr,
                              pending_node_hash_array_ptr + node_block_size_prev);

  size_t node_block_size_new = static_cast<size_t>(unique_end - pending_node_hash_array_ptr);

  if (node_block_size_new >= MAX_GRAPH_NODES_SIZE)
  {
    std::cout << std::endl << "Node block size new exceeds limitation!" << std::endl;
    return false;
  }

  // Step 4: Update Deformation Graph & Block Hierarchy
  grid = min(divUp(node_block_size_new, block), 512);

  updateNodeBlockKernel<<<grid, block>>>(volume_length,
                                         level,
                                         node_block_length,
                                         node_block_dim,
                                         node_block_size,
                                         node_block_size_new,
                                         node_size,
                                         pending_node_hash_array.getHandle(),
                                         node_block_hash_array.getHandle(),
                                         node_block_hash_table.getHandle(),
                                         node_hash_array.getHandle(),
                                         node_array.getHandle(),
                                         node_morton_code_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  int size;
  checkCudaErrors(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));

  if ((node_block_size + size) >= MAX_GRAPH_NODES_SIZE)
  {
    std::cout << std::endl << "Node block size exceeds limitation!" << std::endl;
    return false;
  }

  if ((node_size + size) >= MAX_GRAPH_NODES_SIZE)
  {
    std::cout << std::endl << "Node size exceeds limitation!" << std::endl;
    return false;
  }

  node_block_size += size;
  node_size += size;

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int getNodeVoxelBlending(int node_block_dim,
                         int3 node_block_grid,
                         int voxel_blend_table_entry,
                         float *blend_weight,
                         uint *blend_node_id,
                         Affine3d &blend_transform)
{
  int blend_count = 0;

  int voxel_blend_table_entry_quad = voxel_blend_table_entry * VOXEL_BLENDING_MAX_QUAD_STRIDE;

  for (int quad = 0; quad < VOXEL_BLENDING_MAX_QUAD_STRIDE; quad++)
  {
    float4 weight_entry_quad = tex1Dfetch(blend_weight_table_texture,
                                          voxel_blend_table_entry_quad + quad);
    float *weight_entry = (float *)&weight_entry_quad;

    int4 code_entry_quad = tex1Dfetch(blend_code_table_texture,
                                      voxel_blend_table_entry_quad + quad);
    int *code_entry = (int *)&code_entry_quad;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
      if (blend_count == 4 * NODE_BLENDING_QUAD_STRIDE)
        break;

      float weight_i = weight_entry[i];

      int code_i = code_entry[i];
      int code_i_x = ((code_i & 0x00FF0000) >> 16) - 1;
      int code_i_y = ((code_i & 0x0000FF00) >> 8) - 1;
      int code_i_z = (code_i & 0x000000FF) - 1;

      int3 node_block_grid_i = node_block_grid + make_int3(int(code_i_x), int(code_i_y), int(code_i_z));

      if (node_block_grid_i.x >= 0 ||
          node_block_grid_i.y >= 0 ||
          node_block_grid_i.z >= 0 ||
          node_block_grid_i.x < node_block_dim ||
          node_block_grid_i.y < node_block_dim ||
          node_block_grid_i.z < node_block_dim)
      {
        int node_block_hash_i = getGridHash(node_block_grid_i, node_block_dim);
        HashEntryType node_hash_entry_i = tex1Dfetch(block_hash_table_texture, node_block_hash_i);
        uint node_id_i = node_hash_entry_i.x;
        TransformType transform_col[4];
        Affine3d transform_i;

        if (node_id_i != 0xFFFFFFFF)
        {
          transform_col[0] = tex1Dfetch(transform_texture, 4 * node_id_i + 0);
          transform_col[1] = tex1Dfetch(transform_texture, 4 * node_id_i + 1);
          transform_col[2] = tex1Dfetch(transform_texture, 4 * node_id_i + 2);
          transform_col[3] = tex1Dfetch(transform_texture, 4 * node_id_i + 3);
          transform_i.setValue(transform_col);

          blend_transform += (transform_i * weight_i);

          blend_weight[blend_count] = weight_i;
          blend_node_id[blend_count++] = node_id_i;
        }
      }
    } // i
  } // quad

  // normalization
  float weight_sum = 0.f;

  #pragma unroll
  for (int k = 0; k < blend_count; k++)
    weight_sum += blend_weight[k];

  if (weight_sum > 5e-3)
  {
    blend_transform *= (1.f / weight_sum);

    #pragma unroll
    for (int k = 0; k < blend_count; k++)
      blend_weight[k] /= weight_sum;
  }
  else
  {
    blend_count = 0;
    blend_transform.makeIdentity();
  }

  return blend_count;
}

__global__
void sampleNewNodeTransformKernel_1(float voxel_length,
                                    int voxel_block_inner_dim,
                                    float voxel_block_length,
                                    int node_block_dim,
                                    int dim_ratio,
                                    int voxel_block_inner_hash_size,
                                    size_t node_size_old,
                                    size_t node_size_new,
                                    DeviceArrayHandle<float4> node_array,
                                    DeviceArrayHandle<float4> transform_array)
{
  const int stride = gridDim.x * blockDim.x;

  // grid-stride loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x + node_size_old;
       i < node_size_new;
       i += stride)
  {
    float4 node = node_array.at(i);

    int3 voxel_block_grid;
    voxel_block_grid.x = __float2int_rz(node.x / voxel_block_length);
    voxel_block_grid.y = __float2int_rz(node.y / voxel_block_length);
    voxel_block_grid.z = __float2int_rz(node.z / voxel_block_length);

    float3 voxel_block_origin = getGridCoo(voxel_block_grid, voxel_block_length, 0.f);

    float3 voxel_block_inner_coo = make_float3(node) - voxel_block_origin;

    int3 voxel_block_inner_grid;
    voxel_block_inner_grid.x = __float2int_rz(voxel_block_inner_coo.x / voxel_length);
    voxel_block_inner_grid.y = __float2int_rz(voxel_block_inner_coo.y / voxel_length);
    voxel_block_inner_grid.z = __float2int_rz(voxel_block_inner_coo.z / voxel_length);

    int voxel_block_inner_hash = getGridHash(voxel_block_inner_grid, voxel_block_inner_dim);

    int3 node_block_grid;
    node_block_grid.x = voxel_block_grid.x / dim_ratio;
    node_block_grid.y = voxel_block_grid.y / dim_ratio;
    node_block_grid.z = voxel_block_grid.z / dim_ratio;

    int3 pass_grid;
    pass_grid.x = voxel_block_grid.x % dim_ratio;
    pass_grid.y = voxel_block_grid.y % dim_ratio;
    pass_grid.z = voxel_block_grid.z % dim_ratio;

    int pass = getGridHash(pass_grid, dim_ratio);
    int pass_stride = pass * voxel_block_inner_hash_size;

    int voxel_blend_table_entry = pass_stride + voxel_block_inner_hash;

    float blend_weight[4 * NODE_BLENDING_QUAD_STRIDE];
    uint blend_node_id[4 * NODE_BLENDING_QUAD_STRIDE];
    Affine3d blend_transform(0.0f);

    int blend_count = getNodeVoxelBlending(node_block_dim,
                                           node_block_grid,
                                           voxel_blend_table_entry,
                                           blend_weight,
                                           blend_node_id,
                                           blend_transform);

    blend_transform.getValue(&(transform_array.at(4 * i)));
  }
}

__global__
void sampleNewNodeTransformKernel_2(size_t node_size_old,
                                    size_t node_size_new,
                                    DeviceArrayHandle<float4> transform_array_src,
                                    DeviceArrayHandle<float4> transform_array_dst)
{
  const int stride = gridDim.x * blockDim.x;

  // grid-stride loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x + node_size_old;
       i < node_size_new;
       i += stride)
  {
    Affine3d transform;

    transform.setValue(&(transform_array_src.at(4 * i)));

    transform.getValue(&(transform_array_dst.at(4 * i)));
  } // grid-stride loop
}

void sampleNewNodeTransform(float voxel_length,
                            int voxel_block_inner_dim,
                            float voxel_block_length,
                            int voxel_block_dim,
                            int node_block_dim,
                            size_t node_size_old,
                            size_t node_size_new,
                            const DeviceArray<HashEntryType> &node_block_hash_table,
                            const DeviceArray<float4> &voxel_blend_weight_table,
                            const DeviceArray<int4> &voxel_blend_code_table,
                            const DeviceArray<float4> &node_array,
                            DeviceArray<TransformType> &transform_array)
{
  bindBlendingTextures(transform_array,
                       node_block_hash_table,
                       voxel_blend_weight_table,
                       voxel_blend_code_table);

  int new_node_size = static_cast<int>(node_size_new - node_size_old);

  int dim_ratio = voxel_block_dim / node_block_dim;
  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;

  int block = 256;
  int grid = min(divUp(new_node_size, block), 512);

  sampleNewNodeTransformKernel_1<<<grid, block>>>(voxel_length,
                                                  voxel_block_inner_dim,
                                                  voxel_block_length,
                                                  node_block_dim,
                                                  dim_ratio,
                                                  voxel_block_inner_hash_size,
                                                  node_size_old,
                                                  node_size_new,
                                                  node_array.getHandle(),
                                                  transform_array.getWriteHandle());
  checkCudaErrors(cudaGetLastError());

  unbindBlendingTextures();

  sampleNewNodeTransformKernel_2<<<grid, block>>>(node_size_old,
                                                  node_size_new,
                                                  transform_array.getWriteHandle(),
                                                  transform_array.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void reorderDeformGraphKernel(size_t node_size,
                              const DeviceArrayHandle<uint> node_id_reorder_array,
                              const DeviceArrayHandle<uint> node_hash_array_old,
                              const DeviceArrayHandle<float4> node_array_old,
                              const DeviceArrayHandle<float4> transform_array_old,
                              DeviceArrayHandle<uint> node_hash_array_new,
                              DeviceArrayHandle<float4> node_array_new,
                              DeviceArrayHandle<float4> transform_array_new,
                              DeviceArrayHandle<uint2 *> block_hash_table_hierarchy)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int id_new = blockDim.x * blockIdx.x + threadIdx.x;
       id_new < node_size;
       id_new += stride)
  {
    uint id_old = node_id_reorder_array.at(id_new);

    uint node_hash = node_hash_array_old.at(id_old);
    node_hash_array_new.at(id_new) = node_hash;

    node_array_new.at(id_new) = node_array_old.at(id_old);

    transform_array_new.at(4*id_new + 0) = transform_array_old.at(4*id_old + 0);
    transform_array_new.at(4*id_new + 1) = transform_array_old.at(4*id_old + 1);
    transform_array_new.at(4*id_new + 2) = transform_array_old.at(4*id_old + 2);
    transform_array_new.at(4*id_new + 3) = transform_array_old.at(4*id_old + 3);

    uint block_hash = (node_hash & 0x0FFFFFFF);
    uint level_hash = (node_hash & 0xF0000000) >> 28;

    HashEntryType &hash_entry = (block_hash_table_hierarchy.at(level_hash))[block_hash];
    hash_entry.x = id_new;
  }
}

void reorderDeformGraph(size_t node_size,
                        DeviceArray<uint> &node_hash_array,
                        DeviceArray<float4> &node_array,
                        DeviceArray<TransformType> &transform_array,
                        DeviceArray<uint> &node_morton_code_array,
                        DeviceArray<uint> &node_id_reorder_array,
                        DeviceArray<HashEntryType *> &block_hash_table_hierarchy_handle)
{
  // Step 1: Reorder deformation nodes according to their morton codes
  thrust::device_ptr<uint> node_morton_code_array_ptr(node_morton_code_array.getDevicePtr());
  thrust::device_ptr<uint> node_id_reorder_array_ptr(node_id_reorder_array.getDevicePtr());

  thrust::sequence(node_id_reorder_array_ptr,
                   node_id_reorder_array_ptr + node_size);

  thrust::sort_by_key(node_morton_code_array_ptr,
                      node_morton_code_array_ptr + node_size,
                      node_id_reorder_array_ptr);

  // Step 2: Write new node ids into block hash table
  int block = 256;
  int grid = min(divUp(node_size, block), 256);

  reorderDeformGraphKernel<<<grid, block>>>(node_size,
                                            node_id_reorder_array.getHandle(),
                                            node_hash_array.getHandle(),
                                            node_array.getHandle(),
                                            transform_array.getHandle(),
                                            node_hash_array.getWriteHandle(),
                                            node_array.getWriteHandle(),
                                            transform_array.getWriteHandle(),
                                            block_hash_table_hierarchy_handle.getHandle());
  checkCudaErrors(cudaGetLastError());

  node_hash_array.swap();
  node_array.swap();
  transform_array.swap();
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void copyTransformArrayKernel(size_t array_size,
                              const DeviceArrayHandle<float4> transform_array_src,
                              DeviceArrayHandle<float4> transform_array_dst)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < array_size;
       i += stride)
  {
    transform_array_dst.at(i) = transform_array_src.at(i);
  } // grid-stride loop
}

void copyTransformArray(size_t node_size,
                        DeviceArray<TransformType> &transform_array)
{
  int block = 256;
  int grid = min(divUp(4 * node_size, block), 512);

  copyTransformArrayKernel<<<grid, block>>>(4 * node_size,
                                            transform_array.getHandle(),
                                            transform_array.getWriteHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int getVoxelBlending(int node_block_dim,
                     int3 node_block_grid,
                     int voxel_blend_table_entry,
                     float *blend_weight,
                     uint *blend_node_id,
                     Affine3d &blend_transform)
{
  int blend_count = 0;

  int voxel_blend_table_entry_quad = voxel_blend_table_entry * VOXEL_BLENDING_MAX_QUAD_STRIDE;

  for (int quad = 0; quad < VOXEL_BLENDING_MAX_QUAD_STRIDE; quad++)
  {
    float4 weight_entry_quad = tex1Dfetch(blend_weight_table_texture,
                                          voxel_blend_table_entry_quad + quad);
    float *weight_entry = (float *)&weight_entry_quad;

    int4 code_entry_quad = tex1Dfetch(blend_code_table_texture,
                                      voxel_blend_table_entry_quad + quad);
    int *code_entry = (int *)&code_entry_quad;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
      if (blend_count == 4 * VOXEL_BLENDING_QUAD_STRIDE)
        break;

      float weight_i = weight_entry[i];

      int code_i = code_entry[i];
      int code_i_x = ((code_i & 0x00FF0000) >> 16) - 1;
      int code_i_y = ((code_i & 0x0000FF00) >> 8) - 1;
      int code_i_z = (code_i & 0x000000FF) - 1;

      int3 node_block_grid_i = node_block_grid + make_int3(int(code_i_x), int(code_i_y), int(code_i_z));

      if (node_block_grid_i.x >= 0 &&
          node_block_grid_i.y >= 0 &&
          node_block_grid_i.z >= 0 &&
          node_block_grid_i.x < node_block_dim &&
          node_block_grid_i.y < node_block_dim &&
          node_block_grid_i.z < node_block_dim)
      {
        int node_block_hash_i = getGridHash(node_block_grid_i, node_block_dim);
        HashEntryType node_hash_entry_i = tex1Dfetch(block_hash_table_texture, node_block_hash_i);
        uint node_id_i = node_hash_entry_i.x;
        TransformType transform_col[4];
        Affine3d transform_i;

        if (node_id_i != 0xFFFFFFFF)
        {
          transform_col[0] = tex1Dfetch(transform_texture, 4 * node_id_i + 0);
          transform_col[1] = tex1Dfetch(transform_texture, 4 * node_id_i + 1);
          transform_col[2] = tex1Dfetch(transform_texture, 4 * node_id_i + 2);
          transform_col[3] = tex1Dfetch(transform_texture, 4 * node_id_i + 3);
          transform_i.setValue(transform_col);

          blend_transform += (transform_i * weight_i);

          blend_weight[blend_count] = weight_i;
          blend_node_id[blend_count++] = node_id_i;
        }
      }
    } // i
  } // quad

  // normalization
  float weight_sum = 0.f;

  #pragma unroll
  for (int k = 0; k < blend_count; k++)
    weight_sum += blend_weight[k];

  if (weight_sum > 5e-3)
  {
    blend_transform *= (1.f / weight_sum);

    #pragma unroll
    for (int k = 0; k < blend_count; k++)
      blend_weight[k] /= weight_sum;
  }
  else
  {
    blend_count = 0;
    blend_transform.makeIdentity();
  }

  return blend_count;
}

__global__
void fuseVolumeKernel(bool is_first_frame,
                      float4 color_thresholds,
                      const Intrinsics intrin,
                      const Affine3d transform,
                      const Roi roi,
                      float trunc_dist_inv,
                      float voxel_length,
                      int voxel_block_inner_dim,
                      float voxel_block_length,
                      int voxel_block_dim,
                      int node_block_dim,
                      int dim_ratio,
                      int voxel_block_inner_hash_size,
                      int voxel_block_size,
                      const DeviceArrayHandle<uint> voxel_block_hash_array,
                      const DeviceArrayHandle<uint2> voxel_block_hash_table,
                      DeviceArrayHandle<TsdfVoxelType> tsdf_voxel_block_array,
                      DeviceArrayHandle<RgbaVoxelType> rgba_voxel_block_array)
{
  __shared__ uint voxel_block_hash;
  __shared__ uint2 voxel_block_hash_entry;

  const int voxel_block_stride = gridDim.x;
  const int voxel_block_inner_stride = blockDim.x;

  // block-stride loop
  for (int block_id = blockIdx.x;
       block_id < voxel_block_size;
       block_id += voxel_block_stride)
  {
    if (threadIdx.x == 0)
    {
      voxel_block_hash = voxel_block_hash_array.at(block_id);
      voxel_block_hash_entry = voxel_block_hash_table.at(voxel_block_hash);
    }

    __syncthreads();

    // is active voxel block
    if (voxel_block_hash_entry.x != 1)
      continue;

    int3 voxel_block_grid = getGrid(voxel_block_hash, voxel_block_dim);
    float3 voxel_block_origin = getGridCoo(voxel_block_grid, voxel_block_length, 0.f);

    int3 node_block_grid;
    node_block_grid.x = voxel_block_grid.x / dim_ratio;
    node_block_grid.y = voxel_block_grid.y / dim_ratio;
    node_block_grid.z = voxel_block_grid.z / dim_ratio;

    int3 pass_grid;
    pass_grid.x = voxel_block_grid.x % dim_ratio;
    pass_grid.y = voxel_block_grid.y % dim_ratio;
    pass_grid.z = voxel_block_grid.z % dim_ratio;

    int pass = getGridHash(pass_grid, dim_ratio);
    int pass_stride = pass * voxel_block_inner_hash_size;

    // voxel-stride loop
    for (int voxel_block_inner_hash = threadIdx.x;
         voxel_block_inner_hash < voxel_block_inner_hash_size;
         voxel_block_inner_hash += voxel_block_inner_stride)
    {
      int3 voxel_block_inner_grid = getGrid(voxel_block_inner_hash, voxel_block_inner_dim);
      float3 voxel_block_inner_coo = getGridCoo(voxel_block_inner_grid, voxel_length, 0.5f);
      float4 voxel = make_float4(voxel_block_origin + voxel_block_inner_coo, 1.f);

      int voxel_blend_table_entry = pass_stride + voxel_block_inner_hash;

      // linear blending
      float blend_weight[4 * VOXEL_BLENDING_QUAD_STRIDE];
      uint blend_node_id[4 * VOXEL_BLENDING_QUAD_STRIDE];
      Affine3d blend_transform(0.f);

      int blend_count = getVoxelBlending(node_block_dim,
                                         node_block_grid,
                                         voxel_blend_table_entry,
                                         blend_weight,
                                         blend_node_id,
                                         blend_transform);

      float3 proj_voxel = make_float3(transform * (blend_transform * voxel));

      float proj_voxel_norm = length(proj_voxel);

      int2 map_coo;
      map_coo.x = __float2int_rn(proj_voxel.x * intrin.fx / proj_voxel.z + intrin.cx);
      map_coo.y = __float2int_rn(proj_voxel.y * intrin.fy / proj_voxel.z + intrin.cy);

      bool visible = ((map_coo.x >= roi.start.x) && (map_coo.x < roi.end.x) &&
                      (map_coo.y >= roi.start.y) && (map_coo.y < roi.end.y) &&
                      (proj_voxel.z > 0.0f));

      if (visible && blend_count > 0)
      {
        float depth_scaled = tex2D(depth_texture, map_coo.x, map_coo.y);

        float depth_scaled_dx, depth_scaled_dy;

        depth_scaled_dx  = tex2D(depth_texture, map_coo.x-2, map_coo.y);
        depth_scaled_dx -= tex2D(depth_texture, map_coo.x-1, map_coo.y) * 8.0f;
        depth_scaled_dx += tex2D(depth_texture, map_coo.x+1, map_coo.y) * 8.0f;
        depth_scaled_dx -= tex2D(depth_texture, map_coo.x+2, map_coo.y);
        depth_scaled_dx /= 12.0f;

        depth_scaled_dy  = tex2D(depth_texture, map_coo.x, map_coo.y-2);
        depth_scaled_dy -= tex2D(depth_texture, map_coo.x, map_coo.y-1) * 8.0f;
        depth_scaled_dy += tex2D(depth_texture, map_coo.x, map_coo.y+1) * 8.0f;
        depth_scaled_dy -= tex2D(depth_texture, map_coo.x, map_coo.y+2);
        depth_scaled_dy /= 12.0f;

        float derivative_norm = sqrtf(depth_scaled_dx * depth_scaled_dx + depth_scaled_dy * depth_scaled_dy);

        uchar color_r = tex2D(color_texture, 3*map_coo.x+0, map_coo.y);
        uchar color_g = tex2D(color_texture, 3*map_coo.x+1, map_coo.y);
        uchar color_b = tex2D(color_texture, 3*map_coo.x+2, map_coo.y);

        float color_r_f = __uint2float_rd(color_r);
        float color_g_f = __uint2float_rd(color_g);
        float color_b_f = __uint2float_rd(color_b);

        const float max = fmaxf(color_r_f, fmaxf(color_g_f, color_b_f));
        const float min = fminf(color_r_f, fminf(color_g_f, color_b_f));

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

        if (max == color_r_f) color_h = 60.f * ((color_g_f - color_b_f) / diff);
        else if (max == color_g_f) color_h = 60.f * (2.f + (color_b_f - color_r_f) / diff);
        else color_h = 60.f * (4.f + (color_r_f - color_g_f) / diff);

        if (color_h < 0.f) color_h += 360.f;

        bool target = (color_h > color_thresholds.x &&
                       color_h < color_thresholds.y &&
                       color_s > color_thresholds.z);

        float tsdf = (depth_scaled - proj_voxel_norm) * trunc_dist_inv;
        uchar4 rgba = make_uchar4(color_r, color_g, color_b, 0);
        int weight = 32;

        bool integrate = (depth_scaled != 0.0f &&
                          /*target &&*/
                          /*derivative_norm <= 0.004f &&*/
                          tsdf >= -1.0f &&
                          tsdf <= 1.0f);

        if (!is_first_frame)
          integrate &= (derivative_norm <= 0.002f);

        if (integrate)
        {
          updateVoxel(tsdf, rgba, weight, tsdf_voxel_block_array.at(block_id * voxel_block_inner_hash_size + voxel_block_inner_hash), rgba_voxel_block_array.at(block_id * voxel_block_inner_hash_size + voxel_block_inner_hash));
        }

        if (!is_first_frame && !target && depth_scaled == 0.f)
        {
          updateVoxel_neg(tsdf_voxel_block_array.at(block_id * voxel_block_inner_hash_size + voxel_block_inner_hash));
        }
      } // visible && skinned
    } // voxel-stride loop
  } // block-stride loop
}

void fuseVolume(bool is_first_frame,
                float4 color_thresholds,
                const Intrinsics &intrin,
                const Affine3d &transform,
                const Roi &roi,
                float trunc_dist,
                float voxel_length,
                int voxel_block_inner_dim,
                float voxel_block_length,
                int voxel_block_dim,
                int node_block_dim,
                size_t voxel_block_size,
                const DeviceArray<uint> &voxel_block_hash_array,
                const DeviceArray<HashEntryType> &voxel_block_hash_table,
                DeviceArray<TsdfVoxelType> &tsdf_voxel_block_array,
                DeviceArray<RgbaVoxelType> &rgba_voxel_block_array)
{
  int dim_ratio = voxel_block_dim / node_block_dim;

  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;

  int block = min(voxel_block_inner_hash_size, 256);
  int grid = min(static_cast<int>(voxel_block_size), 1024);

  fuseVolumeKernel<<<grid, block>>>(is_first_frame,
                                    color_thresholds,
                                    intrin,
                                    transform,
                                    roi,
                                    1.0f / trunc_dist,
                                    voxel_length,
                                    voxel_block_inner_dim,
                                    voxel_block_length,
                                    voxel_block_dim,
                                    node_block_dim,
                                    dim_ratio,
                                    voxel_block_inner_hash_size,
                                    voxel_block_size,
                                    voxel_block_hash_array.getHandle(),
                                    voxel_block_hash_table.getHandle(),
                                    tsdf_voxel_block_array.getHandle(),
                                    rgba_voxel_block_array.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
void fetchTsdfVoxelTex(int voxel_entry, float &tsdf, int &weight)
{
  TsdfVoxelType value_raw = tex1Dfetch(tsdf_voxel_texture, voxel_entry);
  tsdf = float(value_raw.x) / DIVISOR;
  weight = int(value_raw.y);
}

__device__ __forceinline__
void fetchRgbaVoxelTex(int voxel_entry, int &rgba)
{
  RgbaVoxelType value_raw = tex1Dfetch(rgba_voxel_texture, voxel_entry);
  int *rgba_ptr = reinterpret_cast<int *>(&value_raw);
  rgba = *rgba_ptr;
}

__device__ __forceinline__
int computeCubeIndex(int voxel_entry_start, int voxel_block_inner_dim, float tsdf_field[8], int rgba_field[8])
{
  int weight;
  int dx = voxel_block_inner_dim * voxel_block_inner_dim;
  int dy = voxel_block_inner_dim;
  int dz = 1;
  int voxel_entry;

  voxel_entry = voxel_entry_start;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[0], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[0]);

  voxel_entry = voxel_entry_start + dz;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[4], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[4]);

  voxel_entry = voxel_entry_start + dy;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[3], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[3]);

  voxel_entry = voxel_entry_start + dy + dz;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[7], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[7]);

  voxel_entry = voxel_entry_start + dx;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[1], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[1]);

  voxel_entry = voxel_entry_start + dx + dz;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[5], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[5]);

  voxel_entry = voxel_entry_start + dx + dy;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[2], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[2]);

  voxel_entry = voxel_entry_start + dx + dy + dz;
  fetchTsdfVoxelTex(voxel_entry, tsdf_field[6], weight); if (weight == 0) return 0;
  fetchRgbaVoxelTex(voxel_entry, rgba_field[6]);

  // calculate flag indicating if each vertex is inside or outside isosurface
  int cubeindex;
  cubeindex  = int(tsdf_field[0] < ISO_VALUE);
  cubeindex += int(tsdf_field[1] < ISO_VALUE) * 2;
  cubeindex += int(tsdf_field[2] < ISO_VALUE) * 4;
  cubeindex += int(tsdf_field[3] < ISO_VALUE) * 8;
  cubeindex += int(tsdf_field[4] < ISO_VALUE) * 16;
  cubeindex += int(tsdf_field[5] < ISO_VALUE) * 32;
  cubeindex += int(tsdf_field[6] < ISO_VALUE) * 64;
  cubeindex += int(tsdf_field[7] < ISO_VALUE) * 128;

  return cubeindex;
}

__global__
void getOccupiedVoxelsKernel(int voxel_block_size,
                             int voxel_block_inner_hash_size,
                             int voxel_block_inner_dim,
                             int compact_voxel_block_inner_dim,
                             int voxel_grid_dim,
                             int voxel_block_dim,
                             const DeviceArrayHandle<uint> voxel_block_hash_array,
                             DeviceArrayHandle<uint2> voxel_block_hash_table,
                             DeviceArrayHandle<int> voxel_grid_hash_array,
                             DeviceArrayHandle<int> vertex_num_array,
                             DeviceArrayHandle<float4> cube_tsdf_field_array,
                             DeviceArrayHandle<int4> cube_rgba_field_array)
{
  __shared__ uint voxel_block_hash;
  __shared__ uint2 voxel_block_hash_entry;

  const int voxel_block_stride = gridDim.x;
  const int voxel_block_inner_stride = blockDim.x;
  const int lane_id = threadIdx.x % warpSize;

  // block-stride loop
  for (int block_id = blockIdx.x;
       block_id < voxel_block_size;
       block_id += voxel_block_stride)
  {
    if (threadIdx.x == 0)
    {
      voxel_block_hash = voxel_block_hash_array.at(block_id);
      voxel_block_hash_entry = voxel_block_hash_table.at(voxel_block_hash);
      voxel_block_hash_table.at(voxel_block_hash) = make_uint2(0xFFFFFFFF, voxel_block_hash_entry.y);
    }

    __syncthreads();

    int3 voxel_block_grid = getGrid(voxel_block_hash, voxel_block_dim);

    int voxel_block_entry = block_id * voxel_block_inner_hash_size;

    // voxel-stride loop
    for (int voxel_block_inner_hash = threadIdx.x;
         voxel_block_inner_hash < voxel_block_inner_hash_size;
         voxel_block_inner_hash += voxel_block_inner_stride)
    {
      int3 voxel_block_inner_grid = getGrid(voxel_block_inner_hash, voxel_block_inner_dim);

      int3 voxel_grid = voxel_block_grid * compact_voxel_block_inner_dim + voxel_block_inner_grid;

      int voxel_grid_hash = getGridHash(voxel_grid, voxel_grid_dim);

      int voxel_entry = voxel_block_entry + voxel_block_inner_hash;

      int vertex_num = 0;
      float tsdf_field[8];
      int rgba_field[8];

      if (voxel_block_inner_grid.x < compact_voxel_block_inner_dim &&
          voxel_block_inner_grid.y < compact_voxel_block_inner_dim &&
          voxel_block_inner_grid.z < compact_voxel_block_inner_dim)
      {
        int cube_index = computeCubeIndex(voxel_entry, voxel_block_inner_dim, tsdf_field, rgba_field);

        vertex_num = (cube_index == 0 || cube_index == 255) ? 0 : tex1Dfetch(num_verts_texture, cube_index);
      }

      /*
       * Warp-Aggregated Atomics
       */

      // Step 1: leader election
#ifdef CUDA8
      unsigned int valid = __ballot(vertex_num > 0);
#else
      unsigned int active = __activemask();
      unsigned int valid = __ballot_sync(active, vertex_num > 0);
#endif

      int leader = __ffs(valid) - 1;

      // Step 2: computing the total increment (in the warp)
      int total = __popc(valid);

      if (total == 0)
        continue;

      // Step 3: performing the atomic add
      unsigned int lane_mask_lt = (1 << lane_id) - 1;
      unsigned int rank = __popc(valid & lane_mask_lt);

      int warp_old = 0, thread_offset = 0;

      if (lane_id == leader)
        warp_old = atomicAdd(&global_count, total);

      // Step 4: broadcasting the result
#ifdef CUDA8
      warp_old = __shfl(warp_old, leader);
#else
      warp_old = __shfl_sync(valid, warp_old, leader);
#endif

      thread_offset = warp_old + (int)rank;

      if (thread_offset < MAX_TRIANGLES_SIZE && vertex_num > 0)
      {
        voxel_grid_hash_array.at(thread_offset) = voxel_grid_hash;

        vertex_num_array.at(thread_offset) = vertex_num;

        cube_tsdf_field_array.at(2 * thread_offset + 0) = make_float4(tsdf_field[0], tsdf_field[1], tsdf_field[2], tsdf_field[3]);
        cube_tsdf_field_array.at(2 * thread_offset + 1) = make_float4(tsdf_field[4], tsdf_field[5], tsdf_field[6], tsdf_field[7]);
        cube_rgba_field_array.at(2 * thread_offset + 0) = make_int4(rgba_field[0], rgba_field[1], rgba_field[2], rgba_field[3]);
        cube_rgba_field_array.at(2 * thread_offset + 1) = make_int4(rgba_field[4], rgba_field[5], rgba_field[6], rgba_field[7]);
      }

      bool full = (warp_old + total >= MAX_TRIANGLES_SIZE);

      if (full)
        break;
    } // voxel-stride loop
  } // block-stride loop

  __syncthreads();

  // prepare for future scans
  if (threadIdx.x == 0)
  {
    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_count = min(MAX_TRIANGLES_SIZE, global_count);
      global_count = 0;
      blocks_done = 0;
    }
  }
}

int getOccupiedVoxels(int voxel_block_inner_dim,
                      int voxel_block_dim,
                      size_t voxel_block_size,
                      const DeviceArray<uint> &voxel_block_hash_array,
                      DeviceArray<HashEntryType> &voxel_block_hash_table,
                      DeviceArray<int> &voxel_grid_hash_array,
                      DeviceArray<int> &vertex_num_array,
                      DeviceArray<float4> &cube_tsdf_field_array,
                      DeviceArray<int4> &cube_rgba_field_array)
{
  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;
  int compact_voxel_block_inner_dim = voxel_block_inner_dim - 1;
  int voxel_grid_dim = voxel_block_dim * compact_voxel_block_inner_dim;

  int block = min(voxel_block_inner_hash_size, 256);
  int grid = min(static_cast<int>(voxel_block_size), 512);

  getOccupiedVoxelsKernel<<<grid, block>>>(voxel_block_size,
                                           voxel_block_inner_hash_size,
                                           voxel_block_inner_dim,
                                           compact_voxel_block_inner_dim,
                                           voxel_grid_dim,
                                           voxel_block_dim,
                                           voxel_block_hash_array.getHandle(),
                                           voxel_block_hash_table.getHandle(),
                                           voxel_grid_hash_array.getHandle(),
                                           vertex_num_array.getHandle(),
                                           cube_tsdf_field_array.getHandle(),
                                           cube_rgba_field_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  int size;
  checkCudaErrors(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));
  return size;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
size_t computeOffsetsAndTotalVertices(int occupied_voxel_size,
                                      DeviceArray<int>& vertex_num_array,
                                      DeviceArray<int>& vertex_offset_array)
{
  thrust::device_ptr<int> beg(vertex_num_array.getDevicePtr());
  thrust::device_ptr<int> end = beg + occupied_voxel_size;
  thrust::device_ptr<int> result(vertex_offset_array.getDevicePtr());

  // prefix sum
  thrust::exclusive_scan(beg, end, result);

  int num_last_elem, offset_last_elem;
  int *vert_num_last = vertex_num_array.getDevicePtr() + occupied_voxel_size - 1;
  int *vert_offset_last = vertex_offset_array.getDevicePtr() + occupied_voxel_size - 1;

  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&num_last_elem),
                             reinterpret_cast<void *>(vert_num_last),
                             sizeof(int), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpy(reinterpret_cast<void *>(&offset_last_elem),
                             reinterpret_cast<void *>(vert_offset_last),
                             sizeof(int), cudaMemcpyDeviceToHost));

  return static_cast<size_t>(num_last_elem + offset_last_elem);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int getCubeIndex(float4 field[2])
{
  // calculate flag indicating if each vertex is inside or outside isosurface
  int cubeindex;
  cubeindex  = int(field[0].x < ISO_VALUE);
  cubeindex += int(field[0].y < ISO_VALUE) * 2;
  cubeindex += int(field[0].z < ISO_VALUE) * 4;
  cubeindex += int(field[0].w < ISO_VALUE) * 8;
  cubeindex += int(field[1].x < ISO_VALUE) * 16;
  cubeindex += int(field[1].y < ISO_VALUE) * 32;
  cubeindex += int(field[1].z < ISO_VALUE) * 64;
  cubeindex += int(field[1].w < ISO_VALUE) * 128;

  return cubeindex;
}

// __device__ __forceinline__
// float3 vertexInterp(float3 p0, float3 p1, float f0, float f1)
// {
//   float t = (ISO_VALUE - f0) / (f1 - f0 + 1e-15f);
//   float x = p0.x + t * (p1.x - p0.x);
//   float y = p0.y + t * (p1.y - p0.y);
//   float z = p0.z + t * (p1.z - p0.z);
// 
//   return make_float3(x, y, z);
// }

__device__ __forceinline__
void fieldInterp(float3 p0, float3 p1, int rgba0, int rgba1, float tsdf0, float tsdf1, float3 &vert_out, int &rgba_out)
{
  float w = (ISO_VALUE - tsdf0) / (tsdf1 - tsdf0 + 1e-15f);
  vert_out = p0 + (p1 - p0) * w;

  uchar4 *rgba0_ptr = reinterpret_cast<uchar4 *>(&rgba0);
  uchar4 *rgba1_ptr = reinterpret_cast<uchar4 *>(&rgba1);

  int rgba_interp = 0;
  uchar4 *rgba_ptr = reinterpret_cast<uchar4 *>(&rgba_interp);
  rgba_ptr[0].x = (uchar)min(__float2int_rn(float(rgba0_ptr[0].x) + (float(rgba1_ptr[0].x) - float(rgba0_ptr[0].x)) * w), 255);
  rgba_ptr[0].y = (uchar)min(__float2int_rn(float(rgba0_ptr[0].y) + (float(rgba1_ptr[0].y) - float(rgba0_ptr[0].y)) * w), 255);
  rgba_ptr[0].z = (uchar)min(__float2int_rn(float(rgba0_ptr[0].z) + (float(rgba1_ptr[0].z) - float(rgba0_ptr[0].z)) * w), 255);
  rgba_ptr[0].w = 0;

  rgba_out = rgba_interp;
}

// Ensure CTA_SIZE leq than 256
#define CTA_SIZE 256
__global__
void generateTrianglesKernel(float volume_center_offset,
                             float voxel_length,
                             int voxel_grid_dim,
                             int occupied_voxel_size,
                             const DeviceArrayHandle<int> voxel_grid_hash_array,
                             const DeviceArrayHandle<int> vertex_num_array,
                             const DeviceArrayHandle<int> vertex_offset_array,
                             const DeviceArrayHandle<float4> cube_tsdf_field_array,
                             const DeviceArrayHandle<int4>   cube_rgba_field_array,
                             DeviceArrayHandle<float4> mesh_vertex_array,
                             DeviceArrayHandle<float4> mesh_normal_array,
                             DeviceArrayHandle<float4> mesh_color_array,
                             DeviceArrayHandle<float4> surfel_vertex_array,
                             DeviceArrayHandle<float4> surfel_normal_array,
                             DeviceArrayHandle<float>  surfel_color_array)
{
  __shared__ float3 vert_list[12][CTA_SIZE];
  __shared__ int    rgba_list[12][CTA_SIZE];

  const int stride = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < occupied_voxel_size;
       i += stride)
  {
    int voxel_grid_hash = voxel_grid_hash_array.at(i);
    int vertex_num = vertex_num_array.at(i);
    int vertex_offset = vertex_offset_array.at(i);
    float4 tsdf_field[2];
    int4 rgba_field[2];
    tsdf_field[0] = cube_tsdf_field_array.at(2 * i + 0);
    tsdf_field[1] = cube_tsdf_field_array.at(2 * i + 1);
    rgba_field[0] = cube_rgba_field_array.at(2 * i + 0);
    rgba_field[1] = cube_rgba_field_array.at(2 * i + 1);

    int3 voxel_grid = getGrid(voxel_grid_hash, voxel_grid_dim);

    int cube_index = getCubeIndex(tsdf_field);

    float3 grid_coo[8];
    grid_coo[0] = getGridCoo(voxel_grid + make_int3(0, 0, 0), voxel_length, 0.5f);
    grid_coo[1] = getGridCoo(voxel_grid + make_int3(1, 0, 0), voxel_length, 0.5f);
    grid_coo[2] = getGridCoo(voxel_grid + make_int3(1, 1, 0), voxel_length, 0.5f);
    grid_coo[3] = getGridCoo(voxel_grid + make_int3(0, 1, 0), voxel_length, 0.5f);
    grid_coo[4] = getGridCoo(voxel_grid + make_int3(0, 0, 1), voxel_length, 0.5f);
    grid_coo[5] = getGridCoo(voxel_grid + make_int3(1, 0, 1), voxel_length, 0.5f);
    grid_coo[6] = getGridCoo(voxel_grid + make_int3(1, 1, 1), voxel_length, 0.5f);
    grid_coo[7] = getGridCoo(voxel_grid + make_int3(0, 1, 1), voxel_length, 0.5f);

    // vertlist[0][tid]  = vertexInterp(grid_coo[0], grid_coo[1], tsdf_field[0].x, tsdf_field[0].y);
    // vertlist[1][tid]  = vertexInterp(grid_coo[1], grid_coo[2], tsdf_field[0].y, tsdf_field[0].z);
    // vertlist[2][tid]  = vertexInterp(grid_coo[2], grid_coo[3], tsdf_field[0].z, tsdf_field[0].w);
    // vertlist[3][tid]  = vertexInterp(grid_coo[3], grid_coo[0], tsdf_field[0].w, tsdf_field[0].x);
    // vertlist[4][tid]  = vertexInterp(grid_coo[4], grid_coo[5], tsdf_field[1].x, tsdf_field[1].y);
    // vertlist[5][tid]  = vertexInterp(grid_coo[5], grid_coo[6], tsdf_field[1].y, tsdf_field[1].z);
    // vertlist[6][tid]  = vertexInterp(grid_coo[6], grid_coo[7], tsdf_field[1].z, tsdf_field[1].w);
    // vertlist[7][tid]  = vertexInterp(grid_coo[7], grid_coo[4], tsdf_field[1].w, tsdf_field[1].x);
    // vertlist[8][tid]  = vertexInterp(grid_coo[0], grid_coo[4], tsdf_field[0].x, tsdf_field[1].x);
    // vertlist[9][tid]  = vertexInterp(grid_coo[1], grid_coo[5], tsdf_field[0].y, tsdf_field[1].y);
    // vertlist[10][tid] = vertexInterp(grid_coo[2], grid_coo[6], tsdf_field[0].z, tsdf_field[1].z);
    // vertlist[11][tid] = vertexInterp(grid_coo[3], grid_coo[7], tsdf_field[0].w, tsdf_field[1].w);

    fieldInterp(grid_coo[0], grid_coo[1], rgba_field[0].x, rgba_field[0].y, tsdf_field[0].x, tsdf_field[0].y, vert_list[0][tid],  rgba_list[0][tid]);
    fieldInterp(grid_coo[1], grid_coo[2], rgba_field[0].y, rgba_field[0].z, tsdf_field[0].y, tsdf_field[0].z, vert_list[1][tid],  rgba_list[1][tid]);
    fieldInterp(grid_coo[2], grid_coo[3], rgba_field[0].z, rgba_field[0].w, tsdf_field[0].z, tsdf_field[0].w, vert_list[2][tid],  rgba_list[2][tid]);
    fieldInterp(grid_coo[3], grid_coo[0], rgba_field[0].w, rgba_field[0].x, tsdf_field[0].w, tsdf_field[0].x, vert_list[3][tid],  rgba_list[3][tid]);
    fieldInterp(grid_coo[4], grid_coo[5], rgba_field[1].x, rgba_field[1].y, tsdf_field[1].x, tsdf_field[1].y, vert_list[4][tid],  rgba_list[4][tid]);
    fieldInterp(grid_coo[5], grid_coo[6], rgba_field[1].y, rgba_field[1].z, tsdf_field[1].y, tsdf_field[1].z, vert_list[5][tid],  rgba_list[5][tid]);
    fieldInterp(grid_coo[6], grid_coo[7], rgba_field[1].z, rgba_field[1].w, tsdf_field[1].z, tsdf_field[1].w, vert_list[6][tid],  rgba_list[6][tid]);
    fieldInterp(grid_coo[7], grid_coo[4], rgba_field[1].w, rgba_field[1].x, tsdf_field[1].w, tsdf_field[1].x, vert_list[7][tid],  rgba_list[7][tid]);
    fieldInterp(grid_coo[0], grid_coo[4], rgba_field[0].x, rgba_field[1].x, tsdf_field[0].x, tsdf_field[1].x, vert_list[8][tid],  rgba_list[8][tid]);
    fieldInterp(grid_coo[1], grid_coo[5], rgba_field[0].y, rgba_field[1].y, tsdf_field[0].y, tsdf_field[1].y, vert_list[9][tid],  rgba_list[9][tid]);
    fieldInterp(grid_coo[2], grid_coo[6], rgba_field[0].z, rgba_field[1].z, tsdf_field[0].z, tsdf_field[1].z, vert_list[10][tid], rgba_list[10][tid]);
    fieldInterp(grid_coo[3], grid_coo[7], rgba_field[0].w, rgba_field[1].w, tsdf_field[0].w, tsdf_field[1].w, vert_list[11][tid], rgba_list[11][tid]);

    for (int j = 0; j < vertex_num; j += 3)
    {
      int index = vertex_offset + j;
      float3 v[3];
      uchar4 *color_ptr;
      float4 rgba[3];
      int edge;

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 0);
      v[0] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[0] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 1);
      v[1] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[1] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 2);
      v[2] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[2] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      // calculate triangle surface normal
      float3 v10, v20;
      v10 = v[1] - v[0];
      v20 = v[2] - v[0];
      float3 n = normalize(cross(v10, v20));

      mesh_vertex_array.at(index + 0) = make_float4(v[0], 1.f);
      mesh_vertex_array.at(index + 1) = make_float4(v[1], 1.f);
      mesh_vertex_array.at(index + 2) = make_float4(v[2], 1.f);

      mesh_normal_array.at(index + 0) = make_float4(n, 0.f);
      mesh_normal_array.at(index + 1) = make_float4(n, 0.f);
      mesh_normal_array.at(index + 2) = make_float4(n, 0.f);

      mesh_color_array.at(index + 0) = rgba[0];
      mesh_color_array.at(index + 1) = rgba[1];
      mesh_color_array.at(index + 2) = rgba[2];

      v[0] += v[1];
      v[0] += v[2];
      v[0] /= 3.0f;
      v[0] += volume_center_offset;
      rgba[0] += rgba[1];
      rgba[0] += rgba[2];
      rgba[0] /= 3.0f;

      surfel_vertex_array.at(index / 3) = make_float4(v[0], 1.f);
      surfel_normal_array.at(index / 3) = make_float4(n, 0.f);

      // const float alpha = 0.4642f;
      // surfel_color_array.at(index / 3) = 0.5f + logf(rgba[0].y) - logf(rgba[0].z) * alpha - logf(rgba[0].x) * (1.f - alpha);

      int r_32 = min(__float2int_rn(rgba[0].x * 255.f), 255);
      int g_32 = min(__float2int_rn(rgba[0].y * 255.f), 255);
      int b_32 = min(__float2int_rn(rgba[0].z * 255.f), 255);
      int rgb_32 = (r_32 << 16 | g_32 << 8 | b_32);

      surfel_color_array.at(index / 3) = *reinterpret_cast<float *>(&rgb_32);
    }
  } // grid-stride loop
}

void generateTriangles(float volume_length,
                       float voxel_length,
                       int voxel_block_inner_dim,
                       int voxel_block_dim,
                       int occupied_voxel_size,
                       const DeviceArray<int> &voxel_grid_hash_array,
                       const DeviceArray<int> &vertex_num_array,
                       const DeviceArray<int> &vertex_offset_array,
                       const DeviceArray<float4> &cube_tsdf_field_array,
                       const DeviceArray<int4>   &cube_rgba_field_array,
                       DeviceArray<float4> &mesh_vertex_array,
                       DeviceArray<float4> &mesh_normal_array,
                       DeviceArray<float4> &mesh_color_array,
                       DeviceArray<float4> &surfel_vertex_array,
                       DeviceArray<float4> &surfel_normal_array,
                       DeviceArray<float>  &surfel_color_array)
{
  // mesh_vertex_array.map();
  // mesh_normal_array.map();
  // mesh_color_array.map();

  int compact_voxel_block_inner_dim = voxel_block_inner_dim - 1;
  int voxel_grid_dim = voxel_block_dim * compact_voxel_block_inner_dim;

  int block = CTA_SIZE;
  int grid = min(divUp(occupied_voxel_size, block), 512);

  generateTrianglesKernel<<<grid, block>>>(volume_length / 2.0f, // volume center offset
                                           voxel_length,
                                           voxel_grid_dim,
                                           occupied_voxel_size,
                                           voxel_grid_hash_array.getHandle(),
                                           vertex_num_array.getHandle(),
                                           vertex_offset_array.getHandle(),
                                           cube_tsdf_field_array.getHandle(),
                                           cube_rgba_field_array.getHandle(),
                                           mesh_vertex_array.getHandle(),
                                           mesh_normal_array.getHandle(),
                                           mesh_color_array.getHandle(),
                                           surfel_vertex_array.getHandle(),
                                           surfel_normal_array.getHandle(),
                                           surfel_color_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // mesh_vertex_array.unmap();
  // mesh_normal_array.unmap();
  // mesh_color_array.unmap();
}
#undef CTA_SIZE

// Ensure CTA_SIZE leq than 256
#define CTA_SIZE 256
__global__
void generateTrianglesKernel(float volume_center_offset,
                             float voxel_length,
                             int voxel_grid_dim,
                             int occupied_voxel_size,
                             const DeviceArrayHandle<int> voxel_grid_hash_array,
                             const DeviceArrayHandle<int> vertex_num_array,
                             const DeviceArrayHandle<int> vertex_offset_array,
                             const DeviceArrayHandle<float4> cube_tsdf_field_array,
                             const DeviceArrayHandle<int4>   cube_rgba_field_array,
                             DeviceArrayHandle<float4> mesh_vertex_array,
                             DeviceArrayHandle<float4> mesh_normal_array,
                             DeviceArrayHandle<float4> mesh_color_array)
{
  __shared__ float3 vert_list[12][CTA_SIZE];
  __shared__ int    rgba_list[12][CTA_SIZE];

  const int stride = blockDim.x * gridDim.x;
  const int tid = threadIdx.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < occupied_voxel_size;
       i += stride)
  {
    int voxel_grid_hash = voxel_grid_hash_array.at(i);
    int vertex_num = vertex_num_array.at(i);
    int vertex_offset = vertex_offset_array.at(i);
    float4 tsdf_field[2];
    int4 rgba_field[2];
    tsdf_field[0] = cube_tsdf_field_array.at(2 * i + 0);
    tsdf_field[1] = cube_tsdf_field_array.at(2 * i + 1);
    rgba_field[0] = cube_rgba_field_array.at(2 * i + 0);
    rgba_field[1] = cube_rgba_field_array.at(2 * i + 1);

    int3 voxel_grid = getGrid(voxel_grid_hash, voxel_grid_dim);

    int cube_index = getCubeIndex(tsdf_field);

    float3 grid_coo[8];
    grid_coo[0] = getGridCoo(voxel_grid + make_int3(0, 0, 0), voxel_length, 0.5f);
    grid_coo[1] = getGridCoo(voxel_grid + make_int3(1, 0, 0), voxel_length, 0.5f);
    grid_coo[2] = getGridCoo(voxel_grid + make_int3(1, 1, 0), voxel_length, 0.5f);
    grid_coo[3] = getGridCoo(voxel_grid + make_int3(0, 1, 0), voxel_length, 0.5f);
    grid_coo[4] = getGridCoo(voxel_grid + make_int3(0, 0, 1), voxel_length, 0.5f);
    grid_coo[5] = getGridCoo(voxel_grid + make_int3(1, 0, 1), voxel_length, 0.5f);
    grid_coo[6] = getGridCoo(voxel_grid + make_int3(1, 1, 1), voxel_length, 0.5f);
    grid_coo[7] = getGridCoo(voxel_grid + make_int3(0, 1, 1), voxel_length, 0.5f);

    // vertlist[0][tid]  = vertexInterp(grid_coo[0], grid_coo[1], tsdf_field[0].x, tsdf_field[0].y);
    // vertlist[1][tid]  = vertexInterp(grid_coo[1], grid_coo[2], tsdf_field[0].y, tsdf_field[0].z);
    // vertlist[2][tid]  = vertexInterp(grid_coo[2], grid_coo[3], tsdf_field[0].z, tsdf_field[0].w);
    // vertlist[3][tid]  = vertexInterp(grid_coo[3], grid_coo[0], tsdf_field[0].w, tsdf_field[0].x);
    // vertlist[4][tid]  = vertexInterp(grid_coo[4], grid_coo[5], tsdf_field[1].x, tsdf_field[1].y);
    // vertlist[5][tid]  = vertexInterp(grid_coo[5], grid_coo[6], tsdf_field[1].y, tsdf_field[1].z);
    // vertlist[6][tid]  = vertexInterp(grid_coo[6], grid_coo[7], tsdf_field[1].z, tsdf_field[1].w);
    // vertlist[7][tid]  = vertexInterp(grid_coo[7], grid_coo[4], tsdf_field[1].w, tsdf_field[1].x);
    // vertlist[8][tid]  = vertexInterp(grid_coo[0], grid_coo[4], tsdf_field[0].x, tsdf_field[1].x);
    // vertlist[9][tid]  = vertexInterp(grid_coo[1], grid_coo[5], tsdf_field[0].y, tsdf_field[1].y);
    // vertlist[10][tid] = vertexInterp(grid_coo[2], grid_coo[6], tsdf_field[0].z, tsdf_field[1].z);
    // vertlist[11][tid] = vertexInterp(grid_coo[3], grid_coo[7], tsdf_field[0].w, tsdf_field[1].w);

    fieldInterp(grid_coo[0], grid_coo[1], rgba_field[0].x, rgba_field[0].y, tsdf_field[0].x, tsdf_field[0].y, vert_list[0][tid],  rgba_list[0][tid]);
    fieldInterp(grid_coo[1], grid_coo[2], rgba_field[0].y, rgba_field[0].z, tsdf_field[0].y, tsdf_field[0].z, vert_list[1][tid],  rgba_list[1][tid]);
    fieldInterp(grid_coo[2], grid_coo[3], rgba_field[0].z, rgba_field[0].w, tsdf_field[0].z, tsdf_field[0].w, vert_list[2][tid],  rgba_list[2][tid]);
    fieldInterp(grid_coo[3], grid_coo[0], rgba_field[0].w, rgba_field[0].x, tsdf_field[0].w, tsdf_field[0].x, vert_list[3][tid],  rgba_list[3][tid]);
    fieldInterp(grid_coo[4], grid_coo[5], rgba_field[1].x, rgba_field[1].y, tsdf_field[1].x, tsdf_field[1].y, vert_list[4][tid],  rgba_list[4][tid]);
    fieldInterp(grid_coo[5], grid_coo[6], rgba_field[1].y, rgba_field[1].z, tsdf_field[1].y, tsdf_field[1].z, vert_list[5][tid],  rgba_list[5][tid]);
    fieldInterp(grid_coo[6], grid_coo[7], rgba_field[1].z, rgba_field[1].w, tsdf_field[1].z, tsdf_field[1].w, vert_list[6][tid],  rgba_list[6][tid]);
    fieldInterp(grid_coo[7], grid_coo[4], rgba_field[1].w, rgba_field[1].x, tsdf_field[1].w, tsdf_field[1].x, vert_list[7][tid],  rgba_list[7][tid]);
    fieldInterp(grid_coo[0], grid_coo[4], rgba_field[0].x, rgba_field[1].x, tsdf_field[0].x, tsdf_field[1].x, vert_list[8][tid],  rgba_list[8][tid]);
    fieldInterp(grid_coo[1], grid_coo[5], rgba_field[0].y, rgba_field[1].y, tsdf_field[0].y, tsdf_field[1].y, vert_list[9][tid],  rgba_list[9][tid]);
    fieldInterp(grid_coo[2], grid_coo[6], rgba_field[0].z, rgba_field[1].z, tsdf_field[0].z, tsdf_field[1].z, vert_list[10][tid], rgba_list[10][tid]);
    fieldInterp(grid_coo[3], grid_coo[7], rgba_field[0].w, rgba_field[1].w, tsdf_field[0].w, tsdf_field[1].w, vert_list[11][tid], rgba_list[11][tid]);

    for (int j = 0; j < vertex_num; j += 3)
    {
      int index = vertex_offset + j;
      float3 v[3];
      uchar4 *color_ptr;
      float4 rgba[3];
      int edge;

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 0);
      v[0] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[0] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 1);
      v[1] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[1] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      edge = tex1Dfetch(tri_texture, (cube_index * 16) + j + 2);
      v[2] = vert_list[edge][tid] - volume_center_offset;
      color_ptr = reinterpret_cast<uchar4 *>(&(rgba_list[edge][tid]));
      rgba[2] = make_float4(float(color_ptr[0].x)/255.f, float(color_ptr[0].y)/255.f, float(color_ptr[0].z)/255.f, 1.f);

      // calculate triangle surface normal
      float3 v10, v20;
      v10 = v[1] - v[0];
      v20 = v[2] - v[0];
      float3 n = normalize(cross(v10, v20));

      mesh_vertex_array.at(index + 0) = make_float4(v[0], 1.f);
      mesh_vertex_array.at(index + 1) = make_float4(v[1], 1.f);
      mesh_vertex_array.at(index + 2) = make_float4(v[2], 1.f);

      mesh_normal_array.at(index + 0) = make_float4(n, 0.f);
      mesh_normal_array.at(index + 1) = make_float4(n, 0.f);
      mesh_normal_array.at(index + 2) = make_float4(n, 0.f);

      mesh_color_array.at(index + 0) = rgba[0];
      mesh_color_array.at(index + 1) = rgba[1];
      mesh_color_array.at(index + 2) = rgba[2];

      // v[0] += v[1];
      // v[0] += v[2];
      // v[0] /= 3.0f;
      // v[0] += volume_center_offset;
      // rgba[0] += rgba[1];
      // rgba[0] += rgba[2];
      // rgba[0] /= 3.0f;

      // surfel_vertex_array.at(index / 3) = make_float4(v[0], 1.f);
      // surfel_normal_array.at(index / 3) = make_float4(n, 0.f);

      // const float alpha = 0.4642f;
      // surfel_color_array.at(index / 3) = 0.5f + logf(rgba[0].y) - logf(rgba[0].z) * alpha - logf(rgba[0].x) * (1.f - alpha);
    }
  } // grid-stride loop
}

void generateTriangles(float volume_length,
                       float voxel_length,
                       int voxel_block_inner_dim,
                       int voxel_block_dim,
                       int occupied_voxel_size,
                       const DeviceArray<int> &voxel_grid_hash_array,
                       const DeviceArray<int> &vertex_num_array,
                       const DeviceArray<int> &vertex_offset_array,
                       const DeviceArray<float4> &cube_tsdf_field_array,
                       const DeviceArray<int4>   &cube_rgba_field_array,
                       DeviceArray<float4> &mesh_vertex_array,
                       DeviceArray<float4> &mesh_normal_array,
                       DeviceArray<float4> &mesh_color_array)
{
  // mesh_vertex_array.map();
  // mesh_normal_array.map();
  // mesh_color_array.map();

  int compact_voxel_block_inner_dim = voxel_block_inner_dim - 1;
  int voxel_grid_dim = voxel_block_dim * compact_voxel_block_inner_dim;

  int block = CTA_SIZE;
  int grid = min(divUp(occupied_voxel_size, block), 512);

  generateTrianglesKernel<<<grid, block>>>(volume_length / 2.0f, // volume center offset
                                           voxel_length,
                                           voxel_grid_dim,
                                           occupied_voxel_size,
                                           voxel_grid_hash_array.getHandle(),
                                           vertex_num_array.getHandle(),
                                           vertex_offset_array.getHandle(),
                                           cube_tsdf_field_array.getHandle(),
                                           cube_rgba_field_array.getHandle(),
                                           mesh_vertex_array.getHandle(),
                                           mesh_normal_array.getHandle(),
                                           mesh_color_array.getHandle());
  checkCudaErrors(cudaGetLastError());

  // mesh_vertex_array.unmap();
  // mesh_normal_array.unmap();
  // mesh_color_array.unmap();
}
#undef CTA_SIZE

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void warpSurfaceKernel(const IcpParamList icp_param,
                       const Intrinsics intrin,
                       const Affine3d transform,
                       float4 volume_center_offset,
                       float voxel_length,
                       int voxel_block_inner_dim,
                       float voxel_block_length,
                       int node_block_dim,
                       int dim_ratio,
                       int voxel_block_inner_hash_size,
                       int triangle_size,
                       float res_sqrt_trunc_inv,
                       const DeviceArrayHandle<float4> ref_mesh_vertex_array,
                       const DeviceArrayHandle<float4> ref_mesh_normal_array,
                       DeviceArrayHandle<float4> ref_mesh_color_array,
                       DeviceArrayHandle<float4> warped_mesh_vertex_array,
                       DeviceArrayHandle<float4> warped_mesh_normal_array,
                       DeviceArrayHandle<float4> warped_surfel_vertex_array,
                       DeviceArrayHandle<float4> warped_surfel_normal_array,
                       DeviceArrayHandle<uint> active_flags)
{
  const int stride = gridDim.x * blockDim.x;

  float energy_local = 0.f;

  // grid-stride loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < triangle_size;
       i += stride)
  {
    float4 warped_surfel_vertex = make_float4(0.f);
    float4 warped_surfel_normal = make_float4(0.f);

    #pragma unroll
    for (int j = 0; j < 3; j += 1)
    {
      float4 ref_mesh_vertex = ref_mesh_vertex_array.at(3 * i + j) + volume_center_offset;
      float4 ref_mesh_normal = ref_mesh_normal_array.at(3 * i + j);

      int3 voxel_block_grid;
      voxel_block_grid.x = __float2int_rz(ref_mesh_vertex.x / voxel_block_length);
      voxel_block_grid.y = __float2int_rz(ref_mesh_vertex.y / voxel_block_length);
      voxel_block_grid.z = __float2int_rz(ref_mesh_vertex.z / voxel_block_length);

      float3 voxel_block_origin = getGridCoo(voxel_block_grid, voxel_block_length, 0.f);

      float3 voxel_block_inner_coo = make_float3(ref_mesh_vertex) - voxel_block_origin;

      int3 voxel_block_inner_grid;
      voxel_block_inner_grid.x = __float2int_rz(voxel_block_inner_coo.x / voxel_length);
      voxel_block_inner_grid.y = __float2int_rz(voxel_block_inner_coo.y / voxel_length);
      voxel_block_inner_grid.z = __float2int_rz(voxel_block_inner_coo.z / voxel_length);

      int voxel_block_inner_hash = getGridHash(voxel_block_inner_grid, voxel_block_inner_dim);

      int3 node_block_grid;
      node_block_grid.x = voxel_block_grid.x / dim_ratio;
      node_block_grid.y = voxel_block_grid.y / dim_ratio;
      node_block_grid.z = voxel_block_grid.z / dim_ratio;

      int3 pass_grid;
      pass_grid.x = voxel_block_grid.x % dim_ratio;
      pass_grid.y = voxel_block_grid.y % dim_ratio;
      pass_grid.z = voxel_block_grid.z % dim_ratio;

      int pass = getGridHash(pass_grid, dim_ratio);
      int pass_stride = pass * voxel_block_inner_hash_size;

      int voxel_blend_table_entry = pass_stride + voxel_block_inner_hash;

      // linear blending
      float blend_weight[4 * VOXEL_BLENDING_QUAD_STRIDE];
      uint blend_node_id[4 * VOXEL_BLENDING_QUAD_STRIDE];
      Affine3d blend_transform(0.0f);

      int blend_count = getVoxelBlending(node_block_dim,
                                         node_block_grid,
                                         voxel_blend_table_entry,
                                         blend_weight,
                                         blend_node_id,
                                         blend_transform);

      float4 warped_mesh_vertex = blend_transform * ref_mesh_vertex;
      float4 warped_mesh_normal = blend_transform * ref_mesh_normal;

      warped_surfel_vertex += warped_mesh_vertex;
      warped_surfel_normal += warped_mesh_normal;

      warped_mesh_vertex_array.at(3 * i + j) = warped_mesh_vertex - volume_center_offset;
      warped_mesh_normal_array.at(3 * i + j) = warped_mesh_normal;
    } // j

    warped_surfel_vertex /= 3.0f;
    warped_surfel_normal /= 3.0f;

    warped_surfel_vertex_array.at(i) = warped_surfel_vertex;
    warped_surfel_normal_array.at(i) = warped_surfel_normal;

    // set active surfel flag
    float3 warped_vertex = make_float3(transform * warped_surfel_vertex);
    float3 warped_normal = make_float3(transform * warped_surfel_normal);

    bool valid = true;

    // is normal valid?
    if (isnan(warped_normal.x) || isnan(warped_normal.y) || isnan(warped_normal.z))
      valid = false;

    int2 map_coo;
    map_coo.x = __float2int_rn(warped_vertex.x * intrin.fx / warped_vertex.z + intrin.cx);
    map_coo.y = __float2int_rn(warped_vertex.y * intrin.fy / warped_vertex.z + intrin.cy);

    // is visible?
    bool visible = valid;
    if (map_coo.x < 0 || map_coo.x >= intrin.width || map_coo.y < 0 || map_coo.y >= intrin.height || warped_vertex.z <= 0.f)
      visible = false;

    float3 proj_vertex = (visible) ? make_float3(tex2D(vertex_true_texture, map_coo.x, map_coo.y)) : make_float3(100.f, 100.f, 100.f);
    float3 proj_normal = (visible) ? make_float3(tex2D(normal_true_texture, map_coo.x, map_coo.y)) : make_float3(0.f, 0.f, 1.f);

    if (isnan(proj_vertex.x) || isnan(proj_normal.x))
      visible = false;

    float dist = length(warped_vertex - proj_vertex);
    float n_inv = 1.f / ( length(proj_normal) * length(warped_normal) );
    float angle = acosf( fabsf( dot(proj_normal, warped_normal) * n_inv ) );
    n_inv = 1 / length(warped_normal);
    float3 view_axis = make_float3(0.f, 0.f, 1.f);
    float view_angle = acosf( fabsf( dot(view_axis, warped_normal) * n_inv ) );

    // reject outliers
    if (dist > icp_param.dist_thres || angle > icp_param.angle_thres || view_angle > icp_param.view_angle_thres)
      visible = false;

    active_flags.at(i) = (visible) ? 1 : 0;

    float residual = (visible) ? dot(proj_normal, proj_vertex - warped_vertex) : 0.f;
    float res_sqrt = sqrtf(residual * residual);

    energy_local += res_sqrt;

    // float color_r = (visible) ? float(tex2D(color_texture, 3*map_coo.x + 0, map_coo.y))/255.0f : 0.9f;
    // float color_g = (visible) ? float(tex2D(color_texture, 3*map_coo.x + 1, map_coo.y))/255.0f : 0.9f;
    // float color_b = (visible) ? float(tex2D(color_texture, 3*map_coo.x + 2, map_coo.y))/255.0f : 0.9f;
    // float color_a = (visible) ? 1.0f : 0.5f;

    // float res_sqrt_2_color = fmaxf(0.f, fminf(res_sqrt * res_sqrt_trunc_inv, 1.f));

    // float color_r = (visible) ? res_sqrt_2_color : 1.0f;
    // float color_g = (visible) ? (1.f - res_sqrt_2_color) : 0.0f;
    // float color_b = (visible) ? 0.1f : 0.1f;
    // float color_a = (visible) ? 1.0f : 1.0f;

    // #pragma unroll
    // for (int j = 0; j < 3; j += 1)
    // {
    //   ref_mesh_color_array.at(3 * i + j) = make_float4(color_r, color_g, color_b, color_a);
    // }
  } // grid-stride loop

  __syncthreads();

  energy_local = blockReduceSum(energy_local);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value, energy_local);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value = global_value;
      global_value = 0.f;
      blocks_done = 0;
    }
  }
}

void warpSurface(const IcpParamList &icp_param,
                 const Intrinsics &intrin,
                 const Affine3d &transform,
                 float volume_length,
                 float voxel_length,
                 int voxel_block_inner_dim,
                 float voxel_block_length,
                 int voxel_block_dim,
                 int node_block_dim,
                 size_t mesh_vertex_size,
                 float res_sqrt_trunc,
                 DeviceArray<float4> &ref_mesh_vertex_array,
                 DeviceArray<float4> &ref_mesh_normal_array,
                 DeviceArray<float4> &ref_mesh_color_array,
                 DeviceArray<float4> &warped_mesh_vertex_array,
                 DeviceArray<float4> &warped_mesh_normal_array,
                 DeviceArray<float4> &warped_surfel_vertex_array,
                 DeviceArray<float4> &warped_surfel_normal_array,
                 DeviceArray<uint> &active_flags,
                 float &res_sqrt)
{
  // ref_mesh_vertex_array.map();
  // ref_mesh_normal_array.map();
  // ref_mesh_color_array.map();
  // warped_mesh_vertex_array.map();
  // warped_mesh_normal_array.map();

  float4 volume_center_offset = make_float4(volume_length / 2.0f, volume_length / 2.0f, volume_length / 2.0f, 0.f);
  int dim_ratio = voxel_block_dim / node_block_dim;
  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;
  int triangle_size = mesh_vertex_size / 3;

  int block = 256;
  int grid = min(divUp(triangle_size, block), 512);

  warpSurfaceKernel<<<grid, block>>>(icp_param,
                                     intrin,
                                     transform,
                                     volume_center_offset,
                                     voxel_length,
                                     voxel_block_inner_dim,
                                     voxel_block_length,
                                     node_block_dim,
                                     dim_ratio,
                                     voxel_block_inner_hash_size,
                                     triangle_size,
                                     1.f / res_sqrt_trunc,
                                     ref_mesh_vertex_array.getHandle(),
                                     ref_mesh_normal_array.getHandle(),
                                     ref_mesh_color_array.getHandle(),
                                     warped_mesh_vertex_array.getHandle(),
                                     warped_mesh_normal_array.getHandle(),
                                     warped_surfel_vertex_array.getHandle(),
                                     warped_surfel_normal_array.getHandle(),
                                     active_flags.getHandle());

  checkCudaErrors(cudaMemcpyFromSymbol(&res_sqrt, output_value, sizeof(res_sqrt)));

  res_sqrt /= float(triangle_size);

  // ref_mesh_vertex_array.unmap();
  // ref_mesh_normal_array.unmap();
  // ref_mesh_color_array.unmap();
  // warped_mesh_vertex_array.unmap();
  // warped_mesh_normal_array.unmap();
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void combineVertexAndColorDataKernel(size_t surfel_size,
         const Affine3d transform,
         const DeviceArrayHandle<float4> surfel_vertex_data,
         const DeviceArrayHandle<float> surfel_color_data,
         DeviceArrayHandle<float> surfel_xyzrgb_data)
{
  const int stride = gridDim.x * blockDim.x;

  // grid-stride loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < surfel_size;
       i += stride)
  {
    float4 surfel_vertex = transform * surfel_vertex_data.at(i);
    float surfel_color = surfel_color_data.at(i);

    surfel_xyzrgb_data.at(8*i + 0) = surfel_vertex.x;
    surfel_xyzrgb_data.at(8*i + 1) = surfel_vertex.y;
    surfel_xyzrgb_data.at(8*i + 2) = surfel_vertex.z;
    surfel_xyzrgb_data.at(8*i + 3) = surfel_vertex.w;
    surfel_xyzrgb_data.at(8*i + 4) = surfel_color;
  }
}

void combineVertexAndColorData(size_t surfel_size,
                               const Affine3d &transform,
                               const DeviceArray<float4> &surfel_vertex_data,
                               const DeviceArray<float> &surfel_color_data,
                               DeviceArray<float> &surfel_xyzrgb_data)
{
  int block = 256;
  int grid = min(divUp(surfel_size, block), 512);

  combineVertexAndColorDataKernel<<<grid, block>>>(surfel_size,
                                                   transform,
                                                   surfel_vertex_data.getHandle(),
                                                   surfel_color_data.getHandle(),
                                                   surfel_xyzrgb_data.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void formIcpGaussNewtonOptKernel(const IcpParamList icp_param,
                                 const Intrinsics intrin,
                                 const Affine3d transform,
                                 size_t vertex_size,
                                 const DeviceArrayHandle<float4> vertex_array,
                                 const DeviceArrayHandle<float4> normal_array,
                                 DeviceArrayHandle<float> sum_buf)
{
  // init sum in each block thread
  float sum_local[27];
  #pragma unroll
  for (int i = 0; i < 24; i+=4)
  {
    *(float4*)(&sum_local[i]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  *(float3*)(&sum_local[24]) = make_float3(0.0f, 0.0f, 0.0f);

  int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < vertex_size;
       i += stride)
  {
    float4 vertex = vertex_array.at(i);
    float4 normal = normal_array.at(i);

    float4 warped_vertex = transform * vertex;
    float4 warped_normal = transform * normal;

    // Data association
    bool valid = true;
    int2 map_coo; // projection
    map_coo.x = __float2int_rn(warped_vertex.x * intrin.fx / warped_vertex.z + intrin.cx);
    map_coo.y = __float2int_rn(warped_vertex.y * intrin.fy / warped_vertex.z + intrin.cy);

    if (map_coo.x < 0 || map_coo.x >= intrin.width || map_coo.y < 0 || map_coo.y >= intrin.height || warped_vertex.z <= 0.0f)
      valid = false;

    float4 proj_vertex = (valid) ? tex2D(vertex_texture, map_coo.x, map_coo.y) : make_float4(100.f, 100.f, 100.f, 1.f);
    float4 proj_normal = (valid) ? tex2D(normal_texture, map_coo.x, map_coo.y) : make_float4(0.f, 0.f, 1.f, 0.f);

    if (isnan(proj_vertex.x) || isnan(proj_normal.x))
      valid = false;

    float dist = length(warped_vertex - proj_vertex);
    float n_inv = 1.f / ( length(proj_normal) * length(warped_normal) );
    float angle = acosf( fabsf( dot(proj_normal, warped_normal) * n_inv ) );
    n_inv = 1 / length(warped_normal);
    float4 view_axis = make_float4(0.f, 0.f, 1.f, 0.f);
    float view_angle = acosf( fabsf( dot(view_axis, warped_normal) * n_inv ) );

    if (dist > icp_param.dist_thres || angle > icp_param.angle_thres || view_angle > icp_param.view_angle_thres)
      valid = false;

    float J_f[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float3 J_w = cross(make_float3(warped_vertex), make_float3(proj_normal));
    float3 J_v = make_float3(proj_normal);
    float f = dot(proj_normal, warped_vertex - proj_vertex);

    if (valid)
    {
      *(float3*)&J_f[0] = J_w;
      *(float3*)&J_f[3] = J_v;
      J_f[6] = -f;
    }

    int shift = 0;
    for (int k = 0; k < 6; ++k)
    {
      #pragma unroll
      for (int j = k; j < 7; ++j)
      {
        sum_local[shift++] += J_f[k] * J_f[j];
      }
    }
  } // grid-stride loop

  __syncthreads();

  for (int i = 0; i < 27; ++i)
  {
    // perform block-level reduction
    sum_local[i] = blockReduceSum(sum_local[i]);

    // reduce block sum through atomic add
    if (threadIdx.x == 0)
    {
      atomicAdd(&sum_buf.at(i), sum_local[i]);
    }
  }
}

void formIcpGaussNewtonOpt(const IcpParamList &icp_param,
                           const Intrinsics &intrin,
                           const Affine3d &transform,
                           size_t vertex_size,
                           const DeviceArray<float4> &vertex_array,
                           const DeviceArray<float4> &normal_array,
                           DeviceArray<float> &sum_buf)
{
  checkCudaErrors(cudaMemset(reinterpret_cast<void *>(sum_buf.getDevicePtr()), 0, 27 * sizeof(float)));

  int block = 256;
  int grid = min(divUp(vertex_size, block), 512);
  formIcpGaussNewtonOptKernel<<<grid, block>>>(icp_param,
                                               intrin,
                                               transform,
                                               vertex_size,
                                               vertex_array.getHandle(),
                                               normal_array.getHandle(),
                                               sum_buf.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
Affine3d getInvSE3(const Affine3d &m)
{
  float3 r[3];
  m.getRotate(r);

  float3 t;
  m.getTranslate(t);

  Affine3d m_inv(m);

  m_inv.at(1, 0) = r[1].x;
  m_inv.at(0, 1) = r[0].y;

  m_inv.at(2, 0) = r[2].x;
  m_inv.at(0, 2) = r[0].z;

  m_inv.at(2, 1) = r[2].y;
  m_inv.at(1, 2) = r[1].z;


  m_inv.at(0, 3) = -dot(r[0], t);
  m_inv.at(1, 3) = -dot(r[1], t);
  m_inv.at(2, 3) = -dot(r[2], t);

  return m_inv;
}

__device__ __forceinline__
int getFitTermBlending(int node_block_dim,
                       int3 node_block_grid,
                       int voxel_blend_table_entry,
                       float *blend_weight,
                       uint *blend_id,
                       Affine3d &blend_transform)
{
  int blend_count = 0;

  int voxel_blend_table_entry_quad = voxel_blend_table_entry * VOXEL_BLENDING_MAX_QUAD_STRIDE;

  for (int quad = 0; quad < VOXEL_BLENDING_MAX_QUAD_STRIDE; quad++)
  {
    float4 weight_entry_quad = tex1Dfetch(blend_weight_table_texture,
                                          voxel_blend_table_entry_quad + quad);
    float *weight_entry = (float *)&weight_entry_quad;

    int4 code_entry_quad = tex1Dfetch(blend_code_table_texture,
                                      voxel_blend_table_entry_quad + quad);
    int *code_entry = (int *)&code_entry_quad;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
      if (blend_count == 4 * VOXEL_BLENDING_QUAD_STRIDE)
        break;

      float weight_i = weight_entry[i];

      int code_i = code_entry[i];
      int code_i_x = ((code_i & 0x00FF0000) >> 16) - 1;
      int code_i_y = ((code_i & 0x0000FF00) >> 8) - 1;
      int code_i_z = (code_i & 0x000000FF) - 1;

      int3 node_block_grid_i = node_block_grid + make_int3(int(code_i_x), int(code_i_y), int(code_i_z));

      if (node_block_grid_i.x >= 0 &&
          node_block_grid_i.y >= 0 &&
          node_block_grid_i.z >= 0 &&
          node_block_grid_i.x < node_block_dim && 
          node_block_grid_i.y < node_block_dim && 
          node_block_grid_i.z < node_block_dim)
      {
        int node_block_hash_i = getGridHash(node_block_grid_i, node_block_dim);
        HashEntryType node_hash_entry_i = tex1Dfetch(block_hash_table_texture, node_block_hash_i);
        uint node_id_i = node_hash_entry_i.x;
        TransformType transform_col[4];
        Affine3d transform_i;

        if (node_id_i != 0xFFFFFFFF)
        {
          transform_col[0] = tex1Dfetch(transform_texture, 4 * node_id_i + 0);
          transform_col[1] = tex1Dfetch(transform_texture, 4 * node_id_i + 1);
          transform_col[2] = tex1Dfetch(transform_texture, 4 * node_id_i + 2);
          transform_col[3] = tex1Dfetch(transform_texture, 4 * node_id_i + 3);
          transform_i.setValue(transform_col);

          blend_transform += (transform_i * weight_i);

          blend_weight[blend_count] = weight_i;
          blend_id[blend_count] = node_id_i;

          blend_count++;
        }
      }
    } // i
  } // quad

  // normalization
  float weight_sum = 0.f;

  #pragma unroll
  for (int k = 0; k < blend_count; k++)
    weight_sum += blend_weight[k];

  if (weight_sum > 5e-3)
  {
    blend_transform *= (1.f / weight_sum);

    #pragma unroll
    for (int k = 0; k < blend_count; k++)
      blend_weight[k] /= weight_sum;
  }
  else
  {
    blend_count = 0;
    blend_transform.makeIdentity();
  }

  return blend_count;
}

__global__
void initFitTermKernel(const IcpParamList icp_param,
                       const Intrinsics intrin,
                       const Affine3d transform,
                       size_t vertex_size,
                       float voxel_length,
                       int voxel_block_inner_dim,
                       float voxel_block_length,
                       int node_block_dim,
                       int dim_ratio,
                       int voxel_block_inner_hash_size,
                       const DetParamList det_param,
                       FitTermDataList fit_term,
                       DetBlockDataList det_block)
{
  const int stride = gridDim.x * blockDim.x;

  const float term_weight = det_param.w_sqrt * det_param.w_sqrt;

  float energy_local = 0.f;

  // grid-stride loop
  for (uint i = threadIdx.x + blockDim.x * blockIdx.x;
       i < vertex_size;
       i += stride)
  {
    float4 vertex = tex1Dfetch(mesh_vert_texture, i);
    float4 normal = tex1Dfetch(mesh_norm_texture, i);

    int3 voxel_block_grid;
    voxel_block_grid.x = __float2int_rz(vertex.x / voxel_block_length);
    voxel_block_grid.y = __float2int_rz(vertex.y / voxel_block_length);
    voxel_block_grid.z = __float2int_rz(vertex.z / voxel_block_length);

    float3 voxel_block_origin = getGridCoo(voxel_block_grid, voxel_block_length, 0.f);

    float3 voxel_block_inner_coo = make_float3(vertex) - voxel_block_origin;

    int3 voxel_block_inner_grid;
    voxel_block_inner_grid.x = __float2int_rz(voxel_block_inner_coo.x / voxel_length);
    voxel_block_inner_grid.y = __float2int_rz(voxel_block_inner_coo.y / voxel_length);
    voxel_block_inner_grid.z = __float2int_rz(voxel_block_inner_coo.z / voxel_length);

    int voxel_block_inner_hash = getGridHash(voxel_block_inner_grid, voxel_block_inner_dim);

    int3 node_block_grid;
    node_block_grid.x = voxel_block_grid.x / dim_ratio;
    node_block_grid.y = voxel_block_grid.y / dim_ratio;
    node_block_grid.z = voxel_block_grid.z / dim_ratio;

    int3 pass_grid;
    pass_grid.x = voxel_block_grid.x % dim_ratio;
    pass_grid.y = voxel_block_grid.y % dim_ratio;
    pass_grid.z = voxel_block_grid.z % dim_ratio;

    int pass = getGridHash(pass_grid, dim_ratio);
    int pass_stride = pass * voxel_block_inner_hash_size;

    int voxel_blend_table_entry = pass_stride + voxel_block_inner_hash;

    // Step 1: Skinning based on linear blending (LB)
    float blend_weight[4 * VOXEL_BLENDING_QUAD_STRIDE];
    uint blend_id[4 * VOXEL_BLENDING_QUAD_STRIDE];

    Affine3d blend_transform(0.f);

    int blend_count = getFitTermBlending(node_block_dim,
                                         node_block_grid,
                                         voxel_blend_table_entry,
                                         blend_weight,
                                         blend_id,
                                         blend_transform);

    float3 warped_vertex = make_float3(transform * (blend_transform * vertex));
    float3 warped_normal = make_float3(transform * (blend_transform * normal));

    // Step 2: Data association
    int2 map_coo;
    map_coo.x = __float2int_rn(warped_vertex.x * intrin.fx / warped_vertex.z + intrin.cx);
    map_coo.y = __float2int_rn(warped_vertex.y * intrin.fy / warped_vertex.z + intrin.cy);

    // is skinned?
    bool valid = (blend_count > 0);

    // is normal valid?
    if (isnan(normal.x) || isnan(normal.y) || isnan(normal.z))
      valid = false;

    // is visible?
    bool visible = valid;
    if (map_coo.x < 0 || map_coo.x >= intrin.width || map_coo.y < 0 || map_coo.y >= intrin.height || warped_vertex.z <= 0.f)
      visible = false;

    float3 proj_vertex = (visible) ? make_float3(tex2D(vertex_texture, map_coo.x, map_coo.y)) : make_float3(100.f, 100.f, 100.f);
    float3 proj_normal = (visible) ? make_float3(tex2D(normal_texture, map_coo.x, map_coo.y)) : make_float3(0.f, 0.f, 1.f);

    // Affine3d transform_inv = getInvSE3(transform * blend_transform);
    // float4 warped_proj_vertex = transform_inv * make_float4(proj_vertex, 1.f);
    // float4 warped_proj_normal = transform_inv * make_float4(proj_normal, 0.f);

    if (isnan(proj_vertex.x) || isnan(proj_normal.x))
      visible = false;

    float dist = length(warped_vertex - proj_vertex);
    float n_inv = 1.f / ( length(proj_normal) * length(warped_normal) );
    float angle = acosf( fabsf( dot(proj_normal, warped_normal) * n_inv ) );
    n_inv = 1 / length(warped_normal);
    float3 view_axis = make_float3(0.f, 0.f, 1.f);
    float view_angle = acosf( fabsf( dot(view_axis, warped_normal) * n_inv ) );

    // reject outliers
    if (dist > icp_param.dist_thres || angle > icp_param.angle_thres || view_angle > icp_param.view_angle_thres)
      visible = false;

    // Step 3: Computing term element data
    // Note: we use negative residual here
    float residual = (visible) ? dot(proj_normal, proj_vertex - warped_vertex) : 0.f;
    // float residual = (visible) ? dot(normal, warped_proj_vertex - vertex) : 0.f;
    // float residual = (visible) ? dot(warped_proj_normal, warped_proj_vertex - vertex) : 0.f;

    energy_local += (term_weight * residual * residual);

    float3 rotate[3];
    transform.getRotate(rotate);

    float4 elem_data;

    elem_data.x = dot(proj_normal, rotate[0]);
    elem_data.y = dot(proj_normal, rotate[1]);
    elem_data.z = dot(proj_normal, rotate[2]);
    // elem_data.x = warped_proj_normal.x;
    // elem_data.y = warped_proj_normal.y;
    // elem_data.z = warped_proj_normal.z;
    elem_data.w = (visible) ? residual : quiet_nanf();

    uint elem_entry_base = det_param.elem_entry_start + i;
    fit_term.elem_data[elem_entry_base] = elem_data;

    // Step 4: Write result
    uint vec_block_stride = 0;
    uint mat_block_stride = 0;

    for (uint a = 0; a < det_param.vec_block_elem_stride; a++)
    {
      uint elem_entry = vec_block_stride + elem_entry_base;
      uint vec_block_entry = det_param.vec_block_entry_start + elem_entry;

      bool valid_a = (visible && a < blend_count);
      // bool valid_a = (valid && a < blend_count);

      fit_term.elem_ids[elem_entry] = (valid_a) ? blend_id[a] : 0xFFFFFFFF;
      fit_term.elem_weights[elem_entry] = (valid_a) ? blend_weight[a] : 0.f;

      uint vec_block_elem_entry_code = (elem_entry_base << 10) + (a << 6) + det_param.term_num;

      det_block.vec_block_elem_ids[vec_block_entry] = (valid_a) ? blend_id[a] : 0xFFFFFFFF;
      det_block.vec_block_elem_entry_codes[vec_block_entry] = vec_block_elem_entry_code;

      for (uint b = a + 1; b < det_param.vec_block_elem_stride; b++)
      {
        uint mat_block_entry = det_param.mat_block_entry_start + mat_block_stride + elem_entry_base;

        bool valid_a_b = (valid_a && b < blend_count);

        // TODO: only for test
        // bool valid_a_b = false;

        uint p = a, q = b;

        if (blend_id[a] > blend_id[b])
        {
          p = b;
          q = a;
        }

        uint mat_block_elem_id = (blend_id[p] << 16) + blend_id[q];
        uint mat_block_elem_entry_code = (elem_entry_base << 10) + (p << 6) + (q << 2) + det_param.term_num;

        det_block.mat_block_elem_ids[mat_block_entry] = (valid_a_b) ? mat_block_elem_id : 0xFFFFFFFF;
        det_block.mat_block_elem_entry_codes[mat_block_entry] = mat_block_elem_entry_code;

        mat_block_stride += det_param.elem_entry_size;
      } // b

      vec_block_stride += det_param.elem_entry_size;
    } // a
  } // i

  __syncthreads();

  energy_local = blockReduceSum(energy_local);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value, energy_local);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value = global_value;
      global_value = 0.f;
      blocks_done = 0;
    }
  }
}

float initFitTerm(const IcpParamList &icp_param,
                  const Intrinsics &intrin,
                  const Affine3d &transform,
                  size_t vertex_size,
                  float voxel_length,
                  int voxel_block_inner_dim,
                  float voxel_block_length,
                  int voxel_block_dim,
                  int node_block_dim,
                  const DetParamList &det_param,
                  FitTermDataList &fit_term,
                  DetBlockDataList &det_block)
{
  int dim_ratio = voxel_block_dim / node_block_dim;
  int voxel_block_inner_hash_size = voxel_block_inner_dim * voxel_block_inner_dim * voxel_block_inner_dim;

  int block = 256;
  int grid = min(divUp(vertex_size, block), 512);

  initFitTermKernel<<<grid, block>>>(icp_param,
                                     intrin,
                                     transform,
                                     vertex_size,
                                     voxel_length,
                                     voxel_block_inner_dim,
                                     voxel_block_length,
                                     node_block_dim,
                                     dim_ratio,
                                     voxel_block_inner_hash_size,
                                     det_param,
                                     fit_term,
                                     det_block);

  checkCudaErrors(cudaGetLastError());

  float fit_energy;
  checkCudaErrors(cudaMemcpyFromSymbol(&fit_energy, output_value, sizeof(fit_energy)));

  return fit_energy;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int loadFitTermBlending(uint elem_entry_base,
                        uint elem_entry_size,
                        uint vec_block_elem_stride,
                        uint *elem_ids,
                        float *elem_weights,
                        uint *blend_id,
                        float *blend_weight,
                        Affine3d &blend_transform)
{
  int blend_count = 0;

  uint vec_block_stride = 0;
  for (uint a = 0; a < vec_block_elem_stride; a++)
  {
    uint elem_entry = vec_block_stride + elem_entry_base;

    blend_id[a] = elem_ids[elem_entry];
    blend_weight[a] = elem_weights[elem_entry];

    TransformType transform_col[4];
    Affine3d transform_i;

    if (blend_id[a] != 0xFFFFFFFF)
    {
      transform_col[0] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 0);
      transform_col[1] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 1);
      transform_col[2] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 2);
      transform_col[3] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 3);
      transform_i.setValue(transform_col);

      blend_transform += (transform_i * blend_weight[a]);

      blend_count++;
    }

    vec_block_stride += elem_entry_size;
  }

  if (blend_count == 0)
    blend_transform.makeIdentity();

  return blend_count;
}


__global__
void updateFitTermKernel(const IcpParamList icp_param,
                         const Intrinsics intrin,
                         const Affine3d transform,
                         size_t vertex_size,
                         const DetParamList det_param,
                         FitTermDataList fit_term)
{
  const int stride = gridDim.x * blockDim.x;

  float energy_local = 0.f;

  // grid-stride loop
  for (uint i = threadIdx.x + blockDim.x * blockIdx.x;
       i < vertex_size;
       i += stride)
  {
    float4 vertex = tex1Dfetch(mesh_vert_texture, i);
    float4 normal = tex1Dfetch(mesh_norm_texture, i);

    float blend_weight[4 * VOXEL_BLENDING_QUAD_STRIDE];
    uint blend_id[4 * VOXEL_BLENDING_QUAD_STRIDE];

    Affine3d blend_transform(0.f);

    uint elem_entry_base = det_param.elem_entry_start + i;

    int blend_count = loadFitTermBlending(elem_entry_base,
                                          det_param.elem_entry_size,
                                          det_param.vec_block_elem_stride,
                                          fit_term.elem_ids,
                                          fit_term.elem_weights,
                                          blend_id,
                                          blend_weight,
                                          blend_transform);

    float3 warped_vertex = make_float3(transform * (blend_transform * vertex));
    float3 warped_normal = make_float3(transform * (blend_transform * normal));

    // Step 2: Data association
    int2 map_coo;
    map_coo.x = __float2int_rn(warped_vertex.x * intrin.fx / warped_vertex.z + intrin.cx);
    map_coo.y = __float2int_rn(warped_vertex.y * intrin.fy / warped_vertex.z + intrin.cy);

    // is skinned?
    bool valid = (blend_count > 0);

    // is normal valid?
    if (isnan(normal.x) || isnan(normal.y) || isnan(normal.z))
      valid = false;

    // is visible?
    bool visible = valid;
    if (map_coo.x < 0 || map_coo.x >= intrin.width || map_coo.y < 0 || map_coo.y >= intrin.height || warped_vertex.z <= 0.f)
      visible = false;

    float3 proj_vertex =(visible) ? make_float3(tex2D(vertex_texture, map_coo.x, map_coo.y)) : make_float3(100.f, 100.f, 100.f);
    float3 proj_normal =(visible) ? make_float3(tex2D(normal_texture, map_coo.x, map_coo.y)) : make_float3(0.f, 0.f, 1.f);

    // Affine3d transform_inv = getInvSE3(transform * blend_transform);
    // float4 warped_proj_vertex = transform_inv * make_float4(proj_vertex, 1.f);
    // float4 warped_proj_normal = transform_inv * make_float4(proj_normal, 0.f);

    if (isnan(proj_vertex.x) || isnan(proj_normal.x))
      visible = false;

    float dist = length(warped_vertex - proj_vertex);
    float n_inv = 1.f / ( length(proj_normal) * length(warped_normal) );
    float angle = acosf( fabsf( dot(proj_normal, warped_normal) * n_inv ) );
    n_inv = 1 / length(warped_normal);
    float3 view_axis = make_float3(0.f, 0.f, 1.f);
    float view_angle = acosf( fabsf( dot(view_axis, warped_normal) * n_inv ) );

    // reject outliers
    if (dist > icp_param.dist_thres || angle > icp_param.angle_thres || view_angle > icp_param.view_angle_thres)
      visible = false;

    // Step 3: Computing term element data
    // Note: we use negative residual here
    float residual = (visible) ? dot(proj_normal, proj_vertex - warped_vertex) : 0.f;
    // float residual = (visible) ? dot(normal, warped_proj_vertex - vertex) : 0.f;
    // float residual = (visible) ? dot(warped_proj_normal, warped_proj_vertex - vertex) : 0.f;

    energy_local += (det_param.w_sqrt * det_param.w_sqrt * residual * residual);

    float3 rotate[3];
    transform.getRotate(rotate);

    float4 elem_data;

    elem_data.x = dot(proj_normal, rotate[0]);
    elem_data.y = dot(proj_normal, rotate[1]);
    elem_data.z = dot(proj_normal, rotate[2]);
    // elem_data.x = warped_proj_normal.x;
    // elem_data.y = warped_proj_normal.y;
    // elem_data.z = warped_proj_normal.z;
    elem_data.w = (visible) ? residual : quiet_nanf();

    fit_term.elem_data[elem_entry_base] = elem_data;
  } // grid-stride loop

  __syncthreads();

  energy_local = blockReduceSum(energy_local);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value, energy_local);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value = global_value;
      global_value = 0.f;
      blocks_done = 0;
    }
  }
}

float updateFitTerm(const IcpParamList &icp_param,
                    const Intrinsics &intrin,
                    const Affine3d &transform,
                    size_t vertex_size,
                    const DetParamList &det_param,
                    FitTermDataList &fit_term)
{
  int block = 256;
  int grid = min(divUp(vertex_size, block), 512);

  updateFitTermKernel<<<grid, block>>>(icp_param,
                                       intrin,
                                       transform,
                                       vertex_size,
                                       det_param,
                                       fit_term);

  checkCudaErrors(cudaGetLastError());

  float fit_energy;
  checkCudaErrors(cudaMemcpyFromSymbol(&fit_energy, output_value, sizeof(fit_energy)));

  return fit_energy;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int getRegTermBlending(int node_block_dim,
                       int3 node_block_grid,
                       int node_blend_table_entry,
                       float4 node,
                       float *blend_weight,
                       uint *blend_id,
                       float3 *blend_node)
{
  int blend_count = 0;

  int node_blend_table_entry_quad = node_blend_table_entry * NODE_BLENDING_MAX_QUAD_STRIDE;

  for (int quad = 0; quad < NODE_BLENDING_MAX_QUAD_STRIDE; quad++)
  {
    float4 weight_entry_quad = tex1Dfetch(blend_weight_table_texture,
                                          node_blend_table_entry_quad + quad);
    float *weight_entry = (float *)&weight_entry_quad;

    int4 code_entry_quad = tex1Dfetch(blend_code_table_texture,
                                      node_blend_table_entry_quad + quad);

    int *code_entry = (int *)&code_entry_quad;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
      if (blend_count == (4 * NODE_BLENDING_QUAD_STRIDE + 1))
        break;

      float weight_i = weight_entry[i];

      int code_i = code_entry[i];
      int code_i_x = ((code_i & 0x00FF0000) >> 16) - 1;
      int code_i_y = ((code_i & 0x0000FF00) >> 8) - 1;
      int code_i_z = (code_i & 0x000000FF) - 1;

      int3 node_block_grid_i = node_block_grid + make_int3(int(code_i_x), int(code_i_y), int(code_i_z));

      if (node_block_grid_i.x >= 0 &&
          node_block_grid_i.y >= 0 &&
          node_block_grid_i.z >= 0 &&
          node_block_grid_i.x < node_block_dim &&
          node_block_grid_i.y < node_block_dim &&
          node_block_grid_i.z < node_block_dim)
      {
        int node_block_hash_i = getGridHash(node_block_grid_i, node_block_dim);
        HashEntryType node_hash_entry_i = tex1Dfetch(block_hash_table_texture, node_block_hash_i);
        uint node_id_i = node_hash_entry_i.x;
        TransformType transform_col[4];
        Affine3d transform_i;

        if (node_id_i != 0xFFFFFFFF)
        {
          transform_col[0] = tex1Dfetch(transform_texture, 4 * node_id_i + 0);
          transform_col[1] = tex1Dfetch(transform_texture, 4 * node_id_i + 1);
          transform_col[2] = tex1Dfetch(transform_texture, 4 * node_id_i + 2);
          transform_col[3] = tex1Dfetch(transform_texture, 4 * node_id_i + 3);
          transform_i.setValue(transform_col);

          blend_weight[blend_count] = weight_i;
          blend_id[blend_count] = node_id_i;
          blend_node[blend_count] = make_float3(transform_i * node);

          blend_count++;
        }
      }
    } // i
  } // quad

  // normalization
  float weight_sum = 0.f;

  #pragma unroll
  for (int k = 1; k < blend_count; k++)
    weight_sum += blend_weight[k];

  if (weight_sum > 5e-3)
  {
    #pragma unroll
    for (int k = 1; k < blend_count; k++)
      blend_weight[k] /= weight_sum;
  }

  return blend_count;
}

__global__
void initRegTermKernel(float w_level,
                       int dim_ratio,
                       float node_block_length,
                       int node_block_dim,
                       size_t node_block_size,
                       const DeviceArrayHandle<uint> node_block_hash_array,
                       /*const DeviceArrayHandle<HashEntryType> node_block_hash_table,*/
                       const DetParamList det_param,
                       RegTermDataList reg_term,
                       DetBlockDataList det_block)
{
  const int stride = gridDim.x * blockDim.x;

  const float term_weight = det_param.w_sqrt * det_param.w_sqrt;

  const float huber_scale = det_param.w_huber * det_param.w_huber;

  float energy_local = 0.f;

  // grid-stride loop
  for (uint i = threadIdx.x + blockDim.x * blockIdx.x;
       i < node_block_size;
       i += stride)
  {
    float blend_weight[4 * NODE_BLENDING_QUAD_STRIDE + 1];
    uint blend_id[4 * NODE_BLENDING_QUAD_STRIDE + 1];
    float3 blend_node[4 * NODE_BLENDING_QUAD_STRIDE + 1];

    // Step 1: Compute central node
    uint node_block_hash = node_block_hash_array.at(i);

    int3 node_block_grid = getGrid(node_block_hash, node_block_dim);
    float4 node = make_float4(getGridCoo(node_block_grid, node_block_length, 0.5f), 1.f);

    int node_blend_table_entry = 0;

    int blend_count = getRegTermBlending(node_block_dim,
                                         node_block_grid,
                                         node_blend_table_entry,
                                         node,
                                         blend_weight,
                                         blend_id,
                                         blend_node);

    bool valid = (blend_count > 1);

    float4 elem_data;

    elem_data.x = node.x;
    elem_data.y = node.y;
    elem_data.z = node.z;
    elem_data.w = w_level;

    // Step 3: Computing term element data
    uint elem_entry_base = det_param.elem_entry_start + i;
    reg_term.elem_data[elem_entry_base] = elem_data;

    // Step 4: Write result
    uint vec_block_stride = 0;
    uint mat_block_stride = 0;

    for (uint a = 1; a < (det_param.vec_block_elem_stride / 2 + 1); a++)
    {
      bool valid_a = (valid && a < blend_count);

      float3 residual = (valid_a) ? (blend_node[0] - blend_node[a]) : make_float3(0.f, 0.f, 0.f);

      // Huber loss function
      float huber_energy = dot(residual, residual) / huber_scale;
      float rho_d_0 = (huber_energy <= 1.f) ? huber_energy : (2.f * sqrtf(huber_energy) - 1.f);
      rho_d_0 *= huber_scale;

      energy_local += w_level * term_weight * rho_d_0;
      // energy_local += w_level * term_weight * blend_weight[a] * rho_d_0;

      // energy_local += w_level * term_weight * dot(residual, residual);

      uint elem_entry = vec_block_stride + elem_entry_base;
      uint vec_block_entry = det_param.vec_block_entry_start + vec_block_stride + elem_entry_base;
      uint vec_block_elem_entry_code = (elem_entry_base << 10) + ((2*a-2) << 6) + ((2*a-1) << 2) + det_param.term_num;

      reg_term.elem_weights[elem_entry] = (valid_a) ? blend_weight[a] : 0.f;
      reg_term.elem_ids[elem_entry] = (valid_a) ? blend_id[0] : 0xFFFFFFFF;
      det_block.vec_block_elem_ids[vec_block_entry] = (valid_a) ? blend_id[0] : 0xFFFFFFFF;
      det_block.vec_block_elem_entry_codes[vec_block_entry] = vec_block_elem_entry_code;

      vec_block_stride += det_param.elem_entry_size;

      elem_entry = vec_block_stride + elem_entry_base;
      vec_block_entry = det_param.vec_block_entry_start + vec_block_stride + elem_entry_base;
      vec_block_elem_entry_code = (elem_entry_base << 10) + ((2*a-1) << 6) + ((2*a-2) << 2) + det_param.term_num;

      reg_term.elem_weights[elem_entry] = (valid_a) ? blend_weight[a] : 0.f;
      reg_term.elem_ids[elem_entry] = (valid_a) ? blend_id[a] : 0xFFFFFFFF;
      det_block.vec_block_elem_ids[vec_block_entry] = (valid_a) ? blend_id[a] : 0xFFFFFFFF;
      det_block.vec_block_elem_entry_codes[vec_block_entry] = vec_block_elem_entry_code;

      vec_block_stride += det_param.elem_entry_size;

      // mat block
      uint mat_block_entry = det_param.mat_block_entry_start + mat_block_stride + elem_entry_base;

      uint p = 0, q = a, pp = 2*a-2, qq = 2*a-1;

      if (blend_id[0] > blend_id[a])
      {
        p = a;
        q = 0;
        pp = 2*a-1;
        qq = 2*a-2;
      }

      uint mat_block_elem_id = (blend_id[p] << 16) + blend_id[q];
      uint mat_block_elem_entry_code = (elem_entry_base << 10) + (pp << 6) + (qq << 2) + det_param.term_num;

      det_block.mat_block_elem_ids[mat_block_entry] = (valid_a) ? mat_block_elem_id : 0xFFFFFFFF;
      det_block.mat_block_elem_entry_codes[mat_block_entry] = mat_block_elem_entry_code;

      mat_block_stride += det_param.elem_entry_size;
    }
  } // grid-stride loop

  __syncthreads();

  energy_local = blockReduceSum(energy_local);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value, energy_local);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value = global_value;
      global_value = 0.f;
      blocks_done = 0;
    }
  }
}

float initRegTerm(float w_level,
                  int dim_ratio,
                  float node_block_length,
                  int node_block_dim,
                  size_t node_block_size,
                  const DeviceArray<uint> &node_block_hash_array,
                  /*const DeviceArray<HashEntryType> &node_block_hash_table,*/
                  const DetParamList &det_param,
                  RegTermDataList &reg_term,
                  DetBlockDataList &det_block)
{
  int block = 256;
  int grid = min(divUp(node_block_size, block), 512);

  initRegTermKernel<<<grid, block>>>(w_level,
                                     dim_ratio,
                                     node_block_length,
                                     node_block_dim,
                                     node_block_size,
                                     node_block_hash_array.getHandle(),
                                     /*node_block_hash_table.getHandle(),*/
                                     det_param,
                                     reg_term,
                                     det_block);

  checkCudaErrors(cudaGetLastError());

  float reg_energy;
  checkCudaErrors(cudaMemcpyFromSymbol(&reg_energy, output_value, sizeof(reg_energy)));

  return reg_energy;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
int loadRegTermBlending(uint elem_entry_base,
                        uint elem_entry_size,
                        uint vec_block_elem_stride,
                        float4 node,
                        float *elem_weights,
                        uint  *elem_ids,
                        float *blend_weight,
                        uint  *blend_id,
                        float3 *blend_node)
{
  int blend_count = 0;

  blend_weight[0] = 0.f;
  blend_id[0] = elem_ids[elem_entry_base];

  if (blend_id[0] != 0xFFFFFFFF)
  {
    TransformType transform_col[4];
    Affine3d transform;

    transform_col[0] = tex1Dfetch(transform_texture, 4 * blend_id[0] + 0);
    transform_col[1] = tex1Dfetch(transform_texture, 4 * blend_id[0] + 1);
    transform_col[2] = tex1Dfetch(transform_texture, 4 * blend_id[0] + 2);
    transform_col[3] = tex1Dfetch(transform_texture, 4 * blend_id[0] + 3);
    transform.setValue(transform_col);

    blend_node[0] = make_float3(transform * node);

    blend_count++;
  }

  uint vec_block_stride = elem_entry_size;

  for (uint a = 1; a < (vec_block_elem_stride / 2 + 1); a++)
  {
    uint elem_entry = vec_block_stride + elem_entry_base;

    blend_weight[a] = elem_weights[elem_entry];
    blend_id[a] = elem_ids[elem_entry];

    if (blend_id[a] != 0xFFFFFFFF)
    {
      TransformType transform_col[4];
      Affine3d transform;

      transform_col[0] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 0);
      transform_col[1] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 1);
      transform_col[2] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 2);
      transform_col[3] = tex1Dfetch(transform_texture, 4 * blend_id[a] + 3);
      transform.setValue(transform_col);

      blend_node[a] = make_float3(transform * node);

      blend_count++;
    }

    vec_block_stride += 2 * elem_entry_size;
  }

  return blend_count;
}

__global__
void updateRegTermKernel(size_t node_block_size,
                         const DetParamList det_param,
                         RegTermDataList reg_term)
{
  const int stride = gridDim.x * blockDim.x;

  const float term_weight = det_param.w_sqrt * det_param.w_sqrt;

  const float huber_scale = det_param.w_huber * det_param.w_huber;

  float energy_local = 0.f;

  // grid-stride loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < node_block_size;
       i += stride)
  {
    uint elem_entry_base = det_param.elem_entry_start + i;

    float4 elem_data = reg_term.elem_data[elem_entry_base];

    float4 node = make_float4(elem_data.x, elem_data.y, elem_data.z, 1.f);

    float w_level = elem_data.w;

    float blend_weight[4 * NODE_BLENDING_QUAD_STRIDE + 1];
    uint blend_id[4 * NODE_BLENDING_QUAD_STRIDE + 1];
    float3 blend_node[4 * NODE_BLENDING_QUAD_STRIDE + 1];

    int blend_count = loadRegTermBlending(elem_entry_base,
                                          det_param.elem_entry_size,
                                          det_param.vec_block_elem_stride,
                                          node,
                                          reg_term.elem_weights,
                                          reg_term.elem_ids,
                                          blend_weight,
                                          blend_id,
                                          blend_node);

    bool valid = (blend_count > 1);

    for (uint a = 1; a < (det_param.vec_block_elem_stride / 2 + 1); a++)
    {
      bool valid_a = (valid && a < blend_count);

      float3 residual = (valid_a) ? (blend_node[0] - blend_node[a]) : make_float3(0.f, 0.f, 0.f);

      // Huber loss function
      float huber_energy = dot(residual, residual) / huber_scale;
      float rho_d_0 = (huber_energy <= 1.f) ? huber_energy : (2.f * sqrtf(huber_energy) - 1.f);
      rho_d_0 *= huber_scale;

      energy_local += w_level * term_weight * rho_d_0;
      // energy_local += w_level * term_weight * blend_weight[a] * rho_d_0;

      // energy_local += w_level * term_weight * dot(residual, residual);
    }
  } // grid-stride loop

  __syncthreads();

  energy_local = blockReduceSum(energy_local);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value, energy_local);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value = global_value;
      global_value = 0.f;
      blocks_done = 0;
    }
  }
}

float updateRegTerm(size_t node_block_size,
                    const DetParamList &det_param,
                    RegTermDataList &reg_term)
{
  int block = 256;
  int grid = min(divUp(node_block_size, block), 512);

  updateRegTermKernel<<<grid, block>>>(node_block_size,
                                       det_param,
                                       reg_term);

  checkCudaErrors(cudaGetLastError());

  float reg_energy;
  checkCudaErrors(cudaMemcpyFromSymbol(&reg_energy, output_value, sizeof(reg_energy)));

  return reg_energy;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__ __forceinline__
void initVecSum(float *vec_sum)
{
  #pragma unroll
  for (int k = 0; k < 6; k++)
    vec_sum[k] = 0.f;
}

__device__ __forceinline__
void initMatSum(float *mat_sum)
{
  #pragma unroll
  for (int k = 0; k < 36; k++)
    mat_sum[k] = 0.f;
}

__device__
void elem_entry_decode(uint code, uint &elem_entry_base, uint &a, uint &b)
{
  elem_entry_base = (code & 0xFFFFFC00) >> 10;
  a = (code & 0x000003C0) >> 6;
  b = (code & 0x0000003C) >> 2;
}

__device__
void computeFitTermJacobian(float elem_weight, float4 elem_data, float3 blend_vertex, float *jacobian)
{
  jacobian[0] = elem_weight * (elem_data.z * blend_vertex.y - elem_data.y * blend_vertex.z);
  jacobian[1] = elem_weight * (elem_data.x * blend_vertex.z - elem_data.z * blend_vertex.x);
  jacobian[2] = elem_weight * (elem_data.y * blend_vertex.x - elem_data.x * blend_vertex.y);
  jacobian[3] = elem_weight * elem_data.x;
  jacobian[4] = elem_weight * elem_data.y;
  jacobian[5] = elem_weight * elem_data.z;
}

__device__
void computeRegTermJacobian(float3 blend_node, float3 *jacobian)
{
  jacobian[0] = make_float3(0.f, -blend_node.z, blend_node.y);
  jacobian[1] = make_float3(blend_node.z, 0.f, -blend_node.x);
  jacobian[2] = make_float3(-blend_node.y, blend_node.x, 0.f);
  jacobian[3] = make_float3(1.f, 0.f, 0.f);
  jacobian[4] = make_float3(0.f, 1.f, 0.f);
  jacobian[5] = make_float3(0.f, 0.f, 1.f);
}

__device__ __forceinline__
void computeRegTermJacobian(const float3 r[3], const float3 r_inv[3], const float3 n[3], float3 jacobian[6])
{
  jacobian[0] = make_float3(dot(r_inv[0], n[0]), dot(r_inv[1], n[0]), dot(r_inv[2], n[0]));
  jacobian[1] = make_float3(dot(r_inv[0], n[1]), dot(r_inv[1], n[1]), dot(r_inv[2], n[1]));
  jacobian[2] = make_float3(dot(r_inv[0], n[2]), dot(r_inv[1], n[2]), dot(r_inv[2], n[2]));
  jacobian[3] = r[0];
  jacobian[4] = r[1];
  jacobian[5] = r[2];
}

__device__
void addFitTermVecSum(float residual, const float *jacobian, float *vec_sum)
{
  #pragma unroll
  for (int k = 0; k < 6; k++)
    vec_sum[k] += residual * jacobian[k];
}

// __device__
// void addRegTermVecSum(float w_level, float3 residual, const float3 *jacobian, float *vec_sum)
// {
//   #pragma unroll
//   for (int k = 0; k < 6; k++)
//     vec_sum[k] += w_level * dot(jacobian[k], residual);
// }

__device__
void addRegTermVecSum(float weight, float3 residual, const float3 *jacobian, float *jacob_res_dot, float *vec_sum)
{
  #pragma unroll
  for (int k = 0; k < 6; k++)
  {
    float dot_result = dot(jacobian[k], residual);
    jacob_res_dot[k] = dot_result;
    vec_sum[k] += weight * dot_result;
  }
}

__device__
void addFitTermMatDiagSum(const float *jacobian, float *mat_sum)
{
  int shift = 0;
  for (int row = 0; row < 6; row++)
  {
    #pragma unroll
    for (int col = 0; col < 6; col++)
    {
      mat_sum[shift++] += jacobian[row] * jacobian[col];
    }
  }
}

// __device__
// void addRegTermMatDiagSum(float w_level, const float3 *jacobian, float *mat_sum)
// {
//   int shift = 0;
//   for (int row = 0; row < 6; row++)
//   {
//     #pragma unroll
//     for (int col = 0; col < 6; col++)
//     {
//       mat_sum[shift++] += w_level * dot(jacobian[row], jacobian[col]);
//     }
//   }
// }

__device__
void addRegTermMatDiagSum(float weight, float rho_d_1, float rho_d_2, const float3 *jacobian, const float *jacob_res_dot, float *mat_sum)
{
  int shift = 0;
  for (int row = 0; row < 6; row++)
  {
    #pragma unroll
    for (int col = 0; col < 6; col++)
    {
      float result = rho_d_1 * dot(jacobian[row], jacobian[col]);
      result += 2.f * rho_d_2 * jacob_res_dot[row] * jacob_res_dot[col];

      mat_sum[shift++] += weight * result;
    }
  }
}

__device__
void addFitTermMat(const float *jacob_a, const float *jacob_b, float *mat_sum)
{
  int shift = 0;
  for (int row = 0; row < 6; row++)
  {
    #pragma unroll
    for (int col = 0; col < 6; col++)
    {
      mat_sum[shift++] += jacob_a[row] * jacob_b[col];
    }
  }
}

// __device__
// void addRegTermMat(float w_level, const float3 *jacob_a, const float3 *jacob_b, float *mat_sum)
// {
//   int shift = 0;
//   for (int row = 0; row < 6; row++)
//   {
//     #pragma unroll
//     for (int col = 0; col < 6; col++)
//     {
//       mat_sum[shift++] += w_level * dot(jacob_a[row], jacob_b[col]);
//     }
//   }
// }

__device__
void addRegTermMat(float weight, float rho_d_1, float rho_d_2, const float3 *jacob_a, const float3 *jacob_b, const float *jacob_res_dot_a, const float *jacob_res_dot_b, float *mat_sum)
{
  int shift = 0;
  for (int row = 0; row < 6; row++)
  {
    #pragma unroll
    for (int col = 0; col < 6; col++)
    {
      float result = rho_d_1 * dot(jacob_a[row], jacob_b[col]);
      result += 2.f * rho_d_2 * jacob_res_dot_a[row] * jacob_res_dot_b[col];

      mat_sum[shift++] += weight * result;
    }
  }
}

__device__
void writeVecSum(float weight, const float *vec_sum, float *dst)
{
  #pragma unroll
  for (int k = 0; k < 6; k++)
    dst[k] = weight * vec_sum[k];
}

__device__
void writeVecSumInc(float weight, const float *vec_sum, float *dst)
{
  #pragma unroll
  for (int k = 0; k < 6; k++)
    dst[k] += weight * vec_sum[k];
}

__device__
void writeMatDiagSum(float weight, const float *mat_sum, float *dst)
{
  #pragma unroll
  for (int k = 0; k < 36; k++)
    dst[k] = weight * mat_sum[k];
}

__device__
void writeMatDiagSumInc(float weight, const float *mat_sum, float *dst)
{
  #pragma unroll
  for (int k = 0; k < 36; k++)
    dst[k] += weight * mat_sum[k];
}

__device__
void writeMatSum(float weight, const float *mat_sum, const uint *block_coos, const uint *block_write_entries, float *dst)
{
  uint up_idx = (block_coos[0] < block_coos[1]) ? 0 : 6;
  uint down_idx = 6 - up_idx;

  for (uint row = 0; row < 6; row++)
  {
    uint up_spmat_entry = block_write_entries[up_idx + row];
    uint down_spmat_entry = block_write_entries[down_idx + row];

    #pragma unroll
    for (uint col = 0; col < 6; col++)
    {
      uint up_mat_entry = 6 * row + col;
      uint down_mat_entry = 6 * col + row;

      dst[up_spmat_entry] = weight * mat_sum[up_mat_entry];
      dst[down_spmat_entry] = weight * mat_sum[down_mat_entry];

      up_spmat_entry += warpSize;
      down_spmat_entry += warpSize;
    }
  }
}

__device__
void writeMatSumInc(float weight, const float *mat_sum, const uint *block_coos, const uint *block_write_entries, float *dst)
{
  uint up_idx = (block_coos[0] < block_coos[1]) ? 0 : 6;
  uint down_idx = 6 - up_idx;

  for (uint row = 0; row < 6; row++)
  {
    uint up_spmat_entry = block_write_entries[up_idx + row];
    uint down_spmat_entry = block_write_entries[down_idx + row];

    #pragma unroll
    for (uint col = 0; col < 6; col++)
    {
      uint up_mat_entry = 6 * row + col;
      uint down_mat_entry = 6 * col + row;

      dst[up_spmat_entry] += weight * mat_sum[up_mat_entry];
      dst[down_spmat_entry] += weight * mat_sum[down_mat_entry];

      up_spmat_entry += warpSize;
      down_spmat_entry += warpSize;
    }
  }
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void formFitTermVecAndMatDiagKernel(const DetParamList det_param_list,
         size_t block_size,
         const DeviceArrayHandle<uint> block_elem_starts,
         const DeviceArrayHandle<uint> block_elem_sizes,
         const DeviceArrayHandle<uint> block_elem_entry_codes,
         const SpMatParamList spmat_param_list,
         DeviceArrayHandle<float> vec_data,
         SpMatDataList spmat_data_list)
{
  __shared__ uint elem_start, elem_size;

  float fit_term_weight = det_param_list.w_sqrt * det_param_list.w_sqrt;
  float vec_sum[6];
  float mat_sum[36]; // row major

  // vec block stride loop
  for (uint i = blockIdx.x; i < block_size; i += gridDim.x)
  {
    if (threadIdx.x == 0)
    {
      elem_start = block_elem_starts.at(i);
      elem_size = block_elem_sizes.at(i);
    }

    __syncthreads();

    // init vec sum in each block thread
    initVecSum(vec_sum);

    // init mat sum in each block thread
    initMatSum(mat_sum);

    // vec element stride loop
    for (uint elem_stride = 0; elem_stride < elem_size; elem_stride += blockDim.x)
    {
      uint j = threadIdx.x + elem_stride;

      if (j < elem_size)
      {
        uint code = block_elem_entry_codes.at(elem_start + j);

        uint elem_entry_base, a, b;

        elem_entry_decode(code, elem_entry_base, a, b);

        float4 vertex = tex1Dfetch(mesh_vert_texture, elem_entry_base);
        // float4 normal = tex1Dfetch(mesh_norm_texture, elem_entry_base);
        float4 elem_data = tex1Dfetch(elem_data_texture, elem_entry_base);

        if (isnan(elem_data.w)) // is outlier
          continue;

        uint elem_entry = elem_entry_base + a * det_param_list.elem_entry_size;

        float elem_weight = tex1Dfetch(elem_weight_texture, elem_entry);
        uint elem_id = tex1Dfetch(elem_id_texture, elem_entry);

        Affine3d transform;
        TransformType transform_col[4];
        transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id + 0);
        transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id + 1);
        transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id + 2);
        transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id + 3);
        transform.setValue(transform_col);

        float3 blend_vertex = make_float3(transform * vertex);

        // float3 warped_proj_normal = make_float3(elem_data.x, elem_data.y, elem_data.z);
        float residual = elem_data.w;

        float jacobian[6];
        computeFitTermJacobian(elem_weight, elem_data, blend_vertex, jacobian);

        // float3 J_w = cross(make_float3(vertex), make_float3(normal)) * elem_weight;
        // float3 J_v = make_float3(normal) * elem_weight;
        // // float3 J_w = cross(make_float3(vertex), warped_proj_normal) * elem_weight;
        // // float3 J_v = warped_proj_normal * elem_weight;
        // float jacobian[6];
        // jacobian[0] = J_w.x;
        // jacobian[1] = J_w.y;
        // jacobian[2] = J_w.z;
        // jacobian[3] = J_v.x;
        // jacobian[4] = J_v.y;
        // jacobian[5] = J_v.z;

        addFitTermVecSum(residual, jacobian, vec_sum);

        addFitTermMatDiagSum(jacobian, mat_sum);
      }
    } // vec element stride loop

    __syncthreads();

    // block reduction based on warp shuffle
    for (int block_row = 0; block_row < 6; block_row++)
    {
      vec_sum[block_row] = blockReduceSum(vec_sum[block_row]);

      for (int block_col = 0; block_col < 6; block_col++)
      {
        int mat_entry = 6 * block_row + block_col; // row major

        mat_sum[mat_entry] = blockReduceSum(mat_sum[mat_entry]);
      }
    }

    __syncthreads();

    // write block sum
    if (threadIdx.x == 0)
    {
      writeVecSum(fit_term_weight, vec_sum, &(vec_data.at(6 * i)));

      writeMatDiagSum(fit_term_weight, mat_sum, &(spmat_data_list.diagonal[36 * i]));
    }
  } // vec block stride loop
}

void formFitTermVecAndMatDiag(const DetParamList &det_param_list,
                              size_t block_size,
                              const DeviceArray<uint> &block_elem_starts,
                              const DeviceArray<uint> &block_elem_sizes,
                              const DeviceArray<uint> &block_elem_entry_codes,
                              const SpMatParamList &spmat_param_list,
                              DeviceArray<float> &vec_data,
                              SpMatDataList &spmat_data_list)
{
  int block = 256;
  int grid = min(static_cast<int>(block_size), 512);

  formFitTermVecAndMatDiagKernel<<<grid, block>>>(det_param_list,
                                                  block_size,
                                                  block_elem_starts.getHandle(),
                                                  block_elem_sizes.getHandle(),
                                                  block_elem_entry_codes.getHandle(),
                                                  spmat_param_list,
                                                  vec_data.getHandle(),
                                                  spmat_data_list);
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void formFitTermMatKernel(const DetParamList det_param_list,
                          size_t block_size,
                          const DeviceArrayHandle<uint> block_elem_starts,
                          const DeviceArrayHandle<uint> block_elem_sizes,
                          const DeviceArrayHandle<uint> block_elem_entry_codes,
                          const DeviceArrayHandle<uint> block_coos,
                          const DeviceArrayHandle<uint> block_write_entries,
                          const SpMatParamList spmat_param_list,
                          SpMatDataList spmat_data_list)
{
  __shared__ uint elem_start, elem_size;

  float fit_term_weight = det_param_list.w_sqrt * det_param_list.w_sqrt;
  float mat_sum[36]; // row major

  // mat block stride loop
  for (uint i = blockIdx.x; i < block_size; i += gridDim.x)
  {
    if (threadIdx.x == 0)
    {
      elem_start = block_elem_starts.at(i);
      elem_size = block_elem_sizes.at(i);
    }

    __syncthreads();

    // init mat sum in each block thread
    initMatSum(mat_sum);

    // vec element stride loop
    for (uint elem_stride = 0; elem_stride < elem_size; elem_stride += blockDim.x)
    {
      uint j = threadIdx.x + elem_stride;

      if (j < elem_size)
      {
        uint code = block_elem_entry_codes.at(elem_start + j);

        uint elem_entry_base, a, b;

        elem_entry_decode(code, elem_entry_base, a, b);

        float4 vertex = tex1Dfetch(mesh_vert_texture, elem_entry_base);
        // float4 normal = tex1Dfetch(mesh_norm_texture, elem_entry_base);
        float4 elem_data = tex1Dfetch(elem_data_texture, elem_entry_base);

        if (isnan(elem_data.w)) // is outlier
          continue;

        uint elem_entry[2];
        elem_entry[0] = elem_entry_base + a * det_param_list.elem_entry_size;
        elem_entry[1] = elem_entry_base + b * det_param_list.elem_entry_size;

        float elem_weight[2];
        elem_weight[0] = tex1Dfetch(elem_weight_texture, elem_entry[0]);
        elem_weight[1] = tex1Dfetch(elem_weight_texture, elem_entry[1]);

        uint elem_id[2];
        elem_id[0] = tex1Dfetch(elem_id_texture, elem_entry[0]);
        elem_id[1] = tex1Dfetch(elem_id_texture, elem_entry[1]);

        float3 blend_vertex[2];

        Affine3d transform;
        TransformType transform_col[4];
        transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id[0] + 0);
        transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id[0] + 1);
        transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id[0] + 2);
        transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id[0] + 3);
        transform.setValue(transform_col);

        blend_vertex[0] = make_float3(transform * vertex);

        transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id[1] + 0);
        transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id[1] + 1);
        transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id[1] + 2);
        transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id[1] + 3);
        transform.setValue(transform_col);

        blend_vertex[1] = make_float3(transform * vertex);

        float jacob_a[6];
        computeFitTermJacobian(elem_weight[0], elem_data, blend_vertex[0], jacob_a);

        float jacob_b[6];
        computeFitTermJacobian(elem_weight[1], elem_data, blend_vertex[1], jacob_a);

        // float3 warped_proj_normal = make_float3(elem_data.x, elem_data.y, elem_data.z);

        // float3 J_w = cross(make_float3(vertex), make_float3(normal));
        // float3 J_v = make_float3(normal);
        // // float3 J_w = cross(make_float3(vertex), warped_proj_normal);
        // // float3 J_v = warped_proj_normal;

        // float jacob_a[6];
        // jacob_a[0] = J_w.x * elem_weight[0];
        // jacob_a[1] = J_w.y * elem_weight[0];
        // jacob_a[2] = J_w.z * elem_weight[0];
        // jacob_a[3] = J_v.x * elem_weight[0];
        // jacob_a[4] = J_v.y * elem_weight[0];
        // jacob_a[5] = J_v.z * elem_weight[0];

        // float jacob_b[6];
        // jacob_b[0] = J_w.x * elem_weight[1];
        // jacob_b[1] = J_w.y * elem_weight[1];
        // jacob_b[2] = J_w.z * elem_weight[1];
        // jacob_b[3] = J_v.x * elem_weight[1];
        // jacob_b[4] = J_v.y * elem_weight[1];
        // jacob_b[5] = J_v.z * elem_weight[1];

        addFitTermMat(jacob_a, jacob_b, mat_sum);
      }
    } // vec element stride loop

    __syncthreads();

    // block reduction based on warp shuffle
    for (int block_row = 0; block_row < 6; block_row++)
    {
      for (int block_col = 0; block_col < 6; block_col++)
      {
        int mat_entry = 6 * block_row + block_col; // row major

        mat_sum[mat_entry] = blockReduceSum(mat_sum[mat_entry]);
      }
    }

    __syncthreads();

    // write block sum
    if (threadIdx.x == 0)
    {
      writeMatSum(fit_term_weight, mat_sum, &(block_coos.at(2 * i)), &(block_write_entries.at(12 * i)), spmat_data_list.data);
    }
  } // vec block stride loop
}

void formFitTermMat(const DetParamList &det_param_list,
                    size_t block_size,
                    const DeviceArray<uint> &block_elem_starts,
                    const DeviceArray<uint> &block_elem_sizes,
                    const DeviceArray<uint> &block_elem_entry_codes,
                    const DeviceArray<uint> &block_coos,
                    const DeviceArray<uint> &block_write_entries,
                    const SpMatParamList &spmat_param_list,
                    SpMatDataList &spmat_data_list)
{
  int block = 256;
  int grid = min(static_cast<int>(block_size), 512);

  formFitTermMatKernel<<<grid, block>>>(det_param_list,
                                        block_size,
                                        block_elem_starts.getHandle(),
                                        block_elem_sizes.getHandle(),
                                        block_elem_entry_codes.getHandle(),
                                        block_coos.getHandle(),
                                        block_write_entries.getHandle(),
                                        spmat_param_list,
                                        spmat_data_list);
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void formRegTermVecAndMatDiagKernel(const DetParamList det_param_list,
                                    size_t block_size,
                                    const DeviceArrayHandle<uint> block_elem_starts,
                                    const DeviceArrayHandle<uint> block_elem_sizes,
                                    const DeviceArrayHandle<uint> block_elem_entry_codes,
                                    const SpMatParamList spmat_param_list,
                                    DeviceArrayHandle<float> vec_data,
                                    SpMatDataList spmat_data_list)
{
  __shared__ uint elem_start, elem_size;

  const float term_weight = det_param_list.w_sqrt * det_param_list.w_sqrt;

  const float huber_scale = det_param_list.w_huber * det_param_list.w_huber;

  float vec_sum[6];
  float mat_sum[36]; // row major

  // vec block stride loop
  for (uint i = blockIdx.x; i < block_size; i += gridDim.x)
  {
    if (threadIdx.x == 0)
    {
      elem_start = block_elem_starts.at(i);
      elem_size = block_elem_sizes.at(i);
    }

    __syncthreads();

    // init vec sum in each block thread
    initVecSum(vec_sum);

    // init mat sum in each block thread
    initMatSum(mat_sum);

    // vec element stride loop
    for (uint elem_stride = 0; elem_stride < elem_size; elem_stride += blockDim.x)
    {
      uint j = threadIdx.x + elem_stride;

      if (j < elem_size)
      {
        uint code = block_elem_entry_codes.at(elem_start + j);

        uint elem_entry_base, a, b;

        elem_entry_decode(code, elem_entry_base, a, b);

        float4 elem_data = tex1Dfetch(elem_data_texture, elem_entry_base);

        float4 node = make_float4(elem_data.x, elem_data.y, elem_data.z, 1.f);

        // float3 node_cross_inv[3];
        // node_cross_inv[0] = make_float3(0.f, -node.z, node.y);
        // node_cross_inv[1] = make_float3(node.z, 0.f, -node.x);
        // node_cross_inv[2] = make_float3(-node.y, node.x, 0.f);

        float w_level = elem_data.w;

        float3 blend_node[2];

        uint elem_entry = elem_entry_base + a * det_param_list.elem_entry_size;

        // TODO: only for test
        // float elem_weight = tex1Dfetch(elem_weight_texture, elem_entry);
        float elem_weight = 1.f;
        uint elem_id = tex1Dfetch(elem_id_texture, elem_entry);

        Affine3d transform;
        TransformType transform_col[4];
        transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id + 0);
        transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id + 1);
        transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id + 2);
        transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id + 3);
        transform.setValue(transform_col);

        blend_node[0] = make_float3(transform * node);

        // float3 r[3];
        // transform.getRotate(r);

        // float3 r_inv[3];
        // r_inv[0] = make_float3(r[0].x, r[1].x, r[2].x);
        // r_inv[1] = make_float3(r[0].y, r[1].y, r[2].y);
        // r_inv[2] = make_float3(r[0].z, r[1].z, r[2].z);

        elem_entry = elem_entry_base + b * det_param_list.elem_entry_size;

        elem_id = tex1Dfetch(elem_id_texture, elem_entry);

        transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id + 0);
        transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id + 1);
        transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id + 2);
        transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id + 3);
        transform.setValue(transform_col);

        blend_node[1] = make_float3(transform * node);

        float3 residual = blend_node[1] - blend_node[0];

        // Huber loss function
        float huber_energy = dot(residual, residual) / huber_scale;
        float huber_energy_inv = 1.f / huber_energy;
        float huber_energy_sqrt = sqrtf(huber_energy);
        float huber_energy_sqrt_inv = 1.f / huber_energy_sqrt;
        float rho_d_1 = (huber_energy <= 1.f) ? 1.f : huber_energy_sqrt_inv;
        float rho_d_2 = (huber_energy <= 1.f) ? 0.f : ( -0.5f * huber_energy_inv * huber_energy_sqrt_inv);
        rho_d_2 /= huber_scale;

        float3 jacobian[6];
        computeRegTermJacobian(blend_node[0], jacobian);

        // float3 jacobian[6];
        // computeRegTermJacobian(r, r_inv, node_cross_inv, jacobian);

        float jacob_res_dot[6];
        addRegTermVecSum(w_level * elem_weight * rho_d_1, residual, jacobian, jacob_res_dot, vec_sum);

        addRegTermMatDiagSum(w_level * elem_weight, rho_d_1, rho_d_2, jacobian, jacob_res_dot, mat_sum);
      }
    } // vec element stride loop

    __syncthreads();

    // block reduction based on warp shuffle
    for (int block_row = 0; block_row < 6; block_row++)
    {
      vec_sum[block_row] = blockReduceSum(vec_sum[block_row]);

      for (int block_col = 0; block_col < 6; block_col++)
      {
        int mat_entry = 6 * block_row + block_col; // row major

        mat_sum[mat_entry] = blockReduceSum(mat_sum[mat_entry]);
      }
    }

    __syncthreads();

    // write block sum
    if (threadIdx.x == 0)
    {
      writeVecSumInc(term_weight, vec_sum, &(vec_data.at(6 * i)));

      writeMatDiagSumInc(term_weight, mat_sum, &(spmat_data_list.diagonal[36 * i]));
    }
  } // vec block stride loop
}

void formRegTermVecAndMatDiag(const DetParamList &det_param_list,
                              size_t block_size,
                              const DeviceArray<uint> &block_elem_starts,
                              const DeviceArray<uint> &block_elem_sizes,
                              const DeviceArray<uint> &block_elem_entry_codes,
                              const SpMatParamList &spmat_param_list,
                              DeviceArray<float> &vec_data,
                              SpMatDataList &spmat_data_list)
{
  int block = 128;
  int grid = min(static_cast<int>(block_size), 1024);

  formRegTermVecAndMatDiagKernel<<<grid, block>>>(det_param_list,
                                                  block_size,
                                                  block_elem_starts.getHandle(),
                                                  block_elem_sizes.getHandle(),
                                                  block_elem_entry_codes.getHandle(),
                                                  spmat_param_list,
                                                  vec_data.getHandle(),
                                                  spmat_data_list);
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void formRegTermMatKernel(const DetParamList det_param_list,
                          size_t block_size,
                          const DeviceArrayHandle<uint> block_elem_starts,
                          const DeviceArrayHandle<uint> block_elem_sizes,
                          const DeviceArrayHandle<uint> block_elem_entry_codes,
                          const DeviceArrayHandle<uint> block_coos,
                          const DeviceArrayHandle<uint> block_write_entries,
                          const SpMatParamList spmat_param_list,
                          SpMatDataList spmat_data_list)
{
  const int stride = gridDim.x * blockDim.x;

  const float term_weight = det_param_list.w_sqrt * det_param_list.w_sqrt;

  const float huber_scale = det_param_list.w_huber * det_param_list.w_huber;

  float mat_sum[36]; // row major

  // grid-stride loop
  for (uint i = threadIdx.x + blockDim.x * blockIdx.x;
       i < block_size;
       i += stride)
  {
    uint elem_start = block_elem_starts.at(i);
    uint elem_size = block_elem_sizes.at(i);

    if (elem_size == 0)
      continue;

    initMatSum(mat_sum);

    for (uint j = 0; j < elem_size; j++)
    {
      uint code = block_elem_entry_codes.at(elem_start + j);

      uint elem_entry_base, a, b;

      elem_entry_decode(code, elem_entry_base, a, b);

      float4 elem_data = tex1Dfetch(elem_data_texture, elem_entry_base);

      float4 node = make_float4(elem_data.x, elem_data.y, elem_data.z, 1.f);

      // float3 node_cross_inv[3];
      // node_cross_inv[0] = make_float3(0.f, -node.z, node.y);
      // node_cross_inv[1] = make_float3(node.z, 0.f, -node.x);
      // node_cross_inv[2] = make_float3(-node.y, node.x, 0.f);

      float w_level = elem_data.w;

      float3 blend_node[2];

      uint elem_entry = elem_entry_base + a * det_param_list.elem_entry_size;

      // TODO: only for test
      // float elem_weight = tex1Dfetch(elem_weight_texture, elem_entry);
      float elem_weight = 1.f;
      uint elem_id = tex1Dfetch(elem_id_texture, elem_entry);

      Affine3d transform;
      TransformType transform_col[4];
      transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id + 0);
      transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id + 1);
      transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id + 2);
      transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id + 3);
      transform.setValue(transform_col);

      blend_node[0] = make_float3(transform * node);

      // float3 ra[3];
      // transform.getRotate(ra);

      // float3 ra_inv[3];
      // ra_inv[0] = make_float3(ra[0].x, ra[1].x, ra[2].x);
      // ra_inv[1] = make_float3(ra[0].y, ra[1].y, ra[2].y);
      // ra_inv[2] = make_float3(ra[0].z, ra[1].z, ra[2].z);

      elem_entry = elem_entry_base + b * det_param_list.elem_entry_size;

      elem_id = tex1Dfetch(elem_id_texture, elem_entry);

      transform_col[0] = tex1Dfetch(transform_texture, 4 * elem_id + 0);
      transform_col[1] = tex1Dfetch(transform_texture, 4 * elem_id + 1);
      transform_col[2] = tex1Dfetch(transform_texture, 4 * elem_id + 2);
      transform_col[3] = tex1Dfetch(transform_texture, 4 * elem_id + 3);
      transform.setValue(transform_col);

      blend_node[1] = make_float3(transform * node);

      // float3 rb[3];
      // transform.getRotate(rb);

      // float3 rb_inv[3];
      // rb_inv[0] = make_float3(rb[0].x, rb[1].x, rb[2].x);
      // rb_inv[1] = make_float3(rb[0].y, rb[1].y, rb[2].y);
      // rb_inv[2] = make_float3(rb[0].z, rb[1].z, rb[2].z);

      float3 residual = blend_node[1] - blend_node[0];

      // Huber loss function
      float huber_energy = dot(residual, residual) / huber_scale;
      float huber_energy_inv = 1.f / huber_energy;
      float huber_energy_sqrt = sqrtf(huber_energy);
      float huber_energy_sqrt_inv = 1.f / huber_energy_sqrt;
      float rho_d_1 = (huber_energy <= 1.f) ? 1.f : huber_energy_sqrt_inv;
      float rho_d_2 = (huber_energy <= 1.f) ? 0.f : ( -0.5f * huber_energy_inv * huber_energy_sqrt_inv);
      rho_d_2 /= huber_scale;

      float3 jacob_a[6];
      computeRegTermJacobian(blend_node[0], jacob_a);

      float3 jacob_b[6];
      computeRegTermJacobian(blend_node[1], jacob_b);

      // float3 jacob_a[6];
      // computeRegTermJacobian(ra, ra_inv, node_cross_inv, jacob_a);

      // float3 jacob_b[6];
      // computeRegTermJacobian(rb, rb_inv, node_cross_inv, jacob_b);

      float jacob_res_dot_a[6];
      float jacob_res_dot_b[6];
      #pragma unroll
      for (int k = 0; k < 6; k++)
      {
        float dot_result; 
        
        dot_result = dot(jacob_a[k], residual);
        jacob_res_dot_a[k] = dot_result;

        dot_result = dot(jacob_b[k], residual);
        jacob_res_dot_b[k] = dot_result;
      }

      addRegTermMat(-w_level * elem_weight, rho_d_1, rho_d_2, jacob_a, jacob_b, jacob_res_dot_a, jacob_res_dot_b, mat_sum);
    } // j < elem_size

    writeMatSumInc(term_weight, mat_sum, &(block_coos.at(2 * i)), &(block_write_entries.at(12 * i)), spmat_data_list.data);
  } // grid-stride loop
}

void formRegTermMat(const DetParamList &det_param_list,
                    size_t block_size,
                    const DeviceArray<uint> &block_elem_starts,
                    const DeviceArray<uint> &block_elem_sizes,
                    const DeviceArray<uint> &block_elem_entry_codes,
                    const DeviceArray<uint> &block_coos,
                    const DeviceArray<uint> &block_write_entries,
                    const SpMatParamList &spmat_param_list,
                    SpMatDataList &spmat_data_list)
{
  int block = 256;
  int grid = min(divUp(block_size, block), 512);

  formRegTermMatKernel<<<grid, block>>>(det_param_list,
                                        block_size,
                                        block_elem_starts.getHandle(),
                                        block_elem_sizes.getHandle(),
                                        block_elem_entry_codes.getHandle(),
                                        block_coos.getHandle(),
                                        block_write_entries.getHandle(),
                                        spmat_param_list,
                                        spmat_data_list);
  checkCudaErrors(cudaGetLastError());
}

} // namespace gpu
} // namespace dynfu
