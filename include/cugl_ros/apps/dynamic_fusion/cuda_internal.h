#ifndef __CUGL_APPS_DYNFU_CUDA_H__
#define __CUGL_APPS_DYNFU_CUDA_H__

#include <cugl_ros/common/containers.h>
#include <cugl_ros/common/geometry.h>

#include <iostream>

using cugl::DeviceArray;
using cugl::DeviceArrayHandle;
using cugl::DeviceArray2D;
using cugl::DeviceArrayHandle2D;
using cugl::Affine3d;
using cugl::VertexImage;
using cugl::NormalImage;
using cugl::DepthImage32f;
using cugl::ColorImage8u;

namespace dynfu
{
namespace gpu
{
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/*
 * Public types
 */
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef short2 TsdfVoxelType;
typedef uchar4 RgbaVoxelType;
typedef uint2 HashEntryType;
typedef float4 TransformType;

const float ISO_VALUE = 0.f;

enum
{
  MAX_TRIANGLES_SIZE = 800000,
  MAX_GRAPH_NODES_SIZE = 5120,
  MAX_VOXEL_BLOCK_SIZE = 30000,
  MAX_NON_ZERO_BLOCKS_SIZE = MAX_GRAPH_NODES_SIZE * 32,
  MAX_SPMAT_ELEMS_SIZE = MAX_NON_ZERO_BLOCKS_SIZE * 2 * 36,
  VOXEL_BLENDING_QUAD_STRIDE = 2, // must <= 4
  NODE_BLENDING_QUAD_STRIDE = 2,  // must <= 2
  VOXEL_BLENDING_MAX_QUAD_STRIDE = 5,
  NODE_BLENDING_MAX_QUAD_STRIDE = 5
};

enum Distortion
{
  DISTORTION_NONE,
  DISTORTION_MODIFIED_BROWN_CONRADY, // projection: 3D to 2D
  DISTORTION_INVERSE_BROWN_CONRADY   // deprojection: 2D to 3D
};

struct Intrinsics
{
  int width;
  int height;
  float cx;
  float cy;
  float fx;
  float fy;
  float depth_scale;
  Distortion model;
  float coeffs[5];
};

struct Roi
{
  int2 start;
  int2 end;
};

struct IcpParamList
{
  float dist_thres;
  float angle_thres;
  float view_angle_thres;
};

struct DetParamList
{
  float w_sqrt;
  float w_huber;
  uint term_num;
  uint elem_entry_start;
  uint elem_entry_size;
  uint vec_block_elem_stride;
  uint mat_block_elem_stride;
  uint vec_block_entry_start;
  uint mat_block_entry_start;
};

struct FitTermDataList
{
  float4 *elem_data;
  uint   *elem_ids;
  float  *elem_weights;
};

struct RegTermDataList
{
  float4 *elem_data;
  uint   *elem_ids;
  float  *elem_weights;
};

struct DetBlockDataList
{
  uint *vec_block_elem_ids;
  uint *vec_block_elem_entry_codes; // *vec_block_elem_entries;
  uint *mat_block_elem_ids;
  uint *mat_block_elem_entry_codes; // *mat_block_elem_entries;
};

struct SpMatParamList
{
  size_t bin_width;
  size_t block_dim;
  size_t rows;
  size_t pad_rows;
};

struct SpMatDataList
{
  uint *row_lengths;
  uint *bin_lengths;
  uint *bin_offsets;

  uint *offsets;

  float *diagonal;
  float *precond;

  uint  *col_indices;
  float *data;
};

static inline int divUp(int total, int grain) { return (total + grain -1) / grain; }

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/*
 * Image Processing
 */
void depthToColor(float2 thresholds,
                  float depth_scale,
                  int width, int height,
                  const DeviceArray2D<ushort> &src_depth,
                  DeviceArray2D<uchar4> &dst_color);

void convertColor(int width, int height,
                  bool draw_mask,
                  int2 mask_start,
                  int2 mask_end,
                  const DeviceArray2D<uchar> &src_color,
                  DeviceArray2D<uchar4> &dst_color);

void truncateDepth(float2 depth_thresholds,
                   float4 color_thresholds,
                   bool draw_mask,
                   int2 mask_start,
                   int2 mask_end,
                   const Intrinsics &intrin,
                   const DeviceArray2D<ushort> &src_depth,
                   const DeviceArray2D<uchar> &src_color,
                   DeviceArray2D<ushort> &dst_depth_16u,
                   DeviceArray2D<float> &dst_depth_32f);

void bilateralFilter(int width, int height,
                     const DeviceArray2D<ushort> &src_depth,
                     DeviceArray2D<ushort> &dst_depth);

void createVertexImage(const Intrinsics &intrin,
                       const DeviceArray2D<ushort> &src_depth,
                       DeviceArray2D<float4> &dst_vertex);

void createNormalImage(int width, int height,
                       const DeviceArray2D<float4> &src_vertex,
                       DeviceArray2D<float4> &dst_normal);


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/*
 * Deformable Voxel Hierarchy
 */
// bind and unbind textures
void bindImageTextures(int width, int height,
                       const VertexImage &vertex_image,
                       const NormalImage &normal_image,
                       const DepthImage32f &depth_image_32f,
                       const ColorImage8u &color_image_8u);

void unbindImageTextures();

void bindImageTrueTextures(int width, int height,
                           const VertexImage &vertex_image,
                           const NormalImage &normal_image);

void unbindImageTrueTextures();

void bindBlendingTextures(const DeviceArray<TransformType> &transform_array,
                          const DeviceArray<HashEntryType> &block_hash_table,
                          const DeviceArray<float4> &blend_weight_table,
                          const DeviceArray<int4> &blend_code_table);

void unbindBlendingTextures();

void bindBlendingTableTextures(const DeviceArray<float4> &blend_weight_table,
                               const DeviceArray<int4> &blend_code_table);

void unbindBlendingTableTextures();

void bindTransformTextures(const DeviceArray<TransformType> &transform_array);

void unbindTransformTextures();

void bindBlockHashTableTextures(const DeviceArray<HashEntryType> &hash_table);

void unbindBlockHashTableTextures();

void bindMarchingCubesTextures(const DeviceArray<int> &tri_table,
                               const DeviceArray<int> &num_verts_table);

void unbindMarchingCubesTextures();

void bindTsdfVoxelTexture(const DeviceArray<TsdfVoxelType> &tsdf_voxel_array);

void unbindTsdfVoxelTexture();

void bindRgbaVoxelTexture(const DeviceArray<RgbaVoxelType> &rgba_voxel_array);

void unbindRgbaVoxelTexture();

void bindMeshTextures(const DeviceArray<float4> &vertex_array,
                      const DeviceArray<float4> &normal_array);

void unbindMeshTextures();

void bindTermElemTextures(const DeviceArray<float4> &elem_data,
                          const DeviceArray<uint> &elem_ids,
                          const DeviceArray<float> &elem_weights);

void unbindTermElemTextures();

// init
void resetSurfelFlags(DeviceArray<uint> &surfel_flags);

void resetVoxelBlockArray(DeviceArray<TsdfVoxelType> &tsdf_voxel_block_array);

void resetVoxelBlockArray(DeviceArray<RgbaVoxelType> &rgba_voxel_block_array);

void resetTransformArray(DeviceArray<TransformType> &transform_array);

void resetBlockHashTable(DeviceArray<HashEntryType> &block_hash_table);

void generateVoxelBlendingTable(float voxel_length,
                                int voxel_block_inner_dim,
                                float voxel_block_length,
                                int voxel_block_dim,
                                float node_block_length,
                                int node_block_dim,
                                DeviceArray<float4> &voxel_blend_weight_table,
                                DeviceArray<int4> &voxel_blend_code_table);

void generateNodeBlendingTable(float node_block_length_prev,
                               int node_block_dim_prev,
                               float node_block_length_curr,
                               int node_block_dim_curr,
                               DeviceArray<float4> &node_blend_weight_table,
                               DeviceArray<int4> &node_blend_code_table);

// init and update deformable voxel hierarchy
bool initVoxelBlock(const Intrinsics &intrin,
                    const Affine3d &transform,
                    float voxel_block_length,
                    int voxel_block_dim,
                    size_t &voxel_block_size,
                    DeviceArray<uint> &pending_node_hash_array,
                    DeviceArray<uint> &voxel_block_hash_array,
                    DeviceArray<HashEntryType> &voxel_block_hash_table);

bool updateVoxelBlock(float voxel_block_length,
                      int voxel_block_dim,
                      size_t vertex_size,
                      const DeviceArray<float4> &vertex_array,
                      const DeviceArray<uint> &active_flags,
                      size_t &voxel_block_size,
                      DeviceArray<uint> &pending_node_hash_array,
                      DeviceArray<uint> &voxel_block_hash_array,
                      DeviceArray<HashEntryType> &voxel_block_hash_table);

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
                            DeviceArray<uint> &node_morton_code_array);

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
                              DeviceArray<uint> &node_morton_code_array);

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
                            DeviceArray<TransformType> &transform_array);

void reorderDeformGraph(size_t node_size,
                        DeviceArray<uint> &node_hash_array,
                        DeviceArray<float4> &node_array,
                        DeviceArray<TransformType> &transform_array,
                        DeviceArray<uint> &node_morton_code_array,
                        DeviceArray<uint> &node_id_reorder_array,
                        DeviceArray<HashEntryType *> &block_hash_table_hierarchy_handle);

void copyTransformArray(size_t node_size,
                        DeviceArray<TransformType> &transform_array);

// volume fusion
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
                DeviceArray<RgbaVoxelType> &rgba_voxel_block_array);

// marching cubes
int getOccupiedVoxels(int voxel_block_inner_dim,
                      int voxel_block_dim,
                      size_t voxel_block_size,
                      const DeviceArray<uint> &voxel_block_hash_array,
                      DeviceArray<HashEntryType> &voxel_block_hash_table,
                      DeviceArray<int> &voxel_grid_hash_array,
                      DeviceArray<int> &vertex_num_array,
                      DeviceArray<float4> &cube_tsdf_field_array,
                      DeviceArray<int4> &cube_rgba_field_array);

size_t computeOffsetsAndTotalVertices(int occupied_voxel_size,
                                      DeviceArray<int> &vertex_num_array,
                                      DeviceArray<int> &vertex_offset_array);

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
                       DeviceArray<float4> &mesh_color_array);

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
                       DeviceArray<float>  &surfel_color_array);

// surface warping
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
                 float &res_sqrt);

void combineVertexAndColorData(size_t surfel_size,
                               const Affine3d &transform,
                               const DeviceArray<float4> &surfel_vertex_data,
                               const DeviceArray<float> &surfel_color_data,
                               DeviceArray<float> &surfel_xyzrgb_data);

// motion tracking
void formIcpGaussNewtonOpt(const IcpParamList &icp_param,
                           const Intrinsics &intrin,
                           const Affine3d &transform,
                           size_t vertex_size,
                           const DeviceArray<float4> &vertex_array,
                           const DeviceArray<float4> &normal_array,
                           DeviceArray<float> &sum_buf);

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
                  DetBlockDataList &det_block);

float updateFitTerm(const IcpParamList &icp_param,
                    const Intrinsics &intrin,
                    const Affine3d &transform,
                    size_t vertex_size,
                    const DetParamList &det_param,
                    FitTermDataList &fit_term);

float initRegTerm(float w_level,
                  int dim_ratio,
                  float node_block_length,
                  int node_block_dim,
                  size_t node_block_size,
                  const DeviceArray<uint> &node_block_hash_array,
                  /*const DeviceArray<HashEntryType> &node_block_hash_table,*/
                  const DetParamList &det_param,
                  RegTermDataList &reg_term,
                  DetBlockDataList &det_block);

float updateRegTerm(size_t node_block_size,
                    const DetParamList &det_param,
                    RegTermDataList &reg_term);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/*
 * Deformable Energy Terms
 */
bool initVecBlock(uint block_elem_size,
                  size_t &vec_block_size,
                  DeviceArray<uint> &block_elem_ids,
                  DeviceArray<uint> &block_elem_entry_codes,
                  DeviceArray<uint> &block_ids,
                  DeviceArray<uint> &block_elem_starts,
                  DeviceArray<uint> &block_elem_sizes,
                  DeviceArray<uint> &block_id2row_lut);

bool initMatBlock(uint block_elem_size,
                  size_t vec_block_size,
                  const DeviceArray<uint> &block_id2row_lut,
                  size_t &mat_block_size,
                  DeviceArray<uint> &block_elem_ids,
                  DeviceArray<uint> &block_elem_entry_codes,
                  DeviceArray<uint> &block_ids,
                  DeviceArray<uint> &block_elem_starts,
                  DeviceArray<uint> &block_elem_sizes,
                  DeviceArray<uint> &block_ext_ids,
                  DeviceArray<uint> &block_entries,
                  DeviceArray<uint> &block_row_lengths,
                  DeviceArray<uint> &block_row_offsets,
                  DeviceArray<uint> &block_coos);

void classifyEnergyTerms(size_t block_size,
                         const DeviceArray<uint> &block_elem_starts,
                         const DeviceArray<uint> &block_elem_sizes,
                         const DeviceArray<uint> &block_elem_entry_codes,
                         DeviceArray<uint> &fit_term_elem_sizes,
                         DeviceArray<uint> &fit_term_elem_entry_codes,
                         DeviceArray<uint> &reg_term_elem_sizes,
                         DeviceArray<uint> &reg_term_elem_entry_codes);

bool initSparseMat(size_t vec_block_size,
                   size_t mat_block_size,
                   DeviceArray<uint> &mat_block_row_lengths,
                   DeviceArray<uint> &mat_block_coos,
                   DeviceArray<uint> &mat_block_write_entries,
                   SpMatParamList &param_list,
                   SpMatDataList &data_list);

void formFitTermVecAndMatDiag(const DetParamList &det_param_list,
                              size_t block_size,
                              const DeviceArray<uint> &block_elem_starts,
                              const DeviceArray<uint> &block_elem_sizes,
                              const DeviceArray<uint> &block_elem_entry_codes,
                              const SpMatParamList &spmat_param_list,
                              DeviceArray<float> &vec_data,
                              SpMatDataList &spmat_data_list);

void formFitTermMat(const DetParamList &det_param_list,
                    size_t block_size,
                    const DeviceArray<uint> &block_elem_starts,
                    const DeviceArray<uint> &block_elem_sizes,
                    const DeviceArray<uint> &block_elem_entry_codes,
                    const DeviceArray<uint> &block_coos,
                    const DeviceArray<uint> &block_write_entries,
                    const SpMatParamList &spmat_param_list,
                    SpMatDataList &spmat_data_list);

void formRegTermVecAndMatDiag(const DetParamList &det_param_list,
                              size_t block_size,
                              const DeviceArray<uint> &block_elem_starts,
                              const DeviceArray<uint> &block_elem_sizes,
                              const DeviceArray<uint> &block_elem_entry_codes,
                              const SpMatParamList &spmat_param_list,
                              DeviceArray<float> &vec_data,
                              SpMatDataList &spmat_data_list);

void formRegTermMat(const DetParamList &det_param_list,
                    size_t block_size,
                    const DeviceArray<uint> &block_elem_starts,
                    const DeviceArray<uint> &block_elem_sizes,
                    const DeviceArray<uint> &block_elem_entry_codes,
                    const DeviceArray<uint> &block_coos,
                    const DeviceArray<uint> &block_write_entries,
                    const SpMatParamList &spmat_param_list,
                    SpMatDataList &spmat_data_list);

void updateTransformArray(float step_size,
                          size_t block_size,
                          const DeviceArray<uint> &block_ids,
                          const DeviceArray<float> &results,
                          DeviceArray<TransformType> &transform_array);

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/*
 * PCG Solver
 */
void bindVecTexture(const DeviceArray<float> &vec_data);

void unbindVecTexture();

void computePreconditioner(float damping,
                           const SpMatParamList &spmat_param_list,
                           SpMatDataList &spmat_data_list);

float initPcgSolver(const SpMatParamList &spmat_param_list,
                    const SpMatDataList &spmat_data_list,
                    const DeviceArray<float> &b,
                    DeviceArray<float> &x,
                    DeviceArray<float> &r,
                    DeviceArray<float> &z,
                    DeviceArray<float> &p);

float iteratePcgSolver(float delta_old,
                       const SpMatParamList &spmat_param_list,
                       const SpMatDataList &spmat_data_list,
                       const DeviceArray<float> &b,
                       DeviceArray<float> &x,
                       DeviceArray<float> &y,
                       DeviceArray<float> &r,
                       DeviceArray<float> &z,
                       DeviceArray<float> &p);

} // namespace gpu
} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_CUDA_H__ */
