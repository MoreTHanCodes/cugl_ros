#ifndef __CUGL_APPS_DYNFU_DEFORM_VOXEL_HIERARCHY_H__
#define __CUGL_APPS_DYNFU_DEFORM_VOXEL_HIERARCHY_H__

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

#include "cuda_internal.h"
#include "deform_energy_terms.h"

namespace dynfu
{
/*
 * Class of the deformable voxel hierarchy data structure.
 *
 * Author: Tao Han, City University of Hong Kong (tao.han@my.cityu.edu.hk)
 *
 * union TsdfVoxel __align__(4)
 * {
 *   short2 data;
 *   struct
 *   {
 *     short tsdf;
 *     short weight;
 *   };
 * };
 *
 * union RgbaVoxel __align__(4)
 * {
 *   uchar4 data;
 *   struct
 *   {
 *     uchar r;
 *     uchar g;
 *     uchar b;
 *     uchar a;
 *   };
 * };
 *
 * union HashEntry __align__(8)
 * {
 *   uint2 data;
 *   struct
 *   {
 *     uint node_id;
 *     uint block_offset;
 *   };
 * };
 *
 */

class DeformVoxelHierarchy
{
 public:
  /*
   * Public Types
   */
  typedef std::shared_ptr<DeformVoxelHierarchy> Ptr;

  /*
   * Public Methods
   */
  DeformVoxelHierarchy(const gpu::IcpParamList &icp_param,
                       const gpu::Intrinsics &intrin,
                       const gpu::Roi &roi,
                       float trunc_dist,
                       float volume_length,
                       int voxel_block_inner_dim_shift,
                       int voxel_block_dim_shift,
                       int node_block_dim_shift_base,
                       float w_fit,
                       float w_fit_huber,
                       float w_reg,
                       float w_reg_huber);

  ~DeformVoxelHierarchy();

  size_t getNodeSize() const { return node_size_; }

  size_t getVoxelBlockSize() const { return voxel_block_size_; }

  void reset();

  bool init(const cugl::Affine3d &transform,
            const cugl::Affine3d &transform_inv,
            float4 color_thres);

  void fuseVolume(const cugl::Affine3d &transform, bool is_first_frame, float4 color_thres);

  size_t fetchIsoSurface(cugl::VertexArray &mesh_vertex_array,
                         cugl::NormalArray &mesh_normal_array,
                         cugl::ColorArray  &mesh_color_array);

  size_t fetchIsoSurface(cugl::VertexArray &mesh_vertex_array,
                         cugl::NormalArray &mesh_normal_array,
                         cugl::ColorArray  &mesh_color_array,
                         cugl::VertexArray &surfel_vertex_array,
                         cugl::NormalArray &surfel_normal_array,
                         cugl::DeviceArray<float> &surfel_color_array);

  bool update(size_t vertex_size,
              const cugl::VertexArray &vertex_array,
              const cugl::DeviceArray<uint> &active_flags);

  void warpSurface(const cugl::Affine3d &transform,
                   size_t mesh_vertex_size,
                   cugl::VertexArray &ref_mesh_vertex_array,
                   cugl::NormalArray &ref_mesh_normal_array,
                   cugl::ColorArray  &ref_mesh_color_array,
                   cugl::VertexArray &warped_mesh_vertex_array,
                   cugl::NormalArray &warped_mesh_normal_array,
                   cugl::VertexArray &warped_surfel_vertex_array,
                   cugl::NormalArray &warped_surfel_normal_array,
                   cugl::DeviceArray<uint> &active_flags,
                   float &res_sqrt);

  void formIcpGaussNewtonOpt(const cugl::Affine3d &transform,
                             size_t vertex_size,
                             const cugl::VertexArray &vertex_array,
                             const cugl::NormalArray &normal_array,
                             cugl::DeviceArray<float> &sum_buf,
                             double *mat_host,
                             double *vec_host);

  bool initDeformEnergyTerms(const cugl::Affine3d &transform,
                             size_t vertex_size,
                             float &total_energy);

  void updateDeformEnergyTerms(const cugl::Affine3d &transform,
                               size_t vertex_size,
                               float &total_energy);

  void formDetGaussNewtonOpt();

  bool solveDetGaussNewtonOpt(float damping);

  // prepare for backtracking line search
  void initTransformSearch();

  // perform backtracking line search
  void runTransformSearchOnce(float step_size);

  void saveDetGaussNewtonOpt(int frame_count);

 private:
  /*
   * Private Types
   */
  typedef std::shared_ptr< cugl::DeviceArray<unsigned int> > HashArrayHandle;
  typedef std::shared_ptr< cugl::DeviceArray<gpu::HashEntryType> > HashTableHandle;

  enum
  {
    HIERARCHY_LEVELS = 1
  };

  /*
   * Parameters
   */
  gpu::IcpParamList icp_param_;

  gpu::Intrinsics intrin_;

  gpu::Roi roi_;

  /*
   * Deformation Energy Terms
   */
  DeformEnergyTerms::Ptr det_ptr_;

  /*
   * Voxel Block Hashing
   */
  float trunc_dist_;

  float volume_length_;

  float voxel_length_;

  float voxel_block_length_;

  int voxel_block_inner_dim_;

  int voxel_block_dim_;

  size_t voxel_block_size_;

  cugl::DeviceArray<unsigned int> voxel_block_hash_array_;

  cugl::DeviceArray<gpu::HashEntryType> voxel_block_hash_table_;

  cugl::DeviceArray<gpu::TsdfVoxelType> tsdf_voxel_block_array_;

  cugl::DeviceArray<gpu::RgbaVoxelType> rgba_voxel_block_array_;

  cugl::DeviceArray<float4> voxel_blend_weight_table_;

  cugl::DeviceArray<int4> voxel_blend_code_table_;

  /*
   * Node Block Hierarchy
   */
  std::vector<float> node_block_length_hierarchy_;

  std::vector<int> node_block_dim_hierarchy_;

  std::vector<size_t> node_block_size_hierarchy_;

  std::vector<HashArrayHandle> node_block_hash_array_hierarchy_;

  std::vector<HashTableHandle> node_block_hash_table_hierarchy_;

  cugl::DeviceArray<gpu::HashEntryType *> node_block_hash_table_hierarchy_handle_;

  cugl::DeviceArray<float4> node_blend_weight_table_;

  cugl::DeviceArray<int4> node_blend_code_table_;

  /*
   * Deformation graph
   */
  size_t node_size_;

  cugl::DeviceArray<unsigned int> node_hash_array_;

  cugl::VertexArray node_array_;

  // Transformations are represented in 4x4 matrices
  cugl::DeviceArray<gpu::TransformType> transform_array_;

  cugl::DeviceArray<unsigned int> node_morton_code_array_;

  cugl::DeviceArray<unsigned int> node_id_reorder_array_;

  /*
   * Marching Cubes
   */
  cugl::DeviceArray<int> voxel_grid_hash_array_;

  cugl::DeviceArray<int> vertex_num_array_;

  cugl::DeviceArray<int> vertex_offset_array_;

  cugl::DeviceArray<float4> cube_tsdf_field_array_;

  cugl::DeviceArray<int4> cube_rgba_field_array_;

  cugl::DeviceArray<int> tri_table_;

  cugl::DeviceArray<int> num_verts_table_;

  /*
   * Others
   */
  cugl::DeviceArray<unsigned int> pending_node_hash_array_;

  /*
   * Private Methods
   */
  void alloc(int image_width, int image_height);

  void free();

  void generateBlendingTable();
};
} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_DEFORM_VOXEL_HIERARCHY_H__ */
