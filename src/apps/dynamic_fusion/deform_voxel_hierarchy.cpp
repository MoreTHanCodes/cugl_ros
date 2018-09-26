#include <cugl_ros/apps/dynamic_fusion/deform_voxel_hierarchy.h>
#include <cugl_ros/apps/dynamic_fusion/marching_cubes_table.h>

extern const int tri_table[256][16]; 
extern const int num_verts_table[256];

namespace dynfu
{
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Public Methods
 */
DeformVoxelHierarchy::DeformVoxelHierarchy(const gpu::IcpParamList &icp_param,
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
                                           float w_reg_huber)
    : icp_param_(icp_param),
      intrin_(intrin),
      roi_(roi),
      trunc_dist_(trunc_dist),
      volume_length_(volume_length),
      voxel_block_size_(0),
      node_size_(0)
{
  // Voxel block hashing
  // std::cout << "voxel_block_dim_shift = " << voxel_block_dim_shift << std::endl;
  voxel_block_dim_ = (1 << voxel_block_dim_shift);

  voxel_block_length_ = volume_length_ / static_cast<float>(voxel_block_dim_);

  voxel_block_inner_dim_ = (1 << voxel_block_inner_dim_shift);

  voxel_length_ = voxel_block_length_ / static_cast<float>(voxel_block_inner_dim_ - 1);

  std::cout << "voxel_length_ = " << int(voxel_length_ * 1000.f) << "mm" << std::endl;

  // Node block hierarchy
  if (node_block_dim_shift_base > voxel_block_dim_shift)
    node_block_dim_shift_base = voxel_block_dim_shift;

  int node_block_dim_base = (1 << node_block_dim_shift_base);

  float node_block_length_base = volume_length_ / static_cast<float>(node_block_dim_base);

  node_block_dim_hierarchy_.push_back(node_block_dim_base);

  node_block_length_hierarchy_.push_back(node_block_length_base);

  node_block_size_hierarchy_.push_back(0);

  for (int level = 1; level < HIERARCHY_LEVELS; level++)
  {
    int node_block_dim_prev = node_block_dim_hierarchy_[level - 1];

    float node_block_length_prev = node_block_length_hierarchy_[level - 1];

    node_block_dim_hierarchy_.push_back(node_block_dim_prev / 2);

    node_block_length_hierarchy_.push_back(node_block_length_prev * 2.f);

    node_block_size_hierarchy_.push_back(0);
  }

  float w_sqrt[2];
  w_sqrt[0] = std::sqrt(w_fit);
  w_sqrt[1] = std::sqrt(w_reg);

  float w_huber[2];
  w_huber[0] = w_fit_huber;
  w_huber[1] = w_reg_huber;

  const size_t energy_terms_block_dim = 6;

  det_ptr_ = DeformEnergyTerms::Ptr(new DeformEnergyTerms(energy_terms_block_dim, w_sqrt, w_huber));

  alloc(intrin_.width, intrin_.height);

  generateBlendingTable();

  reset();
}

DeformVoxelHierarchy::~DeformVoxelHierarchy()
{
  free();
}

void DeformVoxelHierarchy::reset()
{
  // Voxel block hashing
  voxel_block_size_ = 0;

  gpu::resetBlockHashTable(voxel_block_hash_table_);

  gpu::resetVoxelBlockArray(tsdf_voxel_block_array_);

  gpu::resetVoxelBlockArray(rgba_voxel_block_array_);

  // Node block hierarchy
  for (int level = 0; level < HIERARCHY_LEVELS; level++)
  {
    node_block_size_hierarchy_[level] = 0;

    HashTableHandle hash_table_handle = node_block_hash_table_hierarchy_[level];

    gpu::resetBlockHashTable(*hash_table_handle);
  }

  // Deformation graph 
  node_size_ = 0;

  gpu::resetTransformArray(transform_array_);
}

bool DeformVoxelHierarchy::init(const cugl::Affine3d &transform,
                                const cugl::Affine3d &transform_inv,
                                float4 color_thres)
{
  bool success = true;

  success = gpu::initVoxelBlock(intrin_,
                                transform_inv,
                                voxel_block_length_,
                                voxel_block_dim_,
                                voxel_block_size_,
                                pending_node_hash_array_,
                                voxel_block_hash_array_,
                                voxel_block_hash_table_);

  if (!success)
  {
    std::cout << std::endl << "DeformVoxelHierarchy: Init voxel block failed!" << std::endl;

    return success;
  }

  success = gpu::initNodeBlockHierarchy(volume_length_,
                                        voxel_block_dim_,
                                        voxel_block_size_,
                                        voxel_block_hash_array_,
                                        0,
                                        node_block_length_hierarchy_[0],
                                        node_block_dim_hierarchy_[0],
                                        node_block_size_hierarchy_[0],
                                        pending_node_hash_array_,
                                        *(node_block_hash_array_hierarchy_[0]),
                                        *(node_block_hash_table_hierarchy_[0]),
                                        node_size_,
                                        node_hash_array_,
                                        node_array_,
                                        node_morton_code_array_);

  if (!success)
  {
    std::cout << std::endl << "DeformVoxelHierarchy: Init node block hierarchy failed! [level = " << 0 << "]" << std::endl;

    return success;
  }

  for (int level = 1; level < HIERARCHY_LEVELS; level++)
  {
    success = gpu::initNodeBlockHierarchy(volume_length_,
                                          node_block_dim_hierarchy_[level-1],
                                          node_block_size_hierarchy_[level-1],
                                          *(node_block_hash_array_hierarchy_[level-1]),
                                          level,
                                          node_block_length_hierarchy_[level],
                                          node_block_dim_hierarchy_[level],
                                          node_block_size_hierarchy_[level],
                                          pending_node_hash_array_,
                                          *(node_block_hash_array_hierarchy_[level]),
                                          *(node_block_hash_table_hierarchy_[level]),
                                          node_size_,
                                          node_hash_array_,
                                          node_array_,
                                          node_morton_code_array_);

    if (!success)
    {
      std::cout << std::endl << "DeformVoxelHierarchy: Init node block hierarchy failed! [level = " << level << "]" << std::endl;

      return success;
    }
  }

  gpu::reorderDeformGraph(node_size_,
                          node_hash_array_,
                          node_array_,
                          transform_array_,
                          node_morton_code_array_,
                          node_id_reorder_array_,
                          node_block_hash_table_hierarchy_handle_);

  // std::cout << "DeformVoxelHierarchy: node_size = " << node_size_ << std::endl;

  fuseVolume(transform, true, color_thres);

  return success;
}

void DeformVoxelHierarchy::fuseVolume(const cugl::Affine3d &transform, bool is_first_frame, float4 color_thres)
{
  gpu::bindBlendingTextures(transform_array_,
                            *(node_block_hash_table_hierarchy_[0]),
                            voxel_blend_weight_table_,
                            voxel_blend_code_table_);

  gpu::fuseVolume(is_first_frame,
                  color_thres,
                  intrin_,
                  transform,
                  roi_,
                  trunc_dist_,
                  voxel_length_,
                  voxel_block_inner_dim_,
                  voxel_block_length_,
                  voxel_block_dim_,
                  node_block_dim_hierarchy_[0],
                  voxel_block_size_,
                  voxel_block_hash_array_,
                  voxel_block_hash_table_,
                  tsdf_voxel_block_array_,
                  rgba_voxel_block_array_);

  gpu::unbindBlendingTextures();
}

size_t DeformVoxelHierarchy::fetchIsoSurface(cugl::VertexArray &mesh_vertex_array,
                                             cugl::NormalArray &mesh_normal_array,
                                             cugl::ColorArray  &mesh_color_array)
{
  size_t mesh_vertex_size = 0;

  gpu::bindMarchingCubesTextures(tri_table_,
                                 num_verts_table_);

  gpu::bindTsdfVoxelTexture(tsdf_voxel_block_array_);

  gpu::bindRgbaVoxelTexture(rgba_voxel_block_array_);

  int occupied_voxel_size;
  occupied_voxel_size = gpu::getOccupiedVoxels(voxel_block_inner_dim_,
                                               voxel_block_dim_,
                                               voxel_block_size_,
                                               voxel_block_hash_array_,
                                               voxel_block_hash_table_,
                                               voxel_grid_hash_array_,
                                               vertex_num_array_,
                                               cube_tsdf_field_array_,
                                               cube_rgba_field_array_);

  gpu::unbindTsdfVoxelTexture();

  gpu::unbindRgbaVoxelTexture();

  if (!occupied_voxel_size)
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: No occupied voxel detected!" << std::endl;

    return mesh_vertex_size;
  }

  if (occupied_voxel_size >= gpu::MAX_TRIANGLES_SIZE)
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: Occupied voxel size exceeds the limitation!" << std::endl;

    return mesh_vertex_size;
  }

  mesh_vertex_size = gpu::computeOffsetsAndTotalVertices(occupied_voxel_size,
                                                         vertex_num_array_,
                                                         vertex_offset_array_);

  if (mesh_vertex_size >= (3 * gpu::MAX_TRIANGLES_SIZE))
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: Mesh vertex size exceeds the limitation!" << std::endl;

    mesh_vertex_size = 0;

    return mesh_vertex_size;
  }

  gpu::generateTriangles(volume_length_,
                         voxel_length_,
                         voxel_block_inner_dim_,
                         voxel_block_dim_,
                         occupied_voxel_size,
                         voxel_grid_hash_array_,
                         vertex_num_array_,
                         vertex_offset_array_,
                         cube_tsdf_field_array_,
                         cube_rgba_field_array_,
                         mesh_vertex_array,
                         mesh_normal_array,
                         mesh_color_array);

  gpu::unbindMarchingCubesTextures();

  return mesh_vertex_size;
}

size_t DeformVoxelHierarchy::fetchIsoSurface(cugl::VertexArray &mesh_vertex_array,
                                             cugl::NormalArray &mesh_normal_array,
                                             cugl::ColorArray  &mesh_color_array,
                                             cugl::VertexArray &surfel_vertex_array,
                                             cugl::NormalArray &surfel_normal_array,
                                             cugl::DeviceArray<float> &surfel_color_array)
{
  size_t mesh_vertex_size = 0;

  gpu::bindMarchingCubesTextures(tri_table_,
                                 num_verts_table_);

  gpu::bindTsdfVoxelTexture(tsdf_voxel_block_array_);

  gpu::bindRgbaVoxelTexture(rgba_voxel_block_array_);

  int occupied_voxel_size;
  occupied_voxel_size = gpu::getOccupiedVoxels(voxel_block_inner_dim_,
                                               voxel_block_dim_,
                                               voxel_block_size_,
                                               voxel_block_hash_array_,
                                               voxel_block_hash_table_,
                                               voxel_grid_hash_array_,
                                               vertex_num_array_,
                                               cube_tsdf_field_array_,
                                               cube_rgba_field_array_);

  gpu::unbindTsdfVoxelTexture();

  gpu::unbindRgbaVoxelTexture();

  if (!occupied_voxel_size)
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: No occupied voxel detected!" << std::endl;

    return mesh_vertex_size;
  }

  if (occupied_voxel_size >= gpu::MAX_TRIANGLES_SIZE)
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: Occupied voxel size exceeds the limitation!" << std::endl;

    return mesh_vertex_size;
  }

  mesh_vertex_size = gpu::computeOffsetsAndTotalVertices(occupied_voxel_size,
                                                         vertex_num_array_,
                                                         vertex_offset_array_);

  if (mesh_vertex_size >= (3 * gpu::MAX_TRIANGLES_SIZE))
  {
    gpu::unbindMarchingCubesTextures();

    std::cout << std::endl << "MarchingCubes: Mesh vertex size exceeds the limitation!" << std::endl;

    mesh_vertex_size = 0;

    return mesh_vertex_size;
  }

  gpu::generateTriangles(volume_length_,
                         voxel_length_,
                         voxel_block_inner_dim_,
                         voxel_block_dim_,
                         occupied_voxel_size,
                         voxel_grid_hash_array_,
                         vertex_num_array_,
                         vertex_offset_array_,
                         cube_tsdf_field_array_,
                         cube_rgba_field_array_,
                         mesh_vertex_array,
                         mesh_normal_array,
                         mesh_color_array,
                         surfel_vertex_array,
                         surfel_normal_array,
                         surfel_color_array);

  gpu::unbindMarchingCubesTextures();

  return mesh_vertex_size;
}

bool DeformVoxelHierarchy::update(size_t vertex_size,
                                  const cugl::VertexArray &vertex_array,
                                  const cugl::DeviceArray<uint> &active_flags)
{
  bool success = true;

  success = gpu::updateVoxelBlock(voxel_block_length_,
                                  voxel_block_dim_,
                                  vertex_size,
                                  vertex_array,
                                  active_flags,
                                  voxel_block_size_,
                                  pending_node_hash_array_,
                                  voxel_block_hash_array_,
                                  voxel_block_hash_table_);

  if (!success)
  {
    std::cout << std::endl << "DeformVoxelHierarchy: Update voxel block failed!" << std::endl;

    return success;
  }

  size_t node_size_old = node_size_;

  success = gpu::updateNodeBlockHierarchy(volume_length_,
                                          voxel_block_dim_,
                                          voxel_block_size_,
                                          voxel_block_hash_array_,
                                          0,
                                          node_block_length_hierarchy_[0],
                                          node_block_dim_hierarchy_[0],
                                          node_block_size_hierarchy_[0],
                                          pending_node_hash_array_,
                                          *(node_block_hash_array_hierarchy_[0]),
                                          *(node_block_hash_table_hierarchy_[0]),
                                          node_size_,
                                          node_hash_array_,
                                          node_array_,
                                          node_morton_code_array_);

  if (!success)
  {
    std::cout << std::endl << "DeformVoxelHierarchy: Update node block hierarchy failed! [level = " << 0 << "]" << std::endl;

    return success;
  }

  for (int level = 1; level < HIERARCHY_LEVELS; level++)
  {
    success = gpu::updateNodeBlockHierarchy(volume_length_,
                                            node_block_dim_hierarchy_[level-1],
                                            node_block_size_hierarchy_[level-1],
                                            *(node_block_hash_array_hierarchy_[level-1]),
                                            level,
                                            node_block_length_hierarchy_[level],
                                            node_block_dim_hierarchy_[level],
                                            node_block_size_hierarchy_[level],
                                            pending_node_hash_array_,
                                            *(node_block_hash_array_hierarchy_[level]),
                                            *(node_block_hash_table_hierarchy_[level]),
                                            node_size_,
                                            node_hash_array_,
                                            node_array_,
                                            node_morton_code_array_);

    if (!success)
    {
      std::cout << std::endl << "DeformVoxelHierarchy: Update node block hierarchy failed! [level = " << level << "]" << std::endl;

      return success;
    }
  }

  if (node_size_ > node_size_old)
  {
    gpu::sampleNewNodeTransform(voxel_length_,
                                voxel_block_inner_dim_,
                                voxel_block_length_,
                                voxel_block_dim_,
                                node_block_dim_hierarchy_[0],
                                node_size_old,
                                node_size_,
                                *(node_block_hash_table_hierarchy_[0]),
                                voxel_blend_weight_table_,
                                voxel_blend_code_table_,
                                node_array_,
                                transform_array_);

    gpu::reorderDeformGraph(node_size_,
                            node_hash_array_,
                            node_array_,
                            transform_array_,
                            node_morton_code_array_,
                            node_id_reorder_array_,
                            node_block_hash_table_hierarchy_handle_);
  }

  // std::cout << "DeformVoxelHierarchy: node_size = " << node_size_ << std::endl;

  return success;
}

void DeformVoxelHierarchy::warpSurface(const cugl::Affine3d &transform,
                                       size_t mesh_vertex_size,
                                       cugl::VertexArray &ref_mesh_vertex_array,
                                       cugl::NormalArray &ref_mesh_normal_array,
                                       cugl::ColorArray  &ref_mesh_color_array,
                                       cugl::VertexArray &warped_mesh_vertex_array,
                                       cugl::NormalArray &warped_mesh_normal_array,
                                       cugl::VertexArray &warped_surfel_vertex_array,
                                       cugl::NormalArray &warped_surfel_normal_array,
                                       cugl::DeviceArray<uint> &active_flags,
                                       float &res_sqrt)
{
  gpu::bindBlendingTextures(transform_array_,
                            *(node_block_hash_table_hierarchy_[0]),
                            voxel_blend_weight_table_,
                            voxel_blend_code_table_);

  DeformEnergyTerms::ENERGY_TERM term = DeformEnergyTerms::FIT_TERM;

  gpu::warpSurface(icp_param_,
                   intrin_,
                   transform,
                   volume_length_,
                   voxel_length_,
                   voxel_block_inner_dim_,
                   voxel_block_length_,
                   voxel_block_dim_,
                   node_block_dim_hierarchy_[0],
                   mesh_vertex_size,
                   det_ptr_->getParamList(term).w_huber,
                   ref_mesh_vertex_array,
                   ref_mesh_normal_array,
                   ref_mesh_color_array,
                   warped_mesh_vertex_array,
                   warped_mesh_normal_array,
                   warped_surfel_vertex_array,
                   warped_surfel_normal_array,
                   active_flags,
                   res_sqrt);

  gpu::unbindBlendingTextures();
}

void DeformVoxelHierarchy::formIcpGaussNewtonOpt(const cugl::Affine3d &transform,
                                                 size_t vertex_size,
                                                 const cugl::VertexArray &vertex_array,
                                                 const cugl::NormalArray &normal_array,
                                                 cugl::DeviceArray<float> &sum_buf,
                                                 double *mat_host,
                                                 double *vec_host)
{
  gpu::formIcpGaussNewtonOpt(icp_param_,
                             intrin_,
                             transform,
                             vertex_size,
                             vertex_array,
                             normal_array,
                             sum_buf);

  sum_buf.copy(cugl::DeviceArray<float>::DEVICE_TO_HOST);
  const float *sum_buf_host = sum_buf.getHostPtr();

  int shift = 0;
  for (int i = 0; i < 6; ++i)
  {
    for (int j = i; j < 7; ++j)
    {
      float value = sum_buf_host[shift++];

      if (j == 6)
        vec_host[i] = static_cast<double>(value);
      else
        mat_host[j * 6 + i] = mat_host[i * 6 + j] = static_cast<double>(value);
    }
  }
}

bool DeformVoxelHierarchy::initDeformEnergyTerms(const cugl::Affine3d &transform,
                                                 size_t vertex_size,
                                                 float &total_energy)
{
  // bind
  gpu::bindTransformTextures(transform_array_);

  // Fit term
  unsigned int elem_entry_start = 0;
  unsigned int elem_entry_size = static_cast<unsigned int>(vertex_size);
  unsigned int vec_block_entry_start = 0;
  unsigned int mat_block_entry_start = 0;

  DeformEnergyTerms::ENERGY_TERM term = DeformEnergyTerms::FIT_TERM;

  det_ptr_->setElemEntryStart(term, elem_entry_start);
  det_ptr_->setElemEntrySize(term, elem_entry_size);
  det_ptr_->setVecBlockEntryStart(term, vec_block_entry_start);
  det_ptr_->setMatBlockEntryStart(term, mat_block_entry_start);

  // bind
  gpu::bindBlendingTableTextures(voxel_blend_weight_table_,
                                 voxel_blend_code_table_);

  // bind
  gpu::bindBlockHashTableTextures(*(node_block_hash_table_hierarchy_[0]));

  float fit_term_energy = gpu::initFitTerm(icp_param_,
                                           intrin_,
                                           transform,
                                           vertex_size,
                                           voxel_length_,
                                           voxel_block_inner_dim_,
                                           voxel_block_length_,
                                           voxel_block_dim_,
                                           node_block_dim_hierarchy_[0],
                                           det_ptr_->getParamList(term),
                                           det_ptr_->getFitTermDataList(),
                                           det_ptr_->getBlockDataList());

  // unbind
  gpu::unbindBlockHashTableTextures();

  // unbind
  gpu::unbindBlendingTableTextures();

  // Reg Term
  elem_entry_start = 0;
  // elem_entry_size = static_cast<unsigned int>(node_size_ - node_block_size_hierarchy_[HIERARCHY_LEVELS - 1]);
  elem_entry_size = static_cast<unsigned int>(node_size_);
  vec_block_entry_start = vertex_size * det_ptr_->getVecBlockElemStride(term);
  mat_block_entry_start = vertex_size * det_ptr_->getMatBlockElemStride(term);

  term = DeformEnergyTerms::REG_TERM;

  det_ptr_->setElemEntryStart(term, elem_entry_start);
  det_ptr_->setElemEntrySize(term, elem_entry_size);
  det_ptr_->setVecBlockEntryStart(term, vec_block_entry_start);
  det_ptr_->setMatBlockEntryStart(term, mat_block_entry_start);

  int dim_ratio = node_block_dim_hierarchy_[0] / node_block_dim_hierarchy_[1];

  float reg_term_energy = 0.f;

  // bind
  gpu::bindBlendingTableTextures(node_blend_weight_table_,
                                 node_blend_code_table_);

  float w_level = 1.f;

  for (int level = 0; level < 1/*HIERARCHY_LEVELS - 1*/; level++)
  {
    // bind
    // gpu::bindBlockHashTableTextures(*(node_block_hash_table_hierarchy_[level + 1]));
    // TODO: only for test
    gpu::bindBlockHashTableTextures(*(node_block_hash_table_hierarchy_[level]));

    reg_term_energy += gpu::initRegTerm(w_level,
                                        dim_ratio,
                                        node_block_length_hierarchy_[level],
                                        node_block_dim_hierarchy_[level],
                                        node_block_size_hierarchy_[level],
                                        *(node_block_hash_array_hierarchy_[level]),
                                        /*(node_block_hash_table_hierarchy_[level]),*/
                                        det_ptr_->getParamList(term),
                                        det_ptr_->getRegTermDataList(),
                                        det_ptr_->getBlockDataList());

    w_level *= float(dim_ratio);

    elem_entry_start += node_block_size_hierarchy_[level];

    det_ptr_->setElemEntryStart(term, elem_entry_start);

    // unbind
    gpu::unbindBlockHashTableTextures();
  }

  // unbind
  gpu::unbindBlendingTableTextures();

  // unbind
  gpu::unbindTransformTextures();

  total_energy = fit_term_energy + reg_term_energy;

  bool success = det_ptr_->initGaussNewtonOpt();

  if (!success)
  {
    std::cout << std::endl << "DeformVoxelHierarchy: init DET Gauss Newton Opt failed!" << std::endl;
    return false;
  }

  return true;
}

void DeformVoxelHierarchy::updateDeformEnergyTerms(const cugl::Affine3d &transform,
                                                   size_t vertex_size,
                                                   float &total_energy)
{
  // bind
  gpu::bindTransformTextures(transform_array_);

  // Fit term
  unsigned int elem_entry_start = 0;

  DeformEnergyTerms::ENERGY_TERM term = DeformEnergyTerms::FIT_TERM;

  det_ptr_->setElemEntryStart(term, elem_entry_start);

  float fit_term_energy = gpu::updateFitTerm(icp_param_,
                                             intrin_,
                                             transform,
                                             vertex_size,
                                             det_ptr_->getParamList(term),
                                             det_ptr_->getFitTermDataList());

  // Reg Term
  term = DeformEnergyTerms::REG_TERM;

  elem_entry_start = 0;

  det_ptr_->setElemEntryStart(term, elem_entry_start);

  float reg_term_energy = 0.f;

  for (int level = 0; level < 1/*HIERARCHY_LEVELS - 1*/; level++)
  {
    reg_term_energy += gpu::updateRegTerm(node_block_size_hierarchy_[level],
                                          det_ptr_->getParamList(term),
                                          det_ptr_->getRegTermDataList());

    elem_entry_start += node_block_size_hierarchy_[level];

    det_ptr_->setElemEntryStart(term, elem_entry_start);
  }

  // unbind
  gpu::unbindTransformTextures();

  total_energy = fit_term_energy + reg_term_energy;
}

void DeformVoxelHierarchy::formDetGaussNewtonOpt()
{
  // bind
  gpu::bindTransformTextures(transform_array_);

  det_ptr_->formGaussNewtonOpt();

  // unbind
  gpu::unbindTransformTextures();
}

bool DeformVoxelHierarchy::solveDetGaussNewtonOpt(float damping)
{
  return (det_ptr_->solveGaussNewtonOpt(damping));
}

void DeformVoxelHierarchy::initTransformSearch()
{
  gpu::copyTransformArray(node_size_,
                          transform_array_);
}

void DeformVoxelHierarchy::runTransformSearchOnce(float step_size)
{
  transform_array_.swap();

  det_ptr_->updateTransformArray(step_size,
                                 transform_array_);

  transform_array_.swap();
}

void DeformVoxelHierarchy::saveDetGaussNewtonOpt(int frame_count)
{
  det_ptr_->saveGaussNewtonOpt(frame_count);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Private Methods
 */
void DeformVoxelHierarchy::alloc(int image_width, int image_height)
{
  // Voxel Block Hashing
  int voxel_block_inner_hash_size = voxel_block_inner_dim_ * voxel_block_inner_dim_ * voxel_block_inner_dim_;

  int voxel_block_hash_size = voxel_block_dim_ * voxel_block_dim_ * voxel_block_dim_;

  int voxel_block_array_size = (gpu::MAX_VOXEL_BLOCK_SIZE * voxel_block_inner_hash_size);

  voxel_block_hash_array_.alloc((size_t)gpu::MAX_VOXEL_BLOCK_SIZE);

  voxel_block_hash_table_.alloc((size_t)voxel_block_hash_size);

  tsdf_voxel_block_array_.alloc((size_t)voxel_block_array_size);

  rgba_voxel_block_array_.alloc((size_t)voxel_block_array_size);

  int dim_ratio = voxel_block_dim_ / node_block_dim_hierarchy_[0];

  int voxel_blend_pass_num = dim_ratio * dim_ratio * dim_ratio;

  int voxel_blend_table_size = voxel_block_inner_hash_size * voxel_blend_pass_num * gpu::VOXEL_BLENDING_MAX_QUAD_STRIDE;

  voxel_blend_weight_table_.alloc((size_t)voxel_blend_table_size);

  voxel_blend_code_table_.alloc((size_t)voxel_blend_table_size);

  // Node Block Hierarchy
  for (int level = 0; level < HIERARCHY_LEVELS; level++)
  {
    HashArrayHandle hash_array_handle = std::make_shared< cugl::DeviceArray<unsigned int> >();
    hash_array_handle->alloc((size_t)gpu::MAX_GRAPH_NODES_SIZE);

    node_block_hash_array_hierarchy_.push_back(std::move(hash_array_handle));

    int node_block_dim = node_block_dim_hierarchy_[level];
    int node_block_hash_size = node_block_dim * node_block_dim * node_block_dim;

    HashTableHandle hash_table_handle = std::make_shared< cugl::DeviceArray<gpu::HashEntryType> >();
    hash_table_handle->alloc((size_t)node_block_hash_size);

    node_block_hash_table_hierarchy_.push_back(std::move(hash_table_handle));
  }

  std::vector<gpu::HashEntryType *> hierarchy_handle_host;
  for (int level = 0; level < HIERARCHY_LEVELS; level++)
  {
    hierarchy_handle_host.push_back((node_block_hash_table_hierarchy_[level])->getDevicePtr());
  }
  node_block_hash_table_hierarchy_handle_.alloc((size_t)HIERARCHY_LEVELS);
  node_block_hash_table_hierarchy_handle_.upload(hierarchy_handle_host.data());

  // const int node_blend_pass_num = 8;
  // TODO: only for test
  const int node_blend_pass_num = 1;
  const int node_blend_table_size = node_blend_pass_num * gpu::NODE_BLENDING_MAX_QUAD_STRIDE;

  node_blend_weight_table_.alloc((size_t)node_blend_table_size);

  node_blend_code_table_.alloc((size_t)node_blend_table_size);

  // Deformation Graph
  node_hash_array_.alloc((size_t)gpu::MAX_GRAPH_NODES_SIZE, false, false , true); // use double buffer

  node_array_.alloc((size_t)gpu::MAX_GRAPH_NODES_SIZE, false, false, true); // use double buffer

  // We use 4x4 affine transformation matrix here
  transform_array_.alloc((size_t)(gpu::MAX_GRAPH_NODES_SIZE * 4), false, false, true); // use double buffer

  node_morton_code_array_.alloc((size_t)gpu::MAX_GRAPH_NODES_SIZE);

  node_id_reorder_array_.alloc((size_t)gpu::MAX_GRAPH_NODES_SIZE);

  // Marching Cubes
  voxel_grid_hash_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  vertex_num_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  vertex_offset_array_.alloc((size_t)gpu::MAX_TRIANGLES_SIZE);

  cube_tsdf_field_array_.alloc((size_t)(gpu::MAX_TRIANGLES_SIZE * 2));

  cube_rgba_field_array_.alloc((size_t)(gpu::MAX_TRIANGLES_SIZE * 2));

  tri_table_.alloc(256 * 16);
  tri_table_.upload(&tri_table[0][0]);

  num_verts_table_.alloc(256);
  num_verts_table_.upload(num_verts_table);

  // Others
  int pending_node_hash_array_size = std::max(std::max(image_width * image_height, static_cast<int>(gpu::MAX_TRIANGLES_SIZE)), static_cast<int>(gpu::MAX_VOXEL_BLOCK_SIZE) * 27); // 3D block neighbor size = 27
  pending_node_hash_array_.alloc((size_t)pending_node_hash_array_size, false, false, true); // use double buffer
}

void DeformVoxelHierarchy::free()
{
  voxel_block_hash_array_.free();

  voxel_block_hash_table_.free();

  tsdf_voxel_block_array_.free();

  rgba_voxel_block_array_.free();

  voxel_blend_weight_table_.free();

  voxel_blend_code_table_.free();


  for (int level = 0; level < HIERARCHY_LEVELS; level++)
  {
    HashArrayHandle hash_array_handle = node_block_hash_array_hierarchy_[level];

    hash_array_handle->free();

    HashTableHandle hash_table_handle = node_block_hash_table_hierarchy_[level];

    hash_table_handle->free();
  }

  node_blend_weight_table_.free();

  node_blend_code_table_.free();


  node_hash_array_.free();

  node_array_.free();

  transform_array_.free();

  node_morton_code_array_.free();

  node_id_reorder_array_.free();


  voxel_grid_hash_array_.free();

  vertex_num_array_.free();

  vertex_offset_array_.free();

  cube_tsdf_field_array_.free();

  cube_rgba_field_array_.free();

  tri_table_.free();

  num_verts_table_.free();


  pending_node_hash_array_.free();
}

void DeformVoxelHierarchy::generateBlendingTable()
{
  gpu::generateVoxelBlendingTable(voxel_length_,
                                  voxel_block_inner_dim_,
                                  voxel_block_length_,
                                  voxel_block_dim_,
                                  node_block_length_hierarchy_[0],
                                  node_block_dim_hierarchy_[0],
                                  voxel_blend_weight_table_,
                                  voxel_blend_code_table_);

  // gpu::generateNodeBlendingTable(node_block_length_hierarchy_[0],
  //                                node_block_dim_hierarchy_[0],
  //                                node_block_length_hierarchy_[1],
  //                                node_block_dim_hierarchy_[1],
  //                                node_blend_weight_table_,
  //                                node_blend_code_table_);

  // TODO: only for test
  gpu::generateNodeBlendingTable(node_block_length_hierarchy_[0],
                                 node_block_dim_hierarchy_[0],
                                 node_block_length_hierarchy_[0],
                                 node_block_dim_hierarchy_[0],
                                 node_blend_weight_table_,
                                 node_blend_code_table_);
}
} // namespace dynfu
