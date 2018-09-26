#include <cugl_ros/apps/dynamic_fusion/deform_energy_terms.h>

namespace dynfu
{
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Public Types
 */
DeformEnergyTerms::DeformEnergyTerms(size_t block_dim, float w_sqrt[TOTAL_TERM_SIZE], float w_huber[TOTAL_TERM_SIZE])
    : block_dim_(block_dim)
{
  // FIT_TERM
  gpu::DetParamList fit_term_param_list;
  fit_term_param_list.w_sqrt = w_sqrt[FIT_TERM];
  fit_term_param_list.w_huber = w_huber[FIT_TERM];

  fit_term_param_list.term_num = FIT_TERM;

  fit_term_param_list.elem_entry_start = 0;
  fit_term_param_list.elem_entry_size = 0;

  fit_term_param_list.vec_block_entry_start = 0;
  fit_term_param_list.mat_block_entry_start = 0;

  unsigned int vec_block_elem_stride = 4 * gpu::VOXEL_BLENDING_QUAD_STRIDE;
  unsigned int mat_block_elem_stride = vec_block_elem_stride * (vec_block_elem_stride - 1) / 2;

  fit_term_param_list.vec_block_elem_stride = vec_block_elem_stride;
  fit_term_param_list.mat_block_elem_stride = mat_block_elem_stride;

  param_lists_.push_back(fit_term_param_list);

  // REG_TERM
  gpu::DetParamList reg_term_param_list;
  reg_term_param_list.w_sqrt = w_sqrt[REG_TERM];
  reg_term_param_list.w_huber = w_huber[REG_TERM];

  reg_term_param_list.term_num = REG_TERM;

  reg_term_param_list.elem_entry_start = 0;
  reg_term_param_list.elem_entry_size = 0;

  reg_term_param_list.vec_block_entry_start = 0;
  reg_term_param_list.mat_block_entry_start = 0;

  vec_block_elem_stride = 4 * gpu::NODE_BLENDING_QUAD_STRIDE * 2;
  mat_block_elem_stride = 4 * gpu::NODE_BLENDING_QUAD_STRIDE;

  reg_term_param_list.vec_block_elem_stride = vec_block_elem_stride;
  reg_term_param_list.mat_block_elem_stride = mat_block_elem_stride;

  param_lists_.push_back(reg_term_param_list);

  pcg_ptr_ = PcgSolver::Ptr(new PcgSolver(block_dim_));

  alloc();
}

DeformEnergyTerms::~DeformEnergyTerms()
{
  free();
}

bool DeformEnergyTerms::initGaussNewtonOpt()
{
  uint vec_block_elem_size = param_lists_[FIT_TERM].elem_entry_size * param_lists_[FIT_TERM].vec_block_elem_stride;
  vec_block_elem_size += param_lists_[REG_TERM].elem_entry_size * param_lists_[REG_TERM].vec_block_elem_stride;

  bool success = gpu::initVecBlock(vec_block_elem_size,
                                   vec_block_size_,
                                   vec_block_elem_ids_,
                                   *(vec_block_elem_entry_codes_[TOTAL_TERM_SIZE]),
                                   vec_block_ids_,
                                   vec_block_elem_starts_,
                                   *(vec_block_elem_sizes_[TOTAL_TERM_SIZE]),
                                   block_id2row_lut_);

  if (!success)
  {
    std::cout << std::endl << "DeformEnergyTerms: Init vec block failed!" << std::endl;
    return false;
  }

  gpu::classifyEnergyTerms(vec_block_size_,
                           vec_block_elem_starts_,
                           *(vec_block_elem_sizes_[TOTAL_TERM_SIZE]),
                           *(vec_block_elem_entry_codes_[TOTAL_TERM_SIZE]),
                           *(vec_block_elem_sizes_[FIT_TERM]),
                           *(vec_block_elem_entry_codes_[FIT_TERM]),
                           *(vec_block_elem_sizes_[REG_TERM]),
                           *(vec_block_elem_entry_codes_[REG_TERM]));


  uint mat_block_elem_size = param_lists_[FIT_TERM].elem_entry_size * param_lists_[FIT_TERM].mat_block_elem_stride;
  mat_block_elem_size += param_lists_[REG_TERM].elem_entry_size * param_lists_[REG_TERM].mat_block_elem_stride;

  success = gpu::initMatBlock(mat_block_elem_size,
                              vec_block_size_,
                              block_id2row_lut_,
                              mat_block_size_,
                              mat_block_elem_ids_,
                              *(mat_block_elem_entry_codes_[TOTAL_TERM_SIZE]),
                              mat_block_ids_,
                              mat_block_elem_starts_,
                              *(mat_block_elem_sizes_[TOTAL_TERM_SIZE]),
                              mat_block_ext_ids_,
                              mat_block_entries_,
                              mat_block_row_lengths_,
                              mat_block_row_offsets_,
                              mat_block_coos_);

  if (!success)
  {
    std::cout << std::endl << "DeformEnergyTerms: Init mat block failed!" << std::endl;
    return false;
  }

  gpu::classifyEnergyTerms(mat_block_size_,
                           mat_block_elem_starts_,
                           *(mat_block_elem_sizes_[TOTAL_TERM_SIZE]),
                           *(mat_block_elem_entry_codes_[TOTAL_TERM_SIZE]),
                           *(mat_block_elem_sizes_[FIT_TERM]),
                           *(mat_block_elem_entry_codes_[FIT_TERM]),
                           *(mat_block_elem_sizes_[REG_TERM]),
                           *(mat_block_elem_entry_codes_[REG_TERM]));

  success = gpu::initSparseMat(vec_block_size_,
                               mat_block_size_,
                               mat_block_row_lengths_,
                               mat_block_coos_,
                               mat_block_write_entries_,
                               pcg_ptr_->getSpMatParamList(),
                               pcg_ptr_->getSpMatDataList());

  if (!success)
  {
    std::cout << std::endl << "DeformEnergyTerms: Init sparse mat failed!" << std::endl;
    return false;
  }

  return true;
}

void DeformEnergyTerms::formGaussNewtonOpt()
{
  ////////////////////////////////////////
  // Fit Term
  ////////////////////////////////////////
  gpu::bindTermElemTextures(fit_term_elem_data_,
                            fit_term_elem_ids_,
                            fit_term_elem_weights_);

  gpu::formFitTermVecAndMatDiag(param_lists_[FIT_TERM],
                                vec_block_size_,
                                vec_block_elem_starts_,
                                *(vec_block_elem_sizes_[FIT_TERM]),
                                *(vec_block_elem_entry_codes_[FIT_TERM]),
                                pcg_ptr_->getSpMatParamList(),
                                pcg_ptr_->getVecData(),
                                pcg_ptr_->getSpMatDataList());

  gpu::formFitTermMat(param_lists_[FIT_TERM],
                      mat_block_size_,
                      mat_block_elem_starts_,
                      *(mat_block_elem_sizes_[FIT_TERM]),
                      *(mat_block_elem_entry_codes_[FIT_TERM]),
                      mat_block_coos_,
                      mat_block_write_entries_,
                      pcg_ptr_->getSpMatParamList(),
                      pcg_ptr_->getSpMatDataList());

  gpu::unbindTermElemTextures();

  ////////////////////////////////////////
  // Reg Term
  ////////////////////////////////////////
  gpu::bindTermElemTextures(reg_term_elem_data_,
                            reg_term_elem_ids_,
                            reg_term_elem_weights_);

  gpu::formRegTermVecAndMatDiag(param_lists_[REG_TERM],
                                vec_block_size_,
                                vec_block_elem_starts_,
                                *(vec_block_elem_sizes_[REG_TERM]),
                                *(vec_block_elem_entry_codes_[REG_TERM]),
                                pcg_ptr_->getSpMatParamList(),
                                pcg_ptr_->getVecData(),
                                pcg_ptr_->getSpMatDataList());

  gpu::formRegTermMat(param_lists_[REG_TERM],
                      mat_block_size_,
                      mat_block_elem_starts_,
                      *(mat_block_elem_sizes_[REG_TERM]),
                      *(mat_block_elem_entry_codes_[REG_TERM]),
                      mat_block_coos_,
                      mat_block_write_entries_,
                      pcg_ptr_->getSpMatParamList(),
                      pcg_ptr_->getSpMatDataList());

  gpu::unbindTermElemTextures();
}

bool DeformEnergyTerms::solveGaussNewtonOpt(float damping)
{
  const float epsilon = 0.001f;
  const int iter_end = 50; // 10 ~ 20

  return (pcg_ptr_->run(damping, epsilon, iter_end));
}

void DeformEnergyTerms::updateTransformArray(float step_size,
                                             cugl::DeviceArray<gpu::TransformType> &transform_array)
{
  gpu::updateTransformArray(step_size,
                            vec_block_size_,
                            vec_block_ids_,
                            pcg_ptr_->getResult(),
                            transform_array);
}

void DeformEnergyTerms::saveGaussNewtonOpt(int time_stamp)
{
  // save off-diagonal mat
  std::stringstream ss_0;
  std::string file_name_0;
  ss_0 << "sp_mat_" << time_stamp << ".txt";
  file_name_0 = ss_0.str();
  pcg_ptr_->saveSpMat(file_name_0);

  // save diagonal mat
  std::stringstream ss_1;
  std::string file_name_1;
  ss_1 << "diag_mat_" << time_stamp << ".txt";
  file_name_1 = ss_1.str();
  pcg_ptr_->saveDiagMat(file_name_1);

  // save vec
  std::stringstream ss_2;
  std::string file_name_2;
  ss_2 << "vec_" << time_stamp << ".txt";
  file_name_2 = ss_2.str();
  pcg_ptr_->saveVec(file_name_2);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Private Types
 */
void DeformEnergyTerms::alloc()
{
  size_t vec_block_dim = block_dim_;

  size_t mat_block_dim = block_dim_ * block_dim_;

  size_t max_vec_block_elem_size = gpu::MAX_TRIANGLES_SIZE * param_lists_[FIT_TERM].vec_block_elem_stride;
  max_vec_block_elem_size += gpu::MAX_GRAPH_NODES_SIZE * param_lists_[REG_TERM].vec_block_elem_stride;

  size_t max_mat_block_elem_size = gpu::MAX_TRIANGLES_SIZE * param_lists_[FIT_TERM].mat_block_elem_stride;
  max_mat_block_elem_size += gpu::MAX_GRAPH_NODES_SIZE * param_lists_[REG_TERM].mat_block_elem_stride;

  fit_term_elem_data_.alloc(gpu::MAX_TRIANGLES_SIZE);
  fit_term_data_list_.elem_data = fit_term_elem_data_.getDevicePtr();

  fit_term_elem_ids_.alloc(gpu::MAX_TRIANGLES_SIZE * param_lists_[FIT_TERM].vec_block_elem_stride);
  fit_term_data_list_.elem_ids = fit_term_elem_ids_.getDevicePtr();

  fit_term_elem_weights_.alloc(fit_term_elem_ids_.getSize());
  fit_term_data_list_.elem_weights = fit_term_elem_weights_.getDevicePtr();

  reg_term_elem_data_.alloc(gpu::MAX_GRAPH_NODES_SIZE);
  reg_term_data_list_.elem_data = reg_term_elem_data_.getDevicePtr();

  reg_term_elem_ids_.alloc(gpu::MAX_GRAPH_NODES_SIZE * param_lists_[REG_TERM].vec_block_elem_stride);
  reg_term_data_list_.elem_ids = reg_term_elem_ids_.getDevicePtr();

  reg_term_elem_weights_.alloc(reg_term_elem_ids_.getSize());
  reg_term_data_list_.elem_weights = reg_term_elem_weights_.getDevicePtr();

  vec_block_elem_ids_.alloc(max_vec_block_elem_size);
  block_data_list_.vec_block_elem_ids = vec_block_elem_ids_.getDevicePtr();

  mat_block_elem_ids_.alloc(max_mat_block_elem_size);
  block_data_list_.mat_block_elem_ids = mat_block_elem_ids_.getDevicePtr();

  for (int term = 0; term <= TOTAL_TERM_SIZE; term++)
  {
    UintArrayHandle code_array_handle = std::make_shared< cugl::DeviceArray<unsigned int> >();
    code_array_handle->alloc(max_vec_block_elem_size);

    vec_block_elem_entry_codes_.push_back(std::move(code_array_handle));
  }
  block_data_list_.vec_block_elem_entry_codes = (vec_block_elem_entry_codes_[TOTAL_TERM_SIZE])->getDevicePtr();

  for (int term = 0; term <= TOTAL_TERM_SIZE; term++)
  {
    UintArrayHandle code_array_handle = std::make_shared< cugl::DeviceArray<unsigned int> >();
    code_array_handle->alloc(max_mat_block_elem_size);

    mat_block_elem_entry_codes_.push_back(std::move(code_array_handle));
  }
  block_data_list_.mat_block_elem_entry_codes = (mat_block_elem_entry_codes_[TOTAL_TERM_SIZE])->getDevicePtr();

  block_id2row_lut_.alloc(gpu::MAX_GRAPH_NODES_SIZE);

  vec_block_ids_.alloc(gpu::MAX_GRAPH_NODES_SIZE + 1);

  for (int term = 0; term <= TOTAL_TERM_SIZE; term++)
  {
    UintArrayHandle size_array_handle = std::make_shared< cugl::DeviceArray<unsigned int> >();
    size_array_handle->alloc(gpu::MAX_GRAPH_NODES_SIZE + 1);

    vec_block_elem_sizes_.push_back(std::move(size_array_handle));
  }

  vec_block_elem_starts_.alloc(gpu::MAX_GRAPH_NODES_SIZE + 1);

  mat_block_ids_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE / 2 + 1);

  for (int term = 0; term <= TOTAL_TERM_SIZE; term++)
  {
    UintArrayHandle size_array_handle = std::make_shared< cugl::DeviceArray<unsigned int> >();
    size_array_handle->alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE / 2 + 1);

    mat_block_elem_sizes_.push_back(std::move(size_array_handle));
  }

  mat_block_elem_starts_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE / 2 + 1);

  mat_block_ext_ids_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE);

  mat_block_entries_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE);

  mat_block_row_lengths_.alloc(gpu::MAX_GRAPH_NODES_SIZE);

  mat_block_row_offsets_.alloc(gpu::MAX_GRAPH_NODES_SIZE);

  mat_block_coos_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE);

  mat_block_write_entries_.alloc(gpu::MAX_NON_ZERO_BLOCKS_SIZE * vec_block_dim);
}

void DeformEnergyTerms::free()
{
  fit_term_elem_data_.free();
  fit_term_elem_ids_.free();
  fit_term_elem_weights_.free();

  reg_term_elem_data_.free();
  reg_term_elem_ids_.free();
  reg_term_elem_weights_.free();

  vec_block_elem_ids_.free();
  mat_block_elem_ids_.free();

  block_id2row_lut_.free();
  vec_block_ids_.free();
  vec_block_elem_starts_.free();

  mat_block_ids_.free();
  mat_block_elem_starts_.free();
  mat_block_ext_ids_.free();
  mat_block_entries_.free();
  mat_block_row_lengths_.free();
  mat_block_row_offsets_.free();
  mat_block_coos_.free();
  mat_block_write_entries_.free();
}
} // namespace dynfu
