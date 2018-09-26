#ifndef __CUGL_APPS_DYNFU_DEFORM_ENERGY_TERMS_H__
#define __CUGL_APPS_DYNFU_DEFORM_ENERGY_TERMS_H__

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "cuda_internal.h"
#include "pcg_solver.h"

namespace dynfu
{
/*
 * Class of the deformation energy terms.
 *
 * Author: Tao Han, City University of Hong Kong (tao.han@my.cityu.edu.hk)
 */
class DeformEnergyTerms
{
 public:
  /*
   * Public Types
   */
  typedef std::shared_ptr<DeformEnergyTerms> Ptr;

  enum ENERGY_TERM
  {
    FIT_TERM = 0,
    REG_TERM = 1,
    TOTAL_TERM_SIZE = 2
  };

  /*
   * Public Methods
   */
  DeformEnergyTerms(size_t block_dim, float w_sqrt[TOTAL_TERM_SIZE], float w_huber[TOTAL_TERM_SIZE]);

  ~DeformEnergyTerms();

  gpu::DetParamList &getParamList(ENERGY_TERM term)
  {
    return param_lists_[term];
  }

  gpu::FitTermDataList &getFitTermDataList()
  {
    return fit_term_data_list_;
  }

  gpu::RegTermDataList &getRegTermDataList()
  {
    return reg_term_data_list_;
  }

  gpu::DetBlockDataList &getBlockDataList()
  {
    return block_data_list_;
  }

  unsigned int getVecBlockElemStride(ENERGY_TERM term) const
  {
    return param_lists_[term].vec_block_elem_stride;
  }

  unsigned int getMatBlockElemStride(ENERGY_TERM term) const
  {
    return param_lists_[term].mat_block_elem_stride;
  }

  // unsigned int getVecBlockElemOffset(ENERGY_TERM term) const
  // {
  //   return param_lists_[term].vec_block_elem_offset;
  // }

  // unsigned int getMatBlockElemOffset(ENERGY_TERM term) const
  // {
  //   return param_lists_[term].mat_block_elem_offset;
  // }

  void setElemEntryStart(ENERGY_TERM term, unsigned int start)
  {
    param_lists_[term].elem_entry_start = start;
  }

  void setElemEntrySize(ENERGY_TERM term, unsigned int size)
  {
    param_lists_[term].elem_entry_size = size;
  }

  void setVecBlockEntryStart(ENERGY_TERM term, unsigned int start)
  {
    param_lists_[term].vec_block_entry_start = start;
  }

  void setMatBlockEntryStart(ENERGY_TERM term, unsigned int start)
  {
    param_lists_[term].mat_block_entry_start = start;
  }

  ////////////////////////////////////////////////////////////
  bool initGaussNewtonOpt();

  void formGaussNewtonOpt();

  bool solveGaussNewtonOpt(float damping);

  void updateTransformArray(float step_size,
                            cugl::DeviceArray<gpu::TransformType> &transform_array);

  ////////////////////////////////////////////////////////////
  void saveGaussNewtonOpt(int time_stamp);

 private:
  /*
   * Private Types
   */
  typedef std::shared_ptr< cugl::DeviceArray<unsigned int> > UintArrayHandle;

  std::vector<gpu::DetParamList> param_lists_;

  gpu::FitTermDataList fit_term_data_list_;

  gpu::RegTermDataList reg_term_data_list_;

  gpu::DetBlockDataList block_data_list_;

  size_t block_dim_;

  /*
   * Energy Term Data
   */
  cugl::DeviceArray<float4> fit_term_elem_data_;

  cugl::DeviceArray<unsigned int> fit_term_elem_ids_;

  cugl::DeviceArray<float> fit_term_elem_weights_;

  cugl::DeviceArray<float4> reg_term_elem_data_;

  cugl::DeviceArray<unsigned int> reg_term_elem_ids_;

  cugl::DeviceArray<float> reg_term_elem_weights_;

  cugl::DeviceArray<unsigned int> vec_block_elem_ids_;

  cugl::DeviceArray<unsigned int> mat_block_elem_ids_;

  std::vector<UintArrayHandle> vec_block_elem_entry_codes_;

  std::vector<UintArrayHandle> mat_block_elem_entry_codes_;


  /*
   * Normal Equation in Gauss Newton Optimization [ Ax = b ]
   */
  cugl::DeviceArray<unsigned int> block_id2row_lut_;

  size_t vec_block_size_;

  cugl::DeviceArray<unsigned int> vec_block_ids_;

  std::vector<UintArrayHandle> vec_block_elem_sizes_;

  cugl::DeviceArray<unsigned int> vec_block_elem_starts_;

  size_t mat_block_size_;

  cugl::DeviceArray<unsigned int> mat_block_ids_;

  std::vector<UintArrayHandle> mat_block_elem_sizes_;

  cugl::DeviceArray<unsigned int> mat_block_elem_starts_;

  cugl::DeviceArray<unsigned int> mat_block_ext_ids_;

  cugl::DeviceArray<unsigned int> mat_block_entries_;

  cugl::DeviceArray<unsigned int> mat_block_row_lengths_;

  cugl::DeviceArray<unsigned int> mat_block_row_offsets_;

  cugl::DeviceArray<unsigned int> mat_block_coos_;

  cugl::DeviceArray<unsigned int> mat_block_write_entries_; // mat_block_spmat_entries_;

  /*
   * PCG Solver
   */
  PcgSolver::Ptr pcg_ptr_;

  /*
   * Private Methods
   */
  void alloc();

  void free();
};

} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_DEFORM_ENERGY_TERMS_H__ */
