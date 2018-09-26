#ifndef __CUGL_APPS_DYNFU_PCG_SOLVER_H__
#define __CUGL_APPS_DYNFU_PCG_SOLVER_H__

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "cuda_internal.h"

namespace dynfu
{
/*
 * Wrapper class of a pre-conditioned conjugate gradient solver.
 *
 * Ax = b
 *
 * Author: Tao Han, City University of Hong Kong (tao.han@my.cityu.edu.hk)
 */
class PcgSolver
{
 public:
  /*
   * Public Types
   */
  typedef std::shared_ptr<PcgSolver> Ptr;

  /*
   * Public Methods
   */
  PcgSolver(size_t block_dim);

  ~PcgSolver();

  gpu::SpMatParamList &getSpMatParamList()
  {
    return spmat_param_list_;
  }

  gpu::SpMatDataList &getSpMatDataList()
  {
    return spmat_data_list_;
  }

  cugl::DeviceArray<float> &getVecData()
  {
    return vec_data_;
  }

  cugl::DeviceArray<float> &getResult()
  {
    return x_;
  }

  bool run(float damping, float epsilon, int iter_end);

  void saveSpMat(std::string file_name);

  void saveDiagMat(std::string file_name);

  void saveVec(std::string file_name);

 private:
  /*
   * Private Members
   */

  // Sparse matrix [ A ]
  gpu::SpMatParamList spmat_param_list_;

  gpu::SpMatDataList spmat_data_list_;

  cugl::DeviceArray<unsigned int> spmat_row_lengths_;

  cugl::DeviceArray<unsigned int> spmat_bin_lengths_;

  cugl::DeviceArray<unsigned int> spmat_bin_offsets_;

  cugl::DeviceArray<unsigned int> spmat_offsets_;

  cugl::DeviceArray<float> spmat_diagonal_;

  cugl::DeviceArray<float> spmat_precond_;

  cugl::DeviceArray<unsigned int> spmat_col_indices_;

  cugl::DeviceArray<float> spmat_data_;

  // Vector [ b ]
  cugl::DeviceArray<float> vec_data_;

  // Vector [ x ]
  cugl::DeviceArray<float> x_;

  // Other variables in solver
  cugl::DeviceArray<float> y_;

  cugl::DeviceArray<float> r_;

  cugl::DeviceArray<float> z_;

  cugl::DeviceArray<float> p_;

  /*
   * Private Methods
   */
  void alloc();

  void free();
};

} // namespace dynfu

#endif /* __CUGL_APPS_DYNFU_PCG_SOLVER_H__ */
