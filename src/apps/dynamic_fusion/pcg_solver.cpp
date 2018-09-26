#include <cugl_ros/apps/dynamic_fusion/pcg_solver.h>

namespace dynfu
{
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Public Types
 */
PcgSolver::PcgSolver(size_t block_dim)
{
  spmat_param_list_.bin_width = 32;
  spmat_param_list_.block_dim = block_dim;
  spmat_param_list_.rows = 0;
  spmat_param_list_.pad_rows = 0;

  alloc();
}

PcgSolver::~PcgSolver()
{
  free();
}

bool PcgSolver::run(float damping, float epsilon, int iter_end)
{
  gpu::computePreconditioner(damping,
                             spmat_param_list_,
                             spmat_data_list_);

  float delta_init = gpu::initPcgSolver(spmat_param_list_,
                                        spmat_data_list_,
                                        vec_data_,
                                        x_, r_, z_, p_);

  float delta_old = delta_init;
  float delta_new = delta_init;
  int iter = 0;

  const float delta_final = epsilon * epsilon * delta_init;

  while( (iter < iter_end) && (delta_new > delta_final) )
  {
    delta_new = gpu::iteratePcgSolver(delta_old,
                                      spmat_param_list_,
                                      spmat_data_list_,
                                      vec_data_,
                                      x_, y_, r_, z_, p_);

    delta_old = delta_new;
    iter++;
  }

  return (!std::isnan(delta_new));
}

void PcgSolver::saveSpMat(std::string file_name)
{
  spmat_offsets_.copy(cugl::DeviceArray<unsigned int>::DEVICE_TO_HOST);
  const unsigned int *offsets_host = spmat_offsets_.getHostPtr();

  spmat_col_indices_.copy(cugl::DeviceArray<unsigned int>::DEVICE_TO_HOST);
  const unsigned int *col_indices_host = spmat_col_indices_.getHostPtr();

  spmat_data_.copy(cugl::DeviceArray<float>::DEVICE_TO_HOST);
  const float *data_host = spmat_data_.getHostPtr();

  std::ofstream os(file_name.c_str());

  const int bin_width = 32;

  if (os.is_open())
  {
    for (int row = 0; row < spmat_param_list_.rows; row++)
    {
      unsigned int offset_begin = offsets_host[row];
      unsigned int offset_end = offsets_host[row + bin_width];

      for (unsigned int k = offset_begin; k < offset_end; k += bin_width)
      {
        unsigned int col = col_indices_host[k];
        float data = data_host[k];

        if (col != 0xFFFFFFFF)
        {
          os << "<" << row << ", " << col << "> " << data << std::endl;
        }
        else
        {
          break;
        }
      }
    }
  }

  os.close();
}

void PcgSolver::saveDiagMat(std::string file_name)
{
  spmat_diagonal_.copy(cugl::DeviceArray<float>::DEVICE_TO_HOST);
  const float *diagonal_host = spmat_diagonal_.getHostPtr();

  spmat_precond_.copy(cugl::DeviceArray<float>::DEVICE_TO_HOST);
  const float *precond_host = spmat_precond_.getHostPtr();

  std::ofstream os(file_name.c_str());

  if (os.is_open())
  {
    for (int i = 0; i < spmat_param_list_.rows; i++)
    {
      os << i << ": [ ";

      for (int j = 0; j < 6; j++)
      {
        os << diagonal_host[6 * i + j] << " ";
      }

      os << "]   [ ";

      for (int j = 0; j < 6; j++)
      {
        os << precond_host[6 * i + j] << " ";
      }

      os << "]" << std::endl;
    }
  }

  os.close();
}

void PcgSolver::saveVec(std::string file_name)
{
  vec_data_.copy(cugl::DeviceArray<float>::DEVICE_TO_HOST);
  const float *vec_data_host = vec_data_.getHostPtr();

  std::ofstream os(file_name.c_str());

  if (os.is_open())
  {
    for (int i = 0; i < spmat_param_list_.rows; i++)
    {
      os << i << ": " << vec_data_host[i] << std::endl;
    }
  }

  os.close();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/*
 * Private Types
 */
void PcgSolver::alloc()
{
  size_t spmat_max_row_size = gpu::MAX_GRAPH_NODES_SIZE * spmat_param_list_.block_dim;
  size_t spmat_max_bin_size = gpu::divUp(spmat_max_row_size, spmat_param_list_.bin_width) + 1;
  size_t spmat_max_pad_row_size = spmat_max_bin_size * spmat_param_list_.bin_width;

  spmat_row_lengths_.alloc(spmat_max_pad_row_size);
  spmat_data_list_.row_lengths = spmat_row_lengths_.getDevicePtr();

  spmat_bin_lengths_.alloc(spmat_max_bin_size);
  spmat_data_list_.bin_lengths = spmat_bin_lengths_.getDevicePtr();

  spmat_bin_offsets_.alloc(spmat_max_bin_size);
  spmat_data_list_.bin_offsets = spmat_bin_offsets_.getDevicePtr();

  spmat_offsets_.alloc(spmat_max_pad_row_size, true); // use host memory
  spmat_data_list_.offsets = spmat_offsets_.getDevicePtr();

  spmat_diagonal_.alloc(spmat_max_pad_row_size * spmat_param_list_.block_dim, true); // use host memory
  spmat_data_list_.diagonal = spmat_diagonal_.getDevicePtr();

  spmat_precond_.alloc(spmat_max_pad_row_size * spmat_param_list_.block_dim, true); // use host memory
  spmat_data_list_.precond = spmat_precond_.getDevicePtr();

  spmat_col_indices_.alloc(gpu::MAX_SPMAT_ELEMS_SIZE, true); // use host memory
  spmat_data_list_.col_indices = spmat_col_indices_.getDevicePtr();

  spmat_data_.alloc(gpu::MAX_SPMAT_ELEMS_SIZE, true); // use host memory
  spmat_data_list_.data = spmat_data_.getDevicePtr();

  vec_data_.alloc(spmat_max_pad_row_size, true); // use host memory

  x_.alloc(spmat_max_pad_row_size);

  y_.alloc(spmat_max_pad_row_size);

  r_.alloc(spmat_max_pad_row_size);

  z_.alloc(spmat_max_pad_row_size);

  p_.alloc(spmat_max_pad_row_size);
}

void PcgSolver::free()
{
  spmat_row_lengths_.free();

  spmat_bin_lengths_.free();

  spmat_bin_offsets_.free();

  spmat_offsets_.free();

  spmat_diagonal_.free();

  spmat_precond_.free();

  spmat_col_indices_.free();

  spmat_data_.free();

  vec_data_.free();

  x_.free();

  y_.free();

  r_.free();

  z_.free();

  p_.free();
}

} // namespace dynfu
