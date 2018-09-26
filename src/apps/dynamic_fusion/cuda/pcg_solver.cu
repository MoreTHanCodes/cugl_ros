#include <cugl_ros/apps/dynamic_fusion/cuda_internal.h>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/reduce.cuh>

namespace dynfu
{
namespace gpu
{
texture<float, 1, cudaReadModeElementType> vec_texture;

__device__ uint blocks_done_new = 0;
__device__ float global_value_new = 0.f;
__device__ float output_value_new;

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void bindVecTexture(const DeviceArray<float> &vec_data)
{
  cudaChannelFormatDesc vec_desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaBindTexture(0, vec_texture, reinterpret_cast<const float *>(vec_data.getDevicePtr()), vec_desc));
}

void unbindVecTexture()
{
  checkCudaErrors(cudaUnbindTexture(vec_texture));
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__
bool computeMat6x6Inverse(float *s[6])
{
  float scp[6];

  #pragma unroll
  for (int k = 0; k < 6; k++)
  {
    scp[k] = fabsf(s[k][0]);

    for (int j = 1; j < 6; j++)
    {
      if (fabsf(s[k][j]) > scp[k])
      {
        scp[k] = fabsf(s[k][j]);
      }
    }

    if (scp[k] == 0.f)
      return false;
  }

  int pivot_to;
  float scp_max;
  
  #pragma unroll
  for (int k = 0; k < 6; k++)
  {
    // select pivot row
    pivot_to = k;
    scp_max = fabsf(s[k][k] / scp[k]);
    
    // find out which row should be on top
    for (int p = k + 1; p < 6; p++)
    {
      if (fabsf(s[p][k] / scp[p]) > scp_max)
      {
        scp_max = fabsf(s[p][k] / scp[p]);
        pivot_to = p;
      }
    }
    
    // Pivot if necessary
    if (pivot_to != k)
    {
      float *tmprow;
      tmprow = s[k];
      s[k] = s[pivot_to];
      s[pivot_to] = tmprow;
      float tmpscp;
      tmpscp = scp[k];
      scp[k] = scp[pivot_to];
      scp[pivot_to] = tmpscp;
    }
    
    float mjk;
    
    // perform gaussian elimination
    for (int j = k + 1; j < 6; j++)
    {
      mjk = s[j][k] / s[k][k];
      s[j][k] = 0.f;
  
      for (int jj = k + 1; jj < 12; jj++)
      {
        s[j][jj] -= mjk * s[k][jj];
      }
    }
  }
  
  if (fabsf(s[5][5]) < 5e-3)
  {
    return false;    // singular matrix!
  }
  
  //
  // Now we have an upper triangular matrix.
  //
  //  x x x x x x | y y y y y y
  //  0 x x x x x | y y y y y y
  //  0 0 x x x x | y y y y y y
  //  0 0 0 x x x | y y y y y y
  //  0 0 0 0 x x | y y y y y y
  //  0 0 0 0 0 x | y y y y y y
  //
  //  we'll back substitute to get the inverse
  //
  //  1 0 0 0 0 0 | z z z z z z
  //  0 1 0 0 0 0 | z z z z z z
  //  0 0 1 0 0 0 | z z z z z z
  //  0 0 0 1 0 0 | z z z z z z
  //  0 0 0 0 1 0 | z z z z z z
  //  0 0 0 0 0 1 | z z z z z z
  //
  
  float mjk;
  
  for (int k = 5; k > 0; k--)
  {
    for (int j = k - 1; j > -1; j--)
    {
      mjk = s[j][k] / s[k][k];

      for (int jj= j + 1; jj < 12; jj++)
      {
        s[j][jj] -= mjk * s[k][jj];
      }
    }
  }
  
  return true;
}

__device__
void writeDiagBlockInverse(float *src[6], int entry_start, float *dst)
{
  int entry = entry_start;
  for (int row = 0; row < 6; row++)
  {
    for (int col = 0; col < 6; col++)
    {
      dst[entry++] = src[row][col+6] / src[row][row];
    }
  }
}

__device__
void writeDiagInverse(int entry_start, const float *src, float *dst)
{
  int entry = entry_start;
  for (int row = 0; row < 6; row++)
  {
    for (int col = 0; col < 6; col++)
    {
      if (row == col)
      {
        float val = src[entry];
        float val_inv = (fabsf(val) < 1e-4) ? 0.f : (1.f / val);
        dst[entry] = val_inv;
      }
      else
      {
        dst[entry] = 0.f;
      }

      entry++;
    }
  }
}

__global__
void computePrecondKernel(int precond_mat_block_size,
                          float damping,
                          const SpMatParamList spmat_param_list,
                          SpMatDataList spmat_data_list)
{
  const int stride = blockDim.x * gridDim.x;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < precond_mat_block_size;
       i += stride)
  {
    float r1[12], r2[12], r3[12], r4[12], r5[12], r6[12];
    float *s[6];

    s[0] = &r1[0];
    s[1] = &r2[0];
    s[2] = &r3[0];
    s[3] = &r4[0];
    s[4] = &r5[0];
    s[5] = &r6[0];

    // initialization
    int diag_mat_entry_start = 36 * i;
    int diag_mat_entry = diag_mat_entry_start;
    for (int row = 0; row < 6; row++)
    {
      #pragma unroll
      for (int col = 0; col < 6; col++)
      {
        if (row == col)
        {
          float inc = (row < 3) ? (2.f * damping) : damping;

          spmat_data_list.diagonal[diag_mat_entry] += inc;
          s[row][col] = spmat_data_list.diagonal[diag_mat_entry];
          s[row][col+6] = 1.f;
        }
        else
        {
          s[row][col] = spmat_data_list.diagonal[diag_mat_entry];
          s[row][col+6] = 0.f;
        }

        diag_mat_entry++;
      }
    }

    bool find_inverse = computeMat6x6Inverse(s);

    if (find_inverse)
    {
      writeDiagBlockInverse(s, diag_mat_entry_start, spmat_data_list.precond);
    }
    else
    {
      writeDiagInverse(diag_mat_entry_start, spmat_data_list.diagonal, spmat_data_list.precond);
    }

  } // grid stride loop
}

void computePreconditioner(float damping,
                           const SpMatParamList &spmat_param_list,
                           SpMatDataList &spmat_data_list)
{
  int precond_mat_block_size = static_cast<int>(spmat_param_list.rows / spmat_param_list.block_dim);

  int block = 256;
  int grid = min(divUp(precond_mat_block_size, block), 512);

  computePrecondKernel<<<grid, block>>>(precond_mat_block_size,
                                        damping,
                                        spmat_param_list,
                                        spmat_data_list);
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void initPcgSolverKernel(const SpMatParamList spmat_param_list,
                         const SpMatDataList spmat_data_list,
                         DeviceArrayHandle<float> x,
                         DeviceArrayHandle<float> r,
                         DeviceArrayHandle<float> z,
                         DeviceArrayHandle<float> p)
{
  const int stride = blockDim.x * gridDim.x;

  float rz = 0.f;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < spmat_param_list.rows;
       i += stride)
  {
    float z_val = 0.f;

    int diag_mat_col_start = (i / spmat_param_list.block_dim) * spmat_param_list.block_dim;
    #pragma unroll
    for (int k = 0; k < spmat_param_list.block_dim; k++)
    {
      int diag_mat_entry = spmat_param_list.block_dim * i + k; // row major
      int diag_mat_col = diag_mat_col_start + k;

      float precond_val = spmat_data_list.precond[diag_mat_entry];
      float b_val = tex1Dfetch(vec_texture, diag_mat_col);

      z_val += precond_val * b_val;
    }

    float r_val = tex1Dfetch(vec_texture, i);

    x.at(i) = 0.f;
    r.at(i) = r_val;
    z.at(i) = z_val;
    p.at(i) = z_val;

    rz += r_val * z_val;
  } // grid stride loop

  __syncthreads();

  rz = blockReduceSum(rz);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value_new, rz);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done_new, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value_new = global_value_new;
      global_value_new = 0.f;
      blocks_done_new = 0;
    }
  }
}

float initPcgSolver(const SpMatParamList &spmat_param_list,
                    const SpMatDataList &spmat_data_list,
                    const DeviceArray<float> &b,
                    DeviceArray<float> &x,
                    DeviceArray<float> &r,
                    DeviceArray<float> &z,
                    DeviceArray<float> &p)
{
  bindVecTexture(b);

  int block = 256;
  int grid = min(divUp(static_cast<int>(spmat_param_list.pad_rows), block), 512);

  initPcgSolverKernel<<<grid, block>>>(spmat_param_list,
                                       spmat_data_list,
                                       x.getHandle(),
                                       r.getHandle(),
                                       z.getHandle(),
                                       p.getHandle());
  checkCudaErrors(cudaGetLastError());

  unbindVecTexture();

  float delta;
  checkCudaErrors(cudaMemcpyFromSymbol(&delta, output_value_new, sizeof(delta)));

  return delta;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void runMergedPcgStep1(const SpMatParamList spmat_param_list,
                       const SpMatDataList spmat_data_list,
                       DeviceArrayHandle<float> y)
{
  const int stride = blockDim.x * gridDim.x;

  float py = 0.f;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < spmat_param_list.rows;
       i += stride)
  {
    float y_val = 0.f;

    // off-diagonal
    uint index = spmat_data_list.offsets[i];
    uint index_end = spmat_data_list.offsets[i + warpSize];

    while (index < index_end)
    {
      int col_begin = spmat_data_list.col_indices[index];
      int col_end = col_begin + spmat_param_list.block_dim;

      if (col_begin == 0xFFFFFFFF)
        break;

      #pragma unroll
      for (int col = col_begin; col < col_end; col++)
      {
        float p_val = tex1Dfetch(vec_texture, col);
        y_val += spmat_data_list.data[index] * p_val;
        index += warpSize;
      }
    }

    // diagonal
    int diag_mat_col_start = (i / spmat_param_list.block_dim) * spmat_param_list.block_dim;
    #pragma unroll
    for (int k = 0; k < spmat_param_list.block_dim; k++)
    {
      int diag_mat_entry = spmat_param_list.block_dim * i + k;
      int diag_mat_col = diag_mat_col_start + k;

      float diag_val = spmat_data_list.diagonal[diag_mat_entry];
      float p_val = tex1Dfetch(vec_texture, diag_mat_col);

      y_val += diag_val * p_val;
    }

    y.at(i) = y_val;

    float p_val = tex1Dfetch(vec_texture, i);
    py += p_val * y_val;
  } // grid stride loop

  __syncthreads();

  py = blockReduceSum(py);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value_new, py);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done_new, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value_new = global_value_new;
      global_value_new = 0.f;
      blocks_done_new = 0;
    }
  }
}

__global__
void runMergedPcgStep2(float alpha,
                       int rows,
                       const DeviceArrayHandle<float> p,
                       const DeviceArrayHandle<float> y,
                       DeviceArrayHandle<float> x,
                       DeviceArrayHandle<float> r)
{
  const int stride = blockDim.x * gridDim.x;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < rows;
       i += stride)
  {
    float x_val = x.at(i) + alpha * p.at(i);
    float r_val = r.at(i) - alpha * y.at(i);

    x.at(i) = x_val;
    r.at(i) = r_val;
  } // grid stride loop
}

__global__
void runMergedPcgStep3(const SpMatParamList spmat_param_list,
                       const SpMatDataList spmat_data_list,
                       DeviceArrayHandle<float> z)
{
  const int stride = blockDim.x * gridDim.x;

  float rz = 0.f;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < spmat_param_list.rows;
       i += stride)
  {
    float z_val = 0.f;

    int diag_mat_col_start = (i / spmat_param_list.block_dim) * spmat_param_list.block_dim;
    #pragma unroll
    for (int k = 0; k < spmat_param_list.block_dim; k++)
    {
      int diag_mat_entry = spmat_param_list.block_dim * i + k;
      int diag_mat_col = diag_mat_col_start + k;

      float precond_val = spmat_data_list.precond[diag_mat_entry];
      float r_val = tex1Dfetch(vec_texture, diag_mat_col);

      z_val += precond_val * r_val;
    }

    z.at(i) = z_val;

    float r_val = tex1Dfetch(vec_texture, i);

    rz += r_val * z_val;
  } // grid stride loop

  __syncthreads();

  rz = blockReduceSum(rz);

  if (threadIdx.x == 0)
  {
    atomicAdd(&global_value_new, rz);

    unsigned int total_blocks = gridDim.x;
    unsigned int value = atomicInc(&blocks_done_new, total_blocks);

    // last block
    if (value == total_blocks - 1)
    {
      output_value_new = global_value_new;
      global_value_new = 0.f;
      blocks_done_new = 0;
    }
  }
}

__global__
void runMergedPcgStep4(float beta,
                       int rows,
                       const DeviceArrayHandle<float> z,
                       DeviceArrayHandle<float> p)
{
  const int stride = blockDim.x * gridDim.x;

  // grid stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < rows;
       i += stride)
  {
    float p_val = z.at(i) + beta * p.at(i);

    p.at(i) = p_val;
  } // grid stride loop
}

float iteratePcgSolver(float delta_old,
                       const SpMatParamList &spmat_param_list,
                       const SpMatDataList &spmat_data_list,
                       const DeviceArray<float> &b,
                       DeviceArray<float> &x,
                       DeviceArray<float> &y,
                       DeviceArray<float> &r,
                       DeviceArray<float> &z,
                       DeviceArray<float> &p)
{
  bindVecTexture(p);

  int block = 256;
  int grid = min(divUp(static_cast<int>(spmat_param_list.pad_rows), block), 512);

  runMergedPcgStep1<<<grid, block>>>(spmat_param_list,
                                     spmat_data_list,
                                     y.getHandle());
  checkCudaErrors(cudaGetLastError());

  unbindVecTexture();

  float py_dot;
  checkCudaErrors(cudaMemcpyFromSymbol(&py_dot, output_value_new, sizeof(py_dot)));

  float alpha = delta_old / py_dot;

  runMergedPcgStep2<<<grid, block>>>(alpha,
                                     static_cast<int>(spmat_param_list.rows),
                                     p.getHandle(),
                                     y.getHandle(),
                                     x.getHandle(),
                                     r.getHandle());
  checkCudaErrors(cudaGetLastError());

  bindVecTexture(r);

  runMergedPcgStep3<<<grid, block>>>(spmat_param_list,
                                     spmat_data_list,
                                     z.getHandle());
  checkCudaErrors(cudaGetLastError());

  unbindVecTexture();

  float delta_new;
  checkCudaErrors(cudaMemcpyFromSymbol(&delta_new, output_value_new, sizeof(delta_new)));

  float beta = delta_new / delta_old;

  runMergedPcgStep4<<<grid, block>>>(beta,
                                     static_cast<int>(spmat_param_list.rows),
                                     z.getHandle(),
                                     p.getHandle());
  checkCudaErrors(cudaGetLastError());

  return delta_new;
}

} // namespace gpu
} // namespace dynfu
