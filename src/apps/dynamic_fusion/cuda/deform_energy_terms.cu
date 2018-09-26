#include <cugl_ros/apps/dynamic_fusion/cuda_internal.h>
#include <cugl_ros/apps/dynamic_fusion/cuda_impl/reduce.cuh>

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

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void initBlockId2RowLutKernel(size_t vec_block_size,
                              const DeviceArrayHandle<uint> block_ids,
                              DeviceArrayHandle<uint> block_id2row_lut)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < vec_block_size;
       i += stride)
  {
    uint block_id = block_ids.at(i);

    block_id2row_lut.at(block_id) = uint(i);
  } // grid-stride loop
}

bool initVecBlock(uint block_elem_size,
                  size_t &vec_block_size,
                  DeviceArray<uint> &block_elem_ids,
                  DeviceArray<uint> &block_elem_entry_codes,
                  DeviceArray<uint> &block_ids,
                  DeviceArray<uint> &block_elem_starts,
                  DeviceArray<uint> &block_elem_sizes,
                  DeviceArray<uint> &block_id2row_lut)
{
  // sorting
  thrust::device_ptr<uint> block_elem_ids_ptr(block_elem_ids.getDevicePtr());
  thrust::device_ptr<uint> block_elem_entry_codes_ptr(block_elem_entry_codes.getDevicePtr());

  thrust::sort_by_key(block_elem_ids_ptr,
                      block_elem_ids_ptr + block_elem_size,
                      block_elem_entry_codes_ptr);

  // reduction
  thrust::constant_iterator<uint> ones(1);
  thrust::device_ptr<uint> block_ids_ptr(block_ids.getDevicePtr());
  thrust::device_ptr<uint> block_elem_sizes_ptr(block_elem_sizes.getDevicePtr());

  thrust::pair< thrust::device_ptr<uint>, thrust::device_ptr<uint> > pair_end;

  pair_end = thrust::reduce_by_key(block_elem_ids_ptr,
                                   block_elem_ids_ptr + block_elem_size,
                                   ones,
                                   block_ids_ptr,
                                   block_elem_sizes_ptr);

  // scanning
  thrust::device_ptr<uint> block_elem_starts_ptr(block_elem_starts.getDevicePtr());

  thrust::exclusive_scan(block_elem_sizes_ptr,
                         pair_end.second,
                         block_elem_starts_ptr);

  vec_block_size = static_cast<size_t>(pair_end.second - block_elem_sizes_ptr);

  uint block_id_last;
  cudaMemcpy(reinterpret_cast<void *>(&block_id_last),
             reinterpret_cast<void *>(raw_pointer_cast(pair_end.first) - 1),
             sizeof(block_id_last), cudaMemcpyDeviceToHost);

  if (block_id_last == 0xFFFFFFFF)
    vec_block_size--;

  if (vec_block_size >= MAX_GRAPH_NODES_SIZE || vec_block_size <= 0)
  {
    std::cout << std::endl << "vec_block_size exceeds limitation!" << std::endl;
    return false;
  }

  int block = 256;
  int grid = min(divUp(vec_block_size, block), 512);

  initBlockId2RowLutKernel<<<grid, block>>>(vec_block_size,
                                            block_ids.getHandle(),
                                            block_id2row_lut.getHandle());
  checkCudaErrors(cudaGetLastError());

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getMatBlockExtIdsKernel(size_t mat_block_size,
                             const DeviceArrayHandle<uint> block_ids,
                             DeviceArrayHandle<uint> block_ext_ids,
                             DeviceArrayHandle<uint> block_entries)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < mat_block_size;
       i += stride)
  {
    uint block_id = block_ids.at(i);

    uint block_id_elem_x = ((block_id & 0xFFFF0000) >> 16);
    uint block_id_elem_y = (block_id & 0x0000FFFF);

    block_ext_ids.at(i) = block_id_elem_x;
    block_ext_ids.at(i + mat_block_size) = block_id_elem_y;

    block_entries.at(i) = uint(i);
    block_entries.at(i + mat_block_size) = uint(i);
  }
}

__global__
void getBlockCoosKernel(size_t mat_block_size,
                        const DeviceArrayHandle<uint> block_id2row_lut,
                        DeviceArrayHandle<uint> block_ext_ids,
                        DeviceArrayHandle<uint> block_row_offsets,
                        DeviceArrayHandle<uint> block_coos)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < mat_block_size;
       i += stride)
  {
    uint block_id = block_ext_ids.at(i);

    uint block_row = block_id2row_lut.at(block_id);

    uint block_row_offset = block_row_offsets.at(block_row);

    uint block_col = uint(i) - block_row_offset;

    uint block_coo = (block_row << 16) + block_col;

    block_coos.at(i) = block_coo;
  }
}

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
                  DeviceArray<uint> &block_coos)
{
  // sorting
  thrust::device_ptr<uint> block_elem_ids_ptr(block_elem_ids.getDevicePtr());
  thrust::device_ptr<uint> block_elem_entry_codes_ptr(block_elem_entry_codes.getDevicePtr());

  thrust::sort_by_key(block_elem_ids_ptr,
                      block_elem_ids_ptr + block_elem_size,
                      block_elem_entry_codes_ptr);

  // reduction
  thrust::constant_iterator<uint> ones(1);
  thrust::device_ptr<uint> block_ids_ptr(block_ids.getDevicePtr());
  thrust::device_ptr<uint> block_elem_sizes_ptr(block_elem_sizes.getDevicePtr());

  thrust::pair< thrust::device_ptr<uint>, thrust::device_ptr<uint> > pair_end;

  pair_end = thrust::reduce_by_key(block_elem_ids_ptr,
                                   block_elem_ids_ptr + block_elem_size,
                                   ones,
                                   block_ids_ptr,
                                   block_elem_sizes_ptr);

  // scanning
  thrust::device_ptr<uint> block_elem_starts_ptr(block_elem_starts.getDevicePtr());

  thrust::exclusive_scan(block_elem_sizes_ptr,
                         pair_end.second,
                         block_elem_starts_ptr);

  mat_block_size = static_cast<size_t>(pair_end.second - block_elem_sizes_ptr);

  uint block_id_last;
  cudaMemcpy(reinterpret_cast<void *>(&block_id_last),
             reinterpret_cast<void *>(raw_pointer_cast(pair_end.first) - 1),
             sizeof(block_id_last), cudaMemcpyDeviceToHost);

  if (block_id_last == 0xFFFFFFFF)
    mat_block_size--;

  // Note: we only store the upper triangle matrix here
  if (mat_block_size >= MAX_NON_ZERO_BLOCKS_SIZE / 2 || mat_block_size <= 0)
  {
    std::cout << std::endl << "mat_block_size exceeds limitation!" << std::endl;
    return false;
  }

  int block = 256;
  int grid = min(divUp(mat_block_size, block), 512);

  getMatBlockExtIdsKernel<<<grid, block>>>(mat_block_size,
                                           block_ids.getHandle(),
                                           block_ext_ids.getHandle(),
                                           block_entries.getHandle());
  checkCudaErrors(cudaGetLastError());

  // sorting
  thrust::device_ptr<uint> block_ext_ids_ptr(block_ext_ids.getDevicePtr());
  thrust::device_ptr<uint> block_entries_ptr(block_entries.getDevicePtr());

  thrust::sort_by_key(block_ext_ids_ptr,
                      block_ext_ids_ptr + 2 * mat_block_size,
                      block_entries_ptr);

  // reduction
  thrust::device_ptr<uint> block_row_lengths_ptr(block_row_lengths.getDevicePtr());
  thrust::pair< thrust::discard_iterator<thrust::use_default>, thrust::device_ptr<uint> > reduce_end;

  reduce_end = thrust::reduce_by_key(block_ext_ids_ptr,
                                     block_ext_ids_ptr + 2 * mat_block_size,
                                     ones,
                                     thrust::make_discard_iterator(),
                                     block_row_lengths_ptr);

  size_t mat_block_row_size = static_cast<size_t>(reduce_end.second - block_row_lengths_ptr);

  if (mat_block_row_size != vec_block_size)
  {
    std::cout << std::endl << "mat_block_row_size != vec_block_size" << std::endl;
    return false;
  }

  // scanning
  thrust::device_ptr<uint> block_row_offsets_ptr(block_row_offsets.getDevicePtr());

  thrust::exclusive_scan(block_row_lengths_ptr,
                         reduce_end.second,
                         block_row_offsets_ptr);

  block = 256;
  grid = min(divUp(2 * mat_block_size, block), 512);

  getBlockCoosKernel<<<grid, block>>>(2 * mat_block_size,
                                      block_id2row_lut.getHandle(),
                                      block_ext_ids.getHandle(),
                                      block_row_offsets.getHandle(),
                                      block_coos.getHandle());
  checkCudaErrors(cudaGetLastError());

  // sorting
  thrust::device_ptr<uint> block_coos_ptr(block_coos.getDevicePtr());

  thrust::sort_by_key(block_entries_ptr,
                      block_entries_ptr + 2 * mat_block_size,
                      block_coos_ptr);

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__device__
void term_num_decode(uint code, uint &term_num)
{
  term_num = (code & 0x00000003);
}

__global__
void classifyEnergyTermsKernel(size_t block_size,
                               const DeviceArrayHandle<uint> block_elem_starts,
                               const DeviceArrayHandle<uint> block_elem_sizes,
                               const DeviceArrayHandle<uint> block_elem_entry_codes,
                               DeviceArrayHandle<uint> fit_term_elem_sizes,
                               DeviceArrayHandle<uint> fit_term_elem_entry_codes,
                               DeviceArrayHandle<uint> reg_term_elem_sizes,
                               DeviceArrayHandle<uint> reg_term_elem_entry_codes)
{
  const int lane_id = threadIdx.x % warpSize;

  __shared__ uint elem_start, elem_size;
  __shared__ uint fit_term_elem_size, reg_term_elem_size;

  // block stride loop
  for (uint i = blockIdx.x; i < block_size; i += gridDim.x)
  {
    if (threadIdx.x == 0)
    {
      elem_start = block_elem_starts.at(i);
      elem_size = block_elem_sizes.at(i);
      fit_term_elem_size = 0;
      reg_term_elem_size = 0;
    }

    __syncthreads();

    // element stride loop
    for (uint elem_stride = 0; elem_stride < elem_size; elem_stride += blockDim.x)
    {
      uint j = threadIdx.x + elem_stride;

      uint term_num = 0xFFFFFFFF;

      if (j < elem_size)
      {
        uint code = block_elem_entry_codes.at(elem_start + j);

        term_num_decode(code, term_num);

        // Step 1: leader election
#ifdef CUDA8
        unsigned int fit_term_mask = __ballot(term_num == 0);
        unsigned int reg_term_mask = __ballot(term_num == 1);
#else
        unsigned int active = __activemask();
        unsigned int fit_term_mask = __ballot_sync(active, term_num == 0);
        unsigned int reg_term_mask = __ballot_sync(active, term_num == 1);
#endif

        int fit_term_leader = __ffs(fit_term_mask) - 1;
        int reg_term_leader = __ffs(reg_term_mask) - 1;

        // Step 2: computing the total increment (in the warp)
        int fit_term_total = __popc(fit_term_mask);
        int reg_term_total = __popc(reg_term_mask);

        // Step 3: performing the atomic add
        unsigned int lane_mask_lt = (1 << lane_id) - 1;
        unsigned int fit_term_rank = __popc(fit_term_mask & lane_mask_lt);
        unsigned int reg_term_rank = __popc(reg_term_mask & lane_mask_lt);

        uint fit_term_elem_size_old = 0, reg_term_elem_size_old = 0;

        // Step 4: computing the total increment in the block & broadcasting the result
        if (fit_term_total > 0)
        {
          if (lane_id == fit_term_leader)
            fit_term_elem_size_old = atomicAdd(&fit_term_elem_size, fit_term_total);

#ifdef CUDA8
          fit_term_elem_size_old = __shfl(fit_term_elem_size_old,
                                          fit_term_leader);
#else
          fit_term_elem_size_old = __shfl_sync(fit_term_mask,
                                               fit_term_elem_size_old,
                                               fit_term_leader);
#endif

          uint fit_term_elem_offset = elem_start + fit_term_elem_size_old + fit_term_rank;

          if (term_num == 0)
          {
            fit_term_elem_entry_codes.at(fit_term_elem_offset) = code;
          }
        }

        if (reg_term_total > 0)
        {
          if (lane_id == reg_term_leader)
            reg_term_elem_size_old = atomicAdd(&reg_term_elem_size, reg_term_total);

#ifdef CUDA8
          reg_term_elem_size_old = __shfl(reg_term_elem_size_old,
                                          reg_term_leader);
#else
          reg_term_elem_size_old = __shfl_sync(reg_term_mask,
                                               reg_term_elem_size_old,
                                               reg_term_leader);
#endif

          uint reg_term_elem_offset = elem_start + reg_term_elem_size_old + reg_term_rank;

          if (term_num == 1)
          {
            reg_term_elem_entry_codes.at(reg_term_elem_offset) = code;
          }
        }
      } // j < elem_size
    } // element stride loop

    __syncthreads();

    if (threadIdx.x == 0)
    {
      fit_term_elem_sizes.at(i) = fit_term_elem_size;
      reg_term_elem_sizes.at(i) = reg_term_elem_size;
    }
  } // block stride loop
}

void classifyEnergyTerms(size_t block_size,
                         const DeviceArray<uint> &block_elem_starts,
                         const DeviceArray<uint> &block_elem_sizes,
                         const DeviceArray<uint> &block_elem_entry_codes,
                         DeviceArray<uint> &fit_term_elem_sizes,
                         DeviceArray<uint> &fit_term_elem_entry_codes,
                         DeviceArray<uint> &reg_term_elem_sizes,
                         DeviceArray<uint> &reg_term_elem_entry_codes)
{
  int block = 256;
  int grid = min(static_cast<int>(block_size), 512);

  classifyEnergyTermsKernel<<<grid, block>>>(block_size,
                                             block_elem_starts.getHandle(),
                                             block_elem_sizes.getHandle(),
                                             block_elem_entry_codes.getHandle(),
                                             fit_term_elem_sizes.getHandle(),
                                             fit_term_elem_entry_codes.getHandle(),
                                             reg_term_elem_sizes.getHandle(),
                                             reg_term_elem_entry_codes.getHandle());
  checkCudaErrors(cudaGetLastError());
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void getMatRowLengthsKernel(size_t vec_block_size,
                            const DeviceArrayHandle<uint> mat_block_row_lengths,
                            const SpMatParamList param_list,
                            SpMatDataList data_list)
{
  const int stride = blockDim.x * gridDim.x;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < vec_block_size;
       i += stride)
  {
    uint block_row_length = mat_block_row_lengths.at(i);
    uint row_length = block_row_length * param_list.block_dim;

    #pragma unroll
    for (int j = 0; j < param_list.block_dim; j++)
    {
      data_list.row_lengths[i * param_list.block_dim + j] = row_length;
    }
  } // grid-stride loop
}

__global__
void getMatBinLengthsKernel(const SpMatParamList param_list,
                            SpMatDataList data_list)
{
  const int stride = blockDim.x * gridDim.x;

  const int lane_id = threadIdx.x % warpSize;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < param_list.pad_rows;
       i += stride)
  {
    int bin_id = i / warpSize;

    int row_length = (i < param_list.rows) ? data_list.row_lengths[i] : 0;

    int length_max = row_length;
    int length_new = row_length;

    // calculate the maximum length in each warp through warp shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
#ifdef CUDA8
      length_new = __shfl_down(length_max, offset);
#else
      length_new = __shfl_down_sync(0xFFFFFFFF, length_max, offset);
#endif

      length_max = max(length_max, length_new);
    }

    if (lane_id == 0)
      data_list.bin_lengths[bin_id] = length_max;
  } // grid-stride loop
}

__global__
void initSpMatOffsetsAndColIndicesKernel(const SpMatParamList param_list,
                                         SpMatDataList data_list)
{
  const int stride = blockDim.x * gridDim.x;

  const uint lane_id = threadIdx.x % warpSize;

  uint bin_length, bin_offset;

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < param_list.pad_rows;
       i += stride)
  {
    int bin_id = i / warpSize;

    if (lane_id == 0)
    {
      bin_length = data_list.bin_lengths[bin_id];
      bin_offset = data_list.bin_offsets[bin_id];
    }

    // broadcast through warp shuffle
#ifdef CUDA8
    bin_length = __shfl(bin_length, 0);
    bin_offset = __shfl(bin_offset, 0);
#else
    bin_length = __shfl_sync(0xFFFFFFFF, bin_length, 0);
    bin_offset = __shfl_sync(0xFFFFFFFF, bin_offset, 0);
#endif
    bin_offset *= warpSize;

    uint offset = bin_offset + lane_id;

    data_list.offsets[i] = offset;

    uint entry_begin = offset;
    uint entry_end = offset + bin_length * warpSize;

    #pragma unroll
    for (uint spmat_entry = entry_begin; spmat_entry < entry_end; spmat_entry += warpSize)
      data_list.col_indices[spmat_entry] = 0xFFFFFFFF;
  } // grid-stride loop
}

__global__
void setSpMatColIndicesKernel(size_t max_block_size,
                              const DeviceArrayHandle<uint> mat_block_coos,
                              DeviceArrayHandle<uint> mat_block_write_entries,
                              const SpMatParamList param_list,
                              SpMatDataList data_list)
{
  const int stride = blockDim.x * gridDim.x;

  uint mat_block_rows[2];
  uint mat_block_cols[2];

  // grid-stride loop
  for (int i = blockDim.x * blockIdx.x + threadIdx.x;
       i < max_block_size;
       i += stride)
  {
    #pragma unroll
    for (uint j = 0; j < 2; j++)
    {
      uint mat_block_coo = mat_block_coos.at(2*i + j);
      mat_block_rows[j] = (mat_block_coo & 0xFFFF0000) >> 16;
      mat_block_cols[j] = (mat_block_coo & 0x0000FFFF);
    }

    for (uint j = 0; j < 2; j++)
    {
      uint mat_row_base = mat_block_rows[j] * param_list.block_dim;
      uint mat_col_base = mat_block_cols[j] * param_list.block_dim;
      uint col_indice_base = mat_block_rows[1-j] * param_list.block_dim;

      for (uint x = 0; x < param_list.block_dim; x++)
      {
        uint mat_row = mat_row_base + x;

        uint offset = data_list.offsets[mat_row];

        mat_block_write_entries.at((2*i + j)*param_list.block_dim + x) = offset + mat_col_base * warpSize;

        #pragma unroll
        for (uint y = 0; y < param_list.block_dim; y++)
        {
          uint mat_col = mat_col_base + y;
          uint col_indice = col_indice_base + y;

          uint spmat_entry = offset + mat_col * warpSize;

          data_list.col_indices[spmat_entry] = col_indice;
        } // y
      } // x
    } // j
  } // grid-stride loop
}

bool initSparseMat(size_t vec_block_size,
                   size_t mat_block_size,
                   DeviceArray<uint> &mat_block_row_lengths,
                   DeviceArray<uint> &mat_block_coos,
                   DeviceArray<uint> &mat_block_write_entries,
                   SpMatParamList &param_list,
                   SpMatDataList &data_list)
{
  param_list.rows = param_list.block_dim * vec_block_size;

  size_t bins = divUp(param_list.rows, param_list.bin_width) + 1;
  param_list.pad_rows = bins * param_list.bin_width;

  int block = 256;
  int grid = min(divUp(vec_block_size, block), 512);

  getMatRowLengthsKernel<<<grid, block>>>(vec_block_size,
                                          mat_block_row_lengths.getHandle(),
                                          param_list,
                                          data_list);
  checkCudaErrors(cudaGetLastError());

  // block = 256;
  grid = min(divUp(param_list.pad_rows, block), 512);

  getMatBinLengthsKernel<<<grid, block>>>(param_list,
                                          data_list);
  checkCudaErrors(cudaGetLastError());

  thrust::device_ptr<uint> bin_lengths_ptr(data_list.bin_lengths);
  thrust::device_ptr<uint> bin_offsets_ptr(data_list.bin_offsets);

  thrust::exclusive_scan(bin_lengths_ptr,
                         bin_lengths_ptr + bins,
                         bin_offsets_ptr);

  uint bin_offset_last = 0;
  cudaMemcpy(reinterpret_cast<void *>(&bin_offset_last),
             reinterpret_cast<void *>(raw_pointer_cast(bin_offsets_ptr) + bins - 1),
             sizeof(bin_offset_last), cudaMemcpyDeviceToHost);

  size_t spmat_elems_size = bin_offset_last * param_list.bin_width;

  if (spmat_elems_size >= MAX_SPMAT_ELEMS_SIZE)
  {
    std::cout << std::endl << "spmat_elems_size exceeds limitation!" << std::endl;
    return false;
  }

  initSpMatOffsetsAndColIndicesKernel<<<grid, block>>>(param_list,
                                                       data_list);
  checkCudaErrors(cudaGetLastError());

  // block = 256;
  grid = min(divUp(mat_block_size, block), 512);

  setSpMatColIndicesKernel<<<grid, block>>>(mat_block_size,
                                            mat_block_coos.getHandle(),
                                            mat_block_write_entries.getHandle(),
                                            param_list,
                                            data_list);
  checkCudaErrors(cudaGetLastError());

  return true;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__
void updateTransformArrayKernel(float step_size,
                                size_t block_size,
                                const DeviceArrayHandle<uint> block_ids,
                                const DeviceArrayHandle<float> results,
                                const DeviceArrayHandle<TransformType> transform_array_src,
                                DeviceArrayHandle<TransformType> transform_array_dst)
{
  const int stride = blockDim.x * gridDim.x;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < block_size;
       i += stride)
  {
    uint block_id = block_ids.at(i);

    Affine3d transform_src;
    transform_src.setValue(&(transform_array_src.at(4 * block_id)));

    float w_x = step_size * results.at(6 * i + 0);
    float w_y = step_size * results.at(6 * i + 1);
    float w_z = step_size * results.at(6 * i + 2);
    float v_x = step_size * results.at(6 * i + 3);
    float v_y = step_size * results.at(6 * i + 4);
    float v_z = step_size * results.at(6 * i + 5);

    float s_x = sinf(w_x); float c_x = cosf(w_x);
    float s_y = sinf(w_y); float c_y = cosf(w_y);
    float s_z = sinf(w_z); float c_z = cosf(w_z);

    TransformType mat_col[4];

    Affine3d r_z;
    mat_col[0] = make_float4(c_z, s_z, 0.f, 0.f);
    mat_col[1] = make_float4(-s_z, c_z, 0.f, 0.f);
    mat_col[2] = make_float4(0.f, 0.f, 1.f, 0.f);
    mat_col[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    r_z.setValue(mat_col);

    Affine3d r_y;
    mat_col[0] = make_float4(c_y, 0.f, -s_y, 0.f);
    mat_col[1] = make_float4(0.f, 1.f, 0.f, 0.f);
    mat_col[2] = make_float4(s_y, 0.f, c_y, 0.f);
    mat_col[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    r_y.setValue(mat_col);

    Affine3d r_x;
    mat_col[0] = make_float4(1.f, 0.f, 0.f, 0.f);
    mat_col[1] = make_float4(0.f, c_x, s_x, 0.f);
    mat_col[2] = make_float4(0.f, -s_x, c_x, 0.f);
    mat_col[3] = make_float4(0.f, 0.f, 0.f, 1.f);
    r_x.setValue(mat_col);

    // TransformType mat_col[4];
    // mat_col[0] = make_float4(1.f, w_z, -w_y, 0.f);
    // mat_col[1] = make_float4(-w_z, 1.f, w_x, 0.f);
    // mat_col[2] = make_float4(w_y, -w_x, 1.f, 0.f);
    // mat_col[3] = make_float4(v_x, v_y, v_z, 1.f);

    // Affine3d transform_inc;
    // transform_inc.setValue(mat_col);

    Affine3d transform_inc = r_x * (r_y * r_z);
    transform_inc.at(0, 3) = v_x;
    transform_inc.at(1, 3) = v_y;
    transform_inc.at(2, 3) = v_z;

    Affine3d transform_dst = transform_inc * transform_src;
    // Affine3d transform_dst = transform_src * transform_inc;
    transform_dst.getValue(&(transform_array_dst.at(4 * block_id)));
  }
}

void updateTransformArray(float step_size,
                          size_t block_size,
                          const DeviceArray<uint> &block_ids,
                          const DeviceArray<float> &results,
                          DeviceArray<TransformType> &transform_array)
{
  int block = 256;
  int grid = min(divUp(static_cast<int>(block_size), block), 512);

  updateTransformArrayKernel<<<grid, block>>>(step_size,
                                              block_size,
                                              block_ids.getHandle(),
                                              results.getHandle(),
                                              transform_array.getHandle(),
                                              transform_array.getWriteHandle());
  checkCudaErrors(cudaGetLastError());
}

} // namespace gpu
} // namespace dynfu
