// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/impl/sparse_conv/SparseConvCUDAKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

/// Kernel for FillColumn
template <class TReal, class TIndex, class TKernelIndex>
__global__ void FillColumnKernel(
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        TIndex num_inp,
        const TReal* const __restrict__ inp_features,
        const TReal* const __restrict__ inp_importance,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TKernelIndex* const __restrict__ neighbors_kernel_index,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool NORMALIZE,
        bool POINT_IMPORTANCE,
        bool NEIGHBOR_IMPORTANCE) {
    TIndex out_idx = begin_idx + blockIdx.x;
    if (out_idx >= end_idx) return;

    const TIndex col_idx = out_idx - begin_idx;
    TReal* out_column = columns + num_kernel_elements * in_channels * col_idx;
    const int64_t neighbor_start = neighbors_row_splits[out_idx];
    const int64_t neighbor_end = neighbors_row_splits[out_idx + 1];

    TReal normalizer = TReal(0);
    if (NORMALIZE) {
        if (NEIGHBOR_IMPORTANCE) {
            for (int64_t n_idx = neighbor_start + threadIdx.x;
                 n_idx < neighbor_end; n_idx += blockDim.x) {
                TReal n_importance = neighbors_importance[n_idx];
                normalizer += n_importance;
            }
            unsigned int mask = __activemask();
            for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                normalizer += __shfl_down_sync(mask, normalizer, offset);
            normalizer = __shfl_sync(mask, normalizer, 0);
        } else {
            int64_t num_neighbors = neighbor_end - neighbor_start;
            normalizer = num_neighbors;
        }
    }

    for (int64_t n_idx = neighbor_start; n_idx < neighbor_end; ++n_idx) {
        const TIndex inp_idx = neighbors_index[n_idx];
        const TReal n_importance =
                NEIGHBOR_IMPORTANCE ? neighbors_importance[n_idx] : TReal(1);
        int kernel_idx = neighbors_kernel_index[n_idx];

        TReal infeat = 0;
        TReal importance = 1;
        if (POINT_IMPORTANCE) importance = inp_importance[inp_idx];
        if (NEIGHBOR_IMPORTANCE) importance *= n_importance;
        if (NORMALIZE && normalizer != 0) importance /= normalizer;

        for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
            infeat = importance * inp_features[inp_idx * in_channels + ic];
            out_column[kernel_idx * in_channels + ic] = infeat;
        }
    }  // for n
}

template <class TReal, class TIndex, class TKernelIndex>
void FillColumn(const cudaStream_t& stream,
                TReal* columns,
                int in_channels,
                TIndex begin_idx,
                TIndex end_idx,
                TIndex num_out,
                TIndex num_inp,
                const TReal* const __restrict__ inp_features,
                const TReal* const __restrict__ inp_importance,
                size_t neighbors_index_size,
                const TIndex* const __restrict__ neighbors_index,
                const TKernelIndex* const __restrict__ neighbors_kernel_index,
                const TReal* const __restrict__ neighbors_importance,
                const int64_t* const __restrict__ neighbors_row_splits,
                const int num_kernel_elements,
                bool normalize) {
    TIndex num_columns = end_idx - begin_idx;
    cudaMemsetAsync(
            columns, 0,
            sizeof(TReal) * num_kernel_elements * in_channels * num_columns,
            stream);

    const int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = num_columns;

#define FN_PARAMETERS                                                         \
    columns, in_channels, begin_idx, end_idx, num_out, num_inp, inp_features, \
            inp_importance, neighbors_index_size, neighbors_index,            \
            neighbors_kernel_index, neighbors_importance,                     \
            neighbors_row_splits, num_kernel_elements, normalize,             \
            inp_importance != nullptr, neighbors_importance != nullptr

#define CALL_TEMPLATE \
    FillColumnKernel<TReal, TIndex><<<grid, block, 0, stream>>>(FN_PARAMETERS);

    if (grid.x) {
        CALL_TEMPLATE
    }

#undef CALL_TEMPLATE

#undef FN_PARAMETERS
}

template void FillColumn<float, int32_t, int16_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        int32_t num_inp,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_importance,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const int16_t* const __restrict__ neighbors_kernel_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize);

template void FillColumn<float, int32_t, uint8_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        int32_t num_inp,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_importance,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const uint8_t* const __restrict__ neighbors_kernel_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize);

template <class TReal, class TIndex, class TKernelIndex>
__global__ void FillColumnTransposeKernel(
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        TIndex num_inp,
        const TReal* const __restrict__ inp_features,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TKernelIndex* const __restrict__ neighbors_kernel_index,
        const TReal* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool NORMALIZE,
        bool NEIGHBOR_IMPORTANCE) {
    TIndex out_idx = begin_idx + blockIdx.x;
    if (out_idx >= end_idx) return;

    const TIndex col_idx = out_idx - begin_idx;
    TReal* out_column = columns + num_kernel_elements * in_channels * col_idx;
    const int64_t neighbor_start = neighbors_row_splits[out_idx];
    const int64_t neighbor_end = neighbors_row_splits[out_idx + 1];

    for (int64_t n_idx = neighbor_start; n_idx < neighbor_end; ++n_idx) {
        const TIndex inp_idx = neighbors_index[n_idx];
        const int kernel_idx = neighbors_kernel_index[n_idx];

        TReal num_inp_neighbors_normalizer = 1;
        if (NORMALIZE) {
            if (NEIGHBOR_IMPORTANCE) {
                if (inp_neighbors_importance_sum[inp_idx] != 0)
                    num_inp_neighbors_normalizer /=
                            inp_neighbors_importance_sum[inp_idx];
            } else {
                const int64_t inp_neighbor_start =
                        inp_neighbors_prefix_sum[inp_idx];
                const int64_t inp_neighbor_end =
                        inp_idx + 1 < num_inp
                                ? inp_neighbors_prefix_sum[inp_idx + 1]
                                : neighbors_index_size;
                const size_t num_inp_neighbors =
                        inp_neighbor_end - inp_neighbor_start;
                if (num_inp_neighbors > 0)
                    num_inp_neighbors_normalizer /= num_inp_neighbors;
            }
        }

        TReal infeat = 0;
        for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
            infeat = inp_features[inp_idx * in_channels + ic];
            if (NEIGHBOR_IMPORTANCE) infeat *= neighbors_importance[n_idx];
            if (NORMALIZE) infeat *= num_inp_neighbors_normalizer;

            out_column[kernel_idx * in_channels + ic] += infeat;
        }
    }  // for n
}

template <class TReal, class TIndex, class TKernelIndex>
void FillColumnTranspose(
        const cudaStream_t& stream,
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        TIndex num_inp,
        const TReal* const __restrict__ inp_features,
        const TReal* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TKernelIndex* const __restrict__ neighbors_kernel_index,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize) {
    const bool has_neighbors_importance = inp_neighbors_importance_sum;

    TIndex num_columns = end_idx - begin_idx;
    cudaMemsetAsync(
            columns, 0,
            sizeof(TReal) * num_kernel_elements * in_channels * num_columns,
            stream);

    const int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = num_columns;

#define FN_PARAMETERS                                                         \
    columns, in_channels, begin_idx, end_idx, num_out, num_inp, inp_features, \
            neighbors_index_size, neighbors_index, neighbors_kernel_index,    \
            inp_neighbors_importance_sum, inp_neighbors_prefix_sum,           \
            neighbors_importance, neighbors_row_splits, num_kernel_elements,  \
            normalize, has_neighbors_importance

#define CALL_TEMPLATE                        \
    FillColumnTransposeKernel<TReal, TIndex> \
            <<<grid, block, 0, stream>>>(FN_PARAMETERS);

    if (grid.x) {
        CALL_TEMPLATE
    }

#undef CALL_TEMPLATE
#undef FN_PARAMETERS
}

template void FillColumnTranspose<float, int32_t, int16_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        int32_t num_inp,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const int16_t* const __restrict__ neighbors_kernel_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize);

template void FillColumnTranspose<float, int32_t, uint8_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        int32_t num_inp,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const uint8_t* const __restrict__ neighbors_kernel_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
