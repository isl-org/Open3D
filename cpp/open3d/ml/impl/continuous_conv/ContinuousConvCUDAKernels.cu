// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/ml/impl/continuous_conv/ContinuousConvCUDAKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

/// Kernel for FillColumn
template <class TReal,
          class TIndex,
          bool ALIGN_CORNERS,
          CoordinateMapping MAPPING,
          InterpolationMode INTERPOLATION>
__global__ void FillColumnKernel(
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        const TReal* const __restrict__ out_positions,
        TIndex num_inp,
        const TReal* const __restrict__ inp_positions,
        const TReal* const __restrict__ inp_features,
        const TReal* const __restrict__ inp_importance,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const TReal* const __restrict__ extents,
        const TReal* const __restrict__ offsets,
        int filter_size_x,
        int filter_size_y,
        int filter_size_z,
        bool INDIVIDUAL_EXTENT,
        bool ISOTROPIC_EXTENT,
        bool NORMALIZE,
        bool POINT_IMPORTANCE,
        bool NEIGHBOR_IMPORTANCE) {
    TIndex out_idx = begin_idx + blockIdx.x;
    if (out_idx >= end_idx) return;
    const int NUM_INTERP_VALUES =
            (INTERPOLATION == InterpolationMode::LINEAR ||
                             INTERPOLATION == InterpolationMode::LINEAR_BORDER
                     ? 8
                     : 1);
    TReal interp_weights[NUM_INTERP_VALUES];
    TIndex interp_indices[NUM_INTERP_VALUES];

    TReal offset[3] = {offsets[0], offsets[1], offsets[2]};

    const TIndex col_idx = out_idx - begin_idx;
    TReal* out_column = columns + filter_size_x * filter_size_y *
                                          filter_size_z * in_channels * col_idx;
    const int64_t neighbor_start = neighbors_row_splits[out_idx];
    const int64_t neighbor_end = neighbors_row_splits[out_idx + 1];

    TReal out_pos[3] = {out_positions[out_idx * 3 + 0],
                        out_positions[out_idx * 3 + 1],
                        out_positions[out_idx * 3 + 2]};

    TReal inv_extents[3];
    if (INDIVIDUAL_EXTENT) {
        if (ISOTROPIC_EXTENT) {
            inv_extents[0] = TReal(1) / extents[out_idx];
            inv_extents[1] = inv_extents[0];
            inv_extents[2] = inv_extents[0];
        } else {
            inv_extents[0] = TReal(1) / extents[3 * out_idx + 0];
            inv_extents[1] = TReal(1) / extents[3 * out_idx + 1];
            inv_extents[2] = TReal(1) / extents[3 * out_idx + 2];
        }
    } else {
        if (ISOTROPIC_EXTENT) {
            inv_extents[0] = TReal(1) / extents[0];
            inv_extents[1] = inv_extents[0];
            inv_extents[2] = inv_extents[0];
        } else {
            inv_extents[0] = TReal(1) / extents[0];
            inv_extents[1] = TReal(1) / extents[1];
            inv_extents[2] = TReal(1) / extents[2];
        }
    }

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

        TReal x, y, z;
        x = inp_positions[inp_idx * 3 + 0] - out_pos[0];
        y = inp_positions[inp_idx * 3 + 1] - out_pos[1];
        z = inp_positions[inp_idx * 3 + 2] - out_pos[2];

        ComputeFilterCoordinates<ALIGN_CORNERS, MAPPING>(
                x, y, z, filter_size_x, filter_size_y, filter_size_z,
                inv_extents[0], inv_extents[1], inv_extents[2], offset[0],
                offset[1], offset[2]);
        Interpolate<INTERPOLATION>(interp_weights, interp_indices, x, y, z,
                                   filter_size_x, filter_size_y, filter_size_z);

        TReal infeat = 0;
        TReal importance = 1;
        if (POINT_IMPORTANCE) importance = inp_importance[inp_idx];
        if (NEIGHBOR_IMPORTANCE) importance *= n_importance;
        if (NORMALIZE && normalizer != 0) importance /= normalizer;

        for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
            infeat = importance * inp_features[inp_idx * in_channels + ic];
            for (int j = 0; j < NUM_INTERP_VALUES; ++j) {
                TReal value = interp_weights[j] * infeat;
                out_column[interp_indices[j] * in_channels + ic] += value;
            }
        }
    }  // for n
}

template <class TReal, class TIndex>
void FillColumn(const cudaStream_t& stream,
                TReal* columns,
                int in_channels,
                TIndex begin_idx,
                TIndex end_idx,
                TIndex num_out,
                const TReal* const __restrict__ out_positions,
                TIndex num_inp,
                const TReal* const __restrict__ inp_positions,
                const TReal* const __restrict__ inp_features,
                const TReal* const __restrict__ inp_importance,
                size_t neighbors_index_size,
                const TIndex* const __restrict__ neighbors_index,
                const TReal* const __restrict__ neighbors_importance,
                const int64_t* const __restrict__ neighbors_row_splits,
                const TReal* const __restrict__ extents,
                const TReal* const __restrict__ offsets,
                const std::vector<int>& filter_dims,
                InterpolationMode interpolation,
                CoordinateMapping coordinate_mapping,
                bool align_corners,
                bool individual_extent,
                bool isotropic_extent,
                bool normalize) {
    const int filter_size_z = filter_dims[0];
    const int filter_size_y = filter_dims[1];
    const int filter_size_x = filter_dims[2];

    TIndex num_columns = end_idx - begin_idx;
    int filter_spatial_size = filter_size_x * filter_size_y * filter_size_z;
    cudaMemsetAsync(
            columns, 0,
            sizeof(TReal) * filter_spatial_size * in_channels * num_columns,
            stream);

    const int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = num_columns;

#define FN_PARAMETERS                                                          \
    columns, in_channels, begin_idx, end_idx, num_out, out_positions, num_inp, \
            inp_positions, inp_features, inp_importance, neighbors_index_size, \
            neighbors_index, neighbors_importance, neighbors_row_splits,       \
            extents, offsets, filter_size_x, filter_size_y, filter_size_z,     \
            individual_extent, isotropic_extent, normalize,                    \
            inp_importance != nullptr, neighbors_importance != nullptr

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS)                   \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping &&     \
        ALIGN_CORNERS == align_corners)                                        \
        FillColumnKernel<TReal, TIndex, ALIGN_CORNERS, MAPPING, INTERPOLATION> \
                <<<grid, block, 0, stream>>>(FN_PARAMETERS);

#define CALL_TEMPLATE2(INTERPOLATION, MAPPING)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false)

#define CALL_TEMPLATE3(INTERPOLATION)                                     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::BALL_TO_CUBE_RADIAL) \
    CALL_TEMPLATE2(INTERPOLATION,                                         \
                   CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING)     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::IDENTITY)

#define CALL_TEMPLATE4                               \
    CALL_TEMPLATE3(InterpolationMode::LINEAR)        \
    CALL_TEMPLATE3(InterpolationMode::LINEAR_BORDER) \
    CALL_TEMPLATE3(InterpolationMode::NEAREST_NEIGHBOR)

    if (grid.x) {
        CALL_TEMPLATE4
        /*CHECK_CUDA_ERROR*/
    }

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef CALL_TEMPLATE4

#undef FN_PARAMETERS
}

template void FillColumn<float, int32_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        const float* const __restrict__ out_positions,
        int32_t num_inp,
        const float* const __restrict__ inp_positions,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_importance,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const float* const __restrict__ extents,
        const float* const __restrict__ offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize);

template <class TReal,
          class TIndex,
          bool ALIGN_CORNERS,
          CoordinateMapping MAPPING,
          InterpolationMode INTERPOLATION>
__global__ void FillColumnTransposeKernel(
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        const TReal* const __restrict__ out_positions,
        TIndex num_inp,
        const TReal* const __restrict__ inp_positions,
        const TReal* const __restrict__ inp_features,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TReal* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const TReal* const __restrict__ extents,
        const TReal* const __restrict__ offsets,
        int filter_size_x,
        int filter_size_y,
        int filter_size_z,
        bool INDIVIDUAL_EXTENT,
        bool ISOTROPIC_EXTENT,
        bool NORMALIZE,
        bool NEIGHBOR_IMPORTANCE) {
    TIndex out_idx = begin_idx + blockIdx.x;
    if (out_idx >= end_idx) return;
    const int NUM_INTERP_VALUES =
            (INTERPOLATION == InterpolationMode::LINEAR ||
                             INTERPOLATION == InterpolationMode::LINEAR_BORDER
                     ? 8
                     : 1);
    TReal interp_weights[NUM_INTERP_VALUES];
    TIndex interp_indices[NUM_INTERP_VALUES];

    TReal offset[3] = {offsets[0], offsets[1], offsets[2]};

    const TIndex col_idx = out_idx - begin_idx;
    TReal* out_column = columns + filter_size_x * filter_size_y *
                                          filter_size_z * in_channels * col_idx;
    const int64_t neighbor_start = neighbors_row_splits[out_idx];
    const int64_t neighbor_end = neighbors_row_splits[out_idx + 1];

    TReal out_pos[3] = {out_positions[out_idx * 3 + 0],
                        out_positions[out_idx * 3 + 1],
                        out_positions[out_idx * 3 + 2]};

    TReal inv_extents[3];
    if (INDIVIDUAL_EXTENT == false) {
        if (ISOTROPIC_EXTENT) {
            inv_extents[0] = TReal(1) / extents[0];
            inv_extents[1] = inv_extents[0];
            inv_extents[2] = inv_extents[0];
        } else {
            inv_extents[0] = TReal(1) / extents[0];
            inv_extents[1] = TReal(1) / extents[1];
            inv_extents[2] = TReal(1) / extents[2];
        }
    }

    for (int64_t n_idx = neighbor_start; n_idx < neighbor_end; ++n_idx) {
        const TIndex inp_idx = neighbors_index[n_idx];

        TReal x, y, z;
        x = out_pos[0] - inp_positions[inp_idx * 3 + 0];
        y = out_pos[1] - inp_positions[inp_idx * 3 + 1];
        z = out_pos[2] - inp_positions[inp_idx * 3 + 2];

        if (INDIVIDUAL_EXTENT) {
            if (ISOTROPIC_EXTENT) {
                inv_extents[0] = TReal(1) / extents[inp_idx];
                inv_extents[1] = inv_extents[0];
                inv_extents[2] = inv_extents[0];
            } else {
                inv_extents[0] = TReal(1) / extents[3 * inp_idx + 0];
                inv_extents[1] = TReal(1) / extents[3 * inp_idx + 1];
                inv_extents[2] = TReal(1) / extents[3 * inp_idx + 2];
            }
        }

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

        ComputeFilterCoordinates<ALIGN_CORNERS, MAPPING>(
                x, y, z, filter_size_x, filter_size_y, filter_size_z,
                inv_extents[0], inv_extents[1], inv_extents[2], offset[0],
                offset[1], offset[2]);
        Interpolate<INTERPOLATION>(interp_weights, interp_indices, x, y, z,
                                   filter_size_x, filter_size_y, filter_size_z);

        TReal infeat = 0;
        for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
            infeat = inp_features[inp_idx * in_channels + ic];
            if (NEIGHBOR_IMPORTANCE) infeat *= neighbors_importance[n_idx];
            if (NORMALIZE) infeat *= num_inp_neighbors_normalizer;
            for (int j = 0; j < NUM_INTERP_VALUES; ++j) {
                TReal value = interp_weights[j] * infeat;
                out_column[interp_indices[j] * in_channels + ic] += value;
            }
        }
    }  // for n
}

template <class TReal, class TIndex>
void FillColumnTranspose(
        const cudaStream_t& stream,
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        const TReal* const __restrict__ out_positions,
        TIndex num_inp,
        const TReal* const __restrict__ inp_positions,
        const TReal* const __restrict__ inp_features,
        const TReal* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TReal* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const TReal* const __restrict__ extents,
        const TReal* const __restrict__ offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize) {
    const bool has_neighbors_importance = inp_neighbors_importance_sum;
    const int filter_size_z = filter_dims[0];
    const int filter_size_y = filter_dims[1];
    const int filter_size_x = filter_dims[2];

    TIndex num_columns = end_idx - begin_idx;
    int filter_spatial_size = filter_size_x * filter_size_y * filter_size_z;
    cudaMemsetAsync(
            columns, 0,
            sizeof(TReal) * filter_spatial_size * in_channels * num_columns,
            stream);

    const int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = num_columns;

#define FN_PARAMETERS                                                          \
    columns, in_channels, begin_idx, end_idx, num_out, out_positions, num_inp, \
            inp_positions, inp_features, neighbors_index_size,                 \
            neighbors_index, inp_neighbors_importance_sum,                     \
            inp_neighbors_prefix_sum, neighbors_importance,                    \
            neighbors_row_splits, extents, offsets, filter_size_x,             \
            filter_size_y, filter_size_z, individual_extent, isotropic_extent, \
            normalize, has_neighbors_importance

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS)               \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping && \
        ALIGN_CORNERS == align_corners)                                    \
        FillColumnTransposeKernel<TReal, TIndex, ALIGN_CORNERS, MAPPING,   \
                                  INTERPOLATION>                           \
                <<<grid, block, 0, stream>>>(FN_PARAMETERS);

#define CALL_TEMPLATE2(INTERPOLATION, MAPPING)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false)

#define CALL_TEMPLATE3(INTERPOLATION)                                     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::BALL_TO_CUBE_RADIAL) \
    CALL_TEMPLATE2(INTERPOLATION,                                         \
                   CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING)     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::IDENTITY)

#define CALL_TEMPLATE4                               \
    CALL_TEMPLATE3(InterpolationMode::LINEAR)        \
    CALL_TEMPLATE3(InterpolationMode::LINEAR_BORDER) \
    CALL_TEMPLATE3(InterpolationMode::NEAREST_NEIGHBOR)

    if (grid.x) {
        CALL_TEMPLATE4
        /*CHECK_CUDA_ERROR*/
    }

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef CALL_TEMPLATE4

#undef FN_PARAMETERS
}

template void FillColumnTranspose<float, int32_t>(
        const cudaStream_t& stream,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        const float* const __restrict__ out_positions,
        int32_t num_inp,
        const float* const __restrict__ inp_positions,
        const float* const __restrict__ inp_features,
        const float* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const int32_t* const __restrict__ neighbors_index,
        const float* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const float* const __restrict__ extents,
        const float* const __restrict__ offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize);

template <class T>
__global__ void MultiplyColumnsKernel(size_t rows,
                                      size_t cols,
                                      T* __restrict__ col_major_matrix,
                                      const T* const __restrict__ vector) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    size_t col = idx / rows;

    T factor = vector[col];
    col_major_matrix[idx] *= factor;
}

template <class T>
void MultiplyColumns(const cudaStream_t& stream,
                     size_t rows,
                     size_t cols,
                     T* __restrict__ col_major_matrix,
                     const T* const __restrict__ vector) {
    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = DivUp(rows * cols, BLOCKSIZE);

    if (grid.x) {
        MultiplyColumnsKernel<T><<<grid, block, 0, stream>>>(
                rows, cols, col_major_matrix, vector);
    }
}

template void MultiplyColumns<float>(const cudaStream_t& stream,
                                     size_t rows,
                                     size_t cols,
                                     float* __restrict__ col_major_matrix,
                                     const float* const __restrict__ vector);

template <class T>
__global__ void MultiplyAndCopyColumnsKernel(
        size_t rows,
        size_t cols,
        T* __restrict__ out_ptr,
        const T* const __restrict__ col_major_matrix,
        const T* const __restrict__ vector) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    size_t col = idx / rows;

    T factor = vector[col];
    out_ptr[idx] = col_major_matrix[idx] * factor;
}

template <class T>
void MultiplyAndCopyColumns(const cudaStream_t& stream,
                            size_t rows,
                            size_t cols,
                            T* __restrict__ out_ptr,
                            const T* const __restrict__ col_major_matrix,
                            const T* const __restrict__ vector) {
    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = DivUp(rows * cols, BLOCKSIZE);

    if (grid.x) {
        MultiplyAndCopyColumnsKernel<T><<<grid, block, 0, stream>>>(
                rows, cols, out_ptr, col_major_matrix, vector);
    }
}

template void MultiplyAndCopyColumns<float>(
        const cudaStream_t& stream,
        size_t rows,
        size_t cols,
        float* __restrict__ out_ptr,
        const float* const __restrict__ col_major_matrix,
        const float* const __restrict__ vector);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
