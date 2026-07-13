// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/impl/continuous_conv/ContinuousConvSYCLKernels.h"

#include "open3d/ml/impl/continuous_conv/CoordinateTransformationSYCL.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {
constexpr size_t kBlockSize = 128;
// Work-group size for FillColumn(Transpose): one work-group processes one
// output point, matching the CUDA kernel's BLOCKSIZE=32 (one warp per output
// point).
constexpr size_t kFillColumnWGSize = 32;
}  // namespace

template <class T>
void MultiplyColumnsSYCL(sycl::queue& queue,
                         size_t rows,
                         size_t cols,
                         T* col_major_matrix,
                         const T* const vector) {
    const size_t n = rows * cols;
    if (n == 0) return;
    const size_t num_groups = (n + kBlockSize - 1) / kBlockSize;
    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(num_groups * kBlockSize),
                                       sycl::range<1>(kBlockSize)),
                     [=](sycl::nd_item<1> item) {
                         const size_t idx = item.get_global_id(0);
                         if (idx >= n) return;
                         const size_t col = idx / rows;
                         col_major_matrix[idx] *= vector[col];
                     });
         }).wait_and_throw();
}

template <class T>
void MultiplyAndCopyColumnsSYCL(sycl::queue& queue,
                                size_t rows,
                                size_t cols,
                                T* out_ptr,
                                const T* const col_major_matrix,
                                const T* const vector) {
    const size_t n = rows * cols;
    if (n == 0) return;
    const size_t num_groups = (n + kBlockSize - 1) / kBlockSize;
    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(num_groups * kBlockSize),
                                       sycl::range<1>(kBlockSize)),
                     [=](sycl::nd_item<1> item) {
                         const size_t idx = item.get_global_id(0);
                         if (idx >= n) return;
                         const size_t col = idx / rows;
                         out_ptr[idx] = col_major_matrix[idx] * vector[col];
                     });
         }).wait_and_throw();
}

template void MultiplyColumnsSYCL<float>(sycl::queue& queue,
                                         size_t rows,
                                         size_t cols,
                                         float* col_major_matrix,
                                         const float* const vector);

template void MultiplyAndCopyColumnsSYCL<float>(
        sycl::queue& queue,
        size_t rows,
        size_t cols,
        float* out_ptr,
        const float* const col_major_matrix,
        const float* const vector);

namespace {

/// Kernel body for FillColumnSYCL, templated on the same compile-time
/// switches as the CUDA FillColumnKernel (ALIGN_CORNERS, MAPPING,
/// INTERPOLATION), one work-group per output point. Ports FillColumnKernel
/// verbatim (algorithm-for-algorithm): the CUDA warp-shuffle normalizer
/// reduction becomes `sycl::reduce_over_group`, coordinate/interpolation math
/// calls the SYCL ports in CoordinateTransformationSYCL.h.
template <class TFeat,
          class TReal,
          class TIndex,
          bool ALIGN_CORNERS,
          CoordinateMapping MAPPING,
          InterpolationMode INTERPOLATION>
void FillColumnKernelSYCL(sycl::queue& queue,
                          TFeat* columns,
                          int in_channels,
                          TIndex begin_idx,
                          TIndex end_idx,
                          TIndex num_out,
                          const TReal* const out_positions,
                          TIndex num_inp,
                          const TReal* const inp_positions,
                          const TFeat* const inp_features,
                          const TFeat* const inp_importance,
                          size_t neighbors_index_size,
                          const TIndex* const neighbors_index,
                          const TFeat* const neighbors_importance,
                          const int64_t* const neighbors_row_splits,
                          const TReal* const extents,
                          const TReal* const offsets,
                          int filter_size_x,
                          int filter_size_y,
                          int filter_size_z,
                          bool individual_extent,
                          bool isotropic_extent,
                          bool normalize) {
    constexpr int NUM_INTERP_VALUES =
            (INTERPOLATION == InterpolationMode::LINEAR ||
                             INTERPOLATION == InterpolationMode::LINEAR_BORDER
                     ? 8
                     : 1);
    const TIndex num_columns = end_idx - begin_idx;
    if (num_columns <= 0) return;

    const bool point_importance = inp_importance != nullptr;
    const bool neighbor_importance = neighbors_importance != nullptr;

    // Zero the whole columns buffer up front, matching the CUDA
    // cudaMemsetAsync(columns, 0, ...) call before FillColumnKernel launch.
    queue.fill(columns, TFeat(0),
               size_t(filter_size_x) * filter_size_y * filter_size_z *
                       in_channels * size_t(num_columns))
            .wait();

    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(
                             sycl::range<1>(num_columns * kFillColumnWGSize),
                             sycl::range<1>(kFillColumnWGSize)),
                     [=](sycl::nd_item<1> item) {
                         const TIndex out_idx =
                                 begin_idx +
                                 static_cast<TIndex>(item.get_group(0));
                         if (out_idx >= end_idx) return;

                         TReal interp_weights[NUM_INTERP_VALUES];
                         TIndex interp_indices[NUM_INTERP_VALUES];

                         const TReal offset[3] = {offsets[0], offsets[1],
                                                  offsets[2]};

                         const TIndex col_idx = out_idx - begin_idx;
                         TFeat* out_column = columns + size_t(filter_size_x) *
                                                               filter_size_y *
                                                               filter_size_z *
                                                               in_channels *
                                                               size_t(col_idx);
                         const int64_t neighbor_start =
                                 neighbors_row_splits[out_idx];
                         const int64_t neighbor_end =
                                 neighbors_row_splits[out_idx + 1];

                         const TReal out_pos[3] = {
                                 out_positions[out_idx * 3 + 0],
                                 out_positions[out_idx * 3 + 1],
                                 out_positions[out_idx * 3 + 2]};

                         TReal inv_extents[3];
                         if (individual_extent) {
                             if (isotropic_extent) {
                                 inv_extents[0] = TReal(1) / extents[out_idx];
                                 inv_extents[1] = inv_extents[0];
                                 inv_extents[2] = inv_extents[0];
                             } else {
                                 inv_extents[0] =
                                         TReal(1) / extents[3 * out_idx + 0];
                                 inv_extents[1] =
                                         TReal(1) / extents[3 * out_idx + 1];
                                 inv_extents[2] =
                                         TReal(1) / extents[3 * out_idx + 2];
                             }
                         } else {
                             if (isotropic_extent) {
                                 inv_extents[0] = TReal(1) / extents[0];
                                 inv_extents[1] = inv_extents[0];
                                 inv_extents[2] = inv_extents[0];
                             } else {
                                 inv_extents[0] = TReal(1) / extents[0];
                                 inv_extents[1] = TReal(1) / extents[1];
                                 inv_extents[2] = TReal(1) / extents[2];
                             }
                         }

                         auto group = item.get_group();
                         const size_t lid = item.get_local_id(0);
                         const size_t lsize = item.get_local_range(0);

                         TReal normalizer = TReal(0);
                         if (normalize) {
                             if (neighbor_importance) {
                                 TReal local_sum = TReal(0);
                                 for (int64_t n_idx = neighbor_start +
                                                      static_cast<int64_t>(lid);
                                      n_idx < neighbor_end;
                                      n_idx += static_cast<int64_t>(lsize)) {
                                     local_sum += neighbors_importance[n_idx];
                                 }
                                 normalizer = sycl::reduce_over_group(
                                         group, local_sum, sycl::plus<TReal>());
                             } else {
                                 normalizer =
                                         TReal(neighbor_end - neighbor_start);
                             }
                         }

                         for (int64_t n_idx = neighbor_start;
                              n_idx < neighbor_end; ++n_idx) {
                             const TIndex inp_idx = neighbors_index[n_idx];
                             const TFeat n_importance =
                                     neighbor_importance
                                             ? neighbors_importance[n_idx]
                                             : TFeat(1);

                             TReal x, y, z;
                             x = inp_positions[inp_idx * 3 + 0] - out_pos[0];
                             y = inp_positions[inp_idx * 3 + 1] - out_pos[1];
                             z = inp_positions[inp_idx * 3 + 2] - out_pos[2];

                             ComputeFilterCoordinatesSYCL<ALIGN_CORNERS,
                                                          MAPPING>(
                                     x, y, z, filter_size_x, filter_size_y,
                                     filter_size_z, inv_extents[0],
                                     inv_extents[1], inv_extents[2], offset[0],
                                     offset[1], offset[2]);
                             InterpolateSYCL<INTERPOLATION>(
                                     interp_weights, interp_indices, x, y, z,
                                     filter_size_x, filter_size_y,
                                     filter_size_z);

                             TFeat infeat = 0;
                             TFeat importance = 1;
                             if (point_importance)
                                 importance = inp_importance[inp_idx];
                             if (neighbor_importance)
                                 importance *= n_importance;
                             if (normalize && normalizer != 0)
                                 importance /= normalizer;

                             for (int ic = static_cast<int>(lid);
                                  ic < in_channels;
                                  ic += static_cast<int>(lsize)) {
                                 infeat = importance *
                                          inp_features[size_t(inp_idx) *
                                                               in_channels +
                                                       ic];
                                 for (int j = 0; j < NUM_INTERP_VALUES; ++j) {
                                     TFeat value = interp_weights[j] * infeat;
                                     out_column[interp_indices[j] *
                                                        in_channels +
                                                ic] += value;
                                 }
                             }
                         }  // for n
                     });
         }).wait_and_throw();
}

/// Kernel body for FillColumnTransposeSYCL. Ports FillColumnTransposeKernel
/// verbatim (algorithm-for-algorithm); unlike the forward kernel, the CUDA
/// version accumulates into out_column without an initial normalizer
/// reduction, so no reduce_over_group is needed here.
template <class TFeat,
          class TReal,
          class TIndex,
          bool ALIGN_CORNERS,
          CoordinateMapping MAPPING,
          InterpolationMode INTERPOLATION>
void FillColumnTransposeKernelSYCL(
        sycl::queue& queue,
        TFeat* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        const TReal* const out_positions,
        TIndex num_inp,
        const TReal* const inp_positions,
        const TFeat* const inp_features,
        const TFeat* const inp_neighbors_importance_sum,
        const int64_t* const inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* const neighbors_index,
        const TFeat* const neighbors_importance,
        const int64_t* const neighbors_row_splits,
        const TReal* const extents,
        const TReal* const offsets,
        int filter_size_x,
        int filter_size_y,
        int filter_size_z,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize) {
    constexpr int NUM_INTERP_VALUES =
            (INTERPOLATION == InterpolationMode::LINEAR ||
                             INTERPOLATION == InterpolationMode::LINEAR_BORDER
                     ? 8
                     : 1);
    const TIndex num_columns = end_idx - begin_idx;
    if (num_columns <= 0) return;

    const bool has_neighbors_importance = inp_neighbors_importance_sum;
    const bool neighbor_importance = neighbors_importance != nullptr;

    // Zero the whole columns buffer up front, matching the CUDA
    // cudaMemsetAsync(columns, 0, ...) call before FillColumnTransposeKernel
    // launch.
    queue.fill(columns, TFeat(0),
               size_t(filter_size_x) * filter_size_y * filter_size_z *
                       in_channels * size_t(num_columns))
            .wait();

    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(
                             sycl::range<1>(num_columns * kFillColumnWGSize),
                             sycl::range<1>(kFillColumnWGSize)),
                     [=](sycl::nd_item<1> item) {
                         const TIndex out_idx =
                                 begin_idx +
                                 static_cast<TIndex>(item.get_group(0));
                         if (out_idx >= end_idx) return;

                         TReal interp_weights[NUM_INTERP_VALUES];
                         TIndex interp_indices[NUM_INTERP_VALUES];

                         const TReal offset[3] = {offsets[0], offsets[1],
                                                  offsets[2]};

                         const TIndex col_idx = out_idx - begin_idx;
                         TFeat* out_column = columns + size_t(filter_size_x) *
                                                               filter_size_y *
                                                               filter_size_z *
                                                               in_channels *
                                                               size_t(col_idx);
                         const int64_t neighbor_start =
                                 neighbors_row_splits[out_idx];
                         const int64_t neighbor_end =
                                 neighbors_row_splits[out_idx + 1];

                         const TReal out_pos[3] = {
                                 out_positions[out_idx * 3 + 0],
                                 out_positions[out_idx * 3 + 1],
                                 out_positions[out_idx * 3 + 2]};

                         TReal inv_extents[3];
                         if (!individual_extent) {
                             if (isotropic_extent) {
                                 inv_extents[0] = TReal(1) / extents[0];
                                 inv_extents[1] = inv_extents[0];
                                 inv_extents[2] = inv_extents[0];
                             } else {
                                 inv_extents[0] = TReal(1) / extents[0];
                                 inv_extents[1] = TReal(1) / extents[1];
                                 inv_extents[2] = TReal(1) / extents[2];
                             }
                         }

                         const size_t lid = item.get_local_id(0);
                         const size_t lsize = item.get_local_range(0);

                         for (int64_t n_idx = neighbor_start;
                              n_idx < neighbor_end; ++n_idx) {
                             const TIndex inp_idx = neighbors_index[n_idx];

                             TReal x, y, z;
                             x = out_pos[0] - inp_positions[inp_idx * 3 + 0];
                             y = out_pos[1] - inp_positions[inp_idx * 3 + 1];
                             z = out_pos[2] - inp_positions[inp_idx * 3 + 2];

                             if (individual_extent) {
                                 if (isotropic_extent) {
                                     inv_extents[0] =
                                             TReal(1) / extents[inp_idx];
                                     inv_extents[1] = inv_extents[0];
                                     inv_extents[2] = inv_extents[0];
                                 } else {
                                     inv_extents[0] = TReal(1) /
                                                      extents[3 * inp_idx + 0];
                                     inv_extents[1] = TReal(1) /
                                                      extents[3 * inp_idx + 1];
                                     inv_extents[2] = TReal(1) /
                                                      extents[3 * inp_idx + 2];
                                 }
                             }

                             TReal num_inp_neighbors_normalizer = 1;
                             if (normalize) {
                                 if (has_neighbors_importance) {
                                     if (inp_neighbors_importance_sum
                                                 [inp_idx] != 0)
                                         num_inp_neighbors_normalizer /=
                                                 inp_neighbors_importance_sum
                                                         [inp_idx];
                                 } else {
                                     const int64_t inp_neighbor_start =
                                             inp_neighbors_prefix_sum[inp_idx];
                                     const int64_t inp_neighbor_end =
                                             inp_idx + 1 < num_inp
                                                     ? inp_neighbors_prefix_sum
                                                               [inp_idx + 1]
                                                     : static_cast<int64_t>(
                                                               neighbors_index_size);
                                     const size_t num_inp_neighbors =
                                             inp_neighbor_end -
                                             inp_neighbor_start;
                                     if (num_inp_neighbors > 0)
                                         num_inp_neighbors_normalizer /=
                                                 num_inp_neighbors;
                                 }
                             }

                             ComputeFilterCoordinatesSYCL<ALIGN_CORNERS,
                                                          MAPPING>(
                                     x, y, z, filter_size_x, filter_size_y,
                                     filter_size_z, inv_extents[0],
                                     inv_extents[1], inv_extents[2], offset[0],
                                     offset[1], offset[2]);
                             InterpolateSYCL<INTERPOLATION>(
                                     interp_weights, interp_indices, x, y, z,
                                     filter_size_x, filter_size_y,
                                     filter_size_z);

                             TFeat infeat = 0;
                             for (int ic = static_cast<int>(lid);
                                  ic < in_channels;
                                  ic += static_cast<int>(lsize)) {
                                 infeat = inp_features[size_t(inp_idx) *
                                                               in_channels +
                                                       ic];
                                 if (neighbor_importance)
                                     infeat *= neighbors_importance[n_idx];
                                 if (normalize)
                                     infeat *= num_inp_neighbors_normalizer;
                                 for (int j = 0; j < NUM_INTERP_VALUES; ++j) {
                                     TFeat value = interp_weights[j] * infeat;
                                     out_column[interp_indices[j] *
                                                        in_channels +
                                                ic] += value;
                                 }
                             }
                         }  // for n
                     });
         }).wait_and_throw();
}

}  // namespace

// Runtime dispatch over (INTERPOLATION, MAPPING, ALIGN_CORNERS) to the
// compile-time-templated kernel launcher, mirroring the CUDA
// CALL_TEMPLATE/CALL_TEMPLATE2/3/4 macro cascade in ContinuousConvCUDAKernels
// .cu (same set of 3*3*2=18 instantiations selected at runtime).
template <class TFeat, class TReal, class TIndex>
void FillColumnSYCL(sycl::queue& queue,
                    TFeat* columns,
                    int in_channels,
                    TIndex begin_idx,
                    TIndex end_idx,
                    TIndex num_out,
                    const TReal* const out_positions,
                    TIndex num_inp,
                    const TReal* const inp_positions,
                    const TFeat* const inp_features,
                    const TFeat* const inp_importance,
                    size_t neighbors_index_size,
                    const TIndex* const neighbors_index,
                    const TFeat* const neighbors_importance,
                    const int64_t* const neighbors_row_splits,
                    const TReal* const extents,
                    const TReal* const offsets,
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

#define FN_PARAMETERS                                                          \
    queue, columns, in_channels, begin_idx, end_idx, num_out, out_positions,   \
            num_inp, inp_positions, inp_features, inp_importance,              \
            neighbors_index_size, neighbors_index, neighbors_importance,       \
            neighbors_row_splits, extents, offsets, filter_size_x,             \
            filter_size_y, filter_size_z, individual_extent, isotropic_extent, \
            normalize

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS)               \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping && \
        ALIGN_CORNERS == align_corners)                                    \
        FillColumnKernelSYCL<TFeat, TReal, TIndex, ALIGN_CORNERS, MAPPING, \
                             INTERPOLATION>(FN_PARAMETERS);

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

    CALL_TEMPLATE4

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef CALL_TEMPLATE4
#undef FN_PARAMETERS
}

template <class TFeat, class TReal, class TIndex>
void FillColumnTransposeSYCL(sycl::queue& queue,
                             TFeat* columns,
                             int in_channels,
                             TIndex begin_idx,
                             TIndex end_idx,
                             TIndex num_out,
                             const TReal* const out_positions,
                             TIndex num_inp,
                             const TReal* const inp_positions,
                             const TFeat* const inp_features,
                             const TFeat* const inp_neighbors_importance_sum,
                             const int64_t* const inp_neighbors_prefix_sum,
                             size_t neighbors_index_size,
                             const TIndex* const neighbors_index,
                             const TFeat* const neighbors_importance,
                             const int64_t* const neighbors_row_splits,
                             const TReal* const extents,
                             const TReal* const offsets,
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

#define FN_PARAMETERS                                                          \
    queue, columns, in_channels, begin_idx, end_idx, num_out, out_positions,   \
            num_inp, inp_positions, inp_features,                              \
            inp_neighbors_importance_sum, inp_neighbors_prefix_sum,            \
            neighbors_index_size, neighbors_index, neighbors_importance,       \
            neighbors_row_splits, extents, offsets, filter_size_x,             \
            filter_size_y, filter_size_z, individual_extent, isotropic_extent, \
            normalize

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS)               \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping && \
        ALIGN_CORNERS == align_corners)                                    \
        FillColumnTransposeKernelSYCL<TFeat, TReal, TIndex, ALIGN_CORNERS, \
                                      MAPPING, INTERPOLATION>(FN_PARAMETERS);

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

    CALL_TEMPLATE4

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef CALL_TEMPLATE4
#undef FN_PARAMETERS
}

template void FillColumnSYCL<float, float, int32_t>(
        sycl::queue& queue,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        const float* const out_positions,
        int32_t num_inp,
        const float* const inp_positions,
        const float* const inp_features,
        const float* const inp_importance,
        size_t neighbors_index_size,
        const int32_t* const neighbors_index,
        const float* const neighbors_importance,
        const int64_t* const neighbors_row_splits,
        const float* const extents,
        const float* const offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize);

template void FillColumnTransposeSYCL<float, float, int32_t>(
        sycl::queue& queue,
        float* columns,
        int in_channels,
        int32_t begin_idx,
        int32_t end_idx,
        int32_t num_out,
        const float* const out_positions,
        int32_t num_inp,
        const float* const inp_positions,
        const float* const inp_features,
        const float* const inp_neighbors_importance_sum,
        const int64_t* const inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const int32_t* const neighbors_index,
        const float* const neighbors_importance,
        const int64_t* const neighbors_row_splits,
        const float* const extents,
        const float* const offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
