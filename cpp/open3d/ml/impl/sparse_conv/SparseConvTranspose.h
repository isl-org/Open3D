// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <tbb/parallel_for.h>

#include <Eigen/Core>

namespace open3d {
namespace ml {
namespace impl {

/// Implementation of SparseConvTransposeComputeFeatures with template
/// parameters for configuration.
template <class TFeat,
          class TOut,
          class TIndex,
          class TKernelIndex,
          bool NORMALIZE>
void _SparseConvTransposeComputeFeaturesCPU(
        TOut* out_features,
        const std::vector<int>& filter_dims,
        const TFeat* filter,
        size_t num_out,
        const TFeat* out_importance,
        size_t num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        const TIndex* neighbor_index,
        const TKernelIndex* neighbors_kernel_index,
        const TFeat* neighbor_importance,
        const int64_t* neighbors_row_splits) {
    const bool NEIGHBOR_IMPORTANCE = inp_neighbors_importance_sum;

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int num_kernel_elements = 1;
    for (int i = 0; i < filter_dims.size() - 2; ++i)
        num_kernel_elements *= filter_dims[i];

    memset(out_features, 0, sizeof(TOut) * num_out * out_channels);

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_out, 32),
            [&](const tbb::blocked_range<size_t>& r) {
                int range_length = r.end() - r.begin();

                Eigen::Map<Eigen::Matrix<TOut, Eigen::Dynamic, Eigen::Dynamic>>
                        C(out_features + (r.begin() * out_channels),
                          out_channels, range_length);

                for (size_t out_idx = r.begin(); out_idx != r.end();
                     ++out_idx) {
                    const int out_col = out_idx - r.begin();
                    const size_t neighbor_start = neighbors_row_splits[out_idx];
                    const size_t neighbor_end =
                            neighbors_row_splits[out_idx + 1];

                    for (size_t n = neighbor_start; n < neighbor_end; ++n) {
                        const size_t inp_idx = neighbor_index[n];
                        const int kernel_idx = neighbors_kernel_index[n];

                        TFeat n_importance = NEIGHBOR_IMPORTANCE
                                                     ? neighbor_importance[n]
                                                     : TFeat(1);

                        TFeat normalizer(1);
                        if (NORMALIZE) {
                            if (NEIGHBOR_IMPORTANCE) {
                                if (inp_neighbors_importance_sum[inp_idx] !=
                                    TFeat(0))
                                    normalizer /= inp_neighbors_importance_sum
                                            [inp_idx];
                            } else {
                                size_t num_inp_neighbors;
                                const size_t inp_neighbor_start =
                                        inp_neighbors_row_splits[inp_idx];
                                const size_t inp_neighbor_end =
                                        inp_neighbors_row_splits[inp_idx + 1];
                                num_inp_neighbors =
                                        inp_neighbor_end - inp_neighbor_start;
                                if (num_inp_neighbors > 0)
                                    normalizer /= TFeat(num_inp_neighbors);
                            }
                        }

                        Eigen::Map<const Eigen::Matrix<TFeat, Eigen::Dynamic,
                                                       Eigen::Dynamic>>
                                A(filter + kernel_idx * out_channels *
                                                   in_channels,
                                  out_channels, in_channels);

                        Eigen::Map<const Eigen::Matrix<TFeat, Eigen::Dynamic,
                                                       Eigen::Dynamic>>
                                B(inp_features + inp_idx * in_channels,
                                  in_channels, 1);
                        TFeat scale = normalizer * n_importance;
                        C.col(out_col) +=
                                (A * (scale * B)).template cast<TOut>();
                    }

                }  // out_idx

                if (out_importance) {
                    for (int i = 0; i < range_length; ++i)
                        C.col(i) *= TOut(out_importance[r.begin() + i]);
                }
            });
}

/// Computes the output features of a transpose sparse convolution.
///
/// \param out_features    Output array for the computed features with shape
///        [num_out, out channels]
///
/// \param filter_dims    The sizes of the filter dimensions. The size of
///        filter_dims must be >=3. The order is
///        [num kernel elements, inp channels, out channels].
///
/// \param filter    Pointer to the filter values.
///
/// \param num_out    The number of output points.
///
/// \param out_importance    Optional importance for each output point with
///        shape [num_out]. Set to null to disable.
///
/// \param num_inp    The number of input points.
///
/// \param inp_features    The input features with shape
///        [num_inp, in_channels].
///
/// \param inp_neighbors_importance_sum    The sum of the neighbors_importance
///        values for each input with shape [num_inp].
///
/// \param inp_neighbors_row_splits   The prefix sum which defines the start
///        and end of the sublists in \p inp_neighbors_index. The size of the
///        array is \p num_inp + 1.
///
/// \param neighbors_index    The array with lists of neighbors for each
///        output point. The start and end of each sublist is defined by
///        \p neighbors_row_splits.
///
/// \param neighbors_kernel_index    Defines which kernel element to use for
///        each neighbor. This array has the same length as \p neighbors_index.
///
/// \param neighbors_importance    Optional importance for each entry in
///        \p neighbors_index. Set to null to disable.
///
/// \param neighbors_row_splits   The prefix sum which defines the start
///        and end of the sublists in \p neighbors_index. The size of the
///        array is \p num_out + 1.
///
/// \param normalize    If true then the result is normalized either by the
///        number of points (neighbors_importance is null) or by the sum of
///        the respective values in neighbors_importance.
///
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvTransposeComputeFeaturesCPU(
        TOut* out_features,
        const std::vector<int>& filter_dims,
        const TFeat* filter,
        size_t num_out,
        const TFeat* out_importance,
        size_t num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        const TIndex* neighbor_index,
        const TKernelIndex* neighbors_kernel_index,
        const TFeat* neighbor_importance,
        const int64_t* neighbors_row_splits,
        bool normalize) {
#define FN_PARAMETERS                                                         \
    out_features, filter_dims, filter, num_out, out_importance, num_inp,      \
            inp_features, inp_neighbors_importance_sum,                       \
            inp_neighbors_row_splits, neighbor_index, neighbors_kernel_index, \
            neighbor_importance, neighbors_row_splits

#define CALL_TEMPLATE(NORMALIZE)                                         \
    if (NORMALIZE == normalize)                                          \
        _SparseConvTransposeComputeFeaturesCPU<TFeat, TOut, TIndex,      \
                                               TKernelIndex, NORMALIZE>( \
                FN_PARAMETERS);

#define CALL_TEMPLATE2  \
    CALL_TEMPLATE(true) \
    CALL_TEMPLATE(false)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
