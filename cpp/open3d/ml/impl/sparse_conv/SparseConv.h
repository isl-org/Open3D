// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#pragma once
#include <tbb/parallel_for.h>

#include <Eigen/Core>

namespace open3d {
namespace ml {
namespace impl {

/// Implementation of SparseConvComputeFeatures with template parameters for
/// configuration.
template <class TFeat,
          class TOut,
          class TIndex,
          class TKernelIndex,
          bool POINT_IMPORTANCE>
void _SparseConvComputeFeaturesCPU(TOut* out_features,
                                   const std::vector<int>& filter_dims,
                                   const TFeat* filter,
                                   size_t num_out,
                                   size_t num_inp,
                                   const TFeat* inp_features,
                                   const TFeat* inp_importance,
                                   size_t neighbors_index_size,
                                   const TIndex* neighbors_index,
                                   const TKernelIndex* neighbors_kernel_index,
                                   const TFeat* neighbors_importance,
                                   const int64_t* neighbors_row_splits,
                                   bool normalize) {
    const bool NEIGHBOR_IMPORTANCE = neighbors_importance != nullptr;

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

                Eigen::Matrix<TOut, Eigen::Dynamic, 1> normalizers(range_length,
                                                                   1);
                normalizers.setZero();

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
                        const size_t inp_idx = neighbors_index[n];
                        const int kernel_idx = neighbors_kernel_index[n];

                        const TFeat n_importance =
                                (NEIGHBOR_IMPORTANCE ? neighbors_importance[n]
                                                     : TFeat(1));
                        normalizers(out_col) += TOut(n_importance);

                        TFeat importance(1.0);
                        if (POINT_IMPORTANCE)
                            importance = inp_importance[inp_idx];
                        if (NEIGHBOR_IMPORTANCE) importance *= n_importance;

                        Eigen::Map<const Eigen::Matrix<TFeat, Eigen::Dynamic,
                                                       Eigen::Dynamic>>
                                A(filter + kernel_idx * out_channels *
                                                   in_channels,
                                  out_channels, in_channels);

                        Eigen::Map<const Eigen::Matrix<TFeat, Eigen::Dynamic,
                                                       Eigen::Dynamic>>
                                B(inp_features + inp_idx * in_channels,
                                  in_channels, 1);

                        C.col(out_col) +=
                                (A * (importance * B)).template cast<TOut>();
                    }

                }  // out_idx

                if (normalize) {
                    for (int i = 0; i < range_length; ++i) {
                        if (normalizers(i) != TOut(0))
                            C.col(i) /= normalizers(i);
                    }
                }
            });
}

/// Computes the output features of a sparse convolution.
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
/// \param num_inp    The number of input points.
///
/// \param inp_features    The input features with shape
///        [num_inp, in_channels].
///
/// \param inp_importance    Optional importance for each input point with
///        shape [num_inp]. Set to null to disable.
///
/// \param neighbors_index_size    The size of the neighbors_index array.
///
/// \param neighbors_index    The array with lists of neighbors for each
///        output point. The start and end of each sublist is defined by
///        \p neighbors_row_splits.
///
/// \param neighbors_kernel_index    Defines which kernel element to use for
///        each neighbor. This array has the same length as \p neighbors_index.
///
/// \param neighbors_importance    Array of the same length as
///        \p neighbors_importance. Defines which of the kernel elements to use
///        in the matrix multiplication.
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
void SparseConvComputeFeaturesCPU(TOut* out_features,
                                  const std::vector<int>& filter_dims,
                                  const TFeat* filter,
                                  size_t num_out,
                                  size_t num_inp,
                                  const TFeat* inp_features,
                                  const TFeat* inp_importance,
                                  size_t neighbors_index_size,
                                  const TIndex* neighbors_index,
                                  const TKernelIndex* neighbors_kernel_index,
                                  const TFeat* neighbors_importance,
                                  const int64_t* neighbors_row_splits,
                                  bool normalize) {
    // Dispatch all template parameter combinations
    bool has_importance = inp_importance;

#define FN_PARAMETERS                                                  \
    out_features, filter_dims, filter, num_out, num_inp, inp_features, \
            inp_importance, neighbors_index_size, neighbors_index,     \
            neighbors_kernel_index, neighbors_importance,              \
            neighbors_row_splits, normalize

#define CALL_TEMPLATE(HAS_IMPORTANCE)                                    \
    if (HAS_IMPORTANCE == has_importance)                                \
        _SparseConvComputeFeaturesCPU<TFeat, TOut, TIndex, TKernelIndex, \
                                      HAS_IMPORTANCE>(FN_PARAMETERS);

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
