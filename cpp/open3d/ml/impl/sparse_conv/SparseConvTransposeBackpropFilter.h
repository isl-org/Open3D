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
#include <mutex>

namespace open3d {
namespace ml {
namespace impl {

/// Implementation of SparseConvTransposeBackpropFilterCPU with template
/// parameters for configuration.
template <class TFeat,
          class TOut,
          class TIndex,
          class TKernelIndex,
          bool NORMALIZE>
void _SparseConvTransposeBackpropFilterCPU(
        TOut* filter_backprop,
        const std::vector<int>& filter_dims,
        size_t num_out,
        const TFeat* out_importance,
        size_t num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        const TIndex* neighbors_index,
        const TKernelIndex* neighbors_kernel_index,
        const TFeat* neighbors_importance,
        const int64_t* neighbors_row_splits,
        const TFeat* out_features_gradient) {
    const bool NEIGHBOR_IMPORTANCE = neighbors_importance;

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int num_kernel_elements = 1;
    for (int i = 0; i < filter_dims.size() - 2; ++i)
        num_kernel_elements *= filter_dims[i];

    memset(filter_backprop, 0,
           sizeof(TOut) * num_kernel_elements * in_channels * out_channels);
    std::mutex filter_backprop_mutex;

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_out, 32),
            [&](const tbb::blocked_range<size_t>& r) {
                int range_length = r.end() - r.begin();

                Eigen::Matrix<TFeat, Eigen::Dynamic, Eigen::Dynamic> B(
                        in_channels * num_kernel_elements, range_length);
                B.setZero();
                Eigen::Matrix<TFeat, Eigen::Dynamic, Eigen::Dynamic> C(
                        out_channels, range_length);

                Eigen::Array<TFeat, Eigen::Dynamic, 1> infeat(in_channels, 1);

                for (size_t out_idx = r.begin(); out_idx != r.end();
                     ++out_idx) {
                    const int out_col = out_idx - r.begin();
                    const size_t neighbor_start = neighbors_row_splits[out_idx];
                    const size_t neighbor_end =
                            neighbors_row_splits[out_idx + 1];

                    C.col(out_col) = Eigen::Map<
                            const Eigen::Array<TFeat, Eigen::Dynamic, 1>>(
                            out_features_gradient + out_idx * out_channels,
                            out_channels, 1);

                    for (size_t n = neighbor_start; n < neighbor_end; ++n) {
                        const size_t inp_idx = neighbors_index[n];
                        const int kernel_idx = neighbors_kernel_index[n];

                        TFeat n_importance = NEIGHBOR_IMPORTANCE
                                                     ? neighbors_importance[n]
                                                     : TFeat(1);
                        for (int ic = 0; ic < in_channels; ++ic)
                            infeat(ic) =
                                    inp_features[inp_idx * in_channels + ic] *
                                    n_importance;

                        if (NORMALIZE) {
                            TFeat normalizer(1);
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
                            for (int ic = 0; ic < in_channels; ++ic)
                                infeat(ic) *= normalizer;
                        }

                        for (int ic = 0; ic < in_channels; ++ic) {
                            B(kernel_idx * in_channels + ic, out_col) +=
                                    infeat(ic);
                        }
                    }

                }  // out_idx

                if (out_importance) {
                    for (size_t out_idx = r.begin(); out_idx != r.end();
                         ++out_idx) {
                        const int out_col = out_idx - r.begin();
                        C.col(out_col) *= out_importance[out_idx];
                    }
                }

                Eigen::Matrix<TOut, Eigen::Dynamic, Eigen::Dynamic> A(
                        out_channels, num_kernel_elements * in_channels);

                A = (C * B.transpose()).template cast<TOut>();

                {
                    std::lock_guard<std::mutex> lock(filter_backprop_mutex);
                    int linear_i = 0;
                    for (int j = 0; j < num_kernel_elements * in_channels; ++j)
                        for (int i = 0; i < out_channels; ++i, ++linear_i) {
                            filter_backprop[linear_i] += A(i, j);
                        }
                }
            });
}

/// Computes the backprop for the filter of a transpose sparse convolution.
///
/// \param filter_backrop    Output array for the computed filter gradient
///        with shape [depth,height,width, inp channels, out channels]
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
/// \param neighbors_index_size    The size of the neighbors_index array.
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
/// \param out_features_gradient    The gradient from the features with shape
///        [num_out, out_channels]
///
/// \param normalize    If true then the result is normalized either by the
///        number of points (neighbors_importance is null) or by the sum of
///        the respective values in neighbors_importance.
///
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvTransposeBackpropFilterCPU(
        TOut* filter_backprop,
        const std::vector<int>& filter_dims,
        size_t num_out,
        const TFeat* out_importance,
        size_t num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        const TIndex* neighbors_index,
        const TKernelIndex* neighbors_kernel_index,
        const TFeat* neighbors_importance,
        const int64_t* neighbors_row_splits,
        const TFeat* out_features_gradient,
        bool normalize) {
#define FN_PARAMETERS                                                          \
    filter_backprop, filter_dims, num_out, out_importance, num_inp,            \
            inp_features, inp_neighbors_importance_sum,                        \
            inp_neighbors_row_splits, neighbors_index, neighbors_kernel_index, \
            neighbors_importance, neighbors_row_splits, out_features_gradient

#define CALL_TEMPLATE(NORMALIZE)                                        \
    if (NORMALIZE == normalize)                                         \
        _SparseConvTransposeBackpropFilterCPU<TFeat, TOut, TIndex,      \
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
