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

// Implementation of SparseConvBackropFilterCPU
template <class TFeat,
          class TOut,
          class TIndex,
          class TKernelIndex,
          bool POINT_IMPORTANCE>
void _SparseConvBackropFilterCPU(TOut* filter_backprop,
                                 const std::vector<int>& filter_dims,
                                 size_t num_out,
                                 size_t num_inp,
                                 const TFeat* inp_features,
                                 const TFeat* inp_importance,
                                 const TIndex* neighbors_index,
                                 const TKernelIndex* neighbors_kernel_index,
                                 const TFeat* neighbors_importance,
                                 const int64_t* neighbors_row_splits,
                                 const TFeat* out_features_gradient,
                                 bool normalize) {
    const bool NEIGHBOR_IMPORTANCE = neighbors_importance;

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int num_kernel_elements = 1;
    for (int i = 0; i < filter_dims.size() - 2; ++i)
        num_kernel_elements *= filter_dims[i];
    const int total_filter_size =
            num_kernel_elements * in_channels * out_channels;

    memset(filter_backprop, 0, sizeof(TOut) * total_filter_size);
    std::mutex filter_backprop_mutex;

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_out, 10032),
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
                    TFeat normalizer(0);

                    for (size_t n = neighbor_start; n < neighbor_end; ++n) {
                        const size_t inp_idx = neighbors_index[n];
                        const int kernel_idx = neighbors_kernel_index[n];

                        const TFeat n_importance =
                                (NEIGHBOR_IMPORTANCE ? neighbors_importance[n]
                                                     : TFeat(1));
                        normalizer += n_importance;

                        for (int ic = 0; ic < in_channels; ++ic)
                            infeat(ic) =
                                    inp_features[inp_idx * in_channels + ic];

                        TFeat importance(1);
                        if (POINT_IMPORTANCE)
                            importance = inp_importance[inp_idx];
                        if (NEIGHBOR_IMPORTANCE) importance *= n_importance;

                        if (POINT_IMPORTANCE || NEIGHBOR_IMPORTANCE) {
                            for (int ic = 0; ic < in_channels; ++ic)
                                infeat(ic) *= importance;
                        }
                        for (int ic = 0; ic < in_channels; ++ic) {
                            B(kernel_idx * in_channels + ic, out_col) =
                                    infeat(ic);
                        }
                    }

                    C.col(out_col) = Eigen::Map<
                            const Eigen::Array<TFeat, Eigen::Dynamic, 1>>(
                            out_features_gradient + out_idx * out_channels,
                            out_channels, 1);

                    if (normalize && normalizer != TFeat(0))
                        C.col(out_col) /= normalizer;

                }  // out_idx

                Eigen::Matrix<TFeat, Eigen::Dynamic, Eigen::Dynamic> A(
                        out_channels, num_kernel_elements * in_channels);

                A = C * B.transpose();

                {
                    std::lock_guard<std::mutex> lock(filter_backprop_mutex);
                    int linear_i = 0;
                    for (int j = 0; j < num_kernel_elements * in_channels; ++j)
                        for (int i = 0; i < out_channels; ++i, ++linear_i) {
                            filter_backprop[linear_i] += TOut(A(i, j));
                        }
                }
            });
}

/// Computes the backprop for the filter of a sparse convolution.
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done. This function can make use of more memory and
///        returns the maximum size that can be used in max_temp_size.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr and returns the minimum temp
///        size required.
///
/// \param max_temp_size    This is used as an output if temp is nullptr and
///        returns the maximum temp size that can be used.
///
/// \param texture_alignment    The texture alignment in bytes. This is used
///        for allocating segments within the temporary memory.
///
/// \param filter_backrop    Output array for the computed filter gradient
///        with shape [depth,height,width, inp channels, out channels]
///
/// \param filter_dims    The sizes of the filter dimensions. The size of
///        filter_dims must be >=3. The order is
///        [num kernel elements, inp channels, out channels].
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
/// \param out_features_gradient   The gradient signal from the features.
///
/// \param normalize    If true then the output features are normalized either
///        by the number of points (neighbors_importance is null) or by the sum
///        of the respective values in neighbors_importance.
///
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvBackpropFilterCPU(TOut* filter_backprop,
                                 const std::vector<int>& filter_dims,
                                 size_t num_out,
                                 size_t num_inp,
                                 const TFeat* inp_features,
                                 const TFeat* inp_importance,
                                 const TIndex* neighbors_index,
                                 const TKernelIndex* neighbors_kernel_index,
                                 const TFeat* neighbors_importance,
                                 const int64_t* neighbors_row_splits,
                                 const TFeat* out_features_gradient,
                                 bool normalize) {
    bool has_importance = inp_importance;

#define FN_PARAMETERS                                                          \
    filter_backprop, filter_dims, num_out, num_inp, inp_features,              \
            inp_importance, neighbors_index, neighbors_kernel_index,           \
            neighbors_importance, neighbors_row_splits, out_features_gradient, \
            normalize

#define CALL_TEMPLATE(HAS_IMPORTANCE)                                  \
    if (HAS_IMPORTANCE == has_importance)                              \
        _SparseConvBackropFilterCPU<TFeat, TOut, TIndex, TKernelIndex, \
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
