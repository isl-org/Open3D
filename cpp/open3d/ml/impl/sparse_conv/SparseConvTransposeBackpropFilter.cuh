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
#define EIGEN_USE_GPU

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/sgemm_traits.h>

#include "open3d/ml/impl/continuous_conv/ContinuousConvCUDAKernels.h"
#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/ml/impl/sparse_conv/SparseConvCUDAKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

/// Computes the backprop for the filter of a transpose sparse convolution.
///
/// All pointer arguments point to device memory unless stated otherwise.
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
void SparseConvTransposeBackpropFilterCUDA(
        const cudaStream_t& stream,
        void* temp,
        size_t& temp_size,
        size_t& max_temp_size,
        int texture_alignment,
        TOut* filter_backprop,
        const std::vector<int>& filter_dims,
        TIndex num_out,
        const TFeat* out_importance,
        TIndex num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        size_t neighbors_index_size,
        const TIndex* neighbors_index,
        const TKernelIndex* neighbors_kernel_index,
        const TFeat* neighbors_importance,
        const int64_t* neighbors_row_splits,
        const TFeat* out_features_gradient,
        bool normalize) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int num_kernel_elements = 1;
    for (int i = 0; i < filter_dims.size() - 2; ++i)
        num_kernel_elements *= filter_dims[i];

    // this defines how much temporary storage we need at least
    // we want to allocate memory for at least 32 output points.
    const size_t min_num_cols_per_run = std::min(size_t(num_out), size_t(32));
    const size_t max_num_cols_per_run = num_out;
    size_t bytes_per_column =
            sizeof(TFeat) * (num_kernel_elements * in_channels);
    if (out_importance) bytes_per_column += sizeof(TFeat) * out_channels;
    const size_t min_temp_size_bytes = min_num_cols_per_run * bytes_per_column;
    const size_t max_temp_size_bytes = max_num_cols_per_run * bytes_per_column;

    if (get_temp_size) {
        std::pair<char*, size_t> tmp =
                mem_temp.Alloc<char>(min_temp_size_bytes);
        temp_size = mem_temp.MaxUsed();
        mem_temp.Free(tmp);
        mem_temp.Alloc<char>(max_temp_size_bytes);
        max_temp_size = mem_temp.MaxUsed();
        return;
    }

    std::pair<void*, size_t> mem_columns = mem_temp.AllocLargestSegment();

    const size_t num_cols_per_run =
            std::min(mem_columns.second / bytes_per_column, size_t(num_out));

    if (mem_columns.second < min_temp_size_bytes) {
        std::stringstream ss;
        ss << "temp is too small " << mem_columns.second
           << " bytes. Expected at least " << min_temp_size_bytes << " bytes\n";
        throw std::runtime_error(ss.str());
    }

    cudaMemsetAsync(
            filter_backprop, 0,
            sizeof(TOut) * num_kernel_elements * in_channels * out_channels,
            stream);

    typedef cutlass::gemm::SgemmTraits<
            cutlass::MatrixLayout::kColumnMajor,  // layout of A matrix
            cutlass::MatrixLayout::kRowMajor,     // layout of B matrix
            cutlass::Shape<8, 64, 64>             // threadblock tile size
            >
            GemmTraits;

    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

    TFeat* columns = (TFeat*)mem_columns.first;
    TFeat* gradient = ((TFeat*)mem_columns.first) +
                      num_cols_per_run * num_kernel_elements * in_channels;

    // if we cannot process all data at once we need multiple runs
    size_t num_runs = DivUp(num_out, num_cols_per_run);
    for (size_t run_i = 0; run_i < num_runs; ++run_i) {
        const TIndex begin_idx = run_i * num_cols_per_run;
        const TIndex end_idx =
                std::min(size_t(num_out), (run_i + 1) * num_cols_per_run);
        const size_t num_cols_this_run = end_idx - begin_idx;

        if (out_importance) {
            MultiplyAndCopyColumns(
                    stream, out_channels, num_cols_this_run, gradient,
                    out_features_gradient +
                            (run_i * num_cols_per_run * out_channels),
                    out_importance + (run_i * num_cols_per_run));
        } else {
            gradient = const_cast<TFeat*>(
                    out_features_gradient +
                    (run_i * num_cols_per_run * out_channels));
        }

        FillColumnTranspose<TFeat, TIndex>(
                stream, columns, in_channels, begin_idx, end_idx, num_out,
                num_inp, inp_features, inp_neighbors_importance_sum,
                inp_neighbors_row_splits, neighbors_index_size, neighbors_index,
                neighbors_kernel_index, neighbors_importance,
                neighbors_row_splits, num_kernel_elements, normalize);

        typename Gemm::Params params;
        // C is MxN
        // B is KxN
        // A is MxK
        int m = out_channels;
        int k = num_cols_this_run;
        int n = num_kernel_elements * in_channels;
        float alpha = 1;
        const float* const A = gradient;
        int lda = m;
        const float* const B = columns;
        int ldb = n;
        float beta = 1;
        float* C = filter_backprop;
        int ldc = m;

        int result =
                params.initialize(m,      // GEMM M dimension
                                  n,      // GEMM N dimension
                                  k,      // GEMM K dimension
                                  alpha,  // scalar alpha
                                  A,      // matrix A operand
                                  lda,
                                  B,  // matrix B operand
                                  ldb,
                                  beta,  // scalar beta
                                  C,     // source matrix C
                                  ldc,
                                  C,  // destination matrix C (may be different
                                  ldc);

        if (result) {
            throw std::runtime_error(
                    "Failed to initialize CUTLASS Gemm::Params object.");
        }

        Gemm::launch(params, stream);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
