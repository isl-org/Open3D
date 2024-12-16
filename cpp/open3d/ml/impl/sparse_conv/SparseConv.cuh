// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#define EIGEN_USE_GPU

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/sgemm_traits.h>

#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/ml/impl/sparse_conv/SparseConvCUDAKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

/// Computes the output features of a sparse convolution.
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
void SparseConvComputeFeaturesCUDA(const cudaStream_t& stream,
                                   void* temp,
                                   size_t& temp_size,
                                   size_t& max_temp_size,
                                   int texture_alignment,
                                   TOut* out_features,
                                   const std::vector<int>& filter_dims,
                                   const TFeat* filter,
                                   TIndex num_out,
                                   TIndex num_inp,
                                   const TFeat* inp_features,
                                   const TFeat* inp_importance,
                                   size_t neighbors_index_size,
                                   const TIndex* neighbors_index,
                                   const TKernelIndex* neighbors_kernel_index,
                                   const TFeat* neighbors_importance,
                                   const int64_t* neighbors_row_splits,
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

    // this defines how much temporary storage we need at least.
    // we want to allocate memory for at least 32 output points.
    const size_t min_num_cols_per_run = std::min(size_t(num_out), size_t(32));
    const size_t max_num_cols_per_run = num_out;
    const size_t bytes_per_column =
            sizeof(TFeat) * (num_kernel_elements * in_channels);
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

    // Request segment using all of the temporary memory
    std::pair<void*, size_t> mem_columns = mem_temp.AllocLargestSegment();

    if (mem_columns.second < min_temp_size_bytes) {
        std::stringstream ss;
        ss << "temp is too small " << mem_columns.second
           << " bytes. Expected at least " << min_temp_size_bytes << " bytes\n";
        throw std::runtime_error(ss.str());
    }

    // init output
    cudaMemsetAsync(out_features, 0, sizeof(TOut) * num_out * out_channels,
                    stream);

    size_t num_cols_per_run =
            std::min(mem_columns.second / bytes_per_column, size_t(num_out));

    typedef cutlass::gemm::SgemmTraits<
            cutlass::MatrixLayout::kColumnMajor,  // layout of A matrix (filter)
            cutlass::MatrixLayout::kColumnMajor,  // layout of B matrix
                                                  // (columns)
            cutlass::Shape<8, 64, 64>             // threadblock tile size
            >
            GemmTraits;

    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

    // this is the pointer to the patch matrix
    TFeat* columns = (TFeat*)mem_columns.first;

    // if we cannot process all data at once we need multiple runs
    const size_t num_runs = DivUp(num_out, num_cols_per_run);
    for (size_t run_i = 0; run_i < num_runs; ++run_i) {
        const TIndex begin_idx = run_i * num_cols_per_run;
        const TIndex end_idx =
                std::min(size_t(num_out), (run_i + 1) * num_cols_per_run);
        const size_t num_cols_this_run = end_idx - begin_idx;

        // compute the patch matrix
        FillColumn<TFeat, TIndex>(
                stream, columns, in_channels, begin_idx, end_idx, num_out,
                num_inp, inp_features, inp_importance, neighbors_index_size,
                neighbors_index, neighbors_kernel_index, neighbors_importance,
                neighbors_row_splits, num_kernel_elements, normalize);

        // C is MxN
        // B is KxN
        // A is MxK
        int m = out_channels;
        int k = num_kernel_elements * in_channels;
        int n = num_cols_this_run;
        float alpha = 1;
        const float* const A = filter;
        int lda = m;
        const float* const B = columns;
        int ldb = k;
        float beta = 1;
        float* C = out_features + (run_i * num_cols_per_run * out_channels);
        int ldc = m;

        typename Gemm::Params params;
        int result = params.initialize(m,      // GEMM M dimension
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
                                       C,  // destination matrix C
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
