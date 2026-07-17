// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL port of SparseConv.cuh. Structurally identical to the CUDA version:
// same two-pass temp-size query (see MemoryAllocation.h, reused verbatim -
// it is pure host-side bookkeeping, no CUDA/SYCL dependency), same
// chunked-GEMM loop over `num_cols_per_run` output points at a time. Only
// device-specific pieces changed: cudaStream_t -> sycl::queue&,
// cudaMemsetAsync -> queue.fill, CUTLASS 2.x device::Gemm -> the sycl-tla
// backed GemmColumnMajorSYCL shim (see GemmSYCL.h).
#pragma once

#include "open3d/ml/impl/GemmSYCL.h"
#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/ml/impl/sparse_conv/SparseConvSYCLKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

/// SYCL version of SparseConvComputeFeaturesCUDA. See SparseConv.cuh for the
/// full parameter documentation; semantics are identical.
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvComputeFeaturesSYCL(sycl::queue& queue,
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
                                   bool normalize,
                                   bool allow_tf32) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int num_kernel_elements = 1;
    for (size_t i = 0; i < filter_dims.size() - 2; ++i)
        num_kernel_elements *= filter_dims[i];

    // We want to allocate memory for at least 32 output points.
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

    std::pair<void*, size_t> mem_columns = mem_temp.AllocLargestSegment();

    if (mem_columns.second < min_temp_size_bytes) {
        std::stringstream ss;
        ss << "temp is too small " << mem_columns.second
           << " bytes. Expected at least " << min_temp_size_bytes << " bytes\n";
        throw std::runtime_error(ss.str());
    }

    queue.fill(out_features, TOut(0), size_t(num_out) * out_channels).wait();

    size_t num_cols_per_run =
            std::min(mem_columns.second / bytes_per_column, size_t(num_out));

    TFeat* columns = (TFeat*)mem_columns.first;

    size_t num_runs = DivUp(num_out, num_cols_per_run);
    for (size_t run_i = 0; run_i < num_runs; ++run_i) {
        const TIndex begin_idx = TIndex(run_i * num_cols_per_run);
        const TIndex end_idx = TIndex(
                std::min(size_t(num_out), (run_i + 1) * num_cols_per_run));
        const size_t num_cols_this_run = end_idx - begin_idx;

        FillColumnSYCL<TFeat, TIndex, TKernelIndex>(
                queue, columns, in_channels, begin_idx, end_idx, num_out,
                num_inp, inp_features, inp_importance, neighbors_index_size,
                neighbors_index, neighbors_kernel_index, neighbors_importance,
                neighbors_row_splits, num_kernel_elements, normalize);

        // C(MxN) = A(MxK) * B(KxN); A=filter, B=columns, C=out_features.
        const int m = out_channels;
        const int k = num_kernel_elements * in_channels;
        const int n = static_cast<int>(num_cols_this_run);
        const float alpha = 1;
        const float* const A = filter;
        const int lda = m;
        const float* const B = columns;
        const int ldb = k;
        const float beta = 1;
        float* C = out_features + run_i * num_cols_per_run * out_channels;
        const int ldc = m;

        GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
                            cutlass::layout::ColumnMajor>(
                queue, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
                allow_tf32);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
