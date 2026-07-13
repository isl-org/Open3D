// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#define EIGEN_USE_GPU

#include <cutlass/gemm/device/gemm.h>

#include "open3d/ml/impl/continuous_conv/ContinuousConvCUDAKernels.h"
#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/ml/impl/sparse_conv/SparseConvCUDAKernels.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvTransposeComputeFeaturesCUDA(
        const cudaStream_t& stream,
        void* temp,
        size_t& temp_size,
        size_t& max_temp_size,
        int texture_alignment,
        TOut* out_features,
        const std::vector<int>& filter_dims,
        const TFeat* filter,
        TIndex num_out,
        const TFeat* out_importance,
        TIndex num_inp,
        const TFeat* inp_features,
        const TFeat* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_prefix_sum,
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

    // SGEMM: C = alpha * A * B + beta * C using CUTLASS v2 device API.
    using Gemm = cutlass::gemm::device::Gemm<
            float, cutlass::layout::ColumnMajor,  // A: filter (column-major)
            float, cutlass::layout::ColumnMajor,  // B: columns (column-major)
            float, cutlass::layout::ColumnMajor   // C/D: output (column-major)
            >;

    TFeat* columns = (TFeat*)mem_columns.first;

    // if we cannot process all data at once we need multiple runs
    size_t num_runs = DivUp(num_out, num_cols_per_run);
    for (size_t run_i = 0; run_i < num_runs; ++run_i) {
        const TIndex begin_idx = run_i * num_cols_per_run;
        const TIndex end_idx =
                std::min(size_t(num_out), (run_i + 1) * num_cols_per_run);
        const size_t num_cols_this_run = end_idx - begin_idx;

        FillColumnTranspose<TFeat, TIndex, TKernelIndex>(
                stream, columns, in_channels, begin_idx, end_idx, num_out,
                num_inp, inp_features, inp_neighbors_importance_sum,
                inp_neighbors_prefix_sum, neighbors_index_size, neighbors_index,
                neighbors_kernel_index, neighbors_importance,
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

        Gemm gemm_op;
        cutlass::Status status = gemm_op({{m, n, k},
                                          {A, lda},
                                          {B, ldb},
                                          {C, ldc},
                                          {C, ldc},
                                          {alpha, beta}},
                                         nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM failed.");
        }
    }

    if (out_importance) {
        MultiplyColumns(stream, out_channels, num_out, out_features,
                        out_importance);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
