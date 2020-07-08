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

#pragma once
#define EIGEN_USE_GPU

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/sgemm_traits.h>

#include "open3d/ml/impl/continuous_conv/ContinuousConvCUDAKernels.h"
#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/utility/Helper.h"

using open3d::utility::DivUp;

namespace open3d {
namespace ml {
namespace impl {

template <class TReal, class TIndex>
void CConvTransposeComputeFeaturesCUDA(
        const cudaStream_t& stream,
        void* temp,
        size_t& temp_size,
        size_t& max_temp_size,
        int texture_alignment,
        TReal* out_features,
        const std::vector<int>& filter_dims,
        const TReal* filter,
        TIndex num_out,
        const TReal* out_positions,
        const TReal* out_importance,
        TIndex num_inp,
        const TReal* inp_positions,
        const TReal* inp_features,
        const TReal* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* neighbors_index,
        const TReal* neighbors_importance,
        const int64_t* neighbors_row_splits,
        const TReal* extents,
        const TReal* offsets,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int spatial_filter_size = 1;
    for (int i = 0; i < 3; ++i) spatial_filter_size *= filter_dims[i];

    // this defines how much temporary storage we need at least.
    // we want to allocate memory for at least 32 output points.
    const size_t min_num_cols_per_run = std::min(size_t(num_out), size_t(32));
    const size_t max_num_cols_per_run = num_out;
    const size_t bytes_per_column =
            sizeof(TReal) * (spatial_filter_size * in_channels);
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
    cudaMemsetAsync(out_features, 0, sizeof(TReal) * num_out * out_channels,
                    stream);

    size_t num_cols_per_run =
            std::min(mem_columns.second / bytes_per_column, size_t(num_out));

    typedef cutlass::gemm::SgemmTraits<
            cutlass::MatrixLayout::kColumnMajor,  // layout of A matrix
            cutlass::MatrixLayout::kColumnMajor,  // layout of B matrix
            cutlass::Shape<8, 64, 64>             // threadblock tile size
            >
            GemmTraits;

    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

    TReal* columns = (TReal*)mem_columns.first;

    // if we cannot process all data at once we need multiple runs
    size_t num_runs = DivUp(num_out, num_cols_per_run);
    for (size_t run_i = 0; run_i < num_runs; ++run_i) {
        const TIndex begin_idx = run_i * num_cols_per_run;
        const TIndex end_idx =
                std::min(size_t(num_out), (run_i + 1) * num_cols_per_run);
        const size_t num_cols_this_run = end_idx - begin_idx;

        FillColumnTranspose<TReal, TIndex>(
                stream, columns, in_channels, begin_idx, end_idx, num_out,
                out_positions, num_inp, inp_positions, inp_features,
                inp_neighbors_importance_sum, inp_neighbors_prefix_sum,
                neighbors_index_size, neighbors_index, neighbors_importance,
                neighbors_row_splits, extents, offsets, filter_dims,
                interpolation, coordinate_mapping, align_corners,
                individual_extent, isotropic_extent, normalize);

        typename Gemm::Params params;
        // C is MxN
        // B is KxN
        // A is MxK
        int m = out_channels;
        int k = spatial_filter_size * in_channels;
        int n = num_cols_this_run;
        float alpha = 1;
        const float* const A = filter;
        int lda = m;
        const float* const B = columns;
        int ldb = k;
        float beta = 1;
        float* C = out_features + (run_i * num_cols_per_run * out_channels);
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
                                      // memory than source C matrix)
                                  ldc);

        if (result) {
            throw std::runtime_error(
                    "Failed to initialize CUTLASS Gemm::Params object.");
        }

        Gemm::launch(params, stream);
    }

    if (out_importance) {
        MultiplyColumns(stream, out_channels, num_out, out_features,
                        out_importance);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
