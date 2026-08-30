// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL port of ContinuousConvCUDAKernels.{h,cu}. Declares the FillColumn /
// FillColumnTranspose kernels (coordinate mapping + interpolation, one
// work-group per output point, ported from FillColumnKernel /
// FillColumnTransposeKernel) and the small element-wise "multiply by
// per-column scalar" kernels (used by the transpose/importance paths).
#pragma once

#include <sycl/sycl.hpp>
#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"

namespace open3d {
namespace ml {
namespace impl {

/// Copies and transforms the features to a column, which can be multiplied
/// with the filter matrix. SYCL port of FillColumn (see
/// ContinuousConvCUDAKernels.h for the full parameter documentation;
/// semantics are identical, only cudaStream_t -> sycl::queue&). Depends on
/// \p deps and returns the completion event of its last kernel; callers
/// thread this event into e.g. the following GEMM's dependencies instead of
/// blocking-waiting on it.
template <class TFeat, class TReal, class TIndex>
sycl::event FillColumnSYCL(sycl::queue& queue,
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
                           bool normalize,
                           const std::vector<sycl::event>& deps = {});

/// SYCL port of FillColumnTranspose. See ContinuousConvCUDAKernels.h for the
/// full parameter documentation; semantics are identical. Depends on \p deps
/// and returns the completion event of its last kernel.
template <class TFeat, class TReal, class TIndex>
sycl::event FillColumnTransposeSYCL(
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
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize,
        const std::vector<sycl::event>& deps = {});

/// Multiplies each column of a column-major [rows, cols] matrix in place by
/// the corresponding entry of \p vector (length cols). SYCL port of
/// MultiplyColumns. Depends on \p deps and returns the completion event.
template <class T>
sycl::event MultiplyColumnsSYCL(sycl::queue& queue,
                                size_t rows,
                                size_t cols,
                                T* col_major_matrix,
                                const T* const vector,
                                const std::vector<sycl::event>& deps = {});

/// Same as MultiplyColumnsSYCL but writes the result to \p out_ptr instead of
/// modifying \p col_major_matrix in place. SYCL port of
/// MultiplyAndCopyColumns. Depends on \p deps and returns the completion
/// event.
template <class T>
sycl::event MultiplyAndCopyColumnsSYCL(
        sycl::queue& queue,
        size_t rows,
        size_t cols,
        T* out_ptr,
        const T* const col_major_matrix,
        const T* const vector,
        const std::vector<sycl::event>& deps = {});

}  // namespace impl
}  // namespace ml
}  // namespace open3d
