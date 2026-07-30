// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Shared column-major GEMM shim for the SYCL conv ops (ContinuousConv,
// SparseConv and their Backprop/Transpose variants), used exactly like the
// CUDA path's `cutlass::gemm::device::Gemm<float, ColumnMajor, ...>`:
//     D = alpha * op(A) * op(B) + beta * C          (all matrices ColumnMajor)
//
// Declaration only: the sycl-tla (CUTLASS-for-SYCL) template machinery is
// instantiated once in GemmSYCL.cpp, so the eight conv op translation units
// that call this do not each recompile a full set of GEMM kernels through
// both the host and SPIR-V device passes. See GemmSYCL.cpp for the backend
// design notes (TF32/XMX vs. IEEE fp32 paths, the row/column-major transpose
// duality used to expose a column-major output, and tile selection).
#pragma once

#include <cutlass/layout/matrix.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace open3d {
namespace ml {
namespace impl {

/// Column-major GEMM matching the CUDA conv ops' calling convention:
///     C = alpha * A * B + beta * C
/// with A (M x K), B (K x N) and C (M x N) all **column-major** with the
/// given leading dimensions; C is both read and accumulated into in place.
///
/// \p LayoutA / \p LayoutB select whether the corresponding operand is stored
/// column-major (`cutlass::layout::ColumnMajor`) or row-major
/// (`cutlass::layout::RowMajor`). Open3D's conv kernels use
/// <ColumnMajor, ColumnMajor> for the forward ops (see ContinuousConv.cuh)
/// and <ColumnMajor, RowMajor> for the backprop-filter ops (see
/// ContinuousConvBackpropFilter.cuh); only those two combinations are
/// instantiated, so any other combination is a link error rather than a
/// silent fallback.
///
/// \p allow_tf32 selects the Intel XMX TF32 path; false selects the
/// device-agnostic IEEE float32 path. \p deps lists events the GEMM must wait
/// on before reading its operands. Blocks until the GEMM completes.
///
/// Throws std::runtime_error if no supported tile configuration can implement
/// the requested problem shape.
///
/// \code
/// // Accumulate a chunk's contribution into filter_backprop (B row-major):
/// GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
///                     cutlass::layout::RowMajor>(
///         queue, out_channels, spatial_filter_size * in_channels,
///         num_cols_this_run, /*alpha*/ 1.f, out_features_gradient, lda,
///         columns, ldb, /*beta*/ 1.f, filter_backprop, ldc, allow_tf32,
///         {fill_column_event});
/// \endcode
template <class LayoutA = cutlass::layout::ColumnMajor,
          class LayoutB = cutlass::layout::ColumnMajor>
void GemmColumnMajorSYCL(sycl::queue& queue,
                         int m,
                         int n,
                         int k,
                         float alpha,
                         const float* A,
                         int64_t lda,
                         const float* B,
                         int64_t ldb,
                         float beta,
                         float* C,
                         int64_t ldc,
                         bool allow_tf32 = false,
                         const std::vector<sycl::event>& deps = {});

// Explicit instantiation declarations; definitions live in GemmSYCL.cpp.
extern template void GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
                                         cutlass::layout::ColumnMajor>(
        sycl::queue&,
        int,
        int,
        int,
        float,
        const float*,
        int64_t,
        const float*,
        int64_t,
        float,
        float*,
        int64_t,
        bool,
        const std::vector<sycl::event>&);

extern template void GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
                                         cutlass::layout::RowMajor>(
        sycl::queue&,
        int,
        int,
        int,
        float,
        const float*,
        int64_t,
        const float*,
        int64_t,
        float,
        float*,
        int64_t,
        bool,
        const std::vector<sycl::event>&);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
