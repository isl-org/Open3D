// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Column-major GEMM shim for the CUDA conv ops, matching the calling
// convention of GemmSYCL.h's GemmColumnMajorSYCL<LA,LB>():
//     D = alpha * A * B + beta * C
// with A(MxK), B(KxN), C/D(MxN) all column-major (C and D are the same buffer
// — accumulate in place — matching existing Open3D CUDA conv usage).
//
// Backend: CUTLASS v4.2.1 device::Gemm (v2-compatibility shim, kept in v4).
// - OperatorClass: OpClassSimt (SIMT FP32; no Tensor Cores, no alignment req).
// - ArchTag: Sm86 (Ampere GA10x — sm_86 — minimum). CUTLASS uses this tag for
//   algorithm selection; the PTX target arch is set separately via -arch nvcc.
//   Sm86 is the conservative Ampere target that covers all RTX 30xx and A-series
//   consumer GPUs. Tile defaults for OpClassSimt are ArchTag-independent.
// The calling .cuh files previously instantiated this Gemm type inline; this
// shim moves it here so the call sites mirror the GemmColumnMajorSYCL pattern.

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace open3d {
namespace ml {
namespace impl {

/// Runs alpha * A * B + beta * C on CUDA via CUTLASS device::Gemm.
/// A is (M x K, LayoutA), B is (K x N, LayoutB), C/D is (M x N, ColumnMajor).
/// C and D are the same buffer (in-place accumulation).
template <class LayoutA = cutlass::layout::ColumnMajor,
          class LayoutB = cutlass::layout::ColumnMajor>
void GemmColumnMajorCUDA(const cudaStream_t& stream,
                         int m,
                         int n,
                         int k,
                         float alpha,
                         const float* A,
                         int lda,
                         const float* B,
                         int ldb,
                         float beta,
                         float* C,
                         int ldc) {
    // OpClassSimt: SIMT FP32 — no alignment requirement (lda/ldb need not be
    // multiples of 4), so it works with arbitrary channel/kernel counts.
    // Sm86 selects Ampere-era algorithm defaults (tile 128×128×8, 2 stages).
    using Gemm = cutlass::gemm::device::Gemm<float,
                                             LayoutA,
                                             float,
                                             LayoutB,
                                             float,
                                             cutlass::layout::ColumnMajor,
                                             float,  // accumulator
                                             cutlass::arch::OpClassSimt,
                                             cutlass::arch::Sm86>;
    Gemm gemm_op;
    cutlass::Status status =
            gemm_op({{m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
                     {alpha, beta}},
                    nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM failed.");
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
