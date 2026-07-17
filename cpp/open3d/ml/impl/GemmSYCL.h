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
// Backend: sycl-tla (Intel's SYCL fork of CUTLASS, CUTLASS v4.2.1 API,
// CuTe/`CollectiveBuilder`/`GemmUniversalAdapter`). This is the *only* GEMM
// backend for SYCL conv ops (no oneMKL/CPU fallback exists or is used).
//
// Three design points, all verified against the actual sycl-tla v0.9.1 source
// (downloaded and syntax-checked with `icpx -fsycl -fsyntax-only` against a
// prototype using this exact API before writing this file):
//
// 1. Intel Xe DPAS (the XMX tensor-core instruction) has no plain fp32 x fp32
//    MMA mode (see cute/arch/mma_xe.hpp: only tf32/bf16/fp16/int8 combinations
//    are declared). This mirrors NVIDIA Ampere's "TF32" GEMM acceleration
//    trick: `cutlass::tfloat32_t` is a 4-byte type with the same in-memory
//    bit layout as `float` (see cutlass/tfloat32.h — it stores a plain
//    `uint32_t`), so `float*` device buffers can be `reinterpret_cast` to
//    `const tfloat32_t*` with no data conversion/copy; the DPAS unit then
//    reads the full float32 bit pattern and internally truncates the mantissa
//    for the multiply (~10 bit mantissa vs. fp32's 23 bits). This is the
//    standard, precedented way to accelerate fp32 GEMM on tensor-core
//    hardware and keeps accumulation (`ElementAccumulator`) and the output
//    (`ElementC`/`ElementD`) as full `float` — i.e. only the A/B multiply
//    inputs lose precision, not the accumulated result. This is an accepted
//    speed/precision trade-off for ML conv ops. The default IEEE path below
//    remains available when bit-exact fp32 semantics are required.
// 2. The default IEEE path uses sycl-tla's device-agnostic
//    `OpMultiplyAdd` collective with float32 inputs, accumulation, and output.
//    This is the non-tensor/SIMT path and preserves IEEE float32 semantics.
//    Its aliases accept both operand layout combinations used by convolution.
// 3. sycl-tla's Xe epilogue collective builder only supports row-major
//    (N-major) output D/C (verified: `xe_builder.inl` static_asserts on this).
//    To still expose a *column-major* C/D (matching the CUDA calling
//    convention used throughout Open3D's conv kernels) with zero extra data
//    movement, this shim uses the standard row/column-major transpose
//    duality: a ColumnMajor(M,N) matrix with leading dimension ld is
//    bit-identical in memory to a RowMajor(N,M) matrix with the same leading
//    dimension. So `D_colmajor(M,N) = A*B` is computed instead as
//    `D_rowmajor(N,M) = B^T * A^T`, i.e. by swapping the A/B operands (and
//    transposing their layout tags — ColumnMajor<->RowMajor, no data change)
//    and swapping M/N, then telling sycl-tla to produce a RowMajor output
//    with the original `ldc`. The public API below still takes/returns
//    column-major A/B/C exactly like the CUDA path.
//
// GEMM tile shape (workgroup-level tile) is a compile-time performance
// parameter (see plan §9 hyperparameter tuning); the default below is a
// portable, safe choice and is not Xe3-specific tuned.

#pragma once

#include <sstream>

#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/layout/matrix.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace impl {

namespace sycl_gemm_detail {

/// Maps a CUTLASS layout tag to its transpose (ColumnMajor <-> RowMajor);
/// re-interpreting the same buffer/leading-dimension with the transposed tag
/// yields the mathematical transpose with no data movement.
template <class Layout>
struct TransposeLayout;
template <>
struct TransposeLayout<cutlass::layout::ColumnMajor> {
    using type = cutlass::layout::RowMajor;
};
template <>
struct TransposeLayout<cutlass::layout::RowMajor> {
    using type = cutlass::layout::ColumnMajor;
};

/// Builds the CuTe stride for the A operand (canonical CuTe modes [M,K,L])
/// from a runtime leading dimension, matching
/// cutlass::detail::TagToStrideA_t's convention (unit-stride dim is a
/// compile-time cute::Int<1>): RowMajor -> (ld, 1, batch), ColumnMajor ->
/// (1, ld, batch).
template <class Layout>
auto MakeStrideA(int64_t ld) {
    if constexpr (cute::is_same_v<Layout, cutlass::layout::RowMajor>) {
        return cute::make_stride(ld, cute::Int<1>{}, int64_t(0));
    } else {
        return cute::make_stride(cute::Int<1>{}, ld, int64_t(0));
    }
}

/// Builds the CuTe stride for the B operand. Note: cutlass::detail::
/// TagToStrideB_t is in *canonical CuTe modes [N,K,L]* — the *opposite*
/// convention from A/C/D — so RowMajor/ColumnMajor map to the reverse
/// stride pattern compared to MakeStrideA: RowMajor -> (1, ld, batch),
/// ColumnMajor -> (ld, 1, batch). (Verified against
/// cutlass/detail/layout.hpp; this asymmetry is intentional in CUTLASS/CuTe,
/// reflecting that B's mathematical (K,N) shape is expressed as a (N,K)
/// CuTe tensor.)
template <class Layout>
auto MakeStrideB(int64_t ld) {
    if constexpr (cute::is_same_v<Layout, cutlass::layout::RowMajor>) {
        return cute::make_stride(cute::Int<1>{}, ld, int64_t(0));
    } else {
        return cute::make_stride(ld, cute::Int<1>{}, int64_t(0));
    }
}

/// Runs alpha * op(A) * op(B) + beta * C on the Intel Xe DPAS path via
/// sycl-tla, producing a RowMajor (M x N) output D (D may alias C).
/// A is (M x K, LayoutA), B is (K x N, LayoutB). GEMM element type is
/// cutlass::tfloat32_t (see file header); accumulation/output stay `float`.
template <class TileShape, class LayoutA, class LayoutB>
cutlass::Status RunGemmXmxTf32RowMajorOutput(sycl::queue& queue,
                                             int m,
                                             int n,
                                             int k,
                                             float alpha,
                                             const float* A,
                                             int64_t lda,
                                             const float* B,
                                             int64_t ldb,
                                             float beta,
                                             const float* C,
                                             int64_t ldc,
                                             float* D,
                                             int64_t ldd) {
    using ElementA = cutlass::tfloat32_t;
    using ElementB = cutlass::tfloat32_t;
    using ElementAccumulator = float;
    using ElementC = float;
    using ElementOutput = float;
    using ElementComputeEpilogue = float;

    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // Alignment is expressed in elements; tfloat32_t/float are both 4 bytes,
    // and DPAS requires natural alignment for its data type.
    constexpr int AlignmentA = 4;
    constexpr int AlignmentB = 4;
    constexpr int AlignmentC = 4;
    constexpr int AlignmentD = 4;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, ElementA,
            LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
            ElementAccumulator, TileShape,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::gemm::collective::StageCountAuto,
            cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp, TileShape,
            cute::Shape<cute::_1, cute::_1, cute::_1>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementComputeEpilogue, ElementAccumulator, ElementC, LayoutC,
            AlignmentC, ElementOutput, LayoutD, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int, int>, CollectiveMainloop,
            CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    // Build CuTe strides directly from the caller-provided leading
    // dimensions (general strided case, not just the packed/contiguous
    // case): ColumnMajor (rows x cols) -> stride (1, ld); RowMajor
    // (rows x cols) -> stride (ld, 1); batch (L=1) stride is unused (0).
    // The unit-stride dimension's type is a compile-time cute::C<1> (fixed
    // by TagToStrideA_t/TagToStrideB_t), so it must be constructed in place
    // per-layout rather than through a single runtme-typed helper.
    StrideA stride_A = MakeStrideA<LayoutA>(lda);
    StrideB stride_B = MakeStrideB<LayoutB>(ldb);
    // C/D are always RowMajor here (rows=m, cols=n) -> stride (ld, 1).
    StrideC stride_C = cute::make_stride(ldc, cute::Int<1>{}, int64_t(0));
    StrideD stride_D = cute::make_stride(ldd, cute::Int<1>{}, int64_t(0));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                    hw_info.device_id);

    typename Gemm::GemmKernel::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {m, n, k, 1},
            {reinterpret_cast<const ElementA*>(A), stride_A,
             reinterpret_cast<const ElementB*>(B), stride_B},
             {{alpha, beta}, C, stride_C, D, stride_D},
            hw_info};

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = sycl::malloc_device(workspace_size, queue);
    }

    auto status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) sycl::free(workspace, queue);
        return status;
    }
    status = gemm_op.initialize(arguments, workspace, &queue);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) sycl::free(workspace, queue);
        return status;
    }
    status = gemm_op.run(&queue);
    queue.wait_and_throw();
    if (workspace) sycl::free(workspace, queue);
    return status;
}

/// Runs IEEE float32 GEMM through sycl-tla's device-agnostic path.
template <class TileShape, class LayoutA, class LayoutB>
cutlass::Status RunGemmIeeeFp32RowMajorOutput(
        sycl::queue& queue,
        int m,
        int n,
        int k,
        float alpha,
        const float* A,
        int64_t lda,
        const float* B,
        int64_t ldb,
        float beta,
        const float* C,
        int64_t ldc,
        float* D,
        int64_t ldd) {
    using ElementA = float;
    using ElementB = float;
    using ElementAccumulator = float;
    using ElementC = float;
    using ElementOutput = float;
    using ElementComputeEpilogue = float;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    constexpr int AlignmentA = sizeof(ElementA);
    constexpr int AlignmentB = sizeof(ElementB);
    constexpr int AlignmentC = sizeof(ElementC);
    constexpr int AlignmentD = sizeof(ElementOutput);

    using CollectiveMainloop =
            typename cutlass::gemm::collective::CollectiveBuilder<
                    cutlass::arch::Agnostic, cutlass::arch::OpMultiplyAdd,
                    ElementA, LayoutA, AlignmentA, ElementB, LayoutB,
                    AlignmentB, ElementAccumulator, TileShape,
                    cute::Shape<cute::_1, cute::_1, cute::_1>,
                    cutlass::gemm::collective::StageCountAuto,
                    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
            ElementOutput, ElementComputeEpilogue, ElementAccumulator,
            ElementAccumulator>;
    using CollectiveEpilogue =
            typename cutlass::epilogue::collective::CollectiveBuilder<
                    cutlass::arch::Agnostic, cutlass::arch::OpMultiplyAdd,
                    TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>,
                    cutlass::epilogue::collective::EpilogueTileAuto,
                    ElementComputeEpilogue, ElementAccumulator, ElementC,
                    LayoutC, AlignmentC, ElementOutput, LayoutD, AlignmentD,
                    cutlass::epilogue::collective::EpilogueScheduleAuto,
                    EpilogueOp>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int, int>, CollectiveMainloop,
            CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = MakeStrideA<LayoutA>(lda);
    StrideB stride_B = MakeStrideB<LayoutB>(ldb);
    StrideC stride_C = cute::make_stride(ldc, cute::Int<1>{}, int64_t(0));
    StrideD stride_D = cute::make_stride(ldd, cute::Int<1>{}, int64_t(0));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                    hw_info.device_id);

    typename Gemm::GemmKernel::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {m, n, k, 1},
            {A, stride_A, B, stride_B},
            {{alpha, beta}, C, stride_C, D, stride_D},
            hw_info};

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = sycl::malloc_device(workspace_size, queue);
    }

    auto status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) sycl::free(workspace, queue);
        return status;
    }
    status = gemm_op.initialize(arguments, workspace, &queue);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) sycl::free(workspace, queue);
        return status;
    }
    status = gemm_op.run(&queue);
    queue.wait_and_throw();
    if (workspace) sycl::free(workspace, queue);
    return status;
}

}  // namespace sycl_gemm_detail

/// Column-major GEMM shim matching the CUDA conv ops' calling convention:
///     D = alpha * A * B + beta * C
/// with A (M x K), B (K x N), C/D (M x N) all **column-major** with the given
/// leading dimensions (C and D may be the same buffer, i.e. accumulate into
/// C in place, matching the CUDA usage). \p LayoutA / \p LayoutB select
/// whether the corresponding operand is stored column-major
/// (`cutlass::layout::ColumnMajor`) or row-major (`cutlass::layout::RowMajor`)
/// — Open3D's conv kernels use both combinations (see ContinuousConv.cuh:
/// A,B both ColumnMajor; ContinuousConvBackpropFilter.cuh: A ColumnMajor,
/// B RowMajor).
///
/// Backend: sycl-tla only. \p allow_tf32 selects the Intel XMX TF32 path;
/// false selects the device-agnostic IEEE float32 path. Unsupported shapes
/// throw instead of falling back to a custom GEMM kernel.
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
                         bool allow_tf32 = false) {
    using namespace sycl_gemm_detail;
    // D_colmajor(M,N) = A*B  <=>  D_rowmajor(N,M) = B^T * A^T (same memory,
    // ld=ldc; see file header). Swap operands, swap M/N, transpose layout
    // tags; C aliases D (accumulate in place), matching the CUDA behavior.
    using SwappedLayoutA = typename TransposeLayout<LayoutB>::type;
    using SwappedLayoutB = typename TransposeLayout<LayoutA>::type;
    using LargeTile = cute::Shape<cute::_256, cute::_256, cute::_32>;
    using MediumTile = cute::Shape<cute::_64, cute::_64, cute::_16>;
    using SmallTile = cute::Shape<cute::_16, cute::_16, cute::_8>;

    cutlass::Status status = cutlass::Status::kErrorNotSupported;
    if (allow_tf32) {
        status = RunGemmXmxTf32RowMajorOutput<LargeTile, SwappedLayoutA,
                                              SwappedLayoutB>(
                queue, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
        if (status != cutlass::Status::kSuccess) {
            status = RunGemmXmxTf32RowMajorOutput<MediumTile, SwappedLayoutA,
                                                  SwappedLayoutB>(
                    queue, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, C,
                    ldc);
        }
        if (status != cutlass::Status::kSuccess) {
            status = RunGemmXmxTf32RowMajorOutput<SmallTile, SwappedLayoutA,
                                                  SwappedLayoutB>(
                    queue, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, C,
                    ldc);
        }
    }

    if (status != cutlass::Status::kSuccess) {
        status = RunGemmIeeeFp32RowMajorOutput<MediumTile, SwappedLayoutA,
                                               SwappedLayoutB>(
                queue, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, C, ldc);
        if (status != cutlass::Status::kSuccess) {
            status = RunGemmIeeeFp32RowMajorOutput<SmallTile, SwappedLayoutA,
                                                   SwappedLayoutB>(
                    queue, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc, C,
                    ldc);
        }
    }
    if (status != cutlass::Status::kSuccess) {
        std::ostringstream msg;
        msg << "GemmSYCL: sycl-tla GEMM cannot implement problem m=" << m
            << ", n=" << n << ", k=" << k << ", lda=" << lda
            << ", ldb=" << ldb << ", ldc=" << ldc
            << (allow_tf32 ? " (TF32 XMX)" : " (IEEE fp32)");
        throw std::runtime_error(msg.str());
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
