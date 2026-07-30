// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Implementation of the shared column-major GEMM shim declared in GemmSYCL.h,
// used exactly like the CUDA path's
// `cutlass::gemm::device::Gemm<float, ColumnMajor, ...>`:
//     D = alpha * op(A) * op(B) + beta * C          (all matrices ColumnMajor)
//
// All sycl-tla template instantiation is confined to this single translation
// unit (see GemmSYCL.h) so the conv op TUs do not each recompile a full set of
// GEMM kernels through both the host and SPIR-V device passes.
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
//    with the original `ldc`. The public API still takes/returns column-major
//    A/B/C exactly like the CUDA path.
//
// GEMM tile shape (workgroup-level tile) is a compile-time performance
// parameter (see plan §9 hyperparameter tuning); the defaults below are
// portable, safe choices and are not Xe3-specific tuned.

#include "open3d/ml/impl/GemmSYCL.h"

#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/layout/matrix.h>

#include <array>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <sstream>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

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
cutlass::Status RunGemmXmxTf32RowMajorOutput(
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
        int64_t ldd,
        const std::vector<sycl::event>& deps = {}) {
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
    // sycl-tla's GemmUniversalAdapter::run() takes only a sycl::queue*, with
    // no event-dependency parameter, so an explicit barrier is needed to
    // make the GEMM's kernels wait for `deps` on a possibly out-of-order
    // queue (submission order alone would not guarantee this).
    if (!deps.empty()) {
        queue.ext_oneapi_submit_barrier(deps);
    }
    status = gemm_op.run(&queue);
    // sycl-tla's GemmUniversalAdapter::run() returns only a cutlass::Status,
    // not a sycl::event, so there is no completion event this function could
    // return for a caller to chain via depends_on() -- unlike the rest of
    // this codebase's event-based synchronization, this wait is a genuine
    // library-API-boundary necessity, not a lazy default between
    // Open3D-owned device-only stages.
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
        int64_t ldd,
        const std::vector<sycl::event>& deps = {}) {
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
                    cutlass::gemm::collective::KernelScheduleAuto>::
                    CollectiveOp;

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
    // See the analogous comment in RunGemmXmxTf32RowMajorOutput above: an
    // explicit barrier is needed since run() has no event-dependency param.
    if (!deps.empty()) {
        queue.ext_oneapi_submit_barrier(deps);
    }
    status = gemm_op.run(&queue);
    // See the analogous comment in RunGemmXmxTf32RowMajorOutput above: no
    // completion event is available from sycl-tla's run() to chain via
    // depends_on(), so this wait is a genuine library boundary, not a lazy
    // default.
    queue.wait_and_throw();
    if (workspace) sycl::free(workspace, queue);
    return status;
}

/// Collective (work-group) tile shapes. A 256x256 tile only pays off when
/// both M and N are large enough to fill it; the small per-chunk GEMMs
/// typical of the conv ops (~32 output columns per run, see
/// SparseConvSYCL.h/ContinuousConvSYCL.h) should start at SmallTile directly
/// instead of paying for a rejected (or under-filled, if it happens to
/// succeed) probe against the bigger tiles first.
using LargeTile = cute::Shape<cute::_256, cute::_256, cute::_32>;
using MediumTile = cute::Shape<cute::_64, cute::_64, cute::_16>;
using SmallTile = cute::Shape<cute::_16, cute::_16, cute::_8>;

/// Coarse GEMM problem-size class used to pick the starting collective tile
/// shape before falling back to other tiles via `can_implement`.
enum class GemmTileClass { kLarge, kMedium, kSmall };

// Best-guess specialization constants for the GemmTileClass thresholds,
// evaluated on the caller-facing (un-swapped) m/n/k. Tune these thresholds
// on target Intel GPU hardware later.
constexpr int kGemmLargeTileMinMN = 256;
constexpr int kGemmLargeTileMinK = 32;
constexpr int kGemmMediumTileMinMN = 64;
constexpr int kGemmMediumTileMinK = 16;

/// Returns the tile try-order, best fit for the problem size first. The
/// remaining tiles are correctness fallbacks for when `can_implement`
/// rejects the preferred tile, so every tile stays reachable.
std::array<GemmTileClass, 3> TileTryOrder(int m, int n, int k) {
    using T = GemmTileClass;
    if (m >= kGemmLargeTileMinMN && n >= kGemmLargeTileMinMN &&
        k >= kGemmLargeTileMinK) {
        return {T::kLarge, T::kMedium, T::kSmall};
    }
    if (m >= kGemmMediumTileMinMN && n >= kGemmMediumTileMinMN &&
        k >= kGemmMediumTileMinK) {
        return {T::kMedium, T::kLarge, T::kSmall};
    }
    return {T::kSmall, T::kMedium, T::kLarge};
}

/// GEMM arguments bundled to keep the tile dispatch below readable. Holds
/// references/pointers to caller-owned data; scoped to a single
/// GemmColumnMajorSYCL call.
struct GemmArgs {
    sycl::queue& queue;
    int m, n, k;
    float alpha;
    const float* A;
    int64_t lda;
    const float* B;
    int64_t ldb;
    float beta;
    const float* C;
    int64_t ldc;
    float* D;
    int64_t ldd;
    const std::vector<sycl::event>& deps;
};

/// Runs the TF32/XMX path with the requested tile.
template <class LayoutA, class LayoutB>
cutlass::Status RunTf32(GemmTileClass tile, const GemmArgs& a) {
    switch (tile) {
        case GemmTileClass::kLarge:
            return RunGemmXmxTf32RowMajorOutput<LargeTile, LayoutA, LayoutB>(
                    a.queue, a.m, a.n, a.k, a.alpha, a.A, a.lda, a.B, a.ldb,
                    a.beta, a.C, a.ldc, a.D, a.ldd, a.deps);
        case GemmTileClass::kMedium:
            return RunGemmXmxTf32RowMajorOutput<MediumTile, LayoutA, LayoutB>(
                    a.queue, a.m, a.n, a.k, a.alpha, a.A, a.lda, a.B, a.ldb,
                    a.beta, a.C, a.ldc, a.D, a.ldd, a.deps);
        case GemmTileClass::kSmall:
            return RunGemmXmxTf32RowMajorOutput<SmallTile, LayoutA, LayoutB>(
                    a.queue, a.m, a.n, a.k, a.alpha, a.A, a.lda, a.B, a.ldb,
                    a.beta, a.C, a.ldc, a.D, a.ldd, a.deps);
    }
    return cutlass::Status::kErrorNotSupported;
}

/// Runs the IEEE fp32 path with the requested tile. LargeTile is not
/// instantiated here (the SIMT path gains nothing from it), so kLarge is
/// reported unsupported and the caller moves on to the next tile.
template <class LayoutA, class LayoutB>
cutlass::Status RunIeee(GemmTileClass tile, const GemmArgs& a) {
    switch (tile) {
        case GemmTileClass::kMedium:
            return RunGemmIeeeFp32RowMajorOutput<MediumTile, LayoutA, LayoutB>(
                    a.queue, a.m, a.n, a.k, a.alpha, a.A, a.lda, a.B, a.ldb,
                    a.beta, a.C, a.ldc, a.D, a.ldd, a.deps);
        case GemmTileClass::kSmall:
            return RunGemmIeeeFp32RowMajorOutput<SmallTile, LayoutA, LayoutB>(
                    a.queue, a.m, a.n, a.k, a.alpha, a.A, a.lda, a.B, a.ldb,
                    a.beta, a.C, a.ldc, a.D, a.ldd, a.deps);
        case GemmTileClass::kLarge:
            return cutlass::Status::kErrorNotSupported;
    }
    return cutlass::Status::kErrorNotSupported;
}

}  // namespace sycl_gemm_detail

template <class LayoutA, class LayoutB>
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
                         bool allow_tf32,
                         const std::vector<sycl::event>& deps) {
    using namespace sycl_gemm_detail;
    // D_colmajor(M,N) = A*B  <=>  D_rowmajor(N,M) = B^T * A^T (same memory,
    // ld=ldc; see the file header). Swap the operands, swap M/N and transpose
    // the layout tags; C aliases D so we accumulate in place as CUDA does.
    using SwappedLayoutA = typename TransposeLayout<LayoutB>::type;
    using SwappedLayoutB = typename TransposeLayout<LayoutA>::type;

    const GemmArgs args{queue, n,    m, k,   alpha, B,   ldb, A,
                        lda,   beta, C, ldc, C,     ldc, deps};
    const std::array<GemmTileClass, 3> order = TileTryOrder(m, n, k);

    cutlass::Status status = cutlass::Status::kErrorNotSupported;
    if (allow_tf32) {
        for (GemmTileClass tile : order) {
            status = RunTf32<SwappedLayoutA, SwappedLayoutB>(tile, args);
            if (status == cutlass::Status::kSuccess) break;
        }
    }
    if (status != cutlass::Status::kSuccess) {
        for (GemmTileClass tile : order) {
            status = RunIeee<SwappedLayoutA, SwappedLayoutB>(tile, args);
            if (status == cutlass::Status::kSuccess) break;
        }
    }
    if (status != cutlass::Status::kSuccess) {
        std::ostringstream msg;
        msg << "GemmSYCL: sycl-tla GEMM cannot implement problem m=" << m
            << ", n=" << n << ", k=" << k << ", lda=" << lda << ", ldb=" << ldb
            << ", ldc=" << ldc << (allow_tf32 ? " (TF32 XMX)" : " (IEEE fp32)");
        throw std::runtime_error(msg.str());
    }
}

// The only two layout combinations used by Open3D's conv ops: forward ops
// pass both operands column-major (ContinuousConv.cuh), backprop-filter ops
// pass A column-major and B row-major (ContinuousConvBackpropFilter.cuh).
template void GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
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

template void GemmColumnMajorSYCL<cutlass::layout::ColumnMajor,
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
