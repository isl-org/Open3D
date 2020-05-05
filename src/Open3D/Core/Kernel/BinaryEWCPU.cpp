// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "Open3D/Core/Kernel/BinaryEW.h"

#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/CPULauncher.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static void CPUAddElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) +
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUSubElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) -
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUMulElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) *
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUDivElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) /
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename src_t, typename dst_t>
static void CPULogicalAndElementKernel(const void* lhs,
                                       const void* rhs,
                                       void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) &&
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPULogicalOrElementKernel(const void* lhs,
                                      const void* rhs,
                                      void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) ||
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPULogicalXorElementKernel(const void* lhs,
                                       const void* rhs,
                                       void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) !=
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPUGtElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) > *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPULtElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) < *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUGeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) >= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPULeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) <= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUEqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) == *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUNeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) != *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void LaunchBoolBinaryEWCPUKernel(const Tensor& lhs,
                                        const Tensor& rhs,
                                        Tensor& dst,
                                        BinaryEWOpCode op_code,
                                        const Indexer& indexer) {
    switch (op_code) {
        case BinaryEWOpCode::LogicalAnd:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPULogicalAndElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::LogicalOr:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPULogicalOrElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::LogicalXor:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPULogicalXorElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Gt:
            CPULauncher::LaunchBinaryEWKernel(indexer,
                                              CPUGtElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Lt:
            CPULauncher::LaunchBinaryEWKernel(indexer,
                                              CPULtElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Ge:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPUGeqElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Le:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPULeqElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Eq:
            CPULauncher::LaunchBinaryEWKernel(indexer,
                                              CPUEqElementKernel<src_t, dst_t>);
            break;
        case BinaryEWOpCode::Ne:
            CPULauncher::LaunchBinaryEWKernel(
                    indexer, CPUNeqElementKernel<src_t, dst_t>);
            break;
        default:
            break;
    }
}

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code) {
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Indexer indexer({lhs, rhs}, dst, DtypePolicy::ASSERT_SAME_OR_BOOL_OUT);

    if (s_boolean_binary_ew_op_codes.find(op_code) !=
        s_boolean_binary_ew_op_codes.end()) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (src_dtype == dst_dtype) {
                // Inplace boolean op's output type is the same as the
                // input. e.g. np.logical_and(a, b, out=a), where a, b are
                // floats.
                LaunchBoolBinaryEWCPUKernel<scalar_t, scalar_t>(
                        lhs, rhs, dst, op_code, indexer);
            } else {
                // By default, output is boolean type.
                if (dst_dtype != Dtype::Bool) {
                    utility::LogError(
                            "Boolean op's output type must be boolean or the "
                            "same type as the input.");
                }
                LaunchBoolBinaryEWCPUKernel<scalar_t, bool>(lhs, rhs, dst,
                                                            op_code, indexer);
            }

        });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Add:
                    CPULauncher::LaunchBinaryEWKernel(
                            indexer, CPUAddElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Sub:
                    CPULauncher::LaunchBinaryEWKernel(
                            indexer, CPUSubElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Mul:
                    CPULauncher::LaunchBinaryEWKernel(
                            indexer, CPUMulElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Div:
                    CPULauncher::LaunchBinaryEWKernel(
                            indexer, CPUDivElementKernel<scalar_t>);
                    break;
                default:
                    break;
            }
        });
    }
}

}  // namespace kernel
}  // namespace open3d
