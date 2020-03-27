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

#include "Open3D/Core/Kernel/UnaryEW.h"

#include <cmath>

#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/CPULauncher.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

template <typename src_t, typename dst_t>
static void CPUCopyElementKernel(const void* src, void* dst) {
    *static_cast<dst_t*>(dst) =
            static_cast<dst_t>(*static_cast<const src_t*>(src));
}

template <typename scalar_t>
static void CPUSqrtElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            std::sqrt(*static_cast<const scalar_t*>(src));
}

template <typename scalar_t>
static void CPUSinElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = std::sin(*static_cast<const scalar_t*>(src));
}

template <typename scalar_t>
static void CPUCosElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = std::cos(*static_cast<const scalar_t*>(src));
}

template <typename scalar_t>
static void CPUNegElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = -*static_cast<const scalar_t*>(src);
}

template <typename scalar_t>
static void CPUExpElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = std::exp(*static_cast<const scalar_t*>(src));
}

template <typename scalar_t>
static void CPUAbsElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            std::abs(static_cast<double>(*static_cast<const scalar_t*>(src)));
}

template <typename src_t, typename dst_t>
static void CPULogicalNotElementKernel(const void* src, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            !static_cast<bool>(*static_cast<const src_t*>(src)));
}

void CopyCPU(const Tensor& src, Tensor& dst) {
    // src and dst have been checked to have the same shape, dtype, device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    if (src.IsContiguous() && dst.IsContiguous() &&
        src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
        MemoryManager::Memcpy(
                dst.GetDataPtr(), dst.GetDevice(), src.GetDataPtr(),
                src.GetDevice(),
                DtypeUtil::ByteSize(src_dtype) * shape.NumElements());
    } else {
        Indexer indexer({src}, dst, DtypePolicy::NONE);
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            using src_t = scalar_t;
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                using dst_t = scalar_t;
                CPULauncher::LaunchUnaryEWKernel(
                        indexer, CPUCopyElementKernel<src_t, dst_t>);
            });
        });
    }
}

void UnaryEWCPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been chaged to have the same shape, device
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Indexer indexer({src}, dst, DtypePolicy::ASSERT_SAME_OR_BOOL_OUT);

    auto assert_dtype_is_float = [](Dtype dtype) -> void {
        if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
            utility::LogError(
                    "Only supports Float32 and Float64, but {} is used.",
                    DtypeUtil::ToString(dtype));
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            using src_t = scalar_t;
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                using dst_t = scalar_t;
                CPULauncher::LaunchUnaryEWKernel(
                        indexer, CPULogicalNotElementKernel<src_t, dst_t>);
            });
        });
    } else {
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    assert_dtype_is_float(src_dtype);
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUSqrtElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Sin:
                    assert_dtype_is_float(src_dtype);
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUSinElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Cos:
                    assert_dtype_is_float(src_dtype);
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUCosElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Neg:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUNegElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Exp:
                    assert_dtype_is_float(src_dtype);
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUExpElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Abs:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUAbsElementKernel<scalar_t>);
                    break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWCPU");
                    break;
            }
        });
    }
}

}  // namespace kernel
}  // namespace open3d
