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

#include <cmath>
#include <cstring>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename src_t, typename dst_t>
static void CPUCopyElementKernel(const void* src, void* dst) {
    *static_cast<dst_t*>(dst) =
            static_cast<dst_t>(*static_cast<const src_t*>(src));
}

static void CPUCopyObjectElementKernel(const void* src,
                                       void* dst,
                                       int64_t object_byte_size) {
    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    memcpy(dst_bytes, src_bytes, object_byte_size);
}

template <typename scalar_t>
static void CPUSqrtElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            std::sqrt(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUSinElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            static_cast<scalar_t>(std::sin(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUCosElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            static_cast<scalar_t>(std::cos(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUNegElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            static_cast<scalar_t>(-*static_cast<const scalar_t*>(src));
}

template <typename scalar_t>
static void CPUExpElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) =
            static_cast<scalar_t>(std::exp(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUAbsElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            std::abs(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static void CPUFloorElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(std::floor(
            static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static void CPUCeilElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            std::ceil(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static void CPURoundElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(std::round(
            static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static void CPUTruncElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(std::trunc(
            static_cast<double>(*static_cast<const scalar_t*>(src))));
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
        MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                              src.GetDataPtr(), src.GetDevice(),
                              src_dtype.ByteSize() * shape.NumElements());
    } else {
        Indexer indexer({src}, dst, DtypePolicy::NONE);
        if (src.GetDtype().IsObject()) {
            int64_t object_byte_size = src.GetDtype().ByteSize();
            CPULauncher::LaunchUnaryEWKernel(
                    indexer, [&](const void* src, void* dst) {
                        CPUCopyObjectElementKernel(src, dst, object_byte_size);
                    });

        } else {
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
}

void UnaryEWCPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been chaged to have the same shape, device
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    auto assert_dtype_is_float = [](Dtype dtype) -> void {
        if (dtype != Dtype::Float32 && dtype != Dtype::Float64) {
            utility::LogError(
                    "Only supports Float32 and Float64, but {} is used.",
                    dtype.ToString());
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (dst_dtype == src_dtype) {
                Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
                CPULauncher::LaunchUnaryEWKernel(
                        indexer,
                        CPULogicalNotElementKernel<scalar_t, scalar_t>);
            } else if (dst_dtype == Dtype::Bool) {
                Indexer indexer({src}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                CPULauncher::LaunchUnaryEWKernel(
                        indexer, CPULogicalNotElementKernel<scalar_t, bool>);
            } else {
                utility::LogError(
                        "Boolean op's output type must be boolean or the "
                        "same type as the input.");
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
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
                case UnaryEWOpCode::Floor:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUFloorElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Ceil:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUCeilElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Round:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPURoundElementKernel<scalar_t>);
                    break;
                case UnaryEWOpCode::Trunc:
                    CPULauncher::LaunchUnaryEWKernel(
                            indexer, CPUTruncElementKernel<scalar_t>);
                    break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWCPU");
                    break;
            }
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
