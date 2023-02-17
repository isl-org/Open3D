// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_ISPC_MODULE
#include "UnaryEWCPU_ispc.h"
#endif

namespace open3d {
namespace core {
namespace kernel {

template <typename element_func_t>
static void LaunchUnaryEWKernel(const Indexer& indexer,
                                const element_func_t& element_func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &element_func](int64_t i) {
                    element_func(indexer.GetInputPtr(0, i),
                                 indexer.GetOutputPtr(i));
                });
}

template <typename src_t, typename dst_t, typename element_func_t>
static void LaunchUnaryEWKernel(const Indexer& indexer,
                                const element_func_t& element_func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &element_func](int64_t i) {
                    element_func(indexer.GetInputPtr<src_t>(0, i),
                                 indexer.GetOutputPtr<dst_t>(i));
                });
}

template <typename src_t,
          typename dst_t,
          typename element_func_t,
          typename vec_func_t>
static void LaunchUnaryEWKernel(const Indexer& indexer,
                                const element_func_t& element_func,
                                const vec_func_t& vec_func) {
    ParallelFor(
            Device("CPU:0"), indexer.NumWorkloads(),
            [&indexer, &element_func](int64_t i) {
                element_func(indexer.GetInputPtr<src_t>(0, i),
                             indexer.GetOutputPtr<dst_t>(i));
            },
            vec_func);
}

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
static void CPUIsNanElementKernel(const void* src, void* dst) {
    *static_cast<bool*>(dst) =
            std::isnan(static_cast<float>(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUIsInfElementKernel(const void* src, void* dst) {
    *static_cast<bool*>(dst) =
            std::isinf(static_cast<float>(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static void CPUIsFiniteElementKernel(const void* src, void* dst) {
    *static_cast<bool*>(dst) = std::isfinite(
            static_cast<float>(*static_cast<const scalar_t*>(src)));
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
    } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
               src.NumElements() == 1 && !src_dtype.IsObject()) {
        int64_t num_elements = dst.NumElements();

        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
            scalar_t scalar_element = src.To(dst_dtype).Item<scalar_t>();
            scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
            ParallelFor(Device("CPU:0"), num_elements,
                        [&](int64_t workload_idx) {
                            dst_ptr[workload_idx] = scalar_element;
                        });
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::NONE);
        if (src.GetDtype().IsObject()) {
            int64_t object_byte_size = src.GetDtype().ByteSize();
            LaunchUnaryEWKernel(indexer, [&](const void* src, void* dst) {
                CPUCopyObjectElementKernel(src, dst, object_byte_size);
            });

        } else {
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                using src_t = scalar_t;
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                    using dst_t = scalar_t;
                    LaunchUnaryEWKernel<src_t, dst_t>(
                            indexer, CPUCopyElementKernel<src_t, dst_t>);
                });
            });
        }
    }
}

void UnaryEWCPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been changed to have the same shape, device
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    auto assert_dtype_is_float = [](Dtype dtype) -> void {
        if (dtype != core::Float32 && dtype != core::Float64) {
            utility::LogError(
                    "Only supports Float32 and Float64, but {} is used.",
                    dtype.ToString());
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        if (dst_dtype == src_dtype) {
            Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
#ifdef BUILD_ISPC_MODULE
            ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                LaunchUnaryEWKernel<scalar_t, scalar_t>(
                        indexer, CPULogicalNotElementKernel<scalar_t, scalar_t>,
                        OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                   CPULogicalNotElementKernel,
                                                   &ispc_indexer));
            });
        } else if (dst_dtype == core::Bool) {
            Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
#ifdef BUILD_ISPC_MODULE
            ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        indexer, CPULogicalNotElementKernel<scalar_t, bool>,
                        OPEN3D_TEMPLATE_VECTORIZED(
                                scalar_t, CPULogicalNotElementKernel_bool,
                                &ispc_indexer));
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        assert_dtype_is_float(src_dtype);
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
#ifdef BUILD_ISPC_MODULE
        ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (op_code == UnaryEWOpCode::IsNan) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        indexer, CPUIsNanElementKernel<scalar_t>,
                        OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                   CPUIsNanElementKernel,
                                                   &ispc_indexer));
            } else if (op_code == UnaryEWOpCode::IsInf) {
                // A vectorized isinf function is not defined, so use scalar
                // version instead.
                LaunchUnaryEWKernel<scalar_t, bool>(
                        indexer, CPUIsInfElementKernel<scalar_t>);
            } else if (op_code == UnaryEWOpCode::IsFinite) {
                // A vectorized isfinite function is not defined, so use scalar
                // version instead.
                LaunchUnaryEWKernel<scalar_t, bool>(
                        indexer, CPUIsFiniteElementKernel<scalar_t>);
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
#ifdef BUILD_ISPC_MODULE
        ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUSqrtElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUSqrtElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Sin:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUSinElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUSinElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Cos:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUCosElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUCosElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Neg:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUNegElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUNegElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Exp:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUExpElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUExpElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Abs:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUAbsElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUAbsElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Floor:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUFloorElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUFloorElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Ceil:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUCeilElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUCeilElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Round:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPURoundElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPURoundElementKernel,
                                                       &ispc_indexer));
                    break;
                case UnaryEWOpCode::Trunc:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUTruncElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUTruncElementKernel,
                                                       &ispc_indexer));
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
