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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/BinaryEW.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_ISPC_MODULE
#include "BinaryEWCPU_ispc.h"
#endif

namespace open3d {
namespace core {
namespace kernel {

template <typename src_t, typename dst_t, typename element_func_t>
static void LaunchBinaryEWKernel(const Indexer& indexer,
                                 const element_func_t& element_func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &element_func](int64_t i) {
                    element_func(indexer.GetInputPtr<src_t>(0, i),
                                 indexer.GetInputPtr<src_t>(1, i),
                                 indexer.GetOutputPtr<dst_t>(i));
                });
}

template <typename src_t,
          typename dst_t,
          typename element_func_t,
          typename vec_func_t>
static void LaunchBinaryEWKernel(const Indexer& indexer,
                                 const element_func_t& element_func,
                                 const vec_func_t& vec_func) {
    ParallelFor(
            Device("CPU:0"), indexer.NumWorkloads(),
            [&indexer, &element_func](int64_t i) {
                element_func(indexer.GetInputPtr<src_t>(0, i),
                             indexer.GetInputPtr<src_t>(1, i),
                             indexer.GetOutputPtr<dst_t>(i));
            },
            vec_func);
}

template <typename scalar_t>
static void CPUMaxElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = std::max(*static_cast<const scalar_t*>(lhs),
                                            *static_cast<const scalar_t*>(rhs));
}

template <typename scalar_t>
static void CPUMinElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = std::min(*static_cast<const scalar_t*>(lhs),
                                            *static_cast<const scalar_t*>(rhs));
}

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

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code) {
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    if (s_boolean_binary_ew_op_codes.find(op_code) !=
        s_boolean_binary_ew_op_codes.end()) {
        if (dst_dtype == src_dtype) {
            // Inplace boolean op's output type is the same as the
            // input. e.g. np.logical_and(a, b, out=a), where a, b are
            // floats.
            Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
#ifdef BUILD_ISPC_MODULE
            ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalAndElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalAndElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalOrElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalOrElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalXorElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalXorElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Gt:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer, CPUGtElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalGtElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Lt:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer, CPULtElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalLtElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Ge:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUGeqElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalGeqElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Le:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULeqElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalLeqElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Eq:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer, CPUEqElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalEqElementKernel,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Ne:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUNeqElementKernel<scalar_t, scalar_t>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t, CPULogicalNeqElementKernel,
                                        &ispc_indexer));
                        break;
                    default:
                        break;
                }
            });
        } else if (dst_dtype == core::Bool) {
            // By default, output is boolean type.
            Indexer indexer({lhs, rhs}, dst,
                            DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
#ifdef BUILD_ISPC_MODULE
            ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalAndElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalAndElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalOrElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalOrElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalXorElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalXorElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Gt:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUGtElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalGtElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Lt:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPULtElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalLtElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Ge:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUGeqElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalGeqElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Le:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPULeqElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalLeqElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Eq:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUEqElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalEqElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    case BinaryEWOpCode::Ne:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUNeqElementKernel<scalar_t, bool>,
                                OPEN3D_TEMPLATE_VECTORIZED(
                                        scalar_t,
                                        CPULogicalNeqElementKernel_bool,
                                        &ispc_indexer));
                        break;
                    default:
                        break;
                }
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == BinaryEWOpCode::Maximum ||
               op_code == BinaryEWOpCode::Minimum) {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Maximum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMaxElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Minimum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMinElementKernel<scalar_t>);
                    break;
                default:
                    break;
            }
        });
    } else {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
#ifdef BUILD_ISPC_MODULE
        ispc::Indexer ispc_indexer = indexer.ToISPC();
#endif
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Add:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUAddElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUAddElementKernel,
                                                       &ispc_indexer));
                    break;
                case BinaryEWOpCode::Sub:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUSubElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUSubElementKernel,
                                                       &ispc_indexer));
                    break;
                case BinaryEWOpCode::Mul:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMulElementKernel<scalar_t>,
                            OPEN3D_TEMPLATE_VECTORIZED(scalar_t,
                                                       CPUMulElementKernel,
                                                       &ispc_indexer));
                    break;
                case BinaryEWOpCode::Div:
                    // The vectorized Div kernel causes a crash in the Python
                    // tests, so use scalar version instead.
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUDivElementKernel<scalar_t>);
                    break;
                default:
                    break;
            }
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
