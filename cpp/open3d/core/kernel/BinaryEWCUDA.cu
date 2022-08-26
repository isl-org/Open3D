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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/BinaryEW.h"

namespace open3d {
namespace core {
namespace kernel {

// Cannot be a static function since on Windows a function enclosing
// __host__ __device__ lambda function must have external linkage.
template <typename src_t, typename dst_t, typename func_t>
void LaunchBinaryEWKernel(const Device& device,
                          const Indexer& indexer,
                          const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t i) {
        element_kernel(indexer.GetInputPtr<src_t>(0, i),
                       indexer.GetInputPtr<src_t>(1, i),
                       indexer.GetOutputPtr<dst_t>(i));
    };
    ParallelFor(device, indexer.NumWorkloads(), element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchBinaryEWKernel failed.");
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAMaxElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = max(*static_cast<const scalar_t*>(lhs),
                                       *static_cast<const scalar_t*>(rhs));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAMinElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = min(*static_cast<const scalar_t*>(lhs),
                                       *static_cast<const scalar_t*>(rhs));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAAddElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) +
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASubElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) -
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAMulElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) *
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDADivElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) /
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDALogicalAndElementKernel(const void* lhs,
                                                           const void* rhs,
                                                           void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) &&
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDALogicalOrElementKernel(const void* lhs,
                                                          const void* rhs,
                                                          void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) ||
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDALogicalXorElementKernel(const void* lhs,
                                                           const void* rhs,
                                                           void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) !=
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDAGtElementKernel(const void* lhs,
                                                   const void* rhs,
                                                   void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) > *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDALtElementKernel(const void* lhs,
                                                   const void* rhs,
                                                   void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) < *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void OPEN3D_HOST_DEVICE CUDAGeqElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) >= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void OPEN3D_HOST_DEVICE CUDALeqElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) <= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void OPEN3D_HOST_DEVICE CUDAEqElementKernel(const void* lhs,
                                                   const void* rhs,
                                                   void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) == *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void OPEN3D_HOST_DEVICE CUDANeqElementKernel(const void* lhs,
                                                    const void* rhs,
                                                    void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) != *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
void LaunchBoolBinaryEWCUDAKernel(const Tensor& lhs,
                                  const Tensor& rhs,
                                  Tensor& dst,
                                  BinaryEWOpCode op_code,
                                  const Indexer& indexer) {
    Device device = lhs.GetDevice();
    switch (op_code) {
        case BinaryEWOpCode::LogicalAnd:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDALogicalAndElementKernel<src_t, dst_t>(lhs, rhs,
                                                                  dst);
                    });
            break;
        case BinaryEWOpCode::LogicalOr:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDALogicalOrElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::LogicalXor:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDALogicalXorElementKernel<src_t, dst_t>(lhs, rhs,
                                                                  dst);
                    });
            break;
        case BinaryEWOpCode::Gt:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDAGtElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::Lt:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDALtElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::Ge:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDAGeqElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::Le:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDALeqElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::Eq:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDAEqElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        case BinaryEWOpCode::Ne:
            LaunchBinaryEWKernel<src_t, dst_t>(
                    device, indexer,
                    [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                          void* dst) {
                        CUDANeqElementKernel<src_t, dst_t>(lhs, rhs, dst);
                    });
            break;
        default:
            break;
    }
}

void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code) {
    // It has been checked that
    // - lhs, rhs, dst are all in the same CUDA device
    // - lhs, rhs have the same dtype, dst also has the same dtype or is boolean
    Device src_device = lhs.GetDevice();
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    CUDAScopedDevice scoped_device(src_device);

    if (s_boolean_binary_ew_op_codes.find(op_code) !=
        s_boolean_binary_ew_op_codes.end()) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (dst_dtype == src_dtype) {
                // Inplace boolean op's output type is the same as the
                // input. e.g. np.logical_and(a, b, out=a), where a, b are
                // floats.
                Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
                LaunchBoolBinaryEWCUDAKernel<scalar_t, scalar_t>(
                        lhs, rhs, dst, op_code, indexer);
            } else if (dst_dtype == core::Bool) {
                // By default, output is boolean type.
                Indexer indexer({lhs, rhs}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);

                LaunchBoolBinaryEWCUDAKernel<scalar_t, bool>(lhs, rhs, dst,
                                                             op_code, indexer);
            } else {
                utility::LogError(
                        "Boolean op's output type must be boolean or the "
                        "same type as the input.");
            }
        });
    } else if (op_code == BinaryEWOpCode::Maximum ||
               op_code == BinaryEWOpCode::Minimum) {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Maximum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDAMaxElementKernel<scalar_t>(lhs, rhs, dst);
                            });
                    break;
                case BinaryEWOpCode::Minimum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDAMinElementKernel<scalar_t>(lhs, rhs, dst);
                            });
                    break;
                default:
                    break;
            }
        });
    } else {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Add:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDAAddElementKernel<scalar_t>(lhs, rhs, dst);
                            });
                    break;
                case BinaryEWOpCode::Sub:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDASubElementKernel<scalar_t>(lhs, rhs, dst);
                            });
                    break;
                case BinaryEWOpCode::Mul:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDAMulElementKernel<scalar_t>(lhs, rhs, dst);
                            });
                    break;
                case BinaryEWOpCode::Div:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* lhs, void* rhs,
                                                  void* dst) {
                                CUDADivElementKernel<scalar_t>(lhs, rhs, dst);
                            });
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
