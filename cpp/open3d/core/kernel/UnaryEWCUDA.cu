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
#include "open3d/core/kernel/UnaryEW.h"

namespace open3d {
namespace core {
namespace kernel {

// Cannot be a static function since on Windows a function enclosing
// __host__ __device__ lambda function must have external linkage.
template <typename func_t>
void LaunchUnaryEWKernel(const Device& device,
                         const Indexer& indexer,
                         const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t i) {
        element_kernel(indexer.GetInputPtr(0, i), indexer.GetOutputPtr(i));
    };
    core::ParallelFor(device, indexer.NumWorkloads(), element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchUnaryEWKernel failed.");
}

template <typename src_t, typename dst_t, typename func_t>
void LaunchUnaryEWKernel(const Device& device,
                         const Indexer& indexer,
                         const func_t& element_kernel) {
    OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);
    auto element_func = [=] OPEN3D_HOST_DEVICE(int64_t i) {
        element_kernel(indexer.GetInputPtr<src_t>(0, i),
                       indexer.GetOutputPtr<dst_t>(i));
    };
    core::ParallelFor(device, indexer.NumWorkloads(), element_func);
    OPEN3D_GET_LAST_CUDA_ERROR("LaunchUnaryEWKernel failed.");
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDACopyElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<dst_t*>(dst) =
            static_cast<dst_t>(*static_cast<const src_t*>(src));
}

static OPEN3D_HOST_DEVICE void CUDACopyObjectElementKernel(
        const void* src, void* dst, int64_t object_byte_size) {
    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    for (int i = 0; i < object_byte_size; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASqrtElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            sqrt(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASinElementKernel(const void* src,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            sin(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDACosElementKernel(const void* src,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            cos(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDANegElementKernel(const void* src,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = -*static_cast<const scalar_t*>(src);
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAExpElementKernel(const void* src,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            exp(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAAbsElementKernel(const void* src,
                                                    void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            abs(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAIsNanElementKernel(const void* src,
                                                      void* dst) {
    *static_cast<bool*>(dst) =
            isnan(static_cast<float>(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAIsInfElementKernel(const void* src,
                                                      void* dst) {
    *static_cast<bool*>(dst) =
            isinf(static_cast<float>(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAIsFiniteElementKernel(const void* src,
                                                         void* dst) {
    *static_cast<bool*>(dst) =
            isfinite(static_cast<float>(*static_cast<const scalar_t*>(src)));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDAFloorElementKernel(const void* src,
                                                      void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            floor(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDACeilElementKernel(const void* src,
                                                     void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            ceil(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDARoundElementKernel(const void* src,
                                                      void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            round(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDATruncElementKernel(const void* src,
                                                      void* dst) {
    *static_cast<scalar_t*>(dst) = static_cast<scalar_t>(
            trunc(static_cast<double>(*static_cast<const scalar_t*>(src))));
}

template <typename src_t, typename dst_t>
static OPEN3D_HOST_DEVICE void CUDALogicalNotElementKernel(const void* src,
                                                           void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            !static_cast<bool>(*static_cast<const src_t*>(src)));
}

void CopyCUDA(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same dtype
    // - at least one of src or dst is CUDA device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();

    if (src_device.GetType() == Device::DeviceType::CUDA &&
        dst_device.GetType() == Device::DeviceType::CUDA) {
        if (src.IsContiguous() && dst.IsContiguous() &&
            src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
            // MemoryManager handles p2p and non-p2p device copy.
            MemoryManager::Memcpy(dst.GetDataPtr(), dst_device,
                                  src.GetDataPtr(), src_device,
                                  src_dtype.ByteSize() * shape.NumElements());
        } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
                   src.NumElements() == 1 && !src_dtype.IsObject()) {
            int64_t num_elements = dst.NumElements();

            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                scalar_t scalar_element = src.To(dst_dtype).Item<scalar_t>();
                scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
                ParallelFor(src_device, num_elements,
                            [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
                                dst_ptr[workload_idx] = scalar_element;
                            });
            });
        } else if (src_device == dst_device) {
            // For more optimized version, one can check if P2P from src to
            // dst is enabled, then put synchronization with streams on both
            // src and dst to wait for copy kernel to complete.
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            if (src.GetDtype().IsObject()) {
                int64_t object_byte_size = src.GetDtype().ByteSize();
                LaunchUnaryEWKernel(
                        src_device, indexer,
                        [=] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDACopyObjectElementKernel(src, dst,
                                                        object_byte_size);
                        });

            } else {
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        LaunchUnaryEWKernel<src_t, dst_t>(
                                src_device, indexer,
                                // Need to wrap as extended CUDA lambda function
                                [] OPEN3D_HOST_DEVICE(const void* src,
                                                      void* dst) {
                                    CUDACopyElementKernel<src_t, dst_t>(src,
                                                                        dst);
                                });
                    });
                });
            }
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else if (src_device.GetType() == Device::DeviceType::CPU &&
                       dst_device.GetType() == Device::DeviceType::CUDA ||
               src_device.GetType() == Device::DeviceType::CUDA &&
                       dst_device.GetType() == Device::DeviceType::CPU) {
        Tensor src_conti = src.Contiguous();  // No op if already contiguous
        if (dst.IsContiguous() && src.GetShape() == dst.GetShape() &&
            src_dtype == dst_dtype) {
            MemoryManager::Memcpy(dst.GetDataPtr(), dst_device,
                                  src_conti.GetDataPtr(), src_conti.GetDevice(),
                                  src_dtype.ByteSize() * shape.NumElements());
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else {
        utility::LogError("Wrong device type {} -> {}", src_device.ToString(),
                          dst_device.ToString());
    }
}

void UnaryEWCUDA(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been chaged to have the same shape, dtype, device.
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Device src_device = src.GetDevice();

    auto assert_dtype_is_float = [](Dtype dtype) -> void {
        if (dtype != core::Float32 && dtype != core::Float64) {
            utility::LogError(
                    "Only supports Float32 and Float64, but {} is used.",
                    dtype.ToString());
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (dst_dtype == src_dtype) {
                Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
                LaunchUnaryEWKernel<scalar_t, scalar_t>(
                        src_device, indexer,
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDALogicalNotElementKernel<scalar_t, scalar_t>(
                                    src, dst);
                        });
            } else if (dst_dtype == core::Bool) {
                Indexer indexer({src}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDALogicalNotElementKernel<scalar_t, bool>(src,
                                                                        dst);
                        });
            } else {
                utility::LogError(
                        "Boolean op's output type must be boolean or the "
                        "same type as the input.");
            }
        });
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        assert_dtype_is_float(src_dtype);
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (op_code == UnaryEWOpCode::IsNan) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDAIsNanElementKernel<scalar_t>(src, dst);
                        });
            } else if (op_code == UnaryEWOpCode::IsInf) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDAIsInfElementKernel<scalar_t>(src, dst);
                        });
            } else if (op_code == UnaryEWOpCode::IsFinite) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                            CUDAIsFiniteElementKernel<scalar_t>(src, dst);
                        });
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDASqrtElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Sin:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDASinElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Cos:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDACosElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Neg:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDANegElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Exp:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDAExpElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Abs:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDAAbsElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Floor:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDAFloorElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Ceil:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDACeilElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Round:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDARoundElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Trunc:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                                CUDATruncElementKernel<scalar_t>(src, dst);
                            });
                    break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWCUDA");
                    break;
            }
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
