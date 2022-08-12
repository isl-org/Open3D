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
#include "open3d/core/SYCLUtils.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void CopySYCLOld(const Tensor& src, Tensor& dst) {
    // src and dst have been checked to have the same shape, dtype, device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    sy::queue& queue = sycl::GetDefaultQueue(src.GetDevice());

    if (src.IsContiguous() && dst.IsContiguous() &&
        src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
        MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                              src.GetDataPtr(), src.GetDevice(),
                              src_dtype.ByteSize() * shape.NumElements());
    } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
               src.NumElements() == 1 && !src_dtype.IsObject()) {
        int64_t num_workloads = dst.NumElements();
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(dst_dtype, [&]() {
            scalar_t scalar_element = src.To(dst_dtype).Item<scalar_t>();
            scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
            queue.submit([&](sy::handler& h) {
                     h.parallel_for(num_workloads,
                                    [dst_ptr, scalar_element](int64_t i) {
                                        dst_ptr[i] = scalar_element;
                                    });
                 }).wait();
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::NONE);
        const int64_t num_workloads = indexer.NumWorkloads();

        if (src.GetDtype().IsObject()) {
            int64_t object_byte_size = src.GetDtype().ByteSize();
            queue.submit([&](sy::handler& h) {
                     h.parallel_for(num_workloads, [indexer, object_byte_size](
                                                           int64_t i) {
                         const char* src_bytes = static_cast<const char*>(
                                 indexer.GetInputPtr(0, i));
                         char* dst_bytes =
                                 static_cast<char*>(indexer.GetOutputPtr(i));
                         for (int64_t j = 0; j < object_byte_size; j++) {
                             dst_bytes[j] = src_bytes[j];
                         }
                     });
                 }).wait();
        } else {
            // Type-cast is handled here.
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
                using src_t = scalar_t;
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(dst_dtype, [&]() {
                    using dst_t = scalar_t;
                    queue.submit([&](sy::handler& h) {
                             h.parallel_for(num_workloads, [indexer](
                                                                   int64_t i) {
                                 dst_t* dst = indexer.GetOutputPtr<dst_t>(i);
                                 const src_t* src =
                                         indexer.GetInputPtr<src_t>(0, i);
                                 *dst = static_cast<dst_t>(*src);
                             });
                         }).wait();
                });
            });
        }
    }
}

void CopySYCL(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same dtype
    // - at least one of src or dst is SYCL device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();

    if (src_device.IsSYCL() && dst_device.IsSYCL()) {
        if (src.IsContiguous() && dst.IsContiguous() &&
            src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
            // MemoryManager handles p2p and non-p2p device copy.
            MemoryManager::Memcpy(dst.GetDataPtr(), dst_device,
                                  src.GetDataPtr(), src_device,
                                  src_dtype.ByteSize() * shape.NumElements());
        } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
                   src.NumElements() == 1 && !src_dtype.IsObject()) {
            sy::queue& queue = sycl::GetDefaultQueue(dst.GetDevice());
            const int64_t num_workloads = dst.NumElements();
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(dst_dtype, [&]() {
                scalar_t scalar_element = src.To(dst_dtype).Item<scalar_t>();
                scalar_t* dst_ptr = static_cast<scalar_t*>(dst.GetDataPtr());
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads,
                                        [dst_ptr, scalar_element](int64_t i) {
                                            dst_ptr[i] = scalar_element;
                                        });
                     }).wait();
            });
        } else if (src_device == dst_device) {
            sy::queue& queue = sycl::GetDefaultQueue(src.GetDevice());
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            const int64_t num_workloads = indexer.NumWorkloads();
            if (src.GetDtype().IsObject()) {
                int64_t object_byte_size = src.GetDtype().ByteSize();
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer,
                                                        object_byte_size](
                                                               int64_t i) {
                             const char* src_bytes = static_cast<const char*>(
                                     indexer.GetInputPtr(0, i));
                             char* dst_bytes = static_cast<char*>(
                                     indexer.GetOutputPtr(i));
                             for (int64_t j = 0; j < object_byte_size; j++) {
                                 dst_bytes[j] = src_bytes[j];
                             }
                         });
                     }).wait();
            } else {
                // Type-cast is handled here.
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        queue.submit([&](sy::handler& h) {
                                 h.parallel_for(
                                         num_workloads, [indexer](int64_t i) {
                                             dst_t* dst = indexer.GetOutputPtr<
                                                     dst_t>(i);
                                             const src_t* src =
                                                     indexer.GetInputPtr<src_t>(
                                                             0, i);
                                             *dst = static_cast<dst_t>(*src);
                                         });
                             }).wait();
                    });
                });
            }
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else if (src_device.IsCPU() && dst_device.IsSYCL() ||
               src_device.IsSYCL() && dst_device.IsCPU()) {
        Tensor src_contiguous = src.Contiguous();
        if (dst.IsContiguous() && src.GetShape() == dst.GetShape() &&
            src_dtype == dst_dtype) {
            MemoryManager::Memcpy(dst.GetDataPtr(), dst_device,
                                  src_contiguous.GetDataPtr(),
                                  src_contiguous.GetDevice(),
                                  src_dtype.ByteSize() * shape.NumElements());
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else {
        utility::LogError("Wrong device type {} -> {}", src_device.ToString(),
                          dst_device.ToString());
    }
}

void UnaryEWSYCL(const Tensor& src, Tensor& dst, const UnaryEWOpCode& op_code) {
    // src and dst have been changed to have the same shape, device
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    sy::queue& queue = sycl::GetDefaultQueue(src.GetDevice());

    auto assert_dtype_is_float32 = [](Dtype dtype) -> void {
        if (dtype != core::Float32) {
            utility::LogError("Only supports Float32, but {} is used.",
                              dtype.ToString());
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        if (dst_dtype == src_dtype) {
            Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
            const int64_t num_workloads = indexer.NumWorkloads();
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = static_cast<scalar_t>(
                                     !static_cast<bool>(*src));
                         });
                     }).wait();
            });
        } else if (dst_dtype == core::Bool) {
            Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            const int64_t num_workloads = indexer.NumWorkloads();
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             bool* dst = indexer.GetOutputPtr<bool>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = !static_cast<bool>(*src);
                         });
                     }).wait();
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        assert_dtype_is_float32(src_dtype);
        using scalar_t = float;
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
        const int64_t num_workloads = indexer.NumWorkloads();
        if (op_code == UnaryEWOpCode::IsNan) {
            queue.submit([&](sy::handler& h) {
                     h.parallel_for(num_workloads, [indexer](int64_t i) {
                         bool* dst = indexer.GetOutputPtr<bool>(i);
                         const scalar_t* src =
                                 indexer.GetInputPtr<scalar_t>(0, i);
                         *dst = std::isnan(*src);
                     });
                 }).wait();
        } else if (op_code == UnaryEWOpCode::IsInf) {
            queue.submit([&](sy::handler& h) {
                     h.parallel_for(num_workloads, [indexer](int64_t i) {
                         bool* dst = indexer.GetOutputPtr<bool>(i);
                         const scalar_t* src =
                                 indexer.GetInputPtr<scalar_t>(0, i);
                         *dst = std::isinf(*src);
                     });
                 }).wait();
        } else if (op_code == UnaryEWOpCode::IsFinite) {
            queue.submit([&](sy::handler& h) {
                     h.parallel_for(num_workloads, [indexer](int64_t i) {
                         bool* dst = indexer.GetOutputPtr<bool>(i);
                         const scalar_t* src =
                                 indexer.GetInputPtr<scalar_t>(0, i);
                         *dst = std::isfinite(*src);
                     });
                 }).wait();
        }
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
        const int64_t num_workloads = indexer.NumWorkloads();

        DISPATCH_DTYPE_TO_TEMPLATE_SYCL(src_dtype, [&]() {
            if (op_code == UnaryEWOpCode::Sqrt) {
                assert_dtype_is_float32(src_dtype);
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::sqrt(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Sin) {
                assert_dtype_is_float32(src_dtype);
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::sin(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Cos) {
                assert_dtype_is_float32(src_dtype);
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::cos(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Neg) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = -(*src);
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Exp) {
                assert_dtype_is_float32(src_dtype);
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::exp(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Abs) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::abs(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Floor) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::floor(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Ceil) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::ceil(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Round) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::round(static_cast<float>(*src));
                         });
                     }).wait();
            } else if (op_code == UnaryEWOpCode::Trunc) {
                queue.submit([&](sy::handler& h) {
                         h.parallel_for(num_workloads, [indexer](int64_t i) {
                             scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
                             const scalar_t* src =
                                     indexer.GetInputPtr<scalar_t>(0, i);
                             *dst = std::trunc(static_cast<float>(*src));
                         });
                     }).wait();
            } else {
                utility::LogError("Unimplemented op_code for UnaryEWCPU");
            }
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
