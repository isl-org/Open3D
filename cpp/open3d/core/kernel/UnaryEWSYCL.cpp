// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstring>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

namespace {

struct UnaryElementKernel {
    UnaryElementKernel(Indexer indexer_) : indexer(indexer_) {}
    void operator()(int64_t i) {}

protected:
    Indexer indexer;
};

template <typename src_t, typename dst_t>
struct CopyElementKernel : public UnaryElementKernel {
    using UnaryElementKernel::UnaryElementKernel;
    void operator()(int64_t i) {
        const src_t* src = indexer.GetInputPtr<src_t>(0, i);
        dst_t* dst = indexer.GetOutputPtr<dst_t>(i);
        *dst = static_cast<dst_t>(*src);
    }
};

// Math: integers treated as double (C++11)
// no casting needed for float
#define UNARY_ELEMENT_KERNEL(name, elem_op)                                \
    template <typename src_t>                                              \
    struct name##ElementKernel : public UnaryElementKernel {               \
        using UnaryElementKernel::UnaryElementKernel;                      \
        void operator()(int64_t i) {                                       \
            const src_t* src = indexer.GetInputPtr<src_t>(0, i);           \
            src_t* dst = indexer.GetOutputPtr<src_t>(i);                   \
            *dst = static_cast<src_t>(elem_op(static_cast<double>(*src))); \
        }                                                                  \
    };                                                                     \
    template <>                                                            \
    struct name##ElementKernel<float> : public UnaryElementKernel {        \
        using UnaryElementKernel::UnaryElementKernel;                      \
        void operator()(int64_t i) {                                       \
            const float* src = indexer.GetInputPtr<float>(0, i);           \
            float* dst = indexer.GetOutputPtr<float>(i);                   \
            *dst = elem_op(*src);                                          \
        }                                                                  \
    }

UNARY_ELEMENT_KERNEL(Sqrt, sycl::sqrt);
UNARY_ELEMENT_KERNEL(Sin, sycl::sin);
UNARY_ELEMENT_KERNEL(Cos, sycl::cos);
UNARY_ELEMENT_KERNEL(Exp, sycl::exp);
// TODO: Use sycl::abs for integers (no casting)
UNARY_ELEMENT_KERNEL(Abs, sycl::fabs);
UNARY_ELEMENT_KERNEL(Floor, sycl::floor);
UNARY_ELEMENT_KERNEL(Ceil, sycl::ceil);
UNARY_ELEMENT_KERNEL(Round, sycl::round);
UNARY_ELEMENT_KERNEL(Trunc, sycl::trunc);
#undef UNARY_ELEMENT_KERNEL

// No special treatment for unsigned types - we use the SYCL runtime
// default
template <typename scalar_t>
struct NegElementKernel : public UnaryElementKernel {
    using UnaryElementKernel::UnaryElementKernel;
    void operator()(int64_t i) {
        const scalar_t* src = indexer.GetInputPtr<scalar_t>(0, i);
        scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
        *dst = -*src;
    }
};

// Float checkers: integers treated as double (C++11)
// no casting needed for float
#define UNARY_ELEMENT_KERNEL(name, elem_op)                         \
    template <typename src_t>                                       \
    struct name##ElementKernel : public UnaryElementKernel {        \
        using UnaryElementKernel::UnaryElementKernel;               \
        void operator()(int64_t i) {                                \
            const src_t* src = indexer.GetInputPtr<src_t>(0, i);    \
            bool* dst = indexer.GetOutputPtr<bool>(i);              \
            *dst = elem_op(static_cast<double>(*src));              \
        }                                                           \
    };                                                              \
    template <>                                                     \
    struct name##ElementKernel<float> : public UnaryElementKernel { \
        using UnaryElementKernel::UnaryElementKernel;               \
        void operator()(int64_t i) {                                \
            const float* src = indexer.GetInputPtr<float>(0, i);    \
            bool* dst = indexer.GetOutputPtr<bool>(i);              \
            *dst = elem_op(*src);                                   \
        }                                                           \
    }

UNARY_ELEMENT_KERNEL(IsNan, sycl::isnan);
UNARY_ELEMENT_KERNEL(IsInf, sycl::isinf);
UNARY_ELEMENT_KERNEL(IsFinite, sycl::isfinite);
#undef UNARY_ELEMENT_KERNEL

template <typename src_t, typename dst_t /* == bool or src_t */>
struct LogicalNotElementKernel : public UnaryElementKernel {
    using UnaryElementKernel::UnaryElementKernel;
    void operator()(int64_t i) {
        const src_t* src = indexer.GetInputPtr<src_t>(0, i);
        dst_t* dst = indexer.GetOutputPtr<dst_t>(i);
        *dst = static_cast<dst_t>(!static_cast<bool>(*src));
    }
};
}  // namespace

void CopySYCL(const Tensor& src, Tensor& dst) {
    // src and dst have been checked to have the same shape
    // at least one of src and dst is SYCL and the other is SYCL or CPU
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype(), dst_dtype = dst.GetDtype();
    Device src_device = src.GetDevice(), dst_device = dst.GetDevice();
    Device device_with_queue = dst.IsSYCL() ? dst.GetDevice() : src.GetDevice();
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device_with_queue);

    if (src_device.IsSYCL() && dst_device.IsSYCL()) {
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
                scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                queue.fill(dst_ptr, scalar_element, num_elements)
                        ;
            });
        } else if (src_device == dst_device) {  // non-contiguous or broadcast
                                                // on same SYCL device
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            if (src.GetDtype().IsObject()) {
                // TODO: This is likely very slow. Coalesce into less memcpy
                // calls.
                int64_t object_byte_size = src.GetDtype().ByteSize();
                for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
                    const void* src_ptr = indexer.GetInputPtr(0, i);
                    void* dst_ptr = indexer.GetOutputPtr(i);
                    queue.memcpy(dst_ptr, src_ptr, object_byte_size);
                }
                queue;
            } else {
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        const int64_t n = indexer.NumWorkloads();
                            queue.parallel_for(
                                    n, [indexer](int64_t i) {
                                        CopyElementKernel<src_t, dst_t> ef(
                                                indexer);
                                        ef(i);
                                    });
                    });
                });
            }
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else if (src_device.IsCPU() && dst_device.IsSYCL() ||
               src_device.IsSYCL() && dst_device.IsCPU()) {
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

void UnaryEWSYCL(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been changed to have the same shape, device
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Device device = src.GetDevice();  // == dst.GetDevice()
    if (!device.IsSYCL()) {
        utility::LogError("ParallelFor for SYCL cannot run on device {}.",
                          device.ToString());
    }
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    if (op_code == UnaryEWOpCode::LogicalNot) {
        if (dst_dtype == src_dtype) {
            Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
            const int64_t n = indexer.NumWorkloads();
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                LogicalNotElementKernel<scalar_t, scalar_t> ef(indexer);
                                ef(i);
                            });
            });
        } else if (dst_dtype == Bool) {
            Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            const int64_t n = indexer.NumWorkloads();
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                LogicalNotElementKernel<scalar_t, bool> ef(indexer);
                                ef(i);
                            });
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
        const int64_t n = indexer.NumWorkloads();
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (op_code == UnaryEWOpCode::IsNan) {
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                IsNanElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
            } else if (op_code == UnaryEWOpCode::IsInf) {
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                IsInfElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
            } else if (op_code == UnaryEWOpCode::IsFinite) {
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                IsFiniteElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
        const int64_t n = indexer.NumWorkloads();
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                SqrtElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Sin:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                SinElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Cos:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                CosElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Neg:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                NegElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Exp:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                ExpElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Abs:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                AbsElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Floor:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                FloorElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Ceil:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                CeilElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Round:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                RoundElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                case UnaryEWOpCode::Trunc:
                    queue.parallel_for(
                            n, [indexer](int64_t i) {
                                TruncElementKernel<scalar_t> ef(indexer);
                                ef(i);
                            });
                    break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWSYCL");
                    break;
            }
        });
    }
    queue.wait_and_throw();
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
