// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstring>
#include <type_traits>

#include "open3d/core/BlockCopyDispatch.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
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

// Contiguous fast-path helpers: match UNARY_ELEMENT_KERNEL / float-check macros
// (double uses sycl::* on T; other dtypes promote through float).
template <typename T>
struct SYCLFloatCheckers {
#define SYCL_FLOAT_CHECK(Method, sycl_fn, non_float) \
    static inline bool Method(T val) {               \
        if constexpr (std::is_floating_point_v<T>) { \
            return sycl_fn(val);                     \
        }                                            \
        return non_float;                            \
    }
    SYCL_FLOAT_CHECK(IsNan, sycl::isnan, false)
    SYCL_FLOAT_CHECK(IsInf, sycl::isinf, false)
    SYCL_FLOAT_CHECK(IsFinite, sycl::isfinite, true)
#undef SYCL_FLOAT_CHECK
};

// Separate templates so device compilation does not instantiate sycl::abs for
// floating-point T (sycl::abs is integer-only).
template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T>, T> SYCLUnaryAbs(T val) {
    return sycl::fabs(val);
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> SYCLUnaryAbs(T val) {
    return sycl::abs(val);
}

template <typename T>
struct SYCLMath {
#define SYCL_MATH_UNARY(Method, sycl_fn)                             \
    static inline T Method(T val) {                                  \
        if constexpr (std::is_same_v<T, double>) {                   \
            return sycl_fn(val);                                     \
        } else {                                                     \
            return static_cast<T>(sycl_fn(static_cast<float>(val))); \
        }                                                            \
    }
    SYCL_MATH_UNARY(Sqrt, sycl::sqrt)
    SYCL_MATH_UNARY(Sin, sycl::sin)
    SYCL_MATH_UNARY(Cos, sycl::cos)
    SYCL_MATH_UNARY(Exp, sycl::exp)
    SYCL_MATH_UNARY(Floor, sycl::floor)
    SYCL_MATH_UNARY(Ceil, sycl::ceil)
    SYCL_MATH_UNARY(Round, sycl::round)
    SYCL_MATH_UNARY(Trunc, sycl::trunc)
#undef SYCL_MATH_UNARY
    static inline T Abs(T val) { return SYCLUnaryAbs(val); }
};

template <typename scalar_t>
inline scalar_t UnaryEWTransform(UnaryEWOpCode op_code, scalar_t val) {
    switch (op_code) {
        case UnaryEWOpCode::Sqrt:
            return SYCLMath<scalar_t>::Sqrt(val);
        case UnaryEWOpCode::Sin:
            return SYCLMath<scalar_t>::Sin(val);
        case UnaryEWOpCode::Cos:
            return SYCLMath<scalar_t>::Cos(val);
        case UnaryEWOpCode::Neg:
            return -val;
        case UnaryEWOpCode::Exp:
            return SYCLMath<scalar_t>::Exp(val);
        case UnaryEWOpCode::Abs:
            return SYCLMath<scalar_t>::Abs(val);
        case UnaryEWOpCode::Floor:
            return SYCLMath<scalar_t>::Floor(val);
        case UnaryEWOpCode::Ceil:
            return SYCLMath<scalar_t>::Ceil(val);
        case UnaryEWOpCode::Round:
            return SYCLMath<scalar_t>::Round(val);
        case UnaryEWOpCode::Trunc:
            return SYCLMath<scalar_t>::Trunc(val);
        default:
            return val;
    }
}

template <typename scalar_t>
inline bool UnaryEWFloatCheck(UnaryEWOpCode op_code, scalar_t val) {
    switch (op_code) {
        case UnaryEWOpCode::IsNan:
            return SYCLFloatCheckers<scalar_t>::IsNan(val);
        case UnaryEWOpCode::IsInf:
            return SYCLFloatCheckers<scalar_t>::IsInf(val);
        case UnaryEWOpCode::IsFinite:
            return SYCLFloatCheckers<scalar_t>::IsFinite(val);
        default:
            return false;
    }
}

template <typename src_t, typename dst_t>
inline dst_t UnaryEWLogicalNot(src_t val) {
    return static_cast<dst_t>(!static_cast<bool>(val));
}

template <typename scalar_t>
inline void UnaryEWApplyIndexer(const Indexer& indexer,
                                int64_t i,
                                UnaryEWOpCode op_code) {
    const scalar_t* src = indexer.GetInputPtr<scalar_t>(0, i);
    scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
    *dst = UnaryEWTransform(op_code, *src);
}

template <typename scalar_t>
inline void UnaryEWFloatCheckApplyIndexer(const Indexer& indexer,
                                          int64_t i,
                                          UnaryEWOpCode op_code) {
    const scalar_t* src = indexer.GetInputPtr<scalar_t>(0, i);
    bool* dst = indexer.GetOutputPtr<bool>(i);
    *dst = UnaryEWFloatCheck(op_code, *src);
}

template <typename src_t, typename dst_t>
inline void UnaryEWLogicalNotApplyIndexer(const Indexer& indexer, int64_t i) {
    const src_t* src = indexer.GetInputPtr<src_t>(0, i);
    dst_t* dst = indexer.GetOutputPtr<dst_t>(i);
    *dst = UnaryEWLogicalNot<src_t, dst_t>(*src);
}

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
                        .wait_and_throw();
            });
        } else if (src_device == dst_device) {  // non-contiguous or broadcast
                                                // on same SYCL device
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            if (src.GetDtype().IsObject()) {
                const int64_t object_byte_size = src.GetDtype().ByteSize();
                const int64_t block_size =
                        GetLargestAlignedObjectBlockSize(object_byte_size);
                const int64_t n = indexer.NumWorkloads();
                DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(block_size, [&]() {
                    const int64_t blocks = object_byte_size / block_size;
                    queue.parallel_for(n, [indexer, blocks](int64_t i) {
                             // reinterpret_cast required: GetInputPtr
                             // returns char* and block_t may be
                             // sycl::vec<> which is not trivially related
                             // to char via static_cast.
                             const block_t* src =
                                     reinterpret_cast<const block_t*>(
                                             indexer.GetInputPtr(0, i));
                             block_t* dst = reinterpret_cast<block_t*>(
                                     indexer.GetOutputPtr(i));
                             for (int64_t b = 0; b < blocks; ++b) {
                                 dst[b] = src[b];
                             }
                         }).wait_and_throw();
                });
            } else {
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        const int64_t n = indexer.NumWorkloads();
                        queue.parallel_for(n, [indexer](int64_t i) {
                                 CopyElementKernel<src_t, dst_t> ef(indexer);
                                 ef(i);
                             }).wait_and_throw();
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
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    const bool contiguous_same_shape = src.IsContiguous() &&
                                       dst.IsContiguous() &&
                                       src.GetShape() == dst.GetShape();

    if (op_code == UnaryEWOpCode::LogicalNot) {
        if (dst_dtype == src_dtype) {
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                if (contiguous_same_shape) {
                    int64_t n = src.NumElements();
                    const scalar_t* src_ptr = src.GetDataPtr<scalar_t>();
                    scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                        int64_t i = id[0];
                        dst_ptr[i] = UnaryEWLogicalNot<scalar_t, scalar_t>(
                                src_ptr[i]);
                    });
                } else {
                    Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
                    const int64_t n = indexer.NumWorkloads();
                    queue.parallel_for(n, [=](int64_t i) {
                        UnaryEWLogicalNotApplyIndexer<scalar_t, scalar_t>(
                                indexer, i);
                    });
                }
            });
        } else if (dst_dtype == Bool) {
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                if (contiguous_same_shape) {
                    int64_t n = src.NumElements();
                    const scalar_t* src_ptr = src.GetDataPtr<scalar_t>();
                    bool* dst_ptr = dst.GetDataPtr<bool>();
                    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                        int64_t i = id[0];
                        dst_ptr[i] =
                                UnaryEWLogicalNot<scalar_t, bool>(src_ptr[i]);
                    });
                } else {
                    Indexer indexer({src}, dst,
                                    DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                    const int64_t n = indexer.NumWorkloads();
                    queue.parallel_for(n, [=](int64_t i) {
                        UnaryEWLogicalNotApplyIndexer<scalar_t, bool>(indexer,
                                                                      i);
                    });
                }
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (contiguous_same_shape) {
                int64_t n = src.NumElements();
                const scalar_t* src_ptr = src.GetDataPtr<scalar_t>();
                bool* dst_ptr = dst.GetDataPtr<bool>();
                queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    int64_t i = id[0];
                    dst_ptr[i] = UnaryEWFloatCheck(op_code, src_ptr[i]);
                });
            } else {
                Indexer indexer({src}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                const int64_t n = indexer.NumWorkloads();
                queue.parallel_for(n, [=](int64_t i) {
                    UnaryEWFloatCheckApplyIndexer<scalar_t>(indexer, i,
                                                            op_code);
                });
            }
        });
    } else if (dst_dtype == src_dtype) {
        switch (op_code) {
            case UnaryEWOpCode::Sqrt:
            case UnaryEWOpCode::Sin:
            case UnaryEWOpCode::Cos:
            case UnaryEWOpCode::Neg:
            case UnaryEWOpCode::Exp:
            case UnaryEWOpCode::Abs:
            case UnaryEWOpCode::Floor:
            case UnaryEWOpCode::Ceil:
            case UnaryEWOpCode::Round:
            case UnaryEWOpCode::Trunc:
                break;
            default:
                utility::LogError("Unimplemented op_code for UnaryEWSYCL");
        }
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (contiguous_same_shape) {
                int64_t n = src.NumElements();
                const scalar_t* src_ptr = src.GetDataPtr<scalar_t>();
                scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    int64_t i = id[0];
                    dst_ptr[i] = UnaryEWTransform(op_code, src_ptr[i]);
                });
            } else {
                Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
                const int64_t n = indexer.NumWorkloads();
                queue.parallel_for(n, [=](int64_t i) {
                    UnaryEWApplyIndexer<scalar_t>(indexer, i, op_code);
                });
            }
        });
    } else {
        utility::LogError("Unsupported dtype combination for UnaryEWSYCL.");
    }
    queue.wait_and_throw();
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
