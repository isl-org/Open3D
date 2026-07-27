// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/BinaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

namespace {

template <typename T>
struct SYCLMaxMin {
    static inline T Max(T a, T b) { return sycl::max(a, b); }
    static inline T Min(T a, T b) { return sycl::min(a, b); }
};

template <>
struct SYCLMaxMin<bool> {
    static inline bool Max(bool a, bool b) { return a || b; }
    static inline bool Min(bool a, bool b) { return a && b; }
};

template <typename T>
inline bool BinaryEWBooleanResult(T lhs, T rhs, BinaryEWOpCode op_code) {
    switch (op_code) {
        case BinaryEWOpCode::LogicalAnd:
            return static_cast<bool>(lhs) && static_cast<bool>(rhs);
        case BinaryEWOpCode::LogicalOr:
            return static_cast<bool>(lhs) || static_cast<bool>(rhs);
        case BinaryEWOpCode::LogicalXor:
            return static_cast<bool>(lhs) != static_cast<bool>(rhs);
        case BinaryEWOpCode::Gt:
            return lhs > rhs;
        case BinaryEWOpCode::Lt:
            return lhs < rhs;
        case BinaryEWOpCode::Ge:
            return lhs >= rhs;
        case BinaryEWOpCode::Le:
            return lhs <= rhs;
        case BinaryEWOpCode::Eq:
            return lhs == rhs;
        case BinaryEWOpCode::Ne:
            return lhs != rhs;
        default:
            return false;
    }
}

template <typename T>
inline T BinaryEWArithmetic(T lhs, T rhs, BinaryEWOpCode op_code) {
    switch (op_code) {
        case BinaryEWOpCode::Add:
            return lhs + rhs;
        case BinaryEWOpCode::Sub:
            return lhs - rhs;
        case BinaryEWOpCode::Mul:
            return lhs * rhs;
        case BinaryEWOpCode::Div:
            return lhs / rhs;
        default:
            return T{};
    }
}

template <typename T>
inline T BinaryEWMaxMin(T lhs, T rhs, BinaryEWOpCode op_code) {
    if (op_code == BinaryEWOpCode::Maximum) {
        return SYCLMaxMin<T>::Max(lhs, rhs);
    }
    return SYCLMaxMin<T>::Min(lhs, rhs);
}

template <typename src_t, typename dst_t>
inline void BinaryEWBoolApplyIndexer(const Indexer& indexer,
                                     int64_t i,
                                     BinaryEWOpCode op_code) {
    const src_t* lhs = indexer.GetInputPtr<src_t>(0, i);
    const src_t* rhs = indexer.GetInputPtr<src_t>(1, i);
    dst_t* dst = indexer.GetOutputPtr<dst_t>(i);
    *dst = static_cast<dst_t>(BinaryEWBooleanResult(*lhs, *rhs, op_code));
}

template <typename scalar_t>
inline void BinaryEWArithmeticApplyIndexer(const Indexer& indexer,
                                           int64_t i,
                                           BinaryEWOpCode op_code) {
    const scalar_t* lhs = indexer.GetInputPtr<scalar_t>(0, i);
    const scalar_t* rhs = indexer.GetInputPtr<scalar_t>(1, i);
    scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
    *dst = BinaryEWArithmetic(*lhs, *rhs, op_code);
}

template <typename scalar_t>
inline void BinaryEWMaxMinApplyIndexer(const Indexer& indexer,
                                       int64_t i,
                                       BinaryEWOpCode op_code) {
    const scalar_t* lhs = indexer.GetInputPtr<scalar_t>(0, i);
    const scalar_t* rhs = indexer.GetInputPtr<scalar_t>(1, i);
    scalar_t* dst = indexer.GetOutputPtr<scalar_t>(i);
    *dst = BinaryEWMaxMin(*lhs, *rhs, op_code);
}

}  // namespace

void BinaryEWSYCL(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code) {
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Device device = lhs.GetDevice();
    if (!device.IsSYCL()) {
        utility::LogError("ParallelFor for SYCL cannot run on device {}.",
                          device.ToString());
    }
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    const bool contiguous_same_shape =
            lhs.IsContiguous() && rhs.IsContiguous() && dst.IsContiguous() &&
            lhs.GetShape() == rhs.GetShape() &&
            lhs.GetShape() == dst.GetShape();

    const bool is_boolean_op = s_boolean_binary_ew_op_codes.find(op_code) !=
                               s_boolean_binary_ew_op_codes.end();

    if (is_boolean_op) {
        if (dst_dtype == src_dtype) {
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                if (contiguous_same_shape) {
                    int64_t n = lhs.NumElements();
                    const scalar_t* lhs_ptr = lhs.GetDataPtr<scalar_t>();
                    const scalar_t* rhs_ptr = rhs.GetDataPtr<scalar_t>();
                    scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                        int64_t i = id[0];
                        dst_ptr[i] =
                                static_cast<scalar_t>(BinaryEWBooleanResult(
                                        lhs_ptr[i], rhs_ptr[i], op_code));
                    });
                } else {
                    Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
                    const int64_t n = indexer.NumWorkloads();
                    queue.parallel_for(n, [=](int64_t i) {
                        BinaryEWBoolApplyIndexer<scalar_t, scalar_t>(indexer, i,
                                                                     op_code);
                    });
                }
            });
        } else if (dst_dtype == core::Bool) {
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                if (contiguous_same_shape) {
                    int64_t n = lhs.NumElements();
                    const scalar_t* lhs_ptr = lhs.GetDataPtr<scalar_t>();
                    const scalar_t* rhs_ptr = rhs.GetDataPtr<scalar_t>();
                    bool* dst_ptr = dst.GetDataPtr<bool>();
                    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                        int64_t i = id[0];
                        dst_ptr[i] = BinaryEWBooleanResult(lhs_ptr[i],
                                                           rhs_ptr[i], op_code);
                    });
                } else {
                    Indexer indexer({lhs, rhs}, dst,
                                    DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                    const int64_t n = indexer.NumWorkloads();
                    queue.parallel_for(n, [=](int64_t i) {
                        BinaryEWBoolApplyIndexer<scalar_t, bool>(indexer, i,
                                                                 op_code);
                    });
                }
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == BinaryEWOpCode::Maximum ||
               op_code == BinaryEWOpCode::Minimum) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (contiguous_same_shape) {
                int64_t n = lhs.NumElements();
                const scalar_t* lhs_ptr = lhs.GetDataPtr<scalar_t>();
                const scalar_t* rhs_ptr = rhs.GetDataPtr<scalar_t>();
                scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    int64_t i = id[0];
                    dst_ptr[i] =
                            BinaryEWMaxMin(lhs_ptr[i], rhs_ptr[i], op_code);
                });
            } else {
                Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
                const int64_t n = indexer.NumWorkloads();
                queue.parallel_for(n, [=](int64_t i) {
                    BinaryEWMaxMinApplyIndexer<scalar_t>(indexer, i, op_code);
                });
            }
        });
    } else if (dst_dtype == src_dtype) {
        switch (op_code) {
            case BinaryEWOpCode::Add:
            case BinaryEWOpCode::Sub:
            case BinaryEWOpCode::Mul:
            case BinaryEWOpCode::Div:
                break;
            default:
                utility::LogError("Unimplemented op_code for BinaryEWSYCL");
        }
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (contiguous_same_shape) {
                int64_t n = lhs.NumElements();
                const scalar_t* lhs_ptr = lhs.GetDataPtr<scalar_t>();
                const scalar_t* rhs_ptr = rhs.GetDataPtr<scalar_t>();
                scalar_t* dst_ptr = dst.GetDataPtr<scalar_t>();
                queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                    int64_t i = id[0];
                    dst_ptr[i] =
                            BinaryEWArithmetic(lhs_ptr[i], rhs_ptr[i], op_code);
                });
            } else {
                Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
                const int64_t n = indexer.NumWorkloads();
                queue.parallel_for(n, [=](int64_t i) {
                    BinaryEWArithmeticApplyIndexer<scalar_t>(indexer, i,
                                                             op_code);
                });
            }
        });
    } else {
        utility::LogError("Unsupported dtype combination for BinaryEWSYCL.");
    }
    queue.wait_and_throw();
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
