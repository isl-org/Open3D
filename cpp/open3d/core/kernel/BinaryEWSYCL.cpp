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

struct BinaryElementKernel {
    void operator()(int64_t i) {}
    BinaryElementKernel(Indexer indexer_) : indexer(indexer_) {}

protected:
    Indexer indexer;
};

// Min, Max
#define BINARY_ELEMENT_KERNEL(name, elem_fn)                     \
    template <typename src_t, typename dst_t = src_t>            \
    struct name##ElementKernel : public BinaryElementKernel {    \
        using BinaryElementKernel::BinaryElementKernel;          \
        void operator()(int64_t i) {                             \
            const src_t* lhs = indexer.GetInputPtr<src_t>(0, i); \
            const src_t* rhs = indexer.GetInputPtr<src_t>(1, i); \
            dst_t* dst = indexer.GetOutputPtr<dst_t>(i);         \
            *dst = elem_fn(*lhs, *rhs);                          \
        }                                                        \
    }

BINARY_ELEMENT_KERNEL(Max, sycl::max);
BINARY_ELEMENT_KERNEL(Min, sycl::min);
#undef BINARY_ELEMENT_KERNEL

/// Specialize Min, Max for Bool, since sycl::min, sycl::max do not support it.
template <>
struct MaxElementKernel<bool, bool> : public BinaryElementKernel {
    using BinaryElementKernel::BinaryElementKernel;
    void operator()(int64_t i) {
        const bool* lhs = indexer.GetInputPtr<bool>(0, i);
        const bool* rhs = indexer.GetInputPtr<bool>(1, i);
        bool* dst = indexer.GetOutputPtr<bool>(i);
        *dst = *lhs || *rhs;
    }
};
template <>
struct MinElementKernel<bool, bool> : public BinaryElementKernel {
    using BinaryElementKernel::BinaryElementKernel;
    void operator()(int64_t i) {
        const bool* lhs = indexer.GetInputPtr<bool>(0, i);
        const bool* rhs = indexer.GetInputPtr<bool>(1, i);
        bool* dst = indexer.GetOutputPtr<bool>(i);
        *dst = *lhs && *rhs;
    }
};

// Arithmetic and Relational ops.
#define BINARY_ELEMENT_KERNEL(name, elem_op)                     \
    template <typename src_t, typename dst_t = src_t>            \
    struct name##ElementKernel : public BinaryElementKernel {    \
        using BinaryElementKernel::BinaryElementKernel;          \
        void operator()(int64_t i) {                             \
            const src_t* lhs = indexer.GetInputPtr<src_t>(0, i); \
            const src_t* rhs = indexer.GetInputPtr<src_t>(1, i); \
            dst_t* dst = indexer.GetOutputPtr<dst_t>(i);         \
            *dst = (*lhs)elem_op(*rhs);                          \
        }                                                        \
    }

BINARY_ELEMENT_KERNEL(Add, +);
BINARY_ELEMENT_KERNEL(Sub, -);
BINARY_ELEMENT_KERNEL(Mul, *);
BINARY_ELEMENT_KERNEL(Div, /);
BINARY_ELEMENT_KERNEL(Gt, >);
BINARY_ELEMENT_KERNEL(Lt, <);
BINARY_ELEMENT_KERNEL(Geq, >=);
BINARY_ELEMENT_KERNEL(Leq, <=);
BINARY_ELEMENT_KERNEL(Eq, ==);
BINARY_ELEMENT_KERNEL(Neq, !=);
#undef BINARY_ELEMENT_KERNEL

// Logical ops.
#define BINARY_ELEMENT_KERNEL(name, elem_op)                                \
    template <typename src_t, typename dst_t = src_t>                       \
    struct name##ElementKernel : public BinaryElementKernel {               \
        using BinaryElementKernel::BinaryElementKernel;                     \
        void operator()(int64_t i) {                                        \
            const src_t* lhs = indexer.GetInputPtr<src_t>(0, i);            \
            const src_t* rhs = indexer.GetInputPtr<src_t>(1, i);            \
            dst_t* dst = indexer.GetOutputPtr<dst_t>(i);                    \
            *dst = static_cast<bool>(*lhs) elem_op static_cast<bool>(*rhs); \
        }                                                                   \
    }
BINARY_ELEMENT_KERNEL(LogicalAnd, &&);
BINARY_ELEMENT_KERNEL(LogicalOr, ||);
BINARY_ELEMENT_KERNEL(LogicalXor, !=);
#undef BINARY_ELEMENT_KERNEL

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

    if (s_boolean_binary_ew_op_codes.find(op_code) !=
        s_boolean_binary_ew_op_codes.end()) {
        if (dst_dtype == src_dtype) {
            // Inplace boolean op's output type is the same as the
            // input. e.g. np.logical_and(a, b, out=a), where a, b are
            // floats.
            Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                const int64_t n = indexer.NumWorkloads();
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalAndElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalOrElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalXorElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Gt:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            GtElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Lt:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LtElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Ge:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            GeqElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Le:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LeqElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Eq:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            EqElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Ne:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            NeqElementKernel<scalar_t> ef(indexer);
                            ef(i);
                        });
                        break;
                    default:
                        break;
                }
            });
        } else if (dst_dtype == core::Bool) {
            // By default, output is boolean type.
            Indexer indexer({lhs, rhs}, dst,
                            DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                const int64_t n = indexer.NumWorkloads();
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalAndElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalOrElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LogicalXorElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Gt:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            GtElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Lt:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LtElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Ge:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            GeqElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Le:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            LeqElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Eq:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            EqElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
                        break;
                    case BinaryEWOpCode::Ne:
                        queue.parallel_for(n, [indexer](int64_t i) {
                            NeqElementKernel<scalar_t, bool> ef(indexer);
                            ef(i);
                        });
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
            const int64_t n = indexer.NumWorkloads();
            switch (op_code) {
                case BinaryEWOpCode::Maximum:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        MaxElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                case BinaryEWOpCode::Minimum:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        MinElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                default:
                    break;
            }
        });
    } else {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            const int64_t n = indexer.NumWorkloads();
            switch (op_code) {
                case BinaryEWOpCode::Add:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        AddElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                case BinaryEWOpCode::Sub:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        SubElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                case BinaryEWOpCode::Mul:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        MulElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                case BinaryEWOpCode::Div:
                    queue.parallel_for(n, [indexer](int64_t i) {
                        DivElementKernel<scalar_t> ef(indexer);
                        ef(i);
                    });
                    break;
                default:
                    break;
            }
        });
    }
    queue.wait_and_throw();
}
}  // namespace kernel
}  // namespace core
}  // namespace open3d
