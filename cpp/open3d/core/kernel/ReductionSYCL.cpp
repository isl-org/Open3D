// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <limits>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Reduction.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
namespace kernel {

namespace {

template <typename scalar_t>
struct ArgMinReduction {
    using basic_reduction = sycl::minimum<scalar_t>;
    std::pair<int64_t, scalar_t> operator()(int64_t a_idx,
                                            scalar_t a_val,
                                            int64_t b_idx,
                                            scalar_t b_val) const {
        return a_val < b_val ? std::make_pair(a_idx, a_val)
                             : std::make_pair(b_idx, b_val);
    }
};

template <typename scalar_t>
struct ArgMaxReduction {
    using basic_reduction = sycl::maximum<scalar_t>;
    std::pair<int64_t, scalar_t> operator()(int64_t a_idx,
                                            scalar_t a_val,
                                            int64_t b_idx,
                                            scalar_t b_val) const {
        return a_val > b_val ? std::make_pair(a_idx, a_val)
                             : std::make_pair(b_idx, b_val);
    }
};

// TODO: This launches one kernel per output element, which can be inefficient
// in cases where the reduction dim is small but the non-reduced dim is large.
// Unit tests for a large number of outputs are disabled.
// Speed-up by launching one kernel for the entire reduction.
template <class ReductionOp, typename scalar_t>
void SYCLReductionEngine(Device device, Indexer indexer, scalar_t identity) {
    auto device_props =
            sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    auto queue = device_props.queue;
    auto work_group_size = device_props.max_work_group_size;
    size_t log2elements_per_group = 13;
    auto elements_per_group = (1 << log2elements_per_group);  // 8192
    size_t log2workitems_per_group = 8;
    auto workitems_per_group = (1 << log2workitems_per_group);  // 256
    auto elements_per_work_item =
            elements_per_group / workitems_per_group;  // 32 (= max SIMD size)
    auto mask = ~(~0 << log2workitems_per_group);
    ReductionOp red_op;

    for (int64_t output_idx = 0; output_idx < indexer.NumOutputElements();
         output_idx++) {
        // sub_indexer.NumWorkloads() == ipo.
        // sub_indexer's workload_idx is indexer's ipo_idx.
        Indexer scalar_out_indexer = indexer.GetPerOutputIndexer(output_idx);
        auto num_elements = scalar_out_indexer.NumWorkloads();
        auto num_work_groups = num_elements / elements_per_group;
        if (num_elements > elements_per_group * num_work_groups)
            ++num_work_groups;
        // ensure each work group has work_group_size work items
        auto num_work_items = num_work_groups * work_group_size;

        auto red_cg = [&](auto& cgh) {
            auto output = scalar_out_indexer.GetOutputPtr<scalar_t>(0);
            // Setting this still doesn't initialize to identity -
            // output buffer must be initialized separately.
            auto sycl_reducer = sycl::reduction(
                    output, identity, red_op,
                    {sycl::property::reduction::initialize_to_identity()});
            cgh.parallel_for(
                    sycl::nd_range<1>{num_work_items, work_group_size},
                    sycl_reducer, [=](sycl::nd_item<1> item, auto& red_arg) {
                        auto glob_id = item.get_global_id(0);
                        auto offset = ((glob_id >> log2workitems_per_group)
                                       << log2elements_per_group) +
                                      (glob_id & mask);
                        auto item_out = identity;
                        for (size_t i = 0; i < elements_per_work_item; i++) {
                            size_t idx =
                                    (i << log2workitems_per_group) + offset;
                            if (idx >= num_elements) break;
                            auto val =
                                    *scalar_out_indexer.GetInputPtr<scalar_t>(
                                            0, idx);
                            item_out = red_op(item_out, val);
                        }
                        red_arg.combine(item_out);
                    });
        };

        auto e = queue.submit(red_cg);
    }
    queue.wait_and_throw();
}

// Based on OneAPI GPU optimization guide code sample (Blocked access to
// input data + SYCL builtin reduction ops for final reduction)
// TODO: This launches one kernel per output element, which can be inefficient
// in cases where the reduction dim is small but the non-reduced dim is large.
// Speed-up by launching one kernel for the entire reduction.
template <class ReductionOp, typename scalar_t>
void SYCLArgReductionEngine(Device device, Indexer indexer, scalar_t identity) {
    auto device_props =
            sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    auto queue = device_props.queue;
    auto work_group_size = device_props.max_work_group_size;
    size_t log2elements_per_group = 13;
    auto elements_per_group = (1 << log2elements_per_group);  // 8192
    size_t log2workitems_per_group = 8;
    auto workitems_per_group = (1 << log2workitems_per_group);  // 256
    auto elements_per_work_item =
            elements_per_group / workitems_per_group;  // 32 (= max SIMD size)
    auto mask = ~(~0 << log2workitems_per_group);
    ReductionOp red_op;

    // atomic flag. Must be 4 bytes.
    sycl::buffer<int32_t, 1> output_in_use{indexer.NumOutputElements()};
    auto e_fill = queue.submit([&](auto& cgh) {
        auto acc_output_in_use =
                output_in_use.get_access<sycl::access_mode::write>(cgh);
        cgh.fill(acc_output_in_use, 0);
    });

    for (int64_t output_idx = 0; output_idx < indexer.NumOutputElements();
         output_idx++) {
        // sub_indexer.NumWorkloads() == ipo.
        // sub_indexer's workload_idx is indexer's ipo_idx.
        Indexer scalar_out_indexer = indexer.GetPerOutputIndexer(output_idx);
        auto num_elements = scalar_out_indexer.NumWorkloads();
        auto num_work_groups = num_elements / elements_per_group;
        if (num_elements > elements_per_group * num_work_groups)
            ++num_work_groups;
        // ensure each work group has work_group_size work items
        auto num_work_items = num_work_groups * work_group_size;

        sycl::buffer<int32_t, 1> this_output_in_use{output_in_use, output_idx,
                                                    1};
        auto arg_red_cg = [&](auto& cgh) {
            auto acc_in_use =
                    this_output_in_use
                            .get_access<sycl::access_mode::read_write>(cgh);
            cgh.parallel_for(
                    sycl::nd_range<1>{num_work_items, work_group_size},
                    [=](sycl::nd_item<1> item) {
                        auto& out_idx =
                                *scalar_out_indexer.GetOutputPtr<int64_t>(0, 0);
                        auto& out_val =
                                *scalar_out_indexer.GetOutputPtr<scalar_t>(1,
                                                                           0);
                        auto glob_id = item.get_global_id(0);
                        auto this_group = item.get_group();
                        auto offset = ((glob_id >> log2workitems_per_group)
                                       << log2elements_per_group) +
                                      (glob_id & mask);
                        int64_t it_idx = 0;
                        scalar_t it_val = identity;
                        for (size_t i = 0; i < elements_per_work_item; i++) {
                            size_t idx =
                                    (i << log2workitems_per_group) + offset;
                            if (idx >= num_elements) break;
                            auto val =
                                    *scalar_out_indexer.GetInputPtr<scalar_t>(
                                            0, idx);
                            std::tie(it_idx, it_val) =
                                    red_op(it_idx, it_val, idx, val);
                        }
                        auto group_out_val = sycl::reduce_over_group(
                                this_group, it_val, identity,
                                typename ReductionOp::basic_reduction());
                        // atomic (serial) reduction over all groups. SYCL does
                        // not have a barrier over groups. Work item(s) with min
                        // / max value update the output. (non-deterministic)
                        if (it_val == group_out_val) {
                            // TODO: Look for a better option to a spinlock
                            // mutex.
                            auto in_use = sycl::atomic_ref<
                                    int32_t, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(acc_in_use[0]);
                            while (in_use.exchange(1) == 1) {
                            }
                            std::tie(out_idx, out_val) = red_op(
                                    out_idx, out_val, it_idx, group_out_val);
                            in_use.store(0);
                        }
                    });
        };

        auto e = queue.submit(arg_red_cg);
    }
    queue.wait_and_throw();
}
}  // namespace

void ReductionSYCL(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code) {
    Device device = src.GetDevice();
    if (s_regular_reduce_ops.find(op_code) != s_regular_reduce_ops.end()) {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
            scalar_t identity;
            switch (op_code) {
                case ReductionOpCode::Sum:
                    dst.Fill(0);
                    SYCLReductionEngine<sycl::plus<scalar_t>, scalar_t>(
                            device, indexer, 0);
                    break;
                case ReductionOpCode::Prod:
                    dst.Fill(1);
                    SYCLReductionEngine<sycl::multiplies<scalar_t>, scalar_t>(
                            device, indexer, 1);
                    break;
                case ReductionOpCode::Min:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Min.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::max();
                        dst.Fill(identity);
                        SYCLReductionEngine<sycl::minimum<scalar_t>, scalar_t>(
                                device, indexer, identity);
                    }
                    break;
                case ReductionOpCode::Max:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Max.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::lowest();
                        dst.Fill(identity);
                        SYCLReductionEngine<sycl::maximum<scalar_t>, scalar_t>(
                                device, indexer, identity);
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_arg_reduce_ops.find(op_code) != s_arg_reduce_ops.end()) {
        if (dst.GetDtype() != core::Int64) {
            utility::LogError("Arg-reduction must have int64 output dtype.");
        }
        // Accumulation buffer to store temporary min/max values.
        Tensor dst_acc(dst.GetShape(), src.GetDtype(), src.GetDevice());
        Indexer indexer({src}, {dst, dst_acc}, DtypePolicy::INPUT_SAME, dims);
        DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
            scalar_t identity;
            switch (op_code) {
                case ReductionOpCode::ArgMin:
                    identity = std::numeric_limits<scalar_t>::max();
                    dst_acc.Fill(identity);
                    SYCLArgReductionEngine<ArgMinReduction<scalar_t>, scalar_t>(
                            device, indexer, identity);
                    break;
                case ReductionOpCode::ArgMax:
                    identity = std::numeric_limits<scalar_t>::lowest();
                    dst_acc.Fill(identity);
                    SYCLArgReductionEngine<ArgMaxReduction<scalar_t>, scalar_t>(
                            device, indexer, identity);
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_boolean_reduce_ops.find(op_code) !=
               s_boolean_reduce_ops.end()) {
        if (src.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean input tensor.");
        }
        if (dst.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean output tensor.");
        }
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        switch (op_code) {
            case ReductionOpCode::All:
                // Identity == true. 0-sized tensor, returns true.
                dst.Fill(true);
                SYCLReductionEngine<sycl::logical_and<bool>, bool>(
                        device, indexer, true);
                break;
            case ReductionOpCode::Any:
                // Identity == false. 0-sized tensor, returns false.
                dst.Fill(false);
                SYCLReductionEngine<sycl::logical_or<bool>, bool>(
                        device, indexer, false);
                break;
            default:
                utility::LogError("Unsupported op code.");
                break;
        }
    } else {
        utility::LogError("Unsupported op code.");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
