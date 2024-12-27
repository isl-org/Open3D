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
// Based on OneAPI GPU optimization guide code sample (Blocked access to input
// data + SYCL builtin reduction ops for final reduction)
template <class ReductionOp, typename scalar_t>
void SYCLReductionEngine(
        Device device,
        Indexer indexer,
        scalar_t identity =
                sycl::known_identity<ReductionOp, scalar_t>::value) {
    auto device_props =
            sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    auto queue = device_props.queue;
    auto work_group_size = device_props.max_work_group_size;
    size_t log2elements_per_group = 13;
    auto elements_per_group = (1 << log2elements_per_group);  // 8192
    size_t log2workitems_per_group = 8;
    auto workitems_per_group = (1 << log2workitems_per_group);  // 256
    auto elements_per_work_item =
            elements_per_group / workitems_per_group;  // 32 (= max SIMD sizse)
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
        auto output =
                reinterpret_cast<scalar_t*>(scalar_out_indexer.GetOutputPtr(0));
        auto e = queue.submit([&](auto& cgh) {
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
                            scalar_t* val = reinterpret_cast<scalar_t*>(
                                    scalar_out_indexer.GetInputPtr(0, idx));
                            item_out = red_op(item_out, *val);
                        }
                        red_arg.combine(item_out);
                    });
        });
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
            switch (op_code) {
                case ReductionOpCode::Sum:
                    dst.Fill(0);
                    SYCLReductionEngine<sycl::plus<scalar_t>, scalar_t>(
                            device, indexer);
                    break;
                case ReductionOpCode::Prod:
                    dst.Fill(1);
                    SYCLReductionEngine<sycl::multiplies<scalar_t>, scalar_t>(
                            device, indexer);
                    break;
                case ReductionOpCode::Min:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Min.");
                    } else {
                        SYCLReductionEngine<sycl::minimum<scalar_t>, scalar_t>(
                                device, indexer);
                    }
                    break;
                case ReductionOpCode::Max:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Max.");
                    } else {
                        SYCLReductionEngine<sycl::maximum<scalar_t>, scalar_t>(
                                device, indexer);
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_arg_reduce_ops.find(op_code) != s_arg_reduce_ops.end()) {
        utility::LogError("SYCL Arg-reduction is not implemented.");
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
                SYCLReductionEngine<sycl::logical_and<bool>, bool>(device,
                                                                   indexer);
                break;
            case ReductionOpCode::Any:
                // Identity == false. 0-sized tensor, returns false.
                dst.Fill(false);
                SYCLReductionEngine<sycl::logical_or<bool>, bool>(device,
                                                                  indexer);
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
