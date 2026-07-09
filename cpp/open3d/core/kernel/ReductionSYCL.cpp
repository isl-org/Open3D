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

namespace open3d {
namespace core {
namespace kernel {

namespace {

// SYCL tensor reductions use Indexer to map arbitrary strided/broadcast inputs
// to outputs. Two engine implementations cover regular reductions (sum, min,
// etc.) and arg reductions (argmin, argmax).
//
// Work partitioning:
// - Each output element corresponds to reducing over all input elements that
//   match on non-reduction dimensions (e.g. sum along dim 0 fixes other dims).
// - num_outputs = indexer.NumOutputElements(); per-output workload count is
//   indexer.GetPerOutputIndexer(0).NumWorkloads() (size along reduced axes).
//
// SYCLReductionEngine chooses the launch strategy from num_outputs:
// - num_outputs > 1: one kernel, one work-group per output, strided partial
//   sums within the group, then sycl::reduce_over_group. Uses GetInputPtrDevice
//   so a single Indexer can address any (output_idx, reduction_idx) on device.
// - num_outputs == 1: blocked parallel_for + sycl::reduction for large single
//   reductions (e.g. global sum). Each work-item accumulates 32 elements with
//   a blocked index layout; the runtime reduction combines across work-groups.
//
// SYCLArgReductionEngine always uses the multi-output grid (including when
// num_outputs == 1): work-group per output, GetInputPtrDevice, then a local
// tree reduction on (index, value) pairs because sycl::reduction does not
// support arg reductions directly.

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

/// Device-side input pointer for one element of a reduction.
///
/// Equivalent on the host to:
///   indexer.GetPerOutputIndexer(output_idx).GetInputPtr(0,
///   reduction_element_idx)
/// but without building a per-output Indexer in device code (needed when many
/// outputs are reduced in one kernel).
///
/// Loops are bounded by MAX_DIMS with runtime ndims checks so SYCL/JIT sees
/// fixed trip counts.
template <typename scalar_t>
inline const scalar_t* GetInputPtrDevice(const Indexer& indexer,
                                         int64_t output_idx,
                                         int64_t reduction_element_idx) {
    int64_t ndims = indexer.NumDims();
    const int64_t* primary_shape = indexer.GetPrimaryShape();

    // 1. Compute coordinates for non-reduction dimensions from output_idx
    int64_t output_shape[MAX_DIMS] = {0};
    int64_t output_default_strides[MAX_DIMS] = {0};
    int64_t input_coords[MAX_DIMS] = {0};

    for (int64_t i = 0; i < MAX_DIMS; ++i) {
        if (i < ndims) {
            if (indexer.IsReductionDim(i)) {
                output_shape[i] = 1;
            } else {
                output_shape[i] = primary_shape[i];
            }
        }
    }
    int64_t stride = 1;
    for (int64_t i = MAX_DIMS - 1; i >= 0; --i) {
        if (i < ndims) {
            output_default_strides[i] = stride;
            stride = output_shape[i] > 1 ? stride * output_shape[i] : stride;
        }
    }
    int64_t temp_output_idx = output_idx;
    for (int64_t i = 0; i < MAX_DIMS; ++i) {
        if (i < ndims) {
            if (!indexer.IsReductionDim(i)) {
                input_coords[i] = temp_output_idx / output_default_strides[i];
                temp_output_idx = temp_output_idx % output_default_strides[i];
            }
        }
    }

    // 2. Compute coordinates for reduction dimensions from
    // reduction_element_idx
    int64_t reduction_shape[MAX_DIMS] = {0};
    int64_t reduction_default_strides[MAX_DIMS] = {0};
    for (int64_t i = 0; i < MAX_DIMS; ++i) {
        if (i < ndims) {
            if (indexer.IsReductionDim(i)) {
                reduction_shape[i] = primary_shape[i];
            } else {
                reduction_shape[i] = 1;
            }
        }
    }
    stride = 1;
    for (int64_t i = MAX_DIMS - 1; i >= 0; --i) {
        if (i < ndims) {
            reduction_default_strides[i] = stride;
            stride = reduction_shape[i] > 1 ? stride * reduction_shape[i]
                                            : stride;
        }
    }
    int64_t temp_reduction_idx = reduction_element_idx;
    for (int64_t i = 0; i < MAX_DIMS; ++i) {
        if (i < ndims) {
            if (indexer.IsReductionDim(i)) {
                input_coords[i] =
                        temp_reduction_idx / reduction_default_strides[i];
                temp_reduction_idx =
                        temp_reduction_idx % reduction_default_strides[i];
            }
        }
    }

    // 3. Reconstruct workload_idx (linear index in input shape)
    int64_t workload_idx = 0;
    const int64_t* primary_strides = indexer.GetPrimaryStrides();
    for (int64_t i = 0; i < MAX_DIMS; ++i) {
        if (i < ndims) {
            workload_idx += input_coords[i] * primary_strides[i];
        }
    }

    return indexer.GetInputPtr<scalar_t>(0, workload_idx);
}

/// Regular associative reductions (sum, prod, min, max, all, any).
template <class ReductionOp, typename scalar_t>
void SYCLReductionEngine(Device device, Indexer indexer, scalar_t identity) {
    auto device_props =
            sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    auto queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    size_t work_group_size =
            std::min<size_t>(256, device_props.max_work_group_size);
    ReductionOp red_op;

    int64_t num_outputs = indexer.NumOutputElements();
    Indexer first_out_indexer = indexer.GetPerOutputIndexer(0);
    int64_t num_elements = first_out_indexer.NumWorkloads();

    if (num_outputs > 1) {
        // Multi-output path: parallelize across outputs (e.g. sum along one
        // dimension). Favors one launch over per-output submits; inner loop is
        // a simple strided scan with per-element GetInputPtrDevice.
        queue.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(
                         sycl::nd_range<1>{num_outputs * work_group_size,
                                           work_group_size},
                         [=](sycl::nd_item<1> item) {
                             int64_t output_idx = item.get_group(0);
                             if (output_idx >= num_outputs) return;

                             auto local_id = item.get_local_linear_id();

                             scalar_t item_out = identity;
                             for (int64_t idx = local_id; idx < num_elements;
                                  idx += work_group_size) {
                                 const scalar_t* val_ptr =
                                         GetInputPtrDevice<scalar_t>(
                                                 indexer, output_idx, idx);
                                 if (val_ptr) {
                                     item_out = red_op(item_out, *val_ptr);
                                 }
                             }

                             scalar_t grp_val = sycl::reduce_over_group(
                                     item.get_group(), item_out, red_op);

                             if (local_id == 0) {
                                 scalar_t* out_ptr =
                                         indexer.GetOutputPtr<scalar_t>(
                                                 0, output_idx);
                                 if (out_ptr) {
                                     *out_ptr = grp_val;
                                 }
                             }
                         });
             }).wait_and_throw();
    } else {
        // Single-output path: optimize throughput for one large reduction.
        // Block layout: each work-group owns elements_per_group contiguous
        // logical indices; each work-item processes elements_per_work_item (32)
        // indices in a strided pattern within that block for better locality.
        size_t log2workitems_per_group = 0;
        while ((1ULL << log2workitems_per_group) < work_group_size) {
            log2workitems_per_group++;
        }
        auto workitems_per_group = (1 << log2workitems_per_group);
        size_t log2elements_per_group = log2workitems_per_group + 5;
        auto elements_per_group = (1 << log2elements_per_group);
        auto elements_per_work_item = 32;
        auto mask = ~(~0 << log2workitems_per_group);

        // With num_outputs == 1 this runs once; loop supports the same engine
        // if extended to multiple scalar outputs without GetInputPtrDevice.
        for (int64_t output_idx = 0; output_idx < num_outputs; output_idx++) {
            Indexer scalar_out_indexer =
                    indexer.GetPerOutputIndexer(output_idx);
            auto num_elements_scalar = scalar_out_indexer.NumWorkloads();
            if (num_elements_scalar == 0) {
                // nd_range with global size 0 is not well-defined across
                // SYCL backends and can skip the reduction's identity-init
                // write, leaving the output uninitialized. The caller
                // already Fill()-ed dst with the identity, so just skip.
                continue;
            }
            auto num_work_groups = num_elements_scalar / elements_per_group;
            if (num_elements_scalar > elements_per_group * num_work_groups)
                ++num_work_groups;
            auto num_work_items = num_work_groups * work_group_size;

            auto red_cg = [&](auto& cgh) {
                auto output = scalar_out_indexer.GetOutputPtr<scalar_t>(0);
                auto sycl_reducer = sycl::reduction(
                        output, identity, red_op,
                        {sycl::property::reduction::initialize_to_identity()});
                cgh.parallel_for(
                        sycl::nd_range<1>{num_work_items, work_group_size},
                        sycl_reducer,
                        [=](sycl::nd_item<1> item, auto& red_arg) {
                            auto glob_id = item.get_global_id(0);
                            auto offset = ((glob_id >> log2workitems_per_group)
                                           << log2elements_per_group) +
                                          (glob_id & mask);
                            auto item_out = identity;
                            for (size_t i = 0; i < elements_per_work_item;
                                 i++) {
                                size_t idx =
                                        (i << log2workitems_per_group) + offset;
                                if (idx >= num_elements_scalar) break;
                                auto val =
                                        *scalar_out_indexer
                                                 .GetInputPtr<scalar_t>(0, idx);
                                item_out = red_op(item_out, val);
                            }
                            red_arg.combine(item_out);
                        });
            };

            auto e = queue.submit(red_cg);
        }
        queue.wait_and_throw();
    }
}

/// Argmin/argmax: writes index (int64) and value (accumulation tensor) per
/// output. One work-group per output; partial (idx, val) per lane, then binary
/// tree in local memory. reduction_element_idx is stored as the arg index.
template <class ReductionOp, typename scalar_t>
void SYCLArgReductionEngine(Device device, Indexer indexer, scalar_t identity) {
    auto device_props =
            sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    auto queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    size_t work_group_size =
            std::min<size_t>(256, device_props.max_work_group_size);
    ReductionOp red_op;

    int64_t num_outputs = indexer.NumOutputElements();
    Indexer first_out_indexer = indexer.GetPerOutputIndexer(0);
    int64_t num_elements = first_out_indexer.NumWorkloads();

    queue.submit([&](sycl::handler& cgh) {
             sycl::local_accessor<int64_t, 1> local_idx(
                     sycl::range<1>(work_group_size), cgh);
             sycl::local_accessor<scalar_t, 1> local_val(
                     sycl::range<1>(work_group_size), cgh);
             cgh.parallel_for(
                     sycl::nd_range<1>{num_outputs * work_group_size,
                                       work_group_size},
                     [=](sycl::nd_item<1> item) {
                         int64_t output_idx = item.get_group(0);
                         if (output_idx >= num_outputs) return;

                         auto local_id = item.get_local_linear_id();

                         int64_t it_idx = 0;
                         scalar_t it_val = identity;
                         for (size_t idx = local_id; idx < num_elements;
                              idx += work_group_size) {
                             const scalar_t* val_ptr =
                                     GetInputPtrDevice<scalar_t>(
                                             indexer, output_idx, idx);
                             if (val_ptr) {
                                 std::tie(it_idx, it_val) =
                                         red_op(it_idx, it_val, idx, *val_ptr);
                             }
                         }

                         local_idx[local_id] = it_idx;
                         local_val[local_id] = it_val;
                         item.barrier(sycl::access::fence_space::local_space);

                         int stride = 1;
                         while (stride < item.get_local_range(0)) {
                             stride *= 2;
                         }
                         for (stride /= 2; stride > 0; stride /= 2) {
                             if (local_id < stride &&
                                 local_id + stride < item.get_local_range(0)) {
                                 auto other_idx = local_idx[local_id + stride];
                                 auto other_val = local_val[local_id + stride];
                                 std::tie(local_idx[local_id],
                                          local_val[local_id]) =
                                         red_op(local_idx[local_id],
                                                local_val[local_id], other_idx,
                                                other_val);
                             }
                             item.barrier(
                                     sycl::access::fence_space::local_space);
                         }

                         if (local_id == 0) {
                             int64_t* out_idx_ptr =
                                     indexer.GetOutputPtr<int64_t>(0,
                                                                   output_idx);
                             scalar_t* out_val_ptr =
                                     indexer.GetOutputPtr<scalar_t>(1,
                                                                    output_idx);
                             if (out_idx_ptr) *out_idx_ptr = local_idx[0];
                             if (out_val_ptr) *out_val_ptr = local_val[0];
                         }
                     });
         }).wait_and_throw();
}
}  // namespace

/// Dispatches SYCL reduction kernels by op kind: regular, arg, or boolean.
/// Builds an Indexer from src/dst and reduction dims; pre-fills dst (and
/// dst_acc for arg ops) with the reduction identity.
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
