// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

/// @cond
// namespace for code that is common for all kernels
namespace reduce_subarrays_sum_opkernel {

// Base class with common code for the OpKernel implementations
class ReduceSubarraysSumOpKernel : public tensorflow::OpKernel {
public:
    explicit ReduceSubarraysSumOpKernel(
            tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& values = context->input(0);
        OP_REQUIRES(context, values.shape().dims() == 1,
                    errors::InvalidArgument("values must be a rank 1 tensor"));

        const Tensor& row_splits = context->input(1);
        OP_REQUIRES(
                context, row_splits.shape().dims() == 1,
                errors::InvalidArgument("row_splits must be a rank 1 tensor"));

        // special treatment for empty values vector
        if (values.shape().dim_size(0) == 0) {
            Tensor* sums_tensor = 0;
            OP_REQUIRES_OK(context, context->allocate_output(0, values.shape(),
                                                             &sums_tensor));
            return;
        }

        Tensor* sums_tensor = 0;
        TensorShape sums_shape({row_splits.shape().dim_size(0) - 1});
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, sums_shape, &sums_tensor));

        Kernel(context, values, row_splits, *sums_tensor);
    }

    // Function with the device specific code
    virtual void Kernel(tensorflow::OpKernelContext* context,
                        const tensorflow::Tensor& values,
                        const tensorflow::Tensor& row_splits,
                        tensorflow::Tensor& sums) = 0;

private:
};

}  // namespace reduce_subarrays_sum_opkernel
/// @endcond
