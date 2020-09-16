// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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
