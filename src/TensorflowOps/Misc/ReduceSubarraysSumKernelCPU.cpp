// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

#include "Open3D/ML/Misc/Detail/ReduceSubarraysSumCPU.h"

using namespace open3d::ml::detail;
using namespace tensorflow;

template <class T>
class ReduceSubarraysSumOp : public OpKernel {
public:
    explicit ReduceSubarraysSumOp(OpKernelConstruction* construction)
        : OpKernel(construction) {}

    void Compute(OpKernelContext* context) override {
        static_assert(sizeof(int64) == sizeof(int64_t),
                      "int64 type is not compatible");

        const Tensor& values_tensor = context->input(0);
        auto values = values_tensor.flat<T>();
        const TensorShape values_shape(values_tensor.shape());
        const int values_rank = values_shape.dims();
        OP_REQUIRES(context, values_rank == 1,
                    errors::InvalidArgument("values must be a rank 1 tensor"));

        const Tensor& prefix_sum_tensor = context->input(1);
        auto prefix_sum = prefix_sum_tensor.flat<int64>();
        const TensorShape prefix_sum_shape(prefix_sum_tensor.shape());
        const int prefix_sum_rank = prefix_sum_shape.dims();
        OP_REQUIRES(
                context, prefix_sum_rank == 1,
                errors::InvalidArgument("prefix_sum must be a rank 1 tensor"));

        // special treatment for empty values vector
        if (values_shape.dim_size(0) == 0) {
            Tensor* sums_tensor = 0;
            OP_REQUIRES_OK(context, context->allocate_output(0, values_shape,
                                                             &sums_tensor));
            return;
        }

        Tensor* sums_tensor = 0;
        TensorShape sums_shape(prefix_sum_shape);
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, sums_shape, &sums_tensor));
        auto sums = sums_tensor->flat<T>();

        ReduceSubarraysSumCPU(values.data(), values_shape.dim_size(0),
                              (int64_t*)prefix_sum.data(),
                              prefix_sum_shape.dim_size(0), sums.data());
    }

private:
};

#define REG_KB(type)                                            \
    REGISTER_KERNEL_BUILDER(Name("Open3DReduceSubarraysSum")    \
                                    .Device(DEVICE_CPU)         \
                                    .TypeConstraint<type>("T"), \
                            ReduceSubarraysSumOp<type>);
REG_KB(int32_t)
REG_KB(int64)
REG_KB(float)
REG_KB(double)
#undef REG_KB
