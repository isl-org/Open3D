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

// namespace for code that is common for all kernels
namespace reduce_subarrays_sum_opkernel_common {

// struct for providing tensor arguments as member vars with names and shape
// checks
struct TensorArguments {
    TensorArguments(tensorflow::OpKernelContext* context)
        : values(context->input(0)), prefix_sum(context->input(1)) {
        using namespace tensorflow;

        OP_REQUIRES(context, values.shape().dims() == 1,
                    errors::InvalidArgument("values must be a rank 1 tensor"));
        OP_REQUIRES(
                context, prefix_sum.shape().dims() == 1,
                errors::InvalidArgument("prefix_sum must be a rank 1 tensor"));
    }

    const tensorflow::Tensor& values;
    const tensorflow::Tensor& prefix_sum;
};

}  // namespace reduce_subarrays_sum_opkernel_common
