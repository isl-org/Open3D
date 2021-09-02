// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "../TensorFlowHelper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

class BallQueryOpKernel : public tensorflow::OpKernel {
public:
    explicit BallQueryOpKernel(tensorflow::OpKernelConstruction* construction)
        : OpKernel(construction) {
        using namespace tensorflow;

        OP_REQUIRES_OK(construction,
                       construction->GetAttr("nsample", &nsample));
        OP_REQUIRES_OK(construction, construction->GetAttr("radius", &radius));
        OP_REQUIRES(
                construction, nsample > 0,
                errors::InvalidArgument("BallQuery expects positive nsample"));
    }

    void Compute(tensorflow::OpKernelContext* context) override {
        using namespace tensorflow;

        const Tensor& inp_tensor = context->input(0);
        OP_REQUIRES(
                context,
                inp_tensor.dims() == 3 && inp_tensor.shape().dim_size(2) == 3,
                errors::InvalidArgument("BallQuery expects "
                                        "(batch_size,num_points,3) inp shape"));
        int batch_size = inp_tensor.shape().dim_size(0);
        int pts_size = inp_tensor.shape().dim_size(1);
        auto inp_flat = inp_tensor.flat<float>();
        const float* inp = &(inp_flat(0));

        const Tensor& center_tensor = context->input(1);
        OP_REQUIRES(context,
                    center_tensor.dims() == 3 &&
                            center_tensor.shape().dim_size(2) == 3,
                    errors::InvalidArgument(
                            "BallQuery expects "
                            "(batch_size,num_points,3) center shape"));
        int ball_size = center_tensor.shape().dim_size(1);
        auto center_flat = center_tensor.flat<float>();
        const float* center = &(center_flat(0));

        Tensor* out_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                               0, TensorShape{batch_size, ball_size, nsample},
                               &out_tensor));
        auto out_flat = out_tensor->flat<int>();
        int* out = &(out_flat(0));

        Kernel(context, batch_size, pts_size, ball_size, radius, nsample,
               center, inp, out);
    }

    virtual void Kernel(tensorflow::OpKernelContext* context,
                        int b,
                        int n,
                        int m,
                        float radius,
                        int nsample,
                        const float* new_xyz,
                        const float* xyz,
                        int* idx) = 0;

protected:
    int nsample;
    float radius;
};
